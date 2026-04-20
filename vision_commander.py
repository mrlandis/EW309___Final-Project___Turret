import cv2
import depthai as dai
import serial
import math
import time
import numpy as np # REQUIRED FOR RAW DECODING

# ==========================================
# 1. RUBRIC CONFIGURATION 
# ==========================================
COM_PORT = 'COM23' 
S_METERS = 4.191           # Range to target in meters
BIAS_X, BIAS_Y = -0.18, -0.27 
BLOB_PATH = 'landis_dotsv2.blob' 

# INSTRUCTOR TEST PARAMETERS
TARGET_CLASS = 0           
MIN_SIZE = 8               
MAX_SIZE = 25              

# Camera Intrinsics
FX, FY, CX, CY = 492.11, 492.11, 310.60, 240.59 

# ==========================================
# 2. OAK-D Hardware Pipeline (Raw Tensor Mode)
# ==========================================
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.Camera).build()
videoOut = cam_rgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setOutputSize(512, 512, dai.ImageManipConfig.ResizeMode.STRETCH)

# The Raw Neural Network (Hardware Acceleration)
nn = pipeline.create(dai.node.NeuralNetwork)
blob = dai.OpenVINO.Blob(BLOB_PATH)
nn.setBlob(blob)

# LINKING (Notice we completely removed the DetectionParser)
videoOut.link(manip.inputImage)
manip.out.link(nn.input)

q_rgb = videoOut.createOutputQueue(maxSize=1, blocking=False)
q_nn = nn.out.createOutputQueue(maxSize=1, blocking=False)

# ==========================================
# 3. Execution Loop (Host-Side Decoding)
# ==========================================
pipeline.start()

try:
    ser = serial.Serial(COM_PORT, 9600, timeout=0.01)
    ser.write(b"handshake\r\n")
    time.sleep(1)
except Exception as e:
    ser = None

tracking_active = False
tgt_idx = 0
lock_start_time = 0

print("\n========================================")
print("     TRACKING-ONLY CALIBRATION MODE     ")
print("========================================")
print("Press 'p' to begin sequential tracking. Press 'q' to quit.\n")

with pipeline:
    while pipeline.isRunning():
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()
        
        frame = in_rgb.getCvFrame()

        # --- NEW: YOLOv11 RAW TENSOR DECODING ---
        # The AI outputs a 1D array of 37632 numbers. We reshape it to a 7x5376 matrix.
        raw_data = np.array(in_nn.getFirstLayerFp16()).reshape(7, 5376)
        
        boxes_raw = raw_data[0:4, :].T  # X, Y, W, H for 5376 grid points
        scores_raw = raw_data[4:, :].T  # Probabilities for your 3 classes

        # Find the highest probability class for each box
        confidences = np.max(scores_raw, axis=1)
        class_ids = np.argmax(scores_raw, axis=1)

        # Filter out anything below 15% confidence
        mask = confidences > 0.15
        boxes_filtered = boxes_raw[mask]
        conf_filtered = confidences[mask]
        class_filtered = class_ids[mask]

        cv_boxes = []
        for (cx, cy, w, h) in boxes_filtered:
            # Scale coordinates from the AI's 512x512 view back to our 640x480 screen
            x_center = cx * (640 / 512)
            y_center = cy * (480 / 512)
            width = w * (640 / 512)
            height = h * (480 / 512)
            cv_boxes.append([int(x_center - width/2), int(y_center - height/2), int(width), int(height)])

        # Remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(cv_boxes, conf_filtered.tolist(), 0.15, 0.4)

        targets = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w_box, h_box = cv_boxes[i]
                x2, y2 = x1 + w_box, y1 + h_box
                cx_box = x1 + w_box / 2.0
                cy_box = y1 + h_box / 2.0
                label = class_filtered[i]

                # Debug Overlay: Draw Blue boxes around all raw detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame, f"C:{label} W:{w_box}px", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

                # Rubric Filter: Only engage targets of the correct class and size
                if MIN_SIZE <= w_box <= MAX_SIZE and label == TARGET_CLASS: 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green = Valid
                    
                    x_si = (S_METERS * (cx_box - CX)) / FX 
                    y_si = (S_METERS * (cy_box - CY)) / -FY 
                    
                    yaw_err = math.degrees(math.atan2(x_si - BIAS_X, S_METERS)) 
                    pitch_err = math.degrees(math.atan2(y_si - BIAS_Y, S_METERS)) 
                    
                    targets.append({'yaw': yaw_err, 'pitch': pitch_err, 'x': cx_box})

        targets = sorted(targets, key=lambda t: t['x'])

        # STATE MACHINE LOGIC: SEQUENTIAL TRACKING
        if tracking_active and len(targets) > 0:
            if tgt_idx >= len(targets):
                tgt_idx = 0
                
            tgt = targets[tgt_idx]
            
            if ser:
                ser.write(f"TRACK,{tgt['yaw']:.2f},{tgt['pitch']:.2f}\n".encode('utf-8'))

            if abs(tgt['yaw']) <= 0.6 and abs(tgt['pitch']) <= 0.6:
                cv2.putText(frame, f"HOLDING TARGET {tgt_idx+1}...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if time.time() - lock_start_time > 3.0: 
                    print(f">> Steady State verified. Shifting to Target {(tgt_idx + 1) % len(targets) + 1}...")
                    tgt_idx += 1
                    if tgt_idx >= len(targets):
                        tgt_idx = 0 
                    lock_start_time = time.time() 
            else:
                cv2.putText(frame, f"TRACKING TGT {tgt_idx+1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                lock_start_time = time.time()
                
        elif tracking_active and len(targets) == 0:
            cv2.putText(frame, "NO VALID TARGETS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if ser: ser.write(b"TRACK,0.0,0.0\n")

        # Crosshair and Status
        cv2.drawMarker(frame, (int(CX), int(CY)), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.putText(frame, f"VALID TARGETS: {len(targets)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("EW309 Tracking Cycler", frame)

        key = cv2.waitKey(1)
        if key == ord('p'): 
            tracking_active = True
            tgt_idx = 0
            lock_start_time = time.time()
            print("\n>> Sequential Tracking Engaged.")
        if key == ord('q'): 
            break

cv2.destroyAllWindows()
if ser: ser.close()