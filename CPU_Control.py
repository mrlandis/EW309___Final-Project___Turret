import cv2
import depthai as dai
import serial
import math
import time
import numpy as np 

# ==========================================
# 1. CONFIGURATION 
# ==========================================
COM_PORT = 'COM23' 
BLOB_PATH = 'landis_dotsv2.blob' 

S_METERS = 4.572  

# >>> THE FIX: ALLOW BOTH CLASSES <<<
TARGET_CLASSES = [1, 2]   # AI is confused, so accept both!
MIN_SIZE = 5              # Lowered slightly to catch the W:8 dot safely
MAX_SIZE = 25             # Blocks the W:37 red circle completely

# Camera Intrinsics & Biases
CX, CY = 310.60144043, 240.59106445
FX, FY = 492.11883545, 492.11883545 
BIAS_X, BIAS_Y = 0, 0

# ==========================================
# 2. OAK-D HARDWARE PIPELINE
# ==========================================
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.Camera).build()
videoOut = cam_rgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setOutputSize(512, 512, dai.ImageManipConfig.ResizeMode.LETTERBOX)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlob(dai.OpenVINO.Blob(BLOB_PATH))

videoOut.link(manip.inputImage)
manip.out.link(nn.input)

q_rgb = videoOut.createOutputQueue(maxSize=1, blocking=False)
q_nn = nn.out.createOutputQueue(maxSize=1, blocking=False)

# ==========================================
# 3. EXECUTION LOGIC
# ==========================================
pipeline.start()

try:
    ser = serial.Serial(COM_PORT, 9600, timeout=0.01)
    ser.reset_input_buffer()   
    ser.reset_output_buffer()  
    ser.write(b"handshake\r\n")
    time.sleep(1)
except Exception as e:
    print(f">> Serial Error: {e}")
    ser = None

print("\n========================================")
print("     ACQUIRING TARGETS FOR PRE-CALC     ")
print("========================================")
print("Line up the targets. Press 'p' to SEND COMMAND.")

with pipeline:
    while pipeline.isRunning():
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()
        img = in_rgb.getCvFrame()
        
        # --- HUD OVERLAY SETUP ---
        overlay = img.copy()
        
        c_x, c_y = int(CX), int(CY)
        cv2.circle(overlay, (c_x, c_y), 15, (0, 255, 0), 1)
        cv2.line(overlay, (c_x - 25, c_y), (c_x - 5, c_y), (0, 255, 0), 1)
        cv2.line(overlay, (c_x + 5, c_y), (c_x + 25, c_y), (0, 255, 0), 1)
        cv2.line(overlay, (c_x, c_y - 25), (c_x, c_y - 5), (0, 255, 0), 1)
        cv2.line(overlay, (c_x, c_y + 5), (c_x, c_y + 25), (0, 255, 0), 1)
        cv2.circle(overlay, (c_x, c_y), 1, (0, 255, 0), -1)

        # --- RAW TENSOR DECODING ---
        raw_bytes = np.array(in_nn.getData())
        try: raw_data = raw_bytes.view(np.float16).reshape(7, 5376)
        except ValueError: raw_data = raw_bytes.astype(np.float16).reshape(7, 5376)
        
        boxes_raw = raw_data[0:4, :].T  
        scores_raw = raw_data[4:, :].T  

        confidences = np.max(scores_raw, axis=1)
        class_ids = np.argmax(scores_raw, axis=1)

        mask = confidences > 0.05
        boxes_filtered = boxes_raw[mask]
        conf_filtered = confidences[mask]
        class_filtered = class_ids[mask]

        cv_boxes = []
        for (cx_raw, cy_raw, w, h) in boxes_filtered:
            x_center = cx_raw * (640 / 512)
            y_center = (cy_raw - 64) * (640 / 512)
            width = w * (640 / 512)
            height = h * (640 / 512)
            cv_boxes.append([int(x_center - width/2), int(y_center - height/2), int(width), int(height)])

        indices = cv2.dnn.NMSBoxes(cv_boxes, conf_filtered.tolist(), 0.05, 0.4)

        current_targets = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w_box, h_box = cv_boxes[i]
                cx_box = x1 + w_box / 2.0
                cy_box = y1 + h_box / 2.0
                label = class_filtered[i]

                # Draw subtle grey brackets for all raw detections
                cv2.rectangle(overlay, (x1, y1), (x1+w_box, y1+h_box), (100, 100, 100), 1) 
                
                # Debug Text
                cv2.putText(img, f"C:{label} W:{w_box}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # >>> UPDATED FILTER LOGIC: Check if label is IN the list <<<
                if MIN_SIZE <= w_box <= MAX_SIZE and label in TARGET_CLASSES: 
                    
                    # Target acquired: Prominent Green Box + Target Dot
                    cv2.rectangle(overlay, (x1, y1), (x1+w_box, y1+h_box), (0, 255, 0), 2)
                    cv2.circle(overlay, (int(cx_box), int(cy_box)), 3, (0, 255, 255), -1)
                    
                    x_si = (S_METERS * (cx_box - CX)) / FX 
                    y_si = (S_METERS * (cy_box - CY)) / -FY 
                    
                    psi_rad = math.atan2((x_si - BIAS_X), S_METERS) 
                    theta_rad = math.atan2((y_si - BIAS_Y), S_METERS) 
                    
                    current_targets.append({
                        'yaw': math.degrees(psi_rad), 
                        'pitch': math.degrees(theta_rad), 
                        'x': cx_box
                    })

        # Sort targets strictly left-to-right
        current_targets = sorted(current_targets, key=lambda t: t['x'])
        engagement_targets = current_targets[:2]

        # --- DRAW HUD DATA PANEL ---
        cv2.rectangle(overlay, (10, 10), (330, 120), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        if len(engagement_targets) == 2:
            t1, t2 = engagement_targets[0], engagement_targets[1]
            cv2.putText(img, "SYS STATUS: READY TO ENGAGE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"T1 (LEFT)  -> Yaw: {t1['yaw']:+05.1f} | Ptch: {t1['pitch']:+05.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(img, f"T2 (RIGHT) -> Yaw: {t2['yaw']:+05.1f} | Ptch: {t2['pitch']:+05.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(img, "[ PRESS 'P' TO SEND COMMAND ]", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(img, "SYS STATUS: ACQUIRING...", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.putText(img, f"VALID TARGETS LOCKED: {len(engagement_targets)} / 2", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("EW309 Commander HUD", img)

        key = cv2.waitKey(1)
        
        if key == ord('p') and len(engagement_targets) == 2:
            t1, t2 = engagement_targets[0], engagement_targets[1]
            msg = f"{t1['yaw']:.2f},{t1['pitch']:.2f},{t2['yaw']:.2f},{t2['pitch']:.2f}\n"
            
            if ser:
                ser.write(msg.encode('ascii'))
                print(f">> Sent Coordinates to Pico: {msg.strip()}")
            else:
                print(f">> WARNING: Serial Port Not Connected. Tried to send: {msg.strip()}")
                
        if key == ord('q'): 
            break

cv2.destroyAllWindows()
if ser: ser.close()