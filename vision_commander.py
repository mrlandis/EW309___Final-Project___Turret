import cv2
import depthai as dai
import serial
import math
import time
import numpy as np 

# ==========================================
# 1. RUBRIC CONFIGURATION 
# ==========================================
COM_PORT = 'COM23' 
# CRITICAL: Change S_METERS to match your physical test range! 
# (e.g., 10ft = 3.048, 15ft = 4.572, 20ft = 6.096)
S_METERS = 3.048           
BIAS_X, BIAS_Y = -0.18, -0.27 
BLOB_PATH = 'landis_dotsv2.blob' 

# INSTRUCTOR TEST PARAMETERS
REQUIRED_SHOTS = 3         # <-- SET THIS TO MATCH YOUR INSTRUCTOR'S ORDERS
TARGET_CLASS = 2        
MIN_SIZE = 4               
MAX_SIZE = 80              

# Camera Intrinsics
FX, FY, CX, CY = 492.11, 492.11, 310.60, 240.59 

# ==========================================
# 2. OAK-D Hardware Pipeline (Raw Tensor Mode)
# ==========================================
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.Camera).build()
videoOut = cam_rgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setOutputSize(512, 512, dai.ImageManipConfig.ResizeMode.LETTERBOX)

nn = pipeline.create(dai.node.NeuralNetwork)
blob = dai.OpenVINO.Blob(BLOB_PATH)
nn.setBlob(blob)

videoOut.link(manip.inputImage)
manip.out.link(nn.input)

q_rgb = videoOut.createOutputQueue(maxSize=1, blocking=False)
q_nn = nn.out.createOutputQueue(maxSize=1, blocking=False)

# ==========================================
# 3. Execution & Evaluation Loop
# ==========================================
pipeline.start()

try:
    ser = serial.Serial(COM_PORT, 9600, timeout=0.01)
    ser.write(b"handshake\r\n")
    time.sleep(1)
except Exception as e:
    print(f">> Serial Error: {e}")
    ser = None

tracking_state = 0 
lock_start_time = 0
trial_start_time = 0
last_serial_send = 0

last_known_yaw = 0.0
last_known_pitch = 0.0
target_lost_time = 0

print("\n========================================")
print("     ACQUIRING TARGETS FOR PRE-CALC     ")
print("========================================")
print("Line up the targets. Press 'p' to ENGAGE.")

with pipeline:
    while pipeline.isRunning():
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()
        frame = in_rgb.getCvFrame()
        
        overlay = frame.copy()
        alpha = 0.4  

        # --- RAW TENSOR DECODING ---
        raw_bytes = np.array(in_nn.getData())
        try:
            raw_data = raw_bytes.view(np.float16).reshape(7, 5376)
        except ValueError:
            raw_data = raw_bytes.astype(np.float16).reshape(7, 5376)
        
        boxes_raw = raw_data[0:4, :].T  
        scores_raw = raw_data[4:, :].T  

        confidences = np.max(scores_raw, axis=1)
        class_ids = np.argmax(scores_raw, axis=1)

        # DROP CONFIDENCE TO 5% TO FIND THE TABLE TARGET
        mask = confidences > 0.05
        boxes_filtered = boxes_raw[mask]
        conf_filtered = confidences[mask]
        class_filtered = class_ids[mask]

        cv_boxes = []
        for (cx, cy, w, h) in boxes_filtered:
            x_center = cx * (640 / 512)
            y_center = (cy - 64) * (640 / 512)
            width = w * (640 / 512)
            height = h * (640 / 512)
            cv_boxes.append([int(x_center - width/2), int(y_center - height/2), int(width), int(height)])

        # LOWER NMS THRESHOLD TO 5% AS WELL
        indices = cv2.dnn.NMSBoxes(cv_boxes, conf_filtered.tolist(), 0.05, 0.4)

        targets = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w_box, h_box = cv_boxes[i]
                x2, y2 = x1 + w_box, y1 + h_box
                cx_box = x1 + w_box / 2.0
                cy_box = y1 + h_box / 2.0
                label = class_filtered[i]

                # Draw thin blue outlines for all raw detections
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2) 
                cv2.putText(frame, f"C:{label} W:{w_box}px", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

                # Rubric Filter
                if MIN_SIZE <= w_box <= MAX_SIZE and label == TARGET_CLASS: 
                    # Fill valid targets with solid translucent Green
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1) 
                    
                    x_si = (S_METERS * (cx_box - CX)) / FX 
                    y_si = (S_METERS * (cy_box - CY)) / -FY 
                    yaw_err = math.degrees(math.atan2(x_si - BIAS_X, S_METERS)) 
                    pitch_err = math.degrees(math.atan2(y_si - BIAS_Y, S_METERS)) 
                    
                    targets.append({
                        'yaw': yaw_err, 'pitch': pitch_err, 
                        'x': cx_box, 'y': cy_box,
                        'box': (x1, y1, x2, y2)
                    })

        targets = sorted(targets, key=lambda t: t['x'])

        # ==========================================
        # STATE MACHINE LOGIC
        # ==========================================
        current_time = time.time()

        if tracking_state == 0:
            cv2.putText(frame, "PRE-CALCULATION MODE: WAITING FOR 'P'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            for idx, tgt in enumerate(targets):
                if idx > 1: break # Evaluate max 2 targets
                
                color = (0, 255, 0) if idx == 0 else (255, 0, 255)
                name = "T1 (LEFT)" if idx == 0 else "T2 (RIGHT)"
                x1, y1, x2, y2 = tgt['box']
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                cv2.line(overlay, (int(CX), int(CY)), (int(tgt['x']), int(tgt['y'])), color, 2)
                
                text_y = 60 + (idx * 30)
                cv2.putText(frame, f"{name} -> Yaw: {tgt['yaw']:.1f} | Pitch: {tgt['pitch']:.1f}", 
                            (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        elif tracking_state in [1, 3]:
            tgt_idx = 0 if tracking_state == 1 else -1 
            
            if len(targets) > 0:
                try: tgt = targets[tgt_idx]
                except IndexError: tgt = targets[0] 
                    
                last_known_yaw = tgt['yaw']
                last_known_pitch = tgt['pitch']
                target_lost_time = current_time
                
                # Highlight actively engaged target with translucent Red
                cv2.rectangle(overlay, tgt['box'][0:2], tgt['box'][2:4], (0, 0, 255), -1)
            
            if current_time - last_serial_send > 0.033:
                if current_time - target_lost_time > 1.0:
                    if ser: ser.write(b"TRACK,0.0,0.0\n")
                    cv2.putText(frame, "TARGET LOST!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if ser: ser.write(f"TRACK,{last_known_yaw:.2f},{last_known_pitch:.2f}\n".encode('utf-8'))
                    
                    # Target steady state error criteria
                    if abs(last_known_yaw) <= 0.6 and abs(last_known_pitch) <= 0.6:
                        cv2.putText(frame, "HOLDING STEADY STATE...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        if current_time - lock_start_time > 0.8:
                            print(f"\n[RUBRIC] TGT {1 if tracking_state==1 else 2} Steady State Error: Yaw {last_known_yaw:.2f}, Pitch {last_known_pitch:.2f}") 
                            print(f"[RUBRIC] Commanding {REQUIRED_SHOTS} shots.") 
                            if ser: ser.write(f"FIRE,{REQUIRED_SHOTS}\n".encode('utf-8'))
                            tracking_state += 1 
                    else:
                        cv2.putText(frame, f"ENGAGING TGT {1 if tracking_state==1 else 2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        lock_start_time = current_time
                        
                last_serial_send = current_time

        elif tracking_state in [2, 4]:
            cv2.putText(frame, "FIRING SEQUENCE ACTIVE...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if ser and ser.in_waiting > 0:
                msg = ser.readline().decode('utf-8').strip()
                if "MISSION COMPLETE" in msg:
                    tracking_state += 1
                    lock_start_time = current_time
                    if tracking_state == 5:
                        elapsed = current_time - trial_start_time
                        print(f"\n[RUBRIC] TRIAL COMPLETE! Time: {elapsed:.2f} seconds") 
                    else:
                        print(">> Shifting to Target 2")

        elif tracking_state == 5:
            cv2.putText(frame, "TRIAL COMPLETE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.drawMarker(frame, (int(CX), int(CY)), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.imshow("EW309 Commander", frame)

        key = cv2.waitKey(1)
        if key == ord('p') and tracking_state == 0: 
            trial_start_time = time.time()
            tracking_state = 1
            lock_start_time = time.time()
            target_lost_time = time.time()
            print("\n>> TRIAL INITIATED.")
        if key == ord('q'): 
            break

cv2.destroyAllWindows()
if ser: ser.close()