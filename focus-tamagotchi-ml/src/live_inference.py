from pathlib import Path
import cv2
import time
import joblib
import pandas as pd
import os
import psutil
import csv
import argparse
import serial

from vision_pipeline import VisionPipeline
from desktop_pipeline import DesktopPipeline

# config
MODEL_PATH = Path(__file__).resolve().parents[1] / "focus_real_model.pkl"
POLL_INTERVAL_SEC = 60
SMOOTHING_FACTOR = 0.3
LOG_INTERVAL_SEC = 1.0

def main():
    # setup args for the A/B test so we can run cpu or gpu from terminal
    parser = argparse.ArgumentParser(description="Live Focus Inference")
    parser.add_argument('--mode', choices=['baseline', 'optimized'], default='optimized', 
                        help="Choose 'baseline' (CPU only) or 'optimized' (GPU accelerated)")
    args = parser.parse_args()

    # check if we actually have the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first.")
        return

    print("Loading focus model...")
    model = joblib.load(MODEL_PATH)
    
    print(f"Initializing sensors in {args.mode.upper()} mode...")
    # pass the mode to vision pipeline so mediapipe knows what to use
    vision = VisionPipeline(mode=args.mode) 
    desktop = DesktopPipeline()
    cap = cv2.VideoCapture(0)
    
    # STM32 Serial Setup (Initialized once at startup)
    try:
        stm32_serial = serial.Serial('/dev/tty.usbmodem1101', 115200, timeout=0.1)
        print("STM32 connected successfully.")
    except serial.SerialException:
        print("Warning: STM32 not connected. Running offline mode.")
        stm32_serial = None
    
    # dynamically name the csv so we dont overwrite the baseline with the optimized run
    csv_filename = f'telemetry_{args.mode}.csv'
    csv_file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp", "CPU_Percent", "RAM_Percent", "Loop_Time_ms", "Full_Inference_Triggered"])
    
    # kick off the cpu monitor
    psutil.cpu_percent(interval=None)

    last_poll_time = time.time()
    last_log_time = time.time()
    smoothed_score = None

    print(f"Live inference started. Predicting every {POLL_INTERVAL_SEC}s. Saving logs to {csv_filename}. Press 'q' to quit.\n")

    while cap.isOpened():
        loop_start = time.time()
        full_inference_triggered = False

        ret, frame = cap.read()
        if not ret:
            # macs will silently fail if terminal doesnt have camera permissions, so print an error
            print("ERROR: Could not read from the webcam! Check your Mac's Camera permissions in System Settings.")
            break

        # process vision every single frame so our perclos rolling history is accurate
        face_metrics = vision.process_frame(frame)
        
        current_time = time.time()
        time_left = int(POLL_INTERVAL_SEC - (current_time - last_poll_time))
        
        # update the ui overlay on the video feed
        cv2.putText(frame, f"Mode: {args.mode.upper()} | Predicting in: {time_left}s | 'q' to quit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if smoothed_score is not None:
            # Color code the score text
            color = (0, 255, 0) if smoothed_score >= 50 else (0, 0, 255)
            cv2.putText(frame, f"Focus: {int(smoothed_score)}/100", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Live Inference", frame)

        # trigger full ml inference only when the 60s timer hits 0
        if time_left <= 0:
            full_inference_triggered = True
            desktop_metrics = desktop.get_current_metrics()
            
            # bundle current metrics. array order MUST match the training csv exactly
            X_live = [[
                face_metrics['ear'],
                face_metrics['perclos'],
                face_metrics['is_closed'],
                desktop_metrics['active_app_encoded'],
                desktop_metrics['keystrokes'],
                desktop_metrics['mouse_clicks']
            ]]
            
            # 1. get raw prediction from our random forest model
            raw_ml_score = model.predict(pd.DataFrame(X_live, columns=model.feature_names_in_))[0]
            
            # 2. calculate a heuristic baseline just in case the ml model acts up
            base_score = 50.0 
            if face_metrics['is_closed'] == 1 or face_metrics['perclos'] > 0.4:
                base_score = 10.0  # heavy penalty for closed eyes/fatigue
            elif desktop_metrics['active_app_encoded'] == 1:
                base_score = 75.0  # reward for being in a productive app like vscode
            else:
                base_score = 40.0
                
            # 3. ensemble: blend ml with baseline (60% logic / 40% ml) to balance it out
            hybrid_score = (base_score * 0.6) + (raw_ml_score * 0.4)
            
            # 4. apply ema smoothing so the physical tamagotchi doesnt glitch back and forth
            if smoothed_score is None:
                smoothed_score = hybrid_score
            else:
                smoothed_score = (SMOOTHING_FACTOR * hybrid_score) + ((1 - SMOOTHING_FACTOR) * smoothed_score)
            
            # 5. Send binary state to the STM32 to actuate the physical tail
            focus_state = 1 if smoothed_score >= 50 else 0
            if stm32_serial:
                stm32_serial.write(f"{focus_state}\n".encode())

            print(f"App: {desktop_metrics['active_app_name']} | Keys: {desktop_metrics['keystrokes']} | PERCLOS: {face_metrics['perclos']:.2f}")
            print(f"Logic: {base_score} | ML: {raw_ml_score:.1f} | Final: {smoothed_score:.1f} | Sent to STM32: {focus_state}\n")
            
            # reset polling timer and flush the 60s desktop event buffer so we start fresh
            last_poll_time = time.time()
            desktop.reset_metrics()

        current_time = time.time()
        
        # log telemetry every 1 second OR if we just did a full inference
        if full_inference_triggered or (current_time - last_log_time >= LOG_INTERVAL_SEC):
            loop_time_ms = (current_time - loop_start) * 1000
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            
            writer.writerow([current_time, cpu_usage, ram_usage, loop_time_ms, full_inference_triggered])
            last_log_time = current_time

        # exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup everything nicely
    csv_file.close()
    desktop.close()
    
    try:
        vision.detector.close()
    except AttributeError:
        pass # Safely ignore if the custom vision class handles its own cleanup

    cap.release()
    cv2.destroyAllWindows()
    
    if stm32_serial:
        stm32_serial.close()
        
    print("Inference stopped.")

if __name__ == "__main__":
    main()