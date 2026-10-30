from pathlib import Path
import cv2
import time
import joblib
import pandas as pd
import os
import psutil
import csv
from vision_pipeline import VisionPipeline
from desktop_pipeline import DesktopPipeline

# Configuration
MODEL_PATH = Path(__file__).resolve().parents[1] / "focus_real_model.pkl"
POLL_INTERVAL_SEC = 60
SMOOTHING_FACTOR = 0.3
LOG_INTERVAL_SEC = 1.0  # New: Throttle standard logging to 1x per second

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first.")
        return

    print("Loading focus model...")
    model = joblib.load(MODEL_PATH)
    
    print("Initializing sensors...")
    vision = VisionPipeline()
    desktop = DesktopPipeline()
    cap = cv2.VideoCapture(0)
    
    csv_file = open('telemetry_log.csv', mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp", "CPU_Percent", "RAM_Percent", "Loop_Time_ms", "Full_Inference_Triggered"])
    psutil.cpu_percent(interval=None)

    last_poll_time = time.time()
    last_log_time = time.time()  # New: Track when we last wrote to the CSV
    smoothed_score = None

    print(f"Live inference started. Predicting every {POLL_INTERVAL_SEC}s. Press 'q' to quit.\n")

    while cap.isOpened():
        loop_start = time.time()
        full_inference_triggered = False

        ret, frame = cap.read()
        if not ret:
            break

        face_metrics = vision.process_frame(frame)
        
        current_time = time.time()
        time_left = int(POLL_INTERVAL_SEC - (current_time - last_poll_time))
        
        cv2.putText(frame, f"Predicting in: {time_left}s | 'q' to quit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if smoothed_score is not None:
            cv2.putText(frame, f"Focus: {int(smoothed_score)}/100", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Inference", frame)

        if time_left <= 0:
            full_inference_triggered = True
            desktop_metrics = desktop.get_current_metrics()
            
            X_live = [[
                face_metrics['ear'],
                face_metrics['perclos'],
                face_metrics['is_closed'],
                desktop_metrics['active_app_encoded'],
                desktop_metrics['keystrokes'],
                desktop_metrics['mouse_clicks']
            ]]
            
            raw_ml_score = model.predict(pd.DataFrame(X_live, columns=model.feature_names_in_))[0]
            
            base_score = 50.0 
            if face_metrics['is_closed'] == 1 or face_metrics['perclos'] > 0.4:
                base_score = 10.0
            elif desktop_metrics['active_app_encoded'] == 1:
                base_score = 75.0
            else:
                base_score = 40.0
                
            hybrid_score = (base_score * 0.6) + (raw_ml_score * 0.4)
            
            if smoothed_score is None:
                smoothed_score = hybrid_score
            else:
                smoothed_score = (SMOOTHING_FACTOR * hybrid_score) + ((1 - SMOOTHING_FACTOR) * smoothed_score)
            
            print(f"App: {desktop_metrics['active_app_name']} | Keys: {desktop_metrics['keystrokes']} | PERCLOS: {face_metrics['perclos']:.2f}")
            print(f"Logic: {base_score} | ML: {raw_ml_score:.1f} | Final: {smoothed_score:.1f}\n")
            
            last_poll_time = time.time()
            desktop.reset_metrics()

        # --- THROTTLED LOGGING ---
        # Only log if 1 second has passed OR if the heavy ML inference just triggered
        current_time = time.time()
        if full_inference_triggered or (current_time - last_log_time >= LOG_INTERVAL_SEC):
            loop_time_ms = (current_time - loop_start) * 1000
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            
            writer.writerow([current_time, cpu_usage, ram_usage, loop_time_ms, full_inference_triggered])
            last_log_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    csv_file.close()
    desktop.close()
    vision.detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == "__main__":
    main()