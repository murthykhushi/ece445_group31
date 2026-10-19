import cv2
import time
import joblib
import numpy as np
import os
from vision_pipeline import VisionPipeline
from desktop_pipeline import DesktopPipeline

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MODEL_PATH = "focus_real_model.pkl"
POLL_INTERVAL_SEC = 60

# Smoothing Filter Setup
smoothed_score = None
SMOOTHING_FACTOR = 0.3  # Adjust between 0.1 (very smooth/slow) and 1.0 (instant/no smoothing)

def main():
    global smoothed_score

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Could not find {MODEL_PATH}.")
        print("Please run 'python src/train_real.py' first!")
        return

    print("Loading personalized focus model...")
    model = joblib.load(MODEL_PATH)
    
    print("Initializing Sensors...")
    vision = VisionPipeline()
    desktop = DesktopPipeline()
    
    cap = cv2.VideoCapture(0)
    last_poll_time = time.time()

    print("\n" + "="*50)
    print("🧠 HYBRID LIVE INFERENCE STARTED")
    print(f"Predicting focus every {POLL_INTERVAL_SEC} seconds.")
    print("Press 'q' in the video window to quit.")
    print("="*50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process vision continuously to maintain accurate PERCLOS history
        face_metrics = vision.process_frame(frame)
        
        current_time = time.time()
        time_left = int(POLL_INTERVAL_SEC - (current_time - last_poll_time))
        
        # UI Overlay
        cv2.putText(frame, f"Predicting in: {time_left}s | 'q' to quit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if smoothed_score is not None:
            cv2.putText(frame, f"Current Focus: {int(smoothed_score)}/100", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Inference Mode", frame)

        # ==========================================
        # 2. INFERENCE LOOP
        # ==========================================
        if time_left <= 0:
            desktop_metrics = desktop.get_current_metrics()
            
            # Bundle metrics for the ML Model
            X_live = [[
                face_metrics['ear'],
                face_metrics['perclos'],
                face_metrics['is_closed'],
                desktop_metrics['active_app_encoded'],
                desktop_metrics['keystrokes'],
                desktop_metrics['mouse_clicks']
            ]]
            
            # --- HYBRID SCORING SYSTEM ---
            
            # A. Get the pure ML prediction
            raw_ml_score = model.predict(X_live)[0]
            
            # B. Calculate a Logical Base Score
            base_score = 50.0 
            
            # Severe distraction/fatigue penalty
            if face_metrics['is_closed'] == 1 or face_metrics['perclos'] > 0.4:
                base_score = 10.0
            # Reward for productive app (Assume 1 is a coding IDE/productive app. Adjust to match your encoder)
            elif desktop_metrics['active_app_encoded'] == 1:
                base_score = 75.0
            else:
                base_score = 40.0
                
            # C. Blend ML and Logic (60% Logic, 40% ML)
            hybrid_score = (base_score * 0.6) + (raw_ml_score * 0.4)
            
            # D. Apply Kalman/Smoothing filter to prevent rapid UI jumping
            if smoothed_score is None:
                smoothed_score = hybrid_score
            else:
                smoothed_score = (SMOOTHING_FACTOR * hybrid_score) + ((1 - SMOOTHING_FACTOR) * smoothed_score)
            
            # --- TERMINAL OUTPUT ---
            print(f"App: {desktop_metrics['active_app_name']} | Keys: {desktop_metrics['keystrokes']} | PERCLOS: {face_metrics['perclos']:.2f}")
            print(f"Logic: {base_score} | ML: {raw_ml_score:.1f} | Final Smoothed: {smoothed_score:.1f}\n")
            
            # ==========================================
            # 3. HARDWARE HAND-OFF (STM32)
            # ==========================================
            # TODO: Add your pyserial code here to send int(smoothed_score) to hardware
            
            # Reset timer and background metrics
            last_poll_time = time.time()
            desktop.reset_metrics()

        # Quit condition (must have video window selected)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live inference stopped.")

if __name__ == "__main__":
    main()