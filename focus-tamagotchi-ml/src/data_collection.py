import cv2
import time
import pandas as pd
import os
from vision_pipeline import VisionPipeline
from desktop_pipeline import DesktopPipeline

def collect_data():
    print("Initializing Sensors...")
    vision = VisionPipeline()
    desktop = DesktopPipeline()
    
    cap = cv2.VideoCapture(0)
    
    csv_filename = "real_training_data.csv"
    
    if not os.path.exists(csv_filename):
        df = pd.DataFrame(columns=[
            'ear', 'perclos', 'is_closed', 'active_app', 
            'keystrokes', 'mouse_clicks', 'focus_score'
        ])
        df.to_csv(csv_filename, index=False)
        print(f"Created {csv_filename}")

    # ==========================================
    # SET YOUR INTERVAL HERE
    # 60 seconds = 1 minute (Perfect for testing!)
    # ==========================================
    PROMPT_INTERVAL_SEC = 60 
    last_prompt_time = time.time()

    print("\n--- AUTOMATIC DATA COLLECTION STARTED ---")
    print(f"1. The script will automatically prompt you every {PROMPT_INTERVAL_SEC} seconds.")
    print("2. Leave this window open in the background and do your work.")
    print("3. You DO NOT need to press 's'. Just wait for the countdown!")
    print("4. Press 'q' to quit and stop collecting.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_metrics = vision.process_frame(frame)
        
        # Calculate how much time is left in the loop
        current_time = time.time()
        time_left = int(PROMPT_INTERVAL_SEC - (current_time - last_prompt_time))
        
        # Display a live countdown on the camera feed
        cv2.putText(frame, f"Next prompt in: {time_left}s | 'q' to quit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Data Collection Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Trigger AUTOMATICALLY if the countdown hits 0 (or if you manually force it with 's')
        if (time_left <= 0) or (key == ord('s')):
            desktop_metrics = desktop.get_current_metrics()
            
            # \a creates a system "beep" sound to get your attention!
            print('\a') 
            print("\n" + "="*40)
            print(">>> AUTOMATIC FOCUS CHECK! <<<")
            print(f"App: {desktop_metrics['active_app_name']} | Keys: {desktop_metrics['keystrokes']} | EAR: {face_metrics['ear']:.2f}")
            
            try:
                # The script will pause right here and wait for you
                score = input("How focused are you right now? (0-100): ")
                score = float(score)
                
                new_data = {
                    'ear': face_metrics['ear'],
                    'perclos': face_metrics['perclos'],
                    'is_closed': face_metrics['is_closed'],
                    'active_app': desktop_metrics['active_app_encoded'],
                    'keystrokes': desktop_metrics['keystrokes'],
                    'mouse_clicks': desktop_metrics['mouse_clicks'],
                    'focus_score': score
                }
                
                new_df = pd.DataFrame([new_data])
                new_df.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Saved successfully to {csv_filename}!\n")
                
            except ValueError:
                print("Invalid input. Data point skipped.\n")
            
            # Reset the timer AFTER you type your answer so it starts a fresh countdown
            last_prompt_time = time.time()
            print("="*40 + "\n")
            print(f"Resuming background monitoring... Next prompt in {PROMPT_INTERVAL_SEC} seconds.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection stopped.")

if __name__ == "__main__":
    collect_data()