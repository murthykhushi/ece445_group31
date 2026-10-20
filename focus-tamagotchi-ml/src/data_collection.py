import cv2
import time
import pandas as pd
import os
from vision_pipeline import VisionPipeline
from desktop_pipeline import DesktopPipeline

def collect_data():
    # 
    print("Initializing sensors...")
    vision = VisionPipeline()
    desktop = DesktopPipeline()
    
    cap = cv2.VideoCapture(0)
    csv_filename = "real_training_data.csv"
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_filename):
        df = pd.DataFrame(columns=[
            'ear', 'perclos', 'is_closed', 'active_app', 
            'keystrokes', 'mouse_clicks', 'focus_score'
        ])
        df.to_csv(csv_filename, index=False)
        print(f"Created {csv_filename}")

    # Every 60 seconds prompt user for their own focus score 
    PROMPT_INTERVAL_SEC = 60 
    last_prompt_time = time.time()

    print(f"Collection started. Auto-prompting every {PROMPT_INTERVAL_SEC}s. Press 'q' to quit.\n")

    while cap.isOpened():

        # continous while loop to keep gathering data 
        ret, frame = cap.read()
        if not ret:
            break

        # Call vision pipeline to get the proper eye aspect ratio (EAR), other blink (PERCLOS) & facial data
        face_metrics = vision.process_frame(frame)
        
        current_time = time.time()
        time_left = int(PROMPT_INTERVAL_SEC - (current_time - last_prompt_time))
        
        cv2.putText(frame, f"Next prompt in: {time_left}s | 'q' to quit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Data Collection Mode", frame)

        key = cv2.waitKey(1) & 0xFF

        # Can capture at the end of 60s or if interupt of s 
        if (time_left <= 0) or (key == ord('s')):
            desktop_metrics = desktop.get_current_metrics()
            
            print('\a')  # System beep
            print(f"\n--- FOCUS CHECK ---")
            print(f"App: {desktop_metrics['active_app_name']} | Keys: {desktop_metrics['keystrokes']} | EAR: {face_metrics['ear']:.2f}")
            
            try:
                score = float(input("Focus score (0-100): "))
                
                new_data = {
                    'ear': face_metrics['ear'],
                    'perclos': face_metrics['perclos'],
                    'is_closed': face_metrics['is_closed'],
                    'active_app': desktop_metrics['active_app_encoded'],
                    'keystrokes': desktop_metrics['keystrokes'],
                    'mouse_clicks': desktop_metrics['mouse_clicks'],
                    'focus_score': score
                }
                
                pd.DataFrame([new_data]).to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Saved to {csv_filename}\n")
                
            except ValueError:
                print("Invalid input. Skipped.\n")
            
            last_prompt_time = time.time()

        # input key break 
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collection stopped.")

if __name__ == "__main__":
    # Continously grabs data until user quits 
    collect_data()