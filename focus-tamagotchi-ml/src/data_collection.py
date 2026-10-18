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
    
    # File to save our data
    csv_filename = "real_training_data.csv"
    
    # Create the CSV with headers if it doesn't exist yet
    if not os.path.exists(csv_filename):
        df = pd.DataFrame(columns=[
            'ear', 'perclos', 'is_closed', 'active_app', 
            'keystrokes', 'mouse_clicks', 'focus_score'
        ])
        df.to_csv(csv_filename, index=False)
        print(f"Created {csv_filename}")

    print("\n--- DATA COLLECTION STARTED ---")
    print("1. Leave this window open and go do some work.")
    print("2. Click on the camera window and press 's' to save a data sample.")
    print("3. Press 'q' to quit and stop collecting.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Continuously process the webcam so PERCLOS history builds up
        face_metrics = vision.process_frame(frame)
        
        # Display the camera feed
        cv2.putText(frame, "Press 's' to record data | 'q' to quit", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Data Collection Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # If user presses 's', capture the data!
        if key == ord('s'):
            desktop_metrics = desktop.get_current_metrics()
            
            # Prompt the user in the terminal for their focus score
            print("\n>>> DATA POINT CAPTURED! <<<")
            print(f"App: {desktop_metrics['active_app_name']} | Keys: {desktop_metrics['keystrokes']} | EAR: {face_metrics['ear']:.2f}")
            
            try:
                score = input("How focused are you right now? (0-100): ")
                score = float(score)
                
                # Create a new row of data
                new_data = {
                    'ear': face_metrics['ear'],
                    'perclos': face_metrics['perclos'],
                    'is_closed': face_metrics['is_closed'],
                    'active_app': desktop_metrics['active_app_encoded'],
                    'keystrokes': desktop_metrics['keystrokes'],
                    'mouse_clicks': desktop_metrics['mouse_clicks'],
                    'focus_score': score
                }
                
                # Append it to the CSV
                new_df = pd.DataFrame([new_data])
                new_df.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Saved successfully to {csv_filename}!\n")
                
            except ValueError:
                print("Invalid input. Data point skipped. Please enter a number next time.\n")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection stopped.")

if __name__ == "__main__":
    collect_data()