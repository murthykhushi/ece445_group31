import cv2
import serial
import joblib
import time
import warnings
from vision_pipeline import VisionPipeline
from desktop_pipeline import DesktopPipeline

# Suppress feature name warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

def run_live_inference():
    print("Loading trained model from focus_random_forest.pkl...")
    try:
        model = joblib.load("focus_random_forest.pkl")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize both pipelines
    vision = VisionPipeline()
    desktop = DesktopPipeline()
    
    # Initialize serial connection to Arduino/Tamagotchi
    try:
        ser = serial.Serial('/dev/cu.usbmodem14101', 9600, timeout=1) 
    except Exception as e:
        print(f"Warning: Serial connection not found. Running in UI-only mode.")
        ser = None

    cap = cv2.VideoCapture(0)
    print("Starting Live Inference... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Get metrics from both sensors
        face_metrics = vision.process_frame(frame)
        desktop_metrics = desktop.get_current_metrics()
        
        # 2. Combine into a single 6-feature array (MATCH YOUR TRAINING DATA ORDER!)
        features = [[
            face_metrics.get('ear', 0.0), 
            face_metrics.get('perclos', 0.0), 
            face_metrics.get('is_closed', 0),
            desktop_metrics.get('active_app_encoded', 0),
            desktop_metrics.get('keystrokes', 0),
            desktop_metrics.get('mouse_clicks', 0)
        ]]

        # 3. Predict focus score
        try:
            raw_pred = model.predict(features)[0]
        except ValueError as e:
            print(f"Prediction Error: {e}")
            raw_pred = 0.0

        # 4. Display on screen
        status_text = f"Focus: {raw_pred:.2f} | App: {desktop_metrics['active_app_name']}"
        input_text = f"Keys: {desktop_metrics['keystrokes']} | Clicks: {desktop_metrics['mouse_clicks']}"
        
        cv2.putText(frame, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, input_text, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Focus Tamagotchi - Live Inference", frame)

        # 5. Send to hardware
        if ser:
            ser.write(f"{raw_pred:.2f}\n".encode('utf-8'))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    run_live_inference()