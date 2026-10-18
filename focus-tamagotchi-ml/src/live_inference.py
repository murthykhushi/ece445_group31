import cv2
import serial
import joblib
import time
import warnings
import numpy as np
from vision_pipeline import VisionPipeline

# Suppress feature name warnings from scikit-learn for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def run_live_inference():
    print("Loading trained model from focus_random_forest.pkl...")
    try:
        with open("focus_random_forest.pkl", "rb") as f:
            model = joblib.load("focus_random_forest.pkl")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the vision pipeline (matches Line 26 in your traceback)
    vision = VisionPipeline()
    
    # Initialize serial connection
    # Note: Update the port string (e.g., '/dev/cu.usbmodem...') to match your Arduino/Tamagotchi
    try:
        ser = serial.Serial('/dev/cu.usbmodem14101', 9600, timeout=1) 
    except Exception as e:
        print(f"Warning: Serial connection not found. Running in video-only mode.")
        ser = None

    cap = cv2.VideoCapture(0)
    print("Starting Live Inference... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to get the dictionary of metrics
        metrics = vision.process_frame(frame)
        
        # --- THE FIX ---
        # Scikit-learn expects a 2D array (list of lists) with exactly 3 features.
        # We extract just the 3 vision metrics and ignore any extra variables.
        features = [[
            metrics.get('ear', 0.0), 
            metrics.get('perclos', 0.0), 
            metrics.get('is_closed', 0)
        ]]

        # Run prediction (matches Line 70 in your traceback)
        raw_pred = model.predict(features)[0]

        # Display the result on the video feed
        status_text = f"Focus Score: {raw_pred:.2f} | EAR: {metrics['ear']:.2f}"
        cv2.putText(frame, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Focus Tamagotchi - Live Inference", frame)

        # Send prediction to the Tamagotchi hardware via Serial
        if ser:
            # Format the output for the Arduino (adjust string format as needed)
            ser.write(f"{raw_pred:.2f}\n".encode('utf-8'))

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    # (Matches Line 93 in your traceback)
    run_live_inference()