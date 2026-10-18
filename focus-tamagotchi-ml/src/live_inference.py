import cv2
import time
import joblib
import numpy as np
import serial
from vision_pipeline import VisionPipeline

class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-2, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 50.0  
        self.error_covariance = 1.0

    def update(self, measurement):
        self.error_covariance += self.process_variance
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance
        return self.estimate

def run_live_inference(model_path="focus_random_forest.pkl", serial_port="COM3", baud_rate=9600, use_serial=False):
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    
    vision = VisionPipeline()
    kf = SimpleKalmanFilter()
    
    ser = None
    if use_serial:
        try:
            ser = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"Connected to STM32 on {serial_port}")
        except Exception as e:
            print(f"Warning: Could not open serial port {serial_port}: {e}. Running without hardware streaming.")
            
    cap = cv2.VideoCapture(0)
    print("Starting live inference loop (Press 'q' to quit)...")
    
    last_inference_time = 0
    inference_interval = 1.0 
    current_focus = 50.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        metrics = vision.process_frame(frame)
        current_time = time.time()
        
        if current_time - last_inference_time >= inference_interval:
            t_struct = time.localtime(current_time)
            time_of_day_frac = (t_struct.tm_hour * 3600 + t_struct.tm_min * 60 + t_struct.tm_sec) / 86400.0
            time_sin = np.sin(2 * np.pi * time_of_day_frac)
            time_cos = np.cos(2 * np.pi * time_of_day_frac)
            
            keystroke_rate = 0.0 
            mouse_speed = 0.0    
            
            features = np.array([[
                metrics['ear'],
                metrics['perclos'],
                keystroke_rate,
                mouse_speed,
                time_sin,
                time_cos
            ]])
            
            raw_pred = model.predict(features)[0]
            current_focus = kf.update(raw_pred)
            last_inference_time = current_time
            
            if ser and ser.is_open:
                msg = f"{int(current_focus)}\n"
                ser.write(msg.encode('utf-8'))
                
        cv2.putText(frame, f"Focus Score: {current_focus:.1f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"EAR: {metrics['ear']:.3f} | PERCLOS: {metrics['perclos']:.3f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    
        cv2.imshow("Live Focus Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()

if __name__ == "__main__":
    run_live_inference()
