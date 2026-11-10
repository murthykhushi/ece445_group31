"""
Live Focus Inference & Hardware Telemetry (live_inference_test.py)

Capabilities:
- Computes an EMA-smoothed focus score (0-100) every 5s by blending a Random Forest 
  model (40%) with a heuristic baseline (60%) using webcam (EAR/PERCLOS) and OS inputs.
- Classifies scores into 4 hysteresis-buffered behavioral states:
  [0] Highly Focused (>=80) | [1] Focused (60-79) | [2] Distracted (30-59) | [3] Highly Distracted (<30)
- Bi-directional UART: Streams `$GF,<state>,<score>*<checksum>\n` packets to the MCU and
  ingests upstream `EVENT:PET_BOOST:<pts>` ultrasonic petting events to reward focus score.
- Logs CPU/RAM/latency metrics every 1s to `telemetry_<mode>.csv`.

Usage:
  python live_inference_test.py [--mode {optimized,baseline}] [--port PORT]

Examples:
  python live_inference_test.py
  python live_inference_test.py --mode baseline --port COM4
  python live_inference_test.py --port /dev/cu.usbserial-0001

OpenCV Window Hotkeys:
  [1] Force State 0 (Score: 92)   [3] Force State 2 (Score: 45)
  [2] Force State 1 (Score: 72)   [4] Force State 3 (Score: 15)
  [q] Quit and safely close serial/CSV logs
"""

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

# Configuration
MODEL_PATH = Path(__file__).resolve().parents[1] / "focus_real_model.pkl"
POLL_INTERVAL_SEC = 5
SMOOTHING_FACTOR = 0.30  # Proposal Eq (5): alpha = 0.30
LOG_INTERVAL_SEC = 1.0

# Serial Configuration
SERIAL_PORT = "COM3"     # Update to your ESP32/STM32 port (e.g., "COM3" or "/dev/cu.usbserial-0001")
BAUD_RATE = 115200

# Four Behavioral States
STATE_HIGHLY_FOCUSED = 0     # S >= 80
STATE_FOCUSED = 1            # 60 <= S < 80
STATE_DISTRACTED = 2         # 30 <= S < 60
STATE_HIGHLY_DISTRACTED = 3  # S < 30

STATE_NAMES = {
    STATE_HIGHLY_FOCUSED: "Highly Focused (^ w ^)",
    STATE_FOCUSED: "Focused (O w O)",
    STATE_DISTRACTED: "Distracted (- _ -)",
    STATE_HIGHLY_DISTRACTED: "Highly Distracted (> n <)",
}


def classify_with_hysteresis(score: float, prev_state: int) -> int:
    """
    Implements Behavioral Classification with a 5-point hysteresis margin
    to prevent rapid state flipping when the score hovers near 80, 60, or 30.
    """
    if prev_state == STATE_HIGHLY_FOCUSED:
        if score >= 75:
            return STATE_HIGHLY_FOCUSED
        elif score >= 60:
            return STATE_FOCUSED
        elif score >= 30:
            return STATE_DISTRACTED
        return STATE_HIGHLY_DISTRACTED

    elif prev_state == STATE_FOCUSED:
        if score >= 80:
            return STATE_HIGHLY_FOCUSED
        elif score >= 55:
            return STATE_FOCUSED
        elif score >= 30:
            return STATE_DISTRACTED
        return STATE_HIGHLY_DISTRACTED

    elif prev_state == STATE_DISTRACTED:
        if score >= 80:
            return STATE_HIGHLY_FOCUSED
        elif score >= 60:
            return STATE_FOCUSED
        elif score >= 25:
            return STATE_DISTRACTED
        return STATE_HIGHLY_DISTRACTED

    else:  # STATE_HIGHLY_DISTRACTED
        if score >= 80:
            return STATE_HIGHLY_FOCUSED
        elif score >= 60:
            return STATE_FOCUSED
        elif score >= 35:
            return STATE_DISTRACTED
        return STATE_HIGHLY_DISTRACTED


class TamagotchiSerial:
    """Bi-directional UART manager for $GF packets and upstream HC-SR04 petting events."""
    def __init__(self, port: str, baud: int):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            print(f"[Serial] Connected to {port}. Waiting 2s for board boot...")
            time.sleep(2.0)
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            print(f"[Serial Warning] Could not open {port}: {e}")
            print("[Serial Warning] Running in offline/host-only mode.\n")

    def poll_incoming_events(self) -> int:
        """Non-blockingly reads any upstream messages from ESP32 and returns total pet boost points."""
        boost_pts = 0
        if not (self.ser and self.ser.is_open):
            return 0
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("EVENT:PET_BOOST:"):
                    pts = int(line.split(":")[-1])
                    boost_pts += pts
                    print(f"  <- [ESP32 Sensor] Hand Petting Detected! Boosting Focus Score by +{pts} pts")
                elif line.startswith("EVENT:"):
                    print(f"  <- [ESP32 Event] {line}")
                elif line.startswith("ACK:") or line.startswith("NACK:"):
                    print(f"  -> [Board] {line}")
        except (serial.SerialException, ValueError) as e:
            print(f"[Serial Read Warning] {e}")
        return boost_pts

    def send_packet(self, state_id: int, score: float):
        clamped_score = max(0, min(100, int(round(score))))
        payload = f"GF,{state_id},{clamped_score}"
        
        checksum = 0
        for ch in payload:
            checksum ^= ord(ch)
        
        packet = f"${payload}*{checksum:02X}\n"
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(packet.encode("utf-8"))
                self.ser.flush()
            except serial.SerialException as e:
                print(f"[Serial Error] {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.send_packet(STATE_FOCUSED, 70)
            time.sleep(0.1)
            self.ser.close()


def main():
    parser = argparse.ArgumentParser(description="Live Focus Inference")
    parser.add_argument('--mode', choices=['baseline', 'optimized'], default='optimized',
                        help="Choose 'baseline' (CPU only) or 'optimized' (GPU accelerated)")
    parser.add_argument('--port', default=SERIAL_PORT,
                        help=f"Serial port for ESP32/STM32 (default: {SERIAL_PORT})")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first.")
        return

    print("Loading focus model...")
    model = joblib.load(MODEL_PATH)

    print(f"Initializing sensors in {args.mode.upper()} mode...")
    vision = VisionPipeline(mode=args.mode, perclos_window_sec=POLL_INTERVAL_SEC * 3)
    desktop = DesktopPipeline()
    companion = TamagotchiSerial(args.port, BAUD_RATE)
    cap = cv2.VideoCapture(0)

    csv_filename = f'telemetry_{args.mode}.csv'
    csv_file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp", "CPU_Percent", "RAM_Percent", "Loop_Time_ms", "Full_Inference_Triggered"])

    psutil.cpu_percent(interval=None)

    last_poll_time = time.time()
    last_log_time = time.time()
    smoothed_score = 70.0
    current_state = STATE_FOCUSED

    print(f"Live inference started. Predicting every {POLL_INTERVAL_SEC}s. Saving logs to {csv_filename}.")
    print("Demo Hotkeys: [1] Highly Focused | [2] Focused | [3] Distracted | [4] Highly Distracted | [q] Quit\n")

    while cap.isOpened():
        loop_start = time.time()
        full_inference_triggered = False

        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read from the webcam! Check your Camera permissions in System Settings.")
            break

        # Non-blockingly poll ESP32 for upstream ultrasonic petting score boosts
        pet_boost = companion.poll_incoming_events()
        if pet_boost > 0:
            smoothed_score = min(100.0, smoothed_score + pet_boost)
            current_state = classify_with_hysteresis(smoothed_score, current_state)
            companion.send_packet(current_state, smoothed_score)

        # Process vision every frame so PERCLOS rolling history is accurate
        face_metrics = vision.process_frame(frame)

        current_time = time.time()
        time_left = int(POLL_INTERVAL_SEC - (current_time - last_poll_time))

        # Update UI overlay on video feed
        cv2.putText(frame, f"Mode: {args.mode.upper()} | Predicting in: {max(0, time_left)}s | 'q' to quit",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, "Hotkeys: [1] >80 | [2] 60-80 | [3] 30-60 | [4] <30",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if smoothed_score is not None:
            if smoothed_score >= 60:
                color = (0, 255, 0)      # Green (Focused / Highly Focused)
            elif smoothed_score >= 30:
                color = (0, 255, 255)    # Yellow (Distracted)
            else:
                color = (0, 0, 255)      # Red (Highly Distracted)

            cv2.putText(frame, f"Focus: {int(smoothed_score)}/100 | {STATE_NAMES[current_state]}",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Group 31 - Live Focus Inference", frame)

        # Demo Hotkeys for instant verification
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            preset_scores = {ord('1'): 92.0, ord('2'): 72.0, ord('3'): 45.0, ord('4'): 15.0}
            smoothed_score = preset_scores[key]
            current_state = classify_with_hysteresis(smoothed_score, current_state)
            print(f"\n[DEMO OVERRIDE] State: {STATE_NAMES[current_state]} | Score: {smoothed_score:.0f}")
            companion.send_packet(current_state, smoothed_score)
            last_poll_time = time.time()

        # Trigger full ML inference when the polling timer hits 0
        if time_left <= 0:
            full_inference_triggered = True
            desktop_metrics = desktop.get_current_metrics()

            time_multiplier = int(60 / POLL_INTERVAL_SEC)

            X_live = [[
                face_metrics['ear'],
                face_metrics['perclos'],
                face_metrics['is_closed'],
                desktop_metrics['active_app_encoded'],
                desktop_metrics['keystrokes'] * time_multiplier,
                desktop_metrics['mouse_clicks'] * time_multiplier
            ]]

            raw_ml_score = model.predict(pd.DataFrame(X_live, columns=model.feature_names_in_))[0]

            base_score = 50.0
            if face_metrics['is_closed'] == 1 or face_metrics['perclos'] > 0.4:
                base_score = 10.0
            elif desktop_metrics['active_app_encoded'] in [1, 2]:
                base_score = 75.0
            else:
                base_score = 40.0

            hybrid_score = (base_score * 0.6) + (raw_ml_score * 0.4)

            if smoothed_score is None:
                smoothed_score = hybrid_score
            else:
                smoothed_score = (SMOOTHING_FACTOR * hybrid_score) + ((1.0 - SMOOTHING_FACTOR) * smoothed_score)

            current_state = classify_with_hysteresis(smoothed_score, current_state)
            companion.send_packet(current_state, smoothed_score)

            print(f"App: {desktop_metrics['active_app_name']} | Keys: {desktop_metrics['keystrokes']} (x{time_multiplier}) | PERCLOS: {face_metrics['perclos']:.2f}")
            print(f"Logic: {base_score} | ML: {raw_ml_score:.1f} | Final: {smoothed_score:.1f} | State: {STATE_NAMES[current_state]}\n")

            last_poll_time = time.time()
            desktop.reset_metrics()

        current_time = time.time()

        # Log telemetry every 1 second OR if we just did a full inference
        if full_inference_triggered or (current_time - last_log_time >= LOG_INTERVAL_SEC):
            loop_time_ms = (current_time - loop_start) * 1000
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent

            writer.writerow([current_time, cpu_usage, ram_usage, loop_time_ms, full_inference_triggered])
            last_log_time = current_time

    # Cleanup
    csv_file.close()
    companion.close()
    desktop.close()

    try:
        vision.detector.close()
    except AttributeError:
        pass

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == "__main__":
    main()