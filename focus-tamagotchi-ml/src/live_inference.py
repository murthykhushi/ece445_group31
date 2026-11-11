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
POLL_INTERVAL_SEC = 5
SMOOTHING_FACTOR = 0.3
LOG_INTERVAL_SEC = 5.0          # telemetry rows: once per inference instead of every second
SERIAL_PORT = 'COM3'

# blend between heuristic and ML (ML is what actually uses keystrokes/clicks as learned features)
LOGIC_WEIGHT = 0.4
ML_WEIGHT = 1.0 - LOGIC_WEIGHT

# how often the face landmarker runs in optimized mode (baseline runs on every frame).
# This does NOT affect the 5s scoring interval, PERCLOS is a ratio so it stays accurate.
OPTIMIZED_VISION_FPS = 8


def main():
    parser = argparse.ArgumentParser(description="Live Focus Inference")
    parser.add_argument('--mode', choices=['baseline', 'optimized'], default='optimized',
                        help="Choose 'baseline' (unoptimized CPU) or 'optimized' (GPU/tracking/downscaled)")
    parser.add_argument('--no-display', action='store_true',
                        help="Disable the preview window (lowest CPU, quit with Ctrl+C)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first.")
        return

    print("Loading focus model...")
    model = joblib.load(MODEL_PATH)

    print(f"Initializing sensors in {args.mode.upper()} mode...")
    vision = VisionPipeline(mode=args.mode, perclos_window_sec=POLL_INTERVAL_SEC * 3)
    desktop = DesktopPipeline()
    cap = cv2.VideoCapture(0)

    optimized = (args.mode == "optimized")
    if optimized:
        # lower camera resolution/fps = less data to pull off the camera every frame
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
    frame_interval = (1.0 / OPTIMIZED_VISION_FPS) if optimized else 0.0

    # STM32 Serial Setup (initialized once at startup)
    try:
        stm32_serial = serial.Serial(SERIAL_PORT, 115200, timeout=0.1)
        print("STM32 connected successfully.")
    except serial.SerialException:
        print("Warning: STM32 not connected. Running offline mode.")
        stm32_serial = None

    # name the csv per mode so we dont overwrite the baseline with the optimized run
    csv_filename = f'telemetry_{args.mode}.csv'
    csv_file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    # CPU_Percent = THIS process only, normalized to 0-100 across all cores
    # System_CPU_Percent = whole machine (includes chrome, vscode, etc.)
    # Loop_Time_ms = avg time to process one vision frame since the last log row
    writer.writerow(["Timestamp", "CPU_Percent", "System_CPU_Percent", "RAM_Percent",
                     "Loop_Time_ms", "Full_Inference_Triggered"])

    # prime the cpu counters
    proc = psutil.Process()
    n_cores = psutil.cpu_count() or 1
    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    last_poll_time = time.time()
    last_log_time = time.time()
    last_vision_time = 0.0
    smoothed_score = None
    activity_ema = None
    face_metrics = {"ear": 0.0, "perclos": 0.0, "is_closed": 0}

    # vision timing accumulated between log rows
    vision_time_total_ms = 0.0
    vision_frame_count = 0

    quit_hint = "Ctrl+C to quit" if args.no_display else "'q' to quit"
    print(f"Live inference started. Predicting every {POLL_INTERVAL_SEC}s. "
          f"Saving logs to {csv_filename}. {quit_hint}.\n")

    try:
        while cap.isOpened():
            loop_start = time.time()
            full_inference_triggered = False

            # grab() pulls the next frame off the camera cheaply (no decode/copy),
            # so skipped frames cost almost nothing and the buffer never goes stale
            if not cap.grab():
                print("ERROR: Could not read from the webcam! Check camera permissions.")
                break

            time_left = int(POLL_INTERVAL_SEC - (loop_start - last_poll_time))

            # only decode + run vision when it's due (every frame in baseline)
            if loop_start - last_vision_time >= frame_interval:
                ret, frame = cap.retrieve()
                if not ret:
                    print("ERROR: Could not read from the webcam! Check camera permissions.")
                    break

                t_v = time.time()
                face_metrics = vision.process_frame(frame)
                vision_time_total_ms += (time.time() - t_v) * 1000
                vision_frame_count += 1
                last_vision_time = loop_start

                if not args.no_display:
                    cv2.putText(frame, f"Mode: {args.mode.upper()} | Predicting in: {max(time_left, 0)}s | 'q' to quit",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if smoothed_score is not None:
                        color = (0, 255, 0) if smoothed_score >= 50 else (0, 0, 255)
                        cv2.putText(frame, f"Focus: {int(smoothed_score)}/100",
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imshow("Live Inference", frame)

            # trigger full ml inference only when the poll timer hits 0
            if time_left <= 0:
                full_inference_triggered = True
                desktop_metrics = desktop.get_current_metrics()

                time_multiplier = int(60 / POLL_INTERVAL_SEC)

                # array order MUST match the training csv exactly
                X_live = [[
                    face_metrics['ear'],
                    face_metrics['perclos'],
                    face_metrics['is_closed'],
                    desktop_metrics['active_app_encoded'],
                    desktop_metrics['keystrokes'] * time_multiplier,
                    desktop_metrics['mouse_clicks'] * time_multiplier
                ]]

                # 1. raw prediction from the random forest
                raw_ml_score = model.predict(pd.DataFrame(X_live, columns=model.feature_names_in_))[0]

                # 2. heuristic baseline in case the ml model acts up
                # now activity-aware: keystrokes/clicks in this window decide how high it goes
                # continuous score built from three signals instead of fixed buckets:
                #   app fit x recent input activity, then scaled down by fatigue / face absence

                # (a) app weight: how "work-like" the foreground app is (0-1)
                APP_WEIGHTS = {2: 1.0, 3: 0.9, 1: 0.8, 4: 0.6}   # Code, Terminal, Chrome, Safari
                app_w = APP_WEIGHTS.get(desktop_metrics['active_app_encoded'], 0.4)

                # (b) input activity (0-1), using per-minute rates; ~150 keys/min or ~30 clicks/min = fully active
                keys_pm = desktop_metrics['keystrokes'] * time_multiplier
                clicks_pm = desktop_metrics['mouse_clicks'] * time_multiplier
                activity_now = min(1.0, keys_pm / 150.0 + clicks_pm / 30.0)

                # smooth across polls so a quiet 5s (reading, thinking) doesn't tank the score instantly
                if activity_ema is None:
                    activity_ema = activity_now
                else:
                    activity_ema = 0.5 * activity_now + 0.5 * activity_ema

                engagement = 0.35 * app_w + 0.65 * activity_ema        # 0-1
                base_score = 20.0 + 60.0 * engagement                  # 20-80 range

                # (c) fatigue: smooth penalty as PERCLOS climbs from 0.05 (normal blinking) to 0.40 (drowsy)
                fatigue = min(1.0, max(0.0, (face_metrics['perclos'] - 0.05) / 0.35))
                base_score *= (1.0 - 0.8 * fatigue)

                # (d) no face detected (ear == 0) usually means looking away / not at desk
                if face_metrics['ear'] == 0.0:
                    base_score *= 0.75

                # 3. ensemble: weights set by LOGIC_WEIGHT / ML_WEIGHT at the top
                hybrid_score = (base_score * LOGIC_WEIGHT) + (raw_ml_score * ML_WEIGHT)

                # 4. ema smoothing so the physical tamagotchi doesnt glitch
                if smoothed_score is None:
                    smoothed_score = hybrid_score
                else:
                    smoothed_score = (SMOOTHING_FACTOR * hybrid_score) + ((1 - SMOOTHING_FACTOR) * smoothed_score)

                # 5. binary state to the STM32 to actuate the physical tail
                focus_state = 1 if smoothed_score >= 50 else 0
                if stm32_serial:
                    stm32_serial.write(f"{focus_state}\n".encode())
                    stm32_serial.flush()

                # single-line console output per score
                print(f"[{desktop_metrics['active_app_name']}] keys={desktop_metrics['keystrokes']} "
                      f"perclos={face_metrics['perclos']:.2f} | logic={base_score:.0f} ml={raw_ml_score:.1f} "
                      f"final={smoothed_score:.1f} -> STM32={focus_state}")

                # reset polling timer and flush the desktop event buffer
                last_poll_time = time.time()
                desktop.reset_metrics()

            # telemetry: one row per LOG_INTERVAL_SEC (aligned with inference)
            current_time = time.time()
            if full_inference_triggered or (current_time - last_log_time >= LOG_INTERVAL_SEC):
                avg_vision_ms = (vision_time_total_ms / vision_frame_count) if vision_frame_count else 0.0
                proc_cpu = proc.cpu_percent(interval=None) / n_cores
                sys_cpu = psutil.cpu_percent(interval=None)
                ram_usage = psutil.virtual_memory().percent

                writer.writerow([current_time, proc_cpu, sys_cpu, ram_usage,
                                 avg_vision_ms, full_inference_triggered])
                csv_file.flush()

                vision_time_total_ms = 0.0
                vision_frame_count = 0
                last_log_time = current_time

            # exit condition (waitKey also pumps the preview window)
            if not args.no_display:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # cleanup everything nicely
        csv_file.close()
        desktop.close()

        try:
            vision.detector.close()
        except AttributeError:
            pass

        cap.release()
        cv2.destroyAllWindows()

        if stm32_serial:
            stm32_serial.close()

        print("Inference stopped.")


if __name__ == "__main__":
    main()