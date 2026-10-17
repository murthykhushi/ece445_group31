import pandas as pd; import time
import os
import time
import subprocess
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
from pynput import keyboard, mouse

# Configuration
WINDOW_SIZE = 2.0  # 2.0 second sliding window
PROMPT_INTERVAL = 900  # 15 minutes
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

# Globals
key_clicks = 0
mouse_clicks = 0
telemetry_buffer = []

def on_press(key): 
    global key_clicks
    key_clicks += 1

def on_click(x, y, button, pressed): 
    global mouse_clicks
    mouse_clicks += int(pressed)

def get_active_window_mac():
    try:
        # AppleScript to get the active foreground application name
        script = 'tell application "System Events" to get name of first application process whose frontmost is true'
        return subprocess.check_output(['osascript', '-e', script]).decode('utf-8').strip()
    except Exception:
        return "Unknown"

def prompt_user_for_label():
    root = tk.Tk()
    root.withdraw()
    # Force the prompt to the front on macOS
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
    score = simpledialog.askinteger("Focus Tracker", "Rate your focus over the last 15 mins (1-10):", minvalue=1, maxvalue=10)
    root.destroy()
    return score

def main():
    global key_clicks, mouse_clicks, telemetry_buffer
    print(f"Starting logger. Data will be saved to {DATA_DIR}")
    print("Note: macOS will likely prompt you to grant 'Accessibility' permissions to Terminal/Python to track keystrokes.")

    # Start peripheral listeners
    k_listener = keyboard.Listener(on_press=on_press)
    m_listener = mouse.Listener(on_click=on_click)
    k_listener.start()
    m_listener.start()

    last_prompt_time = time.time()

    try:
        while True:
            time.sleep(WINDOW_SIZE)
            current_time = time.time()

            app_name = get_active_window_mac()

            # Store telemetry for this 2.0s window
            telemetry_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "active_app": app_name,
                "keystrokes": key_clicks,
                "mouse_clicks": mouse_clicks
            })

            # Reset counters for the next window
            key_clicks = 0
            mouse_clicks = 0

            # Trigger ESM Pop-up every 15 minutes
            if current_time - last_prompt_time >= PROMPT_INTERVAL:
                label = prompt_user_for_label()
                
                if label is not None and telemetry_buffer:
                    # Apply label to all rows in the buffer and save
                    df = pd.DataFrame(telemetry_buffer)
                    df['focus_score'] = label
                    filename = os.path.join(DATA_DIR, f"focus_data_{int(current_time)}.csv")
                    df.to_csv(filename, index=False)
                    print(f"Saved {len(df)} rows to {filename}")

                # Clear buffer and reset timer
                telemetry_buffer = []
                last_prompt_time = time.time()

    except KeyboardInterrupt:
        print("\nLogging stopped manually.")

if __name__ == "__main__":
    main()
