import time
import threading
from pynput import keyboard, mouse
# AppKit requires pyobjc to be installed on Mac
try:
    from AppKit import NSWorkspace
except ImportError:
    NSWorkspace = None
    print("Warning: AppKit not found. Run 'pip install pyobjc' to track active apps.")

class DesktopPipeline:
    def __init__(self, window_sec=60):
        self.window_sec = window_sec
        self.keystrokes = []
        self.mouse_clicks = []
        
        # Start background listeners
        self.k_listener = keyboard.Listener(on_press=self.on_press)
        self.m_listener = mouse.Listener(on_click=self.on_click)
        self.k_listener.start()
        self.m_listener.start()

        # Update this dictionary to match exactly how you encoded apps during training!
        self.app_encoder = {
            "Google Chrome": 1,
            "Code": 2,          # VS Code usually shows up as "Code"
            "Terminal": 3,
            "Safari": 4,
            "Unknown": 0        # Default fallback
        }

    def on_press(self, key):
        self.keystrokes.append(time.time())

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.mouse_clicks.append(time.time())

    def get_active_app(self):
        if NSWorkspace is None:
            return "Unknown"
        active_app = NSWorkspace.sharedWorkspace().activeApplication()
        return active_app['NSApplicationName'] if active_app else "Unknown"

    def get_current_metrics(self):
        current_time = time.time()
        
        # Filter events to only keep those within the last 60 seconds
        self.keystrokes = [t for t in self.keystrokes if current_time - t <= self.window_sec]
        self.mouse_clicks = [t for t in self.mouse_clicks if current_time - t <= self.window_sec]
        
        active_app_name = self.get_active_app()
        
        # Convert the app name to a number for the Machine Learning model
        encoded_app = self.app_encoder.get(active_app_name, 0)
        
        return {
            "active_app_encoded": encoded_app,
            "keystrokes": len(self.keystrokes),
            "mouse_clicks": len(self.mouse_clicks),
            "active_app_name": active_app_name 
        }