import time
import sys
from pynput import keyboard, mouse

current_os = sys.platform

if current_os == 'darwin':
    try:
        from AppKit import NSWorkspace
    except ImportError:
        NSWorkspace = None
        print("AppKit not found. 'pyobjc' is required for MacOS.")
elif current_os == 'win32':
    try:
        import win32gui
    except ImportError:
        win32gui = None
        print("win32gui not found. 'pywin32' is required for Windows.")
elif current_os.startswith('linux'):
    # TODO: Add Linux support in the future
    pass

class DesktopPipeline:
    def __init__(self, window_sec=60):
        self.window_sec = window_sec
        self.keystrokes = []
        self.mouse_clicks = []

        # Instead of using polling, we use threads for keyboard and mouse so we dont miss inputs
        # These record timestamps when events happen 
        self.k_listener = keyboard.Listener(on_press=self.on_press)
        self.m_listener = mouse.Listener(on_click=self.on_click)
        self.k_listener.start()
        self.m_listener.start()

        # Associate most commonly used apps / apps found in training data with numerical values 
        self.app_encoder = {
            "Google Chrome": 1,
            "Code": 2,
            "Terminal": 3,
            "Safari": 4,
            "Unknown": 0
        }

    def reset_metrics(self):
        self.keystrokes.clear()
        self.mouse_clicks.clear()

    def close(self):
        self.k_listener.stop()
        self.m_listener.stop()

    # Different measurements needed for the desktop portion of pipeline 
    def on_press(self, key):
        self.keystrokes.append(time.time())

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.mouse_clicks.append(time.time())

    def get_active_app(self):
        app_name = "Unknown"

        if current_os == 'darwin' and NSWorkspace:
            active_app = NSWorkspace.sharedWorkspace().activeApplication()
            if active_app:
                app_name = active_app['NSApplicationName']

        elif current_os == 'win32' and win32gui:
            window = win32gui.GetForegroundWindow()
            if window:
                window_title = win32gui.GetWindowText(window)
                if "Chrome" in window_title:
                    app_name = "Google Chrome"
                elif "Code" in window_title or "Visual Studio" in window_title:
                    app_name = "Code"
                elif "Terminal" in window_title or "PowerShell" in window_title or "cmd" in window_title.lower():
                    app_name = "Terminal"
                else:
                    app_name = window_title

        elif current_os.startswith('linux'):
            # TODO: Add Linux support in the future
            pass

        return app_name

    def get_current_metrics(self):
        current_time = time.time()
        
        self.keystrokes = [t for t in self.keystrokes if current_time - t <= self.window_sec]
        self.mouse_clicks = [t for t in self.mouse_clicks if current_time - t <= self.window_sec]
        
        active_app_name = self.get_active_app()
        encoded_app = self.app_encoder.get(active_app_name, 0)
        
        return {
            "active_app_encoded": encoded_app,
            "keystrokes": len(self.keystrokes),
            "mouse_clicks": len(self.mouse_clicks),
            "active_app_name": active_app_name 
        }