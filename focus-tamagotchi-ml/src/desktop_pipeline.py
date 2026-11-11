"""Bounded desktop activity counts. No keystroke contents are retained."""
from collections import deque
from threading import Lock
import sys
import time


class DesktopPipeline:
    def __init__(self, window_sec=60):
        from pynput import keyboard, mouse
        self.window_sec = window_sec
        self.keystrokes, self.mouse_clicks = deque(), deque()
        self.lock = Lock()
        self.app_encoder = {'Google Chrome': 1, 'Code': 2, 'Terminal': 3, 'Safari': 4, 'Unknown': 0}
        self.k_listener = keyboard.Listener(on_press=self.on_press)
        self.m_listener = mouse.Listener(on_click=self.on_click)
        self.k_listener.start()
        self.m_listener.start()

    def _purge(self, now):
        for events in (self.keystrokes, self.mouse_clicks):
            while events and events[0] <= now-self.window_sec:
                events.popleft()

    def reset_metrics(self):
        with self.lock:
            self.keystrokes.clear()
            self.mouse_clicks.clear()

    def close(self):
        self.k_listener.stop()
        self.m_listener.stop()

    def on_press(self, key):
        with self.lock:
            now = time.monotonic()
            self._purge(now)
            self.keystrokes.append(now)

    def on_click(self, x, y, button, pressed):
        if pressed:
            with self.lock:
                now = time.monotonic()
                self._purge(now)
                self.mouse_clicks.append(now)

    def get_active_app(self):
        if sys.platform == 'darwin':
            try:
                from AppKit import NSWorkspace
                app = NSWorkspace.sharedWorkspace().frontmostApplication()
                return app.localizedName() if app else 'Unknown'
            except ImportError:
                return 'Unknown'
        if sys.platform == 'win32':
            import win32gui
            title = win32gui.GetWindowText(win32gui.GetForegroundWindow()).lower()
            for keywords, name in [(('chrome',), 'Google Chrome'), (('visual studio code',), 'Code'),
                                   (('terminal', 'powershell', 'command prompt'), 'Terminal')]:
                if any(word in title for word in keywords):
                    return name
        # Linux window-manager APIs differ (X11/Wayland); missing app identity is explicit.
        return 'Unknown'

    def get_current_metrics(self):
        with self.lock:
            self._purge(time.monotonic())
            keys, clicks = len(self.keystrokes), len(self.mouse_clicks)
        app = self.get_active_app()
        return dict(active_app_encoded=self.app_encoder.get(app, 0), keystrokes=keys,
                    mouse_clicks=clicks, active_app_name=app)
