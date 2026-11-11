from threading import Condition, Event, Thread
import time
from vision.contracts import FramePacket


class LatestQueue:
    """A single pending item; replacing it never mutates a consumer's item."""
    def __init__(self):
        self.condition = Condition()
        self.item = None
        self.closed = False
        self.dropped = 0

    def put(self, item):
        with self.condition:
            if self.closed:
                return
            if self.item is not None:
                self.dropped += 1
            self.item = item
            self.condition.notify()

    def get(self, timeout=None):
        with self.condition:
            self.condition.wait_for(lambda: self.item is not None or self.closed, timeout)
            item, self.item = self.item, None
            return item

    def close(self, discard=False):
        with self.condition:
            self.closed = True
            if discard:
                self.item = None
            self.condition.notify_all()


class CameraCapture:
    def __init__(self, source=0, width=640, height=480, fps=30, queue=None):
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.queue = queue or LatestQueue()
        self.stop_event = Event()
        self.thread = Thread(target=self._run, name='camera-capture', daemon=True)
        self.error = None
        self.started = Event()
        self.manifest = {}

    def start(self):
        self.thread.start()
        if not self.started.wait(10):
            raise RuntimeError('Camera startup timed out')
        if self.error:
            raise RuntimeError(self.error)
        return self

    def _run(self):
        import cv2
        cap = None
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera {self.source}; check permission/device availability')
            for key, value in ((cv2.CAP_PROP_FRAME_WIDTH, self.width), (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                               (cv2.CAP_PROP_FPS, self.fps), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                cap.set(key, value)
            self.manifest = dict(width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                 fps_reported=cap.get(cv2.CAP_PROP_FPS), backend=cap.getBackendName(),
                                 timestamp_origin='host read completion; excludes hidden driver buffering')
            self.started.set()
            seq = 0
            while not self.stop_event.is_set():
                ok, image = cap.read()
                now = time.monotonic()
                if not ok:
                    raise RuntimeError('Camera disconnected or frame read failed')
                seq += 1
                self.queue.put(FramePacket(seq, now, image, now))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.started.set()
            if cap is not None:
                cap.release()
            self.queue.close()

    def close(self):
        self.stop_event.set()
        self.queue.close(discard=True)
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            # Some camera drivers cannot cancel blocking read; the daemon does not hold shutdown.
            self.error = self.error or 'Camera driver did not release within 2 seconds'
