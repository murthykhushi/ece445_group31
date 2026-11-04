from threading import Event, Thread
from vision.capture import CameraCapture, LatestQueue


class VisionService:
    """Camera owns capture; one worker owns all inference and temporal state."""
    def __init__(self, pipeline, source=0, width=640, height=480, fps=30, preview=False):
        self.pipeline = pipeline
        self.capture = CameraCapture(source, width, height, fps)
        self.results = LatestQueue()
        self.stop_event = Event()
        self.worker = Thread(target=self._run, name='vision-worker', daemon=True)
        self.error = None
        self.preview = preview
        self.completed = 0
        self.telemetry = None

    def start(self):
        self.capture.start()
        self.worker.start()
        return self

    def _run(self):
        try:
            while not self.stop_event.is_set():
                frame = self.capture.queue.get(timeout=0.5)
                if frame is None:
                    if self.capture.queue.closed:
                        if self.capture.error:
                            raise RuntimeError(self.capture.error)
                        break
                    continue
                metric = self.pipeline.process(frame)
                self.completed += 1
                if self.telemetry is not None:
                    self.telemetry.record(metric, self.capture.queue.dropped)
                self.results.put((metric, frame.image if self.preview else None))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.results.close()
            self.pipeline.close()

    def get(self, timeout=0.5):
        item = self.results.get(timeout)
        if item is None and self.error:
            raise RuntimeError(self.error)
        return item

    def close(self):
        self.stop_event.set()
        self.capture.close()
        if self.worker.is_alive():
            self.worker.join(timeout=5)
        elif self.worker.ident is None:
            self.pipeline.close()
        self.results.close(discard=True)
        if self.worker.is_alive():
            raise RuntimeError('Vision worker did not stop within 5 seconds')
