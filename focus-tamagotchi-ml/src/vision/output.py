"""Serial writes are isolated from camera/inference, with bounded backlog."""
from threading import Thread
import json
import time
from vision.capture import LatestQueue


class SerialOutput:
    def __init__(self, port, protocol='json-v2', baud=115200):
        import serial
        if protocol not in ('json-v2', 'legacy'):
            raise ValueError('Unsupported serial protocol')
        self.port = serial.Serial(port, baud, timeout=0.1, write_timeout=0.1)
        self.protocol = protocol
        self.queue = LatestQueue()
        self.error = None
        self.worker = Thread(target=self._run, name='serial-output', daemon=True)
        self.worker.start()

    def send(self, state, score=None, ttl_ms=2000):
        self.queue.put(dict(state=state, score=score, ttl_ms=ttl_ms, enqueued=time.monotonic()))

    def _run(self):
        seq = 0
        try:
            while True:
                item = self.queue.get(timeout=0.5)
                if item is None:
                    if self.queue.closed:
                        break
                    continue
                age_ms = (time.monotonic() - item.pop('enqueued')) * 1000
                if age_ms >= item['ttl_ms']:
                    continue
                item['ttl_ms'] = max(1, int(item['ttl_ms'] - age_ms))
                if self.protocol == 'legacy':
                    if item['state'] == 'UNKNOWN':
                        continue  # Old binary firmware cannot express missing data.
                    payload = ('1' if item['state'] == 'FOCUSED' else '0') + '\n'
                else:
                    seq += 1
                    payload = json.dumps(dict(version=2, seq=seq, **item), allow_nan=False) + '\n'
                self.port.write(payload.encode('ascii'))
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.port.close()

    def close(self):
        self.send('UNKNOWN')
        self.queue.close()
        self.worker.join(timeout=1)
