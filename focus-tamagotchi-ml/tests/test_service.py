import time
import unittest
import numpy as np
from vision.capture import LatestQueue
from vision.contracts import FramePacket, MetricPacket
from vision.service import VisionService


class FakeCapture:
    def __init__(self):
        self.queue=LatestQueue()
        self.error=None
    def start(self):
        now=time.monotonic()
        self.queue.put(FramePacket(1,now,np.zeros((4,4,3),dtype=np.uint8),now))
        self.queue.close()
    def close(self):
        self.queue.close()


class FakePipeline:
    def __init__(self, fail=False):
        self.fail=fail
        self.closed=False
    def process(self, frame):
        if self.fail:
            raise RuntimeError('inference failed')
        return MetricPacket(frame.seq,frame.timestamp,'test')
    def close(self):
        self.closed=True


class ServiceTests(unittest.TestCase):
    def test_worker_outputs_and_releases_pipeline(self):
        pipeline=FakePipeline()
        service=VisionService(pipeline)
        service.capture=FakeCapture()
        service.start()
        packet,image=service.get(timeout=1)
        self.assertEqual(packet.seq,1)
        self.assertIsNone(image)
        service.close()
        self.assertTrue(pipeline.closed)

    def test_worker_error_is_propagated_and_closed(self):
        pipeline=FakePipeline(fail=True)
        service=VisionService(pipeline)
        service.capture=FakeCapture()
        service.start()
        with self.assertRaisesRegex(RuntimeError,'inference failed'):
            service.get(timeout=1)
        service.close()
        self.assertTrue(pipeline.closed)
