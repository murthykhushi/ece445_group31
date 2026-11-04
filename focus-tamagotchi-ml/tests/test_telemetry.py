import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from vision.telemetry import Telemetry
from vision.contracts import MetricPacket
from ab_test_visualizer import summarize


class TelemetryTests(unittest.TestCase):
    def test_permission_limited_accounting_never_passes_cpu_gate(self):
        with tempfile.TemporaryDirectory() as parent:
            directory=Path(parent)/'run'
            with patch('psutil.Process.children', side_effect=PermissionError('sandbox')):
                log=Telemetry(directory,{'backend':'test','fallback_reason':None},warmup_sec=0)
                metric=MetricPacket(1,1,'test',frame_age_ms=12,processing_ms=10)
                log.record(metric)
                log.close()
            _,_,report=summarize(directory)
            self.assertFalse(report['cpu_accounting_complete'])
            self.assertFalse(report['cpu_target_met'])
            with (directory/'telemetry.csv').open() as stream:
                row=next(csv.DictReader(stream))
                self.assertEqual(row['cpu_scope'],'process-only')

    def test_existing_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                Telemetry(directory,{'backend':'test'})
