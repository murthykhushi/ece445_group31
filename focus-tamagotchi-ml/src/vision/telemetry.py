import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import psutil


class Telemetry:
    """One-second summaries include ALL completed frame timings, not one sampled loop."""
    FIELDS = ['elapsed_s', 'interval_s', 'backend_actual', 'fallback', 'cpu_core_percent', 'cpu_machine_percent',
              'system_cpu_percent', 'cpu_scope', 'cpu_accounting_complete', 'process_tree_rss_mb', 'completed_fps', 'dropped_total',
              'frame_age_p50_ms', 'frame_age_p95_ms', 'frame_age_max_ms', 'processing_p95_ms',
              'gpu_span_p95_ms', 'gpu_allocated_mb', 'gpu_reserved_mb', 'coverage', 'detection_frames', 'warmup']

    def __init__(self, directory, identity, warmup_sec=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.start = self.last = time.monotonic()
        self.warmup = warmup_sec
        self.process = psutil.Process()
        self.cores = psutil.cpu_count() or 1
        self.previous_cpu = {}
        self.cpu_scope = 'process-tree'
        self.cpu_accounting_complete = True
        self._cpu_snapshot()
        psutil.cpu_percent(None)
        self.file = (self.directory / 'telemetry.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.identity = identity
        self.ages, self.times, self.gpu = [], [], []
        self.detection_frames = 0
        self.last_metric = None
        self.last_dropped = 0
        self.write_manifest(dict(identity=identity, platform=platform.platform(), python=platform.python_version(),
                                 logical_cpus=self.cores, warmup_sec=warmup_sec,
                                 cpu_definition='100 * process-tree CPU seconds / wall seconds / logical CPUs',
                                 gpu_timing='CUDA event span including transfer and host scheduling gaps, not summed kernel time'))

    def write_manifest(self, additions):
        path = self.directory / 'manifest.json'
        data = json.loads(path.read_text()) if path.exists() else {}
        data.update(additions)
        path.write_text(json.dumps(data, indent=2, default=str) + '\n')

    def _cpu_snapshot(self):
        total_delta, rss = 0.0, 0
        current = {}
        self.cpu_scope = 'process-tree'
        self.cpu_accounting_complete = True
        try:
            children = self.process.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
            children = []
            self.cpu_scope = 'process-only'
            self.cpu_accounting_complete = False
        for process in [self.process, *children]:
            try:
                key = (process.pid, process.create_time())
                times = process.cpu_times()
                value = times.user + times.system
                # Newly observed children contribute CPU accumulated since birth.
                previous = self.previous_cpu.get(key, value if key[0] == self.process.pid else 0)
                total_delta += max(0.0, value - previous)
                current[key] = value
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                self.cpu_accounting_complete = False
        self.previous_cpu = current
        return total_delta, rss

    def record(self, metric, dropped=0):
        self.ages.append(metric.frame_age_ms)
        self.times.append(metric.processing_ms)
        if metric.gpu_ms is not None:
            self.gpu.append(metric.gpu_ms)
        self.detection_frames += int(metric.detection_ran)
        self.last_metric, self.last_dropped = metric, dropped
        if time.monotonic() - self.last >= 1:
            self.flush()

    def flush(self):
        if not self.ages:
            return
        now = time.monotonic()
        elapsed = max(now - self.last, 1e-9)
        cpu, rss = self._cpu_snapshot()
        gpu_memory = dict(gpu_allocated_mb='', gpu_reserved_mb='')
        if self.identity.get('backend') == 'torch-cuda':
            import torch
            gpu_memory = dict(gpu_allocated_mb=torch.cuda.memory_allocated()/1024**2,
                              gpu_reserved_mb=torch.cuda.memory_reserved()/1024**2)
        row = dict(**gpu_memory, elapsed_s=now-self.start, interval_s=elapsed, backend_actual=self.last_metric.backend_actual,
                   fallback=bool(self.identity.get('fallback_reason')), cpu_core_percent=100*cpu/elapsed,
                   cpu_machine_percent=100*cpu/elapsed/self.cores, system_cpu_percent=psutil.cpu_percent(None),
                   cpu_scope=self.cpu_scope, cpu_accounting_complete=self.cpu_accounting_complete,
                   process_tree_rss_mb=rss/1024**2, completed_fps=len(self.ages)/elapsed,
                   dropped_total=self.last_dropped, frame_age_p50_ms=float(np.percentile(self.ages, 50)),
                   frame_age_p95_ms=float(np.percentile(self.ages, 95)), frame_age_max_ms=max(self.ages),
                   processing_p95_ms=float(np.percentile(self.times, 95)),
                   gpu_span_p95_ms=float(np.percentile(self.gpu, 95)) if self.gpu else '',
                   coverage=self.last_metric.valid_coverage_60s, detection_frames=self.detection_frames,
                   warmup=self.last - self.start < self.warmup)
        self.writer.writerow(row)
        self.file.flush()
        self.last = now
        self.ages.clear(); self.times.clear(); self.gpu.clear()
        self.detection_frames = 0

    def close(self):
        self.flush()
        self.file.close()
