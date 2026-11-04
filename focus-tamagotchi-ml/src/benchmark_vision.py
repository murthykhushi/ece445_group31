"""Replay a local clip through the same pipeline. Camera capture is tested separately."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import cv2
from vision.cli import ROOT, add_vision_arguments, make_pipeline
from vision.contracts import FramePacket
from vision.telemetry import Telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_vision_arguments(parser)
    parser.add_argument('video', type=Path)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--warmup', type=float, default=30)
    parser.add_argument('--realtime', action='store_true', help='Pace to clip timestamps; drop stale decoded frames when behind')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--trace', type=Path, help='Optional PyTorch CPU/CUDA Chrome trace; invalidates performance acceptance')
    args = parser.parse_args()
    if args.warmup < 0 or args.max_frames < 0:
        parser.error('warmup and max-frames must be nonnegative')
    if args.allow_cpu_fallback:
        parser.error('Benchmarks forbid fallback; explicitly select the intended backend')
    if args.preview:
        parser.error('Replay benchmarks disable preview')
    pipeline = telemetry = cap = profiler = None
    try:
        pipeline = make_pipeline(args)
        if args.trace:
            if not args.backend.startswith('torch-'):
                raise ValueError('--trace supports PyTorch backends only')
            import torch
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            if args.trace.exists():
                raise FileExistsError(f'Trace already exists: {args.trace}')
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.backend == 'torch-cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True,
                schedule=torch.profiler.schedule(wait=0, warmup=2, active=5, repeat=1),
                on_trace_ready=lambda prof: prof.export_chrome_trace(str(args.trace)))
            profiler.start()
        run_dir = args.log_dir or ROOT / 'runs' / datetime.now(timezone.utc).strftime('replay-%Y%m%dT%H%M%S.%fZ')
        telemetry = Telemetry(run_dir, pipeline.identity, args.warmup)
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {args.video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('Clip must declare a positive constant frame rate')
        telemetry.write_manifest(dict(video=str(args.video.resolve()), clip_fps=fps, realtime=args.realtime,
                                      timestamp_origin='frame index / declared FPS; use constant-frame-rate clips',
                                      arguments=vars(args), profiling_enabled=bool(args.trace)))
        started = time.monotonic()
        seq = completed = dropped = 0
        with (run_dir / 'metrics.jsonl').open('w') as output:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                timestamp = seq / fps
                seq += 1
                due = started + timestamp
                if args.realtime:
                    lag = time.monotonic() - due
                    if lag > 1 / fps:
                        dropped += 1
                        continue
                    if lag < 0:
                        time.sleep(-lag)
                read_completed = due if args.realtime else time.monotonic()
                metric = pipeline.process(FramePacket(seq, timestamp, image, read_completed))
                output.write(json.dumps(metric.to_dict(), allow_nan=False) + '\n')
                telemetry.record(metric, dropped)
                if profiler:
                    profiler.step()
                completed += 1
                if args.max_frames and completed >= args.max_frames:
                    break
        telemetry.write_manifest(dict(completed_frames=completed, dropped_frames=dropped, decoded_frames=seq,
                                      wall_seconds=time.monotonic()-started))
        print(f'{completed} frames completed; {dropped} dropped; logs: {run_dir}')
    finally:
        if profiler:
            profiler.stop()
        if cap is not None:
            cap.release()
        if pipeline:
            pipeline.close()
        if telemetry:
            telemetry.close()


if __name__ == '__main__':
    main()
