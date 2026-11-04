import argparse
from pathlib import Path
from vision.config import Calibration
from vision.pipeline import VisionPipeline, ROOT


def add_vision_arguments(parser):
    parser.add_argument('--backend', choices=['torch-cuda', 'torch-cpu', 'mediapipe-cpu', 'mediapipe-gpu'], default='torch-cuda')
    parser.add_argument('--profile', choices=['naive', 'optimized'], default='optimized')
    parser.add_argument('--model-path', type=Path, help='FAN TorchScript or MediaPipe task, depending on backend')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--allow-cpu-fallback', action='store_true', help='Explicit degraded mode; invalidates accelerated benchmark')
    parser.add_argument('--detector-hz', type=float, default=5)
    parser.add_argument('--fp16', action='store_true', help='Experimental CUDA precision; validate accuracy separately')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--preview', action='store_true', help='Enable CPU video overlay; off by default')


def make_pipeline(args):
    import cv2
    cv2.setNumThreads(args.threads)
    if min(args.width, args.height, args.fps, args.threads) <= 0 or args.detector_hz <= 0:
        raise ValueError('Dimensions, FPS, threads and detector rate must be positive')
    if args.fp16 and args.backend != 'torch-cuda':
        raise ValueError('--fp16 requires --backend torch-cuda')
    return VisionPipeline(backend=args.backend, calibration=Calibration.load(args.calibration),
                          model_path=args.model_path, profile=args.profile, detector_hz=args.detector_hz,
                          fp16=args.fp16, threads=args.threads, allow_cpu_fallback=args.allow_cpu_fallback)
