"""YuNet + 2DFAN-4, batch size one, shared CPU/CUDA tensor implementation.

Kornia 0.8.2's FaceDetector.preprocess is identity: its YuNet weights take
BGR float pixels in [0,255]. FAN takes RGB [0,1]. Do not upgrade Kornia
without checking this contract (newer releases change preprocessing).
"""
from importlib.metadata import version
from pathlib import Path
import hashlib
import numpy as np
from vision.contracts import LandmarkPacket
from vision.model_artifacts import ARTIFACTS


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def crop_geometry(box):
    """FAN reference center/scale, continuous full-image pixel coordinates."""
    x1, y1, x2, y2 = map(float, box)
    width, height = x2 - x1, y2 - y1
    if width < 10 or height < 10 or not np.isfinite(box).all():
        raise ValueError('Invalid face box')
    side = 200 * (width + height) / 195.0
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2 - 0.12 * height
    return center_x - side / 2, center_y - side / 2, side


def decode_heatmaps(heatmaps):
    """Return subpixel heatmap coordinates and uncalibrated peak scores on device."""
    import torch
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 68 or min(heatmaps.shape[2:]) < 3:
        raise ValueError('FAN must return [B,68,H,W] heatmaps')
    b, n, h, w = heatmaps.shape
    flat = heatmaps.flatten(2)
    scores, indices = flat.max(dim=-1)
    x, y = indices % w, indices // w
    interior = (x > 0) & (x < w - 1) & (y > 0) & (y < h - 1)
    dx = flat.gather(2, (indices + 1).clamp(0, h*w-1).unsqueeze(-1)).squeeze(-1) - flat.gather(2, (indices - 1).clamp(0, h*w-1).unsqueeze(-1)).squeeze(-1)
    dy = flat.gather(2, (indices + w).clamp(0, h*w-1).unsqueeze(-1)).squeeze(-1) - flat.gather(2, (indices - w).clamp(0, h*w-1).unsqueeze(-1)).squeeze(-1)
    points = torch.stack((x + 0.5 + dx.sign() * 0.25 * interior,
                          y + 0.5 + dy.sign() * 0.25 * interior), dim=-1)
    return points, scores


class TorchBackend:
    def __init__(self, model_path, device='cuda', profile='optimized', detector_hz=5.0,
                 min_landmark_score=0.15, fp16=False, threads=1):
        import torch
        if device not in ('cuda', 'cpu') or profile not in ('naive', 'optimized'):
            raise ValueError('Unsupported torch device or profile')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Install a CUDA-enabled PyTorch build on an NVIDIA host; use --backend torch-cpu for an explicit CPU reference.')
        if fp16 and device != 'cuda':
            raise ValueError('FP16 requires CUDA')
        if detector_hz <= 0 or threads < 1:
            raise ValueError('detector_hz and threads must be positive')
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f'FAN model missing: {path}. Run src/download_models.py --fan.')
        if version('kornia') != '0.8.2':
            raise RuntimeError('This adapter requires kornia==0.8.2 (verified BGR [0,255] detector contract).')
        from kornia.contrib import FaceDetector
        torch.hub.set_dir(str(path.parent / 'torch-cache'))
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        torch.set_num_threads(threads)
        fan_hash = file_hash(path)
        if path.name == ARTIFACTS['fan']['filename'] and fan_hash != ARTIFACTS['fan']['sha256']:
            raise ValueError('FAN artifact checksum mismatch')
        detector_path = path.parent / ARTIFACTS['yunet']['filename']
        if not detector_path.is_file():
            raise FileNotFoundError('YuNet cache missing. Run src/download_models.py --yunet (same directory as FAN).')
        if file_hash(detector_path) != ARTIFACTS['yunet']['sha256']:
            raise ValueError('YuNet artifact checksum mismatch')
        self.net = torch.jit.load(str(path), map_location='cpu').eval()
        # YuNet is initialized by the pinned Kornia package and cached by torch.hub.
        self.detector = FaceDetector(confidence_threshold=0.7, top_k=200, keep_top_k=10).eval()
        detector_hash = model_hash(self.detector)
        self.net.to(device=self.device, dtype=self.dtype)
        self.detector.to(self.device)
        self.profile = profile
        self.detector_period = 1 / detector_hz
        self.min_score = min_landmark_score
        self.roi = None
        self.last_detection = float('-inf')
        self.last_timestamp = None
        self.staging = self.device_image = None
        self.actual_backend = 'torch-' + device
        self.identity = dict(backend=self.actual_backend, torch_version=torch.__version__, kornia_version=version('kornia'),
                             fan_sha256=fan_hash, detector_state_sha256=detector_hash, profile=profile,
                             detector_hz=detector_hz, min_landmark_score=min_landmark_score,
                             precision='fp16' if fp16 else 'fp32', threads=threads, schema='ibug68',
                             cuda_runtime=torch.version.cuda,
                             device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu')
        self.grid_axis = (torch.arange(256, device=self.device, dtype=torch.float32) + 0.5) / 256
        if device == 'cuda':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def _upload(self, image):
        torch = self.torch
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected uint8 HWC BGR input')
        source = torch.from_numpy(image)
        if self.profile == 'optimized' and self.device.type == 'cuda':
            if self.staging is None or tuple(self.staging.shape) != image.shape:
                self.staging = torch.empty(image.shape, dtype=torch.uint8, pin_memory=True)
                self.device_image = torch.empty(image.shape, dtype=torch.uint8, device=self.device)
                self.roi = None
            self.staging.copy_(source)
            self.device_image.copy_(self.staging, non_blocking=True)
            return self.device_image
        return source.to(self.device)

    def _crop(self, bgr, roi):
        torch = self.torch
        x, y, side = crop_geometry(roi)
        height, width = bgr.shape[-2:]
        # align_corners=False grid addresses pixel centers; zero padding handles image edges.
        xs = 2 * (x + self.grid_axis * side + 0.5) / width - 1
        ys = 2 * (y + self.grid_axis * side + 0.5) / height - 1
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        crop = torch.nn.functional.grid_sample(bgr, grid, align_corners=False, padding_mode='zeros')
        return crop[:, [2, 1, 0]].div(255).to(self.dtype), (x, y, side)

    def infer(self, frame):
        torch = self.torch
        if self.last_timestamp is not None and frame.timestamp <= self.last_timestamp:
            raise ValueError('Frame timestamps must increase')
        if self.last_timestamp is not None and frame.timestamp - self.last_timestamp > 0.2:
            self.roi = None
        self.last_timestamp = frame.timestamp
        with torch.inference_mode():
            if self.device.type == 'cuda':
                self.start_event.record()
            image = self._upload(frame.image).permute(2, 0, 1).unsqueeze(0).float()
            h, w = frame.image.shape[:2]
            detect = self.profile == 'naive' or self.roi is None or frame.timestamp - self.last_detection >= self.detector_period
            if detect:
                self.last_detection = frame.timestamp
                detector_image = torch.nn.functional.interpolate(image, size=(320, 320), mode='bilinear', align_corners=False)
                boxes = self.detector(detector_image)[0]
                # At most ten compact detections; no full-frame download.
                boxes = boxes.detach().cpu().numpy()
                if len(boxes) != 1:
                    self.roi = None
                    return self._finish(LandmarkPacket(flags=['multiple_faces' if len(boxes) else 'no_face'], detection_ran=True))
                self.roi = boxes[0, :4] * np.array([w / 320, h / 320, w / 320, h / 320])
            try:
                crop, (x, y, side) = self._crop(image, self.roi)
            except ValueError:
                self.roi = None
                return self._finish(LandmarkPacket(flags=['invalid_roi'], detection_ran=detect))
            heatmaps = self.net(crop)
            if isinstance(heatmaps, (tuple, list)):
                heatmaps = heatmaps[-1]
            points, scores = decode_heatmaps(heatmaps.float())
            points[..., 0] = x + points[..., 0] * side / heatmaps.shape[-1]
            points[..., 1] = y + points[..., 1] * side / heatmaps.shape[-2]
            compact = torch.cat((points[0], scores[0, :, None]), dim=-1).cpu().numpy()
            needed = [8, 30, *range(36, 48), 48, 54]
            if not np.isfinite(compact).all() or np.min(compact[needed, 2]) < self.min_score:
                self.roi = None
                return self._finish(LandmarkPacket(flags=['low_landmark_score'], detection_ran=detect))
            coords = compact[:, :2]
            low, high = coords.min(axis=0), coords.max(axis=0)
            if min(high - low) < 20 or (low < 0).any() or (high >= [w, h]).any():
                self.roi = None
                return self._finish(LandmarkPacket(flags=['clipped_or_small_face'], detection_ran=detect))
            # Move the detector box with landmarks, preserving its dimensions until refresh.
            # Avoid shrinking the box to eyebrows/jaw on each tracking update.
            center = coords[[27, 30]].mean(axis=0)
            if not detect and hasattr(self, 'previous_center'):
                shift = center - self.previous_center
                if np.linalg.norm(shift) > 0.3 * max(self.roi[2:] - self.roi[:2]):
                    self.roi = None
                    return self._finish(LandmarkPacket(flags=['rapid_motion'], detection_ran=detect))
                self.roi = self.roi + np.tile(shift, 2)
            self.previous_center = center
            return self._finish(LandmarkPacket(points=coords, detection_ran=detect))

    def _finish(self, packet):
        if self.device.type == 'cuda':
            self.end_event.record()
            self.end_event.synchronize()
            packet.gpu_ms = self.start_event.elapsed_time(self.end_event)
        # Synchronous completion makes staging reusable. Capture runs independently;
        # this implementation intentionally does not claim multi-stream overlap.
        return packet

    def close(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize()
        self.staging = self.device_image = None
