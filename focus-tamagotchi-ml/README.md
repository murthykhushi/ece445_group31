# Focus Tamagotchi vision pipeline (v2)

A local Python pipeline for an NVIDIA Windows/Linux host: bounded camera capture → GPU preprocessing → YuNet detection → FAN eyelid landmarks → CPU head-pose geometry → time-weighted closure/blink metrics → optional trained focus model → optional STM32 output.

This branch implements the architecture, but **30 FPS, <10% CPU, blink accuracy and NVIDIA execution must be measured on your target machine**. The development machine is an Apple Silicon Mac. A successful CPU smoke test is not a CUDA performance result. Precise screen gaze and anatomically validated PERCLOS80 are not implemented; `perclos80_60s` remains null. The reported `closure_fraction_60s` is a calibrated EAR-based proxy.

## Setup (Python 3.12)

From `focus-tamagotchi-ml/`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`. Install a CUDA-enabled PyTorch build appropriate to your NVIDIA driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -r requirements-cuda.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python src/download_models.py --fan --yunet
```

Weights are cached in the ignored `models/` directory. Downloads use verified TLS, an atomic temporary file, and a recorded SHA256 sidecar. FAN and YuNet checksums are pinned in `vision/model_artifacts.py` and verified again at backend startup. After the first download, use `--sha256 <recorded hash>` with one model to enforce that artifact on another machine. Run manifests record the FAN file hash and YuNet state hash. Only load trusted model files. Inference startup requires the cached YuNet artifact and never downloads weights implicitly.

The optional CUDA requirements pin `kornia==0.8.2` because its detector takes BGR float pixels in [0,255]. Later versions change preprocessing. FAN takes RGB floats in [0,1]. The adapter rejects an unverified Kornia version rather than silently using the wrong colors. CPU and CUDA use the same pretrained models and decoding code. FAN is a relatively large reference model; profile it before claiming the target rate or selecting lighter weights.

For the MediaPipe reference, the existing `face_landmarker.task` is reused, or run `python src/download_models.py --mediapipe`. MediaPipe's GPU delegate is a separate backend, **not CUDA**; its documented Python GPU support is Ubuntu-specific. There is no silent GPU fallback.

## Run the vision service

```sh
python src/live_inference.py --backend torch-cuda --vision-only --duration 120
python src/live_inference.py --backend torch-cuda --vision-only --preview
python src/live_inference.py --backend torch-cpu --vision-only --duration 120
python src/live_inference.py --backend mediapipe-cpu --vision-only --duration 120
```

Preview is off by default. Use Ctrl+C to stop, or `q` in the preview window. `--camera`, `--width`, `--height`, and `--fps` configure capture (default 0 / 640 / 480 / 30). Requested camera settings are verified by recording the backend-reported configuration; actual throughput comes from completed frames. Permissions are requested by the OS when you run the camera. On macOS, grant Camera permission and, only for desktop activity, Accessibility/Input Monitoring.

`--allow-cpu-fallback` permits an explicit degraded CPU run. Logs include requested/actual backend and the failure reason; the run cannot count as successful GPU validation. The benchmark command rejects fallback. FP16 is experimental and opt-in (`--fp16`); compare accuracy against FP32 before adopting it.

No focus model or desktop input hooks are needed in `--vision-only` mode. Linux active-app identity currently reports `Unknown`; desktop input hooks depend on your X11/Wayland environment. The vision service is independent of those hooks.

## Runtime behavior and metric definitions

- A dedicated camera thread owns capture. A one-item queue replaces pending frames and counts drops; frames already owned by inference are never mutated. The inference worker owns the model and temporal state. Output/serial is a separate worker with bounded backlog. Native camera calls that do not unblock at shutdown are reported and cannot keep the process alive.
- The optimized CUDA backend reuses one pinned staging tensor and one device upload buffer. GPU preprocessing, cropping, detection and heatmap decoding follow one full-frame upload. Only compact boxes (on detector frames) and landmarks return to the CPU. Completion is synchronized before reusing staging memory. Capture overlaps computation, but **multi-stream copy/compute overlap is not claimed**.
- `--profile naive` allocates/uploads normally and detects on every frame. `--profile optimized` reuses upload buffers and refreshes detection at `--detector-hz 5` with landmark-based ROI translation between refreshes. Invalid/rapid-motion/clipped landmarks force reacquisition. Multiple faces cause UNKNOWN rather than choosing another person.
- A long camera gap, missed face, invalid eye, or disagreeing eye states is UNKNOWN. Long gaps are not filled with the previous eye state. For gaps ≤100 ms between valid samples, the previous state applies to that interval. Missing intervals are excluded from the closure denominator and reduce coverage.
- `closure_fraction_60s = closed valid seconds / valid seconds`. The full 60-second window and ≥80% valid coverage are required. Startup values are null. At 30 FPS, blink timing remains quantized at roughly 33 ms.
- Eye names follow the landmark convention: `left` is the image-left eye in an unmirrored, frontal frame. Eye thresholds have hysteresis. A complete OPEN→CLOSED→OPEN event lasting 60–800 ms is a blink. Closures crossing missing intervals are discarded as events. Longer closures remain in the closure fraction. `blink_event_duration_ms` is emitted on reopening; `blink_count_60s` and median duration cover the rolling window. No completed blinks yields a null median, not a fabricated duration.
- Pose requires camera intrinsics. Without calibration, angles are null and `pose_uncalibrated` is recorded. Pose uses six canonical face correspondences, PnP, reprojection rejection, a ±70° pitch / ±75° yaw / ±80° roll operating gate, and time-aware smoothing. Angles follow OpenCV rotation decomposition: pitch about camera x, yaw about y, roll about z; x image-right, y down, z forward. Neutral offsets are `[pitch, yaw, roll]`. Generic face geometry still needs reference validation.
- Frame age runs from host read completion to metric completion, so hidden camera/driver buffering is excluded. A frame older than 100 ms is marked stale and cannot produce a focus prediction. A newer result is also required at consumption time.

## Calibration

Default EAR thresholds are **uncalibrated** and explicitly marked. Copy `config/calibration.example.json` to your own file or generate it with the helper.

For camera intrinsics, capture at least ten sharp checkerboard images across positions and angles, at the actual camera resolution. Use interior corner counts:

```sh
python src/calibrate_vision.py --output config/my-camera.json camera \
  --images 'calibration/*.jpg' --columns 9 --rows 6 --square-size 25
```

For eye thresholds, independently label open and closed samples (do not reuse the system's predicted state as ground truth). Input CSV columns: `eye_state,ear_left,ear_right`, with at least 20 samples of each state per eye. Replay JSONL provides EAR values to annotate. Exclude transitions and occlusions.

```sh
python src/calibrate_vision.py --existing config/my-camera.json \
  --output config/my-calibration.json eyes --samples calibration/eyes.csv
python src/live_inference.py --backend torch-cuda --vision-only \
  --calibration config/my-calibration.json
```

The helper rejects overlapping eye distributions and poor camera reprojection fits. Eye calibration does not validate the anatomical “80% closed” criterion. Pose intrinsics can be scaled to the same aspect ratio; changed aspect ratios are rejected. Validate lens cropping/zoom and exposure settings separately.

## Collection, training, and focus inference

The original six-feature model and CSV remain untouched. They **cannot be used with v2 metrics**. New collection writes `data/focus_v2.csv`; new training writes `focus_v2.joblib` and an evaluation report.

```sh
python src/data_collection.py --backend torch-cuda --calibration config/my-calibration.json
python src/train_real.py
python src/live_inference.py --backend torch-cuda --calibration config/my-calibration.json
```

Each collection run gets a separate session ID. Every minute, a prompt asks you to score the minute that just ended. The feature snapshot is frozen at the prompt while vision continues in its worker. Invalid/low-coverage/stale windows are skipped; blank input skips a label; `q` exits. Never mix legacy rows into this file.

Training requires at least ten labeled rows from two sessions and holds out entire sessions. The minimum is a software guard, not a sufficient sample-size claim. Collect multiple people/sessions and report participant-level generalization separately. The saved model remains fitted only on the training split. The report compares held-out MAE with a constant training-mean baseline. Synthetic unit-test data is never saved as the user's focus model.

The shared feature schema contains current per-eye EAR, 60-second closure fraction, blink count/median, current head angles, current application code, and 60-second key/click counts. Missing pose or no-blink duration is imputed by a fitted training pipeline with missingness indicators. Pose is currently a filtered current-frame feature, not a full-window pose aggregate. Model schema and landmark-family compatibility are checked before inference. Application identity alone is not rewarded with a hard-coded focus score.

Focus prediction stays at the existing 60-second cadence. UNKNOWN clears the visible score; stale predictions expire after 65 seconds. After a gap, a new valid prediction is required rather than reviving an old score.

## STM32 integration

Serial is disabled unless `--serial-port` is supplied. For firmware implementing the new contract:

```sh
python src/live_inference.py --backend torch-cuda --serial-port COM3 --serial-protocol json-v2
```

The host sends newline-delimited JSON once a second:

```json
{"version":2,"seq":1,"state":"FOCUSED","score":72.0,"ttl_ms":1999}
```

`state` is `FOCUSED`, `UNFOCUSED`, or `UNKNOWN`; UNKNOWN score is null. The firmware must track sequence, expire the state `ttl_ms` after receipt using its local clock, and handle UNKNOWN explicitly. No MCU firmware exists in this repository, so this host change does not claim that behavior is installed on the STM32.

`--serial-protocol legacy` preserves `0\n`/`1\n`. It sends nothing for UNKNOWN. **Legacy firmware can retain a stale physical state** because it has no TTL/unknown field; use it only when that limitation is acceptable. Bounded queues and 100 ms serial write timeouts prevent a stalled MCU from blocking inference.

## Reproducible benchmarks

Use a consented constant-frame-rate video, the same weights, calibration, resolution and precision across runs. Frame index / declared FPS supplies source time; variable-frame-rate recordings need conversion first. Replay measures processing, not camera capture/driver latency.

```sh
python src/benchmark_vision.py clip.mp4 --backend torch-cpu --profile naive --log-dir runs/cpu
python src/benchmark_vision.py clip.mp4 --backend torch-cuda --profile naive --log-dir runs/gpu-naive
python src/benchmark_vision.py clip.mp4 --backend torch-cuda --profile optimized --log-dir runs/gpu-optimized
python src/ab_test_visualizer.py runs/cpu runs/gpu-naive runs/gpu-optimized \
  --plot runs/comparison.png --report runs/comparison.json
```

Existing run directories are rejected so logs are not overwritten. Default warm-up is 30 seconds of wall time. Use long enough clips; `--warmup 0` is for explicit smoke tests. Default replay processes every frame as fast as possible. `--realtime` paces by clip time and counts skipped frames when behind. Run live camera trials separately using `--duration 630` and repeat three times, with both idle and representative foreground GPU workloads. Record driver, power mode, camera exposure and hardware in addition to the generated manifest.

Each run contains a manifest and one-second telemetry. Replay additionally saves `metrics.jsonl` for labels. CPU is process-tree user+system CPU time / wall seconds / logical CPUs ×100; raw core-equivalent CPU and system CPU are also reported. RSS is the sum across observed processes (shared pages may be double-counted). Short-lived child processes that exit between samples can be missed; the vision architecture itself uses threads. Sampling includes telemetry overhead. If OS permissions prevent enumerating the process tree, logs explicitly mark process-only accounting as incomplete and the report refuses to declare the CPU target met.

All completed frame ages enter each interval's p50/p95/max. The report calls the maximum of interval p95 values exactly that; it does not claim it is a pooled p95. CUDA events measure an elapsed device timeline span including transfers and host scheduling gaps, not pure summed kernel execution. Use `benchmark_vision.py clip.mp4 --backend torch-cuda --max-frames 10 --trace runs/cuda-trace.json` to export a short Chrome trace (after two warm-up frames, five frames are recorded). This is a separate diagnostic run; profiling overhead invalidates performance acceptance. Inspect its CUDA kernels/transfers in a trace viewer or use Nsight on the NVIDIA machine. PyTorch allocated/reserved VRAM is recorded once per interval on CUDA (it is not total board memory). GPU utilization, board power and thermal counters are not collected automatically; capture those with your platform profiler. The initial implementation does not automatically lower cadence to pass the CPU limit.

The proposed CPU gate is post-warmup duration-weighted mean and sampled-interval p95 <10%; the final interval may be shorter than one second and its duration is logged. If the requirement means every sample or whole-system CPU, change the acceptance convention explicitly. Compare matching naive CPU/CUDA runs to isolate acceleration; compare naive/optimized CUDA to isolate scheduling and buffering. FP16 is a separate ablation. GPU execution does not by itself improve accuracy.

Accuracy labels are CSV with `timestamp` in clip seconds and any of `eye_state`, `closure_fraction_60s`, `yaw_deg`, `pitch_deg`, `roll_deg`, `blink_event_duration_ms`. Blink duration is populated only at each manually labeled reopening timestamp. Closure references must use the same valid-interval definition. Pose references must be independent of this model.

```sh
python src/evaluate_vision.py --predictions runs/gpu-optimized/metrics.jsonl \
  --labels labels.csv --output runs/accuracy.json
```

The evaluator reports evaluated pair counts and alignment coverage alongside errors; missing predictions are not silently scored as correct. Frame tolerance is 50 ms; blink event matching is one-to-one within 100 ms, with false-positive/negative counts and duration error. Report those choices with results. Proposed gates from the design: blink F1 ≥0.90, duration MAE ≤50 ms, closure MAE ≤0.05 fraction, pose MAE ≤5°, p95 frame age <100 ms. These are targets, not measured achievements.

## Verification and code map

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src
```

On PowerShell set `$env:PYTHONPATH='src'` before the unittest command. Tests need the base dependencies; tensor tests use optional torch and the CUDA parity check skips when no NVIDIA device is available. Unit tests do not open the camera or download weights.

- `src/vision/backends/`: same-model CPU/CUDA adapter and separate MediaPipe adapter.
- `capture.py`, `service.py`: bounded queues, worker ownership, failure propagation.
- `geometry.py`, `temporal.py`, `config.py`: calibration, pose, EAR, blink/closure windows.
- `features.py`: single collection/training/inference contract; no v1 reinterpretation.
- `telemetry.py`, `benchmark_vision.py`, `evaluate_vision.py`: resource and accuracy evidence.
- `output.py`: bounded serial adapter; firmware changes remain a separate integration step.

Raw video is not stored during live operation. Normal logs contain metrics and configuration, not keystroke contents. Keep consented test clips and calibration data local and define a retention policy.

Model/preprocessing references: [Kornia 0.8.2 detector source](https://github.com/kornia/kornia/blob/v0.8.2/kornia/contrib/face_detection.py), [FAN reference implementation](https://github.com/1adrianb/face-alignment/blob/v1.4.1/face_alignment/api.py), [FAN paper](https://arxiv.org/abs/1703.07332), [MediaPipe Python delegate support](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions), [PyTorch transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html). GPU crop sampling uses continuous pixel-center coordinates, so compare it against the reference implementation's integer-rounded crop as part of landmark accuracy validation. Review upstream weight/data licenses before redistribution; weights are not committed.
