# Implementation verification — 4 November 2026

Development host: macOS / Apple Silicon, Python 3.12.1, PyTorch 2.15.0, Kornia 0.8.2. No NVIDIA CUDA device was available.

Completed checks:

- `PYTHONPATH=src python -m unittest discover -s tests -v`: 38 tests discovered, 37 passed, 1 skipped (CUDA numerical parity requires NVIDIA hardware).
- `python -m compileall -q src` and `git diff --check`: passed.
- Actual pretrained YuNet + FAN CPU inference on the upstream face-alignment test image: valid eyelid landmarks and OPEN state, explicit null pose without camera calibration.
- Actual MediaPipe CPU inference on the same image: valid eyelid landmarks and OPEN state.
- Deterministic replay through the optimized CPU backend: 12 frames completed; metric JSONL, telemetry CSV and run manifest written.
- Naive CPU replay with profiling: 10 frames completed; Chrome trace exported.
- Comparison-report and plot generation: completed. Profiling and permission-limited CPU accounting are explicitly excluded from successful CPU acceptance.
- Cached model downloads: pinned SHA256 checks passed for FAN and YuNet.
- Training contract and held-out-session split: verified with disposable synthetic unit-test fixtures. No user focus model was trained or overwritten.

The test image and generated smoke artifacts were kept outside source control. The test image source was https://raw.githubusercontent.com/1adrianb/face-alignment/master/test/assets/aflw-test.jpg. It was repeated into a short local video to exercise the replay/report interfaces; this is not an accuracy dataset or a sustained performance experiment.

Not established by these checks:

- CUDA model execution, host CPU <10%, sustained 30 FPS, target-machine frame age, or foreground workload impact.
- Held-out blink, anatomical PERCLOS80, pose, or focus accuracy.
- Live camera behavior on Windows/Linux or STM32 firmware handling of JSON/TTL/UNKNOWN.

The Mac sandbox denied process-tree enumeration. The logger correctly fell back to explicitly incomplete process-only accounting, and the report refused to declare the CPU target satisfied. Full target-hardware experiments should follow README instructions, using consented clips, measured camera calibration, and independent labels. No camera was opened for these development checks.
