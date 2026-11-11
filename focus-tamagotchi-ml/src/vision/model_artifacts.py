"""Known model artifacts verified during the implementation smoke test."""
ARTIFACTS = {
    'fan': {
        'filename': '2DFAN4-cd938726ad.zip',
        'url': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
        'sha256': 'cd938726adb1f15f361263cce2db9cb820c42585fa8796ec72ce19107f369a46',
    },
    'yunet': {
        'filename': 'torch-cache/checkpoints/yunet_final.pth',
        'url': 'https://github.com/kornia/data/raw/main/yunet_final.pth',
        'sha256': 'ee0a3035c5cbfc9484a31332223c9fd9c0ddcdb7c46b6a19e558ed393b6f2c67',
    },
    'mediapipe': {
        'filename': 'face_landmarker.task',
        'url': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        'sha256': None,
    },
}
