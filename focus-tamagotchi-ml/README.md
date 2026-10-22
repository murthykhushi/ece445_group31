# Focus Tamagotchi ML Pipeline

Use Python 3.12. From this directory, set up the environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, allow the app running Python (Terminal, VS Code, or Codex) access under **System Settings → Privacy & Security → Camera, Accessibility, and Input Monitoring**. Restart the app after changing permissions if needed.

1. Collect personal training data: `python src/data_collection.py`. Press `s` to enter a focus score (0–100), or wait for the 60-second prompt. Press `q` in the camera window to quit.
2. Train with the included data or your collected samples: `python src/train_real.py`.
3. Run focus tracking: `python src/live_inference.py`. Predictions update every 60 seconds. Press `q` in the camera window to quit.

Data is stored in `data/real_training_data.csv`; collection appends to this file. The trained model and downloaded MediaPipe face model are stored in this project directory. Scripts resolve these paths independently of the current working directory. The face model downloads automatically on first use and requires internet access.
