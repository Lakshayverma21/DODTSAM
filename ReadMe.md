# DODTSAM

Dynamic Object Detection and Tracking System with Assistive Mobility.

DODTSAM is a real-time webcam object detection assistant built with Ultralytics YOLO, OpenCV, Matplotlib, and text-to-speech. It detects common objects, estimates approximate distance with a calibrated pinhole-camera model, identifies dominant object color, announces object information aloud, and triggers a proximity alarm when something is too close.

## Features

- Real-time object detection with YOLO11n
- Automatic CPU, CUDA, or Apple MPS device selection
- Approximate distance estimation in centimeters
- Dominant color detection for detected objects
- Left, right, and center position feedback
- Non-blocking voice announcements with `pyttsx3`
- Repeating proximity alarm for nearby objects
- Matplotlib display window for Mac-friendly rendering

## Project Structure

```text
DODTSAM/
├── object_detection_blind.py   # Main application
├── requirements.txt            # Python runtime dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Files excluded from GitHub
```

The script uses `MODEL_PATH = "yolo11n.pt"` by default. Ultralytics downloads this model automatically on first run if it is not already present. Keep local model weights such as `*.pt` out of Git unless you intentionally use Git LFS.

## Requirements

- Python 3.10 recommended
- Webcam or USB camera
- Working audio output for text-to-speech and alerts
- Tkinter support for Matplotlib's `TkAgg` backend

Platform notes:

- macOS: alarm playback uses built-in system sounds through `afplay`.
- Windows: alarm playback uses `winsound`.
- Linux: install system speech/Tk packages if needed, for example `espeak` and `python3-tk`.

## Setup

```sh
git clone https://github.com/<your-username>/DODTSAM.git
cd DODTSAM

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bat
.venv\Scripts\activate
```

## Run

```sh
python object_detection_blind.py
```

Close the Matplotlib figure window or press `Ctrl+C` in the terminal to stop the app.

## Configuration

Common settings are near the top of `object_detection_blind.py`:

```python
MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 0
FOCAL = 580.0
ALARM_DIST = 80.0
CONF = 0.25
```

- Change `CAMERA_INDEX` to `1` or higher for an external camera.
- Tune `FOCAL` for better distance estimates with your own camera.
- Lower `ALARM_DIST` for fewer danger alerts, or raise it for earlier warnings.
- Adjust `KNOWN_WIDTHS` to improve distance estimates for specific object classes.

## GitHub Checklist

Before pushing:

```sh
git init
git add README.md requirements.txt .gitignore .python-version object_detection_blind.py
git commit -m "Prepare DODTSAM for GitHub"
git branch -M main
git remote add origin https://github.com/<your-username>/DODTSAM.git
git push -u origin main
```

Do not commit:

- `venv/` or `.venv/`
- downloaded model weights such as `*.pt`
- cache folders such as `__pycache__/`
- local videos, logs, or runtime outputs

If you need to version a custom trained model, use Git LFS:

```sh
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

## Troubleshooting

If the camera does not open, try changing `CAMERA_INDEX` in `object_detection_blind.py`.

If the display window does not open on Linux, install Tkinter support:

```sh
sudo apt install python3-tk
```

If speech does not work on Linux, install a speech backend:

```sh
sudo apt install espeak
```

If `ultralytics`, `cv2`, `torch`, or another module is missing, reactivate your virtual environment and reinstall:

```sh
pip install -r requirements.txt
```

## Notes

Distance values are approximate and depend on camera calibration, object orientation, and object size assumptions. This project is intended as an assistive prototype and should not be used as the only source of navigation safety.
