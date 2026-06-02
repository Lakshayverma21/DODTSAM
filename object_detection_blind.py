"""
╔══════════════════════════════════════════════════════════════════╗
║         DODTSAM — Dynamic Object Detection & Tracking            ║
║              System with Assistive Mobility                      ║
║                                                                  ║
║  Model      : YOLO11n (auto-downloads on first run ~5MB)         ║
║  Display    : Matplotlib (works on all Mac setups)               ║
║  TTS        : pyttsx3 (background thread, non-blocking)          ║
║  Alarm      : afplay (macOS) / winsound (Windows)                ║
║  Distance   : Pinhole camera model (calibrated for 1280px cam)   ║
║                                                                  ║
║  Run:                                                            ║
║    cd ~/Downloads/Real-Time_Object_Detection_and_Validation-main ║
║    source venv_mac/bin/activate                                  ║
║    python object_detection_blind.py                              ║
║                                                                  ║
║  Quit: Close the Figure window                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import platform
import queue
import subprocess
import sys
import threading
import time

# ── Dependency check with helpful error messages ──────────────────
try:
    import cv2
    import numpy as np
    import pyttsx3
    import torch
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use('TkAgg')          # Works on all Mac Python builds
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    pkg_map = {
        "cv2":         "opencv-python --only-binary=:all:",
        "numpy":       "numpy",
        "pyttsx3":     "pyttsx3",
        "torch":       "torch",
        "ultralytics": "ultralytics",
        "matplotlib":  "matplotlib",
    }
    missing = e.name or "unknown"
    print(f"\n[ERROR] Missing module: {missing}")
    print(f"Fix:  pip install {pkg_map.get(missing, missing)}\n")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these values to tune behaviour
# ════════════════════════════════════════════════════════════════════

MODEL_PATH   = "yolo11n.pt"   # YOLO11 nano — best speed/accuracy for CPU
CAMERA_INDEX = 0              # 0 = built-in webcam, 1 = external

# ── Distance calibration ──────────────────────────────────────────
# Formula: FOCAL = (pixel_width_of_object × known_distance_cm) / real_width_cm
# Calibrated for a 1280px wide MacBook camera:
#   Person at 100cm → ~755px wide, real shoulder width = 65cm
#   FOCAL = (755 × 100) / 65 ≈ 580 ← use this value
FOCAL        = 580.0

# Distance (cm) at which alarm triggers and box turns red
ALARM_DIST   = 80.0

# YOLO detection confidence threshold (0.0–1.0)
CONF         = 0.25

# Default object width (cm) for classes not in KNOWN_WIDTHS
DEF_WIDTH    = 30.0

# ── Speech timing ─────────────────────────────────────────────────
SPEECH_RATE  = 155            # Words per minute for TTS
ANNOUNCE_CD  = 5.0            # Seconds between repeat announcements (stationary object)
MOVE_CD      = 1.0            # Seconds between distance updates (moving object)
WARN_CD      = 1.5            # Seconds between danger warnings
MOVE_DELTA   = 5.0            # Min cm change to count as "moving"
STALE_SEC    = 3.0            # Seconds before removing a lost track

# ── Real-world widths (cm) for common COCO classes ────────────────
# Adjust these for better distance accuracy in your environment
KNOWN_WIDTHS = {
    "person":       65.0,   # shoulder width (adjust to your build)
    "bicycle":      60.0,
    "car":         180.0,
    "motorcycle":   80.0,
    "bus":         250.0,
    "truck":       250.0,
    "cat":          25.0,
    "dog":          40.0,
    "bottle":        7.0,
    "cup":           8.0,
    "chair":        45.0,
    "laptop":       33.0,
    "cell phone":    7.0,
    "book":         15.0,
    "tv":           90.0,
    "backpack":     30.0,
    "couch":       180.0,
    "bed":         190.0,
    "dining table":120.0,
    "clock":        30.0,
    "vase":         20.0,
    "teddy bear":   25.0,
    "sports ball":  22.0,
    "apple":         8.0,
    "orange":        8.0,
    "refrigerator": 75.0,
    "keyboard":     43.0,
    "mouse":         6.0,
    "remote":        5.0,
    "umbrella":     90.0,
    "handbag":      30.0,
    "suitcase":     45.0,
    "bowl":         15.0,
    "banana":        4.0,
    "pizza":        30.0,
    "cake":         25.0,
    "microwave":    50.0,
    "oven":         60.0,
    "sink":         55.0,
    "scissors":      8.0,
    "toothbrush":    2.0,
}


# ════════════════════════════════════════════════════════════════════
#  DEVICE SELECTION
# ════════════════════════════════════════════════════════════════════

def select_device():
    """Pick the best available compute device: CUDA > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ════════════════════════════════════════════════════════════════════
#  SPEAKER — Non-blocking text-to-speech
#  Runs in a background thread so voice never freezes the video feed.
#  Each detected object queues its own message independently.
# ════════════════════════════════════════════════════════════════════

class Speaker:
    def __init__(self):
        self._q    = queue.Queue(maxsize=40)   # Up to 40 queued phrases
        self._stop = threading.Event()
        # Daemon=True means this thread dies when main program exits
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        """Background thread: drains the queue and speaks each phrase."""
        engine = pyttsx3.init()
        engine.setProperty("rate", SPEECH_RATE)
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:        # Poison pill — shut down
                break
            engine.say(text)
            engine.runAndWait()
            self._q.task_done()
        engine.stop()

    def say(self, text):
        """Queue a message. Never blocks — drops silently if queue is full."""
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass                    # Skip rather than delay video

    def close(self):
        """Gracefully stop the TTS thread."""
        self._stop.set()
        try:
            self._q.put_nowait(None)    # Wake the thread to exit
        except queue.Full:
            pass


# ════════════════════════════════════════════════════════════════════
#  ALARM — Proximity beep alert
#  Uses macOS afplay (no extra library needed).
#  Runs in a background thread — beeps continuously while active.
# ════════════════════════════════════════════════════════════════════

class Alarm:
    def __init__(self):
        self._active = False
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def set(self, active: bool):
        """Turn alarm on or off from the main thread."""
        with self._lock:
            self._active = active

    def _is_active(self):
        with self._lock:
            return self._active

    def _beep(self):
        """Play one beep sound using the best available method."""
        system = platform.system()

        if system == "Darwin":      # macOS
            # Try built-in system sounds (no install needed)
            for sound in [
                "/System/Library/Sounds/Ping.aiff",
                "/System/Library/Sounds/Tink.aiff",
                "/System/Library/Sounds/Pop.aiff",
            ]:
                try:
                    subprocess.Popen(
                        ["afplay", sound],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    ).wait()
                    time.sleep(0.15)
                    return
                except Exception:
                    continue
            # Fallback: AppleScript beep
            try:
                subprocess.Popen(
                    ["osascript", "-e", "beep 1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                ).wait()
                time.sleep(0.4)
                return
            except Exception:
                pass

        elif system == "Windows":
            try:
                import winsound
                winsound.Beep(1800, 200)
                return
            except Exception:
                pass

        # Linux / last resort fallback
        print("\a", end="", flush=True)
        time.sleep(0.3)

    def _worker(self):
        """Background thread: beeps repeatedly while alarm is active."""
        while not self._stop.is_set():
            if self._is_active():
                self._beep()
            else:
                time.sleep(0.05)    # Sleep when not alarming

    def close(self):
        self._stop.set()


# ════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def clamp_box(x1, y1, x2, y2, fw, fh):
    """Clamp bounding box coordinates to stay within frame boundaries."""
    return (
        max(0, min(int(x1), fw - 1)),
        max(0, min(int(y1), fh - 1)),
        max(0, min(int(x2), fw - 1)),
        max(0, min(int(y2), fh - 1)),
    )


def get_class_name(names, class_id):
    """Safely get class name from YOLO model (handles both list and dict)."""
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    return names[class_id] if class_id < len(names) else str(class_id)


def estimate_distance(class_name, x1, x2):
    """
    Estimate distance using the pinhole camera model:
        distance = (real_width_cm × focal_length_px) / box_width_px

    Returns distance in centimeters, clamped to [1, 999] cm.
    """
    box_px   = max(1.0, float(x2 - x1))
    real_w   = KNOWN_WIDTHS.get(class_name, DEF_WIDTH)
    distance = (real_w * FOCAL) / box_px
    return round(max(1.0, min(distance, 999.0)), 1)


def detect_color(img, x1, y1, x2, y2, class_name):
    """
    Detect dominant color of an object using HSV color space.
    For people, samples the torso area to get clothing color.
    Returns a color name string.
    """
    fh, fw = img.shape[:2]
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, fw, fh)
    if x2 <= x1 or y2 <= y1:
        return "Unknown"

    bw, bh = x2 - x1, y2 - y1

    # For people: sample torso area (avoids skin affecting color)
    if class_name == "person":
        roi = img[
            y1 + int(bh * 0.25): y1 + int(bh * 0.70),
            x1 + int(bw * 0.25): x1 + int(bw * 0.75)
        ]
    else:
        # For objects: sample central 70% to avoid background bleed
        roi = img[
            y1 + int(bh * 0.15): y2 - int(bh * 0.15),
            x1 + int(bw * 0.15): x2 - int(bw * 0.15)
        ]

    if roi.size == 0:
        return "Unknown"

    # Convert to HSV for reliable color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_s = float(np.mean(hsv[:, :, 1]))   # Saturation
    mean_v = float(np.mean(hsv[:, :, 2]))   # Value (brightness)

    # Check for achromatic colors first
    if mean_v < 45:               return "Black"
    if mean_s < 35 and mean_v > 190: return "White"
    if mean_s < 35:               return "Gray"

    # Find dominant hue among saturated pixels
    saturated = hsv[:, :, 1] > 45
    if not np.any(saturated):
        return "Unknown"

    hue_values = hsv[:, :, 0][saturated]
    dominant_hue = int(np.argmax(np.bincount(
        hue_values.flatten(), minlength=180
    )))

    # Map hue angle to color name
    if dominant_hue < 10 or dominant_hue >= 170: return "Red"
    if dominant_hue < 22:  return "Orange"
    if dominant_hue < 35:  return "Yellow"
    if dominant_hue < 85:  return "Green"
    if dominant_hue < 130: return "Blue"
    if dominant_hue < 160: return "Purple"
    return "Pink"


def get_position(x1, x2, frame_width):
    """Return horizontal position of object relative to frame center."""
    cx = (x1 + x2) / 2.0
    if cx < frame_width / 3:
        return "on your left"
    if cx > frame_width * 2 / 3:
        return "on your right"
    return "ahead of you"


# ════════════════════════════════════════════════════════════════════
#  DRAWING — Render labels onto the frame
# ════════════════════════════════════════════════════════════════════

def draw_detection(frame, x1, y1, x2, y2, class_name, color,
                   distance, position, confidence, danger):
    """
    Draw bounding box and 3-line label on the frame.
    Red box + orange distance text = danger zone.
    Green box + green distance text = safe.
    """
    box_color  = (0, 0, 255)   if danger else (30, 220, 80)
    dist_color = (0, 50, 255)  if danger else (0, 255, 130)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 3 label lines drawn INSIDE the box from the top
    lx = x1 + 4
    ly = max(y1 + 24, 80)   # Clamp so text never goes above frame top

    # Line 1: Class name + confidence
    cv2.putText(frame, f"{class_name} {confidence:.0%}",
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 2: DISTANCE — large and prominent (most important for visually impaired)
    cv2.putText(frame, f"DIST:{distance:.0f}cm",
                (lx, ly + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, dist_color, 2, cv2.LINE_AA)

    # Line 3: Color + horizontal position
    cv2.putText(frame, f"{color} | {position}",
                (lx, ly + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)

    # Center dot for tracking reference
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 5, box_color, -1)


def draw_hud(frame, fps, alarm_active, n_objects):
    """Draw bottom status bar and top danger banner."""
    h, w = frame.shape[:2]

    # Bottom HUD bar
    cv2.rectangle(frame, (0, h - 26), (w, h), (20, 20, 20), -1)
    cv2.putText(frame,
        f"DODTSAM | YOLO11 | Objects:{n_objects} | "
        f"FPS:{fps:.1f} | Alarm<{ALARM_DIST:.0f}cm | Close window to quit",
        (6, h - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Top danger banner (only shown when alarm is active)
    if alarm_active:
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 180), -1)
        cv2.putText(frame,
            f"DANGER: OBJECT WITHIN {ALARM_DIST:.0f} cm",
            (8, 26),
            cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ════════════════════════════════════════════════════════════════════

def run():
    # ── Setup ────────────────────────────────────────────────────
    device = select_device()
    print(f"[DODTSAM] Using device : {device}")
    print(f"[DODTSAM] Loading YOLO11n model...")

    model = YOLO(MODEL_PATH)
    model.to(device)

    speaker = Speaker()
    alarm   = Alarm()

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.")
        speaker.close()
        alarm.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)   # Give camera time to warm up

    # ── Per-object tracking state ─────────────────────────────────
    # Key: stable string based on class + grid cell position
    # Value: dict of timing and distance state for that object
    # This ensures every object has INDEPENDENT speech timers —
    # object A's cooldown never blocks object B from being announced.
    tracked = {}

    # ── Matplotlib display setup ──────────────────────────────────
    # cv2.imshow doesn't work on all Mac setups, so we use matplotlib.
    # The frame is rendered as an image inside a matplotlib figure.
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor("black")
    ax.axis("off")
    fig.canvas.manager.set_window_title("DODTSAM — YOLO11 Object Detection")

    speaker.say("DODTSAM ready. Detection active.")
    prev_time = time.time()
    print("[DODTSAM] Running — close the Figure window to quit.")

    try:
        # Keep running while the matplotlib window is open
        while plt.fignum_exists(fig.number):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            now            = time.time()
            frame_h, frame_w = frame.shape[:2]
            fps            = 1.0 / max(now - prev_time, 0.001)
            prev_time      = now

            # ── Run YOLO11 inference ──────────────────────────────
            # model.predict() — no external tracker needed.
            # We do our own stable grid-based tracking below.
            results = model.predict(
                frame,
                conf=CONF,
                verbose=False,
                device=device,
            )

            alarm_active = False
            n_objects    = 0

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    # ── Extract detection info ────────────────────
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, frame_w, frame_h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    n_objects   += 1
                    class_id     = int(box.cls[0])
                    class_name   = get_class_name(model.names, class_id)
                    confidence   = float(box.conf[0])

                    # ── Compute properties ────────────────────────
                    distance = estimate_distance(class_name, x1, x2)
                    color    = detect_color(frame, x1, y1, x2, y2, class_name)
                    position = get_position(x1, x2, frame_w)
                    danger   = distance <= ALARM_DIST

                    if danger:
                        alarm_active = True

                    # ── Draw on frame ─────────────────────────────
                    draw_detection(frame, x1, y1, x2, y2,
                                   class_name, color, distance,
                                   position, confidence, danger)

                    # ── Stable tracking key ───────────────────────
                    # Divides frame into a grid. Objects in the same
                    # grid cell share state. This avoids needing bytetrack.
                    grid_x = int((x1 + x2) / 2) // 120
                    grid_y = int((y1 + y2) / 2) // 100
                    key    = f"{class_name}_{grid_x}_{grid_y}"

                    # Create state for new tracks
                    state = tracked.setdefault(key, {
                        "prev_dist":    None,   # Last known distance
                        "t_announce":   0.0,    # Last periodic announcement
                        "t_movement":   0.0,    # Last movement update
                        "t_warning":    0.0,    # Last danger warning
                        "first_seen":   True,   # Flag for first detection
                        "last_seen":    now,    # For stale track cleanup
                    })
                    state["last_seen"] = now
                    prev_dist = state["prev_dist"]

                    # ── Speech logic ──────────────────────────────
                    # Priority 1: Danger warning (object too close)
                    if danger:
                        if now - state["t_warning"] >= WARN_CD:
                            speaker.say(
                                f"Warning! {class_name}, "
                                f"{distance:.0f} centimeters, {position}."
                            )
                            state["t_warning"] = now

                    # Priority 2: First time this object is detected
                    elif state["first_seen"]:
                        speaker.say(
                            f"{class_name} detected. {color}. "
                            f"{distance:.0f} centimeters. {position}."
                        )
                        state["first_seen"]  = False
                        state["t_announce"]  = now

                    # Priority 3: Object is moving (distance changing)
                    # This is the key feature for visually impaired users —
                    # speaks every second as distance changes in real time.
                    elif (prev_dist is not None and
                          abs(distance - prev_dist) >= MOVE_DELTA):
                        if now - state["t_movement"] >= MOVE_CD:
                            direction = ("moving away"
                                         if distance > prev_dist
                                         else "approaching")
                            speaker.say(
                                f"{class_name} {direction}. "
                                f"{distance:.0f} centimeters."
                            )
                            state["t_movement"] = now

                    # Priority 4: Periodic reminder for stationary objects
                    elif now - state["t_announce"] >= ANNOUNCE_CD:
                        speaker.say(
                            f"{class_name}. {color}. "
                            f"{distance:.0f} centimeters. {position}."
                        )
                        state["t_announce"] = now

                    # Update stored distance for movement detection
                    state["prev_dist"] = distance

            # ── Alarm and HUD ─────────────────────────────────────
            alarm.set(alarm_active)
            draw_hud(frame, fps, alarm_active, n_objects)

            # ── Remove stale tracks ───────────────────────────────
            # Objects not seen for STALE_SEC seconds are removed
            for k in list(tracked.keys()):
                if now - tracked[k]["last_seen"] > STALE_SEC:
                    del tracked[k]

            # ── Render frame via matplotlib ───────────────────────
            # Convert BGR (OpenCV) to RGB (matplotlib) before display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.axis("off")
            ax.imshow(rgb)
            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n[DODTSAM] Interrupted by user.")

    finally:
        # ── Clean shutdown ────────────────────────────────────────
        alarm.set(False)
        cap.release()
        plt.close("all")
        speaker.close()
        alarm.close()
        print("[DODTSAM] Exited cleanly.")


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()