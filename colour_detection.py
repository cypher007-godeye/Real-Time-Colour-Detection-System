"""
Real-Time Colour Detection System
==================================
Detects 8 colours live via webcam using HSV masking.
Pipeline: Camera → Resize → Gaussian blur → HSV convert →
          inRange masks → morphological cleanup → contour filter →
          temporal smoothing → bounding box + label overlay.

Requirements:
    pip install opencv-python numpy
Usage:
    python colour_detection.py
    Press Q to quit.
"""

import cv2
import numpy as np
from collections import deque, defaultdict

# ---------------------------------------------------------------------------
# 1. COLOUR DEFINITIONS  (HSV lower/upper bounds + BGR display colour)
#    HSV in OpenCV: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255]
#    Red wraps around 0°, so it needs two ranges.
# ---------------------------------------------------------------------------
COLOURS = {
    "Red": {
        "ranges": [
            (np.array([0,  120, 70]),  np.array([10, 255, 255])),
            (np.array([170,120, 70]),  np.array([179,255, 255])),
        ],
        "bgr": (0, 0, 220),
    },
    "Green": {
        "ranges": [
            (np.array([36, 80, 60]),   np.array([86, 255, 255])),
        ],
        "bgr": (0, 200, 0),
    },
    "Blue": {
        "ranges": [
            (np.array([100,120, 70]),  np.array([130, 255, 255])),
        ],
        "bgr": (220, 60, 0),
    },
    "Yellow": {
        "ranges": [
            (np.array([22, 120, 100]), np.array([35, 255, 255])),
        ],
        "bgr": (0, 220, 220),
    },
    "Orange": {
        "ranges": [
            (np.array([10, 150, 100]), np.array([22, 255, 255])),
        ],
        "bgr": (0, 130, 255),
    },
    "Purple": {
        "ranges": [
            (np.array([130, 60, 60]),  np.array([160, 255, 255])),
        ],
        "bgr": (180, 0, 180),
    },
    "Cyan": {
        "ranges": [
            (np.array([80,  80, 80]),  np.array([100, 255, 255])),
        ],
        "bgr": (200, 200, 0),
    },
    "White": {
        "ranges": [
            (np.array([0,   0, 200]),  np.array([179, 40, 255])),
        ],
        "bgr": (160, 160, 160),
    },
}

# ---------------------------------------------------------------------------
# 2. TUNING PARAMETERS
# ---------------------------------------------------------------------------
RESIZE_W        = 640          # Processing width (height auto-scaled)
BLUR_KERNEL     = (7, 7)       # Gaussian blur kernel (odd numbers only)
MORPH_KERNEL    = np.ones((6, 6), np.uint8)
MIN_AREA        = 1800         # px² — ignore blobs smaller than this
SMOOTH_WINDOW   = 8            # Frames to majority-vote label stability
TEXT_FONT       = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE      = 0.65
TEXT_THICKNESS  = 2

# ---------------------------------------------------------------------------
# 3. HELPER: build mask for one colour entry
# ---------------------------------------------------------------------------
def build_mask(hsv_frame: np.ndarray, colour_def: dict) -> np.ndarray:
    """Union of all HSV range masks for a colour (handles red's wrap-around)."""
    combined = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lo, hi in colour_def["ranges"]:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv_frame, lo, hi))
    return combined

# ---------------------------------------------------------------------------
# 4. HELPER: morphological cleanup to reduce noise
# ---------------------------------------------------------------------------
def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Open (remove speckles) then close (fill gaps)."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    return mask

# ---------------------------------------------------------------------------
# 5. HELPER: draw a labelled bounding box on the frame
# ---------------------------------------------------------------------------
def draw_box(frame: np.ndarray, x: int, y: int, w: int, h: int,
             label: str, bgr: tuple) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
    # Semi-transparent label background
    label_text = f" {label} "
    (tw, th), baseline = cv2.getTextSize(label_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    lx, ly = x, max(y - th - baseline - 4, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx, ly), (lx + tw, ly + th + baseline + 4), bgr, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, label_text, (lx, ly + th + 2),
                TEXT_FONT, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# 6. TEMPORAL SMOOTHER — per-region label stability via majority vote
# ---------------------------------------------------------------------------
class LabelSmoother:
    """
    Keeps a short sliding window of detected labels for each spatial cell.
    Outputs the most common label in the window; suppresses flickering.
    """
    def __init__(self, window: int = SMOOTH_WINDOW):
        self.window = window
        self._buffers: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=window))

    def vote(self, cell: tuple, label: str) -> str:
        self._buffers[cell].append(label)
        buf = self._buffers[cell]
        return max(set(buf), key=buf.count)

    def grid_cell(self, cx: int, cy: int, grid: int = 64) -> tuple:
        """Snap centre to a coarse grid so nearby detections share a buffer."""
        return (cx // grid, cy // grid)

# ---------------------------------------------------------------------------
# 7. MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check device index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    smoother = LabelSmoother()

    print("[INFO] Colour detection running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame dropped.")
            continue

        # ── Resize for consistent processing speed ──────────────────────────
        h_orig, w_orig = frame.shape[:2]
        scale = RESIZE_W / w_orig
        proc = cv2.resize(frame, (RESIZE_W, int(h_orig * scale)))

        # ── Gaussian blur to suppress sensor noise ───────────────────────────
        blurred = cv2.GaussianBlur(proc, BLUR_KERNEL, 0)

        # ── BGR → HSV ────────────────────────────────────────────────────────
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ── Per-colour detection ─────────────────────────────────────────────
        for colour_name, colour_def in COLOURS.items():
            mask = build_mask(hsv, colour_def)
            mask = clean_mask(mask)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA:
                    continue                     # skip tiny false positives

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

                # Temporal smoothing via majority-vote buffer
                cell  = smoother.grid_cell(cx, cy)
                label = smoother.vote(cell, colour_name)

                draw_box(proc, x, y, w, h, label, colour_def["bgr"])

        # ── FPS overlay ──────────────────────────────────────────────────────
        cv2.putText(proc, "Colour Detector | Q to quit", (8, 22),
                    TEXT_FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Real-Time Colour Detection", proc)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
