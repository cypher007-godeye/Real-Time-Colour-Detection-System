import cv2
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# HSV Ranges for Colors
# Format: (Lower Bound, Upper Bound)
# Note: Red is split into two ranges because it wraps around the Hue spectrum (0-180)
COLOR_RANGES = {
    "Red": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (180, 255, 255))
    ],
    "Orange": [
        ((11, 100, 100), (25, 255, 255))
    ],
    "Yellow": [
        ((26, 100, 100), (34, 255, 255))
    ],
    "Green": [
        ((35, 100, 100), (85, 255, 255))
    ],
    "Cyan": [
        ((86, 100, 100), (100, 255, 255))
    ],
    "Blue": [
        ((101, 100, 100), (130, 255, 255))
    ],
    "Purple": [
        ((131, 50, 50), (159, 255, 255))
    ],
    "Magenta": [
        ((160, 100, 100), (179, 255, 255))
    ],
    "Pink": [
        ((140, 50, 100), (170, 255, 255))
    ],
    "Brown": [
        ((10, 100, 20), (20, 255, 130))
    ],
    "White": [
        ((0, 0, 200), (180, 30, 255))
    ],
    "Black": [
        ((0, 0, 0), (180, 255, 50))
    ],
    "Gray": [
        ((0, 0, 50), (180, 50, 200))
    ]
}

# BGR Colors for Labels and Bounding Boxes
COLOR_BGR = {
    "Red": (0, 0, 255),
    "Orange": (0, 165, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "Cyan": (255, 255, 0),
    "Blue": (255, 0, 0),
    "Purple": (128, 0, 128),
    "Magenta": (255, 0, 255),
    "Pink": (255, 192, 203),
    "Brown": (42, 165, 139),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Gray": (128, 128, 128)
}

# System Constants
MIN_CONTOUR_AREA = 500  # Minimum area of a detected object to avoid noise
CAMERA_INDEX = 0         # Default camera index
GAUSSIAN_BLUR_KERNEL = (5, 5)
MORPH_KERNEL = (5, 5)
MIRROR_STREAM = True     # Flip the image horizontally for a mirror-like effect
INVERT_COLORS = False    # Invert image colors (negative image)

# ==============================================================================
# CORE DETECTION LOGIC
# ==============================================================================

class ColorDetector:
    """
    Handles color detection in frames using HSV color space,
    masking, morphological operations, and contour analysis.
    """
    def __init__(self):
        self.kernel = np.ones(MORPH_KERNEL, np.uint8)

    def preprocess_frame(self, frame):
        """Apply Gaussian Blur to reduce high-frequency noise."""
        return cv2.GaussianBlur(frame, GAUSSIAN_BLUR_KERNEL, 0)

    def get_mask(self, hsv_frame, color_name):
        """
        Generate a binary mask for a specific color.
        Handles dual-range colors like Red.
        """
        ranges = COLOR_RANGES.get(color_name, [])
        mask = np.zeros(hsv_frame.shape[:2], np.uint8)

        for (lower, upper) in ranges:
            lower_arr = np.array(lower)
            upper_arr = np.array(upper)
            mask += cv2.inRange(hsv_frame, lower_arr, upper_arr)

        # Ensure binary mask (0 or 255)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return mask

    def apply_morphology(self, mask):
        """
        Use Morphological Opening and Closing to remove noise and fill holes.
        """
        # Opening: Remove small dots (noise)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # Closing: Fill small holes inside objects
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def detect_colors(self, frame):
        """
        Detects colors in the frame and returns bounding boxes and labels.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred_hsv = self.preprocess_frame(hsv_frame)

        detections = []

        for color_name in COLOR_RANGES.keys():
            mask = self.get_mask(blurred_hsv, color_name)
            mask = self.apply_morphology(mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append({
                        "color": color_name,
                        "box": (x, y, w, h),
                        "bgr": COLOR_BGR.get(color_name, (255, 255, 255))
                    })

        return detections

# ==============================================================================
# CAMERA STREAM MANAGEMENT
# ==============================================================================

class CameraStream:
    """
    Manages the OpenCV VideoCapture lifecycle.
    """
    def __init__(self, index=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {index}")

    def read(self):
        """Returns the next frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("Starting Real-Time Colour Detection System...")
    print("Press 'q' to quit.")

    try:
        stream = CameraStream()
        detector = ColorDetector()
    except Exception as e:
        print(f"Error initializing system: {e}")
        return

    while True:
        frame = stream.read()
        if frame is None:
            print("Failed to read frame from camera.")
            break

        # Invert/Mirror the stream
        if MIRROR_STREAM:
            frame = cv2.flip(frame, 1)
        if INVERT_COLORS:
            frame = cv2.bitwise_not(frame)

        detections = detector.detect_colors(frame)

        for det in detections:
            x, y, w, h = det["box"]
            color = det["bgr"]
            label = det["color"]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            cv2.rectangle(frame, (x, y - 20), (x + w, y), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Colour Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()
    print("System shut down successfully.")

if __name__ == "__main__":
    main()
