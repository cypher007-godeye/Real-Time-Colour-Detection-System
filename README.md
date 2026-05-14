# Real-Time Colour Detection System

A computer vision-based system that detects and identifies colors in real-time using a live camera feed. The system identifies target colors, draws precise bounding boxes around detected objects, and displays corresponding color labels.

## 🚀 Features
- **Real-time Detection**: High-performance color masking and contour analysis.
- **Robustness**: Uses HSV color space to minimize the impact of lighting variations.
- **Noise Reduction**: Implements Gaussian Blur and Morphological operations (Opening/Closing) to eliminate flickering and artifacts.
- **Multi-Color Support**: Pre-configured ranges for Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta, Pink, Brown, White, Black, and Gray.
- **Calibration Tool**: Includes a web-based color tester to easily find and verify OpenCV-compatible HSV values.

## 🛠️ System Architecture
The system follows a linear processing pipeline:
**Camera Input** $\rightarrow$ **Frame Pre-processing** $\rightarrow$ **HSV Masking** $\rightarrow$ **Noise Filtering** $\rightarrow$ **Contour Detection** $\rightarrow$ **Bounding Box & Labeling**

### Process Flow Diagram
```text
+-----------+       +------------+       +------------+       +-------------+       +------------+
|  Camera   | ----> |    Frame    | ----> |  Detection  | ----> | Bounding Box | ----> |    Label    |
|   Input   |       | Pre-process |       |   (HSV)     |       |  Calculation  |       |  Display    |
+-----------+       +------------+       +------------+       +-------------+       +------------+
                                               ^
                                               |
                                     +-------------------+
                                     |   Noise Reduction |
                                     | (Blur & Morphology)|
                                     +-------------------+
```

## 📚 Technical Details

### 1. HSV Color Space
The system converts BGR frames into **HSV (Hue, Saturation, Value)**.
- **Hue**: The color type.
- **Saturation**: The intensity of the color.
- **Value**: The brightness.
HSV is used because it separates color information from lighting, making detection more stable across different environments.

### 2. Stability & Noise Reduction
To ensure smooth performance and prevent "flickering" detections:
- **Gaussian Blur**: Smoothes the frame to remove high-frequency noise.
- **Morphological Opening**: Removes small white noise dots from the mask.
- **Morphological Closing**: Fills small holes within detected objects.
- **Area Filtering**: Only contours larger than a specific pixel threshold are processed, ignoring tiny artifacts.

## 📁 Project Structure
- `colour_detection.py`: The main application containing the detector logic, camera stream management, and visualization loop.
- `color_tester.html`: A handy utility to pick a color and get its corresponding OpenCV HSV values for calibration.
- `README.md`: Project documentation.

## 🏁 Getting Started

### Prerequisites
- Python 3.x
- OpenCV
- NumPy

### Installation
```bash
pip install opencv-python numpy
```

### Running the System
```bash
python colour_detection.py
```
*Press **'q'** to quit the application.*

### Using the Calibration Tool
Simply open `color_tester.html` in any modern web browser to pick colors and find their HSV values to tune the `COLOR_RANGES` in `colour_detection.py`.
