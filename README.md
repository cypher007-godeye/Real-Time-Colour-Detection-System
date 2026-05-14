# 🎨 Real-Time Colour Detection System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

> **"Teaching a computer to see colors, one pixel at a time."** 🌈

This isn't just another OpenCV project. It's a high-precision, real-time color sensing engine that turns your webcam into a scientific instrument. Whether you're sorting M&Ms or building a color-tracking robot, this system has you covered.

---

## 🚀 The "Magic" Features

- ⚡ **Blazing Fast**: Optimized pipeline for smooth, real-time performance.
- 🌈 **Spectrum Master**: Pre-configured for **13+ colors**, from the deep blues of the ocean to the neon pinks of a 90s arcade.
- 🛠️ **Calibration Suite**: Includes a dedicated web-based tool to find the exact HSV "sweet spot" for any color.
- 🛡️ **Anti-Flicker Shield**: Advanced noise reduction that stops bounding boxes from dancing around the screen.
- 🎯 **Smart Labeling**: Dynamic bounding boxes that follow objects as they move.

---

## 🗺️ The Logic Journey (Flowchart)

Here is how a single frame travels from your camera to your screen:

```text
  [ 📷 LIVE CAMERA ]
          |
          v
  [ 🌫️ GAUSSIAN BLUR ]  <--- "The Squint": Removes grainy noise
          |
          v
  [ 🎨 HSV CONVERSION ] <--- BGR -> HSV (The secret to lighting independence)
          |
          v
  [ 🎭 BINARY MASKING ]  <--- "Is it the right color?" (Yes=White / No=Black)
          |
          v
  [ 🧹 MORPHOLOGY ]      <--- Opening (Removes dots) & Closing (Fills holes)
          |
          v
  [ 🔍 CONTOUR FINDER ]  <--- Grouping pixels into solid objects
          |
          v
  [ 📏 AREA FILTERING ]  <--- "Too small to be real?" -> Toss it!
          |
          v
  [ 🖼️ VISUAL OVERLAY ]  <--- Draw Bounding Box + Color Label
          |
          v
  [ 🖥️ FINAL DISPLAY ]   <--- Result: "Look! A Red Ball!"
```

---

## 🧪 The Secret Sauce (Technical Deep Dive)

### 1. Why HSV and not RGB? 🤯
RGB (Red-Green-Blue) is how screens *show* color, but it's terrible for *detecting* it. If you dim the lights, the RGB values change completely.
**HSV (Hue, Saturation, Value)** separates the "what" (Hue) from the "how bright" (Value). This means the system recognizes "Yellow" whether it's under a bright lamp or in a shadow.

### 2. The "Silly" Small Details 🤏
- **The Red Rebel**: Did you know Red is the only color that lives in *two places* on the hue wheel? It's at the very start (0°) and the very end (180°). We use two separate masks just to keep Red happy.
- **The Mirror Effect**: We flip the camera horizontally. Why? Because if you move your hand to the right and the screen shows it moving left, you'll feel like you're in a glitchy parallel dimension.
- **Morphology = Digital Eraser**: We use "Opening" and "Closing." Think of "Opening" as an eraser that rubs out tiny white specks, and "Closing" as a glue that fills in the gaps.

---

## 📁 Project Map

| File | Role | Personality |
| :--- | :--- | :--- |
| `colour_detection.py` | 🚀 Core Engine | The "Brain". Does all the heavy lifting and math. |
| `color_tester.html` | 🎨 Calibration Tool | The "Eye". Helps you pick the perfect colors. |
| `README.md` | 📖 Documentation | The "Guide". Explains everything you see here. |

---

## 🏁 Getting Started

### 🛠️ Prerequisites
- **Python 3.x**
- **OpenCV** (`pip install opencv-python`)
- **NumPy** (`pip install numpy`)

### 🚀 Launch Sequence
```bash
# 1. Install the essentials
pip install opencv-python numpy

# 2. Ignition!
python colour_detection.py
```
> **Pro Tip**: Press **'q'** to escape the matrix and shut down the system.

---

## ⚙️ Tuning the Beast

### Adding a New Color
1. Open `color_tester.html` $\rightarrow$ Pick a color $\rightarrow$ Copy the **OpenCV HSV**.
2. Go to `colour_detection.py` $\rightarrow$ `COLOR_RANGES` $\rightarrow$ Add your color!
   ```python
   "ElectricLime": [((50, 100, 100), (70, 255, 255))]
   ```

### Fighting Noise
- **Too many tiny boxes?** Bump up `MIN_CONTOUR_AREA` to `1000` or `2000`.
- **Boxes flickering?** Try a larger `GAUSSIAN_BLUR_KERNEL` like `(9, 9)`.

---

## 🗺️ Future Roadmap
- [ ] 🖼️ **Custom GUI**: Change colors without touching the code.
- [ ] 📊 **Color Logger**: Export detected color timelines to CSV.
- [ ] 🤖 **AI Upgrade**: Use YOLOv8 for "Object + Color" detection (e.g., "Red Apple" instead of just "Red").
- [ ] 📱 **Mobile App**: Port the logic to Kivy or Flutter.

## 📜 License
This project is licensed under the **MIT License**. Go wild, fork it, and make it your own!
