# 🎨 Air Canvas — Draw in the Air with Hand Gestures

A real-time computer vision application that lets you **draw on a virtual canvas using hand gestures** captured through your webcam. Built with **Python**, **OpenCV**, and **MediaPipe**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

---

## ✨ Features

- **Real-time hand tracking** — MediaPipe detects 21 hand landmarks at high FPS
- **Gesture-based drawing** — Raise your index finger to draw, no mouse needed
- **Color palette** — 9 vibrant colors to choose from, select by hovering
- **Eraser mode** — Open your palm to erase parts of the drawing
- **Clear canvas** — Hold thumb + pinky gesture to clear everything
- **Navigate mode** — Raise index + middle finger to move without drawing
- **Adjustable brush size** — Use `+`/`-` keys to resize the brush
- **Undo support** — Press `z` to undo strokes
- **Save your art** — Press `s` to export as PNG
- **Smooth drawing** — Line interpolation between frames for clean strokes
- **Mirror mode** — Frame is flipped for natural, intuitive interaction

---

## 🖐️ Gesture Guide

| Gesture | Fingers | Action |
|---|---|---|
| ☝️ **Index finger only** | Index up, others down | ✏️ **Draw** — trace lines on canvas |
| ✌️ **Peace sign** | Index + Middle up | 🖱️ **Navigate** — move cursor without drawing |
| 🖐️ **Open palm** | All five fingers up | 🧹 **Erase** — circular eraser around palm |
| ✊ **Fist** | No fingers up | ⏸️ **Idle** — pause, do nothing |
| 🤙 **Hang loose** | Thumb + Pinky up | 🗑️ **Clear** — hold for 0.6s to clear canvas |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `q` | Quit the application |
| `c` | Clear the entire canvas |
| `+` / `=` | Increase brush size |
| `-` | Decrease brush size |
| `s` | Save canvas as PNG |
| `z` | Undo last stroke |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8 or higher
- A webcam

### Install Dependencies

```bash
# Navigate to the project folder
cd "AirCanvas-HandDraw"

# Install required packages
pip install -r requirements.txt
```

### Run the Application

```bash
python air_canvas.py
```

---

## 🏗️ Project Structure

```
AirCanvas-HandDraw/
├── air_canvas.py        # Main application (all classes + entry point)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

### Architecture

| Class | Responsibility |
|---|---|
| `HandDetector` | Wraps MediaPipe Hands — detects landmarks, counts fingers, classifies gestures |
| `Canvas` | Manages the drawing buffer (NumPy array) — draw, erase, clear, undo, overlay |
| `ColorPalette` | Renders the color selection bar — handles hover-based selection |
| `AirCanvasApp` | Main loop — capture → detect → gesture logic → draw → composite → display |

---

## 📸 How It Works

1. **Capture** — OpenCV grabs frames from the webcam and flips horizontally
2. **Detect** — MediaPipe Hands finds 21 landmark points on the hand
3. **Classify** — Finger positions are analyzed to determine the gesture
4. **Act** — Based on the gesture, draw lines / erase / navigate / clear
5. **Composite** — The drawing canvas (black buffer) is overlaid on the live feed
6. **Display** — The final composited frame with UI elements is shown

---

## 🛠️ Customization

You can adjust these parameters in `air_canvas.py`:

```python
# In AirCanvasApp.__init__()
self.brush_size = 8          # Default brush thickness (pixels)
self.eraser_radius = 40      # Eraser circle radius
self.min_brush = 2           # Minimum brush size
self.max_brush = 30          # Maximum brush size

# In HandDetector.__init__()
detection_conf = 0.7         # Hand detection confidence threshold
tracking_conf = 0.7          # Hand tracking confidence threshold
```

---

## 📝 License

This project is open source and available for educational and personal use.

---

*Built using Python, OpenCV, and MediaPipe*
