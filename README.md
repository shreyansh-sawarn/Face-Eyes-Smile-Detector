# Face Eyes Smile Detector

A Python project for real-time face, eyes, and smile detection using [OpenCV](https://opencv.org/) Haar Cascade classifiers. It opens a webcam (or video file) and draws bounding rectangles around detected faces (blue), eyes (green), and smiles (red).

---

## Requirements

- Python 3.10+
- OpenCV (includes NumPy as a dependency)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the project

### Default (internal webcam)

```bash
python face_eye_smile_detection.py
```

### External webcam

```bash
python face_eye_smile_detection.py --source 1
```

### Video file

```bash
python face_eye_smile_detection.py --source path/to/video.mp4
```

### Start with specific detectors disabled

```bash
python face_eye_smile_detection.py --no-eyes --no-smiles
```

### All CLI options

```
usage: face_eye_smile_detection.py [-h] [--source SOURCE] [--no-eyes] [--no-smiles]

options:
  -h, --help       Show this help message and exit
  --source SOURCE  Camera index (0, 1, …) or path to a video file. Default: 0
  --no-eyes        Start with eye detection disabled
  --no-smiles      Start with smile detection disabled
```

---

## Controls

| Key   | Action                          |
|-------|---------------------------------|
| `s`   | Save screenshot to `imgs/`      |
| `p`   | Pause / Resume                  |
| `f`   | Toggle face detection on/off    |
| `e`   | Toggle eye detection on/off     |
| `m`   | Toggle smile detection on/off   |
| `Esc` | Quit                            |

Screenshots are saved to the `imgs/` folder as `screenshot-0.jpeg`, `screenshot-1.jpeg`, etc. A brief on-screen notification confirms each save.

---

## HUD overlay

The live video feed includes a heads-up display showing:

- **FPS** and **number of faces** currently detected
- **Toggle status** of each detector (ON / OFF)
- **Keyboard shortcut** reference bar at the bottom

The HUD is shown on-screen only — screenshots are saved without the HUD so they stay clean.

---

## Detection colours

| Feature | Rectangle colour |
|---------|-----------------|
| Face    | 🔵 Blue          |
| Eyes    | 🟢 Green         |
| Smile   | 🔴 Red           |

---

## Cascade classifiers

The detector uses OpenCV's bundled Haar cascade files by default (via `cv2.data.haarcascades`). The local `.xml` files in this repository are kept as a fallback if the built-in path is unavailable.
