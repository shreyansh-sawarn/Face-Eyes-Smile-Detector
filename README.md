# Face Eyes Smile Detector

A python based project to detect an user's face, eyes and smile. It uses Haar-like features and the open source computer vision library, OpenCV.
Users can detect face, eyes and smile using their webcam and screenshot using **`s`**-key on your keyboard and save it into the `imgs` folder.

### Steps for running the project:
1. Clone the repository using the command 'git clone '
2. Install dependencies/requirements using the command 'pip install opencv-python' which installs the packages numpy and opencv-python.
3. Run the project using the command 'python face_eye_smile_detection.py'

If you don't have an internal webcam and you use an external webcam, change the parameter of the `cv2.VideoCapture()` function on line 90 in 'face_eye_smile_detection.py' from `0` to `1` like so:

```python
# 0 = internal webcam, 1 = external webcam
VIDEO_CAPTURE = cv2.VideoCapture(1)
```

To close and exit the webcam video stream press the **`Esc`**-key on your keyboard while the window of your webcam video stream is active.
