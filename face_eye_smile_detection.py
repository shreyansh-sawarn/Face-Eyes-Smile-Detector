"""
Face, Eyes, and Smile Detector — entry point.
Developed by Shreyansh Sawarn.

Usage
-----
    python face_eye_smile_detection.py [--source 0] [--no-eyes] [--no-smiles]

Controls:  s=screenshot  p=pause  f/e/m=toggle detectors  Esc=quit
"""

import cv2

from detector.capture import parse_args, start_video_capturing
from detector.config  import DetectorConfig


if __name__ == '__main__':
    args = parse_args()

    # Resolve source: integer camera index or video file path string
    source = int(args.source) if args.source.isdigit() else args.source

    cfg = DetectorConfig(
        detect_eyes   = not args.no_eyes,
        detect_smiles = not args.no_smiles,
    )

    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        raise RuntimeError(
            f"Could not open video source '{source}'. "
            "Check that the camera is connected or the file path is correct."
        )

    print(
        f"Detector started  |  source={source}\n"
        "Controls: s=screenshot  p=pause  f/e/m=toggle  Esc=quit"
    )
    start_video_capturing(video_capture, cfg)
