"""
Face, Eyes, and Smile Detector using OpenCV Haar Cascades.
Developed by Shreyansh Sawarn.

Controls
--------
  s       Save screenshot (detection rectangles included, HUD excluded)
  p       Pause / Resume
  f       Toggle face detection on/off
  e       Toggle eye detection on/off
  m       Toggle smile (mouth) detection on/off
  Esc     Quit
"""

import os
import time
import argparse

import cv2

# ---------------------------------------------------------------------------
# Detection parameters  (tune to adjust sensitivity / false-positive rate)
# ---------------------------------------------------------------------------
FACE_SCALE_FACTOR   = 1.3
FACE_MIN_NEIGHBORS  = 5

EYE_SCALE_FACTOR    = 1.1
EYE_MIN_NEIGHBORS   = 22

SMILE_SCALE_FACTOR  = 1.7
SMILE_MIN_NEIGHBORS = 22

# ---------------------------------------------------------------------------
# Drawing colours  (BGR)
# ---------------------------------------------------------------------------
COLOR_FACE   = (255,   0,   0)   # blue
COLOR_EYE    = (  0, 255,   0)   # green
COLOR_SMILE  = (  0,   0, 255)   # red
COLOR_HUD    = (255, 255, 255)   # white
COLOR_NOTIFY = (  0, 220, 100)   # bright green
COLOR_OFF    = ( 80,  80,  80)   # dim grey

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs')


# ---------------------------------------------------------------------------
# Cascade loader  (#13 – use OpenCV built-ins, fall back to local copies)
# ---------------------------------------------------------------------------
def _load_cascade(builtin_name: str, local_name: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade classifier.

    Tries OpenCV's bundled data directory first; if that fails, falls back to
    a local copy stored alongside this script.

    :param builtin_name: Filename inside ``cv2.data.haarcascades``
    :param local_name:   Filename in the same directory as this script
    :raises IOError: If neither source yields a valid classifier
    :return: Loaded CascadeClassifier
    :rtype: cv2.CascadeClassifier
    """
    candidates = [
        os.path.join(cv2.data.haarcascades, builtin_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), local_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():  # (#2) validate the load succeeded
                return clf
    raise IOError(
        f"Could not load cascade classifier '{builtin_name}'. "
        f"Tried paths: {candidates}"
    )


# ---------------------------------------------------------------------------
# Detector configuration / state  (#9 – encapsulated class, no bare globals)
# ---------------------------------------------------------------------------
class DetectorConfig:
    """Holds cascade classifiers and runtime detection toggles.

    :param detect_faces:  Start with face detection enabled
    :param detect_eyes:   Start with eye detection enabled
    :param detect_smiles: Start with smile detection enabled
    """

    def __init__(
        self,
        detect_faces:  bool = True,
        detect_eyes:   bool = True,
        detect_smiles: bool = True,
    ):
        self.face_cascade  = _load_cascade(
            'haarcascade_frontalface_default.xml', 'haarcascade_frontface.xml')
        self.eye_cascade   = _load_cascade(
            'haarcascade_eye.xml', 'haarcascade_eye.xml')
        self.smile_cascade = _load_cascade(
            'haarcascade_smile.xml', 'haarcascade_smile.xml')

        self.detect_faces  = detect_faces
        self.detect_eyes   = detect_eyes
        self.detect_smiles = detect_smiles


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------
def face_detection(bw_img, orig_img, cfg: DetectorConfig):
    """Detect faces, eyes, and smiles; draw bounding rectangles on *orig_img*.

    Detection of each feature respects the toggle flags in *cfg*.
    Eyes and smiles are only searched within each detected face region.

    :param bw_img:  Grayscale frame used for detection
    :param orig_img: BGR frame that rectangles are drawn on (modified in-place)
    :param cfg:     DetectorConfig controlling active detectors and parameters
    :return: Tuple of (annotated_frame, face_count)
    :rtype: tuple[numpy.ndarray, int]
    """
    face_count = 0

    if not cfg.detect_faces:
        return orig_img, face_count

    faces = cfg.face_cascade.detectMultiScale(   # (#10) named-constant params
        bw_img, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS)

    for fx, fy, fw, fh in faces:
        face_count += 1
        cv2.rectangle(orig_img, (fx, fy), (fx + fw, fy + fh), COLOR_FACE, 2)

        roi_bw    = bw_img[fy:fy + fh, fx:fx + fw]
        roi_color = orig_img[fy:fy + fh, fx:fx + fw]

        if cfg.detect_eyes:
            eyes = cfg.eye_cascade.detectMultiScale(
                roi_bw, EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex + ew, ey + eh), COLOR_EYE, 2)

        if cfg.detect_smiles:
            smiles = cfg.smile_cascade.detectMultiScale(
                roi_bw, SMILE_SCALE_FACTOR, SMILE_MIN_NEIGHBORS)
            for sx, sy, sw, sh in smiles:
                cv2.rectangle(roi_color, (sx, sy),
                              (sx + sw, sy + sh), COLOR_SMILE, 2)

    return orig_img, face_count


# ---------------------------------------------------------------------------
# Screenshot  (#11 – f-string)
# ---------------------------------------------------------------------------
def make_screenshot(img, counter: int) -> str:
    """Save the current annotated frame as a JPEG screenshot.

    :param img:     BGR frame to save
    :param counter: Sequential index to avoid overwriting previous screenshots
    :return: Full filesystem path of the saved file
    :rtype: str
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOT_DIR, f'screenshot-{counter}.jpeg')
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# HUD overlay  (#4)
# ---------------------------------------------------------------------------
def _draw_hud(
    canvas,
    fps: float,
    face_count: int,
    cfg: DetectorConfig,
    paused: bool,
    notify_text: str,
    notify_until: float,
) -> None:
    """Render the heads-up display onto *canvas* in-place.

    Displays FPS, face count, detector toggle states, keyboard shortcuts,
    and a timed screenshot notification.
    """
    h, _w     = canvas.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    small     = 0.48
    line_h    = 22
    margin    = 10
    y         = margin + line_h

    def put(text, x, yy, scale=small, color=COLOR_HUD, thickness=1):
        # Black outline for legibility on any background
        cv2.putText(canvas, text, (x, yy), font, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, yy), font, scale,
                    color, thickness, cv2.LINE_AA)

    # Status line
    status = "[ PAUSED ]" if paused else f"FPS: {fps:.1f}  |  Faces: {face_count}"
    put(status, margin, y, scale=0.55)
    y += line_h + 6

    # Detector toggle indicators
    for label, active in [
        ("Face  (f)", cfg.detect_faces),
        ("Eyes  (e)", cfg.detect_eyes),
        ("Smile (m)", cfg.detect_smiles),
    ]:
        color = COLOR_NOTIFY if active else COLOR_OFF
        badge = "ON " if active else "OFF"
        put(f"[{badge}] {label}", margin, y, color=color)
        y += line_h

    # Bottom shortcut bar
    put("s:screenshot  p:pause  f/e/m:toggle  Esc:quit",
        margin, h - margin, scale=0.42, color=(200, 200, 200))

    # Timed screenshot notification  (#8)
    if notify_text and time.time() < notify_until:
        put(notify_text, margin, h - margin - line_h - 4,
            scale=0.52, color=COLOR_NOTIFY)


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
def start_video_capturing(
    video_capture: cv2.VideoCapture,
    cfg: DetectorConfig,
) -> None:
    """Run the main video-capture and detection loop.

    :param video_capture: Already-opened ``cv2.VideoCapture`` object
    :param cfg:           :class:`DetectorConfig` instance
    """
    screenshot_counter = 0
    paused             = False
    notify_text        = ""
    notify_until       = 0.0
    last_face_count    = 0
    annotated          = None   # last annotated frame (used while paused)

    # High-resolution FPS tracking
    prev_tick = cv2.getTickCount()
    fps       = 0.0

    while True:
        key = cv2.waitKey(1) & 0xFF

        # ------------------------------------------------------------------ #
        # Keyboard handling
        # ------------------------------------------------------------------ #
        if key == 27:                            # Esc → quit
            break
        elif key == ord('p'):                    # (#6) pause / resume
            paused = not paused
        elif key == ord('f'):                    # (#7) toggle face
            cfg.detect_faces = not cfg.detect_faces
        elif key == ord('e'):                    # (#7) toggle eyes
            cfg.detect_eyes = not cfg.detect_eyes
        elif key == ord('m'):                    # (#7) toggle smile/mouth
            cfg.detect_smiles = not cfg.detect_smiles

        # ------------------------------------------------------------------ #
        # Paused: keep HUD refreshed without reading new frames  (#6)
        # ------------------------------------------------------------------ #
        if paused:
            if annotated is not None:
                display = annotated.copy()
                _draw_hud(display, fps, last_face_count, cfg,
                          paused, notify_text, notify_until)
                cv2.imshow('Face · Eyes · Smile Detector', display)
            continue

        # ------------------------------------------------------------------ #
        # Grab frame  (#1 – check return value)
        # ------------------------------------------------------------------ #
        ret, img = video_capture.read()
        if not ret or img is None:
            print("Warning: failed to grab frame – retrying…")
            continue

        # ------------------------------------------------------------------ #
        # FPS calculation
        # ------------------------------------------------------------------ #
        now_tick  = cv2.getTickCount()
        fps       = cv2.getTickFrequency() / max(now_tick - prev_tick, 1)
        prev_tick = now_tick

        # ------------------------------------------------------------------ #
        # Detection
        # ------------------------------------------------------------------ #
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        annotated, last_face_count = face_detection(bw_img, img, cfg)

        # ------------------------------------------------------------------ #
        # Screenshot  (#8 – save BEFORE HUD is drawn)
        # ------------------------------------------------------------------ #
        if key == ord('s'):
            path = make_screenshot(annotated.copy(), screenshot_counter)
            screenshot_counter += 1
            notify_text  = f"Saved: {os.path.basename(path)}"
            notify_until = time.time() + 2.5

        # ------------------------------------------------------------------ #
        # HUD on a display copy (so HUD is NOT baked into screenshots)  (#4)
        # ------------------------------------------------------------------ #
        display = annotated.copy()
        _draw_hud(display, fps, last_face_count, cfg,
                  paused, notify_text, notify_until)
        cv2.imshow('Face · Eyes · Smile Detector', display)

    video_capture.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI argument parsing  (#5)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed namespace with ``source``, ``no_eyes``, ``no_smiles``
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Real-time face, eyes, and smile detector using OpenCV Haar Cascades.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls\n"
            "--------\n"
            "  s       Save screenshot\n"
            "  p       Pause / Resume\n"
            "  f       Toggle face detection\n"
            "  e       Toggle eye detection\n"
            "  m       Toggle smile detection\n"
            "  Esc     Quit\n"
        ),
    )
    parser.add_argument(
        '--source', default='0',
        metavar='SOURCE',
        help=(
            'Camera index (0, 1, 2, …) or path to a video file. '
            'Use 0 for the internal webcam, 1 for an external webcam. '
            'Default: 0'
        ),
    )
    parser.add_argument(
        '--no-eyes', action='store_true',
        help='Start with eye detection disabled',
    )
    parser.add_argument(
        '--no-smiles', action='store_true',
        help='Start with smile detection disabled',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()

    # Convert source to int when it looks like a camera index
    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Build config – cascades are loaded here (#9, #13)
    cfg = DetectorConfig(
        detect_eyes   = not args.no_eyes,
        detect_smiles = not args.no_smiles,
    )

    # Open the video source  (#3 – validate before entering the loop)
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
