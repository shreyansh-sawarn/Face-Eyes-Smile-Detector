"""
Video capture loop and CLI argument parser.
"""

import os
import time
import argparse

import cv2

from .config     import DetectorConfig
from .detection  import face_detection
from .hud        import draw_hud
from .screenshot import make_screenshot
from .tracker    import FaceTracker

WINDOW_TITLE = 'Face · Eyes · Smile Detector'


def _fit_to_window(frame, window_title: str):
    """Resize *frame* to fill the current inner dimensions of *window_title*.

    Uses :func:`cv2.getWindowImageRect` (available since OpenCV 4.5.1) to
    query the actual display area of the window.  If the window has not been
    laid out yet, or the call fails, the original frame is returned unchanged.

    :param frame:        BGR image to resize
    :param window_title: Name passed to :func:`cv2.namedWindow`
    :return: Resized frame (or the original if sizing is unavailable)
    :rtype: numpy.ndarray
    """
    try:
        _x, _y, win_w, win_h = cv2.getWindowImageRect(window_title)
        if win_w > 0 and win_h > 0:
            return cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error:
        pass
    return frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Namespace with ``source``, ``no_eyes``, and ``no_smiles``
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


def start_video_capturing(
    video_capture: cv2.VideoCapture,
    cfg: DetectorConfig,
) -> None:
    """Run the main video-capture and detection loop.

    Key bindings:

    ======  =====================================
    Key     Action
    ======  =====================================
    ``s``   Save screenshot to ``imgs/``
    ``p``   Pause / Resume
    ``f``   Toggle face detection on/off
    ``e``   Toggle eye detection on/off
    ``m``   Toggle smile (mouth) detection on/off
    Esc     Quit
    ======  =====================================

    :param video_capture: Already-opened :class:`cv2.VideoCapture` object
    :param cfg:           :class:`DetectorConfig` instance
    """
    screenshot_counter = 0
    paused             = False
    notify_text        = ""
    notify_until       = 0.0
    last_face_count    = 0
    annotated          = None   # last annotated frame, used to redraw while paused

    prev_tick = cv2.getTickCount()
    fps       = 0.0

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    tracker = FaceTracker()

    while True:
        key = cv2.waitKey(1) & 0xFF

        # ------------------------------------------------------------------ #
        # Keyboard handling
        # ------------------------------------------------------------------ #
        if key == 27:                        # Esc → quit
            break

        # Exit if the user closed the window with the X button.
        # Must come AFTER waitKey() — that call is what processes the OS
        # WM_DESTROY event. WND_PROP_AUTOSIZE returns -1.0 on a destroyed
        # window and is more reliable than WND_PROP_VISIBLE on Windows.
        try:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break

        if key == ord('p'):                  # pause / resume
            paused = not paused
        elif key == ord('f'):                # toggle face detection
            cfg.detect_faces = not cfg.detect_faces
        elif key == ord('e'):                # toggle eye detection
            cfg.detect_eyes = not cfg.detect_eyes
        elif key == ord('m'):                # toggle smile detection
            cfg.detect_smiles = not cfg.detect_smiles

        # ------------------------------------------------------------------ #
        # Paused: keep HUD refreshed without reading new frames
        # ------------------------------------------------------------------ #
        if paused:
            if annotated is not None:
                display = annotated.copy()
                draw_hud(display, fps, last_face_count, cfg,
                         paused, notify_text, notify_until)
                cv2.imshow(WINDOW_TITLE, _fit_to_window(display, WINDOW_TITLE))
            continue

        # ------------------------------------------------------------------ #
        # Grab frame
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
        annotated, last_face_count = face_detection(bw_img, img, cfg, tracker)

        # ------------------------------------------------------------------ #
        # Screenshot (saved BEFORE HUD is drawn so HUD stays off screenshots)
        # ------------------------------------------------------------------ #
        if key == ord('s'):
            path = make_screenshot(annotated.copy(), screenshot_counter)
            screenshot_counter += 1
            notify_text  = f"Saved: {os.path.basename(path)}"
            notify_until = time.time() + 2.5

        # ------------------------------------------------------------------ #
        # Render HUD onto a display copy, then show
        # ------------------------------------------------------------------ #
        display = annotated.copy()
        draw_hud(display, fps, last_face_count, cfg,
                 paused, notify_text, notify_until)
        cv2.imshow(WINDOW_TITLE, _fit_to_window(display, WINDOW_TITLE))

    video_capture.release()
    cv2.destroyAllWindows()
