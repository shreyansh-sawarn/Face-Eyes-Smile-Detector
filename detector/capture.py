"""
Video capture loop and CLI argument parser.
"""

import os
import time
import json
import argparse
import cv2
import numpy as np

from .config import DetectorConfig
from .detection import face_detection
from .hud import draw_hud
from .screenshot import make_screenshot
from .tracker import FaceTracker
from .renderer import draw_face_results
from .server import start_web_server, shared_state, HAS_FLASK

WINDOW_TITLE = 'Face · Eyes · Smile Detector'


def _fit_to_window(frame, window_title: str):
    """Resize *frame* to fill the current inner dimensions of *window_title*."""
    try:
        _x, _y, win_w, win_h = cv2.getWindowImageRect(window_title)
        if win_w > 0 and win_h > 0:
            return cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error:
        pass
    return frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time face, eyes, and smile detector using OpenCV and MediaPipe.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls\n"
            "--------\n"
            "  s       Save screenshot\n"
            "  r       Start / Stop session recording\n"
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
            'Use 0 for the internal webcam. Default: 0'
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
    parser.add_argument(
        '--web', action='store_true',
        help='Start live web dashboard server at http://localhost:5000',
    )
    return parser.parse_args()


def start_video_capturing(
    video_capture: cv2.VideoCapture,
    cfg: DetectorConfig,
    web: bool = False,
) -> None:
    """Run the main video-capture, detection, rendering, and web streaming loop."""
    screenshot_counter = 0
    paused = False
    notify_text = ""
    notify_until = 0.0
    last_face_count = 0
    annotated = None

    prev_tick = cv2.getTickCount()
    fps = 0.0

    # Session recording variables
    recording = False
    video_writer = None
    recording_timestamp = ""
    metadata_log = []
    frame_count = 0

    # Ensure directories exist
    os.makedirs(cfg.screenshot_dir, exist_ok=True)
    os.makedirs(cfg.recordings_dir, exist_ok=True)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    tracker = FaceTracker(
        alpha=cfg.tracker_alpha,
        iou_threshold=cfg.tracker_iou_threshold,
        detect_interval=cfg.tracker_detect_interval,
        max_misses=cfg.tracker_max_misses
    )

    # Initialize background web server
    if web and HAS_FLASK:
        start_web_server(cfg)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Esc key to quit
        if key == 27:
            break

        # Check if window was closed
        try:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break

        # Key bindings
        if key == ord('p'):
            paused = not paused
        elif key == ord('f'):
            cfg.detect_faces = not cfg.detect_faces
        elif key == ord('e'):
            cfg.detect_eyes = not cfg.detect_eyes
        elif key == ord('m'):
            cfg.detect_smiles = not cfg.detect_smiles
        elif key == ord('r'):
            # Start/Stop recording session
            if not recording:
                recording = True
                recording_timestamp = time.strftime("%Y%m%d_%H%M%S")
                # Resolve frame size dynamically from first grab
                width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                
                video_path = os.path.join(cfg.recordings_dir, f"session_{recording_timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                
                metadata_log = []
                frame_count = 0
                notify_text = "REC Start"
                notify_until = time.time() + 2.0
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                
                # Flush metadata sidecar JSON
                meta_path = os.path.join(cfg.recordings_dir, f"session_{recording_timestamp}.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata_log, f, indent=4)
                
                notify_text = f"Saved session: session_{recording_timestamp}"
                notify_until = time.time() + 3.0

        # Paused processing
        if paused:
            if annotated is not None:
                display = annotated.copy()
                draw_hud(display, fps, last_face_count, cfg, paused, notify_text, notify_until, recording)
                cv2.imshow(WINDOW_TITLE, _fit_to_window(display, WINDOW_TITLE))
                if web and HAS_FLASK:
                    shared_state.update_frame(display)
                    shared_state.update_stats(fps, last_face_count, recording)
            continue

        # Grab a frame
        ret, img = video_capture.read()
        if not ret or img is None:
            continue

        # FPS calculation
        now_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / max(now_tick - prev_tick, 1)
        prev_tick = now_tick

        # Grayscale for Haar backend
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Run decoupled face detection
        results, last_face_count = face_detection(bw_img, img, cfg, tracker)

        # Draw overlays on the annotated copy
        annotated = img.copy()
        draw_face_results(annotated, results, cfg)

        # Log recording frame & metadata
        if recording and video_writer:
            frame_count += 1
            # Resize frame to writer bounds if there is a mismatch
            video_writer.write(annotated)
            
            # Log sidecar frame telemetry
            metadata_log.append({
                'frame': frame_count,
                'timestamp': time.time(),
                'faces_count': len(results),
                'detections': [
                    {
                        'face_id': face.face_id,
                        'box': face.box,
                        'smile_score': face.smile_score,
                        'emotion': face.emotion
                    } for face in results
                ]
            })

        # Capture screenshot
        if key == ord('s'):
            path = make_screenshot(annotated.copy(), screenshot_counter)
            screenshot_counter += 1
            notify_text = f"Saved: {os.path.basename(path)}"
            notify_until = time.time() + 2.5

        # Render HUD onto final display frame
        display = annotated.copy()
        draw_hud(display, fps, last_face_count, cfg, paused, notify_text, notify_until, recording)
        cv2.imshow(WINDOW_TITLE, _fit_to_window(display, WINDOW_TITLE))

        # Push to Flask web server daemon
        if web and HAS_FLASK:
            shared_state.update_frame(display)
            shared_state.update_stats(fps, last_face_count, recording)

    # Cleanup
    if video_writer:
        video_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()
