"""
Core Haar-cascade detection logic.
"""

import cv2

from .config    import DetectorConfig
from .constants import (
    COLOR_FACE, COLOR_EYE, COLOR_SMILE,
    FACE_SCALE_FACTOR,  FACE_MIN_NEIGHBORS,
    EYE_SCALE_FACTOR,   EYE_MIN_NEIGHBORS,
    SMILE_SCALE_FACTOR, SMILE_MIN_NEIGHBORS,
)

# FaceTracker is optional — imported lazily to avoid circular dependency
try:
    from .tracker import FaceTracker as _FaceTracker
except ImportError:
    _FaceTracker = None  # type: ignore


def face_detection(bw_img, orig_img, cfg: DetectorConfig, tracker=None):
    """Detect faces, eyes, and smiles; draw bounding rectangles on *orig_img*.

    Face detection always runs internally to locate the ROIs needed for
    eye and smile detection. The blue face rectangle is only drawn when
    ``cfg.detect_faces`` is ``True``.

    When a :class:`~detector.tracker.FaceTracker` is supplied, face
    coordinates are smoothed across frames, dramatically reducing flicker.

    :param bw_img:   Grayscale frame used for detection
    :param orig_img: BGR frame that rectangles are drawn on (modified in-place)
    :param cfg:      :class:`DetectorConfig` controlling active detectors
    :param tracker:  Optional :class:`~detector.tracker.FaceTracker` instance
    :return: Tuple of (annotated_frame, face_count)
    :rtype: tuple[numpy.ndarray, int]
    """
    face_count = 0

    # Skip everything only when all three detectors are off
    if not cfg.detect_faces and not cfg.detect_eyes and not cfg.detect_smiles:
        if tracker is not None:
            tracker.reset()
        return orig_img, face_count

    # Use tracker for smoothed face boxes, or raw detectMultiScale if no tracker
    if tracker is not None:
        faces = tracker.process(
            bw_img, cfg.face_cascade, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS)
    else:
        faces = cfg.face_cascade.detectMultiScale(
            bw_img, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS)

    for fx, fy, fw, fh in faces:
        face_count += 1

        # Only draw the face box when the face toggle is on
        if cfg.detect_faces:
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
