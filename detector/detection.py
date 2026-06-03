"""
Detection engine supporting Haar Cascades and MediaPipe backends.
"""

import math
from typing import List, Tuple, Optional
import cv2
import numpy as np

# Try to import MediaPipe
try:
    import mediapipe as mp
    # Ensure solutions module exists (not present in some light/stub versions or Python 3.14+)
    _ = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError):
    HAS_MEDIAPIPE = False

from .config import DetectorConfig
from .types import FaceResult
from .tracker import FaceTracker


# Initialize MediaPipe solutions lazily
_mp_face_mesh = None

def get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None and HAS_MEDIAPIPE:
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _mp_face_mesh


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def process_mediapipe_mesh(
    img: np.ndarray,
    cfg: DetectorConfig,
    tracker: FaceTracker
) -> Tuple[List[FaceResult], int]:
    """Detect faces and facial attributes using MediaPipe Face Mesh."""
    face_mesh = get_face_mesh()
    if face_mesh is None:
        return [], 0

    h, w, _ = img.shape
    # MediaPipe requires RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    raw_boxes = []
    landmarks_per_face = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get bounding box from landmarks
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]

            xmin, xmax = int(min(xs) * w), int(max(xs) * w)
            ymin, ymax = int(min(ys) * h), int(max(ys) * h)

            # Pad the bounding box slightly
            pad_x = int((xmax - xmin) * 0.05)
            pad_y = int((ymax - ymin) * 0.05)
            
            xmin = max(0, xmin - pad_x)
            ymin = max(0, ymin - pad_y)
            xmax = min(w, xmax + pad_x)
            ymax = min(h, ymax + pad_y)

            box_w = xmax - xmin
            box_h = ymax - ymin
            raw_boxes.append((xmin, ymin, box_w, box_h))
            landmarks_per_face.append(face_landmarks.landmark)

    # Smooth boxes via tracker
    smoothed_boxes = tracker.process(raw_boxes)

    face_results = []
    
    # Match smoothed boxes back to landmarks to calculate metrics
    for sx, sy, sw, sh, track_id in smoothed_boxes:
        # Find closest raw box to associate landmarks
        best_idx = -1
        best_dist = 99999.0
        sc_center = (sx + sw / 2, sy + sh / 2)

        for idx, (rx, ry, rw, rh) in enumerate(raw_boxes):
            rx_center = (rx + rw / 2, ry + rh / 2)
            d = _dist(sc_center, rx_center)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        smile_score = 0.0
        emotion = "neutral"
        eyes_boxes = []
        pts_px = []

        if best_idx >= 0 and best_idx < len(landmarks_per_face):
            lms = landmarks_per_face[best_idx]
            
            # Convert normalized landmarks to pixel coordinates
            pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

            # Smile detection heuristics using mesh landmarks:
            # 61: Left mouth corner, 291: Right mouth corner
            # 33: Left eye outer corner, 263: Right eye outer corner
            # 13: Inner top lip, 14: Inner bottom lip
            p61, p291 = pts_px[61], pts_px[291]
            p33, p263 = pts_px[33], pts_px[263]
            p13, p14 = pts_px[13], pts_px[14]

            w_mouth = _dist(p61, p291)
            w_eyes = _dist(p33, p263)
            h_mouth = _dist(p13, p14)

            if w_eyes > 0:
                mouth_eye_ratio = w_mouth / w_eyes
                # Heuristic mapping for smile (typically resting ratio is ~0.72, smile is > 0.85)
                smile_score = min(max((mouth_eye_ratio - 0.72) / 0.18, 0.0), 1.0)
            
            # Simple geometric emotion rules:
            if smile_score > 0.55:
                emotion = "happy"
            elif h_mouth / max(w_mouth, 1.0) > 0.20:
                emotion = "surprised"
            else:
                # Furrowed brows estimation
                # 70: Left eyebrow inner, 300: Right eyebrow inner
                # 168: Point between eyes (nasion)
                p70, p300 = pts_px[70], pts_px[300]
                p168 = pts_px[168]
                brow_dist = _dist(p70, p300)
                if w_eyes > 0 and (brow_dist / w_eyes) < 0.20:
                    emotion = "angry"

            # Derive eye bounding boxes for visualization (if requested)
            if cfg.detect_eyes:
                # Left eye landmarks: around 33 (outer), 133 (inner)
                # Right eye landmarks: around 362 (inner), 263 (outer)
                # Let's compute bounding box around eye landmarks
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                for eye_indices in (left_eye_indices, right_eye_indices):
                    eye_pts = [pts_px[idx] for idx in eye_indices]
                    exs = [p[0] for p in eye_pts]
                    eys = [p[1] for p in eye_pts]
                    ex1, ex2 = min(exs), max(exs)
                    ey1, ey2 = min(eys), max(eys)
                    ew_pad = int((ex2 - ex1) * 0.2)
                    eh_pad = int((ey2 - ey1) * 0.2)
                    eyes_boxes.append((ex1 - ew_pad, ey1 - eh_pad, (ex2 - ex1) + 2 * ew_pad, (ey2 - ey1) + 2 * eh_pad))

        face_results.append(
            FaceResult(
                box=(sx, sy, sw, sh),
                face_id=track_id,
                eyes=eyes_boxes,
                smile_score=smile_score,
                emotion=emotion,
                landmarks=pts_px
            )
        )

    return face_results, len(smoothed_boxes)


def process_haar_cascades(
    bw_img: np.ndarray,
    img: np.ndarray,
    cfg: DetectorConfig,
    tracker: FaceTracker
) -> Tuple[List[FaceResult], int]:
    """Detect faces and components using classical OpenCV Haar Cascades."""
    # Run face detection
    raw_faces = cfg.face_cascade.detectMultiScale(
        bw_img, cfg.face_scale_factor, cfg.face_min_neighbors
    )
    raw_boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw_faces]

    # Smooth boxes
    smoothed_boxes = tracker.process(raw_boxes)

    face_results = []
    for sx, sy, sw, sh, track_id in smoothed_boxes:
        # Boundaries check
        fy, fh = max(0, sy), min(img.shape[0] - sy, sh)
        fx, fw = max(0, sx), min(img.shape[1] - sx, sw)

        roi_bw = bw_img[fy:fy + fh, fx:fx + fw]

        eyes_detected = []
        smiles_detected = []
        smile_score = 0.0
        emotion = "neutral"

        if cfg.detect_eyes and roi_bw.size > 0:
            eyes = cfg.eye_cascade.detectMultiScale(
                roi_bw, cfg.eye_scale_factor, cfg.eye_min_neighbors
            )
            for ex, ey, ew, eh in eyes:
                eyes_detected.append((sx + ex, sy + ey, ew, eh))

        if cfg.detect_smiles and roi_bw.size > 0:
            smiles = cfg.smile_cascade.detectMultiScale(
                roi_bw, cfg.smile_scale_factor, cfg.smile_min_neighbors
            )
            for sx_s, sy_s, sw_s, sh_s in smiles:
                smiles_detected.append((sx + sx_s, sy + sy_s, sw_s, sh_s))
            
            if len(smiles) > 0:
                smile_score = 1.0
                emotion = "happy"

        face_results.append(
            FaceResult(
                box=(sx, sy, sw, sh),
                face_id=track_id,
                eyes=eyes_detected,
                smiles=smiles_detected,
                smile_score=smile_score,
                emotion=emotion
            )
        )

    return face_results, len(smoothed_boxes)


def face_detection(
    bw_img: np.ndarray,
    orig_img: np.ndarray,
    cfg: DetectorConfig,
    tracker: FaceTracker
) -> Tuple[List[FaceResult], int]:
    """Unified entrypoint for detection. Returns list of FaceResult and count."""
    # Check toggles
    if not cfg.detect_faces and not cfg.detect_eyes and not cfg.detect_smiles:
        tracker.reset()
        return [], 0

    if cfg.backend == "mediapipe" and HAS_MEDIAPIPE:
        return process_mediapipe_mesh(orig_img, cfg, tracker)
    else:
        return process_haar_cascades(bw_img, orig_img, cfg, tracker)
