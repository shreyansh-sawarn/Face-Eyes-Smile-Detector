"""
Renderer module to draw premium visual indicators and bounding boxes on frames.
"""

from typing import List, Tuple
import cv2
import numpy as np

from .config import DetectorConfig
from .types import FaceResult


def draw_rounded_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    line_type: int = cv2.LINE_AA,
    corner_radius: int = 10,
) -> None:
    """Draw a rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1

    # Keep radius within bounds
    r = min(corner_radius, abs(w) // 2, abs(h) // 2)

    if r <= 0:
        cv2.rectangle(img, pt1, pt2, color, thickness, line_type)
        return

    # Draw lines
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, line_type)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, line_type)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, line_type)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, line_type)

    # Draw arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, line_type)


def draw_filled_rounded_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    corner_radius: int = 10,
) -> None:
    """Draw a filled rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    r = min(corner_radius, abs(w) // 2, abs(h) // 2)

    if r <= 0:
        cv2.rectangle(img, pt1, pt2, color, -1)
        return

    # Draw the central shapes (body of the rectangle)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)

    # Draw the four corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1)


def draw_face_results(frame: np.ndarray, results: List[FaceResult], cfg: DetectorConfig) -> None:
    """Draw bounding boxes, facial components, and per-face HUD labels."""
    overlay = frame.copy()
    has_overlay = False

    for result in results:
        x, y, w, h = result.box

        # 1. Semi-transparent overlay inside face box
        if cfg.detect_faces:
            draw_filled_rounded_rect(overlay, (x, y), (x + w, y + h), cfg.color_face, corner_radius=12)
            has_overlay = True

    # Blend overlay with original frame for glass/semi-transparent effect
    if has_overlay:
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # 2. Draw outer borders and components
    for result in results:
        x, y, w, h = result.box

        if cfg.detect_faces:
            # Main Face Bounding Box
            draw_rounded_rect(frame, (x, y), (x + w, y + h), cfg.color_face, thickness=2, corner_radius=12)

            # Per-Face Label Badge
            # Construct text: e.g. "Face #1 | Smile: 85% [Happy]"
            smile_pct = int(result.smile_score * 100)
            smile_label = f"Smile: {smile_pct}%" if cfg.detect_smiles else ""
            emotion_label = f"[{result.emotion.upper()}]" if result.emotion else ""
            label_text = f"ID: {result.face_id}"
            
            detail_parts = []
            if smile_label:
                detail_parts.append(smile_label)
            if emotion_label:
                detail_parts.append(emotion_label)
            
            if detail_parts:
                label_text += f" | {' '.join(detail_parts)}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # Badge background and drawing
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            badge_y = max(y - 8, 15)
            badge_x = x
            
            cv2.rectangle(frame, (badge_x, badge_y - th - 6), (badge_x + tw + 10, badge_y + 4), (30, 30, 30), -1)
            cv2.rectangle(frame, (badge_x, badge_y - th - 6), (badge_x + tw + 10, badge_y + 4), cfg.color_face, 1)
            cv2.putText(frame, label_text, (badge_x + 5, badge_y - 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 3. Draw Eyes
        if cfg.detect_eyes:
            for ex, ey, ew, eh in result.eyes:
                draw_rounded_rect(frame, (ex, ey), (ex + ew, ey + eh), cfg.color_eye, thickness=1, corner_radius=4)

        # 4. Draw Smiles (if Haar backend is used or if bounding boxes exist)
        if cfg.detect_smiles and cfg.backend == "haar":
            for sx, sy, sw, sh in result.smiles:
                draw_rounded_rect(frame, (sx, sy), (sx + sw, sy + sh), cfg.color_smile, thickness=1, corner_radius=4)
        
        # Draw landmarks if using mediapipe (subtle dots for premium feel)
        if cfg.backend == "mediapipe" and result.landmarks:
            for pt in result.landmarks:
                cv2.circle(frame, pt, 1, cfg.color_smile, -1, cv2.LINE_AA)
