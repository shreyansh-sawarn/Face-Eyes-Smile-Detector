"""
Heads-up display (HUD) overlay renderer.
"""

import time
from typing import List, Tuple
import cv2
import numpy as np

from .config import DetectorConfig

# Shortcut rows shown in the bottom help panel
_SHORTCUTS = [
    [("S", "Screenshot"), ("R", "Record"), ("P", "Pause"), ("ESC", "Quit")],
    [("F", "Face"),       ("E", "Eyes"),   ("M", "Smile")],
]


def _draw_key_badge(canvas, label: str, desc: str, x: int, y: int) -> int:
    """Draw a single [KEY] Description badge and return the x position after it."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    key_scale = 0.40
    desc_scale = 0.38
    thickness = 1

    key_text = f" {label} "
    desc_text = f" {desc}  "

    (kw, kh), _ = cv2.getTextSize(key_text, font, key_scale, thickness)
    (dw, _), _ = cv2.getTextSize(desc_text, font, desc_scale, thickness)

    pad = 2
    box_top = y - kh - pad
    box_bottom = y + pad

    # Key box background (dark, slightly transparent look)
    cv2.rectangle(canvas, (x, box_top), (x + kw, box_bottom), (60, 60, 60), -1)
    cv2.rectangle(canvas, (x, box_top), (x + kw, box_bottom), (160, 160, 160), 1)

    # Key label (bright)
    cv2.putText(canvas, key_text, (x, y), font, key_scale, (230, 230, 230), thickness, cv2.LINE_AA)

    # Description text (dimmer)
    cv2.putText(canvas, desc_text, (x + kw, y), font, desc_scale, (180, 180, 180), thickness, cv2.LINE_AA)

    return x + kw + dw


def _draw_help_panel(canvas) -> None:
    """Draw the keyboard shortcut reference panel at the bottom of *canvas*."""
    h, w = canvas.shape[:2]
    row_h = 22
    margin = 10
    rows = len(_SHORTCUTS)
    panel_h = rows * row_h + margin

    # Semi-transparent dark background strip
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - panel_h - 4), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    for row_idx, row in enumerate(_SHORTCUTS):
        y = h - panel_h + row_idx * row_h + row_h - 4
        x = margin
        for key, desc in row:
            x = _draw_key_badge(canvas, key, desc, x, y)


def draw_hud(
    canvas: np.ndarray,
    fps: float,
    face_count: int,
    cfg: DetectorConfig,
    paused: bool,
    notify_text: str,
    notify_until: float,
    recording: bool = False,
) -> None:
    """Render the heads-up display onto *canvas* in-place."""
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.48
    line_h = 22
    margin = 10
    y = margin + line_h

    def put(text, x, yy, scale=small, color=cfg.color_hud, thickness=1):
        # Black stroke behind text for legibility on any background
        cv2.putText(canvas, text, (x, yy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, yy), font, scale, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Top-left: status line
    # ------------------------------------------------------------------ #
    status = "[ PAUSED ]" if paused else f"FPS: {fps:.1f}  |  Faces: {face_count}  |  Backend: {cfg.backend.upper()}"
    put(status, margin, y, scale=0.55)
    y += line_h + 6

    # ------------------------------------------------------------------ #
    # Top-left: per-detector ON/OFF toggle badges
    # ------------------------------------------------------------------ #
    for label, active in [
        ("Face  (f)", cfg.detect_faces),
        ("Eyes  (e)", cfg.detect_eyes),
        ("Smile (m)", cfg.detect_smiles),
    ]:
        color = cfg.color_notify if active else cfg.color_off
        badge = "ON " if active else "OFF"
        put(f"[{badge}] {label}", margin, y, color=color)
        y += line_h

    # ------------------------------------------------------------------ #
    # Top-right: Recording status indicator
    # ------------------------------------------------------------------ #
    if recording:
        # Flashing red dot effect based on current system time
        dot_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (50, 50, 150)
        cv2.circle(canvas, (w - 75, 20), 5, dot_color, -1, cv2.LINE_AA)
        put("REC", w - 62, 25, scale=0.5, color=(0, 0, 255), thickness=2)

    # ------------------------------------------------------------------ #
    # Bottom: keyboard shortcut panel
    # ------------------------------------------------------------------ #
    _draw_help_panel(canvas)

    # ------------------------------------------------------------------ #
    # Timed screenshot notification (above the shortcut panel)
    # ------------------------------------------------------------------ #
    if notify_text and time.time() < notify_until:
        panel_h = len(_SHORTCUTS) * line_h + margin
        put(notify_text, margin, h - panel_h - 10, scale=0.52, color=cfg.color_notify)
