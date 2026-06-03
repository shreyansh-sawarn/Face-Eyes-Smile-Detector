"""
Heads-up display (HUD) overlay renderer.
"""

import time

import cv2
import numpy as np

from .config    import DetectorConfig
from .constants import COLOR_HUD, COLOR_NOTIFY, COLOR_OFF

# Shortcut rows shown in the bottom help panel
_SHORTCUTS = [
    [("S", "Screenshot"), ("P", "Pause"),      ("ESC", "Quit")],
    [("F", "Face"),       ("E", "Eyes"),        ("M",   "Smile")],
]


def _draw_key_badge(canvas, label: str, desc: str, x: int, y: int) -> int:
    """Draw a single [KEY] Description badge and return the x position after it.

    :param canvas: BGR frame to draw on (modified in-place)
    :param label:  Key character(s) shown inside brackets, e.g. ``"S"``
    :param desc:   Description shown after the bracket, e.g. ``"Screenshot"``
    :param x:      Left edge x coordinate
    :param y:      Baseline y coordinate
    :return:       x position directly after this badge (for chaining)
    :rtype:        int
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    key_scale  = 0.40
    desc_scale = 0.38
    thickness  = 1

    key_text  = f" {label} "
    desc_text = f" {desc}  "

    (kw, kh), _ = cv2.getTextSize(key_text,  font, key_scale,  thickness)
    (dw, _),  _ = cv2.getTextSize(desc_text, font, desc_scale, thickness)

    pad = 2
    box_top    = y - kh - pad
    box_bottom = y + pad

    # Key box background (dark, slightly transparent look)
    cv2.rectangle(canvas, (x, box_top), (x + kw, box_bottom),
                  (60, 60, 60), -1)
    cv2.rectangle(canvas, (x, box_top), (x + kw, box_bottom),
                  (160, 160, 160), 1)

    # Key label (bright)
    cv2.putText(canvas, key_text, (x, y), font, key_scale,
                (230, 230, 230), thickness, cv2.LINE_AA)

    # Description text (dimmer)
    cv2.putText(canvas, desc_text, (x + kw, y), font, desc_scale,
                (180, 180, 180), thickness, cv2.LINE_AA)

    return x + kw + dw


def _draw_help_panel(canvas) -> None:
    """Draw the keyboard shortcut reference panel at the bottom of *canvas*.

    :param canvas: BGR frame to draw on (modified in-place)
    """
    h, w    = canvas.shape[:2]
    row_h   = 22
    margin  = 10
    rows    = len(_SHORTCUTS)
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
    canvas,
    fps: float,
    face_count: int,
    cfg: DetectorConfig,
    paused: bool,
    notify_text: str,
    notify_until: float,
) -> None:
    """Render the heads-up display onto *canvas* in-place.

    Displays FPS, detected face count, per-detector toggle states, a
    keyboard shortcut reference panel, and a timed screenshot notification.

    :param canvas:       BGR frame to draw on (modified in-place)
    :param fps:          Current frames per second
    :param face_count:   Number of faces detected in the current frame
    :param cfg:          :class:`DetectorConfig` for reading toggle states
    :param paused:       Whether the capture loop is currently paused
    :param notify_text:  Short message to flash on screen (e.g. after screenshot)
    :param notify_until: ``time.time()`` timestamp at which *notify_text* expires
    """
    h, _w  = canvas.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    small  = 0.48
    line_h = 22
    margin = 10
    y      = margin + line_h

    def put(text, x, yy, scale=small, color=COLOR_HUD, thickness=1):
        # Black stroke behind text for legibility on any background
        cv2.putText(canvas, text, (x, yy), font, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, yy), font, scale,
                    color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Top-left: status line
    # ------------------------------------------------------------------ #
    status = "[ PAUSED ]" if paused else f"FPS: {fps:.1f}  |  Faces: {face_count}"
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
        color = COLOR_NOTIFY if active else COLOR_OFF
        badge = "ON " if active else "OFF"
        put(f"[{badge}] {label}", margin, y, color=color)
        y += line_h

    # ------------------------------------------------------------------ #
    # Bottom: keyboard shortcut panel
    # ------------------------------------------------------------------ #
    _draw_help_panel(canvas)

    # ------------------------------------------------------------------ #
    # Timed screenshot notification (above the shortcut panel)
    # ------------------------------------------------------------------ #
    if notify_text and time.time() < notify_until:
        panel_h = len(_SHORTCUTS) * line_h + margin
        put(notify_text, margin, h - panel_h - 10,
            scale=0.52, color=COLOR_NOTIFY)
