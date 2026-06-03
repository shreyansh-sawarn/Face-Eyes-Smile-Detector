"""
Screenshot utility.
"""

import os

import cv2

from .constants import SCREENSHOT_DIR


def make_screenshot(img, counter: int) -> str:
    """Save the current annotated frame as a JPEG screenshot.

    The HUD overlay is intentionally excluded — call this before
    :func:`draw_hud` so screenshots contain only detection rectangles.

    :param img:     BGR frame to save
    :param counter: Sequential index used in the filename to avoid overwrites
    :return: Full filesystem path of the saved file
    :rtype: str
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOT_DIR, f'screenshot-{counter}.jpeg')
    cv2.imwrite(path, img)
    return path
