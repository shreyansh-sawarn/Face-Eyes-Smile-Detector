"""
Shared constants for the detector package.

All tuneable values live here — adjust sensitivity, colours, or paths
without touching any detection or rendering logic.
"""

import os

# ---------------------------------------------------------------------------
# Detection parameters  (tune to adjust sensitivity / false-positive rate)
# ---------------------------------------------------------------------------
FACE_SCALE_FACTOR   = 1.2
FACE_MIN_NEIGHBORS  = 6

EYE_SCALE_FACTOR    = 1.1
EYE_MIN_NEIGHBORS   = 28

SMILE_SCALE_FACTOR  = 1.7
SMILE_MIN_NEIGHBORS = 28

# ---------------------------------------------------------------------------
# Drawing colours  (BGR)
# ---------------------------------------------------------------------------
COLOR_FACE   = (255,   0,   0)   # blue
COLOR_EYE    = (  0, 255,   0)   # green
COLOR_SMILE  = (  0,   0, 255)   # red
COLOR_HUD    = (255, 255, 255)   # white
COLOR_NOTIFY = (  0, 220, 100)   # bright green
COLOR_OFF    = ( 80,  80,  80)   # dim grey

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root is one level above this file (which lives inside detector/)
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_DIR = os.path.join(_PROJECT_ROOT, 'imgs')
