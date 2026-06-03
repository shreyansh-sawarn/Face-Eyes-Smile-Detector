"""
DetectorConfig: loads settings from config.toml (or config.json) or uses defaults.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple

import cv2

# Determine standard library TOML support
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# Project root is one level above this file (which lives inside detector/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cascade(builtin_name: str, local_name: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade classifier.

    Tries OpenCV's bundled data directory first; if that fails, falls back to
    a local copy stored in the project root directory.
    """
    candidates = [
        os.path.join(cv2.data.haarcascades, builtin_name),
        os.path.join(_PROJECT_ROOT, local_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                return clf
    raise IOError(
        f"Could not load cascade classifier '{builtin_name}'. "
        f"Tried paths: {candidates}"
    )


class DetectorConfig:
    """Holds cascade classifiers, detector backends, and runtime parameters."""

    def __init__(
        self,
        detect_faces: bool = True,
        detect_eyes: bool = True,
        detect_smiles: bool = True,
    ) -> None:
        # Toggles
        self.detect_faces = detect_faces
        self.detect_eyes = detect_eyes
        self.detect_smiles = detect_smiles

        # Default Parameters
        self.backend = "mediapipe"  # "mediapipe" or "haar"
        self.face_scale_factor = 1.2
        self.face_min_neighbors = 6
        self.eye_scale_factor = 1.1
        self.eye_min_neighbors = 28
        self.smile_scale_factor = 1.7
        self.smile_min_neighbors = 28

        # Colors (BGR)
        self.color_face = (255, 0, 0)
        self.color_eye = (0, 255, 0)
        self.color_smile = (0, 0, 255)
        self.color_hud = (255, 255, 255)
        self.color_notify = (0, 220, 100)
        self.color_off = (80, 80, 80)

        # Session Recording
        self.screenshot_dir = os.path.join(_PROJECT_ROOT, "imgs")
        self.recordings_dir = os.path.join(_PROJECT_ROOT, "recordings")

        # Tracker configs
        self.tracker_alpha = 0.35
        self.tracker_iou_threshold = 0.30
        self.tracker_detect_interval = 2
        self.tracker_max_misses = 6

        # Cascade Cache (loaded lazily only if needed)
        self._face_cascade = None
        self._eye_cascade = None
        self._smile_cascade = None

        # Load configurations from file if available
        self.load_config()

    @property
    def face_cascade(self) -> cv2.CascadeClassifier:
        if self._face_cascade is None:
            self._face_cascade = _load_cascade(
                'haarcascade_frontalface_default.xml', 'haarcascade_frontface.xml')
        return self._face_cascade

    @property
    def eye_cascade(self) -> cv2.CascadeClassifier:
        if self._eye_cascade is None:
            self._eye_cascade = _load_cascade(
                'haarcascade_eye.xml', 'haarcascade_eye.xml')
        return self._eye_cascade

    @property
    def smile_cascade(self) -> cv2.CascadeClassifier:
        if self._smile_cascade is None:
            self._smile_cascade = _load_cascade(
                'haarcascade_smile.xml', 'haarcascade_smile.xml')
        return self._smile_cascade

    def load_config(self) -> None:
        """Attempt to load configuration from config.toml, config.json, or config_default.toml."""
        toml_path = os.path.join(_PROJECT_ROOT, "config.toml")
        json_path = os.path.join(_PROJECT_ROOT, "config.json")

        config_data = {}

        if os.path.exists(toml_path) and tomllib is not None:
            try:
                with open(toml_path, "rb") as f:
                    config_data = tomllib.load(f)
            except Exception as e:
                print(f"Error loading {toml_path}: {e}. Using defaults.")
        elif os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Error loading {json_path}: {e}. Using defaults.")

        if not config_data:
            return

        # Parse sections
        detection = config_data.get("detection", {})
        self.backend = detection.get("backend", self.backend)
        self.face_scale_factor = detection.get("face_scale_factor", self.face_scale_factor)
        self.face_min_neighbors = detection.get("face_min_neighbors", self.face_min_neighbors)
        self.eye_scale_factor = detection.get("eye_scale_factor", self.eye_scale_factor)
        self.eye_min_neighbors = detection.get("eye_min_neighbors", self.eye_min_neighbors)
        self.smile_scale_factor = detection.get("smile_scale_factor", self.smile_scale_factor)
        self.smile_min_neighbors = detection.get("smile_min_neighbors", self.smile_min_neighbors)

        tracker = config_data.get("tracker", {})
        self.tracker_alpha = tracker.get("alpha", self.tracker_alpha)
        self.tracker_iou_threshold = tracker.get("iou_threshold", self.tracker_iou_threshold)
        self.tracker_detect_interval = tracker.get("detect_interval", self.tracker_detect_interval)
        self.tracker_max_misses = tracker.get("max_misses", self.tracker_max_misses)

        ui = config_data.get("ui", {})
        self.screenshot_dir = ui.get("screenshot_dir", self.screenshot_dir)
        self.recordings_dir = ui.get("recordings_dir", self.recordings_dir)
        
        # Color parsing helper
        def parse_color(color_list, default):
            if isinstance(color_list, list) and len(color_list) == 3:
                return tuple(color_list)
            return default

        colors = ui.get("colors", {})
        self.color_face = parse_color(colors.get("face"), self.color_face)
        self.color_eye = parse_color(colors.get("eye"), self.color_eye)
        self.color_smile = parse_color(colors.get("smile"), self.color_smile)
        self.color_hud = parse_color(colors.get("hud"), self.color_hud)
        self.color_notify = parse_color(colors.get("notify"), self.color_notify)
        self.color_off = parse_color(colors.get("off"), self.color_off)
