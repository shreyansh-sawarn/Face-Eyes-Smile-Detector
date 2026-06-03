"""
DetectorConfig: cascade classifiers and runtime detection toggles.
"""

import os

import cv2

# Project root is one level above this file (which lives inside detector/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cascade(builtin_name: str, local_name: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade classifier.

    Tries OpenCV's bundled data directory first; if that fails, falls back to
    a local copy stored in the project root directory.

    :param builtin_name: Filename inside ``cv2.data.haarcascades``
    :param local_name:   Filename in the project root directory
    :raises IOError: If neither source yields a valid classifier
    :return: Loaded CascadeClassifier
    :rtype: cv2.CascadeClassifier
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
    """Holds cascade classifiers and runtime detection toggles.

    On construction the three Haar cascade classifiers are loaded once.
    The boolean toggle flags can be flipped freely at runtime (e.g. via
    keyboard shortcuts) without reloading the classifiers.

    :param detect_faces:  Start with face detection enabled
    :param detect_eyes:   Start with eye detection enabled
    :param detect_smiles: Start with smile detection enabled
    """

    def __init__(
        self,
        detect_faces:  bool = True,
        detect_eyes:   bool = True,
        detect_smiles: bool = True,
    ) -> None:
        self.face_cascade  = _load_cascade(
            'haarcascade_frontalface_default.xml', 'haarcascade_frontface.xml')
        self.eye_cascade   = _load_cascade(
            'haarcascade_eye.xml', 'haarcascade_eye.xml')
        self.smile_cascade = _load_cascade(
            'haarcascade_smile.xml', 'haarcascade_smile.xml')

        self.detect_faces  = detect_faces
        self.detect_eyes   = detect_eyes
        self.detect_smiles = detect_smiles
