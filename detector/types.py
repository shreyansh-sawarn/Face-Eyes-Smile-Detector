"""
Data structures for face detection results.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class FaceResult:
    """Represents a single detected face and its components."""
    box: Tuple[int, int, int, int]  # (x, y, w, h) absolute coordinates in frame
    face_id: int
    eyes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h) absolute coordinates
    smiles: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h) absolute coordinates
    smile_score: float = 0.0  # smile confidence metric in range 0.0 to 1.0
    emotion: str = "neutral"  # classified emotion/expression
    landmarks: List[Tuple[int, int]] = field(default_factory=list)  # key facial landmarks (if available)
