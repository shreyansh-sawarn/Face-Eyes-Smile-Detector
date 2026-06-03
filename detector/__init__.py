"""
detector – Face, Eyes, and Smile detection package.

Public API
----------
    from detector import DetectorConfig, face_detection
    from detector import draw_hud, make_screenshot
    from detector import parse_args, start_video_capturing
"""

from .config     import DetectorConfig
from .detection  import face_detection
from .hud        import draw_hud
from .screenshot import make_screenshot
from .capture    import parse_args, start_video_capturing
from .tracker    import FaceTracker

__all__ = [
    'DetectorConfig',
    'FaceTracker',
    'face_detection',
    'draw_hud',
    'make_screenshot',
    'parse_args',
    'start_video_capturing',
]
