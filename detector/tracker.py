"""
Temporal bounding-box smoother and tracker for face detections.
Supports backend-agnostic coordinates and tracks consistent IDs.
"""

from typing import List, Tuple, Dict, Any

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class FaceTracker:
    """Temporal EMA smoother and ID persistent tracker for face bounding boxes."""

    def __init__(
        self,
        alpha: float = 0.35,
        iou_threshold: float = 0.30,
        detect_interval: int = 1,  # Default to 1 (every frame) for MediaPipe/live feel, customizable
        max_misses: int = 6,
    ) -> None:
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.detect_interval = detect_interval
        self.max_misses = max_misses

        self._tracks: List[Dict[str, Any]] = []
        self._frame_idx: int = 0
        self._next_id: int = 1

    def process(self, raw_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
        """Match raw detections to tracks, smooth coordinates, and return (x, y, w, h, track_id)."""
        self._frame_idx += 1

        matched_track_ids = set()
        matched_det_ids = set()

        # Match raw detections to existing tracks
        for di, det in enumerate(raw_boxes):
            best_iou = self.iou_threshold
            best_tidx = -1

            for ti, track in enumerate(self._tracks):
                if ti in matched_track_ids:
                    continue
                score = _iou(det, (int(track['x']), int(track['y']), int(track['w']), int(track['h'])))
                if score > best_iou:
                    best_iou = score
                    best_tidx = ti

            if best_tidx >= 0:
                t = self._tracks[best_tidx]
                dx, dy, dw, dh = (float(v) for v in det)
                t['x'] = self.alpha * dx + (1 - self.alpha) * t['x']
                t['y'] = self.alpha * dy + (1 - self.alpha) * t['y']
                t['w'] = self.alpha * dw + (1 - self.alpha) * t['w']
                t['h'] = self.alpha * dh + (1 - self.alpha) * t['h']
                t['misses'] = 0
                matched_track_ids.add(best_tidx)
                matched_det_ids.add(di)

        # --- age unmatched existing tracks; drop the stale ones --- #
        for ti, t in enumerate(self._tracks):
            if ti not in matched_track_ids:
                t['misses'] += 1
        self._tracks = [t for t in self._tracks if t['misses'] <= self.max_misses]

        # --- unmatched detections become new tracks --- #
        for di, det in enumerate(raw_boxes):
            if di not in matched_det_ids:
                dx, dy, dw, dh = (float(v) for v in det)
                self._tracks.append({
                    'x': dx,
                    'y': dy,
                    'w': dw,
                    'h': dh,
                    'misses': 0,
                    'id': self._next_id
                })
                self._next_id += 1

        return [
            (int(t['x']), int(t['y']), int(t['w']), int(t['h']), t['id'])
            for t in self._tracks
        ]

    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks = []
        self._frame_idx = 0
        self._next_id = 1
