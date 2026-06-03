"""
Temporal bounding-box smoother for Haar-cascade face detections.

Reduces flicker by combining three techniques:

1. **Detection skipping** – runs ``detectMultiScale`` only every
   *detect_interval* frames; cached results fill the gaps.
2. **IoU matching** – each new detection is paired with the closest
   existing track using Intersection-over-Union overlap.
3. **EMA smoothing** – matched track coordinates are blended toward the
   new detection with an Exponential Moving Average, eliminating jitter.

Tracks that go unmatched for more than *max_misses* consecutive frames
are aged out automatically.
"""

from __future__ import annotations

import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    """Compute Intersection-over-Union between two (x, y, w, h) boxes.

    :param a: First box as (x, y, w, h)
    :param b: Second box as (x, y, w, h)
    :return:  IoU score in [0, 1]
    :rtype:   float
    """
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


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """Temporal EMA smoother for face bounding boxes.

    :param alpha:           EMA weight for new detections.
                            0 = frozen (ignore new data),
                            1 = raw (no smoothing). Default 0.35.
    :param iou_threshold:   Minimum IoU to match a detection to a track.
    :param detect_interval: Run ``detectMultiScale`` every N frames.
                            In-between frames reuse the cached result.
    :param max_misses:      Drop a track after this many unmatched frames.
    """

    def __init__(
        self,
        alpha:           float = 0.35,
        iou_threshold:   float = 0.30,
        detect_interval: int   = 2,
        max_misses:      int   = 6,
    ) -> None:
        self.alpha           = alpha
        self.iou_threshold   = iou_threshold
        self.detect_interval = detect_interval
        self.max_misses      = max_misses

        self._tracks:   list[dict] = []
        self._frame_idx: int       = 0
        self._last_raw:  list      = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        bw_img,
        cascade:     cv2.CascadeClassifier,
        scale_factor: float,
        min_neighbors: int,
    ) -> list[tuple[int, int, int, int]]:
        """Run detection (if due) and return smoothed bounding boxes.

        :param bw_img:        Grayscale frame for ``detectMultiScale``
        :param cascade:       Loaded Haar cascade classifier
        :param scale_factor:  ``scaleFactor`` for ``detectMultiScale``
        :param min_neighbors: ``minNeighbors`` for ``detectMultiScale``
        :return: List of smoothed ``(x, y, w, h)`` integer tuples
        :rtype:  list[tuple[int, int, int, int]]
        """
        self._frame_idx += 1

        if self._frame_idx % self.detect_interval == 0:
            raw = cascade.detectMultiScale(bw_img, scale_factor, min_neighbors)
            self._last_raw = list(raw) if len(raw) > 0 else []

        return self._match_and_smooth(self._last_raw)

    def reset(self) -> None:
        """Clear all tracks (e.g. when toggling detectors off then on)."""
        self._tracks    = []
        self._last_raw  = []
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_and_smooth(
        self, raw: list
    ) -> list[tuple[int, int, int, int]]:
        """Match *raw* detections to tracks; apply EMA; age out stale tracks."""

        matched_track_ids = set()
        matched_det_ids   = set()

        # --- match each raw detection to the best existing track --- #
        for di, det in enumerate(raw):
            best_iou  = self.iou_threshold
            best_tidx = -1

            for ti, track in enumerate(self._tracks):
                if ti in matched_track_ids:
                    continue
                score = _iou(det, (track['x'], track['y'],
                                   track['w'], track['h']))
                if score > best_iou:
                    best_iou  = score
                    best_tidx = ti

            if best_tidx >= 0:
                t          = self._tracks[best_tidx]
                dx, dy, dw, dh = (float(v) for v in det)
                t['x']     = self.alpha * dx + (1 - self.alpha) * t['x']
                t['y']     = self.alpha * dy + (1 - self.alpha) * t['y']
                t['w']     = self.alpha * dw + (1 - self.alpha) * t['w']
                t['h']     = self.alpha * dh + (1 - self.alpha) * t['h']
                t['misses'] = 0
                matched_track_ids.add(best_tidx)
                matched_det_ids.add(di)

        # --- unmatched detections become new tracks --- #
        for di, det in enumerate(raw):
            if di not in matched_det_ids:
                dx, dy, dw, dh = (float(v) for v in det)
                self._tracks.append(
                    {'x': dx, 'y': dy, 'w': dw, 'h': dh, 'misses': 0}
                )

        # --- age unmatched existing tracks; drop the stale ones --- #
        for ti, t in enumerate(self._tracks):
            if ti not in matched_track_ids:
                t['misses'] += 1
        self._tracks = [t for t in self._tracks if t['misses'] <= self.max_misses]

        return [
            (int(t['x']), int(t['y']), int(t['w']), int(t['h']))
            for t in self._tracks
        ]
