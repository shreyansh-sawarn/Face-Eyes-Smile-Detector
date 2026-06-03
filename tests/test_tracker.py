import unittest
from detector.tracker import FaceTracker, _iou


class TestTracker(unittest.TestCase):
    def test_iou_calculation(self):
        # Identical boxes
        box_a = (10, 10, 50, 50)
        box_b = (10, 10, 50, 50)
        self.assertEqual(_iou(box_a, box_b), 1.0)

        # No overlap
        box_c = (100, 100, 50, 50)
        self.assertEqual(_iou(box_a, box_c), 0.0)

        # Partial overlap (half overlap)
        box_d = (10, 10, 10, 10)
        box_e = (15, 10, 10, 10)
        self.assertAlmostEqual(_iou(box_d, box_e), 0.333, places=3)

    def test_tracker_assignment(self):
        tracker = FaceTracker(alpha=1.0, detect_interval=1, max_misses=3)
        
        # First frame detections
        dets_f1 = [(10, 10, 50, 50)]
        res_f1 = tracker.process(dets_f1)
        
        self.assertEqual(len(res_f1), 1)
        x, y, w, h, track_id = res_f1[0]
        self.assertEqual((x, y, w, h), dets_f1[0])
        self.assertEqual(track_id, 1)

        # Second frame detections (close box, should keep ID 1)
        dets_f2 = [(12, 11, 48, 51)]
        res_f2 = tracker.process(dets_f2)
        self.assertEqual(len(res_f2), 1)
        x, y, w, h, track_id = res_f2[0]
        self.assertEqual(track_id, 1)

        # New separate detection (should create ID 2)
        dets_f3 = [(12, 11, 48, 51), (200, 200, 40, 40)]
        res_f3 = tracker.process(dets_f3)
        self.assertEqual(len(res_f3), 2)
        ids = [r[4] for r in res_f3]
        self.assertIn(1, ids)
        self.assertIn(2, ids)

    def test_tracker_aging(self):
        tracker = FaceTracker(alpha=1.0, detect_interval=1, max_misses=2)
        
        # Start detection
        tracker.process([(10, 10, 50, 50)])
        
        # Miss 1 frame
        res = tracker.process([])
        self.assertEqual(len(res), 1)
        self.assertEqual(len(tracker._tracks), 1)
        
        # Miss 2 frames
        res = tracker.process([])
        self.assertEqual(len(res), 1)
        self.assertEqual(len(tracker._tracks), 1)
        
        # Miss 3 frames (stale, exceeds max_misses of 2)
        res = tracker.process([])
        self.assertEqual(len(res), 0)
        self.assertEqual(len(tracker._tracks), 0)


if __name__ == '__main__':
    unittest.main()
