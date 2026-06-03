import unittest
from detector.config import DetectorConfig


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = DetectorConfig()
        self.assertTrue(cfg.detect_faces)
        self.assertTrue(cfg.detect_eyes)
        self.assertTrue(cfg.detect_smiles)
        self.assertIn(cfg.backend, ("mediapipe", "haar"))
        self.assertIsInstance(cfg.color_face, tuple)
        self.assertEqual(len(cfg.color_face), 3)

    def test_default_constants(self):
        cfg = DetectorConfig()
        self.assertEqual(cfg.face_scale_factor, 1.2)
        self.assertEqual(cfg.tracker_alpha, 0.35)


if __name__ == '__main__':
    unittest.main()
