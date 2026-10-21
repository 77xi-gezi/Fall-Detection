"""Static YOLO detection and drawing."""
import numpy as np

from src.detector import DetectedPerson, PersonDetector
from src.utils import draw_bbox, draw_skeleton


def test_person_record():
    person = DetectedPerson(1, (10, 20, 30, 60), np.zeros((17, 3)), 0.9)
    assert person.person_id == 1
    assert person.bbox == (10, 20, 30, 60)


def test_predict_only_without_tracking(monkeypatch):
    calls = []
    models = []

    class FakeYOLO:
        def __init__(self, model_name):
            models.append(model_name)

        def predict(self, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setattr("src.detector.YOLO", FakeYOLO)
    detector = PersonDetector(device="cpu")
    assert detector.detect_image(np.zeros((50, 60, 3), dtype=np.uint8)) == []
    assert len(calls) == 1
    assert models == ["yolo26s-pose.pt"]
    assert "tracker" not in calls[0] and "persist" not in calls[0]


def test_skeleton_and_bbox():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    keypoints = np.full((17, 3), [25, 25, 0.9], dtype=np.float32)
    draw_skeleton(image, keypoints)
    draw_bbox(image, (10, 5, 40, 45), label="Person:1")
    assert image.any()


def test_low_confidence_keypoints_not_drawn():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    keypoints = np.full((17, 3), [25, 25, 0.1], dtype=np.float32)
    draw_skeleton(image, keypoints)
    assert not image.any()
