"""Single-image I/O, immediate states and one actual classifier observation."""
import json
import sys

import numpy as np
import pytest
import torch

import detect_falls
from src.detector import DetectedPerson
from src.feature_extraction import FeatureExtractor
from src.image_io import read_image, write_image
from src.pipeline import FallDetectionPipeline


class FakeDetector:
    def __init__(self, persons):
        self.persons = persons

    def detect_image(self, image):
        return self.persons


class FakeLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, batch):
        self.batches.append(batch.clone())
        return torch.tensor([[-2.0, 0.0, 3.0]]).repeat(len(batch), 1)


def make_person(person_id=1, lying=False):
    keypoints = np.full((17, 3), [30, 30, 0.95], dtype=np.float32)
    keypoints[0] = [10, 40 if lying else 5, 0.95]
    keypoints[11] = [30, 40, 0.95]
    keypoints[12] = [35, 40, 0.95]
    bbox = (5, 30, 60, 45) if lying else (10, 5, 40, 45)
    return DetectedPerson(person_id, bbox, keypoints, 0.95)


def fake_pipeline(persons):
    pipeline = FallDetectionPipeline.__new__(FallDetectionPipeline)
    pipeline.device = "cpu"
    pipeline.detector = FakeDetector(persons)
    pipeline.feature_extractor = FeatureExtractor()
    pipeline.lstm = FakeLSTM()
    return pipeline


def test_unicode_image_io(tmp_path):
    path = tmp_path / "中文图片.png"
    original = np.full((50, 60, 3), 123, dtype=np.uint8)
    write_image(path, original)
    assert np.array_equal(read_image(path), original)


def test_reject_directory_empty_and_corrupt_files(tmp_path):
    with pytest.raises(ValueError, match="single image"):
        read_image(tmp_path)
    path = tmp_path / "test.png"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Empty"):
        read_image(path)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="decode"):
        read_image(path)


def test_classifies_first_image_with_one_step_and_separate_people():
    pipeline = fake_pipeline([make_person(1), make_person(2, lying=True)])
    image = np.zeros((50, 70, 3), dtype=np.uint8)
    _, persons = pipeline.classify_image(image)
    assert pipeline.lstm.batches[0].shape == (2, 1, 5)
    assert not torch.equal(pipeline.lstm.batches[0][0], pipeline.lstm.batches[0][1])
    expected = torch.softmax(torch.tensor([-2.0, 0.0, 3.0]), dim=0).tolist()
    for person in persons:
        assert person["state"] == "fallen"
        assert person["features"][2] == 0
        assert list(person["probabilities"].values()) == pytest.approx(expected)
    pipeline.classify_image(image)
    assert len(pipeline.lstm.batches) == 2
    assert pipeline.lstm.batches[1].shape == (2, 1, 5)
    assert not hasattr(pipeline, "track_manager")


def test_output_states_metadata_and_annotation(tmp_path):
    source = tmp_path / "source.png"
    write_image(source, np.zeros((50, 70, 3), dtype=np.uint8))
    output = tmp_path / "result.png"
    result = fake_pipeline([make_person()]).process_image(source, output)
    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert result == saved
    assert result["state"] == "fallen"
    assert result["persons_detected"] == 1
    assert result["inference"]["input_frames"] == 1
    assert result["inference"]["single_image_accuracy_validated"] is False
    assert read_image(output).any()


def test_no_person_does_not_invent_state_or_call_lstm(tmp_path):
    source = tmp_path / "blank.png"
    write_image(source, np.zeros((50, 70, 3), dtype=np.uint8))
    pipeline = fake_pipeline([])
    result = pipeline.process_image(source, tmp_path / "result.png")
    assert result["state"] == "no_person"
    assert result["persons"] == []
    assert not pipeline.lstm.batches


def test_reject_overwriting_source_or_json_overwriting_image(tmp_path):
    source = tmp_path / "source.png"
    write_image(source, np.zeros((50, 70, 3), dtype=np.uint8))
    pipeline = fake_pipeline([])
    with pytest.raises(ValueError, match="different files"):
        pipeline.process_image(source, source)
    with pytest.raises(ValueError, match="different files"):
        pipeline.process_image(source, tmp_path / "result.png", source)
    with pytest.raises(ValueError, match=".json"):
        pipeline.process_image(source, tmp_path / "result.png", tmp_path / "bad.txt")


def test_cli_rejects_folder_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["detect_falls.py", "--input", str(tmp_path)])
    monkeypatch.setattr(detect_falls, "FallDetectionPipeline", lambda **kwargs: pytest.fail())
    assert detect_falls.main() == 2


def test_cli_image_outputs_state_summary(tmp_path, monkeypatch, capsys):
    source = tmp_path / "photo.png"
    write_image(source, np.zeros((50, 70, 3), dtype=np.uint8))
    output = tmp_path / "result.png"
    monkeypatch.setattr(
        sys, "argv", ["detect_falls.py", "--input", str(source), "--output", str(output)],
    )
    monkeypatch.setattr(
        detect_falls, "FallDetectionPipeline", lambda **kwargs: fake_pipeline([make_person()]),
    )
    assert detect_falls.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "fallen"
    assert result["persons"][0]["state"] == "fallen"
    assert "keypoints" not in result["persons"][0]
    assert result["single_image_accuracy_validated"] is False
