"""Independent image batches, output isolation and failure reporting."""
import csv
import json

import numpy as np
import pytest

from detect_folder import collect_images, detect_folder
from src.image_io import write_image
from tests.test_single_image import fake_pipeline, make_person


def test_independent_images_no_minimum_and_distinct_extensions(tmp_path):
    source = tmp_path / "图片"
    source.mkdir()
    image = np.zeros((50, 70, 3), dtype=np.uint8)
    write_image(source / "同名.jpg", image)
    write_image(source / "同名.png", image)
    output = tmp_path / "结果"
    pipeline = fake_pipeline([make_person()])
    summary = detect_folder(source, output, pipeline=pipeline)
    assert summary["succeeded"] == 2 and summary["failed"] == 0
    assert len(pipeline.lstm.batches) == 2
    assert all(batch.shape == (1, 1, 5) for batch in pipeline.lstm.batches)
    assert (output / "images/同名.jpg_result.png").is_file()
    assert (output / "images/同名.png_result.png").is_file()
    assert (output / "results/同名.jpg.json").is_file()
    with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["state"] == "fallen" for row in rows)
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary


def test_bad_image_recorded_and_remaining_images_processed(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    (source / "1_bad.jpg").write_bytes(b"broken")
    write_image(source / "2_good.png", np.zeros((50, 70, 3), dtype=np.uint8))
    (source / "ignore.txt").write_text("not an image")
    output = tmp_path / "results"
    summary = detect_folder(source, output, pipeline=fake_pipeline([]))
    assert summary["total_images"] == 2
    assert summary["failed"] == summary["succeeded"] == 1
    assert summary["states"]["no_person"] == 1
    with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["state"] == "error" and rows[0]["error"]
    assert rows[1]["state"] == "no_person"


def test_recursive_excludes_output_and_preserves_relative_paths(tmp_path):
    source = tmp_path / "images"
    (source / "sub").mkdir(parents=True)
    write_image(source / "sub/photo.png", np.zeros((50, 70, 3), dtype=np.uint8))
    output = source / "output"
    with pytest.raises(ValueError, match="No supported"):
        collect_images(source, output)
    pipeline = fake_pipeline([])
    for _ in range(2):
        summary = detect_folder(source, output, recursive=True, pipeline=pipeline)
        assert summary["total_images"] == 1
    assert (output / "images/sub/photo.png_result.png").is_file()


def test_model_loaded_once_and_empty_folder_rejected_first(tmp_path, monkeypatch):
    source = tmp_path / "images"
    source.mkdir()
    pipeline = fake_pipeline([])
    loads = []

    def factory(**kwargs):
        loads.append(kwargs)
        return pipeline

    monkeypatch.setattr("detect_folder.FallDetectionPipeline", factory)
    with pytest.raises(ValueError, match="No supported"):
        detect_folder(source, tmp_path / "output")
    assert not loads
    write_image(source / "one.png", np.zeros((50, 70, 3), dtype=np.uint8))
    write_image(source / "two.png", np.zeros((50, 70, 3), dtype=np.uint8))
    detect_folder(source, tmp_path / "output")
    assert len(loads) == 1


def test_input_folder_cannot_be_output(tmp_path):
    with pytest.raises(ValueError, match="Output folder"):
        collect_images(tmp_path, tmp_path)
