"""Temporal continuity, real MP4 encoding, asynchronous jobs and cancellation."""
import io
import json
import threading
import time
import zipfile

import cv2
import numpy as np
import pytest

import src.video_web
import web_app
from src.video_pipeline import VideoCancelled, VideoSession, inspect_video, process_video
from tests.test_single_image import fake_pipeline, make_person
from tests.test_web_app import image_upload


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "原始视频.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (70, 50))
    assert writer.isOpened()
    for index in range(34):
        writer.write(np.full((50, 70, 3), index * 5, np.uint8))
    writer.release()
    return path


def test_real_windows_start_at_frame_30_and_reset_on_gap():
    pipeline = fake_pipeline([make_person()])
    session = VideoSession(pipeline)
    image = np.zeros((50, 70, 3), np.uint8)
    for index in range(29):
        _, people = session.process_frame(image, index)
        assert people[0]["state"] == "collecting"
    assert not pipeline.lstm.batches
    pipeline.detector.persons[0].keypoints[11:13, 1] += 2
    _, people = session.process_frame(image, 29)
    assert people[0]["state"] == "fallen"
    sequence = pipeline.lstm.batches[-1]
    assert sequence.shape == (1, 30, 5)
    assert sequence[0, -1, 2] > 0  # Genuine inter-frame motion.
    original_id = people[0]["track_id"]
    pipeline.detector.persons = []
    session.process_frame(image, 30)
    pipeline.detector.persons = [make_person()]
    _, people = session.process_frame(image, 31)
    assert people[0]["track_id"] != original_id
    assert people[0]["window_frames"] == 1
    assert people[0]["features"][2] == 0
    assert len(pipeline.lstm.batches) == 1


@pytest.mark.parametrize("fps,count,width,height", [
    (120, 300, 640, 360), (25, 9000, 640, 360), (25, 300, 7680, 4320),
])
def test_video_limits_are_checked_before_decoding(monkeypatch, fps, count, width, height):
    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: count,
                    cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height}[prop]

        def read(self):
            pytest.fail("Oversized or unsupported video should be rejected before decoding")

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())
    with pytest.raises(ValueError):
        inspect_video("oversized.mp4")


def test_detection_order_does_not_mix_people_and_videos_reset():
    a, b = make_person(), make_person()
    b.bbox = (45, 5, 65, 45)
    pipeline = fake_pipeline([a, b])
    session = VideoSession(pipeline)
    frame = np.zeros((50, 70, 3), np.uint8)
    _, first = session.process_frame(frame, 0)
    pipeline.detector.persons = [b, a]
    _, second = session.process_frame(frame, 1)
    assert [p["track_id"] for p in second] == [first[1]["track_id"], first[0]["track_id"]]
    _, fresh = VideoSession(pipeline).process_frame(frame, 0)
    assert all(p["window_frames"] == 1 for p in fresh)


def test_real_video_output_and_frame_records(sample_video, tmp_path):
    result_dir = tmp_path / "results"
    summary = process_video(fake_pipeline([make_person()]), sample_video, result_dir)
    assert summary["total_frames"] == 34
    assert summary["frames_with_classification"] == 5
    assert summary["inference"]["input_frames"] == 30
    assert inspect_video(result_dir / "result.mp4")["total_frames"] == 34
    records = [json.loads(line) for line in (result_dir / "predictions.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()]
    assert records[28]["persons"][0]["state"] == "collecting"
    assert records[29]["persons"][0]["state"] == "fallen"
    assert sample_video.is_file()


def wait_job(client, job):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(job["status_url"])
        assert response.status_code == 200
        if response.json["status"] not in ("queued", "processing"):
            return response.json
        time.sleep(0.03)
    pytest.fail("Video worker did not finish")


def test_browser_video_job_downloads_ranges_and_cleanup(sample_video, tmp_path):
    app = web_app.create_app(pipeline=fake_pipeline([make_person()]),
                             output_root=tmp_path / "jobs",
                             health_checker=lambda: {"ready": True})
    client = app.test_client()
    response = client.post("/api/videos", data={
        "video": (io.BytesIO(sample_video.read_bytes()), "../../视频.mp4"),
    })
    assert response.status_code == 202
    done = wait_job(client, response.json)
    assert done["status"] == "completed", done
    result = done["result"]
    assert result["filename"] == "视频.mp4"
    assert client.get(result["video_url"]).mimetype == "video/mp4"
    assert client.get(result["video_url"], headers={"Range": "bytes=0-99"}).status_code == 206
    assert client.get(result["csv_url"]).data.startswith(b"\xef\xbb\xbf")
    assert len(client.get(result["predictions_url"]).text.splitlines()) == 34
    archive = zipfile.ZipFile(io.BytesIO(client.get(result["zip_url"]).data))
    assert set(archive.namelist()) == {
        "result.mp4", "predictions.jsonl", "summary.csv", "summary.json",
    }
    directory = tmp_path / "jobs" / result["job_id"]
    assert not list(directory.glob("input*"))
    assert client.get(f"/results/{result['job_id']}/status.json").status_code == 404


def test_busy_cancel_and_failure_release_inference_lock(sample_video, tmp_path, monkeypatch):
    entered = threading.Event()

    def blocking_video(pipeline, source, directory, *, progress, cancelled):
        entered.set()
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise VideoCancelled()

    monkeypatch.setattr(src.video_web, "process_video", blocking_video)
    app = web_app.create_app(pipeline=fake_pipeline([]), output_root=tmp_path / "jobs",
                             health_checker=lambda: {"ready": True})
    client = app.test_client()

    def upload():
        return client.post("/api/videos", data={
            "video": (io.BytesIO(sample_video.read_bytes()), "clip.mp4"),
        })

    job = upload().json
    assert entered.wait(3)
    assert client.post("/api/detect", data={"images": image_upload()}).status_code == 429
    assert upload().status_code == 429
    assert client.post(job["status_url"] + "/cancel").json["cancel_requested"]
    assert wait_job(client, job)["status"] == "cancelled"
    assert not list((tmp_path / "jobs" / job["job_id"]).glob("input*"))
    assert client.post("/api/detect", data={"images": image_upload()}).status_code == 200

    def failing_video(*args, **kwargs):
        raise ValueError("视频解码失败")
    monkeypatch.setattr(src.video_web, "process_video", failing_video)
    job = upload().json
    assert wait_job(client, job)["status"] == "failed"
    assert client.post("/api/detect", data={"images": image_upload()}).status_code == 200


def test_video_upload_validation_and_restart_interruption(tmp_path, sample_video):
    app = web_app.create_app(pipeline=fake_pipeline([]), output_root=tmp_path / "jobs",
                             health_checker=lambda: {"ready": True})
    client = app.test_client()
    for name, content in [("bad.mp4", b"bad"), ("bad.html", b"bad"), ("empty.mp4", b"")]:
        assert client.post("/api/videos", data={
            "video": (io.BytesIO(content), name),
        }).status_code == 400
    assert not list((tmp_path / "jobs").iterdir())
    assert client.get("/api/videos/not-an-id").status_code == 404
    response = client.post("/api/videos", headers={"Origin": "https://other.example"})
    assert response.status_code == 403
    job_id = "a" * 32
    directory = tmp_path / "jobs" / job_id
    directory.mkdir()
    (directory / "status.json").write_text(json.dumps({"status": "processing"}))
    status = client.get(f"/api/videos/{job_id}").json
    assert status["status"] == "failed"
    assert "重启" in status["error"]
