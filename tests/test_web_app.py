"""Browser upload boundaries, independent results, and result downloads."""
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

import web_app
from tests.test_single_image import fake_pipeline, make_person


def image_upload(name="人物.png"):
    stream = io.BytesIO()
    Image.fromarray(np.zeros((50, 70, 3), dtype=np.uint8)).save(stream, format="PNG")
    stream.seek(0)
    return stream, name


@pytest.fixture
def browser_app(tmp_path):
    app = web_app.create_app(
        pipeline=fake_pipeline([make_person()]), output_root=tmp_path,
        health_checker=lambda: {"ready": True, "models": dict.fromkeys(web_app.MODELS, True)},
    )
    app.config["TESTING"] = True
    return app


def test_home_and_health_are_available_without_upload(browser_app):
    client = browser_app.test_client()
    assert "上传图片" in client.get("/").text
    health = client.get("/api/health")
    assert health.json["ready"] is True
    assert health.json["app"] == "fall-detection-web"
    assert "no-store" in health.headers["Cache-Control"]


def test_single_upload_and_all_downloads(browser_app, tmp_path):
    client = browser_app.test_client()
    response = client.post("/api/detect", data={"images": image_upload()})
    assert response.status_code == 200
    data = response.json
    assert (data["total"], data["succeeded"], data["failed"]) == (1, 1, 0)
    result = data["results"][0]
    assert result["state"] == "fallen"
    assert result["filename"] == "人物.png"
    assert result["inference"]["input_frames"] == 1
    assert data["single_image_accuracy_validated"] is False
    assert client.get(result["image_url"]).mimetype == "image/png"
    assert client.get(result["json_url"]).json["state"] == "fallen"
    assert "attachment" in client.get(result["image_url"] + "?download=1").headers[
        "Content-Disposition"
    ]
    assert client.get(data["csv_url"]).data.startswith(b"\xef\xbb\xbf")
    assert client.get(data["json_url"]).json["total"] == 1
    archive = zipfile.ZipFile(io.BytesIO(client.get(data["zip_url"]).data))
    assert set(archive.namelist()) == {
        "001_result.png", "001_result.json", "summary.csv", "summary.json",
    }
    assert not list((tmp_path / data["job_id"]).glob("*input*"))


def test_mixed_uploads_keep_separate_results_and_continue_after_error(browser_app):
    response = browser_app.test_client().post("/api/detect", data={"images": [
        image_upload("same.png"), (io.BytesIO(b"broken"), "bad.jpg"), image_upload("same.png"),
    ]})
    data = response.json
    assert (data["total"], data["succeeded"], data["failed"]) == (3, 2, 1)
    assert data["results"][1]["state"] == "error"
    assert data["results"][0]["image_url"] != data["results"][2]["image_url"]


@pytest.mark.parametrize("data", [
    {}, {"images": (io.BytesIO(b""), "empty.jpg")},
    {"images": (io.BytesIO(b"<html>"), "test.html")},
])
def test_invalid_uploads_have_readable_errors(browser_app, data):
    response = browser_app.test_client().post("/api/detect", data=data)
    assert response.status_code in (200, 400)
    assert response.json.get("error") or response.json["results"][0]["error"]


def test_upload_limits(browser_app, monkeypatch):
    client = browser_app.test_client()
    response = client.post("/api/detect", data={"images": [image_upload() for _ in range(51)]})
    assert response.status_code == 400
    monkeypatch.setattr(web_app, "MAX_FILE_BYTES", 10)
    assert client.post("/api/detect", data={"images": image_upload()}).json["failed"] == 1
    browser_app.config["MAX_CONTENT_LENGTH"] = 10
    assert client.post("/api/detect", data={"images": image_upload()}).status_code == 413


def test_unavailable_models_reject_upload_without_creating_job(tmp_path):
    app = web_app.create_app(output_root=tmp_path, health_checker=lambda: {"ready": False})
    response = app.test_client().post("/api/detect", data={"images": image_upload()})
    assert response.status_code == 503
    assert not list(tmp_path.iterdir())


def test_paths_cross_origin_and_untrusted_hosts_are_rejected(browser_app, tmp_path):
    client = browser_app.test_client()
    assert client.post("/api/detect", data={"images": image_upload()}, headers={
        "Origin": "https://another.example",
    }).status_code == 403
    assert client.get("/api/health", headers={"Host": "another.example"}).status_code == 400
    assert client.get("/results/not-a-job/summary.json").status_code == 404
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    data = client.post("/api/detect", data={"images": image_upload("../../photo.png")}).json
    assert data["results"][0]["filename"] == "photo.png"
    assert client.get(f"/results/{data['job_id']}/../private.txt").status_code == 404
    assert outside.read_text() == "private"


def test_csv_formula_names_are_neutralized_and_not_rendered_as_html(browser_app):
    client = browser_app.test_client()
    data = client.post("/api/detect", data={"images": image_upload("=1+2.png")}).json
    assert "'=1+2.png" in client.get(data["csv_url"]).data.decode("utf-8-sig")
    assert json.loads(client.get(data["json_url"]).text)["results"][0]["filename"] == "=1+2.png"


def test_failed_initialization_releases_lock_for_retry(tmp_path, monkeypatch):
    import src.pipeline

    app = web_app.create_app(output_root=tmp_path, health_checker=lambda: {"ready": True})
    monkeypatch.setattr(src.pipeline, "FallDetectionPipeline", lambda **kw: 1 / 0)
    client = app.test_client()
    assert client.post("/api/detect", data={"images": image_upload()}).status_code == 503
    monkeypatch.setattr(src.pipeline, "FallDetectionPipeline", lambda **kw: fake_pipeline([]))
    response = client.post("/api/detect", data={"images": image_upload()})
    assert response.status_code == 200
    assert response.json["results"][0]["state"] == "no_person"
