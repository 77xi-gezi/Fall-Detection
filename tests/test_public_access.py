"""Public HTTPS access keeps inference and saved results behind authentication."""
import pytest

import web_app
from tests.test_single_image import fake_pipeline
from tests.test_web_app import image_upload

PUBLIC = "https://fall-test.trycloudflare.com"
PASSWORD = "test-password-123456"


@pytest.fixture
def public_client(tmp_path):
    app = web_app.create_app(
        pipeline=fake_pipeline([]), output_root=tmp_path, public_url=PUBLIC,
        access_password=PASSWORD, health_checker=lambda: {"ready": True},
    )
    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize("path", [
    "/", "/api/health", "/api/detect", "/results/any/result.png",
    "/api/videos", "/api/videos/" + "a" * 32, "/results/any/result.mp4",
])
def test_no_public_route_exposes_inference_or_results_without_auth(public_client, path):
    response = public_client.get(path, base_url=PUBLIC)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic ")
    assert "no-store" in response.headers["Cache-Control"]


def test_password_and_host_checks(public_client):
    assert public_client.get("/", base_url=PUBLIC, auth=("fall", "wrong")).status_code == 401
    assert public_client.get("/", base_url=PUBLIC, auth=("fall", "错误密码")).status_code == 401
    assert public_client.get("/", base_url=PUBLIC, auth=("fall", PASSWORD)).status_code == 200
    assert public_client.get("/", base_url="https://other.example",
                             auth=("fall", PASSWORD)).status_code == 400


def test_https_tunnel_origin_upload_and_authenticated_downloads(public_client):
    # The local tunnel forwards HTTP with the browser's public HTTPS origin.
    response = public_client.post(
        "/api/detect", base_url=PUBLIC.replace("https://", "http://"),
        headers={"Origin": PUBLIC, "X-Forwarded-Proto": "https"},
        auth=("fall", PASSWORD), data={"images": image_upload()},
    )
    assert response.status_code == 200
    assert response.json["succeeded"] == 1
    assert response.json["results"][0]["state"] == "no_person"
    image_url = response.json["results"][0]["image_url"]
    assert public_client.get(image_url, base_url=PUBLIC).status_code == 401
    assert public_client.get(image_url, base_url=PUBLIC,
                             auth=("fall", PASSWORD)).mimetype == "image/png"
    assert public_client.post(
        "/api/detect", base_url=PUBLIC, auth=("fall", PASSWORD),
        headers={"Origin": "https://other.example"}, data={"images": image_upload()},
    ).status_code == 403


@pytest.mark.parametrize("url", ["http://example.com", "https://example.com/path",
                                "https://user:pass@example.com", "https://example.com?x=1"])
def test_invalid_public_urls_are_rejected(url):
    with pytest.raises(ValueError):
        web_app.create_app(public_url=url, access_password=PASSWORD)


def test_public_mode_requires_a_password():
    with pytest.raises(ValueError, match="password"):
        web_app.create_app(public_url=PUBLIC)


def test_real_waitress_preserves_https_origin_for_tunnel_upload(tmp_path):
    import threading

    import requests
    from waitress import create_server

    app = web_app.create_app(
        pipeline=fake_pipeline([]), output_root=tmp_path, public_url=PUBLIC,
        access_password=PASSWORD, health_checker=lambda: {"ready": True},
    )
    server = create_server(app, port=0, **web_app.server_options(True))
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    client = requests.Session()
    client.trust_env = False
    try:
        headers = {"Host": PUBLIC.split("//")[1], "Origin": PUBLIC,
                   "X-Forwarded-Proto": "https"}
        stream, name = image_upload()
        response = client.post(
            f"http://127.0.0.1:{port}/api/detect", headers=headers,
            auth=("fall", PASSWORD), files={"images": (name, stream.getvalue(), "image/png")},
            timeout=5,
        )
        assert response.status_code == 200
        assert response.json()["succeeded"] == 1
        headers["Origin"] = "https://another.example"
        assert client.post(f"http://127.0.0.1:{port}/api/detect", headers=headers,
                           auth=("fall", PASSWORD), timeout=5).status_code == 403
    finally:
        client.close()
        server.close()
        server.task_dispatcher.shutdown()
        thread.join(timeout=2)
