"""Local browser upload UI backed by the existing Triton single-image pipeline."""
import argparse
import csv
import io
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
import warnings
import webbrowser
import zipfile
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from src.image_io import IMAGE_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_FILES = 50
MAX_PIXELS = 25_000_000
MODELS = ("yolo26s_pose", "fall_lstm")
logger = logging.getLogger(__name__)


def triton_health(url: str) -> dict:
    """Check both specific models, without relying on a browser or a system proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    models = {}
    for model in MODELS:
        try:
            with opener.open(f"{url.rstrip('/')}/v2/models/{model}/ready", timeout=2) as response:
                models[model] = response.status == 200
        except (OSError, urllib.error.URLError):
            models[model] = False
    return {"ready": all(models.values()), "models": models}


def validate_image(path: Path) -> None:
    """Reject corrupt uploads and excessive decoded sizes before OpenCV decoding."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("图片分辨率过大，请缩小至 2500 万像素以内。")
                image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("无法读取图片，请选择有效的 JPG、PNG、BMP、TIFF 或 WEBP 图片。") from error


def create_app(*, pipeline=None, output_root: Path | None = None,
               triton_url: str = "http://127.0.0.1:8000", health_checker=None) -> Flask:
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.json.ensure_ascii = False
    app.config.update(
        MAX_CONTENT_LENGTH=104 * 1024 * 1024,
        MAX_FORM_PARTS=MAX_FILES + 10,
        TRUSTED_HOSTS=["127.0.0.1", "localhost"],
    )
    output_root = Path(output_root or PROJECT_ROOT / "output/web_results").resolve()
    inference_lock = threading.Lock()
    pipelines = [pipeline]
    check_health = health_checker or (lambda: triton_health(triton_url))

    @app.before_request
    def protect_local_upload():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if request.headers.get("Sec-Fetch-Site") == "cross-site" or (
                origin and origin.rstrip("/") != request.host_url.rstrip("/")
            ):
                return jsonify(error="请从本机检测网页提交图片。"), 403

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'self'; form-action 'self'; object-src 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {
            413: "本次上传总大小超过 100 MB，请减少图片数量。",
            404: "未找到页面或结果文件。",
            400: "上传请求无效，请重新选择图片。",
        }
        return jsonify(error=messages.get(error.code, error.description)), error.code

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(**check_health(), app="fall-detection-web", backend="triton",
                       max_files=MAX_FILES,
                       max_file_mb=MAX_FILE_BYTES // (1024 * 1024))

    @app.post("/api/detect")
    def detect():
        uploads = request.files.getlist("images")
        if not uploads or len(uploads) > MAX_FILES:
            return jsonify(error=f"请选择 1 至 {MAX_FILES} 张图片。"), 400
        if not inference_lock.acquire(blocking=False):
            return jsonify(error="正在处理另一批图片，请等待完成后重试。"), 429
        try:
            if not check_health()["ready"]:
                message = "Triton 模型尚未就绪。请启动 Docker Desktop 和检测服务，再重试。"
                return jsonify(error=message), 503
            if pipelines[0] is None:
                from src.pipeline import FallDetectionPipeline

                pipelines[0] = FallDetectionPipeline(backend="triton", triton_url=triton_url)
            job_id = uuid.uuid4().hex
            job_root = output_root / job_id
            job_root.mkdir(parents=True)
            started = time.perf_counter()
            results = []
            for number, upload in enumerate(uploads, 1):
                filename = (upload.filename or "未命名图片").replace("\\", "/").rsplit("/", 1)[-1]
                source = job_root / f"{number:03d}_input{Path(filename).suffix.lower()}"
                record = {"filename": filename, "state": "error", "state_label": "检测失败",
                          "persons_detected": 0}
                try:
                    if source.suffix not in IMAGE_EXTENSIONS:
                        message = "不支持该文件格式，请选择 JPG、PNG、BMP、TIFF 或 WEBP 图片。"
                        raise ValueError(message)
                    data = upload.stream.read(MAX_FILE_BYTES + 1)
                    if not data:
                        raise ValueError("图片文件为空。")
                    if len(data) > MAX_FILE_BYTES:
                        raise ValueError("单张图片不能超过 16 MB。")
                    source.write_bytes(data)
                    validate_image(source)
                    image_name = f"{number:03d}_result.png"
                    json_name = f"{number:03d}_result.json"
                    result = pipelines[0].process_image(
                        source, job_root / image_name, job_root / json_name,
                    )
                    record.update(result)
                    record.update(
                        image_url=f"/results/{job_id}/{image_name}",
                        json_url=f"/results/{job_id}/{json_name}",
                    )
                except ValueError as error:
                    record["error"] = str(error)
                except Exception:
                    logger.exception("Browser inference failed for %s", filename)
                    record["error"] = "检测失败，请检查 Triton 服务是否正常运行，然后重试。"
                finally:
                    source.unlink(missing_ok=True)
                results.append(record)
            summary = {
                "job_id": job_id, "total": len(results),
                "succeeded": sum(item["state"] != "error" for item in results),
                "failed": sum(item["state"] == "error" for item in results),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "results": results, "single_image_accuracy_validated": False,
                "csv_url": f"/results/{job_id}/summary.csv",
                "json_url": f"/results/{job_id}/summary.json",
                "zip_url": f"/results/{job_id}/results.zip",
            }
            (job_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            with (job_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["图片", "状态", "人数", "标注图片", "错误"])
                for item in results:
                    # Neutralize spreadsheet formulas in user-controlled filenames.
                    name = item["filename"]
                    if name.lstrip().startswith(("=", "+", "-", "@")):
                        name = "'" + name
                    writer.writerow([name, item["state_label"], item["persons_detected"],
                                     item.get("image_url", "").rsplit("/", 1)[-1],
                                     item.get("error", "")])
            return jsonify(summary)
        except Exception:
            logger.exception("Could not initialize browser detection")
            return jsonify(error="检测服务初始化失败，请检查终端日志后重试。"), 503
        finally:
            inference_lock.release()

    @app.get("/results/<job_id>/<filename>")
    def result_file(job_id, filename):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            abort(404)
        job_root = output_root / job_id
        if not (job_root / "summary.json").is_file():
            abort(404)
        if filename == "results.zip":
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for path in sorted(job_root.iterdir()):
                    if path.name in {"summary.csv", "summary.json"} or re.fullmatch(
                        r"\d{3}_result\.(png|json)", path.name,
                    ):
                        zip_file.write(path, path.name)
            archive.seek(0)
            return send_file(archive, mimetype="application/zip", as_attachment=True,
                             download_name=f"fall_results_{job_id[:8]}.zip")
        if filename not in {"summary.csv", "summary.json"} and not re.fullmatch(
            r"\d{3}_result\.(png|json)", filename,
        ):
            abort(404)
        return send_from_directory(job_root, filename, as_attachment="download" in request.args)

    return app


def main():
    parser = argparse.ArgumentParser(description="浏览器图片跌倒检测，使用现有 Triton GPU 服务")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--triton-url", default="http://127.0.0.1:8000")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port 必须在 1 至 65535 范围内")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    from waitress import serve

    print(f"Browser detection: http://127.0.0.1:{args.port}", flush=True)
    if args.open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    serve(create_app(triton_url=args.triton_url), host="127.0.0.1", port=args.port,
          threads=4, max_request_body_size=104 * 1024 * 1024)


if __name__ == "__main__":
    main()
