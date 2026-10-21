"""Asynchronous upload jobs, polling and cancellation for the browser video UI."""
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

from flask import abort, jsonify, request

from src.video_pipeline import (
    MAX_VIDEO_BYTES,
    VIDEO_EXTENSIONS,
    VideoCancelled,
    inspect_video,
    process_video,
)

logger = logging.getLogger(__name__)


def register_video_routes(app, *, output_root, inference_lock, get_pipeline, check_health):
    active = {}
    active_lock = threading.Lock()
    status_lock = threading.RLock()

    def write_status(directory, status):
        # Windows does not replace a destination while another thread has it open.
        with status_lock:
            temporary = directory / "status.tmp"
            temporary.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
            for attempt in range(10):
                try:
                    temporary.replace(directory / "status.json")
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.02)

    def worker(job_id, source, status, cancel):
        directory = output_root / job_id
        last_update = 0

        def progress(done, total):
            nonlocal last_update
            now = time.monotonic()
            if now - last_update >= 0.5 or done == total:
                status.update(processed_frames=done, total_frames=total,
                              progress=min(99, round(done * 100 / max(1, total))))
                write_status(directory, status)
                last_update = now

        try:
            status["status"] = "processing"
            write_status(directory, status)
            result = process_video(get_pipeline(), source, directory, progress=progress,
                                   cancelled=cancel.is_set)
            if cancel.is_set():
                raise VideoCancelled()
            result.update(filename=status["filename"], job_id=job_id,
                          video_url=f"/results/{job_id}/result.mp4",
                          csv_url=f"/results/{job_id}/summary.csv",
                          json_url=f"/results/{job_id}/summary.json",
                          predictions_url=f"/results/{job_id}/predictions.jsonl",
                          zip_url=f"/results/{job_id}/results.zip")
            (directory / "summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            status.update(status="completed", progress=100,
                          processed_frames=result["total_frames"], result=result)
        except VideoCancelled:
            status.update(status="cancelled", error="视频检测已取消，可以选择其他视频。")
        except ValueError as error:
            status.update(status="failed", error=str(error))
        except Exception:
            logger.exception("Video job %s failed", job_id)
            status.update(status="failed", error="视频检测失败，请检查 Triton 服务和终端日志。")
        finally:
            try:
                source.unlink(missing_ok=True)
                if status["status"] != "completed":
                    for name in ("result.mp4", "predictions.jsonl", "summary.csv", "summary.json"):
                        (directory / name).unlink(missing_ok=True)
            finally:
                with status_lock:
                    try:
                        write_status(directory, status)
                    finally:
                        with active_lock:
                            active.pop(job_id, None)
                        inference_lock.release()

    @app.post("/api/videos")
    def upload_video():
        uploads = request.files.getlist("video")
        if len(uploads) != 1:
            return jsonify(error="请每次选择一个视频。"), 400
        upload = uploads[0]
        filename = (upload.filename or "video").replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            return jsonify(error="请选择 MP4、MOV、AVI、MKV、WEBM 或 M4V 视频。"), 400
        if not inference_lock.acquire(blocking=False):
            return jsonify(error="检测服务正在处理其他任务，请稍后重试。"), 429
        directory = None
        transferred = False
        try:
            if not check_health()["ready"]:
                return jsonify(error="Triton 模型尚未就绪，请先启动检测服务。"), 503
            job_id = uuid.uuid4().hex
            directory = output_root / job_id
            directory.mkdir(parents=True)
            source = directory / f"input{suffix}"
            size = 0
            with source.open("wb") as handle:
                while chunk := upload.stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise ValueError("单个视频不能超过 100 MB。")
                    handle.write(chunk)
            if not size:
                raise ValueError("视频文件为空。")
            info = inspect_video(source)
            status = {"job_id": job_id, "filename": filename, "mode": "video",
                      "status": "queued", "progress": 0, "processed_frames": 0,
                      "total_frames": info["total_frames"],
                      "status_url": f"/api/videos/{job_id}"}
            cancel = threading.Event()
            write_status(directory, status)
            with active_lock:
                active[job_id] = cancel
            thread = threading.Thread(target=worker, args=(job_id, source, status, cancel),
                                      daemon=True, name=f"video-{job_id[:8]}")
            thread.start()
            transferred = True
            # Return a stable initial snapshot; worker updates the shared status immediately.
            return jsonify(job_id=job_id, status="queued", status_url=status["status_url"]), 202
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            logger.exception("Video upload failed")
            return jsonify(error="视频上传失败，请重试。"), 503
        finally:
            if not transferred:
                if directory:
                    for path in directory.iterdir():
                        path.unlink(missing_ok=True)
                    directory.rmdir()
                    with active_lock:
                        active.pop(directory.name, None)
                inference_lock.release()

    def load_job(job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            abort(404)
        path = output_root / job_id / "status.json"
        if not path.is_file():
            abort(404)
        with status_lock:
            status = json.loads(path.read_text(encoding="utf-8"))
        with active_lock:
            running = job_id in active
        if status["status"] in ("queued", "processing") and not running:
            status.update(status="failed", error="网页服务已重启，任务中断，请重新上传视频。")
        return status

    @app.get("/api/videos/<job_id>")
    def video_status(job_id):
        return jsonify(load_job(job_id))

    @app.post("/api/videos/<job_id>/cancel")
    def cancel_video(job_id):
        status = load_job(job_id)
        with active_lock:
            cancel = active.get(job_id)
            if cancel:
                cancel.set()
        return jsonify(status=status["status"], cancel_requested=cancel is not None)
