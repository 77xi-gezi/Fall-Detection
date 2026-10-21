"""Video inference with independent tracks and genuine 30-frame feature windows."""
import csv
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from models.fall_detector import FallDetector
from src.pipeline import STATE_LABELS
from src.utils import COLOR_YELLOW, STATUS_COLORS, draw_bbox, draw_skeleton

WINDOW_SIZE = 30
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_SECONDS = 300
MAX_VIDEO_FRAMES = 18000
MAX_VIDEO_PIXELS = 1920 * 1080


class VideoCancelled(Exception):
    pass


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("无法读取视频，请使用有效的 MP4、MOV、AVI、MKV 或 WEBM 文件。")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not 1 <= fps <= 60:
            raise ValueError("视频帧率必须在 1 至 60 FPS 之间，请先转换视频。")
        if not math.isfinite(count) or count < 1:
            raise ValueError("无法确定视频帧数，请先转换为 MP4 后重试。")
        if count > MAX_VIDEO_FRAMES or count / fps > MAX_VIDEO_SECONDS + 0.05:
            raise ValueError("视频不能超过 5 分钟或 18000 帧，请截取后再上传。")
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(value) and value > 0 for value in (width, height)):
            raise ValueError("视频画面尺寸无效，请转换为 MP4 后重试。")
        if (width * height > MAX_VIDEO_PIXELS or max(width, height) > 1920
                or min(width, height) > 1080):
            raise ValueError("视频分辨率过大，请先缩小至 1920×1080 像素以内。")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("视频没有可读取的画面。")
        if frame.shape[:2] != (round(height), round(width)):
            raise ValueError("视频画面尺寸与文件信息不符，请转换为 MP4 后重试。")
        height, width = frame.shape[:2]
        return {"fps": fps, "total_frames": round(count), "width": width, "height": height,
                "duration_seconds": round(count / fps, 3)}
    finally:
        capture.release()


def bbox_iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]),
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class VideoSession:
    """Conservative one-to-one IoU association; gaps restart the temporal window.

    IDs only associate adjacent frames, not identity recognition. Reset per video.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tracks = {}
        self.next_id = 1

    def process_frame(self, frame, frame_index):
        detections = self.pipeline.detector.detect_image(frame)
        candidates = sorted(
            ((bbox_iou(person.bbox, track["bbox"]), index, track_id)
             for index, person in enumerate(detections)
             for track_id, track in self.tracks.items()
             if track["last_frame"] == frame_index - 1), reverse=True,
        )
        matched, used_tracks = {}, set()
        for score, index, track_id in candidates:
            if score >= 0.2 and index not in matched and track_id not in used_tracks:
                matched[index] = track_id
                used_tracks.add(track_id)
        records, sequences, ready_records = [], [], []
        current_tracks = {}
        for index, person in enumerate(detections):
            track_id = matched.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                track = {"features": deque(maxlen=WINDOW_SIZE), "cog": None}
            else:
                track = self.tracks[track_id]
            vector = self.pipeline.feature_extractor.extract(
                person.keypoints, person.bbox, frame.shape[0], track["cog"],
            )
            track.update(bbox=person.bbox, last_frame=frame_index, cog=vector.cog_height)
            track["features"].append(vector.to_array())
            current_tracks[track_id] = track
            record = {"track_id": track_id, "bbox": list(person.bbox),
                      "window_frames": len(track["features"]), "state": "collecting",
                      "state_label": "收集连续帧", "confidence": None, "probabilities": None,
                      "features": vector.to_array().tolist()}
            records.append(record)
            if len(track["features"]) == WINDOW_SIZE:
                sequences.append(np.stack(track["features"]))
                ready_records.append(record)
        self.tracks = current_tracks  # No features cross a missed detection.
        if sequences:
            batch = torch.from_numpy(np.stack(sequences)).to(self.pipeline.device)
            with torch.inference_mode():
                probabilities = torch.softmax(self.pipeline.lstm(batch), dim=1).cpu().numpy()
            for record, values in zip(ready_records, probabilities):
                state = FallDetector.CLASS_NAMES[int(np.argmax(values))]
                record.update(state=state, state_label=STATE_LABELS[state],
                              confidence=float(np.max(values)), probabilities={
                                  FallDetector.CLASS_NAMES[i]: float(v)
                                  for i, v in enumerate(values)
                              })
        annotated = frame.copy()
        for person, record in zip(detections, records):
            color = STATUS_COLORS.get(record["state"], COLOR_YELLOW)
            label = f"ID:{record['track_id']} {record['state'].upper()}"
            label += (f" {record['window_frames']}/30" if record["state"] == "collecting"
                      else f" {record['confidence']:.2f}")
            draw_skeleton(annotated, person.keypoints, color=color)
            draw_bbox(annotated, person.bbox, color=color, label=label)
        return annotated, records


def process_video(pipeline, source: Path, output_dir: Path, *, progress=None,
                  cancelled=None) -> dict:
    """Write silent browser-compatible H.264 MP4 and streamed frame records."""
    import imageio_ffmpeg

    info = inspect_video(source)
    if source.resolve() in {
        (output_dir / name).resolve()
        for name in ("result.mp4", "predictions.jsonl", "summary.csv", "summary.json")
    }:
        raise ValueError("输入视频与结果文件不能是同一个文件。")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "result.mp4"
    capture = cv2.VideoCapture(str(source))
    writer = imageio_ffmpeg.write_frames(
        str(video_path), (info["width"], info["height"]), fps=info["fps"],
        pix_fmt_in="bgr24", pix_fmt_out="yuv420p", codec="libx264", quality=7,
        macro_block_size=2, output_params=["-movflags", "+faststart", "-preset", "veryfast"],
        ffmpeg_log_level="error",
    )
    session = VideoSession(pipeline)
    counts = Counter()
    classified_frames, frames = 0, 0
    started = time.perf_counter()
    initialized = False
    try:
        writer.send(None)
        initialized = True
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as records_file, (
            output_dir / "summary.csv"
        ).open("w", encoding="utf-8-sig", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "帧编号", "时间（秒）", "人物编号", "状态", "模型分数", "窗口帧数",
            ])
            while True:
                if cancelled and cancelled():
                    raise VideoCancelled("视频检测已取消。")
                ok, frame = capture.read()
                if not ok:
                    break
                if (frames >= MAX_VIDEO_FRAMES or (frames + 1) / info["fps"] > 300.05
                        or frame.shape[:2] != (info["height"], info["width"])):
                    raise ValueError("视频超过时长或帧数限制，或画面尺寸在中途发生变化。")
                annotated, persons = session.process_frame(frame, frames)
                writer.send(np.ascontiguousarray(annotated))
                timestamp = round(frames / info["fps"], 4)
                records_file.write(json.dumps({
                    "frame_index": frames, "time_seconds": timestamp, "persons": persons,
                }, ensure_ascii=False, allow_nan=False) + "\n")
                for person in persons:
                    counts[person["state"]] += 1
                    csv_writer.writerow([frames + 1, timestamp, person["track_id"],
                                         person["state_label"], person["confidence"],
                                         person["window_frames"]])
                if not persons:
                    counts["no_person"] += 1
                    csv_writer.writerow([frames + 1, timestamp, "", "未检测到人体", "", ""])
                classified_frames += any(p["probabilities"] is not None for p in persons)
                frames += 1
                if progress:
                    progress(frames, info["total_frames"])
    finally:
        capture.release()
        if initialized:
            writer.close()
    if frames == 0 or frames < info["total_frames"] - 1:
        raise ValueError("视频提前中断或解码失败，请转换视频后重试。")
    # FFmpeg errors can surface only during close; check the resulting stream too.
    encoded = inspect_video(video_path)
    if abs(encoded["total_frames"] - frames) > 1:
        raise ValueError("标注视频编码失败，请重试。")
    state = next((s for s in ("fallen", "falling", "normal") if counts[s]),
                 "collecting" if counts["collecting"] else "no_person")
    summary = {
        "mode": "video", "state": state,
        "state_label": STATE_LABELS.get(state, "连续帧不足，尚未分类"),
        "total_frames": frames, "fps": info["fps"],
        "duration_seconds": round(frames / info["fps"], 3),
        "frames_with_classification": classified_frames,
        "persons_tracked": session.next_id - 1, "person_frame_counts": dict(counts),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "inference": {"input_frames": 30, "window_size": 30, "motion_observed": True,
                      "tracking": "adjacent-frame IoU; reset on missed detection",
                      "backend": getattr(pipeline, "backend", "torch"),
                      "accuracy_validated": False, "audio_preserved": False,
                      "note": "按原始帧率逐帧处理，未补帧；人物连续出现30帧后分类。"},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return summary
