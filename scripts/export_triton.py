"""Export both inference networks and generate a Triton ONNX model repository."""
# ruff: noqa: E402
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tools/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib"))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from models.fall_detector import FallDetector
from ultralytics import YOLO


def describe(path: Path) -> dict:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def tensor_info(value):
        return {
            "name": value.name,
            "shape": [d.dim_value if d.HasField("dim_value") else -1
                      for d in value.type.tensor_type.shape.dim],
        }

    return {
        "inputs": [tensor_info(value) for value in model.graph.input],
        "outputs": [tensor_info(value) for value in model.graph.output],
    }


def write_config(model_dir: Path, info: dict) -> None:
    lines = [f'name: "{model_dir.name}"', 'platform: "onnxruntime_onnx"', "max_batch_size: 0"]
    for key, field in (("inputs", "input"), ("outputs", "output")):
        lines.append(f"{field} [")
        for tensor in info[key]:
            dims = ", ".join(str(d) for d in tensor["shape"])
            lines.append(f'  {{ name: "{tensor["name"]}" data_type: TYPE_FP32 dims: [ {dims} ] }}')
        lines.append("]")
    lines.extend([
        "instance_group [ { kind: KIND_GPU count: 1 gpus: [ 0 ] } ]",
        'version_policy: { specific: { versions: [ 1 ] } }',
    ])
    (model_dir / "config.pbtxt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(repository: Path) -> dict:
    repository = repository.resolve()
    pose_dir = repository / "yolo26s_pose"
    lstm_dir = repository / "fall_lstm"
    for directory in (pose_dir / "1", lstm_dir / "1"):
        directory.mkdir(parents=True, exist_ok=True)
    staging = ROOT / ".tools/onnx_export"
    staging.mkdir(parents=True, exist_ok=True)
    pose_weights = staging / "yolo26s-pose.pt"
    shutil.copy2(ROOT / "yolo26s-pose.pt", pose_weights)
    pose = YOLO(str(pose_weights))
    exported = pose.export(
        format="onnx", imgsz=640, batch=1, device="cpu", opset=17,
        dynamic=False, simplify=False, nms=False,
    )
    pose_path = pose_dir / "1/model.onnx"
    shutil.copy2(exported, pose_path)
    pose_info = describe(pose_path)
    assert pose_info["inputs"][0]["shape"] == [1, 3, 640, 640]
    assert pose_info["outputs"][0]["shape"] == [1, 300, 57], pose_info

    lstm = FallDetector().cpu().eval()
    checkpoint = torch.load(ROOT / "models/checkpoints/best.pth", map_location="cpu",
                            weights_only=False)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm_path = lstm_dir / "1/model.onnx"
    torch.onnx.export(
        lstm, torch.zeros(2, 1, 5), str(lstm_path), opset_version=17, dynamo=False,
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "people", 1: "frames"}, "logits": {0: "people"}},
    )
    lstm_info = describe(lstm_path)
    session = ort.InferenceSession(str(lstm_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    errors = {}
    for people in (1, 2, 5):
        for frames in (1, 30):
            features = rng.random((people, frames, 5), dtype=np.float32)
            with torch.inference_mode():
                expected = lstm(torch.from_numpy(features)).numpy()
            actual = session.run(["logits"], {"features": features})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            errors[f"{people}x{frames}"] = float(np.max(np.abs(actual - expected)))
    # Validate the pose ONNX can actually execute, not only pass the graph checker.
    pose_session = ort.InferenceSession(str(pose_path), providers=["CPUExecutionProvider"])
    pose_output = pose_session.run(None, {"images": np.zeros((1, 3, 640, 640), np.float32)})[0]
    assert pose_output.shape == (1, 300, 57) and np.isfinite(pose_output).all()
    write_config(pose_dir, pose_info)
    write_config(lstm_dir, lstm_info)
    manifest = {
        "pose": pose_info, "lstm": lstm_info, "opset": 17,
        "imgsz": 640, "pose_end2end": True, "lstm_input_frames": 1,
        "lstm_supported_frames": [1, 30], "video_window_frames": 30,
        "lstm_training_frames": 30, "single_image_accuracy_validated": False,
        "lstm_onnx_max_abs_error": errors,
    }
    manifest_path = repository.parent / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path,
                        default=ROOT / "deploy/triton/model_repository")
    export(parser.parse_args().repository)
