"""Check live Triton ONNX inference against local ONNX Runtime and PyTorch."""
# ruff: noqa: E402
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tools/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib"))

import numpy as np
import onnxruntime as ort

from detect_folder import detect_folder
from src.image_io import read_image
from src.pipeline import FallDetectionPipeline
from src.triton_backend import TritonHTTP, TritonPoseDetector, preprocess
from ultralytics import YOLO


def verify(url: str, image_path: Path | None = None) -> dict:
    source = (image_path or ROOT / "input/demo.jpg").resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Test image not found: {source}. Use --image to select your own image."
        )
    client = TritonHTTP(url)
    with urllib.request.urlopen(f"{url}/v2/health/ready", timeout=10) as response:
        assert response.status == 200
    with urllib.request.urlopen(f"{url}/v2", timeout=10) as response:
        server_info = json.load(response)
    repository = ROOT / "deploy/triton/model_repository"
    session = ort.InferenceSession(str(repository / "fall_lstm/1/model.onnx"),
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    errors = {}
    for people in (1, 2, 5):
        features = rng.random((people, 1, 5), dtype=np.float32)
        local = session.run(["logits"], {"features": features})[0]
        remote = client.infer("fall_lstm", "features", "logits", features)
        np.testing.assert_allclose(remote, local, rtol=1e-4, atol=1e-5)
        errors[str(people)] = float(np.max(np.abs(remote - local)))

    image = read_image(source)
    tensor, _, _ = preprocess(image)
    pose_session = ort.InferenceSession(str(repository / "yolo26s_pose/1/model.onnx"),
                                        providers=["CPUExecutionProvider"])
    local_pose = pose_session.run(["output0"], {"images": tensor})[0]
    remote_pose = client.infer("yolo26s_pose", "images", "output0", tensor)
    valid_local = local_pose[0][local_pose[0, :, 4] >= 0.5]
    valid_remote = remote_pose[0][remote_pose[0, :, 4] >= 0.5]
    assert valid_local.shape == valid_remote.shape
    np.testing.assert_allclose(valid_remote, valid_local, rtol=1e-3, atol=0.05)
    # Compare against the original weights using the same square letterbox.
    torch_result = YOLO(str(ROOT / "yolo26s-pose.pt")).predict(
        image, imgsz=640, device="cpu", rect=False, conf=0.5, nms=False, verbose=False,
    )[0]
    remote_persons = TritonPoseDetector(client).detect_image(image)
    assert len(remote_persons) == len(torch_result.boxes)
    for person, box, keypoints in zip(remote_persons, torch_result.boxes.xyxy,
                                      torch_result.keypoints.data):
        np.testing.assert_allclose(person.bbox, box.numpy(), rtol=1e-3, atol=0.1)
        np.testing.assert_allclose(person.keypoints, keypoints.numpy(), rtol=1e-3, atol=0.1)
    pipeline = FallDetectionPipeline(backend="triton", triton_url=url)
    result = pipeline.process_image(source, ROOT / "output/triton_demo_result.png")
    assert result["inference"]["backend"] == "triton"
    report = {
        "server_version": server_info["version"],
        "server_url": url,
        "models_ready": ["yolo26s_pose", "fall_lstm"],
        "lstm_remote_vs_local_max_abs_error": errors,
        "pose_valid_detections": len(valid_remote),
        "pose_remote_vs_local_max_abs_error": float(np.max(np.abs(valid_remote - valid_local)))
        if len(valid_remote) else 0,
        "single_image_state": result["state"],
        "single_image_accuracy_validated": False,
    }
    batch_input = ROOT / ".tools/batch_smoke_input"
    if batch_input.is_dir():
        batch = detect_folder(batch_input, ROOT / "output/triton_batch_smoke", pipeline=pipeline)
        assert batch["succeeded"] == 2 and batch["failed"] == 1
        report["batch_smoke"] = {key: batch[key] for key in ("total_images", "succeeded", "failed")}
    (ROOT / "deploy/triton/validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, help="Validation image; defaults to input/demo.jpg")
    arguments = parser.parse_args()
    verify(arguments.url.rstrip("/"), arguments.image)
