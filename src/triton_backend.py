"""Triton HTTP binary protocol plus YOLO26 pose pre/postprocessing."""
import json
import urllib.error
import urllib.request

import cv2
import numpy as np
import torch

from src.detector import DetectedPerson


class TritonHTTP:
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: int = 60):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def infer(self, model: str, input_name: str, output_name: str, data: np.ndarray) -> np.ndarray:
        data = np.ascontiguousarray(data, dtype="<f4")
        header = json.dumps({
            "inputs": [{"name": input_name, "shape": list(data.shape), "datatype": "FP32",
                        "parameters": {"binary_data_size": data.nbytes}}],
            "outputs": [{"name": output_name, "parameters": {"binary_data": True}}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/v2/models/{model}/infer", data=header + data.tobytes(),
            headers={"Content-Type": "application/octet-stream",
                     "Inference-Header-Content-Length": str(len(header))}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                header_size = int(response.headers.get("Inference-Header-Content-Length", 0))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Triton {model}: HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Cannot connect to Triton at {self.url}: {error.reason}") from error
        info = json.loads(body[:header_size] if header_size else body)
        output = info["outputs"][0]
        if output["name"] != output_name or output["datatype"] != "FP32":
            raise RuntimeError(f"Unexpected Triton output: {output}")
        if header_size:
            byte_count = output["parameters"]["binary_data_size"]
            values = np.frombuffer(body[header_size:header_size + byte_count], dtype="<f4").copy()
        else:
            values = np.asarray(output["data"], dtype=np.float32)
        return values.reshape(output["shape"])


def preprocess(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Match Ultralytics square letterbox, BGR->RGB and [0,1] normalization."""
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    resized_w, resized_h = round(width * ratio), round(height * ratio)
    left = round((size - resized_w) / 2 - 0.1)
    top = round((size - resized_h) / 2 - 0.1)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, size - resized_h - top, left, size - resized_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255
    return tensor, ratio, (left, top)


def postprocess(output: np.ndarray, image_shape: tuple, ratio: float,
                padding: tuple[int, int], confidence: float) -> list[DetectedPerson]:
    if output.shape != (1, 300, 57):
        raise ValueError(f"Expected YOLO26 end-to-end output (1,300,57), got {output.shape}")
    height, width = image_shape[:2]
    left, top = padding
    persons = []
    for row in output[0]:
        if not np.isfinite(row).all() or row[4] < confidence or int(row[5]) != 0:
            continue
        bbox = row[:4].copy()
        bbox[[0, 2]] = np.clip((bbox[[0, 2]] - left) / ratio, 0, width)
        bbox[[1, 3]] = np.clip((bbox[[1, 3]] - top) / ratio, 0, height)
        keypoints = row[6:].reshape(17, 3).copy()
        keypoints[:, 0] = np.clip((keypoints[:, 0] - left) / ratio, 0, width)
        keypoints[:, 1] = np.clip((keypoints[:, 1] - top) / ratio, 0, height)
        persons.append(DetectedPerson(
            person_id=len(persons) + 1, bbox=tuple(float(v) for v in bbox),
            keypoints=keypoints, confidence=float(row[4]),
        ))
    return persons


class TritonPoseDetector:
    def __init__(self, client: TritonHTTP, confidence_threshold: float = 0.5, imgsz: int = 640):
        if imgsz != 640:
            raise ValueError("The exported Triton pose model requires --imgsz 640")
        self.client = client
        self.confidence_threshold = confidence_threshold
        self.model_name = "yolo26s_pose (ONNX/Triton)"

    def detect_image(self, image: np.ndarray) -> list[DetectedPerson]:
        tensor, ratio, padding = preprocess(image)
        output = self.client.infer("yolo26s_pose", "images", "output0", tensor)
        return postprocess(output, image.shape, ratio, padding, self.confidence_threshold)


class TritonLSTM(torch.nn.Module):
    def __init__(self, client: TritonHTTP):
        super().__init__()
        self.client = client

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.client.infer("fall_lstm", "features", "logits", features.cpu().numpy())
        return torch.from_numpy(logits)
