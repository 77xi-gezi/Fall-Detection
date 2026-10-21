"""YOLO26s-Pose detection on one image, without tracking."""
from dataclasses import dataclass

import numpy as np
import torch

from ultralytics import YOLO

DEFAULT_POSE_MODEL = "yolo26s-pose.pt"


@dataclass
class DetectedPerson:
    person_id: int
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    confidence: float


class PersonDetector:
    def __init__(self, model_name: str = DEFAULT_POSE_MODEL, device: str | None = None,
                 confidence_threshold: float = 0.5, imgsz: int = 640):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.model_name = model_name
        self.model = YOLO(model_name)

    def detect_image(self, image: np.ndarray) -> list[DetectedPerson]:
        results = self.model.predict(
            source=image, verbose=False, device=self.device,
            conf=self.confidence_threshold, imgsz=self.imgsz,
        )
        persons = []
        if not results or results[0].boxes is None or results[0].keypoints is None:
            return persons
        result = results[0]
        for i in range(len(result.boxes)):
            keypoints = result.keypoints[i].data.cpu().numpy().reshape(-1, 3)
            if keypoints.shape != (17, 3) or not np.isfinite(keypoints).all():
                continue
            persons.append(DetectedPerson(
                person_id=len(persons) + 1,
                bbox=tuple(float(v) for v in result.boxes.xyxy[i].cpu().numpy()),
                keypoints=keypoints,
                confidence=float(result.boxes.conf[i].item()),
            ))
        return persons
