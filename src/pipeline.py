"""Single image -> YOLO26s-Pose -> five features -> one-step LSTM output."""
import json
import logging
from pathlib import Path

import numpy as np
import torch

from models.fall_detector import FallDetector
from src.detector import DEFAULT_POSE_MODEL, PersonDetector
from src.feature_extraction import FeatureExtractor
from src.image_io import IMAGE_EXTENSIONS, read_image, write_image
from src.utils import STATUS_COLORS, draw_bbox, draw_skeleton

logger = logging.getLogger(__name__)
STATE_LABELS = {
    "normal": "正常", "falling": "疑似跌倒", "fallen": "倒地", "no_person": "未检测到人体",
}


class FallDetectionPipeline:
    """Reuse temporal weights for experimental single-image classification."""

    def __init__(self, yolo_model: str = DEFAULT_POSE_MODEL,
                 lstm_checkpoint: str = "models/checkpoints/best.pth",
                 device: str | None = None, detection_confidence: float = 0.5,
                 imgsz: int = 640, backend: str = "torch",
                 triton_url: str = "http://127.0.0.1:8000"):
        self.backend = backend
        self.triton_url = triton_url
        if backend == "triton":
            from src.triton_backend import TritonHTTP, TritonLSTM, TritonPoseDetector

            self.device = "cpu"  # Host feature tensors; both networks execute on Triton's GPU.
            client = TritonHTTP(triton_url)
            self.detector = TritonPoseDetector(client, detection_confidence, imgsz)
            self.lstm = TritonLSTM(client)
            self.feature_extractor = FeatureExtractor()
            return
        if backend != "torch":
            raise ValueError(f"Unsupported inference backend: {backend}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = PersonDetector(
            model_name=yolo_model, device=self.device,
            confidence_threshold=detection_confidence, imgsz=imgsz,
        )
        self.lstm = FallDetector()
        # This bundled, trusted checkpoint also contains NumPy training metrics.
        checkpoint = torch.load(lstm_checkpoint, map_location=self.device, weights_only=False)
        self.lstm.load_state_dict(checkpoint["model_state_dict"])
        self.lstm.to(self.device).eval()
        self.feature_extractor = FeatureExtractor()
        logger.info("Single-image %s + LSTM initialized on %s", yolo_model, self.device)

    def classify_image(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """One real observation per person; no repeated frames or persistent history."""
        detections = self.detector.detect_image(image)
        annotated = image.copy()
        persons = []
        features = []
        for person in detections:
            vector = self.feature_extractor.extract(
                person.keypoints, person.bbox, image.shape[0], prev_cog_height=None,
            )
            features.append(vector.to_array())
            persons.append({
                "person_id": person.person_id,
                "bbox": list(person.bbox),
                "keypoints": person.keypoints.tolist(),
                "detection_confidence": float(person.confidence),
                "features": vector.to_array().tolist(),
            })
        if features:
            # Shape: (people, 1 actual observation, 5 features).
            batch = torch.from_numpy(np.stack(features)[:, None, :]).to(self.device)
            with torch.inference_mode():
                probabilities = torch.softmax(self.lstm(batch), dim=1).cpu().numpy()
            for person, record, values in zip(detections, persons, probabilities):
                state = FallDetector.CLASS_NAMES[int(np.argmax(values))]
                record.update({
                    "state": state,
                    "state_label": STATE_LABELS[state],
                    "confidence": float(np.max(values)),
                    "probabilities": {
                        FallDetector.CLASS_NAMES[i]: float(v) for i, v in enumerate(values)
                    },
                })
                color = STATUS_COLORS[state]
                draw_skeleton(annotated, person.keypoints, color=color)
                draw_bbox(
                    annotated, person.bbox, color=color,
                    label=f"Person:{person.person_id} {state.upper()} {record['confidence']:.2f}",
                )
        return annotated, persons

    def process_image(self, input_path: str | Path, output_path: str | Path,
                      json_path: str | Path | None = None) -> dict:
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        json_path = Path(json_path).resolve() if json_path else output_path.with_suffix(".json")
        if output_path == input_path or json_path == input_path or json_path == output_path:
            raise ValueError("Input image, output image and output JSON must be different files")
        if (output_path.suffix.lower() not in IMAGE_EXTENSIONS
                or json_path.suffix.lower() != ".json"):
            raise ValueError("Output must be an image file and JSON output must end with .json")
        image = read_image(input_path)
        annotated, persons = self.classify_image(image)
        states = {person["state"] for person in persons}
        state = next((s for s in ("fallen", "falling", "normal") if s in states), "no_person")
        result = {
            "input_image": str(input_path),
            "annotated_image": str(output_path),
            "state": state,
            "state_label": STATE_LABELS[state],
            "persons_detected": len(persons),
            "persons": persons,
            "inference": {
                "pose_model": getattr(self.detector, "model_name", DEFAULT_POSE_MODEL),
                "backend": getattr(self, "backend", "torch"),
                "input_frames": 1,
                "checkpoint_training_frames": 30,
                "motion_observed": False,
                "single_image_accuracy_validated": False,
                "note": "复用30帧训练权重做单时间步推理；单图准确率未经验证。",
                "device": "cuda:0 (Triton)" if getattr(self, "backend", "torch") == "triton"
                else self.device,
            },
        }
        write_image(output_path, annotated)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Image state: %s; annotated image: %s; JSON: %s", state, output_path, json_path)
        return result
