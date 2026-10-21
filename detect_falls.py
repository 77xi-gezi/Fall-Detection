"""Input a single image and output its per-person model state."""
import argparse
import json
import logging
from pathlib import Path

from src.detector import DEFAULT_POSE_MODEL
from src.image_io import read_image
from src.pipeline import FallDetectionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="单张图片：YOLO26s-Pose → 5 特征 → LSTM 状态输出")
    parser.add_argument("--input", "-i", required=True, help="输入单张图片文件路径")
    parser.add_argument("--output", "-o", help="标注图片路径，默认 output/<图片名>_result.png")
    parser.add_argument("--json-output", help="状态 JSON 路径，默认与标注图片同名")
    parser.add_argument("--model", default="models/checkpoints/best.pth", help="LSTM 权重")
    parser.add_argument("--yolo-model", default=DEFAULT_POSE_MODEL, help="YOLO26s-Pose 权重")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理分辨率")
    parser.add_argument(
        "--detection-confidence", type=float, default=0.5, help="人体检测置信度阈值",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], help="默认自动选择 CUDA / CPU")
    parser.add_argument("--backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--triton-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.imgsz <= 0 or not 0 < args.detection_confidence <= 1:
        parser.error("imgsz 必须为正数，detection-confidence 必须在 (0, 1] 范围内")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    input_path = Path(args.input)
    if not input_path.exists():
        input_path = Path("input") / args.input
    try:
        # Reject directories and invalid files before loading any models.
        read_image(input_path)
        pipeline = FallDetectionPipeline(
            yolo_model=args.yolo_model, lstm_checkpoint=args.model, device=args.device,
            detection_confidence=args.detection_confidence, imgsz=args.imgsz,
            backend=args.backend, triton_url=args.triton_url,
        )
        result = pipeline.process_image(
            input_path, args.output or Path("output") / f"{input_path.stem}_result.png",
            args.json_output,
        )
        summary = {
            "state": result["state"],
            "state_label": result["state_label"],
            "persons": [
                {key: person[key] for key in (
                    "person_id", "state", "state_label", "confidence", "probabilities",
                )}
                for person in result["persons"]
            ],
            "annotated_image": result["annotated_image"],
            "single_image_accuracy_validated": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
