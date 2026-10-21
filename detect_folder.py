"""Batch independent single-image predictions with one shared model load."""
import argparse
import csv
import json
import logging
from pathlib import Path

from src.detector import DEFAULT_POSE_MODEL
from src.image_io import IMAGE_EXTENSIONS
from src.pipeline import FallDetectionPipeline

logger = logging.getLogger(__name__)


def collect_images(input_dir: Path, output_dir: Path, recursive: bool = False) -> list[Path]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input image folder does not exist: {input_dir}")
    if output_dir == input_dir or output_dir in input_dir.parents:
        raise ValueError("Output folder must be separate from, or inside, the input folder")
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted(
        (path for path in candidates
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
         and not path.resolve().is_relative_to(output_dir)),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if not images:
        raise ValueError(f"No supported images found in {input_dir}")
    return images


def detect_folder(input_dir: str | Path, output_dir: str | Path, *, recursive: bool = False,
                  pipeline: FallDetectionPipeline | None = None,
                  model_options: dict | None = None) -> dict:
    """Treat every image independently; record failures and continue the batch."""
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    images = collect_images(input_dir, output_dir, recursive)
    if pipeline is None:
        pipeline = FallDetectionPipeline(**(model_options or {}))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    counts = dict.fromkeys(("normal", "falling", "fallen", "no_person", "error"), 0)
    columns = [
        "image", "state", "state_label", "persons_detected", "output_image", "output_json", "error",
    ]
    # UTF-8 BOM lets Excel display Chinese filenames and state labels correctly.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for index, source in enumerate(images, start=1):
            relative = source.relative_to(input_dir)
            # Keep the original extension to distinguish e.g. photo.jpg and photo.png.
            image_path = output_dir / "images" / relative.parent / f"{relative.name}_result.png"
            json_path = output_dir / "results" / relative.parent / f"{relative.name}.json"
            row = dict.fromkeys(columns, "")
            row["image"] = relative.as_posix()
            try:
                result = pipeline.process_image(source, image_path, json_path)
                row.update({
                    "state": result["state"],
                    "state_label": result["state_label"],
                    "persons_detected": result["persons_detected"],
                    "output_image": image_path.relative_to(output_dir).as_posix(),
                    "output_json": json_path.relative_to(output_dir).as_posix(),
                })
            except Exception as error:
                # One unreadable image must not discard the rest of the folder.
                row.update({"state": "error", "state_label": "检测失败", "error": str(error)})
                logger.error("[%d/%d] %s: %s", index, len(images), relative, error)
            else:
                logger.info("[%d/%d] %s: %s", index, len(images), relative, row["state_label"])
            counts[row["state"]] += 1
            writer.writerow(row)
            csv_file.flush()
    summary = {
        "input_folder": str(input_dir),
        "output_folder": str(output_dir),
        "total_images": len(images),
        "succeeded": len(images) - counts["error"],
        "failed": counts["error"],
        "states": counts,
        "recursive": recursive,
        "results_csv": str(csv_path),
        "single_image_accuracy_validated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="批量单图检测：独立处理文件夹中的每一张图片")
    parser.add_argument("--input", "-i", default="input/images", help="图片文件夹")
    parser.add_argument("--output", "-o", default="output/batch_results", help="结果文件夹")
    parser.add_argument("--recursive", action="store_true", help="同时处理子文件夹")
    parser.add_argument("--model", default="models/checkpoints/best.pth", help="LSTM 权重")
    parser.add_argument("--yolo-model", default=DEFAULT_POSE_MODEL, help="姿态模型权重")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理分辨率")
    parser.add_argument("--detection-confidence", type=float, default=0.5, help="人体检测阈值")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="默认自动选择 CUDA / CPU")
    parser.add_argument("--backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--triton-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.imgsz <= 0 or not 0 < args.detection_confidence <= 1:
        parser.error("imgsz 必须为正数，detection-confidence 必须在 (0, 1] 范围内")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    input_dir = Path(args.input)
    if not input_dir.exists():
        input_dir = Path("input") / args.input
    try:
        summary = detect_folder(input_dir, args.output, recursive=args.recursive, model_options={
            "yolo_model": args.yolo_model, "lstm_checkpoint": args.model, "device": args.device,
            "detection_confidence": args.detection_confidence, "imgsz": args.imgsz,
            "backend": args.backend, "triton_url": args.triton_url,
        })
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
