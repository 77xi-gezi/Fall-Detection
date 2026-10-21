"""Detect falls in a video using Torch or the deployed Triton models."""
import argparse
import json
from pathlib import Path

from src.pipeline import FallDetectionPipeline
from src.video_pipeline import process_video


def main():
    parser = argparse.ArgumentParser(description="视频人体跌倒检测，连续30帧滑动窗口")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/video_results"))
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--device", default=None)
    parser.add_argument("--triton-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    source = args.input.resolve()
    if source.parent == args.output.resolve() or args.output.resolve() in source.parents:
        parser.error("请将结果保存到输入视频所在目录之外。")
    try:
        pipeline = FallDetectionPipeline(backend=args.backend, device=args.device,
                                         triton_url=args.triton_url)
        def progress(done, total):
            if done % 30 == 0 or done == total:
                print(f"处理进度：{done}/{total} 帧", flush=True)
        result = process_video(pipeline, source, args.output, progress=progress)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"检测失败：{error}\n")


if __name__ == "__main__":
    main()
