#!/usr/bin/env python3
"""知识类视频拉片分析工具 CLI 入口。

用法：
    python main.py --input /path/to/video.mp4
    python main.py --input video.mp4 --output my_output --model gemini-3.1-pro
"""

import argparse
import logging
import sys
from pathlib import Path

from lapian.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="知识类视频拉片分析工具")
    parser.add_argument("--input", "-i", required=True, help="输入视频文件路径")
    parser.add_argument("--output", "-o", default=None,
                        help="输出目录（默认 output/<视频名>）")
    parser.add_argument("--config", default=Path(__file__).parent / "config.yaml",
                        help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument("--prompts", default=Path(__file__).parent / "prompts.yaml",
                        help="Prompt 文件路径（默认 ./prompts.yaml）")
    parser.add_argument("--model", "-m", default=None,
                        help="覆盖 config.yaml 中的模型（如 gemini-3.7-flash）")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    video = Path(args.input).expanduser().resolve()
    if not video.exists():
        sys.exit(f"视频不存在: {video}")
    out_dir = Path(args.output) if args.output else Path("output") / video.stem

    pipeline = Pipeline(args.config, args.prompts, model_override=args.model)
    result = pipeline.run(video, out_dir)
    print(f"\n✅ 拉片完成，结构化数据: {result}")


if __name__ == "__main__":
    main()
