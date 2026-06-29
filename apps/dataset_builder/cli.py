from __future__ import annotations

import argparse
from pathlib import Path

from .settings import BuilderSettings, SourceType, StepMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build face portrait datasets from videos or YouTube URLs.")
    parser.add_argument("source", help="Path to a video file or public YouTube URL.")
    parser.add_argument("--output", default="dataset", help="Output dataset directory.")
    parser.add_argument("--step-frames", type=int, help="Read every N frames.")
    parser.add_argument("--step-seconds", type=float, default=1.0, help="Read every N seconds.")
    parser.add_argument("--keep-all-faces", action="store_true", help="Save every detected face, not only the largest.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not move perceptual duplicates.")
    parser.add_argument("--no-portrait-json", action="store_true", help="Do not write per-face portrait JSON files.")
    parser.add_argument("--keep-temp-video", action="store_true", help="Keep downloaded YouTube video after analysis.")
    parser.add_argument("--no-rejected", action="store_true", help="Do not save rejected crops.")
    return parser.parse_args()


def is_youtube_source(source: str) -> bool:
    lowered = source.lower()
    return "youtube.com/" in lowered or "youtu.be/" in lowered


def main() -> int:
    from .worker import DatasetBuildPipeline

    args = parse_args()
    step_mode = StepMode.FRAMES if args.step_frames else StepMode.SECONDS
    step_value = args.step_frames if args.step_frames else args.step_seconds
    settings = BuilderSettings(
        source=args.source,
        source_type=SourceType.YOUTUBE if is_youtube_source(args.source) else SourceType.FILE,
        output_dir=Path(args.output),
        step_mode=step_mode,
        step_value=step_value,
        save_best_face_only=not args.keep_all_faces,
        remove_duplicates=not args.keep_duplicates,
        create_portrait_json=not args.no_portrait_json,
        delete_temp_video=not args.keep_temp_video,
        save_rejected=not args.no_rejected,
    )
    pipeline = DatasetBuildPipeline(settings, log=print)
    summary = pipeline.run()
    print(f"Готово: {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
