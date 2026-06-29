from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from .downloader import download_youtube_video
from .duplicate_filter import PerceptualHashDuplicateFilter
from .extractor import VideoFaceExtractor
from .quality_filter import QualityFilter, QualityResult
from .settings import BuilderSettings, SourceType


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class BuildSummary:
    total_frames: int = 0
    detected_faces: int = 0
    saved: int = 0
    rejected: int = 0
    duplicates: int = 0
    warnings: int = 0
    processing_time_seconds: float = 0.0
    output_dir: str = ""
    log: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "общее количество кадров": self.total_frames,
            "обнаружено лиц": self.detected_faces,
            "сохранено": self.saved,
            "отброшено": self.rejected,
            "дубликатов": self.duplicates,
            "предупреждений": self.warnings,
            "время обработки": self.processing_time_seconds,
            "output_dir": self.output_dir,
        }


class StopRequested(RuntimeError):
    pass


class DatasetBuildPipeline:
    def __init__(
        self,
        settings: BuilderSettings,
        log: LogCallback | None = None,
        progress: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.settings = settings
        self.log_callback = log
        self.progress_callback = progress
        self.should_stop = should_stop or (lambda: False)
        self.summary = BuildSummary(output_dir=str(settings.output_dir))

    def run(self) -> BuildSummary:
        self.settings.validate()
        started = time.monotonic()
        video_path: Path | None = None
        temp_video = False

        self._prepare_output()
        try:
            if self.settings.source_type == SourceType.YOUTUBE:
                self._log("Скачивание YouTube-видео")
                video_path = download_youtube_video(
                    self.settings.source,
                    self.settings.temp_dir,
                    lambda text, percent: self._log(f"{text}: {percent:.1f}%" if percent is not None else text),
                )
                temp_video = True
            else:
                video_path = Path(self.settings.source)

            self._process_video(video_path)
        finally:
            if temp_video and video_path and self.settings.delete_temp_video and video_path.exists():
                video_path.unlink()
                self._log("Временное видео удалено")
            self.summary.processing_time_seconds = round(time.monotonic() - started, 3)
            self._write_summary()

        return self.summary

    def _process_video(self, video_path: Path) -> None:
        extractor = VideoFaceExtractor(video_path)
        quality_filter = QualityFilter(self.settings.min_face_size, self.settings.blur_threshold)
        duplicate_filter = PerceptualHashDuplicateFilter(self.settings.duplicate_threshold)
        self.summary.total_frames = extractor.count_sampled_frames(self.settings.step_frames, self.settings.step_seconds)
        self._log(f"К анализу кадров: {self.summary.total_frames}")

        for index, face in enumerate(
            extractor.iter_faces(self.settings.step_frames, self.settings.step_seconds, self.settings.save_best_face_only),
            start=1,
        ):
            if self.should_stop():
                raise StopRequested("Остановлено пользователем.")
            self.summary.detected_faces += 1
            metadata = {
                "frame_index": face.frame_index,
                "timestamp_seconds": face.timestamp_seconds,
                "source_frame_size": face.source_frame_size,
                "box": face.box,
            }
            result = quality_filter.analyze(face.image, metadata)
            duplicate = duplicate_filter.check(face.image) if self.settings.remove_duplicates else None
            if duplicate and duplicate.duplicate:
                self.summary.duplicates += 1
                self._save_face("duplicates", face.frame_index, face.image, result, quality_filter)
            else:
                self._store_quality_result(face.frame_index, face.image, result, quality_filter)

            if self.progress_callback:
                self.progress_callback(index, max(index, self.summary.total_frames))

        self._log("Анализ завершен")

    def _store_quality_result(
        self,
        frame_index: int,
        image,
        result: QualityResult,
        quality_filter: QualityFilter,
    ) -> None:
        if result.status == "rejected":
            self.summary.rejected += 1
            if self.settings.save_rejected:
                self._save_face("rejected", frame_index, image, result, quality_filter)
            return
        if result.status == "warning":
            self.summary.warnings += 1
        self.summary.saved += 1
        self._save_face(result.status, frame_index, image, result, quality_filter)

    def _save_face(
        self,
        category: str,
        frame_index: int,
        image,
        result: QualityResult,
        quality_filter: QualityFilter,
    ) -> None:
        folder = self.settings.output_dir / category
        image_path = folder / f"frame{frame_index:06d}.jpg"
        cv2.imwrite(str(image_path), image)
        if self.settings.create_portrait_json:
            quality_filter.write_portrait_json(image_path.with_name(f"{image_path.stem}_portrait.json"), result)
        self._log(f"{category}: {image_path.name}")

    def _prepare_output(self) -> None:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("passed", "warning", "rejected", "duplicates"):
            (self.settings.output_dir / name).mkdir(exist_ok=True)

    def _write_summary(self) -> None:
        output = self.settings.output_dir
        (output / "summary.json").write_text(
            json.dumps(self.summary.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "log.txt").write_text("\n".join(self.summary.log), encoding="utf-8")

    def _log(self, message: str) -> None:
        self.summary.log.append(message)
        if self.log_callback:
            self.log_callback(message)
