from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SourceType(str, Enum):
    FILE = "file"
    YOUTUBE = "youtube"


class StepMode(str, Enum):
    FRAMES = "frames"
    SECONDS = "seconds"


@dataclass(slots=True)
class BuilderSettings:
    source: str
    output_dir: Path = Path("dataset")
    source_type: SourceType = SourceType.FILE
    step_mode: StepMode = StepMode.SECONDS
    step_value: float = 1.0
    save_best_face_only: bool = True
    remove_duplicates: bool = True
    create_portrait_json: bool = True
    delete_temp_video: bool = True
    save_rejected: bool = True
    duplicate_threshold: int = 6
    min_face_size: int = 96
    blur_threshold: float = 70.0
    temp_dir: Path = Path("temp_downloads")

    def validate(self) -> None:
        if not self.source:
            raise ValueError("Источник не указан.")
        if self.step_value <= 0:
            raise ValueError("Шаг кадров должен быть больше нуля.")
        if self.step_mode == StepMode.FRAMES and int(self.step_value) != self.step_value:
            raise ValueError("Шаг в кадрах должен быть целым числом.")

    @property
    def step_frames(self) -> int | None:
        if self.step_mode == StepMode.FRAMES:
            return max(1, int(self.step_value))
        return None

    @property
    def step_seconds(self) -> float | None:
        if self.step_mode == StepMode.SECONDS:
            return float(self.step_value)
        return None
