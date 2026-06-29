from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(slots=True)
class ExtractedFace:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    box: tuple[int, int, int, int]
    source_frame_size: tuple[int, int]


class VideoFaceExtractor:
    def __init__(self, video_path: Path, margin: float = 0.25) -> None:
        self.video_path = Path(video_path)
        self.margin = margin
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("Не удалось загрузить OpenCV Haar cascade для face detection.")

    def iter_faces(
        self,
        step_frames: int | None = None,
        step_seconds: float | None = None,
        best_only: bool = True,
    ) -> Iterator[ExtractedFace]:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval = step_frames or max(1, int(round((step_seconds or 1.0) * fps)))
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % interval != 0:
                    frame_index += 1
                    continue

                faces = self._detect(frame)
                if best_only and faces:
                    faces = [max(faces, key=lambda item: item[2] * item[3])]

                height, width = frame.shape[:2]
                for box in faces:
                    crop = self._crop(frame, box)
                    yield ExtractedFace(
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                        image=crop,
                        box=box,
                        source_frame_size=(width, height),
                    )
                frame_index += 1
        finally:
            capture.release()

    def count_sampled_frames(self, step_frames: int | None = None, step_seconds: float | None = None) -> int:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            return 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        interval = step_frames or max(1, int(round((step_seconds or 1.0) * fps)))
        return (total + interval - 1) // interval if total else 0

    def _detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return [tuple(map(int, face)) for face in faces]

    def _crop(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = box
        pad_x = int(width * self.margin)
        pad_y = int(height * self.margin)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(frame.shape[1], x + width + pad_x)
        bottom = min(frame.shape[0], y + height + pad_y)
        return frame[top:bottom, left:right].copy()
