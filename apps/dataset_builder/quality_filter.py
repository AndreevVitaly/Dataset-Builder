from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class QualityResult:
    status: str
    quality: float
    warnings: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    portrait: dict[str, Any] = field(default_factory=dict)


class QualityFilter:
    def __init__(self, min_face_size: int = 96, blur_threshold: float = 70.0) -> None:
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold

    def analyze(self, image: np.ndarray, metadata: dict[str, Any]) -> QualityResult:
        warnings: list[str] = []
        rejection_reasons: list[str] = []
        height, width = image.shape[:2]

        if min(width, height) < self.min_face_size:
            rejection_reasons.append("лицо слишком маленькое")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.blur_threshold:
            rejection_reasons.append("размытие")

        portrait = self._analyze_with_portrait_core(image, metadata)
        core_warnings = portrait.get("warnings") or []
        if isinstance(core_warnings, list):
            warnings.extend(str(item) for item in core_warnings)
        if portrait.get("landmarks_detected") is False:
            rejection_reasons.append("не удалось определить landmarks")
        if portrait.get("occluded") is True:
            rejection_reasons.append("лицо частично закрыто")
        pose = portrait.get("pose") or {}
        if isinstance(pose, dict) and max(abs(float(pose.get(key, 0.0))) for key in ("yaw", "pitch", "roll")) > 35:
            rejection_reasons.append("сильный наклон")

        quality = float(portrait.get("quality", min(1.0, blur_score / max(self.blur_threshold * 2, 1.0))))
        if rejection_reasons:
            status = "rejected"
        elif warnings:
            status = "warning"
        else:
            status = "passed"

        portrait.setdefault("quality", quality)
        portrait.setdefault("warnings", warnings)
        portrait.setdefault("metadata", metadata)
        portrait.setdefault("opencv_metrics", {"blur_score": blur_score, "width": width, "height": height})
        return QualityResult(status, quality, warnings, rejection_reasons, portrait)

    def write_portrait_json(self, path: Path, result: QualityResult) -> None:
        data = dict(result.portrait)
        data["dataset_builder"] = {
            "status": result.status,
            "warnings": result.warnings,
            "rejection_reasons": result.rejection_reasons,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _analyze_with_portrait_core(self, image: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
        try:
            portrait_core = importlib.import_module("portrait_core")
        except ImportError:
            return {
                "portrait_core_available": False,
                "warnings": ["portrait_core недоступен, использована базовая OpenCV-оценка"],
                "metadata": metadata,
            }

        for name in ("analyze_portrait", "analyze_image", "analyze"):
            func = getattr(portrait_core, name, None)
            if callable(func):
                return self._normalize_result(func(image))

        analyzer_cls = getattr(portrait_core, "PortraitAnalyzer", None)
        if analyzer_cls is not None:
            analyzer = analyzer_cls()
            for name in ("analyze_image", "analyze"):
                func = getattr(analyzer, name, None)
                if callable(func):
                    return self._normalize_result(func(image))

        return {
            "portrait_core_available": True,
            "warnings": ["В portrait_core не найдена публичная функция анализа изображения"],
            "metadata": metadata,
        }

    def _normalize_result(self, result: Any) -> dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        if hasattr(result, "__dict__"):
            return dict(result.__dict__)
        return {"result": str(result)}
