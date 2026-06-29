from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class DuplicateMatch:
    duplicate: bool
    distance: int | None = None


class PerceptualHashDuplicateFilter:
    def __init__(self, threshold: int = 6, hash_size: int = 8) -> None:
        self.threshold = threshold
        self.hash_size = hash_size
        self._seen: list[int] = []

    def check(self, image: np.ndarray) -> DuplicateMatch:
        value = self.hash_image(image, self.hash_size)
        distances = [(value ^ existing).bit_count() for existing in self._seen]
        if distances:
            distance = min(distances)
            if distance <= self.threshold:
                return DuplicateMatch(True, distance)
        self._seen.append(value)
        return DuplicateMatch(False, None)

    @staticmethod
    def hash_image(image: np.ndarray, hash_size: int = 8) -> int:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = small[:, 1:] > small[:, :-1]
        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bit)
        return value
