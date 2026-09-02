from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScanStats:
    total_images: int = 0
    total_txt: int = 0
    total_detections: int = 0
    person_count: int = 0
    vehicle_count: int = 0
    review_target_count: int = 0
    review_target_images: int = 0
