from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from src.config.constants import CLASS_NAMES
from src.enums.detection_source import DetectionSource
from src.enums.review_status import ReviewStatus


@dataclass
class Detection:
    bbox_id: int
    class_id: int
    confidence: Optional[float]
    bbox: list
    source: str
    status: str

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Detection(
            bbox_id=int(d["bbox_id"]),
            class_id=int(d["class_id"]),
            confidence=d.get("confidence"),
            bbox=[float(v) for v in d["bbox"]],
            source=d.get("source", DetectionSource.AI),
            status=d.get("status", ReviewStatus.UNCONFIRMED),
        )

    @property
    def color_category(self) -> str:
        if self.source == DetectionSource.MANUAL:
            return "manual"
        return "review"

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))
