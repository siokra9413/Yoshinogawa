from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.models.detection import Detection


@dataclass
class ImageRecord:
    path: Path
    root: Path
    rel_key: str
    txt_path: Optional[Path]
    has_txt: bool
    detections: list[Detection] = field(default_factory=list)
    # confidence>=REVIEW_MAX、既定でstatus=OK。UI非表示・出力時にそのまま含める
    high_conf_detections: list[Detection] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name
