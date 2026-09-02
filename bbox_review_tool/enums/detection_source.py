from __future__ import annotations

from enum import StrEnum


class DetectionSource(StrEnum):
    """検出結果の由来。"""
    AI = "ai"
    MANUAL = "manual"
