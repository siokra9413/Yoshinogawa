from __future__ import annotations

from enum import IntEnum


class DetectionClass(IntEnum):
    """検出クラスID。"""
    PERSON = 0
    VEHICLE = 2
