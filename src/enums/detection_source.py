from __future__ import annotations

from enum import Enum


class DetectionSource(str, Enum):
    """検出結果の由来。"""
    AI = "ai"
    MANUAL = "manual"

    def __str__(self) -> str:
        # (str, Enum)のみだとstr()が"DetectionSource.AI"のような表記になり、
        # JSON/CSV出力の内容が変わってしまうため、値そのものを返すよう明示する。
        return str(self.value)
