from __future__ import annotations

from enum import Enum


class ReviewStatus(str, Enum):
    """BBoxのレビュー判定状態。"""
    OK = "OK"  # 採用
    NG = "NG"  # 却下
    UNCONFIRMED = "未確認"

    def __str__(self) -> str:
        # (str, Enum)のみだとstr()が"ReviewStatus.OK"のような表記になり、
        # JSON/CSV出力の内容が変わってしまうため、値そのものを返すよう明示する。
        return str(self.value)
