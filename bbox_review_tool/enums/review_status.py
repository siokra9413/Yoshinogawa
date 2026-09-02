from __future__ import annotations

from enum import StrEnum


class ReviewStatus(StrEnum):
    """BBoxのレビュー判定状態。"""
    OK = "OK"  # 採用
    NG = "NG"  # 却下
    UNCONFIRMED = "未確認"
