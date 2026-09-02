from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from bbox_review_tool.config.constants import RESULT_CSV_FILENAME

if TYPE_CHECKING:
    from bbox_review_tool.state.app_state import AppState


class CsvRepository:
    """検出ごとのレビュー結果(review_result.csv)の書き出しを担当する。"""

    def path_for(self, root: Path) -> Path:
        return root / RESULT_CSV_FILENAME

    def save(self, state: "AppState") -> None:
        for root in state.roots:
            recs = [r for r in state.records if r.root == root]
            rows = []
            for r in recs:
                for d in r.detections:
                    rows.append({
                        "image_name": r.name,
                        "bbox_id": d.bbox_id,
                        "status": d.status,
                    })
            df = pd.DataFrame(rows, columns=["image_name", "bbox_id", "status"])
            try:
                df.to_csv(self.path_for(root), index=False, encoding="utf-8-sig")
            except OSError as e:
                state.warnings.append(f"{RESULT_CSV_FILENAME} の保存に失敗しました ({root}): {e}")
