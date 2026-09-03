from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.config.constants import SUMMARY_CSV_FILENAME, SUMMARY_JSON_FILENAME

if TYPE_CHECKING:
    from src.state.app_state import AppState


class ExportSummaryBuilder:
    """教師データ出力の集計サマリーを構築する。"""

    def build(self, state: "AppState", export_result: dict) -> dict:
        return {
            "総画像数": state.stats.total_images,
            "出力画像数": export_result["final_image_count"],
            "総検出数": state.stats.total_detections,
            "採用検出数": export_result["final_bbox_count"],
            "Person件数": export_result["person_count"],
            "Vehicle件数": export_result["vehicle_count"],
            "train画像数": export_result["train_image_count"],
            "val画像数": export_result["val_image_count"],
        }


def save_summary(summary: dict, out_dir: Path) -> None:
    df = pd.DataFrame([summary])
    df.to_csv(out_dir / SUMMARY_CSV_FILENAME, index=False, encoding="utf-8-sig")
    (out_dir / SUMMARY_JSON_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
