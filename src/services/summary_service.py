from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.config.constants import SUMMARY_CSV_FILENAME, SUMMARY_JSON_FILENAME
from src.enums.detection_class import DetectionClass
from src.enums.detection_source import DetectionSource
from src.enums.review_status import ReviewStatus

if TYPE_CHECKING:
    from src.state.app_state import AppState


class ExportSummaryBuilder:
    """教師データ出力の集計サマリーを構築する。"""

    def build(self, state: "AppState", export_result: dict) -> dict:
        ok = ng = person = vehicle = added = 0
        for r in state.records:
            for d in r.detections:
                if d.status == ReviewStatus.OK:
                    ok += 1
                elif d.status == ReviewStatus.NG:
                    ng += 1
                if d.class_id == DetectionClass.PERSON:
                    person += 1
                elif d.class_id == DetectionClass.VEHICLE:
                    vehicle += 1
                if d.source == DetectionSource.MANUAL:
                    added += 1
        review_target = state.stats.review_target_count
        adopt_rate = (ok / review_target * 100) if review_target else 0.0
        reject_rate = (ng / review_target * 100) if review_target else 0.0
        return {
            "総画像数": state.stats.total_images,
            "総検出数": state.stats.total_detections,
            "レビュー対象数": review_target,
            "採用件数": ok,
            "却下件数": ng,
            "採用率(%)": round(adopt_rate, 1),
            "却下率(%)": round(reject_rate, 1),
            "Person件数": person,
            "Vehicle件数": vehicle,
            "追加BBox数": added,
            "削除BBox数": state.deleted_count,
            "最終教師データ画像数": export_result["final_image_count"],
            "最終教師データBBox数": export_result["final_bbox_count"],
        }


def save_summary(summary: dict, out_dir: Path) -> None:
    df = pd.DataFrame([summary])
    df.to_csv(out_dir / SUMMARY_CSV_FILENAME, index=False, encoding="utf-8-sig")
    (out_dir / SUMMARY_JSON_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
