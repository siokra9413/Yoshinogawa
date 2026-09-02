from __future__ import annotations

from src.enums.detection_class import DetectionClass

CLASS_NAMES = {DetectionClass.PERSON: "Person", DetectionClass.VEHICLE: "Vehicle"}

REVIEW_MIN = 0.3
REVIEW_MAX = 0.5  # この値以上の検出はレビュー対象外・既定で採用済み扱い（教師データ出力には含める）

MAX_IMAGES = 1000  # これを超える画像数のフォルダは読み込まずアラートを出す

STATE_FILENAME = "review_state.json"
RESULT_CSV_FILENAME = "review_result.csv"
SUMMARY_CSV_FILENAME = "review_summary.csv"
SUMMARY_JSON_FILENAME = "review_summary.json"

FILTER_OPTIONS = ["全件", "Personのみ", "Vehicleのみ", "採用のみ", "却下のみ", "未確認のみ"]
