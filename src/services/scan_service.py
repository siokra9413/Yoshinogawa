from __future__ import annotations

import re
from pathlib import Path

from src.config.constants import MAX_IMAGES, REVIEW_MAX, REVIEW_MIN
from src.enums.detection_class import DetectionClass
from src.enums.detection_source import DetectionSource
from src.enums.review_status import ReviewStatus
from src.models.detection import Detection
from src.models.image_record import ImageRecord
from src.models.scan_stats import ScanStats

DETECTION_LINE_RE = re.compile(
    r"^\s*([-+\d.eE]+)\s+(\d+)\s+\[\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,"
    r"\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\]\s*$"
)


def parse_detection_line(line: str):
    line = line.strip()
    if not line:
        return None
    m = DETECTION_LINE_RE.match(line)
    if not m:
        return None
    try:
        conf = float(m.group(1))
        cls = int(m.group(2))
        xmin, ymin, xmax, ymax = (float(m.group(i)) for i in range(3, 7))
    except ValueError:
        return None
    return {"confidence": conf, "class_id": cls, "bbox": [xmin, ymin, xmax, ymax]}


def parse_detection_file(txt_path: Path, warnings: list) -> list:
    results = []
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                parsed = parse_detection_line(line)
                if parsed is None:
                    warnings.append(f"{txt_path.name}:{lineno} 行を解析できません: {line.strip()}")
                    continue
                results.append(parsed)
    except OSError as e:
        warnings.append(f"{txt_path} を読み込めません: {e}")
    return results


def find_images(root: Path):
    seen = set()
    for pattern in ("*.jpg", "*.JPG", "*.Jpg", "*.jpeg", "*.JPEG"):
        for p in root.rglob(pattern):
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield p


class TooManyImagesError(Exception):
    def __init__(self):
        super().__init__(f"画像数が上限（{MAX_IMAGES}枚）を超えています")


def check_image_count_within_limit(roots: list):
    """全フォルダ合計の画像数を数える。上限を超えた時点で走査を打ち切って例外を送出する
    （全件走査してから比較すると、巨大フォルダでハング/メモリ枯渇の原因になる）。"""
    total = 0
    for root in roots:
        for _ in find_images(root):
            total += 1
            if total > MAX_IMAGES:
                raise TooManyImagesError()


class FolderScanner:
    """対象フォルダ配下の画像・AI検出結果txtを走査し、ImageRecord一覧とScanStatsを構築する。"""

    def scan(self, roots: list, warnings: list):
        check_image_count_within_limit(roots)
        combined_stats = ScanStats()
        records = []
        for root in roots:
            sub_records, sub_stats = self._scan_folder(root, warnings)
            records.extend(sub_records)
            for key in vars(combined_stats):
                setattr(combined_stats, key, getattr(combined_stats, key) + getattr(sub_stats, key))
        return records, combined_stats

    def _scan_folder(self, root: Path, warnings: list):
        stats = ScanStats()
        records = []
        image_paths = sorted(find_images(root), key=lambda p: str(p).lower())
        for img_path in image_paths:
            stats.total_images += 1
            txt_path = img_path.with_suffix(".txt")
            has_txt = txt_path.exists()
            rel_key = str(img_path.relative_to(root))
            detections = []
            high_conf_detections = []
            has_review_target = False
            if has_txt:
                stats.total_txt += 1
                raw = parse_detection_file(txt_path, warnings)
                bbox_id = 0
                high_conf_bbox_id = 0
                for d in raw:
                    conf = d["confidence"]
                    if conf >= REVIEW_MAX:
                        # レビュー対象外・既定で採用済み扱い。UIには表示せず、出力時にそのまま含める。
                        high_conf_detections.append(Detection(
                            bbox_id=high_conf_bbox_id, class_id=d["class_id"], confidence=conf,
                            bbox=d["bbox"], source=DetectionSource.AI, status=ReviewStatus.OK,
                        ))
                        high_conf_bbox_id += 1
                        continue
                    if conf < REVIEW_MIN:
                        continue
                    stats.total_detections += 1
                    if d["class_id"] == DetectionClass.PERSON:
                        stats.person_count += 1
                    elif d["class_id"] == DetectionClass.VEHICLE:
                        stats.vehicle_count += 1
                    stats.review_target_count += 1
                    has_review_target = True
                    detections.append(Detection(
                        bbox_id=bbox_id, class_id=d["class_id"], confidence=conf,
                        bbox=d["bbox"], source=DetectionSource.AI, status=ReviewStatus.UNCONFIRMED,
                    ))
                    bbox_id += 1
            if has_review_target:
                stats.review_target_images += 1
            records.append(ImageRecord(
                path=img_path, root=root, rel_key=rel_key,
                txt_path=txt_path if has_txt else None,
                has_txt=has_txt, detections=detections,
                high_conf_detections=high_conf_detections,
            ))
        return records, stats
