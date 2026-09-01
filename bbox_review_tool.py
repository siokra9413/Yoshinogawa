#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI物体検出結果 レビュー・アノテーションツール（単一ファイル版）

対象フォルダ配下の *.jpg / *.txt（AI検出結果）を再帰的に読み込み、
信頼値0.3〜0.5の境界検出のみを対象に目視レビューを行い、
YOLO形式の教師データセット（train/val分割済み）を出力する。
信頼値0.5以上の検出はそもそも読み込まない。BBoxの位置編集は行わない
（操作はショートカットキーのみで完結する）。
"""

from __future__ import annotations

import json
import random
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, QRectF, QSettings, Signal, QObject, QThread
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QShortcut, QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsItem, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLabel, QPushButton, QFileDialog, QSplitter,
    QTableWidget, QTableWidgetItem, QComboBox, QLineEdit, QMessageBox,
    QStatusBar, QAbstractItemView, QHeaderView, QProgressDialog,
    QGroupBox, QSizePolicy, QTabWidget, QButtonGroup, QFrame,
)


# ============================================================
# 定数
# ============================================================

CLASS_NAMES = {0: "Person", 2: "Vehicle"}

REVIEW_MIN = 0.3
REVIEW_MAX = 0.5  # この値以上の検出はそもそも読み込まない（仕様確定事項）

MAX_IMAGES = 1000  # これを超える画像数のフォルダは読み込まずアラートを出す

STATUS_OK = "OK"
STATUS_NG = "NG"
STATUS_HOLD = "保留"
STATUS_UNCONFIRMED = "未確認"

SOURCE_AI = "ai"
SOURCE_MANUAL = "manual"

STATE_FILENAME = "review_state.json"
RESULT_CSV_FILENAME = "review_result.csv"
SUMMARY_CSV_FILENAME = "review_summary.csv"
SUMMARY_JSON_FILENAME = "review_summary.json"

FILTER_OPTIONS = ["全件", "Personのみ", "Vehicleのみ", "OKのみ", "NGのみ", "保留のみ", "未確認のみ"]

SELECTED_COLOR = QColor(0, 255, 255)

DETECTION_LINE_RE = re.compile(
    r"^\s*([-+\d.eE]+)\s+(\d+)\s+\[\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,"
    r"\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\]\s*$"
)


def imread_unicode(path: Path):
    """Windowsの非ASCIIパスでも安全に画像を読み込む。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ============================================================
# データモデル
# ============================================================

@dataclass
class Detection:
    bbox_id: int
    class_id: int
    confidence: Optional[float]
    bbox: list
    source: str
    status: str

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Detection(
            bbox_id=int(d["bbox_id"]),
            class_id=int(d["class_id"]),
            confidence=d.get("confidence"),
            bbox=[float(v) for v in d["bbox"]],
            source=d.get("source", SOURCE_AI),
            status=d.get("status", STATUS_UNCONFIRMED),
        )

    @property
    def color_category(self) -> str:
        if self.source == SOURCE_MANUAL:
            return "manual"
        return "review"

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))


def bbox_color(detection: Detection) -> QColor:
    if detection.color_category == "manual":
        return QColor(40, 200, 60)
    return QColor(230, 200, 0)


@dataclass
class ImageRecord:
    path: Path
    rel_key: str
    txt_path: Optional[Path]
    has_txt: bool
    detections: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class ScanStats:
    total_images: int = 0
    total_txt: int = 0
    total_detections: int = 0
    person_count: int = 0
    vehicle_count: int = 0
    review_target_count: int = 0
    review_target_images: int = 0


# ============================================================
# 検出結果パース・フォルダスキャン
# ============================================================

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


def scan_folder(root: Path, warnings: list):
    stats = ScanStats()
    records = []
    image_paths = []
    for p in find_images(root):
        image_paths.append(p)
        if len(image_paths) > MAX_IMAGES:
            # 上限を超えた時点で走査を打ち切る（全件走査すると巨大フォルダでハング/メモリ枯渇の原因になる）
            raise TooManyImagesError()
    image_paths.sort(key=lambda p: str(p).lower())
    for img_path in image_paths:
        stats.total_images += 1
        txt_path = img_path.with_suffix(".txt")
        has_txt = txt_path.exists()
        rel_key = str(img_path.relative_to(root))
        detections = []
        has_review_target = False
        if has_txt:
            stats.total_txt += 1
            raw = parse_detection_file(txt_path, warnings)
            bbox_id = 0
            for d in raw:
                conf = d["confidence"]
                if not (REVIEW_MIN <= conf < REVIEW_MAX):
                    continue
                stats.total_detections += 1
                if d["class_id"] == 0:
                    stats.person_count += 1
                elif d["class_id"] == 2:
                    stats.vehicle_count += 1
                stats.review_target_count += 1
                has_review_target = True
                detections.append(Detection(
                    bbox_id=bbox_id, class_id=d["class_id"], confidence=conf,
                    bbox=d["bbox"], source=SOURCE_AI, status=STATUS_UNCONFIRMED,
                ))
                bbox_id += 1
        if has_review_target:
            stats.review_target_images += 1
        records.append(ImageRecord(
            path=img_path, rel_key=rel_key,
            txt_path=txt_path if has_txt else None,
            has_txt=has_txt, detections=detections,
        ))
    return records, stats


# ============================================================
# アプリケーション状態
# ============================================================

class AppState:
    def __init__(self, root: Path):
        self.root = root
        self.records: list = []
        self.stats = ScanStats()
        self.warnings: list = []
        self.current_index: int = 0
        self.filter_mode: str = "全件"
        self.no_detection_mode: bool = False
        self.deleted_count: int = 0

    def state_path(self) -> Path:
        return self.root / STATE_FILENAME

    def result_csv_path(self) -> Path:
        return self.root / RESULT_CSV_FILENAME

    def load_or_scan(self):
        self.records, self.stats = scan_folder(self.root, self.warnings)
        self._try_load_state()

    def _try_load_state(self):
        path = self.state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.warnings.append(f"状態ファイルの読込に失敗しました: {e}")
            return
        by_key = {r.rel_key: r for r in self.records}
        for img_data in data.get("images", []):
            rec = by_key.get(img_data.get("rel_key"))
            if rec is None:
                continue
            try:
                rec.detections = [Detection.from_dict(d) for d in img_data.get("detections", [])]
            except (KeyError, TypeError, ValueError):
                continue
        if self.records:
            self.current_index = max(0, min(data.get("current_index", 0), len(self.records) - 1))
        self.filter_mode = data.get("filter_mode", self.filter_mode)
        self.no_detection_mode = bool(data.get("no_detection_mode", False))
        self.deleted_count = int(data.get("deleted_count", 0))

    def save_state(self):
        data = {
            "root": str(self.root),
            "current_index": self.current_index,
            "filter_mode": self.filter_mode,
            "no_detection_mode": self.no_detection_mode,
            "deleted_count": self.deleted_count,
            "images": [
                {"rel_key": r.rel_key, "detections": [d.to_dict() for d in r.detections]}
                for r in self.records
            ],
        }
        tmp_path = self.state_path().with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp_path.replace(self.state_path())
        except OSError as e:
            self.warnings.append(f"状態の自動保存に失敗しました: {e}")

    def save_result_csv(self):
        rows = []
        for r in self.records:
            for d in r.detections:
                rows.append({
                    "image_name": r.name,
                    "bbox_id": d.bbox_id,
                    "status": d.status,
                })
        df = pd.DataFrame(rows, columns=["image_name", "bbox_id", "status"])
        try:
            df.to_csv(self.result_csv_path(), index=False, encoding="utf-8-sig")
        except OSError as e:
            self.warnings.append(f"{RESULT_CSV_FILENAME} の保存に失敗しました: {e}")

    def _matches_filter(self, r: ImageRecord) -> bool:
        f = self.filter_mode
        if f == "全件":
            return True
        if f == "Personのみ":
            return any(d.class_id == 0 for d in r.detections)
        if f == "Vehicleのみ":
            return any(d.class_id == 2 for d in r.detections)
        if f == "OKのみ":
            return any(d.status == STATUS_OK for d in r.detections)
        if f == "NGのみ":
            return any(d.status == STATUS_NG for d in r.detections)
        if f == "保留のみ":
            return any(d.status == STATUS_HOLD for d in r.detections)
        if f == "未確認のみ":
            return any(d.status == STATUS_UNCONFIRMED for d in r.detections)
        return True

    def filtered_indices(self) -> list:
        idxs = []
        for i, r in enumerate(self.records):
            if self.no_detection_mode:
                if r.has_txt and len(r.detections) > 0:
                    continue
                idxs.append(i)
            else:
                if not r.has_txt or not r.detections:
                    continue
                if self._matches_filter(r):
                    idxs.append(i)
        return idxs

    def find_next_matching(self, query: str, start_after: int):
        idxs = self.filtered_indices()
        if not idxs:
            return None
        query = query.lower()
        pos = idxs.index(start_after) if start_after in idxs else -1
        n = len(idxs)
        for step in range(1, n + 1):
            i = idxs[(pos + step) % n]
            if query in self.records[i].name.lower():
                return i
        return None

    def progress(self):
        denom = 0
        numer = 0
        for r in self.records:
            for d in r.detections:
                if d.source == SOURCE_AI and d.confidence is not None and REVIEW_MIN <= d.confidence < REVIEW_MAX:
                    denom += 1
                    if d.status != STATUS_UNCONFIRMED:
                        numer += 1
        return numer, denom


# ============================================================
# 教師データ出力
# ============================================================

def export_dataset(state: AppState, out_dir: Path, warnings: list) -> dict:
    entries = []
    for r in state.records:
        included = [d for d in r.detections if d.status == STATUS_OK]
        if included:
            entries.append((r, included))

    keys = [r.rel_key for r, _ in entries]
    shuffled = list(keys)
    random.Random(42).shuffle(shuffled)
    split_point = int(len(shuffled) * 0.8)
    train_keys = set(shuffled[:split_point])

    images_train = out_dir / "images" / "train"
    images_val = out_dir / "images" / "val"
    labels_train = out_dir / "labels" / "train"
    labels_val = out_dir / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    final_bbox_count = 0
    final_image_count = 0

    for r, included in entries:
        is_train = r.rel_key in train_keys
        img_dest_dir = images_train if is_train else images_val
        lbl_dest_dir = labels_train if is_train else labels_val
        dest_img_path = img_dest_dir / r.rel_key
        dest_lbl_path = (lbl_dest_dir / r.rel_key).with_suffix(".txt")
        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl_path.parent.mkdir(parents=True, exist_ok=True)

        img = imread_unicode(r.path)
        if img is None:
            warnings.append(f"{r.path} を読み込めず出力から除外しました")
            continue
        h, w = img.shape[:2]

        lines = []
        for d in included:
            xmin, ymin, xmax, ymax = d.bbox
            xmin_c, xmax_c = sorted((max(0.0, min(xmin, w)), max(0.0, min(xmax, w))))
            ymin_c, ymax_c = sorted((max(0.0, min(ymin, h)), max(0.0, min(ymax, h))))
            bw = xmax_c - xmin_c
            bh = ymax_c - ymin_c
            if bw <= 0 or bh <= 0:
                continue
            cx = (xmin_c + xmax_c) / 2.0 / w
            cy = (ymin_c + ymax_c) / 2.0 / h
            lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")

        if not lines:
            continue

        try:
            shutil.copy2(r.path, dest_img_path)
        except OSError as e:
            warnings.append(f"{r.path} のコピーに失敗しました: {e}")
            continue

        dest_lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        final_bbox_count += len(lines)
        final_image_count += 1

    yaml_text = (
        "path: dataset\n"
        "\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: Person\n"
        "  2: Vehicle\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    return {"final_image_count": final_image_count, "final_bbox_count": final_bbox_count}


def build_summary(state: AppState, export_result: dict) -> dict:
    ok = ng = hold = person = vehicle = added = 0
    for r in state.records:
        for d in r.detections:
            if d.status == STATUS_OK:
                ok += 1
            elif d.status == STATUS_NG:
                ng += 1
            elif d.status == STATUS_HOLD:
                hold += 1
            if d.class_id == 0:
                person += 1
            elif d.class_id == 2:
                vehicle += 1
            if d.source == SOURCE_MANUAL:
                added += 1
    return {
        "総画像数": state.stats.total_images,
        "総検出数": state.stats.total_detections,
        "レビュー対象数": state.stats.review_target_count,
        "OK件数": ok,
        "NG件数": ng,
        "保留件数": hold,
        "Person件数": person,
        "Vehicle件数": vehicle,
        "追加BBox数": added,
        "削除BBox数": state.deleted_count,
        "最終教師データ画像数": export_result["final_image_count"],
        "最終教師データBBox数": export_result["final_bbox_count"],
    }


def save_summary(summary: dict, out_dir: Path):
    df = pd.DataFrame([summary])
    df.to_csv(out_dir / SUMMARY_CSV_FILENAME, index=False, encoding="utf-8-sig")
    (out_dir / SUMMARY_JSON_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ============================================================
# GUI: BBoxアイテム
# ============================================================

class BBoxItem(QGraphicsRectItem):
    """レビュー専用の矩形表示アイテム（位置・サイズ編集は不可、選択のみ）。"""

    def __init__(self, rect: QRectF, color: QColor):
        super().__init__(rect)
        self.detection: Optional[Detection] = None
        self.canvas = None
        self._base_color = color
        self.selected_highlight = False
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_pen()

    def set_color(self, color: QColor):
        self._base_color = color
        self._apply_pen()

    def set_selected_highlight(self, flag: bool):
        self.selected_highlight = flag
        self._apply_pen()
        self.setZValue(1 if flag else 0)

    def _apply_pen(self):
        if self.selected_highlight:
            self.setPen(QPen(SELECTED_COLOR, 5))
            self.setBrush(QBrush(QColor(0, 255, 255, 60)))
        else:
            self.setPen(QPen(self._base_color, 2))
            self.setBrush(QBrush(Qt.NoBrush))

    def scene_rect(self) -> QRectF:
        p = self.pos()
        r = self.rect()
        return QRectF(p.x() + r.x(), p.y() + r.y(), r.width(), r.height())

    def mousePressEvent(self, event):
        if self.canvas is not None:
            self.canvas.select_bbox_item(self)
        super().mousePressEvent(event)


# ============================================================
# GUI: 画像キャンバス（ズーム／パン／BBox選択・表示）
# ============================================================

class ImageCanvas(QGraphicsView):
    bboxSelected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._bbox_items: list = []
        self.selected_item: Optional[BBoxItem] = None
        self._panning = False
        self._pan_start = None
        self._image_size = (0, 0)
        self.setMouseTracking(True)

    # ---- 画像の読込・表示 ----
    def load_record(self, record: ImageRecord):
        self._scene.clear()
        self._bbox_items = []
        self.selected_item = None
        img = imread_unicode(record.path)
        if img is None:
            self._pixmap_item = None
            self._image_size = (0, 0)
            return False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(0, 0, w, h)
        self._image_size = (w, h)
        for det in record.detections:
            self._add_bbox_item(det)
        self.fit_view()
        return True

    def image_size(self):
        return self._image_size

    def fit_view(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_view()

    # ---- BBoxアイテム管理 ----
    def _add_bbox_item(self, det: Detection):
        x1, y1, x2, y2 = det.bbox
        item = BBoxItem(QRectF(0, 0, max(x2 - x1, 1.0), max(y2 - y1, 1.0)), bbox_color(det))
        item.setPos(x1, y1)
        item.detection = det
        item.canvas = self
        self._scene.addItem(item)
        self._bbox_items.append(item)
        return item

    def select_bbox_item(self, item: Optional[BBoxItem]):
        if self.selected_item is not None:
            self.selected_item.set_selected_highlight(False)
        self.selected_item = item
        if item is not None:
            item.set_selected_highlight(True)
        self.bboxSelected.emit(item.detection if item else None)

    def select_detection(self, detection: Optional[Detection]):
        if detection is None:
            self.select_bbox_item(None)
            return
        for item in self._bbox_items:
            if item.detection is detection:
                self.select_bbox_item(item)
                return

    def refresh_colors(self):
        for item in self._bbox_items:
            item.set_color(bbox_color(item.detection))

    # ---- マウス操作: ズーム／パン ----
    def zoom_in(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1.15, 1.15)

    def zoom_out(self):
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(1 / 1.15, 1 / 1.15)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            on_background = item is None or item is self._pixmap_item
            if on_background:
                self._panning = True
                self._pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                self.select_bbox_item(None)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)


# ============================================================
# GUI: 右側レビューパネル
# ============================================================

TOGGLE_ON_STYLE = "QPushButton:checked { background-color: #2e8b57; color: white; font-weight: bold; }"
STATUS_BTN_STYLE = (
    "QPushButton:checked#okBtn { background-color: #2e8b57; color: white; }"
    "QPushButton:checked#ngBtn { background-color: #c0392b; color: white; }"
    "QPushButton:checked#holdBtn { background-color: #b8860b; color: white; }"
)


class ReviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # ---- 常時表示: 画像情報・色の凡例・検出一覧 ----
        self.image_name_label = QLabel("画像: -")
        self.image_name_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.image_name_label)

        self.position_label = QLabel("0 / 0 件表示中")
        layout.addWidget(self.position_label)

        legend = QLabel(
            '<span style="color:#e6c800;">■</span> レビュー対象(0.3〜0.5) '
            '&nbsp;&nbsp;<span style="color:#28c83c;">■</span> 追加アノテーション(過去データ) '
            '&nbsp;&nbsp;<span style="color:#00ffff;">■</span> 選択中'
        )
        legend.setStyleSheet("color: #444;")
        layout.addWidget(legend)
        shortcut_hint = QLabel(
            "検出選択: W(前) / S(次)　画像切替: A(前) / D(次)　状態: O=OK N=NG H=保留"
        )
        shortcut_hint.setStyleSheet("color: #666; font-size: 11px;")
        shortcut_hint.setWordWrap(True)
        layout.addWidget(shortcut_hint)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "信頼値", "クラス", "BBox座標", "状態"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(150)
        # テーブルにキーボードフォーカスが残るとW/S/A/D等のショートカットキーを
        # 行の頭文字検索として奪ってしまうため、フォーカスを持たせない。
        self.table.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.table)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)

        # ---- タブ1: レビュー操作（普段の作業はここだけで完結） ----
        self.tabs = QTabWidget()
        review_tab = QWidget()
        review_layout = QVBoxLayout(review_tab)

        status_box = QGroupBox("選択中BBoxの状態 (O=OK / N=NG / H=保留)")
        status_layout = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK (O)")
        self.ok_btn.setObjectName("okBtn")
        self.ng_btn = QPushButton("NG (N)")
        self.ng_btn.setObjectName("ngBtn")
        self.hold_btn = QPushButton("保留 (H)")
        self.hold_btn.setObjectName("holdBtn")
        for b in (self.ok_btn, self.ng_btn, self.hold_btn):
            b.setCheckable(True)
            b.setStyleSheet(STATUS_BTN_STYLE)
            btn_row.addWidget(b)
        self.status_button_group = QButtonGroup(self)
        self.status_button_group.setExclusive(True)
        for b in (self.ok_btn, self.ng_btn, self.hold_btn):
            self.status_button_group.addButton(b)
        status_layout.addLayout(btn_row)
        status_box.setLayout(status_layout)
        review_layout.addWidget(status_box)

        nav_box = QGroupBox("ナビゲーション")
        nav_layout = QVBoxLayout()
        nav_btn_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 前へ (A)")
        self.next_btn = QPushButton("次へ ▶ (D)")
        nav_btn_row.addWidget(self.prev_btn)
        nav_btn_row.addWidget(self.next_btn)
        nav_layout.addLayout(nav_btn_row)
        self.progress_label = QLabel("0 / 0件確認済（レビュー対象BBox基準）")
        nav_layout.addWidget(self.progress_label)
        nav_box.setLayout(nav_layout)
        review_layout.addWidget(nav_box)

        review_layout.addStretch(1)
        self.tabs.addTab(review_tab, "レビュー操作")

        # ---- タブ2: フィルタ・検索・特殊モード（普段は開かない設定類） ----
        filter_tab = QWidget()
        filter_layout = QVBoxLayout(filter_tab)
        form = QFormLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTER_OPTIONS)
        # コンボボックスにフォーカスが残ると文字キーが項目検索に奪われるため、フォーカスを持たせない。
        self.filter_combo.setFocusPolicy(Qt.NoFocus)
        form.addRow("フィルタ ([ / ]):", self.filter_combo)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("画像名検索 (Ctrl+F, Enter)")
        form.addRow("検索:", self.search_edit)
        filter_layout.addLayout(form)

        self.no_detection_btn = QPushButton("検出なし画像レビューモード：OFF (M)")
        self.no_detection_btn.setCheckable(True)
        self.no_detection_btn.setStyleSheet(TOGGLE_ON_STYLE)
        self.no_detection_btn.toggled.connect(self._on_no_detection_toggled)
        filter_layout.addWidget(self.no_detection_btn)
        filter_layout.addStretch(1)
        self.tabs.addTab(filter_tab, "フィルタ・表示設定")

        layout.addWidget(self.tabs)

        self.export_btn = QPushButton("教師データ出力... (Ctrl+E)")
        self.export_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; background-color: #34495e; color: white; }"
        )
        layout.addWidget(self.export_btn)

        self.setMinimumWidth(260)
        self.setMaximumWidth(280)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def _on_no_detection_toggled(self, checked: bool):
        self.no_detection_btn.setText(f"検出なし画像レビューモード：{'ON' if checked else 'OFF'} (M)")

    def populate_table(self, detections: list):
        self.table.setRowCount(len(detections))
        for row, det in enumerate(detections):
            conf_text = f"{det.confidence:.3f}" if det.confidence is not None else "-"
            coord_text = ", ".join(f"{v:.1f}" for v in det.bbox)
            values = [str(det.bbox_id), conf_text, det.class_name, coord_text, det.status]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, det)
                self.table.setItem(row, col, item)

    def selected_detection(self) -> Optional[Detection]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return item.data(Qt.UserRole) if item else None

    def select_detection_row(self, det: Optional[Detection]):
        if det is None:
            self.table.clearSelection()
            self.sync_status_controls(None)
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) is det:
                self.table.selectRow(row)
                break
        self.sync_status_controls(det)

    def sync_status_controls(self, det: Optional[Detection]):
        """選択中BBoxの状態をOK/NG/保留ボタンに反映する。"""
        buttons = {STATUS_OK: self.ok_btn, STATUS_NG: self.ng_btn, STATUS_HOLD: self.hold_btn}
        self.status_button_group.setExclusive(False)
        for b in buttons.values():
            b.setChecked(False)
        self.status_button_group.setExclusive(True)
        enabled = det is not None
        for b in buttons.values():
            b.setEnabled(enabled)
        if det is not None and det.status in buttons:
            buttons[det.status].setChecked(True)


# ============================================================
# フォルダ読込ワーカー（別スレッドで実行しUIをブロックしない）
# ============================================================

class FolderLoadWorker(QObject):
    succeeded = Signal(object)
    tooManyImages = Signal()
    failed = Signal(str)

    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def run(self):
        try:
            state = AppState(self.root)
            state.load_or_scan()
        except TooManyImagesError:
            self.tooManyImages.emit()
            return
        except Exception:
            # 別スレッドでの未捕捉例外はアプリ全体をクラッシュさせうるため、
            # ここで必ず捕捉してメインスレッド側にエラーとして通知する。
            self.failed.emit(traceback.format_exc())
            return
        self.succeeded.emit(state)


# ============================================================
# GUI: メインウィンドウ
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI物体検出結果 レビュー・アノテーションツール")
        self.settings = QSettings("BBoxReviewTool", "BBoxReviewTool")
        self.state: Optional[AppState] = None

        self._build_ui()
        self._build_shortcuts()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)

        last_root = self.settings.value("last_root", "")
        if last_root and Path(last_root).exists():
            self.load_folder(Path(last_root))

    # ---- UI構築 ----
    def _build_ui(self):
        central = QWidget()
        root_layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.open_btn = QPushButton("対象フォルダを開く... (Ctrl+O)")
        self.open_btn.clicked.connect(self.open_folder_dialog)
        top_bar.addWidget(self.open_btn)

        self.stats_box = QGroupBox("起動時集計")
        stats_layout = QHBoxLayout()
        self.stat_labels = {}
        for key in ["総画像数", "txtファイル数", "総検出数", "Person件数", "Vehicle件数",
                    "0.3～0.5の検出数", "レビュー対象画像数"]:
            item_box = QVBoxLayout()
            item_box.setSpacing(0)
            caption = QLabel(key)
            caption.setStyleSheet("color: #666; font-size: 10px;")
            caption.setAlignment(Qt.AlignCenter)
            value = QLabel("-")
            value.setStyleSheet("font-weight: bold; font-size: 13px;")
            value.setAlignment(Qt.AlignCenter)
            self.stat_labels[key] = value
            item_box.addWidget(caption)
            item_box.addWidget(value)
            stats_layout.addLayout(item_box)
        self.stats_box.setLayout(stats_layout)
        top_bar.addWidget(self.stats_box)
        top_bar.addStretch(1)
        root_layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)
        self.canvas = ImageCanvas()
        self.panel = ReviewPanel()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 12)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter)

        self.setCentralWidget(central)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.panel.ok_btn.clicked.connect(lambda: self.set_current_status(STATUS_OK))
        self.panel.ng_btn.clicked.connect(lambda: self.set_current_status(STATUS_NG))
        self.panel.hold_btn.clicked.connect(lambda: self.set_current_status(STATUS_HOLD))
        self.panel.no_detection_btn.toggled.connect(self.on_no_detection_mode_toggled)
        self.panel.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        self.panel.search_edit.returnPressed.connect(self.on_search)
        self.panel.prev_btn.clicked.connect(self.go_prev)
        self.panel.next_btn.clicked.connect(self.go_next)
        self.panel.export_btn.clicked.connect(self.export_dataset_dialog)
        self.panel.table.itemSelectionChanged.connect(self.on_table_selection_changed)

        self.canvas.bboxSelected.connect(self.on_canvas_selection_changed)

    def _build_shortcuts(self):
        # 画像切替
        QShortcut(QKeySequence("A"), self, activated=self.go_prev)
        QShortcut(QKeySequence("D"), self, activated=self.go_next)
        # 検出（BBox）切替
        QShortcut(QKeySequence("W"), self, activated=self.select_prev_detection)
        QShortcut(QKeySequence("S"), self, activated=self.select_next_detection)
        # 状態設定
        QShortcut(QKeySequence("O"), self, activated=lambda: self.set_current_status(STATUS_OK))
        QShortcut(QKeySequence("N"), self, activated=lambda: self.set_current_status(STATUS_NG))
        QShortcut(QKeySequence("H"), self, activated=lambda: self.set_current_status(STATUS_HOLD))
        # フィルタ・表示モード
        QShortcut(QKeySequence("["), self, activated=lambda: self.cycle_filter(-1))
        QShortcut(QKeySequence("]"), self, activated=lambda: self.cycle_filter(1))
        QShortcut(QKeySequence("M"), self, activated=self.panel.no_detection_btn.toggle)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.panel.search_edit.setFocus)
        # フォルダ・出力・保存
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_folder_dialog)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.export_dataset_dialog)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._do_autosave)
        # ズーム
        QShortcut(QKeySequence("+"), self, activated=self.canvas.zoom_in)
        QShortcut(QKeySequence("="), self, activated=self.canvas.zoom_in)
        QShortcut(QKeySequence("-"), self, activated=self.canvas.zoom_out)
        QShortcut(QKeySequence("0"), self, activated=self.canvas.fit_view)

    # ---- フォルダ読込 ----
    def open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "対象フォルダを選択")
        if path:
            self.load_folder(Path(path))

    def load_folder(self, root: Path):
        if getattr(self, "_load_thread", None) is not None:
            return
        self.open_btn.setEnabled(False)
        self._loading_dialog = QProgressDialog("フォルダを読み込み中...", None, 0, 0, self)
        self._loading_dialog.setWindowTitle("読み込み中")
        self._loading_dialog.setWindowModality(Qt.WindowModal)
        self._loading_dialog.setCancelButton(None)
        self._loading_dialog.setMinimumDuration(0)
        self._loading_dialog.show()

        self._load_thread = QThread(self)
        self._load_worker = FolderLoadWorker(root)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.succeeded.connect(self._on_folder_load_succeeded)
        self._load_worker.tooManyImages.connect(self._on_folder_load_too_many_images)
        self._load_worker.failed.connect(self._on_folder_load_failed)
        for sig in (self._load_worker.succeeded, self._load_worker.tooManyImages, self._load_worker.failed):
            sig.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._on_load_thread_finished)
        self._load_thread.start()

    def _on_load_thread_finished(self):
        self._loading_dialog.close()
        self._loading_dialog = None
        self._load_worker.deleteLater()
        self._load_thread.deleteLater()
        self._load_worker = None
        self._load_thread = None
        self.open_btn.setEnabled(True)

    def _on_folder_load_too_many_images(self):
        QMessageBox.warning(
            self, "画像数が多すぎます",
            f"対象フォルダの画像数が上限（{MAX_IMAGES}枚）を超えています。\n\n"
            f"フォルダを分割するなどして上限以下にしてから開いてください。",
        )

    def _on_folder_load_failed(self, message: str):
        print(message, file=sys.stderr)
        first_line = message.strip().splitlines()[-1] if message.strip() else message
        QMessageBox.critical(self, "エラー", f"フォルダの読込に失敗しました:\n{first_line}")

    def _on_folder_load_succeeded(self, state: AppState):
        self.state = state
        self.settings.setValue("last_root", str(state.root))
        self._update_stat_labels()
        if state.warnings:
            self.status_bar.showMessage(f"警告 {len(state.warnings)}件（詳細はログ参照）", 5000)
            for w in state.warnings[:20]:
                print(f"[警告] {w}", file=sys.stderr)
        self.panel.filter_combo.setCurrentText(state.filter_mode)
        self.panel.no_detection_btn.setChecked(state.no_detection_mode)
        idxs = state.filtered_indices()
        if idxs:
            if state.current_index not in idxs:
                state.current_index = idxs[0]
        self.show_current()

    def _update_stat_labels(self):
        s = self.state.stats
        mapping = {
            "総画像数": s.total_images,
            "txtファイル数": s.total_txt,
            "総検出数": s.total_detections,
            "Person件数": s.person_count,
            "Vehicle件数": s.vehicle_count,
            "0.3～0.5の検出数": s.review_target_count,
            "レビュー対象画像数": s.review_target_images,
        }
        for key, value in mapping.items():
            self.stat_labels[key].setText(str(value))

    # ---- 画像表示 ----
    def current_record(self) -> Optional[ImageRecord]:
        if self.state is None or not self.state.records:
            return None
        return self.state.records[self.state.current_index]

    def show_current(self):
        record = self.current_record()
        if record is None:
            self.panel.image_name_label.setText("画像: -")
            self.panel.position_label.setText("0 / 0 件表示中")
            self.panel.populate_table([])
            self.panel.select_detection_row(None)
            return
        ok = self.canvas.load_record(record)
        if not ok:
            self.status_bar.showMessage(f"画像を読み込めませんでした: {record.path}", 5000)
        self.panel.image_name_label.setText(f"画像: {record.name}  ({record.rel_key})")
        idxs = self.state.filtered_indices()
        if self.state.current_index in idxs:
            pos_text = f"{idxs.index(self.state.current_index) + 1} / {len(idxs)} 件表示中"
        else:
            pos_text = f"- / {len(idxs)} 件表示中"
        self.panel.position_label.setText(pos_text)
        self.panel.populate_table(record.detections)
        self.canvas.select_detection(record.detections[0] if record.detections else None)
        self._update_progress_label()
        self.setWindowTitle(f"AI物体検出結果 レビュー・アノテーションツール - {record.rel_key}")

    def _update_progress_label(self):
        if self.state is None:
            return
        numer, denom = self.state.progress()
        self.panel.progress_label.setText(f"{numer} / {denom}件確認済")

    # ---- ナビゲーション ----
    def _navigate(self, step: int):
        if self.state is None or not self.state.records:
            return
        idxs = self.state.filtered_indices()
        if not idxs:
            self.status_bar.showMessage("条件に一致する画像がありません", 3000)
            return
        if self.state.current_index in idxs:
            pos = idxs.index(self.state.current_index)
            new_pos = pos + step
            if 0 <= new_pos < len(idxs):
                self.state.current_index = idxs[new_pos]
                self.show_current()
            else:
                self.status_bar.showMessage("これ以上ありません", 2000)
        else:
            self.state.current_index = idxs[0] if step > 0 else idxs[-1]
            self.show_current()
        self.mark_dirty()

    def go_prev(self):
        self._navigate(-1)

    def go_next(self):
        self._navigate(1)

    def _refresh_after_filter_change(self):
        if self.state is None:
            return
        idxs = self.state.filtered_indices()
        if idxs and self.state.current_index not in idxs:
            self.state.current_index = idxs[0]
        self.show_current()

    def on_filter_changed(self, text: str):
        if self.state is None:
            return
        self.state.filter_mode = text
        self._refresh_after_filter_change()
        self.mark_dirty()

    def on_no_detection_mode_toggled(self, checked: bool):
        if self.state is None:
            return
        self.state.no_detection_mode = checked
        self._refresh_after_filter_change()
        self.mark_dirty()

    def on_search(self):
        if self.state is None:
            return
        query = self.panel.search_edit.text().strip()
        if not query:
            return
        result = self.state.find_next_matching(query, self.state.current_index)
        if result is None:
            self.status_bar.showMessage("一致する画像が見つかりません", 3000)
            return
        self.state.current_index = result
        self.show_current()

    # ---- BBox選択の同期 ----
    def on_table_selection_changed(self):
        det = self.panel.selected_detection()
        self.canvas.select_detection(det)

    def on_canvas_selection_changed(self, det: Optional[Detection]):
        self.panel.select_detection_row(det)

    # ---- ステータス設定 ----
    def set_current_status(self, status: str):
        det = self.panel.selected_detection() or (self.canvas.selected_item.detection if self.canvas.selected_item else None)
        if det is None:
            self.status_bar.showMessage("BBoxが選択されていません", 2000)
            return
        det.status = status
        record = self.current_record()
        if record is not None:
            self.panel.populate_table(record.detections)
            self.panel.select_detection_row(det)
        self._update_progress_label()
        self.mark_dirty()
        if status in (STATUS_OK, STATUS_NG):
            self._advance_after_status(det)

    def _advance_after_status(self, det: Detection):
        """OK/NG確定後、次のBBoxへ。画像内の最後のBBoxなら次の画像へ進む。"""
        record = self.current_record()
        dets = record.detections if record is not None else []
        idx = dets.index(det) if det in dets else -1
        if idx != -1 and idx + 1 < len(dets):
            self.canvas.select_detection(dets[idx + 1])
        else:
            self.go_next()

    # ---- 検出（BBox）選択のキーボード操作 ----
    def _select_detection_by_offset(self, offset: int):
        record = self.current_record()
        if record is None or not record.detections:
            return
        dets = record.detections
        current = self.panel.selected_detection()
        idx = dets.index(current) if current in dets else (-1 if offset > 0 else 0)
        new_idx = (idx + offset) % len(dets)
        self.canvas.select_detection(dets[new_idx])

    def select_next_detection(self):
        self._select_detection_by_offset(1)

    def select_prev_detection(self):
        self._select_detection_by_offset(-1)

    # ---- フィルタのキーボード操作 ----
    def cycle_filter(self, direction: int):
        current = self.panel.filter_combo.currentText()
        idx = FILTER_OPTIONS.index(current) if current in FILTER_OPTIONS else 0
        new_idx = (idx + direction) % len(FILTER_OPTIONS)
        self.panel.filter_combo.setCurrentText(FILTER_OPTIONS[new_idx])

    # ---- 自動保存 ----
    def mark_dirty(self):
        self._autosave_timer.start(1000)

    def _do_autosave(self):
        if self.state is None:
            return
        self.state.save_state()
        self.state.save_result_csv()

    def closeEvent(self, event):
        if self.state is not None:
            self._autosave_timer.stop()
            self.state.save_state()
            self.state.save_result_csv()
        super().closeEvent(event)

    # ---- 教師データ出力 ----
    def export_dataset_dialog(self):
        if self.state is None:
            QMessageBox.warning(self, "教師データ出力", "先に対象フォルダを開いてください。")
            return
        default_dir = str(self.state.root / "dataset")
        path = QFileDialog.getExistingDirectory(self, "出力先フォルダ (dataset) を選択", default_dir)
        if not path:
            return
        out_dir = Path(path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            export_warnings = []
            result = export_dataset(self.state, out_dir, export_warnings)
            summary = build_summary(self.state, result)
            save_summary(summary, out_dir)
        except OSError as e:
            QMessageBox.critical(self, "教師データ出力", f"出力中にエラーが発生しました:\n{e}")
            return
        msg = (
            f"教師データを出力しました。\n\n"
            f"出力先: {out_dir}\n"
            f"最終教師データ画像数: {result['final_image_count']}\n"
            f"最終教師データBBox数: {result['final_bbox_count']}"
        )
        if export_warnings:
            msg += f"\n\n警告 {len(export_warnings)}件（詳細はターミナルログ参照）"
            for w in export_warnings[:20]:
                print(f"[警告] {w}", file=sys.stderr)
        QMessageBox.information(self, "教師データ出力 完了", msg)


# ============================================================
# エントリポイント
# ============================================================

def install_excepthook():
    def hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(text, file=sys.stderr)
        try:
            QMessageBox.critical(
                None, "予期しないエラー",
                f"{exc_type.__name__}: {exc_value}\n\n詳細はターミナルログを確認してください。",
            )
        except Exception:
            pass
    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI物体検出結果レビューツール")
    install_excepthook()
    win = MainWindow()
    win.resize(1920, 1080)
    win.showMaximized()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
