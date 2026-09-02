from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListView,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSplitter,
    QStatusBar, QTreeView, QVBoxLayout, QWidget,
)

from src.config.constants import FILTER_OPTIONS, MAX_IMAGES
from src.enums.review_status import ReviewStatus
from src.models.detection import Detection
from src.models.image_record import ImageRecord
from src.services.dataset_exporter import DatasetExporter
from src.services.scan_service import TooManyImagesError
from src.services.summary_service import ExportSummaryBuilder, save_summary
from src.state.app_state import AppState
from src.ui.image_canvas import ImageCanvas
from src.ui.review_panel import ReviewPanel


class FolderLoadWorker(QObject):
    succeeded = Signal(object)
    tooManyImages = Signal()
    failed = Signal(str)

    def __init__(self, roots: list):
        super().__init__()
        self.roots = roots

    def run(self):
        try:
            state = AppState(self.roots)
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


def select_multiple_folders(parent) -> list:
    """複数フォルダを選択できるダイアログ（Qtの標準ダイアログは単一選択のみのため、
    非ネイティブダイアログの内部ビューを拡張選択モードにして実現する）。"""
    dialog = QFileDialog(parent, "対象フォルダを選択（Ctrl/Shiftクリックで複数選択可）")
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    for view in list(dialog.findChildren(QListView)) + list(dialog.findChildren(QTreeView)):
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    if dialog.exec() != QFileDialog.Accepted:
        return []
    return [Path(p) for p in dialog.selectedFiles()]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI物体検出結果 レビュー・アノテーションツール")
        self.settings = QSettings("BBoxReviewTool", "BBoxReviewTool")
        self.state: Optional[AppState] = None
        self._dataset_exporter = DatasetExporter()
        self._summary_builder = ExportSummaryBuilder()

        self._build_ui()
        self._build_shortcuts()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)

        try:
            last_roots = [Path(p) for p in json.loads(self.settings.value("last_roots", "[]"))]
        except (json.JSONDecodeError, TypeError):
            last_roots = []
        last_roots = [p for p in last_roots if p.exists()]
        if last_roots:
            self.load_folder(last_roots)

    # ---- UI構築 ----
    def _build_ui(self):
        central = QWidget()
        root_layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.open_btn = QPushButton("対象フォルダを開く（複数選択可）... (Ctrl+O)")
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

        self.panel.ok_btn.clicked.connect(lambda: self.set_current_status(ReviewStatus.OK))
        self.panel.ng_btn.clicked.connect(lambda: self.set_current_status(ReviewStatus.NG))
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
        QShortcut(QKeySequence("O"), self, activated=lambda: self.set_current_status(ReviewStatus.OK))
        QShortcut(QKeySequence("N"), self, activated=lambda: self.set_current_status(ReviewStatus.NG))
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
        roots = select_multiple_folders(self)
        if roots:
            self.load_folder(roots)

    def load_folder(self, roots: list):
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
        self._load_worker = FolderLoadWorker(roots)
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
        self.settings.setValue("last_roots", json.dumps([str(r) for r in state.roots]))
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
        multi_root = len(self.state.roots) > 1
        display_key = f"{record.root.name}/{record.rel_key}" if multi_root else record.rel_key
        self.panel.image_name_label.setText(f"画像: {record.name}  ({display_key})")
        idxs = self.state.filtered_indices()
        if self.state.current_index in idxs:
            pos_text = f"{idxs.index(self.state.current_index) + 1} / {len(idxs)} 件表示中"
        else:
            pos_text = f"- / {len(idxs)} 件表示中"
        self.panel.position_label.setText(pos_text)
        self.panel.populate_table(record.detections)
        self.canvas.select_detection(record.detections[0] if record.detections else None)
        self._update_progress_label()
        self.setWindowTitle(f"AI物体検出結果 レビュー・アノテーションツール - {display_key}")

    def _update_progress_label(self):
        if self.state is None:
            return
        numer, denom = self.state.progress()
        rates = self.state.review_rates()
        self.panel.progress_label.setText(
            f"{numer} / {denom}件確認済　"
            f"レビュー対象 {rates['target']}件（採用率 {rates['adopt_rate']:.0f}% / 却下率 {rates['reject_rate']:.0f}%）"
        )

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
    def set_current_status(self, status: ReviewStatus):
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
        if status in (ReviewStatus.OK, ReviewStatus.NG):
            self._advance_after_status(det)

    def _advance_after_status(self, det: Detection):
        """採用/却下確定後、次のBBoxへ。画像内の最後のBBoxなら次の画像へ進む。"""
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
        default_dir = str(self.state.roots[0] / "dataset")
        path = QFileDialog.getExistingDirectory(self, "出力先フォルダ (dataset) を選択", default_dir)
        if not path:
            return
        out_dir = Path(path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            export_warnings = []
            result = self._dataset_exporter.export(self.state, out_dir, export_warnings)
            summary = self._summary_builder.build(self.state, result)
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
