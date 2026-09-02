from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from src.config.constants import FILTER_OPTIONS
from src.enums.review_status import ReviewStatus
from src.models.detection import Detection

TOGGLE_ON_STYLE = "QPushButton:checked { background-color: #2e8b57; color: white; font-weight: bold; }"
STATUS_BTN_STYLE = (
    "QPushButton:checked#okBtn { background-color: #2e8b57; color: white; }"
    "QPushButton:checked#ngBtn { background-color: #c0392b; color: white; }"
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
            "検出選択: W(前) / S(次)　画像切替: A(前) / D(次)　状態: O=採用 N=却下"
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

        status_box = QGroupBox("選択中BBoxの状態 (O=採用 / N=却下)")
        status_layout = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("採用 (O)")
        self.ok_btn.setObjectName("okBtn")
        self.ng_btn = QPushButton("却下 (N)")
        self.ng_btn.setObjectName("ngBtn")
        for b in (self.ok_btn, self.ng_btn):
            b.setCheckable(True)
            b.setStyleSheet(STATUS_BTN_STYLE)
            btn_row.addWidget(b)
        self.status_button_group = QButtonGroup(self)
        self.status_button_group.setExclusive(True)
        for b in (self.ok_btn, self.ng_btn):
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
        self.progress_label = QLabel("レビュー対象 0件（採用率 0% / 却下率 0%）")
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
        """選択中BBoxの状態を採用/却下ボタンに反映する。"""
        buttons = {ReviewStatus.OK: self.ok_btn, ReviewStatus.NG: self.ng_btn}
        self.status_button_group.setExclusive(False)
        for b in buttons.values():
            b.setChecked(False)
        self.status_button_group.setExclusive(True)
        enabled = det is not None
        for b in buttons.values():
            b.setEnabled(enabled)
        if det is not None and det.status in buttons:
            buttons[det.status].setChecked(True)
