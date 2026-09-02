from __future__ import annotations

from typing import Optional

import cv2

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
)

from bbox_review_tool.models.detection import Detection
from bbox_review_tool.models.image_record import ImageRecord
from bbox_review_tool.services.image_io import imread_unicode

SELECTED_COLOR = QColor(0, 255, 255)


def bbox_color(detection: Detection) -> QColor:
    if detection.color_category == "manual":
        return QColor(40, 200, 60)
    return QColor(230, 200, 0)


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
