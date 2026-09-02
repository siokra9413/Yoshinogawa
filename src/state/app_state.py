from __future__ import annotations

from bbox_review_tool.config.constants import REVIEW_MAX, REVIEW_MIN
from bbox_review_tool.enums.detection_class import DetectionClass
from bbox_review_tool.enums.detection_source import DetectionSource
from bbox_review_tool.enums.review_status import ReviewStatus
from bbox_review_tool.models.image_record import ImageRecord
from bbox_review_tool.models.scan_stats import ScanStats
from bbox_review_tool.repositories.csv_repository import CsvRepository
from bbox_review_tool.repositories.state_repository import StateRepository
from bbox_review_tool.services.scan_service import FolderScanner


class AppState:
    """フォルダ走査結果・フィルタ状態・ナビゲーション位置を保持する。
    永続化(状態ファイル/CSV)や走査処理そのものは各サービス/リポジトリへ委譲する。"""

    def __init__(self, roots: list):
        self.roots = roots
        self.records: list = []
        self.stats = ScanStats()
        self.warnings: list = []
        self.current_index: int = 0
        self.filter_mode: str = "全件"
        self.no_detection_mode: bool = False
        self.deleted_count: int = 0
        self._scanner = FolderScanner()
        self._state_repo = StateRepository()
        self._csv_repo = CsvRepository()

    def load_or_scan(self):
        self.records, self.stats = self._scanner.scan(self.roots, self.warnings)
        self._state_repo.load(self)

    def save_state(self):
        self._state_repo.save(self)

    def save_result_csv(self):
        self._csv_repo.save(self)

    def _matches_filter(self, r: ImageRecord) -> bool:
        f = self.filter_mode
        if f == "全件":
            return True
        if f == "Personのみ":
            return any(d.class_id == DetectionClass.PERSON for d in r.detections)
        if f == "Vehicleのみ":
            return any(d.class_id == DetectionClass.VEHICLE for d in r.detections)
        if f == "採用のみ":
            return any(d.status == ReviewStatus.OK for d in r.detections)
        if f == "却下のみ":
            return any(d.status == ReviewStatus.NG for d in r.detections)
        if f == "未確認のみ":
            return any(d.status == ReviewStatus.UNCONFIRMED for d in r.detections)
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
                if d.source == DetectionSource.AI and d.confidence is not None and REVIEW_MIN <= d.confidence < REVIEW_MAX:
                    denom += 1
                    if d.status != ReviewStatus.UNCONFIRMED:
                        numer += 1
        return numer, denom

    def review_rates(self):
        """レビュー対象数・採用率・却下率を返す。"""
        target = adopted = rejected = 0
        for r in self.records:
            for d in r.detections:
                if d.source == DetectionSource.AI and d.confidence is not None and REVIEW_MIN <= d.confidence < REVIEW_MAX:
                    target += 1
                    if d.status == ReviewStatus.OK:
                        adopted += 1
                    elif d.status == ReviewStatus.NG:
                        rejected += 1
        adopt_rate = (adopted / target * 100) if target else 0.0
        reject_rate = (rejected / target * 100) if target else 0.0
        return {"target": target, "adopted": adopted, "rejected": rejected,
                "adopt_rate": adopt_rate, "reject_rate": reject_rate}
