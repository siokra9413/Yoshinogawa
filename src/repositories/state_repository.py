from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.config.constants import STATE_FILENAME
from src.models.detection import Detection

if TYPE_CHECKING:
    from src.state.app_state import AppState


class StateRepository:
    """レビュー状態(review_state.json)の読み書きを担当する。"""

    def path_for(self, root: Path) -> Path:
        return root / STATE_FILENAME

    def load(self, state: "AppState") -> None:
        # フィルタ・表示モード・カーソル位置はセッション共通設定として、
        # 保存済み状態ファイルが見つかった最初のフォルダの値を採用する。
        primary_loaded = False
        for root in state.roots:
            path = self.path_for(root)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                state.warnings.append(f"状態ファイルの読込に失敗しました: {e}")
                continue
            by_key = {r.rel_key: r for r in state.records if r.root == root}
            for img_data in data.get("images", []):
                rec = by_key.get(img_data.get("rel_key"))
                if rec is None:
                    continue
                try:
                    rec.detections = [Detection.from_dict(d) for d in img_data.get("detections", [])]
                except (KeyError, TypeError, ValueError):
                    continue
            state.deleted_count += int(data.get("deleted_count", 0))
            if not primary_loaded:
                state.filter_mode = data.get("filter_mode", state.filter_mode)
                state.no_detection_mode = bool(data.get("no_detection_mode", False))
                if state.records:
                    state.current_index = max(0, min(data.get("current_index", 0), len(state.records) - 1))
                primary_loaded = True

    def save(self, state: "AppState") -> None:
        for root in state.roots:
            recs = [r for r in state.records if r.root == root]
            data = {
                "root": str(root),
                "current_index": state.current_index,
                "filter_mode": state.filter_mode,
                "no_detection_mode": state.no_detection_mode,
                "deleted_count": state.deleted_count,
                "images": [
                    {"rel_key": r.rel_key, "detections": [d.to_dict() for d in r.detections]}
                    for r in recs
                ],
            }
            path = self.path_for(root)
            tmp_path = path.with_suffix(".json.tmp")
            try:
                tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp_path.replace(path)
            except OSError as e:
                state.warnings.append(f"状態の自動保存に失敗しました ({root}): {e}")
