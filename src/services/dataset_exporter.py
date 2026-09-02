from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from src.enums.review_status import ReviewStatus
from src.models.image_record import ImageRecord
from src.services.image_io import imread_unicode

if TYPE_CHECKING:
    from src.state.app_state import AppState


class DatasetExporter:
    """レビューで採用したBBox（および既定採用の高信頼度BBox）をYOLO形式で書き出す。"""

    def export(self, state: "AppState", out_dir: Path, warnings: list) -> dict:
        # 複数フォルダを開いている場合、同名ファイルの衝突を避けるため出力先を
        # フォルダごとのサブディレクトリに分ける。単一フォルダの場合は従来通りの構成のまま。
        multi_root = len(state.roots) > 1
        root_tags = {root: (f"{i:02d}_{root.name}" if multi_root else "") for i, root in enumerate(state.roots)}

        def dataset_rel(r: ImageRecord) -> Path:
            tag = root_tags[r.root]
            return (Path(tag) / r.rel_key) if tag else Path(r.rel_key)

        entries = []
        for r in state.records:
            included = [d for d in r.detections if d.status == ReviewStatus.OK]
            if included:
                entries.append((r, included))

        keys = [str(dataset_rel(r)) for r, _ in entries]
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
            rel = dataset_rel(r)
            is_train = str(rel) in train_keys
            img_dest_dir = images_train if is_train else images_val
            lbl_dest_dir = labels_train if is_train else labels_val
            dest_img_path = img_dest_dir / rel
            dest_lbl_path = (lbl_dest_dir / rel).with_suffix(".txt")
            dest_img_path.parent.mkdir(parents=True, exist_ok=True)
            dest_lbl_path.parent.mkdir(parents=True, exist_ok=True)

            img = imread_unicode(r.path)
            if img is None:
                warnings.append(f"{r.path} を読み込めず出力から除外しました")
                continue
            h, w = img.shape[:2]

            # レビューで採用したBBoxに加え、0.5以上（レビュー対象外＝高信頼度）のBBoxも
            # そのまま含める。レビュー対象外のBBoxをラベルから欠落させると、実際には
            # 物体が写っている領域が学習上「背景」として扱われ、精度低下につながるため。
            boxes_to_write = [(d.class_id, d.bbox) for d in included]
            boxes_to_write += [(d.class_id, d.bbox) for d in r.high_conf_detections]

            lines = []
            for class_id, bbox in boxes_to_write:
                xmin, ymin, xmax, ymax = bbox
                xmin_c, xmax_c = sorted((max(0.0, min(xmin, w)), max(0.0, min(xmax, w))))
                ymin_c, ymax_c = sorted((max(0.0, min(ymin, h)), max(0.0, min(ymax, h))))
                bw = xmax_c - xmin_c
                bh = ymax_c - ymin_c
                if bw <= 0 or bh <= 0:
                    continue
                cx = (xmin_c + xmax_c) / 2.0 / w
                cy = (ymin_c + ymax_c) / 2.0 / h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")

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
