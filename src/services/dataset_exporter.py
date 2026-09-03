from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from src.enums.detection_class import DetectionClass
from src.enums.review_status import ReviewStatus
from src.models.image_record import ImageRecord
from src.services.image_io import imread_unicode

if TYPE_CHECKING:
    from src.state.app_state import AppState


class DatasetExporter:
    """レビューで採用したBBox（および既定採用の高信頼度BBox）をYOLO形式（フラット構成）で書き出す。"""

    def export(self, state: "AppState", out_dir: Path, warnings: list) -> dict:
        multi_root = len(state.roots) > 1
        root_tags = {root: (f"{i:02d}_{root.name}" if multi_root else "") for i, root in enumerate(state.roots)}

        def dest_name(r: ImageRecord) -> str:
            tag = root_tags[r.root]
            return f"{tag}_{r.path.name}" if tag else r.path.name

        entries = []
        for r in state.records:
            included = [d for d in r.detections if d.status == ReviewStatus.OK]
            if included or r.high_conf_detections:
                entries.append((r, included))

        prepared = []
        used_names = set()
        for r, included in entries:
            name = dest_name(r)
            if name in used_names:
                warnings.append(f"{r.path} はファイル名衝突のため出力から除外しました: {name}")
                continue
            used_names.add(name)
            prepared.append((r, included, name))

        shuffled = [name for _, _, name in prepared]
        random.Random(42).shuffle(shuffled)
        split_point = int(len(shuffled) * 0.8)
        train_keys = set(shuffled[:split_point])

        images_dir = out_dir / "images"
        labels_dir = out_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        final_bbox_count = 0
        final_image_count = 0
        person_count = 0
        vehicle_count = 0
        train_files = []
        val_files = []

        for r, included, name in prepared:
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
            written_classes = []
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
                written_classes.append(class_id)

            if not lines:
                continue

            dest_img_path = images_dir / name
            dest_lbl_path = (labels_dir / name).with_suffix(".txt")

            try:
                shutil.copy2(r.path, dest_img_path)
            except OSError as e:
                warnings.append(f"{r.path} のコピーに失敗しました: {e}")
                continue

            dest_lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            final_bbox_count += len(lines)
            final_image_count += 1
            person_count += written_classes.count(DetectionClass.PERSON)
            vehicle_count += written_classes.count(DetectionClass.VEHICLE)

            rel_img = f"images/{name}"
            if name in train_keys:
                train_files.append(rel_img)
            else:
                val_files.append(rel_img)

        yaml_lines = ["path: .", "", "train:"]
        yaml_lines += [f"  - {p}" for p in train_files]
        yaml_lines += ["", "val:"]
        yaml_lines += [f"  - {p}" for p in val_files]
        yaml_lines += ["", "names:", "  0: Person", "  2: Vehicle"]
        (out_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        return {
            "final_image_count": final_image_count,
            "final_bbox_count": final_bbox_count,
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "train_image_count": len(train_files),
            "val_image_count": len(val_files),
        }
