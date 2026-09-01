#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bbox_review_tool.py のローカル動作確認用テストデータ生成スクリプト。

./test_data/images/3001/ 以下に合成jpg画像とAI検出結果txtを生成する。
- 信頼値 0.3未満 / 0.3〜0.5 / 0.5以上 を混在させる
- Person(0) / Vehicle(2) を混在させる
- 再帰探索確認用のサブフォルダ (3001/sub/) を用意
- 一部の画像は txt を生成しない（検出なし画像レビューモードの確認用）
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

OUT_ROOT = Path(__file__).parent / "test_data" / "images" / "3001"
SUB_DIR = OUT_ROOT / "sub"
IMG_W, IMG_H = 640, 480
N_MAIN_IMAGES = 24
N_SUB_IMAGES = 6
N_NO_TXT_IMAGES = 6  # 検出なし画像レビュー確認用（txtを作らない）

RNG = random.Random(42)


def make_image(rng: random.Random):
    img = np.full((IMG_H, IMG_W, 3), (60, 60, 60), dtype=np.uint8)
    cv2.rectangle(img, (0, IMG_H - 60), (IMG_W, IMG_H), (30, 60, 30), -1)
    boxes = []
    n_boxes = rng.randint(1, 4)
    for _ in range(n_boxes):
        cls = rng.choice([0, 2])
        if cls == 0:
            w, h = rng.randint(20, 45), rng.randint(45, 90)
            color = (60, 180, 240)
        else:
            w, h = rng.randint(60, 120), rng.randint(35, 65)
            color = (240, 160, 60)
        x1 = rng.randint(0, max(IMG_W - w, 1))
        y1 = rng.randint(0, max(IMG_H - h, 1))
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        conf_bucket = rng.random()
        if conf_bucket < 0.25:
            conf = rng.uniform(0.05, 0.29)   # 読込対象外レンジ
        elif conf_bucket < 0.65:
            conf = rng.uniform(0.30, 0.499)  # レビュー対象レンジ
        else:
            conf = rng.uniform(0.50, 0.97)   # 高信頼度レンジ
        boxes.append((conf, cls, [float(x1), float(y1), float(x2), float(y2)]))
    return img, boxes


def write_pair(rng: random.Random, out_dir: Path, name: str, with_txt: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    img, boxes = make_image(rng)
    jpg_path = out_dir / f"{name}.jpg"
    ok, buf = cv2.imencode(".jpg", img)
    buf.tofile(str(jpg_path))
    if with_txt:
        txt_path = out_dir / f"{name}.txt"
        lines = [f"{conf:.10f}    {cls}    [{b[0]}, {b[1]}, {b[2]}, {b[3]}]" for conf, cls, b in boxes]
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print(f"生成先: {OUT_ROOT.parent.parent}")
    for i in range(N_MAIN_IMAGES):
        name = f"81811311_{200 + i}"
        write_pair(RNG, OUT_ROOT, name, with_txt=True)
    for i in range(N_SUB_IMAGES):
        name = f"92200211_{100 + i}"
        write_pair(RNG, SUB_DIR, name, with_txt=True)
    for i in range(N_NO_TXT_IMAGES):
        name = f"70000000_{i}"
        write_pair(RNG, OUT_ROOT, name, with_txt=False)
    total = N_MAIN_IMAGES + N_SUB_IMAGES + N_NO_TXT_IMAGES
    print(f"画像 {total}枚を生成しました（txtなし: {N_NO_TXT_IMAGES}枚）。")
    print(f"bbox_review_tool.py で次のフォルダを開いてください:\n  {OUT_ROOT}")


if __name__ == "__main__":
    main()
