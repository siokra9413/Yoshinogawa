from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path):
    """Windowsの非ASCIIパスでも安全に画像を読み込む。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
