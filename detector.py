# detector.py
import cv2
import numpy as np
from typing import Tuple, Dict

def _ensure_bgr(img):
    if img is None:
        raise ValueError("Failed to decode image.")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def detect_whitehair_bytes(
    image_bytes: bytes,
    thresh_l: int = 200,
    min_line_len_px: int = 10,
    morph_open: int = 1,
) -> Dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return detect_whitehair_img(img, thresh_l, min_line_len_px, morph_open)

def detect_whitehair_img(
    img_bgr,
    thresh_l: int = 200,
    min_line_len_px: int = 10,
    morph_open: int = 1,
) -> Dict:
    img_bgr = _ensure_bgr(img_bgr)

    # LAB空間で明度(L)の高い細線を狙う
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 白っぽさ：高L & 低彩度（A,Bが中庸に近い）をざっくりマスク
    _, mL = cv2.threshold(L, thresh_l, 255, cv2.THRESH_BINARY)
    A_center = cv2.inRange(A, 115, 145)
    B_center = cv2.inRange(B, 115, 145)
    whiteish = cv2.bitwise_and(mL, cv2.bitwise_and(A_center, B_center))

    # 細い線を強調（Sobel→Canny）
    edges = cv2.Canny(L, 60, 150)
    candidate = cv2.bitwise_and(whiteish, edges)

    # ノイズ除去（モルフォロジ）
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open, morph_open))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k, iterations=1)

    # 小領域除去
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    mask = np.zeros_like(candidate)
    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max(1, min_line_len_px):
            mask[labels == i] = 255
            kept += area

    ratio = float(kept) / float(candidate.size)

    # 可視化（赤で上書き）
    vis = img_bgr.copy()
    vis[mask == 255] = (0, 0, 255)

    return {
        "whitehair_ratio": round(ratio, 4),
        "whitehair_pixels": int(kept),
        "mask": mask,          # 0/255
        "visual": vis,         # BGR
    }
