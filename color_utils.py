"""
color_utils.py
----------------
Modul untuk menentukan warna dominan dari sebuah crop gambar (hasil bounding box YOLO).

Algoritma:
1. Ambil area bounding box (crop) dari frame/gambar asli.
2. Konversi crop dari BGR (format OpenCV) ke HSV, karena HSV lebih stabil
   terhadap perubahan pencahayaan dibanding RGB/BGR.
3. Hitung rata-rata (atau nilai piksel terbanyak) dari Hue, Saturation, Value
   pada area crop tersebut (dengan sedikit "cropping tengah" agar tidak
   kebawa warna background/pinggiran objek).
4. Cocokkan nilai HSV tersebut ke salah satu rentang warna yang sudah
   didefinisikan (merah, hijau, biru, kuning, dst).
5. Kembalikan nama warna sebagai string.

Kamu bebas menambah/mengubah rentang HSV di COLOR_RANGES sesuai kebutuhan
(misal warna khusus barang yang mau kamu deteksi).
"""

import cv2
import numpy as np

# Rentang HSV untuk tiap warna.
# Format OpenCV: H (0-179), S (0-255), V (0-255)
# Catatan: warna "merah" dipecah jadi 2 rentang karena hue merah ada di
# ujung awal (0) dan ujung akhir (180) dari lingkaran warna HSV.
COLOR_RANGES = {
    "merah":  [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "oranye": [((11, 70, 50), (25, 255, 255))],
    "kuning": [((26, 70, 50), (34, 255, 255))],
    "hijau":  [((35, 50, 50), (85, 255, 255))],
    "cyan":   [((86, 50, 50), (95, 255, 255))],
    "biru":   [((96, 50, 50), (130, 255, 255))],
    "ungu":   [((131, 50, 50), (155, 255, 255))],
    "pink":   [((156, 50, 50), (169, 255, 255))],
    "putih":  [((0, 0, 200), (180, 40, 255))],
    "hitam":  [((0, 0, 0), (180, 255, 40))],
    "abu-abu": [((0, 0, 41), (180, 40, 199))],
}


def _crop_center(image, ratio=0.6):
    """Ambil bagian tengah crop (biar tidak kebawa background/tepi objek)."""
    h, w = image.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return image[y0:y0 + ch, x0:x0 + cw]


def detect_dominant_color(crop_bgr):
    """
    Menentukan nama warna dominan dari sebuah crop BGR (hasil bounding box).

    Parameters
    ----------
    crop_bgr : np.ndarray
        Potongan gambar (H x W x 3) format BGR dari OpenCV.

    Returns
    -------
    str
        Nama warna dominan, misal "merah", "biru", dll. Mengembalikan
        "tidak diketahui" jika crop kosong/tidak valid.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "tidak diketahui"

    # Ambil bagian tengah saja supaya lebih representatif untuk objeknya
    region = _crop_center(crop_bgr)
    if region.size == 0:
        region = crop_bgr

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    best_color = "tidak diketahui"
    best_count = 0

    for color_name, ranges in COLOR_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            mask_total = cv2.bitwise_or(mask_total, mask)

        count = int(cv2.countNonZero(mask_total))
        if count > best_count:
            best_count = count
            best_color = color_name

    # Kalau tidak ada piksel yang cocok sama sekali ke salah satu rentang
    total_pixels = hsv.shape[0] * hsv.shape[1]
    if total_pixels == 0 or best_count / total_pixels < 0.05:
        return "tidak diketahui"

    return best_color
