"""
auto_calibrate_color.py
------------------------
Analisis sample warna dari `color_samples/` (hasil generate_color_samples.py)
dan otomatis update COLOR_RANGES di color_utils.py.

Cara pakai:
    python auto_calibrate_color.py
    python auto_calibrate_color.py --preview    # Lihat rentang baru tanpa apply
    python auto_calibrate_color.py --backup     # Backup color_utils.py sebelum update

Cara Kerja:
1. Baca semua file .npy di folder color_samples/
2. Untuk tiap warna, hitung distribusi H/S/V dari semua piksel sample
3. Hitung rentang [mean - N*std, mean + N*std] untuk mendapat batas bawah & atas
4. Khusus Merah: pecah jadi 2 rentang karena berada di ujung lingkaran Hue (0 & 180)
5. Update COLOR_RANGES di color_utils.py dengan rentang baru
"""

import argparse
import ast
import os
import re
import shutil
from datetime import datetime

import numpy as np

SAMPLE_DIR = "color_samples"
COLOR_UTILS_PATH = "color_utils.py"

# Berapa std dev yang dipakai untuk menentukan batas rentang.
# Lebih besar = lebih longgar (lebih banyak warna terdeteksi, tapi lebih banyak noise)
# Lebih kecil = lebih ketat (lebih presisi, tapi bisa miss kalau cahaya berubah)
STD_MULTIPLIER = 2.5

# Minimum piksel yang diperlukan agar kalibrasi valid
MIN_PIXELS_REQUIRED = 2000


def load_all_samples(color_name: str) -> np.ndarray | None:
    """Muat semua batch sample untuk satu warna, gabungkan jadi satu array."""
    path = os.path.join(SAMPLE_DIR, f"{color_name}.npy")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True).tolist()
    if not isinstance(data, list) or len(data) == 0:
        return None
    all_pixels = np.vstack(data)  # shape: (N_total, 3)
    return all_pixels.astype(np.float32)


def compute_hsv_range(pixels: np.ndarray, color_name: str):
    """
    Hitung rentang HSV dari array piksel.
    
    Returns
    -------
    list of tuple ((lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v))
    Bisa 1 atau 2 pasang (merah punya 2 pasang karena wrap-around hue).
    """
    h_vals = pixels[:, 0].astype(np.float32)
    s_vals = pixels[:, 1].astype(np.float32)
    v_vals = pixels[:, 2].astype(np.float32)

    s_mean, s_std = s_vals.mean(), s_vals.std()
    v_mean, v_std = v_vals.mean(), v_vals.std()

    s_lo = int(np.clip(s_mean - STD_MULTIPLIER * s_std, 0, 255))
    s_hi = int(np.clip(s_mean + STD_MULTIPLIER * s_std, 0, 255))
    v_lo = int(np.clip(v_mean - STD_MULTIPLIER * v_std, 0, 255))
    v_hi = int(np.clip(v_mean + STD_MULTIPLIER * v_std, 0, 255))

    # Merah: hue ada di sekitar 0-10 DAN 170-180 (ujung lingkaran)
    # Deteksi dengan melihat apakah banyak piksel di ujung-ujung tersebut
    if color_name == "merah":
        mask_low = h_vals <= 15
        mask_high = h_vals >= 165

        if mask_low.sum() > 0 and mask_high.sum() > 0:
            # Dua rentang: hue rendah & hue tinggi
            h_lo1 = int(np.clip(h_vals[mask_low].mean() - STD_MULTIPLIER * h_vals[mask_low].std(), 0, 15))
            h_hi1 = int(np.clip(h_vals[mask_low].mean() + STD_MULTIPLIER * h_vals[mask_low].std(), 0, 20))
            h_lo2 = int(np.clip(h_vals[mask_high].mean() - STD_MULTIPLIER * h_vals[mask_high].std(), 160, 180))
            h_hi2 = int(np.clip(h_vals[mask_high].mean() + STD_MULTIPLIER * h_vals[mask_high].std(), 165, 180))
            return [
                ((h_lo1, s_lo, v_lo), (h_hi1, s_hi, v_hi)),
                ((h_lo2, s_lo, v_lo), (h_hi2, s_hi, v_hi)),
            ]
        elif mask_low.sum() > mask_high.sum():
            h_mean = h_vals[mask_low].mean() if mask_low.sum() > 0 else h_vals.mean()
            h_std = h_vals[mask_low].std() if mask_low.sum() > 0 else h_vals.std()
            h_lo = int(np.clip(h_mean - STD_MULTIPLIER * h_std, 0, 180))
            h_hi = int(np.clip(h_mean + STD_MULTIPLIER * h_std, 0, 20))
            return [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]
        else:
            h_mean = h_vals[mask_high].mean() if mask_high.sum() > 0 else h_vals.mean()
            h_std = h_vals[mask_high].std() if mask_high.sum() > 0 else h_vals.std()
            h_lo = int(np.clip(h_mean - STD_MULTIPLIER * h_std, 160, 180))
            h_hi = int(np.clip(h_mean + STD_MULTIPLIER * h_std, 165, 180))
            return [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]

    # Warna lain: satu rentang
    # Khusus Putih & Hitam: saturation dan value yang lebih penting dari hue
    if color_name == "putih":
        return [((0, 0, int(np.clip(v_mean - STD_MULTIPLIER * v_std, 150, 255))),
                 (180, int(np.clip(s_mean + STD_MULTIPLIER * s_std, 0, 60)), 255))]

    if color_name == "hitam":
        return [((0, 0, 0),
                 (180, 255, int(np.clip(v_mean + STD_MULTIPLIER * v_std, 0, 60))))]

    if color_name == "abu-abu":
        return [((0, 0, int(np.clip(v_mean - STD_MULTIPLIER * v_std, 30, 200))),
                 (180, int(np.clip(s_mean + STD_MULTIPLIER * s_std, 0, 50)),
                  int(np.clip(v_mean + STD_MULTIPLIER * v_std, 40, 210))))]

    h_mean, h_std = h_vals.mean(), h_vals.std()
    h_lo = int(np.clip(h_mean - STD_MULTIPLIER * h_std, 0, 179))
    h_hi = int(np.clip(h_mean + STD_MULTIPLIER * h_std, 0, 179))
    if h_lo >= h_hi:
        h_lo = max(0, h_lo - 5)
        h_hi = min(179, h_hi + 5)

    return [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]


def format_color_ranges_code(new_ranges: dict) -> str:
    """Format dict COLOR_RANGES menjadi string kode Python yang rapi."""
    lines = ["COLOR_RANGES = {\n"]
    for color, ranges in new_ranges.items():
        range_str = ", ".join(
            f"({tuple(lo)}, {tuple(hi)})" for lo, hi in ranges
        )
        lines.append(f'    "{color}":  [{range_str}],\n')
    lines.append("}\n")
    return "".join(lines)


def update_color_utils(new_ranges: dict, backup: bool = True):
    """
    Update blok COLOR_RANGES di color_utils.py dengan rentang baru.
    """
    with open(COLOR_UTILS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Backup dulu jika diminta
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{COLOR_UTILS_PATH}.backup_{ts}"
        shutil.copy2(COLOR_UTILS_PATH, backup_path)
        print(f"[BACKUP] Disimpan ke: {backup_path}")

    # Cari dan ganti blok COLOR_RANGES menggunakan regex
    pattern = r"COLOR_RANGES\s*=\s*\{.*?\}"
    new_code = format_color_ranges_code(new_ranges)
    new_content, count = re.subn(pattern, new_code.rstrip(), content, flags=re.DOTALL)

    if count == 0:
        print("[ERROR] Tidak menemukan blok COLOR_RANGES di color_utils.py!")
        print("        Pastikan format-nya sesuai dan coba lagi.")
        return False

    with open(COLOR_UTILS_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[OK] color_utils.py berhasil diupdate dengan {len(new_ranges)} warna terkalibrasi.")
    return True


def run(preview_only: bool = False, backup: bool = True):
    all_colors = [
        "merah", "oranye", "coklat", "kuning", "hijau", "cyan",
        "biru", "ungu", "pink", "putih", "hitam", "abu-abu"
    ]

    print("\n" + "=" * 60)
    print("  AUTO CALIBRATE COLOR — Kalibrasi Otomatis HSV Ranges")
    print("=" * 60)

    new_ranges = {}
    skipped = []

    for color_name in all_colors:
        pixels = load_all_samples(color_name)

        if pixels is None:
            print(f"  [SKIP] '{color_name}': tidak ada sample.")
            skipped.append(color_name)
            continue

        n_px = len(pixels)
        if n_px < MIN_PIXELS_REQUIRED:
            print(f"  [WARN] '{color_name}': hanya {n_px} piksel (min: {MIN_PIXELS_REQUIRED}). "
                  f"Hasil mungkin kurang akurat.")

        ranges = compute_hsv_range(pixels, color_name)
        new_ranges[color_name] = ranges

        print(f"  [OK]   '{color_name}': {n_px} piksel → ", end="")
        for lo, hi in ranges:
            print(f"H{lo[0]}-{hi[0]} S{lo[1]}-{hi[1]} V{lo[2]}-{hi[2]}  ", end="")
        print()

    if not new_ranges:
        print("\n[ERROR] Tidak ada sample yang bisa diproses!")
        print("         Jalankan dulu: python generate_color_samples.py")
        return

    # Untuk warna yang tidak ada sample-nya, gunakan range default
    DEFAULT_RANGES = {
        "merah":   [((0, 110, 100), (10, 255, 255)), ((170, 110, 100), (180, 255, 255))],
        "oranye":  [((11, 110, 100), (25, 255, 255))],
        "coklat":  [((0, 40, 30), (22, 150, 140)), ((170, 40, 30), (180, 150, 140))],
        "kuning":  [((26, 70, 50), (34, 255, 255))],
        "hijau":   [((35, 50, 50), (85, 255, 255))],
        "cyan":    [((86, 50, 50), (95, 255, 255))],
        "biru":    [((96, 50, 50), (130, 255, 255))],
        "ungu":    [((131, 50, 50), (155, 255, 255))],
        "pink":    [((145, 50, 80), (169, 255, 255)), ((170, 30, 120), (180, 120, 255)), ((0, 30, 120), (10, 120, 255))],
        "putih":   [((0, 0, 180), (180, 40, 255))],
        "hitam":   [((0, 0, 0), (180, 255, 40))],
        "abu-abu": [((0, 0, 41), (180, 40, 179))],
    }

    for color_name in skipped:
        new_ranges[color_name] = DEFAULT_RANGES.get(color_name, [((0, 0, 0), (180, 255, 255))])
        print(f"  [DEFAULT] '{color_name}': memakai rentang default bawaan.")

    # Urutkan sesuai urutan asli
    ordered_ranges = {c: new_ranges[c] for c in all_colors if c in new_ranges}

    print("\n" + "─" * 60)
    print("  COLOR_RANGES BARU (preview):")
    print("─" * 60)
    print(format_color_ranges_code(ordered_ranges))

    if preview_only:
        print("[PREVIEW ONLY] Tidak ada perubahan yang diterapkan ke color_utils.py.")
        print("               Jalankan tanpa --preview untuk apply perubahan.")
        return

    print("─" * 60)
    confirm = input("  Terapkan rentang baru ke color_utils.py? [y/N]: ").strip().lower()
    if confirm == "y":
        success = update_color_utils(ordered_ranges, backup=backup)
        if success:
            print("\n  ✓ SELESAI! Coba jalankan detect_webcam.py untuk cek akurasi warna.")
    else:
        print("  Dibatalkan. Tidak ada perubahan.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kalibrasi otomatis COLOR_RANGES di color_utils.py dari sample yang dikumpulkan"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Hanya tampilkan rentang baru tanpa mengubah color_utils.py"
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Jangan buat backup color_utils.py sebelum diupdate"
    )
    args = parser.parse_args()
    run(preview_only=args.preview, backup=not args.no_backup)
