"""
generate_color_samples.py
--------------------------
Tool interaktif untuk mengambil SAMPLE WARNA dari webcam.

Cara pakai:
    python generate_color_samples.py

Alur:
1. Arahkan objek berwarna ke tengah kamera (area di dalam kotak kuning).
2. Tekan tombol warna yang sesuai (lihat panduan di layar) untuk SIMPAN sample.
3. Tekan 'q' untuk selesai.

Semua sample tersimpan di folder `color_samples/` sebagai file .npy
Setelah selesai, jalankan auto_calibrate_color.py untuk update COLOR_RANGES.

Kontrol keyboard:
    r = Merah         o = Oranye        k = Kuning
    h = Hijau         c = Cyan          b = Biru
    u = Ungu          p = Pink          w = Putih
    t = Hitam (blacK) a = Abu-abu
    q = Keluar / Selesai
    d = Hapus sample terakhir
    s = Lihat statistik sample saat ini
"""

import argparse
import os
import time

import cv2
import numpy as np

# =============================================================================
# Konfigurasi
# =============================================================================
SAMPLE_DIR = "color_samples"
SAMPLE_BOX_RATIO = 0.25  # Ukuran kotak sampling relatif terhadap frame (25%)

# Mapping tombol keyboard → nama warna
KEY_COLOR_MAP = {
    ord("r"): "merah",
    ord("o"): "oranye",
    ord("k"): "kuning",
    ord("h"): "hijau",
    ord("c"): "cyan",
    ord("b"): "biru",
    ord("u"): "ungu",
    ord("p"): "pink",
    ord("w"): "putih",
    ord("t"): "hitam",
    ord("a"): "abu-abu",
}

# Warna tampilan BGR untuk tiap label (untuk display)
DISPLAY_COLORS = {
    "merah":   (0, 0, 220),
    "oranye":  (0, 128, 255),
    "kuning":  (0, 215, 255),
    "hijau":   (0, 200, 0),
    "cyan":    (255, 200, 0),
    "biru":    (255, 50, 50),
    "ungu":    (200, 0, 200),
    "pink":    (180, 60, 180),
    "putih":   (220, 220, 220),
    "hitam":   (60, 60, 60),
    "abu-abu": (150, 150, 150),
}


# =============================================================================
# Fungsi Bantu
# =============================================================================
def get_sample_path(color_name: str) -> str:
    return os.path.join(SAMPLE_DIR, f"{color_name}.npy")


def load_samples(color_name: str) -> list:
    """Muat sample yang sudah ada dari file .npy."""
    path = get_sample_path(color_name)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True).tolist()
        return data if isinstance(data, list) else [data]
    return []


def save_samples(color_name: str, samples: list):
    """Simpan daftar sample HSV ke file .npy."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    np.save(get_sample_path(color_name), np.array(samples, dtype=object), allow_pickle=True)


def get_sample_region(frame):
    """Kembalikan koordinat dan crop area kotak sampling di tengah frame."""
    h, w = frame.shape[:2]
    box_h = int(h * SAMPLE_BOX_RATIO)
    box_w = int(w * SAMPLE_BOX_RATIO)
    y1 = (h - box_h) // 2
    x1 = (w - box_w) // 2
    y2 = y1 + box_h
    x2 = x1 + box_w
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def extract_hsv_values(crop_bgr) -> np.ndarray:
    """Konversi crop BGR ke HSV, kembalikan array nilai HSV semua piksel."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3)  # shape: (N, 3) → setiap baris = [H, S, V]


def draw_ui(frame, sample_counts: dict, last_action: str, last_action_time: float):
    """Gambar overlay UI pada frame."""
    h, w = frame.shape[:2]

    # ── Kotak Sampling ────────────────────────────────────────────────────────
    _, (x1, y1, x2, y2) = get_sample_region(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        frame, "ARAHKAN OBJEK KE SINI",
        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
    )

    # ── Panel Kiri: Panduan Tombol ────────────────────────────────────────────
    panel_w = 230
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "TOMBOL WARNA:", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    keys_info = [
        ("r", "Merah"),   ("o", "Oranye"),  ("k", "Kuning"),
        ("h", "Hijau"),   ("c", "Cyan"),    ("b", "Biru"),
        ("u", "Ungu"),    ("p", "Pink"),    ("w", "Putih"),
        ("t", "Hitam"),   ("a", "Abu-abu"),
    ]

    for i, (key, label) in enumerate(keys_info):
        color_name = label.lower().replace("-", "-")
        bgr = DISPLAY_COLORS.get(label.lower(), (200, 200, 200))
        count = sample_counts.get(label.lower(), 0)
        text = f"[{key}] {label}: {count} px"
        cv2.putText(frame, text, (10, 50 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, bgr, 1)

    cv2.putText(frame, "─────────────────────", (10, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
    cv2.putText(frame, "[d] Hapus sample terakhir", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 255), 1)
    cv2.putText(frame, "[s] Statistik sample", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 255), 1)
    cv2.putText(frame, "[q] Selesai & Simpan", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 80), 1)

    # ── Notifikasi Aksi Terakhir ──────────────────────────────────────────────
    if last_action and (time.time() - last_action_time) < 2.5:
        cv2.putText(frame, last_action, (w // 2 - 180, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)


def print_statistics(sample_counts: dict):
    """Cetak statistik sample ke terminal."""
    print("\n" + "=" * 45)
    print("  STATISTIK SAMPLE SAAT INI")
    print("=" * 45)
    total = 0
    for color, count in sorted(sample_counts.items()):
        bar = "█" * min(count // 500, 30)
        print(f"  {color:<10} {count:>7} piksel  {bar}")
        total += count
    print("─" * 45)
    print(f"  Total: {total} piksel")
    recommended = 5000
    for color, count in sample_counts.items():
        if count < recommended:
            print(f"  ⚠  '{color}' masih kurang ({count}/{recommended} px). Tambah lebih banyak!")
    print("=" * 45 + "\n")


# =============================================================================
# Main
# =============================================================================
def run(camera_index: int = 0):
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Muat sample yang sudah ada
    samples_data: dict[str, list] = {}
    sample_counts: dict[str, int] = {}
    for color_name in KEY_COLOR_MAP.values():
        existing = load_samples(color_name)
        samples_data[color_name] = existing
        sample_counts[color_name] = sum(len(s) for s in existing)

    # Riwayat aksi untuk undo
    action_history: list[str] = []  # menyimpan nama warna yang terakhir di-sample

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera index {camera_index}")

    print("=" * 55)
    print("  GENERATE COLOR SAMPLES — Pengambilan Sample Warna")
    print("=" * 55)
    print("  Arahkan objek berwarna ke KOTAK KUNING di tengah layar.")
    print("  Tekan tombol huruf yang sesuai warna objek tersebut.")
    print("  Tekan 'q' kalau sudah selesai.")
    print("=" * 55 + "\n")

    last_action = ""
    last_action_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Ambil region sampling
        crop, _ = get_sample_region(frame)

        # Gambar UI
        draw_ui(frame, sample_counts, last_action, last_action_time)
        cv2.imshow("Generate Color Samples", frame)

        key = cv2.waitKey(1) & 0xFF

        # ── Tombol Keluar ─────────────────────────────────────────────────────
        if key == ord("q"):
            break

        # ── Tombol Statistik ──────────────────────────────────────────────────
        elif key == ord("s"):
            print_statistics(sample_counts)

        # ── Tombol Undo (hapus sample batch terakhir) ─────────────────────────
        elif key == ord("d"):
            if action_history:
                last_color = action_history.pop()
                if samples_data[last_color]:
                    removed = samples_data[last_color].pop()
                    sample_counts[last_color] -= len(removed)
                    save_samples(last_color, samples_data[last_color])
                    msg = f"Dihapus: 1 batch '{last_color}'"
                    last_action = msg
                    last_action_time = time.time()
                    print(f"[UNDO] {msg}")
            else:
                print("[UNDO] Tidak ada sample untuk dihapus.")

        # ── Tombol Sample Warna ───────────────────────────────────────────────
        elif key in KEY_COLOR_MAP:
            color_name = KEY_COLOR_MAP[key]
            if crop.size == 0:
                print(f"[SKIP] Crop kosong, coba lagi.")
                continue

            hsv_pixels = extract_hsv_values(crop)  # shape: (N, 3)
            samples_data[color_name].append(hsv_pixels)
            sample_counts[color_name] += len(hsv_pixels)
            action_history.append(color_name)

            # Simpan ke disk langsung
            save_samples(color_name, samples_data[color_name])

            n_px = len(hsv_pixels)
            total = sample_counts[color_name]
            last_action = f"✓ Tersimpan: {color_name} (+{n_px} px → total {total})"
            last_action_time = time.time()
            print(f"[SIMPAN] {color_name}: +{n_px} piksel (total: {total} px)")

    cap.release()
    cv2.destroyAllWindows()

    # Ringkasan akhir
    print("\n" + "=" * 55)
    print("  SELESAI — Ringkasan Sample yang Tersimpan")
    print("=" * 55)
    print_statistics(sample_counts)
    print("  Langkah berikutnya:")
    print("  → python auto_calibrate_color.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ambil sample warna dari webcam untuk kalibrasi HSV"
    )
    parser.add_argument("--camera", type=int, default=0, help="Index kamera (default 0)")
    args = parser.parse_args()
    run(args.camera)
