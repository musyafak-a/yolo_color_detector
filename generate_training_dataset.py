"""
generate_training_dataset.py
-----------------------------
Tool untuk membuat dataset YOLO secara interaktif dari webcam.
Dipakai kalau objek yang ingin kamu deteksi BUKAN bagian dari 80 kelas COCO
(misal: produk custom, jenis dompet tertentu, kemasan spesifik, dll).

Cara pakai:
    python generate_training_dataset.py
    python generate_training_dataset.py --classes "dompet,botol_custom,kotak_merah"
    python generate_training_dataset.py --output dataset_custom --split 0.8

Kontrol saat mode labeling:
    KLIK + DRAG = gambar bounding box
    ENTER / SPASI = konfirmasi bounding box & pilih kelas
    r = Ulangi (hapus bounding box saat ini)
    c = Capture frame baru
    s = Skip frame ini (tanpa menyimpan)
    q = Keluar / Selesai

Output:
    dataset/
    ├── train/
    │   ├── images/   ← file .jpg
    │   └── labels/   ← file .txt (format YOLO)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml
"""

import argparse
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Konfigurasi Default
# =============================================================================
DEFAULT_CLASSES = ["objek"]  # Ganti dengan nama kelas yang kamu inginkan
DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_TRAIN_SPLIT = 0.8  # 80% train, 20% valid
WINDOW_NAME = "Generate Training Dataset"

# Warna untuk tiap kelas (BGR) — hingga 20 kelas
CLASS_COLORS = [
    (0, 255, 0),    (0, 0, 255),    (255, 165, 0),  (255, 0, 255),
    (0, 255, 255),  (255, 255, 0),  (128, 0, 255),  (255, 128, 0),
    (0, 128, 255),  (128, 255, 0),  (255, 0, 128),  (0, 255, 128),
    (100, 100, 255),(255, 100, 100),(100, 255, 100), (200, 50, 200),
    (50, 200, 200), (200, 200, 50), (150, 75, 0),   (75, 0, 150),
]


# =============================================================================
# State Global untuk Mouse Callback
# =============================================================================
class DrawingState:
    def __init__(self):
        self.drawing = False
        self.start_pt = (-1, -1)
        self.end_pt = (-1, -1)
        self.confirmed = False
        self.boxes: list[dict] = []  # list of {x1,y1,x2,y2,cls_id}

    def reset_current(self):
        self.drawing = False
        self.start_pt = (-1, -1)
        self.end_pt = (-1, -1)
        self.confirmed = False

    def has_valid_box(self):
        x1, y1 = self.start_pt
        x2, y2 = self.end_pt
        return (x1 >= 0 and y1 >= 0 and
                abs(x2 - x1) > 10 and abs(y2 - y1) > 10)


state = DrawingState()


def mouse_callback(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.start_pt = (x, y)
        state.end_pt = (x, y)
        state.confirmed = False
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        state.end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state.end_pt = (x, y)
        state.drawing = False


# =============================================================================
# Fungsi Bantu
# =============================================================================
def setup_output_dirs(output_dir: str) -> dict[str, Path]:
    """Buat struktur folder dataset."""
    dirs = {
        "train_img": Path(output_dir) / "train" / "images",
        "train_lbl": Path(output_dir) / "train" / "labels",
        "valid_img": Path(output_dir) / "valid" / "images",
        "valid_lbl": Path(output_dir) / "valid" / "labels",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_data_yaml(output_dir: str, class_names: list[str]):
    """Buat file data.yaml untuk training YOLO."""
    yaml_content = f"""# Dataset konfigurasi untuk YOLOv8
# Generated oleh generate_training_dataset.py

train: {output_dir}/train/images
val: {output_dir}/valid/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[YAML] Disimpan: {yaml_path}")
    return yaml_path


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Konversi pixel coords ke format YOLO (cx, cy, w, h) dalam range 0-1."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    return cx, cy, w, h


def save_sample(frame, boxes: list[dict], img_path: Path, lbl_path: Path):
    """Simpan gambar dan file label YOLO."""
    cv2.imwrite(str(img_path), frame)
    h, w = frame.shape[:2]
    with open(str(lbl_path), "w") as f:
        for box in boxes:
            cx, cy, bw, bh = bbox_to_yolo(box["x1"], box["y1"], box["x2"], box["y2"], w, h)
            f.write(f"{box['cls_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def draw_ui(display, class_names, total_saved, mode_text=""):
    """Gambar UI overlay pada frame."""
    h, w = display.shape[:2]

    # Panel atas
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    cv2.putText(display, f"Total tersimpan: {total_saved} gambar",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(display, mode_text,
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1)

    # Panel bawah: panduan kelas
    panel_h = 22 + len(class_names) * 22
    cv2.rectangle(overlay, (0, h - panel_h - 10), (300, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    cv2.putText(display, "KELAS:", (10, h - panel_h + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, name in enumerate(class_names):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        cv2.putText(display, f"[{i}] {name}", (10, h - panel_h + 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # Panduan singkat pojok kanan bawah
    hints = [
        "DRAG = Buat bbox",
        "ENTER = Konfirmasi",
        "r = Reset bbox",
        "c = Frame baru",
        "s = Skip frame",
        "q = Keluar",
    ]
    for i, hint in enumerate(hints):
        cv2.putText(display, hint, (w - 200, h - (len(hints) - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)


def draw_boxes(display, boxes, class_names):
    """Gambar semua bounding box yang sudah dikonfirmasi."""
    for box in boxes:
        color = CLASS_COLORS[box["cls_id"] % len(CLASS_COLORS)]
        cv2.rectangle(display, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        label = class_names[box["cls_id"]]
        cv2.putText(display, label, (box["x1"], max(20, box["y1"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def pick_class(class_names: list[str]) -> int | None:
    """
    Tampilkan window pemilihan kelas dan tunggu user memilih (tekan angka).
    Returns index kelas atau None jika dibatalkan.
    """
    print("\n  Pilih kelas untuk bounding box ini:")
    for i, name in enumerate(class_names):
        print(f"    [{i}] {name}")
    print("    [ESC] Batalkan")

    # Buat window kecil pilihan kelas
    panel = np.zeros((60 + len(class_names) * 40, 350, 3), dtype=np.uint8)
    cv2.putText(panel, "Tekan angka untuk pilih kelas:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    for i, name in enumerate(class_names):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        cv2.putText(panel, f"[{i}] {name}", (10, 60 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow("Pilih Kelas", panel)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow("Pilih Kelas")
            return None
        digit = key - ord("0")
        if 0 <= digit < len(class_names):
            cv2.destroyWindow("Pilih Kelas")
            return digit
        print(f"  Masukkan angka 0–{len(class_names)-1}")


# =============================================================================
# Main Loop
# =============================================================================
def run(class_names: list[str], output_dir: str, train_split: float, camera_index: int):
    global state

    dirs = setup_output_dirs(output_dir)
    yaml_path = save_data_yaml(output_dir, class_names)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera index {camera_index}")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\n" + "=" * 60)
    print("  GENERATE TRAINING DATASET — Labeling Interaktif")
    print("=" * 60)
    print(f"  Kelas: {class_names}")
    print(f"  Output: {output_dir}/")
    print(f"  Train/Valid split: {train_split:.0%}/{1-train_split:.0%}")
    print("─" * 60)
    print("  CARA PAKAI:")
    print("  1. Tekan [c] untuk CAPTURE frame dari webcam")
    print("  2. KLIK + DRAG untuk gambar bounding box di objek")
    print("  3. Tekan [ENTER] untuk konfirmasi & pilih kelas")
    print("  4. Ulangi untuk bounding box tambahan di frame yang sama")
    print("  5. Tekan [c] lagi untuk capture frame berikutnya")
    print("  6. Tekan [q] kalau sudah selesai")
    print("=" * 60 + "\n")

    total_saved = 0
    frame_buffer = None  # Frame yang di-capture untuk dilabeli
    mode = "preview"  # "preview" atau "labeling"
    mode_text = "Mode: PREVIEW — Tekan [c] untuk capture frame"

    while True:
        if mode == "preview":
            ret, live_frame = cap.read()
            if not ret:
                break
            display = live_frame.copy()
            state.boxes = []
            state.reset_current()
        else:
            display = frame_buffer.copy()

        # Gambar bounding box yang sedang digambar (belum dikonfirmasi)
        if state.has_valid_box() and mode == "labeling":
            x1 = min(state.start_pt[0], state.end_pt[0])
            y1 = min(state.start_pt[1], state.end_pt[1])
            x2 = max(state.start_pt[0], state.end_pt[0])
            y2 = max(state.start_pt[1], state.end_pt[1])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(display, "Tekan ENTER utk konfirmasi",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        draw_boxes(display, state.boxes, class_names)
        draw_ui(display, class_names, total_saved, mode_text)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        # ── Keluar ────────────────────────────────────────────────────────────
        if key == ord("q"):
            break

        # ── Capture Frame ─────────────────────────────────────────────────────
        elif key == ord("c"):
            ret, live_frame = cap.read()
            if ret:
                frame_buffer = live_frame.copy()
                state.boxes = []
                state.reset_current()
                mode = "labeling"
                mode_text = "Mode: LABELING — Drag bbox, ENTER utk konfirmasi, [c] frame baru"
                print("\n[CAPTURE] Frame baru di-capture. Mulai labeling!")
            else:
                print("[ERROR] Gagal capture frame.")

        # ── Skip Frame ────────────────────────────────────────────────────────
        elif key == ord("s"):
            mode = "preview"
            mode_text = "Mode: PREVIEW — Tekan [c] untuk capture frame"
            state.reset_current()
            state.boxes = []
            print("[SKIP] Frame dilewati.")

        # ── Reset Bbox Saat Ini ───────────────────────────────────────────────
        elif key == ord("r"):
            state.reset_current()
            print("[RESET] Bounding box dihapus.")

        # ── Konfirmasi Bbox ───────────────────────────────────────────────────
        elif key in (13, 32) and mode == "labeling":  # ENTER atau SPASI
            if not state.has_valid_box():
                # Tidak ada bbox: simpan gambar dengan label yang sudah ada
                if state.boxes:
                    ts = int(time.time() * 1000)
                    is_train = random.random() < train_split
                    split = "train" if is_train else "valid"
                    img_path = dirs[f"{split}_img"] / f"frame_{ts}.jpg"
                    lbl_path = dirs[f"{split}_lbl"] / f"frame_{ts}.txt"
                    save_sample(frame_buffer, state.boxes, img_path, lbl_path)
                    total_saved += 1
                    n_boxes = len(state.boxes)
                    print(f"[SIMPAN] {split}: frame_{ts}.jpg ({n_boxes} bbox) — Total: {total_saved}")
                    state.boxes = []
                    state.reset_current()
                    mode = "preview"
                    mode_text = "Mode: PREVIEW — Tekan [c] untuk capture frame"
                else:
                    print("[INFO] Tidak ada bounding box. Gambar bbox dulu!")
            else:
                # Ada bbox baru: konfirmasi & pilih kelas
                x1 = min(state.start_pt[0], state.end_pt[0])
                y1 = min(state.start_pt[1], state.end_pt[1])
                x2 = max(state.start_pt[0], state.end_pt[0])
                y2 = max(state.start_pt[1], state.end_pt[1])

                cls_id = pick_class(class_names)
                if cls_id is not None:
                    state.boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls_id": cls_id})
                    print(f"[BBOX] Kelas '{class_names[cls_id]}' ditambahkan. "
                          f"Total bbox: {len(state.boxes)}. "
                          f"[ENTER] lagi untuk simpan, atau gambar bbox tambahan.")
                state.reset_current()

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    train_count = len(list((dirs["train_img"]).glob("*.jpg")))
    valid_count = len(list((dirs["valid_img"]).glob("*.jpg")))

    print("\n" + "=" * 60)
    print("  SELESAI — Ringkasan Dataset")
    print("=" * 60)
    print(f"  Total gambar tersimpan: {total_saved}")
    print(f"  Train: {train_count} gambar → {dirs['train_img']}")
    print(f"  Valid: {valid_count} gambar → {dirs['valid_img']}")
    print(f"  Config: {yaml_path}")
    print("─" * 60)
    print("  Langkah berikutnya:")
    print("  → python train_custom_model.py")
    print("    ATAU")
    print(f"  → python train_custom_model.py --data {yaml_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Buat dataset YOLO secara interaktif dari webcam"
    )
    parser.add_argument(
        "--classes", type=str,
        default=",".join(DEFAULT_CLASSES),
        help="Nama kelas dipisah koma, contoh: 'dompet,botol_custom,kotak_merah'"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Folder output dataset (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--split", type=float, default=DEFAULT_TRAIN_SPLIT,
        help=f"Proporsi data untuk training (default: {DEFAULT_TRAIN_SPLIT})"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Index kamera (default: 0)"
    )
    args = parser.parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    run(class_names, args.output, args.split, args.camera)
