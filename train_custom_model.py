"""
train_custom_model.py
----------------------
Wrapper untuk training model YOLOv8 custom dari dataset yang sudah dilabeli.

Cara pakai:
    python train_custom_model.py
    python train_custom_model.py --data dataset/data.yaml --epochs 50
    python train_custom_model.py --data dataset/data.yaml --epochs 100 --model yolov8s.pt

Setelah training selesai, model terbaik tersimpan di:
    runs/detect/train/weights/best.pt

Untuk pakai model baru di deteksi:
    python detect_webcam.py --model runs/detect/train/weights/best.pt
    python detect_image.py --source foto.jpg --model runs/detect/train/weights/best.pt
"""

import argparse
import os
import sys
from pathlib import Path


def check_dataset(data_yaml: str) -> bool:
    """Validasi bahwa file data.yaml dan struktur dataset ada."""
    if not os.path.exists(data_yaml):
        print(f"[ERROR] File tidak ditemukan: {data_yaml}")
        print("        Jalankan dulu: python generate_training_dataset.py")
        return False

    # Baca dan cek isi yaml
    with open(data_yaml, "r") as f:
        content = f.read()

    required_keys = ["train:", "val:", "nc:", "names:"]
    missing = [k for k in required_keys if k not in content]
    if missing:
        print(f"[ERROR] data.yaml tidak lengkap. Key yang hilang: {missing}")
        return False

    return True


def count_dataset_size(data_yaml: str) -> dict:
    """Hitung jumlah gambar di train dan valid."""
    dataset_dir = Path(data_yaml).parent
    counts = {}
    for split in ["train", "valid"]:
        img_dir = dataset_dir / split / "images"
        if img_dir.exists():
            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            lbl_dir = dataset_dir / split / "labels"
            lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
            counts[split] = {"images": len(imgs), "labels": len(lbls)}
        else:
            counts[split] = {"images": 0, "labels": 0}
    return counts


def recommend_epochs(total_images: int) -> int:
    """Rekomendasikan jumlah epoch berdasarkan ukuran dataset."""
    if total_images < 50:
        return 100
    elif total_images < 200:
        return 75
    elif total_images < 500:
        return 50
    else:
        return 30


def run(
    data_yaml: str,
    base_model: str,
    epochs: int | None,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
):
    print("\n" + "=" * 60)
    print("  TRAIN CUSTOM MODEL - Training YOLOv8")
    print("=" * 60)

    # ── Validasi Dataset ──────────────────────────────────────────────────────
    if not check_dataset(data_yaml):
        sys.exit(1)

    counts = count_dataset_size(data_yaml)
    train_n = counts["train"]["images"]
    valid_n = counts["valid"]["images"]
    total_n = train_n + valid_n

    print(f"\n  Dataset: {data_yaml}")
    print(f"  Train  : {train_n} gambar")
    print(f"  Valid  : {valid_n} gambar")
    print(f"  Total  : {total_n} gambar")

    if train_n < 10:
        print(f"\n  [WARN] Dataset terlalu kecil ({train_n} gambar train)!")
        print("          Disarankan minimal 50 gambar per kelas untuk hasil yang layak.")
        print("          Kumpulkan lebih banyak data dengan generate_training_dataset.py")
        ans = input("  Lanjutkan training? [y/N]: ").strip().lower()
        if ans != "y":
            print("  Dibatalkan.")
            sys.exit(0)

    # ── Tentukan Epochs ───────────────────────────────────────────────────────
    if epochs is None:
        epochs = recommend_epochs(total_n)
        print(f"\n  [AUTO] Epochs direkomendasikan: {epochs} (berdasarkan {total_n} gambar)")

    # ── Summary Konfigurasi ───────────────────────────────────────────────────
    print(f"\n  Base model : {base_model}")
    print(f"  Epochs     : {epochs}")
    print(f"  Image size : {imgsz}x{imgsz}")
    print(f"  Batch size : {batch}")
    print(f"  Device     : {device}")
    print(f"  Output     : {project}/{name}/")
    print("-" * 60)

    # Estimasi waktu kasar
    sec_per_epoch = max(5, total_n * 0.05)  # estimasi kasar: ~0.05 detik/gambar/epoch di CPU
    est_minutes = (epochs * sec_per_epoch) / 60
    print(f"  Estimasi waktu (CPU): ~{est_minutes:.0f} menit")
    print("  (Lebih cepat jika ada GPU)")
    print("-" * 60)

    confirm = input("\n  Mulai training? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Dibatalkan.")
        sys.exit(0)

    # ── Import & Training ─────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics tidak terinstall!")
        print("        Jalankan: pip install ultralytics")
        sys.exit(1)

    print("\n  Memulai training...")
    print("=" * 60 + "\n")

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # ── Hasil Training ────────────────────────────────────────────────────────
    best_weights = Path(project) / name / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("  TRAINING SELESAI!")
    print("=" * 60)

    if best_weights.exists():
        print(f"\n  ✓ Model terbaik: {best_weights}")
        print(f"\n  Untuk pakai model baru:")
        print(f"    python detect_webcam.py --model {best_weights}")
        print(f"    python detect_image.py --source foto.jpg --model {best_weights}")
    else:
        print(f"\n  [WARN] best.pt tidak ditemukan di {best_weights}")
        print(f"          Cek folder: {project}/{name}/weights/")

    # Cetak metrik validasi terakhir jika tersedia
    try:
        metrics = results.results_dict
        mAP50 = metrics.get("metrics/mAP50(B)", None)
        mAP5095 = metrics.get("metrics/mAP50-95(B)", None)
        if mAP50 is not None:
            print(f"\n  Metrik Akhir:")
            print(f"    mAP@50     : {mAP50:.4f}")
            if mAP5095 is not None:
                print(f"    mAP@50-95  : {mAP5095:.4f}")

        # Interpretasi mAP
        if mAP50 is not None:
            if mAP50 >= 0.8:
                print("    → Akurasi BAIK! Model siap dipakai.")
            elif mAP50 >= 0.5:
                print("    → Akurasi CUKUP. Tambah data atau naikkan epochs untuk lebih baik.")
            else:
                print("    → Akurasi KURANG. Disarankan kumpulkan lebih banyak data.")
    except Exception:
        pass

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training custom YOLOv8 model dari dataset lokal"
    )
    parser.add_argument(
        "--data", type=str, default="dataset/data.yaml",
        help="Path ke data.yaml (default: dataset/data.yaml)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Base model YOLO (default: yolov8n.pt). "
             "Pilihan: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Jumlah epoch training (default: otomatis berdasarkan ukuran dataset)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Ukuran gambar untuk training (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16, kurangi ke 8 atau 4 kalau RAM/VRAM kurang)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu', '0' (GPU pertama), '0,1' (multi-GPU) (default: cpu)"
    )
    parser.add_argument(
        "--project", type=str, default="runs/detect",
        help="Folder output training (default: runs/detect)"
    )
    parser.add_argument(
        "--name", type=str, default="custom_train",
        help="Nama subfolder training (default: custom_train)"
    )
    args = parser.parse_args()

    run(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
