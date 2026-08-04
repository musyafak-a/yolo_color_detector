"""
detect_video.py
-----------------
Deteksi objek (YOLOv8) + warna dominan pada file video.
Mendukung Tiled Detection (SAHI-like) untuk mendeteksi objek kecil & banyak.

Cara pakai:
    python detect_video.py --source video_anda.mp4
    python detect_video.py --source video_anda.mp4 --model yolov8n.pt --conf 0.4
    python detect_video.py --source video_anda.mp4 --conf 0.15 --imgsz 1280
    python detect_video.py --source video_anda.mp4 --tiles 2  # bagi frame jadi 2x2 tile

Output:
    - Menampilkan video realtime dengan deteksi.
    - Video hasil otomatis tersimpan di folder ./output/
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from color_utils import detect_dominant_color


def compute_iou(box1, box2):
    """Hitung Intersection over Union antara dua bounding box [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def merge_nms(detections, iou_threshold=0.4):
    """
    Non-Maximum Suppression sederhana untuk menggabungkan deteksi dari
    beberapa tile yang mungkin mendeteksi objek yang sama.
    detections = list of (x1, y1, x2, y2, conf, cls_id)
    """
    if not detections:
        return []

    # Urutkan berdasarkan confidence (tertinggi dulu)
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            iou = compute_iou(best[:4], det[:4])
            # Hanya suppress jika IoU tinggi DAN kelas sama
            if iou < iou_threshold or best[5] != det[5]:
                remaining.append(det)
        detections = remaining

    return keep


def detect_with_tiles(model, frame, conf, iou, imgsz, max_det, tiles):
    """
    Jalankan deteksi menggunakan pendekatan tiling (SAHI-like).
    Frame dibagi menjadi tiles x tiles potongan yang saling overlap,
    lalu deteksi dijalankan di setiap potongan + frame penuh.
    Hasil digabungkan dengan NMS untuk menghapus duplikat.
    """
    h, w = frame.shape[:2]
    all_detections = []  # list of (x1, y1, x2, y2, conf, cls_id)

    # ── 1. Deteksi pada frame penuh (untuk objek besar) ──────────────────────
    results_full = model.predict(
        frame, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
        agnostic_nms=True, verbose=False,
    )[0]

    for box in results_full.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        all_detections.append((x1, y1, x2, y2, confidence, cls_id))

    # ── 2. Deteksi pada setiap tile (untuk objek kecil) ──────────────────────
    if tiles > 1:
        overlap_ratio = 0.25  # 25% overlap antar tile
        tile_h = h // tiles
        tile_w = w // tiles
        overlap_h = int(tile_h * overlap_ratio)
        overlap_w = int(tile_w * overlap_ratio)

        for row in range(tiles):
            for col in range(tiles):
                # Hitung koordinat tile dengan overlap
                y_start = max(0, row * tile_h - overlap_h)
                y_end = min(h, (row + 1) * tile_h + overlap_h)
                x_start = max(0, col * tile_w - overlap_w)
                x_end = min(w, (col + 1) * tile_w + overlap_w)

                tile = frame[y_start:y_end, x_start:x_end]

                if tile.size == 0:
                    continue

                results_tile = model.predict(
                    tile, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                    agnostic_nms=True, verbose=False,
                )[0]

                # Konversi koordinat tile kembali ke koordinat frame penuh
                for box in results_tile.boxes:
                    tx1, ty1, tx2, ty2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Offset koordinat ke posisi asli di frame
                    abs_x1 = tx1 + x_start
                    abs_y1 = ty1 + y_start
                    abs_x2 = tx2 + x_start
                    abs_y2 = ty2 + y_start

                    all_detections.append((abs_x1, abs_y1, abs_x2, abs_y2,
                                          confidence, cls_id))

    # ── 3. Gabungkan & hapus duplikat dengan NMS ─────────────────────────────
    final_detections = merge_nms(all_detections, iou_threshold=0.4)

    return final_detections


def run(source: str, model_path: str, conf: float, imgsz: int, iou: float,
        max_det: int, tiles: int, save_dir: str = "output"):
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(source):
        raise FileNotFoundError(f"Video tidak ditemukan: {source}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {source}")

    # Siapkan VideoWriter untuk menyimpan hasil
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # Handle nan or 0 fps
        fps = 30.0

    filename = os.path.basename(source)
    out_path = os.path.join(save_dir, f"detected_{filename}")

    # Gunakan codec mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"\nMemproses video: {source}")
    print(f"Resolusi: {width}x{height} | FPS: {fps:.1f}")
    print(f"Pengaturan deteksi:")
    print(f"  Confidence : {conf}")
    print(f"  Image size : {imgsz}")
    print(f"  IoU thresh : {iou}")
    print(f"  Max detect : {max_det}")
    print(f"  Tiling     : {tiles}x{tiles}" + (" (aktif - untuk objek kecil)" if tiles > 1 else " (nonaktif)"))
    print("Tombol kontrol (pastikan klik jendela video dulu):")
    print("  'q'     : Berhenti / Keluar")
    print("  'p'/' ' : Pause / Play")
    print("  'd'     : Skip maju 10 detik")
    print("  'a'     : Skip mundur 10 detik\n")

    is_paused = False
    frame_to_display = None

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("Selesai memproses seluruh video.")
                break

            # Deteksi dengan tiling (jika tiles > 1) atau deteksi biasa
            detections = detect_with_tiles(
                model, frame, conf, iou, imgsz, max_det, tiles
            )

            for (x1, y1, x2, y2, confidence, cls_id) in detections:
                class_name = model.names[cls_id]

                # Hindari error jika koordinat box melebihi ukuran gambar
                crop = frame[max(0, y1):y2, max(0, x1):x2]

                # Abaikan jika crop kosong
                if crop.size == 0:
                    continue

                color_name = detect_dominant_color(crop)

                label = f"{class_name} ({color_name}) {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Simpan frame ke file output (hanya jika video berjalan)
            out.write(frame)
            frame_to_display = frame

        # Tampilkan di layar
        if frame_to_display is not None:
            cv2.imshow("Deteksi Objek + Warna (Video)", frame_to_display)

        delay = 0 if is_paused else 1
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            print("Proses dihentikan oleh pengguna.")
            break
        elif key == ord("p") or key == 32:  # 'p' atau Spasi
            is_paused = not is_paused
        elif key == ord("d"):  # skip maju 10s
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 10 * fps)
            is_paused = False
        elif key == ord("a"):  # skip mundur 10s
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 10 * fps))
            is_paused = False

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo hasil deteksi berhasil disimpan di: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi objek + warna pada video")
    parser.add_argument("--source", required=True,
                        help="Path ke file video (contoh: video.mp4)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Path/nama model YOLO")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Confidence threshold (default: 0.15)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Resolusi inferensi (default: 1280)")
    parser.add_argument("--iou", type=float, default=0.3,
                        help="IoU threshold untuk NMS (default: 0.3)")
    parser.add_argument("--max-det", type=int, default=100,
                        help="Jumlah maksimal deteksi per frame (default: 100)")
    parser.add_argument("--tiles", type=int, default=2,
                        help="Jumlah tile per sisi (default: 2 = 2x2 tile). "
                             "Gunakan 1 untuk nonaktifkan tiling. "
                             "Gunakan 3 untuk objek yang sangat kecil (3x3 tile).")
    args = parser.parse_args()

    run(args.source, args.model, args.conf, args.imgsz, args.iou,
        args.max_det, args.tiles)
