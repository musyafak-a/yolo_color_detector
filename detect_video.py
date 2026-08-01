"""
detect_video.py
-----------------
Deteksi objek (YOLOv8) + warna dominan pada file video.

Cara pakai:
    python detect_video.py --source video_anda.mp4
    python detect_video.py --source video_anda.mp4 --model yolov8n.pt --conf 0.4

Output:
    - Menampilkan video realtime dengan deteksi.
    - Video hasil otomatis tersimpan di folder ./output/
"""

import argparse
import os
import cv2
from ultralytics import YOLO
from color_utils import detect_dominant_color

def run(source: str, model_path: str, conf: float, save_dir: str = "output"):
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
    if fps == 0 or fps != fps: # Handle nan or 0 fps
        fps = 30.0

    filename = os.path.basename(source)
    out_path = os.path.join(save_dir, f"detected_{filename}")
    
    # Gunakan codec mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"\nMemproses video: {source}")
    print(f"Resolusi: {width}x{height} | FPS: {fps:.1f}")
    print(f"Tekan 'q' pada jendela video untuk berhenti memproses.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Selesai memproses seluruh video.")
            break

        results = model.predict(frame, conf=conf, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

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

        # Simpan frame ke file output
        out.write(frame)

        # Tampilkan di layar
        cv2.imshow("Deteksi Objek + Warna (Video)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Proses dihentikan oleh pengguna.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo hasil deteksi berhasil disimpan di: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi objek + warna pada video")
    parser.add_argument("--source", required=True, help="Path ke file video (contoh: video.mp4)")
    parser.add_argument("--model", default="yolov8n.pt", help="Path/nama model YOLO")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run(args.source, args.model, args.conf)
