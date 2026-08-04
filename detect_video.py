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
        elif key == ord("p") or key == 32: # 'p' atau Spasi
            is_paused = not is_paused
        elif key == ord("d"): # skip maju 10s
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 10 * fps)
            is_paused = False
        elif key == ord("a"): # skip mundur 10s
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 10 * fps))
            is_paused = False

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
