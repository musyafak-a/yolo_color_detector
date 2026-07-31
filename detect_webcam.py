"""
detect_webcam.py
------------------
Deteksi objek (YOLOv8) + warna dominan secara realtime dari webcam.

Cara pakai:
    python detect_webcam.py
    python detect_webcam.py --model yolov8n.pt --conf 0.4 --camera 0

Tekan 'q' untuk keluar.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from color_utils import detect_dominant_color


def open_camera(camera_index: int):
    """
    Buka kamera dengan mencoba berbagai konfigurasi secara otomatis.
    Mengembalikan (cap, lebar, tinggi) atau raise RuntimeError.
    """
    # Urutan backend yang dicoba: DSHOW → MSMF → default
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("MSMF",       cv2.CAP_MSMF),
        ("Default",    cv2.CAP_ANY),
    ]

    # Resolusi yang dicoba dari kecil ke besar (biar kompatibel lebih banyak kamera)
    resolutions = [
        (640, 480),
        (1280, 720),
        (320, 240),
    ]

    for backend_name, backend in backends:
        print(f"  Mencoba backend: {backend_name}...", end=" ")
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            print("GAGAL")
            continue

        for w, h in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            # Baca frame uji — retry hingga 40x agar kamera punya waktu warm-up
            valid_frame = False
            for _ in range(40):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Pastikan frame tidak sepenuhnya hitam (mean > 5)
                    if frame.mean() > 5:
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"OK ({actual_w}x{actual_h})")
                        return cap, actual_w, actual_h

            # Frame masih hitam, coba resolusi berikutnya
        print(f"GAGAL (semua resolusi)")
        cap.release()

    raise RuntimeError(
        f"Tidak bisa membuka kamera index {camera_index} dengan konfigurasi apapun.\n"
        "Pastikan: (1) tidak ada app lain yang pakai kamera, (2) driver kamera terinstall.\n"
        "Coba juga: python detect_webcam.py --camera 1"
    )


def run(model_path: str, conf: float, camera_index: int):
    model = YOLO(model_path)

    print(f"\nMembuka kamera index {camera_index}...")
    cap, width, height = open_camera(camera_index)
    print(f"Kamera aktif [{width}x{height}]. Tekan 'q' untuk keluar.\n")

    consecutive_failures = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                print("Gagal membaca frame dari kamera (10x berturut-turut). Menghentikan.")
                break
            time.sleep(0.05)
            continue

        # Reset counter kegagalan kalau frame berhasil
        consecutive_failures = 0

        results = model.predict(frame, conf=conf, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            color_name = detect_dominant_color(crop)

            label = f"{class_name} ({color_name}) {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.imshow("Deteksi Objek + Warna (Realtime)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi objek + warna realtime via webcam")
    parser.add_argument("--model", default="yolov8n.pt", help="Path/nama model YOLO")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Index kamera (default 0)")
    args = parser.parse_args()

    run(args.model, args.conf, args.camera)
