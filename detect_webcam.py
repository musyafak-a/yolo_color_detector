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

import cv2
from ultralytics import YOLO

from color_utils import detect_dominant_color


def run(model_path: str, conf: float, camera_index: int):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera index {camera_index}")

    print("Kamera aktif. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

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
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Index kamera (default 0)")
    args = parser.parse_args()

    run(args.model, args.conf, args.camera)
