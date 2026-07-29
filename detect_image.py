"""
detect_image.py
-----------------
Deteksi objek (YOLOv8) + deteksi warna dominan pada tiap objek yang terdeteksi,
dijalankan pada SATU gambar (file .jpg/.png).

Cara pakai:
    python detect_image.py --source path/ke/gambar.jpg
    python detect_image.py --source path/ke/gambar.jpg --model yolov8n.pt --conf 0.4

Output:
    - Jendela gambar dengan bounding box + label "nama_objek (warna)"
    - File hasil tersimpan otomatis di folder ./output/
"""

import argparse
import os

import cv2
from ultralytics import YOLO

from color_utils import detect_dominant_color


def run(source: str, model_path: str, conf: float, save_dir: str = "output"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load model YOLO (pretrained COCO, otomatis download kalau belum ada)
    model = YOLO(model_path)

    # 2. Baca gambar
    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Tidak bisa membaca gambar: {source}")

    # 3. Jalankan deteksi objek
    results = model.predict(image, conf=conf, verbose=False)[0]

    # 4. Untuk tiap objek yang terdeteksi: crop -> deteksi warna -> gambar box+label
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        crop = image[max(0, y1):y2, max(0, x1):x2]
        color_name = detect_dominant_color(crop)

        label = f"{class_name} ({color_name}) {confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # 5. Simpan & tampilkan hasil
    out_path = os.path.join(save_dir, os.path.basename(source))
    cv2.imwrite(out_path, image)
    print(f"Hasil disimpan di: {out_path}")

    cv2.imshow("Deteksi Objek + Warna", image)
    print("Tekan tombol apa saja di jendela gambar untuk keluar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi objek + warna pada gambar")
    parser.add_argument("--source", required=True, help="Path ke file gambar")
    parser.add_argument("--model", default="yolov8n.pt", help="Path/nama model YOLO")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    run(args.source, args.model, args.conf)
