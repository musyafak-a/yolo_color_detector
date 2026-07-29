# Object Detector: Deteksi Barang Berdasarkan Warna (YOLO + OpenCV)

Project ini mendeteksi objek menggunakan **YOLOv8** (via library `ultralytics`),
lalu untuk setiap objek yang terdeteksi, sistem menganalisis **warna dominan**-nya
menggunakan **OpenCV (HSV color space)**. Hasil akhirnya adalah label seperti:

```
botol (merah) 0.87
cangkir (biru) 0.91
```

---

## 1. Konsep / Algoritma Keseluruhan

Karena kamu baru pertama kali, penting untuk paham dulu **alur besarnya**
sebelum masuk ke kode:

```
[Gambar/Frame Input]
        |
        v
[YOLO mendeteksi objek] --> menghasilkan bounding box (x1,y1,x2,y2) + nama kelas + confidence
        |
        v
[Crop area bounding box dari gambar asli]
        |
        v
[Konversi crop ke HSV] --> hitung piksel mana yang masuk rentang warna tertentu
        |
        v
[Cocokkan ke warna dengan piksel terbanyak] --> "merah", "biru", "hijau", dst
        |
        v
[Gabungkan label] --> "nama_objek (warna)" digambar di atas bounding box
```

**Kenapa HSV, bukan RGB?**
RGB gampang berubah drastis kalau pencahayaan berubah. HSV memisahkan
"warna murni" (Hue) dari "kecerahan" (Value), jadi lebih stabil untuk
klasifikasi warna sederhana seperti ini.

**Kenapa pakai YOLO pretrained (bukan training dari nol)?**
YOLOv8 pretrained (`yolov8n.pt`) sudah dilatih di dataset COCO (80 kelas
objek umum: botol, cangkir, tas, orang, mobil, dll). Untuk mendeteksi
"barang" secara umum + warnanya, kamu **tidak perlu training ulang** —
cukup pakai model pretrained + tambahkan lapisan analisis warna di atasnya
(itulah yang dilakukan project ini).

Kamu baru perlu training custom (dijelaskan di bagian 6) **hanya jika**
barang yang mau kamu deteksi bukan bagian dari 80 kelas COCO (misal:
kemasan produk spesifik, jenis kotak tertentu, dll).

---

## 2. Struktur Project

```
yolo_color_detector/
├── requirements.txt      # daftar library yang dibutuhkan
├── color_utils.py        # fungsi analisis warna dominan (HSV)
├── detect_image.py        # jalankan deteksi pada 1 file gambar
├── detect_webcam.py       # jalankan deteksi realtime dari webcam
└── README.md              # panduan ini
```

---

## 3. Setup dari Nol (Step-by-Step)

### Langkah 1 — Install Python
Pastikan Python 3.9–3.12 sudah terinstall. Cek dengan:
```bash
python --version
```

### Langkah 2 — Buat virtual environment (biar rapi, tidak bentrok dengan project lain)
Karena kamu biasa pakai `.venv` di VS Code, langkahnya sama seperti biasa:
```bash
python -m venv .venv
```
Aktifkan:
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

### Langkah 3 — Install dependencies
Taruh semua file di atas dalam satu folder, lalu:
```bash
pip install -r requirements.txt
```
Ini akan menginstall `ultralytics` (YOLOv8), `opencv-python`, dan `numpy`.

### Langkah 4 — Jalankan deteksi pada gambar
```bash
python detect_image.py --source contoh.jpg
```
- Saat pertama kali dijalankan, `ultralytics` akan **otomatis download**
  model `yolov8n.pt` (~6MB) dari internet. Pastikan ada koneksi internet.
- Hasil akan muncul di jendela gambar, dan otomatis tersimpan di folder
  `output/`.

### Langkah 5 — Jalankan deteksi realtime dari webcam
```bash
python detect_webcam.py
```
Tekan `q` untuk berhenti.

---

## 4. Penjelasan Parameter Penting

| Parameter | Fungsi | Default |
|---|---|---|
| `--model` | Model YOLO yang dipakai. `yolov8n.pt` = paling ringan/cepat (nano). Ada juga `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` — makin besar makin akurat tapi makin lambat | `yolov8n.pt` |
| `--conf` | Confidence threshold (0–1). Makin tinggi, makin sedikit false positive tapi bisa miss objek yang kurang jelas | `0.4` |
| `--camera` | Index kamera (0 = kamera default laptop) | `0` |

---

## 5. Cara Kerja `color_utils.py` (Detail)

1. Ambil bagian **tengah** dari crop bounding box (60% tengah), supaya
   tidak kebawa warna background di pinggir kotak deteksi.
2. Convert ke HSV.
3. Untuk tiap warna yang didefinisikan di `COLOR_RANGES` (merah, oranye,
   kuning, hijau, biru, ungu, pink, putih, hitam, abu-abu), hitung berapa
   piksel yang masuk ke rentang HSV warna tersebut (`cv2.inRange`).
4. Warna dengan jumlah piksel terbanyak = warna dominan.
5. Kalau tidak ada warna yang dominan (piksel yang cocok < 5% dari total),
   dianggap `"tidak diketahui"`.

Kalau warna barangmu sering salah terdeteksi, kamu tinggal **tweak** angka
di `COLOR_RANGES` — ini bagian yang paling sering perlu disesuaikan karena
tiap kamera/pencahayaan beda karakteristik.

---

## 6. (Opsional) Training Custom Model — Kalau Barangmu Bukan Kelas COCO

Kalau kamu mau deteksi barang spesifik yang **bukan** salah satu dari 80
kelas COCO (misal: kemasan produk tertentu, jenis kotak custom), ikuti ini:

### a. Kumpulkan & Label Dataset
1. Kumpulkan minimal 100–200 foto per kelas barang (makin banyak makin baik).
2. Beri label bounding box pakai tool seperti:
   - [Roboflow](https://roboflow.com) (paling mudah, ada auto-export ke format YOLO)
   - [LabelImg](https://github.com/HumanSignal/labelImg)
3. Export dataset dalam format **YOLOv8** — nanti kamu dapat struktur folder:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### b. File `data.yaml` (contoh)
```yaml
train: dataset/train/images
val: dataset/valid/images
nc: 2
names: ['botol_a', 'kotak_b']
```

### c. Training
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # mulai dari pretrained (transfer learning)
model.train(data='dataset/data.yaml', epochs=50, imgsz=640)
```

### d. Pakai Model Hasil Training
Setelah training selesai, model tersimpan di `runs/detect/train/weights/best.pt`.
Tinggal ganti argumen `--model`:
```bash
python detect_image.py --source contoh.jpg --model runs/detect/train/weights/best.pt
```

---

## 7. Troubleshooting Umum

- **Error download model gagal** → cek koneksi internet, atau download manual
  `yolov8n.pt` dari [rilis resmi ultralytics](https://github.com/ultralytics/assets/releases)
  dan taruh di folder project.
- **Webcam tidak terbuka** → coba ganti `--camera 1` atau `--camera 2`.
- **Warna sering salah** → sesuaikan rentang HSV di `color_utils.py`, atau
  ambil sample warna asli barangmu pakai color picker untuk cek nilai HSV-nya.
- **Deteksi terlalu banyak salah (false positive)** → naikkan `--conf`
  misal ke `0.5` atau `0.6`.
