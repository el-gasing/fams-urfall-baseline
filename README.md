# FAMS vs UR Fall: baseline Kwolek & Kepski (2014)

Implementasi ulang **algoritma baseline UR Fall Detection Dataset** yang bisa dijalankan di dataset jatuh
mana pun. Kode yang sama dipakai untuk dua dataset:

- **UR Fall Detection Dataset** (URFD): replikasi hasil paper aslinya, sebagai patokan.
- **FAMS**: dataset jatuh dengan 3 IMU (pinggang, dada, paha kanan) dan 4 kamera.

Tiap dataset punya adapter kecil sendiri. Algoritma, evaluasi dan laporannya dipakai bersama.

| Bagian | Paper | Yang diuji |
|---|---|---|
| **Algoritma** | Kwolek & Kepski, "Human fall detection on embedded platform using depth maps and wireless accelerometer", *Computer Methods and Programs in Biomedicine* 117 (2014) 489–501 | Pemicu akselerometer (SV_total > 3 g), lalu verifikasi pose dengan SVM linear / k-NN pada 4 fitur orang |
| **Pembanding threshold** | Bourke *et al.*, "Evaluation of a threshold-based tri-axial accelerometer fall detection algorithm", *Gait & Posture* 26(2) (2007) 194–199 | UFT (puncak) dan LFT (lembah) pada SV_total, seperti di Tabel 2 paper Kwolek |

Hanya dua hal yang berbeda antar dataset: **pembaca data** (adapter) dan **sumber fitur pose**.
- UR Fall: fitur dari depth map Kinect, dirilis bersama dataset.
- FAMS: fitur yang sama dihitung dari **skeleton 3D hasil triangulasi 4 kamera** (`extract-3d`), karena
  FAMS tidak punya kamera depth.

> **Status.** UR Fall sudah dijalankan penuh (hasil di [§8](#8-hasil)). Untuk FAMS:
> - Kodenya lengkap.
> - Metode threshold sudah dicoba di 2 sesi.
> - `extract-3d` butuh GPU dan belum dijalankan untuk semua sesi.

---

## Daftar isi

1. [Mulai cepat](#1-mulai-cepat)
2. [Alur, langkah demi langkah](#2-alur-langkah-demi-langkah)
3. [Dataset](#3-dataset)
4. [Perintah](#4-perintah)
5. [Konfigurasi](#5-konfigurasi)
6. [Menambah dataset baru](#6-menambah-dataset-baru)
7. [Keluaran](#7-keluaran)
8. [Hasil](#8-hasil)
9. [Detail metode](#9-detail-metode)
10. [Penyimpangan dari paper](#10-penyimpangan-dari-paper)
11. [Catatan metodologi](#11-catatan-metodologi)
12. [Struktur proyek](#12-struktur-proyek)
13. [Troubleshooting](#13-troubleshooting)
14. [Referensi](#14-referensi)

---

## 1. Mulai cepat

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q                       # cek cepat, tanpa dataset

# --- UR Fall (beberapa detik) ---
python -m urfall_bench download -d configs/datasets/urfall.yaml          # ~110 MB ke data/urfall/
python -m urfall_bench run -d configs/datasets/urfall.yaml               # 30 jatuh + 40 ADL
python -m urfall_bench run -d configs/datasets/urfall_paper.yaml         # 30 + 30, seperti paper

# --- FAMS (zip/folder sesi di data/fams/) ---
python -m urfall_bench inspect    -d configs/datasets/fams.yaml          # ringkasan + % terpicu per label
python -m urfall_bench extract-3d -d configs/datasets/fams.yaml -s pose3d.device=cuda   # GPU, lama
python -m urfall_bench run        -d configs/datasets/fams.yaml

# --- atau semuanya sekaligus ---
bash scripts/run_all.sh -- -s pose3d.device=cuda
```

Hasil setiap run ada di `results/<dataset>/<benchmark>_<waktu>/report.md`. Panduan menjalankan di server
GPU ada di [SERVER_GUIDE.md](SERVER_GUIDE.md).

Kebutuhan:
- Python 3.9+.
- Benchmark cukup `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyyaml`.
- `extract-3d` juga butuh `torch`, `torchvision`, `av` (PyAV) dan `opencv-python`. GPU sangat disarankan.

---

## 2. Alur, langkah demi langkah

```
UR Fall
  acc.csv ─► SV_total ─► pemicu > 3 g ─────────────────────┐
                                                           ├─► fitur di [t, t+1 s] ─► SVM / k-NN ─► jatuh / ADL per urutan
  urfall-cam0-*.csv (fitur depth per frame) ───────────────┘

FAMS
  waist_consolidated.csv ─► SV_total ─► pemicu > 3 g ──────┐
                                                           ├─► fitur di [t, t+1 s] ─► SVM / k-NN ─► jatuh / ADL per urutan
  video 4 kamera ─► skeleton 2D ─► triangulasi 3D ─► fitur ┘   (extract-3d)
```

1. **Bagi data jadi urutan** (`Sequence`). Di UR Fall satu urutan = satu file `fall-XX` / `adl-XX`. Di
   FAMS satu urutan = satu segmen label ≠ 0 di `label_timeline`. Label urutan biner: **jatuh** atau **ADL**.
2. **Tahap akselerometer.** Hitung SV_total = √(Ax² + Ay² + Az²) dari IMU pinggang. Sampel di atas 3 g
   menjadi kejadian "potential fall". Sampel di atas ambang yang berjarak < 1 s digabung jadi satu
   kejadian, dengan waktu = puncaknya.
3. **Tahap verifikasi pose.** Ambil fitur pose di [t, t + 1 s] setelah pemicu, lalu klasifier memutuskan
   apakah ini jatuh. Ada dua mode, lihat [§9.4](#94-mode-pose-persis-paper) dan
   [§9.5](#95-mode-kandidat-untuk-perbandingan).
4. **Keputusan per urutan.** Urutan dinyatakan jatuh kalau ada pemicu yang lolos verifikasi. Urutan tanpa
   pemicu = ADL.
5. **Pembanding.** UFT, LFT, dan "pemicu saja" (tahap 2 tanpa tahap 3) dihitung pada urutan yang sama.
6. **Metrik.** Accuracy, precision, sensitivity, specificity (definisi Bagian 7 paper), ditambah F1.
   Semuanya per urutan dan dinilai *out-of-fold*.

---

## 3. Dataset

### 3.1 UR Fall Detection Dataset (`urfall`)

Sumber: <https://fenix.ur.edu.pl/mkepski/ds/uf.html>, lisensi CC BY-NC-SA 4.0. Isinya 30 urutan jatuh dan
40 urutan ADL. Direkam dengan 2 Kinect (cam0 depan, cam1 atap) dan akselerometer di pinggang.
Unduh dengan `python -m urfall_bench download -d configs/datasets/urfall.yaml`. Video bisa dilewati dengan
`-s dataset.download_video=false`.

| File | Isi | Dipakai untuk |
|---|---|---|
| `<seq>-acc.csv` | `waktu_ms, SV_total, Ax, Ay, Az` (g) | Pemicu, UFT, LFT |
| `<seq>-data.csv` | `nomor_frame, waktu_ms, SV_total di frame itu`, satu jam dengan `acc.csv` | Waktu tiap frame depth |
| `urfall-cam0-falls.csv`, `urfall-cam0-adls.csv` | `seq, frame, postur, HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40` | Fitur pose dan label postur |
| `<seq>-cam0.mp4` | Video RGB (opsional) | Tidak dipakai kode |

Label postur: **−1** = tidak berbaring, **0** = sedang jatuh (diabaikan saat latih), **1** = berbaring.
UR Fall tidak memberi ID subjek per urutan, jadi kelompok CV = urutan itu sendiri.

`urfall_paper.yaml` = `urfall.yaml` dengan `include` 30 jatuh + `adl-01..30`, karena paper memakai 30
urutan ADL tanpa menyebut yang mana.

### 3.2 FAMS (`fams`)

Satu sesi = satu subjek yang menjalankan seluruh protokol. Taruh zip ekspor sesi (atau folder hasil
ekstraknya) di `data/fams/`:

```
<sesi>/
├── cameras.json                         # session_id, daftar kamera, started_at_ms
├── videos/<sid>_camN_video_sync.webm    # 4 kamera, 1280x720
└── data/
    ├── manifest.json                    # nama subjek (kelompok LOSO)
    ├── <sid>_integrity_report.json      # label_timeline (titik pergantian label), session_end_ms
    ├── <sid>_waist_consolidated.csv     # timestamp_ms, acc_x_g, acc_y_g, acc_z_g, gyro_*, sample_kind
    ├── <sid>_chest_consolidated.csv
    └── <sid>_thigh_right_consolidated.csv
```

Aturan pembacaan (`urfall_bench/datasets/fams.py`):
- Satu urutan = satu segmen di antara dua titik `label_timeline` dengan label ≠ 0. Label 0 (persiapan
  atau jeda) dibuang. Id urutan: `<sesi>_L<label>_<ulangan>`.
- Kalau `<sid>_integrity_report.json` tidak ada, dipakai `manifest.json → integrity.label_timeline`.
- IMU: hanya sampel asli (`sample_kind == 0`), diurutkan menurut `timestamp_ms`. Defaultnya IMU pinggang
  (`dataset.imu_role: waist`), sesuai paper yang memasang sensor di dekat pelvis.
- Kelompok CV = nama subjek dari `manifest.json`. Kalau satu orang punya dua sesi, keduanya masuk satu
  kelompok.
- Waktu video = `started_at_ms` kamera + waktu frame di video, jadi satu jam dengan `timestamp_ms` IMU.

**Label** ada di [`configs/datasets/fams.yaml`](configs/datasets/fams.yaml) (`classes`, `fall_classes`),
diambil dari Lampiran I formulir informed consent RSUI. Ada 38 label: **22 FALL** dan **16 ADL**. Di data
yang sudah direkam baru ada label 1–34; label tangga 35–38 belum ada.

| Label | Aktivitas | Jenis | | Label | Aktivitas | Jenis |
|---|---|---|---|---|---|---|
| 1 | Berdiri tegak sembari melakukan aktifitas | ADL | | 20 | Bangun dari matras | ADL |
| 2 | Jatuh ke depan | FALL | | 21 | Berjalan dan jatuh ke depan | FALL |
| 3 | Tidur di matras | ADL | | 22 | Berjalan dan jatuh ke samping kiri | FALL |
| 4 | Jatuh ke belakang | FALL | | 23 | Berjalan dan jatuh ke samping kanan | FALL |
| 5 | Near Fall | ADL | | 24 | Berjalan dan jatuh ke belakang | FALL |
| 6 | Jatuh ke depan dengan dorongan | FALL | | 25 | Jatuh ke depan dan bangun kembali | FALL |
| 7 | Berjalan | ADL | | 26 | Jatuh ke belakang dan bangun kembali | FALL |
| 8 | Jatuh ke belakang dengan dorongan | FALL | | 27 | Jatuh ke kiri dan bangun kembali | FALL |
| 9 | Merangkak | ADL | | 28 | Jatuh ke kanan dan bangun kembali | FALL |
| 10 | Jatuh ke kiri | FALL | | 29 | Berdiri, duduk, dan jatuh ke depan | FALL |
| 11 | Duduk di kursi secara perlahan | ADL | | 30 | Berdiri, duduk, dan jatuh ke kiri | FALL |
| 12 | Berlari kecil | ADL | | 31 | Berdiri, duduk, dan jatuh ke kanan | FALL |
| 13 | Duduk di atas matras | ADL | | 32 | Berdiri, duduk, dan jatuh ke belakang | FALL |
| 14 | Jatuh ke kanan | FALL | | 33 | Berjalan cepat | ADL |
| 15 | Melompat | ADL | | 34 | Duduk di kursi dengan cepat | ADL |
| 16 | Berdiri dari kursi dan jatuh ke depan | FALL | | 35 | Menaiki tangga perlahan | ADL |
| 17 | Berdiri dari kursi dan jatuh ke samping kiri | FALL | | 36 | Menuruni tangga perlahan | ADL |
| 18 | Berdiri dari kursi dan jatuh ke samping kanan | FALL | | 37 | Menaiki tangga dengan cepat | ADL |
| 19 | Duduk dan jatuh | FALL | | 38 | Menuruni tangga dengan cepat | ADL |

Kalau sebuah sesi berisi label yang tidak ada di `classes`, adapter berhenti dengan pesan galat.

---

## 4. Perintah

```
python -m urfall_bench <perintah> -d <dataset.yaml> [-b <benchmark.yaml>] [-s key=value ...]
```

| Perintah | Fungsi |
|---|---|
| `list` | Adapter terdaftar dan config dataset/benchmark yang tersedia |
| `download` | Unduh dataset (UR Fall). File yang sudah ada dilewati |
| `inspect` | Jumlah urutan dan kelompok CV, durasi, puncak SV_total dan % urutan terpicu per label (atau per jatuh/ADL). CSV ke `results/<dataset>/inspect/` |
| `extract-3d` | Skeleton 3D multi-kamera di sekitar tiap pemicu (FAMS, GPU). Hasil ke `cache/<dataset>/pose3d/` |
| `run` | **Benchmark**: threshold + mode pose (bila ada label postur) + mode kandidat (bila ada fitur pose) |
| `report --run <folder>` | Tulis ulang `report.md` dari CSV sebuah run |

Flag yang sering dipakai: `run --out results/x/y` untuk menentukan folder run (defaultnya
`results/<dataset>/<benchmark>_<waktu>`).

---

## 5. Konfigurasi

Config dibangun dari tiga lapis, berurutan:

1. [`configs/benchmarks/defaults.yaml`](configs/benchmarks/defaults.yaml), yang mendokumentasikan semua opsi;
2. file **benchmark** (`-b`, default `kwolek2014.yaml`);
3. file **dataset** (`-d`).

Setelah itu, sejumlah override `-s kunci.bertitik=nilai` (nilai diparse sebagai YAML). File dataset boleh
diawali `extends: lain.yaml` untuk mewarisi file dataset lain.

| File benchmark | Isi |
|---|---|
| `kwolek2014.yaml` | Protokol paper: 4 fitur pose, SVM linear + k-NN (k = 5), pemicu 3 g (default) |
| `p40.yaml` | Seperti di atas + fitur P40 |
| `kfold.yaml` | `GroupKFold` 10 fold alih-alih LOSO (untuk FAMS, supaya skema CV setara dengan UR Fall) |

| Kunci | Default | Arti |
|---|---|---|
| `trigger.threshold_g` | `3.0` | Ambang pemicu SV_total |
| `trigger.merge_ms` | `1000` | Sampel di atas ambang yang berjarak < ini digabung |
| `thresholds.uft_g` / `lft_g` | `3.52` / `0.41` | Ambang pembanding Bourke 2007 |
| `verify.features` | `paper` | `paper` = 4 fitur; `+P40` = tambah P40 |
| `verify.window_s` | `[0, 1]` | Frame yang dinilai: [t + a, t + b] detik setelah pemicu |
| `verify.min_lying_s` | `0.5` | Mode pose "pose saja": lama berbaring minimum |
| `verify.lying_fraction` | `0.5` | Mode pose "pose + acc": proporsi frame berbaring minimum di jendela |
| `models` | `SVM: {kernel: linear, C: 1}`, `kNN: {n_neighbors: 5}` | Parameter diteruskan ke scikit-learn |
| `evaluation.protocol` | `group-kfold` (UR Fall), `loso` (FAMS) | Pembagian CV per kelompok |
| `evaluation.folds` | `10` | Jumlah fold untuk `group-kfold` |
| `pose3d.*` | lihat `defaults.yaml` | Klip (`before_s`, `after_s`), kalibrasi (`calib_interval_s`, `height_m`), `rotate`, `fps`, `device`, `batch` |
| `dataset.include` | — | Hanya urutan yang id-nya cocok dengan salah satu pola glob |
| `dataset.imu_role` | `waist` | FAMS: IMU tahap akselerometer (`waist`, `chest`, `thigh_right`) |
| `dataset.pad_s` | `0` | FAMS: lebarkan tiap segmen label (detik) |

Contoh:

```bash
# pemicu 2.5 g dan jendela verifikasi 2 detik
python -m urfall_bench run -d configs/datasets/urfall.yaml -s trigger.threshold_g=2.5 -s "verify.window_s=[0, 2]"
# FAMS dengan IMU dada sebagai pemicu
python -m urfall_bench run -d configs/datasets/fams.yaml -s dataset.imu_role=chest
# FAMS, hanya urutan label 2 dan 12
python -m urfall_bench inspect -d configs/datasets/fams.yaml -s "dataset.include=['*_L02_*', '*_L12_*']"
# SVM RBF saja, tanpa k-NN
python -m urfall_bench run -d configs/datasets/fams.yaml -s "models={SVM: {kernel: rbf}}"
```

Model yang dikenal adalah `SVM` dan `kNN` (`urfall_bench/kwolek/models.py`). Model lain bisa ditambahkan
di file itu.

---

## 6. Menambah dataset baru

1. Salin [`urfall_bench/datasets/_template.py`](urfall_bench/datasets/_template.py) ke
   `urfall_bench/datasets/<nama>.py`. Implementasikan `load_sequences()`, yang mengembalikan daftar
   `Sequence` (`id`, `group`, `is_fall`, `t_acc` ms, `sv` g).
2. Opsional: `attach_pose()` untuk mengisi `t_frame` dan `F` (kolom = `self.features`). Set
   `has_posture = True` kalau ada label postur per frame (mode pose paper).
3. Daftarkan dengan `@register("<nama>")` dan import di `urfall_bench/datasets/__init__.py`.
4. Salin [`configs/datasets/_template.yaml`](configs/datasets/_template.yaml) ke
   `configs/datasets/<nama>.yaml`.
5. `python -m urfall_bench inspect -d configs/datasets/<nama>.yaml`, lalu `run`.

Dataset tanpa fitur pose tetap bisa dijalankan; yang keluar hanya metode threshold. Kalau dataset punya
video multi-kamera seperti FAMS, adapter cukup menyediakan `session_paths()` dan `load_session()` agar
`extract-3d` bisa dipakai.

---

## 7. Keluaran

```
results/<dataset>/<run>/
├── config.yaml                 # konfigurasi lengkap yang dipakai
├── run_info.json               # jumlah urutan/kelompok, fitur, protokol, jumlah kandidat
├── report.md                   # semua tabel (+ angka paper untuk dibandingkan)
├── table_pose_per_frame.csv    # klasifikasi pose berbaring per frame (Tabel 1 paper)    [mode pose]
├── table_pose.csv              # threshold + mode pose per urutan (Tabel 2 paper)       [mode pose]
├── table_candidate.csv         # threshold + mode kandidat per urutan (perbandingan antar dataset)
├── candidates.csv              # tiap kandidat: urutan, waktu pemicu, ada fitur?, label, prediksi per model
├── sequences.csv               # per urutan: kelompok, label, SV maks/min, jumlah pemicu, info dataset
└── per_label.csv               # per label aktivitas: n, median SV maks, % terpicu         [FAMS]

cache/<dataset>/
├── pose3d/                     # extract-3d
│   ├── candidates.csv          # session, sequence, trigger_ms, npz
│   └── <sid>/calibration.json, clip_<ms>.npz
└── work/<sid>/                 # zip yang diekstrak, skeleton 2D sementara, log.txt (boleh dihapus)
```

Isi `clip_<ms>.npz` (dari `triangulasi.py`):
- `t_ms` (T,): waktu, satu jam dengan IMU
- `X` (T, 12, 3): posisi sendi dalam meter; z = atas, lantai z = 0; NaN = tidak tertriangulasi
- `reproj_px`, `n_view` (T, 12)
- `kamera`, `joints`

Kolom metrik di semua tabel: `accuracy, precision, sensitivity, specificity, f1` (persen) dan
`TP, TN, FP, FN, n`. Kelas positif = jatuh (di Tabel 1: berbaring).

---

## 8. Hasil

### 8.1 UR Fall: replikasi paper

Benchmark `kwolek2014`, pemicu 3 g, jendela [0, 1] s, CV 10-fold per urutan. Format sel:
**acc / prec / sens / spec**.

**Tabel 1: klasifikasi pose berbaring per frame**

| | `urfall` (9 805 frame) | `urfall_paper` (7 800 frame) | Paper (612 sampel) |
|---|---|---|---|
| SVM | 99.17 / 98.18 / 98.39 / 99.42 | 98.97 / 97.01 / 97.66 / 99.29 | 99.67 / 100 / 99.05 / 100 |
| k-NN | 98.83 / 98.27 / 96.81 / 99.46 | 99.49 / 98.53 / 98.80 / 99.65 | 99.67 / 100 / 99.05 / 100 |

**Tabel 2: deteksi jatuh per urutan**

| Metode | `urfall` (30 + 40) | `urfall_paper` (30 + 30) | Paper (30 + 30) |
|---|---|---|---|
| Threshold UFT | 92.86 / 90.32 / 93.33 / 92.50 | 96.67 / 100 / 93.33 / 100 | 95.00 / 90.91 / 100 / 90.00 |
| Threshold LFT | 80.00 / 73.53 / 83.33 / 77.50 | 83.33 / 83.33 / 83.33 / 83.33 | 86.67 / 82.35 / 93.33 / 80.00 |
| Pemicu saja (> 3 g) | 94.29 / 90.62 / 96.67 / 92.50 | 98.33 / 100 / 96.67 / 100 | — |
| SVM pose saja | 65.71 / 57.89 / 73.33 / 60.00 | 76.67 / 78.57 / 73.33 / 80.00 | 90.00 / 83.30 / 100 / 80.00 |
| **SVM pose + acc** | **94.29** / 90.62 / 96.67 / 92.50 | **98.33** / 100 / 96.67 / 100 | **98.33** / 96.77 / 100 / 96.67 |
| k-NN pose + acc | 94.29 / 90.62 / 96.67 / 92.50 | 98.33 / 100 / 96.67 / 100 | — |
| SVM pose + acc (kandidat) | 91.43 / 90.00 / 90.00 / 92.50 | 98.33 / 100 / 96.67 / 100 | — |
| k-NN pose + acc (kandidat) | 94.29 / 90.62 / 96.67 / 92.50 | 98.33 / 100 / 96.67 / 100 | — |

Dengan setting 30 + 30, akurasi metode utama (**98.33**) sama dengan paper. Komposisi kesalahannya
berbeda:

| | Salah deteksi | Penyebab |
|---|---|---|
| Di sini | 1 jatuh terlewat (`fall-10`) | Puncak SV_total cuma 2.92 g, jadi tidak pernah terpicu |
| Paper | 1 ADL dianggap jatuh | — |

Dengan `-b configs/benchmarks/p40.yaml` hasilnya hampir sama. Kandidat SVM di `urfall` naik ke
95.71 / 93.55 / 96.67 / 95.00.

### 8.2 FAMS

Belum ada hasil lengkap (butuh `extract-3d` di semua sesi). Uji awal `inspect` dan metode threshold di
2 sesi (68 urutan) menunjukkan dua hal:
- Hampir semua urutan jatuh tembus 3 g.
- Ada dua ADL yang juga **selalu** tembus 3 g: **Berlari kecil (12)** dan **Melompat (15)**, dengan
  median SV maks 4.8 g dan 6.5 g.

Ini berbeda dengan UR Fall (lihat [§11](#11-catatan-metodologi)).

---

## 9. Detail metode

### 9.1 Tahap akselerometer (paper Bagian 2.3 dan 4)

`urfall_bench/kwolek/trigger.py`

- **SV_total(t) = √(Ax² + Ay² + Az²)** (Persamaan 1 paper). Nilainya ≈ 1 g saat diam.
- **Pemicu.** Kejadian SV_total > **3 g**. Sampel di atas ambang yang berjarak < 1 s digabung, dan waktu
  pemicu = puncak SV_total di kejadian itu. Satu urutan bisa punya beberapa pemicu.
- **UFT.** Jatuh kalau max SV_total > 3.52 g.
- **LFT.** Jatuh kalau min SV_total < 0.41 g (fase jatuh bebas).

  Angka UFT/LFT adalah ambang sensor batang tubuh dari Bourke 2007. Paper Kwolek tidak menuliskannya.

### 9.2 Fitur pose UR Fall (paper Bagian 5.3)

Paper memakai 4 fitur yang dihitung dari titik-titik orang di depth map:

| Fitur | Arti |
|---|---|
| `HeightWidthRatio` | h/w kotak pembatas orang |
| `HHmaxRatio` | tinggi sekarang dibagi tinggi orang saat berdiri |
| `D` | jarak centroid orang ke bidang lantai (lantai dari v-disparity + Hough + RANSAC) |
| `MaxStdXZ` | max(σx, σz): sebaran titik pada dua sumbu horizontal |

Halaman dataset juga menyediakan `P40` (proporsi titik < 40 cm dari lantai). P40 tidak dipakai di paper,
tapi tersedia lewat `p40.yaml`.

### 9.3 Fitur pose FAMS dari skeleton 3D

`urfall_bench/kwolek/features.py`. Fitur yang sama dihitung dari 12 sendi badan hasil triangulasi.
Koordinat dunia dalam meter, z = atas, lantai di z = 0.

| Fitur | Dari skeleton 3D |
|---|---|
| `HeightWidthRatio` | h = rentang z sendi; w = rentang sendi pada sumbu utama bidang horizontal (PCA xy) |
| `HHmaxRatio` | z sendi tertinggi / h_max. h_max = persentil 95 dari z sendi tertinggi pada semua frame milik subjek itu |
| `D` | z centroid sendi |
| `MaxStdXZ` | max(std x, std y) sendi |
| `P40` | proporsi sendi dengan z < 0.40 m |

Frame dengan kurang dari 6 sendi valid tidak dihitung. Bedanya dengan depth map:
- Titiknya 12 sendi, bukan ribuan piksel.
- Kepala dan wajah tidak ada, jadi h dan h_max diukur sampai bahu. Karena h_max juga sampai bahu,
  HHmaxRatio tetap sebanding.
- `D` dan `P40` bergantung pada skala kalibrasi (`pose3d.height_m`), sedangkan `HeightWidthRatio`,
  `HHmaxRatio` dan rasio antar-sumbu tidak.

Cara `extract-3d` menghasilkan skeleton ini (`urfall_bench/pose3d/extract.py`), per sesi:

1. **Ekstrak zip** ke `cache/<dataset>/work/<sid>/session/`.
2. **Kandidat.** Pemicu > 3 g per urutan. Fungsinya sama dengan yang dipakai `run`, jadi waktu kandidat
   pasti cocok.
3. **Kalibrasi kamera tanpa papan.**
   - `ekstrak_skeleton.py --interval-s 2` di seluruh sesi, lalu `triangulasi.py kalibrasi`.
   - Badan subjek dipakai sebagai papan kalibrasi: sendi yang terlihat di dua kamera menjadi pasangan
     titik.
   - Matriks esensial + PnP, lalu bundle adjustment (pose kamera, focal length, distorsi k1, titik 3D).
   - Arah atas = rata-rata arah pergelangan kaki → bahu. Lantai di z = 0. Skala dari `pose3d.height_m`
     (jarak bahu–pergelangan kaki ≈ 0.78 × tinggi badan).
4. **Skeleton 2D.**
   - Model: Keypoint R-CNN (torchvision, bobot COCO). 5 titik wajah dibuang, jadi tersisa 12 sendi badan.
   - Klip = [t − 1 s, t + 3 s] untuk semua frame dan semua kamera. Klip yang tumpang tindih digabung.
   - Tiap video hanya di-decode **sekali** per sesi (`--rentang`), lalu hasilnya dipotong per klip.
5. **Triangulasi** tiap klip (grid 15 fps) → `clip_<ms>.npz`.
6. **`candidates.csv`** ditulis bertahap per sesi. Kandidat yang gagal punya kolom `npz` kosong.

Aman diulang: kalibrasi, skeleton 2D dan klip yang sudah ada dilewati, dan ekstraksi 2D melanjutkan dari
checkpoint. Kalau daftar kandidat berubah (mis. `trigger.threshold_g` diganti), skeleton 2D klip dibuat
ulang di folder baru.

### 9.4 Mode pose (persis paper)

Paper melatih klasifier **pose berbaring vs tidak berbaring**, lalu memakainya setelah pemicu. Mode ini
butuh label postur per frame, jadi **hanya jalan di UR Fall** (`has_posture = True`).

1. Klasifier dilatih per frame (postur 1 vs −1; postur 0 diabaikan).
2. **Pose saja** (paper: "SVM – depth only"). Urutan jatuh kalau ada ≥ `min_lying_s` detik berturut-turut
   frame yang diklasifikasi berbaring, tanpa melihat akselerometer.
3. **Pose + acc** (paper: "SVM – depth + acc."). Untuk tiap pemicu, frame di jendela verifikasi
   diklasifikasi. Pemicu lolos kalau ≥ `lying_fraction` frame-nya berbaring.

### 9.5 Mode kandidat (untuk perbandingan)

FAMS tidak punya label postur per frame, jadi perbandingan UR Fall vs FAMS memakai mode ini di kedua
dataset:

1. Tiap pemicu = satu **kandidat**.
2. Fitur kandidat = median tiap fitur pose pada frame di jendela verifikasi.
3. Label kandidat = label urutannya (jatuh / ADL).
4. Urutan jatuh kalau ada kandidat yang diputuskan jatuh.
5. Kandidat tanpa fitur diputuskan **bukan jatuh**, dan jumlahnya dilaporkan di `report.md`.

Klasifier di sini menjalankan peran yang sama dengan paper, yaitu memverifikasi alarm akselerometer.
Bedanya, labelnya diambil dari urutan, bukan dari postur.

### 9.6 Validasi silang dan metrik

Semua klasifier dinilai **out-of-fold**: urutan yang dinilai tidak pernah ikut dilatih.

| Dataset | Kelompok | Skema default |
|---|---|---|
| UR Fall | urutan (tidak ada ID subjek) | `group-kfold`, 10 fold |
| FAMS | subjek | `loso` |

- Kalau data latih sebuah fold hanya berisi satu kelas, prediksi fold itu = kelas tersebut.
- Fitur distandardisasi (`StandardScaler`) di dalam pipeline, jadi statistiknya dihitung dari data latih
  saja.

Metrik (paper Bagian 7):
- sensitivity = TP / (TP + FN)
- specificity = TN / (TN + FP)
- precision = TP / (TP + FP)
- accuracy = (TP + TN) / n
- F1 = 2·P·S / (P + S)

---

## 10. Penyimpangan dari paper

| Hal | Paper | Di sini | Alasan |
|---|---|---|---|
| Data latih klasifier pose | Dataset terpisah, 612 sampel | Frame UR Fall sendiri (postur −1 / 1), out-of-fold per urutan | Dataset latih paper tidak dirilis |
| Fitur depth | Diekstrak sendiri dari depth map | Fitur per frame yang dirilis bersama UR Fall | Definisinya sama, tanpa mengulang segmentasi depth |
| Jumlah ADL | 30 (yang mana tidak disebut) | `urfall` = 40; `urfall_paper` = `adl-01..30` | Dilaporkan dua-duanya |
| Penghentian urutan | Sisa urutan dibuang setelah respons salah | Satu keputusan per urutan | Lebih sederhana, hasil tetap per urutan |
| Skala fitur | Tidak disebut (LIBSVM) | `StandardScaler` | Satuan fitur berbeda (mm vs rasio); disarankan panduan LIBSVM |
| SVM | LIBSVM linear | `sklearn.svm.SVC(kernel="linear", C=1)` | Sama-sama libsvm |
| Jendela verifikasi | Tidak disebut | [t, t + 1 s] | Banyak urutan jatuh UR Fall berakhir < 0.5 s setelah benturan |
| Aturan "pose saja" | Tidak dirinci | Berbaring ≥ 0.5 s berturut-turut | |
| Nilai UFT / LFT | Merujuk Bourke 2007, angkanya tidak ditulis | 3.52 g / 0.41 g (batang tubuh) | **Cek ulang ke paper Bourke** |
| Fitur pose FAMS | Depth map Kinect | Skeleton 3D 4 kamera | FAMS tidak punya kamera depth ([§9.3](#93-fitur-pose-fams-dari-skeleton-3d)) |
| Mode kandidat | — | Tambahan | FAMS tidak punya label postur per frame |

---

## 11. Catatan metodologi

- **Di UR Fall, tahap verifikasi hampir tidak berpengaruh.** "Pose + acc" sama persis dengan "pemicu
  saja". Hanya 3 ADL yang tembus 3 g (`adl-31`, `adl-33`, `adl-40`). Menurut label postur UR Fall sendiri,
  ketiganya memang berakhir berbaring (rebahan cepat), jadi menurut definisi Kwolek tidak bisa ditolak.
  Di 30 ADL pertama tidak ada yang terpicu, sehingga hasil 98.33 praktis ditentukan oleh akselerometer
  saja.
- **Di FAMS justru tahap verifikasi yang menentukan.** Berlari kecil dan melompat tembus 3 g, tapi
  orangnya tetap berdiri. Kemampuan fitur pose dari skeleton 3D menolak dua aktivitas ini adalah inti
  perbandingannya. `inspect` menampilkan % terpicu per label.
- **Yang dibandingkan bukan hanya datasetnya.** Sumber fitur pose juga berbeda: depth map 1 kamera di
  UR Fall, triangulasi 4 kamera di FAMS. Ini perlu ditulis eksplisit. Metode threshold (UFT/LFT/pemicu)
  adalah satu-satunya bagian yang benar-benar setara, karena sama-sama IMU di pinggang.
- **Unit evaluasi beda dari baseline UP-Fall.** Di sini keputusannya biner per urutan, bukan multikelas
  per jendela, jadi angkanya tidak bisa disandingkan langsung dengan F1 multikelas.
- **Cara pembagian CV memengaruhi angka.** UR Fall tidak punya ID subjek, jadi CV-nya per urutan dan
  subjek yang sama bisa muncul di data latih dan uji. FAMS memakai LOSO, yang lebih ketat. Untuk
  pembanding yang setara, jalankan juga FAMS dengan `-b configs/benchmarks/kfold.yaml`.
- **Segmen label FAMS** dicatat operator secara langsung. Panjangnya (3–30 s) berbeda dari durasi di
  protokol. Kalau benturan bisa jatuh di luar segmen karena label ditekan terlambat, coba
  `-s dataset.pad_s=1`.
- **Skala skeleton 3D** bergantung pada `pose3d.height_m`, yang sama untuk semua subjek. Fitur rasio tidak
  terpengaruh, sedangkan `D` dan `P40` terpengaruh.

---

## 12. Struktur proyek

```
fams_urfall_baseline/
├── README.md
├── SERVER_GUIDE.md         rsync + tmux untuk server GPU
├── requirements.txt
├── configs/
│   ├── benchmarks/         defaults.yaml · kwolek2014.yaml · p40.yaml · kfold.yaml
│   └── datasets/           urfall.yaml · urfall_paper.yaml · fams.yaml · _template.yaml
├── urfall_bench/
│   ├── cli.py              perintah (python -m urfall_bench …)
│   ├── config.py           muat / gabung YAML, override -s
│   ├── core.py             Sequence (satu urutan yang dinilai)
│   ├── datasets/           SATU FILE PER DATASET: urfall.py · fams.py (+ base.py registry, _template.py)
│   ├── kwolek/             trigger · features · models · evaluation · detection · runner · report
│   └── pose3d/             extract.py (extract-3d) · ekstrak_skeleton.py · skeleton_utils.py · triangulasi.py
├── scripts/run_all.sh      seluruh perbandingan UR Fall vs FAMS
├── tests/test_smoke.py     python -m pytest tests -q
├── data/      (git-ignored) dataset mentah
├── cache/     (git-ignored) skeleton 3D dan file sementara
└── results/   (git-ignored) hasil run
```

---

## 13. Troubleshooting

| Masalah | Solusi |
|---|---|
| `download` gagal di beberapa file | Jalankan ulang. File yang sudah ada dilewati |
| `KeyError: label N tidak ada di dataset.classes` | Tambahkan label itu di `configs/datasets/fams.yaml` (`classes`, dan `fall_classes` kalau jatuh) |
| `run` FAMS hanya mengeluarkan threshold | Belum ada `cache/fams/pose3d/candidates.csv`. Jalankan `extract-3d` dulu |
| Banyak kandidat "tanpa fitur pose" | Cek `cache/fams/work/<sid>/log.txt`. Biasanya kalibrasi sesi itu gagal (reprojection error besar) atau subjek tidak terlihat di ≥ 2 kamera |
| Kalibrasi gagal di satu sesi | Jalankan `triangulasi.py kalibrasi` secara manual dengan `--awal` dari sesi lain di ruangan yang sama, simpan sebagai `cache/fams/pose3d/<sid>/calibration.json`, lalu ulangi `extract-3d` (sesi lain dilewati) |
| Kamera keempat terdeteksi buruk | Sesuaikan `pose3d.rotate` di `fams.yaml` (90/180/270) |
| `ImportError` PyAV dan OpenCV bentrok | `extract-3d` sudah menjalankan `ekstrak_skeleton.py` (PyAV) dan `triangulasi.py` (OpenCV) di proses terpisah. Jangan import keduanya di satu skrip |
| Fold LOSO cuma punya satu kelas | Terjadi kalau subjek sedikit. Prediksinya = kelas itu. Tambah sesi atau pakai `kfold.yaml` |

---

## 14. Referensi

1. B. Kwolek, M. Kepski. Human fall detection on embedded platform using depth maps and wireless
   accelerometer. *Computer Methods and Programs in Biomedicine*, 117(3):489–501, 2014.
   doi:10.1016/j.cmpb.2014.09.005
2. A. K. Bourke, J. V. O'Brien, G. M. Lyons. Evaluation of a threshold-based tri-axial accelerometer fall
   detection algorithm. *Gait & Posture*, 26(2):194–199, 2007.
3. UR Fall Detection Dataset. <https://fenix.ur.edu.pl/mkepski/ds/uf.html> (CC BY-NC-SA 4.0).
4. L. Martínez-Villaseñor *et al.* UP-Fall Detection Dataset: A Multimodal Approach. *Sensors*,
   19(9):1988, 2019.
5. K. He, G. Gkioxari, P. Dollár, R. Girshick. Mask R-CNN. *ICCV*, 2017 (dasar Keypoint R-CNN di
   torchvision).
