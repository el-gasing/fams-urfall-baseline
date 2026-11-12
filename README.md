# FAMS vs UR Fall: baseline Kwolek & Kepski (2014)

Perbandingan dataset **FAMS** dengan **UR Fall Detection Dataset** memakai algoritma baseline UR Fall:

> B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps and wireless
> accelerometer*, Computer Methods and Programs in Biomedicine 117 (2014) 489–501.

Kode yang sama dijalankan di kedua dataset. Perbedaannya hanya di pembaca data (`kwolek/data_*.py`) dan
sumber fitur pose: depth map Kinect di UR Fall, skeleton 3D multi-kamera di FAMS.

## Algoritma (paper Bagian 3–7)

1. **Tahap akselerometer.** SV_total = √(Ax² + Ay² + Az²) dari IMU di pinggang. Kalau SV_total > **3 g**,
   ada "potential fall".
2. **Tahap verifikasi pose.** Setelah pemicu, fitur orang dihitung lalu diklasifikasi:
   - `HeightWidthRatio`: h/w, rasio tinggi dan lebar orang
   - `HHmaxRatio`: h/h_max, tinggi sekarang dibagi tinggi saat berdiri
   - `D`: jarak centroid orang ke lantai
   - `MaxStdXZ`: max(σx, σz), sebaran titik di dua sumbu horizontal

   Klasifiernya **SVM linear** (LIBSVM di paper), dengan **k-NN (k = 5)** sebagai pembanding.
3. **Pembanding threshold murni** (Tabel 2 paper, dari Bourke et al. 2007):
   - UFT: jatuh kalau puncak SV_total > 3.52 g
   - LFT: jatuh kalau lembah SV_total < 0.41 g
4. **Evaluasi per urutan**: accuracy, precision, sensitivity, specificity (F1 ditambahkan).

## Dua mode verifikasi

| mode | apa | dataset |
|---|---|---|
| `pose` | Persis paper. Klasifier pose *berbaring vs tidak* dilatih per frame dari label postur, lalu (a) "pose saja": jatuh kalau berbaring ≥ 0.5 s; (b) "pose + acc": jatuh kalau mayoritas frame di [t, t+1 s] setelah pemicu berbaring | UR Fall saja (FAMS tidak punya label postur per frame) |
| `kandidat` | Tiap pemicu 3 g = satu kandidat. Fitur kandidat = median fitur pose di [t, t+1 s]. Klasifier memutuskan kandidat jatuh atau bukan (label = urutan itu jatuh) | **UR Fall dan FAMS**. Mode ini yang dipakai untuk perbandingan |

Semua klasifier dinilai *out-of-fold*: urutan yang dinilai tidak pernah ikut dilatih.
- UR Fall tidak punya ID subjek, jadi CV-nya 10-fold per urutan.
- FAMS memakai **LOSO** (leave-one-subject-out).

## Data

**UR Fall**, ~110 MB, lisensi CC BY-NC-SA 4.0:
```bash
bash scripts/download_urfall.sh data/urfall
```
Yang dipakai: `*-acc.csv` (SV_total), `*-data.csv` (waktu frame), `urfall-cam0-{falls,adls}.csv` (8 fitur
depth per frame dan label postur). Ada 30 urutan jatuh dan 40 ADL. Paper memakai 30 + 30.

**FAMS**: zip ekspor sesi (atau folder hasil ekstraknya) di `data/fams/`. Per sesi yang dipakai:
- `cameras.json`
- `data/manifest.json` (nama subjek → kelompok LOSO)
- `data/<sid>_integrity_report.json` (`label_timeline`)
- `data/<sid>_waist_consolidated.csv` (hanya `sample_kind == 0`)
- `videos/`

Satu urutan = satu segmen label ≠ 0. Label 0 (persiapan/jeda) dibuang. Jenis FALL/ADL tiap label ada di
[`labels_fams.yaml`](labels_fams.yaml), diambil dari Lampiran I formulir informed consent RSUI:
22 label FALL dan 16 ADL. Di data yang sudah direkam baru ada label 1–34.

## Menjalankan

```bash
pip install -r requirements.txt

# UR Fall: replikasi Tabel 1-2 + mode kandidat (beberapa detik)
python jalankan_urfall.py --data data/urfall --out hasil/urfall

# FAMS, threshold saja (tanpa video, beberapa detik)
python jalankan_fams.py data/fams --out hasil/fams_threshold

# FAMS, skeleton 3D di sekitar tiap pemicu (server GPU, lama), lalu verifikasi pose
python fams3d/jalankan.py data/fams --out hasil/fams_3d --kerja kerja/fams --device cuda
python jalankan_fams.py data/fams --skeleton3d hasil/fams_3d --out hasil/fams

# atau semuanya sekaligus
bash scripts/jalankan_semua.sh data/fams -- --device cuda
```

Opsi yang sering dipakai (kedua runner):
- `--fitur +P40`: tambah fitur P40 (proporsi titik < 40 cm dari lantai, ada di dataset UR Fall)
- `--jendela A B`: jendela verifikasi setelah pemicu
- `--pemicu-g`, `--uft-g`, `--lft-g`

Khusus FAMS:
- `--peran chest|thigh_right`: IMU lain untuk tahap akselerometer
- `--skema 10-fold`
- `--pad-s`: lebarkan segmen label

Keluaran tiap run ada di `<out>/`:
- `hasil.md`: semua tabel
- `tabel_*.csv`
- `kandidat.csv`: keputusan per kandidat
- FAMS: `urutan.csv` dan `per_label.csv` (puncak SV dan persen terpicu per label)

## Skeleton 3D FAMS (`fams3d/`)

`fams3d/jalankan.py` per sesi:
1. Ekstrak zip.
2. Hitung kandidat. Fungsi pemicunya sama dengan runner, jadi waktu kandidat cocok.
3. Kalibrasi kamera tanpa papan:
   - `ekstrak_skeleton.py --interval-s 2` di seluruh sesi
   - `triangulasi.py kalibrasi`
   - badan subjek dipakai sebagai papan, bundle adjustment, lantai di z = 0
   - skala dari `--tinggi-m`
4. Skeleton 2D di [t − 1 s, t + 3 s] tiap kandidat, untuk semua frame dan semua kamera. Tiap video hanya
   di-decode sekali (`--rentang`), lalu triangulasi per klip.
5. Tulis `kandidat.csv` yang dibaca `jalankan_fams.py --skeleton3d`.

Aman diulang: langkah yang sudah selesai dilewati. `ekstrak_skeleton.py`, `skeleton_utils.py` dan
`triangulasi.py` adalah salinan pipeline skeleton 3D proyek ini. Satu-satunya tambahan adalah opsi
`--rentang` di `ekstrak_skeleton.py`.

**Fitur Kwolek dari skeleton 3D** ([`kwolek/fitur.py`](kwolek/fitur.py)). Titik orang = 12 sendi badan
(tanpa wajah), meter, z = atas.
- h = rentang z sendi
- w = rentang sendi di sumbu utama bidang horizontal (PCA xy)
- h_max = persentil 95 sendi tertinggi per subjek
- D = z centroid
- MaxStdXZ = max(std x, std y)
- P40 = proporsi sendi dengan z < 0.4 m

## Hasil UR Fall (data lengkap, 70 urutan)

Fitur paper (4). Pemicu 3 g, jendela [0, 1] s, 10-fold per urutan:

| metode | acc | prec | sens | spec | paper (acc/prec/sens/spec) |
|---|---|---|---|---|---|
| Threshold UFT (> 3.52 g) | 92.86 | 90.32 | 93.33 | 92.50 | 95.00 / 90.91 / 100 / 90.00 |
| Threshold LFT (< 0.41 g) | 80.00 | 73.53 | 83.33 | 77.50 | 86.67 / 82.35 / 93.33 / 80.00 |
| SVM pose saja | 65.71 | 57.89 | 73.33 | 60.00 | 90.00 / 83.30 / 100 / 80.00 |
| **SVM pose + acc** | **94.29** | 90.62 | 96.67 | 92.50 | **98.33** / 96.77 / 100 / 96.67 |
| SVM pose + acc (kandidat) | 91.43 | 90.00 | 90.00 | 92.50 | — |
| kNN pose + acc (kandidat) | 94.29 | 90.62 | 96.67 | 92.50 | — |

Klasifikasi pose per frame (Tabel 1): SVM akurasi 99.17 (paper 99.67), kNN 98.83.
Tabel lengkap ada di `hasil/urfall/hasil.md`.

## Referensi: UP-Fall-Algorithm (Karina Maharani)

Repo [KarinaMaharani/UP-Fall-Algorithm](https://github.com/KarinaMaharani/UP-Fall-Algorithm) (commit
`d4a6e64`, 11 Nov 2026) menjalankan **algoritma baseline UP-Fall** (Martínez-Villaseñor et al., Sensors 2019)
di UP-Fall, UR Fall dan FAMS. Folder ini melengkapinya dari arah sebaliknya: **algoritma baseline UR Fall**
(Kwolek & Kepski 2014), yang tidak ada di repo tersebut, dijalankan di UR Fall dan FAMS.

| hal | UP-Fall-Algorithm | folder ini |
|---|---|---|
| Algoritma | ARC UP-Fall: jendela 1/2/3 s, 18 fitur, seleksi fitur, RF/SVM/MLP/kNN. Ditambah CNN dan skeleton (Koffi 2024) | Kwolek 2014: pemicu SV_total > 3 g, lalu SVM linear / k-NN pada fitur pose |
| Unit evaluasi | Jendela (multikelas dan biner) | Urutan (biner jatuh vs ADL), seperti Tabel 2 paper UR Fall |
| Label FAMS | `configs/datasets/ra_pak_edy.yaml` | `labels_fams.yaml`. Isinya **sama** (38 nama, 22 label FALL), keduanya dari Lampiran I formulir RSUI |
| Daftar file UR Fall | `upfall_bench/datasets/urfall.py` (`download`) | `scripts/download_urfall.sh`, daftar dan URL yang sama |
| Pembacaan sesi FAMS | `upfall_bench/datasets/imu_session.py`: segmen label ≠ 0, `sample_kind == 0`, waktu dari `timestamp_ms` | Aturan yang sama (`kwolek/data_fams.py`) |
| Skeleton FAMS | 2D (Keypoint R-CNN / MediaPipe) | 3D, triangulasi multi-kamera |

Karena label, segmentasi dan sumber data UR Fall-nya sama, hasil kedua repo bisa disandingkan langsung.
Contohnya: task biner UP-Fall di repo Karina vs baseline Kwolek di sini, di dataset yang sama.

## Penyimpangan dari paper

| hal | paper | di sini |
|---|---|---|
| Data latih klasifier pose | Dataset terpisah, 612 sampel | Frame UR Fall sendiri (label postur −1 / 1), out-of-fold per urutan |
| Fitur depth | Diekstrak sendiri dari depth map | Fitur per frame yang dirilis bersama UR Fall |
| Jumlah ADL | 30 | Semua 40 |
| Penghentian urutan | Urutan dihentikan setelah respons salah | Tidak; satu keputusan per urutan |
| Skala fitur | Tidak disebut | `StandardScaler` (satuan fitur berbeda: mm, rasio) |
| Jendela verifikasi | Tidak disebut | [t, t + 1 s]. Banyak urutan jatuh UR Fall berakhir < 0.5 s setelah benturan |
| Nilai UFT/LFT | Tidak ditulis, merujuk Bourke 2007 | 3.52 g / 0.41 g (trunk, Bourke 2007). **Cek ulang ke paper Bourke** |
| Fitur pose FAMS | Depth map Kinect | Skeleton 3D multi-kamera (lihat atas) |

Catatan lain:
- "Pose saja" jauh di bawah paper. Penyebab utamanya: ADL UR Fall berisi berbaring di sofa (posturnya
  sama dengan setelah jatuh), dan sebagian urutan jatuh berhenti hanya sesaat setelah benturan.
- Di FAMS, **Berlari kecil (12)** dan **Melompat (15)** juga tembus 3 g. Tahap verifikasi pose yang harus
  menolak kedua aktivitas itu. Lihat `per_label.csv`.
- Segmen label FAMS dicatat operator secara langsung, dan lamanya (3–30 s) berbeda dari durasi di
  protokol.
