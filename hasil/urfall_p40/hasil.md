# Baseline UR Fall (Kwolek & Kepski 2014) di UR Fall
70 urutan (30 jatuh, 40 ADL). Fitur: HeightWidthRatio, HHmaxRatio, D, MaxStdXZ, P40. CV: 10-fold per urutan. Pemicu 3.0 g, jendela verifikasi [0.0, 1.0] s setelah pemicu.

## Tabel 1: klasifikasi pose berbaring per frame
Paper (dataset latih sendiri, 612 sampel, 10-fold): SVM dan k-NN akurasi 99.67, presisi 100, sensitivitas 99.05, spesifisitas 100.

| metode | accuracy | precision | sensitivity | specificity | f1 | TP | TN | FP | FN | n |
|---|---|---|---|---|---|---|---|---|---|---|
| Pose per frame (SVM) | 99.22 | 98.14 | 98.64 | 99.41 | 98.39 | 2321 | 7408 | 44 | 32 | 9805 |
| Pose per frame (kNN) | 98.74 | 98.23 | 96.47 | 99.45 | 97.34 | 2270 | 7411 | 41 | 83 | 9805 |

## Tabel 2: deteksi jatuh per urutan (mode pose, persis paper)
Paper (30 jatuh + 30 ADL): SVM depth saja 90.00/83.30/100/80.00, SVM depth + acc 98.33/96.77/100/96.67, UFT 95.00/90.91/100/90.00, LFT 86.67/82.35/93.33/80.00 (akurasi/presisi/sensitivitas/spesifisitas).

| metode | accuracy | precision | sensitivity | specificity | f1 | TP | TN | FP | FN | n |
|---|---|---|---|---|---|---|---|---|---|---|
| Threshold UFT (>3.52 g) | 92.86 | 90.32 | 93.33 | 92.5 | 91.8 | 28 | 37 | 3 | 2 | 70 |
| Threshold LFT (<0.41 g) | 80.0 | 73.53 | 83.33 | 77.5 | 78.12 | 25 | 31 | 9 | 5 | 70 |
| Pemicu saja (>3.0 g) | 94.29 | 90.62 | 96.67 | 92.5 | 93.55 | 29 | 37 | 3 | 1 | 70 |
| SVM fitur pose saja | 65.71 | 57.89 | 73.33 | 60.0 | 64.71 | 22 | 24 | 16 | 8 | 70 |
| SVM fitur pose + acc | 94.29 | 90.62 | 96.67 | 92.5 | 93.55 | 29 | 37 | 3 | 1 | 70 |
| kNN fitur pose saja | 67.14 | 59.46 | 73.33 | 62.5 | 65.67 | 22 | 25 | 15 | 8 | 70 |
| kNN fitur pose + acc | 94.29 | 90.62 | 96.67 | 92.5 | 93.55 | 29 | 37 | 3 | 1 | 70 |

## Mode kandidat (dipakai untuk perbandingan dengan FAMS)

| metode | accuracy | precision | sensitivity | specificity | f1 | TP | TN | FP | FN | n |
|---|---|---|---|---|---|---|---|---|---|---|
| Threshold UFT (>3.52 g) | 92.86 | 90.32 | 93.33 | 92.5 | 91.8 | 28 | 37 | 3 | 2 | 70 |
| Threshold LFT (<0.41 g) | 80.0 | 73.53 | 83.33 | 77.5 | 78.12 | 25 | 31 | 9 | 5 | 70 |
| Pemicu saja (>3.0 g) | 94.29 | 90.62 | 96.67 | 92.5 | 93.55 | 29 | 37 | 3 | 1 | 70 |
| SVM fitur pose + acc (kandidat) | 95.71 | 93.55 | 96.67 | 95.0 | 95.08 | 29 | 38 | 2 | 1 | 70 |
| kNN fitur pose + acc (kandidat) | 94.29 | 90.62 | 96.67 | 92.5 | 93.55 | 29 | 37 | 3 | 1 | 70 |

Kandidat: {"n_kandidat": 32, "n_kandidat_tanpa_fitur": 0}
