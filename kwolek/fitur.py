"""
Fitur pose Kwolek & Kepski 2014 (Bagian 5.3), dari depth map (UR Fall) atau dari skeleton 3D (FAMS).

Paper (4 fitur, dihitung dari titik-titik orang di depth map):
  HeightWidthRatio  h/w      tinggi : lebar kotak pembatas orang
  HHmaxRatio        h/h_max  tinggi orang sekarang : tinggi orang saat berdiri
  D                          jarak centroid orang ke lantai
  MaxStdXZ                   max(sigma_x, sigma_z): sebaran titik di sekitar centroid pada dua sumbu
                             horizontal (di koordinat Kinect y = atas, jadi x dan z = horizontal)
Tambahan dari halaman dataset UR Fall (tidak dipakai di paper, opsional --fitur +P40):
  P40                        proporsi titik orang yang berada < 40 cm dari lantai

UR Fall menyediakan fitur ini per frame (urfall-cam0-*.csv). Untuk FAMS fitur yang sama dihitung dari
skeleton 3D hasil triangulasi (fams3d/), koordinat dunia z = atas, lantai di z = 0, satuan meter:
  titik orang   = sendi yang berhasil ditriangulasi (12 sendi badan, tanpa wajah)
  h             = rentang z sendi;  w = rentang sendi pada sumbu utama bidang horizontal (PCA xy)
  h_max         = persentil 95 dari z sendi tertinggi, dihitung per subjek dari semua frame miliknya
  D             = z centroid sendi
  MaxStdXZ      = max(std x, std y) sendi
  P40           = proporsi sendi dengan z < 0.40 m
Bedanya dengan depth map: titiknya 12 sendi, bukan ribuan piksel; wajah/kepala tidak ada, jadi h dan
h_max diukur sampai bahu. Karena h_max juga sampai bahu, HHmaxRatio tetap sebanding.
"""
import numpy as np

FITUR_PAPER = ["HeightWidthRatio", "HHmaxRatio", "D", "MaxStdXZ"]
FITUR_TAMBAHAN = ["P40"]
MIN_SENDI = 6  # frame dengan sendi valid lebih sedikit dari ini tidak dihitung


def daftar_fitur(pilihan):
    """'paper' -> 4 fitur paper; '+P40' -> paper + P40."""
    return FITUR_PAPER + (FITUR_TAMBAHAN if pilihan == "+P40" else [])


def puncak_z(X):
    """z sendi tertinggi tiap frame (untuk h_max). X: (T, J, 3)."""
    z = np.where(np.isfinite(X[..., 2]), X[..., 2], -np.inf).max(axis=1)
    return np.where(np.isfinite(z), z, np.nan)


def fitur_skeleton(X, h_max):
    """X: (T, J, 3) meter, z = atas, NaN = sendi tidak ada. Kembalikan (T, 5) dengan kolom
    HeightWidthRatio, HHmaxRatio, D, MaxStdXZ, P40; NaN untuk frame dengan sendi < MIN_SENDI."""
    T = X.shape[0]
    out = np.full((T, 5), np.nan)
    for f in range(T):
        P = X[f][np.all(np.isfinite(X[f]), axis=1)]
        if len(P) < MIN_SENDI:
            continue
        z = P[:, 2]
        xy = P[:, :2] - P[:, :2].mean(0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        proj = xy @ vt[0]
        h = z.max() - z.min()
        w = max(proj.max() - proj.min(), 1e-3)
        out[f] = [h / w, z.max() / h_max, z.mean(), max(P[:, 0].std(), P[:, 1].std()), np.mean(z < 0.40)]
    return out


KOLOM_SKELETON = ["HeightWidthRatio", "HHmaxRatio", "D", "MaxStdXZ", "P40"]
