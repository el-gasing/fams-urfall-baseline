"""
Fitur pose Kwolek & Kepski 2014 (Bagian 5.3), dari depth map (UR Fall) atau dari skeleton 3D (FAMS).

Paper (4 fitur, dihitung dari titik-titik orang di depth map):
  HeightWidthRatio  h/w      tinggi : lebar kotak pembatas orang
  HHmaxRatio        h/h_max  tinggi orang sekarang : tinggi orang saat berdiri
  D                          jarak centroid orang ke lantai
  MaxStdXZ                   max(sigma_x, sigma_z): sebaran titik pada dua sumbu horizontal
Tambahan dari halaman dataset UR Fall (tidak dipakai di paper, ``verify.features: +P40``):
  P40                        proporsi titik orang yang berada < 40 cm dari lantai

UR Fall menyediakan fitur ini per frame (urfall-cam0-*.csv). Untuk FAMS fitur yang sama dihitung dari
skeleton 3D (pose3d/), koordinat dunia z = atas, lantai z = 0, satuan meter:
  titik orang   = sendi yang berhasil ditriangulasi (12 sendi badan, tanpa wajah)
  h             = rentang z sendi;  w = rentang sendi pada sumbu utama bidang horizontal (PCA xy)
  h_max         = persentil 95 dari z sendi tertinggi, per subjek dari semua frame miliknya
  D             = z centroid sendi
  MaxStdXZ      = max(std x, std y) sendi
  P40           = proporsi sendi dengan z < 0.40 m
"""
import numpy as np

PAPER = ["HeightWidthRatio", "HHmaxRatio", "D", "MaxStdXZ"]
EXTRA = ["P40"]
SKELETON_COLUMNS = ["HeightWidthRatio", "HHmaxRatio", "D", "MaxStdXZ", "P40"]
MIN_JOINTS = 6  # frame dengan sendi valid lebih sedikit dari ini tidak dihitung


def feature_names(choice):
    """'paper' -> 4 fitur paper; '+P40' -> paper + P40."""
    if choice not in ("paper", "+P40"):
        raise ValueError(f"verify.features harus 'paper' atau '+P40', bukan {choice!r}")
    return PAPER + (EXTRA if choice == "+P40" else [])


def top_z(X):
    """z sendi tertinggi tiap frame (untuk h_max). X: (T, J, 3)."""
    z = np.where(np.isfinite(X[..., 2]), X[..., 2], -np.inf).max(axis=1)
    return np.where(np.isfinite(z), z, np.nan)


def skeleton_features(X, h_max):
    """X: (T, J, 3) meter, z = atas, NaN = sendi tidak ada. Kembalikan (T, 5) dengan kolom
    SKELETON_COLUMNS; NaN untuk frame dengan sendi < MIN_JOINTS."""
    out = np.full((X.shape[0], 5), np.nan)
    for f in range(X.shape[0]):
        P = X[f][np.all(np.isfinite(X[f]), axis=1)]
        if len(P) < MIN_JOINTS:
            continue
        z = P[:, 2]
        xy = P[:, :2] - P[:, :2].mean(0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        proj = xy @ vt[0]
        h = z.max() - z.min()
        w = max(proj.max() - proj.min(), 1e-3)
        out[f] = [h / w, z.max() / h_max, z.mean(), max(P[:, 0].std(), P[:, 1].std()), np.mean(z < 0.40)]
    return out
