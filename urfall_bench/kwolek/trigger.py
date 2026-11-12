"""
Tahap akselerometer (Kwolek & Kepski 2014, Bagian 2.3 dan 4).

  SV_total(t) = sqrt(Ax^2 + Ay^2 + Az^2)        (Persamaan 1)
  SV_total > 3 g  -> "potential fall", lalu tahap verifikasi pose dijalankan.

Pembanding threshold murni (Tabel 2 paper, dari Bourke et al. 2007, Gait & Posture 26(2):194-199;
ambang sensor batang tubuh: UFT 3.52 g, LFT 0.41 g):
  UFT: jatuh bila puncak SV_total > UFT
  LFT: jatuh bila lembah SV_total < LFT (fase jatuh bebas)
Nilai UFT/LFT ada di configs/benchmarks/defaults.yaml.
"""
import numpy as np


def sv_total(ax, ay, az):
    return np.sqrt(np.asarray(ax, float) ** 2 + np.asarray(ay, float) ** 2 + np.asarray(az, float) ** 2)


def triggers(t_ms, sv, threshold_g=3.0, merge_ms=1000.0):
    """Waktu (ms) tiap kejadian SV_total > threshold_g. Sampel di atas ambang yang berjarak < merge_ms
    digabung jadi satu kejadian; waktunya = puncak SV_total di kejadian itu."""
    t_ms, sv = np.asarray(t_ms, float), np.asarray(sv, float)
    idx = np.flatnonzero(sv > threshold_g)
    if len(idx) == 0:
        return np.empty(0)
    cut = np.flatnonzero(np.diff(t_ms[idx]) > merge_ms) + 1
    return np.array([t_ms[g[np.argmax(sv[g])]] for g in np.split(idx, cut)])


def uft(sv, threshold_g):
    return bool(np.nanmax(sv) > threshold_g) if len(sv) else False


def lft(sv, threshold_g):
    return bool(np.nanmin(sv) < threshold_g) if len(sv) else False
