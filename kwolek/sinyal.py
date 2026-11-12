"""
Bagian akselerometer dari baseline UR Fall.

Kwolek & Kepski 2014, Bagian 2.3 dan 4:
  SV_total(t) = sqrt(Ax^2 + Ay^2 + Az^2)                     (Persamaan 1)
  SV_total > 3 g  -> "potential fall", lalu tahap verifikasi pose dijalankan.

Pembanding threshold murni (Tabel 2 paper, dari Bourke et al. 2007, Gait & Posture 26(2)):
  UFT (upper fall threshold): jatuh bila puncak SV_total > UFT
  LFT (lower fall threshold): jatuh bila lembah SV_total < LFT (fase jatuh bebas)
Nilai default UFT/LFT di bawah adalah ambang sensor di batang tubuh (trunk) dari Bourke 2007.
Paper Kwolek tidak menuliskan angkanya, jadi angka ini perlu dicek ulang ke paper Bourke.
"""
import numpy as np

AMBANG_PEMICU_G = 3.0   # Kwolek 2014 Bagian 4
UFT_G = 3.52            # Bourke 2007, trunk (cek ulang)
LFT_G = 0.41            # Bourke 2007, trunk (cek ulang)


def sv_total(ax, ay, az):
    return np.sqrt(np.asarray(ax, float) ** 2 + np.asarray(ay, float) ** 2 + np.asarray(az, float) ** 2)


def pemicu(t_ms, sv, ambang=AMBANG_PEMICU_G, jeda_ms=1000.0):
    """Waktu (ms) tiap kejadian SV_total > ambang. Sampel di atas ambang yang berjarak < jeda_ms
    digabung jadi satu kejadian; waktunya = puncak SV_total di kejadian itu."""
    t_ms, sv = np.asarray(t_ms, float), np.asarray(sv, float)
    idx = np.flatnonzero(sv > ambang)
    if len(idx) == 0:
        return np.empty(0)
    pecah = np.flatnonzero(np.diff(t_ms[idx]) > jeda_ms) + 1
    return np.array([t_ms[g[np.argmax(sv[g])]] for g in np.split(idx, pecah)])


def uft(sv, ambang=UFT_G):
    return bool(np.nanmax(sv) > ambang) if len(sv) else False


def lft(sv, ambang=LFT_G):
    return bool(np.nanmin(sv) < ambang) if len(sv) else False
