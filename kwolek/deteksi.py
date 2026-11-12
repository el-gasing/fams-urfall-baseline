"""
Prosedur deteksi jatuh per urutan (sequence), dipakai sama untuk UR Fall dan FAMS.

Satu Urutan = satu percobaan: satu file fall-XX/adl-XX di UR Fall, satu segmen label != 0 di FAMS.
Paper menilai per urutan (Tabel 2: 30 urutan jatuh + 30 urutan ADL), jadi semua metrik di sini juga
per urutan, kecuali eksperimen klasifikasi pose per frame (Tabel 1).

Metode:
  UFT / LFT        threshold murni pada SV_total (Bourke 2007), pembanding di Tabel 2.
  Pemicu 3g        hanya tahap pertama Kwolek (SV_total > 3 g), tanpa verifikasi.
  mode "pose"      (persis paper, butuh label postur per frame -> hanya UR Fall)
                   klasifier pose berbaring vs tidak berbaring dilatih per frame, lalu
                   - "fitur pose saja" : jatuh bila ada >= min_berbaring_s detik berturut-turut berbaring
                   - "fitur pose + acc": jatuh bila setelah pemicu 3g, mayoritas frame di
                                         [t + jendela_a, t + jendela_b] diklasifikasi berbaring
  mode "kandidat"  (bisa untuk kedua dataset, dipakai untuk perbandingan UR Fall vs FAMS)
                   tiap pemicu 3g = satu kandidat; fitur kandidat = median fitur pose per frame di
                   [t + jendela_a, t + jendela_b]; klasifier memutuskan kandidat jatuh atau bukan
                   (label = urutan itu jatuh); urutan jatuh bila ada kandidat yang diputuskan jatuh.
Semua klasifier dinilai out-of-fold dengan pembagian per kelompok (urutan/subjek), jadi urutan yang dinilai
tidak pernah dilihat saat latih.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from . import sinyal
from .evaluasi import MODEL, buat_model, metrik


@dataclass
class Urutan:
    id: str
    kelompok: str            # untuk pembagian CV: urutan (UR Fall) atau subjek (FAMS)
    jatuh: bool
    t_acc: np.ndarray        # ms
    sv: np.ndarray           # g
    t_frame: np.ndarray = field(default_factory=lambda: np.empty(0))    # ms, jam yang sama dengan t_acc
    F: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))     # (frame, fitur), boleh NaN
    postur: np.ndarray = None  # per frame: 1 berbaring, 0 tidak, -1 diabaikan (hanya UR Fall)
    info: dict = field(default_factory=dict)


def pembagi(kelompok, skema):
    kelompok = np.asarray(kelompok)
    if skema == "loso":
        return list(LeaveOneGroupOut().split(kelompok, groups=kelompok))
    k = min(int(skema.split("-")[0]), len(np.unique(kelompok)))  # "10-fold"
    return list(GroupKFold(n_splits=k).split(kelompok, groups=kelompok))


def prediksi_oof(X, y, kelompok, skema, nama_model):
    """Prediksi out-of-fold; fold yang data latihnya cuma satu kelas -> tebak kelas itu."""
    pred = np.zeros(len(y), bool)
    for tr, te in pembagi(kelompok, skema):
        if len(np.unique(y[tr])) < 2:
            pred[te] = bool(y[tr][0]) if len(tr) else False
            continue
        m = buat_model(nama_model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te]).astype(bool)
    return pred


# ------------------------------------------------------------------------------------------------
def metode_threshold(urutan, uft_g, lft_g, pemicu_g):
    y = [u.jatuh for u in urutan]
    return [
        (f"Threshold UFT (>{uft_g} g)", metrik(y, [sinyal.uft(u.sv, uft_g) for u in urutan])),
        (f"Threshold LFT (<{lft_g} g)", metrik(y, [sinyal.lft(u.sv, lft_g) for u in urutan])),
        (f"Pemicu saja (>{pemicu_g} g)", metrik(y, [len(sinyal.pemicu(u.t_acc, u.sv, pemicu_g)) > 0 for u in urutan])),
    ]


def _jendela(u, t, a, b):
    return (u.t_frame >= t + a * 1000) & (u.t_frame <= t + b * 1000)


def _run_terpanjang_s(t_frame, berbaring):
    best, mulai = 0.0, None
    for i, v in enumerate(berbaring):
        if v and mulai is None:
            mulai = i
        if (not v or i == len(berbaring) - 1) and mulai is not None:
            akhir = i if v else i - 1
            best = max(best, (t_frame[akhir] - t_frame[mulai]) / 1000)
            mulai = None
    return best


def metode_pose(urutan, skema, pemicu_g, a, b, min_berbaring_s, model=MODEL):
    """Mode 'pose' (persis paper). Hanya urutan yang punya label postur."""
    X, y, g, asal = [], [], [], []
    for i, u in enumerate(urutan):
        ok = (u.postur >= 0) & np.all(np.isfinite(u.F), axis=1)
        X.append(u.F[ok]); y.append(u.postur[ok] == 1); g += [u.kelompok] * ok.sum(); asal.append((i, ok))
    X, y, g = np.vstack(X), np.concatenate(y), np.array(g)
    hasil, label_y = [], [u.jatuh for u in urutan]
    for nm in model:
        pf = prediksi_oof(X, y, g, skema, nm)
        hasil.append((f"Pose per frame ({nm})", metrik(y, pf)))
        # kembalikan prediksi ke tiap urutan (frame tanpa prediksi = tidak berbaring)
        k, berb = 0, []
        for i, ok in asal:
            p = np.zeros(len(ok), bool); n = ok.sum(); p[ok] = pf[k:k + n]; k += n; berb.append(p)
        pose_saja = [_run_terpanjang_s(u.t_frame, p) >= min_berbaring_s for u, p in zip(urutan, berb)]
        pose_acc = []
        for u, p in zip(urutan, berb):
            putus = False
            for t in sinyal.pemicu(u.t_acc, u.sv, pemicu_g):
                w = _jendela(u, t, a, b)
                putus |= bool(w.any() and p[w].mean() >= 0.5)
            pose_acc.append(putus)
        hasil.append((f"{nm} fitur pose saja", metrik(label_y, pose_saja)))
        hasil.append((f"{nm} fitur pose + acc", metrik(label_y, pose_acc)))
    return hasil


def kandidat(urutan, pemicu_g, a, b):
    """Daftar kandidat: (indeks urutan, waktu pemicu, vektor fitur median atau NaN)."""
    out = []
    for i, u in enumerate(urutan):
        for t in sinyal.pemicu(u.t_acc, u.sv, pemicu_g):
            w = _jendela(u, t, a, b) if len(u.t_frame) else np.zeros(0, bool)
            with np.errstate(all="ignore"):
                v = np.nanmedian(u.F[w], axis=0) if w.any() else np.full(u.F.shape[1] if u.F.ndim == 2 else 0, np.nan)
            out.append((i, t, v))
    return out


def metode_kandidat(urutan, skema, pemicu_g, a, b, model=MODEL):
    """Mode 'kandidat'. Kandidat tanpa fitur (tidak ada frame/skeleton) diputuskan bukan jatuh."""
    kd = kandidat(urutan, pemicu_g, a, b)
    label_y = [u.jatuh for u in urutan]
    info = dict(n_kandidat=len(kd), n_kandidat_tanpa_fitur=sum(not np.all(np.isfinite(v)) for _, _, v in kd))
    if not kd:
        return [], info, []
    idx = np.array([i for i, _, _ in kd])
    X = np.array([v for _, _, v in kd]); ok = np.all(np.isfinite(X), axis=1)
    y = np.array([urutan[i].jatuh for i in idx]); g = np.array([urutan[i].kelompok for i in idx])
    hasil, rinci = [], []
    for nm in model:
        p = np.zeros(len(kd), bool)
        if ok.sum() and len(np.unique(g[ok])) > 1:
            p[ok] = prediksi_oof(X[ok], y[ok], g[ok], skema, nm)
        per_urutan = np.zeros(len(urutan), bool)
        for i, pi in zip(idx, p):
            per_urutan[i] |= pi
        hasil.append((f"{nm} fitur pose + acc (kandidat)", metrik(label_y, per_urutan)))
        rinci += [dict(model=nm, urutan=urutan[i].id, t_pemicu_ms=t, ada_fitur=bool(o), jatuh=bool(urutan[i].jatuh),
                       prediksi=bool(pi)) for (i, t, _), o, pi in zip(kd, ok, p)]
    return hasil, info, rinci
