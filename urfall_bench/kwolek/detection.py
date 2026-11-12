"""
Metode deteksi per urutan, dipakai sama untuk semua dataset.

  threshold   UFT / LFT (Bourke 2007) dan "pemicu saja" (tahap 1 Kwolek tanpa verifikasi).
  mode pose   (persis paper; butuh label postur per frame -> hanya UR Fall)
              klasifier berbaring vs tidak berbaring dilatih per frame, lalu
              - "pose saja" : jatuh bila >= verify.min_lying_s detik berturut-turut berbaring
              - "pose + acc": jatuh bila setelah pemicu, >= verify.lying_fraction frame di
                              [t + window_s[0], t + window_s[1]] diklasifikasi berbaring
  mode kandidat (untuk perbandingan antar dataset)
              tiap pemicu = satu kandidat; fitur = median fitur pose per frame di jendela itu;
              label = urutan itu jatuh; urutan jatuh bila ada kandidat yang diputuskan jatuh.
              Kandidat tanpa fitur diputuskan bukan jatuh.
Semua klasifier dinilai out-of-fold dengan pembagian per kelompok (evaluation.protocol).
"""
import numpy as np

from . import trigger
from .evaluation import metrics, oof_predict


def _trig(s, cfg):
    return trigger.triggers(s.t_acc, s.sv, cfg["trigger"]["threshold_g"], cfg["trigger"]["merge_ms"])


def _window(s, t, cfg):
    a, b = cfg["verify"]["window_s"]
    return (s.t_frame >= t + a * 1000) & (s.t_frame <= t + b * 1000)


def threshold_methods(seqs, cfg):
    y = [s.is_fall for s in seqs]
    u, l, p = cfg["thresholds"]["uft_g"], cfg["thresholds"]["lft_g"], cfg["trigger"]["threshold_g"]
    return [
        (f"Threshold UFT (>{u} g)", metrics(y, [trigger.uft(s.sv, u) for s in seqs])),
        (f"Threshold LFT (<{l} g)", metrics(y, [trigger.lft(s.sv, l) for s in seqs])),
        (f"Pemicu saja (>{p} g)", metrics(y, [len(_trig(s, cfg)) > 0 for s in seqs])),
    ]


def _longest_run_s(t_frame, lying):
    best, start = 0.0, None
    for i, v in enumerate(lying):
        if v and start is None:
            start = i
        if (not v or i == len(lying) - 1) and start is not None:
            end = i if v else i - 1
            best = max(best, (t_frame[end] - t_frame[start]) / 1000)
            start = None
    return best


def pose_methods(seqs, cfg):
    """Mode pose (persis paper). Kembalikan (baris per frame, baris per urutan)."""
    X, y, g, where = [], [], [], []
    for s in seqs:
        ok = (s.posture >= 0) & np.all(np.isfinite(s.F), axis=1)
        X.append(s.F[ok]); y.append(s.posture[ok] == 1); g += [s.group] * int(ok.sum()); where.append(ok)
    X, y, g = np.vstack(X), np.concatenate(y), np.array(g)
    labels = [s.is_fall for s in seqs]
    per_frame, per_seq = [], []
    v = cfg["verify"]
    for name, params in cfg["models"].items():
        pf = oof_predict(X, y, g, cfg["evaluation"], name, params)
        per_frame.append((f"Pose per frame ({name})", metrics(y, pf)))
        k, lying = 0, []
        for ok in where:
            p = np.zeros(len(ok), bool); n = int(ok.sum()); p[ok] = pf[k:k + n]; k += n; lying.append(p)
        only = [_longest_run_s(s.t_frame, p) >= v["min_lying_s"] for s, p in zip(seqs, lying)]
        acc = []
        for s, p in zip(seqs, lying):
            hit = False
            for t in _trig(s, cfg):
                w = _window(s, t, cfg)
                hit |= bool(w.any() and p[w].mean() >= v["lying_fraction"])
            acc.append(hit)
        per_seq.append((f"{name} pose saja", metrics(labels, only)))
        per_seq.append((f"{name} pose + acc", metrics(labels, acc)))
    return per_frame, per_seq


def candidates(seqs, cfg):
    """[(indeks urutan, waktu pemicu, vektor fitur median atau NaN)]"""
    out = []
    nf = max((s.F.shape[1] for s in seqs if s.F.ndim == 2), default=0)  # urutan tanpa fitur: F (0, 0)
    for i, s in enumerate(seqs):
        for t in _trig(s, cfg):
            w = _window(s, t, cfg) if len(s.t_frame) else np.zeros(0, bool)
            with np.errstate(all="ignore"):
                v = np.nanmedian(s.F[w], axis=0) if w.any() else np.full(nf, np.nan)
            out.append((i, t, v))
    return out


def candidate_methods(seqs, cfg):
    """Mode kandidat. Kembalikan (baris, info, rincian per kandidat)."""
    cd = candidates(seqs, cfg)
    labels = [s.is_fall for s in seqs]
    info = dict(n_candidates=len(cd), n_candidates_without_features=sum(not np.all(np.isfinite(v)) for _, _, v in cd))
    if not cd:
        return [], info, []
    idx = np.array([i for i, _, _ in cd])
    X = np.array([v for _, _, v in cd])
    ok = np.all(np.isfinite(X), axis=1) & (X.shape[1] > 0)
    y = np.array([seqs[i].is_fall for i in idx]); g = np.array([seqs[i].group for i in idx])
    rows, detail = [], []
    for name, params in cfg["models"].items():
        p = np.zeros(len(cd), bool)
        if ok.sum() and len(np.unique(g[ok])) > 1:
            p[ok] = oof_predict(X[ok], y[ok], g[ok], cfg["evaluation"], name, params)
        per_seq = np.zeros(len(seqs), bool)
        for i, pi in zip(idx, p):
            per_seq[i] |= pi
        rows.append((f"{name} pose + acc (kandidat)", metrics(labels, per_seq)))
        detail += [dict(model=name, sequence=seqs[i].id, trigger_ms=t, has_features=bool(o),
                        is_fall=bool(seqs[i].is_fall), predicted=bool(pi))
                   for (i, t, _), o, pi in zip(cd, ok, p)]
    return rows, info, detail
