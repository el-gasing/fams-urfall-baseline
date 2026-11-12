"""
Pembaca sesi FAMS (ekspor data-collector-IMU-mobile), zip atau folder:
  cameras.json                              started_at_ms tiap kamera
  data/manifest.json                        nama subjek
  data/<sid>_integrity_report.json          label_timeline (titik pergantian label, otoritatif)
  data/<sid>_waist_consolidated.csv         timestamp_ms, acc_x_g, acc_y_g, acc_z_g, ..., sample_kind
  videos/<sid>_camN_video_sync.webm

Satu Urutan = satu segmen label != 0 (label 0 = persiapan/jeda dibuang). Sensor yang dipakai
untuk tahap akselerometer = IMU pinggang (waist), sesuai paper (x-IMU di dekat pelvis).
Hanya sampel asli (sample_kind == 0) yang dipakai. Kelompok CV = nama subjek dari manifest.
"""
import glob
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import yaml

from . import fitur as FT
from .sinyal import sv_total
from .deteksi import Urutan


class Sesi:
    """Akses seragam ke sesi FAMS dalam bentuk zip atau folder."""

    def __init__(self, path):
        self.path = path
        if zipfile.is_zipfile(path):
            self.z = zipfile.ZipFile(path)
            names = [n for n in self.z.namelist() if not n.startswith("__MACOSX")]
            akar = [n[: -len("cameras.json")] for n in names if n.endswith("cameras.json")]
            self.root = min(akar, key=len) if akar else ""
            self._names = set(names)
        else:
            self.z, self.root = None, ""
        cams = json.load(self.open("cameras.json"))
        self.sid = str(cams["session_id"])
        self.kamera = cams["cameras"]

    def open(self, rel):
        if self.z:
            return io.BytesIO(self.z.read(self.root + rel))
        return open(os.path.join(self.path, rel), "rb")

    def exists(self, rel):
        return (self.root + rel) in self._names if self.z else os.path.exists(os.path.join(self.path, rel))

    def label_timeline(self):
        rel = f"data/{self.sid}_integrity_report.json"
        rep = json.load(self.open(rel)) if self.exists(rel) else json.load(self.open("data/manifest.json"))["integrity"]
        return rep["label_timeline"], rep.get("session_end_ms")

    def subjek(self):
        return json.load(self.open("data/manifest.json")).get("subject") or self.sid

    def imu(self, peran="waist"):
        d = pd.read_csv(self.open(f"data/{self.sid}_{peran}_consolidated.csv"), comment="#")
        d = d[d.sample_kind == 0].sort_values("timestamp_ms")
        return d.timestamp_ms.to_numpy(float), d[["acc_x_g", "acc_y_g", "acc_z_g"]].to_numpy(float)


def baca_label(path):
    lab = yaml.safe_load(open(path))["labels"]
    return {int(k): v for k, v in lab.items()}


def segmen(sesi):
    tl, akhir = sesi.label_timeline()
    akhir = akhir or tl[-1]["timestamp_ms"]
    out, ke = [], {}
    for i, e in enumerate(tl):
        lab = int(e["label_id"])
        if lab == 0:
            continue
        b = tl[i + 1]["timestamp_ms"] if i + 1 < len(tl) else akhir
        ke[lab] = ke.get(lab, 0) + 1
        out.append(dict(label=lab, ulang=ke[lab], a=float(e["timestamp_ms"]), b=float(b)))
    return out


def daftar_sesi(paths):
    """Argumen bisa zip, folder sesi, atau folder berisi banyak zip/folder sesi."""
    hasil = []
    for p in paths:
        if os.path.isdir(p) and not os.path.exists(os.path.join(p, "cameras.json")):
            hasil += sorted(q for q in glob.glob(os.path.join(p, "*"))
                            if zipfile.is_zipfile(q) or os.path.exists(os.path.join(q, "cameras.json")))
        else:
            hasil.append(p)
    return hasil


def muat(paths, label_path, peran="waist", pad_s=0.0):
    """Urutan FAMS (tanpa fitur pose). Fitur pose ditempel dengan tempel_skeleton()."""
    label = baca_label(label_path)
    urutan, info = [], []
    for p in daftar_sesi(paths):
        s = Sesi(p)
        subj = s.subjek()
        t, A = s.imu(peran)
        sv = sv_total(A[:, 0], A[:, 1], A[:, 2])
        segs = segmen(s)
        for g in segs:
            if g["label"] not in label:
                raise KeyError(f"{p}: label {g['label']} tidak ada di {label_path}")
            m = (t >= g["a"] - pad_s * 1000) & (t < g["b"] + pad_s * 1000)
            if m.sum() < 2:
                continue
            urutan.append(Urutan(
                id=f"{s.sid}_L{g['label']:02d}_{g['ulang']}", kelompok=subj,
                jatuh=label[g["label"]]["jenis"] == "FALL", t_acc=t[m], sv=sv[m],
                info=dict(sesi=s.sid, subjek=subj, label=g["label"], ulang=g["ulang"], a_ms=g["a"], b_ms=g["b"],
                          durasi_s=round((g["b"] - g["a"]) / 1000, 2))))
        info.append(dict(sesi=s.sid, file=os.path.basename(os.path.normpath(p)), subjek=subj, segmen=len(segs)))
    return urutan, pd.DataFrame(info)


def tempel_skeleton(urutan, folder_3d, fitur):
    """Tempel fitur pose dari skeleton 3D (keluaran fams3d/jalankan.py) ke tiap urutan.
    folder_3d/kandidat.csv: sesi, urutan, t_pemicu_ms, npz (relatif ke folder_3d)."""
    idx = pd.read_csv(os.path.join(folder_3d, "kandidat.csv"), dtype={"sesi": str})
    idx = idx[idx.npz.notna()]
    data = {}
    for _, r in idx.iterrows():
        f = os.path.join(folder_3d, r.npz)
        if os.path.exists(f):
            d = np.load(f)
            data.setdefault(r.urutan, []).append((d["t_ms"].astype(float), d["X"].astype(float)))
    # h_max per subjek: persentil 95 dari sendi tertinggi di semua frame subjek itu
    per_subj = {}
    for u in urutan:
        for _, X in data.get(u.id, []):
            per_subj.setdefault(u.kelompok, []).append(FT.puncak_z(X))
    h_max = {k: np.nanpercentile(np.concatenate(v), 95) for k, v in per_subj.items()}
    kol = [FT.KOLOM_SKELETON.index(f) for f in fitur]
    for u in urutan:
        if u.id not in data:
            u.t_frame, u.F = np.empty(0), np.empty((0, len(fitur)))
            continue
        t = np.concatenate([a for a, _ in data[u.id]])
        F = np.vstack([FT.fitur_skeleton(X, h_max[u.kelompok]) for _, X in data[u.id]])[:, kol]
        o = np.argsort(t, kind="stable")
        u.t_frame, u.F = t[o], F[o]
    return h_max
