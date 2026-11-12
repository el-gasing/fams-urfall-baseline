"""
Sesi FAMS (ekspor data-collector), zip atau folder, di ``dataset.root``:
  cameras.json                              session_id, kamera, started_at_ms
  data/manifest.json                        nama subjek (kelompok CV)
  data/<sid>_integrity_report.json          label_timeline (titik pergantian label), session_end_ms
  data/<sid>_<peran>_consolidated.csv       timestamp_ms, acc_x_g, acc_y_g, acc_z_g, ..., sample_kind
  videos/<sid>_camN_video_sync.webm

Satu Sequence = satu segmen label != 0 (label 0 = persiapan/jeda dibuang). Tahap akselerometer memakai
IMU ``dataset.imu_role`` (default waist, sesuai paper: sensor di dekat pelvis). Hanya sampel asli
(sample_kind == 0). Kelompok CV = nama subjek dari manifest. Label FALL/ADL = ``dataset.fall_classes``.
Fitur pose dibaca dari skeleton 3D di ``cache/<nama>/pose3d/`` (perintah ``extract-3d``).
"""
import glob
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

from ..core import Sequence
from ..kwolek import features as FT
from ..kwolek.trigger import sv_total
from .base import DatasetAdapter, register


class Session:
    """Akses seragam ke sesi FAMS dalam bentuk zip atau folder."""

    def __init__(self, path):
        self.path = str(path)
        if zipfile.is_zipfile(self.path):
            self.z = zipfile.ZipFile(self.path)
            names = [n for n in self.z.namelist() if not n.startswith("__MACOSX")]
            roots = [n[: -len("cameras.json")] for n in names if n.endswith("cameras.json")]
            self.root = min(roots, key=len) if roots else ""
            self._names = set(names)
        else:
            self.z, self.root = None, ""
        cams = json.load(self.open("cameras.json"))
        self.sid = str(cams["session_id"])
        self.cameras = cams["cameras"]

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

    def subject(self):
        return json.load(self.open("data/manifest.json")).get("subject") or self.sid

    def imu(self, role):
        d = pd.read_csv(self.open(f"data/{self.sid}_{role}_consolidated.csv"), comment="#")
        d = d[d.sample_kind == 0].sort_values("timestamp_ms")
        return d.timestamp_ms.to_numpy(float), d[["acc_x_g", "acc_y_g", "acc_z_g"]].to_numpy(float)

    def segments(self):
        tl, end = self.label_timeline()
        end = end or tl[-1]["timestamp_ms"]
        out, count = [], {}
        for i, e in enumerate(tl):
            lab = int(e["label_id"])
            if lab == 0:
                continue
            b = tl[i + 1]["timestamp_ms"] if i + 1 < len(tl) else end
            count[lab] = count.get(lab, 0) + 1
            out.append(dict(label=lab, rep=count[lab], a=float(e["timestamp_ms"]), b=float(b)))
        return out


@register("fams")
class FAMSAdapter(DatasetAdapter):

    def session_paths(self):
        if os.path.exists(os.path.join(self.root, "cameras.json")):
            return [str(self.root)]
        return sorted(q for q in glob.glob(os.path.join(self.root, "*"))
                      if zipfile.is_zipfile(q) or os.path.exists(os.path.join(q, "cameras.json")))

    def load_session(self, path):
        s = Session(path)
        subj = s.subject()
        t, A = s.imu(self.dcfg.get("imu_role", "waist"))
        sv = sv_total(A[:, 0], A[:, 1], A[:, 2])
        pad = float(self.dcfg.get("pad_s", 0.0)) * 1000
        seqs = []
        for g in s.segments():
            if self.classes and g["label"] not in self.classes:
                raise KeyError(f"{path}: label {g['label']} tidak ada di dataset.classes")
            m = (t >= g["a"] - pad) & (t < g["b"] + pad)
            if m.sum() < 2:
                continue
            seqs.append(Sequence(
                id=f"{s.sid}_L{g['label']:02d}_{g['rep']}", group=subj, is_fall=g["label"] in self.fall_classes,
                t_acc=t[m], sv=sv[m],
                info=dict(session=s.sid, subject=subj, label=g["label"], rep=g["rep"],
                          duration_s=round((g["b"] - g["a"]) / 1000, 2))))
        return s, seqs

    def load_sequences(self):
        seqs = []
        for p in self.session_paths():
            seqs += self.load_session(p)[1]
        return seqs

    def attach_pose(self, seqs):
        """Fitur pose dari skeleton 3D (``extract-3d``): cache/<nama>/pose3d/candidates.csv
        (session, sequence, trigger_ms, npz relatif ke folder itu)."""
        d3 = self.cache / "pose3d"
        names = self.features
        if not (d3 / "candidates.csv").exists():
            return names
        idx = pd.read_csv(d3 / "candidates.csv", dtype={"session": str})
        idx = idx[idx.npz.notna()]
        data = {}
        for _, r in idx.iterrows():
            f = d3 / r.npz
            if f.exists():
                d = np.load(f)
                data.setdefault(r.sequence, []).append((d["t_ms"].astype(float), d["X"].astype(float)))
        per_subj = {}
        for s in seqs:
            for _, X in data.get(s.id, []):
                per_subj.setdefault(s.group, []).append(FT.top_z(X))
        h_max = {k: np.nanpercentile(np.concatenate(v), 95) for k, v in per_subj.items()}
        cols = [FT.SKELETON_COLUMNS.index(f) for f in names]
        for s in seqs:
            if s.id not in data:
                continue
            t = np.concatenate([a for a, _ in data[s.id]])
            F = np.vstack([FT.skeleton_features(X, h_max[s.group]) for _, X in data[s.id]])[:, cols]
            o = np.argsort(t, kind="stable")
            s.t_frame, s.F = t[o], F[o]
            s.info["h_max_m"] = round(float(h_max[s.group]), 3)
        return names
