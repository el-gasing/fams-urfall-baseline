"""
UR Fall Detection Dataset (Kwolek & Kepski 2014), https://fenix.ur.edu.pl/mkepski/ds/uf.html, CC BY-NC-SA 4.0.

File di ``dataset.root`` (``python -m urfall_bench download -d configs/datasets/urfall.yaml``):
  <seq>-acc.csv   waktu_ms, SV_total, Ax, Az, Ay (g; urutan kolom sesuai halaman dataset, hanya SV_total dipakai)
  <seq>-data.csv  nomor_frame, waktu_ms, SV_total interpolasi di frame itu (jam sama dengan acc.csv)
  urfall-cam0-falls.csv / urfall-cam0-adls.csv
                  seq, nomor_frame, postur (-1 tidak berbaring, 0 sedang jatuh, 1 berbaring),
                  HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40
  <seq>-cam0.mp4  video (opsional, tidak dipakai)
UR Fall tidak memberi ID subjek per urutan, jadi kelompok CV = urutan itu sendiri.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

from ..core import Sequence
from .base import DatasetAdapter, register

BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data"
DEPTH_COLUMNS = ["HeightWidthRatio", "MajorMinorRatio", "BoundingBoxOccupancy", "MaxStdXZ", "HHmaxRatio", "H", "D",
                 "P40"]


def file_list(video=True, cam1=False):
    files = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
    for i in range(1, 31):
        files += [f"fall-{i:02d}-acc.csv", f"fall-{i:02d}-data.csv"] + ([f"fall-{i:02d}-cam0.mp4"] if video else [])
        files += [f"fall-{i:02d}-cam1.mp4"] if cam1 else []
    for i in range(1, 41):
        files += [f"adl-{i:02d}-acc.csv", f"adl-{i:02d}-data.csv"] + ([f"adl-{i:02d}-cam0.mp4"] if video else [])
    return files


@register("urfall")
class URFallAdapter(DatasetAdapter):
    has_posture = True

    def load_sequences(self):
        seqs = []
        for f in sorted(os.listdir(self.root)):
            if not f.endswith("-acc.csv"):
                continue
            sid = f[: -len("-acc.csv")]
            acc = pd.read_csv(self.root / f, header=None).to_numpy(float)
            seqs.append(Sequence(id=sid, group=sid, is_fall=sid.startswith("fall"), t_acc=acc[:, 0], sv=acc[:, 1]))
        return seqs

    def attach_pose(self, seqs):
        names = self.features
        fe = pd.concat([pd.read_csv(self.root / f"urfall-cam0-{k}.csv", header=None,
                                    names=["seq", "frame", "posture"] + DEPTH_COLUMNS) for k in ("falls", "adls")])
        by_seq = dict(tuple(fe.groupby("seq")))
        for s in seqs:
            d = by_seq.get(s.id)
            if d is None:
                continue
            sync = pd.read_csv(self.root / f"{s.id}-data.csv", header=None).to_numpy(float)
            t_of = dict(zip(sync[:, 0].astype(int), sync[:, 1]))
            d = d[d.frame.isin(t_of)].sort_values("frame")
            p = d.posture.to_numpy(int)
            s.t_frame = d.frame.map(t_of).to_numpy(float)
            s.F = d[names].to_numpy(float)
            s.posture = np.where(p == 1, 1, np.where(p == -1, 0, -1))  # postur 0 (sedang jatuh) diabaikan
        return names

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        files = file_list(video=self.dcfg.get("download_video", True), cam1=self.dcfg.get("download_cam1", False))
        failed = []
        for k, f in enumerate(files, 1):
            dst = self.root / f
            if dst.exists() and dst.stat().st_size > 0:
                continue
            print(f"[{k}/{len(files)}] {f}", flush=True)
            try:
                urllib.request.urlretrieve(f"{BASE_URL}/{f}", str(dst) + ".part")
                os.replace(str(dst) + ".part", dst)
            except Exception as e:  # noqa: BLE001
                failed.append(f)
                print(f"  gagal: {e}")
        print(f"selesai: {len(files) - len(failed)}/{len(files)} file di {self.root}"
              + (f"; gagal {len(failed)}, jalankan ulang" if failed else ""))
