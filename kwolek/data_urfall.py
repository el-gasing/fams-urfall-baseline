"""
Pembaca UR Fall Detection Dataset (https://fenix.ur.edu.pl/mkepski/ds/uf.html).

File yang dipakai (unduh dengan scripts/download_urfall.sh):
  <seq>-acc.csv   waktu_ms, SV_total, Ax, Ay, Az (g)   -- SV_total = sqrt(Ax^2+Ay^2+Az^2), dicek di sini
  <seq>-data.csv  nomor_frame, waktu_ms, SV_total di frame itu (jam sama dengan acc.csv)
  urfall-cam0-falls.csv / urfall-cam0-adls.csv
                  seq, nomor_frame, postur (-1 tidak berbaring, 0 sedang jatuh, 1 berbaring),
                  HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40
UR Fall tidak memberi ID subjek per urutan, jadi kelompok CV = urutan itu sendiri.
"""
import os

import numpy as np
import pandas as pd

from .deteksi import Urutan

KOLOM_DEPTH = ["HeightWidthRatio", "MajorMinorRatio", "BoundingBoxOccupancy", "MaxStdXZ", "HHmaxRatio", "H", "D", "P40"]


def muat(root, fitur):
    fe = pd.concat([pd.read_csv(os.path.join(root, f"urfall-cam0-{k}.csv"), header=None,
                                names=["seq", "frame", "postur"] + KOLOM_DEPTH) for k in ("falls", "adls")])
    urutan = []
    for seq, d in fe.groupby("seq", sort=True):
        acc = pd.read_csv(os.path.join(root, f"{seq}-acc.csv"), header=None).to_numpy(float)
        sinkron = pd.read_csv(os.path.join(root, f"{seq}-data.csv"), header=None).to_numpy(float)
        t_of = dict(zip(sinkron[:, 0].astype(int), sinkron[:, 1]))
        d = d[d.frame.isin(t_of)].sort_values("frame")
        postur = d.postur.to_numpy(int)
        urutan.append(Urutan(
            id=seq, kelompok=seq, jatuh=seq.startswith("fall"),
            t_acc=acc[:, 0], sv=acc[:, 1],
            t_frame=d.frame.map(t_of).to_numpy(float), F=d[fitur].to_numpy(float),
            postur=np.where(postur == 1, 1, np.where(postur == -1, 0, -1)),  # postur 0 (sedang jatuh) diabaikan
        ))
    return urutan
