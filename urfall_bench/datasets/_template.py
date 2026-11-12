"""
Template adapter dataset baru. Salin ke ``urfall_bench/datasets/<nama>.py``, isi ``load_sequences``,
tambahkan import-nya di ``datasets/__init__.py``, lalu buat ``configs/datasets/<nama>.yaml``
dari ``configs/datasets/_template.yaml``.

Yang dibutuhkan algoritma per urutan:
  t_acc, sv     waktu (ms) dan SV_total (g) dari akselerometer di pinggang / batang tubuh
  is_fall       label urutan
  group         kelompok CV (subjek; atau id urutan bila subjek tidak diketahui)
Opsional (lewat ``attach_pose``): t_frame (ms, jam sama dengan t_acc) dan F (frame x fitur pose,
kolom = ``self.features``). Tanpa fitur pose hanya metode threshold yang dijalankan.
"""
import numpy as np

from ..core import Sequence
from .base import DatasetAdapter, register


@register("my_dataset")
class MyDatasetAdapter(DatasetAdapter):
    has_posture = False  # True bila ada label postur per frame (mode pose paper)

    def load_sequences(self):
        seqs = []
        # for f in sorted(self.root.glob("*.csv")):
        #     t, ax, ay, az = ...
        #     seqs.append(Sequence(id=f.stem, group=subject, is_fall=..., t_acc=t,
        #                          sv=np.sqrt(ax**2 + ay**2 + az**2)))
        return seqs

    def attach_pose(self, seqs):
        # for s in seqs: s.t_frame, s.F = ..., ...   # kolom F mengikuti self.features
        return []
