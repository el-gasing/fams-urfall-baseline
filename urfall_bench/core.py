"""Struktur data bersama: satu ``Sequence`` = satu percobaan (urutan) yang dinilai jatuh / ADL.

UR Fall: satu file ``fall-XX`` / ``adl-XX``. FAMS: satu segmen label != 0 di ``label_timeline``.
Paper Kwolek & Kepski menilai per urutan (Tabel 2: 30 urutan jatuh + 30 urutan ADL), jadi semua metrik
deteksi di paket ini juga per urutan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sequence:
    id: str
    group: str                 # kelompok CV: urutan itu sendiri (UR Fall) atau subjek (FAMS)
    is_fall: bool
    t_acc: np.ndarray          # ms
    sv: np.ndarray             # SV_total (g)
    t_frame: np.ndarray = field(default_factory=lambda: np.empty(0))       # ms, jam yang sama dengan t_acc
    F: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))        # (frame, fitur pose), boleh NaN
    posture: np.ndarray | None = None  # per frame: 1 berbaring, 0 tidak, -1 diabaikan (hanya UR Fall)
    info: dict = field(default_factory=dict)

    @property
    def has_features(self) -> bool:
        return self.F.ndim == 2 and len(self.F) > 0
