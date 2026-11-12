"""Kelas dasar adapter dan registry.

Untuk dataset baru cukup tulis SATU subclass ``DatasetAdapter`` yang mengimplementasikan
``load_sequences()`` (dan ``attach_pose()`` bila ada fitur pose), lalu daftarkan dengan
``@register("nama")``. Lihat ``_template.py``. Semua isi ``dataset:`` di YAML tersedia sebagai ``self.dcfg``.
"""
from __future__ import annotations

import fnmatch

from ..config import cache_dir, resolve
from ..kwolek.features import feature_names

_REGISTRY: dict[str, type["DatasetAdapter"]] = {}


def register(name: str):
    def deco(cls):
        cls.adapter_name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(cfg: dict) -> "DatasetAdapter":
    key = cfg["dataset"]["adapter"]
    if key not in _REGISTRY:
        raise KeyError(f"adapter dataset '{key}' tidak dikenal. Terdaftar: {available()}")
    return _REGISTRY[key](cfg)


class DatasetAdapter:
    """Mengubah file mentah menjadi daftar ``Sequence``.

    Subclass WAJIB mengimplementasikan ``load_sequences``. BOLEH meng-override ``attach_pose``,
    ``download`` dan ``has_posture`` (True bila dataset punya label postur per frame -> mode pose paper).
    """
    adapter_name = "base"
    has_posture = False

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dcfg = cfg["dataset"]
        self.name = self.dcfg["name"]
        self.root = resolve(self.dcfg.get("root", f"data/{self.name}"))
        self.cache = cache_dir(cfg)
        self.classes = {int(k): v for k, v in (self.dcfg.get("classes") or {}).items()}
        self.fall_classes = set(int(x) for x in self.dcfg.get("fall_classes") or [])

    @property
    def features(self) -> list[str]:
        return feature_names(self.cfg["verify"]["features"])

    # --- wajib -------------------------------------------------------------------------------
    def load_sequences(self):
        raise NotImplementedError

    # --- opsional ----------------------------------------------------------------------------
    def attach_pose(self, seqs) -> list[str]:
        """Isi ``t_frame`` dan ``F`` (dan ``posture``) tiap urutan. Kembalikan nama kolom fitur."""
        return []

    def download(self):
        raise NotImplementedError(f"{self.name}: download otomatis tidak tersedia, siapkan data di {self.root}")

    # --- umum --------------------------------------------------------------------------------
    def sequences(self):
        seqs = self.load_sequences()
        pats = self.dcfg.get("include")
        if pats:
            seqs = [s for s in seqs if any(fnmatch.fnmatch(s.id, p) for p in pats)]
        if not seqs:
            raise RuntimeError(f"{self.name}: tidak ada urutan di {self.root} (cek dataset.root / dataset.include)")
        return seqs
