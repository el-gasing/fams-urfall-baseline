"""Klasifier verifikasi (paper Bagian 6): SVM linear (LIBSVM) dan k-NN 5 tetangga sebagai pembanding.

Diatur lewat ``models:`` di config, mis. ``models: {SVM: {kernel: linear, C: 1.0}, kNN: {n_neighbors: 5}}``.
Fitur distandardisasi di dalam pipeline (tidak disebut di paper; fitur berbeda satuan, lihat README §9).
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_KELAS = {"SVM": SVC, "kNN": KNeighborsClassifier}


def build(name, params):
    if name not in _KELAS:
        raise KeyError(f"model '{name}' tidak dikenal. Pilihan: {sorted(_KELAS)}")
    return make_pipeline(StandardScaler(), _KELAS[name](**(params or {})))
