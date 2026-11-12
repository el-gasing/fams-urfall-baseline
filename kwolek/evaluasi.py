"""Klasifier dan metrik baseline UR Fall (Kwolek & Kepski 2014, Bagian 6-7)."""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def buat_model(nama):
    """SVM linear (paper memakai LIBSVM linear) dan k-NN 5 tetangga (pembanding di paper).
    Standardisasi fitur tidak disebut di paper; dipakai karena fitur berbeda satuan (lihat README)."""
    if nama == "SVM":
        return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    if nama == "kNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    raise ValueError(nama)


MODEL = ["SVM", "kNN"]


def metrik(y_true, y_pred):
    """Accuracy, precision, sensitivity, specificity (definisi Bagian 7 paper) + F1, dalam persen.
    Kelas positif = jatuh / berbaring."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int(np.sum(y_true & y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred)); fn = int(np.sum(y_true & ~y_pred))
    bagi = lambda a, b: 100.0 * a / b if b else np.nan
    prec, sens = bagi(tp, tp + fp), bagi(tp, tp + fn)
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and np.isfinite(prec) and np.isfinite(sens) else np.nan
    return dict(accuracy=bagi(tp + tn, len(y_true)), precision=prec, sensitivity=sens,
                specificity=bagi(tn, tn + fp), f1=f1, TP=tp, TN=tn, FP=fp, FN=fn, n=len(y_true))


def tabel(baris):
    """baris: list of (nama_metode, dict metrik) -> DataFrame dibulatkan."""
    df = pd.DataFrame([dict(metode=n, **m) for n, m in baris])
    num = ["accuracy", "precision", "sensitivity", "specificity", "f1"]
    df[num] = df[num].round(2)
    return df


def ke_markdown(df):
    kol = list(df.columns)
    garis = ["| " + " | ".join(kol) + " |", "|" + "---|" * len(kol)]
    for _, r in df.iterrows():
        garis.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(garis)
