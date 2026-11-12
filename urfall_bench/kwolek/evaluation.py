"""Pembagian CV berkelompok, prediksi out-of-fold, dan metrik (paper Bagian 7)."""
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from . import models


def splits(groups, protocol, folds=10):
    """protocol: 'loso' (leave-one-group-out) atau 'group-kfold' (GroupKFold, ``folds`` fold)."""
    groups = np.asarray(groups)
    if protocol == "loso":
        return list(LeaveOneGroupOut().split(groups, groups=groups))
    if protocol == "group-kfold":
        k = min(int(folds), len(np.unique(groups)))
        return list(GroupKFold(n_splits=k).split(groups, groups=groups))
    raise ValueError(f"evaluation.protocol harus 'loso' atau 'group-kfold', bukan {protocol!r}")


def oof_predict(X, y, groups, ev, name, params):
    """Prediksi out-of-fold: data yang dinilai tidak pernah ikut dilatih.
    Fold yang data latihnya cuma satu kelas -> tebak kelas itu."""
    pred = np.zeros(len(y), bool)
    for tr, te in splits(groups, ev["protocol"], ev.get("folds", 10)):
        if len(np.unique(y[tr])) < 2:
            pred[te] = bool(y[tr][0]) if len(tr) else False
            continue
        pred[te] = models.build(name, params).fit(X[tr], y[tr]).predict(X[te]).astype(bool)
    return pred


def metrics(y_true, y_pred):
    """Accuracy, precision, sensitivity, specificity (definisi paper) + F1, dalam persen.
    Kelas positif = jatuh (atau berbaring, untuk klasifikasi pose per frame)."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int(np.sum(y_true & y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred)); fn = int(np.sum(y_true & ~y_pred))
    div = lambda a, b: 100.0 * a / b if b else np.nan
    prec, sens = div(tp, tp + fp), div(tp, tp + fn)
    f1 = 2 * prec * sens / (prec + sens) if np.isfinite(prec) and np.isfinite(sens) and prec + sens else np.nan
    return dict(accuracy=div(tp + tn, len(y_true)), precision=prec, sensitivity=sens,
                specificity=div(tn, tn + fp), f1=f1, TP=tp, TN=tn, FP=fp, FN=fn, n=len(y_true))
