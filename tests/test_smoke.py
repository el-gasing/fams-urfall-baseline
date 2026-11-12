"""Cek end-to-end tanpa dataset asli:  python -m pytest tests -q

Data sintetis dalam format UR Fall: urutan jatuh punya lonjakan SV_total > 3 g lalu berbaring,
urutan ADL tidak. Juga beberapa ADL "lompat" yang tembus 3 g tapi tetap berdiri (kasus FAMS).
"""
import numpy as np
import pandas as pd
import pytest

from urfall_bench.config import load_config
from urfall_bench.datasets import get_adapter
from urfall_bench.kwolek import features, trigger
from urfall_bench.kwolek.evaluation import metrics
from urfall_bench.kwolek.runner import run


def _write_urfall(root, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("fall", "adl"):
        for i in range(1, n + 1):
            sid = f"{kind}-{i:02d}"
            t = np.arange(0, 4000, 10.0)
            sv = 1 + 0.05 * rng.standard_normal(len(t))
            jump = kind == "adl" and i <= 4
            if kind == "fall" or jump:
                sv[(t > 1990) & (t < 2030)] = 5.0
            acc = np.c_[t, sv, sv, np.zeros_like(t), np.zeros_like(t)]
            pd.DataFrame(acc).to_csv(root / f"{sid}-acc.csv", header=False, index=False)
            tf = np.arange(0, 4000, 33.0)
            pd.DataFrame(np.c_[np.arange(1, len(tf) + 1), tf, np.ones_like(tf)]).to_csv(
                root / f"{sid}-data.csv", header=False, index=False)
            for k, tt in enumerate(tf, 1):
                lying = kind == "fall" and tt > 2000
                hw = (0.4 if lying else 2.5) + 0.1 * rng.standard_normal()
                hh = (0.3 if lying else 1.0) + 0.05 * rng.standard_normal()
                d = (200 if lying else 1000) + 30 * rng.standard_normal()
                rows.append([kind + "s", sid, k, 1 if lying else -1, hw, 1.5, 0.6, 100, hh, 1700, d, 0.9 if lying else 0.05])
    df = pd.DataFrame(rows)
    for kind in ("falls", "adls"):
        df[df[0] == kind].iloc[:, 1:].to_csv(root / f"urfall-cam0-{kind}.csv", header=False, index=False)


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / "urfall"
    root.mkdir()
    _write_urfall(root)
    return load_config("configs/datasets/urfall.yaml", "configs/benchmarks/kwolek2014.yaml",
                       [f"dataset.root={root}", f"results_dir={tmp_path / 'results'}", "evaluation.folds=4"])


def test_trigger_and_sv():
    t = np.arange(0, 3000, 10.0)
    sv = np.ones_like(t); sv[100:103] = 4.0; sv[250] = 3.5
    assert np.allclose(trigger.sv_total([3], [4], [0]), [5])
    assert len(trigger.triggers(t, sv, 3.0, 1000)) == 2
    assert trigger.uft(sv, 3.52) and not trigger.lft(sv, 0.41)


def test_metrics():
    m = metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m["accuracy"] == 50 and m["sensitivity"] == 50 and m["specificity"] == 50 and m["TP"] == 1


def test_skeleton_features_standing_vs_lying():
    stand = np.zeros((1, 12, 3)); stand[0, :, 2] = np.linspace(0.05, 1.4, 12); stand[0, :, 0] = np.linspace(-.2, .2, 12)
    lie = np.zeros((1, 12, 3)); lie[0, :, 0] = np.linspace(0, 1.4, 12); lie[0, :, 2] = 0.1
    fs, fl = features.skeleton_features(stand, 1.4)[0], features.skeleton_features(lie, 1.4)[0]
    assert fs[0] > 1 > fl[0]            # HeightWidthRatio
    assert fs[1] > 0.9 and fl[1] < 0.2  # HHmaxRatio
    assert fl[4] == 1.0                 # P40


def test_run_end_to_end(cfg, tmp_path):
    ad = get_adapter(cfg)
    out = run(cfg, ad, tmp_path / "run", log=lambda *_: None)
    pose = pd.read_csv(out / "table_pose.csv").set_index("method")
    cand = pd.read_csv(out / "table_candidate.csv").set_index("method")
    # pemicu saja tertipu oleh 4 ADL "lompat"; verifikasi pose menolaknya
    assert pose.loc["Pemicu saja (>3.0 g)", "FP"] == 4
    assert pose.loc["SVM pose + acc", "FP"] == 0 and pose.loc["SVM pose + acc", "sensitivity"] == 100
    sv = cand.loc["SVM pose + acc (kandidat)"]
    assert sv["sensitivity"] == 100 and sv["FP"] < cand.loc["Pemicu saja (>3.0 g)", "FP"]
    assert (out / "report.md").exists()


def test_include_filter(cfg):
    cfg["dataset"]["include"] = ["fall-0[1-3]", "adl-01"]
    assert len(get_adapter(cfg).sequences()) == 4
