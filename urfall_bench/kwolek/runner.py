"""Satu run benchmark: threshold + mode pose (bila ada label postur) + mode kandidat (bila ada fitur pose).

Keluaran di ``<out>/``:
  config.yaml                konfigurasi lengkap yang dipakai
  run_info.json              ringkasan data dan kandidat
  table_pose_per_frame.csv   klasifikasi pose berbaring per frame (Tabel 1 paper)       [mode pose]
  table_pose.csv             threshold + mode pose per urutan (Tabel 2 paper)          [mode pose]
  table_candidate.csv        threshold + mode kandidat per urutan (untuk perbandingan)
  candidates.csv             keputusan tiap kandidat
  sequences.csv              per urutan: label, SV maks/min, jumlah pemicu, info dataset
  per_label.csv              per label aktivitas (bila dataset punya label aktivitas)
  report.md                  semua tabel
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from . import detection, report, trigger


def table(rows):
    df = pd.DataFrame([dict(method=n, **m) for n, m in rows])
    if len(df):
        num = ["accuracy", "precision", "sensitivity", "specificity", "f1"]
        df[num] = df[num].round(2)
    return df


def run(cfg, adapter, out: Path, log=print):
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False, allow_unicode=True)
    seqs = adapter.sequences()
    n_fall = sum(s.is_fall for s in seqs)
    groups = len({s.group for s in seqs})
    log(f"{adapter.name}: {len(seqs)} urutan ({n_fall} jatuh, {len(seqs) - n_fall} ADL), {groups} kelompok CV")
    feats = adapter.attach_pose(seqs)
    has_feat = any(s.has_features for s in seqs)
    info = dict(dataset=cfg["dataset"]["name"], sequences=len(seqs), falls=n_fall, adl=len(seqs) - n_fall,
                cv_groups=groups, features=feats if has_feat else [], protocol=cfg["evaluation"]["protocol"])

    thr = detection.threshold_methods(seqs, cfg)
    if adapter.has_posture and has_feat:
        per_frame, per_seq = detection.pose_methods(seqs, cfg)
        table(per_frame).to_csv(out / "table_pose_per_frame.csv", index=False)
        table(thr + per_seq).to_csv(out / "table_pose.csv", index=False)
    if has_feat:
        rows, cinfo, detail = detection.candidate_methods(seqs, cfg)
        info.update(cinfo)
        pd.DataFrame(detail).to_csv(out / "candidates.csv", index=False)
    else:
        rows = []
        log("  tidak ada fitur pose -> hanya metode threshold (FAMS: jalankan extract-3d dulu)")
    table(thr + rows).to_csv(out / "table_candidate.csv", index=False)

    tcfg = cfg["trigger"]
    seq_df = pd.DataFrame([dict(sequence=s.id, group=s.group, is_fall=s.is_fall, sv_max=round(float(s.sv.max()), 2),
                                sv_min=round(float(s.sv.min()), 2),
                                n_triggers=len(trigger.triggers(s.t_acc, s.sv, tcfg["threshold_g"], tcfg["merge_ms"])),
                                **s.info) for s in seqs])
    seq_df.to_csv(out / "sequences.csv", index=False)
    if "label" in seq_df:
        names = adapter.classes
        pl = seq_df.groupby("label").agg(n=("sequence", "size"), sv_max_median=("sv_max", "median"),
                                         pct_triggered=("n_triggers", lambda x: round(100 * (x > 0).mean(), 1)))
        pl["sv_max_median"] = pl.sv_max_median.round(2)
        pl.insert(0, "type", ["FALL" if i in adapter.fall_classes else "ADL" for i in pl.index])
        pl.insert(0, "name", [names.get(i, str(i)) for i in pl.index])
        pl.to_csv(out / "per_label.csv")
    json.dump(info, open(out / "run_info.json", "w"), indent=2, default=str)
    text = report.write(out)
    log(text)
    return out
