"""report.md dari CSV sebuah run (dipanggil di akhir run, atau ulang lewat ``report --run``)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_T1 = ("Paper (dataset latih sendiri, 612 sampel, 10-fold): SVM dan k-NN accuracy 99.67, precision 100, "
            "sensitivity 99.05, specificity 100.")
PAPER_T2 = ("Paper (30 jatuh + 30 ADL), accuracy/precision/sensitivity/specificity: SVM depth saja "
            "90.00/83.30/100/80.00; SVM depth + acc 98.33/96.77/100/96.67; UFT 95.00/90.91/100/90.00; "
            "LFT 86.67/82.35/93.33/80.00.")


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if isinstance(v, float) and np.isnan(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write(run_dir) -> str:
    d = Path(run_dir)
    info = json.load(open(d / "run_info.json"))
    md = [f"# Baseline Kwolek & Kepski (2014): {info['dataset']}", "",
          f"{info['sequences']} urutan ({info['falls']} jatuh, {info['adl']} ADL), {info['cv_groups']} kelompok CV, "
          f"protokol `{info['protocol']}`. Fitur pose: {', '.join(info['features']) or '-'}.", ""]
    if (d / "table_pose_per_frame.csv").exists():
        md += ["## Tabel 1: klasifikasi pose berbaring per frame", PAPER_T1, "",
               to_markdown(pd.read_csv(d / "table_pose_per_frame.csv")), "",
               "## Tabel 2: deteksi jatuh per urutan (mode pose, persis paper)", PAPER_T2, "",
               to_markdown(pd.read_csv(d / "table_pose.csv")), ""]
    md += ["## Deteksi jatuh per urutan (mode kandidat, untuk perbandingan antar dataset)", "",
           to_markdown(pd.read_csv(d / "table_candidate.csv")), ""]
    if "n_candidates" in info:
        md += [f"Kandidat: {info['n_candidates']}, tanpa fitur pose: {info['n_candidates_without_features']}.", ""]
    if (d / "per_label.csv").exists():
        md += ["## Per label aktivitas", "", to_markdown(pd.read_csv(d / "per_label.csv")), ""]
    text = "\n".join(md)
    (d / "report.md").write_text(text + "\n", encoding="utf-8")
    return text
