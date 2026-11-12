"""
Baseline UR Fall (Kwolek & Kepski 2014) di dataset UR Fall: replikasi Tabel 1 dan Tabel 2 paper,
plus mode "kandidat" yang dipakai untuk dibandingkan dengan FAMS.

Cara pakai:
    python jalankan_urfall.py --data data/urfall --out hasil/urfall
        [--fitur paper|+P40] [--skema 10-fold] [--jendela 0.5 2.0]

Keluaran: <out>/hasil.md (semua tabel), tabel_*.csv, kandidat.csv (keputusan per kandidat).
"""
import argparse
import json
import os

from kwolek import deteksi as DT
from kwolek import data_urfall, fitur as FT, sinyal
from kwolek.evaluasi import ke_markdown, tabel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/urfall")
    ap.add_argument("--out", default="hasil/urfall")
    ap.add_argument("--fitur", default="paper", choices=["paper", "+P40"])
    ap.add_argument("--skema", default="10-fold", help="CV per urutan (UR Fall tidak punya ID subjek)")
    ap.add_argument("--pemicu-g", type=float, default=sinyal.AMBANG_PEMICU_G)
    ap.add_argument("--uft-g", type=float, default=sinyal.UFT_G)
    ap.add_argument("--lft-g", type=float, default=sinyal.LFT_G)
    ap.add_argument("--jendela", nargs=2, type=float, default=[0.0, 1.0], metavar=("A", "B"),
                    help="frame yang dinilai: [t_pemicu + A, t_pemicu + B] detik")
    ap.add_argument("--min-berbaring-s", type=float, default=0.5, help="mode pose saja: lama berbaring minimum")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fitur = FT.daftar_fitur(a.fitur)
    U = data_urfall.muat(a.data, fitur)
    nj = sum(u.jatuh for u in U)
    print(f"UR Fall: {len(U)} urutan ({nj} jatuh, {len(U) - nj} ADL), fitur {fitur}")

    pose = DT.metode_pose(U, a.skema, a.pemicu_g, *a.jendela, a.min_berbaring_s)
    t1 = tabel([(n, m) for n, m in pose if n.startswith("Pose per frame")])
    t2 = tabel(DT.metode_threshold(U, a.uft_g, a.lft_g, a.pemicu_g) + [(n, m) for n, m in pose if not n.startswith("Pose")])
    kd, info, rinci = DT.metode_kandidat(U, a.skema, a.pemicu_g, *a.jendela)
    t3 = tabel(DT.metode_threshold(U, a.uft_g, a.lft_g, a.pemicu_g) + kd)

    t1.to_csv(os.path.join(a.out, "tabel1_pose_per_frame.csv"), index=False)
    t2.to_csv(os.path.join(a.out, "tabel2_deteksi_per_urutan.csv"), index=False)
    t3.to_csv(os.path.join(a.out, "tabel_kandidat.csv"), index=False)
    import pandas as pd
    pd.DataFrame(rinci).to_csv(os.path.join(a.out, "kandidat.csv"), index=False)
    md = [f"# Baseline UR Fall (Kwolek & Kepski 2014) di UR Fall",
          f"{len(U)} urutan ({nj} jatuh, {len(U) - nj} ADL). Fitur: {', '.join(fitur)}. CV: {a.skema} per urutan. "
          f"Pemicu {a.pemicu_g} g, jendela verifikasi [{a.jendela[0]}, {a.jendela[1]}] s setelah pemicu.",
          "", "## Tabel 1: klasifikasi pose berbaring per frame",
          "Paper (dataset latih sendiri, 612 sampel, 10-fold): SVM dan k-NN akurasi 99.67, presisi 100, "
          "sensitivitas 99.05, spesifisitas 100.", "", ke_markdown(t1),
          "", "## Tabel 2: deteksi jatuh per urutan (mode pose, persis paper)",
          "Paper (30 jatuh + 30 ADL): SVM depth saja 90.00/83.30/100/80.00, SVM depth + acc 98.33/96.77/100/96.67, "
          "UFT 95.00/90.91/100/90.00, LFT 86.67/82.35/93.33/80.00 (akurasi/presisi/sensitivitas/spesifisitas).",
          "", ke_markdown(t2),
          "", "## Mode kandidat (dipakai untuk perbandingan dengan FAMS)", "", ke_markdown(t3), "",
          f"Kandidat: {json.dumps(info)}"]
    open(os.path.join(a.out, "hasil.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
