"""
Baseline UR Fall (Kwolek & Kepski 2014) di FAMS, dengan prosedur yang sama dengan jalankan_urfall.py.

Tahap akselerometer memakai IMU pinggang FAMS (SV_total, pemicu 3 g, UFT/LFT).
Tahap verifikasi pose memakai fitur Kwolek yang dihitung dari skeleton 3D multi-kamera (pengganti depth
map Kinect), keluaran fams3d/jalankan.py. Tanpa --skeleton3d hanya metode threshold yang dijalankan.

Cara pakai:
    python jalankan_fams.py data/fams --out hasil/fams                         # threshold saja
    python jalankan_fams.py data/fams --skeleton3d hasil/fams_3d --out hasil/fams [--skema loso]

Argumen sesi: zip, folder sesi, atau folder yang berisi banyak zip/folder sesi.
"""
import argparse
import json
import os

import pandas as pd

from kwolek import data_fams, deteksi as DT, fitur as FT, sinyal
from kwolek.evaluasi import ke_markdown, tabel

AKAR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sesi", nargs="+")
    ap.add_argument("--out", default="hasil/fams")
    ap.add_argument("--label", default=os.path.join(AKAR, "labels_fams.yaml"))
    ap.add_argument("--skeleton3d", help="folder keluaran fams3d/jalankan.py (berisi kandidat.csv)")
    ap.add_argument("--fitur", default="paper", choices=["paper", "+P40"])
    ap.add_argument("--skema", default="loso", help="loso (per subjek) atau 10-fold (per subjek)")
    ap.add_argument("--peran", default="waist", help="IMU untuk tahap akselerometer: waist | chest | thigh_right")
    ap.add_argument("--pad-s", type=float, default=0.0, help="lebarkan tiap segmen label (detik)")
    ap.add_argument("--pemicu-g", type=float, default=sinyal.AMBANG_PEMICU_G)
    ap.add_argument("--uft-g", type=float, default=sinyal.UFT_G)
    ap.add_argument("--lft-g", type=float, default=sinyal.LFT_G)
    ap.add_argument("--jendela", nargs=2, type=float, default=[0.0, 1.0], metavar=("A", "B"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    U, info_sesi = data_fams.muat(a.sesi, a.label, a.peran, a.pad_s)
    nj = sum(u.jatuh for u in U)
    nsubj = len({u.kelompok for u in U})
    print(f"FAMS: {len(info_sesi)} sesi, {nsubj} subjek, {len(U)} urutan ({nj} jatuh, {len(U) - nj} ADL)")
    info_sesi.to_csv(os.path.join(a.out, "sesi.csv"), index=False)

    baris = DT.metode_threshold(U, a.uft_g, a.lft_g, a.pemicu_g)
    info, rinci = {}, []
    fitur = FT.daftar_fitur(a.fitur)
    if a.skeleton3d:
        h_max = data_fams.tempel_skeleton(U, a.skeleton3d, fitur)
        kd, info, rinci = DT.metode_kandidat(U, a.skema, a.pemicu_g, *a.jendela)
        baris += kd
        info["h_max_m"] = {k: round(float(v), 3) for k, v in h_max.items()}
    t = tabel(baris)
    t.to_csv(os.path.join(a.out, "tabel_kandidat.csv" if a.skeleton3d else "tabel_threshold.csv"), index=False)

    # rincian per urutan: puncak SV, jumlah pemicu, per label
    per = pd.DataFrame([dict(**u.info, urutan=u.id, jatuh=u.jatuh, sv_maks=round(float(u.sv.max()), 2),
                             sv_min=round(float(u.sv.min()), 2),
                             n_pemicu=len(sinyal.pemicu(u.t_acc, u.sv, a.pemicu_g))) for u in U])
    per.to_csv(os.path.join(a.out, "urutan.csv"), index=False)
    lab = data_fams.baca_label(a.label)
    per_label = per.groupby("label").agg(n=("urutan", "size"), sv_maks_median=("sv_maks", "median"),
                                         persen_terpicu=("n_pemicu", lambda x: round(100 * (x > 0).mean(), 1)))
    per_label["sv_maks_median"] = per_label.sv_maks_median.round(2)
    per_label.insert(0, "jenis", [lab[i]["jenis"] for i in per_label.index])
    per_label.insert(0, "nama", [lab[i]["nama"] for i in per_label.index])
    per_label.to_csv(os.path.join(a.out, "per_label.csv"))
    if rinci:
        pd.DataFrame(rinci).to_csv(os.path.join(a.out, "kandidat.csv"), index=False)

    md = ["# Baseline UR Fall (Kwolek & Kepski 2014) di FAMS",
          f"{len(info_sesi)} sesi, {nsubj} subjek, {len(U)} urutan ({nj} jatuh, {len(U) - nj} ADL). "
          f"IMU: {a.peran}. Pemicu {a.pemicu_g} g, jendela verifikasi [{a.jendela[0]}, {a.jendela[1]}] s. "
          + (f"Fitur pose dari skeleton 3D: {', '.join(fitur)}. CV: {a.skema} per subjek." if a.skeleton3d
             else "Tanpa skeleton 3D: hanya metode threshold."),
          "", ke_markdown(t), "", "## Per label", "", ke_markdown(per_label.reset_index()), ""]
    if info:
        md.append(f"Kandidat: {json.dumps(info)}")
    open(os.path.join(a.out, "hasil.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
