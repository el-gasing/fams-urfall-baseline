"""
Skeleton 3D FAMS di sekitar tiap pemicu 3 g, sebagai pengganti depth map Kinect di baseline UR Fall.
Jalankan di server GPU. Aman diulang: langkah yang sudah selesai dilewati.

Cara pakai (dari folder repo):
    python fams3d/jalankan.py data/fams --out hasil/fams_3d --kerja kerja/fams [--device cuda]
        [--sebelum-s 1 --sesudah-s 3] [--interval-kal-s 2] [--tinggi-m 1.65] [--putar cam4=90]

Per sesi:
  1. Zip diekstrak ke <kerja>/<sid>/ (video dibutuhkan).
  2. Kandidat = pemicu SV_total > 3 g di IMU pinggang, per segmen label != 0. Fungsinya sama persis dengan
     yang dipakai jalankan_fams.py (kwolek.sinyal.pemicu), jadi waktu kandidat cocok.
  3. Kalibrasi kamera tanpa papan (badan subjek sebagai papan): ekstrak_skeleton.py --interval-s seluruh
     sesi, lalu triangulasi.py kalibrasi -> <out>/<sid>/kalibrasi.json.
  4. Tiap kandidat: skeleton 2D di [t - sebelum, t + sesudah] detik (semua frame, semua kamera; satu kali
     decode per kamera dengan ekstrak_skeleton.py --rentang), dipotong per klip, lalu triangulasi.py triangulasi -> <out>/<sid>/<urutan>_<k>.npz (t_ms, X (T, 12, 3) meter, z = atas,
     lantai z = 0). Kandidat yang berdekatan (< sebelum + sesudah) memakai klip yang sama.
  5. <out>/kandidat.csv: sesi, urutan, t_pemicu_ms, npz -> dibaca jalankan_fams.py --skeleton3d.

ekstrak_skeleton.py, skeleton_utils.py dan triangulasi.py di folder ini adalah salinan dari pipeline
skeleton 3D proyek ini (Keypoint R-CNN COCO, 12 sendi badan, kalibrasi diri + bundle adjustment).
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

DI_SINI = os.path.dirname(os.path.abspath(__file__))
AKAR = os.path.dirname(DI_SINI)
sys.path.insert(0, AKAR)
from kwolek import data_fams, sinyal  # noqa: E402


def jalan(cmd, log):
    print("  $", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        r = subprocess.run(cmd, cwd=DI_SINI, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"  GAGAL (kode {r.returncode}), lihat {log}", flush=True)
    return r.returncode == 0


def siapkan_folder(path, kerja):
    """Folder sesi siap pakai (zip diekstrak sekali)."""
    s = data_fams.Sesi(path)
    if not s.z:
        return os.path.abspath(path), s
    tujuan = os.path.join(kerja, s.sid, "sesi")
    if not os.path.exists(os.path.join(tujuan, "cameras.json")):
        print(f"  ekstrak {os.path.basename(path)} -> {tujuan}", flush=True)
        for n in s.z.namelist():
            if n.startswith(s.root) and not n.endswith("/") and "__MACOSX" not in n:
                dst = os.path.join(tujuan, n[len(s.root):])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with s.z.open(n) as src, open(dst, "wb") as out:
                    while True:
                        b = src.read(1 << 22)
                        if not b:
                            break
                        out.write(b)
    return os.path.abspath(tujuan), s


PER_FRAME = ("t_ms", "frame_idx", "kpts", "boxes", "scores")


def potong_klip(src, dst, a_ms, b_ms):
    """Salin frame dalam [a_ms, b_ms] dari tiap *_skeleton.npz di src ke dst (format sama)."""
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not f.endswith("_skeleton.npz"):
            continue
        d = dict(np.load(os.path.join(src, f)))
        m = (d["t_ms"] >= a_ms) & (d["t_ms"] <= b_ms)
        np.savez_compressed(os.path.join(dst, f), **{k: (v[m] if k in PER_FRAME else v) for k, v in d.items()})


def selesai_skeleton(folder, n_kamera):
    return os.path.isdir(folder) and len([f for f in os.listdir(folder) if f.endswith("_skeleton.npz")]) >= n_kamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sesi", nargs="+", help="zip, folder sesi, atau folder berisi banyak sesi")
    ap.add_argument("--out", default="hasil/fams_3d")
    ap.add_argument("--kerja", default="kerja/fams", help="tempat ekstrak zip dan skeleton 2D sementara")
    ap.add_argument("--label", default=os.path.join(AKAR, "labels_fams.yaml"))
    ap.add_argument("--peran", default="waist")
    ap.add_argument("--pemicu-g", type=float, default=sinyal.AMBANG_PEMICU_G)
    ap.add_argument("--sebelum-s", type=float, default=1.0)
    ap.add_argument("--sesudah-s", type=float, default=3.0)
    ap.add_argument("--interval-kal-s", type=float, default=2.0)
    ap.add_argument("--tinggi-m", type=float, default=1.65, help="tinggi badan untuk skala (perkiraan)")
    ap.add_argument("--putar", nargs="*", default=["cam4=90"], help="kamera yang dipasang miring")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fps", type=float, default=15, help="grid waktu triangulasi")
    a = ap.parse_args()
    out, kerja = os.path.abspath(a.out), os.path.abspath(a.kerja)
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    umum = ["--device", a.device, "--batch", str(a.batch)] + (["--putar", *a.putar] if a.putar else [])

    baris = []
    for p in data_fams.daftar_sesi(a.sesi):
        U, _ = data_fams.muat([p], a.label, a.peran)
        folder, s = siapkan_folder(p, kerja)
        sid, ncam = s.sid, len(s.kamera)
        t0 = min(c["started_at_ms"] for c in s.kamera)
        d_out, d_kerja = os.path.join(out, sid), os.path.join(kerja, sid)
        os.makedirs(d_out, exist_ok=True); os.makedirs(d_kerja, exist_ok=True)
        log = os.path.join(d_kerja, "log.txt")
        kand = [(u.id, t) for u in U for t in sinyal.pemicu(u.t_acc, u.sv, a.pemicu_g)]
        print(f"sesi {sid} ({s.subjek()}): {len(U)} urutan, {len(kand)} kandidat", flush=True)

        kal = os.path.join(d_out, "kalibrasi.json")
        if not os.path.exists(kal):
            sk = os.path.join(d_kerja, "skeleton_kalibrasi")
            if not selesai_skeleton(sk, ncam):
                jalan([py, "ekstrak_skeleton.py", folder, "--interval-s", str(a.interval_kal_s), "--out", sk] + umum, log)
            jalan([py, "triangulasi.py", "kalibrasi", sk, "--out", kal, "--tinggi-m", str(a.tinggi_m)], log)
        if not os.path.exists(kal):
            baris += [dict(sesi=sid, urutan=uid, t_pemicu_ms=t, npz=None) for uid, t in kand]
            continue

        # klip: gabungkan kandidat yang berdekatan
        klip = []
        for uid, t in sorted(kand, key=lambda x: x[1]):
            a0, a1 = t - a.sebelum_s * 1000, t + a.sesudah_s * 1000
            if klip and a0 <= klip[-1]["b"]:
                klip[-1]["b"] = max(klip[-1]["b"], a1); klip[-1]["anggota"].append((uid, t))
            else:
                klip.append(dict(a=a0, b=a1, anggota=[(uid, t)]))
        # satu kali decode per kamera: inference hanya di dalam klip (--rentang), lalu dipotong per klip
        sk_all = os.path.join(d_kerja, "skeleton_klip")
        belum = [k for k in range(len(klip)) if not os.path.exists(os.path.join(d_out, f"klip{k:03d}.npz"))]
        if belum and not selesai_skeleton(sk_all, ncam):
            rentang = [f"{(c['a'] - t0) / 1000:.3f}:{(c['b'] - t0) / 1000:.3f}" for c in klip]
            jalan([py, "ekstrak_skeleton.py", folder, "--out", sk_all, "--rentang", *rentang] + umum, log)
        for k, c in enumerate(klip):
            nama = f"klip{k:03d}"
            npz = os.path.join(d_out, f"{nama}.npz")
            if not os.path.exists(npz) and selesai_skeleton(sk_all, ncam):
                sk = os.path.join(d_kerja, nama)
                potong_klip(sk_all, sk, c["a"], c["b"])
                jalan([py, "triangulasi.py", "triangulasi", sk, "--kalibrasi", kal, "--out", npz,
                       "--fps", str(a.fps)], log)
            ada = os.path.exists(npz)
            baris += [dict(sesi=sid, urutan=uid, t_pemicu_ms=t, npz=os.path.relpath(npz, out) if ada else None)
                      for uid, t in c["anggota"]]
        pd.DataFrame(baris).to_csv(os.path.join(out, "kandidat.csv"), index=False)  # simpan bertahap

    df = pd.DataFrame(baris)
    df.to_csv(os.path.join(out, "kandidat.csv"), index=False)
    print(f"selesai: {len(df)} kandidat, {df.npz.notna().sum()} punya skeleton 3D -> {out}/kandidat.csv")


if __name__ == "__main__":
    main()
