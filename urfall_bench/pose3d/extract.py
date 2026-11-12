"""
Skeleton 3D di sekitar tiap pemicu, sebagai pengganti depth map Kinect (perintah ``extract-3d``).
Jalan di server GPU. Aman diulang: langkah yang sudah selesai dilewati.

Per sesi (adapter harus punya ``session_paths()`` dan ``load_session()``, saat ini: fams):
  1. Zip diekstrak ke cache/<nama>/work/<sid>/session/ (video dibutuhkan).
  2. Kandidat = pemicu SV_total > trigger.threshold_g per urutan, fungsi yang sama dengan ``run``.
  3. Kalibrasi kamera tanpa papan (badan subjek sebagai papan): ekstrak_skeleton.py --interval-s seluruh
     sesi, lalu triangulasi.py kalibrasi -> cache/<nama>/pose3d/<sid>/calibration.json.
  4. Skeleton 2D di [t - before_s, t + after_s] tiap kandidat (semua frame, semua kamera). Tiap video hanya
     di-decode sekali (ekstrak_skeleton.py --rentang), lalu dipotong per klip.
  5. triangulasi.py triangulasi tiap klip -> cache/<nama>/pose3d/<sid>/clip_<ms>.npz
     (t_ms, X (T, 12, 3) meter, z = atas, lantai z = 0).
  6. cache/<nama>/pose3d/candidates.csv: session, sequence, trigger_ms, npz -> dibaca ``run``.

ekstrak_skeleton.py, skeleton_utils.py, triangulasi.py di folder ini: pipeline skeleton 3D proyek
(Keypoint R-CNN COCO, 12 sendi badan, kalibrasi diri + bundle adjustment), ditambah opsi --rentang.
"""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from ..kwolek.trigger import triggers

HERE = os.path.dirname(os.path.abspath(__file__))
PER_FRAME = ("t_ms", "frame_idx", "kpts", "boxes", "scores")


def _run(cmd, log):
    print("  $", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"  GAGAL (kode {r.returncode}), lihat {log}", flush=True)
    return r.returncode == 0


def _prepare_folder(sess, work):
    """Folder sesi siap pakai (zip diekstrak sekali)."""
    if not sess.z:
        return os.path.abspath(sess.path)
    dst_root = os.path.join(work, "session")
    if not os.path.exists(os.path.join(dst_root, "cameras.json")):
        print(f"  ekstrak {os.path.basename(sess.path)} -> {dst_root}", flush=True)
        for n in sess.z.namelist():
            if n.startswith(sess.root) and not n.endswith("/") and "__MACOSX" not in n:
                dst = os.path.join(dst_root, n[len(sess.root):])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with sess.z.open(n) as src, open(dst, "wb") as out:
                    while True:
                        b = src.read(1 << 22)
                        if not b:
                            break
                        out.write(b)
    return os.path.abspath(dst_root)


def cut_clip(src, dst, a_ms, b_ms):
    """Salin frame dalam [a_ms, b_ms] dari tiap *_skeleton.npz di src ke dst (format sama)."""
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith("_skeleton.npz"):
            d = dict(np.load(os.path.join(src, f)))
            m = (d["t_ms"] >= a_ms) & (d["t_ms"] <= b_ms)
            np.savez_compressed(os.path.join(dst, f), **{k: (v[m] if k in PER_FRAME else v) for k, v in d.items()})


def _done(folder, n_cam):
    return os.path.isdir(folder) and sum(f.endswith("_skeleton.npz") for f in os.listdir(folder)) >= n_cam


def clips_from(cands, before_s, after_s):
    """Gabungkan kandidat yang klipnya tumpang tindih. cands: [(sequence_id, t_ms)]."""
    clips = []
    for sid, t in sorted(cands, key=lambda x: x[1]):
        a0, a1 = t - before_s * 1000, t + after_s * 1000
        if clips and a0 <= clips[-1]["b"]:
            clips[-1]["b"] = max(clips[-1]["b"], a1); clips[-1]["members"].append((sid, t))
        else:
            clips.append(dict(a=a0, b=a1, members=[(sid, t)]))
    return clips


def extract(cfg, adapter):
    if not hasattr(adapter, "session_paths"):
        raise SystemExit(f"extract-3d: adapter '{adapter.adapter_name}' tidak punya video multi-kamera")
    p3, tr = cfg["pose3d"], cfg["trigger"]
    out_root, work_root = adapter.cache / "pose3d", adapter.cache / "work"
    out_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    common = ["--device", str(p3["device"]), "--batch", str(p3["batch"])]
    rot = [f"{k}={v}" for k, v in (p3.get("rotate") or {}).items()]
    common += ["--putar", *rot] if rot else []

    rows = []
    for path in adapter.session_paths():
        sess, seqs = adapter.load_session(path)
        sid, ncam = sess.sid, len(sess.cameras)
        t0 = min(c["started_at_ms"] for c in sess.cameras)
        d_out, d_work = out_root / sid, work_root / sid
        d_out.mkdir(parents=True, exist_ok=True); d_work.mkdir(parents=True, exist_ok=True)
        log = str(d_work / "log.txt")
        cands = [(s.id, t) for s in seqs for t in triggers(s.t_acc, s.sv, tr["threshold_g"], tr["merge_ms"])]
        print(f"sesi {sid} ({sess.subject()}): {len(seqs)} urutan, {len(cands)} kandidat", flush=True)
        folder = _prepare_folder(sess, str(d_work))

        cal = d_out / "calibration.json"
        if not cal.exists():
            sk = str(d_work / "skeleton_calibration")
            if not _done(sk, ncam):
                _run([py, "ekstrak_skeleton.py", folder, "--interval-s", str(p3["calib_interval_s"]), "--out", sk]
                     + common, log)
            _run([py, "triangulasi.py", "kalibrasi", sk, "--out", str(cal), "--tinggi-m", str(p3["height_m"])], log)
        if not cal.exists():
            rows += [dict(session=sid, sequence=q, trigger_ms=t, npz=None) for q, t in cands]
            continue

        clips = clips_from(cands, p3["before_s"], p3["after_s"])
        ranges = [f"{(c['a'] - t0) / 1000:.3f}:{(c['b'] - t0) / 1000:.3f}" for c in clips]
        sk_all = str(d_work / ("skeleton_clips_" + hashlib.md5(" ".join(ranges).encode()).hexdigest()[:8]))
        pending = [c for c in clips if not (d_out / f"clip_{int(c['a'])}.npz").exists()]
        if pending and not _done(sk_all, ncam):
            _run([py, "ekstrak_skeleton.py", folder, "--out", sk_all, "--rentang", *ranges] + common, log)
        for c in clips:
            npz = d_out / f"clip_{int(c['a'])}.npz"
            if not npz.exists() and _done(sk_all, ncam):
                sk = str(d_work / f"clip_{int(c['a'])}")
                cut_clip(sk_all, sk, c["a"], c["b"])
                _run([py, "triangulasi.py", "triangulasi", sk, "--kalibrasi", str(cal), "--out", str(npz),
                      "--fps", str(p3["fps"])], log)
            rel = os.path.relpath(npz, out_root) if npz.exists() else None
            rows += [dict(session=sid, sequence=q, trigger_ms=t, npz=rel) for q, t in c["members"]]
        pd.DataFrame(rows).to_csv(out_root / "candidates.csv", index=False)  # simpan bertahap

    df = pd.DataFrame(rows, columns=["session", "sequence", "trigger_ms", "npz"])
    df.to_csv(out_root / "candidates.csv", index=False)
    print(f"selesai: {len(df)} kandidat, {df.npz.notna().sum()} punya skeleton 3D -> {out_root}/candidates.csv")
