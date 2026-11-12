"""
Triangulasi skeleton 2D multi-kamera -> skeleton 3D, tanpa papan kalibrasi.

Cara pakai:
    # 1. kalibrasi dari skeleton seluruh sesi (ekstrak_skeleton.py --interval-s 2)
    .venv/bin/python triangulasi.py kalibrasi <folder_skeleton_kalibrasi> --out <kalibrasi.json> [--tinggi-m 1.65]
    # 2. skeleton 3D untuk satu segmen (bisa pakai sebagian kamera)
    .venv/bin/python triangulasi.py triangulasi <folder_skeleton> --kalibrasi <kalibrasi.json> --out <x.npz>
        [--kamera cam1 cam2]
    # 3. ablasi jumlah kamera (semua kombinasi >= 2 kamera) + metrik kualitas 3D
    .venv/bin/python triangulasi.py ablasi <folder_skeleton> [<folder_skeleton> ...] --kalibrasi <kalibrasi.json>
        --out <hasil.csv>

Kalibrasi diri (self-calibration): badan subjek dipakai sebagai "papan kalibrasi".
  1. Sendi subjek yang terlihat bersamaan di dua kamera = pasangan titik.
  2. Pasangan kamera dengan titik bersama terbanyak -> matriks esensial -> pose relatif.
     Kamera lain ditambahkan lewat PnP terhadap titik 3D yang sudah ada.
  3. Bundle adjustment: pose kamera, focal length, distorsi radial k1, dan semua titik 3D
     dioptimasi bersama (loss Huber, observasi pencilan dibuang lalu diulang).
  4. Skala & arah: arah "atas" = rata-rata arah pergelangan kaki -> bahu saat berdiri,
     skala dari --tinggi-m (jarak bahu-pergelangan kaki ~ 0.78 x tinggi badan, Drillis & Contini).
     Lantai di z = 0.
Skala hanya perkiraan (tinggi badan asli tidak diketahui); metrik yang dilaporkan dibuat
tidak bergantung skala bila memungkinkan (reprojection error dalam piksel, CV panjang tulang).
"""
import argparse
import glob
import itertools
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from skeleton_utils import track_subject

JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
BONES = {"bahu": (0, 1), "pinggul": (6, 7), "lengan_atas_L": (0, 2), "lengan_atas_R": (1, 3),
         "lengan_bawah_L": (2, 4), "lengan_bawah_R": (3, 5), "paha_L": (6, 8), "paha_R": (7, 9),
         "betis_L": (8, 10), "betis_R": (9, 11), "badan_L": (0, 6), "badan_R": (1, 7)}
PAIRS_LR = [("lengan_atas_L", "lengan_atas_R"), ("lengan_bawah_L", "lengan_bawah_R"),
            ("paha_L", "paha_R"), ("betis_L", "betis_R"), ("badan_L", "badan_R")]


# ---------------------------------------------------------------- data 2D

def load_cams(folder, skor_min=0.8, kp_min=2.0, offsets=None):
    """Per kamera: t_ms, xy (F,12,2), ok (F,12), tinggi kotak subjek (F,). Hanya subjek (tracking)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*_skeleton.npz"))):
        d = dict(np.load(p))
        cam = str(d["cam_id"])
        pick = track_subject(d, skor_min)
        F = len(pick)
        xy = np.full((F, 12, 2), np.nan)
        ok = np.zeros((F, 12), bool)
        bh = np.full(F, np.nan)
        for f in np.flatnonzero(pick >= 0):
            k = d["kpts"][f, pick[f]]
            good = k[:, 2] > kp_min
            xy[f, good], ok[f, good] = k[good, :2], True
            b = d["boxes"][f, pick[f]]
            bh[f] = max(b[2] - b[0], b[3] - b[1])  # sisi terpanjang: tetap wajar saat berbaring
        t = d["t_ms"] - (offsets or {}).get(cam, 0.0)
        out[cam] = dict(t=t, xy=xy, ok=ok, bh=bh)
    return out


def align(cams, grid, tol_ms=70, gap_ms=150):
    """Sampel 2D tiap kamera di grid waktu bersama: interpolasi linear antar 2 frame yang mengapit
    (kalau jaraknya <= gap_ms dan sendi ada di keduanya), selain itu frame terdekat (<= tol_ms)."""
    res = {}
    for c, d in cams.items():
        t, xy, ok = d["t"], d["xy"], d["ok"]
        T = len(grid)
        X = np.full((T, 12, 2), np.nan)
        M = np.zeros((T, 12), bool)
        B = np.full(T, np.nan)
        j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        a, b = j - 1, j
        span = t[b] - t[a]
        w = np.clip((grid - t[a]) / np.where(span > 0, span, 1), 0, 1)
        near = np.where(np.abs(t[a] - grid) <= np.abs(t[b] - grid), a, b)
        near_ok = np.abs(t[near] - grid) <= tol_ms
        inside = (grid >= t[a]) & (grid <= t[b]) & (span <= gap_ms)
        for k in range(12):
            both = inside & ok[a, k] & ok[b, k]
            X[both, k] = xy[a[both], k] * (1 - w[both, None]) + xy[b[both], k] * w[both, None]
            M[both, k] = True
            alt = ~both & near_ok & ok[near, k]
            X[alt, k] = xy[near[alt], k]
            M[alt, k] = True
        B[near_ok] = d["bh"][near[near_ok]]
        res[c] = (X, M, B)
    return res


def common_grid(cams, fps=None):
    """Grid waktu: fps tetap di rentang yang diliput semua kamera, atau (fps=None) waktu frame
    kamera dengan frame terbanyak (untuk data sampling --interval-s)."""
    t0 = max(d["t"][0] for d in cams.values())
    t1 = min(d["t"][-1] for d in cams.values())
    if fps:
        return np.arange(t0, t1, 1000 / fps)
    ref = max(cams.values(), key=lambda d: len(d["t"]))["t"]
    return ref[(ref >= t0) & (ref <= t1)]


# ---------------------------------------------------------------- geometri

def project(Xw, rvec, tvec, f, k1, c):
    """Xw (N,3) -> piksel (N,2), model pinhole + distorsi radial 1 suku."""
    Xc = Rotation.from_rotvec(rvec).apply(Xw) + tvec
    z = Xc[:, 2:3]
    xn = Xc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    r2 = (xn ** 2).sum(1, keepdims=True)
    return f * xn * (1 + k1 * r2) + c


def undistort(px, f, k1, c, iters=5):
    """Kebalikan dari distorsi radial (iterasi titik tetap) -> koordinat ternormalisasi."""
    xd = (px - c) / f
    xn = xd.copy()
    for _ in range(iters):
        xn = xd / (1 + k1 * (xn ** 2).sum(-1, keepdims=True))
    return xn


def proj_matrix(cam):
    R = Rotation.from_rotvec(cam["rvec"]).as_matrix()
    return np.hstack([R, cam["tvec"][:, None]])  # untuk koordinat ternormalisasi (K = I)


def triangulate(views, max_err_px=None, cams=None):
    """views: list of (P (3,4) ternormalisasi, xn (2,), cam_name, px (2,)). DLT; kalau max_err_px
    diberi dan ada >= 3 view, view dengan error terbesar dibuang selama errornya > max_err_px.
    Kembalikan (X (3,), error rata-rata px, nama kamera yang dipakai) atau None."""
    views = list(views)
    while len(views) >= 2:
        A = []
        for P, x, _, _ in views:
            A += [x[0] * P[2] - P[0], x[1] * P[2] - P[1]]
        _, _, Vt = np.linalg.svd(np.array(A))
        X = Vt[-1][:3] / Vt[-1][3]
        if cams is None:
            return X, np.nan, [v[2] for v in views]
        errs = []
        for P, _, name, px in views:
            cm = cams[name]
            if (P[2, :3] @ X + P[2, 3]) <= 0:  # di belakang kamera
                errs.append(np.inf)
                continue
            errs.append(np.linalg.norm(project(X[None], cm["rvec"], cm["tvec"], cm["f"], cm["k1"], cm["c"])[0] - px))
        errs = np.array(errs)
        if max_err_px is None or errs.max() <= max_err_px:
            return X, float(errs.mean()), [v[2] for v in views]
        if len(views) == 2:
            return None
        views.pop(int(np.argmax(errs)))
    return None


# ---------------------------------------------------------------- kalibrasi

def bundle_adjust(cams, obs, X3, fix, f_scale=4.0, f_fix=None):
    """cams: dict nama -> {rvec,tvec,f,k1,c}; obs: list (nama, idx_titik, px); X3 (N,3).
    Kamera `fix` posenya dikunci (menghilangkan kebebasan rotasi/translasi global)."""
    names = list(cams)
    free = [n for n in names if n != fix]
    ci = {n: i for i, n in enumerate(names)}
    oc = np.array([ci[o[0]] for o in obs])
    op = np.array([o[1] for o in obs])
    opx = np.array([o[2] for o in obs])
    N, C = len(X3), len(names)
    # parameter: tiap kamera [f, k1] + kamera bebas [rvec, tvec] + titik
    def unpack(v):
        fk = v[:2 * C].reshape(C, 2)
        pose = v[2 * C:2 * C + 6 * len(free)].reshape(len(free), 6)
        X = v[2 * C + 6 * len(free):].reshape(N, 3)
        return fk, pose, X

    def poses(pose):
        rv = np.zeros((C, 3)); tv = np.zeros((C, 3))
        for n in names:
            if n == fix:
                rv[ci[n]], tv[ci[n]] = cams[n]["rvec"], cams[n]["tvec"]
            else:
                p = pose[free.index(n)]
                rv[ci[n]], tv[ci[n]] = p[:3], p[3:]
        return rv, tv

    cc = np.array([cams[n]["c"] for n in names])

    def resid(v):
        fk, pose, X = unpack(v)
        rv, tv = poses(pose)
        Xc = Rotation.from_rotvec(rv[oc]).apply(X[op]) + tv[oc]
        z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])[:, None]
        xn = Xc[:, :2] / z
        r2 = (xn ** 2).sum(1, keepdims=True)
        px = fk[oc, 0:1] * xn * (1 + fk[oc, 1:2] * r2) + cc[oc]
        return (px - opx).ravel()

    v0 = np.concatenate([np.array([[cams[n]["f"], cams[n]["k1"]] for n in names]).ravel(),
                         np.concatenate([np.r_[cams[n]["rvec"], cams[n]["tvec"]] for n in free]) if free else [],
                         X3.ravel()])
    S = lil_matrix((2 * len(obs), len(v0)), dtype=int)
    r = np.arange(len(obs))
    for d in range(2):
        S[2 * r + d, 2 * oc] = 1
        S[2 * r + d, 2 * oc + 1] = 1
        for n in free:
            m = r[oc == ci[n]]
            base = 2 * C + 6 * free.index(n)
            for q in range(6):
                S[2 * m + d, base + q] = 1
        for q in range(3):
            S[2 * r + d, 2 * C + 6 * len(free) + 3 * op + q] = 1
    lb = np.full(len(v0), -np.inf); ub = np.full(len(v0), np.inf)
    lb[0:2 * C:2], ub[0:2 * C:2] = 200, 5000      # focal length wajar (piksel)
    if f_fix:  # focal length diketahui (spesifikasi kamera) -> dikunci
        v0[0:2 * C:2] = f_fix
        lb[0:2 * C:2], ub[0:2 * C:2] = f_fix - 0.5, f_fix + 0.5
    lb[1:2 * C:2], ub[1:2 * C:2] = -0.6, 0.6      # distorsi radial
    # skala tiap jenis parameter (focal ~ ratusan piksel, rotasi/translasi/titik ~ 0.01-0.1)
    xs = np.r_[np.tile([50, 0.02], C), np.tile([0.02] * 3 + [0.05] * 3, len(free)), np.full(3 * N, 0.05)]
    sol = least_squares(resid, v0, jac_sparsity=S, loss="huber", f_scale=f_scale, x_scale=xs,
                        method="trf", bounds=(lb, ub), max_nfev=1000, ftol=1e-10, xtol=1e-10, verbose=0)
    fk, pose, X = unpack(sol.x)
    rv, tv = poses(pose)
    for n in names:
        cams[n].update(f=fk[ci[n], 0], k1=fk[ci[n], 1], rvec=rv[ci[n]], tvec=tv[ci[n]])
    err = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    return X, err


def calibrate(al, size, f0, thr_e=6.0, thr_pnp=15.0, thr_tri=40.0, thr_akhir=15.0, awal=None, f_fix=None):
    names = sorted(al)
    W, H = size
    c0 = np.array([W / 2, H / 2])
    cams = {n: dict(f=float(f0), k1=0.0, c=c0.copy(), rvec=np.zeros(3), tvec=np.zeros(3)) for n in names}

    def tri_all(placed, max_err=None):
        """Triangulasi semua (waktu, sendi) yang terlihat di >= 2 kamera yang sudah ditempatkan."""
        T = len(al[names[0]][0])
        pts, keys, used = [], [], []
        P = {n: proj_matrix(cams[n]) for n in placed}
        for ti in range(T):
            for k in range(12):
                vs = [(P[n], undistort(al[n][0][ti, k], cams[n]["f"], cams[n]["k1"], cams[n]["c"]), n,
                       al[n][0][ti, k]) for n in placed if al[n][1][ti, k]]
                if len(vs) < 2:
                    continue
                r = triangulate(vs, max_err, cams)
                if r is not None:
                    pts.append(r[0]); keys.append((ti, k)); used.append(r[2])
        return np.array(pts), keys, used

    if awal is not None:
        # mulai dari kalibrasi lain (mis. sesi lain di ruangan yang sama) -> langsung bundle adjustment
        for n in names:
            cams[n].update({k: np.array(v, float) if k in ("rvec", "tvec", "c") else float(v)
                            for k, v in awal[n].items() if k in ("f", "k1", "c", "rvec", "tvec")})
        a, placed = names[0], list(names)
        X3, keys, used = tri_all(placed, max_err=thr_tri)
        print(f"mulai dari kalibrasi awal: {len(X3)} titik tertriangulasi")
    else:
        # pasangan awal: titik bersama terbanyak
        shared = {(a, b): int((al[a][1] & al[b][1]).sum()) for a, b in itertools.combinations(names, 2)}
        (a, b), n_ab = max(shared.items(), key=lambda kv: kv[1])
        print(f"titik bersama per pasangan: " + ", ".join(f"{p[0]}-{p[1]}:{v}" for p, v in shared.items()))
        m = al[a][1] & al[b][1]
        pa, pb = al[a][0][m], al[b][0][m]
        na, nb = (pa - c0) / f0, (pb - c0) / f0
        E, inl = cv2.findEssentialMat(na, nb, np.eye(3), cv2.RANSAC, 0.999, thr_e / f0)
        _, R, t, inl = cv2.recoverPose(E, na, nb, np.eye(3), mask=inl)
        print(f"pasangan awal {a}-{b}: {n_ab} titik, inlier esensial {int((inl > 0).sum())}")
        cams[b]["rvec"], cams[b]["tvec"] = Rotation.from_matrix(R).as_rotvec(), t.ravel()
        placed = [a, b]

        X3, keys, used = tri_all(placed)
        for n in sorted(set(names) - set(placed), key=lambda n: -sum(al[n][1][ti, k] for ti, k in keys)):
            sel = [i for i, (ti, k) in enumerate(keys) if al[n][1][ti, k]]
            obj = X3[sel].astype(np.float64)
            img = np.array([al[n][0][keys[i][0], keys[i][1]] for i in sel], np.float64)
            K = np.array([[f0, 0, c0[0]], [0, f0, c0[1]], [0, 0, 1]], float)
            ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=thr_pnp, iterationsCount=2000)
            print(f"tambah {n} lewat PnP: {len(sel)} titik, inlier {0 if inl is None else len(inl)}")
            cams[n]["rvec"], cams[n]["tvec"] = rv.ravel(), tv.ravel()
            placed.append(n)
            X3, keys, used = tri_all(placed, max_err=thr_tri)

    # bundle adjustment, 2 putaran (buang observasi pencilan di antaranya)
    def observations():
        return [(n, i, al[n][0][ti, k]) for i, (ti, k) in enumerate(keys) for n in used[i]]

    for rnd, thr in enumerate([thr_tri, thr_akhir]):
        obs = observations()
        X3, err = bundle_adjust(cams, obs, X3, fix=a, f_fix=f_fix)
        print(f"BA putaran {rnd + 1}: {len(obs)} observasi, reprojection error median {np.median(err):.1f} px, "
              f"rata-rata {err.mean():.1f} px")
        X3, keys, used = tri_all(names, max_err=thr)
    obs = observations()
    X3, err = bundle_adjust(cams, obs, X3, fix=a, f_fix=f_fix)
    per_cam = {n: err[[o[0] == n for o in obs]] for n in names}
    return cams, X3, keys, err, per_cam


def to_world(cams, X3, keys, tinggi_m):
    """Putar & skala supaya z = atas, lantai z = 0, satuan meter (perkiraan dari tinggi badan)."""
    T = max(ti for ti, _ in keys) + 1
    J = np.full((T, 12, 3), np.nan)
    for x, (ti, k) in zip(X3, keys):
        J[ti, k] = x
    sh = np.nanmean(J[:, 0:2], 1)
    an = np.nanmean(J[:, 10:12], 1)
    v = sh - an
    L = np.linalg.norm(v, axis=1)
    good = np.isfinite(L)
    stand = good & (L >= np.nanpercentile(L[good], 60))  # frame paling "tegak"
    up = np.median(v[stand] / L[stand, None], 0)
    up /= np.linalg.norm(up)
    s = 0.78 * tinggi_m / np.median(L[stand])
    # sumbu x: arah kamera acuan (sumbu x kamera pertama) diproyeksikan ke bidang lantai
    ref = sorted(cams)[0]
    xr = Rotation.from_rotvec(cams[ref]["rvec"]).as_matrix()[0]
    x = xr - (xr @ up) * up
    x /= np.linalg.norm(x)
    Rw = np.vstack([x, np.cross(up, x), up])
    o = np.median(an[stand], 0)
    o = o - up * (0.039 * tinggi_m / s)  # titik pergelangan kaki ~3.9% tinggi badan di atas lantai
    for n, cm in cams.items():
        R = Rotation.from_rotvec(cm["rvec"]).as_matrix()
        Rn = R @ Rw.T
        cm["rvec"] = Rotation.from_matrix(Rn).as_rotvec()
        cm["tvec"] = s * (R @ o + cm["tvec"])
    return (Rw @ (X3 - o).T).T * s


# ---------------------------------------------------------------- 3D per segmen

def load_calib(path):
    js = json.load(open(path))
    return {n: dict(f=c["f"], k1=c["k1"], c=np.array(c["c"]), rvec=np.array(c["rvec"]), tvec=np.array(c["tvec"]))
            for n, c in js["kamera"].items()}, js


def triangulate_seq(al, cams, use, max_err_px=None, max_err_frac=0.05):
    """max_err_frac: ambang buang view = fraksi tinggi badan subjek di gambar (sisi terpanjang kotak),
    supaya setara antar dataset beresolusi beda; max_err_px (kalau diberi) = ambang piksel tetap."""
    T = len(next(iter(al.values()))[0])
    X = np.full((T, 12, 3), np.nan)
    err = np.full((T, 12), np.nan)
    nv = np.zeros((T, 12), int)
    P = {n: proj_matrix(cams[n]) for n in use}
    for ti in range(T):
        for k in range(12):
            vs = [(P[n], undistort(al[n][0][ti, k], cams[n]["f"], cams[n]["k1"], cams[n]["c"]), n, al[n][0][ti, k])
                  for n in use if al[n][1][ti, k]]
            if len(vs) < 2:
                continue
            thr = max_err_px
            if thr is None:
                bh = np.nanmedian([al[n][2][ti] for n in use]) if any(np.isfinite(al[n][2][ti]) for n in use) else np.nan
                thr = max_err_frac * bh if np.isfinite(bh) else 25.0
            r = triangulate(vs, thr, cams)
            if r is not None:
                X[ti, k], err[ti, k], nv[ti, k] = r[0], r[1], len(r[2])
    return X, err, nv


def metrics(X, err, nv, al, use, fps):
    """Metrik kualitas skeleton 3D (sebagian besar tidak bergantung skala)."""
    valid = np.isfinite(X[..., 0])
    bh = np.nanmedian(np.concatenate([al[n][2] for n in use]))
    out = dict(cakupan_sendi_pct=100 * valid.mean(),
               frame_lengkap_pct=100 * valid.all(1).mean(),
               reproj_px_median=float(np.nanmedian(err)),
               reproj_pct_tinggi_badan=float(100 * np.nanmedian(err) / bh),
               view_per_sendi=float(nv[valid].mean()) if valid.any() else np.nan)
    lens = {}
    for name, (i, j) in BONES.items():
        L = np.linalg.norm(X[:, i] - X[:, j], axis=1)
        lens[name] = L[np.isfinite(L)]
    cv = [np.std(l) / np.mean(l) for l in lens.values() if len(l) > 5]
    out["cv_panjang_tulang_pct"] = float(100 * np.median(cv)) if cv else np.nan
    asym = [abs(np.median(lens[a]) - np.median(lens[b])) / np.mean([np.median(lens[a]), np.median(lens[b])])
            for a, b in PAIRS_LR if len(lens[a]) > 5 and len(lens[b]) > 5]
    out["asimetri_LR_pct"] = float(100 * np.median(asym)) if asym else np.nan
    # jitter: percepatan antar-frame (mm/frame^2), hanya untuk data rapat
    if fps:
        acc = np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=-1) * 1000
        out["jitter_mm"] = float(np.nanmedian(acc))
    return out


# ---------------------------------------------------------------- CLI

def seg_name(folder):
    """Nama segmen = <sesi>/<folder>, mis. Pak_Edy_.../skeleton_fall_L4."""
    parts = os.path.normpath(folder).split(os.sep)
    return "/".join(parts[-2:])


def parse_offsets(xs):
    return {k: float(v) for k, v in (x.split("=") for x in xs or [])}


def cmd_kalibrasi(a):
    cams2d = load_cams(a.folder, offsets=parse_offsets(a.offset))
    if a.kamera:
        cams2d = {k: v for k, v in cams2d.items() if k in a.kamera}
    grid = common_grid(cams2d)
    al = align(cams2d, grid, tol_ms=a.tol_ms, gap_ms=0)
    print(f"{len(grid)} momen, kamera {sorted(al)}")
    if a.f_tetap:
        a.f0 = a.f_tetap
    cams, X3, keys, err, per_cam = calibrate(al, (a.lebar, a.tinggi), a.f0, a.ambang_esensial_px,
                                             a.ambang_pnp_px, a.ambang_tri_px, a.ambang_akhir_px,
                                             json.load(open(a.awal))["kamera"] if a.awal else None,
                                             a.f_tetap)
    Xw = to_world(cams, X3, keys, a.tinggi_m)
    res = {"sumber": os.path.abspath(a.folder), "tinggi_badan_m": a.tinggi_m, "ukuran_gambar": [a.lebar, a.tinggi],
           "offset_ms": parse_offsets(a.offset), "n_titik_3d": len(Xw), "n_observasi": int(len(err)),
           "reproj_px_median": float(np.median(err)), "reproj_px_mean": float(err.mean()), "kamera": {}}
    print("\nhasil kalibrasi:")
    for n in sorted(cams):
        cm = cams[n]
        R = Rotation.from_rotvec(cm["rvec"]).as_matrix()
        C = -R.T @ cm["tvec"]
        fov = np.degrees(2 * np.arctan(a.lebar / 2 / cm["f"]))
        res["kamera"][n] = dict(f=float(cm["f"]), k1=float(cm["k1"]), c=cm["c"].tolist(), rvec=cm["rvec"].tolist(),
                                tvec=cm["tvec"].tolist(), posisi_m=C.tolist(), fov_horizontal_deg=float(fov),
                                reproj_px_median=float(np.median(per_cam[n])), n_observasi=int(len(per_cam[n])))
        print(f"  {n}: posisi (x,y,z) = ({C[0]:+.2f}, {C[1]:+.2f}, {C[2]:+.2f}) m, f = {cm['f']:.0f} px "
              f"(FOV {fov:.0f} deg), k1 = {cm['k1']:+.3f}, reproj median {np.median(per_cam[n]):.1f} px")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    np.save(a.out.replace(".json", "_titik3d.npy"), Xw)
    print(f"\nreprojection error total: median {np.median(err):.1f} px, rata-rata {err.mean():.1f} px -> {a.out}")


def cmd_triangulasi(a):
    cams, js = load_calib(a.kalibrasi)
    cams2d = load_cams(a.folder, offsets=js.get("offset_ms"))
    use = a.kamera or sorted(cams)
    al = align({k: cams2d[k] for k in use}, common_grid({k: cams2d[k] for k in use}, a.fps))
    grid = common_grid({k: cams2d[k] for k in use}, a.fps)
    X, err, nv = triangulate_seq(al, cams, use, a.maks_err_px, a.maks_err_tinggi)
    np.savez_compressed(a.out, t_ms=grid, X=X, reproj_px=err, n_view=nv, kamera=np.array(use), joints=np.array(JOINTS))
    m = metrics(X, err, nv, al, use, a.fps)
    print(f"{len(grid)} frame, kamera {use} -> {a.out}")
    for k, v in m.items():
        print(f"  {k}: {v:.2f}")


def cmd_ablasi(a):
    import pandas as pd
    cams, js = load_calib(a.kalibrasi)
    rows = []
    for folder in a.folder:
        cams2d = load_cams(folder, offsets=js.get("offset_ms"))
        names = sorted(set(cams) & set(cams2d))
        grid = common_grid({k: cams2d[k] for k in names}, a.fps)
        al = align({k: cams2d[k] for k in names}, grid)
        seg = seg_name(folder)
        res = {}
        for r in range(2, len(names) + 1):
            for use in itertools.combinations(names, r):
                res[use] = triangulate_seq(al, cams, list(use), a.maks_err_px, a.maks_err_tinggi)
        # sendi/frame yang berhasil di SEMUA kombinasi -> pembanding yang adil untuk akurasi
        common = np.logical_and.reduce([np.isfinite(X[..., 0]) for X, _, _ in res.values()])
        for use, (X, err, nv) in res.items():
            m = metrics(X, err, nv, al, use, a.fps)
            Xc = np.where(common[..., None], X, np.nan)
            mc = metrics(Xc, np.where(common, err, np.nan), nv, al, use, a.fps)
            rows.append(dict(segmen=seg, n_kamera=len(use), kamera="+".join(use), **m,
                             **{f"{k}_bersama": v for k, v in mc.items()
                                if k in ("reproj_px_median", "cv_panjang_tulang_pct", "asimetri_LR_pct")}))
            print(f"{seg:>24} {'+'.join(use):>20}  cakupan {m['cakupan_sendi_pct']:5.1f}%  "
                  f"lengkap {m['frame_lengkap_pct']:5.1f}%  reproj {m['reproj_px_median']:5.1f}px  "
                  f"CV tulang {m['cv_panjang_tulang_pct']:5.1f}% (bersama {mc['cv_panjang_tulang_pct']:5.1f}%)",
                  flush=True)
        print(f"{seg}: {100 * common.mean():.1f}% sendi berhasil di semua kombinasi (dasar metrik *_bersama)")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    cols = ["cakupan_sendi_pct", "frame_lengkap_pct", "reproj_px_median", "cv_panjang_tulang_pct",
            "cv_panjang_tulang_pct_bersama", "asimetri_LR_pct"]
    print("\nrata-rata per jumlah kamera:")
    print(df.groupby(["segmen", "n_kamera"])[cols].mean().round(1).to_string())
    print(f"-> {a.out}")


def cmd_ketersediaan(a):
    """Metrik tanpa kalibrasi: seberapa banyak sendi yang terlihat yakin di >= 2 / >= 3 kamera
    (batas atas sendi yang bisa ditriangulasi), per kombinasi jumlah kamera."""
    import pandas as pd
    rows = []
    for folder in a.folder:
        cams2d = load_cams(folder)
        names = sorted(cams2d)
        grid = common_grid(cams2d, a.fps)
        al = align(cams2d, grid)
        seg = seg_name(folder)
        for c in names:
            rows.append(dict(segmen=seg, n_kamera=1, kamera=c, deteksi_2d_pct=100 * al[c][1].mean()))
        for r in range(2, len(names) + 1):
            for use in itertools.combinations(names, r):
                nv = sum(al[c][1].astype(int) for c in use)
                rows.append(dict(segmen=seg, n_kamera=r, kamera="+".join(use),
                                 sendi_min2view_pct=100 * (nv >= 2).mean(),
                                 sendi_min3view_pct=100 * (nv >= 3).mean(),
                                 frame_lengkap_min2view_pct=100 * (nv >= 2).all(1).mean(),
                                 view_per_sendi=float(nv.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["segmen", "n_kamera"]).mean(numeric_only=True).round(1).to_string())
    print(f"-> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kalibrasi")
    k.add_argument("folder")
    k.add_argument("--out", required=True)
    k.add_argument("--kamera", nargs="*")
    k.add_argument("--tinggi-m", type=float, default=1.65, help="tinggi badan subjek (untuk skala)")
    k.add_argument("--f0", type=float, default=900, help="tebakan awal focal length (piksel)")
    k.add_argument("--lebar", type=int, default=1280)
    k.add_argument("--tinggi", type=int, default=720)
    k.add_argument("--tol-ms", type=float, default=100, help="selisih waktu maks antar kamera untuk satu momen")
    k.add_argument("--offset", nargs="*", metavar="CAM=MS", help="koreksi waktu kamera (ms, + = video telat)")
    k.add_argument("--f-tetap", type=float, help="kunci focal length semua kamera (piksel), mis. dari spesifikasi")
    k.add_argument("--awal", help="kalibrasi awal (json) untuk diperhalus, mis. dari sesi lain di ruangan sama")
    k.add_argument("--ambang-esensial-px", type=float, default=6, help="ambang RANSAC matriks esensial (px)")
    k.add_argument("--ambang-pnp-px", type=float, default=15, help="ambang RANSAC PnP saat menambah kamera (px)")
    k.add_argument("--ambang-tri-px", type=float, default=40, help="ambang buang view saat triangulasi awal (px)")
    k.add_argument("--ambang-akhir-px", type=float, default=15, help="ambang buang view sebelum BA terakhir (px)")
    for name in ("triangulasi", "ablasi"):
        p = sub.add_parser(name)
        p.add_argument("folder", nargs="+" if name == "ablasi" else None)
        p.add_argument("--kalibrasi", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--fps", type=float, default=15)
        p.add_argument("--maks-err-px", type=float, default=None,
                       help="ambang piksel tetap untuk membuang view (default: pakai --maks-err-tinggi)")
        p.add_argument("--maks-err-tinggi", type=float, default=0.05,
                       help="ambang buang view = fraksi tinggi badan subjek di gambar (default 5%%)")
        if name == "triangulasi":
            p.add_argument("--kamera", nargs="*")
    q = sub.add_parser("ketersediaan", help="metrik tanpa kalibrasi: sendi terlihat di >= 2 kamera")
    q.add_argument("folder", nargs="+")
    q.add_argument("--out", required=True)
    q.add_argument("--fps", type=float, default=15)
    a = ap.parse_args()
    {"kalibrasi": cmd_kalibrasi, "triangulasi": cmd_triangulasi, "ablasi": cmd_ablasi,
     "ketersediaan": cmd_ketersediaan}[a.cmd](a)


if __name__ == "__main__":
    main()
