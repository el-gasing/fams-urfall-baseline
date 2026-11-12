"""Fungsi bersama (numpy saja, tanpa PyAV/OpenCV) untuk memilih subjek dari hasil ekstrak_skeleton.py."""
import numpy as np


def track_subject(d, skor_min):
    """Pilih indeks orang yang sama di tiap frame: mulai dari kotak terbesar ber-skor tinggi,
    lalu tiap frame ambil orang yang pusat kotaknya paling dekat dengan frame sebelumnya."""
    boxes, scores = d["boxes"], d["scores"]
    F = len(scores)
    pick = np.full(F, -1)
    prev, areas, miss = None, [], 0
    for f in range(F):
        ok = np.flatnonzero(np.nan_to_num(scores[f], nan=0) >= skor_min)
        if len(ok) == 0:
            continue
        b = boxes[f, ok]
        ctr = (b[:, :2] + b[:, 2:]) / 2
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        if prev is None or miss > 10:  # awal, atau subjek lama tak terlihat -> kotak terbesar
            k = np.argmax(area)
        else:
            dist = np.linalg.norm(ctr - prev, axis=1)
            k = np.argmin(dist)
            if dist[k] > 400:
                # lompatan besar (mis. saat jatuh): tangkap ulang kotak terbesar, tapi hanya kalau
                # ukurannya mirip subjek -- supaya tidak pindah ke operator/orang lain yang kecil
                k = np.argmax(area)
                if area[k] < 0.6 * np.median(areas[-15:]):
                    miss += 1
                    continue
        miss = 0
        pick[f] = ok[k]
        prev = ctr[k]
        areas.append(area[k])
    return pick


def subject_joints(d, pick, kp_min, max_gap_s):
    """Kembalikan xy (F,12,2), status (F,12): 2 = terdeteksi, 1 = hasil isi celah, 0 = tidak ada."""
    F = len(pick)
    xy = np.full((F, 12, 2), np.nan)
    st = np.zeros((F, 12), int)
    for f in np.flatnonzero(pick >= 0):
        k = d["kpts"][f, pick[f]]
        good = k[:, 2] > kp_min
        xy[f, good] = k[good, :2]
        st[f, good] = 2
    t = d["t_ms"] / 1000
    for j in range(12):
        idx = np.flatnonzero(st[:, j] == 2)
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1 and t[b] - t[a] <= max_gap_s:
                w = (t[a + 1:b] - t[a]) / (t[b] - t[a])
                xy[a + 1:b, j] = xy[a, j] + w[:, None] * (xy[b, j] - xy[a, j])
                st[a + 1:b, j] = 1
    return xy, st
