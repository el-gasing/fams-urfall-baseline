"""
Ekstrak skeleton 2D dari video multi-kamera satu sesi FAMS.

Cara pakai:
    .venv/bin/python ekstrak_skeleton.py <folder_sesi> [--kamera cam1 cam2] [--batch 8]
        [--mulai-s 0 --durasi-s 60] [--putar cam4=90] [--video-cek]

Model: Keypoint R-CNN (torchvision keypointrcnn_resnet50_fpn, bobot COCO17).
Titik wajah (nose, mata, telinga = 5 titik pertama COCO) dibuang, jadi tersisa 12 titik badan.

Alur:
  1. Baca cameras.json -> daftar kamera + started_at_ms.
  2. Decode tiap video (PyAV), waktu dunia tiap frame = started_at_ms + waktu frame di video.
  3. Deteksi semua orang per frame (multi-person), simpan maks --maks-orang dengan skor tertinggi.
  4. Simpan per kamera: <folder_sesi>/skeleton/<sid>_<cam>_skeleton.npz
       t_ms      (F,)          waktu dunia tiap frame (ms, jam yang sama dengan IMU)
       frame_idx (F,)          nomor frame di video
       kpts      (F, P, 12, 3) x, y (piksel), skor keyakinan tiap titik; NaN = tidak ada orang
       boxes     (F, P, 4)     x1, y1, x2, y2
       scores    (F, P)        skor deteksi orang
       joints    (12,)         nama titik
     Ditulis bertahap (checkpoint), jadi bisa dilanjutkan kalau terputus.
  5. Opsional --video-cek: video mp4 dengan skeleton digambar di atasnya, untuk cek visual.

Triangulasi ke 3D (butuh kalibrasi kamera) dikerjakan terpisah.
"""
import argparse
import json
import os
import time

import av
import numpy as np
import torch
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

COCO_KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
FACE = 5  # 5 titik pertama = wajah, dibuang
JOINTS = COCO_KPTS[FACE:]
# tulang untuk digambar (indeks dalam JOINTS, wajah sudah dibuang)
BONES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
         (6, 8), (8, 10), (7, 9), (9, 11)]


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device, min_size):
    m = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, min_size=min_size)
    return m.eval().to(device)


def unrotate(xy, rot, H, W):
    """Koordinat di frame yang sudah diputar (rot derajat searah jarum jam) -> koordinat frame asli."""
    x, y = xy[..., 0].copy(), xy[..., 1].copy()
    if rot == 90:
        xy[..., 0], xy[..., 1] = y, H - x
    elif rot == 180:
        xy[..., 0], xy[..., 1] = W - x, H - y
    elif rot == 270:
        xy[..., 0], xy[..., 1] = W - y, x
    return xy


def run_batch(model, imgs, device, max_people, min_score, rot=0):
    """imgs: list of HxWx3 uint8. Kembalikan kpts (B,P,12,3), boxes (B,P,4), scores (B,P).
    rot: putar frame sebelum inference (untuk kamera yang dipasang miring); koordinat hasil
    tetap dalam frame asli."""
    B = len(imgs)
    H, W = imgs[0].shape[:2]
    if rot:
        imgs = [np.ascontiguousarray(np.rot90(i, k=-rot // 90)) for i in imgs]
    kp = np.full((B, max_people, len(JOINTS), 3), np.nan, np.float32)
    bx = np.full((B, max_people, 4), np.nan, np.float32)
    sc = np.full((B, max_people), np.nan, np.float32)
    x = [torch.from_numpy(i).to(device).permute(2, 0, 1).float().div_(255) for i in imgs]
    with torch.inference_mode():
        outs = model(x)
    for b, o in enumerate(outs):
        keep = (o["scores"] >= min_score).nonzero().flatten()[:max_people]  # sudah urut skor
        n = len(keep)
        if n == 0:
            continue
        k = o["keypoints"][keep, FACE:, :2]
        s = o["keypoints_scores"][keep, FACE:, None]
        kp[b, :n] = torch.cat([k, s], -1).cpu().numpy()
        bx[b, :n] = o["boxes"][keep].cpu().numpy()
        sc[b, :n] = o["scores"][keep].cpu().numpy()
    if rot:
        unrotate(kp[..., :2], rot, H, W)
        c = unrotate(bx.reshape(B, max_people, 2, 2), rot, H, W)
        bx = np.concatenate([c.min(2), c.max(2)], -1)
    return kp, bx, sc


def draw(img, kpts, min_kp):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for p, person in enumerate(kpts):
        if np.isnan(person[0, 0]):
            continue
        col = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 200, 60), (220, 120, 255)][p % 5]
        for a, b in BONES:
            if person[a, 2] > min_kp and person[b, 2] > min_kp:
                d.line([tuple(person[a, :2]), tuple(person[b, :2])], fill=col, width=4)
        for x, y, c in person:
            if c > min_kp:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)
    return np.asarray(im)


def process_camera(cam, folder, sid, out_dir, model, device, a):
    out_path = os.path.join(out_dir, f"{sid}_{cam['cam_id']}_skeleton.npz")
    part_path = out_path.replace(".npz", ".part.npz")
    if os.path.exists(out_path) and not a.ulang:
        print(f"[{cam['cam_id']}] sudah ada, lewati ({out_path})")
        return

    # lanjutkan dari checkpoint kalau ada
    done = {"t_ms": [], "frame_idx": [], "kpts": [], "boxes": [], "scores": []}
    if os.path.exists(part_path) and not a.ulang:
        d = np.load(part_path)
        done = {k: [d[k]] for k in done}
        print(f"[{cam['cam_id']}] lanjut dari frame {int(d['frame_idx'][-1]) + 1}")
    last_idx = int(done["frame_idx"][0][-1]) if done["frame_idx"] else -1

    c = av.open(os.path.join(folder, cam["file"]))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    writer = None
    if a.video_cek:
        writer = av.open(out_path.replace(".npz", "_cek.mp4"), "w")
        vs = writer.add_stream("libx264", rate=15)
        vs.width, vs.height, vs.pix_fmt = s.width // 2, s.height // 2, "yuv420p"

    t_lo = a.mulai_s
    rentang = [tuple(float(v) for v in r.split(":")) for r in a.rentang]
    t_hi = a.mulai_s + a.durasi_s if a.durasi_s else float("inf")
    buf_img, buf_t, buf_i = [], [], []
    n_done, t_start, last_save = 0, time.time(), time.time()

    def flush():
        nonlocal n_done
        kp, bx, sc = run_batch(model, buf_img, device, a.maks_orang, a.skor_min, a.putar.get(cam["cam_id"], 0))
        done["t_ms"].append(np.array(buf_t)); done["frame_idx"].append(np.array(buf_i))
        done["kpts"].append(kp); done["boxes"].append(bx); done["scores"].append(sc)
        if writer:
            for img, k in zip(buf_img, kp):
                f = av.VideoFrame.from_ndarray(draw(img, k, a.kp_min), format="rgb24").reformat(
                    width=vs.width, height=vs.height)
                for pkt in vs.encode(f):
                    writer.mux(pkt)
        n_done += len(buf_img)
        buf_img.clear(); buf_t.clear(); buf_i.clear()

    def save(path):
        np.savez_compressed(path, joints=np.array(JOINTS), cam_id=cam["cam_id"],
                            started_at_ms=cam["started_at_ms"],
                            **{k: np.concatenate(v) if v else np.empty(0) for k, v in done.items()})

    # --interval-s: ambil frame terdekat dengan titik grid jam dunia (kelipatan interval),
    # jadi semua kamera mengambil momen yang sama (untuk kalibrasi/triangulasi)
    step = a.interval_s * 1000
    prev = None
    for i, f in enumerate(c.decode(s)):
        if i <= last_idx or f.time < t_lo:
            prev = None
            continue
        if rentang and not any(lo <= f.time < hi for lo, hi in rentang):
            prev = None
            continue
        if f.time >= t_hi:
            break
        tw = cam["started_at_ms"] + f.time * 1000
        if step:
            g = np.ceil(tw / step) * step if prev is None else np.floor(tw / step) * step
            if prev is None or g <= prev[1]:
                prev = (i, tw, f)
                continue
            # titik grid g ada di antara frame sebelumnya dan frame ini -> pilih yang lebih dekat
            pi, pt, pf = prev
            prev = (i, tw, f)
            i, tw, f = (pi, pt, pf) if abs(pt - g) < abs(tw - g) else (i, tw, f)
            if buf_i and i == buf_i[-1]:
                continue
        buf_img.append(f.to_ndarray(format="rgb24"))
        buf_t.append(tw)
        buf_i.append(i)
        if len(buf_img) == a.batch:
            flush()
            if time.time() - last_save > 120:
                save(part_path); last_save = time.time()
            if n_done % (a.batch * 25) == 0:
                el = time.time() - t_start
                print(f"[{cam['cam_id']}] {n_done} frame, t_video={f.time:.0f}s, {n_done / el:.1f} fps", flush=True)
    if buf_img:
        flush()
    c.close()
    if writer:
        for pkt in vs.encode():
            writer.mux(pkt)
        writer.close()
    save(out_path)
    if os.path.exists(part_path):
        os.remove(part_path)
    el = time.time() - t_start
    print(f"[{cam['cam_id']}] selesai: {n_done} frame dalam {el / 60:.1f} menit ({n_done / max(el, 1e-9):.1f} fps) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--kamera", nargs="*", help="default: semua kamera di cameras.json")
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--min-size", type=int, default=720, help="sisi pendek input model (default torchvision 800)")
    ap.add_argument("--maks-orang", type=int, default=5)
    ap.add_argument("--skor-min", type=float, default=0.5, help="skor minimum deteksi orang")
    ap.add_argument("--kp-min", type=float, default=2.0, help="ambang skor titik untuk digambar di video cek")
    ap.add_argument("--mulai-s", type=float, default=0.0, help="mulai dari detik ke- (waktu video)")
    ap.add_argument("--durasi-s", type=float, default=0.0, help="0 = sampai habis")
    ap.add_argument("--rentang", nargs="*", default=[], metavar="A:B",
                    help="hanya proses frame di rentang waktu video ini (detik); frame lain hanya di-decode")
    ap.add_argument("--interval-s", type=float, default=0.0,
                    help="ambil 1 frame tiap N detik (grid jam dunia, sama untuk semua kamera); 0 = semua frame")
    ap.add_argument("--putar", nargs="*", default=[], metavar="CAM=DERAJAT",
                    help="putar frame searah jarum jam sebelum inference, mis. cam4=90 (90/180/270)")
    ap.add_argument("--video-cek", action="store_true")
    ap.add_argument("--ulang", action="store_true", help="abaikan hasil/checkpoint lama")
    ap.add_argument("--out", default=None, help="default: <folder>/skeleton")
    a = ap.parse_args()
    a.putar = {k: int(v) for k, v in (x.split("=") for x in a.putar)}

    cams = json.load(open(os.path.join(a.folder, "cameras.json")))["cameras"]
    sid = cams[0]["session_id"]
    if a.kamera:
        cams = [c for c in cams if c["cam_id"] in a.kamera]
    out_dir = a.out or os.path.join(a.folder, "skeleton")
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device(a.device)
    print(f"device: {device}, kamera: {[c['cam_id'] for c in cams]}, putar: {a.putar}")
    model = load_model(device, a.min_size)
    for cam in sorted(cams, key=lambda c: c["cam_id"]):
        process_camera(cam, a.folder, sid, out_dir, model, device, a)


if __name__ == "__main__":
    main()
