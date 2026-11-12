# Menjalankan di server GPU (rsync + tmux)

Hanya `extract-3d` yang butuh GPU. `download`, `inspect`, `run` dan `report` jalan dalam hitungan detik
di laptop. Ganti `USER@HOST` dan `PORT` dengan server kamu. `[laptop]` = terminal laptop,
`[server]` = setelah `ssh`.

> **Selalu awali perintah laptop dengan `cd ... &&`.** Di rsync, `./` berarti "folder saat ini". Kalau
> dijalankan dari folder yang salah, seluruh folder itu ikut terkirim. Dengan `&&`, tidak ada yang
> terkirim kalau `cd` gagal. Untuk melihat dulu apa yang akan dikirim, tambahkan `-n` (dry run).

## 1. Kirim kode (kecil, ulangi setiap kali kode berubah)

```bash
# [laptop]
cd ~/Documents/datasetpaper/fams_urfall_baseline && rsync -avP -e "ssh -p PORT" \
  --exclude data/ --exclude cache/ --exclude results/ --exclude __pycache__/ --exclude .git/ \
  ./ USER@HOST:~/fams_urfall_baseline/
```

## 2. Kirim data FAMS

Cukup zip ekspor sesinya (video ada di dalamnya). Folder tujuan dibuat dulu:

```bash
# [server] sekali
mkdir -p ~/fams_urfall_baseline/data/fams
# [laptop]
cd ~/Documents/datasetpaper && rsync -avP -e "ssh -p PORT" ./*.zip USER@HOST:~/fams_urfall_baseline/data/fams/
```

UR Fall diunduh langsung di server: `python -m urfall_bench download -d configs/datasets/urfall.yaml`.

## 3. Siapkan lingkungan (sekali)

```bash
# [server]
cd ~/fams_urfall_baseline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pasang torch yang sesuai CUDA server bila perlu
python -m pytest tests -q           # cek cepat, tanpa data
nvidia-smi                          # pastikan GPU terlihat
```

## 4. Jalankan di tmux

```bash
# [server]
tmux new -A -s fams
cd ~/fams_urfall_baseline && source .venv/bin/activate
python -m urfall_bench extract-3d -d configs/datasets/fams.yaml -s pose3d.device=cuda -s pose3d.batch=8 \
  2>&1 | tee -a cache/extract3d.log
# lepas: Ctrl-b lalu d. Masuk lagi: tmux attach -t fams
```

`extract-3d` aman diulang. Kalau terputus, jalankan perintah yang sama dan langkah yang sudah selesai
akan dilewati. Pantau dengan:

```bash
tail -f cache/fams/work/*/log.txt
ls cache/fams/pose3d/*/clip_*.npz | wc -l
```

Setelah selesai, jalankan benchmark (beberapa detik):

```bash
bash scripts/run_all.sh          # atau satu per satu, lihat README §4
```

## 5. Ambil hasil

```bash
# [laptop]
cd ~/Documents/datasetpaper/fams_urfall_baseline && rsync -avP -e "ssh -p PORT" \
  USER@HOST:~/fams_urfall_baseline/results/ ./results/
# opsional, skeleton 3D (untuk dianalisis ulang tanpa GPU):
cd ~/Documents/datasetpaper/fams_urfall_baseline && rsync -avP -e "ssh -p PORT" \
  USER@HOST:~/fams_urfall_baseline/cache/fams/pose3d/ ./cache/fams/pose3d/
```

`cache/fams/work/` (zip yang diekstrak dan skeleton 2D sementara) bisa dihapus di server setelah
`cache/fams/pose3d/` lengkap.
