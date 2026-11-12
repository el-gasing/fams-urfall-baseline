#!/usr/bin/env bash
# Seluruh perbandingan UR Fall vs FAMS dengan baseline Kwolek & Kepski 2014.
#   bash scripts/jalankan_semua.sh [folder_sesi_fams] [-- argumen tambahan untuk fams3d/jalankan.py]
# Langkah skeleton 3D (2) butuh GPU dan lama; langkah lain beberapa detik.
set -eu
cd "$(dirname "$0")/.."
FAMS="${1:-data/fams}"; shift || true; [ "${1:-}" = "--" ] && shift
PY="${PY:-python}"

# 1. UR Fall: replikasi Tabel 1-2 paper + mode kandidat
[ -s data/urfall/urfall-cam0-falls.csv ] || bash scripts/download_urfall.sh data/urfall
$PY jalankan_urfall.py --data data/urfall --out hasil/urfall
$PY jalankan_urfall.py --data data/urfall --out hasil/urfall_p40 --fitur +P40

# 2. FAMS: skeleton 3D di sekitar tiap pemicu 3 g (GPU)
$PY fams3d/jalankan.py "$FAMS" --out hasil/fams_3d --kerja kerja/fams "$@"

# 3. FAMS: threshold + verifikasi pose dari skeleton 3D
$PY jalankan_fams.py "$FAMS" --out hasil/fams_threshold
$PY jalankan_fams.py "$FAMS" --skeleton3d hasil/fams_3d --out hasil/fams
$PY jalankan_fams.py "$FAMS" --skeleton3d hasil/fams_3d --out hasil/fams_p40 --fitur +P40
echo "selesai: hasil/urfall/hasil.md dan hasil/fams/hasil.md"
