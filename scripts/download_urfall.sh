#!/usr/bin/env bash
# Unduh UR Fall Detection Dataset (Kwolek & Kepski 2014) ke data/urfall/ (atau folder argumen pertama).
# Sumber: https://fenix.ur.edu.pl/mkepski/ds/uf.html  (lisensi CC BY-NC-SA 4.0)
#   bash scripts/download_urfall.sh [folder]          # CSV + video cam0 (~110 MB)
#   CAM1=1 bash scripts/download_urfall.sh [folder]   # + video cam1 (kamera atap, urutan jatuh saja)
# Aman diulang: file yang sudah ada dilewati.
set -u
BASE="https://fenix.ur.edu.pl/mkepski/ds/data"
DEST="${1:-$(dirname "$0")/../data/urfall}"
mkdir -p "$DEST" && cd "$DEST" || exit 1

files=(urfall-cam0-falls.csv urfall-cam0-adls.csv)
for i in $(seq -w 1 30); do
  files+=("fall-$i-acc.csv" "fall-$i-data.csv" "fall-$i-cam0.mp4")
  [ "${CAM1:-0}" = 1 ] && files+=("fall-$i-cam1.mp4")
done
for i in $(seq -w 1 40); do
  files+=("adl-$i-acc.csv" "adl-$i-data.csv" "adl-$i-cam0.mp4")
done

gagal=(); n=${#files[@]}; k=0
for f in "${files[@]}"; do
  k=$((k+1)); [ -s "$f" ] && continue
  printf "[%d/%d] %s\n" "$k" "$n" "$f"
  if curl -fL --retry 3 -C - -o "$f.part" "$BASE/$f"; then mv "$f.part" "$f"; else gagal+=("$f"); fi
done
echo "Selesai: $(ls *.csv *.mp4 2>/dev/null | wc -l | tr -d ' ') file di $(pwd)"
[ ${#gagal[@]} -gt 0 ] && echo "GAGAL (${#gagal[@]}): ${gagal[*]} -- jalankan ulang."
exit 0
