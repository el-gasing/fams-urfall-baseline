#!/usr/bin/env bash
# Seluruh perbandingan UR Fall vs FAMS dengan baseline Kwolek & Kepski (2014).
#   bash scripts/run_all.sh [-- override untuk extract-3d, mis. -s pose3d.device=cuda]
# extract-3d butuh GPU dan lama; langkah lain beberapa detik. Hasil: results/<dataset>/<run>/report.md
set -eu
cd "$(dirname "$0")/.."
PY="${PY:-python}"
[ "${1:-}" = "--" ] && shift
B=configs/benchmarks

# 1. UR Fall: data + replikasi paper (40 ADL dan 30 ADL seperti paper), dengan dan tanpa P40
[ -s data/urfall/urfall-cam0-falls.csv ] || $PY -m urfall_bench download -d configs/datasets/urfall.yaml
for ds in urfall urfall_paper; do
  $PY -m urfall_bench run -d configs/datasets/$ds.yaml -b $B/kwolek2014.yaml --out results/$ds/kwolek2014
  $PY -m urfall_bench run -d configs/datasets/$ds.yaml -b $B/p40.yaml        --out results/$ds/p40
done

# 2. FAMS: ringkasan data, skeleton 3D di sekitar tiap pemicu (GPU), lalu benchmark
$PY -m urfall_bench inspect    -d configs/datasets/fams.yaml
$PY -m urfall_bench extract-3d -d configs/datasets/fams.yaml "$@"
$PY -m urfall_bench run -d configs/datasets/fams.yaml -b $B/kwolek2014.yaml --out results/fams/kwolek2014
$PY -m urfall_bench run -d configs/datasets/fams.yaml -b $B/p40.yaml        --out results/fams/p40
$PY -m urfall_bench run -d configs/datasets/fams.yaml -b $B/kfold.yaml      --out results/fams/kfold
echo "selesai: lihat results/*/*/report.md"
