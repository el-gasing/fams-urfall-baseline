"""Baris perintah: ``python -m urfall_bench <perintah> -d configs/datasets/<nama>.yaml [-b ...] [-s a.b=c ...]``"""
from __future__ import annotations

import argparse
import datetime
import sys
import warnings
from pathlib import Path

import pandas as pd

from .config import load_config, resolve, results_dir

DEFAULT_BENCH = "configs/benchmarks/kwolek2014.yaml"


def _setup(args):
    from .datasets import get_adapter
    cfg = load_config(args.dataset, args.benchmark or DEFAULT_BENCH, args.set)
    return cfg, get_adapter(cfg)


def cmd_list(args):
    from .datasets import available
    print("adapter dataset terdaftar:", ", ".join(available()))
    print("config dataset:", ", ".join(p.stem for p in sorted(resolve("configs/datasets").glob("*.yaml"))
                                       if not p.stem.startswith("_")))
    print("config benchmark:", ", ".join(p.stem for p in sorted(resolve("configs/benchmarks").glob("*.yaml"))))


def cmd_download(args):
    _, ad = _setup(args)
    ad.download()


def cmd_inspect(args):
    """Ringkasan data: urutan, kelompok, puncak SV_total dan % terpicu (per label bila ada)."""
    from .kwolek.trigger import triggers
    cfg, ad = _setup(args)
    seqs = ad.sequences()
    tr = cfg["trigger"]
    df = pd.DataFrame([dict(sequence=s.id, group=s.group, is_fall=s.is_fall, duration_s=(s.t_acc[-1] - s.t_acc[0]) / 1000,
                            sv_max=float(s.sv.max()), sv_min=float(s.sv.min()),
                            triggered=len(triggers(s.t_acc, s.sv, tr["threshold_g"], tr["merge_ms"])) > 0,
                            label=s.info.get("label")) for s in seqs])
    print(f"\n{len(df)} urutan ({int(df.is_fall.sum())} jatuh, {int((~df.is_fall).sum())} ADL), "
          f"{df.group.nunique()} kelompok CV")
    key = "label" if df.label.notna().any() else "is_fall"
    agg = df.groupby(key).agg(n=("sequence", "size"), duration_s=("duration_s", "mean"), sv_max_median=("sv_max", "median"),
                              sv_max_max=("sv_max", "max"), pct_triggered=("triggered", lambda x: 100 * x.mean()))
    if key == "label":
        agg.insert(0, "type", ["FALL" if i in ad.fall_classes else "ADL" for i in agg.index])
        agg.insert(0, "name", [ad.classes.get(int(i), str(i)) for i in agg.index])
    pd.set_option("display.width", 200)
    print(f"\nper {key} (SV dalam g, pemicu > {tr['threshold_g']} g):")
    print(agg.round(2).to_string())
    out = results_dir(cfg) / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sequences.csv", index=False)
    agg.round(2).to_csv(out / f"per_{key}.csv")
    print(f"\n-> {out}")


def cmd_extract_3d(args):
    from .pose3d.extract import extract
    cfg, ad = _setup(args)
    extract(cfg, ad)


def cmd_run(args):
    from .kwolek.runner import run
    cfg, ad = _setup(args)
    bench = Path(args.benchmark or DEFAULT_BENCH).stem
    out = resolve(args.out) if args.out else \
        results_dir(cfg) / f"{bench}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run(cfg, ad, out)
    print(f"\n-> {out}")


def cmd_report(args):
    from .kwolek.report import write
    print(write(resolve(args.run)))


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(prog="python -m urfall_bench",
                                 description="Baseline UR Fall (Kwolek & Kepski 2014) untuk berbagai dataset jatuh")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, helptext, dataset=True):
        p = sub.add_parser(name, help=helptext)
        if dataset:
            p.add_argument("-d", "--dataset", required=True, help="configs/datasets/<nama>.yaml")
            p.add_argument("-b", "--benchmark", help=f"configs/benchmarks/<nama>.yaml (default {DEFAULT_BENCH})")
            p.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE",
                           help="override config, mis. -s trigger.threshold_g=2.5 (boleh berkali-kali)")
        p.set_defaults(fn=fn)
        return p

    add("list", cmd_list, "adapter dan config yang tersedia", dataset=False)
    add("download", cmd_download, "unduh dataset (UR Fall)")
    add("inspect", cmd_inspect, "ringkasan urutan, puncak SV_total, % terpicu per label")
    add("extract-3d", cmd_extract_3d, "skeleton 3D multi-kamera di sekitar tiap pemicu (FAMS, GPU)")
    add("run", cmd_run, "benchmark: threshold + mode pose + mode kandidat").add_argument(
        "--out", help="folder run (default results/<dataset>/<benchmark>_<waktu>)")
    add("report", cmd_report, "tulis ulang report.md dari CSV sebuah run", dataset=False).add_argument(
        "--run", required=True, help="folder run")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
