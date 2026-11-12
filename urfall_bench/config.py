"""Config YAML: file dataset + file benchmark, digabung, ditambah override ``-s a.b=c``.

Urutan lapisan: configs/benchmarks/defaults.yaml <- benchmark (-b) <- dataset (-d) <- override (-s).
Path relatif di config diresolve terhadap akar proyek (folder yang berisi ``urfall_bench/``),
jadi perintah bisa dijalankan dari folder mana pun.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict, overrides) -> dict:
    """``["trigger.threshold_g=2.5", "verify.window_s=[0,2]"]`` -> isi bersarang (nilai diparse sebagai YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(dataset, benchmark=None, overrides=None) -> dict:
    """defaults <- benchmark <- dataset (setelah ``extends``) <- override.

    File dataset boleh diawali ``extends: lain.yaml`` untuk memakai ulang file dataset lain dan
    hanya mengubah yang berbeda (mis. urfall_paper.yaml = urfall.yaml + subset ADL).
    """
    cfg = load_yaml(PROJECT_ROOT / "configs" / "benchmarks" / "defaults.yaml")
    if benchmark:
        cfg = deep_merge(cfg, load_yaml(benchmark))
    ds = load_yaml(dataset)
    base = ds.pop("extends", None)
    if base:
        bp = Path(dataset).parent / base
        if not bp.exists():
            bp = PROJECT_ROOT / "configs" / "datasets" / base
        cfg = deep_merge(cfg, load_yaml(bp))
    cfg = deep_merge(cfg, ds)
    return apply_overrides(cfg, overrides)


def resolve(path) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def cache_dir(cfg: dict) -> Path:
    return resolve(cfg.get("cache_dir", "cache")) / cfg["dataset"]["name"]


def results_dir(cfg: dict) -> Path:
    return resolve(cfg.get("results_dir", "results")) / cfg["dataset"]["name"]
