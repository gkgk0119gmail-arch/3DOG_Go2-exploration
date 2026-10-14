"""YAML configuration for the POSE-inspection pipeline.

Every tunable number lives in ``configs/pose_inspection/*.yaml``. A file may name parents with ``_base_: [a.yaml, ...]``
(paths relative to the file); parents are merged first, then the file itself (dicts merge recursively, everything else
replaces). ``load_config(path, overrides)`` also applies dotted overrides such as ``{"planner.posture.enabled": False}``.

The result is an attribute-access dict (``cfg.planner.posture.enabled``) that still behaves like a plain dict, so it
can be dumped next to every run for provenance.
"""

from __future__ import annotations

import copy
import glob
import os
from pathlib import Path

import yaml

SIM_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = SIM_DIR / "configs" / "pose_inspection"


class Cfg(dict):
    """dict with attribute access (recursively wraps nested dicts)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def __deepcopy__(self, memo):
        return Cfg(copy.deepcopy(dict(self), memo))


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    bases = d.pop("_base_", [])
    merged: dict = {}
    for b in [bases] if isinstance(bases, str) else bases:
        merged = _merge(merged, _load((path.parent / b).resolve()))
    return _merge(merged, d)


def set_dotted(d: dict, key: str, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> Cfg:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    d = _load(p.resolve())
    for k, v in (overrides or {}).items():
        set_dotted(d, k, v)
    cfg = _wrap(d)
    cfg["_source_"] = str(p.resolve())
    return cfg


def sim_path(p: str) -> str:
    """Resolve a path relative to sim/ (glob patterns allowed: the first match is returned)."""
    q = p if os.path.isabs(p) else str(SIM_DIR / p)
    if any(c in q for c in "*?["):
        hits = sorted(glob.glob(q, recursive=True))
        if not hits:
            raise FileNotFoundError(q)
        return hits[0]
    return q
