"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

def _clean_argv():
    """Strip invisible characters that sneak in when commands are copy-pasted
    (non-breaking spaces, Unicode line/paragraph separators, zero-width spaces)."""
    bad = "\u00a0\u2028\u2029\u200b\ufeff"
    out = []
    for a in sys.argv:
        for ch in bad:
            a = a.replace(ch, " ")
        out += [x for x in a.split(" ") if x]
    sys.argv[:] = out


_clean_argv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


@dataclass
class Budget:
    """Simulation budgets. 'quick' runs on a laptop in well under an hour;
    'full' is the setting to report (use the M4 with --jobs)."""
    tune_E: int
    tune_T: int
    tune_warm: int
    val_E: int
    test_E: int
    test_T: int
    test_warm: int
    es_iters: int
    es_pairs: int
    es_seeds: int
    saa_E: int
    saa_T: int
    saa_warm: int
    saa_M: int

    @classmethod
    def preset(cls, name: str) -> "Budget":
        if name == "smoke":
            return cls(8, 300, 50, 8, 8, 600, 50, 3, 2, 2, 1, 40, 10, 4)
        if name == "quick":
            return cls(32, 1000, 100, 32, 64, 3000, 200, 30, 8, 3, 2, 250, 30, 8)
        if name == "full":
            return cls(64, 3000, 200, 64, 200, 10000, 500, 120, 16, 5, 8, 1000, 100, 20)
        raise ValueError(name)


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    return path


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """cols: (key, header, format)"""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _, fmt in cols:
            v = r.get(k, "")
            if isinstance(v, float) and fmt:
                cells.append(format(v, fmt) if np.isfinite(v) else "–")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
