"""Shared helpers for the N2 exact-set identity test (2026-10-26). CPU only, read-only on models/.

Reuses the architecture-generalisation review's helpers (`arch_generalisation_2026-10-26/scripts`: `common`,
`static_probe`) and its archived decision rows (sha256-pinned there)."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
AG = HERE.parents[1] / "arch_generalisation_2026-10-26"
sys.path.insert(0, str(AG / "scripts"))
import common as C  # noqa: E402  (the review's: PSTAR, panel texts, pack, load_model, forward)

ROWS_DIR = Path(os.environ.get("N2_ROWS_DIR", "/home/goodlad/gen3ai_archive/arch_generalisation_2026-10-26"))
#: the release cdylib the probe's Rust encodes go through (see README, F-N2-H1)
LIB = Path(os.environ.get("N2_LIB", "/home/goodlad/dev/gen3ai/src/rust_env/target/release/libpokesim_env.so"))


def pinned_sha() -> dict:
    out = {}
    for ln in (AG / "rows_sha256.txt").read_text().split("\n"):
        if ln.strip():
            h, name = ln.split()
            out[name] = h
    return out


def load_rows(panel: str) -> dict:
    p = ROWS_DIR / f"rows_{panel}.npz"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    want = pinned_sha()[p.name]
    if h != want:
        raise RuntimeError(f"{p}: sha256 {h} != the pinned {want}")
    return dict(np.load(p))


def lib():
    from utils.rust_env import ffi
    return ffi.load(LIB)


C.lib = lib          # the review's helpers (static_probe.turn1_slots callers) take the lib we pass them
