"""X14's advantage-normalisation BIAS, before and after the fix (core review 2026-10-20 A2; `gen3_overlap_norm_then_weight_v1`).

Runs the harness `agents.training.rust_rollout.overlap_test._decoupled_logit_push` — a real toy-learner ``train()``
under a decoupled update, reading what R1 is handed — against WHICHEVER tree's ``src`` is first on ``PYTHONPATH``.
The harness file itself is loaded by path from ``--harness`` (the fixed tree's test file), and it touches only names
that exist before the fix, so the same harness measures both trees:

    # after (the fix's worktree)
    PYTHONPATH=<fixed>/src  python adv_norm_bias.py --harness <fixed>/src/agents/training/rust_rollout/overlap_test.py --label after
    # before (a checkout of the parent commit)
    PYTHONPATH=<parent>/src python adv_norm_bias.py --harness <fixed>/src/agents/training/rust_rollout/overlap_test.py --label before

The fixture: mu = (0.6, 0.4), pi_prox = (0.5, 0.5) over the toy's two actions (TV 0.1; w = (0.833, 1.25), below the
cap), an advantage that carries NO information about the action (mean 1, std 1), so the right push is ZERO. Cells:
  * ``stratified``: exact proportions, one micro-batch of 200 rows — deterministic, rounding only;
  * ``sampled_s<seed>``: 20,000 rows drawn i.i.d., ten micro-batches of 2,048 (production's micro-batch size, the
    last one ragged), per-micro-batch normalisation as in production — the push carries sampling noise of
    ~1/sqrt(rows) ≈ 0.007, so it is read against the closed form (E[A]/S)(pi_prox − mu) of the as-coded order.
Writes ``result/adv_norm_bias_<label>.json``. CPU only; seconds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np


def _harness(path: str):
    spec = importlib.util.spec_from_file_location("x14_bias_harness", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._decoupled_logit_push


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("result")))
    a = ap.parse_args()
    import agents.training.rust_rollout.overlap as ov
    push = _harness(a.harness)
    cells = {"stratified": dict()}
    for seed in (1, 2, 3):
        cells[f"sampled_s{seed}"] = dict(n_steps=2000, n_envs=10, batch_size=2048, seed=seed)
    out = {"label": a.label, "overlap_module": ov.__file__,
           "has_decoupled_advantages": hasattr(ov, "decoupled_advantages"), "cells": {}}
    for name, kw in cells.items():
        got = push(**kw)
        out["cells"][name] = {"push": [float(x) for x in got["push"]],
                              "closed_form_as_coded": [float(x) for x in got["closed_form"]],
                              "normalize_in_r1": bool(got["normalize_in_r1"]), "n_micro": int(got["n_micro"]),
                              "max_abs_push": float(np.abs(got["push"]).max())}
        print(f"{a.label:7s} {name:12s} push {np.array2string(got['push'], precision=5)}  "
              f"closed form (as coded) {np.array2string(got['closed_form'], precision=5)}  "
              f"R1 normalises: {got['normalize_in_r1']}  micro-batches {got['n_micro']}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"adv_norm_bias_{a.label}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
