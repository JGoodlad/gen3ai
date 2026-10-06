"""X5 A/B LOOK 1: the read, at the registered n (design_x5_belief_tokens.md §7.4, §7.6–§7.9). Reads the four
families' look-1 requests from the eval COUNT ledger through ``main.h2h.cross``'s declared reads, and writes
``result.json`` + ``result.md`` beside this file.

    PYTHONPATH=<tree>/src python read_look1.py [--root <ledger root>] [--progress]

``--progress`` prints per-family cell counts only (interim reads are PROGRESS, never a verdict). Without it every
read is made; any cell missing / short / aborted makes that read INCONCLUSIVE (never interpreted)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

from agents.training import eval_ledger as L
from main.h2h import cross as X

HERE = Path(__file__).resolve().parent
M = Path("/home/goodlad/dev/gen3ai/models")
P_OFF = "gen3_eval_protocol_v1_h2h"
P_ONE = "gen3_eval_protocol_v1_h2h_oracle_one_sided"
P_BOTH = "gen3_eval_protocol_v1_h2h_oracle_both_sided"
READS = {"steps": ("x5ab_strength_steps", "x5ab_look1_steps", P_OFF),
         "wall": ("x5ab_strength_wall", "x5ab_look1_wall", P_OFF),
         "one": ("x5ab_oracle_one_sided", "x5ab_look1_oracle_one", P_ONE),
         "both": ("x5ab_oracle_both_sided", "x5ab_look1_oracle_both", P_BOTH)}
FINAL = "final_model.zip"
RUNS = {"blob": ["rb_x5ab_blob_s1001", "rb_x5ab_blob_s1002", "rb_x5ab_blob_s1003"],
        "fm": ["rb_x5ab_fm_s1001", "rb_x5ab_fm_s1002", "rb_x5ab_fm_s1003"],
        "sp": ["rb_x5ab_oracle_sp_s1001b", "rb_x5ab_oracle_sp_s1002", "rb_x5ab_oracle_sp_s1003"],
        "full": ["rb_x5ab_oracle_full_s1001b", "rb_x5ab_oracle_full_s1002", "rb_x5ab_oracle_full_s1003"]}
FM12 = ["rb_x5ab_fm_s1001/checkpoints/checkpoint_12000021_steps.zip",
        "rb_x5ab_fm_s1002/checkpoints/checkpoint_12000070_steps.zip",
        "rb_x5ab_fm_s1003/checkpoints/checkpoint_12000119_steps.zip"]
#: the late-registered predictions (ledger 2026-10-05, `43e0955d`), 80 % intervals, pp
PRED = {"C_species": (4.0, 0.0, 10.0), "C_full": (8.0, 2.0, 16.0), "C_full_minus_C_species": (4.0, -2.0, 10.0),
        "both_minus_one_nonoracle": (None, -5.0, 0.0), "H": (None, 0.2, 0.6)}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def preconditions() -> Dict[str, Any]:
    """The look's checkable preconditions: every final is ≥ 15M steps (its checkpoint twin's step) and every run's
    metadata exists; the blob-path identity, init threads and restart counts are the BANK entries' (cited)."""
    fails: List[str] = []
    steps: Dict[str, int] = {}
    for arm, runs in RUNS.items():
        for r in runs:
            ck = sorted((M / r / "checkpoints").glob("checkpoint_*_steps.zip"),
                        key=lambda p: int(re.findall(r"(\d+)_steps", p.name)[0]))
            last = int(re.findall(r"(\d+)_steps", ck[-1].name)[0]) if ck else 0
            steps[r] = last
            if last < 15_000_000:
                fails.append(f"{r}: last checkpoint {last} < 15M")
            if not (M / r / FINAL).exists():
                fails.append(f"{r}: no {FINAL}")
    for p in FM12:
        if not (M / p).exists():
            fails.append(f"missing {p}")
    return {"failures": fails, "last_checkpoint_steps": steps,
            "cited": ["blob-path identity at 708dcb0a vs e5e660dd (ledger 2026-10-06 BANK, fixed_mass arm)",
                      "0 restarts / 0 crashes in all 12 look-1 runs (ledger BANK entries 2026-10-04/05/06)",
                      "init_num_threads not recorded (ledger 2026-10-04 BANK); accepted by construction there"]}


def shas(arm: str, wall: bool = False) -> List[str]:
    if wall:
        return [sha(M / p) for p in FM12]
    return [sha(M / r / FINAL) for r in RUNS[arm]]


def read(root: Any, key: str) -> List[Any]:
    fam, req, prot = READS[key]
    return X.look_cells(root, fam, req, prot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    root = Path(a.root) if a.root else L.archive_ledger_root()
    cells = {k: read(root, k) for k in READS}
    if a.progress:
        for k, cs in cells.items():
            done = sum(1 for c in cs if c.n_pairs >= X.MIN_PAIRS)
            print(f"{k}: {len(cs)} cells, {done} at >= {X.MIN_PAIRS} pairs, "
                  f"{sum(c.n_pairs for c in cs)} pairs, aborted {sum(c.aborted for c in cs)}")
        return 0
    pre = preconditions()
    S = {arm: shas(arm) for arm in RUNS}
    S["fm12"] = shas("fm", wall=True)
    out: Dict[str, Any] = {"preconditions": pre, "shas": S}

    def mat(key: str, rows: str, cols: str) -> X.MatrixFromCells:
        sub = [c for c in cells[key] if c.player in S[rows] and c.opponent in S[cols]]
        return X.matrix_from_cells(sub, S[rows], S[cols])

    # the two registered strength reads (each its own verdict)
    for name, key, rows in (("matched_steps", "steps", "fm"), ("matched_wall_time", "wall", "fm12")):
        m = mat(key, rows, "blob")
        d = X.summarize(1, m, pre["failures"])
        out[name] = {"decision": d.as_dict(), "verdict": X.verdict_text(d), "cells": list(m.cells)}

    # the oracle reads: C (one-sided primary) and the both-sided cells (descriptive)
    stats: Dict[str, Any] = {}
    for mode in ("one", "both"):
        for o in ("sp", "full"):
            for ref in ("blob", "fm"):
                m = mat(mode, o, ref)
                reasons = list(m.inconclusive) + pre["failures"]
                st = X.cross_stat(m.h) if m.h is not None and not reasons else None
                stats[f"{mode}:{o}:{ref}"] = st
                out[f"oracle_{mode}_{o}_vs_{ref}"] = {"inconclusive": reasons, "stat": st.as_dict() if st else None,
                                                     "cells": list(m.cells)}
    steps_stat = X.cross_stat(mat("steps", "fm", "blob").h) if out["matched_steps"]["decision"]["stat"] else None
    c_sp, c_full = stats["one:sp:blob"], stats["one:full:blob"]
    grades: Dict[str, Any] = {}
    if c_sp and c_full and steps_stat:
        floor = X.replicate_floor(steps_stat)
        out["floor"] = floor
        out["contrast_full_minus_species"] = X.shared_column_contrast(c_full, c_sp).as_dict()
        out["H_species"] = X.headroom(steps_stat, c_sp, floor["floor_pp"])
        out["H_full"] = X.headroom(steps_stat, c_full, floor["floor_pp"])
        for k, v in (("C_species", c_sp.delta_hat), ("C_full", c_full.delta_hat),
                     ("C_full_minus_C_species", c_full.delta_hat - c_sp.delta_hat)):
            p, lo, hi = PRED[k]
            grades[k] = {"predicted": p, "interval80": [lo, hi], "observed": v, "grade": X.grade(v, lo, hi)}
        for hk in ("H_species", "H_full"):
            h = out[hk]["H"]
            grades[hk] = ({"predicted": "0.2-0.6 plausible (no confident prediction)", "observed": h,
                           "grade": X.grade(h, 0.2, 0.6)} if h is not None else
                          {"observed": None, "grade": "NOT GRADED (H not reported: C does not clear the floor)"})
    # both-sided vs one-sided, the non-oracle side's rate change = h_one − h_both (same seeds, same cells)
    diffs: Dict[str, float] = {}
    for o in ("sp", "full"):
        for ref in ("blob", "fm"):
            s1, s2 = stats[f"one:{o}:{ref}"], stats[f"both:{o}:{ref}"]
            if s1 and s2:
                diffs[f"{o}_vs_{ref}"] = s1.delta_hat - s2.delta_hat
    if len(diffs) == 4:
        allm = math.fsum(diffs.values()) / 4
        grades["both_minus_one_nonoracle"] = {
            "interval80": [-5.0, 0.0], "observed_by_group": diffs, "observed_mean": allm,
            "grade": X.grade(allm, -5.0, 0.0),
            "grade_by_group": {k: X.grade(v, -5.0, 0.0) for k, v in diffs.items()}}
    out["prediction_grades"] = grades
    (HERE / "result.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: out[k]["verdict"] for k in ("matched_steps", "matched_wall_time")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
