"""X67, the entropy screen (PREREG §5.2): the PLAN of the head-to-head cross.

Rows = the three X67 forks ``rb_x35_x67ent_s6701..s6703``. Columns = the C0 finals ``rb_x35_c0_s3501..s3508`` that
pass the guards when the plan is made: the five banked (3501-3505) are REQUIRED; a top-up seed (3506-3508, X60-T's
C0 top-up) with no ``final_model.zip`` yet is left out and named, and one that has a final but fails a guard REFUSES
the plan (a control that exists and is not clean is a question for the orchestrator, never a silent drop).

Every run in the plan must have: ``final_model.zip``; a ``pin_history`` of exactly ONE entry at the production side
pin, from the fork step 280,000,170 to at least 300,000,170; the lineage record of a FORK of P* (the parent
checkpoint's sha256); its own seed; ``n_epochs`` 10; ``ent_coef`` 0.0125 (a row) or 0.05 (a column); K9(b) PASS over
its own rows (``main.x35.batch.k9b_read``, AM-6). All runs must share one ``torch_version`` and one
``init_num_threads``. REFUSES (exit 3) otherwise.

Writes ``--out`` (the [player .zip, opponent .zip] list ``main.h2h play-many --cells`` reads) and ``--shas`` (row and
column sha256 lists and seeds for the reader). Stdout: the cell count. ``--columns-only`` checks the columns alone
and writes nothing (the registration's own check, run before any X67 fork exists).

Run with ``PYTHONPATH=<a checkout>/src`` (``main.x35.batch`` supplies the K9(b) read).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

MODELS = Path(os.environ.get("GEN3AI_MODELS_DIR") or "/home/goodlad/dev/gen3ai/models")
PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
PSTAR_SHA256 = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
FORK_STEP, MIN_LAST_STEP = 280_000_170, 300_000_170
ROWS = ("rb_x35_x67ent_s{}", (6701, 6702, 6703), 0.0125)
COLS = ("rb_x35_c0_s{}", (3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508), 0.05)
REQUIRED_COLS = (3501, 3502, 3503, 3504, 3505)
N_EPOCHS = 10


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_run(run: Path, seed: int, ent: float) -> Tuple[List[str], Dict[str, Any]]:
    """``(problems, facts)`` for one finished run; the caller has checked ``final_model.zip`` exists."""
    import main.x35.batch as B

    probs: List[str] = []
    md = json.loads((run / "metadata.json").read_text())
    ph = md.get("pin_history") or []
    seen = [(str(p.get("git_hash"))[:8], p.get("first_step"), p.get("last_step")) for p in ph]
    if len(ph) != 1 or ph[0].get("git_hash") != PIN:
        probs.append(f"pin_history {seen} is not one entry at {PIN[:8]}")
    else:
        if int(ph[0].get("first_step") or -1) != FORK_STEP:
            probs.append(f"its pin entry starts at {ph[0].get('first_step')}, not the fork step {FORK_STEP}")
        if int(ph[0].get("last_step") or 0) < MIN_LAST_STEP:
            probs.append(f"last_step {ph[0].get('last_step')} < {MIN_LAST_STEP}")
    lin = md.get("lineage") or {}
    parent = lin.get("fork_parent") or {}
    if lin.get("role") != "fork" or parent.get("sha256") != PSTAR_SHA256:
        probs.append(f"lineage is not a FORK of P* (role {lin.get('role')!r}, parent sha {str(parent.get('sha256'))[:12]})")
    cli = md.get("cli_args") or {}
    if cli.get("seed") != seed:
        probs.append(f"seed {cli.get('seed')!r} != {seed}")
    if cli.get("n_epochs") != N_EPOCHS:
        probs.append(f"n_epochs {cli.get('n_epochs')!r} != {N_EPOCHS}")
    for where, got in (("metadata.ent_coef", md.get("ent_coef")), ("cli_args.ent_coef", cli.get("ent_coef"))):
        if not isinstance(got, (int, float)) or abs(float(got) - ent) > 1e-12:
            probs.append(f"{where} {got!r} != {ent}")
    k = B.k9b_read(str(run))
    if k.verdict != "PASS":
        probs.append(f"K9(b) {k.verdict}: {'; '.join(k.reasons + k.fires) or 'no reason given'}")
    return probs, {"torch_version": md.get("torch_version"), "init_num_threads": md.get("init_num_threads"),
                   "num_timesteps": md.get("num_timesteps")}


def collect(fmt: str, seeds: Tuple[int, ...], ent: float, required: Tuple[int, ...], fails: List[str],
            notes: List[str]) -> List[Tuple[int, Path, Dict[str, Any]]]:
    out = []
    for s in seeds:
        run = MODELS / fmt.format(s)
        fz = run / "final_model.zip"
        if not fz.is_file():
            (fails if s in required else notes).append(f"{run.name}: no final_model.zip"
                                                       + ("" if s in required else " (left out of the columns)"))
            continue
        probs, facts = check_run(run, s, ent)
        fails.extend(f"{run.name}: {p}" for p in probs)
        out.append((s, fz, facts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--shas")
    ap.add_argument("--columns-only", action="store_true")
    a = ap.parse_args()
    if not a.columns_only and not (a.out and a.shas):
        ap.error("--out and --shas are required unless --columns-only")
    fails: List[str] = []
    notes: List[str] = []
    cols = collect(COLS[0], COLS[1], COLS[2], REQUIRED_COLS, fails, notes)
    rows = [] if a.columns_only else collect(ROWS[0], ROWS[1], ROWS[2], ROWS[1], fails, notes)
    for what in ("torch_version", "init_num_threads"):
        seen = sorted({str(f[what]) for _, _, f in cols + rows})
        if len(seen) > 1:
            fails.append(f"{what} differs across the runs: {seen}")
    for n in notes:
        print("NOTE: " + n, file=sys.stderr)
    if fails:
        print("REFUSED: " + "; ".join(fails), file=sys.stderr)
        return 3
    if a.columns_only:
        print(f"columns OK: {[s for s, _, _ in cols]} ({len(cols)} C0 finals pass every guard)")
        return 0
    cells = [[str(r), str(c)] for _, r, _ in rows for _, c, _ in cols]
    Path(a.out).write_text(json.dumps(cells, indent=1))
    Path(a.shas).write_text(json.dumps({
        "rows": [sha(p) for _, p, _ in rows], "cols": [sha(p) for _, p, _ in cols],
        "row_seeds": [s for s, _, _ in rows], "col_seeds": [s for s, _, _ in cols],
        "row_runs": [str(p) for _, p, _ in rows], "col_runs": [str(p) for _, p, _ in cols],
        "left_out": notes}, indent=1))
    print(len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
