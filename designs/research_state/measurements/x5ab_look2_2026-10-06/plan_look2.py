"""X5 A/B LOOK 2: the PLAN — which cells look 2 plays, resolved at PLAY time, with the uniqueness guard.

``design_x5_belief_tokens.md`` §7.4: look 2 reads the FULL 5 × 5 cross (fixed_mass seeds 1001–1005 as rows, blob
seeds 1001–1005 as columns) and "only new cells are played": look 1's 3 × 3 cells (request ``x5ab_look1_*``) are
REUSED from the eval ledger; look 2's request (``x5ab_look2_*``, same family) holds the 16 new ones. Two reads
(§7.9, Amendment 5), each its own family (registered at ``bcb0296c`` with all three looks' boundaries in its rule):

    steps  family x5ab_strength_steps  rows fixed_mass final (15M)            cols blob final (15M)
    wall   family x5ab_strength_wall   rows fixed_mass 12M checkpoint (§7.9)  cols blob final (15M)

    PYTHONPATH=<tree>/src python plan_look2.py --read steps|wall --out cells.json [--root R] [--models M]
        [--seeds ...] [--look1-seeds ...] [--emit look2|look1]

Exit 0 = ``--out`` holds the cells to play (a JSON list of [player .zip, opponent .zip], the form ``main.h2h
play-many --cells`` reads). Exit 3 = REFUSED (a run missing or short of 15M, no unique 12M checkpoint, the ledger's
look-1 cells are not exactly the registered 3 × 3, or look 2's request holds a cell outside the plan). Every guard
is by checkpoint sha256, the key the ledger's cells carry. ``--emit look1`` exists ONLY for a dry run into a scratch
ledger (it plays look 1's cells there first) and refuses the run archive's ledger and the archive's models."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.h2h import cross as X

ARCHIVE_MODELS = Path("/home/goodlad/dev/gen3ai/models")
P_OFF = "gen3_eval_protocol_v1_h2h"
#: read → (family, look-1 request, look-2 request)
READS: Dict[str, Tuple[str, str, str]] = {
    "steps": ("x5ab_strength_steps", "x5ab_look1_steps", "x5ab_look2_steps"),
    "wall": ("x5ab_strength_wall", "x5ab_look1_wall", "x5ab_look2_wall")}
#: the families' registered terms (ledger events of 2026-10-06; `family-register` is idempotent ONLY on these exact
#: terms, so the driver registers with them, never with HEAD).
FAMILY_COMMIT = "bcb0296c0edf9d9f6602f23bcfbc6b14e5df8ef7"
SEEDS = (1001, 1002, 1003, 1004, 1005)
LOOK1_SEEDS = (1001, 1002, 1003)
FINAL = "final_model.zip"
STEPS_15M = 15_000_000
#: §7.9: s = +16.7 % → 15M / 1.167 = 12.85M → the 1M checkpoint at or below = the 12M one (step in [12M, 13M)).
WALL_MILLION = 12


class Refused(Exception):
    pass


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ckpts(run: Path) -> List[Tuple[int, Path]]:
    out = []
    for p in (run / "checkpoints").glob("checkpoint_*_steps.zip"):
        m = re.fullmatch(r"checkpoint_(\d+)_steps\.zip", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def resolve(models: Path, seeds: Sequence[int]) -> Tuple[Dict[str, Dict[int, Path]], List[str]]:
    """The checkpoints of every arm × seed, and every precondition failure (a missing / short run is a failure,
    never a dropped seed): ``{"blob": {seed: final}, "fm": {seed: final}, "fm12": {seed: 12M ckpt}}``."""
    fails: List[str] = []
    got: Dict[str, Dict[int, Path]] = {"blob": {}, "fm": {}, "fm12": {}}
    for arm in ("blob", "fm"):
        for s in seeds:
            run = models / f"rb_x5ab_{arm}_s{s}"
            if not run.is_dir():
                fails.append(f"{run.name}: no run directory (not trained yet?)")
                continue
            ck = _ckpts(run)
            last = ck[-1][0] if ck else 0
            if last < STEPS_15M:
                fails.append(f"{run.name}: last checkpoint {last} < 15M")
            if not (run / FINAL).is_file():
                fails.append(f"{run.name}: no {FINAL}")
            else:
                got[arm][s] = run / FINAL
            if arm == "fm":
                w = [p for st, p in ck if st // 1_000_000 == WALL_MILLION]
                if len(w) != 1:
                    fails.append(f"{run.name}: {len(w)} checkpoint(s) in [12M, 13M), need exactly 1")
                else:
                    got["fm12"][s] = w[0]
    return got, fails


def label(p: Path) -> str:
    """``<run>/<file>`` (a checkpoint under ``checkpoints/`` keeps its run's name)."""
    run = p.parent.parent if p.parent.name == "checkpoints" else p.parent
    return f"{run.name}/{p.relative_to(run)}"


def rows_cols(got: Dict[str, Dict[int, Path]], read: str) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    return (got["fm12"] if read == "wall" else got["fm"]), got["blob"]


def look_cells(root: Path, family: str, request: str) -> List[object]:
    try:
        return X.look_cells(root, family, request, P_OFF)
    except L.ReaderDeclError as e:     # family not registered under this root: no cells
        print(f"[plan] {family}: {e}", file=sys.stderr)
        return []


def plan(root: Path, models: Path, read: str, seeds: Sequence[int], look1_seeds: Sequence[int],
         emit: str = "look2") -> Tuple[List[Tuple[str, str]], List[str]]:
    """``(cells to play as [player path, opponent path], log lines)``; raises :class:`Refused`."""
    if read not in READS:
        raise Refused(f"unknown read {read!r}")
    if not set(look1_seeds) < set(seeds):
        raise Refused(f"look-1 seeds {list(look1_seeds)} must be a proper subset of {list(seeds)}")
    fam, req1, req2 = READS[read]
    got, fails = resolve(models, seeds)
    if fails:
        raise Refused("preconditions: " + "; ".join(fails))
    rows, cols = rows_cols(got, read)
    shas = {p: sha(p) for p in list(rows.values()) + list(cols.values())}
    full = [(r, c) for r, c in itertools.product(seeds, seeds)]
    old = [(r, c) for r, c in itertools.product(look1_seeds, look1_seeds)]
    key = {rc: (shas[rows[rc[0]]], shas[cols[rc[1]]]) for rc in full}
    log = [f"[plan] {read}: family {fam}; rows {[label(rows[s]) for s in seeds]}"]
    if emit == "look1":
        return [(str(rows[r]), str(cols[c])) for r, c in old], log + [f"[plan] DRY RUN: emitting look 1's {len(old)}"]
    l1 = look_cells(root, fam, req1)
    have1 = {(c.player, c.opponent) for c in l1}
    want1 = {key[rc] for rc in old}
    if have1 != want1:
        raise Refused(f"{req1}: the ledger holds {len(have1)} cell(s), {len(have1 & want1)} of the registered "
                      f"{len(want1)} look-1 cells and {len(have1 - want1)} other(s); look 2 reuses EXACTLY look 1's "
                      f"{len(old)} cells (fix look 1 first; never replay it under look 2)")
    short = [f"{c.player_id} vs {c.opponent_id} ({c.n_pairs} pairs, {c.verdict})" for c in l1
             if c.n_pairs < X.MIN_PAIRS or c.verdict != "OK"]
    if short:
        log.append(f"[plan] WARNING: look-1 cell(s) short/invalid — the read will be INCONCLUSIVE: {short}")
    new = [rc for rc in full if rc not in old]
    # THE UNIQUENESS GUARD: no planned cell is a look-1 cell (by sha), and no two planned cells coincide.
    clash = [rc for rc in new if key[rc] in have1]
    if clash or len({key[rc] for rc in new}) != len(new):
        raise Refused(f"uniqueness: planned cell(s) {clash} already in {req1}, or two planned cells share a sha pair")
    l2 = look_cells(root, fam, req2)
    planned = {key[rc] for rc in new}
    stray = [f"{c.player_id} vs {c.opponent_id}" for c in l2 if (c.player, c.opponent) not in planned]
    if stray:
        raise Refused(f"{req2} holds cell(s) outside look 2's plan: {stray}")
    done = sum(1 for c in l2 if c.n_pairs >= X.MIN_PAIRS)
    log.append(f"[plan] {read}: {len(full)} cells in the {len(seeds)} x {len(seeds)} matrix = {len(old)} reused from "
               f"{req1} + {len(new)} new under {req2} ({len(l2)} already in the ledger, {done} at >= {X.MIN_PAIRS} "
               f"pairs; the driver's resume skips every recorded batch)")
    return [(str(rows[r]), str(cols[c])) for r, c in new], log


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True, choices=sorted(READS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="ledger root (default: the run archive's)")
    ap.add_argument("--models", default=str(ARCHIVE_MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--look1-seeds", type=int, nargs="+", default=list(LOOK1_SEEDS))
    ap.add_argument("--emit", default="look2", choices=("look2", "look1"))
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else L.archive_ledger_root()
    models = Path(a.models)
    if a.emit == "look1" and (a.root is None or models.resolve() == ARCHIVE_MODELS.resolve()):
        print("REFUSED: --emit look1 is a DRY-RUN tool (scratch --root and scratch --models only)", file=sys.stderr)
        return 3
    try:
        cells, log = plan(root, models, a.read, a.seeds, a.look1_seeds, a.emit)
    except Refused as e:
        print(f"REFUSED ({a.read}): {e}", file=sys.stderr)
        return 3
    for line in log:
        print(line, file=sys.stderr)
    Path(a.out).write_text(json.dumps([list(c) for c in cells], indent=1) + "\n")
    print(len(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
