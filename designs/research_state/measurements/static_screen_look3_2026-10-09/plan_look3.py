"""Static-token screen LOOK 3 (the FINAL look): the PLAN — the 16 seeds, their preconditions, and the 39 NEW cells.

``designs/endstate/design_static_tokens.md`` §8.1 / §8.2 (P_st ``6c6d2e09``): look 3 reads the FULL 8 × 8 cross
(static seeds 1001–1008 as rows, legacy seeds 1001–1008 as columns, each run's 15M ``final_model.zip``). As at look 2
(``../static_screen_look2_2026-10-09/plan_look2.py``), only NEW cells are played: look 1's 9 cells (request
``st_look1_steps``) and look 2's 16 (request ``st_look2_steps``) are REUSED from the eval COUNT ledger (same engine at
P_st, same regime, same pin, same checkpoints by sha256), and look 3's request ``st_look3_steps`` (same family
``st_screen_strength_steps``) holds the 39 new ones: rows 1006–1008 × all eight columns (24) + rows 1001–1005 ×
columns 1006–1008 (15).

    PYTHONPATH=<6c6d2e09 checkout>/src python plan_look3.py --out cells.json [--root R] [--models M]

Exit 0 = ``--out`` holds the 39 cells (a JSON list of [player .zip, opponent .zip], the form ``main.h2h play-many
--cells`` reads) and stdout is their count. Exit 3 = REFUSED: any of look 1's preconditions on any of the SIXTEEN runs
(``plan_look2.resolve``'s body with the seed list widened: a run missing, short of 15M by its final's
``num_timesteps``, without ``final_model.zip``, not at the registered training pin, the wrong ``token_encoding`` /
``belief_tokens`` / ``policy_readout``, ``init_num_threads`` unequal across the sixteen, S3's amendment-2 check not
VALID); the ledger's look-1 request not holding EXACTLY the registered 3 × 3 cells or look 2's not EXACTLY its 16; a
planned cell that is already a look-1 / look-2 cell, or two planned cells with one sha pair; or the look-3 request
holding a cell outside the plan. A missing / short seed is a refusal, never a dropped seed."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "static_screen_look2_2026-10-09"))
sys.path.insert(1, str(HERE.parent / "static_screen_look1_2026-10-08"))
import plan_look2 as P2  # noqa: E402  (look 2's resolve / look_cells, look 1's helpers as P2.P1)

from agents.training import eval_ledger as L  # noqa: E402

P1 = P2.P1
ARCHIVE_MODELS = P1.ARCHIVE_MODELS
PIN = P1.PIN
SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008)
LOOK1_SEEDS = P1.SEEDS                     # 1001–1003
LOOK2_SEEDS = P2.SEEDS                     # 1001–1005
ARMS = P1.ARMS
FAMILY = P2.FAMILY
REQ1, REQ2, REQ3 = P2.REQ1, P2.REQ2, "st_look3_steps"
MIN_PAIRS = P2.X.MIN_PAIRS


class Refused(Exception):
    pass


def resolve(models: Path, pin_check: bool = True):
    """``plan_look2.resolve`` over all SIXTEEN runs (it reads its module-level seed list)."""
    old = P2.SEEDS
    P2.SEEDS = SEEDS
    try:
        return P2.resolve(models, pin_check)
    finally:
        P2.SEEDS = old


def look_cells(root: Path, request: str) -> List[object]:
    return P2.look_cells(root, request)


def plan(root: Path, models: Path, pin_check: bool = True,
         s3_check: bool = True) -> Tuple[List[Tuple[str, str]], List[str]]:
    got, fails, _ = resolve(models, pin_check)
    if s3_check:
        ok, line = P1.s3_valid()
        if not ok:
            fails.append(f"S3 (rb_st_static_s1003) amendment-2 validity check did not read VALID ({line})")
    if fails:
        raise Refused("preconditions: " + "; ".join(fails))
    rows, cols = got["static"], got["legacy"]
    shas = {p: P1.sha(p) for p in list(rows.values()) + list(cols.values())}
    full = list(itertools.product(SEEDS, SEEDS))
    old1 = list(itertools.product(LOOK1_SEEDS, LOOK1_SEEDS))
    old2 = [rc for rc in itertools.product(LOOK2_SEEDS, LOOK2_SEEDS) if rc not in old1]
    key = {rc: (shas[rows[rc[0]]], shas[cols[rc[1]]]) for rc in full}
    log = [f"[plan] family {FAMILY}; rows static {list(SEEDS)}, cols legacy {list(SEEDS)} (final_model.zip each)"]
    have: Dict[str, set] = {}
    for req, want_rc in ((REQ1, old1), (REQ2, old2)):
        cs = look_cells(root, req)
        h = {(c.player, c.opponent) for c in cs}
        w = {key[rc] for rc in want_rc}
        if h != w:
            raise Refused(f"{req}: the ledger holds {len(h)} cell(s), {len(h & w)} of the registered {len(w)} and "
                          f"{len(h - w)} other(s); look 3 reuses EXACTLY those {len(w)} cells")
        short = [f"{c.player_id} vs {c.opponent_id} ({c.n_pairs} pairs, {c.verdict})" for c in cs
                 if c.n_pairs < MIN_PAIRS or c.verdict != "OK"]
        if short:
            log.append(f"[plan] WARNING: {req} cell(s) short/invalid — the read will be INCONCLUSIVE: {short}")
        have[req] = h
    if have[REQ1] & have[REQ2]:
        raise Refused(f"uniqueness: {REQ1} and {REQ2} share {len(have[REQ1] & have[REQ2])} cell(s)")
    reused = have[REQ1] | have[REQ2]
    new = [rc for rc in full if rc not in old1 and rc not in old2]
    clash = [rc for rc in new if key[rc] in reused]
    if clash or len({key[rc] for rc in new}) != len(new):
        raise Refused(f"uniqueness: planned cell(s) {clash} already reused, or two planned cells share a sha pair")
    l3 = look_cells(root, REQ3)
    planned = {key[rc] for rc in new}
    stray = [f"{c.player_id} vs {c.opponent_id}" for c in l3 if (c.player, c.opponent) not in planned]
    if stray:
        raise Refused(f"{REQ3} holds cell(s) outside look 3's plan: {stray}")
    done = sum(1 for c in l3 if c.n_pairs >= MIN_PAIRS)
    log.append(f"[plan] {len(full)} cells in the 8 x 8 matrix = {len(old1)} from {REQ1} + {len(old2)} from {REQ2} "
               f"(reused) + {len(new)} new under {REQ3} ({len(l3)} already in the ledger, {done} at >= {MIN_PAIRS} "
               f"pairs; the driver's resume skips every recorded batch)")
    return [(str(rows[r]), str(cols[c])) for r, c in new], log


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="ledger root (default: the run archive's)")
    ap.add_argument("--models", default=str(ARCHIVE_MODELS))
    ap.add_argument("--no-pin-check", action="store_true")
    ap.add_argument("--no-s3-check", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else L.archive_ledger_root()
    try:
        cells, log = plan(root, Path(a.models), not a.no_pin_check, not a.no_s3_check)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    for line in log:
        print(line, file=sys.stderr)
    Path(a.out).write_text(json.dumps([list(c) for c in cells], indent=1) + "\n")
    print(len(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
