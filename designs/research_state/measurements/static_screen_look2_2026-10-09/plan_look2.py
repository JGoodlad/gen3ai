"""Static-token screen LOOK 2: the PLAN — the 10 seeds, their preconditions, and the 16 NEW cells of the 5 × 5 cross.

``designs/endstate/design_static_tokens.md`` §8.1 / §8.2 (P_st ``6c6d2e09``): look 2 reads the FULL 5 × 5 cross
(static seeds 1001–1005 as rows, legacy seeds 1001–1005 as columns, each run's 15M ``final_model.zip``), and, as in
the X5 A/B's look 2 (``../x5ab_look2_2026-10-06/plan_look2.py``), only NEW cells are played: look 1's 3 × 3 cells
(request ``st_look1_steps``) are REUSED from the eval COUNT ledger (same engine at P_st, same regime, same pin, same
checkpoints by sha256), and look 2's request ``st_look2_steps`` (same family ``st_screen_strength_steps``) holds the
16 new ones: rows 1004–1005 × all five columns (10) + rows 1001–1003 × columns 1004–1005 (6).

    PYTHONPATH=<6c6d2e09 checkout>/src python plan_look2.py --out cells.json [--root R] [--models M] [--no-pin-check]

Exit 0 = ``--out`` holds the 16 cells (a JSON list of [player .zip, opponent .zip], the form ``main.h2h play-many
--cells`` reads) and stdout is their count. Exit 3 = REFUSED: any of look 1's preconditions on any of the ten runs
(``plan_look1.py``: a run missing, short of 15M by its final's ``num_timesteps``, without ``final_model.zip``, not at
the registered training pin, the wrong ``token_encoding`` / ``belief_tokens`` / ``policy_readout``, ``init_num_threads``
unequal across the ten, S3's amendment-2 check not VALID); the ledger's look-1 request not holding EXACTLY the
registered 3 × 3 cells; a planned cell that is already a look-1 cell or two planned cells with one sha pair; or the
look-2 request holding a cell outside the plan. A missing / short seed is a refusal, never a dropped seed."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "static_screen_look1_2026-10-08"))
import plan_look1 as P1  # noqa: E402  (look 1's helpers: PIN, WANT, sha, final_steps, s3_valid)

from agents.training import eval_ledger as L  # noqa: E402
from main.h2h import cross as X  # noqa: E402

ARCHIVE_MODELS = P1.ARCHIVE_MODELS
PIN = P1.PIN
SEEDS = (1001, 1002, 1003, 1004, 1005)
LOOK1_SEEDS = P1.SEEDS
ARMS = P1.ARMS
FAMILY = "st_screen_strength_steps"
REQ1, REQ2 = "st_look1_steps", "st_look2_steps"
PROTOCOL = "gen3_eval_protocol_v1_h2h"


class Refused(Exception):
    pass


def resolve(models: Path, pin_check: bool = True) -> Tuple[Dict[str, Dict[int, Path]], List[str], Dict[str, dict]]:
    """``plan_look1.resolve``'s checks over all ten runs (its body, with the seed list a parameter)."""
    fails: List[str] = []
    got: Dict[str, Dict[int, Path]] = {a: {} for a in ARMS}
    facts: Dict[str, dict] = {}
    threads = set()
    for arm in ARMS:
        for s in SEEDS:
            run = P1.run_dir(models, arm, s)
            if not run.is_dir():
                fails.append(f"{run.name}: no run directory")
                continue
            meta = json.loads((run / "metadata.json").read_text())
            cfg = json.loads((run / "model_config.json").read_text())
            pins = [str(p.get("git_hash", "")) for p in meta.get("pin_history") or [] if isinstance(p, dict)]
            if pin_check and (not pins or any(p != PIN for p in pins)):
                fails.append(f"{run.name}: pin_history {pins} is not the registered P_st {PIN[:8]}")
            if cfg.get("token_encoding") != arm:
                fails.append(f"{run.name}: token_encoding {cfg.get('token_encoding')!r} != {arm!r}")
            for k, v in P1.WANT.items():
                if cfg.get(k) != v:
                    fails.append(f"{run.name}: {k} {cfg.get(k)!r} != {v!r}")
            threads.add(meta.get("init_num_threads"))
            f = run / P1.FINAL
            if not f.is_file():
                fails.append(f"{run.name}: no {P1.FINAL}")
                continue
            n = P1.final_steps(f)
            if n < P1.STEPS_15M:
                fails.append(f"{run.name}: {P1.FINAL} at {n} steps < 15M")
            got[arm][s] = f
            facts[run.name] = {"final": str(f), "final_num_timesteps": n, "pin_history": pins,
                               "init_num_threads": meta.get("init_num_threads"),
                               "torch_version": meta.get("torch_version"),
                               "token_encoding": cfg.get("token_encoding"),
                               "belief_tokens": cfg.get("belief_tokens"), "policy_readout": cfg.get("policy_readout"),
                               "behaviour_check_warn_in_metadata": "behaviour-check warn" in json.dumps(meta)}
    if len(threads) != 1:
        fails.append(f"init_num_threads differ across the ten: {sorted(map(str, threads))}")
    return got, fails, facts


def look_cells(root: Path, request: str) -> List[object]:
    try:
        return X.look_cells(root, FAMILY, request, PROTOCOL)
    except L.ReaderDeclError as e:     # family not registered under this root: no cells
        print(f"[plan] {FAMILY}: {e}", file=sys.stderr)
        return []


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
    old = list(itertools.product(LOOK1_SEEDS, LOOK1_SEEDS))
    key = {rc: (shas[rows[rc[0]]], shas[cols[rc[1]]]) for rc in full}
    log = [f"[plan] family {FAMILY}; rows static {list(SEEDS)}, cols legacy {list(SEEDS)} (final_model.zip each)"]
    l1 = look_cells(root, REQ1)
    have1 = {(c.player, c.opponent) for c in l1}
    want1 = {key[rc] for rc in old}
    if have1 != want1:
        raise Refused(f"{REQ1}: the ledger holds {len(have1)} cell(s), {len(have1 & want1)} of the registered "
                      f"{len(want1)} look-1 cells and {len(have1 - want1)} other(s); look 2 reuses EXACTLY look 1's "
                      f"{len(old)} cells")
    short = [f"{c.player_id} vs {c.opponent_id} ({c.n_pairs} pairs, {c.verdict})" for c in l1
             if c.n_pairs < X.MIN_PAIRS or c.verdict != "OK"]
    if short:
        log.append(f"[plan] WARNING: look-1 cell(s) short/invalid — the read will be INCONCLUSIVE: {short}")
    new = [rc for rc in full if rc not in old]
    clash = [rc for rc in new if key[rc] in have1]
    if clash or len({key[rc] for rc in new}) != len(new):
        raise Refused(f"uniqueness: planned cell(s) {clash} already in {REQ1}, or two planned cells share a sha pair")
    l2 = look_cells(root, REQ2)
    planned = {key[rc] for rc in new}
    stray = [f"{c.player_id} vs {c.opponent_id}" for c in l2 if (c.player, c.opponent) not in planned]
    if stray:
        raise Refused(f"{REQ2} holds cell(s) outside look 2's plan: {stray}")
    done = sum(1 for c in l2 if c.n_pairs >= X.MIN_PAIRS)
    log.append(f"[plan] {len(full)} cells in the 5 x 5 matrix = {len(old)} reused from {REQ1} + {len(new)} new under "
               f"{REQ2} ({len(l2)} already in the ledger, {done} at >= {X.MIN_PAIRS} pairs; the driver's resume "
               f"skips every recorded batch)")
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
