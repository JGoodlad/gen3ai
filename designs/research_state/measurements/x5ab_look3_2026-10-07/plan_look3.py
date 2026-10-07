"""X5 A/B LOOK 3: the PLAN — which cells look 3 plays, resolved at PLAY time, with the uniqueness guard.

``design_x5_belief_tokens.md`` §7.4: look 3 reads the FULL 8 × 8 cross (fixed_mass seeds 1001–1008 as rows, blob
seeds 1001–1008 as columns) and "only new cells are played": look 1's 3 × 3 cells (request ``x5ab_look1_*``) and
look 2's 16 (``x5ab_look2_*``) are REUSED from the eval ledger; look 3's request (``x5ab_look3_*``, same family)
holds the 8 × 8 − 5 × 5 = 39 new ones. Two registered reads (§7.9, Amendment 5) + one SENSITIVITY line (Decision
record 2026-10-06, "MATCHED-WALL-TIME definition fixed BEFORE look 3 is read": matched wall = END-TO-END, the 12M
checkpoint; the 13M steady-state checkpoint is reported beside it, never as the read):

    steps   family x5ab_strength_steps      rows fixed_mass final (15M)        cols blob final (15M)   39 new
    wall    family x5ab_strength_wall       rows fixed_mass 12M checkpoint     cols blob final (15M)   39 new
    wall13  family x5ab_sensitivity_wall13  rows fixed_mass 13M checkpoint     cols blob final (15M)   64 new (all)

    PYTHONPATH=<706fa536 checkout>/src python plan_look3.py --read steps|wall|wall13 --out cells.json
        [--root R] [--models M] [--seeds ...] [--look-seeds 3 5] [--no-pin-check]

Exit 0 = ``--out`` holds the cells to play (a JSON list of [player .zip, opponent .zip], the form ``main.h2h
play-many --cells`` reads). Exit 3 = REFUSED: a run missing or short of 15M, no unique 12M / 13M checkpoint, a run
not at its registered training pin, an earlier look's request not holding EXACTLY its registered block, or look 3's
request holding a cell outside the plan. Every guard is by checkpoint sha256, the key the ledger's cells carry.

THE SEEDS (registered; ledger BANK entries + the look-3 brief): seed 1006 of fixed_mass is the run
``rb_x5ab_fm_s1006b`` (708dcb0a, DEVIATION ``--behaviour-check warn``); ``rb_x5ab_fm_s1006`` stopped at update 1
and is kept as the INCONCLUSIVE artifact, NOT a seed. The oracle arms (``rb_x5ab_oracle_*``) are reference arms
(§7.6), not extended at look 3, and never enter this cross."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.h2h import cross as X

ARCHIVE_MODELS = Path("/home/goodlad/dev/gen3ai/models")
P_OFF = "gen3_eval_protocol_v1_h2h"
#: the look-3 play/read pin (Decision record 2026-10-06, "The LOOK-3 CROSS plays at the TRAINING code")
PIN = "706fa536ef53d9a680d2461f16613e0d1430d176"
#: the families' registered commit (x5ab_strength_*; ledger events of 2026-10-06)
FAMILY_COMMIT = "bcb0296c0edf9d9f6602f23bcfbc6b14e5df8ef7"
SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008)
#: the seeds per arm at looks 1 and 2 (§7.4: 3, 5); look 3 = all of SEEDS (8)
LOOK_SEEDS = (3, 5)
FINAL = "final_model.zip"
STEPS_15M = 15_000_000
#: run directory per (arm, seed) where it is not ``rb_x5ab_<arm>_s<seed>``
RUN_NAME = {("fm", 1006): "rb_x5ab_fm_s1006b"}
#: each run's registered TRAINING pin (metadata.json pin_history; ledger BANK entries; the look-3 brief)
TRAIN_PIN = {**{("blob", s): "e5e660dd" for s in range(1001, 1007)},
             **{("fm", s): "708dcb0a" for s in range(1001, 1007)},
             **{(a, s): "706fa536" for a in ("blob", "fm") for s in (1007, 1008)}}


class Read(NamedTuple):
    family: str
    earlier: Tuple[str, ...]      # the earlier looks' requests, look 1 first (reused, never replayed)
    request: str                  # look 3's request
    rows: str                     # "fm" (final) | "fm12" | "fm13"
    registered: bool              # False = the SENSITIVITY line (reported, never a verdict)


READS: Dict[str, Read] = {
    "steps": Read("x5ab_strength_steps", ("x5ab_look1_steps", "x5ab_look2_steps"), "x5ab_look3_steps", "fm", True),
    "wall": Read("x5ab_strength_wall", ("x5ab_look1_wall", "x5ab_look2_wall"), "x5ab_look3_wall", "fm12", True),
    "wall13": Read("x5ab_sensitivity_wall13", (), "x5ab_look3_wall13", "fm13", False),
}
#: rows key → the million the fixed_mass checkpoint sits in ([M, M+1) million steps)
ROW_MILLION = {"fm12": 12, "fm13": 13}


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


def run_dir(models: Path, arm: str, seed: int) -> Path:
    return models / RUN_NAME.get((arm, seed), f"rb_x5ab_{arm}_s{seed}")


def _pins(run: Path) -> List[str]:
    try:
        ph = json.loads((run / "metadata.json").read_text()).get("pin_history")
    except (OSError, ValueError):
        return []
    if not isinstance(ph, list):
        return []
    return [str(p.get("git_hash", ""))[:8] for p in ph if isinstance(p, dict)]


def resolve(models: Path, seeds: Sequence[int], pin_check: bool = True
            ) -> Tuple[Dict[str, Dict[int, Path]], List[str]]:
    """The checkpoints of every arm × seed, and every precondition failure (a missing / short run is a failure,
    never a dropped seed): ``{"blob": {seed: final}, "fm": {seed: final}, "fm12": {...}, "fm13": {...}}``."""
    fails: List[str] = []
    got: Dict[str, Dict[int, Path]] = {"blob": {}, "fm": {}, "fm12": {}, "fm13": {}}
    for arm in ("blob", "fm"):
        for s in seeds:
            run = run_dir(models, arm, s)
            if not run.is_dir():
                fails.append(f"{run.name}: no run directory (not trained yet?)")
                continue
            if pin_check:
                pins, want = _pins(run), TRAIN_PIN.get((arm, s))
                if want is None or not pins or any(p != want for p in pins):
                    fails.append(f"{run.name}: pin_history {pins} is not the registered training pin {want}")
            ck = _ckpts(run)
            last = ck[-1][0] if ck else 0
            if last < STEPS_15M:
                fails.append(f"{run.name}: last checkpoint {last} < 15M")
            if not (run / FINAL).is_file():
                fails.append(f"{run.name}: no {FINAL}")
            else:
                got[arm][s] = run / FINAL
            if arm == "fm":
                for key, mil in ROW_MILLION.items():
                    w = [p for st, p in ck if st // 1_000_000 == mil]
                    if len(w) != 1:
                        fails.append(f"{run.name}: {len(w)} checkpoint(s) in [{mil}M, {mil + 1}M), need exactly 1")
                    else:
                        got[key][s] = w[0]
    return got, fails


def arch_key(p: Path) -> str:
    """A hash of the checkpoint's recorded ``policy_kwargs`` (the non-serialised part of sb3's ``data`` member): the
    constructor parameters ``agents.inference.service.slots.forward_fingerprint`` hashes, so two checkpoints with
    one key are one h2h slot group, and two keys may be two (an engine serves at most two, ``main.h2h.arch``).
    WHY (2026-10-07, the first look-3 launch): the 706fa536-trained seeds (1007, 1008) record two defaulted kwargs the
    older ones lack (``policy_readout: tower`` in both arms, ``oracle_reveal: off`` in blob), so the engine refused
    to co-serve them (``CellArchMismatch`` on the forward fingerprint) — the cells are played on one engine per
    (row key, column key) instead, each checkpoint by its own template."""
    with zipfile.ZipFile(p) as z:
        pk = json.loads(z.read("data"))["policy_kwargs"]
    clean = {k: v for k, v in pk.items() if not k.startswith(":")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:12]


def split_by_engine(cells: Sequence[Tuple[str, str]]) -> List[Tuple[Tuple[str, str], List[Tuple[str, str]]]]:
    """``cells`` partitioned by (player arch key, opponent arch key), in first-appearance order: one engine each."""
    keys: Dict[str, str] = {}
    out: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for pl, op in cells:
        for x in (pl, op):
            if x not in keys:
                keys[x] = arch_key(Path(x))
        out.setdefault((keys[pl], keys[op]), []).append((pl, op))
    return list(out.items())


def label(p: Path) -> str:
    """``<run>/<file>`` (a checkpoint under ``checkpoints/`` keeps its run's name)."""
    run = p.parent.parent if p.parent.name == "checkpoints" else p.parent
    return f"{run.name}/{p.relative_to(run)}"


def rows_cols(got: Dict[str, Dict[int, Path]], read: str) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    return got[READS[read].rows], got["blob"]


def look_cells(root: Path, family: str, request: str) -> List[object]:
    try:
        return X.look_cells(root, family, request, P_OFF)
    except L.ReaderDeclError as e:     # family not registered under this root: no cells
        print(f"[plan] {family}: {e}", file=sys.stderr)
        return []


def blocks(seeds: Sequence[int], look_seeds: Sequence[int], n_earlier: int
           ) -> Tuple[List[List[Tuple[int, int]]], List[Tuple[int, int]]]:
    """``(each earlier look's NEW block as (row seed, col seed) pairs, look 3's new block)``. Look k's block is
    seeds[:n_k]² minus seeds[:n_(k-1)]²; with no earlier looks (the sensitivity line) look 3's block is all."""
    sizes = list(look_seeds[:n_earlier])
    prev: set = set()
    out: List[List[Tuple[int, int]]] = []
    for n in sizes:
        full = set(itertools.product(seeds[:n], seeds[:n]))
        out.append(sorted(full - prev))
        prev = full
    last = [rc for rc in itertools.product(seeds, seeds) if rc not in prev]
    return out, last


def plan(root: Path, models: Path, read: str, seeds: Sequence[int], look_seeds: Sequence[int],
         pin_check: bool = True, emit_look: int = 3) -> Tuple[List[Tuple[str, str]], List[str]]:
    """``(cells to play as [player path, opponent path], log lines)``; raises :class:`Refused`."""
    if read not in READS:
        raise Refused(f"unknown read {read!r}")
    R = READS[read]
    if list(look_seeds) != sorted(set(look_seeds)) or (look_seeds and look_seeds[-1] >= len(seeds)):
        raise Refused(f"look seed counts {list(look_seeds)} must increase and stay below {len(seeds)}")
    got, fails = resolve(models, seeds, pin_check)
    if fails:
        raise Refused("preconditions: " + "; ".join(fails))
    rows, cols = rows_cols(got, read)
    shas = {p: sha(p) for p in list(rows.values()) + list(cols.values())}
    key = {rc: (shas[rows[rc[0]]], shas[cols[rc[1]]]) for rc in itertools.product(seeds, seeds)}
    earlier_blocks, new = blocks(seeds, look_seeds, len(R.earlier))
    log = [f"[plan] {read}: family {R.family}; rows {[label(rows[s]) for s in seeds]}"]
    if emit_look != 3:                  # DRY RUN ONLY: an earlier look's block, into a scratch ledger
        blk = earlier_blocks[emit_look - 1]
        return [(str(rows[r]), str(cols[c])) for r, c in blk], log + [f"[plan] DRY RUN: emitting look {emit_look}'s "
                                                                      f"{len(blk)} cells"]
    held: set = set()
    for req, blk in zip(R.earlier, earlier_blocks):
        have = {(c.player, c.opponent) for c in look_cells(root, R.family, req)}
        want = {key[rc] for rc in blk}
        if have != want:
            raise Refused(f"{req}: the ledger holds {len(have)} cell(s), {len(have & want)} of its registered "
                          f"{len(want)} and {len(have - want)} other(s); look 3 reuses EXACTLY the earlier looks' "
                          f"cells (fix that look first; never replay it under look 3)")
        held |= have
        short = [f"{c.player_id} vs {c.opponent_id} ({c.n_pairs} pairs, {c.verdict})"
                 for c in look_cells(root, R.family, req) if c.n_pairs < X.MIN_PAIRS or c.verdict != "OK"]
        if short:
            log.append(f"[plan] WARNING: {req} cell(s) short/invalid — the read will be INCONCLUSIVE: {short}")
    # THE UNIQUENESS GUARD: no planned cell is an earlier look's cell (by sha), and no two planned cells coincide.
    clash = [rc for rc in new if key[rc] in held]
    if clash or len({key[rc] for rc in new}) != len(new):
        raise Refused(f"uniqueness: planned cell(s) {clash} already in {list(R.earlier)}, or two planned cells "
                      f"share a sha pair")
    l3 = look_cells(root, R.family, R.request)
    planned = {key[rc] for rc in new}
    stray = [f"{c.player_id} vs {c.opponent_id}" for c in l3 if (c.player, c.opponent) not in planned]
    if stray:
        raise Refused(f"{R.request} holds cell(s) outside look 3's plan: {stray}")
    done = sum(1 for c in l3 if c.n_pairs >= X.MIN_PAIRS)
    n = len(seeds)
    log.append(f"[plan] {read}: {n * n} cells in the {n} x {n} matrix = {n * n - len(new)} reused from "
               f"{list(R.earlier) or 'none'} + {len(new)} new under {R.request} ({len(l3)} already in the ledger, "
               f"{done} at >= {X.MIN_PAIRS} pairs; the driver's resume skips every recorded batch)")
    return [(str(rows[r]), str(cols[c])) for r, c in new], log


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True, choices=sorted(READS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="ledger root (default: the run archive's)")
    ap.add_argument("--models", default=str(ARCHIVE_MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--look-seeds", type=int, nargs="*", default=list(LOOK_SEEDS),
                    help="seeds per arm at looks 1 and 2 (registered: 3 5)")
    ap.add_argument("--no-pin-check", action="store_true", help="DRY RUN only (scratch --root)")
    ap.add_argument("--emit-look", type=int, default=3, choices=(1, 2, 3),
                    help="DRY RUN only (scratch --root): emit an earlier look's block to play it first")
    ap.add_argument("--split-dir", default=None,
                    help="also write engine_<i>.json there: the cells split by (player, opponent) architecture key")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else L.archive_ledger_root()
    if (a.no_pin_check or a.emit_look != 3) and (a.root is None
                                                  or Path(a.models).resolve() == ARCHIVE_MODELS.resolve()):
        print("REFUSED: --no-pin-check / --emit-look are DRY-RUN tools (scratch --root and --models only)",
              file=sys.stderr)
        return 3
    try:
        cells, log = plan(root, Path(a.models), a.read, a.seeds, a.look_seeds, not a.no_pin_check, a.emit_look)
    except Refused as e:
        print(f"REFUSED ({a.read}): {e}", file=sys.stderr)
        return 3
    for line in log:
        print(line, file=sys.stderr)
    Path(a.out).write_text(json.dumps([list(c) for c in cells], indent=1) + "\n")
    if a.split_dir:                      # one cell list per engine (per (row, column) architecture key pair)
        sd = Path(a.split_dir)
        sd.mkdir(parents=True, exist_ok=True)
        for old in sd.glob("engine_*.json"):
            old.unlink()
        for i, ((kp, ko), part) in enumerate(split_by_engine(cells)):
            (sd / f"engine_{i}.json").write_text(json.dumps([list(c) for c in part], indent=1) + "\n")
            print(f"[plan] engine {i}: player arch {kp} x opponent arch {ko}: {len(part)} cell(s)", file=sys.stderr)
    print(len(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
