"""G-C POOLED OVER SEEDS (AM-X47b-1, registered 2026-10-20 BEFORE any X35 / X47 read; launch sheet §0): the collapse
battery's guard (``design_compute_efficiency.md`` §5.6.5) read on every seed pair of FINALS and pooled.

    python -m main.x35.gc pool --side toy --bank <bank_x26> --truth <rows.jsonl[.gz]> --reads <dir> \\
        --pair 3601=<ON label>:<OFF label> --pair 3602=... --pair 3603=... --out <gc.json>

THE PAIRS. Each arm's FINAL (``final_model.zip``) against its SAME-SEED control's final: A1 / A2 / D1 s350x against
C0 s350x; X47 (its final, not its s_eq node, as ruled) and toy A1 / A2 s360x against toy C0 s360x. ``<label>`` is a
``main.policy_spectrum read`` label in ``--reads`` (``<label>.probs.npz``, read on ``--bank``).

PER PAIR, ON − OFF on identical turns:
* B1 = the mean over every bank decision of H(π_ON) − H(π_OFF) (nats, the entropy over the legal actions);
* B2 = the mean over the truth's DECISIVE turns (some action DOMINATED, ``truth.turn_truth``) of starved(ON) −
  starved(OFF), where a turn is starved when some NEAR-BEST action holds < 1 % of the policy's mass. Rule 8 per pair:
  a turn where EITHER policy puts mass within :data:`BOUNDARY` of the 1 % line on a branched action is EXCLUDED for
  both (counted in ``excluded_boundary``).
Each pair carries its own battle-clustered 95 % interval (the crossed bootstrap on one pair = the battle bootstrap).

THE POOLED READ. Point = the equal-weight mean over the seed pairs of the per-pair means. Interval = a CROSSED
(two-way) cluster bootstrap: each of :data:`N_BOOT` replicates (seed :data:`BOOT_SEED`) draws the seed pairs with
replacement AND the bank's battles with replacement (ONE battle draw shared by every pair, because every pair reads the
same bank); the replicate's statistic is the mean over the drawn pairs of each pair's turn-weighted mean over the drawn
battles; the interval is the replicates' 2.5 / 97.5 percentiles, widened to contain the point estimate when a skewed
replicate distribution leaves it outside (so the triple is always ordered). §5.6.5 registers "battle-clustered 95 %, ON against
OFF, pooled over seeds", which names the clustering but no seed pooling; this is the ruled seed-pair bootstrap combined
with it.

THE TRUTH PER SIDE (AM-X47b-2, AM-X47c-2). Each side's B2 reads the same 306-turn subset re-branched under its own
plateau checkpoint: production under P* (X26@280M; §5.6.5's refresh rule retired READ A's X26@150M truth), the toy
under T*. :data:`TRUTH_BY_SIDE` is each side's continuation stamp, and :func:`pool` refuses rows of another
continuation (the 150M truth included).

THE RECORD (schema :data:`GC_SCHEMA`) is what ``readout decide`` / ``decide-d1`` / ``x47 decide`` take as ``g_c``;
:func:`check_record` refuses (``ReadoutInputError``) a record whose ON or OFF seeds are not the read's R1 seeds, whose
side is not those seeds' side, or whose truth is not the side's. A bare (estimate, lower, upper) triple cannot show
it was pooled, so the readers do not accept one.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.x35.readout import ReadoutInputError

GC_SCHEMA = "x35_gc_pooled_v1"
METHOD = ("crossed cluster bootstrap: seed pairs x bank battles (one battle draw shared by every pair), "
          "equal weight per pair, 95 % percentile (AM-X47b-1)")
N_BOOT = 2000
BOOT_SEED = 20261020
#: rule 8 at the 1 % starve line (the collapse pilot's ``starvation_readout.py`` value).
BOUNDARY = 1e-6
#: the seeds of each side (``kit.PROD_SEEDS`` / ``kit.TOY_SEEDS``; ``gc_test`` pins the equality).
SIDE_SEEDS: Dict[str, Tuple[int, ...]] = {"production": tuple(range(3501, 3506)), "toy": tuple(range(3601, 3606))}
#: AM-X47b-2 / AM-X47c-2: each side's B2 truth, by the continuation every truth row is stamped with (the committed
#: rows: ``x35_prod_b2_truth_2026-10-20/`` and ``x35_toy_b2_truth_2026-10-20/``, :data:`TRUTH_ROWS_REL`).
TRUTH_BY_SIDE: Dict[str, str] = {"production": "X26@280M greedy (both sides)",
                                 "toy": "toyTstar@290M greedy (both sides)"}
#: each side's committed truth rows, relative to the repo root (``gc_test`` pins each file's stamp to its side's).
_MEAS = "designs/research_state/measurements"
TRUTH_ROWS_REL: Dict[str, str] = {
    "production": f"{_MEAS}/x35_prod_b2_truth_2026-10-20/rows_X26at280M_S32.compact.jsonl.gz",
    "toy": f"{_MEAS}/x35_toy_b2_truth_2026-10-20/rows_toyTstar290M_S32.compact.jsonl.gz"}


# ---------------------------------------------------------------------------------------------- per turn
def entropy(probs: np.ndarray) -> np.ndarray:
    """Per-row entropy (nats) of masked probabilities (illegal actions exactly 0), as ``policy_spectrum.spectrum``."""
    p = np.asarray(probs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)


def b1_deltas(p_on: np.ndarray, p_off: np.ndarray, battles: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """``(H_ON − H_OFF per decision, battle id per decision)``."""
    if p_on.shape != p_off.shape or len(battles) != len(p_on):
        raise ReadoutInputError(f"B1: shapes {p_on.shape} / {p_off.shape} / {len(battles)} battles differ")
    return entropy(p_on) - entropy(p_off), np.asarray(battles, dtype=object)


def b2_deltas(rows: Sequence[Mapping[str, Any]], index: Mapping[str, int], battles: Sequence[str],
              p_on: np.ndarray, p_off: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """``(starved_ON − starved_OFF per decisive turn, battle id per turn, turns excluded by rule 8)``."""
    from main.policy_spectrum.truth import STARVE, turn_truth

    d: List[float] = []
    cl: List[str] = []
    excluded = 0
    for r in rows:
        if not r.get("ok"):
            continue
        t = turn_truth(dict(r))
        if not t["dominated"].any():
            continue
        i = index[r["id"]]
        acts = t["actions"]
        a, b = p_on[i, acts], p_off[i, acts]
        if np.any(np.abs(a - STARVE) < BOUNDARY) or np.any(np.abs(b - STARVE) < BOUNDARY):
            excluded += 1
            continue
        d.append(float((t["near"] & (a < STARVE)).any()) - float((t["near"] & (b < STARVE)).any()))
        cl.append(battles[i])
    return np.asarray(d, dtype=np.float64), np.asarray(cl, dtype=object), excluded


# ---------------------------------------------------------------------------------------------- the pool
def crossed_bootstrap(pairs: Sequence[Tuple[np.ndarray, np.ndarray]], *, n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED) -> Tuple[float, float, float]:
    """``(estimate, lower, upper)`` of the equal-weight mean over pairs of each pair's mean, by the crossed cluster
    bootstrap (module docs). ``pairs`` = ``[(values, battle ids)]``, one per seed pair."""
    if not pairs:
        raise ReadoutInputError("G-C: no seed pair")
    for v, c in pairs:
        if len(v) == 0 or len(v) != len(c):
            raise ReadoutInputError(f"G-C: a pair has {len(v)} values for {len(c)} battle ids")
        if not np.all(np.isfinite(v)):
            raise ReadoutInputError("G-C: a non-finite per-turn value")
    universe = sorted({str(b) for _, c in pairs for b in c})
    pos = {b: k for k, b in enumerate(universe)}
    nb, npair = len(universe), len(pairs)
    if nb < 2:
        raise ReadoutInputError(f"G-C: {nb} battle(s); the battle bootstrap needs >= 2")
    sums = np.zeros((npair, nb))
    cnts = np.zeros((npair, nb))
    for s, (v, c) in enumerate(pairs):
        k = np.fromiter((pos[str(b)] for b in c), dtype=np.int64, count=len(c))
        np.add.at(sums[s], k, np.asarray(v, dtype=np.float64))
        np.add.at(cnts[s], k, 1.0)
    est = math.fsum(sums[s].sum() / cnts[s].sum() for s in range(npair)) / npair
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nb, size=(n_boot, nb))
    w = np.zeros((n_boot, nb))
    np.add.at(w, (np.arange(n_boot)[:, None], draws), 1.0)
    num, den = w @ sums.T, w @ cnts.T                                        # [n_boot, npair]
    if np.any(den <= 0):
        raise ReadoutInputError("G-C: a replicate drew no turn of some pair (too few battles per pair)")
    means = num / den
    pick = rng.integers(0, npair, size=(n_boot, npair))
    pooled = np.take_along_axis(means, pick, axis=1).mean(axis=1)
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    return float(est), float(min(lo, est)), float(max(hi, est))


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _side_of(seeds: Sequence[int]) -> Optional[str]:
    for side, pool in SIDE_SEEDS.items():
        if seeds and all(int(s) in pool for s in seeds):
            return side
    return None


def pool_arrays(side: str, truth_continuation: str, pairs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    """The record from per-pair arrays: ``pairs = {seed: {"on": label, "off": label, "b1": (values, battles),
    "b2": (values, battles), "b2_excluded": n}}``. Refuses a side / seed / truth mismatch (module docs)."""
    seeds = sorted(int(s) for s in pairs)
    if side not in SIDE_SEEDS:
        raise ReadoutInputError(f"G-C: side {side!r} is not one of {sorted(SIDE_SEEDS)}")
    if _side_of(seeds) != side:
        raise ReadoutInputError(f"G-C: seeds {seeds} are not all {side} seeds {list(SIDE_SEEDS[side])}")
    if truth_continuation != TRUTH_BY_SIDE[side]:
        raise ReadoutInputError(f"G-C: the {side} side's B2 truth is {TRUTH_BY_SIDE[side]!r} (AM-X47b-2, AM-X47c-2), "
                                f"not {truth_continuation!r}")
    per_pair: Dict[str, Any] = {}
    for s in seeds:
        p = pairs[s] if s in pairs else pairs[str(s)]  # type: ignore[index]
        per_pair[str(s)] = {"on": p["on"], "off": p["off"], "b1": list(crossed_bootstrap([p["b1"]])),
                            "b1_decisions": int(len(p["b1"][0])), "b2": list(crossed_bootstrap([p["b2"]])),
                            "b2_turns": int(len(p["b2"][0])), "b2_excluded_boundary": int(p.get("b2_excluded", 0))}
    get = [pairs[s] if s in pairs else pairs[str(s)] for s in seeds]  # type: ignore[index]
    return {"schema": GC_SCHEMA, "method": METHOD, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "side": side,
            "seeds": seeds, "on_seeds": seeds, "off_seeds": seeds,
            "truth": {"continuation": truth_continuation},
            "b1": list(crossed_bootstrap([g["b1"] for g in get])),
            "b2": list(crossed_bootstrap([g["b2"] for g in get])),
            "per_pair": per_pair}


def pool(side: str, bank_dir: Path, truth_path: Path, reads_dir: Path, pairs: Mapping[int, Tuple[str, str]]
         ) -> Dict[str, Any]:
    """The whole read from files (read-only): ``pairs = {seed: (ON label, OFF label)}``."""
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import load_probs
    from main.policy_spectrum.truth import load_rows

    bank = load_bank(Path(bank_dir))
    battles = [d["battle"] for d in bank.decisions]
    index = {d["id"]: i for i, d in enumerate(bank.decisions)}
    rows = load_rows(Path(truth_path))
    conts = sorted({str(r.get("continuation")) for r in rows})
    if len(conts) != 1:
        raise ReadoutInputError(f"G-C: the truth file holds {len(conts)} continuations {conts}; one is required")
    missing = [r["id"] for r in rows if r["id"] not in index]
    if missing:
        raise ReadoutInputError(f"G-C: {len(missing)} truth turns are not in the bank (first {missing[0]!r})")
    arrays: Dict[int, Dict[str, Any]] = {}
    shas: Dict[str, Optional[str]] = {}
    for s, (on, off) in sorted(pairs.items()):
        p_on, p_off = load_probs(Path(reads_dir), on, bank), load_probs(Path(reads_dir), off, bank)
        d2, c2, ex = b2_deltas(rows, index, battles, p_on, p_off)
        arrays[int(s)] = {"on": on, "off": off, "b1": b1_deltas(p_on, p_off, battles), "b2": (d2, c2),
                          "b2_excluded": ex}
        for lab in (on, off):
            meta = Path(reads_dir) / f"{lab.replace('/', '_').replace(' ', '_')}.json"
            try:
                shas[lab] = json.loads(meta.read_text())["checkpoint"]["sha256"]
            except (OSError, ValueError, KeyError):
                shas[lab] = None
    rec = pool_arrays(side, conts[0], arrays)
    rec["truth"].update({"path": str(truth_path), "sha256": _file_sha(Path(truth_path)), "rows": len(rows),
                         "ok_rows": sum(1 for r in rows if r.get("ok"))})
    rec["bank"] = {"path": str(bank_dir), "content_sha256": bank.manifest.get("content_sha256"),
                   "decisions": len(bank.decisions)}
    rec["checkpoint_sha256"] = shas
    return rec


def constructed_record(side: str, seeds: Sequence[int], b1: Sequence[float], b2: Sequence[float], *,
                       per_battle: int = 3, n_battles: int = 4) -> Dict[str, Any]:
    """A record of CONSTRUCTED data through :func:`pool_arrays` (self-checks and tests only): pair i's per-turn deltas
    are ``b1[i]`` / ``b2[i]`` plus a battle-indexed offset (−s, …, +s with s = 0.001) so every pair has spread."""
    out: Dict[int, Dict[str, Any]] = {}
    for i, seed in enumerate(seeds):
        offs = np.linspace(-0.001, 0.001, n_battles).repeat(per_battle)
        cl = np.array([f"b{k}" for k in range(n_battles)], dtype=object).repeat(per_battle)
        out[int(seed)] = {"on": f"ON_s{seed}", "off": f"OFF_s{seed}", "b1": (b1[i] + offs, cl),
                          "b2": (b2[i] + offs, cl)}
    return pool_arrays(side, TRUTH_BY_SIDE[side], out)


# ---------------------------------------------------------------------------------------------- the check
def check_record(rec: Any, on_seeds: Sequence[Any], off_seeds: Sequence[Any]) -> Tuple[List[float], List[float]]:
    """``(b1, b2)`` triples of a G-C record fit for a read whose R1 seeds are ``on_seeds`` / ``off_seeds``; raises
    :class:`ReadoutInputError` naming every reason it is not (module docs)."""
    if not isinstance(rec, Mapping) or rec.get("schema") != GC_SCHEMA:
        raise ReadoutInputError(f"G-C needs the seed-pooled record (schema {GC_SCHEMA}, `python -m main.x35.gc pool`, "
                                "AM-X47b-1); a bare (estimate, lower, upper) triple cannot show it was pooled")
    why: List[str] = []
    want_on = sorted(int(s) for s in on_seeds)
    want_off = sorted(int(s) for s in off_seeds)
    got_on = sorted(int(s) for s in rec.get("on_seeds") or ())
    got_off = sorted(int(s) for s in rec.get("off_seeds") or ())
    if got_on != want_on or got_off != want_off:
        why.append(f"the record pools ON seeds {got_on} / OFF seeds {got_off}, the read's R1 has {want_on} / {want_off}")
    pp = rec.get("per_pair") or {}
    if sorted(int(s) for s in pp) != got_on:
        why.append(f"the record's per-pair seeds {sorted(pp)} are not its pooled seeds {got_on}")
    side = rec.get("side")
    if _side_of(want_on) != side:
        why.append(f"the record's side {side!r} is not the R1 seeds' side {_side_of(want_on)!r}")
    elif (rec.get("truth") or {}).get("continuation") != TRUTH_BY_SIDE.get(str(side)):
        why.append(f"the record's B2 truth {(rec.get('truth') or {}).get('continuation')!r} is not the {side} side's "
                   f"{TRUTH_BY_SIDE.get(str(side))!r} (AM-X47b-2, AM-X47c-2)")
    if why:
        raise ReadoutInputError("G-C: " + "; ".join(why))
    return list(rec["b1"]), list(rec["b2"])


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m main.x35.gc", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="the seed-pooled G-C record (read-only on its inputs)")
    p.add_argument("--side", required=True, choices=sorted(SIDE_SEEDS))
    p.add_argument("--bank", required=True)
    p.add_argument("--truth", required=True, help="the side's truth rows (TRUTH_BY_SIDE)")
    p.add_argument("--reads", required=True, help="main.policy_spectrum read's output dir (<label>.probs.npz)")
    p.add_argument("--pair", action="append", required=True, help="<seed>=<ON label>:<OFF label>")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pairs: Dict[int, Tuple[str, str]] = {}
    try:
        for spec in a.pair:
            seed, _, labs = spec.partition("=")
            on, _, off = labs.partition(":")
            if not on or not off or int(seed) in pairs:
                raise ReadoutInputError(f"--pair {spec!r}: <seed>=<ON>:<OFF>, each seed once")
            pairs[int(seed)] = (on, off)
        rec = pool(a.side, Path(a.bank), Path(a.truth), Path(a.reads), pairs)
    except (ReadoutInputError, ValueError) as exc:
        print(f"[x35 gc] INCONCLUSIVE: {exc}", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: rec[k] for k in ("side", "seeds", "b1", "b2")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
