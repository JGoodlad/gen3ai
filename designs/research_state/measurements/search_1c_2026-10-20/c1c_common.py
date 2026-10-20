"""STAGE 1c's shared constants and PURE helpers (``PREREGISTRATION.md``). Importable by the driver (``run_1c.py``),
the play side (``play_1c.py``) and the read (``read_1c.py``) alike: numpy and the stdlib at import; the gen-3 helpers
are imported inside the functions that need them.

THE BANK is stage 1's (``../world_sampler_stage1_2026-10-19/s1_common.py``: the maturity battery's interim LEAN states
at X26 225M that the banked F could build, 287 ids, sha256 ``cf8d8c6f…``), in stage 1's registered order.

ARCHIVE (``$S1C_ARCHIVE``, default ``~/gen3ai_archive/search_1c``)::

    rows/rows.jsonl          one row per state: the proposals' weights, likelihoods, per-world facts, the resamples
    worlds/worlds.jsonl      one row per state: every proposal's team, root prefix and rolls (what a playout needs)
    recheck/rows.jsonl       G3: fixed states rebuilt and rescored in a fresh process
    play/<arm>/rows.jsonl    phase B: the pilot's playout rows of each played arm
    ws1/                     stage 1's helper modules' caches (the corpus), never stage 1's own archive
    logs/, STOP_AT, A_DONE, A_STOPPED
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
REPO = HERE.parents[3]
STAGE1 = MEAS / "world_sampler_stage1_2026-10-19"
L1DIR = MEAS / "search_l1_l5_2026-10-20"
L3DIR = MEAS / "search_l3_2026-10-20"
ARCH = Path(os.environ.get("S1C_ARCHIVE") or (Path.home() / "gen3ai_archive" / "search_1c"))
L3_ARCH = Path.home() / "gen3ai_archive" / "search_l3"

#: the proposal sets (PREREGISTRATION §2): N_F honest F worlds and N_C corpus (C_ladder) worlds per state
N_F = 128
N_C = 128
#: worlds a search plays (the resample's k; F8 / C8 are the first J of a set)
J = 8
MAX_TRIES = 60

#: the ARMS (PREREGISTRATION §3): name -> (the proposals: [(set, the first n of it)], the weighting).
#: "uniform" = equal weights; "dice" = exp(log_w); "dice*opp" = exp(log_w + log L). The first n of a set is itself a
#: draw of n worlds (each world j has its own key-seeded rng), so the 64-world arms are nested in the 128-world ones.
ARMS: Dict[str, tuple] = {
    "F8": ([("F", 8)], "uniform"),
    "F128L": ([("F", 128)], "dice*opp"),
    "C128L": ([("C", 128)], "dice*opp"),
    "M256L": ([("F", 128), ("C", 128)], "dice*opp"),
    "F64L": ([("F", 64)], "dice*opp"),
    "C64L": ([("C", 64)], "dice*opp"),
    "M128L": ([("F", 64), ("C", 64)], "dice*opp"),
    "F128d": ([("F", 128)], "dice"),
    "C128d": ([("C", 128)], "dice"),
    "C8": ([("C", 8)], "uniform"),
}
PRIMARY = ("F128L", "C128L", "M256L")
#: the brief's N = 64 (nested), read as a SECONDARY family and for the N-scaling
SECONDARY_64 = ("F64L", "C64L", "M128L")
BASELINE = "F8"
#: the weighted arms whose resampled 8 are stored (F8 / C8 play their first 8)
RESAMPLED = ("F128L", "C128L", "M256L", "F64L", "C64L", "M128L", "F128d", "C128d")
#: the DESCRIPTIVE tempered likelihood (fixed in advance, never fitted): exp(log_w + TEMPER * log L)
TEMPER = 0.5


def arm_sets(arm: str) -> List[str]:
    return [s for s, _ in ARMS[arm][0]]


def arm_slice(row_sets: Dict[str, list], arm: str) -> list:
    """The arm's proposals' entries of a per-world list (``row_sets[set]``), in the arm's order (F before C)."""
    return [x for s, n in ARMS[arm][0] for x in row_sets[s][:n]]


RESAMPLE_KEY = "search_1c:{arm}:{id}"
RECHECK_SALT = "search_1c_recheck_v1:"
SMOKE_SALT = "search_1c_smoke_v1:"
IDENTITY_SALT = "search_1c_identity_v1:"
N_RECHECK = 6
N_IDENTITY = 12


def d(*parts) -> Path:
    p = ARCH.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:          # a torn last line (a kill mid-write)
                    pass
    return out


def read_by_id(path: Path) -> Dict[str, dict]:
    """{id: the FIRST row with that id} (a resumed run never rewrites a state)."""
    out: Dict[str, dict] = {}
    for r in read_rows(path):
        out.setdefault(r["id"], r)
    return out


def write_row(f, row: dict) -> None:
    f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    f.flush()
    os.fsync(f.fileno())


def stop_at() -> Optional[float]:
    p = ARCH / "STOP_AT"
    return float(p.read_text().split()[0]) if p.exists() else None


# ------------------------------------------------------------------------------------------------
# the facts meter (PREREGISTRATION §4): one world's agreement with the truth on every hidden class
# ------------------------------------------------------------------------------------------------

def canon(m: str) -> str:
    """A Hidden Power is ONE move whatever its type (its type is a spread fact)."""
    return "hiddenpower" if m.startswith("hiddenpower") else m


def world_facts(true_team: str, world_team: str, kept: Dict[str, Sequence[str]],
                item_revealed: Dict[str, bool]) -> Dict[str, int]:
    """The world's agreement with the TRUTH on what the searched side has not seen (counts; the read forms the
    shares). ``kept``: {seen opponent species: its revealed moves}; ``item_revealed``: {seen species: play named its
    original item}.

      mh / mt   the seen mons' UNREVEALED true moves the world's same mon holds / their number
      me        1 iff every seen mon's unrevealed move set is exact (1 when there is none)
      uh / ut   the true never-seen species the world holds among its never-seen mons / their number
      ue        1 iff the world's never-seen species SET is the truth's (1 when there is none)
      ih / it   the seen mons with an UNREVEALED item whose world item is the truth's / their number
      sh / st   the seen mons whose world spread (nature, EVs, IVs) is exactly the truth's / their number
      umh       the true never-seen mons whose same-species world mon holds their exact move set
    """
    from agents.info_levers.packed import Spread, item_id, parse_set, species_id, typed_moves

    tm = {species_id(f): f for f in (parse_set(e) for e in true_team.split("]"))}
    wm = {species_id(f): f for f in (parse_set(e) for e in world_team.split("]"))}
    ks = {s: {canon(m) for m in v} for s, v in kept.items()}
    out = dict(mh=0, mt=0, me=1, uh=0, ut=0, ue=1, ih=0, it=0, sh=0, st=0, umh=0)
    for sid, rev in ks.items():
        if sid not in tm:
            raise ValueError(f"a seen species {sid!r} the true team does not hold")
        if sid not in wm:
            raise ValueError(f"a seen species {sid!r} the world does not hold")
        ft, fw = tm[sid], wm[sid]
        tu = {canon(m) for m in typed_moves(ft)} - rev
        wu = {canon(m) for m in typed_moves(fw)} - rev
        if tu:
            out["mh"] += len(tu & wu)
            out["mt"] += len(tu)
            if tu != wu:
                out["me"] = 0
        if not item_revealed.get(sid, False):
            out["it"] += 1
            out["ih"] += int(item_id(fw) == item_id(ft))
        out["st"] += 1
        out["sh"] += int(Spread.of(fw).same(Spread.of(ft)))
    never_t = [s for s in tm if s not in ks]
    never_w = {s for s in wm if s not in ks}
    out["ut"] = len(never_t)
    out["uh"] = sum(1 for s in never_t if s in never_w)
    out["ue"] = int(set(never_t) == never_w)
    out["umh"] = sum(1 for s in never_t if s in wm and {canon(m) for m in typed_moves(wm[s])}
                     == {canon(m) for m in typed_moves(tm[s])})
    return out


def near_copy(true_team: str, world_team: str, near: int = 4) -> bool:
    """The world shares >= ``near`` whole mon sets (species + move set, Hidden Power untyped) with the truth
    (stage 1's near-copy meter)."""
    from agents.info_levers.packed import parse_set, species_id, typed_moves

    def sets(team: str):
        return {species_id(f): frozenset(canon(m) for m in typed_moves(f)) for f in (parse_set(e) for e in team.split("]"))}

    t, w = sets(true_team), sets(world_team)
    return sum(1 for s, mv in w.items() if t.get(s) == mv) >= near


def ids_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(ids).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------
# AMENDMENT A (2026-10-20; PREREGISTRATION §11): the STRICT corpus read, the permuted-L placebo, the play indices
# ------------------------------------------------------------------------------------------------

#: A1: the arms holding corpus proposals, read STRICT (every near4 proposal's weight 0, the rest renormalised)
CORPUS_ARMS = ("C128L", "M256L", "C64L", "M128L", "C128d", "C8")
#: A3: the permuted-L placebo's key
PLACEBO_SALT = "search_1c_placebo_v1:"
#: A1: a corpus arm's phase-B worlds are resampled from its STRICT weights under this key
STRICT_KEY = "search_1c:{arm}:strict:{id}"


def placebo_perm(n: int, arm: str, did: str) -> List[int]:
    """A3: the fixed permutation of a state's n proposals (sha256 of ``search_1c_placebo_v1:<arm>:<id>``)."""
    import numpy as np

    seed = int(hashlib.sha256(f"{PLACEBO_SALT}{arm}:{did}".encode()).hexdigest()[:16], 16)
    return [int(x) for x in np.random.default_rng(seed).permutation(n)]


def arm_weights(row: dict, arm: str, strict: bool = False, temper: float = 1.0, placebo: bool = False,
                weighting: Optional[str] = None):
    """(the arm's normalised weights over its proposals, their facts, their near4 flags), ``"strict-excluded"`` when the
    STRICT filter leaves no proposal, or None when the arm is not valid on the state. ``weighting`` overrides the arm's
    own ("uniform" / "dice" / "dice*opp"); ``temper`` scales log L; ``placebo`` permutes log L across the proposals
    (A3) before the weighting; ``strict`` zeroes every near4 proposal (A1)."""
    import numpy as np

    from main.value_search.opp_likelihood import normalise, posterior_log_weights

    parts, wt = ARMS[arm]
    wt = weighting or wt
    sets = [s for s, _ in parts]
    if not row.get("ok") or row.get("defect") or any(not (row.get(s) or {}).get("ok") for s in sets):
        return None
    if any("facts" not in row[s] for s in sets):
        return None
    facts = arm_slice({s: row[s]["facts"] for s in sets}, arm)
    near = arm_slice({s: row[s]["near4"] for s in sets}, arm)
    n = len(facts)
    if wt == "uniform":
        lw = np.zeros(n)
    else:
        dice = arm_slice({s: row[s]["log_w"] for s in sets}, arm)
        ll = None
        if wt == "dice*opp":
            ll = arm_slice({s: row[s]["ll"] for s in sets}, arm)
            if placebo:
                p = placebo_perm(n, arm, row["id"])
                ll = [ll[i] for i in p]
            ll = [temper * x for x in ll]
        lw = posterior_log_weights(n, dice, ll)
    if strict:
        keep = np.array([not x for x in near], dtype=bool)
        if not keep.any():
            return "strict-excluded"
        lw = np.where(keep, lw, -np.inf)
    return normalise(lw), facts, near


def play_indices(row: dict, arm: str) -> Optional[List[int]]:
    """The proposal indices (into the arm's own proposal list) an arm PLAYS in phase B: a corpus arm (A1) resampled
    from its STRICT weights (systematic, key ``STRICT_KEY``); F8 its first 8; any other weighted arm the driver's own
    stored resample. None when the arm is not valid (or STRICT-excluded) on the state."""
    from main.value_search.opp_likelihood import resample_uniform, systematic

    if arm in CORPUS_ARMS:
        p = arm_weights(row, arm, strict=True)
        if p is None or isinstance(p, str):
            return None
        return systematic(p[0], J, resample_uniform(STRICT_KEY.format(arm=arm, id=row["id"])))
    if ARMS[arm][1] == "uniform":
        return list(range(J))
    rec = (row.get("resample") or {}).get(arm)
    return None if rec is None else list(rec["idx"])
