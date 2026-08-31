"""Base-rate + decision-class census over CURRENT-ERA eval traces (obs dim 2501).

Feeds an out-of-distribution control study for the constructed "Starmie" risk probe,
which sits in the 5-5 faint class (both sides down to their last mon). The question
this answers is: how RARE is that class in the distribution the model actually plays,
and is the accuracy/KO-race decision shape it exercises available at COMMON faint
counts too?

Everything here is model-free — it reads only the recorded `obs` rows out of
`<run>/eval_traces/step_<N>/<opponent>/<result>_s<seed>_<idx>_states.npz`.

Run:
    python designs/research_state/measurements/starmie_ood_control_traces.py
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Writes:
    starmie_ood_control_traces_2026-08-31.json   (the report)
    starmie_ood_control_traces_stats.npz         (~50 MB scratch: sufficient stats +
                                                  raw obs subsamples; NOT for commit)
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.gen3_data import moves as gen3_moves
from agents.observation.constants import (
    MOVE_SLOT_DIM,
    POKEMON_ACTIVE_OFFSET,
    POKEMON_FULL_DIM,
    POKEMON_HP_OFFSET,
    POKEMON_MOVES_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
)

# --------------------------------------------------------------------------- config

MODELS_DIR = "/home/goodlad/dev/gen3ai/models"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUT_DIR, "starmie_ood_control_traces_2026-08-31.json")
OUT_NPZ = os.path.join(OUT_DIR, "starmie_ood_control_traces_stats.npz")

RUNS = [
    "ai_v9_70_R3ACTION_0828",  # the run the risk probe's checkpoint came from
    "ai_v9_72_R3SELF_0828",
    "ai_v9_75_R4S3c_0829",
    "ai_v9_76_R4ACTION_0830",
    "ai_v9_77_G1LEAN_0830",
]

OBS_DIM = 2501
SEED = 0
N_SAMPLE = 4000
N_SAMPLE_LOW = 2000
LOWFAINT_SET = (2, 3)  # our_faints == opp_faints in {2,3}

BLOCKS: Dict[str, Tuple[int, int]] = {
    "our_team": (0, 732),
    "opp_team": (732, 1464),
    "active_ctx": (1464, 1580),
    "global_env": (1580, 1600),
    "board": (1600, 1617),
    "pair_history": (1617, 1797),
    "event_window": (1797, 2501),
}

# Move-slot layout inside a mon slot (agents/observation/moves.py::get_layout).
MS_ID, MS_POWER, MS_CATEGORY, MS_KNOWN, MS_ACCURACY, MS_NEVER_MISS = 0, 1, 5, 6, 9, 10

# accuracy/KO-race pair thresholds (the Surf-vs-Hydro-Pump shape)
ACC_HI = 95.0
ACC_LO = 85.0


# --------------------------------------------------------------------- small helpers


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=OUT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def opponent_class(opp: str) -> str:
    """bot | sentinel | ext_snapshot — the coarse identity of the eval opponent."""
    if opp.startswith("sentinel_"):
        return "sentinel"
    if opp.startswith("ext_"):
        return "ext_snapshot"
    return "bot"


def is_model_opponent(opp: str) -> bool:
    """True when the opponent is another NETWORK (self-play sentinel or a frozen
    cross-run snapshot) rather than a scripted bot."""
    return opponent_class(opp) != "bot"


def r(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


# ------------------------------------------------------------------ obs decoding


def faint_counts(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(our_faints, opp_faints) per row.

    A slot counts as a faint only when it is REVEALED (species_known == 1) and its HP
    fraction is exactly 0. An unrevealed opponent slot carries species_known == 0 and
    hp == 0 and must not count.
    """
    n = obs.shape[0]
    hp = np.empty((n, 12), dtype=np.float32)
    known = np.empty((n, 12), dtype=np.float32)
    for i in range(12):
        base = i * POKEMON_FULL_DIM
        hp[:, i] = obs[:, base + POKEMON_HP_OFFSET]
        known[:, i] = obs[:, base + POKEMON_SPECIES_KNOWN_OFFSET]
    fainted = (known >= 0.5) & (hp == 0.0)
    return (
        fainted[:, 0:6].sum(axis=1).astype(np.int8),
        fainted[:, 6:12].sum(axis=1).astype(np.int8),
    )


def our_active_slot(obs_row: np.ndarray) -> Optional[int]:
    """Index (0..5) of OUR active mon, or None when no slot is flagged active."""
    for i in range(6):
        if obs_row[i * POKEMON_FULL_DIM + POKEMON_ACTIVE_OFFSET] >= 0.5:
            return i
    return None


def active_move_table(obs_row: np.ndarray, slot: int) -> List[Tuple[int, float, float, float]]:
    """[(move_num, base_power, accuracy_pct, category)] for the active mon's known slots.

    Values are read straight off the OBS (power = raw/200, accuracy = raw/100), which is
    exactly what the network sees; a spot check against `agents.gen3_data.moves` is run
    separately and reported in the JSON.

    Both are ROUNDED back to integers. Gen 3 base powers and accuracies are integers, but
    the obs stores them divided down into float32: `float32(0.85) * 100 == 85.0000024`, so
    an un-rounded `accuracy <= 85` silently REJECTS every 85%-accuracy move — Fire Blast,
    Megahorn and Meteor Mash all fell out of the risky set that way before this rounding
    was added, and the resulting census named Hydro Pump as the pool's only risky move.
    """
    out: List[Tuple[int, float, float, float]] = []
    base0 = slot * POKEMON_FULL_DIM + POKEMON_MOVES_OFFSET
    for k in range(4):
        b = base0 + k * MOVE_SLOT_DIM
        if obs_row[b + MS_KNOWN] < 0.5:
            continue
        out.append(
            (
                int(round(float(obs_row[b + MS_ID]))),
                float(round(float(obs_row[b + MS_POWER]) * 200.0)),
                float(round(float(obs_row[b + MS_ACCURACY]) * 100.0)),
                float(obs_row[b + MS_CATEGORY]),
            )
        )
    return out


def acc_race_pair(
    table: List[Tuple[int, float, float, float]]
) -> Optional[Tuple[int, int]]:
    """The witnessing (accurate_move_num, risky_move_num) pair, or None.

    Returns the WEAKEST qualifying accurate move against the STRONGEST qualifying
    risky one — the widest version of the trade the mon is actually offered.
    """
    dmg = [(num, bp, acc) for (num, bp, acc, cat) in table if cat >= 0.5 and bp > 0.0]
    hi = [(bp, num) for (num, bp, acc) in dmg if acc >= ACC_HI]
    lo = [(bp, num) for (num, bp, acc) in dmg if acc <= ACC_LO]
    if not hi or not lo:
        return None
    lo_bp, lo_num = max(lo)
    hi_bp, hi_num = min(hi)
    if lo_bp > hi_bp:
        return (hi_num, lo_num)
    return None


def has_acc_race_pair(table: List[Tuple[int, float, float, float]]) -> bool:
    """Does the active mon hold an accuracy/KO-race pair?

    Two DAMAGING moves (category physical|special AND base power > 0) where one is
    accurate (acc >= 95) and the other is shakier (acc <= 85) but strictly stronger.
    Type effectiveness is deliberately ignored — this is the shape of the choice, not
    its resolution against a particular defender.
    """
    return acc_race_pair(table) is not None


# ------------------------------------------------------------------- accumulators


class BlockAcc:
    """Streaming per-block mean L2 norm + mean exact-zero fraction."""

    def __init__(self) -> None:
        self.n = 0
        self.l2 = defaultdict(float)
        self.zero = defaultdict(float)

    def add(self, obs: np.ndarray) -> None:
        if obs.shape[0] == 0:
            return
        self.n += obs.shape[0]
        for name, (lo, hi) in BLOCKS.items():
            blk = obs[:, lo:hi]
            self.l2[name] += float(np.linalg.norm(blk.astype(np.float64), axis=1).sum())
            self.zero[name] += float((blk == 0.0).sum()) / (hi - lo)

    def report(self) -> Dict[str, Any]:
        if self.n == 0:
            return {"n": 0, "blocks": {}}
        return {
            "n": self.n,
            "blocks": {
                name: {
                    "mean_l2_norm": r(self.l2[name] / self.n, 4),
                    "mean_zero_fraction": r(self.zero[name] / self.n, 5),
                }
                for name in BLOCKS
            },
        }


class Reservoir:
    """Uniform reservoir over rows, carrying the faint labels alongside."""

    def __init__(self, k: int, rng: np.random.Generator) -> None:
        self.k = k
        self.rng = rng
        self.seen = 0
        self.buf = np.zeros((k, OBS_DIM), dtype=np.float32)
        self.of = np.zeros(k, dtype=np.int8)
        self.pf = np.zeros(k, dtype=np.int8)

    def add(self, obs: np.ndarray, of: np.ndarray, pf: np.ndarray) -> None:
        for i in range(obs.shape[0]):
            if self.seen < self.k:
                j = self.seen
            else:
                j = int(self.rng.integers(0, self.seen + 1))
                if j >= self.k:
                    self.seen += 1
                    continue
            self.buf[j] = obs[i]
            self.of[j] = of[i]
            self.pf[j] = pf[i]
            self.seen += 1

    def filled(self) -> int:
        return min(self.seen, self.k)


# ------------------------------------------------------------------------- main


def main() -> None:
    rng = np.random.default_rng(SEED)

    joint = np.zeros((6, 6), dtype=np.int64)
    joint_by_class: Dict[str, np.ndarray] = {
        c: np.zeros((6, 6), dtype=np.int64) for c in ("bot", "sentinel", "ext_snapshot")
    }
    race_by_cell = np.zeros((6, 6), dtype=np.int64)  # decisions WITH an acc-race pair
    evaluable_by_cell = np.zeros((6, 6), dtype=np.int64)  # decisions with an active mon
    n_total = 0
    n_no_active = 0
    n_1v1 = 0
    n_1v1_and_55 = 0
    race_total = 0
    race_evaluable = 0

    per_run: Dict[str, Dict[str, Any]] = {}
    per_opponent: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"decisions": 0, "battles": 0, "n_55": 0, "n_1v1": 0}
    )
    monotone_violations: List[Dict[str, Any]] = []
    opp_monotone_violations = 0

    acc_pooled, acc_f5, acc_f23 = BlockAcc(), BlockAcc(), BlockAcc()
    sums = np.zeros(OBS_DIM, dtype=np.float64)
    sqs = np.zeros(OBS_DIM, dtype=np.float64)

    res_all = Reservoir(N_SAMPLE, rng)
    res_low = Reservoir(N_SAMPLE_LOW, rng)

    # move-decode spot check: obs-read (bp, acc) vs the dex, on the first 40 distinct nums
    spot: Dict[int, Tuple[float, float]] = {}

    # which move pair actually WITNESSES the acc-race class (pooled / F5 / F2-F3)
    pair_census: Dict[str, Counter] = {
        "pooled": Counter(),
        "F5": Counter(),
        "F2_F3": Counter(),
    }
    risky_census: Dict[str, Counter] = {
        "pooled": Counter(),
        "F5": Counter(),
        "F2_F3": Counter(),
    }
    # every damaging move PRESENT on our active mon, regardless of whether it witnesses
    # a pair — this is what says whether the class is pool-limited or pairing-limited
    present_risky: Counter = Counter()  # accuracy <= ACC_LO
    present_gap: Counter = Counter()  # ACC_LO < accuracy < ACC_HI (the dead band)
    n_with_risky_move = 0

    for run in RUNS:
        pat = os.path.join(MODELS_DIR, run, "eval_traces", "*", "*", "*_states.npz")
        files = sorted(glob.glob(pat))
        run_dec = 0
        run_batt = 0
        run_joint = np.zeros((6, 6), dtype=np.int64)
        run_race = 0
        run_race_eval = 0
        run_steps: Counter = Counter()

        for f in files:
            opp = os.path.basename(os.path.dirname(f))
            step = os.path.basename(os.path.dirname(os.path.dirname(f)))
            try:
                with np.load(f) as d:
                    obs = np.asarray(d["obs"], dtype=np.float32)
                    hs = np.asarray(d["has_state"]).astype(bool) if "has_state" in d else None
            except Exception:
                continue
            if obs.ndim != 2 or obs.shape[1] != OBS_DIM or obs.shape[0] == 0:
                continue
            if hs is not None and hs.shape[0] == obs.shape[0]:
                obs = obs[hs]
            if obs.shape[0] == 0:
                continue

            of, pf = faint_counts(obs)

            # self-check: our_faints must be monotone non-decreasing within a battle
            dif = np.diff(of.astype(np.int16))
            if dif.size and dif.min() < 0:
                monotone_violations.append(
                    {
                        "file": os.path.relpath(f, MODELS_DIR),
                        "n_drops": int((dif < 0).sum()),
                        "seq": [int(x) for x in of.tolist()],
                    }
                )
            dop = np.diff(pf.astype(np.int16))
            if dop.size and dop.min() < 0:
                opp_monotone_violations += 1

            run_batt += 1
            run_steps[step] += 1
            n = obs.shape[0]
            run_dec += n
            n_total += n

            np.add.at(joint, (of, pf), 1)
            np.add.at(run_joint, (of, pf), 1)
            np.add.at(joint_by_class[opponent_class(opp)], (of, pf), 1)

            is55 = (of == 5) & (pf == 5)
            is1v1 = (of == 5) & (pf == 5)  # 6-of == 1 and 6-pf == 1 <=> of == pf == 5
            n_1v1 += int(is1v1.sum())
            n_1v1_and_55 += int((is55 & is1v1).sum())

            po = per_opponent[opp]
            po["decisions"] += n
            po["battles"] += 1
            po["n_55"] += int(is55.sum())
            po["n_1v1"] += int(is1v1.sum())

            # sufficient stats
            o64 = obs.astype(np.float64)
            sums += o64.sum(axis=0)
            sqs += (o64 * o64).sum(axis=0)
            acc_pooled.add(obs)
            m5 = (of == 5) & (pf == 5)
            if m5.any():
                acc_f5.add(obs[m5])
            m23 = (of == pf) & np.isin(of, LOWFAINT_SET)
            if m23.any():
                acc_f23.add(obs[m23])

            res_all.add(obs, of, pf)
            if m23.any():
                res_low.add(obs[m23], of[m23], pf[m23])

            # accuracy/KO-race decision class
            for i in range(n):
                slot = our_active_slot(obs[i])
                if slot is None:
                    n_no_active += 1
                    continue
                table = active_move_table(obs[i], slot)
                for num, bp, acc, _cat in table:
                    if len(spot) < 40 and num not in spot:
                        spot[num] = (bp, acc)
                evaluable_by_cell[of[i], pf[i]] += 1
                run_race_eval += 1
                race_evaluable += 1
                any_risky = False
                for num, bp, acc, cat in table:
                    if cat < 0.5 or bp <= 0.0:
                        continue
                    if acc <= ACC_LO:
                        present_risky[num] += 1
                        any_risky = True
                    elif acc < ACC_HI:
                        present_gap[num] += 1
                if any_risky:
                    n_with_risky_move += 1
                pair = acc_race_pair(table)
                if pair is not None:
                    race_by_cell[of[i], pf[i]] += 1
                    run_race += 1
                    race_total += 1
                    buckets = ["pooled"]
                    if of[i] == 5 and pf[i] == 5:
                        buckets.append("F5")
                    elif of[i] == pf[i] and int(of[i]) in LOWFAINT_SET:
                        buckets.append("F2_F3")
                    for bkt in buckets:
                        pair_census[bkt][pair] += 1
                        risky_census[bkt][pair[1]] += 1

        per_run[run] = {
            "battles": run_batt,
            "decisions": run_dec,
            "steps": dict(sorted(run_steps.items())),
            "n_55": int(run_joint[5, 5]),
            "frac_55": r(run_joint[5, 5] / run_dec, 6) if run_dec else None,
            "diagonal_counts": [int(run_joint[k, k]) for k in range(6)],
            "diagonal_frac": [r(run_joint[k, k] / run_dec, 6) for k in range(6)] if run_dec else None,
            "acc_race_evaluable": run_race_eval,
            "acc_race_frac": r(run_race / run_race_eval, 6) if run_race_eval else None,
        }

    # ------------------------------------------------------------------ report math
    N = float(n_total)
    joint_frac = (joint / N).tolist() if N else []

    total_faints = Counter()
    min_faints = Counter()
    for a in range(6):
        for b in range(6):
            c = int(joint[a, b])
            if c:
                total_faints[a + b] += c
                min_faints[min(a, b)] += c

    def cell_race(a: int, b: int) -> Dict[str, Any]:
        ev = int(evaluable_by_cell[a, b])
        return {
            "decisions": int(joint[a, b]),
            "evaluable": ev,
            "with_acc_race_pair": int(race_by_cell[a, b]),
            "frac": r(race_by_cell[a, b] / ev, 5) if ev else None,
        }

    top_cells = sorted(
        ((int(joint[a, b]), a, b) for a in range(6) for b in range(6)),
        reverse=True,
    )[:5]

    # move-decode spot check against the dex
    by_num: Dict[int, Any] = {}
    for mid in gen3_moves.raw():
        md = gen3_moves.get(mid)
        if md is not None:
            by_num.setdefault(md.num, md)
    spot_rows = []
    spot_mismatch = 0
    for num, (bp, acc) in sorted(spot.items())[:40]:
        md = by_num.get(num)
        if md is None:
            spot_rows.append({"num": num, "dex": None, "obs_bp": r(bp, 2), "obs_acc": r(acc, 2)})
            continue
        # EXACT after rounding — a loose tolerance here is what let the float32
        # 85 -> 85.0000024 threshold bug pass unnoticed.
        ok = md.base_power == int(bp) and md.accuracy == int(acc)
        # bare hiddenpower is overridden to 70bp/Normal by the encoder — expected drift
        if not ok and md.id != "hiddenpower":
            spot_mismatch += 1
        spot_rows.append(
            {
                "num": num,
                "id": md.id,
                "obs_bp": r(bp, 2),
                "dex_bp": md.base_power,
                "obs_acc": r(acc, 2),
                "dex_acc": md.accuracy,
                "match": bool(ok),
            }
        )

    def name_of(num: int) -> str:
        md = by_num.get(num)
        return md.id if md is not None else f"move#{num}"

    def render_pairs(bkt: str) -> List[Dict[str, Any]]:
        tot = sum(pair_census[bkt].values())
        return [
            {
                "accurate": name_of(a),
                "risky": name_of(b),
                "count": c,
                "frac_of_class": r(c / tot, 4) if tot else None,
            }
            for (a, b), c in pair_census[bkt].most_common(12)
        ]

    def render_risky(bkt: str) -> List[Dict[str, Any]]:
        tot = sum(risky_census[bkt].values())
        return [
            {"risky": name_of(m), "count": c, "frac_of_class": r(c / tot, 4) if tot else None}
            for m, c in risky_census[bkt].most_common(12)
        ]

    mean = (sums / N).astype(np.float32) if N else np.zeros(OBS_DIM, np.float32)
    var = np.maximum(sqs / N - (sums / N) ** 2, 0.0) if N else np.zeros(OBS_DIM)
    std = np.sqrt(var).astype(np.float32)

    np.savez_compressed(
        OUT_NPZ,
        mean=mean,
        std=std,
        n=np.int64(n_total),
        sample=res_all.buf[: res_all.filled()],
        sample_our_faints=res_all.of[: res_all.filled()],
        sample_opp_faints=res_all.pf[: res_all.filled()],
        sample_lowfaint=res_low.buf[: res_low.filled()],
        sample_lowfaint_our_faints=res_low.of[: res_low.filled()],
        sample_lowfaint_opp_faints=res_low.pf[: res_low.filled()],
    )

    report: Dict[str, Any] = {
        "meta": {
            "generated": "2026-08-31",
            "git_head": git_head(),
            "models_dir": MODELS_DIR,
            "runs": RUNS,
            "obs_dim": OBS_DIM,
            "n_runs": len(RUNS),
            "n_battles": sum(v["battles"] for v in per_run.values()),
            "n_decisions": n_total,
            "per_run": per_run,
            "per_opponent": {
                k: {
                    **v,
                    "class": opponent_class(k),
                    "frac_55": r(v["n_55"] / v["decisions"], 6) if v["decisions"] else None,
                }
                for k, v in sorted(per_opponent.items())
            },
            "seed": SEED,
            "companion_npz": OUT_NPZ,
            "decoding": {
                "POKEMON_FULL_DIM": POKEMON_FULL_DIM,
                "hp_offset": POKEMON_HP_OFFSET,
                "species_known_offset": POKEMON_SPECIES_KNOWN_OFFSET,
                "moves_offset": POKEMON_MOVES_OFFSET,
                "move_slot_dim": MOVE_SLOT_DIM,
                "active_offset": POKEMON_ACTIVE_OFFSET,
                "move_slot_fields_used": {
                    "id": MS_ID,
                    "base_power_norm200": MS_POWER,
                    "category": MS_CATEGORY,
                    "known": MS_KNOWN,
                    "accuracy_norm100": MS_ACCURACY,
                    "never_miss": MS_NEVER_MISS,
                },
                "source_of_bp_and_accuracy": "OBS (raw = obs*200 / obs*100); spot-checked against agents.gen3_data.moves",
                "spot_check_mismatches": spot_mismatch,
                "spot_check_rows": spot_rows,
            },
        },
        "self_check": {
            "our_faints_monotone_violations": len(monotone_violations),
            "our_faints_monotone_violation_examples": monotone_violations[:10],
            "opp_faints_monotone_violation_battles": opp_monotone_violations,
            "decisions_with_no_active_our_slot": n_no_active,
        },
        "base_rate": {
            "n_decisions": n_total,
            "joint_counts_our_by_opp": joint.tolist(),
            "joint_fractions_our_by_opp": [[r(x, 7) for x in row] for row in joint_frac],
            "joint_readable": {
                f"our{a}_opp{b}": {
                    "count": int(joint[a, b]),
                    "frac": r(joint[a, b] / N, 7) if N else None,
                }
                for a in range(6)
                for b in range(6)
            },
            "joint_by_opponent_class": {
                cls: {
                    "n_decisions": int(m.sum()),
                    "counts": m.tolist(),
                    "frac_55": r(m[5, 5] / m.sum(), 7) if m.sum() else None,
                    "diagonal_frac": [r(m[k, k] / m.sum(), 6) for k in range(6)]
                    if m.sum()
                    else None,
                }
                for cls, m in joint_by_class.items()
            },
            "bot_vs_model_opponent": {
                "bot": {
                    "n_decisions": int(joint_by_class["bot"].sum()),
                    "frac_55": r(
                        joint_by_class["bot"][5, 5] / max(joint_by_class["bot"].sum(), 1), 7
                    ),
                },
                "model_opponent_sentinel_plus_ext": {
                    "n_decisions": int(
                        joint_by_class["sentinel"].sum() + joint_by_class["ext_snapshot"].sum()
                    ),
                    "frac_55": r(
                        (joint_by_class["sentinel"][5, 5] + joint_by_class["ext_snapshot"][5, 5])
                        / max(
                            joint_by_class["sentinel"].sum()
                            + joint_by_class["ext_snapshot"].sum(),
                            1,
                        ),
                        7,
                    ),
                },
            },
            "total_faints_marginal": {
                str(k): {"count": int(v), "frac": r(v / N, 7)}
                for k, v in sorted(total_faints.items())
            },
            "min_faints_marginal": {
                str(k): {"count": int(v), "frac": r(v / N, 7)}
                for k, v in sorted(min_faints.items())
            },
            "headline": {
                "P_our5_opp5": r(joint[5, 5] / N, 7),
                "P_our5_opp5_and_1v1_endgame": r(n_1v1_and_55 / N, 7),
                "P_1v1_endgame": r(n_1v1 / N, 7),
                "note_5_5_is_identically_1v1": "our_alive = 6-our_faints, opp_alive = 6-opp_faints, so (5,5) IS the 1v1 endgame; the three numbers coincide by construction",
                "P_diagonal_F": {
                    str(k): r(joint[k, k] / N, 7) for k in range(6)
                },
                "P_total_faints_ge_10": r(
                    sum(v for k, v in total_faints.items() if k >= 10) / N, 7
                ),
                "percentile_of_5_5_cell": {
                    "frac_decisions_with_total_faints_ge_10": r(
                        sum(v for k, v in total_faints.items() if k >= 10) / N, 7
                    ),
                    "percentile": r(
                        100.0
                        * (1.0 - sum(v for k, v in total_faints.items() if k >= 10) / N),
                        4,
                    ),
                },
            },
        },
        "acc_race_class": {
            "definition": (
                "the active mon's KNOWN move slots contain two damaging moves "
                f"(category in {{physical,special}} and base power > 0) with one at accuracy >= {ACC_HI} "
                f"and another at accuracy <= {ACC_LO} carrying strictly higher base power "
                "(the Surf-vs-Hydro-Pump shape). Type effectiveness ignored."
            ),
            "overall": {
                "evaluable_decisions": race_evaluable,
                "with_acc_race_pair": race_total,
                "frac": r(race_total / race_evaluable, 5) if race_evaluable else None,
            },
            "diagonal": {f"F{k}": cell_race(k, k) for k in range(6)},
            "top5_cells": [
                {"our_faints": a, "opp_faints": b, **cell_race(a, b)}
                for (_c, a, b) in top_cells
            ],
            "witnessing_pairs": {
                "pooled": render_pairs("pooled"),
                "diagonal_F5": render_pairs("F5"),
                "diagonal_F2_F3": render_pairs("F2_F3"),
            },
            "availability": {
                "note": (
                    "Is the class rare because the PAIRING is rare, or because the risky "
                    "move itself is absent from the eval team pool? These are the moves "
                    "present on our active mon at all."
                ),
                "decisions_with_any_risky_damaging_move": n_with_risky_move,
                "frac_of_evaluable": r(n_with_risky_move / race_evaluable, 5)
                if race_evaluable
                else None,
                "risky_moves_present_acc_le_85": [
                    {"move": name_of(m), "count": c}
                    for m, c in present_risky.most_common(20)
                ],
                "dead_band_moves_present_85_lt_acc_lt_95": [
                    {"move": name_of(m), "count": c}
                    for m, c in present_gap.most_common(20)
                ],
            },
            "witnessing_risky_move": {
                "pooled": render_risky("pooled"),
                "diagonal_F5": render_risky("F5"),
                "diagonal_F2_F3": render_risky("F2_F3"),
            },
            "full_grid_frac": [
                [
                    r(race_by_cell[a, b] / evaluable_by_cell[a, b], 5)
                    if evaluable_by_cell[a, b]
                    else None
                    for b in range(6)
                ]
                for a in range(6)
            ],
        },
        "obs_blocks": {
            "block_offsets": {k: list(v) for k, v in BLOCKS.items()},
            "pooled": acc_pooled.report(),
            "diagonal_F5": acc_f5.report(),
            "diagonal_F2_F3": acc_f23.report(),
        },
        "npz_companion": {
            "path": OUT_NPZ,
            "arrays": {
                "mean": [OBS_DIM],
                "std": [OBS_DIM],
                "n": n_total,
                "sample": [res_all.filled(), OBS_DIM],
                "sample_our_faints": [res_all.filled()],
                "sample_opp_faints": [res_all.filled()],
                "sample_lowfaint": [res_low.filled(), OBS_DIM],
                "sample_lowfaint_our_faints": [res_low.filled()],
                "sample_lowfaint_opp_faints": [res_low.filled()],
            },
            "lowfaint_pool_size": acc_f23.n,
            "lowfaint_sample_truncated": res_low.filled() < N_SAMPLE_LOW,
            "committed": False,
            "note": "scratch artifact (~tens of MB); left on disk, NOT to be committed",
        },
        "caveats": [
            "Eval traces are a SELECTED sample of play: they come from the periodic eval "
            "callback (fixed bot opponents + self-play sentinels + frozen cross-run "
            "snapshots), not from the training rollout distribution. Base rates here are "
            "'what the model does in eval', not 'what it does in training'.",
            "A trace file records only the TRAINEE's decision points, so the per-decision "
            "faint counts are sampled at OUR decision times — a faint that happens and is "
            "immediately followed by the battle ending may never appear as a decision row.",
            "(our_faints==5, opp_faints==5) and 'exactly 1 alive each side' are the SAME "
            "predicate given opp_alive = 6 - opp_faints, so those two headline numbers are "
            "identical by construction rather than by measurement.",
            "opp_faints undercounts a side whose mons are unrevealed only in the sense that "
            "an unrevealed mon cannot be fainted in gen3 singles as recorded here; a REVIVED "
            "mon (not a gen3 OU mechanic) would break the monotonicity self-check.",
            "The accuracy/KO-race class is computed from the OBS move slots, which for OUR "
            "own team are always known. Fixed-damage moves (Seismic Toss, Endeavor) carry "
            "base_power 0 in the dex and are therefore NOT counted as damaging here.",
            "Never-miss moves encode accuracy 100 and so land on the ACCURATE side of the "
            "pair; that is intended (they are the safe option in a KO race).",
            "The pair test does NOT require the accurate move to be a serious attacking "
            "alternative — a low-power utility move that happens to be damaging (Rapid Spin, "
            "20 bp / 100 acc) satisfies the accurate side. See acc_race_class.witnessing_pairs "
            "for what actually carries the class before treating the frequency as a count of "
            "genuine KO races.",
            "The acc >= 95 / acc <= 85 thresholds leave a DEAD BAND at 85 < acc < 95 that "
            "excludes Rock Slide (90) and Overheat (90) outright — a move in the band can "
            "neither open nor close a pair. That is the spec's definition, not an accident, "
            "but it is why the measured class is narrower than 'any accuracy trade-off'.",
            "Base power / accuracy are read from the obs (obs*200 / obs*100) rather than "
            "re-resolved per move id, then ROUNDED to integers before thresholding: float32 "
            "makes 0.85*100 == 85.0000024, so an un-rounded 'accuracy <= 85' drops Fire "
            "Blast, Megahorn and Meteor Mash entirely. A 40-move spot check against "
            "agents.gen3_data.moves asserts EXACT equality after rounding and is reported in "
            "meta.decoding.",
            "Block L2 norms mix raw-scalar channels (species nums, move nums, type ids) with "
            "normalized ones, so they are a reference fingerprint for an OOD distance, not a "
            "calibrated magnitude.",
            "Runs differ in eval opponent SET (the 0828 runs carry sentinel_0..4; the 0829/0830 "
            "runs carry an ext_ frozen snapshot instead), so the pooled opponent mix is not "
            "balanced across runs.",
        ],
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)

    # ------------------------------------------------------------------- stdout
    print(f"runs={len(RUNS)} battles={report['meta']['n_battles']} decisions={n_total}")
    print("diagonal P(our==opp==F):")
    for k in range(6):
        print(f"  F={k}: {joint[k, k] / N:.5f}  (n={int(joint[k, k])})")
    print(f"P(5,5) = {joint[5, 5] / N:.6f}")
    print(f"P(1v1 endgame) = {n_1v1 / N:.6f}")
    print(f"P(total faints >= 10) = {sum(v for k, v in total_faints.items() if k >= 10) / N:.6f}")
    print(f"acc-race class overall: {race_total}/{race_evaluable} = {race_total / max(race_evaluable,1):.4f}")
    print("acc-race by diagonal cell:")
    for k in range(6):
        c = cell_race(k, k)
        print(f"  F={k}: {c['with_acc_race_pair']}/{c['evaluable']} = {c['frac']}")
    print(f"monotone violations: {len(monotone_violations)}  no-active rows: {n_no_active}")
    print(f"spot-check mismatches: {spot_mismatch}")
    print(f"decisions with ANY risky (acc<=85) damaging move: {n_with_risky_move} "
          f"({n_with_risky_move / max(race_evaluable,1):.4f})")
    print("risky moves present:", [(name_of(m), c) for m, c in present_risky.most_common(8)])
    print(
        "dead-band (85<acc<95) moves present:",
        [(name_of(m), c) for m, c in present_gap.most_common(8)],
    )
    print(f"lowfaint pool={acc_f23.n} sample={res_low.filled()}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
