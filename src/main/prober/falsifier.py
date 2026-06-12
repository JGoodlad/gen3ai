"""Dice-attribution falsifier (Phase 1) — was a loss decision bad luck or a
reducible mistake?

Consumes the Phase-0 reconstruction layer (``utils.bridge.reconstruction`` +
``agents.training.obs_materializer``) to answer, for an anchored decision in a
recorded battle, two separable questions:

**Luck** — *holding both players' actions fixed*, re-roll the turn under N
fresh PRNG seeds and ask where the REALIZED outcome sits in that dice
distribution. The realized line is produced by the special ``"original"`` seed
(no PRNG swap + exact recorded follow-ups), scored through the same outcome
pipeline as the re-rolls, so its margin is directly comparable. A realized
margin in the bad tail ⇒ the dice broke bad.

**Mistake** — under the SAME N seeds (common random numbers: each alternative
faces the identical dice stream, so the comparison is *paired* and the
advantage estimate is far tighter than two independent samples), replay the
turn with each of the top-k alternative legal actions (ranked by the saved
policy logits) instead of the chosen one. A paired advantage that clears both a
material threshold and a z-test ⇒ a better action existed ⇒ reducible mistake.

Both axes score a **material margin** computed from the omniscient re-roll
outcome — ``(our_alive − opp_alive) + (our_hp_frac − opp_hp_frac)``. This is
referee-side ANALYSIS (what actually happened on the board), which is allowed
to be omniscient; the one-sided/omniscient wall constrains only what feeds the
encoder/critic. (A critic-scored variant — V(s′) over materialized one-sided
obs, the distributional-critic calibration probe — is a deliberate follow-up,
not built here.)

Verdicts (v1 heuristics, constants below):
- ``MISTAKE`` — a significantly better alternative existed, dice were fair.
- ``LUCK``    — realized outcome in the worst ``_LUCK_PCTL_MAX`` tail, no
  better alternative.
- ``MIXED``   — both: a better line existed AND the dice also broke bad.
- ``NEUTRAL`` — neither: outcome near the middle of the dice distribution and
  no alternative beats the chosen action.

Anchoring: re-rolls anchor at START-OF-TURN move rounds (Phase-0 limit), so
only ``move_selection`` invocations are falsifiable; a forced-switch anchor is
remapped to its turn's move decision when one exists. Default anchors = the
``worst`` most-negative TD-residual decisions (δ = r + γV(s′) − V(s), the same
formula the prober uses everywhere) on distinct turns.

Alternatives that the sim refuses (a maybe-trapped switch the request couldn't
rule out) are detected from the ``[Unavailable choice]`` error in that arm's
one-sided suffix protocol and excluded from the verdict when mostly refused —
a refused "alternative" was never actually available.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np

from agents.training.obs_materializer import materialize_from_record
from utils.bridge.reconstruction import ReconstructionRecord, reroll_turn

# v1 verdict thresholds — deliberately simple, documented, and centralized.
_LUCK_PCTL_MAX = 0.20   # realized outcome in the worst quintile of the dice dist
_ADV_MIN = 0.5          # a better alt must win ≥ half a mon-equivalent of material
_Z = 1.96               # ...and clear a 95% paired z-test
_REFUSED_MAX = 0.5      # an alt refused on most seeds was never really available

DEFAULT_SEEDS = 40
DEFAULT_ALTS = 3


# ---------------------------------------------------------------------------
# Pure scoring helpers (unit-tested directly)
# ---------------------------------------------------------------------------

def material_margin(outcome: dict, side: str) -> float:
    """Material standing from ``side``'s perspective after a (re-)rolled turn:
    ``(our_alive − opp_alive) + (our_hp_frac − opp_hp_frac)``. Alive counts
    dominate (a mon ≈ 1.0); the HP-fraction term breaks ties within equal
    material. Computed from the omniscient outcome — referee-side analysis."""
    other = "p2" if side == "p1" else "p1"
    us, them = outcome[side], outcome[other]
    alive = float(us["alive"] - them["alive"])
    hp = (us["team_hp"] / max(1.0, us["team_maxhp"])
          - them["team_hp"] / max(1.0, them["team_maxhp"]))
    return alive + hp


def percentile_of(value: float, samples: Sequence[float]) -> Optional[float]:
    """Midrank percentile of ``value`` within ``samples`` ∈ [0, 1]; low = the
    realized outcome was worse than most of the dice distribution."""
    if not samples:
        return None
    below = sum(1 for s in samples if s < value - 1e-12)
    ties = sum(1 for s in samples if abs(s - value) <= 1e-12)
    return (below + 0.5 * ties) / len(samples)


def paired_stats(diffs: Sequence[float]) -> "tuple[float, float]":
    """Mean and standard error of paired per-seed differences (alt − chosen on
    the same seed). Common random numbers make this the right estimator — the
    dice noise cancels within each pair instead of inflating both arms."""
    n = len(diffs)
    if n == 0:
        return 0.0, float("inf")
    mean = float(np.mean(diffs))
    if n == 1:
        return mean, float("inf")
    se = float(np.std(diffs, ddof=1) / math.sqrt(n))
    return mean, se


def classify(luck_pctl: Optional[float], best_adv: Optional[float],
             best_se: Optional[float]) -> str:
    """The v1 verdict matrix (module-header semantics)."""
    bad_luck = luck_pctl is not None and luck_pctl <= _LUCK_PCTL_MAX
    better_alt = (
        best_adv is not None and best_se is not None
        and best_adv > _ADV_MIN and best_adv > _Z * best_se
    )
    if better_alt and bad_luck:
        return "MIXED"
    if better_alt:
        return "MISTAKE"
    if bad_luck:
        return "LUCK"
    return "NEUTRAL"


def fresh_seeds(n: int, salt: str) -> List[str]:
    """``n`` deterministic gen-5 seeds derived from ``salt`` (battle+decision),
    so a re-run of the same falsification reproduces exactly."""
    out = []
    for i in range(n):
        h = hashlib.sha256(f"{salt}:{i}".encode()).digest()
        out.append(",".join(str(int.from_bytes(h[j:j + 2], "big")) for j in (0, 2, 4, 6)))
    return out


def _was_refused(reroll, side: str) -> bool:
    chunks = reroll.p1_chunks if side == "p1" else reroll.p2_chunks
    return any("[Unavailable choice]" in c or "[Invalid choice]" in c for c in chunks)


# ---------------------------------------------------------------------------
# Decision-level falsification
# ---------------------------------------------------------------------------

def falsify_decision(
    record: ReconstructionRecord,
    summary: dict,
    npz: dict,
    inv_index: int,
    *,
    n_seeds: int = DEFAULT_SEEDS,
    n_alts: int = DEFAULT_ALTS,
    followup: str = "random",
    mappings=None,
) -> dict:
    """Luck/mistake attribution for one ``move_selection`` decision."""
    invs = summary.get("invocations", [])
    if not (0 <= inv_index < len(invs)):
        raise IndexError(f"inv {inv_index} out of range (battle has {len(invs)})")
    inv = invs[inv_index]
    if inv.get("phase") != "move_selection":
        raise ValueError(
            f"inv {inv_index} is a {inv.get('phase')!r} decision — re-rolls anchor at "
            f"start-of-turn move rounds; falsify that turn's move_selection invocation"
        )
    turn = int(inv["turn"])
    if "actions" not in npz:
        raise ValueError("states.npz has no 'actions' array — trace predates the "
                         "reconstruction layer; only newer bridge-eval traces are falsifiable")
    if not record.trainee_username:
        raise ValueError("reconstruction record lacks trainee_username")
    side = record.side_of(record.trainee_username)

    actions = np.asarray(npz["actions"], dtype=int)
    chosen_idx = int(actions[inv_index])

    # One materializer pass to the anchored decision: rebuilds the agent's exact
    # state and maps every LEGAL action index → the sim choice string the live
    # mapper would emit (zero mapping reimplementation; legality = the live mask).
    trace = materialize_from_record(
        record, actions=actions, mappings=mappings,
        map_actions_at=inv_index, stop_after_decision=inv_index,
    )
    if len(trace.decisions) != inv_index + 1:
        raise RuntimeError(
            f"replay desync: materializer produced {len(trace.decisions)} decisions "
            f"for inv {inv_index} — record and trace disagree")
    if trace.decisions[-1].turn != turn:
        raise RuntimeError(
            f"replay desync: decision {inv_index} replays at turn "
            f"{trace.decisions[-1].turn}, summary says {turn}")
    choice_map: Dict[int, str] = trace.action_choices or {}
    if chosen_idx not in choice_map:
        raise RuntimeError(f"chosen action {chosen_idx} not legal in replayed state")

    # Alternatives: top-k legal actions by the saved policy logits, minus chosen.
    logits = npz.get("logits")
    alt_order = sorted(
        (j for j in choice_map if j != chosen_idx),
        key=(lambda j: -float(logits[inv_index][j])) if logits is not None else int,
    )
    alt_idxs = alt_order[:max(0, n_alts)]

    seeds = fresh_seeds(n_seeds, salt=f"{record.battle_tag}:{inv_index}")
    fix_both = {f"{side}_action": "recorded",
                f"{'p2' if side == 'p1' else 'p1'}_action": "recorded"}

    # Chosen arm: fix both actions; fresh seeds = the dice distribution,
    # "original" = the realized line.
    chosen_rr = reroll_turn(record, turn, seeds=seeds + ["original"],
                            followup=followup, **fix_both)
    by_seed = {r.seed: r for r in chosen_rr.rerolls}
    realized = material_margin(by_seed["original"].outcome, side)
    chosen_margins = [material_margin(by_seed[s].outcome, side)
                      for s in seeds if not by_seed[s].outcome.get("stuck")]
    luck_pctl = percentile_of(realized, chosen_margins)

    # Alternative arms: same seeds (paired), our side plays the alt.
    other_side = "p2" if side == "p1" else "p1"
    alts = []
    for alt in alt_idxs:
        rr = reroll_turn(
            record, turn, seeds=seeds, followup=followup,
            **{f"{side}_action": choice_map[alt], f"{other_side}_action": "recorded"},
        )
        alt_by_seed = {r.seed: r for r in rr.rerolls}
        diffs, refused = [], 0
        for s in seeds:
            a = alt_by_seed[s]
            if _was_refused(a, side):
                refused += 1
                continue
            if a.outcome.get("stuck") or by_seed[s].outcome.get("stuck"):
                continue
            diffs.append(material_margin(a.outcome, side)
                         - material_margin(by_seed[s].outcome, side))
        adv, se = paired_stats(diffs)
        alts.append({
            "action": alt, "label": _label_of(inv, alt), "choice": choice_map[alt],
            "paired_advantage": round(adv, 4), "advantage_se": round(se, 4),
            "n_pairs": len(diffs), "refused_frac": round(refused / max(1, n_seeds), 3),
        })

    usable = [a for a in alts if a["refused_frac"] <= _REFUSED_MAX and a["n_pairs"] > 1]
    best = max(usable, key=lambda a: a["paired_advantage"], default=None)
    verdict = classify(
        luck_pctl,
        best["paired_advantage"] if best else None,
        best["advantage_se"] if best else None,
    )

    return {
        "inv": inv_index,
        "turn": turn,
        "chosen": {"action": chosen_idx, "label": _label_of(inv, chosen_idx),
                   "choice": choice_map[chosen_idx]},
        "n_seeds": n_seeds,
        "realized_margin": round(realized, 4),
        "dice_distribution": {
            "mean": round(float(np.mean(chosen_margins)), 4) if chosen_margins else None,
            "std": round(float(np.std(chosen_margins)), 4) if chosen_margins else None,
            "n": len(chosen_margins),
        },
        "luck_percentile": round(luck_pctl, 4) if luck_pctl is not None else None,
        "alternatives": alts,
        "best_alternative": best["label"] if best else None,
        "verdict": verdict,
    }


def _label_of(inv: dict, action_idx: int) -> Optional[str]:
    """Action label from the summary's per-invocation actions dict (insertion
    order == action-index order, the recorder's contract)."""
    labels = list((inv.get("actions") or {}).keys())
    return labels[action_idx] if 0 <= action_idx < len(labels) else None


# ---------------------------------------------------------------------------
# Battle-level falsification (anchor selection + loop)
# ---------------------------------------------------------------------------

def anchor_deltas(summary: dict, npz: dict, *, gamma: float) -> Dict[int, float]:
    """Map each turn's ``move_selection`` anchor inv → the **worst** TD-residual δ
    attributed to its turn (δ = r + γV(s′) − V(s) — the prober's formula). A
    forced-switch crater attributes to its turn's move decision (the re-rollable
    anchor). This δ both **ranks** anchors (:func:`select_anchors`) and, surfaced
    onto each falsified decision, **weights** it in the run-level
    ``falsify_scan`` — one source of truth so the ranking and the magnitude
    weighting can never disagree."""
    invs = summary.get("invocations", [])
    values = npz.get("values")
    if values is None or not len(invs):
        raise ValueError("trace has no recorded values — cannot rank anchors by δ")
    move_inv_by_turn: Dict[int, int] = {}
    for i, inv in enumerate(invs):
        if inv.get("phase") == "move_selection" and inv["turn"] not in move_inv_by_turn:
            move_inv_by_turn[inv["turn"]] = i

    scored: Dict[int, float] = {}   # anchor inv -> worst δ attributed to it
    for i, inv in enumerate(invs[:-1]):
        reward = (inv.get("outcome") or {}).get("reward")
        rtotal = reward.get("total") if isinstance(reward, dict) else reward
        if rtotal is None or i + 1 >= len(values):
            continue
        td = float(rtotal) + gamma * float(values[i + 1]) - float(values[i])
        anchor = move_inv_by_turn.get(inv["turn"])
        if anchor is None:
            continue
        if anchor not in scored or td < scored[anchor]:
            scored[anchor] = td
    return scored


def select_anchors(summary: dict, npz: dict, *, gamma: float, worst: int) -> List[int]:
    """The ``worst`` most-negative-δ ``move_selection`` decisions on distinct
    turns. A thin ranking over :func:`anchor_deltas`."""
    scored = anchor_deltas(summary, npz, gamma=gamma)
    return sorted(scored, key=scored.get)[:max(1, worst)]


def falsify_battle(
    record: ReconstructionRecord,
    summary: dict,
    npz: dict,
    *,
    invs: Optional[Sequence[int]] = None,
    worst: int = 3,
    gamma: float = 0.99,
    n_seeds: int = DEFAULT_SEEDS,
    n_alts: int = DEFAULT_ALTS,
    followup: str = "random",
    mappings=None,
) -> dict:
    """Falsify the chosen (or worst-δ) decisions of one battle."""
    anchors = list(invs) if invs else select_anchors(summary, npz, gamma=gamma, worst=worst)
    try:
        scored = anchor_deltas(summary, npz, gamma=gamma)   # {} weight when values are absent
    except ValueError:
        scored = {}
    decisions, errors = [], []
    for inv_index in anchors:
        try:
            d = falsify_decision(
                record, summary, npz, int(inv_index),
                n_seeds=n_seeds, n_alts=n_alts, followup=followup, mappings=mappings)
            # Surface the TD-residual δ that selected (and now weights) this anchor —
            # one source of truth for falsify_scan's crater-magnitude weighting.
            d["anchor_delta"] = round(scored[int(inv_index)], 4) if int(inv_index) in scored else None
            decisions.append(d)
        except (ValueError, RuntimeError, IndexError, subprocess.TimeoutExpired) as e:
            # A wedged Node re-roll (TimeoutExpired) becomes a recorded decision-error
            # like a node crash (RuntimeError) — it must not discard the battle's other
            # already-falsified decisions.
            errors.append({"inv": int(inv_index), "error": str(e)})
    counts: Dict[str, int] = {}
    for d in decisions:
        counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1
    return {
        "battle": record.battle_tag,
        "result": (summary.get("meta") or {}).get("result"),
        "trainee_side": record.side_of(record.trainee_username) if record.trainee_username else None,
        "anchors": [int(a) for a in anchors],
        "decisions": decisions,
        "verdict_counts": counts,
        "errors": errors,
        "thresholds": {"luck_pctl_max": _LUCK_PCTL_MAX, "adv_min": _ADV_MIN, "z": _Z},
    }
