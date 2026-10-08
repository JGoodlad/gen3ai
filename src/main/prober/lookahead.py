"""One-ply lookahead — how the critic values the state resulting from each candidate action.

Feature 1 of the prober's counterfactual analysis. For an anchored ``move_selection`` decision in a
recorded bridge battle it RE-ROLLS the turn under each LEGAL action (OUR side plays the candidate, the
opponent plays its RECORDED action), reads the resulting one-sided successor obs off the RUST CORE's
parse chain (the chain the training rows are encoded on — ``core_walk.read_streams``, P5), and reads the loaded model's **V(s′)** (plus the distributional / win-prob heads when the
run trained them). The result is per-action ΔV — "what would the critic have valued each alternative
at" — the value readout the model-free, material-margin :mod:`falsifier` deliberately defers.

Two faithful modes share one call:

- **CRN (the headline, the ``"original"`` seed)** — hold the realized dice stream and vary ONLY our
  action, so ΔV isolates the action's *effect*, not dice variance. The CHOSEN action's CRN successor
  reproduces the REAL next state, so its V(s′) ≈ the recorded next V — a built-in consistency anchor
  surfaced as ``recorded_next_value``.
- **Dice-averaged (``n_seeds`` > 0)** — mean V(s′) ± std over fresh common-random-number seeds, the
  robustness annotation when the modal line is dice-sensitive.

Like the falsifier this rides the reconstruction / re-roll path, so it needs the trace's
``*_reconstruction.json`` sibling (bridge-eval traces). V is read from the loaded model on the
materialized ONE-SIDED obs — never the omniscient re-roll outcome (the one-sided / omniscient wall).
A candidate whose turn ENDS the battle has no successor state; it is reported as a terminal
(``terminal_frac`` > 0 / ``terminal`` win-loss) rather than a numeric V.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional, Sequence

import numpy as np

from main.prober.core_walk import StreamRequest, decision_choices, read_streams
from main.prober.engine import _npz_value
from main.prober.falsifier import _label_of, fresh_seeds
from utils.bridge.reconstruction import ReconstructionRecord, reroll_many

DEFAULT_SEEDS = 0   # CRN-only by default (fast, deterministic); >0 dice-averages the value


def _stats(xs: Sequence[float]) -> "tuple[Optional[float], Optional[float]]":
    """Mean and population std of a value sample, or ``(None, None)`` when empty."""
    if not xs:
        return None, None
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std())


def lookahead_decision(
    model,
    record: ReconstructionRecord,
    summary: dict,
    npz: dict,
    inv_index: int,
    *,
    n_seeds: int = DEFAULT_SEEDS,
    candidates: Optional[Sequence[int]] = None,
    followup: str = "random",
    mappings=None,
    impl: str = "node",
) -> dict:
    """One-ply value-delta for one ``move_selection`` decision (see module header).

    ``candidates`` restricts the swept actions to a subset of legal action indices (the chosen action
    is always included); ``None`` sweeps every legal action. ``n_seeds`` > 0 adds the dice-averaged
    value alongside the CRN headline. ``impl`` (``"node"`` default | ``"rust"``) selects the offline
    replay/re-roll driver. Raises (like the falsifier) on a non-``move_selection`` anchor, a trace
    missing the ``actions`` array, or a replay desync.
    """
    invs = summary.get("invocations", [])
    if not (0 <= inv_index < len(invs)):
        raise IndexError(f"inv {inv_index} out of range (battle has {len(invs)})")
    inv = invs[inv_index]
    if inv.get("phase") != "move_selection":
        raise ValueError(
            f"inv {inv_index} is a {inv.get('phase')!r} decision — re-rolls anchor at start-of-turn "
            f"move rounds; look ahead from that turn's move_selection invocation"
        )
    turn = int(inv["turn"])
    if "actions" not in npz:
        raise ValueError("states.npz has no 'actions' array — trace predates the reconstruction "
                         "layer; only newer bridge-eval traces support lookahead")
    if not record.trainee_username:
        raise ValueError("reconstruction record lacks trainee_username")
    side = record.side_of(record.trainee_username)
    other_side = "p2" if side == "p1" else "p1"

    actions = np.asarray(npz["actions"], dtype=int)
    chosen_idx = int(actions[inv_index])

    # Every LEGAL action index → the sim choice string at the decision, off the RUST CORE's walk of the
    # record (P5: the core's choice tokens, the mapper training's rows use — the falsifier's map).
    choice_map, walked_turn = decision_choices(record, side, inv_index)
    if walked_turn != turn:
        raise RuntimeError(
            f"replay desync: decision {inv_index} replays at turn {walked_turn}, summary says {turn}")
    if chosen_idx not in choice_map:
        raise RuntimeError(f"chosen action {chosen_idx} not legal in replayed state")

    # Candidate set: every legal action by default (chosen always included), else the caller's subset.
    if candidates is None:
        cand = [c for c in choice_map]
    else:
        cand = [int(c) for c in candidates if int(c) in choice_map]
    if chosen_idx not in cand:
        cand.append(chosen_idx)

    seeds = fresh_seeds(n_seeds, salt=f"{record.battle_tag}:{inv_index}:la") if n_seeds > 0 else []
    seed_list = list(seeds) + ["original"]          # "original" = the CRN realized-dice line (headline)
    prefix_actions = [int(a) for a in actions[:inv_index]]
    recorded_v = _npz_value(npz, inv_index)
    recorded_next_v = _npz_value(npz, inv_index + 1) if inv_index + 1 < len(invs) else None

    username = record.username(side)
    packed = record.packed_team(side)

    # Resolve EVERY (candidate × seed) arm in ONE Node process (reroll_many) — the sweep pays the
    # ~677ms Node-spawn / pokemon-showdown require cost ONCE instead of once per candidate (~3.8× on the
    # re-roll step, measured). The CHOSEN action is sourced "recorded" so its "original" line reproduces
    # the realized turn EXACTLY (resolveTurnExact ⇒ value_crn == recorded_next_value even with a mid-turn
    # forced switch); alternatives play their explicit choice. Each arm is byte-identical (modulo the
    # state-invisible |t:| timestamp) to a single reroll_turn — pinned by reroll_many_parity_fuzz_test.
    arms = []
    for a in cand:
        our_action = "recorded" if a == chosen_idx else choice_map[a]
        for s in seed_list:
            arms.append({f"{side}_action": our_action, f"{other_side}_action": "recorded",
                         "seed": s, "label": int(a)})
    rr = reroll_many(record, turn, arms, followup=followup, impl=impl)
    prefix_chunks = rr.prefix_p1_chunks if side == "p1" else rr.prefix_p2_chunks
    by_arm: dict = {}                          # action index → {seed → ArmReroll}
    for arm in rr.arms:
        by_arm.setdefault(int(arm.label), {})[arm.seed] = arm

    # Each (candidate × seed) arm's SUCCESSOR row: the side's text (the recorded prefix + the arm's own
    # ply) read by the RUST CORE's parse chain with the trackers on — the chain the training rows are
    # encoded on — with our actions replayed by index, encoding only decision inv+1 (P5,
    # `core_walk.read_streams`; every arm in ONE core process). Byte-equal to the poke-env
    # materializer it replaces (`designs/research_state/measurements/pokeenv_p5_prober_2026-10-07/`).
    branch_keys: List[tuple] = []
    requests: List[StreamRequest] = []
    for a in cand:
        for s in seed_list:
            r = by_arm.get(int(a), {}).get(s)
            if r is None or r.outcome.get("stuck") or r.outcome.get("ended"):
                continue
            branch_keys.append((int(a), s))
            requests.append(StreamRequest(
                username=username, packed_team=packed, side=side,
                chunks=list(prefix_chunks) + list(r.p1_chunks if side == "p1" else r.p2_chunks),
                actions=prefix_actions + [int(a)], encode_at=[inv_index + 1]))
    succ_by_key = {}
    for key, decs in zip(branch_keys, read_streams(requests)):
        if len(decs) > inv_index + 1:               # else: no successor request in the window
            succ_by_key[key] = decs[inv_index + 1]

    rows = []
    for a in cand:
        by_seed = by_arm.get(int(a), {})
        vals: List[float] = []
        crn_v: Optional[float] = None
        crn_wp: Optional[float] = None
        terminal = 0
        terminal_winner: Optional[str] = None
        for s in seed_list:
            r = by_seed.get(s)
            if r is None or r.outcome.get("stuck"):
                continue
            if r.outcome.get("ended"):
                terminal += 1
                if s == "original":
                    w = r.outcome.get("winner")
                    terminal_winner = ("win" if w == username else "loss" if w else "tie")
                continue
            succ = succ_by_key.get((int(a), s))
            if succ is None:
                continue
            obs, mask = np.array(succ.obs, dtype=np.float32), np.asarray(succ.mask, dtype=np.int8)
            v = model.value(obs, mask)
            vals.append(v)
            if s == "original":
                crn_v = v
                crn_wp = getattr(model, "win_prob_at", lambda *_: None)(obs, mask)
        v_mean, v_std = _stats(vals)
        rows.append({
            "action": int(a), "label": _label_of(inv, a), "choice": choice_map[a],
            "is_chosen": a == chosen_idx,
            "value_crn": round(crn_v, 4) if crn_v is not None else None,
            "value_mean": round(v_mean, 4) if v_mean is not None else None,
            "value_std": round(v_std, 4) if v_std is not None else None,
            "n_evaluated": len(vals),
            "terminal_frac": round(terminal / max(1, len(seed_list)), 3),
            "terminal": terminal_winner,
            "win_prob_crn": round(crn_wp, 4) if crn_wp is not None else None,
        })

    # ΔV vs the chosen action's CRN successor value (the cleanest "how much better/worse" read). Falls
    # back to ΔV vs the recorded V(s) when the chosen line ended terminally (no successor V).
    chosen_row = next((r for r in rows if r["is_chosen"]), None)
    baseline = chosen_row["value_crn"] if chosen_row and chosen_row["value_crn"] is not None else recorded_v
    for r in rows:
        r["delta_v"] = (round(r["value_crn"] - baseline, 4)
                        if (r["value_crn"] is not None and baseline is not None) else None)

    # Rank by the resulting-state value (terminal wins float to the top, losses sink): a candidate the
    # critic values higher than the chosen line is the headline "a better move existed" signal.
    def _rank(r):
        if r["terminal"] == "win":
            return float("inf")
        if r["terminal"] == "loss":
            return float("-inf")
        return r["value_crn"] if r["value_crn"] is not None else float("-inf")

    rows.sort(key=_rank, reverse=True)
    best = next((r for r in rows if not r["is_chosen"]), None)

    return {
        "inv": inv_index,
        "turn": turn,
        "side": side,
        "chosen": {"action": chosen_idx, "label": _label_of(inv, chosen_idx),
                   "choice": choice_map[chosen_idx]},
        "recorded_value": round(recorded_v, 4) if recorded_v is not None else None,
        "recorded_next_value": round(recorded_next_v, 4) if recorded_next_v is not None else None,
        "baseline_value": round(baseline, 4) if baseline is not None else None,
        "n_seeds": n_seeds,
        "candidates": rows,
        "best_alternative": best["label"] if best else None,
        "best_delta_v": best["delta_v"] if best else None,
    }


def lookahead_battle(
    model,
    record: ReconstructionRecord,
    summary: dict,
    npz: dict,
    *,
    invs: Optional[Sequence[int]] = None,
    worst: int = 3,
    gamma: float = 0.99,
    n_seeds: int = DEFAULT_SEEDS,
    followup: str = "random",
    mappings=None,
    impl: str = "node",
) -> dict:
    """Look ahead from the chosen (or worst-δ) decisions of one battle — a thin loop over
    :func:`lookahead_decision` mirroring :func:`falsifier.falsify_battle`."""
    from main.prober.falsifier import select_anchors

    anchors = list(invs) if invs else select_anchors(summary, npz, gamma=gamma, worst=worst)
    decisions, errors = [], []
    for inv_index in anchors:
        try:
            decisions.append(lookahead_decision(
                model, record, summary, npz, int(inv_index),
                n_seeds=n_seeds, followup=followup, mappings=mappings, impl=impl))
        except (ValueError, RuntimeError, IndexError, subprocess.TimeoutExpired) as e:
            errors.append({"inv": int(inv_index), "error": str(e)})
    return {
        "battle": record.battle_tag,
        "result": (summary.get("meta") or {}).get("result"),
        "trainee_side": record.side_of(record.trainee_username) if record.trainee_username else None,
        "anchors": [int(a) for a in anchors],
        "decisions": decisions,
        "errors": errors,
    }
