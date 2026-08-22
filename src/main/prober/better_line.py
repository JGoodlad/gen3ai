"""Search for a BETTER LINE than the model actually played — a shallow, CRN-anchored beam over the
critic that returns ONE human-legible contrastive trajectory ("at turn T, switch to X instead — here
is the line, here is the per-ply ΔV / ΔP(win), here is where the recorded play went wrong").

This is the depth-≥2 generalization of :mod:`main.prober.lookahead` (which is exactly its depth-1
instance). Where lookahead RE-ROLLS each candidate ONE ply from the recorded turn, ``better_line``
branches a search TREE: it opens the decision point as a search root in the warm clone-and-branch
search-server (:class:`utils.bridge.search_session.SearchSession`, ~1.7 ms/clone, constant in depth —
the only primitive that makes depth>1 feasible), expands our top-k actions by policy prior, scores
each successor's V(s′) through the critic on the materialized ONE-SIDED obs, keeps the top-``beam``,
and recurses. Backup is max-over-our-continuations; the returned LINE is the principal variation.

THE FAITHFUL-CONDITIONAL OPPONENT (the user-chosen model):
- **divergence ply (the turn under review)** — the opponent plays its RECORDED move (it committed not
  knowing our change). The CHOSEN action is reproduced EXACTLY (``recorded_exact`` → the realized next
  state), giving the built-in ``value_crn == recorded_next_value`` faithfulness anchor.
- **interior plies (past the divergence)** — the RELOADED opponent (``opp_model``) reacts **GREEDILY**
  (argmax) on ITS OWN one-sided materialized obs (the search privilege: we hold the referee record, so
  we drive the real hidden team). With no ``opp_model`` the interior opponent falls back to the sim's
  default legal move (flagged in ``opp_model_used``) — depth-1 is faithful regardless; only depth≥2
  leans on it.

⚠️ **The interior regime is GREEDY BY DECISION, and it is a MIX with the divergence ply's recorded
one.** That is deliberate and it is declared here because the mix is otherwise invisible: the
divergence ply reproduces what the opponent actually did (stochastic, if the sentinel played
stochastic — the regime `prober.replay.build_opponent` now honours), while every ply past it plays
the opponent's single most likely reply. The reason is what the beam is FOR: a "better line" is a
claim about a line that survives the opponent's BEST answer, so sampling the opponent would report a
line that beats one draw of a die and call it better — an optimistic search bias, in the one direction
this tree already pays for (the optimizer's curse). Greedy is the standard worst-case-opponent search
assumption, and it makes the returned ΔV a lower bound rather than a lucky draw.

The two consequences a reader must hold: (1) the line is scored against a **deterministic** opponent,
so a line whose refutation is a low-probability reply will not be found; (2) the beam's numbers are
**not** comparable ply-for-ply with `replay-counterfactual`'s Monte-Carlo win rate, which plays the
recorded stochastic regime to the end. Every payload carries
``interior_opponent_regime: "greedy"`` so no artifact can hide which of the two produced it.

DICE: CRN throughout (``seed="original"`` — the realized dice at the divergence turn, the natural
continuation deeper), so ΔV isolates the ACTION effect, not dice variance. THE WALL: every obs the
policy/critic scores is materialized from ONLY the per-side protocol chunks the search-server emits;
the omniscient clone/outcome/teams/dice drive only the opponent and the dice, never the obs encoder.
Requires the trace's ``*_reconstruction.json`` sibling (bridge-eval traces), like lookahead/falsify.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from agents.training.obs_materializer import infer_action_indices, materialize_decisions
from main.prober.engine import _npz_value
from main.prober.falsifier import _label_of
from utils.bridge.reconstruction import ReconstructionRecord, replay_battle
from utils.bridge.search_session import SearchSession

_WIN, _LOSS = 1e6, -1e6      # terminal backup sentinels (win floats up, loss sinks)


@dataclass
class _Node:
    """One search node: a board state reached by a sequence of OUR actions (the opponent responding
    per the faithful-conditional model), plus its composed one-sided chunks + action histories."""

    label: str
    depth: int                       # number of OUR plies from the root (root = 0)
    parent: Optional["_Node"]
    sid: Optional[str]               # search-server node id (None once terminal)
    action: Optional[int]            # OUR action taken from parent to reach here
    our_suffixes: List[list]         # per-ply our-side suffix chunks along the path
    opp_suffixes: List[list]
    our_actions: List[int]           # full our action-index history (prefix + path)
    opp_actions: List[int]
    ended: bool = False
    terminal: Optional[str] = None   # "win" / "loss" / "tie"
    value: Optional[float] = None    # critic V(s) on OUR materialized obs at this node
    win_prob: Optional[float] = None
    value_dist: Optional[list] = None
    backup: float = 0.0
    children: List["_Node"] = field(default_factory=list)


def _terminal_label(outcome: dict, username: str) -> str:
    w = outcome.get("winner")
    return "win" if w == username else ("tie" if not w else "loss")


def better_line_decision(
    model,
    record: ReconstructionRecord,
    summary: dict,
    npz: dict,
    inv_index: int,
    *,
    depth: int = 2,
    beam: int = 3,
    top_k: int = 4,
    followup: str = "random",
    opp_model=None,
    mappings=None,
    timeout: float = 300.0,
    session=None,
    impl: str = "node",
) -> dict:
    """Search for a better line from one ``move_selection`` decision (module header). Returns the
    recommended contrastive trajectory + per-ply deltas. ``depth`` = how many OUR plies to look ahead
    (1 == lookahead), ``beam`` = nodes kept per ply, ``top_k`` = our candidate actions expanded per
    node, ``opp_model`` = the reloaded opponent for interior plies (None → sim-default, flagged).

    ``impl`` (``"node"`` default | ``"rust"``) selects the offline replay/search driver, the same
    way ``--use-bridge={node,rust}`` selects the live transport. It is IGNORED when ``session`` is
    injected — that warm ``SearchSession`` already carries its own impl, and silently re-spawning
    it on a different one would defeat the reuse the injection exists for."""
    invs = summary.get("invocations", [])
    if not (0 <= inv_index < len(invs)):
        raise IndexError(f"inv {inv_index} out of range (battle has {len(invs)})")
    inv = invs[inv_index]
    if inv.get("phase") != "move_selection":
        raise ValueError(
            f"inv {inv_index} is a {inv.get('phase')!r} decision — better-line anchors at a "
            "start-of-turn move round; pick that turn's move_selection invocation")
    if "actions" not in npz:
        raise ValueError("states.npz has no 'actions' array — trace predates the reconstruction layer")
    if not record.trainee_username:
        raise ValueError("reconstruction record lacks trainee_username")
    if depth < 1:
        raise ValueError("depth must be >= 1")

    turn = int(inv["turn"])
    side = record.side_of(record.trainee_username)
    other = "p2" if side == "p1" else "p1"
    username = record.username(side)
    actions = np.asarray(npz["actions"], dtype=int)
    chosen_idx = int(actions[inv_index])
    our_prefix_actions = [int(a) for a in actions[:inv_index]]
    recorded_v = _npz_value(npz, inv_index)
    recorded_next_v = _npz_value(npz, inv_index + 1) if inv_index + 1 < len(invs) else None

    # Replay the recorded battle ONCE (Node spawn) and reuse BOTH sides' chunks below — the anchor
    # choice-map (our side) and the opponent's action history (opp side) — instead of replaying twice.
    rep = replay_battle(record, impl=impl)
    our_full = rep.p1_chunks if side == "p1" else rep.p2_chunks
    opp_full = rep.p1_chunks if other == "p1" else rep.p2_chunks

    # Map our legal actions → sim choices at the anchor (the real mapper). encode_only_at=set() ⇒ no obs
    # encode anywhere (we need only the choice-map + the decision's turn — the anchor obs is in npz).
    trace = materialize_decisions(
        our_full, username=username, packed_team=record.packed_team(side), side=side,
        actions=our_prefix_actions + [chosen_idx], battle_format=record.format_id,
        battle_tag=record.battle_tag, mappings=mappings,
        map_actions_at=inv_index, stop_after_decision=inv_index, encode_only_at=set())
    if not trace.decisions or trace.decisions[-1].turn != turn:
        raise RuntimeError(f"replay desync materializing the anchor (inv {inv_index}, turn {turn})")
    choice_map: dict = trace.action_choices or {}
    if chosen_idx not in choice_map:
        raise RuntimeError(f"chosen action {chosen_idx} not legal in replayed state")

    use_opp = opp_model is not None and depth >= 2
    opp_model_used = "reloaded" if use_opp else ("none(sim-default)" if depth >= 2 else "recorded@divergence")

    # A caller (the search-teacher worker) can inject a WARM SearchSession reused across battles — then
    # we don't close it (nullcontext); else we own a one-shot session and close it on exit.
    ctx = (nullcontext(session) if session is not None
           else SearchSession(record, timeout=timeout, impl=impl))
    with ctx as ss:
        root = ss.open_root(turn, record=record)
        our_prefix = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
        opp_prefix = root.prefix_p1_chunks if other == "p1" else root.prefix_p2_chunks
        opp_recorded = root.recorded_choices.get(other)

        # Opp action-index history up to the divergence turn (for materializing the opp's interior
        # obs). The opp's decision count in the prefix = its pre-turn-T decisions + the (unactioned)
        # turn-T request, so drop the last. Only needed for the reloaded interior opponent.
        opp_pre: List[int] = []
        opp_recorded_idx: Optional[int] = None
        if use_opp:
            opp_all = infer_action_indices(record, other, mappings=mappings, chunks=opp_full)
            opp_prefix_dec = materialize_decisions(
                opp_prefix, username=record.username(other), packed_team=record.packed_team(other),
                side=other, actions=opp_all, battle_format=record.format_id,
                battle_tag=record.battle_tag, mappings=mappings, encode_only_at=set())
            opp_pre = opp_all[:max(0, len(opp_prefix_dec.decisions) - 1)]
            # the opp's divergence-ply action index (it played its recorded move there)
            if len(opp_all) > len(opp_pre):
                opp_recorded_idx = opp_all[len(opp_pre)]

        # ---- helpers bound to this search -------------------------------------------------
        def our_view(n: _Node):
            """Materialize OUR successor obs + mask + legal-action choice map at node ``n``."""
            chunks = list(our_prefix)
            for s in n.our_suffixes:
                chunks += s
            dec_i = inv_index + n.depth
            mt = materialize_decisions(
                chunks, username=username, packed_team=record.packed_team(side), side=side,
                actions=n.our_actions, battle_format=record.format_id, battle_tag=record.battle_tag,
                mappings=mappings, map_actions_at=dec_i, stop_after_decision=dec_i,
                encode_only_at={dec_i})        # encode ONLY the successor; the prefix is track-only
            if len(mt.decisions) <= dec_i:
                return None, None, None
            d = mt.decisions[dec_i]
            return d.obs, d.mask, (mt.action_choices or {})

        def opp_view(n: _Node):
            """Materialize the reloaded opponent's interior obs + mask + choice-map at node ``n`` (its
            greedy choice is picked from a BATCHED forward by the caller). (None, None, None) when the
            opp obs doesn't materialize → the caller falls back to the sim default (flagged)."""
            chunks = list(opp_prefix)
            for s in n.opp_suffixes:
                chunks += s
            dec_i = len(opp_pre) + n.depth        # opp acts once per turn, same cadence as us
            mt = materialize_decisions(
                chunks, username=record.username(other), packed_team=record.packed_team(other),
                side=other, actions=opp_pre + n.opp_actions[len(opp_pre):],
                battle_format=record.format_id, battle_tag=record.battle_tag, mappings=mappings,
                map_actions_at=dec_i, stop_after_decision=dec_i,
                encode_only_at={dec_i})        # encode ONLY the opp's interior decision
            if len(mt.decisions) <= dec_i:
                return None, None, None
            d = mt.decisions[dec_i]
            return d.obs, d.mask, (mt.action_choices or {})

        # ---- ply 1: the divergence (opponent plays its RECORDED move) ---------------------
        cand = list(choice_map)
        arms = [{"node_id": root.node_id, "recorded_exact": True, "seed": "original",
                 "followup": followup, "label": chosen_idx}]
        for a in cand:
            if a == chosen_idx:
                continue
            arms.append({"node_id": root.node_id, f"{side}_action": choice_map[a],
                         f"{other}_action": opp_recorded or "default", "seed": "original",
                         "followup": followup, "label": a})
        expanded = ss.expand_many(arms)

        depth1: List[_Node] = []
        for e in expanded:
            a = int(e.label)
            # The opponent played its RECORDED move at the divergence ply for EVERY candidate (it
            # committed not knowing our change), so the opp action-history extends by opp_recorded_idx
            # for ALL depth-1 nodes — not just the chosen one — else an alternative's depth-2 opp obs
            # desyncs by a ply.
            opp_a = opp_recorded_idx if use_opp else None
            n = _Node(
                label=str(a), depth=1, parent=None, sid=e.node_id, action=a,
                our_suffixes=[e.p1_chunks if side == "p1" else e.p2_chunks],
                opp_suffixes=[e.p1_chunks if other == "p1" else e.p2_chunks],
                our_actions=our_prefix_actions + [a],
                opp_actions=(opp_pre + [opp_a]) if (use_opp and opp_a is not None) else list(opp_pre),
                ended=e.ended, terminal=_terminal_label(e.outcome, username) if e.ended else None)
            depth1.append(n)

        _score_frontier(model, depth1, our_view)
        # The root (divergence) ply expands EVERY candidate so all are fairly ranked at full depth;
        # the beam only limits OUR branching at interior plies (below).
        frontier = [n for n in depth1 if not n.ended and n.value is not None]

        # ---- plies 2..depth ----------------------------------------------------------------
        opp_fallbacks = 0   # interior plies where the reloaded opponent couldn't be resolved → sim default
        for d in range(2, depth + 1):
            # Pass 1: materialize every frontier parent's OUR + (reloaded) OPP view (prefix track-only,
            # one target encode each). Collect for ONE batched policy forward per side (vs a forward per
            # node) — the policy forwards are batched; the obs builds are the cost lever-3 already cut.
            pend = []   # (parent, our_obs, our_mask, our_cmap, opp_obs, opp_mask, opp_cmap)
            for parent in frontier:
                if parent.sid is None:
                    continue
                o, m, cm = our_view(parent)
                if o is None:
                    continue
                oo, om, ocm = opp_view(parent) if use_opp else (None, None, None)
                pend.append((parent, o, m, cm, oo, om, ocm))
            if not pend:
                break
            our_probs = model.action_probs_batch(
                np.stack([p[1] for p in pend]), np.stack([p[2] for p in pend]))
            opp_str_per = [None] * len(pend)
            opp_idx_per = [None] * len(pend)
            if use_opp:
                live = [(i, p) for i, p in enumerate(pend) if p[4] is not None]
                if live:
                    op = opp_model.action_probs_batch(
                        np.stack([p[4] for (_, p) in live]), np.stack([p[5] for (_, p) in live]))
                    for k, (i, p) in enumerate(live):
                        # GREEDY, deliberately — the declared interior regime (module docstring).
                        # NOT an oversight of `prober.replay.build_opponent`'s regime seam, which
                        # governs the DIVERGENCE ply (where the opponent's real, possibly
                        # stochastic, committed move is what happened). A beam that sampled here
                        # would return lines that beat one draw of a die: the search's job is the
                        # opponent's BEST reply, so argmax makes the reported ΔV a lower bound
                        # instead of a lucky one. Stamped into the payload as
                        # `interior_opponent_regime` so no artifact can hide it.
                        idx = int(np.argmax(op[k]))
                        choice = (p[6] or {}).get(idx)
                        if choice is not None:    # argmax landed on a legal/mapped action
                            opp_str_per[i] = choice
                            opp_idx_per[i] = idx

            # Pass 2: rank our top-k per parent (from the batched priors) + build the expand arms.
            arms = []
            arm_meta = []   # (parent, our_action_idx, opp_idx)
            for i, (parent, o, m, cm, oo, om, ocm) in enumerate(pend):
                opp_str, opp_idx = opp_str_per[i], opp_idx_per[i]
                if use_opp and opp_str is None:
                    opp_fallbacks += 1          # the reloaded opp fell back to the sim default here
                ranked = sorted(list(cm), key=lambda a: our_probs[i][a], reverse=True)[:top_k]
                for a in ranked:
                    arms.append({"node_id": parent.sid, f"{side}_action": cm[a],
                                 f"{other}_action": opp_str or "default", "seed": "original",
                                 "followup": followup, "label": f"{parent.label}>{a}"})
                    arm_meta.append((parent, a, opp_idx))
            if not arms:
                break
            results = ss.expand_many(arms)
            kids: List[_Node] = []
            for e, (parent, a, opp_idx) in zip(results, arm_meta):
                child = _Node(
                    label=str(e.label), depth=d, parent=parent, sid=e.node_id, action=a,
                    our_suffixes=parent.our_suffixes + [e.p1_chunks if side == "p1" else e.p2_chunks],
                    opp_suffixes=parent.opp_suffixes + [e.p1_chunks if other == "p1" else e.p2_chunks],
                    our_actions=parent.our_actions + [a],
                    opp_actions=(parent.opp_actions + [opp_idx]) if (use_opp and opp_idx is not None)
                    else list(parent.opp_actions),
                    ended=e.ended, terminal=_terminal_label(e.outcome, username) if e.ended else None)
                parent.children.append(child)
                kids.append(child)
            _score_frontier(model, kids, our_view)
            frontier = _keep_beam(kids, beam)

    # Honesty: if the reloaded interior opponent couldn't be resolved at some plies (its obs failed to
    # materialize, or its greedy action was illegal), those plies played the sim DEFAULT — say so, don't
    # silently claim a full "reloaded" line.
    if use_opp and opp_fallbacks:
        opp_model_used += f" (+{opp_fallbacks} interior plies → sim default)"

    # ---- backup (max over our continuations) + principal variation ------------------------
    for n in depth1:
        _backup(n)
    chosen_node = next((n for n in depth1 if n.action == chosen_idx), None)
    baseline = (chosen_node.value if chosen_node and chosen_node.value is not None else recorded_v)

    ranked = sorted(depth1, key=lambda n: n.backup, reverse=True)
    best = next((n for n in ranked if n.action != chosen_idx), None)

    rows = []
    for n in sorted(depth1, key=lambda n: n.backup, reverse=True):
        rows.append({
            "action": n.action, "label": _label_of(inv, n.action), "choice": choice_map[n.action],
            "is_chosen": n.action == chosen_idx,
            "value": round(n.value, 4) if n.value is not None else None,
            "backup": (None if abs(n.backup) >= _WIN else round(n.backup, 4)),
            "terminal": n.terminal,
            "delta_v": (round(n.value - baseline, 4) if (n.value is not None and baseline is not None)
                        else None),
            "win_prob": round(n.win_prob, 4) if n.win_prob is not None else None,
            "principal_variation": _pv_labels(n, inv, invs),
        })

    pv = _pv_nodes(best) if best is not None else []
    return {
        "inv": inv_index, "turn": turn, "side": side, "depth": depth, "beam": beam, "top_k": top_k,
        "opp_model_used": opp_model_used,
        # The DECLARED interior-ply opponent regime (module docstring). A constant today, and a
        # field rather than a docstring line precisely because it is one: the divergence ply plays
        # the RECORDED regime while every ply past it plays argmax, and a payload that did not say
        # so left the mix for a reader to rediscover. If this is ever made configurable, the value
        # follows the configuration; nothing downstream may assume "greedy" from its absence.
        "interior_opponent_regime": "greedy",
        "chosen": {"action": chosen_idx, "label": _label_of(inv, chosen_idx),
                   "choice": choice_map[chosen_idx]},
        "recorded_value": round(recorded_v, 4) if recorded_v is not None else None,
        "recorded_next_value": round(recorded_next_v, 4) if recorded_next_v is not None else None,
        "baseline_value": round(baseline, 4) if baseline is not None else None,
        "candidates": rows,
        "best_alternative": {
            "action": best.action, "label": _label_of(inv, best.action), "choice": choice_map[best.action],
            "backup": (None if abs(best.backup) >= _WIN else round(best.backup, 4)),
            "terminal": best.terminal,
            "delta_v": (round(best.value - baseline, 4)
                        if (best.value is not None and baseline is not None) else None),
            "win_prob": round(best.win_prob, 4) if best.win_prob is not None else None,
            "principal_variation": [_pv_step(x, invs) for x in pv],
        } if best is not None else None,
    }


def _score_frontier(model, nodes: Sequence[_Node], our_view) -> None:
    """Batch-score every non-terminal node's V(s) (one critic forward) + the win-prob head."""
    pend = [(n, our_view(n)) for n in nodes if not n.ended]
    live = [(n, o, m) for (n, (o, m, _)) in pend if o is not None]
    if not live:
        return
    obs = np.stack([o for (_, o, _) in live])
    masks = np.stack([m for (_, _, m) in live])
    vals = model.values_batch(obs, masks)
    for (n, o, m), v in zip(live, vals):
        n.value = float(v)
        wp = getattr(model, "win_prob_at", lambda *_: None)(o, m)
        n.win_prob = float(wp) if wp is not None else None
        vd = getattr(model, "value_dist_at", lambda *_: None)(o, m)
        n.value_dist = vd.tolist() if vd is not None else None


def _keep_beam(nodes: List[_Node], beam: int) -> List[_Node]:
    """Top-``beam`` non-terminal nodes by value (to extend); terminal nodes are leaves."""
    live = [n for n in nodes if not n.ended and n.value is not None]
    live.sort(key=lambda n: n.value, reverse=True)
    return live[:beam]


def _backup(n: _Node) -> float:
    """Max over our continuations; terminal/leaf is its own value."""
    if n.terminal == "win":
        n.backup = _WIN
    elif n.terminal == "loss":
        n.backup = _LOSS
    elif n.terminal == "tie":
        n.backup = 0.0
    elif n.children:
        n.backup = max(_backup(c) for c in n.children)
    else:
        n.backup = n.value if n.value is not None else _LOSS
    return n.backup


def _pv_nodes(n: _Node) -> List[_Node]:
    """The principal-variation node chain from ``n`` down (greedy argmax-backup)."""
    chain = [n]
    while n.children:
        n = max(n.children, key=lambda c: c.backup)
        chain.append(n)
    return chain


def _pv_step(n: _Node, invs) -> dict:
    return {"depth": n.depth, "action": n.action, "value": round(n.value, 4) if n.value is not None else None,
            "terminal": n.terminal}


def _pv_labels(n: _Node, inv, invs) -> List[dict]:
    return [_pv_step(x, invs) for x in _pv_nodes(n)]
