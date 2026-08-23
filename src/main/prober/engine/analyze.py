"""The engine's top-level entry — `analyze_invocation`, plus the trace meta and value-dist reads."""

from __future__ import annotations

import numpy as np

from main.prober.engine.beliefs import (belief_view_from_logits, build_belief, build_belief_truth,
    build_exclusive_belief, build_opp_full_team, move_belief_view, revealed_opp_species)
from main.prober.engine.board import _merge_team, _our_items, build_board
from main.prober.engine.decode import (_faithfulness, _intervention_sweep, _matchups, _saliency,
    _threats, _value_saliency, decode_incoming_belief)
from main.prober.engine.flags import self_cure_options, summary_flags
from main.prober.engine.intent import build_opp_intent
from main.prober.engine.spread import build_spread_belief
from main.prober.engine.switch_in import build_switch_in_outgoing
from main.prober.engine.timeline import _timeline_for, opp_voluntary_switch
from main.prober.engine.util import _has_state, _npz_value, _npz_win_prob
from main.prober.engine.views import (InvocationAnalysis, TraceMeta, ValueDistView, ValueView,
    WinProbView)


def _dist_quantile(support, cdf, t: float) -> float:
    idx = int(np.searchsorted(cdf, t))
    return float(support[min(idx, len(support) - 1)])


def build_value_dist(npz, i: int, support, popart=None) -> "ValueDistView | None":
    """The distributional value head's per-decision return distribution at decision ``i``. None when the
    array is absent (old trace / the run had no value-dist head → the KeyError "unavailable" path), this
    row wasn't captured (all-NaN), or the support/trace bin counts disagree (config drift). ``support`` =
    ``(vmin, vmax, bins)`` from the loaded model; ``popart`` = optional ``(mu, sigma)`` to denormalize
    E[Z]. Pure (numpy only) — the single source the app histogram + the ``analyze`` CLI both render."""
    try:
        arr = npz["value_dist"]
    except KeyError:
        return None
    if not (0 <= i < len(arr)):
        return None
    probs = np.asarray(arr[i], dtype=np.float64)
    if probs.size == 0 or np.isnan(probs).any():
        return None
    vmin, vmax, bins = support
    if int(bins) != probs.size:
        return None
    z = np.linspace(float(vmin), float(vmax), int(bins))
    p = probs / max(float(probs.sum()), 1e-8)
    mean = float((p * z).sum())
    std = float(np.sqrt(max(float((p * (z - mean) ** 2).sum()), 0.0)))
    cdf = np.cumsum(p)
    peak = int(np.argmax(p))
    lo, hi = max(0, peak - 2), min(len(p), peak + 3)
    bimodality = float(max(0.0, 1.0 - float(p[lo:hi].sum())))
    mean_real = None
    if popart is not None and popart[1]:
        mean_real = mean * float(popart[1]) + float(popart[0])
    return ValueDistView(
        probs=tuple(float(x) for x in p), support=tuple(float(x) for x in z),
        mean=mean, std=std,
        p10=_dist_quantile(z, cdf, 0.10), p50=_dist_quantile(z, cdf, 0.50),
        p90=_dist_quantile(z, cdf, 0.90),
        entropy=float(-(p * np.log(p + 1e-12)).sum()), bimodality=bimodality, mean_real=mean_real,
    )


def build_meta(summary: dict, summary_path: str = "", npz_path: "str | None" = None) -> TraceMeta:
    m = summary.get("meta", {})
    return TraceMeta(
        step=int(m.get("step", 0)),
        battle_id=str(m.get("battle_id", "")),
        result=str(m.get("result", "")),
        turns=int(m.get("turns", 0)),
        n_invocations=int(m.get("invocations", len(summary.get("invocations", [])))),
        summary_path=summary_path,
        npz_path=npz_path,
    )


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def analyze_invocation(model, summary: dict, npz, inv_index: int,
                       summary_path: str = "", npz_path: "str | None" = None,
                       opp_team: "tuple[str, ...] | None" = None,
                       our_hp_types: "dict | None" = None,
                       opp_team_details: "list | None" = None,
                       our_team_details: "list | None" = None) -> InvocationAnalysis:
    """Analyze a single decision point. Pure given ``model`` (the torch boundary).

    ``opp_team`` is the opponent's PRIVILEGED full team (species ids from the trace's
    `reconstruction.json` sibling, loaded by the caller — kept out of this pure engine); when given
    AND the model exposes the belief, the result carries the slot-MATCHED `belief_truth`.
    ``opp_team_details`` is the richer per-mon `team_details()` list ({species, evs, ivs, nature, …}) from
    the same `reconstruction.json`; when given AND the model exposes the spread belief, the result carries
    the believed-vs-true `spread_belief`."""
    meta = build_meta(summary, summary_path, npz_path)
    inv = summary["invocations"][inv_index]
    chosen = inv["chosen"]
    common = dict(
        meta=meta, inv_index=inv_index, turn=int(inv["turn"]), phase=inv.get("phase", ""),
        our_species=inv["our"]["species"], opp_species=inv["opp"]["species"], chosen=chosen,
    )

    outcome = inv.get("outcome", {}) or {}
    # Team (item + moveset): our side (items only, exact, end-of-battle) from the summary; superseded
    # /extended per-turn by the obs decode below once a state is captured — which also surfaces the
    # OPP's revealed items + both sides' revealed movesets.
    team = _our_items(summary)
    board = build_board(inv, team, our_hp_types)   # model-free; available even without captured state
    # The NEXT decision's board is the RESOLVED "after" state — read it (model-free) so the UI can
    # show before→after HP. None on the last decision (no following invocation).
    invs = summary["invocations"]
    next_board = (build_board(invs[inv_index + 1], team, our_hp_types)
                  if inv_index + 1 < len(invs) else None)
    belief = build_belief(inv)       # model-free summary fallback (re-computed below when a model + state exist)
    exclusive_belief = None          # MODEL-ONLY: needs the full posterior, not the summary's top-3
    belief_truth = None
    # α/β — model-free from the trace, and it STAYS model-free (unlike `belief`, which prefers a
    # re-computed read): the intent heads are supervised against what the opponent then did, so the
    # honest question is what THIS decision's model expected, not what a later checkpoint would.
    opp_intent = build_opp_intent(inv)
    outcome = {**outcome, "timeline": _timeline_for(inv, next_board, outcome)}   # model-free RESULT lines
    opp_full_team = build_opp_full_team(opp_team_details, board)   # model-free; available without state
    # Forced-switch panel: what each ALIVE candidate would DO to the opp active (the op's outgoing is
    # all-zero here). Model-free / privileged — available even without captured state.
    switch_in_outgoing = None
    if common["phase"] == "forced_switch":
        try:
            switch_in_outgoing = build_switch_in_outgoing(board, our_team_details, opp_team_details)
        except Exception:  # noqa: BLE001 — never break the analysis
            switch_in_outgoing = None
    if not _has_state(npz, inv_index):
        return InvocationAnalysis(
            **common, has_state=False, actions=(), matchups=None, sweep=None,
            saliency=None, value_saliency=None, threats=None, incoming=None,
            warnings=(f"invocation {inv_index} has no captured state",), opp_full_team=opp_full_team,
            outcome=outcome, flags=summary_flags(inv), cure_options=self_cure_options(inv),
            board=board, next_board=next_board, belief=belief, opp_intent=opp_intent,
            switch_in_outgoing=switch_in_outgoing, opp_switched_to=opp_voluntary_switch(inv),
        )

    obs = npz["obs"][inv_index].astype(np.float32)
    # Obs-version guard: if the trace's obs is a different length than the CURRENT encoder, every
    # obs-OFFSET decode past the divergence point (incoming/threat/matchups/turn-history crit etc.)
    # is misaligned. Flag it so the UI warns instead of showing garbage. (Team blocks + global are
    # at the front and still decode; the model forward still runs on the trace's native obs.)
    enc_dim = getattr(model.offsets, "total_dim", 0)
    obs_mismatch = (int(obs.shape[0]), int(enc_dim)) if enc_dim and obs.shape[0] != enc_dim else None
    decode_team = getattr(model, "describe_team", None)
    if decode_team is not None:
        obs_team = decode_team(obs)
        if obs_team:
            board = build_board(inv, _merge_team(team, obs_team), our_hp_types)   # both sides, per-turn revealed
    # What actually happened on THIS turn (crit + couldn't-move reason): the realized events are
    # recorded in the NEXT decision's most-recent TurnDelta (turn T's events land in decision T+1's
    # obs). Adds our_crit/opp_crit + our_cant/opp_cant to the outcome so the RESULT/happened line can
    # tag '⚡crit' and 'couldn't move (asleep)'.
    decode_turn = getattr(model, "describe_turn_outcome", None)
    if decode_turn is not None:
        obs_all = npz["obs"]
        nxt = inv_index + 1
        if nxt < len(obs_all) and _has_state(npz, nxt):
            to = decode_turn(obs_all[nxt].astype(np.float32))
            if to:
                outcome = {**outcome, **to}
    # Rebuild the RESULT timeline now that crit / move_order / cant are merged in (the no-state path
    # above already built a model-free one without them).
    outcome = {**outcome, "timeline": _timeline_for(inv, next_board, outcome)}
    acts = inv["actions"]
    # The recorded `actions` dict is ALREADY in ACTION-INDEX order: `BattleRecorder._all_action_labels`
    # iterates action index 0..10 and keys move slot m (action 6+m) on `legal.move_ids[m]` — the SAME
    # request-slot order the action mask, the DamageOperator's per-move blocks, and the policy logits
    # (action 6+k) use. So `labels[i]` ↔ action index i ↔ `probs[i]` directly; every action-6+k-indexed
    # consumer below (matchups ×mult, op outgoing, the re-run argmax) is correctly paired with NO realign.
    # (A prior `_reorder_move_labels` step re-sorted these to the per-mon block's MOVESET order via
    # `our_active_move_slots`, which differs from request order after a server reorder — that SCRAMBLED the
    # already-correct labels and produced spurious `disagree` flags + transposed matchup/op labels. Removed.)
    labels = list(acts.keys())
    mask = np.array([1 if acts[k]["valid"] else 0 for k in labels], dtype=np.int8)

    # Hidden-opp belief: prefer the loaded model's RE-COMPUTED belief (works for ANY belief-on
    # checkpoint, incl. runs whose recorder predates the summary `belief` block) over the summary
    # fallback; with the privileged opponent team, also build the slot-MATCHED truth view.
    belief_fn = getattr(model, "belief", None)
    if belief_fn is not None:
        raw = belief_fn(obs, mask)
        if raw is not None:
            sp_logits, bmask = raw
            mb = belief_view_from_logits(sp_logits, bmask)
            if mb is not None:
                belief = mb
            # The species-clause READING, beside the raw marginals — never instead of them. It needs
            # the FULL posterior, which is why it hangs off the re-computed branch and not off the
            # summary fallback: the summary's `belief` block carries only the top-3 per slot, whose
            # rows do not sum to 1, so an operator run on them would answer a different question
            # while looking identical.
            exclusive_belief = build_exclusive_belief(
                sp_logits, bmask, revealed_opp_species(board))
            if opp_team:
                belief_truth = build_belief_truth(sp_logits, bmask, revealed_opp_species(board), opp_team)

    probs, _ = model.action_dist(obs, mask)
    actions = _faithfulness(probs, labels, acts, chosen, mask)
    matchups = _matchups(obs, labels, model.offsets, inv["our"].get("species", ""), our_hp_types)
    sweep = _intervention_sweep(model, obs, mask, labels, chosen, model.offsets)
    saliency = _saliency(model, obs, mask, labels.index(chosen), model.offsets)
    value_saliency = _value_saliency(model, obs, mask, model.offsets)
    threats = _threats(obs, model.offsets)
    incoming = decode_incoming_belief(obs, model.offsets)

    # Value: recorded V(s), the loaded model's re-run V(s), V at the next captured
    # decision, and ΔV. (A big ΔV is where the critic's expectation shifted.)
    recorded_v = _npz_value(npz, inv_index)
    value = None
    if recorded_v is not None:
        n = len(summary["invocations"])
        nxt = inv_index + 1
        next_v = _npz_value(npz, nxt) if (nxt < n and _has_state(npz, nxt)) else None
        rerun_v = model.value(obs, mask)
        # PopArt-normalized companions (the critic's learning scale), when the model exposes stats.
        mu = sigma = norm_rec = norm_rerun = None
        pa = getattr(model, "popart_stats", lambda: None)()
        if pa is not None and pa[1]:
            mu, sigma = pa
            norm_rec = (recorded_v - mu) / sigma
            norm_rerun = (rerun_v - mu) / sigma if rerun_v is not None else None
        value = ValueView(
            recorded=recorded_v, rerun=rerun_v, next_recorded=next_v,
            delta=(next_v - recorded_v) if next_v is not None else None,
            popart_mu=mu, popart_sigma=sigma,
            normalized_recorded=norm_rec, normalized_rerun=norm_rerun,
        )

    # Win probability (--win-prob-mode): recorded P(win|s) + ΔP(win) to the next captured decision —
    # the calibrated analog of the value/ΔV above. None on a run without the head (win_probs NaN/absent).
    recorded_wp = _npz_win_prob(npz, inv_index)
    win_prob = None
    if recorded_wp is not None:
        n = len(summary["invocations"])
        nxt = inv_index + 1
        next_wp = _npz_win_prob(npz, nxt) if (nxt < n and _has_state(npz, nxt)) else None
        win_prob = WinProbView(
            recorded=recorded_wp, next_recorded=next_wp,
            delta=(next_wp - recorded_wp) if next_wp is not None else None,
        )

    # Distributional value head (--value-dist-mode): the predicted RETURN DISTRIBUTION at this state —
    # the interpretability read the scalar V collapses (sharp=confident, wide=uncertain, bimodal=coinflip).
    # Model-free from the trace's per-atom `value_dist` array; the support (atoms) + PopArt denorm come
    # from the loaded model. None on a run without the head (array absent / NaN).
    value_dist = None
    _vds = getattr(model, "value_dist_support", lambda: None)()
    if _vds is not None:
        value_dist = build_value_dist(
            npz, inv_index, _vds, getattr(model, "popart_stats", lambda: None)())

    # Does the loaded model still pick what was recorded? (Exact tier ≈ always; on
    # nearest/recent a disagreement is the interesting case.)
    rerun_argmax = labels[int(np.argmax(probs))]
    agrees = rerun_argmax == chosen
    flags = summary_flags(inv) + (() if agrees else ("disagree",))

    # Field state (weather/spikes/screens) decoded from the obs global block, when
    # the model can decode it (a fake/stub model simply won't have describe_global).
    field = None
    describe = getattr(model, "describe_global", None)
    if describe is not None:
        try:
            field = describe(obs)
        except Exception:  # noqa: BLE001 — field decode is best-effort
            field = None

    # Unified DamageOperator view (per-mon incoming + outgoing per-move), re-computed from the loaded
    # model — None when the checkpoint has no damage op or the model can't expose it (a fake/stub model).
    damage_op = None
    dop = getattr(model, "damage_op_view", None)
    if dop is not None:
        try:
            damage_op = dop(obs, mask)
        except Exception:  # noqa: BLE001 — best-effort, never break the analysis
            damage_op = None

    # Move belief: what the model thinks the REVEALED opponent's still-UNSEEN moves are (+ the per-our-mon
    # labels for the op damage rows). Best-effort, like damage_op; None on a move-belief-off checkpoint.
    move_belief = None
    mbfn = getattr(model, "move_belief", None)
    if mbfn is not None:
        try:
            move_belief = move_belief_view(mbfn(obs, mask))
        except Exception:  # noqa: BLE001 — never break the analysis
            move_belief = None

    # Spread belief: the believed opp DERIVED stats (the op's stat input) vs the true derived stats from the
    # privileged team_details. Best-effort; None on a --spread-belief-off checkpoint.
    spread_belief = None
    sbfn = getattr(model, "spread_belief_view", None)
    if sbfn is not None:
        try:
            spread_belief = build_spread_belief(sbfn(obs, mask), opp_team_details)
        except Exception:  # noqa: BLE001 — never break the analysis
            spread_belief = None

    # Rebuilt from the FINAL board (with the obs-decoded revealed moves), so a move shows revealed the
    # moment it fires.
    opp_full_team = build_opp_full_team(opp_team_details, board)
    return InvocationAnalysis(
        **common, has_state=True, actions=actions, matchups=matchups, sweep=sweep,
        saliency=saliency, value_saliency=value_saliency, threats=threats, incoming=incoming,
        warnings=(), outcome=outcome, value=value, win_prob=win_prob, value_dist=value_dist,
        rerun_argmax=rerun_argmax, agrees=agrees, flags=flags, cure_options=self_cure_options(inv),
        board=board, next_board=next_board,
        obs_mismatch=obs_mismatch, field=field, belief=belief,
        exclusive_belief=exclusive_belief, belief_truth=belief_truth,
        opp_intent=opp_intent,
        damage_op=damage_op, move_belief=move_belief, spread_belief=spread_belief,
        opp_full_team=opp_full_team,
        switch_in_outgoing=switch_in_outgoing, opp_switched_to=opp_voluntary_switch(inv),
    )
