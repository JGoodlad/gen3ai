"""R-ROLLOUT Monte-Carlo win-probability TARGETS (`--win-prob-rollout-target`, arm 10).

**THE LEVER, in one line of arithmetic.** Under ``--critic winprob`` every state of an episode is
trained against the SAME terminal bit: one outcome bit copied to ~30 states. That bit carries at
most 1 bit of information about the whole GAME and nothing at all about the individual STATE — and
the head refit (``designs/research_state/measurements/winprob_head_refit_2026-09-09/`` §6) showed
the consequence is a TARGET defect: only 10.2 % / 14.4 % of the label's variance lies BETWEEN
(cycle, opponent) cells, so a learner minimising a proper scoring rule shrinks the weak axes toward
the marginal. ``--win-prob-lambda`` (arm 8) attacked that by moving the network's OWN later
estimates backward — a cheaper channel, but a self-referential one. This flag attacks it by BUYING
NEW BITS: play ``R`` continuations from the state itself and make ``wins / R`` the target. R
rollouts from a state give R bits about THAT state's own win probability; the terminal bit gives
one bit about the game.

**It replaces the buffer's OWN target on the sampled states.** That is the whole point and the
distinction from the counterfactual label factory: ``cf_winprob_term`` applies the head to FOREIGN
recorded states from ``<run>/cf_labels/`` with their own labels and never touches ``win_target`` —
and that arm read NULL. Here the labelled row is a row of THIS rollout buffer, inside the ordinary
PPO objective, at the ordinary ``win_prob_coef``.

WHAT IT COSTS, AND WHY THE FRACTION IS SMALL
--------------------------------------------
A continuation is a real battle played by a real policy: measured on `cf_producer`
(§ *Where the time goes*), ~104 live ``choose_move`` calls per continuation and 93 % of the wall in
those forwards. So one labelled state at ``R = 8`` costs ~832 policy decisions, against the
**98,304** the trainee itself makes in a production rollout (48 envs x 2,048 steps). The identity
that governs the flag is therefore

    labelling budget / collection budget  =  fraction x R x mean_continuation_decisions

so ``--win-prob-rollout-target 1/32`` at ``R = 8`` would cost **~26x** the run's entire simulation
budget, and the fraction that buys the labels for ``1x`` is ``1 / (R x mean_continuation) ~ 1/832``.
:func:`budget_multiple` computes this from the MEASURED per-rollout numbers rather than from the
banked constant, and it is published every rollout as ``win_prob/rollout_budget_multiple`` — a lever
whose cost is not on the dashboard is a lever that gets left on.

WHY THE LABELLING IS SYNCHRONOUS
--------------------------------
The label must land on the buffer row it describes, and the rollout buffer is a RING that is
refilled from scratch every iteration. A label that arrives one rollout LATE (the `cf_records`
pattern — a producer that ages labels onto disk and a consumer that folds them later) has no row
left to write into; it can only become an auxiliary loss on foreign states, which is the arm that
already read null. So the labelling runs BETWEEN ``_on_rollout_end`` and ``train()``, it BLOCKS,
and the stall is measured and published (``win_prob/rollout_seconds``) rather than hidden.

THE SELECTION (`winprob_rollout_select_v1`)
-------------------------------------------
UNIFORM over eligible rows, at most ONE per EPISODE SLICE, seeded from the run seed and the
rollout index. Uniform and not priority-ranked on purpose: a priority sampler (the `cf_producer`
shape) re-weights WHICH states carry the new target, which is a distribution-shift confound for
exactly the read the arm exists to make. At most one per episode is the bits-per-state argument
applied to the sample itself — R rollouts from two states of one game are far more correlated than
R rollouts from two different games.

Eligible means all four of:

* ``win_mask == 1`` — the episode TERMINATED inside this buffer. That is also what makes the state
  reconstructible: the ``__RECON__`` record is written at episode END, so a trailing in-progress
  episode has no record on disk yet.
* a reconstruction HANDLE was captured for the row (see :func:`record_key`).
* ``turn >= MIN_LABELABLE_TURN`` — `cf_producer`'s bound, inherited rather than re-derived.
* a MOVE ROUND, read off the buffer's own ``action_mask`` (`cf_producer_sampler.is_move_round`).
  A mid-turn forced switch cannot be the divergence point of a counterfactual replay, which cuts
  at a TURN boundary; labelling one would label the turn's move decision instead.

THE ECOLOGY APPROXIMATION — the arm's largest declared caveat
-------------------------------------------------------------
A continuation plays the recorded state forward with the **CURRENT policy on BOTH sides, sampling
at temperature 1.0**. The trainee side is then exactly the regime the training actor plays in (a
GREEDY rollout would bias every label LOW — measured +0.037 [+0.007, +0.066] over 477 sentinel
states when the prober got this wrong). The OPPONENT's TEAM is exact (it comes out of the record's
packed team), but its POLICY is not: a training ``__RECON__`` record carries no opponent identity
at all — not the bot name, not the pool snapshot's step — so there is nothing to reload. Every
label is therefore measured against a self-like opponent, which is right for the ~90 % self-play
share of the mixture and biased LOW on the rest. This module does not hide that: the buffer's own
``opp_class`` key is carried through the selection and the win rate is published PER CLASS, with
``win_prob/rollout_bot_share`` naming how much of the labelled mass is drawn from the biased
population.

Pure numpy and pure arithmetic — no torch, no subprocess, no bridge. The driver that actually plays
the continuations is :mod:`agents.training.win_prob_rollout_labeller`; the child process it spawns
is :mod:`agents.training.win_prob_rollout_worker`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.action.constants import MOVE_START
from agents.training.cf_producer_sampler import MIN_LABELABLE_TURN

#: ``--win-prob-rollout-target 0.0`` is OFF, and OFF is bit-identical by SKIPPING the whole path
#: rather than by computing an identity — the same discipline `--win-prob-lambda 1.0` keeps.
ROLLOUT_OFF = 0.0

#: Legal ``--win-prob-rollout-mode`` values. ``replace``: the labelled state's target BECOMES the
#: rollout win fraction. ``blend``: the mean of the rollout fraction and the episode's terminal bit
#: — half the target shift, and it keeps some of the RECORDED ecology (the terminal bit was played
#: against the real opponent; the rollout fraction was not — see *THE ECOLOGY APPROXIMATION*).
ROLLOUT_MODES = ("replace", "blend")

#: Default ``--win-prob-rollout-r``. 8 is `cf_producer`'s own default and the point where a tight-MC
#: win fraction has a standard error of ~0.18 at p = 0.5 — still noisy per state, but the noise is
#: INDEPENDENT across states, where the terminal bit's noise is shared by every state of the game.
DEFAULT_ROLLOUT_R = 8

#: HARD CAP on states labelled per rollout, whatever the fraction asks for. The cost of this lever
#: is linear in the state count and it is paid as a STALL on the training loop, so an operator who
#: fat-fingers a fraction gets a bounded bill rather than a wedged run. 256 states at R = 8 is
#: ~2,048 continuations — already ~17x a production rollout's own simulation budget, so the cap is
#: a guard rail and never an operating point.
MAX_STATES_PER_ROLLOUT = 256

#: The declared selection rule. Bump it if the rule changes: a silent change here is a
#: distribution-shift confound for every read taken across it.
SELECTOR_VERSION = "winprob_rollout_select_v1"

#: The banked mean number of LIVE policy decisions in one continuation (both sides), from
#: `cf_producer`'s profile (832 `choose_move` calls over R = 8 arms, 2026-08-23). Used ONLY as the
#: fallback when a rollout measured none of its own; :func:`budget_multiple` prefers the measurement.
BANKED_CONTINUATION_DECISIONS = 104.0

#: The four `opp_class` codes the env publishes, in code order. Mirrors
#: `MaskableAgentWrapper.OPP_CLASS_*`; used only to NAME the per-class metrics.
OPP_CLASS_NAMES = ("bot", "pool", "stable", "exploiter")

#: ``--win-prob-rollout-weight 1.0`` is OFF, and OFF is bit-identical by not building the weight
#: vector at all — the loss then takes its unweighted expression UNCHANGED, the same discipline
#: `--win-prob-strata-weight 0` and `--win-prob-lambda 1.0` keep.
ROLLOUT_WEIGHT_OFF = 1.0

#: The obs Dict key the per-row BCE weight rides (`gen3_winprob_rollout_weight_v1`). A LABEL key
#: like `win_target` / `win_mask`: the env emits a placeholder every step and the
#: `WinProbLabelCallback` overwrites it post-collection, which is the only carrier that survives
#: `RolloutBuffer.get()`'s shuffle aligned to its own row. Declared ONLY when the flag is above
#: 1.0, so an unflagged run's obs space is untouched.
ROLLOUT_WEIGHT_KEY = "win_row_w"


def anchor_row_weights(anchor_mask, win_mask, weight: float):
    """``[n_steps, n_envs]`` float32 per-row BCE weight for `gen3_winprob_rollout_weight_v1`.

    **WHY THE FLAG EXISTS — the mass arithmetic.** At the fraction that costs 1x the run's own
    simulation budget (``1 / (R x ~104)``, i.e. ~0.0012 at R = 8) the rollout-labelled rows are
    ~0.12 % of the win-prob BCE's rows. A treatment carrying 0.12 % of the objective cannot move
    the head BY ARITHMETIC, whatever the labels say — so the read of `--win-prob-rollout-target`
    at a feasible fraction is a read of nothing. The weight buys MASS at FIXED simulation cost:
    it is the one lever on this loss that changes the anchored rows' share of the objective
    without changing how many continuations get played.

    **THE NORMALISATION IS THE SAME CONVENTION `_win_prob_strata_weights` USES**: the returned
    weight has mean exactly 1 over the buffer's SCORED rows, so the loss SCALE does not move with
    the fraction and the value gradient is not rescaled by a lever that is supposed to re-price a
    mix. With ``a`` anchored rows among ``N`` scored ones (``f = a / N``) and a flag value ``k``::

        Z        = 1 + f * (k - 1)          # the raw mean of {k on anchors, 1 elsewhere}
        w_anchor = k / Z ,  w_other = 1 / Z
        anchored share of the weighted mass = f * k / Z

    At ``f = 0.0012`` and ``k = 64`` that is ``0.0768 / 1.0756 = 7.1 %`` — from 0.12 %.

    🚨 **ONLY THE ANCHOR ROWS ARE WEIGHTED, never the rows that bootstrap toward them.** Under
    ``--win-prob-lambda < 1`` an anchor TERMINATES the recursion and the earlier rows of its
    episode blend toward it at ``λ^k`` (`win_prob_callback.lambda_return_targets`). Those rows are
    a DIFFERENT quantity: their target is a λ-mixture of the anchor, the network's own later
    values and — past the next anchor or terminal — the copied outcome bit. Up-weighting them
    would up-weight the network's own estimates in the same stroke, which is arm 8's lever and not
    this one, and it would make the delivered dose a function of the episode-length distribution.
    The λ-propagated influence is MEASURED instead, and published as
    ``win_prob/rollout_influence_lambda``.

    ``anchor_mask`` is the ``[n_steps, n_envs]`` bool `apply_rollout_labels` returned; ``win_mask``
    is the buffer's FINAL ``win_mask[:, :, 0]`` (after λ's own re-masking, so the normaliser is
    computed over exactly the rows the BCE will score). Returns ``None`` — and the caller then
    writes nothing and the loss stays bit-identical — when the flag is at 1.0, when no row was
    anchored, or when nothing is scored.
    """
    k = float(weight)
    if k <= ROLLOUT_WEIGHT_OFF:
        return None
    anchors = np.asarray(anchor_mask, dtype=bool) if anchor_mask is not None else None
    if anchors is None or not anchors.any():
        return None
    scored = np.asarray(win_mask, dtype=np.float64) >= 0.5
    if scored.shape != anchors.shape:
        raise ValueError("anchor_mask and win_mask must have the same [n_steps, n_envs] shape")
    n_scored = float(scored.sum())
    if n_scored <= 0.0:
        return None
    raw = np.where(anchors & scored, k, 1.0).astype(np.float64)
    # `z` is exactly the mean of `raw` over the SCORED rows, so mean(w) == 1 there by construction.
    # An anchor that λ somehow left unscored contributes to neither side of the ratio.
    z = float((raw * scored).sum() / n_scored)
    if not np.isfinite(z) or z <= 0.0:
        return None
    return (raw / z).astype(np.float32)


def weighted_mass(anchor_fraction: float, weight: float) -> float:
    """The anchored rows' share of the WEIGHTED objective: ``f * k / (1 + f * (k - 1))``.

    The closed form of what :func:`anchor_row_weights` produces, so a design doc, a test and a
    pre-launch cost calculation can all quote the same number without a buffer. ``f`` is
    ``rollout_mass`` (anchored rows / scored rows) and ``k`` the flag.
    """
    f = float(anchor_fraction)
    k = float(weight)
    z = 1.0 + f * (k - 1.0)
    return float("nan") if z <= 0.0 else float(f * k / z)


def record_key(pid: int, battle_tag: Optional[str]) -> str:
    """The handle that joins a BUFFER ROW to a reconstruction record on disk.

    `cf_records.CfRecordRing` names every file ``<ns:019d>_<pid>_<tag>_reconstruction.json``, so
    ``<pid>_<tag>`` identifies the episode uniquely: the tag counter is per BridgeSession (it
    repeats across env workers) and the pid disambiguates the workers. Captured in the env worker,
    where both halves are known, and matched by SUFFIX in :func:`index_records`.
    """
    from agents.training.cf_records import safe_tag
    return f"{int(pid)}_{safe_tag(battle_tag)}"


def index_records(records_dir) -> Dict[str, str]:
    """``{record_key: path}`` over a `cf_records` ring directory, NEWEST wins.

    One ``readdir`` per rollout, not one ``stat`` per handle: the ring holds ``--cf-records-keep``
    files (512 by default) and the filenames sort chronologically by construction, so the newest
    record for a key is simply the last one seen in sorted order.
    """
    import os
    from utils.bridge.reconstruction import RECON_SUFFIX
    out: Dict[str, str] = {}
    try:
        names = sorted(os.listdir(str(records_dir)))
    except OSError:
        return out
    for name in names:
        if not name.endswith(RECON_SUFFIX):
            continue
        stem = name[: -len(RECON_SUFFIX)]
        # `<ns>_<pid>_<tag>` — drop the 19-digit timestamp, keep `<pid>_<tag>`.
        cut = stem.find("_")
        if cut < 0:
            continue
        out[stem[cut + 1:]] = os.path.join(str(records_dir), name)
    return out


def is_move_round_mask(mask: Sequence[float]) -> bool:
    """A start-of-turn MOVE round, read off the buffer's own ``action_mask``.

    Identical predicate to `cf_producer_sampler.is_move_round`, spelled here over a buffer row
    rather than a materialized decision's mask so the selection needs no replay to filter.
    """
    return bool(np.asarray(mask)[MOVE_START:].sum() > 0)


def slice_ids(episode_starts: np.ndarray) -> np.ndarray:
    """``[n_steps, n_envs]`` int array: which EPISODE SLICE of its env each row belongs to.

    A slice is a contiguous run of rows between episode boundaries within this buffer. Two rows of
    the same env with the same id came from the same game — which is exactly the correlation the
    one-per-slice rule exists to avoid sampling twice.
    """
    es = np.asarray(episode_starts, dtype=np.float64)
    return np.cumsum(es >= 0.5, axis=0).astype(np.int64)


def eligible_mask(win_mask: np.ndarray, turns: np.ndarray, has_handle: np.ndarray,
                  action_mask: np.ndarray) -> np.ndarray:
    """``[n_steps, n_envs]`` bool: the rows this arm may label. See *THE SELECTION* above.

    ``action_mask`` is ``[n_steps, n_envs, n_actions]`` straight off the rollout buffer's obs dict;
    ``turns`` is ``-1`` where no handle was captured.
    """
    wm = np.asarray(win_mask, dtype=np.float64) >= 0.5
    tn = np.asarray(turns, dtype=np.int64) >= int(MIN_LABELABLE_TURN)
    hh = np.asarray(has_handle, dtype=bool)
    am = np.asarray(action_mask)
    move = am[..., MOVE_START:].sum(axis=-1) > 0
    return wm & tn & hh & move


def select_rows(eligible: np.ndarray, episode_starts: np.ndarray, n_pick: int,
                rng: np.random.Generator) -> List[Tuple[int, int]]:
    """``winprob_rollout_select_v1`` — up to ``n_pick`` ``(t, env)`` rows, <= 1 per episode slice.

    Uniform over the eligible SLICES, then uniform within the chosen slice. Deterministic given
    ``rng``: the caller seeds it from the run seed and the rollout index, so a rerun of the same
    argv on the same collected buffer picks the same states.
    """
    n_pick = int(n_pick)
    if n_pick <= 0:
        return []
    el = np.asarray(eligible, dtype=bool)
    if not el.any():
        return []
    sids = slice_ids(episode_starts)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    ts, es = np.nonzero(el)
    for t, e in zip(ts.tolist(), es.tolist()):
        buckets.setdefault((int(e), int(sids[t, e])), []).append(int(t))
    # Sorted so the ORDER the rng consumes is a property of the buffer, never of dict insertion.
    keys = sorted(buckets)
    take = min(n_pick, len(keys))
    chosen = rng.choice(len(keys), size=take, replace=False)
    picks: List[Tuple[int, int]] = []
    for k in sorted(int(i) for i in np.atleast_1d(chosen)):
        env_i, _sid = keys[k]
        rows = buckets[keys[k]]
        picks.append((int(rows[int(rng.integers(len(rows)))]), int(env_i)))
    return sorted(picks)


def n_states_for(fraction: float, n_steps: int, n_envs: int, cap: int = MAX_STATES_PER_ROLLOUT
                 ) -> int:
    """How many states the fraction asks for, capped. The fraction is of the WHOLE buffer.

    Of the whole buffer and not of the eligible subset on purpose: the flag's value is the number a
    cost calculation uses, and a denominator that moves with the episode mix would make the same
    flag value cost different amounts on different days.
    """
    if float(fraction) <= ROLLOUT_OFF:
        return 0
    want = int(np.floor(float(fraction) * int(n_steps) * int(n_envs)))
    return int(max(0, min(want, int(cap))))


def apply_rollout_labels(wt: np.ndarray, wm: np.ndarray, picks: Sequence[Tuple[int, int]],
                         labels: Sequence[Optional[float]], mode: str, terminal_y: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float]]]:
    """Write the rollout targets into the buffer's ``win_target`` / ``win_mask``, IN PLACE.

    ``labels[i]`` is ``wins / R`` for ``picks[i]``, or ``None`` for a state whose rollouts failed
    (a vanished record, a wedged bridge child) — a failed state keeps its terminal bit and is never
    given a fabricated one. Returns ``(anchor_mask, anchor_value, applied)``; the first two are the
    ``[n_steps, n_envs]`` arrays the λ recursion consumes as ANCHORS (see *precedence* in
    `win_prob_callback`), the third is ``(t, env, label)`` for the metrics.
    """
    if mode not in ROLLOUT_MODES:
        raise ValueError(f"mode must be one of {ROLLOUT_MODES}, got {mode!r}")
    shape = wt[:, :, 0].shape
    anchor_mask = np.zeros(shape, dtype=bool)
    anchor_value = np.zeros(shape, dtype=np.float64)
    applied: List[Tuple[int, int, float]] = []
    y = np.asarray(terminal_y, dtype=np.float64)
    for (t, e), lab in zip(picks, labels):
        if lab is None or not np.isfinite(float(lab)):
            continue
        val = float(lab) if mode == "replace" else 0.5 * (float(lab) + float(y[t, e]))
        wt[t, e, 0] = np.asarray(val, dtype=wt.dtype)
        wm[t, e, 0] = np.asarray(1.0, dtype=wm.dtype)
        anchor_mask[t, e] = True
        anchor_value[t, e] = val
        applied.append((int(t), int(e), float(lab)))
    return anchor_mask, anchor_value, applied


def budget_multiple(n_states: int, rollouts: int, arm_decisions: float,
                    n_steps: int, n_envs: int) -> float:
    """Labelling decisions / the trainee's own collection decisions, for THIS rollout.

    The number the flag lives or dies by (see *WHAT IT COSTS*). ``arm_decisions`` is the mean number
    of live policy decisions one continuation took — MEASURED by the workers when they report it,
    :data:`BANKED_CONTINUATION_DECISIONS` when they do not. The denominator is the trainee's own
    ``n_steps x n_envs`` decisions; the sim also runs an opponent, so reading this against the FULL
    simulation budget halves it, and it is deliberately the conservative half.
    """
    denom = float(int(n_steps) * int(n_envs))
    if denom <= 0.0:
        return float("nan")
    return float(int(n_states) * int(rollouts) * float(arm_decisions)) / denom


def fraction_for_unit_budget(rollouts: int, arm_decisions: float) -> float:
    """The ``--win-prob-rollout-target`` that costs exactly 1x the collection budget.

    ``1 / (R x mean_continuation_decisions)`` — independent of the buffer shape, because both sides
    of the ratio scale with it. This is the number that sets an arm's REGISTERED fraction.
    """
    d = float(rollouts) * float(arm_decisions)
    return float("nan") if d <= 0.0 else 1.0 / d


def weight_metrics(*, anchor_mask, new_mask, row_w, weight: float, anchor_share=None
                   ) -> Dict[str, float]:
    """The `win_prob/rollout_weight` + `rollout_mass_weighted` + `rollout_influence_lambda` meters.

    These three are the DELIVERED DOSE, and the read quotes them rather than the flag — the flag
    is what was asked for, these are what the objective actually carried.

    * ``rollout_weight`` — the flag value, so an absent-vs-1.0 reading is never a guess.
    * ``rollout_mass_weighted`` — ``(sum of the row weight over the ANCHOR rows) / (sum of the row
      weight over every row the BCE scores)``. An unscored row is in neither sum because it
      contributes nothing to the loss at ANY weight. Equals :func:`weighted_mass` of
      ``rollout_mass``; both are computed so a drift between them is a bug, not a convention.
    * ``rollout_influence_lambda`` — the same ratio over the anchor rows PLUS every row whose
      λ-target received an anchor contribution, each such row counted by its **λ^k SHARE** of that
      target rather than as a whole row (the share is exact and free: the recursion already walks
      the buffer backward, so `lambda_return_targets` fills it in the same pass). It is the honest
      TOTAL treatment mass under `--win-prob-lambda < 1`, where an anchor's information reaches
      the ~``1/(1-λ)`` rows before it. With λ off (``anchor_share=None``) there is no propagation
      and it equals ``rollout_mass_weighted`` — published anyway, because an absent tag must mean
      "the weight is off" and nothing else.
    """
    out: Dict[str, float] = {"rollout_weight": float(weight)}
    if row_w is None or anchor_mask is None:
        return out
    w = np.asarray(row_w, dtype=np.float64)
    scored = np.asarray(new_mask, dtype=np.float64) >= 0.5
    anchors = np.asarray(anchor_mask, dtype=bool)
    denom = float((w * scored).sum())
    if denom <= 0.0:
        return out
    out["rollout_mass_weighted"] = float((w * (anchors & scored)).sum() / denom)
    share = (np.asarray(anchor_share, dtype=np.float64) if anchor_share is not None
             else np.zeros(w.shape, dtype=np.float64))
    # An anchor row is its own full share; every other row carries the λ^k it inherited.
    share = np.where(anchors, 1.0, np.clip(share, 0.0, 1.0))
    out["rollout_influence_lambda"] = float((w * scored * share).sum() / denom)
    return out


def rollout_metrics(*, applied, picks, labels, eligible, terminal_y, new_mask, opp_class,
                    fraction: float, rollouts: int, mode: str, seconds: float,
                    arms_played: int, arms_capped: int, arm_decisions: float,
                    records_missing: int, n_steps: int, n_envs: int) -> Dict[str, float]:
    """The ``win_prob/rollout_*`` TB family: what the labelling DID and what it COST, per rollout.

    Every entry is a fact about THIS rollout. ``rollout_shift`` is the dose meter — the mean
    ``|rollout label - terminal bit|`` on the labelled states, the exact analogue of
    ``lambda_target_shift``: 0 means the new target agreed with the bit it replaced and the arm is
    buying nothing.
    """
    y = np.asarray(terminal_y, dtype=np.float64)
    nm = np.asarray(new_mask, dtype=np.float64) >= 0.5
    n_scored = float(nm.sum())
    labs = np.asarray([a[2] for a in applied], dtype=np.float64)
    bits = np.asarray([y[a[0], a[1]] for a in applied], dtype=np.float64)
    cls = np.asarray([int(opp_class[a[0], a[1]]) for a in applied], dtype=np.int64) \
        if len(applied) else np.zeros(0, dtype=np.int64)
    out: Dict[str, float] = {
        "rollout_target": float(fraction),
        "rollout_r": float(int(rollouts)),
        "rollout_mode_blend": 1.0 if mode == "blend" else 0.0,
        "rollout_eligible": float(np.asarray(eligible, dtype=bool).sum()),
        "rollout_requested": float(len(picks)),
        "rollout_states": float(len(applied)),
        "rollout_failed": float(len(picks) - len(applied)),
        "rollout_records_missing": float(int(records_missing)),
        "rollout_seconds": float(seconds),
        "rollout_seconds_per_state": (float(seconds) / len(applied)) if applied else float("nan"),
        "rollout_arms": float(int(arms_played)),
        "rollout_capped_frac": (float(arms_capped) / float(arms_played)) if arms_played else 0.0,
        "rollout_arm_decisions": float(arm_decisions),
        # What share of the objective's SCORED rows now carries a rollout-derived target. It is the
        # honest headline: a lever applied to 0.1% of the mass cannot move a loss by much, and this
        # is the number that says so before a read tries to.
        "rollout_mass": (float(len(applied)) / n_scored) if n_scored > 0 else float("nan"),
        "rollout_budget_multiple": budget_multiple(len(applied), rollouts, arm_decisions,
                                                   n_steps, n_envs),
        "rollout_unit_budget_fraction": fraction_for_unit_budget(rollouts, arm_decisions),
    }
    if len(labs):
        out["rollout_win_rate"] = float(labs.mean())
        out["rollout_shift"] = float(np.abs(labs - bits).mean())
        out["rollout_label_std"] = float(labs.std())
        for code, name in enumerate(OPP_CLASS_NAMES):
            sel = cls == code
            if sel.any():
                out[f"rollout_win_rate_{name}"] = float(labs[sel].mean())
        # The ECOLOGY caveat, priced: the share of labelled states whose real opponent was a BOT,
        # i.e. the share of the new labels measured against a substituted (stronger) opponent and
        # therefore biased LOW.
        out["rollout_bot_share"] = float((cls == 0).mean())
    return out
