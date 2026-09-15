"""THE FORK ARM (`--fork-fraction`, `gen3_fork_v1`) — contested-state EXPLORING STARTS.

**WHY IT EXISTS, in one measurement.** `designs/research_state/measurements/`
`paired_refit_discrimination_2026-09-14/` measured, for the first time, the property a one-ply
search consumes: does ``sign(V(s'_a) - V(s'_b))`` agree with which of two successors ONE MOVE APART
actually wins, on the SAME dice? The promoted win-prob critic scores **0.5169 [0.4800, 0.5524]** —
a coin. With the trunk FROZEN and only the head's four tensors moving, a plain BCE on
COUNTERFACTUAL SUCCESSOR STATES reaches **0.6032 [0.5690, 0.6374]** (+0.0863, DETECTED), and a
pairwise RANKING term on the same rows buys **nothing** (-0.0107, NOT DETECTED, negative on
points). Pairwise accuracy is a RANK statistic, invariant to monotone recalibration, so the
ordering was already in ``value_pooled`` and the on-policy PPO stream never asked the head for it.

**The conclusion that lands here: the DATA is the lever, not the loss form.** The refit's data —
sibling successor states, with outcomes, under shared dice — does not exist in a training rollout,
because a rollout visits exactly ONE successor per decision. This arm manufactures it: at a
CONTESTED decision the battle is FORKED, the branches are played to a terminal under the current
policy, and their transitions enter the SAME PPO buffer. No new loss, no ranking term (CLOSED as a
lever at this depth), no auxiliary objective — the branch rows carry the ordinary GAE/λ-return of
their own branch and the ordinary win-prob BCE against their own outcome.

THE THREE BRANCHES, and why the random one is the point
-------------------------------------------------------
Three facts about the policy fell out of that dataset before any head was fitted, on 5,076 forks:

* top-1 and top-2 are **outcome-interchangeable** on identical dice — 0.7082 vs 0.7078, a gap of
  0.0004. So a two-branch fork spends its whole simulation budget on a distinction the outcome
  cannot see.
* a uniformly random legal alternative wins 0.6795 — throwing the decision away costs **2.9 pp**.
* in **4.5 % [3.99, 5.16]** of forks that random alternative beat BOTH policy candidates.

That 4.5 % is the BLIND-SPOT RATE and it is where the new information is: it is the only branch
whose outcome the policy's own ordering did not predict. ``--fork-branches 2`` (top-2 only) exists
as the control that isolates it — it is expected to buy little, and `fork/random_wins` is the meter
that says so.

⚠️ **79.7 % of branch pairs are TIED** (same outcome) — the structural tax on any terminal-outcome
sibling signal. `fork/tie_rate` publishes it every rollout; a run whose tie rate climbs is a run
whose forks are buying less, whatever the fork count says.

COMMON RANDOM NUMBERS — dice AND draws (`--fork-crn`)
----------------------------------------------------
🚨 **A concrete account of the `cf_q_labels` null, and the reason this flag has two settings.**
`cf_q_labels` pairs the sim DICE and leaves both sides sampling at temperature 1.0 — so two arms
of the same sweep differ in the dice they did not draw AND in every policy coin either side
flipped afterwards. A label factory that pairs the dice but not the draws may be teaching the head
noise. The forks in the measurement above were greedy on both sides, which is the only reason 104
determinism re-runs came back identical.

``dice`` — one sim seed for the whole line (the record's own resolved START seed, ``post_t_seed``
None), exactly `replay_counterfactual`'s pairing. That is the `cf_q_labels` regime, kept as the
comparator.

``dice_and_draws`` (the DEFAULT) — that, PLUS both players' policy sampling streams seeded
IDENTICALLY per branch, so branch A's k-th decision and branch B's k-th decision consume the SAME
uniform (`agents.training.fork_crn`). The branches then differ in exactly one thing: the action at
the fork. It is VERIFIABLE rather than asserted — `fork_crn_sim_test` replays two branches with
IDENTICAL actions through the real bridge and asserts byte-identical protocol.

Sampling stays at temperature 1.0 on both sides. A greedy continuation would be a different
ecology from the one the training actor plays in, and the prober measured that error at
+0.037 [+0.007, +0.066] when it was made.

WHAT COSTS WHAT
---------------
A fork plays ``branches`` continuations from the fork turn to a terminal, so

    fork sim steps  ~=  forks x branches x mean_remaining_decisions

against the ``n_steps x n_envs`` decisions the trainee itself makes. :func:`sim_steps_share`
computes it from the MEASURED remaining-turn counts and it is published every rollout as
``fork/sim_steps_share`` — a lever whose cost is not on the dashboard is a lever that gets left on.
At ``F = 0.02`` with 3 branches the expectation is ~1.5-2x a plain run's simulation.

THE ECOLOGY APPROXIMATION — this arm's largest declared caveat
--------------------------------------------------------------
A training ``__RECON__`` record carries the resolved seed, both packed teams and the committed
choices, and **nothing that says which policy sat on the other side**. So a branch is played
against a SELF-LIKE opponent (the current snapshot), not the episode's real one — right for the
~90 % self-play share of the training mixture and biased for the rest, in the same direction
`win_prob_rollout` documents. ``fork/branch_share`` prices how much of the buffer that population
substitution now occupies, and ``fork/bot_share`` names how much of it replaced a BOT.

Pure numpy and pure arithmetic — no torch, no subprocess, no bridge. The child that plays the
branches is :mod:`agents.training.fork_worker`, the parent-side fan-out
:mod:`agents.training.fork_driver`, the buffer surgery :mod:`agents.training.fork_buffer` and the
loop hook :mod:`agents.training.fork_callback`.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.action.constants import MOVE_START
from agents.training.cf_producer_sampler import MIN_LABELABLE_TURN

#: ``--fork-fraction 0.0`` is OFF, and OFF is bit-identical by SKIPPING the whole path rather than
#: by computing an identity — the discipline `--win-prob-rollout-target 0.0` and
#: `--win-prob-lambda 1.0` keep. At 0.0 no module below is imported, no obs key is declared, no
#: callback is attached and no row is injected.
FORK_OFF = 0.0

#: Legal ``--fork-branches``. 3 (the default) = the policy's top-2 candidates + ONE uniformly
#: random legal action; 2 = the top-2 alone, the control that isolates the random branch's value.
BRANCH_CHOICES = (2, 3)
DEFAULT_BRANCHES = 3

#: Branch NAMES in a fixed order — the order every row, meter and test reads them in.
BRANCH_NAMES = ("top1", "top2", "rand")

#: ``--fork-contested-gap``: the QUANTILE, over this rollout's own candidate pool, of the policy's
#: top-2 masked-logit gap below which a decision counts as CONTESTED. 0.40 is
#: `paired_refit_discrimination_2026-09-14/forks.py`'s ``--gap-quantile`` default, taken verbatim
#: so the arm forks the same population the baseline 0.5169 was measured on.
#:
#: 🚨 A QUANTILE and not an absolute logit gap, because the gap's SCALE moves with the policy's
#: temperature over a run: a fixed threshold would fork 40 % of decisions early and ~0 % once the
#: logits sharpen, i.e. the treatment would silently anneal itself off.
DEFAULT_CONTESTED_GAP = 0.40

#: ``--fork-contested-absv``: the |V - 0.5| band that ALSO admits a decision. 0.0 = OFF, and OFF is
#: the default **on purpose**.
#:
#: 🚨 THE PAIRED-REFIT SELECTOR NEVER READS V, and its own words are the reason: *"|V-0.5| is
#: RECORDED per decision but never selects — selecting on V would make the held-out read partly a
#: measurement of the selector."* The arm's registered endpoint is held-out pairwise accuracy
#: against that baseline, so selecting on V here would make the endpoint incomparable to the
#: number it is being read against. The knob exists because the ledger's spec names it; it is OFF,
#: and turning it on forfeits the comparison.
DEFAULT_CONTESTED_ABSV = 0.0

#: ``--fork-max-per-battle``: at most this many forks per EPISODE SLICE. 1 by default, and it is
#: the bits-per-state argument applied to the sample: two forks of one game share a prefix, an
#: opponent and a team draw, so they are far more correlated than two forks of different games.
#: `forks.py` used 2 over a 957-battle tree; a production rollout has ~2,400 episodes, so 1 is not
#: a binding constraint on the count.
DEFAULT_MAX_PER_BATTLE = 1

#: Legal ``--fork-crn``. See *COMMON RANDOM NUMBERS* above.
CRN_MODES = ("dice", "dice_and_draws")
DEFAULT_CRN = "dice_and_draws"

#: HARD CAP on forks per rollout, whatever the fraction asks for. Cost is linear in this and it is
#: paid as a STALL on the training loop, so a fat-fingered fraction gets a bounded bill rather than
#: a wedged run.
#:
#: The arithmetic it is set from: a production rollout is 2,048 x 48 = **98,304** decisions and a
#: branch is ~35 live decisions (measured in the `--debug` smoke), so
#: ``share = forks x 3 x 35 / 98,304 = forks / 936``. 4,096 is therefore ~4.4x the run's own
#: simulation — a GUARD RAIL and never an operating point.
#:
#: 🚨 **It is not what binds in practice, and neither is the fraction.** The ROW BUDGET
#: (`fork_callback.ROW_BUDGET_MULTIPLE`) caps the injection at one buffer's worth of rows, i.e.
#: ~98,304 / ~125 rows-per-fork ~= **790 forks** at the production shape — below the 1,966 that
#: ``--fork-fraction 0.02`` asks for. So above ~0.008 the fraction is INERT and the delivered fork
#: count is the budget's. Both numbers are published (`fork/requested` vs `fork/forks`,
#: `fork/row_budget`, `fork/rows_per_fork`) precisely so that ceiling is visible rather than
#: inferred from a flat `fork/rate`.
MAX_FORKS_PER_ROLLOUT = 4096

#: A fork later than this is near-terminal and, on a stalling game, already inside a loop.
#: `forks.py`'s bound, and its reason: the smoke put two forks at turns 178/183 and five of their
#: six branches hit the 250-turn cap.
MAX_FORKABLE_TURN = 40

#: The minimum number of LEGAL actions a contested decision must offer. 3 is what makes a random
#: branch distinct from the top-2 at all (`forks.py --min-legal 3`).
MIN_LEGAL_ACTIONS = 3

#: The declared selection rule. Bump it if the rule changes: a silent change here is a
#: distribution-shift confound for every read taken across it.
SELECTOR_VERSION = "gen3_fork_select_v1"

#: The obs Dict key carrying the per-row POLICY-TERM mask (`gen3_fork_v1`). A LABEL key on the
#: same plumbing as `win_target` / `win_row_w`: the env emits a placeholder every step and the
#: injected rows carry their own value, which is the only carrier that survives
#: `RolloutBuffer.get()`'s shuffle aligned to its own row. Declared ONLY when the flag is on, so an
#: unflagged run's observation space is untouched.
PG_MASK_KEY = "fork_pg_m"

#: The placeholder the env writes into :data:`PG_MASK_KEY`. 1.0, not 0.0: it is a MULTIPLIER on the
#: policy term, and a zero reaching the loss would delete the entire policy gradient and read as a
#: dead run rather than as a plumbing break.
PG_MASK_PLACEHOLDER = 1.0

#: How many eligible candidates are scored for every fork the fraction asks for, before the
#: quantile is taken. The pool has to be big enough that its ``--fork-contested-gap`` quantile is
#: the population's; scoring EVERY eligible row would mean a forward over most of a 98,304-row
#: buffer, which is a cost the arm does not need to pay to place its threshold.
SELECTION_POOL_FLOOR = 32.0


def _sha(*parts) -> int:
    """The decision-keyed hash `forks.py` uses, verbatim — reproducible and independent of the
    policy's own ordering (a probability-ordered draw would spend the random branch on what the
    policy already covers)."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def is_move_round_mask(mask: Sequence[float]) -> bool:
    """A start-of-turn MOVE round, read off the buffer's own ``action_mask``.

    The same predicate `win_prob_rollout.is_move_round_mask` and `cf_producer_sampler.is_move_round`
    spell, and for the same reason: a mid-turn FORCED SWITCH cannot be the divergence point of a
    counterfactual replay, which cuts at a TURN boundary.
    """
    return bool(np.asarray(mask)[MOVE_START:].sum() > 0)


def eligible_mask(win_mask, turns, action_mask) -> np.ndarray:
    """``[n_steps, n_envs]`` bool: the rows this arm MAY fork, before the contested test.

    Four conditions, every one of them inherited rather than re-derived:

    * ``win_mask == 1`` — the episode TERMINATED inside this buffer. That is also what makes the
      state replayable: the ``__RECON__`` record is written at episode END, so a trailing
      in-progress episode has no record on disk yet (`win_prob_rollout.eligible_mask`).
    * a reconstruction HANDLE was captured for the row (``turns >= 0``).
    * ``MIN_LABELABLE_TURN <= turn <= MAX_FORKABLE_TURN`` — `cf_producer`'s lower bound and
      `forks.py`'s upper one.
    * a MOVE ROUND with at least :data:`MIN_LEGAL_ACTIONS` legal actions.
    """
    wm = np.asarray(win_mask, dtype=np.float64) >= 0.5
    tn = np.asarray(turns, dtype=np.int64)
    am = np.asarray(action_mask)
    in_turn_band = (tn >= int(MIN_LABELABLE_TURN)) & (tn <= int(MAX_FORKABLE_TURN))
    move = am[..., MOVE_START:].sum(axis=-1) > 0
    enough = am.sum(axis=-1) >= float(MIN_LEGAL_ACTIONS)
    return wm & (tn >= 0) & in_turn_band & move & enough


def slice_ids(episode_starts) -> np.ndarray:
    """``[n_steps, n_envs]`` int: which EPISODE SLICE of its env each row belongs to.

    Identical to `win_prob_rollout.slice_ids`; two rows of the same env with the same id came from
    the same game, which is exactly the correlation ``--fork-max-per-battle`` bounds.
    """
    es = np.asarray(episode_starts, dtype=np.float64)
    return np.cumsum(es >= 0.5, axis=0).astype(np.int64)


def n_forks_for(fraction: float, n_steps: int, n_envs: int,
                cap: int = MAX_FORKS_PER_ROLLOUT) -> int:
    """How many forks the fraction asks for, capped.

    Of the WHOLE buffer and not of the eligible subset, for `win_prob_rollout.n_states_for`'s
    reason: the flag's value is the number a cost calculation uses, and a denominator that moved
    with the episode mix would make the same flag value cost different amounts on different days.
    """
    if float(fraction) <= FORK_OFF:
        return 0
    want = int(np.floor(float(fraction) * int(n_steps) * int(n_envs)))
    return int(max(0, min(want, int(cap))))


def candidate_pool(eligible, episode_starts, n_pool: int, max_per_battle: int,
                   rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Up to ``n_pool`` ``(t, env)`` rows drawn UNIFORMLY over eligible episode SLICES.

    The rows whose policy top-2 gap the caller will score in order to place this rollout's
    contested threshold. Uniform over SLICES first and then within a slice, so a long game does not
    dominate the pool — the same shape `win_prob_rollout.select_rows` uses, generalised from "one
    per slice" to ``--fork-max-per-battle`` per slice.

    Deterministic given ``rng``: the caller seeds it from the run seed and ``num_timesteps``, so a
    rerun of the same argv on the same collected buffer forks the same decisions.
    """
    n_pool = int(n_pool)
    if n_pool <= 0:
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
    order = rng.permutation(len(keys))
    per = max(1, int(max_per_battle))
    picks: List[Tuple[int, int]] = []
    for k in order.tolist():
        env_i, _sid = keys[int(k)]
        rows = buckets[keys[int(k)]]
        take = min(per, len(rows), n_pool - len(picks))
        if take <= 0:
            break
        chosen = rng.choice(len(rows), size=take, replace=False)
        for j in np.atleast_1d(chosen).tolist():
            picks.append((int(rows[int(j)]), int(env_i)))
        if len(picks) >= n_pool:
            break
    return sorted(picks)


def pool_size_for(n_forks: int, quantile: float) -> int:
    """How many candidates to SCORE so the ``quantile`` threshold yields ~``n_forks`` forks.

    ``n_forks / quantile``, floored at :data:`SELECTION_POOL_FLOOR` candidates so a small rollout
    still places its threshold on more than a handful of gaps. Scoring every eligible row instead
    would mean a policy forward over most of a 98,304-row buffer — a cost the arm does not need to
    pay to place a threshold.
    """
    q = max(float(quantile), 1e-3)
    return int(max(SELECTION_POOL_FLOOR, np.ceil(float(int(n_forks)) / q)))


def top2_gaps(logits, masks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gap, top1, top2)`` over a batch of masked action logits.

    The gap `forks.py` computes verbatim: illegal actions are set to ``-inf``, the rows are sorted
    descending, and the gap is ``logit[a1] - logit[a2]``. A row with fewer than two legal actions
    reports ``inf`` (never contested) rather than raising — the eligibility filter has already
    removed those, and a NaN here would poison the quantile.
    """
    lg = np.array(logits, dtype=np.float64, copy=True)
    mk = np.asarray(masks)
    lg[mk <= 0] = -np.inf
    order = np.argsort(-lg, axis=1)
    a1 = order[:, 0].astype(np.int64)
    a2 = order[:, 1].astype(np.int64) if lg.shape[1] > 1 else a1
    v1 = np.take_along_axis(lg, a1[:, None], axis=1).reshape(-1)
    v2 = np.take_along_axis(lg, a2[:, None], axis=1).reshape(-1)
    gap = v1 - v2
    gap = np.where(np.isfinite(gap), gap, np.inf)
    return gap, a1, a2


def contested_threshold(gaps, quantile: float) -> float:
    """The shard's OWN ``quantile`` of the top-2 gap — `forks.py`'s ``np.quantile(gaps, q)``.

    Infinite gaps (a row with one legal action) are dropped before the quantile so they cannot drag
    it to ``inf``; with nothing finite left the threshold is ``-inf``, i.e. nothing is contested.
    """
    g = np.asarray(gaps, dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return float("-inf")
    return float(np.quantile(g, float(np.clip(quantile, 0.0, 1.0))))


def contested_select(pool, gaps, absv, threshold: float, absv_band: float,
                     n_forks: int) -> List[int]:
    """Indices into ``pool`` of the decisions to FORK — TIGHTEST GAP FIRST, capped at ``n_forks``.

    A row qualifies when ``gap <= threshold`` — the paired-refit's rule, verbatim — or, when
    ``--fork-contested-absv > 0``, when ``|V - 0.5| <= absv_band``. That second arm is OFF by
    default and forfeits the comparison to the registered baseline when it is not; see
    :data:`DEFAULT_CONTESTED_ABSV`.

    Ties in the gap are broken by the pool's own ``(t, env)`` order, so the selection is a function
    of the buffer and not of a sort's stability.
    """
    g = np.asarray(gaps, dtype=np.float64)
    keep = g <= float(threshold)
    if float(absv_band) > 0.0 and absv is not None:
        keep = keep | (np.abs(np.asarray(absv, dtype=np.float64)) <= float(absv_band))
    idx = [int(i) for i in np.flatnonzero(keep)]
    idx.sort(key=lambda i: (float(g[i]), tuple(pool[i])))
    return idx[: max(0, int(n_forks))]


def branch_actions(legal: Sequence[int], top1: int, top2: int, n_branches: int, *,
                   salt: str, seed: int) -> Dict[str, int]:
    """``{branch_name: action}`` for one fork — `forks.py`'s draw, verbatim.

    The random branch is drawn from ``legal \\ {top1, top2}`` with a DECISION-KEYED hash, so it is
    reproducible and independent of the policy's own ordering: a probability-ordered draw would
    spend the branch on what the policy already covers, and the whole reason the third branch
    exists is the 4.5 % of forks where a uniformly random legal alternative beat BOTH policy
    candidates.

    Returns only ``top1``/``top2`` at ``n_branches == 2``, and also when there is no third legal
    action to draw (the caller counts that as ``no_rand`` rather than silently forking two).
    """
    acts = {"top1": int(top1), "top2": int(top2)}
    if int(n_branches) < 3:
        return acts
    others = [int(a) for a in legal if int(a) not in (int(top1), int(top2))]
    if not others:
        return acts
    acts["rand"] = others[_sha(salt, seed) % len(others)]
    return acts


# ── meters ────────────────────────────────────────────────────────────────────────────────────
def pairwise_rows(forks: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """``(dv, dy)`` over every NON-TIED branch pair of every fork.

    ``dv`` is ``V(s'_a) - V(s'_b)`` on the two branches' SUCCESSOR states — the first state after
    the fork action resolves, which is the state a one-ply search would score — and ``dy`` is
    ``outcome_a - outcome_b``. A pair is dropped when the two branches share an outcome (the
    79.7 % structural tie tax), when either branch has no successor (it ended AT the fork turn),
    or when either branch was CAPPED: a 250-turn cap is decided by SEAT and not by the position
    (`counterfactual._battle_outcome`), so it is excluded, never scored 0.5.
    """
    dvs, dys = [], []
    for f in forks:
        br = [b for b in (f.get("branches") or {}).values()
              if b.get("succ_value") is not None and b.get("outcome") is not None
              and not b.get("capped")]
        for i in range(len(br)):
            for j in range(i + 1, len(br)):
                ya, yb = float(br[i]["outcome"]), float(br[j]["outcome"])
                if ya == yb:
                    continue
                dvs.append(float(br[i]["succ_value"]) - float(br[j]["succ_value"]))
                dys.append(ya - yb)
    return np.asarray(dvs, dtype=np.float64), np.asarray(dys, dtype=np.float64)


def pairwise_accuracy(forks: Sequence[dict]) -> Tuple[float, int]:
    """``(accuracy, n_pairs)`` — THE DIRECT DISCRIMINATION METRIC, on this rollout's own forks.

    The identical quantity `paired_refit_discrimination_2026-09-14` measured offline at
    **0.5169 [0.4800, 0.5524]** for the promoted head and **0.6032** for a frozen-trunk refit: the
    fraction of non-tied sibling pairs on which ``sign(V(s'_a) - V(s'_b))`` agrees with which
    branch actually won. A pair whose ``dv`` is exactly 0 scores 0.5 — the honest value of a
    coin-flip tie-break, and it is counted rather than dropped so a head that collapses to a
    constant reads 0.5 instead of reading "no data".

    ⚠️ **It is an ON-STREAM ESTIMATE, not the endpoint.** These pairs come from the states the arm
    TRAINED on, so it is an in-sample number and drifts optimistic; the registered endpoint is a
    held-out read on FRESH contested forks. It is here because it is nearly free and because a run
    whose ``fork/pairwise_acc`` never leaves 0.5 is a run whose treatment is not landing.
    """
    dv, dy = pairwise_rows(forks)
    if dv.size == 0:
        return float("nan"), 0
    agree = np.where(dv == 0.0, 0.5, (np.sign(dv) == np.sign(dy)).astype(np.float64))
    return float(agree.mean()), int(dv.size)


def tie_rate(forks: Sequence[dict]) -> float:
    """Fraction of forks whose branches ALL share one outcome — the structural tax, measured.

    The offline dataset read 79.7 % of branch PAIRS tied; this is the per-FORK version (every
    branch the same), which is the number that says whether a fork bought any ordering at all.
    Forks with fewer than two scorable branches are excluded from both halves of the ratio.
    """
    n = tied = 0
    for f in forks:
        ys = [float(b["outcome"]) for b in (f.get("branches") or {}).values()
              if b.get("outcome") is not None and not b.get("capped")]
        if len(ys) < 2:
            continue
        n += 1
        tied += int(len(set(ys)) == 1)
    return float(tied) / n if n else float("nan")


def random_wins_rate(forks: Sequence[dict]) -> float:
    """THE BLIND-SPOT RATE: the fraction of three-branch forks where the RANDOM branch beat BOTH
    policy candidates. The offline dataset read **4.5 % [3.99, 5.16]**.

    It is the one meter that says whether the third branch is earning its simulation: at 0 the arm
    is paying 1.5x for a distinction (top-1 vs top-2) the outcome already showed is worth 0.0004.
    """
    n = beats = 0
    for f in forks:
        br = f.get("branches") or {}
        r = br.get("rand")
        ps = [br.get("top1"), br.get("top2")]
        if r is None or r.get("outcome") is None or r.get("capped"):
            continue
        ys = [float(p["outcome"]) for p in ps
              if p is not None and p.get("outcome") is not None and not p.get("capped")]
        if len(ys) < 2:
            continue
        n += 1
        beats += int(float(r["outcome"]) > max(ys))
    return float(beats) / n if n else float("nan")


def sim_steps_share(forks: Sequence[dict], n_steps: int, n_envs: int) -> float:
    """Branch decisions / the trainee's own collection decisions, for THIS rollout.

    The number the flag lives or dies by. The numerator is MEASURED — each branch reports how many
    live decisions it actually took — and the denominator is ``n_steps x n_envs``. The sim also
    runs an opponent on both sides of both halves, so the ratio is scale-free in that factor and
    reads as "this rollout's forks cost X times the rollout's own simulation".
    """
    denom = float(int(n_steps) * int(n_envs))
    if denom <= 0.0:
        return float("nan")
    total = 0.0
    for f in forks:
        for b in (f.get("branches") or {}).values():
            total += float(b.get("decisions") or 0.0)
    return total / denom


def fork_metrics(*, forks: Sequence[dict], requested: int, eligible: int, pool: int,
                 threshold: float, injected_rows: int, buffer_rows: int, masked_rows: int,
                 fraction: float, branches: int, crn: str, seconds: float,
                 records_missing: int, n_steps: int, n_envs: int,
                 opp_class: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """The ``fork/*`` TB family: what the forking DID and what it COST, per rollout.

    Every entry is a fact about THIS rollout. The six the design registers are ``rate``,
    ``branch_share``, ``tie_rate``, ``random_wins``, ``pairwise_acc`` and ``sim_steps_share``; the
    rest are the plumbing a reader needs before trusting those six.
    """
    acc, n_pairs = pairwise_accuracy(forks)
    done = [f for f in forks if f.get("branches")]
    out: Dict[str, float] = {
        "fraction": float(fraction),
        "branches": float(int(branches)),
        "crn_draws": 1.0 if str(crn) == "dice_and_draws" else 0.0,
        "eligible": float(int(eligible)),
        "pool": float(int(pool)),
        "gap_threshold": float(threshold),
        "requested": float(int(requested)),
        # `rate` is forks per BATTLE — forks divided by the episodes that FINISHED in this buffer,
        # which is the denominator the design's "forks per battle" names. It is NOT forks per
        # decision: that is `fraction`, which is already published.
        "forks": float(len(done)),
        "failed": float(int(requested) - len(done)),
        "records_missing": float(int(records_missing)),
        "tie_rate": tie_rate(forks),
        "random_wins": random_wins_rate(forks),
        "pairwise_acc": acc,
        "pairwise_pairs": float(n_pairs),
        "sim_steps_share": sim_steps_share(forks, n_steps, n_envs),
        "seconds": float(seconds),
        "injected_rows": float(int(injected_rows)),
        "masked_rows": float(int(masked_rows)),
        # What share of the rows the PPO objective sees came from a branch rather than from the
        # trainee's own play. It is the honest headline for the population substitution: these rows
        # were measured against a self-like opponent, not the episode's real one.
        "branch_share": (float(injected_rows) / float(int(buffer_rows) + int(injected_rows))
                         if (int(buffer_rows) + int(injected_rows)) > 0 else float("nan")),
    }
    if opp_class is not None and len(opp_class):
        cls = np.asarray(opp_class, dtype=np.int64)
        out["bot_share"] = float((cls == 0).mean())
    return out


def forks_per_battle(n_forks: int, episode_starts) -> float:
    """``fork/rate`` — forks divided by the episodes that STARTED in this buffer.

    Episode starts and not terminals: the buffer's ``episode_starts`` plane is the one count that
    needs no outcome scan, and over a full rollout the two differ by at most one per env.
    """
    es = np.asarray(episode_starts, dtype=np.float64)
    n_ep = float((es >= 0.5).sum())
    return float(int(n_forks)) / n_ep if n_ep > 0 else float("nan")
