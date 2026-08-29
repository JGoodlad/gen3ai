"""The search itself: branch the current decision, score the successors, pick an action.

**Shape of the first ply.** For each determinized world *w*, each of OUR legal actions *a*, each
α-weighted OPPONENT candidate *c* and each CRN dice draw *r*, roll the turn forward once and read
the critic on the successor. The action taken is

    argmax_a  Σ_w p(w) · Σ_c α(c) · (1/R) · Σ_r  V(s'(w, a, c, r))

**And then it DEEPENS, while the clock allows** (the owner's depth amendment, minutes after the
registration: "fixed depth" was shorthand for cheap, not a constraint). The depth-1 sweep above
runs first and unchanged; then, ply by ply, the top-*m* actions are expanded a ply deeper and the
values are backed up MAX-over-ours / α-weighted-over-theirs — the same alternation the expression
above already is, which is the point. The tree, the backup and the beam live in :mod:`deepen`; the
budget decides the realized depth and the row records it, so a 0.5 s cell reporting depth 1 and an
8 s cell reporting depth 2 is the finding rather than a configuration.

Three properties of the first-ply expression are the whole experiment and none is incidental:

* **the opponent is MARGINALIZED, the dice are AVERAGED, and OUR action is MAXIMIZED** — the
  three-axis variance measurement put the behavior-weighted split at OPP 59.7% >> DICE 26.5% >
  OUR 10.0%, so the axis that buys the most per unit of compute is the one α is spent on;
* **the dice are COMMON across arms** (CRN): draw *r* uses ONE seed shared by every (a, c) pair in
  the world, so a difference between two actions is not a difference between two dice streams.
  Every draw is a FRESHLY MINTED seed — the sim's own ``"original"`` stream is never used, because
  it is the dice the turn is actually about to be resolved with and reading it is one ply of
  clairvoyance (see :meth:`SearchEngine._run`, where the measurement and the reading it corrupted
  are recorded).
  ⚠️ CRN here shares the dice STREAM, not the roll→event MAPPING — two arms that consume a
  different number of draws desynchronize after the first divergence, which is why a one-seed
  sweep over-reads the OUR×OPP interaction by roughly 2× and why R>1 is on the budget ladder at
  all rather than pinned to 1;
* **a fallback is COUNTED, never silent.** Every path that fails to produce a scored arm returns
  the policy's own action tagged with a reason from :data:`FALLBACK_REASONS`. A search that
  quietly degraded to the policy would report a null dividend that is indistinguishable from a
  real one — the single most important thing this module can get wrong.

**Where the arms come from.** ``SearchSession`` (the warm clone-and-branch search server) opens a
root at the current turn from a LIVE-synthesized reconstruction record and expands every arm in
one round trip. The successor OBS is materialized through :func:`materialize_branches`, which
replays the shared battle prefix ONCE for the whole arm set rather than once per arm — the prefix
is the measured majority of a counterfactual arm's cost and grows linearly in the turn number, so
without prefix sharing the realized widths would collapse in exactly the late-game positions the
probe cares about.

**The one-sided / omniscient wall holds.** ``expand_many`` returns per-side chunks AND an
omniscient ``outcome``. Only the chunks reach the encoder — that is the same wall the re-roll path
keeps. The ORACLE arm's privilege is the true opponent TEAM in the record it searches, nothing
more; it never reads a referee-view board into an observation.

**Threading.** ``materialize_branches`` refuses to run on ``POKE_LOOP`` (it drives a replay player
through that loop and would deadlock on its own result), and ``choose_move`` runs there. So the
whole search executes in a worker thread and the player's ``choose_move`` is an ``async def`` that
awaits it — poke-env awaits an awaitable choice (``player.py``: ``if isinstance(choice, Awaitable)``),
which frees POKE_LOOP for exactly as long as the search holds it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.search_dividend import defensive as dfn
from main.search_dividend import determinize as dz
from main.search_dividend.alpha import (AlphaPublication, build_candidates,
                                        legal_choices_from_request)
from main.search_dividend.budget import (CostModel, Deadline, RealizedWidths, WidthCaps,
                                         WidthPlan, allocate)
from main.search_dividend.deepen import TreeNode, per_action_values, plan_beam
from main.search_dividend.racing import Racer, RacingConfig

ARMS = ("base", "honest", "oracle", "playoff")

#: How the budget is allocated ACROSS OUR ROOT ACTIONS. ``grid`` is the registered allocator — a
#: fixed K x R sweep scoring every action on every sample. ``racing`` is the adaptive alternative
#: (:mod:`racing`): the same samples, but candidates whose CRN-paired difference CI separates below
#: the leader stop being scored, and the saved arm evaluations buy more samples instead.
#: ``defensive`` (:mod:`defensive`) wraps the racer in the two REFUSALS the G/H/I probe trio
#: measured — a triage gate that never searches a decided position, and a futility stop that never
#: overrules without separation. ``grid`` is the DEFAULT and its code path is untouched.
ROOT_STRATEGIES = ("grid", "racing", "defensive")

#: The arms that search the TRUE opponent team. ``playoff`` inherits the oracle's world because its
#: depth-1 sweep is only a SCREEN — the hidden-information question is deliberately held fixed so
#: the cell measures the leaf ESTIMATOR (critic vs rollout) and nothing else.
TRUE_WORLD_ARMS = ("oracle", "playoff")

# Terminal successors, on the WIN-PROB scale. On the value scale the same constants are used and
# the caveat is reported with the row: a shaped-return V and a ±1 terminal are not the same units,
# so `--score win_prob` is the default whenever the head exists.
TERMINAL_VALUE = {"win": 1.0, "loss": 0.0, "tie": 0.5}


class _NoWorld(RuntimeError):
    """The honest arm could not build a single pool-consistent world. Its own exception type so
    the caller reports ``no_world`` rather than the generic ``search_error`` — "the pool has no
    completion for this information set" and "something threw" are different findings."""


@dataclass
class SearchConfig:
    arm: str = "base"
    budget_s: float = 1.0
    caps: WidthCaps = field(default_factory=WidthCaps)
    score: str = "auto"                 # auto | value | win_prob
    search_impl: str = "node"
    honest_swap_moves: bool = False     # axis M — see determinize.swap_unused_moves
    seed: int = 0
    # The iterative-deepening CAP, not a target: the wall-clock budget governs the realized depth,
    # and a ply that does not fit is simply not spent. 3 is the amendment's own example of what an
    # 8 s cell might reach on contenders; raising it costs nothing when the budget is small.
    max_depth: int = 3
    #: ``grid`` (the registered fixed sweep, DEFAULT), ``racing`` or ``defensive``.
    #: See :data:`ROOT_STRATEGIES`.
    root_strategy: str = "grid"
    #: Consulted on ``root_strategy`` in ``("racing", "defensive")`` — the defensive strategy
    #: RACES, so it inherits this config rather than carrying a second copy of an elimination
    #: threshold that could drift from it.
    racing: RacingConfig = field(default_factory=RacingConfig)
    #: Only consulted on ``root_strategy="defensive"``.
    defensive: dfn.DefensiveConfig = field(default_factory=dfn.DefensiveConfig)

    def __post_init__(self) -> None:
        if self.root_strategy not in ROOT_STRATEGIES:
            raise ValueError(f"unknown root_strategy {self.root_strategy!r} "
                             f"(want one of {ROOT_STRATEGIES})")

    def defensive_cfg(self) -> Optional[dfn.DefensiveConfig]:
        """The defensive config IF this is a defensive search, else ``None``.

        Every defensive-only rule reads through this rather than testing the strategy string at
        each site, so a grid/racing run cannot accidentally acquire one of them."""
        return self.defensive if self.root_strategy == "defensive" else None

    def effective_score(self) -> str:
        """The leaf mode ``batch_scores`` is actually asked for.

        Identical to ``self.score`` on every strategy but ``defensive``, which names its head
        EXPLICITLY instead of taking the battery's ``auto`` — because ``auto`` silently degrades
        to the scalar value head on a checkpoint without a win-prob one, and that is the exact arm
        probe G measured as NOT beating the played action. See
        :func:`~main.search_dividend.defensive.resolve_score_mode`.
        """
        cfg = self.defensive_cfg()
        return self.score if cfg is None else dfn.resolve_score_mode(cfg.leaf)

    def resolved_caps(self) -> WidthCaps:
        """The ORACLE arm is ONE world by construction (the true state), and the BASE arm has no
        search at all. Encoding that here rather than at every call site means a budget can never
        be spent on a world axis that does not exist."""
        if self.arm == "playoff":
            # The sweep is a SCREEN, not the verdict — it nominates two candidates and the paired
            # rollouts settle them. So it is pinned to the CHEAPEST honest configuration (one true
            # world, one dice draw) and every second the budget can spare goes to the rollouts,
            # which are the unbiased half. Spending the budget on screen width instead would buy
            # more of the estimator this arm exists to stop trusting.
            return WidthCaps(m_opp=self.caps.m_opp, k_worlds=1, r_dice=1)
        if self.arm == "oracle":
            return WidthCaps(m_opp=self.caps.m_opp, k_worlds=1, r_dice=self.caps.r_dice)
        if self.arm == "base":
            return WidthCaps(m_opp=0, k_worlds=0, r_dice=0)
        return self.caps

    def effective_max_depth(self) -> int:
        """The deepening cap this arm may actually use.

        ``playoff`` is pinned to 1 and it is not a tuning choice. The MAX backup at interior plies
        is a biased-high estimator under leaf noise (``E[max] >= max E``, the reconciliation memo's
        candidate (e)), so a deeper screen would ADD exactly the bias the rollouts are here to
        remove, and the cell would no longer isolate the leaf estimator.
        """
        return 1 if self.arm == "playoff" else max(1, int(self.max_depth))


@dataclass
class DecisionResult:
    action: int
    fallback: Optional[str]
    widths: RealizedWidths
    diagnostics: dict = field(default_factory=dict)
    scores: Optional[dict] = None       # {action_index: aggregated score}
    policy_action: Optional[int] = None
    #: The ``playoff`` arm's second-stage block, or ``None`` on every other arm. An ADDITIVE row
    #: field (ladder requirement 3, 87a3f91): one schema, extended, never a second one.
    playoff: Optional[dict] = None

    @property
    def changed(self) -> bool:
        return (self.policy_action is not None and self.fallback is None
                and int(self.action) != int(self.policy_action))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def batch_scores(model, obs: np.ndarray, masks: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """``(scores, mode_used)`` for a batch of successor observations.

    ``win_prob`` reads the head's stashed logit from the SAME forward that produced the values,
    so the two never come from different states. ``auto`` prefers it and falls back to the shaped
    value when the run trained no win-prob head.

    **ONE forward for the whole arm set, and that is the single largest perf fact here.** Measured
    2026-08-23 on the live checkpoint with BLAS pinned: the extractor costs 27.95 ms at B=1 and
    1.53-1.73 ms per row at B=64-256, so scoring N arms one at a time would cost ~16-18x what this
    does. It is already the cheapest of the three per-arm costs (1.45-1.95 ms/row against the
    materializer's 4.8 and the sim branch's 2.3).

    Both tensors are cast to **float32** — the mask arrives from the materializer as whatever the
    encoder produced, while every LIVE forward passes a float32 one. A dtype difference is
    invisible to eager and a fresh 1.5 s TRACE under ``--compile-extractor`` (measured), which is
    the same hazard class as the dict key set. Normalizing here costs a copy of an 11-wide row.

    🚨 **THE STASH IS SHARED MUTABLE STATE AND ITS WIDTH IS CHECKED, NEVER ASSUMED.**
    ``last_win_prob_logits`` is an attribute of ONE extractor object, and in the MIRROR mode both
    sides play the same ``model``: the searched side runs this call in a worker thread (so
    ``materialize_branches`` is off ``POKE_LOOP``) while the unsearched side's own B=1 forward runs
    on ``POKE_LOOP``. A forward that lands between ``predict_values`` returning and the ``getattr``
    below leaves a stash describing a DIFFERENT state — and at B=1 against an N-arm batch the
    consequence is not a wrong number but a SHORT one, which ``zip`` in ``_expand_ply`` would
    silently truncate to one scored arm, handing the decision to whichever action happened to sort
    first. Every other tie between a value and a stash in this tree is width-checked (α's clause 3
    raises on exactly this shape); this one was not. A mismatch now RAISES, which
    :meth:`SearchEngine.choose` records as a counted ``search_error`` fallback — visible in the
    histogram, never a silently mis-scored decision.
    """
    import torch

    policy = model.policy
    dev = model.device
    n = int(np.asarray(obs).shape[0])
    with torch.no_grad():
        inp = {"observation": torch.as_tensor(obs, dtype=torch.float32).to(dev),
               "action_mask": torch.as_tensor(masks, dtype=torch.float32).to(dev)}
        values = policy.predict_values(inp).squeeze(-1).float().cpu().numpy()
        wp_logits = getattr(getattr(policy, "features_extractor", None),
                            "last_win_prob_logits", None)
        if wp_logits is not None and int(wp_logits.shape[0]) != n:
            raise RuntimeError(
                f"win-prob stash width {int(wp_logits.shape[0])} != scored batch {n} — the "
                "stash belongs to a DIFFERENT forward. In the mirror both players share one "
                "policy object and the unsearched side forwards on POKE_LOOP while this call "
                "runs in the search's worker thread; give each side its own model, or score "
                "with --score value.")
        wp = (torch.sigmoid(wp_logits.float()).squeeze(-1).cpu().numpy()
              if wp_logits is not None else None)
    if mode == "value" or (wp is None and mode in ("auto", "win_prob")):
        return np.asarray(values, dtype=np.float64), "value"
    if wp is None:
        return np.asarray(values, dtype=np.float64), "value"
    return np.asarray(wp, dtype=np.float64).reshape(-1), "win_prob"


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


@dataclass
class _PlyContext:
    """The per-world facts every ply of one decision's expansion shares."""

    side: str
    other: str
    record: object
    prefix: Sequence[str]
    our_history: List[int]
    pub: Optional[AlphaPublication]
    seeds: Sequence[str]
    m_opp: int

    def decision_index(self, ply: int) -> int:
        """Which decision row a ply-``ply`` successor's observation is.

        ``our_history`` holds the actions BEFORE the live decision, so the live decision is row
        ``len(our_history)`` and each ply adds one. The whole depth generalization is this line:
        depth 1 encodes at ``+1``, depth 2 at ``+2``, and everything else is unchanged."""
        return len(self.our_history) + int(ply)


class SearchEngine:
    """One engine per battle-playing player. Holds the WARM search session and the running cost
    model, so the ~0.6 s driver spawn is paid once per battery rather than once per decision."""

    def __init__(self, *, model, mappings, cfg: SearchConfig,
                 pool_packed: Optional[Sequence[str]] = None, playoff=None):
        self.model = model
        self.mappings = mappings
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        #: The second-stage scorer, or ``None``. Present only on the ``playoff`` arm — injected
        #: rather than constructed here so a test can drive the decision rule with a scripted sim.
        self.playoff = playoff
        self._session = None
        # An EWMA of the two measured costs. Seeded from the CostModel defaults and updated after
        # every decision, so a battery that starts fast and slows down (the prefix grows with the
        # turn) re-plans its widths instead of missing the deadline by a constant factor.
        self._cost = CostModel()
        self._pool_packed = pool_packed
        self._pool_mons: Optional[List[List[dz.MonSet]]] = None
        self._gender_tbl: Optional[Dict[str, str]] = None
        self._move_bank: Optional[Dict[str, list]] = None

    # -- lifecycle ----------------------------------------------------------

    def session(self):
        from utils.bridge.search_session import SearchSession

        if self._session is None:
            self._session = SearchSession(impl=self.cfg.search_impl)
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _pool(self) -> Tuple[List[List[dz.MonSet]], Dict[str, str], Dict[str, list]]:
        if self._pool_mons is None:
            packed, gender = dz.load_pool(self._pool_packed)
            self._pool_mons = [dz.split_team(p, gender) for p in packed]
            self._gender_tbl = gender
            self._move_bank = dz.species_move_bank(self._pool_mons)
        return self._pool_mons, self._gender_tbl, self._move_bank   # type: ignore[return-value]

    # -- the decision -------------------------------------------------------

    def choose(self, *, record, side: str, turn: int, our_history: Sequence[int],
               our_tokens: Dict[int, str], observed_our_lines: Sequence[str],
               pub: Optional[AlphaPublication], policy_action: int,
               opp_true_packed: Optional[str] = None,
               root_win_prob: Optional[float] = None) -> DecisionResult:
        """Run the depth-1 search for ONE decision and return the action to play.

        ``our_tokens`` maps each legal action INDEX to its sim choice string, built live from the
        real action mapper — the search never re-derives legality, it inherits it.
        ``opp_true_packed`` is the ORACLE arm's privilege and must be ``None`` on every other arm;
        :meth:`_worlds` raises rather than silently searching the truth.

        ``root_win_prob`` is the win-prob head's read at the LIVE decision, captured off the same
        forward that produced ``policy_action`` (the search's own forwards clobber that stash, so
        it cannot be re-read here). It is the defensive gate's only input and is ignored on every
        other strategy.
        """
        caps = self.cfg.resolved_caps()
        widths = RealizedWidths(planned={}, n_our_actions=len(our_tokens))
        if self.cfg.arm == "base" or caps.k_worlds <= 0:
            widths.planned = WidthPlan(0, 0, 0).as_dict()
            return DecisionResult(policy_action, "no_search", widths,
                                  policy_action=policy_action)
        if not our_tokens:
            return DecisionResult(policy_action, "not_move_selection", widths,
                                  policy_action=policy_action)

        # THE TRIAGE GATE, and it runs BEFORE the clock starts — a gated decision costs nothing at
        # all, which is what makes the banked seconds real rather than an accounting entry.
        gated = self._gate(widths, len(our_tokens), root_win_prob)
        if gated is not None:
            widths.planned = WidthPlan(0, 0, 0).as_dict()
            return DecisionResult(policy_action, gated, widths, policy_action=policy_action)

        deadline = Deadline(self.cfg.budget_s)
        plan = allocate(self.cfg.budget_s, len(our_tokens), self._cost, caps)
        widths.planned = plan.as_dict()
        widths.dice = plan.r_dice
        widths.worlds_requested = plan.k_worlds
        try:
            res = self._run(record, side, turn, our_history, our_tokens, observed_our_lines,
                            pub, policy_action, opp_true_packed, plan, widths, deadline)
        except _NoWorld as e:
            widths.elapsed_s = round(deadline.elapsed(), 4)
            return DecisionResult(policy_action, "no_world", widths,
                                  diagnostics={"error": str(e)}, policy_action=policy_action)
        except Exception as e:                       # noqa: BLE001
            # A search failure must never cost the battle: the battery's whole point is a
            # per-arm win rate, and a crashed decision would silently drop a game.
            widths.elapsed_s = round(deadline.elapsed(), 4)
            return DecisionResult(policy_action, "search_error", widths,
                                  diagnostics={"error": f"{type(e).__name__}: {e}"},
                                  policy_action=policy_action)
        if self.cfg.defensive_cfg() is not None:
            if res.fallback is None:
                res = self._defensive_confirm(res, record=record, turn=turn,
                                              our_tokens=our_tokens,
                                              policy_action=policy_action, deadline=deadline)
            # The clock the strategy handed back. On a futility stop the race stopped early and
            # this is real time a time manager could move to a contested decision; on an overrule
            # it is the granularity residual. Recorded either way so the two can be told apart by
            # the verdict beside it rather than by inference.
            widths.defensive_banked_s = round(max(0.0, deadline.remaining()), 4)
        if self.cfg.arm == "playoff" and res.fallback is None:
            # THE SECOND STAGE. The sweep above is now only a nomination: it is the same biased
            # critic estimator the R-ladder convicted, so its argmax is deliberately NOT acted on.
            # `_adjudicate` replaces the top1-vs-top2 comparison — the one the memo measured at 54%
            # noise-to-margin — with paired rollouts to a terminal, and hands the decision back to
            # the POLICY whenever those do not resolve it.
            res = self._adjudicate(res, record=record, turn=turn, our_tokens=our_tokens,
                                   policy_action=policy_action, deadline=deadline)
        widths.elapsed_s = round(deadline.elapsed(), 4)
        self._update_cost(widths)
        return res

    def _gate(self, widths: RealizedWidths, n_legal: int,
              root_win_prob: Optional[float]) -> Optional[str]:
        """The DEFENSIVE triage gate. ``None`` = search; a string = the counted fallback reason.

        ``None`` on every other strategy, so ``grid`` and ``racing`` never see this at all.

        The two refusals are separate branches because they answer different questions, and the
        second one is the one probe H's whole sweep was about: `|P(win) - 0.5| >= wp_margin` does
        NOT claim the search would agree with the policy there (the flip rate is flat at ~0.69
        everywhere and RISES in the forced class), it claims that being overruled in a decided
        position is worth almost nothing — 83.0% of the claimed dividend sits in the 22.7% of
        decisions worth >= 5 pp, and this feature is the only cheap one that finds them.

        A checkpoint with no win-prob head is REFUSED rather than imputed to 0.5. Imputing would
        route every decision on such a run into the searched class — the most expensive possible
        reading of a missing measurement, and one that would produce a full set of healthy
        counters while measuring a different strategy.
        """
        cfg = self.cfg.defensive_cfg()
        if cfg is None:
            return None
        if root_win_prob is None:
            widths.defensive_no_win_prob = True
            widths.defensive_banked_s = round(float(self.cfg.budget_s), 4)
            return "defensive_no_win_prob"
        widths.defensive_root_win_prob = round(float(root_win_prob), 6)
        reason = dfn.gate(int(n_legal), float(root_win_prob), cfg)
        if reason == dfn.GATE_SEARCH:
            return None
        widths.defensive_gate_reason = reason
        widths.defensive_verdict = dfn.VERDICT_FORCED
        # The WHOLE budget, because the gate runs before the clock starts. That is the point of
        # placing it there: a forced decision costs the policy's own forward and nothing else.
        widths.defensive_banked_s = round(float(self.cfg.budget_s), 4)
        return "defensive_forced"

    def _defensive_confirm(self, res: DecisionResult, *, record, turn: int,
                           our_tokens: Dict[int, str], policy_action: int,
                           deadline: Deadline) -> DecisionResult:
        """The OPTIONAL fourth stage: settle a proposed overrule with paired terminal rollouts.

        Off by default (``confirm_rollouts=0``), and the first registered cell runs without it —
        one new mechanism at a time, so the race's own verdict is measured before a second filter
        is stacked on it. When on, it reuses the ``playoff`` machinery verbatim rather than
        reimplementing pairing: the two candidates are rolled out under the SAME post-divergence
        dice and the same policy-sampling RNG, and the playoff may act only when the paired
        difference clears ``2·SE`` over ``>= MIN_PAIRS`` pairs. An inconclusive confirm returns the
        POLICY's action, which is the same refusal the futility stop makes one level up.

        ⚠️ **The screen scores handed to the playoff are NOMINATIONS, not the race's means.** The
        pair is ``{race_winner: 1.0, policy_action: 0.0}`` because ``top_two`` orders by score and
        an ELIMINATED action's mean is FROZEN at the round it went out — it can exceed the final
        leader's mean without ever having dominated it, which would silently swap the pair. The
        race's real means stay in ``scores``/``diagnostics``; the ``playoff.margin`` field on a
        confirmed decision is therefore 1.0 by construction and is not a margin.
        """
        from main.search_dividend.playoff import STAGE_ERROR, PlayoffResult

        cfg = self.cfg.defensive_cfg()
        if cfg is None or int(cfg.confirm_rollouts) <= 0 or self.playoff is None:
            return res
        if res.widths.defensive_verdict != dfn.VERDICT_OVERRULED:
            return res
        a1, a2 = int(res.action), int(policy_action)
        if a1 not in our_tokens or a2 not in our_tokens:
            return res
        try:
            po = self.playoff.adjudicate(
                scores={a1: 1.0, a2: 0.0}, policy_action=policy_action, our_tokens=our_tokens,
                record=record, turn=int(turn), deadline=deadline, rng=self.rng)
        except Exception as e:                           # noqa: BLE001
            po = PlayoffResult(int(policy_action), STAGE_ERROR,
                               error=f"{type(e).__name__}: {e}")
        action = int(po.action)
        res.widths.defensive_confirm_stage = po.stage
        # The verdict follows the ACTION, not the stage: a confirm that declined played the
        # policy's move, so the decision `kept` it. Which of the two ways it was kept is the
        # `defensive_confirm_stage` beside it — folded apart, never summed together.
        res.widths.defensive_verdict = dfn.verdict(True, action, int(policy_action))
        diag = dict(res.diagnostics or {})
        diag["playoff"] = po.as_dict()
        return DecisionResult(action, None, res.widths, diagnostics=diag, scores=res.scores,
                              policy_action=policy_action, playoff=po.as_dict())

    def _adjudicate(self, res: DecisionResult, *, record, turn: int,
                    our_tokens: Dict[int, str], policy_action: int,
                    deadline: Deadline) -> DecisionResult:
        """Run the top-2 playoff over a successful screen and fold its verdict into the result.

        A playoff that RAISES is caught here for the same reason a search failure is caught in
        :meth:`choose` — the battery's unit of account is a finished game, and a crashed decision
        would drop one. It becomes a counted ``playoff_error`` fallback to the policy's action.
        """
        from main.search_dividend.playoff import PlayoffResult, STAGE_ERROR

        if self.playoff is None:
            return res
        try:
            po = self.playoff.adjudicate(
                scores=res.scores or {}, policy_action=policy_action, our_tokens=our_tokens,
                record=record, turn=int(turn), deadline=deadline, rng=self.rng)
        except Exception as e:                           # noqa: BLE001
            po = PlayoffResult(int(policy_action), STAGE_ERROR,
                               error=f"{type(e).__name__}: {e}")
        diag = dict(res.diagnostics or {})
        diag["playoff"] = po.as_dict()
        return DecisionResult(int(po.action), po.fallback, res.widths, diagnostics=diag,
                              scores=res.scores, policy_action=policy_action, playoff=po.as_dict())

    # -- internals ----------------------------------------------------------

    def _run(self, record, side, turn, our_history, our_tokens, observed_our_lines, pub,
             policy_action, opp_true_packed, plan: WidthPlan, widths: RealizedWidths,
             deadline: Deadline) -> DecisionResult:
        if self.cfg.root_strategy in ("racing", "defensive"):
            # A WHOLE separate allocator, not a mode flag threaded through this one. The grid body
            # below is left byte-identical on purpose: `--root-strategy racing` is an experiment,
            # and an experiment that also perturbs its own control is not one.
            res = self._run_racing(record, side, turn, our_history, our_tokens,
                                   observed_our_lines, pub, policy_action, opp_true_packed,
                                   plan, widths, deadline)
            # DEFENSIVE is a POST-RULE over the same race, not a third allocator. The race is the
            # measurement (probe G's pairing + probe I's seq elimination); what defensive adds is
            # the refusal to act on a race that did not separate. Layering it keeps the racing arm
            # available unchanged as its own control.
            return res if self.cfg.defensive_cfg() is None else \
                self._apply_defensive(res, policy_action)
        other = "p2" if side == "p1" else "p1"
        ss = self.session()

        worlds = self._worlds(record, other, observed_our_lines, plan.k_worlds, opp_true_packed)
        widths.worlds_requested = len(worlds)

        # CRN: ONE seed per dice draw r, SHARED by every (action, candidate) arm — and across
        # worlds too, so two worlds disagree about the TEAM rather than about the dice.
        #
        # 🚨 EVERY DRAW IS A FRESH SEED. Draw 0 used to be the sim's ``"original"`` seed, which
        # `search_driver.js` honours by NOT swapping the PRNG (`if (!isOriginal) b.prng = new
        # PRNG(seed)`) — and `open_root` replays the record to the start of turn T, so that arm
        # resolved turn T from the battle's OWN mid-game PRNG state. Measured 2026-08-24 over 12
        # consecutive live decisions: expanding the REALIZED (our choice, their choice) pair under
        # ``"original"`` reproduced the real turn's our-side protocol BYTE-FOR-BYTE **11 of 12**
        # times, against **14 of 36** for fresh seeds (and those 14 are the turns with no dice in
        # them). That is not a sample of the dice — it IS the dice, one ply of clairvoyance no
        # player has.
        #
        # It also made the WIDTH LADDER measure the wrong thing, which is how it was found. Every
        # arm's score is a mean over the R draws, so the realized draw's share is 1/R: the leak
        # SHRANK as the budget bought resamples. `resolved_caps` pins the ORACLE arm to
        # ``k_worlds=1`` (the truth is one world) and `WIDTH_ORDER` spends dice LAST, so the oracle
        # arm is the only one whose leftover budget has nowhere to go but the dice axis — it ran at
        # R≈2.1 (1 s) and R≈7.2 (3 s) while the honest arm sat at R≈1.05. The mirror cells then
        # read oracle 0.383 / honest 0.554 against a 0.500 null, and the difference was the
        # dilution of a leak rather than the value of knowing the hidden team.
        seeds = [self._crn_seed(turn, r) for r in range(max(1, plan.r_dice))]

        actions = sorted(our_tokens)
        scores: Dict[int, float] = {a: 0.0 for a in actions}
        weight_sum = 0.0
        arm_diag: List[dict] = []
        alpha_diag: List[dict] = []
        world_diag: List[dict] = []
        beams: List[List[int]] = []
        widths.depth_planned = self.cfg.effective_max_depth()
        # ⚠️ ONE WORLD AT A TIME, opened AND consumed before the next. `open_root` CLEARS the
        # driver's node cache (that is what bounds a warm session's memory across battles), so
        # opening K roots up front invalidates every earlier one and `expand_many` then fails with
        # `unknown node nNN`. Found by the smoke: the oracle arm (K=1) was clean while the honest
        # arm lost 13 of 17 searched decisions to it.
        for wi, (wrec, wmeta) in enumerate(worlds):
            world_diag.append(wmeta)
            batch_cost = (self._cost.world_open_s
                          + self._cost.arm_s * len(actions) * plan.m_opp * len(seeds))
            if wi and not deadline.fits(batch_cost):
                widths.deadline_truncated = True
                break
            t_open = time.monotonic()
            try:
                root = ss.open_root(turn, record=wrec)
            except Exception as e:                   # noqa: BLE001
                widths.open_s += time.monotonic() - t_open
                widths.worlds_open_failed += 1
                wmeta["gate"] = f"open_failed: {type(e).__name__}: {e}"
                continue
            widths.open_s += time.monotonic() - t_open
            prefix = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
            if not dz.prefix_matches(observed_our_lines, prefix, turn=turn):
                # THE GATE. A world whose replay does not reproduce the protocol we actually saw
                # is a different battle; keeping it would answer about a position we were never
                # in. Dropped WITH A COUNTER (the precedent ran 535/535 and 615/615 clean, so a
                # non-zero count here is news, not noise).
                widths.worlds_gate_failed += 1
                wmeta["gate"] = "prefix_mismatch"
                continue
            widths.worlds_gated_ok += 1
            wmeta["gate"] = "ok"

            # The legal surface differs between worlds (a determinized bench holds different
            # mons), so the marginalization set is rebuilt per world — reusing one would branch
            # on switch targets that do not exist.
            legal = legal_choices_from_request((root.requests or {}).get(other))
            cands, diag = build_candidates(legal, pub, m_opp=plan.m_opp)
            alpha_diag.append(diag)
            if not cands:
                continue
            widths.opp_candidates = max(widths.opp_candidates, len(cands))
            ctx = _PlyContext(side=side, other=other, record=record, prefix=prefix,
                              our_history=list(our_history), pub=pub, seeds=seeds,
                              m_opp=plan.m_opp)
            got = self._score_world({"root": root, "meta": wmeta, "cands": cands},
                                    ctx, our_tokens, actions, widths, deadline)
            if got is None:
                continue
            per_action, beam, adiag = got
            arm_diag.append(adiag)
            if beam:
                beams.append(beam)
            for a in actions:
                scores[a] += per_action.get(a, 0.0)
            weight_sum += 1.0
        if weight_sum <= 0:
            return DecisionResult(policy_action, _no_arm_reason(widths, alpha_diag), widths,
                                  diagnostics={"worlds": world_diag, "alpha": alpha_diag},
                                  policy_action=policy_action)
        for a in actions:
            scores[a] /= weight_sum

        # WHICH actions the argmax may pick from. With no deepening: all of them, unchanged. Once
        # a ply was spent, only the beam — the actions whose values sit at the deepened depth. The
        # rule ACROSS worlds is the INTERSECTION of the per-world beams, because an action must be
        # deepened everywhere for its cross-world mean to be depth-consistent; an empty
        # intersection degrades to the union and says so rather than silently mixing depths.
        choices, beam_rule = _selectable_across_worlds(actions, beams)
        widths.beam_m = len(choices) if beams else 0

        # Tie-break TOWARD THE POLICY. A search that breaks an exact tie away from the policy's
        # own pick would report a `changed` decision it has no evidence for, and `change_rate` is
        # one of the two headline numbers here.
        best = max(choices, key=lambda a: (scores[a], a == policy_action))
        return DecisionResult(
            best, None, widths,
            diagnostics={"worlds": world_diag, "alpha": alpha_diag, "arms": arm_diag,
                         "beam": choices, "beam_rule": beam_rule,
                         "depth_realized": widths.depth_realized,
                         "score_mode": arm_diag[0].get("score_mode") if arm_diag else None},
            scores={int(a): round(float(v), 5) for a, v in scores.items()},
            policy_action=policy_action)

    def _run_racing(self, record, side, turn, our_history, our_tokens, observed_our_lines, pub,
                    policy_action, opp_true_packed, plan: WidthPlan, widths: RealizedWidths,
                    deadline: Deadline) -> DecisionResult:
        """The ADAPTIVE root allocator — successive elimination over CRN-paired samples.

        The width ORDER is inherited, not redesigned. ``m_opp`` is still whatever
        :func:`~main.search_dividend.budget.allocate` spent the first axis on, and a round is still
        one determinized world; what changes is that ``k_worlds`` and ``r_dice`` stop being a fixed
        grid and become a round SUPPLY, drawn in the registered order — every world once, then
        every world again with fresh CRN dice, up to ``k_worlds * r_dice`` rounds. The clock
        decides how many are actually spent, and because an eliminated action stops being scored,
        each later round is cheaper than the one before it. That is the entire mechanism: the same
        seconds buy more SAMPLES on fewer ACTIONS.

        **A round is depth 1.** Racing and iterative deepening are two different ways to spend the
        same clock and this build does not compose them — a first round allowed to deepen would
        consult ``deadline.remaining()`` and swallow the budget the race needs, so the race would
        be over a single sample. ``max_depth=1`` here is that decision made explicitly rather than
        emerging from whichever call happened to ask for the clock first.

        **An INCOMPLETE round is discarded rather than folded in.** If any live action failed to
        materialize a successor, the grid path's ``+= 0.0`` merely dilutes a mean; here it would
        eliminate that action permanently on a value that was never measured. The round is dropped
        with a counter instead — pairing integrity is the premise of every CI in this method.
        """
        other = "p2" if side == "p1" else "p1"
        ss = self.session()
        caps = self.cfg.resolved_caps()

        # The round SUPPLY asks for the CAP's worth of worlds, not the grid plan's. `plan.k_worlds`
        # is what a uniform sweep could afford over the full action set — precisely the number
        # racing exists to beat — so planning to it would cap the experiment at its own control.
        worlds = self._worlds(record, other, observed_our_lines, max(1, caps.k_worlds),
                              opp_true_packed)
        widths.worlds_requested = len(worlds)
        max_rounds = max(1, caps.k_worlds) * max(1, caps.r_dice)

        actions = sorted(our_tokens)
        racer = Racer(actions, self.cfg.racing)
        widths.depth_planned = 1
        widths.dice = 1
        arm_diag: List[dict] = []
        alpha_diag: List[dict] = []
        world_diag: List[dict] = []
        stop = "rounds"
        for j in range(max_rounds):
            if racer.resolved():
                stop = "resolved"
                break
            live = racer.live
            batch_cost = (self._cost.world_open_s
                          + self._cost.arm_s * len(live) * plan.m_opp)
            if j and not deadline.fits(batch_cost):
                widths.deadline_truncated = True
                stop = "deadline"
                break
            # Worlds cycle: rounds beyond the supply re-open an earlier world with FRESH dice,
            # which is the `r_dice` axis of the registered order arriving after `k_worlds`.
            wrec, wmeta = worlds[j % len(worlds)]
            wmeta = dict(wmeta)
            wmeta["round"] = j
            world_diag.append(wmeta)
            t_open = time.monotonic()
            try:
                root = ss.open_root(turn, record=wrec)
            except Exception as e:                   # noqa: BLE001
                widths.open_s += time.monotonic() - t_open
                widths.worlds_open_failed += 1
                wmeta["gate"] = f"open_failed: {type(e).__name__}: {e}"
                continue
            widths.open_s += time.monotonic() - t_open
            prefix = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
            if not dz.prefix_matches(observed_our_lines, prefix, turn=turn):
                widths.worlds_gate_failed += 1
                wmeta["gate"] = "prefix_mismatch"
                continue
            widths.worlds_gated_ok += 1
            wmeta["gate"] = "ok"

            legal = legal_choices_from_request((root.requests or {}).get(other))
            cands, diag = build_candidates(legal, pub, m_opp=plan.m_opp)
            alpha_diag.append(diag)
            if not cands:
                continue
            widths.opp_candidates = max(widths.opp_candidates, len(cands))
            ctx = _PlyContext(side=side, other=other, record=record, prefix=prefix,
                              our_history=list(our_history), pub=pub,
                              seeds=[self._crn_seed(turn, j)], m_opp=plan.m_opp)
            got = self._score_world({"root": root, "meta": wmeta, "cands": cands},
                                    ctx, {a: our_tokens[a] for a in live}, live, widths,
                                    deadline, max_depth=1)
            if got is None:
                continue
            per_action, _beam, adiag = got
            arm_diag.append(adiag)
            if not set(adiag.get("valued") or ()) >= set(live):
                # Not every live action produced a value; folding a 0.0 in would eliminate it on a
                # measurement that never happened.
                widths.racing_rounds_incomplete += 1
                wmeta["gate"] = "incomplete_round"
                continue
            racer.observe({a: per_action[a] for a in live})
        else:
            stop = "resolved" if racer.resolved() else "rounds"

        if racer.rounds <= 0:
            return DecisionResult(policy_action, _no_arm_reason(widths, alpha_diag), widths,
                                  diagnostics={"worlds": world_diag, "alpha": alpha_diag},
                                  policy_action=policy_action)
        out = racer.outcome(prefer=policy_action, stop_reason=stop)
        widths.racing_rounds = out.rounds
        widths.racing_eliminated = len(out.eliminated)
        widths.racing_resolved = out.stop_reason == "resolved"
        widths.racing_arms_saved = out.arms_grid - out.arms_spent
        return DecisionResult(
            out.action, None, widths,
            diagnostics={"worlds": world_diag, "alpha": alpha_diag, "arms": arm_diag,
                         "racing": out.as_dict(), "depth_realized": 1,
                         "score_mode": arm_diag[0].get("score_mode") if arm_diag else None},
            scores={int(a): round(float(v), 5) for a, v in out.means.items()},
            policy_action=policy_action)

    def _apply_defensive(self, res: DecisionResult, policy_action: int) -> DecisionResult:
        """THE FUTILITY STOP — the rule that makes this strategy defensive rather than adaptive.

        The racer always HAS a leader (its best empirical mean), and acting on that leader
        regardless of separation is precisely what every losing arm in this battery did. Probe I
        measured the separation distribution as U-shaped with an empty middle — **52.2% of root
        decisions never separate at all** within 32 paired samples, and among those that do the
        MEDIAN separates at the minimum-samples floor. So "did not separate" is not an unfinished
        race that more budget would settle; it is the search reporting that it cannot tell these
        actions apart, and the honest action there is the one the policy already chose.

        A search FAILURE (any counted fallback) is passed through untouched — the strategy has
        nothing to say about a decision that never produced a race, and re-labelling one would
        hide a driver problem behind a design choice.
        """
        if res.fallback is not None:
            return res
        separated = bool(res.widths.racing_resolved)
        v = dfn.verdict(separated, int(res.action), int(policy_action))
        res.widths.defensive_verdict = v
        action = dfn.resolve_action(v, int(res.action), int(policy_action))
        diag = dict(res.diagnostics or {})
        diag["defensive"] = {
            "verdict": v, "separated": separated,
            "race_action": int(res.action), "policy_action": int(policy_action),
            "root_win_prob": res.widths.defensive_root_win_prob,
            "leaf": (self.cfg.defensive.leaf if self.cfg.defensive_cfg() else None),
        }
        return DecisionResult(action, None, res.widths, diagnostics=diag, scores=res.scores,
                              policy_action=policy_action)

    def _score_batch(self, obs: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, str]:
        """THE LEAF SEAM — one place where the scoring head is chosen and then VERIFIED.

        ``batch_scores`` reports the mode it *used*, which may differ from the mode it was
        *asked for*: with no win-prob head it returns the scalar value readout and no error. On a
        defensive search that silent substitution swaps the arm probe G measured as beating the
        played action (+0.0219 [+0.0089, +0.0364]) for the one that does not clear zero (+0.0135
        [-0.0007, +0.0280]), with every counter in the battery reading identically. So the check
        happens here, at the return, rather than at the request.
        """
        scores, mode_used = batch_scores(self.model, obs, masks, self.cfg.effective_score())
        dfn.check_leaf(mode_used, self.cfg.defensive_cfg())
        return scores, mode_used

    def _score_world(self, w: dict, ctx: _PlyContext, our_tokens: Dict[int, str],
                     actions: Sequence[int], widths: RealizedWidths,
                     deadline: Deadline, *,
                     max_depth: Optional[int] = None
                     ) -> Optional[Tuple[Dict[int, float], List[int], dict]]:
        """Expand + score ONE world, deepening while the clock allows.

        Returns ``({action: E[score]}, beam, diagnostics)`` — ``beam`` empty when the world stayed
        at depth 1, which is the width-only reference the amendment kept alive inside the same run.
        """
        root, cands = w["root"], w["cands"]
        vroot = TreeNode(node_id=root.node_id, ended=False, our_tokens=dict(our_tokens),
                         requests=root.requests, path=())
        first = self._expand_ply(ctx, [(vroot, cands)], ply=1, widths=widths, deep=False)
        if first["n_scored"] <= 0:
            return None

        values = per_action_values(vroot)
        beam: List[int] = []
        ply = 1
        n_opp_cache: Dict[int, int] = {}

        def n_opp_at(leaf: TreeNode) -> int:
            key = id(leaf)
            if key not in n_opp_cache:
                n_opp_cache[key] = len(
                    legal_choices_from_request((leaf.requests or {}).get(ctx.other)))
            return n_opp_cache[key]

        md = self.cfg.effective_max_depth() if max_depth is None else max(1, int(max_depth))
        while ply < md:
            cand_beam, leaves, _n_arms = plan_beam(
                vroot, values, depth=ply, m_opp=ctx.m_opp, n_opp_at=n_opp_at,
                arm_cost_s=self._cost.arm_s, ply_overhead_s=self._cost.world_open_s,
                remaining_s=deadline.remaining())
            if not cand_beam or not leaves:
                break
            frontier: List[Tuple[TreeNode, list]] = []
            for leaf in leaves:
                legal = legal_choices_from_request((leaf.requests or {}).get(ctx.other))
                lc, _diag = build_candidates(legal, ctx.pub, m_opp=ctx.m_opp)
                if lc:
                    frontier.append((leaf, lc))
            if not frontier:
                break
            grown = self._expand_ply(ctx, frontier, ply=ply + 1, widths=widths, deep=True)
            if grown["n_scored"] <= 0:
                # The ply produced nothing scorable; the tree is unchanged in value terms, so the
                # decision stays at the depth it had rather than claiming one it did not reach.
                break
            ply += 1
            beam = cand_beam
            values = per_action_values(vroot)
            widths.depth_realized = max(widths.depth_realized, ply)

        per_action = {int(a): float(values.get(int(a), 0.0)) for a in actions}
        return per_action, beam, {
            "score_mode": first["score_mode"], "n_scored": first["n_scored"],
            "n_terminal": first["n_terminal"], "tier": w["meta"].get("tier"),
            "depth": ply, "beam": list(beam),
            # WHICH actions actually backed up to a value, as opposed to which were asked for.
            # `per_action` defaults a missing action to 0.0, which the grid's running mean merely
            # dilutes but a racer would read as a measurement and eliminate on.
            "valued": sorted(int(a) for a in values)}

    def _expand_ply(self, ctx: _PlyContext, frontier: Sequence[Tuple[TreeNode, list]], *,
                    ply: int, widths: RealizedWidths, deep: bool) -> dict:
        """Grow ONE ply: expand every (our action x their candidate x dice) arm of every frontier
        node, materialize the successors' observations in a single shared-prefix replay, score
        them, and hang the results on the tree.

        The root ply and a deepening ply are the SAME operation — that is why deepening is a loop
        over this method rather than a second code path. Two things differ and both are arguments:
        the decision row the observation is read at (``ctx.decision_index``) and the dice, which at
        a deeper ply is ONE freshly minted CRN seed shared across the whole ply (the dice axis is
        last in the registered width order, so a deeper ply never spends budget resampling it).
        """
        from agents.training.obs_materializer import Branch, materialize_branches

        seeds = list(ctx.seeds) if ply == 1 else [self._crn_seed(0, ply)]
        parents: List[TreeNode] = []
        acts: List[int] = []
        weights: List[float] = []
        payload: List[dict] = []
        for parent, cands in frontier:
            for a in sorted(parent.our_tokens):
                for c in cands:
                    for sd in seeds:
                        label = len(parents)
                        parents.append(parent)
                        acts.append(int(a))
                        weights.append(float(c.weight))
                        payload.append({"node_id": parent.node_id,
                                        f"{ctx.side}_action": parent.our_tokens[a],
                                        f"{ctx.other}_action": c.token,
                                        "seed": sd, "label": label})
        if not payload:
            return {"n_scored": 0, "n_terminal": 0, "score_mode": self.cfg.effective_score()}

        expanded = self.session().expand_many(payload)
        widths.arms_expanded += len(expanded)
        if deep:
            widths.deep_arms_expanded += len(expanded)

        username = ctx.record.username(ctx.side)
        branches: List[Branch] = []
        branch_of: List[int] = []
        n_terminal = 0
        n_scored = 0
        for e in expanded:
            li = int(e.label)
            parent = parents[li]
            if e.ended:
                lab = _terminal_label(e.outcome, username)
                parent.add_child(acts[li], weights[li],
                                 TreeNode(node_id=None, ended=True, value=TERMINAL_VALUE[lab],
                                          path=parent.path + (acts[li],)))
                widths.arms_terminal += 1
                n_terminal += 1
                n_scored += 1
                continue
            suffix = e.p1_chunks if ctx.side == "p1" else e.p2_chunks
            branches.append(Branch(chunks=suffix,
                                   actions=list(parent.path) + [acts[li]], label=li))
            branch_of.append(li)

        score_mode = self.cfg.effective_score()
        if branches:
            dec_i = ctx.decision_index(ply)
            traces = materialize_branches(
                ctx.prefix, branches, username=username,
                packed_team=ctx.record.packed_team(ctx.side), side=ctx.side,
                prefix_actions=list(ctx.our_history), battle_format=ctx.record.format_id,
                battle_tag=ctx.record.battle_tag, mappings=self.mappings,
                map_actions_at=dec_i, stop_after_decision=dec_i, encode_only_at={dec_i})
            obs_rows, mask_rows, keys, kept = [], [], [], []
            for li, mt in zip(branch_of, traces):
                if len(mt.decisions) <= dec_i:
                    continue                 # the successor never produced a request (rare)
                d = mt.decisions[dec_i]
                obs_rows.append(d.obs)
                mask_rows.append(d.mask)
                keys.append(li)
                kept.append(mt)
            if obs_rows:
                sc, score_mode = self._score_batch(np.stack(obs_rows), np.stack(mask_rows))
                by_label = {int(e.label): e for e in expanded}
                for li, mt, v in zip(keys, kept, sc):
                    parent = parents[li]
                    e = by_label[li]
                    # The child's OWN legal surface, from the REAL mapper — this is what makes a
                    # deeper ply possible at all, and it is already a by-product of the
                    # materialization the depth-1 pass ran. EMPTIED on a node that is not a clean
                    # move selection (see `branchable`), which makes such a node a leaf by
                    # construction everywhere downstream — `expandable()`, `leaves_under` and the
                    # cost estimate all agree without any of them having to know the rule.
                    tokens = (dict(mt.action_choices or {})
                              if branchable(e.requests, ctx.side) else {})
                    parent.add_child(
                        acts[li], weights[li],
                        TreeNode(node_id=e.node_id, ended=False, value=float(v),
                                 our_tokens=tokens,
                                 requests=e.requests, path=parent.path + (acts[li],)))
                    n_scored += 1
        widths.arms_scored += n_scored
        if deep:
            widths.deep_arms_scored += n_scored
        return {"n_scored": n_scored, "n_terminal": n_terminal, "score_mode": score_mode}

    def _worlds(self, record, opp_side: str, observed_our_lines: Sequence[str], k: int,
                opp_true_packed: Optional[str]) -> List[Tuple[object, dict]]:
        """The records to search. ORACLE = the truth, K=1. HONEST = K pool-consistent
        determinizations of the never-revealed slots."""
        if self.cfg.arm in TRUE_WORLD_ARMS:
            if opp_true_packed is None:
                raise ValueError(f"the {self.cfg.arm} arm needs the opponent's true packed team")
            return [(dz.record_with_team(record, opp_side, opp_true_packed, "-oracle"),
                     {"kind": "oracle", "tier": 0})]
        if opp_true_packed is not None:
            # A guard, not politeness: an honest arm handed the truth would produce an oracle
            # result under an honest label, which is the one confusion this experiment cannot
            # survive.
            raise ValueError("the honest arm must not be given the opponent's true team")

        pool, gender, bank = self._pool()
        base_packed = record.packed_team(opp_side)
        base = dz.split_team(base_packed, gender)
        revealed = dz.revealed_species(observed_our_lines, opp_side)
        if self.cfg.honest_swap_moves:
            used = dz.used_moves_by_species(observed_our_lines, opp_side)
            base, _n = dz.swap_unused_moves(base, revealed, used, bank, self.rng)
        dets, stats = dz.build_determinizations(base, revealed, pool, gender, k=k, rng=self.rng)
        out: List[Tuple[object, dict]] = []
        for i, d in enumerate(dets):
            out.append((dz.record_with_team(record, opp_side, d["packed"], f"-w{i}"),
                        {"kind": "honest", "tier": d["tier"],
                         "hidden": d["hidden_species"], "n_hidden": stats["n_hidden"]}))
        if not out:
            # ⚠️ The record's ``>player`` team IS the truth (the live builder read it off the
            # opponent object), so falling back to it is only legitimate when there is nothing
            # hidden to determinize — then that team IS our information set. With hidden slots
            # still open, the same fallback would be an ORACLE search wearing the honest label,
            # which is the one confusion this experiment cannot survive. Decline instead, and let
            # the caller count it.
            if stats["n_hidden"] == 0:
                out.append((dz.record_with_team(record, opp_side, base_packed, "-w0"),
                            {"kind": "honest_complete", "tier": 0, "n_hidden": 0}))
            else:
                raise _NoWorld(
                    f"no pool-consistent completion for {stats['n_hidden']} hidden slot(s) "
                    f"(tier1 donors={stats['tier1_donors']}, tier2={stats['tier2_donors']})")
        return out

    def _crn_seed(self, turn: int, r: int) -> str:
        return f"sodium,{self.rng.getrandbits(128):032x}"

    def _update_cost(self, widths: RealizedWidths) -> None:
        """EWMA BOTH measured costs so the next decision's plan reflects THIS battle's turn depth.

        A constant cost model would over-plan late game (the prefix replay grows linearly in the
        turn) and under-plan early, which shows up as a budget that is never spent.

        ⚠️ ``world_open_s`` used to be exempt from that and carried its 0.05 s default forever,
        which is not merely imprecise — it is a BIAS, and in the direction that costs width. A real
        open measures 0.055-0.064 s early and grows, and the arm term is derived by SUBTRACTION
        (``elapsed - opens*world_open_s``), so every second the model failed to attribute to the
        opens was attributed to the arms instead. ``arm_s`` is what the allocator divides the
        budget by, so an inflated one buys fewer arms — and it compounds on the honest arm, where
        K opens are the term being mis-estimated K times: measured 2026-08-23 interleaved against
        the un-measured model, the honest arm at 3 s went **K 3.23 → 4.62 worlds** (arms/decision
        149.7 → 222.5), i.e. the belief marginalization this arm exists for roughly doubled per
        second of the same budget. (That is the two changes of 2026-08-23 together — this and the
        materializer's per-arm restore.)

        The budget still under-runs (~65-85% spent) and that residual is GRANULARITY, not waste: a
        bump on any axis costs a whole world or a whole dice sweep, so the last partial one cannot
        be bought. Nothing here should "fix" it by over-planning — a committed world's arm set is
        not interruptible, so an optimistic plan overruns the deadline instead of truncating.
        """
        if widths.arms_scored <= 0 or widths.elapsed_s <= 0:
            return
        opens = max(1, widths.worlds_gated_ok + widths.worlds_gate_failed
                    + widths.worlds_open_failed)
        # The MEASURED open time when we have it; the running estimate otherwise (a decision that
        # opened nothing must not drive the term to zero).
        open_total = widths.open_s if widths.open_s > 0 else opens * self._cost.world_open_s
        arm_share = max(0.0, widths.elapsed_s - open_total)
        arm_s = arm_share / max(1, widths.arms_expanded)
        a = 0.3
        world_open_s = self._cost.world_open_s
        if widths.open_s > 0:
            world_open_s = max(1e-4, (1 - a) * world_open_s + a * (widths.open_s / opens))
        self._cost = CostModel(
            world_open_s=world_open_s,
            arm_s=max(1e-4, (1 - a) * self._cost.arm_s + a * arm_s))


def _no_arm_reason(widths: RealizedWidths, alpha_diag: Sequence[dict]) -> str:
    """Why a decision produced no scored arm — the counted fallback reason.

    A broken DRIVER and a wrong WORLD are different diagnoses; report the one that actually
    happened. Ties go to the driver — if it died, nothing else is trustworthy.
    """
    if widths.worlds_gated_ok == 0:
        return "root_failed" if widths.worlds_open_failed else "prefix_gate_failed"
    if not any(d.get("n_legal") for d in alpha_diag):
        return "no_candidates"
    return "no_scored_arm"


def branchable(requests: Optional[dict], side: str) -> bool:
    """Is this node a clean MOVE-SELECTION for ``side`` — the only kind the search may branch on?

    The root decision already obeys this rule: :meth:`SearchDividendPlayer._search` declines with
    ``not_move_selection`` on a ``force_switch``, because a forced switch has no branchable choice
    surface and a team preview none at all. Iterative deepening has to obey it at EVERY ply, and
    the reason is not symmetry — it is that a deeper node's ``action_choices`` come from the action
    MAPPER, which happily enumerates switch targets at a forced switch. Branching there sends the
    sim a move for a side that was never asked for one, and the successor replay then narrates a
    board the observation encoder does not agree with (measured 2026-08-23: the deepening arm's
    first live run logged ~1770 poke-env "message thinks p1: X is active, but it's not" warnings,
    a class the depth-1 arm produces exactly zero of).
    """
    return bool(legal_choices_from_request((requests or {}).get(side)))


def _selectable_across_worlds(actions: Sequence[int],
                              beams: Sequence[Sequence[int]]) -> Tuple[List[int], str]:
    """``(choices, rule)`` — which actions the cross-world argmax may pick from.

    ``rule`` is recorded rather than inferred, because the three cases mean different things and a
    reader must be able to tell them apart:

    * ``"depth1"`` — no world deepened; every scored action is comparable, unchanged behaviour.
    * ``"intersection"`` — every deepened world agreed on these, so their cross-world means are all
      at the same depth.
    * ``"union"`` — the beams did not overlap. The means then MIX depths, and that is a defect of
      the reading, not of the search; naming it keeps it out of the "it just worked" bucket.
    """
    if not beams:
        return sorted(actions), "depth1"
    inter = set(beams[0])
    for b in beams[1:]:
        inter &= set(b)
    if inter:
        return sorted(inter), "intersection"
    union: set = set()
    for b in beams:
        union |= set(b)
    return (sorted(union), "union") if union else (sorted(actions), "depth1")


def _terminal_label(outcome: dict, username: str) -> str:
    w = (outcome or {}).get("winner")
    return "win" if w == username else ("tie" if not w else "loss")
