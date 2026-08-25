"""The TOP-2 PLAYOFF — the one search variant the R-ladder verdict does not rule out.

**Why this arm exists.** The R-ladder (2026-08-24) swept the dice axis 32x and the harm did not
move: win rates 0.125 / 0.138 / 0.266 / 0.188 / 0.253 / 0.205 across R ∈ {1,2,4,8,16,32}. Averaging
removes VARIANCE, so a flat ladder says the disease is **leaf-estimator BIAS** — the critic's (and
the α opponent-model's) systematic error, which search amplifies by maximizing over it. Every
remaining fix that averages harder is therefore dead on arrival.

A playoff attacks the bias instead of the variance, and it attacks it at exactly the place the
reconciliation memo measured it worst: the **top1-vs-top2 comparison**, where our leaf noise
``sd = 0.0115`` sits at **54% of the ``top1−top2`` margin of 0.0213**
(``designs/research_state/wang_search_reconciliation.md`` §4). That single comparison decides the
action; everything else in the sweep is a ranking that never gets acted on. So this arm keeps the
depth-1 critic sweep as a **SCREEN** — cheap, biased, and used only to nominate two candidates —
and settles the nomination with **paired rollouts to a terminal**.

**Why rollouts are unbiased HERE and only here.** A rollout to terminal is scored by the sim, not
by the critic, so it carries no critic bias at all. What it does carry is *opponent-model* bias:
the estimand "what happens if I play a instead of b" depends on who the opponent is. In the
**mirror** the opponent IS the same network — so a self-rollout is the exact estimand, not an
approximation of it. That is why this arm is registered as a mirror cell and why its reading does
not transfer to the scripted roster unchanged.

**Why the rollouts are PAIRED, and what pairing buys.** Two candidates are rolled out under the
SAME post-divergence dice (``post_t_seed``) and the SAME policy-sampling RNG, and the statistic is
the per-draw DIFFERENCE ``d_r = score(a₁, r) − score(a₂, r)``. Any bias shared by the two arms —
the opponent's own imperfection, a mis-modelled matchup, the stall cap — enters both terms and
cancels in the difference. The residual is what actually distinguishes the two moves. (This is the
same common-random-numbers discipline the dice axis already uses inside the screen; here it is
applied to the axis that decides.)

**Why it can decline, and why that is the point.** The week's lesson is that a search which
overrides on noise is worse than no search: every negative cell in the battery is an override
budget spent on a coin flip. So the playoff must clear a bar before it acts —
``|mean(d)| ≥ 2·SE(d)`` on the paired difference — and otherwise plays the POLICY's own action and
counts ``playoff_inconclusive``. A high inconclusive rate is not a failure of the instrument; it is
the instrument saying the rollout budget did not resolve the pair, which is a reportable finding.

Three counted outcomes, and a cell is read on all three together:

* ``screen_decisive`` — the screen already agreed with the policy by more than the measured leaf
  noise, so there was nothing for a playoff to overturn and no rollout was spent. Not a fallback:
  the search decided, and it decided to keep the policy's move.
* ``played`` — the pair was resolved; the winner is played (which may or may not be the policy's).
* ``inconclusive`` / ``no_budget`` — the policy's action, counted, never silent.

**The scoring rule is inherited, not re-derived.** ``rollout_score`` delegates to
``cf_producer.rollout_outcome_score`` (``gen3_cf_draw_at_cap_v1``): win 1.0, loss 0.0, tie 0.5, and
a line that reached the 250-turn stall-forfeit CAP is 0.5 **whatever ``outcome`` says**, because at
the cap both sides forfeit and the winner is decided by which ``FORCELOSE`` the sim processes
first — a systematic 0 against p1, not a coin flip. Sharing the function rather than copying it is
deliberate: this is precisely the constant that a second hand-written copy would get wrong.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: R — paired rollouts per playoff. 12 is the registration's number; the REALIZED count is what
#: gets reported, because the per-decision deadline may buy fewer and a plan is not a measurement.
DEFAULT_ROLLOUTS = 12

#: The screen's own noise bar, on the win-prob scale. 2 x the measured per-leaf ``sd = 0.0115``
#: (``wang_search_reconciliation`` §4). A screen margin WIDER than this, in the policy's own
#: favour, is treated as settled and spends no rollouts; anything narrower is exactly the blurry
#: comparison this arm exists to replace.
DEFAULT_SCREEN_MARGIN = 0.023

#: How many standard errors the paired difference must clear before the playoff may override.
SE_MULTIPLE = 2.0

#: The playoff needs at least this many PAIRS before it is allowed to conclude anything. Without
#: it a 2-pair sweep that happened to agree twice reports ``sd = 0`` and therefore ``SE = 0``, and
#: an infinitely-confident verdict off two draws is the exact failure the 2·SE gate exists to stop.
MIN_PAIRS = 4

#: A floor on the paired SE, in units of the difference. The per-rollout score is atomic in halves
#: (0, ±0.5, ±1), so a run of identical draws measures ``sd = 0`` while carrying no information
#: that the true spread is zero. ``0.5 / n`` is the resolution the scoring scheme itself has — one
#: half-unit spread over the n draws — and it keeps a degenerate sample from reading as certainty.
SE_FLOOR_UNIT = 0.5

#: The stages a playoff decision can end in. ``screen_decisive`` and ``played`` are DECISIONS (the
#: search chose); the rest hand the decision back to the policy and are counted as fallbacks.
STAGE_SCREEN_DECISIVE = "screen_decisive"
STAGE_PLAYED = "played"
STAGE_INCONCLUSIVE = "inconclusive"
STAGE_NO_BUDGET = "no_budget"
STAGE_ERROR = "error"

#: The fallback reasons this module contributes to ``search.FALLBACK_REASONS``.
FALLBACK_INCONCLUSIVE = "playoff_inconclusive"
FALLBACK_NO_BUDGET = "playoff_no_budget"
FALLBACK_ERROR = "playoff_error"


# ---------------------------------------------------------------------------
# the decision rule — pure, so it is testable without a sim
# ---------------------------------------------------------------------------


def top_two(scores: Dict[int, float]) -> Optional[Tuple[int, int, float]]:
    """``(a1, a2, margin)`` — the two best-scoring actions and the gap between them.

    ``None`` when fewer than two actions were scored, which is not a playoff situation at all (the
    player already declines a root with fewer than two legal tokens; this covers the case where the
    screen managed to score only one arm).

    Ties break on the action INDEX so the pair is a deterministic function of the score dict — two
    processes handed the same screen must nominate the same two candidates, or a rerun is not a
    rerun.
    """
    if not scores or len(scores) < 2:
        return None
    ranked = sorted(scores.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))
    (a1, s1), (a2, s2) = ranked[0], ranked[1]
    return int(a1), int(a2), float(s1) - float(s2)


def screen_is_decisive(a1: int, margin: float, policy_action: int,
                       margin_threshold: float) -> bool:
    """Is the screen already settled in the POLICY's favour by more than the leaf noise?

    Both halves are required and they say different things. ``a1 == policy_action`` means a playoff
    could only ever move us AWAY from the policy — the screen's own top pick is what we would play
    anyway. ``margin > threshold`` means the screen's preference for it is larger than the noise the
    screen is known to carry. Together they say: there is nothing here worth spending a rollout on.

    When ``a1 != policy_action`` the playoff runs no matter how wide the margin, because that is
    precisely the override case, and the whole finding of the week is that a wide screen margin is
    not evidence — it is a biased estimator being confident.
    """
    return int(a1) == int(policy_action) and float(margin) > float(margin_threshold)


def rollout_score(res: dict) -> float:
    """One rollout's score in [0, 1] — win 1.0, loss 0.0, draw 0.5, **draw AT CAP 0.5**.

    Delegated to :func:`agents.training.cf_producer.rollout_outcome_score` rather than reimplemented
    (``gen3_cf_draw_at_cap_v1``, f8eec73). A second copy of this rule is how the capped-line bias
    got shipped the first time: at the 250-turn stall cap both sides forfeit and p1's ``FORCELOSE``
    is always processed first, so ``outcome`` there is a fact about seat order, not about the
    position.
    """
    from agents.training.cf_producer import rollout_outcome_score

    return float(rollout_outcome_score(res))


def paired_stats(diffs: Sequence[float]) -> Tuple[float, float, int]:
    """``(mean, se, n)`` of the paired difference.

    ``se`` uses the SAMPLE standard deviation (``ddof=1``) — the pairs are the independent unit
    here, not the individual rollouts — floored at :data:`SE_FLOOR_UNIT` ``/ n`` so a degenerate
    all-identical sample cannot report certainty. ``n < 2`` returns an infinite SE, which makes
    every downstream gate refuse rather than divide by zero.
    """
    n = len(diffs)
    if n <= 0:
        return 0.0, math.inf, 0
    mean = sum(float(d) for d in diffs) / n
    if n < 2:
        return mean, math.inf, n
    var = sum((float(d) - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return mean, max(se, SE_FLOOR_UNIT / n), n


def is_conclusive(mean: float, se: float, n: int, *, k: float = SE_MULTIPLE,
                  min_pairs: int = MIN_PAIRS) -> bool:
    """May the playoff act on this paired difference?

    Two independent bars, and both exist because of a specific way this could lie:

    * ``n >= min_pairs`` — a sample too small to have a spread cannot certify one;
    * ``|mean| >= k·SE`` — the difference must clear its own noise. The search never overrides on
      noise; that is the single lesson the whole battery has produced.
    """
    if n < int(min_pairs) or not math.isfinite(se):
        return False
    return abs(float(mean)) >= float(k) * float(se)


def decide(a1: int, a2: int, mean: float, se: float, n: int, policy_action: int, *,
           k: float = SE_MULTIPLE, min_pairs: int = MIN_PAIRS) -> Tuple[int, str]:
    """``(action, stage)`` from a finished paired sweep.

    ``mean`` is ``score(a1) − score(a2)``, so a positive difference favours ``a1``. An inconclusive
    sweep returns the POLICY's action — not ``a1``, and not the screen's winner. Returning the
    screen's pick on an inconclusive playoff would smuggle the biased estimator back in through the
    tie-break, which is the one thing this arm is built to avoid.
    """
    if not is_conclusive(mean, se, n, k=k, min_pairs=min_pairs):
        return int(policy_action), STAGE_INCONCLUSIVE
    return (int(a1) if mean > 0 else int(a2)), STAGE_PLAYED


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


@dataclass
class PlayoffConfig:
    rollouts: int = DEFAULT_ROLLOUTS
    screen_margin: float = DEFAULT_SCREEN_MARGIN
    se_multiple: float = SE_MULTIPLE
    min_pairs: int = MIN_PAIRS
    impl: str = "node"
    #: Seeded EWMA of one rollout's measured wall cost, used to decide whether the NEXT pair fits
    #: the remaining budget. Starts pessimistic: over-planning a pair costs the deadline, and a
    #: decision that overruns badly enough trips the LIVE battle's own idle watchdog.
    rollout_cost_s: float = 1.0


@dataclass
class PlayoffResult:
    action: int
    stage: str
    r: int = 0
    margin: float = 0.0
    mean: float = 0.0
    se: float = 0.0
    mean_a: float = 0.0
    mean_b: float = 0.0
    cand: Tuple[int, int] = (0, 0)
    wall_s: float = 0.0
    capped: int = 0
    failed: int = 0
    error: Optional[str] = None

    @property
    def fallback(self) -> Optional[str]:
        return {STAGE_INCONCLUSIVE: FALLBACK_INCONCLUSIVE,
                STAGE_NO_BUDGET: FALLBACK_NO_BUDGET,
                STAGE_ERROR: FALLBACK_ERROR}.get(self.stage)

    def as_dict(self) -> dict:
        d = {"stage": self.stage, "r": int(self.r), "margin": round(float(self.margin), 6),
             "mean": round(float(self.mean), 4), "se": round(float(self.se), 4),
             "mean_a": round(float(self.mean_a), 4), "mean_b": round(float(self.mean_b), 4),
             "cand": [int(self.cand[0]), int(self.cand[1])],
             "wall_s": round(float(self.wall_s), 3),
             "capped": int(self.capped), "failed": int(self.failed)}
        if self.error:
            d["error"] = self.error
        return d


#: The signature the runner drives. Injected so the decision rule can be tested against a scripted
#: sim without a bridge child — the rollout is the expensive, unmockable half and everything above
#: it is arithmetic.
RolloutFn = Callable[..., dict]


class PlayoffRunner:
    """Runs the paired rollouts for one decision. One instance per battle-playing engine.

    It owns no persistent player objects **on purpose**: every rollout builds a fresh trainee /
    opponent pair, the way ``cf_producer`` does. Reusing them would need the scripted-prefix
    override removed after each line (``install_scripted_prefix`` captures ``player.choose_move``
    and would otherwise nest a scripted callable inside the next install) and would accumulate a
    ``_battles`` entry per rollout for the life of the cell — tens of thousands of them. Player
    construction is cheap next to a rollout; the leak and the nesting are not.
    """

    def __init__(self, *, model, mappings, battle_format: str, cfg: PlayoffConfig,
                 tag: str = "", rollout_fn: Optional[RolloutFn] = None,
                 player_factory: Optional[Callable[[object, str, str], object]] = None):
        self.model = model
        self.mappings = mappings
        self.battle_format = battle_format
        self.cfg = cfg
        self.tag = tag
        self._rollout_fn = rollout_fn
        self._player_factory = player_factory
        self._n_players = 0

    # -- the entry point ----------------------------------------------------

    def adjudicate(self, *, scores: Dict[int, float], policy_action: int, our_tokens: Dict[int, str],
                   record, turn: int, deadline, rng) -> PlayoffResult:
        """Screen → (maybe) playoff → an action, with every branch counted.

        ``deadline`` is the SAME per-decision :class:`~main.search_dividend.budget.Deadline` the
        screen ran under: the playoff spends what the screen left, which is what makes ``--budget``
        one number rather than two that can disagree.
        """
        pair = top_two(scores)
        if pair is None:
            return PlayoffResult(int(policy_action), STAGE_NO_BUDGET,
                                 error="fewer than two scored actions")
        a1, a2, margin = pair
        if screen_is_decisive(a1, margin, policy_action, self.cfg.screen_margin):
            return PlayoffResult(int(a1), STAGE_SCREEN_DECISIVE, margin=margin, cand=(a1, a2))
        if a1 not in our_tokens or a2 not in our_tokens:
            # The screen scored an action the live mapper will not produce a sim token for. That is
            # not a legality bug (the tokens came from the real mapper), it is a candidate the
            # rollout cannot be scripted with — decline rather than substitute a token we guessed.
            return PlayoffResult(int(policy_action), STAGE_NO_BUDGET, margin=margin, cand=(a1, a2),
                                 error="candidate has no sim token")

        t0 = time.monotonic()
        sa: List[float] = []
        sb: List[float] = []
        diffs: List[float] = []
        capped = 0
        failed = 0
        err: Optional[str] = None
        for r in range(max(1, int(self.cfg.rollouts))):
            pair_cost = 2.0 * self.cfg.rollout_cost_s
            if r and not deadline.fits(pair_cost):
                break
            # CRN, on BOTH randomness axes. `post_t_seed` pins the sim's post-divergence dice; the
            # torch seed pins the two stochastic players' sampling. Sharing them across the two
            # candidates is the whole reason the difference cancels shared bias — with either axis
            # free, `d_r` would also contain "the two lines drew different dice".
            sim_seed = _mint_seed(rng)
            torch_seed = int(rng.getrandbits(31))
            try:
                ra = self._rollout(record, turn, our_tokens[a1], sim_seed, torch_seed)
                rb = self._rollout(record, turn, our_tokens[a2], sim_seed, torch_seed)
            except Exception as e:                       # noqa: BLE001
                # A failed PAIR contributes nothing — never a half-pair, which would leave the two
                # arms measured under different dice and quietly break the pairing.
                failed += 1
                err = err or f"{type(e).__name__}: {e}"
                continue
            va, vb = rollout_score(ra), rollout_score(rb)
            capped += int(bool(ra.get("capped"))) + int(bool(rb.get("capped")))
            sa.append(va)
            sb.append(vb)
            diffs.append(va - vb)
            elapsed = time.monotonic() - t0
            self.cfg.rollout_cost_s = max(1e-3, 0.7 * self.cfg.rollout_cost_s
                                          + 0.3 * (elapsed / max(1, 2 * len(diffs))))

        wall = time.monotonic() - t0
        mean, se, n = paired_stats(diffs)
        if n == 0:
            return PlayoffResult(int(policy_action), STAGE_ERROR if failed else STAGE_NO_BUDGET,
                                 margin=margin, cand=(a1, a2), wall_s=wall, failed=failed,
                                 error=err or "the deadline bought no rollout pair")
        action, stage = decide(a1, a2, mean, se, n, policy_action,
                               k=self.cfg.se_multiple, min_pairs=self.cfg.min_pairs)
        return PlayoffResult(
            action=action, stage=stage, r=n, margin=margin, mean=mean, se=se,
            mean_a=sum(sa) / len(sa), mean_b=sum(sb) / len(sb), cand=(a1, a2),
            wall_s=wall, capped=capped, failed=failed, error=err)

    # -- the rollout --------------------------------------------------------

    def _rollout(self, record, turn: int, choice: str, sim_seed: str, torch_seed: int) -> dict:
        if self._rollout_fn is not None:
            return self._rollout_fn(record=record, turn=turn, choice=choice,
                                    sim_seed=sim_seed, torch_seed=torch_seed)
        return self._live_rollout(record, turn, choice, sim_seed, torch_seed)

    def _live_rollout(self, record, turn: int, choice: str, sim_seed: str,
                      torch_seed: int) -> dict:
        """Play ONE line to a terminal: replay ``record`` to ``turn``, substitute ``choice``, then
        both sides live on the current net.

        🚨 **The live battle's CHOICE TAP is suspended for the duration by ROOM, not by switch.**
        ``record.install_choice_tap`` patches ``BattleStreamClient._write_choice`` for the whole
        process, so a rollout's commands would otherwise be appended to the LIVE battle's
        reconstruction record — which would make every later search in that battle branch from a
        position that never existed. It cannot be a global on/off flag either: in the mirror the
        UNSEARCHED side commits its own live choice on ``POKE_LOOP`` while this rollout runs (our
        ``choose_move`` awaited an executor, so the loop is free), and a blanket suspend would DROP
        that command from the record. Filtering on the battle room keeps the live command and
        discards the rollout's. See :meth:`LiveRecordBuilder.accepts_room`.
        """
        import torch as th

        from utils.bridge.counterfactual import replay_counterfactual

        trainee = self._make_player(record, record.side_of(record.trainee_username), "T")
        opp_side = "p2" if record.side_of(record.trainee_username) == "p1" else "p1"
        opponent = self._make_player(record, opp_side, "O")
        state = th.random.get_rng_state()
        try:
            th.manual_seed(int(torch_seed))
            return replay_counterfactual(
                record, trainee=trainee, opponent=opponent, divergence_turn=int(turn),
                substitute_choice=choice, post_t_seed=sim_seed, impl=self.cfg.impl)
        finally:
            th.random.set_rng_state(state)

    def _make_player(self, record, side: str, role: str):
        if self._player_factory is not None:
            return self._player_factory(record, side, role)
        from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
        from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

        from agents.inference.player import RLPlayer

        self._n_players += 1
        return RLPlayer(
            model=self.model, team=ConstantTeambuilder(record.packed_team(side)),
            battle_format=self.battle_format,
            server_configuration=LocalhostServerConfiguration, mappings=self.mappings,
            account_configuration=AccountConfiguration(
                f"SDivP{role}{self.tag}{self._n_players % 99999}", "password"),
            max_concurrent_battles=1, start_listening=False,
            # STOCHASTIC at temperature 1 on BOTH sides: the mirror's estimand is "what happens
            # against this network playing its own distribution", and a greedy self-rollout would
            # measure a different, degenerate opponent.
            stochastic=True, temperature=1.0)


def _mint_seed(rng) -> str:
    return f"sodium,{rng.getrandbits(128):032x}"


def fold_playoff(decisions: Sequence[dict]) -> dict:
    """Fold per-decision ``playoff`` blocks into the counters a results row carries.

    ADDITIVE by construction (ladder requirement 3, 87a3f91): a row without any playoff block folds
    to zeros and reads exactly as it always did, so one schema still covers the whole battery.

    SUMS are carried rather than means, so a cell's report can pool many games EXACTLY instead of
    averaging per-game averages over unequal decision counts — the same rule
    ``eval_sharding`` follows (Σwon/Σfinished, never a mean of rates).
    """
    out = {"n_screen_decisive": 0, "n_playoff": 0, "n_playoff_inconclusive": 0,
           "n_playoff_no_budget": 0, "n_playoff_capped": 0, "n_playoff_failed": 0,
           "n_playoff_ran": 0, "playoff_r_total": 0, "playoff_wall_s": 0.0}
    for d in decisions:
        p = d.get("playoff")
        if not p:
            continue
        stage = p.get("stage")
        if stage == STAGE_SCREEN_DECISIVE:
            out["n_screen_decisive"] += 1
        elif stage == STAGE_PLAYED:
            out["n_playoff"] += 1
        elif stage == STAGE_INCONCLUSIVE:
            out["n_playoff_inconclusive"] += 1
        elif stage in (STAGE_NO_BUDGET, STAGE_ERROR):
            out["n_playoff_no_budget"] += 1
        out["n_playoff_capped"] += int(p.get("capped", 0) or 0)
        out["n_playoff_failed"] += int(p.get("failed", 0) or 0)
        if int(p.get("r", 0) or 0):
            out["n_playoff_ran"] += 1
            out["playoff_r_total"] += int(p["r"])
            out["playoff_wall_s"] += float(p.get("wall_s", 0.0) or 0.0)
    out["playoff_wall_s"] = round(out["playoff_wall_s"], 3)
    return out
