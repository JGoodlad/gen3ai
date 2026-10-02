"""The eval PLAYER and its forensic capture quota — split out of ``eval_callback.py`` (2026-10-01).

``EvalRLPlayer`` (the greedy trainee every Python eval path pilots, with per-battle reward tracking and
the bounded forensic-trace capture), the ``ForensicQuota`` it enforces, the trace-filename contract the
prober inverts, and ``episode_length_sum``. ``eval_callback`` re-exports every public name here, so the
historical import site still resolves; a test that STUBS a name this module READS (``write_battle_record``,
``register_trace_prefix``) must patch it HERE (``src/test_stub_vacuity_gate_test.py``).
"""
import math
import os
from typing import NamedTuple

from agents.inference.player import RLPlayer
from agents.training.reward_tracker import RewardTrackingMixin
from agents.training.battle_recorder import BattleRecorder, write_battle_record
from agents.training.eval_sharding import td_tail
from utils.bridge.reconstruction import register_trace_prefix

# `gen3_trace_selection_manifest_v1` — the trace-SELECTION contract. Declared ONCE, in a pure
# stdlib module the prober and the scaffolding gauge import too, so the recorder and every
# consumer read one declaration rather than three copies of it.
from agents.training.trace_selection import (  # noqa: E402
    forensic_selection_rule as _selection_rule_text,
)
# `gen3_trace_result_v2` — the RESULT vocabulary (WIN / LOSS / DRAW, a DRAW naming tie vs
# timeout). Pure stdlib, shared with the recorder and every reader.
from agents.training.trace_result import (  # noqa: E402
    DRAW, LOSS, WIN, classify_result, outcome_prefix,
)
# The forfeit deadline (== MAX_TURNS, `gen3_deadline_clock_v1`) — the SAME number the training
# reward reads to decide a terminal pays `draw_penalty`. Imported, never re-typed.
from agents.training.reward_weights import _TIMEOUT_TURN_CAP  # noqa: E402


def trace_filename_stem(outcome: str, trace_tag: str, idx: int) -> str:
    """The forensic-trace filename stem ``<outcome>_<trace_tag><idx>`` (no suffix).

    THE single source of the naming contract the prober's `discovery._FNAME_RE`
    must invert. ``trace_tag`` is the work-stealing eval's per-shard namespace
    (``""`` un-sharded, or ``f"s{shard}_"`` from `eval_worker`). When sharding was
    added this stem grew an ``s<shard>_`` infix the prober's regex didn't match, so
    every sharded-eval trace parsed as outcome ``"?"`` and the WHOLE prober went
    blind — `eval_callback_test.test_trace_naming_contract` now pins that
    `discovery` parses exactly what this returns, so the two can't drift again."""
    return f"{outcome}_{trace_tag}{idx:03d}"

# Forensic-trace sample caps per opponent per eval cycle. Once every bucket is filled the
# remaining battles run the cheap fast path and only feed the win-rate count.
_FORENSIC_LOSS_QUOTA = 10
_FORENSIC_WIN_QUOTA = 5
# 🚨 WHERE DRAWS SIT IN THE QUOTA: their OWN bucket, independent of both others.
#
# The two rejected alternatives, and why. (a) Fold draws into the LOSS quota — that is what the
# code did until 2026-09-07, and it means a stall storm evicts the decisive losses the prober
# exists to study; the loss slice would silently change meaning in exactly the cycles where a
# regression is worth reading. (b) Give draws no quota — that was the OTHER half of the old
# behaviour (a true tie matched no branch and its capture was dropped), and it is how "0 draws in
# every eval trace" became a number nobody could question.
#
# An independent bucket keeps the loss slice's meaning FIXED across cycles and makes the draw rate
# readable off the tree. It is set to the win quota rather than the loss quota because a draw is
# not loss forensics — it is a rate to notice, not a game to dissect — and because at the observed
# draw frequency (0 persisted in 145k archived traces) the bucket costs nothing in the normal case
# and bounds a pathological stall cycle at 5 extra traces per opponent.
_FORENSIC_DRAW_QUOTA = 5


class ForensicQuota(NamedTuple):
    """One eval cycle's per-opponent capture quota — the THREE numbers, carried together.

    They travel as a unit because they are only ever meaningful together: the manifest states all
    three as the selection RULE, the recorder enforces all three, and a consumer reweights by all
    three. Threading them as three loose ints through the worker boundary is how the manifest's
    stated rule and the recorder's actual behaviour drift apart — the exact defect
    `trace_selection` exists to close.

    🚨 **THE DEFAULT IS THE SHIPPED CONTROL'S QUOTA AND MUST NOT MOVE** (`gen3_forensic_quota_flag_v1`,
    2026-09-08). `ai_v12_11_ladder_ctrl10M` — the win-prob ladder's registered control — ran to
    completion at 5/10/5, so a change of default would put every later arm on a different capture
    regime than the control it is differenced against. The flags exist so an arm RAISES it
    explicitly and the raise is recorded in the manifest; they do not exist to retune the default.
    """

    win: int = _FORENSIC_WIN_QUOTA
    loss: int = _FORENSIC_LOSS_QUOTA
    draw: int = _FORENSIC_DRAW_QUOTA

    @classmethod
    def coerce(cls, value: "ForensicQuota | dict | None") -> "ForensicQuota":
        """A quota from a cfg dict, an existing quota, or the default for ``None``.

        ``None`` means "the caller said nothing", which is the DEFAULT — never zero. A zero quota
        is a legal, meaningful setting (capture nothing of that outcome, as the search teacher
        uses), so it can never be the fallback for an absent one.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(win=int(value["win"]), loss=int(value["loss"]), draw=int(value["draw"]))

    def clamped(self) -> "ForensicQuota":
        """Negatives folded to 0 — a negative quota is not a smaller one, it is nonsense."""
        return ForensicQuota(max(0, int(self.win)), max(0, int(self.loss)), max(0, int(self.draw)))

    def per_shard(self, n_shards: int) -> "ForensicQuota":
        """This quota divided across ``n_shards`` work-steal units.

        ⚠️ **THE PER-SHARD FLOOR OVERSHOOTS THE GLOBAL QUOTA AND ALWAYS HAS.** Each unit gets
        ``max(1, ceil(q / n_shards))``, so at the 5/10/5 default over 4 shards the cycle can write
        up to 8/12/8 rather than 5/10/5 — which is why a measured trace count sits ABOVE the
        stated quota (observed: 32-40 traces per opponent on `ai_v12_11_ladder_ctrl10M` against a
        nominal 20). The floor of 1 is deliberate: a shard that may capture nothing turns a
        low-quota opponent into a silent hole. The manifest states the GLOBAL quota, which is what
        a reweighting consumer needs; this is the enforcement detail beneath it.

        A quota of exactly 0 stays 0 — "capture none" must survive the division intact, or the
        floor of 1 would silently turn it into "capture one per shard".
        """
        n = max(1, int(n_shards))
        return ForensicQuota(*(0 if q <= 0 else max(1, math.ceil(q / n))
                               for q in self.clamped()))


def forensic_selection_rule(win_quota: int = _FORENSIC_WIN_QUOTA,
                            loss_quota: int = _FORENSIC_LOSS_QUOTA,
                            draw_quota: int = _FORENSIC_DRAW_QUOTA) -> str:
    """This recorder's rule in words, bound to ITS OWN quota constants.

    The sentence lives in `trace_selection`; the DEFAULTS live here, beside the quotas they
    describe — so a change to `_FORENSIC_*_QUOTA` cannot leave the recorded rule saying the old
    numbers.
    """
    return _selection_rule_text(win_quota=win_quota, loss_quota=loss_quota,
                                draw_quota=draw_quota)


def _rule_for(quota: "ForensicQuota | dict | None") -> str:
    """The rule sentence for a run's CONFIGURED quota (the default when it says nothing)."""
    q = ForensicQuota.coerce(quota)
    return forensic_selection_rule(win_quota=q.win, loss_quota=q.loss, draw_quota=q.draw)


def episode_length_sum(player) -> float:
    """Σ of turn counts over this player's finished battles (the additive numerator a sharded eval
    pools — the parent recovers the mean as Σturns / Σn_finished). Denominator is the FINISHED-battle
    count (poke-env ``n_finished_battles``), distinct from the reward count (``n_reward_episodes``) —
    see ``ShardResult`` / ``aggregate``. Reads ``_battles`` directly (the convention this file + its
    test fakes use)."""
    return float(sum(b.turn for b in player._battles.values() if b.finished))


class EvalRLPlayer(RewardTrackingMixin, RLPlayer):
    """RLPlayer with per-battle reward tracking for eval metrics.

    Optionally captures forensic traces (full per-decision model I/O → JSON +
    .npz, the same format the replay recorder writes) for a bounded sample of
    each cycle's battles: up to `win_quota` wins and `loss_quota` losses. The
    quota is the whole point of the cheap-vs-heavy split — only battles being
    captured pay the aux cost (`predict_values`, probs/`_last_prediction`); once
    both quotas are filled every remaining battle runs the fast path and just
    counts toward the win rate. Call `begin_forensic_cycle(dir, step)` before a
    cycle to (re)arm capture; leave the dir None to disable forensics entirely.
    """

    def __init__(self, *args, reward_fn_factory, gamma: float = 0.99,
                 loss_quota=_FORENSIC_LOSS_QUOTA, win_quota=_FORENSIC_WIN_QUOTA,
                 draw_quota=_FORENSIC_DRAW_QUOTA, **kwargs):
        # reward_fn_factory is REQUIRED (no silent default): eval MUST measure with the run's actual
        # RewardConfig, not a bare Gen3RewardManager() — a default here once silently scored eval with
        # bias_redesign=False (the old anti-spam reward), making the eval reward meaningless. Callers
        # build it from RewardConfig.from_dict(<model_dir>/model_config.json). See build_eval_players.
        super().__init__(*args, **kwargs)
        self._init_reward_tracking(reward_fn_factory)
        self._gamma = float(gamma)
        self._loss_quota = loss_quota
        self._win_quota = win_quota
        self._draw_quota = draw_quota
        self._forensic_dir: str | None = None
        self._forensic_step = 0
        self._trace_tag = ""
        self._wins_kept = 0
        self._losses_kept = 0
        self._draws_kept = 0
        # Every DRAW this player SAW, quota or no quota — the denominator the manifest's
        # `capture_rate_draw` needs. poke-env counts wins and finished battles but has no draw
        # count (a tie leaves `_won` None and a timeout is booked as our forfeit), so unless we
        # count it here the draw rate is unrecoverable from the trace tree — which is the exact
        # absence this whole change exists to close.
        self._draws_seen = 0
        self._trace_idx = 0
        self._recorders: dict[str, BattleRecorder] = {}
        # δ residuals pooled across THIS matchup's captured battles (one EvalRLPlayer per
        # opponent), folded into a tail statistic at collect via td_tail(). Reset each cycle.
        self._td_pool: list[float] = []
        # M5 Lane H's gate (the per-game-seeded Python path): when a dict, every decision appends
        # ``(action, top-2 legal log-prob margin)`` under its battle tag. None (every live eval) = off.
        self.decision_log: "dict | None" = None

    def begin_forensic_cycle(self, forensic_dir: str | None, step: int, *,
                             trace_tag: str = "", win_quota: int | None = None,
                             loss_quota: int | None = None,
                             draw_quota: int | None = None) -> None:
        """Arm (or disable, if dir is None) forensic capture for one eval cycle / shard.

        ``trace_tag`` namespaces this player's persisted trace filenames. Under battle-level
        work-stealing several shard units of the SAME opponent write into the same
        ``eval_traces/step_<N>/<opponent>/`` dir, each from a fresh player whose ``_trace_idx``
        restarts at 0 — so without a per-shard tag the files (``win_001`` …) would collide and
        overwrite. ``win_quota`` / ``loss_quota`` override the per-cycle defaults so a sharded
        opponent's per-unit quota can be scaled down (≈ global ÷ shard_count) to keep the total
        traces per opponent bounded."""
        self._forensic_dir = forensic_dir
        self._forensic_step = step
        self._trace_tag = trace_tag
        if win_quota is not None:
            self._win_quota = win_quota
        if loss_quota is not None:
            self._loss_quota = loss_quota
        if draw_quota is not None:
            self._draw_quota = draw_quota
        self._wins_kept = 0
        self._losses_kept = 0
        self._draws_kept = 0
        self._draws_seen = 0
        self._trace_idx = 0
        self._recorders.clear()
        self._td_pool = []

    def td_tail(self):
        """Lower-tail (CVaR) of this matchup's per-decision critic surprise, or None if no
        captured battles produced residuals this cycle. The eval cycle records it as
        eval/td_resid_tail_vs_<opponent>."""
        return td_tail(self._td_pool)

    @property
    def traces_written(self) -> int:
        """Traces this player PERSISTED this cycle/shard — the selection, as a count.

        The manifest records it beside the battles played so a consumer can read the quota's
        outcome skew off the trace tree instead of inheriting it silently
        (`gen3_trace_selection_manifest_v1`)."""
        return self._wins_kept + self._losses_kept + self._draws_kept

    @property
    def traces_won(self) -> int:
        """How many of :attr:`traces_written` were WINS (≤ the battles won, by construction)."""
        return self._wins_kept

    @property
    def traces_drawn(self) -> int:
        """How many of :attr:`traces_written` were DRAWS — a tie or a 250-turn timeout."""
        return self._draws_kept

    @property
    def draws_seen(self) -> int:
        """Every DRAW this player PLAYED this cycle/shard, whether or not its trace was kept.

        The denominator of `capture_rate_draw`. Counted here because no other layer counts it:
        poke-env's `n_won_battles` / `n_finished_battles` book a tie as neither and a timeout as
        a loss."""
        return self._draws_seen

    def td_residuals(self) -> list[float]:
        """The raw per-decision δ samples pooled this matchup (a COPY). A sharded eval ships these
        from each shard and pools them in the parent before the single CVaR — a CVaR cannot be
        averaged across shards, so the raw samples (not a per-shard tail) are the unit of exchange."""
        return list(self._td_pool)

    @property
    def _quota_open(self) -> bool:
        """Whether either outcome still needs forensic samples this cycle."""
        return self._forensic_dir is not None and (
            self._wins_kept < self._win_quota or self._losses_kept < self._loss_quota
            or self._draws_kept < self._draw_quota
        )

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "EVAL_STALL")
        if forfeit:
            return forfeit
        # Capture a battle in full only if we already started capturing it (so its
        # trace stays whole even if the quota fills mid-battle) or the quota is still
        # open when it begins. Everything else takes the fast path (need_aux=False).
        capturing = battle.battle_tag in self._recorders or self._quota_open
        idx, probs, mask = self._predict_best_action(
            battle, stochastic=False, need_aux=capturing
        )
        if idx is None:
            return self.choose_default_move()
        self._track_reward(battle, idx, mask)
        if self.decision_log is not None:
            ml = getattr(self, "_last_masked_logits", None)
            legal = [float(x) for x, m in zip(ml[0].tolist(), mask) if m] if ml is not None else []
            top = sorted(legal, reverse=True)
            margin = top[0] - top[1] if len(top) > 1 else float("inf")
            # the chosen action's legal log-prob (log-softmax over the legal logits, in float64)
            lse = max(legal) + math.log(sum(math.exp(x - max(legal)) for x in legal)) if legal else 0.0
            chosen = float(ml[0, idx].item()) - lse if ml is not None else float("nan")
            self.decision_log.setdefault(battle.battle_tag, []).append((int(idx), margin, chosen))
        if capturing:
            rec = self._recorders.get(battle.battle_tag)
            if rec is None:
                rec = BattleRecorder(battle.battle_tag, self._reward_fn_factory, gamma=self._gamma)
                self._recorders[battle.battle_tag] = rec
            rec.record(battle, idx, probs, mask, state=getattr(self, "_last_prediction", None))
        return self.action_to_order(idx, battle)

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)  # reward finalize (mixin)
        rec = self._recorders.pop(battle.battle_tag, None)
        if rec is None:
            return
        # Harvest δ from EVERY captured battle (even one whose trace we drop below for quota) —
        # the tail metric wants signal from all the V(s) we paid for, not just the persisted sample.
        self._td_pool.extend(rec.td_residuals())
        # `gen3_trace_result_v2`: WIN / LOSS / DRAW, classified ONCE (here) by the same function
        # the summary writer uses, so the filename prefix and `meta.result` cannot disagree. The
        # ORDER inside `classify_result` is what fixes the old defect — a 250-turn TIMEOUT arrives
        # with `lost=True` and used to be booked as a decisive loss, and a true TIE matched
        # neither branch and was dropped without a file, a count or a trace of having happened.
        result, _draw_kind = classify_result(
            won=battle.won, lost=battle.lost, finished=battle.finished,
            turn=battle.turn, turn_cap=_TIMEOUT_TURN_CAP,
        )
        if result == DRAW:
            # Counted BEFORE the quota test: the draw RATE must not depend on the capture quota.
            self._draws_seen += 1
        # Persist this trace only if its outcome is one we still want a sample of;
        # otherwise drop the buffered capture (we already have enough of that result).
        _kept, _quota = {
            WIN:  (self._wins_kept,   self._win_quota),
            LOSS: (self._losses_kept, self._loss_quota),
            DRAW: (self._draws_kept,  self._draw_quota),
        }[result]
        if _kept >= _quota:
            return
        outcome = outcome_prefix(result)
        self._trace_idx += 1
        # `_trace_tag` namespaces the file so concurrent shard units of the same opponent (each a
        # fresh player with _trace_idx restarting at 0, all writing this one dir) never collide.
        # `trace_filename_stem` is the single source of the naming contract the prober inverts.
        prefix = os.path.join(self._forensic_dir,
                              trace_filename_stem(outcome, self._trace_tag, self._trace_idx))
        write_battle_record(prefix, rec, battle, self._forensic_step)
        # Bridge battles also have a full-information reconstruction record (seed + both
        # teams + command log, captured at the bridge layer). Registering this trace's
        # prefix lets the registry join the two and persist it as the SEPARATE
        # `<prefix>_reconstruction.json` artifact — deliberately not part of the
        # summary/states the obs pipeline reads (the one-sided/omniscient wall).
        # Websocket eval never produces a record, so the registration just evicts.
        register_trace_prefix(battle.battle_tag, prefix,
                              extra={"trainee_username": self.username})
        if result == WIN:
            self._wins_kept += 1
        elif result == LOSS:
            self._losses_kept += 1
        else:
            self._draws_kept += 1
