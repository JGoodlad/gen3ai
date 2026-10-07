"""The eval FORENSIC-capture quota and the trace-filename contract — poke-env-free, split out of ``eval_player.py``
(P1 of the poke-env retirement, ``T27``).

``ForensicQuota`` (the per-opponent capture quota an eval cycle enforces and its manifest states), the
``_FORENSIC_*_QUOTA`` defaults, ``forensic_selection_rule`` / ``_rule_for`` (the rule sentence a manifest records),
``trace_filename_stem`` (the filename contract the prober inverts) and ``episode_length_sum``. None of it touches a
poke-env object, but it lived beside ``EvalRLPlayer`` (an ``RLPlayer`` subclass), so importing the quota imported the
whole poke-env package — onto the import path of the trainer's argument parser, ``main.h2h`` and ``main.plateau``.
``eval_player`` (and through it ``eval_callback``) re-exports every public name here, so every historical import
site still resolves.
"""
import math
from typing import NamedTuple

# `gen3_trace_selection_manifest_v1` — the trace-SELECTION contract. Declared ONCE, in a pure
# stdlib module the prober and the scaffolding gauge import too, so the recorder and every
# consumer read one declaration rather than three copies of it.
from agents.training.trace_selection import (
    forensic_selection_rule as _selection_rule_text,
)


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
