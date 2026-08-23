"""Shadow-mode verification for the reward manager's suppressed-term fast path.

**The change this guards** (`gen3_reward_skip_suppressed_v1`). ``process_turn_reward`` used to
compute every BIAS helper and then hand the results to ``_apply_pbrs_suppression``, which zeroes
~20 of them under the production composition — a movedex walk, two effectiveness loops and a
12-mon status scan per decision, all for numbers immediately overwritten with 0.0. The manager
now derives its ACTIVE BIAS set once at construction (activeness is a per-run constant: every
flag ``_bias_term_active`` reads is resume-immutable and value-checked) and skips the pure value
computations the composition forces to zero.

**Why it needs a shadow mode.** This is THE OBJECTIVE. A wrong reward trains a wrong policy with
no error anywhere and no metric that moves — strictly worse than a slow one. Unlike most perf
work, though, the oracle here is trivially cheap: the full computation is one function call, so a
second manager with the skip disabled can be driven in lockstep and every field of every
breakdown compared, at scale, on real battles.

**How to use it.** ``GEN3AI_REWARD_VERIFY=1`` makes every ``Gen3RewardManager`` build a twin with
the skip off and assert bit-identity each turn (the ``GEN3AI_OBS_VERIFY`` shape). There is no
CLI flag — the skip is an internal swap, and a default branch nobody runs is the untested one.

Gates: ``reward_skip_parity_fuzz_test.py`` (real bridge battles, per-field bit-identity across
three compositions) and ``reward_skip_parity_test.py`` (the derivation + this comparator).
"""

import os
from typing import Iterable, Optional


class RewardVerifyMismatch(AssertionError):
    """The skip-suppressed fast path disagreed with the full computation on some field.

    An ``AssertionError`` because it can only mean a bug in the skip derivation: every skipped
    term is one this run's config forces to 0.0, so a difference means a term is being skipped
    that the composition still charges.
    """


def reward_verify_enabled() -> bool:
    """Is compute-both-and-compare shadow mode on (``GEN3AI_REWARD_VERIFY=1``)?

    Read per construction rather than at import, so a test can flip it with
    ``patch.dict(os.environ)`` and get a verifying manager.
    """
    return os.environ.get("GEN3AI_REWARD_VERIFY", "") == "1"


def compare_breakdowns(fast, full, field_names: Iterable[str],
                       fast_total: float, full_total: float,
                       active_bias: Iterable[str], where: str) -> None:
    """Raise ``RewardVerifyMismatch`` unless two breakdowns agree on every field and on total.

    Compared with ``!=``, deliberately not ``isclose``: this is a BIT-identity claim. A skip that
    moves a reward by 1e-16 is still a skip that changed the objective, and since every skipped
    term is structurally zero there is no legitimate source of a small difference.
    """
    bad = [(n, getattr(fast, n), getattr(full, n)) for n in field_names
           if getattr(fast, n) != getattr(full, n)]
    if not bad and fast_total == full_total:
        return
    detail = "; ".join(f"{n}: fast={a!r} full={b!r}" for n, a, b in bad) or "total only"
    raise RewardVerifyMismatch(
        f"skip-suppressed reward diverged from the full computation at {where} ({detail}); "
        f"fast total={fast_total!r} full total={full_total!r}; "
        f"active BIAS = {sorted(active_bias)}")


def verify_turn(twin, battle, delta, bd, field_names: Iterable[str],
                active_bias: Iterable[str]) -> None:
    """Replay one turn on the full-computation `twin` and demand bit-identity with `bd`.

    The twin is driven in lockstep (`reset` / `record_action` / `process_turn_reward`) by the
    manager, so it carries its OWN cross-turn counters — this comparison therefore covers state
    divergence as well as the turn's arithmetic.
    """
    ref = twin.process_turn_reward(battle, delta)
    compare_breakdowns(bd, twin._last_breakdown, field_names, bd.total, ref, active_bias,
                       f"turn {getattr(battle, 'turn', '?')}")


def build_twin(manager_cls, config, progress_clock, log_level) -> Optional[object]:
    """The full-computation oracle for `manager_cls`, or None when shadow mode is off.

    Same config, same clock (the clock is READ-only for the reward, so sharing it is what
    production does), skip disabled, and no twin of its own — so this cannot recurse.
    """
    if not reward_verify_enabled():
        return None
    return manager_cls(log_level, config=config, progress_clock=progress_clock, _shadow=True)
