"""Every OTHER live lever's supply as a DECLARED resource — `gen3_supply_guard_v2` (2026-09-30).

`gen3_supply_guard_v1` closed the cf label producer (deleted with the cf training half, deletion pass
L4): a coefficient that was ON while its supplier was absent trained a whole run on nothing, and every
counter that could have said so was a scalar nobody thresholded. The supply inventory of the same day
found the same SHAPE in more places — a flag that is set, a mechanism that silently delivers nothing,
and a run that then reads as a result about the lever (the ones still live):

=====================  ===========================================  ==================================
lever (``key``)        what silently happened instead               measured victim
=====================  ===========================================  ==================================
``self_play_pool``     `--self-play` with a pool that never seeds:   `ai_v12_27_ladder_ctrl10M_shaped_
                       every episode falls back to the BOT pool      dense` (10M, 0 snapshots),
                                                                     `ai_v6_10_unified_obs_0618`
``pfsp``               `--pfsp-scale` with no sentinel win-rate:     (inventory)
                       the pool sample stays uniform / stale
``fork``               `--fork-fraction` whose arm disables itself   (inventory)
                       or forks nothing, with a print
=====================  ===========================================  ==================================

**One mechanism, two failure classes.**

* A **deterministic mis-wiring** (the fork arm's buffer is not a `ForkRolloutBuffer`, an obs key the
  arm reads is missing) is the same on every restart, so
  it raises :class:`LeverConfigError` → ``FATAL_CONFIG`` (3) the first time it is seen.
* A **dry streak** — the lever is LIVE (it is supposed to be delivering) and delivered ZERO units
  for ``N`` consecutive cycles — raises :class:`LeverStarvedError` → ``FATAL_SUPPLY`` (5). ``N`` is
  DECLARED per lever in :data:`LEVERS` (the cycle UNIT differs: an eval cycle is ~2M steps, a
  rollout ~100k) and overridable with ``--supply-starve-cycles key=N[,key=N]``; ``key=0`` disables
  that lever's FATAL, which is ANNOUNCED at training start and never silent.

Either way the launcher does not restart. And at the end of every segment each guard prints its
totals — ``🚨🚨 [SUPPLY] ZERO …`` when a live lever delivered nothing — so a run that finished
inside the floor still says so.

**What "live" means is the lever's business, not this module's.** A fresh self-play run below the
win-rate gate legitimately has an empty pool; PFSP has nothing to weight until the pool has a
sentinel. The caller passes ``live=False`` for those cycles: they are counted (and reported) but
neither arm nor break a streak.

Nothing here imports torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

from main.exit_codes import FatalConfigError, SupplyStarvedError


def loud(msg: str) -> None:
    """A supply line goes to the run's OWN log (stdout, which the launcher captures into the child
    log) AND, under the launcher, to its event stream. `main.launcher.ipc.emit` alone does the
    second only — so a self-play run's "pool is EMPTY" warning was invisible in the run directory
    (audit 2026-09-30, finding F5): a supply verdict must be readable from the run itself."""
    print(msg, flush=True)
    try:
        from main.launcher.ipc import _get_pipe, send_event
        if _get_pipe() is not None:
            send_event(msg)
    except Exception:  # noqa: BLE001 — the event stream is a convenience; the log line is the record
        pass


def record_scalar(callback, key: str, value: float) -> None:
    """``callback.logger.record`` when the model HAS a logger (every real SB3 model does; the
    unit-test fakes that drive one callback hook in isolation do not)."""
    logger = getattr(getattr(callback, "model", None), "logger", None)
    if logger is not None:
        logger.record(key, float(value))


class LeverStarvedError(SupplyStarvedError):
    """A LIVE lever delivered nothing for its declared floor → ``FATAL_SUPPLY`` (5)."""


class LeverConfigError(FatalConfigError):
    """A lever that is mis-wired the same way on every restart → ``FATAL_CONFIG`` (3)."""


@dataclass(frozen=True)
class Lever:
    key: str            # the --supply-starve-cycles key
    flag: str           # the flag that turns the lever on
    unit: str           # one observation cycle, in words
    delivers: str       # what one delivered unit is
    fallback: str       # what silently happens when it delivers nothing
    default_cycles: int


#: THE DECLARED FLOORS. Sized so a healthy run never trips them and a dead supply is caught inside a
#: fraction of a short arm: an eval cycle is 2M steps by default, so 3 is 6M of a self-play run with
#: no self-opponent — every v9+ fresh run in the archive seeded at its FIRST cycle (2M); a rollout is
#: ~100k steps at the production shape, so 5 dry rollouts is ~0.5M.
LEVERS: Dict[str, Lever] = {lv.key: lv for lv in (
    Lever("self_play_pool", "--self-play", "eval cycle", "self-play opponent (a seeded pool)",
          "every training episode falls back to the BOT pool — no self-opponent is ever played",
          3),
    Lever("pfsp", "--pfsp-scale", "eval cycle", "a measured sentinel win-rate",
          "PFSP has nothing to weight — the pool sample stays uniform (or frozen at stale rates)",
          3),
    Lever("fork", "--fork-fraction", "rollout", "an injected fork row",
          "the buffer is exactly what collection made it — the arm forks nothing", 5),
)}


def parse_starve_overrides(spec: Optional[str]) -> Dict[str, int]:
    """``"self_play_pool=5,fork=0"`` → ``{"self_play_pool": 5, "fork": 0}``. Raises ValueError on an
    unknown key, a token that is not ``key=N``, or a negative N — the parser turns that into an
    argparse error, so a typo cannot silently leave the default in force."""
    out: Dict[str, int] = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        key, sep, val = tok.partition("=")
        key = key.strip()
        if not sep or key not in LEVERS:
            raise ValueError(f"--supply-starve-cycles token {tok!r}: expected key=N with key in "
                             f"{sorted(LEVERS)}")
        try:
            n = int(val)
        except ValueError:
            raise ValueError(f"--supply-starve-cycles token {tok!r}: N must be an integer") from None
        if n < 0:
            raise ValueError(f"--supply-starve-cycles token {tok!r}: N must be >= 0 (0 = off)")
        out[key] = n
    return out


def starve_cycles_for(args, key: str) -> int:
    """The floor in force for ``key`` on this run: the override if typed, else the declared one."""
    try:
        overrides = parse_starve_overrides(getattr(args, "supply_starve_cycles", None))
    except ValueError:
        overrides = {}
    return int(overrides.get(key, LEVERS[key].default_cycles))


class DryStreakGuard:
    """Count consecutive LIVE cycles in which a lever delivered nothing; raise at the floor.

    Pure state machine. :meth:`observe` once per cycle. The counters can be seeded (and read back
    through :meth:`state`) so a lever whose cycle is longer than a launcher segment — an eval cycle
    is 2M steps — keeps a RUN-level streak across restarts instead of resetting every segment, which
    would let a floor of 3 never trip on a run that restarts every 2 cycles."""

    def __init__(self, key: str, starve_cycles: Optional[int] = None, *,
                 state: Optional[Mapping[str, int]] = None,
                 emit: Callable[[str], None] = loud) -> None:
        self.lever = LEVERS[key]
        self.starve_cycles = (self.lever.default_cycles if starve_cycles is None
                              else int(starve_cycles))
        st = dict(state or {})
        self.streak = int(st.get("streak", 0))       # consecutive LIVE dry cycles
        self.cycles = int(st.get("cycles", 0))       # every observed cycle
        self.live_cycles = int(st.get("live_cycles", 0))
        self.dry_cycles = int(st.get("dry_cycles", 0))
        self.total = int(st.get("total", 0))         # units delivered
        self._emit = emit
        self.last_why = ""

    @property
    def enabled(self) -> bool:
        return self.starve_cycles > 0

    def state(self) -> Dict[str, int]:
        return {"streak": self.streak, "cycles": self.cycles, "live_cycles": self.live_cycles,
                "dry_cycles": self.dry_cycles, "total": self.total}

    def announce(self) -> str:
        lv = self.lever
        if not self.enabled:
            return (f"⚠️  [SUPPLY] {lv.flag}: the in-flight supply guard is DISABLED "
                    f"(--supply-starve-cycles {lv.key}=0) — if it delivers nothing, {lv.fallback}, "
                    f"and this run will NOT stop.")
        return (f"🛡️  [SUPPLY] {lv.flag}: must deliver {lv.delivers} within {self.starve_cycles} "
                f"consecutive live {lv.unit}s, else FATAL_SUPPLY (5)")

    def observe(self, delivered: int, *, live: bool = True, why: str = "",
                may_raise: bool = True) -> None:
        """One cycle. ``delivered`` = units this cycle; ``live`` = was the lever SUPPOSED to deliver;
        ``why`` = the cause, carried into the FATAL; ``may_raise=False`` records without raising
        (a graceful-shutdown drain must not turn a completed run into a FATAL)."""
        self.cycles += 1
        delivered = max(0, int(delivered))
        self.total += delivered
        if not live:
            return
        self.live_cycles += 1
        if delivered > 0:
            self.streak = 0
            return
        self.streak += 1
        self.dry_cycles += 1
        self.last_why = why
        lv = self.lever
        self._emit(f"⚠️  [SUPPLY] {lv.flag}: {lv.unit} delivered NO {lv.delivers} "
                   f"({self.streak} consecutive"
                   + (f" of {self.starve_cycles} allowed" if self.enabled else ", guard disabled")
                   + f"){': ' + why if why else ''}")
        if may_raise and self.enabled and self.streak >= self.starve_cycles:
            raise LeverStarvedError(self.fatal_message())

    def fatal_message(self) -> str:
        lv = self.lever
        return (f"\n[SUPPLY] FATAL: {lv.flag} is live but delivered NO {lv.delivers} for "
                f"{self.streak} consecutive {lv.unit}s (floor --supply-starve-cycles "
                f"{lv.key}={self.starve_cycles}; {self.total} delivered over {self.live_cycles} "
                f"live of {self.cycles} observed).\n"
                f"  Cause: {self.last_why or 'not reported'}.\n"
                f"  Meanwhile {lv.fallback}.\n"
                f"  Training on would make this run read as a result about a lever that never "
                f"engaged. Not restarting.")

    def summary_lines(self) -> List[str]:
        """The end-of-segment line — LOUD when the lever was live and delivered nothing."""
        lv = self.lever
        if self.cycles == 0:
            return [f"🚨 [SUPPLY] {lv.flag}: NO {lv.unit} was observed this segment — nothing "
                    f"here says the lever ever ran."]
        if self.total == 0:
            return [f"🚨🚨 [SUPPLY] ZERO {lv.delivers} delivered by {lv.flag} over {self.cycles} "
                    f"{lv.unit}(s) ({self.live_cycles} live, {self.dry_cycles} dry). {lv.fallback}. "
                    f"This segment is NOT evidence about that lever."]
        return [f"🏭 [SUPPLY] {lv.flag}: {self.total} × {lv.delivers} over {self.cycles} "
                f"{lv.unit}(s) ({self.dry_cycles} live-dry)"]
