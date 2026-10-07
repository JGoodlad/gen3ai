"""THE gen-3 MOVE-ORDER rule — who acts first in a turn — torch-free, ONE declaration for every reader.

Architecture audit F7b (`designs/endstate/design_arch_audit.md` F7; owner 2026-10-06, "F7b IN"): P(we move first)
from PHYSICS instead of the hand-picked logistic ``sigmoid(Δspeed / _DMG_SPEED_SCALE)``. The order rule has two
layers, and both live here so no reader can fork either:

* **the priority bracket** (:func:`p_seat_first`) — the higher-priority action always goes first; only at EQUAL
  priority does speed decide. This was `move_resolution_rules.p_seat_first` (the move-resolution family's and
  `intent_conditional`'s rule); it moved here unchanged so the bracket and the speed physics are one module.
* **within a bracket** (:func:`p_first_same_priority`) — P(our mon acts before theirs at equal priority): the
  speed BELIEF integral with the speed-tie coin flip (:func:`p_outspeed_belief`), then the gen-3 Quick Claw roll
  (:func:`p_first_quick_claw`). Our speed is EXACT (:func:`gen3_speed_stat` → :func:`gen3_stage_speed` →
  :func:`gen3_para_speed`, Showdown's integer arithmetic); theirs is a belief, so its stage / paralysis factor
  scales the belief's mean AND spread (:func:`belief_speed_scale`).

Every rule below is verified in `deps/pokemon-showdown` with the gen-3 inheritance resolved (gen3 → gen4 → gen5 →
… → base; paths relative to `deps/pokemon-showdown/`):

* ORDER — `sim/battle.ts` ``comparePriority``: order, then ``priority``, then ``speed``; ``speedSort`` SHUFFLES
  equal entries (``this.prng.shuffle``), so an exact speed tie is a coin flip.
* SPEED — `data/mods/gen3/scripts.ts` ``getActionSpeed``: ``getStat('spe')``; with the turn's Quick Claw roll and a
  held Quick Claw the speed becomes 65535 (no 13-bit truncation in gen 3).
* QUICK CLAW — `sim/battle.ts` (turn start, ``makeRequest('move')``): ``if (this.gen === 3) this.quickClawRoll =
  this.randomChance(1, 5)`` — ONE roll per turn, SHARED by every holder (`data/mods/gen3/items.ts` quickclaw:
  ``onFractionalPriority: undefined``, the later-gen per-holder roll is not inherited). Two holders on a
  successful roll both read 65535: a tie, a coin flip.
* STAT — `sim/battle.ts` ``statModify``: ``tr(tr(2·base + iv + tr(ev/4))·level/100 + 5)``, then the nature
  ``tr(tr(stat·110, 16)/100)`` / ``·90``. Level 100 (gen3ou), so the level factor is exact.
* STAGES — `sim/pokemon.ts` ``getStat``: ``boostTable = [1, 1.5, 2, 2.5, 3, 3.5, 4]``, ``floor(stat·t[b])`` for a
  boost, ``floor(stat / t[−b])`` for a drop, clamped to ±6 (no gen-3 override of the table).
* PARALYSIS — `data/mods/gen4/conditions.ts` par ``onModifySpe``: ``chainModify(0.25)`` (Quick Feet is a gen-4
  ability, unreachable in gen 3), applied at the end of ``runEvent`` by ``modify(spe, 0.25)`` =
  ``tr((tr(spe·1024) + 2047) / 4096)`` (`sim/battle.ts` ``modify``): it rounds HALF DOWN, not a floor.
* CHOICE BAND does NOT touch speed — `data/items.ts` choiceband has ``onModifyAtk`` only (gen 4's override only
  re-wires the choice lock); Choice Scarf (the speed item) is gen 4.

**QUICK CLAW IS BANNED IN GEN 3 OU** — format-gated, never silently dropped (:func:`quick_claw_live`). The
vendored simulator predates the ban (its `config/formats.ts` gen3ou banlist, submodule `e0551883`, 2026-05-09,
has no Quick Claw); Showdown master's `config/formats.ts` ``[Gen 3] OU`` banlist carries ``'Quick Claw'`` (read
2026-10-07), and the owner confirmed it as current policy the same day. The model plays gen3ou only
(the default of every transport), so the op's Quick Claw term is OFF there — no holder can
exist on a legal team — while :func:`p_first_quick_claw` stays implemented and tested for a format that allows it.

What is NOT modelled (named residuals, not silently wrong): Swift Swim / Chlorophyll (speed ×2 in rain / sun),
Macho Brace (×0.5), and the opponent's speed is a GAUSSIAN over the integer lattice (the belief's mean and
spread), not its exact usage mixture.

Duck-typed over tensors (tensor METHODS only — ``floor``, ``round``, ``erf``, ``clamp``, ``to``), so this module
imports no torch, exactly like the rule tables it serves.
"""
from __future__ import annotations

import math
from typing import Any

#: Gen 3's Quick Claw: ONE roll per turn, ``randomChance(1, 5)`` (`sim/battle.ts`), shared by every holder.
QUICK_CLAW_P = 1.0 / 5.0
#: The speed a successful Quick Claw roll gives its holder (`data/mods/gen3/scripts.ts` ``getActionSpeed``).
QUICK_CLAW_SPEED = 65535.0
#: Paralysis' speed modifier as Showdown applies it: ``chainModify(0.25)`` → a 4096-based modifier of 1024.
PARA_SPEED_MOD_4096 = 1024.0
def quick_claw_live() -> bool:
    """Whether Quick Claw can be held in the format the model plays (gen3ou — every transport's default). FALSE:
    it is BANNED in Gen 3 OU — Showdown master `config/formats.ts` ``[Gen 3] OU`` banlist carries ``'Quick Claw'``
    (read 2026-10-07); owner, 2026-10-07: "Quick Claw is now BANNED in Gen 3 OU". The vendored
    `deps/pokemon-showdown` (`e0551883`, 2026-05-09) predates the ban.

    THE ONE READ of the ban for the speed physics, deliberately minimal: when the declared gen-3 OU FORMAT SPEC
    (`designs/endstate/design_format_spec.md`, being built 2026-10-07) lands, this body becomes a read of its
    banlist — no second banlist mechanism lives here."""
    return False


#: The modes of `--speed-physics` (gen3_speed_physics_v1). 'off' is the logistic, production.
SPEED_PHYSICS_MODES = ("off", "on")

_SQRT2 = math.sqrt(2.0)


# ------------------------------------------------------------------------------------- the priority bracket
def p_seat_first(prio_m: Any, prio_k: Any, p_out: Any) -> Any:
    """P(seat k acts BEFORE our move m) — the gen-3 order rule, ONE declaration for every reader (the move-
    resolution family's `pre`, `intent_conditional`'s Protect): the higher PRIORITY moves first; at equal
    priority the faster mon, ``p_out`` = P(we act first within the bracket) (`sim/battle.ts` ``comparePriority``
    sorts by priority, then speed). All three broadcast."""
    return (prio_k > prio_m).to(p_out.dtype) + (prio_k == prio_m).to(p_out.dtype) * (1.0 - p_out)


# ------------------------------------------------------------------------------------ our EXACT speed stat
def gen3_speed_stat(base: Any, iv: Any, ev: Any, nature_mult: Any) -> Any:
    """Our mon's level-100 Speed STAT, exactly as Showdown's ``statModify`` computes it. ``iv`` / ``ev`` are the
    observation's (de-normalised, so ROUNDED back to their integers first); ``nature_mult`` is 0.9 / 1.0 / 1.1
    (the nature's percentage, rounded, so the float32 product cannot land a hair under an integer)."""
    inner = 2.0 * base + iv.round() + (ev.round() / 4.0).floor() + 5.0
    pct = (nature_mult * 100.0).round()
    return (inner * pct / 100.0).floor()


def gen3_stage_speed(stat: Any, stage: Any) -> Any:
    """``getStat``'s stage step: ``floor(stat · (2+s)/2)`` for a boost, ``floor(stat · 2/(2−s))`` for a drop, the
    stage clamped to ±6 (``stage`` is the observation's, rounded to its integer). Both quotients are of integers,
    so the floor is exact in float."""
    s = stage.round().clamp(-6.0, 6.0)
    up = (stat * (2.0 + s.clamp(min=0.0)) / 2.0).floor()
    down = (stat * 2.0 / (2.0 - s.clamp(max=0.0))).floor()
    pos = (s >= 0).to(stat.dtype)
    return pos * up + (1.0 - pos) * down


def gen3_para_speed(stat: Any, para: Any) -> Any:
    """Paralysis' ``modify(spe, 0.25)`` = ``tr((tr(spe·1024) + 2047) / 4096)`` — round HALF DOWN (302 → 75, 303 → 76),
    applied where ``para`` (0/1) is set."""
    paralysed = ((stat * PARA_SPEED_MOD_4096 + 2047.0) / 4096.0).floor()
    p = (para > 0.5).to(stat.dtype)
    return p * paralysed + (1.0 - p) * stat


def gen3_final_speed(stat: Any, stage: Any, para: Any) -> Any:
    """The order ``getStat`` applies them in: the stage (inside ``getStat``), then ``ModifySpe`` (paralysis)."""
    return gen3_para_speed(gen3_stage_speed(stat, stage), para)


def belief_speed_scale(stage: Any, para: Any) -> Any:
    """The factor a stage and paralysis put on a BELIEVED speed (its mean and its spread scale together): the
    stage multiplier ``(2+s)/2`` / ``2/(2−s)`` times 1/4 under paralysis. A belief is continuous, so the integer
    floors of the exact path have nothing to act on."""
    s = stage.round().clamp(-6.0, 6.0)
    pos = (s >= 0).to(s.dtype)
    mult = pos * (2.0 + s.clamp(min=0.0)) / 2.0 + (1.0 - pos) * 2.0 / (2.0 - s.clamp(max=0.0))
    p = (para > 0.5).to(s.dtype)
    return mult * (1.0 - 0.75 * p)


# ---------------------------------------------------------------------------------- the speed integral
def _phi(z: Any) -> Any:
    """The standard normal CDF."""
    return 0.5 * (1.0 + (z / _SQRT2).erf())


def p_outspeed_belief(ours: Any, mu: Any, sigma: Any) -> Any:
    """P(our EXACT speed ``ours`` beats theirs) + ½ P(a tie), their FINAL speed believed ~ N(``mu``, ``sigma``²) on
    the integer lattice (a speed stat is an integer): with T the believed speed rounded to an integer,

        P(T < s) + ½ P(T = s) = ½ [Φ((s − ½ − μ)/σ) + Φ((s + ½ − μ)/σ)]

    — the tie is a coin flip (`speedSort`'s shuffle), and nothing here is a free constant (the ½ is half a stat
    point, the lattice). ``sigma = 0`` (a point belief) is the exact step ``1[s > μ] + ½·1[s = μ]``; a wider
    spread pulls the probability toward ½. All three broadcast."""
    ok = (sigma > 0).to(mu.dtype)
    safe = sigma * ok + (1.0 - ok)                       # 1 where σ = 0: that branch is discarded below
    smooth = 0.5 * (_phi((ours - 0.5 - mu) / safe) + _phi((ours + 0.5 - mu) / safe))
    step = (ours > mu).to(mu.dtype) + 0.5 * (ours == mu).to(mu.dtype)
    return ok * smooth + (1.0 - ok) * step


def p_first_quick_claw(p_speed: Any, our_qc: Any, opp_qc: Any) -> Any:
    """P(we act first at equal priority) with gen 3's Quick Claw: one roll per turn (prob 1/5) shared by both
    holders. On the roll a holder's speed is 65535 — a lone holder goes first, two holders tie (a coin flip);
    without it, or with no holder, the speed order ``p_speed`` stands. ``our_qc`` / ``opp_qc`` are P(holds it)
    (ours known exactly, theirs a belief)."""
    u, t = our_qc, opp_qc
    on_roll = u * t * 0.5 + u * (1.0 - t) + (1.0 - u) * (1.0 - t) * p_speed
    return (1.0 - QUICK_CLAW_P) * p_speed + QUICK_CLAW_P * on_roll


def p_first_same_priority(ours: Any, mu: Any, sigma: Any, our_qc: Any, opp_qc: Any) -> Any:
    """THE within-bracket rule: the speed integral, then Quick Claw. Every reader of P(we act first at equal
    priority) under `--speed-physics on` goes through this one function."""
    return p_first_quick_claw(p_outspeed_belief(ours, mu, sigma), our_qc, opp_qc)
