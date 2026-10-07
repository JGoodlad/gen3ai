"""THE gen-3 MOVE-ORDER rule — who acts first in a turn — torch-free, ONE declaration for every reader.

Architecture audit F7b (`designs/endstate/design_arch_audit.md` F7; owner 2026-10-06, "F7b IN"): P(we move first)
from PHYSICS instead of the hand-picked logistic ``sigmoid(Δspeed / _DMG_SPEED_SCALE)``. The order rule has two
layers, and both live here so no reader can fork either:

* **the priority bracket** (:func:`p_seat_first`) — the higher-priority action always goes first; only at EQUAL
  priority does speed decide. This was `move_resolution_rules.p_seat_first` (the move-resolution family's and
  `intent_conditional`'s rule); it moved here unchanged so the bracket and the speed physics are one module.
* **within a bracket** (:func:`p_first_same_priority`) — P(our mon acts before theirs at equal priority): the
  DISCRETE speed mixture with the speed-tie coin flip (:func:`p_outspeed_mixture`), then the gen-3 Quick Claw
  roll (:func:`p_first_quick_claw`). Our speed is EXACT (:func:`gen3_speed_stat` → :func:`gen3_stage_speed` →
  :func:`gen3_para_speed`, Showdown's integer arithmetic); theirs is a DISCRETE distribution over the Speed STAT
  (the Smogon spreads mixture, gen3_speed_mixture_v1), and each support point takes the SAME exact stage and
  paralysis arithmetic as ours (:func:`gen3_final_speed`), so no rounding is approximated on either side.

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

**QUICK CLAW IS BANNED IN GEN 3 OU** — format-gated through the FORMAT SPEC (:func:`quick_claw_live` reads `agents.gen3_data.format_spec`), never silently dropped. The
vendored simulator predates the ban (its `config/formats.ts` gen3ou banlist, submodule `e0551883`, 2026-05-09,
has no Quick Claw); Showdown master's `config/formats.ts` ``[Gen 3] OU`` banlist carries ``'Quick Claw'`` (read
2026-10-07), and the owner confirmed it as current policy the same day. The model plays gen3ou only
(the default of every transport), so the op's Quick Claw term is OFF there — no holder can
exist on a legal team — while :func:`p_first_quick_claw` stays implemented and tested for a format that allows it.

What is NOT modelled (named residuals, not silently wrong): Swift Swim / Chlorophyll (speed ×2 in rain / sun),
Macho Brace (×0.5). Their speed is the species' Smogon USAGE mixture: what this battle has revealed about it (an
observed move order) does not condition it, and the learned spread belief is not read here.

Duck-typed over tensors (tensor METHODS only — ``floor``, ``round``, ``clamp``, ``to``, ``sum``, ``gather``), so this module
imports no torch, exactly like the rule tables it serves.
"""
from __future__ import annotations

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

    THE ONE READ of the ban for the speed physics: a read of the gen3ou FORMAT SPEC's banlist
    (`agents.gen3_data.format_spec`, `designs/endstate/design_format_spec.md` §5.4) — no second banlist lives here.
    Read at call time, so a planted spec moves it."""
    from agents.gen3_data import format_spec
    return not format_spec.active().is_banned("item", "quickclaw")


#: The modes of `--speed-physics` (gen3_speed_physics_v1). 'off' is the logistic, production.
SPEED_PHYSICS_MODES = ("off", "on")


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


# ------------------------------------------------------------------------------ the speed MIXTURE
def p_outspeed_mixture(ours: Any, their_final: Any, cum: Any) -> Any:
    """P(our EXACT speed beats theirs) + ½ P(a tie), their FINAL speed a DISCRETE distribution
    (gen3_speed_mixture_v1):

        Σ_v w_v · (1[ours > f(v)] + ½ · 1[ours = f(v)])

    — the tie is a coin flip (`speedSort`'s shuffle); nothing is a free constant. ``their_final`` ``[..., V]`` is
    each support point's FINAL speed f(v), NON-DECREASING along the last axis (the lattice through the exact,
    monotone stage / paralysis arithmetic), and ``cum`` ``[..., V+1]`` the cumulative weights with a leading 0
    (``cum[..., n]`` = the mass of the first n points). ``ours`` ``[..., Q]`` (Q values per row, its leading
    axes ``their_final``'s). Because f is sorted, the mass below ``ours`` is a PREFIX whose length is a count, so
    the sum is two ``gather``s. Returns ``[..., Q]`` in ``ours``'s dtype."""
    below = (their_final.unsqueeze(-2) < ours.unsqueeze(-1)).sum(-1)           # [..., Q]: f(v) <  ours
    at_or_below = (their_final.unsqueeze(-2) <= ours.unsqueeze(-1)).sum(-1)    # [..., Q]: f(v) <= ours
    p = 0.5 * (cum.gather(-1, below) + cum.gather(-1, at_or_below))
    return p.to(ours.dtype)


def p_first_quick_claw(p_speed: Any, our_qc: Any, opp_qc: Any) -> Any:
    """P(we act first at equal priority) with gen 3's Quick Claw: one roll per turn (prob 1/5) shared by both
    holders. On the roll a holder's speed is 65535 — a lone holder goes first, two holders tie (a coin flip);
    without it, or with no holder, the speed order ``p_speed`` stands. ``our_qc`` / ``opp_qc`` are P(holds it)
    (ours known exactly, theirs a belief)."""
    u, t = our_qc, opp_qc
    on_roll = u * t * 0.5 + u * (1.0 - t) + (1.0 - u) * (1.0 - t) * p_speed
    return (1.0 - QUICK_CLAW_P) * p_speed + QUICK_CLAW_P * on_roll


def p_first_same_priority(ours: Any, their_final: Any, cum: Any, our_qc: Any, opp_qc: Any) -> Any:
    """THE within-bracket rule: the speed mixture (:func:`p_outspeed_mixture`), then Quick Claw. Every reader of
    P(we act first at equal priority) under `--speed-physics on` goes through this one function. ``our_qc`` /
    ``opp_qc`` broadcast against the ``[..., Q]`` result."""
    return p_first_quick_claw(p_outspeed_mixture(ours, their_final, cum), our_qc, opp_qc)
