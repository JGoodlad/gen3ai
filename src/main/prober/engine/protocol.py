"""Reading the RAW Showdown protocol out of a trace's `*_replay.html` sibling.

These are the readers that answer a question from the log itself rather than inferring it: who
moved first, whether a side moved at all, and whether its move was immune / missed. They exist
because a model-free trace has no decoded TurnDelta, so every such fact would otherwise be a
guess (measured: 9.8% of move lines were mis-explained without them).
"""

from __future__ import annotations

import re

from main.prober.engine.util import _norm_species


_LOG_BLOCK_RE = re.compile(r'class="battle-log-data"[^>]*>(.*?)</script>', re.DOTALL)


def parse_protocol_log(html: str) -> "tuple[str, ...]":
    """Extract the raw Showdown protocol lines (``|move|…`` / ``|-damage|…`` / ``|turn|N``) from a
    ``replay.html``'s ``battle-log-data`` script block — the same browser-watchable log, surfaced in
    the prober so each decision's raw events are visible. Returns the ``|``-lines in order (the whole
    body if the expected block isn't found, so a format tweak degrades rather than blanks)."""
    m = _LOG_BLOCK_RE.search(html or "")
    body = m.group(1) if m else (html or "")
    return tuple(ln.rstrip("\r\n") for ln in body.splitlines() if ln.startswith("|"))


def protocol_for_turn(lines: "tuple[str, ...]", turn: int) -> "tuple[str, ...]":
    """The protocol slice for one decision's turn: every line from ``|turn|N`` up to (not incl.)
    ``|turn|N+1``. Pure — pairs `parse_protocol_log` (the caller does the file IO) with a decision's
    ``turn`` so the raw events between this decision and the next are shown in order."""
    out, cur = [], 0
    for ln in lines or ():
        if ln.startswith("|turn|"):
            try:
                n = int(ln.split("|")[2])
            except (IndexError, ValueError):
                n = cur
            if n > turn:
                break
            cur = n
        if cur == turn:
            out.append(ln)
    return tuple(out)


def move_order_from_protocol(lines: "tuple[str, ...] | None",
                             our_active: str, opp_active: str) -> "str | None":
    """Who moved FIRST this turn, read off the raw Showdown log: ``"we_first"`` / ``"opp_first"``,
    or ``None`` when it cannot be established beyond doubt.

    The TurnDelta records `move_order`, but only a decision captured WITH state carries one — on a
    model-free / older trace it is absent, and the timeline then has to drop the implied sequence
    rather than guess. The protocol slice for that same turn does not have that gap: the sim emits
    ``|move|`` lines IN EXECUTION ORDER, which is the fact itself rather than an inference from it.

        |move|p1a: Jirachi|Protect|p1a: Jirachi
        |move|p2a: Raikou|Thunderbolt|p1a: Jirachi

    Sides are identified by matching the actor's NICKNAME to the two active species, not by
    assuming we are `p1`. Our teams are packed without nicknames, so Showdown uses the species —
    but a nicknamed mon would not match, and the honest answer there is `None` (keep today's
    behaviour) rather than a 50/50 guess dressed up as a reading.

    Returns `None` when: there are no move lines, the first actor matches neither active, or BOTH
    actives are the same species (a mirror, where the nickname cannot disambiguate the sides).
    """
    ours, opps = _norm_species(our_active), _norm_species(opp_active)
    if not ours or not opps or ours == opps:
        return None                      # a mirror: the nickname names both sides equally
    for ln in lines or ():
        if not ln.startswith("|move|"):
            continue
        parts = ln.split("|")
        if len(parts) < 3:
            continue
        actor = parts[2]                 # e.g. "p1a: Jirachi"
        nick = _norm_species(actor.split(":", 1)[1] if ":" in actor else actor)
        if not nick:
            continue
        if nick == ours:
            return "we_first"
        if nick == opps:
            return "opp_first"
        return None                      # a nickname we cannot place — do not guess from later lines
    return None


def protocol_action_fate(lines: "tuple[str, ...] | None",
                         actor: str, other: str) -> "str | None":
    """What the log says a side's active actually DID this turn: ``"moved"``, ``"cant:<reason>"``,
    ``"absent"`` (chose a move, never got to act), or ``None`` when it cannot be established.

    The recorded action says what a side CHOSE. Nothing in a model-free trace says whether the
    choice ever executed, so the timeline assumed it did and explained the missing damage as an
    ineffective move — which reads as a claim about the move rather than about the turn. Measured
    on gen-16 `loss_s0_004`: Forretress CHOSE Explosion on turn 6, was outsped by a +2 Tyranitar
    and killed before acting, and the timeline said `we explosion — no effect`. The protocol for
    that turn contains no `|move|` line for our side at all.

    ``"absent"`` is only returned when the OTHER side was positively identified in the same slice —
    that is the guard against a nickname that does not match its species (this pool contains teams
    whose nicknames are LOCALIZED species names). Without that evidence the answer is `None`, and
    the caller keeps today's behaviour rather than inventing a stronger claim from a failed match.
    """
    a, o = _norm_species(actor), _norm_species(other)
    if not a or not o or a == o:
        return None                       # a mirror: the nickname names both sides equally
    other_seen = False
    for ln in lines or ():
        if not (ln.startswith("|move|") or ln.startswith("|cant|")):
            continue
        parts = ln.split("|")
        if len(parts) < 3:
            continue
        who = parts[2]
        nick = _norm_species(who.split(":", 1)[1] if ":" in who else who)
        if nick == a:
            if ln.startswith("|move|"):
                return "moved"
            reason = parts[3].strip() if len(parts) > 3 else ""
            return f"cant:{reason}" if reason else "cant:"
        if nick == o:
            other_seen = True
    return "absent" if other_seen else None


def protocol_move_result(lines: "tuple[str, ...] | None",
                         actor: str, other: str) -> "str | None":
    """Why a move that executed produced nothing — ``"immune"`` / ``"missed"`` — read off the log,
    or ``None`` when it does not say.

    The sibling of :func:`protocol_action_fate`: that one answers "did it act at all", this one
    "what came of it". `_no_effect_reason` can already name both, but it keys on the TurnDelta's
    decoded effectiveness/outcome, which a MODEL-FREE trace does not have — so a genuine immunity
    degraded to a bare "no effect". Measured on gen-16 `loss_s0_004` turn 7: Earthquake into a
    Levitate Gengar, `|-immune|p2a: Gengar` right there in the log, rendered `— no effect`; across
    100 battles, 67 of 400 bare no-effect lines sat on a turn carrying an `|-immune|`.

    Scoped to the actor's OWN move: the scan starts at that `|move|` line and stops at the next
    one (or the turn's end), so the other side's immunity on the same turn cannot be borrowed. A
    `|-miss|` names the ATTACKER first, so it is attributed only when that is this actor.
    """
    a, o = _norm_species(actor), _norm_species(other)
    if not a or not o or a == o:
        return None
    def _nick(field: str) -> str:
        return _norm_species(field.split(":", 1)[1] if ":" in field else field)

    started = False
    for ln in lines or ():
        parts = ln.split("|")
        if ln.startswith("|move|") and len(parts) > 2:
            if started:
                break                      # the next move — out of this move's window
            started = _nick(parts[2]) == a
            continue
        if not started:
            continue
        if ln.startswith(("|turn|", "|upkeep")):
            break
        if ln.startswith("|-immune|"):
            return "immune"
        if ln.startswith("|-miss|") and len(parts) > 2 and _nick(parts[2]) == a:
            return "missed"
    return None
