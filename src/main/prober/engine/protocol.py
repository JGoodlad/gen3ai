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


#: Protocol tags that PROVE the actor's move DID something, whether or not the timeline can name it.
#: Deliberately broad — the point is not to classify the effect but to know one EXISTS, so a bare
#: "no effect" is never asserted over a log that shows otherwise. Results (`-fail`/`-immune`/`-miss`)
#: are excluded on purpose: those say the move did nothing, and they are read as `result` instead.
_VISIBLE_EFFECT_TAGS = frozenset({
    "-status", "-damage", "-heal", "-sethp", "-boost", "-unboost", "-setboost", "-swapboost",
    "-copyboost", "-invertboost", "-clearboost", "-clearnegativeboost", "-clearallboost",
    "-sidestart", "-sideend", "-start", "-end", "-weather", "-fieldstart", "-fieldend",
    "-item", "-enditem", "-curestatus", "-cureteam", "-transform", "-mustrecharge",
    "-singleturn", "-singlemove", "-activate", "-crit", "-supereffective", "-resisted",
    "-hitcount", "-formechange", "-ohko", "-mega", "-primal", "-burst", "-zpower", "faint",
    "swap", "detailschange", "replace",
    # A two-turn move's CHARGE turn (Solar Beam / Fly / Dig) emits only `-prepare`: the move did
    # exactly what it was supposed to, and calling that "no effect" is the same error in a new place.
    "-prepare", "-anim", "-ability", "-endability", "-fieldactivate",
})
#: A `-status` line's status code → the timeline's display spelling (the recorder's own vocabulary,
#: `battle_recorder._mon_display_status`), so a protocol-sourced status reads identically to a
#: recorded one. Toxic's counter is not on the line, so it renders bare — a counter we do not have
#: is not one to invent.
_PROTOCOL_STATUS_DISPLAY = {"brn": "BRN", "par": "PAR", "slp": "SLP", "frz": "FRZ",
                            "psn": "PSN", "tox": "TOX"}


def protocol_move_effects(lines: "tuple[str, ...] | None",
                          actor: str, other: str) -> "dict | None":
    """Everything the LOG says the ACTOR's own move did this turn, or ``None`` when the move's
    window cannot be located (no window ⇒ no evidence, which is a different answer from "nothing").

    ``{"result": "immune"|"missed"|"failed"|None, "status": (name, "PAR", self_targeted)|None,
       "effects": tuple[str, ...]}`` — ``effects`` are the raw tags that PROVE something happened.

    A status carries ``self_targeted`` (Rest/Toxic Orb land on the actor, Thunder Wave on the other
    side) read from the ``pNa:`` player tag of the actor's own ``|move|`` line, NOT from the name:
    this pool's teams carry LOCALIZED nicknames (``Leuphorie`` = Blissey), so a caller must be able
    to say WHICH SIDE was hit and then use its own species spelling. ``name`` is the raw nickname,
    a last-resort fallback only.

    The superset of :func:`protocol_move_result`, and it exists because "no effect" was being
    ASSERTED over a log that said otherwise. Measured on `ai_v12_02_winprob_critic`
    `step_50000016/sentinel_0/loss_s0_002` turn 1: the opponent pivoted Suicune → Cloyster, our
    Thunder Wave resolved against the SWITCH-IN and paralyzed it (``|-status|p2a: Cloyster|par``
    right there in the log, and the next decision's obs carries the PAR bit) — and the timeline read
    ``we thunderwave — no effect``. The recorded ``events`` list could not have saved it: the
    recorder only ever wrote a status event for the mon that was ACTIVE WHEN THE DECISION WAS MADE,
    so the arrival's paralysis reached no trace ever written (fixed the same day in
    `battle_recorder._append_status_events`, which does nothing for the traces already on disk).

    Windowing is :func:`protocol_move_result`'s, tightened: the scan starts at the actor's own
    ``|move|`` line and stops at the next one, at the turn's blank separator (the residual phase
    begins there — a Leftovers heal is not the move's doing), or at ``|upkeep``/``|turn|``. Sides
    are matched by nickname exactly as its siblings do, and a mirror returns ``None``.
    """
    a, o = _norm_species(actor), _norm_species(other)
    if not a or not o or a == o:
        return None

    def _nick(field: str) -> str:
        return _norm_species(field.split(":", 1)[1] if ":" in field else field)

    def _player(field: str) -> str:
        return field.split(":", 1)[0].strip() if ":" in field else ""

    started = False
    actor_tag = ""
    result: "str | None" = None
    status: "tuple[str, str, bool] | None" = None
    effects: "list[str]" = []
    for ln in lines or ():
        parts = ln.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag == "move" and len(parts) > 2:
            if started:
                break                          # the next move — out of this move's window
            started = _nick(parts[2]) == a
            if started:
                actor_tag = _player(parts[2])  # "p1a" — which SIDE this move came from
            continue
        if not started:
            continue
        if tag in ("", "turn", "upkeep"):
            break                              # the action block ended; residuals are not the move
        if tag == "-immune":
            result = result or "immune"
        elif tag == "-miss" and len(parts) > 2 and _nick(parts[2]) == a:
            result = result or "missed"
        elif tag == "-fail":
            result = result or "failed"
        elif tag == "-status" and len(parts) > 3:
            code = parts[3].strip().lower()
            if status is None and code in _PROTOCOL_STATUS_DISPLAY:
                status = (_nick(parts[2]), _PROTOCOL_STATUS_DISPLAY[code],
                          bool(actor_tag) and _player(parts[2]) == actor_tag)
        if tag in _VISIBLE_EFFECT_TAGS:
            effects.append(tag)
    if not started:
        return None
    return {"result": result, "status": status, "effects": tuple(effects)}
