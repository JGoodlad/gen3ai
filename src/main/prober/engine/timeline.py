"""The RESULT timeline — an ordered, one-line-per-action battle log for one decision.

The recorder stores each side's action + that mon's OWN net HP change, which renders as nonsense
("we icebeam (−72%)") because a mon's HP loss is dealt by the OPPONENT's move. This module
re-attributes each loss to the move that caused it, and explains a move that did nothing visible
instead of leaving it blank.
"""

from __future__ import annotations

import re

from agents import gen3_data

from main.prober.engine.protocol import (move_order_from_protocol, protocol_action_fate,
    protocol_move_result)
from main.prober.engine.util import _loss_pct, _multiplier_meaningful, _norm_species, _pct
from main.prober.engine.views import BoardView


# ── Result timeline (the RESULT panel's data model) ──────────────────────────────────────────────
# The recorder stores each side's action + that mon's OWN net HP change ("hp_delta"). Rendered
# naively ("we icebeam (-72%)") it misreads: a mon's HP loss is dealt by the OPPONENT's move, not its
# own, so the damage shows on the wrong line and move order needs a confusing "«1st»" tag. This builds
# an ordered, one-line-per-action timeline that RE-ATTRIBUTES each mon's HP loss to the opponent's
# move that caused it, Showdown-battle-log style — "opp rockslide did 73% (salamence 100% → 27%)" —
# so the lines read top-to-bottom in execution order.

_SENT_IN = "_sent_in"
_SEP = " → "                       # the " → " the recorder uses to pack a forced replacement
_FAINT_EVENT_RE = re.compile(r"^(our|opp):(.+):fainted$")
_STATUS_EVENT_RE = re.compile(r"^(our|opp):(.+):([A-Z]{3})$")   # our:milotic:PAR, opp:swampert:TOX
_SWITCH_IN_HIT_MAX_HP = 90.0       # a switched-in mon below this clearly took our hit (vs a sand tick)


def _is_attack(move_id: "str | None") -> bool:
    """Whether a recorded move id is a damaging ATTACK — positive-BP, fixed-damage (Seismic Toss /
    Night Shade), variable-power (Return), or Hidden Power (the bare id reads BP 0 but is a real
    attack). Uses the same ``_multiplier_meaningful`` predicate the matchup panel does, so a real hit
    is never dropped from the attribution and a whiffed attack can be flagged 'missed'/'no effect'."""
    return _multiplier_meaningful((move_id or "").lower())


def _no_effect_reason(move_id: "str | None", effectiveness: "str | None",
                      outcome: "str | None" = None) -> "str | None":
    """Why a move that produced NO visible effect did nothing — 'immune' / 'missed' / 'failed' — or
    None for a move whose effect is legitimately invisible here (hazards / heal / boost / Protect),
    which must NOT be flagged. Only an ATTACK (should deal damage) or a status-inflicting move (should
    apply a status) is annotated. ``outcome`` is the RECORDED move fate (gen3_move_outcome_v1:
    'hit'/'miss'/'fail') — preferred when present, so 'missed'/'failed' is a fact; a type immunity
    (a hit that did nothing) reads 'immune' first; only when the outcome wasn't decoded (model-free /
    older trace) do we fall back to inferring a miss from the move's accuracy."""
    mid = (move_id or "").lower()
    md = gen3_data.moves.get(mid)
    if not (_is_attack(mid) or (md and md.status_inflicted)):
        return None
    if effectiveness == "immune":
        return "immune"
    if outcome == "miss":
        return "missed"
    if outcome == "fail":
        return "failed"
    if outcome == "hit":                 # connected but no damage/status landed — not a miss
        return "failed"
    if md and md.accuracy and md.accuracy < 100 and not md.never_miss:   # fallback: no recorded fate
        return "missed"
    return "failed"


def _parse_outcome_action(action: "str | None") -> dict:
    """Structure a recorded action string: a bare move (``icebeam``), a voluntary switch
    (``switched_to:skarmory``), a move that ended in a forced replacement
    (``hiddenpower → metagross_sent_in``), a bare post-faint replacement with no move
    (``claydol_sent_in`` — the mon fainted before acting), or a no-op (``none``/``unknown``)."""
    a = (action or "").strip()
    if a in ("", "none", "unknown", "-"):
        return {"kind": "none"}
    if a.startswith("switched_to:"):
        return {"kind": "switch", "switch_to": a.split(":", 1)[1]}
    if a.endswith(_SENT_IN):
        head, sep, tail = a.partition(_SEP)
        if sep:                                   # "<move> → <X>_sent_in"
            return {"kind": "move", "move": head, "sent_in": tail[:-len(_SENT_IN)]}
        return {"kind": "send_in", "sent_in": a[:-len(_SENT_IN)]}   # bare "<X>_sent_in"
    return {"kind": "move", "move": a}


def opp_voluntary_switch(inv: dict) -> "str | None":
    """The species the OPPONENT VOLUNTARILY switched IN on this decision's turn (model-free, from the
    recorded opp action), else `None`. A voluntary pivot means our move RESOLVED against this switch-in —
    NOT the active mon we computed our damage against — the source of the "computed-vs ≠ resolved-vs"
    confusion when reading a decision (e.g. Earthquake computed vs Snorlax, then resolved into a Levitate
    Claydol the opp pivoted in). A forced post-faint replacement (`_sent_in`) is NOT a voluntary pivot."""
    opp_action = ((inv.get("outcome") or {}).get("opp") or {}).get("action")
    pa = _parse_outcome_action(opp_action)
    return pa.get("switch_to") if pa.get("kind") == "switch" else None


def build_result_timeline(outcome: dict, our_species: str, opp_species: str, phase: str = "",
                          our_hp_before=None, opp_hp_before=None,
                          our_hp_after=None, opp_hp_after=None,
                          move_order_hint: "str | None" = None,
                          our_fate_hint: "str | None" = None,
                          opp_fate_hint: "str | None" = None,
                          our_result_hint: "str | None" = None,
                          opp_result_hint: "str | None" = None) -> "list[dict]":
    """Ordered, one-line-per-action model of what HAPPENED after a decision (the RESULT panel). Pure.

    Re-attributes each side's HP loss to the OPPONENT's move that dealt it, and pairs it with the
    target's before→after HP (``before = after + damage``), so each line reads like a battle log:
    ``{side, kind, move, damage, target, hp_before, hp_after, crit, boost, cant, status, resulting,
    no_effect, never_moved, switch_to, sent_in}``. ``resulting`` marks a hit on a switch-IN where only the after-HP
    is known (the recorded delta can't price it across the switch); ``no_effect`` (``immune`` /
    ``missed`` / ``failed``) explains a move that did NOTHING visible. Ordered by execution: a voluntary
    switch resolves before a move; otherwise
    the TurnDelta ``move_order`` (folded from the real event-log sequence). When BOTH sides moved but
    ``move_order`` is absent (a no-state / model-free decision) the order is unknown — the move entries
    carry ``order_certain=False`` so the renderer drops the implied sequence rather than guessing."""
    out = outcome or {}
    our, opp = out.get("our") or {}, out.get("opp") or {}
    if not (our or opp):
        return []

    events = [str(e) for e in (out.get("events") or [])]
    faints = {(m.group(1), _norm_species(m.group(2)))
              for m in (_FAINT_EVENT_RE.match(e) for e in events) if m}
    statuses = {(m.group(1), _norm_species(m.group(2))): m.group(3)
                for m in (_STATUS_EVENT_RE.match(e) for e in events) if m}
    consumed: set = set()

    pa_our, pa_opp = _parse_outcome_action(our.get("action")), _parse_outcome_action(opp.get("action"))
    # A move lands on the opponent's active — redirected to its switch-IN when the opponent
    # VOLUNTARILY switched (that resolves first); a forced "_sent_in" does NOT redirect (the fainting
    # mon ate the hit, the replacement is a separate line).
    our_target = pa_opp.get("switch_to") if pa_opp["kind"] == "switch" else opp_species
    opp_target = pa_our.get("switch_to") if pa_our["kind"] == "switch" else our_species

    def _move_entry(side, pa, crit, boost, cant, target, delta, before, after, switched_in, eff,
                    fate, actor="", fate_hint=None, result_hint=None):
        e = {"side": side, "kind": "move", "move": pa.get("move", ""), "crit": bool(crit),
             "boost": boost or "", "cant": cant, "target": "", "damage": "", "hp_before": "",
             "hp_after": "", "status": "", "resulting": False, "no_effect": "",
             "never_moved": False, "switch_to": "", "sent_in": ""}
        # WHAT THE LOG SAYS HAPPENED beats what the recorder says was CHOSEN. Without it a move
        # that never executed is explained as one that executed and achieved nothing — a claim
        # about the MOVE rather than about the TURN (see `protocol_action_fate`).
        hint = fate_hint or ""
        if hint.startswith("cant:") and not cant:
            # The `|cant|` reason the recorded outcome did not carry. A model-free trace has no
            # decoded TurnDelta, so `our_cant`/`opp_cant` are empty and a blocked move used to fall
            # through to "— no effect" — MEASURED on the same battle: a FROZEN Forretress read that
            # way, and an 843-battle sweep saw a full-para Surf do the same.
            e["cant"] = hint.split(":", 1)[1] or "unknown"
            return e
        if cant:
            return e
        if hint == "absent":
            # It chose a move and never got to make it. Claimed ONLY when the mon FAINTED this turn:
            # otherwise "no move line for this side" has other explanations (a partial slice, a turn
            # that could not be aligned) and saying nothing is the honest answer.
            actor_side = "our" if side == "we" else "opp"
            if (actor_side, _norm_species(actor)) in faints:
                e["never_moved"] = True
            return e
        recip_side = "opp" if side == "we" else "our"
        key = (recip_side, _norm_species(target))
        fainted = key in faints or str(delta).strip() == "-100%"
        dmg = _loss_pct(delta)
        if _is_attack(pa.get("move")) and (fainted or dmg is not None):
            e["target"] = target
            if fainted:
                consumed.add(key)
                b = dmg if dmg is not None else before
                e["damage"] = f"{dmg:.0f}%" if dmg is not None else ""
                e["hp_before"] = f"{b:.0f}%" if isinstance(b, (int, float)) else ""
                e["hp_after"] = "faint"
            else:
                aft = after if after is not None else max(0.0, (before or 0.0) - dmg)
                e["damage"], e["hp_after"] = f"{dmg:.0f}%", f"{aft:.0f}%"
                e["hp_before"] = f"{min(100.0, aft + dmg):.0f}%"
        elif _is_attack(pa.get("move")) and switched_in and after is not None and after < _SWITCH_IN_HIT_MAX_HP:
            # The opponent VOLUNTARILY switched, so the recorded hp_delta (≈0, it compares the mon
            # that LEFT) can't price the hit on the switch-IN. The next board's HP is still truth —
            # show the RESULTING hp ("→ celebi (now 11%)") rather than dropping the attack entirely.
            e["target"], e["hp_after"], e["resulting"] = target, f"{after:.0f}%", True
        if key in statuses:                          # a status this move applied (own line or alongside dmg)
            e["target"], e["status"] = target, statuses[key]
        # A move that produced NOTHING visible: say WHY (missed / no effect / immune) so a blank line
        # never reads as "data missing". Silent for utility moves (hazards/heal/boost — reason None).
        if not (e["damage"] or e["status"] or e["hp_after"]):
            # The RECORDED fate/effectiveness wins — it comes from the TurnDelta the analysis was
            # built on. The protocol only fills what the recorder did not decode, which on a
            # model-free trace is everything.
            # The RECORDED fate/effectiveness wins OUTRIGHT — it comes from the TurnDelta the
            # analysis was built on, and `_no_effect_reason` ranks immune ABOVE miss, so feeding a
            # protocol immunity alongside a recorded miss would silently overrule the recorder.
            # The protocol fills only when the recorder decoded NOTHING, which on a model-free
            # trace is every decision.
            if eff or fate:
                reason = _no_effect_reason(pa.get("move"), eff, fate)
            else:
                reason = _no_effect_reason(
                    pa.get("move"),
                    "immune" if result_hint == "immune" else None,
                    "miss" if result_hint == "missed" else None)
            # ...UNLESS the target SWITCHED IN and nothing recorded the move's fate. Then the
            # recorded hp_delta compares the mon that LEFT, so it cannot price the hit either way —
            # and `_no_effect_reason`'s last resort is to infer a miss from the move's accuracy,
            # which turns "we have no evidence" into the confident claim "it missed". MEASURED: a
            # Meteor Mash into a Jirachi switch-in read `— missed` while the battle's own protocol
            # log showed -resisted / -damage 84/100 / -boost atk (the switch-in then healed to
            # exactly 90% on Leftovers, so the resulting-HP branch above just missed its threshold).
            # An absent explanation is honest; a wrong one is worse than none.
            if switched_in and not fate and not result_hint and reason == "missed":
                reason = None
            e["no_effect"] = reason or ""
        return e

    def _entry_for(side):
        if side == "we":
            pa, crit, boost, cant = pa_our, out.get("our_crit"), out.get("our_boost", ""), out.get("our_cant")
            actor, target, delta = our_species, our_target, opp.get("hp_delta")
            before, after, eff = _pct(opp_hp_before), _pct(opp_hp_after), out.get("our_effectiveness")
            switched_in, fate = pa_opp["kind"] == "switch", out.get("our_move_outcome")  # our move lands on the opp's switch-IN
        else:
            pa, crit, boost, cant = pa_opp, out.get("opp_crit"), out.get("opp_boost", ""), out.get("opp_cant")
            actor, target, delta = opp_species, opp_target, our.get("hp_delta")
            before, after, eff = _pct(our_hp_before), _pct(our_hp_after), out.get("opp_effectiveness")
            switched_in, fate = pa_our["kind"] == "switch", out.get("opp_move_outcome")
        if pa["kind"] == "none":
            return None
        if pa["kind"] == "switch":
            return {"side": side, "kind": "switch", "actor": actor, "switch_to": pa.get("switch_to", "")}
        if pa["kind"] == "send_in":
            return {"side": side, "kind": "send_in", "sent_in": pa.get("sent_in", "")}
        e = _move_entry(side, pa, crit, boost, cant, target, delta, before, after, switched_in,
                        eff, fate, actor=actor,
                        fate_hint=(our_fate_hint if side == "we" else opp_fate_hint),
                        result_hint=(our_result_hint if side == "we" else opp_result_hint))
        e["actor"] = actor
        return e

    entries: "list[dict]" = []
    if phase == "forced_switch":
        # Our post-faint replacement choice; the opponent doesn't act → just the send-in.
        if pa_our["kind"] == "switch":
            entries.append({"side": "we", "kind": "send_in", "sent_in": pa_our.get("switch_to", "")})
    else:
        we_e, opp_e = _entry_for("we"), _entry_for("opp")
        # Execution order: a voluntary switch precedes a move; else the recorded move_order; else
        # ours. `move_order_hint` is the SAME fact read off the raw protocol log when the TurnDelta
        # did not record one (see `move_order_from_protocol`) — the recorded value still wins,
        # since it comes from the event log the fold was built on.
        recorded_order = out.get("move_order") or move_order_hint
        we_first = True
        if pa_opp["kind"] == "switch" and pa_our["kind"] == "move":
            we_first = False
        elif pa_our["kind"] != "switch" and recorded_order == "opp_first":
            we_first = False
        entries = [e for e in ([we_e, opp_e] if we_first else [opp_e, we_e]) if e is not None]
        # Order certainty: top-to-bottom is REAL only when a voluntary switch fixes it (switches
        # resolve first) or the TurnDelta recorded move_order. If BOTH sides actually moved (a canted
        # side didn't) and move_order is absent — a no-state / model-free decision — we don't know who
        # went first, so flag it and let the renderer drop the implied sequence instead of guessing.
        def _moved(e):
            return bool(e and e.get("kind") == "move" and not e.get("cant"))
        order_certain = (recorded_order in ("we_first", "opp_first")) or not (
            _moved(we_e) and _moved(opp_e))
        for e in entries:
            if e.get("kind") == "move":
                e["order_certain"] = order_certain
        # A move that ended in a forced replacement → surface the send-in as its own trailing line.
        for pa, sd in ((pa_our, "we"), (pa_opp, "opp")):
            if pa["kind"] == "move" and pa.get("sent_in"):
                entries.append({"side": sd, "kind": "send_in", "sent_in": pa["sent_in"]})

    # Any faint NOT already shown as a move's target (self-KO, residual, opp action unknown).
    for sd, sp in faints:
        if (sd, sp) not in consumed:
            entries.append({"side": ("we" if sd == "our" else "opp"), "kind": "faint", "actor": sp})
    return entries


def _timeline_for(inv: dict, next_board: "BoardView | None", outcome: dict,
                  protocol: "tuple[str, ...] | None" = None) -> "list[dict]":
    """`build_result_timeline` wired to a decision: the actives' HP this turn (before) + the resolved
    HP at the next decision (after, from ``next_board``).

    `protocol` is this turn's raw Showdown slice, and it answers two questions the recorded
    outcome cannot: the execution ORDER when no `move_order` was recorded, and whether each side's
    chosen move EXECUTED AT ALL (`protocol_action_fate`) — a mon KO'd before acting, or blocked by
    a `|cant|`, used to render as a move that ran and did nothing."""
    our_sp = inv.get("our", {}).get("species", "")
    opp_sp = inv.get("opp", {}).get("species", "")
    return build_result_timeline(
        outcome, our_sp, opp_sp,
        inv.get("phase", ""),
        our_hp_before=inv.get("our", {}).get("hp"), opp_hp_before=inv.get("opp", {}).get("hp"),
        our_hp_after=(next_board.ours.active_hp if next_board else None),
        opp_hp_after=(next_board.opp.active_hp if next_board else None),
        move_order_hint=move_order_from_protocol(protocol, our_sp, opp_sp),
        # Whether each side's chosen move actually EXECUTED. Same slice, same species matching as
        # the order hint — a recorded action says what was chosen, only the log says what happened.
        our_fate_hint=protocol_action_fate(protocol, our_sp, opp_sp),
        opp_fate_hint=protocol_action_fate(protocol, opp_sp, our_sp),
        our_result_hint=protocol_move_result(protocol, our_sp, opp_sp),
        opp_result_hint=protocol_move_result(protocol, opp_sp, our_sp),
    )


# Plain-language for a "couldn't move" (cant) reason decoded from the TurnDelta, and for a move that
# produced nothing visible. Both live HERE rather than in a renderer: the TUI paints them with Rich
# styles, the web/CLI want the same words as plain text, and two copies of this vocabulary would
# drift the moment one surface learns a new reason.
CANT_PHRASE = {"slp": "asleep", "frz": "frozen", "par": "fully paralyzed", "flinch": "flinched",
               "recharge": "recharging", "nopp": "no PP", "truant": "loafing",
               "attract": "immobilized", "taunt": "taunted", "disable": "disabled",
               "flinched": "flinched",
               # The PROTOCOL's own spellings, which reach this map now that a `|cant|` reason can
               # come straight off the log (`protocol_action_fate`) rather than only from the
               # decoded TurnDelta. Measured over a run's replays, the live set is
               # par/slp/frz/flinch/`Focus Punch`/`move: Taunt`/recharge.
               "focus punch": "lost its focus"}
NO_EFFECT_TEXT = {"immune": "no effect (immune)", "missed": "missed", "failed": "no effect"}


def cant_phrase(cant: str) -> str:
    """Plain language for a couldn't-move reason, from either source: the TurnDelta's decoded code
    (``slp``) or the protocol's own field (``move: Taunt``). The ``move: `` prefix is stripped so
    both spellings land on one entry; an unknown reason is returned as-is rather than dropped."""
    key = str(cant).strip().lower()
    if key.startswith("move: "):
        key = key[len("move: "):]
    return CANT_PHRASE.get(key, str(cant))


def surprise_phrase(td: float) -> str:
    """Plain-language reading of the TD-surprise δ = r + γV(s′) − V(s), so the ML term is
    self-explaining wherever it is shown: negative δ = the turn went worse than the critic predicted.

    Lives HERE, beside `timeline_entry_text` and `CANT_PHRASE`, for the same reason they do — it is
    VOCABULARY, and every surface must say it the same way. (It began as a TUI-local helper; a web
    view re-wording "much worse than the critic expected" is precisely the drift the engine/renderer
    split exists to prevent.)"""
    mag = abs(td)
    if mag < 0.5:
        return "about what the critic expected"
    much = "much " if mag >= 3.0 else ""
    return f"{much}{'better' if td > 0 else 'worse'} than the critic expected"


def timeline_entry_text(e: dict) -> str:
    """One :func:`build_result_timeline` entry as a PLAIN-TEXT battle-log line — the unstyled
    sibling of the TUI's Rich renderer (``app._append_timeline_entry``), with the same wording:

        ``we thunderbolt did 31% (suicune 31% → faint)`` · ``opp rockslide did 73% (salamence 100%
        → 27%)`` · ``we switch tyranitar → skarmory`` · ``opp sends in metagross`` ·
        ``we hypnosis — missed``

    Pure. It exists so the JSON CLI and the web view render the timeline without either of them
    re-deriving the sentence (the engine is the single source of truth for what happened, including
    how it reads)."""
    side = str(e.get("side", ""))
    kind = e.get("kind")
    if kind == "switch":
        return f"{side} switch {e.get('actor', '')} → {e.get('switch_to', '')}".strip()
    if kind == "send_in":
        verb = "send in" if side == "we" else "sends in"
        return f"{side} {verb} {e.get('sent_in', '')}".strip()
    if kind == "faint":
        return f"{side} {e.get('actor', '')} fainted".strip()

    parts = [side, str(e.get("move") or "?")]
    text = " ".join(p for p in parts if p)
    if e.get("never_moved"):
        # It chose the move and was KO'd before it could make it. The old rendering said
        # "— no effect", which describes a move that executed and achieved nothing — a claim about
        # the MOVE, on a turn where the move never happened at all.
        text += " — never moved (fainted first)"
        return text
    if e.get("cant"):
        text += f" — couldn't move ({cant_phrase(e['cant'])})"
    elif e.get("damage"):
        text += (f" did {e['damage']}  ({e.get('target', '')} "
                 f"{e.get('hp_before', '')} → {e.get('hp_after', '')})")
    elif e.get("resulting"):        # hit a switch-IN; only the resulting HP is known
        text += f" → {e.get('target', '')} (now {e.get('hp_after', '')})"
    elif e.get("status"):
        text += f" → {e.get('target', '')} {e['status']}"
    elif e.get("no_effect"):        # nothing happened — say WHY, never leave it blank
        text += f" — {NO_EFFECT_TEXT.get(e['no_effect'], 'no effect')}"
    if e.get("boost"):
        text += f"  ·  {e['boost']}"
    if e.get("crit"):
        text += "  ⚡CRIT"
    return text
