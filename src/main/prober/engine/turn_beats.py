"""A turn's events grouped into ordered BEATS — one actor's action plus every consequence it caused,
in the order the simulator ran them (`designs/prober/battle_viewer_ux_2026-10-09.md` §5).

Pure: the turn's protocol lines (structural ones included — the framing is what tells the phases
apart) paired with the event `turn_events.TurnFold` made of each, in; plain dicts out.

THE PHASES, recognised from the protocol's own framing (Showdown `sim/battle.ts`, which the Rust core
emits line for line):

* ``lead``     — turn 0's switch-ins.
* ``start``    — start-of-turn effects (Focus Punch tightening its focus): lines before the turn's
                 first action.
* ``switch``   — a switch BY CHOICE: a ``|switch|`` for a side whose active has not fainted.
* ``move``     — a ``|move|`` or a ``|cant|``, numbered in execution ORDER; ``order == 1`` moved first.
                 A ``[from]`` move (Sleep Talk's call, Magic Coat's bounce) stays in its caller's beat.
* ``replace``  — a FORCED replacement: a ``|switch|`` for a side whose active fainted since its last
                 switch. In Gen 3 that is sent in straight after the action that caused the faint,
                 before the next move ("in gen 3 or earlier, switching in fainted pokemon is done after
                 every move", `runAction`), so a replace beat can sit BETWEEN two moves.
* ``residual`` — end of turn: from the residual action's opening blank line (``this.add('')``) to
                 ``|upkeep|``, plus the faints the residuals caused (announced after ``upkeep``).
* ``end``      — ``|win|`` / ``|tie|``.

Inside a beat the ``effects`` are its consequences: a damage effect carries ``hp_before`` →
``hp_after`` and its cause, the effectiveness / critical-hit lines that preceded it become its
``tags``, consecutive unsourced hits of a multi-hit move merge (``hits``), and a faint carries its
``cause`` — the move that KO'd it, the residual that did, or "its own" move for a self-KO.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

#: Effect kinds that QUALIFY the next damage line on the same target rather than standing alone.
_QUALIFIERS = {"supereffective": "super effective", "resisted": "not very effective", "crit": "critical hit"}

#: The beat phases, in the words the page prints.
PHASE_WORDS = {"lead": "Lead", "start": "Start of turn", "switch": "Switch", "move": "Move",
               "replace": "Replacement", "residual": "End of turn", "end": "Result"}

_EFFECT_FIELDS = ("pct", "hp_before", "hp_after", "source", "fainted", "status", "stat", "amount", "stage",
                  "condition", "start", "effect", "item", "ability", "weather", "consumed", "hits", "reason")


def _residual_start(seq: Sequence[Tuple[str, Optional[dict]]], up: Optional[int]) -> int:
    """Index where the end-of-turn residuals begin: the LAST blank framing line before ``|upkeep``
    (the residual action opens with one). With no blank there (a trimmed log), the first
    ``[upkeep]``-tagged line; with no ``|upkeep`` at all (the battle ended mid-turn), none."""
    if up is None:
        return len(seq)
    blank = next((i for i in range(up - 1, -1, -1) if seq[i][0].strip() == "|"), None)
    if blank is not None:
        return blank
    tagged = next((i for i, (ln, _) in enumerate(seq[:up]) if "[upkeep]" in ln), None)
    return tagged if tagged is not None else up


def _faint_cause(beat: dict, ev: dict) -> str:
    """Why this mon fainted, read off the beat it fainted in: the last damage it took there (its
    source — Sandstorm, recoil, Spikes — or the beat's move), or the move itself for a self-KO."""
    side, sp = ev.get("side"), ev.get("species")
    hit = next((e for e in reversed(beat["effects"])
                if e.get("kind") == "damage" and e.get("side") == side and e.get("species") == sp), None)
    if hit is not None and hit.get("source"):
        return str(hit["source"])
    if beat["kind"] == "move":
        mv = beat.get("move") or "?"
        if beat.get("side") == side and beat.get("species") == sp:
            return f"its own {mv}"
        who = {"we": "our", "opp": "their"}.get(beat.get("side") or "", "")
        return f"{who} {beat.get('species')}'s {mv}".strip()
    if beat["phase"] == "residual":
        return "end-of-turn damage"
    return ""


class _Builder:
    def __init__(self) -> None:
        self.beats: List[dict] = []
        self.cur: Optional[dict] = None
        self.residual: Optional[dict] = None
        self.pending_faint: Dict[str, bool] = {}     # side -> its active fainted, not yet replaced
        self.quals: Dict[Tuple, List[str]] = {}      # (side, species) -> qualifiers awaiting a damage line
        self.order = 0

    def push(self, phase: str, kind: str, ev: Optional[dict], **kw) -> dict:
        b = {"index": len(self.beats), "phase": phase, "kind": kind,
             "side": (ev or {}).get("side"), "species": (ev or {}).get("species"),
             "text": (ev or {}).get("text"), "effects": []}
        b.update(kw)
        self.beats.append(b)
        return b

    def get_residual(self) -> dict:
        if self.residual is None:
            self.residual = self.push("residual", "residual", None, text="end of turn")
        return self.residual

    @staticmethod
    def stamp(b: dict, ev: dict) -> None:
        ev["beat"], ev["phase"] = b["index"], b["phase"]

    def attach(self, b: dict, ev: dict) -> None:
        self.stamp(b, ev)
        k = ev.get("kind")
        key = (ev.get("side"), ev.get("species"))
        if k in _QUALIFIERS:
            self.quals.setdefault(key, []).append(_QUALIFIERS[k])
            return
        eff = {"kind": k, "side": ev.get("side"), "species": ev.get("species"), "text": ev.get("text")}
        for f in _EFFECT_FIELDS:
            if ev.get(f) is not None:
                eff[f] = ev[f]
        if k == "damage":
            tags = self.quals.pop(key, [])
            prev = b["effects"][-1] if b["effects"] else None
            if (b["kind"] == "move" and prev is not None and prev.get("kind") == "damage"
                    and prev.get("side") == eff["side"] and prev.get("species") == eff["species"]
                    and not prev.get("source") and not eff.get("source")):
                prev["hp_after"] = eff.get("hp_after")      # a multi-hit move: one line, its total
                if prev.get("hp_before") is not None and eff.get("hp_after") is not None:
                    prev["pct"] = round(eff["hp_after"] - prev["hp_before"], 1)
                prev["hits"] = int(prev.get("hits") or 1) + 1
                prev["fainted"] = bool(eff.get("fainted"))
                prev["tags"] = sorted(set(prev.get("tags") or []) | set(tags))
                return
            eff["tags"] = tags
            if b["kind"] == "move" and not eff.get("source"):
                eff["cause"] = b.get("move")
        elif k == "hitcount":
            for e in reversed(b["effects"]):
                if e.get("kind") == "damage" and e.get("side") == eff["side"]:
                    try:
                        e["hits"] = int(ev.get("hits"))
                    except (TypeError, ValueError):
                        pass
                    break
            return
        elif k == "faint":
            eff["cause"] = _faint_cause(b, ev)
            if ev.get("side"):
                self.pending_faint[ev["side"]] = True
        elif k in ("immune", "miss", "fail"):
            q = self.quals.pop(key, [])
            if q:
                eff["tags"] = q
        b["effects"].append(eff)


def build_beats(seq: Sequence[Tuple[str, Optional[dict]]], *, turn: int = 0) -> List[dict]:
    """Group one turn's ``(protocol line, event | None)`` pairs into ordered beats; each event is
    stamped with ``beat`` (its index) and ``phase``. See the module docstring for the rules."""
    B = _Builder()
    up = next((i for i, (ln, _) in enumerate(seq) if ln.startswith("|upkeep")), None)
    res_at = _residual_start(seq, up)
    for i, (_ln, ev) in enumerate(seq):
        if ev is None:
            continue
        k, side = ev.get("kind"), ev.get("side")
        after_upkeep = up is not None and i > up
        in_residual = (i >= res_at and not after_upkeep)
        if k == "end":
            B.attach(B.push("end", "end", ev), ev)
        elif turn == 0 and k in ("switch", "drag"):
            B.cur = B.push("lead", "lead", ev, hp_after=ev.get("hp_after"))
            B.stamp(B.cur, ev)
        elif k == "switch":
            cur = B.cur
            if (cur is not None and cur["kind"] == "move" and cur.get("side") == side
                    and cur.get("move") == "Baton Pass"
                    and not any(e.get("kind") == "switch" for e in cur["effects"])):
                B.attach(cur, dict(ev, text=f"{ev.get('text')} (Baton Pass)"))
                B.stamp(cur, ev)
                continue
            forced = bool(side and B.pending_faint.get(side))
            kind = "replace" if forced else "switch"
            B.cur = B.push(kind, kind, ev, from_species=ev.get("replaced_species"), hp_after=ev.get("hp_after"))
            B.stamp(B.cur, ev)
            if side:
                B.pending_faint[side] = False
        elif k == "drag":
            host = B.cur if B.cur is not None else B.push("move", "drag", ev)
            B.attach(host, ev)
            if side:
                B.pending_faint[side] = False
        elif in_residual or after_upkeep:
            B.attach(B.get_residual(), ev)
        elif k == "move" and ev.get("via") and B.cur is not None and B.cur["kind"] == "move" \
                and B.cur.get("side") == side:
            B.attach(B.cur, ev)
        elif k in ("move", "cant"):
            B.order += 1
            B.cur = B.push("move", k, ev, move=ev.get("move"), order=B.order, first=B.order == 1,
                           target_side=ev.get("target_side"), target_species=ev.get("target_species"),
                           missed=bool(ev.get("missed")), reason=ev.get("reason"))
            B.stamp(B.cur, ev)
        else:
            if B.cur is None:
                B.cur = B.push("start", "start", None, text="start of turn")
            B.attach(B.cur, ev)
    for b in B.beats:
        b["phase_word"] = PHASE_WORDS.get(b["phase"], b["phase"])
    return B.beats


def turn_summary(beats: Sequence[dict]) -> Dict[str, dict]:
    """Per side, what the turn rail shows: the side's PRIMARY action this turn (its first switch /
    move / can't-move / lead, as a glyph + a label), whether its mon fainted, and the species it sent in
    after a faint. ``action`` is None when the side did nothing this turn."""
    out: Dict[str, dict] = {}
    for side in ("we", "opp"):
        first = next((b for b in beats if b.get("side") == side and b["kind"] in ("move", "cant", "switch", "lead")),
                     None)
        if first is None:
            act = None
        elif first["kind"] == "move":
            act = {"glyph": "▸", "label": first.get("move"), "kind": "move", "species": first.get("species")}
        elif first["kind"] == "cant":
            act = {"glyph": "⏸", "label": "can't move", "kind": "cant", "species": first.get("species")}
        else:
            act = {"glyph": "⇄" if first["kind"] == "switch" else "", "label": first.get("species"),
                   "kind": first["kind"], "species": first.get("species")}
        fainted = [e.get("species") for b in beats for e in b["effects"]
                   if e.get("kind") == "faint" and e.get("side") == side]
        sent = [b.get("species") for b in beats if b["kind"] == "replace" and b.get("side") == side]
        out[side] = {"action": act, "fainted": fainted, "sent_in": sent}
    return out


__all__ = ["PHASE_WORDS", "build_beats", "turn_summary"]
