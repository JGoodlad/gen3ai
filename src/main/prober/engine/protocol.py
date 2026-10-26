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


def _seat(side: "str | None") -> "str | None":
    """``"p1"`` / ``"p2"`` when a caller passes a RECORDED seat, else ``None`` — a reader then falls
    back to nickname matching. A seat is a fact (``meta.trace_source.trainee_side``); a nickname is
    not: this pool ships LOCALIZED ones (``Staross`` = Starmie, ``Leuphorie`` = Blissey) and mojibake
    ones (``MagnÃ©ton``), and a mirror names both sides alike. MEASURED 2026-10-26 over 12.5k
    side-turns of four current runs: the nickname reading said ``absent`` (never moved) for 389
    sides that did move, and nothing at all for 886 (mirrors), blanking the line."""
    return side if side in ("p1", "p2") else None


def _player_of(field: str) -> str:
    """``"p1a: Gengar"`` → ``"p1"``; ``""`` when the field names no player."""
    return field.split(":", 1)[0].strip()[:2] if ":" in field else ""


def move_order_from_protocol(lines: "tuple[str, ...] | None",
                             our_active: str, opp_active: str,
                             our_side: "str | None" = None) -> "str | None":
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
    ``our_side`` (the trainee's RECORDED seat) reads the first mover's player tag instead, so
    none of those three can happen.
    """
    seat = _seat(our_side)
    ours, opps = _norm_species(our_active), _norm_species(opp_active)
    if not seat and (not ours or not opps or ours == opps):
        return None                      # a mirror: the nickname names both sides equally
    for ln in lines or ():
        if not ln.startswith("|move|"):
            continue
        parts = ln.split("|")
        if len(parts) < 3:
            continue
        actor = parts[2]                 # e.g. "p1a: Jirachi"
        if seat:
            who = _player_of(actor)
            return None if not who else ("we_first" if who == seat else "opp_first")
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
                         actor: str, other: str,
                         actor_side: "str | None" = None) -> "str | None":
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

    ``actor_side`` (``"p1"`` / ``"p2"``, the actor's RECORDED seat) replaces the nickname match by the
    player tag, which is what makes the guard unnecessary: a localized or mojibake nickname no longer
    reads as a side that never moved, and a mirror reads like any other match.
    """
    seat = _seat(actor_side)
    a, o = _norm_species(actor), _norm_species(other)
    if not seat and (not a or not o or a == o):
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
        if (_player_of(who) == seat) if seat else (nick == a):
            if ln.startswith("|move|"):
                return "moved"
            reason = parts[3].strip() if len(parts) > 3 else ""
            return f"cant:{reason}" if reason else "cant:"
        if (_player_of(who) != "" and _player_of(who) != seat) if seat else (nick == o):
            other_seen = True
    return "absent" if other_seen else None


def protocol_move_result(lines: "tuple[str, ...] | None",
                         actor: str, other: str,
                         actor_side: "str | None" = None) -> "str | None":
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
    ``actor_side`` (the actor's RECORDED seat) finds the actor by player tag, as in
    :func:`protocol_action_fate`.
    """
    seat = _seat(actor_side)
    a, o = _norm_species(actor), _norm_species(other)
    if not seat and (not a or not o or a == o):
        return None
    def _nick(field: str) -> str:
        return _norm_species(field.split(":", 1)[1] if ":" in field else field)

    def _is_actor(field: str) -> bool:
        return (_player_of(field) == seat) if seat else (_nick(field) == a)

    started = False
    for ln in lines or ():
        parts = ln.split("|")
        if ln.startswith("|move|") and len(parts) > 2:
            if started:
                break                      # the next move — out of this move's window
            started = _is_actor(parts[2])
            continue
        if not started:
            continue
        if ln.startswith(("|turn|", "|upkeep")):
            break
        if ln.startswith("|-immune|"):
            return "immune"
        if ln.startswith("|-miss|") and len(parts) > 2 and _is_actor(parts[2]):
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
                          actor: str, other: str,
                          actor_side: "str | None" = None) -> "dict | None":
    """Everything the LOG says the ACTOR's own move did this turn, or ``None`` when the move's
    window cannot be located (no window ⇒ no evidence, which is a different answer from "nothing").

    ``{"result": "immune"|"missed"|"failed"|None, "status": (name, "PAR", self_targeted)|None,
       "statuses": ((name, "PAR", self_targeted, via), ...), "effects": tuple[str, ...]}`` —
    ``effects`` are the raw tags that PROVE something happened; ``status`` is the first of
    ``statuses`` (all of them, in log order, ``via`` the ability that relayed one, else ``""``).

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
    are matched by nickname exactly as its siblings do, and a mirror returns ``None`` —
    UNLESS ``actor_side`` (``"p1"`` / ``"p2"``, the actor's RECORDED seat: the trainee's is
    :func:`trainee_side_of`, the opponent's the other) is given, which finds the actor's ``|move|``
    line by player tag alone: a mirror and a localized nickname (``Electhor`` = Zapdos) then read
    like any other match. ``actor`` / ``other`` are ignored in that case.
    """
    a, o = _norm_species(actor), _norm_species(other)
    seat = actor_side if actor_side in ("p1", "p2") else None
    if not seat and (not a or not o or a == o):
        return None

    def _nick(field: str) -> str:
        return _norm_species(field.split(":", 1)[1] if ":" in field else field)

    def _player(field: str) -> str:
        return field.split(":", 1)[0].strip() if ":" in field else ""

    def _is_actor(field: str) -> bool:
        return (_player_of(field) == seat) if seat else (_nick(field) == a)

    started = False
    actor_tag = ""
    result: "str | None" = None
    status: "tuple[str, str, bool] | None" = None
    statuses: "list[tuple[str, str, bool, str]]" = []
    effects: "list[str]" = []
    for ln in lines or ():
        parts = ln.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag == "move" and len(parts) > 2:
            if started:
                break                          # the next move — out of this move's window
            started = _is_actor(parts[2])
            if started:
                actor_tag = _player(parts[2])  # "p1a" — which SIDE this move came from
            continue
        if not started:
            continue
        if tag in ("", "turn", "upkeep"):
            break                              # the action block ended; residuals are not the move
        if tag == "-immune":
            result = result or "immune"
        elif tag == "-miss" and len(parts) > 2 and _is_actor(parts[2]):
            result = result or "missed"
        elif tag == "-fail":
            result = result or "failed"
        elif tag == "-status" and len(parts) > 3:
            code = parts[3].strip().lower()
            if code in _PROTOCOL_STATUS_DISPLAY:
                hit = (_nick(parts[2]), _PROTOCOL_STATUS_DISPLAY[code],
                       bool(actor_tag) and _player(parts[2]) == actor_tag)
                if status is None:
                    status = hit
                # EVERY status the window applied, with the ability that relayed it. A Body Slam
                # that paralyzes a SYNCHRONIZE mon puts a second `-status` on the ATTACKER
                # (`[from] ability: Synchronize|[of] p2a: Umbreon`); `status` alone named only the first.
                statuses.append(hit + (_from_cause(parts[4:]),))
        if tag in _VISIBLE_EFFECT_TAGS:
            effects.append(tag)
    if not started:
        return None
    return {"result": result, "status": status, "statuses": tuple(statuses), "effects": tuple(effects)}


def _from_cause(tags: "list[str]") -> str:
    """``[from] ability: Synchronize`` → ``"Synchronize"`` (an item, ability or move named by a
    protocol line's ``[from]`` tag), else ``""``."""
    for t in tags:
        if t.startswith("[from]"):
            src = t[len("[from]"):].strip()
            return src.split(":", 1)[1].strip() if ":" in src else src
    return ""


# ── Stat STAGES ──────────────────────────────────────────────────────────────────────────────────
# Showdown's protocol (its `sim/SIM-PROTOCOL.md`, the "Minor actions" list) has ten messages
# that move a stat stage. The timeline used to read stages only from the recorder's TurnDelta
# (`model.describe_turn_outcome`), whose block is gone from every current run, so a Swords Dance, an
# Intimidate, a Belly Drum or a Curse never reached a line (MEASURED 2026-10-26: 0 of 92 `-boost` /
# `-unboost` lines over 14 battles of one run). The log is the source now.
#: The messages the gen-3 sim EMITS (the Rust port's `protocol.rs`, which `engine_stages_test.py`
#: reads: boost / unboost incl. the `[from] item:` berries, `-setboost` Belly Drum, `-copyboost`
#: Psych Up, `-clearallboost` Haze, `-clearnegativeboost` White Herb).
BOOST_TAGS_GEN3 = frozenset({"-boost", "-unboost", "-setboost", "-clearallboost", "-copyboost",
                             "-clearnegativeboost"})
#: Messages the protocol defines that no gen-3 move, ability or item emits (Heart / Guard / Power
#: Swap, Topsy-Turvy, Spectral Thief, Clear Smog). Folded anyway, as a NOTE: a stage message this
#: reader did not know must never vanish silently. `engine_stages_test.py` pins
#: `BOOST_TAGS_GEN3 | BOOST_TAGS_LATER_GEN` against every `*boost*` message SIM-PROTOCOL.md lists, so
#: a new one fails a test instead.
BOOST_TAGS_LATER_GEN = frozenset({"-clearboost", "-swapboost", "-invertboost", "-clearpositiveboost"})
_BOOST_TAGS = BOOST_TAGS_GEN3 | BOOST_TAGS_LATER_GEN
#: The protocol's stat ids → the recorder's spelling (`model._BOOST_STATS`), in its order.
_STAT_SHORT = {"atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe",
               "accuracy": "acc", "evasion": "eva"}
_STAT_ORDER = tuple(_STAT_SHORT.values())


class _MonStages:
    """One mon's stage changes over a slice, in the recorder's spelling (``atk+2``) plus the
    changes a stage DELTA cannot express (a set, a clear, a copy) as NOTES.

    A stat's entry is ``("d", n)`` — a net change by ``n`` — or ``("s", n)`` — the stage is now
    exactly ``n`` (Belly Drum sets it; after a clear the base is KNOWN to be 0, so a later change is
    absolute too). A slice never says where a stage STARTED, so a set is never turned into a delta."""
    __slots__ = ("label", "stats", "notes", "base_zero", "said_clear")

    def __init__(self, label: str, base_zero: bool = False) -> None:
        self.label = label
        self.stats: "dict[str, tuple[str, int]]" = {}
        self.notes: "list[str]" = []
        self.base_zero = base_zero         # every stage was 0 at some point in the slice
        self.said_clear = False

    def change(self, stat: str, n: int) -> None:
        kind, cur = self.stats.get(stat, ("s", 0) if self.base_zero else ("d", 0))
        self.stats[stat] = (kind, cur + n)

    def clear(self) -> None:
        self.stats, self.base_zero, self.said_clear = {}, True, True

    def text(self) -> str:
        out = ["boosts cleared"] if self.said_clear else []
        for stat in _STAT_ORDER:
            if stat in self.stats:
                kind, n = self.stats[stat]
                if kind == "s":
                    out.append(f"{stat}={n:+d}" if n else f"{stat}=0")
                elif n:
                    out.append(f"{stat}{n:+d}")
        return " ".join(out + self.notes)


def _side_tags(lines: "tuple[str, ...]", ours: str, opps: str) -> "tuple[str, str] | None":
    """``(our player tag, their player tag)`` — ``("p1", "p2")`` — read off the ``|move|`` /
    ``|cant|`` lines of the two decision-time actives, matched by NICKNAME exactly as
    :func:`move_order_from_protocol` does. One anchor is enough (the other side is the other
    player). ``None`` when no actor matches, when a mirror makes the nickname name both sides, or
    when the matches CONTRADICT each other."""
    if not ours or not opps or ours == opps:
        return None
    we: "set[str]" = set()
    opp: "set[str]" = set()
    for ln in lines or ():
        if not (ln.startswith("|move|") or ln.startswith("|cant|")):
            continue
        parts = ln.split("|")
        if len(parts) < 3 or ":" not in parts[2]:
            continue
        who, nick = parts[2].split(":", 1)
        tag, nick = who.strip()[:2], _norm_species(nick)
        if nick == ours:
            we.add(tag)
        elif nick == opps:
            opp.add(tag)
    if len(we) > 1 or len(opp) > 1 or (we and we == opp):
        return None
    if we and opp:
        return next(iter(we)), next(iter(opp))
    only = next(iter(we or opp), None)
    if only not in ("p1", "p2"):
        return None
    other = "p2" if only == "p1" else "p1"
    return (only, other) if we else (other, only)


def trainee_side_of(summary: "dict | None") -> "str | None":
    """The trainee's player tag (``"p1"`` / ``"p2"``) a core trace RECORDS
    (``meta.trace_source.trainee_side``), else ``None`` — an older trace names no seat, and a caller
    then falls back to the nickname matching its siblings use. Fact, not a heuristic: the seat is
    what makes a MIRROR (both actives the same species) readable at all."""
    side = (((summary or {}).get("meta") or {}).get("trace_source") or {}).get("trainee_side")
    return side if side in ("p1", "p2") else None


def protocol_boost_changes(lines: "tuple[str, ...] | None",
                           our_active: str, opp_active: str,
                           our_side: "str | None" = None) -> "dict | None":
    """The stat-stage changes the LOG records this turn, or ``None`` when it carries some but the
    two sides cannot be told apart (a mirror with no recorded seat, or no ``|move|`` / ``|cant|``
    line to anchor the nicknames on) — no reading rather than a guessed one, like its siblings.

    ``{"side": {"we": "atk+2 spe-1", "opp": "atk-1"}, "caused": {"we": "spd-1"}}`` (only the keys
    with a change; both dicts empty when the slice carries no stage message):

    * ``side`` — the NET change to each side's mon: its own Swords Dance, an Intimidate on arrival,
      a berry, a Belly Drum. The recorder's own vocabulary (``model._boost_delta_str``:
      ``atk+1 spe+2``), so a recorded and a protocol-sourced boost read identically; ``-boost`` and
      ``-unboost`` of one stat cancel. What a delta cannot say is said in words: Belly Drum's
      ``-setboost`` is ``atk=+6`` (the stage it SET), Haze / a herb / Psych Up are notes. A change
      on more than one mon of a side (a switch mid-turn) is labelled by species. A zero-amount line
      (``-boost|…|atk|0``) reads ``atk at its limit``.
    * ``caused`` — the change a side's MOVE put on the OTHER side's mon (Psychic's Sp. Def drop,
      Screech, Icy Wind), keyed by the side that made it. Told from ``side`` by the log's own
      framing: inside that move's window (to the next ``|move|``, the blank before the residuals),
      with no ``[from]`` tag (a berry is the victim's) and not straight after an Intimidate's
      ``|-ability|`` line. Without it ``opp thunderwave — no effect · spd-1`` read as if the
      Thunder Wave had done it.

    ``our_side`` is the trainee's recorded seat (:func:`trainee_side_of`); given it, the sides are
    known and a mirror reads like any other match."""
    lines = tuple(lines or ())
    if not any(len(p) > 1 and p[1] in _BOOST_TAGS for p in (ln.split("|", 2) for ln in lines)):
        return {"side": {}, "caused": {}}
    if our_side in ("p1", "p2"):
        tags = (our_side, "p2" if our_side == "p1" else "p1")
    else:
        tags = _side_tags(lines, _norm_species(our_active), _norm_species(opp_active))
    if tags is None:
        return None
    side_of = {tags[0]: "we", tags[1]: "opp"}

    species: "dict[tuple[str, str], str]" = {}        # (player, nickname) → species, from the switch-in lines
    net: "dict[tuple[str, str], _MonStages]" = {}     # insertion order = first-change order
    caused: "dict[str, dict[tuple[str, str], _MonStages]]" = {"we": {}, "opp": {}}
    all_cleared = False                               # a Haze: every mon, named or not, is at 0 from here
    win: "str | None" = None                          # the side whose move's window we are inside
    from_intimidate = False                           # the last line was an Intimidate's `-ability`

    def _ref(field: str) -> "tuple[str, str] | None":
        if ":" not in field:
            return None
        who, nick = field.split(":", 1)
        return who.strip()[:2], nick.strip()

    def _mon(book: dict, ref: "tuple[str, str]") -> "_MonStages":
        if ref not in book:
            book[ref] = _MonStages(species.get(ref) or ref[1].lower(), base_zero=all_cleared)
        return book[ref]

    for ln in lines:
        parts = ln.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag in ("switch", "drag", "replace") and len(parts) > 3:
            ref = _ref(parts[2])
            if ref:
                species[ref] = parts[3].split(",", 1)[0].strip().lower()
        # WINDOWS, exactly as `protocol_move_effects` frames them: a move opens one, the next
        # action, the blank before the residuals, or a switch-in (a Roar's drag) closes it.
        if tag == "move" and len(parts) > 2:
            ref = _ref(parts[2])
            win = side_of.get(ref[0]) if ref else None
        elif tag in ("", "turn", "upkeep", "switch", "drag", "replace", "cant"):
            win = None
        if tag == "-ability":
            from_intimidate = len(parts) > 4 and parts[4].strip() == "boost"
            continue
        was_intimidate, from_intimidate = from_intimidate, (from_intimidate and tag in _BOOST_TAGS)
        if tag not in _BOOST_TAGS:
            continue
        if tag == "-clearallboost":
            all_cleared = True
            for m in net.values():
                m.clear()
            continue
        ref = _ref(parts[2]) if len(parts) > 2 else None
        if ref is None or ref[0] not in side_of:
            continue
        has_from = any(p.startswith("[from]") for p in parts[5:])
        book = net
        if (tag in ("-boost", "-unboost") and win is not None and side_of[ref[0]] != win
                and not has_from and not was_intimidate):
            book = caused[win]
        m = _mon(book, ref)
        if tag in ("-boost", "-unboost", "-setboost"):
            stat = _STAT_SHORT.get(parts[3].strip() if len(parts) > 3 else "", "")
            try:
                n = int(parts[4]) if len(parts) > 4 else None
            except ValueError:
                n = None
            if not stat or n is None:
                m.notes.append("stat change")
            elif tag == "-setboost":
                m.stats[stat] = ("s", n)
            elif n == 0:
                m.notes.append(f"{stat} at its limit")
            else:
                m.change(stat, n if tag == "-boost" else -n)
        elif tag == "-clearboost":
            m.clear()
        elif tag == "-clearnegativeboost":
            m.stats = {k: v for k, v in m.stats.items() if v[1] >= 0}
            m.notes.append("drops restored")
        elif tag == "-copyboost":
            other = _ref(parts[3]) if len(parts) > 3 else None
            m.stats, m.base_zero = {}, False
            m.notes.append(f"copied {_mon(net, other).label if other else 'the target'}'s boosts")
        elif tag == "-swapboost":
            other = _ref(parts[3]) if len(parts) > 3 else None
            m.notes.append(f"swapped boosts with {_mon(net, other).label if other else 'the target'}")
        elif tag == "-invertboost":
            m.notes.append("boosts inverted")
        elif tag == "-clearpositiveboost":
            m.notes.append("gains cleared")

    def _text(book: dict, side: "str | None", haze: bool) -> str:
        mine = [m for ref, m in book.items() if side is None or side_of.get(ref[0]) == side]
        shown = [m for m in mine if m.text()]
        # A Haze clears mons this slice never named: say so once for the side, not per mon.
        said = ["boosts cleared"] if haze and not any(m.said_clear for m in mine) else []
        if len(shown) == 1 and not said:
            return shown[0].text()
        return "; ".join(said + [f"{m.label} {m.text()}" for m in shown])

    out: "dict[str, dict[str, str]]" = {"side": {}, "caused": {}}
    for side in ("we", "opp"):
        if t := _text(net, side, all_cleared):
            out["side"][side] = t
        if t := _text(caused[side], None, False):
            out["caused"][side] = t
    return out
