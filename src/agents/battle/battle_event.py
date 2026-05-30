"""Revealed-order battle event log: schema + completeness policy.

This is the **gen3ai** layer that sits on top of poke-env's state tracker. poke-env
overwrites "current board" fields as each ``|...|`` protocol line arrives; that is the
right shape for picking a move, but RL / reward shaping / replay analysis need the
opposite — *what happened, in order, between two states*.

``Gen3Battle`` (see ``gen3_battle.py``) captures that ordered stream into a list of
:class:`BattleEvent` **in the same parse pass**, recording attribution *before* the
line mutates state. This module owns the immutable schema and the message-policy
registry that guarantees **every** protocol line is accounted for (see §4 of
``designs/ai_v4/design_event_sourced_battle.md``).

Design rules:
  * **Self-contained.** An event carries enough to be interpreted without re-reading
    battle state (a ``MOVE`` carries its ``move_id`` + resolved target).
  * **Order is meaning.** ``seq`` is the revealed order; "who moved first" is read
    from order, not a derived flag.
  * **No silent drops.** Every line is classified by :data:`MESSAGE_POLICY` into
    exactly one :class:`Policy` bucket; an unclassified or non-gen3 line *raises*.

Side convention matches poke-env's internal parser (``_current_move_user_side``):
``"ours"`` / ``"opp"``. Use the :data:`OURS` / :data:`OPP` constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

# Side labels, matching poke-env's internal `_current_move_user_side`.
OURS = "ours"
OPP = "opp"


class EventKind(IntEnum):
    """What a :class:`BattleEvent` represents. Stable integer values so the log can
    be serialised (replay archives, datasets) without a name lookup."""

    MOVE = 1
    SWITCH = 2
    DRAG = 3
    FAINT = 4
    DAMAGE = 5
    HEAL = 6
    BOOST = 7
    UNBOOST = 8
    SETBOOST = 9
    CLEARBOOST = 10  # clear/clearall/clearnegative/clearpositive/invert/copy/swapboost
    STATUS = 11
    CURESTATUS = 12  # -curestatus and -cureteam
    CANT = 13
    CRIT = 14
    MISS = 15
    FAIL = 16  # -fail / -notarget / -nothing
    IMMUNE = 17
    RESISTED = 18
    SUPEREFFECTIVE = 19
    ITEM = 20
    ENDITEM = 21
    ABILITY = 22  # -ability and -endability
    WEATHER = 23
    FIELD = 24  # -fieldstart / -fieldend
    SIDE = 25  # hazards / screens: -sidestart / -sideend / -swapsideconditions
    VOLATILE_START = 26  # -start / -singleturn / -singlemove (Substitute, Leech Seed, …)
    VOLATILE_END = 27  # -end
    ACTIVATE = 28  # -activate (Trick, Knock Off effect, Pressure, …)
    PREPARE = 29  # -prepare (two-turn move charge)
    MUSTRECHARGE = 30  # -mustrecharge (Hyper Beam)
    TRANSFORM = 31  # -transform
    FORMECHANGE = 32  # -formechange / detailschange / replace
    SWAP = 33  # swap (Ally Switch-style; rare in singles)
    SETHP = 34  # -sethp absolute HP correction (Pain Split) — an event so the log is
    #            HP-complete on its own (the reward manager reads HP deltas from it)
    # Turn boundaries are NOT events: Gen3Battle indexes them in `_turn_start` for
    # O(1) per-turn slicing, keeping the stream free of synthetic markers consumers
    # would have to filter. (The design's earlier "TurnBoundary marker" idea was
    # dropped in favour of the index.)
    UNKNOWN = 99  # defensive sentinel for offline tooling; live gen3ou never emits one


# Which EventKind a given protocol keyword maps to (EVENT-policy types only).
EVENT_KIND: Dict[str, EventKind] = {
    "move": EventKind.MOVE,
    "switch": EventKind.SWITCH,
    "drag": EventKind.DRAG,
    "faint": EventKind.FAINT,
    "-damage": EventKind.DAMAGE,
    "-heal": EventKind.HEAL,
    "-boost": EventKind.BOOST,
    "-unboost": EventKind.UNBOOST,
    "-setboost": EventKind.SETBOOST,
    "-clearboost": EventKind.CLEARBOOST,
    "-clearallboost": EventKind.CLEARBOOST,
    "-clearnegativeboost": EventKind.CLEARBOOST,
    "-clearpositiveboost": EventKind.CLEARBOOST,
    "-invertboost": EventKind.CLEARBOOST,
    "-copyboost": EventKind.CLEARBOOST,
    "-swapboost": EventKind.CLEARBOOST,
    "-status": EventKind.STATUS,
    "-curestatus": EventKind.CURESTATUS,
    "-cureteam": EventKind.CURESTATUS,
    "cant": EventKind.CANT,
    "-crit": EventKind.CRIT,
    "-miss": EventKind.MISS,
    "-fail": EventKind.FAIL,
    "-notarget": EventKind.FAIL,
    "-nothing": EventKind.FAIL,
    "-immune": EventKind.IMMUNE,
    "-resisted": EventKind.RESISTED,
    "-supereffective": EventKind.SUPEREFFECTIVE,
    "-item": EventKind.ITEM,
    "-enditem": EventKind.ENDITEM,
    "-ability": EventKind.ABILITY,
    "-endability": EventKind.ABILITY,
    "-weather": EventKind.WEATHER,
    "-fieldstart": EventKind.FIELD,
    "-fieldend": EventKind.FIELD,
    "-sidestart": EventKind.SIDE,
    "-sideend": EventKind.SIDE,
    "-swapsideconditions": EventKind.SIDE,
    "-start": EventKind.VOLATILE_START,
    "-singleturn": EventKind.VOLATILE_START,
    "-singlemove": EventKind.VOLATILE_START,
    "-end": EventKind.VOLATILE_END,
    "-activate": EventKind.ACTIVATE,
    "-prepare": EventKind.PREPARE,
    "-mustrecharge": EventKind.MUSTRECHARGE,
    "-transform": EventKind.TRANSFORM,
    "-formechange": EventKind.FORMECHANGE,
    "detailschange": EventKind.FORMECHANGE,
    "replace": EventKind.FORMECHANGE,
    "swap": EventKind.SWAP,
    "-sethp": EventKind.SETHP,
}


# Per-kind REQUIRED payload keys. The schema test (battle_event_test.py) asserts that
# every emitted event of a given kind carries at least these keys in ``value`` — the
# structural guard against the Focus-Punch class of loss (a fact silently dropped
# because the consumer's vocabulary didn't have a slot for it). Optional keys
# (``from`` / ``of`` / ``from_move`` / ``target_status`` / ``hp_before`` …) are not
# listed; add a key here the moment a consumer must be able to rely on it.
EVENT_VALUE_KEYS: Dict[EventKind, frozenset] = {
    EventKind.MOVE: frozenset({"move_id"}),
    EventKind.SWITCH: frozenset({"prev_active"}),
    EventKind.DRAG: frozenset({"prev_active"}),
    EventKind.FAINT: frozenset(),
    EventKind.DAMAGE: frozenset({"amount"}),
    EventKind.HEAL: frozenset({"amount"}),
    EventKind.BOOST: frozenset({"stat", "amount"}),
    EventKind.UNBOOST: frozenset({"stat", "amount"}),
    EventKind.SETBOOST: frozenset({"stat", "amount"}),
    EventKind.CLEARBOOST: frozenset({"op"}),
    EventKind.STATUS: frozenset({"status"}),
    EventKind.CURESTATUS: frozenset({"status"}),
    EventKind.CANT: frozenset({"reason"}),
    EventKind.CRIT: frozenset({"op"}),
    EventKind.MISS: frozenset({"op"}),
    EventKind.FAIL: frozenset({"op"}),
    EventKind.IMMUNE: frozenset({"multiplier"}),
    EventKind.RESISTED: frozenset({"multiplier"}),
    EventKind.SUPEREFFECTIVE: frozenset({"multiplier"}),
    EventKind.ITEM: frozenset({"item"}),
    EventKind.ENDITEM: frozenset({"item"}),
    EventKind.ABILITY: frozenset({"ability"}),
    EventKind.WEATHER: frozenset({"weather"}),
    EventKind.FIELD: frozenset({"effect"}),
    EventKind.SIDE: frozenset({"condition"}),
    EventKind.VOLATILE_START: frozenset({"effect"}),
    EventKind.VOLATILE_END: frozenset({"effect"}),
    EventKind.ACTIVATE: frozenset({"effect"}),
    EventKind.PREPARE: frozenset({"move"}),
    EventKind.MUSTRECHARGE: frozenset(),
    EventKind.TRANSFORM: frozenset(),
    EventKind.FORMECHANGE: frozenset({"details"}),
    EventKind.SWAP: frozenset({"detail"}),
    EventKind.SETHP: frozenset({"hp"}),
}


@dataclass(frozen=True)
class BattleEvent:
    """One immutable, ordered record of something the protocol revealed.

    Attribution (``side`` / ``actor_species`` / ``target_species``) is resolved at
    parse time, *before* the line mutates state — which is exactly why the gap=0
    snapshot desync the old diff-based ``TurnDelta`` defended against can no longer
    corrupt it.
    """

    seq: int  # monotonic across the whole battle
    turn: int  # game turn this line belongs to (self.turn at emit time)
    kind: EventKind
    side: Optional[str] = None  # "our" / "opp" / None, resolved at parse time
    actor_species: Optional[str] = None  # the mon that acted / was affected
    target_species: Optional[str] = None
    value: Dict[str, Any] = field(default_factory=dict)  # kind-specific payload
    raw: Tuple[str, ...] = ()  # original split_message — provenance + debugging

    # ---- typed accessors --------------------------------------------------- #
    # Consumers read these instead of indexing ``value`` directly: the keys are an
    # internal detail, and a typed surface means a missing key reads as ``None``
    # rather than a KeyError, while :data:`EVENT_VALUE_KEYS` + the schema test keep
    # the underlying dict honest (no silently-dropped fact — the Focus Punch lesson).
    @property
    def move_id(self) -> Optional[str]:
        """MOVE: the executed move id (delegation-aware — the move that fired)."""
        return self.value.get("move_id")

    @property
    def from_move(self) -> Optional[str]:
        """MOVE: the delegating move (Sleep Talk / Metronome / …), if any."""
        return self.value.get("from_move")

    @property
    def amount(self) -> Optional[float]:
        """DAMAGE/HEAL: signed HP-fraction delta (negative = HP lost).
        BOOST/UNBOOST/SETBOOST: signed stat-stage delta."""
        return self.value.get("amount")

    @property
    def reason(self) -> Optional[str]:
        """Free-form cause string (CANT reason verbatim, ``[from]`` effect, …).
        Deliberately untyped: a never-before-seen reason is preserved as-is rather
        than collapsed into a fixed vocabulary."""
        return self.value.get("reason")

    @property
    def cant_move(self) -> Optional[str]:
        """CANT: the move that was prevented (e.g. ``focuspunch``)."""
        return self.value.get("move")

    @property
    def stat(self) -> Optional[str]:
        """BOOST/UNBOOST/SETBOOST: the stat changed (atk/def/spa/spd/spe/accuracy/evasion)."""
        return self.value.get("stat")

    @property
    def status(self) -> Optional[str]:
        """STATUS/CURESTATUS: the status id (brn/par/slp/frz/psn/tox)."""
        return self.value.get("status")

    @property
    def item(self) -> Optional[str]:
        """ITEM/ENDITEM: the item id."""
        return self.value.get("item")

    @property
    def ability(self) -> Optional[str]:
        """ABILITY: the ability id."""
        return self.value.get("ability")

    @property
    def weather(self) -> Optional[str]:
        """WEATHER: the weather id."""
        return self.value.get("weather")

    @property
    def effect(self) -> Optional[str]:
        """VOLATILE_*/ACTIVATE/FIELD/SIDE: the effect / condition id."""
        return self.value.get("effect") or self.value.get("condition")

    @property
    def multiplier(self) -> Optional[float]:
        """IMMUNE/RESISTED/SUPEREFFECTIVE: 0.0 / 0.5 / 2.0."""
        return self.value.get("multiplier")

    @property
    def from_cause(self) -> Optional[str]:
        """The ``[from] …`` source clause (e.g. ``move: Knock Off``, ``Spikes``)."""
        return self.value.get("from")

    @property
    def of_source(self) -> Optional[str]:
        """The ``[of] …`` source-mon identifier clause."""
        return self.value.get("of")


class Policy(IntEnum):
    """How the dispatcher treats a protocol keyword. Every keyword poke-env can emit
    is classified into exactly one of these (see :data:`MESSAGE_POLICY`)."""

    EVENT = 1  # recorded as a BattleEvent (poke-env also mutates state)
    STATE_ONLY = 2  # mutates state, intentionally not an event (e.g. |-sethp| correction)
    CONTROL = 3  # protocol control, not battle content (|turn| |request| |player| …)
    COSMETIC = 4  # explicitly irrelevant, with a reason (|-anim| |c| chat |inactive| …)
    UNSUPPORTED = 5  # a real protocol line that CANNOT occur in gen3ou — raise if seen


class UnsupportedMessageType(Exception):
    """A protocol line classified ``UNSUPPORTED`` (a non-gen3 mechanic) was seen. By
    design this is fatal: gen3ou's protocol surface is fixed and finite, so a
    ``-mega`` / ``-zpower`` / ``-terastallize`` line means we are parsing the wrong
    format and any state we'd produce is garbage."""


class UnknownMessageType(Exception):
    """A protocol line not present in :data:`MESSAGE_POLICY` was seen. Coverage is
    stale; fail loudly so the §4.4 audit forces it to be classified before shipping."""


# ---------------------------------------------------------------------------
# The message policy registry — one entry per protocol keyword poke-env can emit.
#
# Keep in sync with AbstractBattle.parse_message's dispatch and MESSAGES_TO_IGNORE.
# The registry-audit unit test (battle_event_test.py) asserts that union is fully
# covered, so a future poke-env keyword change fails a test rather than slipping
# through as an UnknownMessageType at runtime.
# ---------------------------------------------------------------------------
_E = Policy.EVENT
_S = Policy.STATE_ONLY
_C = Policy.CONTROL
_K = Policy.COSMETIC
_U = Policy.UNSUPPORTED

MESSAGE_POLICY: Dict[str, Tuple[Policy, str]] = {
    # ---- battle content: recorded as events ----
    "move": (_E, "move use"),
    "switch": (_E, "switch in"),
    "drag": (_E, "forced switch (phaze)"),
    "faint": (_E, "faint"),
    "-damage": (_E, "hp loss"),
    "-heal": (_E, "hp gain"),
    "-boost": (_E, "stat stage up"),
    "-unboost": (_E, "stat stage down"),
    "-setboost": (_E, "stat stage set (Belly Drum)"),
    "-clearboost": (_E, "clear one mon's boosts"),
    "-clearallboost": (_E, "clear all boosts (Haze)"),
    "-clearnegativeboost": (_E, "clear negative boosts"),
    "-clearpositiveboost": (_E, "clear positive boosts"),
    "-invertboost": (_E, "invert boosts"),
    "-copyboost": (_E, "copy boosts (Psych Up)"),
    "-swapboost": (_E, "swap boosts (Heart Swap-style)"),
    "-status": (_E, "status inflicted"),
    "-curestatus": (_E, "status cured"),
    "-cureteam": (_E, "team status cured (Heal Bell/Aromatherapy)"),
    "cant": (_E, "prevented from acting"),
    "-crit": (_E, "critical hit on current mover's target"),
    "-miss": (_E, "move missed"),
    "-fail": (_E, "move failed"),
    "-notarget": (_E, "no target"),
    "-nothing": (_E, "move did nothing"),
    "-immune": (_E, "target immune (0x)"),
    "-resisted": (_E, "target resisted (0.5x)"),
    "-supereffective": (_E, "super effective (2x)"),
    "-item": (_E, "item revealed/gained"),
    "-enditem": (_E, "item consumed/removed"),
    "-ability": (_E, "ability revealed/triggered"),
    "-endability": (_E, "ability suppressed"),
    "-weather": (_E, "weather set/continue"),
    "-fieldstart": (_E, "field effect start"),
    "-fieldend": (_E, "field effect end"),
    "-sidestart": (_E, "side condition start (hazards/screens)"),
    "-sideend": (_E, "side condition end"),
    "-swapsideconditions": (_E, "swap side conditions (Court Change-style)"),
    "-start": (_E, "volatile/effect start"),
    "-singleturn": (_E, "single-turn effect (Protect/Endure)"),
    "-singlemove": (_E, "single-move effect (Destiny Bond/Grudge)"),
    "-end": (_E, "volatile/effect end"),
    "-activate": (_E, "effect activation (Trick, Knock Off, Pressure, …)"),
    "-prepare": (_E, "two-turn move charging"),
    "-mustrecharge": (_E, "must recharge (Hyper Beam)"),
    "-transform": (_E, "Transform"),
    "-formechange": (_E, "forme change"),
    "detailschange": (_E, "permanent details change"),
    "replace": (_E, "illusion reveal / replace"),
    "swap": (_E, "position swap"),
    "-sethp": (_E, "absolute hp set (Pain Split) — evented so the log is HP-complete"),
    # ---- state-only: mutates state, intentionally not an event ----
    # (empty in gen3ou: every state-mutating line we see is also battle content and
    # is recorded as an EVENT. The bucket is kept for protocol lines that only
    # correct internal state without being a game action.)
    # ---- protocol control: handled, not battle content ----
    "turn": (_C, "turn boundary (indexed in Gen3Battle._turn_start, not an event)"),
    "start": (_C, "battle start"),
    "rule": (_C, "format rule"),
    "gen": (_C, "generation declaration"),
    "tier": (_C, "tier declaration"),
    "teamsize": (_C, "team size"),
    "player": (_C, "player identity"),
    "poke": (_C, "team preview entry"),
    "clearpoke": (_C, "clear team preview"),
    "teampreview": (_C, "team preview phase"),
    "inactive": (_C, "inactivity timer on"),
    "inactiveoff": (_C, "inactivity timer off"),
    "upkeep": (_C, "end-of-turn upkeep control"),
    "title": (_C, "room title"),
    "raw": (_C, "raw html control"),
    "rated": (_C, "rated battle marker"),
    "gametype": (_C, "game type (singles)"),
    "init": (_C, "room init"),
    "deinit": (_C, "room deinit"),
    "sentchoice": (_C, "choice acknowledgement"),
    "split": (_C, "spectator/player split marker"),
    "askreg": (_C, "registration prompt"),
    "": (_C, "empty/blank line"),
    # ---- cosmetic: explicitly irrelevant flavor ----
    ":": (_K, "timestamp"),
    "t:": (_K, "timestamp"),
    "-anim": (_K, "move animation"),
    "-hint": (_K, "rules hint"),
    "-hitcount": (_K, "multi-hit count"),
    "-waiting": (_K, "waiting on other mon"),
    "-block": (_K, "move blocked animation"),
    "-center": (_K, "triple-battle centering"),
    "-combine": (_K, "combined move animation"),
    "-fieldactivate": (_K, "field activation flavor"),
    "-ohko": (_K, "one-hit KO flavor"),
    "message": (_K, "server message"),
    "-message": (_K, "server message"),
    "html": (_K, "html flavor"),
    "uhtml": (_K, "updatable html flavor"),
    "hidelines": (_K, "hide lines flavor"),
    "debug": (_K, "debug line"),
    "badge": (_K, "user badge"),
    "c": (_K, "chat"),
    "chat": (_K, "chat"),
    "J": (_K, "user join"),
    "L": (_K, "user leave"),
    "j": (_K, "user join"),
    "l": (_K, "user leave"),
    "join": (_K, "user join"),
    "leave": (_K, "user leave"),
    "n": (_K, "user rename"),
    "name": (_K, "user rename"),
    # legacy no-dash duplicates poke-env keeps in MESSAGES_TO_IGNORE (the real
    # signals are the dashed forms above):
    "crit": (_K, "legacy crit alias (dashed -crit is the real one)"),
    "immune": (_K, "legacy immune alias (dashed -immune is the real one)"),
    "resisted": (_K, "legacy resisted alias (dashed -resisted is the real one)"),
    "supereffective": (_K, "legacy alias (dashed -supereffective is the real one)"),
    # ---- unsupported: non-gen3 mechanics, fatal if ever seen ----
    "-mega": (_U, "Mega Evolution — Gen 6+"),
    "-primal": (_U, "Primal Reversion — Gen 6"),
    "-zpower": (_U, "Z-move — Gen 7"),
    "-zbroken": (_U, "Z-move shield break — Gen 7"),
    "zbroken": (_U, "Z-move shield break (legacy) — Gen 7"),
    "-burst": (_U, "Ultra Burst — Gen 7"),
    "-terastallize": (_U, "Terastallization — Gen 9"),
}


def classify(keyword: str) -> Tuple[Policy, str]:
    """Look up the policy for a protocol keyword, raising on unsupported/unknown.

    Raising is *correct by design* (see design §4.2): an ``UNSUPPORTED`` line means
    we are in the wrong game; an unclassified line means our coverage is stale. Both
    should be **zero** across the fuzz corpus — they are tripwires, not a runtime path.
    """
    entry = MESSAGE_POLICY.get(keyword)
    if entry is None:
        raise UnknownMessageType(
            f"protocol keyword {keyword!r} is not in MESSAGE_POLICY — classify it "
            f"(EVENT / STATE_ONLY / CONTROL / COSMETIC / UNSUPPORTED) before shipping"
        )
    policy, reason = entry
    if policy is Policy.UNSUPPORTED:
        raise UnsupportedMessageType(f"protocol keyword {keyword!r}: {reason}")
    return policy, reason
