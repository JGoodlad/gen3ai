"""``ViewEventFolder`` — ONE ply's :class:`~agents.battle.battle_event.BattleEvent` stream,
folded from the ply's one-sided protocol WITHOUT poke-env's state tracker
(``gen3_view_event_fold_v1``).

Why this exists
---------------
``designs/rust_sim/one_sided_view.md`` DEFERRAL **D5**: the port's one-sided view hands Python a
successor's BOARD, so the read-models need no protocol replay — but the per-decision TRACKERS
(recency, pair history, the 32-row event window, the progress clock, the Hidden-Power belief) fold
the EVENT LOG, and a board is not an event log. Without them a view-path successor's observation
carries ZEROED tracker blocks, which is a silently different observation, which is why
``materialize_from_view()`` could not be built.

The event log is not available from the payload, but the **raw protocol of that one ply already
is** — ``expand_many`` returns the arm's own one-sided ``p1_chunks`` / ``p2_chunks``. So the ply's
events can be folded from bytes we are already being sent. What made that expensive before is that
the only fold in the tree ran *through* poke-env: ``Gen3Battle.parse_message`` calls
``super().parse_message`` for every line, and a per-arm poke-env battle means a per-arm CLONE of
the whole battle graph (measured 0.77 ms/arm of restore against a 0.65 ms/arm total for the whole
view path).

**The split this module rests on, and it is the contract's own split one level down.** Building a
``BattleEvent`` reads exactly FIVE facts about the board — a mon's species, its status, its HP
fraction, which mon is active on a side, and which side's move is currently resolving — plus the
turn number. Everything else on the event comes from the LINE. So this class keeps
:meth:`Gen3Battle._build_event` and :meth:`Gen3Battle._capture_pre` **verbatim, by inheritance**
and replaces only those five reads with a LIGHT BOARD folded from the same lines.

🚨 **There is no second event builder here, and that is deliberate.** Every ``value`` key, every
attribution rule, every one of the ``_build_event`` special cases (the Damp ``[of]`` redirection,
the Sleep-Talk delegation, the move-suffix ``[miss]`` synthetics, the effectiveness mover
fallback) is the inherited method running unchanged. A second implementation of that 50-keyword
schema is exactly how two paths come to disagree, and the schema is the part with the history of
findings. What IS new is the light board — six lines of state per mon — and the differential gate
``event_fold_parity_fuzz_test.py`` compares this fold's events against ``Gen3Battle``'s own on the
same bytes, field by field.

The seed is EXACT, not reconstructed
------------------------------------
:meth:`seed_from` copies the five facts out of the real ``Gen3Battle`` the shared prefix already
built, so the ply starts from the same board poke-env is on. Nothing is re-derived from the
prefix — the prefix is replayed once per decision either way, and re-folding it here would be a
second chance to disagree for no benefit.

What it does NOT do
-------------------
This is an EVENT fold, not a state tracker: after the ply its board is only as complete as the
five facts. The successor's BOARD comes from the one-sided view payload
(:mod:`agents.battle.view_adapter`), which is the whole point — the two are used together and
neither is asked for the other's job.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.battle.battle_event import OPP, OURS, BattleEvent, Policy, classify
from agents.battle.gen3_battle import Gen3Battle

#: Protocol keywords whose CONDITION field (``"52/100 par"``) is at index 3 and states both the
#: HP and the status of the mon named at index 2.
_CONDITION_AT_3 = frozenset({"switch", "drag", "replace", "-damage", "-heal", "-sethp"})

#: Keywords ``poke_env.player.Player._handle_battle_message`` handles ITSELF and never forwards
#: to ``Battle.parse_message`` — so they are absent from the live event log and must be absent
#: here. ``win`` / ``tie`` reach ``won_by`` / ``tied``, which emit no event; ``error`` reaches
#: ``record_choice_rejected`` OUT OF BAND and is not part of a search ply's protocol (a search
#: arm's choice is never rejected — the driver validates it).
_PLAYER_INTERCEPTED = frozenset(
    {"request", "showteam", "win", "tie", "error", "bigerror", "t:", "expire", "uhtmlchange"})


class _LightMon:
    """The four reads ``_capture_pre`` / ``_build_event`` make of a ``Pokemon``.

    ``status`` mirrors poke-env's ``Status`` enum only in the one property that is read —
    ``.name`` — because ``_capture_pre`` stores ``tmon.status.name`` and nothing downstream
    re-resolves it to the enum."""

    __slots__ = ("species", "status", "hp_fraction")

    def __init__(self, species: Optional[str], status: "Optional[_LightStatus]",
                 hp_fraction: float) -> None:
        self.species = species
        self.status = status
        self.hp_fraction = hp_fraction


class _LightStatus:
    """``Status``-shaped: only ``.name`` is ever read (``_capture_pre``)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


def _split_condition(cond: str) -> Tuple[float, Optional[str]]:
    """``"52/100 par"`` → ``(0.52, "PAR")``; ``"0 fnt"`` → ``(0.0, None)``.

    This is the wire's own spelling of ``Pokemon.current_hp_fraction`` — ``current_hp / max_hp``
    — and the numerator/denominator are exactly the pair poke-env stores, for our side (true
    integers) and the opponent's (the gen3ou ``ceil%`` fold) alike. A fainted mon reads 0.0 and
    poke-env clears its status, which ``Pokemon.faint`` does too."""
    parts = str(cond).strip().split(" ")
    hp_part = parts[0]
    status = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    if status == "fnt" or hp_part == "0":
        return 0.0, None
    if "/" in hp_part:
        num, _, den = hp_part.partition("/")
        try:
            d = float(den)
            return (float(num) / d if d else 0.0), (status.upper() if status else None)
        except ValueError:
            return 0.0, (status.upper() if status else None)
    try:
        return float(hp_part) / 100.0, (status.upper() if status else None)
    except ValueError:
        return 0.0, (status.upper() if status else None)


def _norm_ident(ident: str) -> str:
    """``"p1a: Nick"`` → ``"p1: Nick"`` — poke-env's own key normalisation
    (``AbstractBattle.get_pokemon``'s first three lines), so the light board is keyed the way the
    battle it is seeded from is keyed."""
    if len(ident) > 3 and ident[3] != " ":
        return ident[:2] + ident[3:]
    return ident


class ViewEventFolder(Gen3Battle):
    """Fold ONE ply's one-sided protocol lines into ``BattleEvent``s, poke-env-free.

    Construct with :meth:`seed_from`, feed the ply's chunks with :meth:`fold`, read the result
    with ``events_since(0)`` (the log starts empty, so that IS the ply)."""

    #: Set when a line names an ident the seed never saw AND no ``|switch|`` introduced it. It is
    #: a diagnostic, not a fallback: the species then follows poke-env's own "create from the
    #: identifier" rule, which is what ``get_pokemon`` does.
    unknown_idents: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mons: Dict[str, _LightMon] = {}
        self._active: Dict[str, Optional[str]] = {OURS: None, OPP: None}
        self.unknown_idents = 0

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def seed_from(cls, battle: Gen3Battle) -> "ViewEventFolder":
        """A folder standing exactly where ``battle`` stands, with an EMPTY event log.

        The ``_seq`` counter continues from the battle's so a folded event's ``seq`` keeps the
        ordering the live log would have given it; ``_turn`` and ``_current_move_user_side``
        carry over because a ply can open mid-turn (a replacement round)."""
        folder = cls(battle.battle_tag, battle.player_username, battle.logger, gen=3)
        folder._player_role = battle.player_role
        folder._turn = battle.turn
        folder._seq = battle._seq
        folder._current_move_user_side = battle._current_move_user_side
        for side, team in ((OURS, battle.team), (OPP, battle.opponent_team)):
            for ident, mon in team.items():
                key = _norm_ident(ident)
                folder._mons[key] = _LightMon(
                    species=mon.species,
                    status=_LightStatus(mon.status.name) if mon.status else None,
                    hp_fraction=float(mon.current_hp_fraction),
                )
                if mon.active:
                    folder._active[side] = key
        return folder

    def branch(self) -> "ViewEventFolder":
        """A folder standing where THIS one stands, with an empty log — one arm's starting point.

        A ply's arms all fork from the same board, and folding an arm MUTATES the five facts, so
        each arm needs its own. The copy is twelve small objects and six scalars, which is why
        the view road pays no per-arm battle clone at all: the whole reason the protocol road
        restores a pickled poke-env graph is that its fold mutates a state tracker, and this one
        does not have a state tracker to mutate.

        It is also what makes DEPTH work. A folder that has just folded ply *d* IS the board ply
        *d+1* forks from, so a deeper ply needs no new seed and no poke-env battle — which is the
        only reason the deepener can leave the protocol road behind."""
        other = ViewEventFolder(
            self.battle_tag, self.player_username, self.logger, gen=3)
        other._player_role = self._player_role
        other._turn = self._turn
        other._seq = self._seq
        other._current_move_user_side = self._current_move_user_side
        other._mons = {k: _LightMon(m.species, m.status, m.hp_fraction)
                       for k, m in self._mons.items()}
        other._active = dict(self._active)
        return other

    # ------------------------------------------------------------------ #
    # The five reads `_build_event` / `_capture_pre` make                 #
    # ------------------------------------------------------------------ #
    def _get_light(self, ident: Optional[str]) -> Optional[_LightMon]:
        if not ident:
            return None
        key = _norm_ident(ident)
        mon = self._mons.get(key)
        if mon is None:
            # poke-env's ``get_pokemon`` CREATES the mon from the identifier, taking the species
            # from the name part. Mirror that rather than returning None, so the two folds agree
            # on a line that names a mon neither has seen switch in.
            self.unknown_idents += 1
            from poke_env.data.normalize import to_id_str

            name = key.split(": ", 1)[1] if ": " in key else key
            mon = self._mons[key] = _LightMon(to_id_str(name), None, 1.0)
        return mon

    def get_pokemon(self, identifier: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._get_light(identifier)

    @property
    def active_pokemon(self) -> Any:            # type: ignore[override]
        return self._get_light(self._active[OURS]) if self._active[OURS] else None

    @property
    def opponent_active_pokemon(self) -> Any:   # type: ignore[override]
        return self._get_light(self._active[OPP]) if self._active[OPP] else None

    def _species_of(self, ident: Optional[str]) -> Optional[str]:
        mon = self._get_light(ident) if ident and len(ident) >= 2 and ident[0] == "p" else None
        return mon.species if mon else None

    def _active_species(self, side: Optional[str]) -> Optional[str]:
        if side is None:
            return None
        mon = self._get_light(self._active.get(side))
        return mon.species if mon else None

    def _hp_fraction(self, ident: str) -> float:
        mon = self._get_light(ident)
        return float(mon.hp_fraction) if mon else 0.0

    # ------------------------------------------------------------------ #
    # The fold                                                            #
    # ------------------------------------------------------------------ #
    def fold(self, chunks: Sequence[str]) -> List[BattleEvent]:
        """Feed the ply's one-sided chunks; return the events they produced, in order.

        🚨 **The line FILTER is poke-env's, not a convenience.** ``Player._handle_battle_message``
        intercepts seven keywords BEFORE ``parse_message`` ever sees them — ``request``,
        ``showteam``, ``win``, ``tie``, ``error``, ``bigerror`` and the ``MESSAGES_TO_IGNORE``
        set — so a fold that handed them to ``classify`` would raise ``UnknownMessageType`` on
        the first ``|request|`` of every ply. The live battle's log does not contain them either,
        which is what makes skipping them the FAITHFUL move rather than a dropped fact."""
        start = len(self._events)
        for chunk in chunks:
            for line in str(chunk).split("\n"):
                if not line.startswith("|"):
                    continue
                sm = line.split("|")
                if len(sm) < 2 or sm[1] in _PLAYER_INTERCEPTED:
                    continue
                self.parse_message(sm)
        return self._events[start:]

    def parse_message(self, split_message: List[str]) -> None:
        """``Gen3Battle.parse_message``'s shape with the LIGHT board in place of poke-env's.

        The scaffolding is spelled out rather than delegated because the one thing that must
        change is the mutation call in the middle: ``Gen3Battle`` calls
        ``super().parse_message`` (poke-env's whole state tracker), this calls :meth:`_apply`
        (the five facts). Classification, the pre-capture, the event build, the record and the
        move-suffix synthetics are all the inherited ones."""
        self._state_epoch += 1
        keyword = split_message[1] if len(split_message) > 1 else ""
        policy, _reason = classify(keyword)     # raises on UNSUPPORTED / UNKNOWN, as live
        self._lines_seen += 1
        self._policy_counts[policy] += 1

        if policy is not Policy.EVENT:
            self._apply(keyword, split_message)
            if keyword == "turn":
                self._turn_start.setdefault(self._turn, len(self._events))
            return

        pre = self._capture_pre(keyword, split_message)
        self._apply(keyword, split_message)
        ev = self._build_event(keyword, split_message, pre)
        if ev is not None:
            self._record(ev)
        if keyword == "move":
            for extra in self._move_suffix_events(split_message):
                self._record(extra)

    def _apply(self, keyword: str, sm: List[str]) -> None:
        """Mutate the five facts, exactly as the line would mutate poke-env's board.

        Only the keywords that move one of the five appear here; every other line is inert for
        this fold BY CONSTRUCTION, not by omission — ``_build_event`` reads nothing else."""
        if keyword == "turn":
            if len(sm) > 2 and sm[2].isdigit():
                self._turn = int(sm[2])
            return
        if keyword == "move":
            # The resolving side owns every following |-crit| / |-miss| / |-fail| /
            # effectiveness line until the next |move|. poke-env sets this off the ident's
            # PREFIX, not off the team lookup, so the two cannot disagree on a Zoroark.
            if len(sm) > 2 and len(sm[2]) >= 2:
                self._current_move_user_side = OURS if sm[2][:2] == self._player_role else OPP
            return
        if keyword in ("switch", "drag", "replace"):
            ident = sm[2] if len(sm) > 2 else ""
            if not ident:
                return
            key = _norm_ident(ident)
            side = self._side_of(ident)
            species = self._species_from_details(sm[3] if len(sm) > 3 else "")
            hp, status = _split_condition(sm[4] if len(sm) > 4 else "100/100")
            mon = self._mons.get(key)
            if mon is None:
                mon = self._mons[key] = _LightMon(species, None, hp)
            mon.species = species or mon.species
            mon.hp_fraction = hp
            mon.status = _LightStatus(status) if status else None
            if side is not None and keyword != "replace":
                self._active[side] = key
            elif side is not None:
                # `replace` (Illusion break) swaps WHICH mon is active without a switch.
                self._active[side] = key
            return
        if keyword in ("-damage", "-heal", "-sethp"):
            mon = self._get_light(sm[2] if len(sm) > 2 else None)
            if mon is None:
                return
            hp, status = _split_condition(sm[3] if len(sm) > 3 else "")
            mon.hp_fraction = hp
            # A `|-damage|` line RESTATES the status; poke-env's `set_hp_status` writes it, and
            # clears nothing when the field is absent.
            if status:
                mon.status = _LightStatus(status)
            elif hp == 0.0:
                mon.status = None
            return
        if keyword == "-status":
            mon = self._get_light(sm[2] if len(sm) > 2 else None)
            if mon is not None and len(sm) > 3 and sm[3]:
                mon.status = _LightStatus(sm[3].strip().upper())
            return
        if keyword in ("-curestatus", "-cureteam"):
            if keyword == "-cureteam":
                for m in self._mons.values():
                    m.status = None
                return
            mon = self._get_light(sm[2] if len(sm) > 2 else None)
            if mon is not None:
                mon.status = None
            return
        if keyword == "faint":
            mon = self._get_light(sm[2] if len(sm) > 2 else None)
            if mon is not None:
                mon.hp_fraction = 0.0
                mon.status = None
            return
        if keyword in ("detailschange", "-formechange"):
            mon = self._get_light(sm[2] if len(sm) > 2 else None)
            species = self._species_from_details(sm[3] if len(sm) > 3 else "")
            if mon is not None and species:
                mon.species = species
            return

    @staticmethod
    def _species_from_details(details: str) -> Optional[str]:
        """``"Blissey, F"`` → ``"blissey"`` — poke-env takes the species from the DETAILS string
        (``Pokemon._update_from_details``), never from the ident, which is the nickname."""
        from poke_env.data.normalize import to_id_str

        head = str(details).split(",")[0].strip()
        return to_id_str(head) if head else None
