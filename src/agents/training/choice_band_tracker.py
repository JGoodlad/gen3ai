"""Per-opponent-species Choice-Band belief tracker (System A of the hidden-attribute
inference design, ``designs/ai_v5/design_hidden_attribute_inference.md`` §3).

Choice Band is **never revealed by a protocol message** the way Hidden Power's type is, so the
incoming-damage block can't price its ×1.5 physical boost without a *belief* about whether the
opponent is even holding it. This tracker maintains, per opponent species this episode, a
calibrated ``p_cb ∈ [0, 1]`` — seeded from the Smogon item usage prior and **re-derived every
decision** from the currently-observed state (held/consumed item + the moves selected this
"stint"). It mirrors :class:`~agents.training.hidden_power_tracker.HiddenPowerTracker`: owned by
``EpisodeTracker``, updated in ``record()`` *before* the encoder runs, fed to the model as a
calibrated belief (never an argmax), excluded from snapshot/restore.

What proves a mon is NOT holding Choice Band (Gen 3, Showdown sim):
  * **A non-CB item is held or was consumed.** Items are mutually exclusive. A passive reveal
    (Leftovers/Sitrus heal sets ``mon.item``; a popped Berry / Knock-Off / Thief-victim sets
    ``mon.consumed_item`` and clears ``mon.item``) ⇒ ``p_cb = 0``. We read **both** ``item`` and
    ``consumed_item``.
  * **≥2 distinct SELECTED moves in one uninterrupted stint.** In the Showdown sim a Choice item
    locks the holder into its first selected move via the ``choicelock`` volatile until it
    switches; selecting a second distinct move is impossible. **Choice Band is the only Choice
    item in Gen 3** (Scarf/Specs are gen 4), so any choice-lock violation unambiguously implies
    "not CB" — no disambiguation needed.

What raises it:
  * **The item is *known* to be Choice Band** — e.g. Trick / Switcheroo / Thief handed it over
    (``mon.item == 'choiceband'``) ⇒ ``p_cb = 1`` (and the mon is now choice-locked).

So the belief is **idempotent, not monotone**: Trick can *raise* ``p_cb``, and it is re-derived
from the cumulative observation each decision rather than ratcheted one way. Re-running the
derivation on the same events is a no-op ⇒ it is excluded from snapshot/restore for the same
reason the HP tracker is, just via idempotent re-derivation rather than monotone elimination.

Sim-vs-cartridge note (do NOT delete on a "this is wrong for gen3" refactor): the move-lock
is a **Showdown-sim mechanic**, not cartridge Gen-3 (real Gen-3 Choice Band only boosts Atk;
the choice-lock is a later-gen mechanic Showdown also applies to its gen3 mod). It is valid here
*only* because we exclusively play the Showdown sim. Verified against
``deps/pokemon-showdown/data/conditions.ts`` (``choicelock``) + ``data/items.ts`` (``choiceband``
``gen: 3``, ``isChoice: true``; Scarf/Specs ``gen: 4``) — never modern-gen knowledge.

There is **no protocol tag** for the opponent's lock (Showdown's ``[from] lockedmove`` suffix is
for *our own* forced moves only); the tell is **reconstructed from the cumulative move log**, not
detected. A lone status move is NOT an elimination — the lock can be a status move; only ≥2
*distinct* selections (any category) prove it.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

from agents.battle.battle_event import OPP, BattleEvent, EventKind, from_clause_move_source

# The id-form of the one Choice item legal in Gen 3 (Scarf/Specs are gen 4). Matches the
# ``item`` / ``consumed_item`` id form poke-env stores (``to_id_str("Choice Band")``).
CHOICE_BAND = "choiceband"

# Forced 0-PP move — not a real selection, so it never counts toward the distinct-move set.
_STRUGGLE = "struggle"


def _is_delegated(event: BattleEvent) -> bool:
    """True if a MOVE event is a *delegated execution* — the move was CALLED by a DIFFERENT move
    (Sleep Talk / Metronome → X, or a Snatch-stolen move) — rather than a free selection.

    Reads the move-call off the shared :func:`~agents.battle.battle_event.from_clause_move_source`
    wire parser (NOT ``event.from_move``, which is unreliable for gen3 — see that function), then
    applies the free-selection reading: delegation ⇔ the ``[from]`` names a move *different* from
    the one that executed. A ``[from]`` that names the SAME move (Pursuit-on-switch's self-tag
    ``[from] Pursuit``) is the mon's own free selection ⇒ keep it; a ``lockedmove`` continuation
    marker names a non-move and so reads as delegated ⇒ skipped, which is harmless (the locked move
    was already counted from its selection turn). Counting Pursuit-on-switch as delegation (the
    original bug) silently under-counts selections and never eliminates a non-CB Pursuit user."""
    src = from_clause_move_source(event.raw)
    return src is not None and src != event.move_id

# An opponent's per-mon item state as the tracker reads it: (held_item_id, consumed_item_id),
# both id-form or None. In production this is ``(LivePokemon.item, LivePokemon.consumed_item)``.
ItemState = Tuple[Optional[str], Optional[str]]


class ChoiceBandTracker:
    """Tracks a per-opponent-species Choice-Band belief within one episode.

    Call :meth:`observe` each decision with the opponent's current per-mon item state and the
    cumulative battle event log; it re-derives ``p_cb`` for every observed species. Read
    :meth:`p_cb` for the calibrated belief and :meth:`is_known` for whether that belief rests on
    in-battle evidence (item revealed/consumed, or move-eliminated) vs the bare prior.
    """

    def __init__(self, _priors: Optional[dict] = None) -> None:
        """``_priors`` (tests): ``{species: {item_id: prob}}`` — bypasses the data facade. Default
        loads the project's Smogon item usage priors via ``gen3_data.priors`` (parsed once, shared).
        """
        if _priors is not None:
            self._priors: dict = _priors
        else:
            from agents.gen3_data import priors as _gen3_priors
            self._priors = _gen3_priors.item_raw()
        # Re-derived from scratch each observe() — so they always reflect the cumulative log.
        self._p_cb: dict[str, float] = {}
        self._known: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Update                                                              #
    # ------------------------------------------------------------------ #
    def observe(
        self,
        item_state: Mapping[str, ItemState],
        events: Sequence[BattleEvent],
    ) -> None:
        """Re-derive ``p_cb`` for every opponent species in ``item_state`` from the cumulative
        event log + current item state.

        ``item_state`` maps each revealed opponent species → ``(held_item, consumed_item)`` (both
        id-form or None). ``events`` is the whole-battle event log (``battle.events_since(0)``);
        only opponent-side MOVE / SWITCH / DRAG / ITEM / ENDITEM events are consulted.

        Contract: ``item_state`` must include **every** opponent species you want a belief for.
        Beliefs are re-derived from scratch each call (the dicts are cleared), so a species omitted
        from ``item_state`` falls back to its bare prior. The production caller builds it from
        ``live.opp.mons`` (all revealed mons — a monotonically growing set), so a once-eliminated
        mon never silently reverts; a partial ``item_state`` would.

        Pure function of (``item_state``, ``events``) ⇒ **idempotent**: calling it again with the
        same (or a grown) log produces the same beliefs, which is why it needs no snapshot/restore.

        Raises ``ValueError`` on a contradiction (a mon known to hold Choice Band that nonetheless
        selected ≥2 distinct moves since acquiring it) — a should-never-happen tripwire that means
        either a tracker bug or a sim/data gap, mirroring the HP tracker's self-check.
        """
        move_window = self._distinct_moves_per_stint(events)
        self._p_cb.clear()
        self._known.clear()
        for species, (held, consumed) in item_state.items():
            n_distinct = len(move_window.get(species, ()))
            self._p_cb[species], self._known[species] = self._derive(
                species, held, consumed, n_distinct
            )

    @staticmethod
    def _distinct_moves_per_stint(
        events: Sequence[BattleEvent],
    ) -> dict[str, set]:
        """Per opponent species, the set of distinct **selected** move ids in its current
        item-window — the moves it has chosen since the later of (its last switch-in, its last
        item change). Reconstructed by walking the cumulative log once.

        The window resets on the species' SWITCH / DRAG (a new stint clears the choice-lock) and
        on its ITEM / ENDITEM (an item change means the move-evidence no longer pertains to the
        *current* item — e.g. a mon Tricked a Choice Band must not be "eliminated" by moves it
        selected before receiving it).

        Excluded from the count (they are not free selections that could break a choice-lock):
          * **delegated** executions — Sleep Talk / Metronome call another move; the SELECTION was
            the delegating move (its own MOVE event), not the called one (see :func:`_is_delegated`,
            which handles both the modern ``[from]move: X`` and the gen3 ``[from] X`` wire forms).
          * **Struggle** — forced when all PP is gone.
        """
        window: dict[str, set] = {}
        for e in events:
            if e.side != OPP:
                continue
            if e.kind in (
                EventKind.SWITCH,
                EventKind.DRAG,
                EventKind.ITEM,
                EventKind.ENDITEM,
            ):
                if e.actor_species:
                    window[e.actor_species] = set()
            elif e.kind is EventKind.MOVE:
                if (
                    e.actor_species
                    and e.move_id
                    and e.move_id != _STRUGGLE
                    and not _is_delegated(e)
                ):
                    window.setdefault(e.actor_species, set()).add(e.move_id)
        return window

    def _prior(self, species: str) -> float:
        return float(self._priors.get(species, {}).get(CHOICE_BAND, 0.0))

    def _derive(
        self, species: str, held: Optional[str], consumed: Optional[str], n_distinct: int
    ) -> Tuple[float, bool]:
        """Return ``(p_cb, is_known)`` for one species from its currently-observed state.

        Order matters — a *known* item dominates the move evidence, and a known Choice Band is the
        only thing that can raise the belief above the prior."""
        if held == CHOICE_BAND:
            # Known to hold Choice Band (revealed via Trick / Thief). It is now choice-locked, so
            # it cannot have selected ≥2 distinct moves since acquiring it (any reveal is an ITEM
            # event, which resets the move-window). A contradiction here is a should-never-happen
            # tripwire — fail loud rather than emit an incoherent belief, mirroring the HP tracker.
            #
            # STEP-3 NOTE (production wiring): the only known way this is reachable on a *legal*
            # battle is the astronomically-rare combo of a Trick-revealed Choice Band whose first
            # post-Trick move was Magic-Coat-bounced / Snatched (the sim's choicelock skips
            # bounced/snatched moves) AND a 2nd distinct move with no intervening switch — and only
            # if those "Past"-tier moves are even present. The 20-min fuzz confirms it never fires
            # across the team pool, so the raise is a safe tripwire here. When this tracker is owned
            # by ``EpisodeTracker.record()``, wrap the ``observe`` call to catch+log+clamp (trust the
            # revealed item → 1.0) rather than let a raise abort a SubprocVecEnv rollout.
            if n_distinct >= 2:
                raise ValueError(
                    f"ChoiceBandTracker: '{species}' is known to hold Choice Band yet selected "
                    f"{n_distinct} distinct moves since acquiring it — impossible under the "
                    f"choice-lock. Either a tracker bug (move-window not reset on the item event) "
                    f"or a sim/data gap (the item-change event was not recorded)."
                )
            return 1.0, True
        if held is not None:
            # A known, held, non-CB item.
            return 0.0, True
        if consumed is not None:
            # The held item was spent / removed (Berry eaten, Knock Off, Thief, Trick-away) — the
            # mon either ate a non-CB consumable or is now itemless. Either way: not Choice Band.
            return 0.0, True
        if n_distinct >= 2:
            # ≥2 distinct selected moves in one stint — impossible under any Choice item.
            return 0.0, True
        return self._prior(species), False

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #
    def p_cb(self, species: str) -> float:
        """Calibrated P(``species`` holds Choice Band) ∈ [0, 1]. Falls back to the bare usage
        prior for a species not yet observed this episode (the natural no-evidence belief)."""
        if species in self._p_cb:
            return self._p_cb[species]
        return self._prior(species)

    def is_known(self, species: str) -> bool:
        """True once the belief rests on in-battle evidence — the item was revealed/consumed, or
        the mon was move-eliminated (or confirmed CB). False while ``p_cb`` is still the prior."""
        return self._known.get(species, False)

    def reset(self) -> None:
        self._p_cb.clear()
        self._known.clear()
