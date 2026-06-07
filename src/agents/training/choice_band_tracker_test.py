"""Unit tests for ChoiceBandTracker — the narrowing math on hand-built event logs.

Mirrors ``hidden_power_tracker_test.py``: pure, fast, no battle. Builds opponent-side
:class:`BattleEvent` sequences + per-mon item state by hand and asserts the re-derived ``p_cb``,
including the bug-vs-data-gap ValueError self-check.
"""
import pytest

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind
from agents.training.choice_band_tracker import CHOICE_BAND, ChoiceBandTracker

# A small explicit prior pool so tests are independent of the on-disk usage stats.
PRIORS = {
    "heracross": {"choiceband": 0.20, "leftovers": 0.50, "salacberry": 0.30},
    "snorlax": {"choiceband": 0.05, "leftovers": 0.80, "chestoberry": 0.15},
    "aerodactyl": {"choiceband": 0.76, "leftovers": 0.24},
    # 'nopriormon' deliberately absent → prior P(CB) == 0.0
}


def make_tracker() -> ChoiceBandTracker:
    return ChoiceBandTracker(_priors=PRIORS)


# ---- event-builder helpers (opponent side by default) ------------------------ #
_SEQ = [0]


def _ev(kind, species, value, side=OPP, raw=()):
    _SEQ[0] += 1
    return BattleEvent(seq=_SEQ[0], turn=1, kind=kind, side=side,
                       actor_species=species, value=dict(value), raw=tuple(raw))


def mv(species, move_id, from_move=None, side=OPP):
    """A MOVE event in the MODERN ``[from]move: X`` form (delegation carried on ``from_move``)."""
    val = {"move_id": move_id}
    raw = ("", "move", f"p2a: {species}", move_id)
    if from_move is not None:
        val["from_move"] = from_move
        raw = raw + ("p1a: X", f"[from]move: {from_move}")
    return _ev(EventKind.MOVE, species, val, side=side, raw=raw)


def mv_gen3_called(species, move_id, delegator, side=OPP):
    """A delegated MOVE in the bundled-gen3 ``[from] <MoveName>`` form: ``from_move`` is None (the
    shared parser doesn't capture this wire form), so only the raw line marks the delegation."""
    raw = ("", "move", f"p2a: {species}", move_id, "p1a: X", f"[from] {delegator}")
    return _ev(EventKind.MOVE, species, {"move_id": move_id}, side=side, raw=raw)


def switch(species, side=OPP, drag=False):
    return _ev(EventKind.DRAG if drag else EventKind.SWITCH, species,
               {"prev_active": None}, side=side)


def item_ev(species, item, side=OPP, end=False):
    return _ev(EventKind.ENDITEM if end else EventKind.ITEM, species,
               {"item": item}, side=side)


def obs(tracker, species, held=None, consumed=None, events=()):
    """Convenience: feed one opponent mon's item state + the event log, return p_cb."""
    tracker.observe({species: (held, consumed)}, list(events))
    return tracker.p_cb(species)


# ---------------------------------------------------------------------------
# Prior seed / no-evidence
# ---------------------------------------------------------------------------

def test_prior_seed_no_evidence():
    """With no item revealed and ≤1 move, p_cb is the bare usage prior and not 'known'."""
    t = make_tracker()
    p = obs(t, "heracross", events=[switch("heracross")])
    assert p == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_unobserved_species_returns_prior():
    """p_cb for a species never passed to observe() still returns its prior (no-evidence belief)."""
    t = make_tracker()
    t.observe({}, [])
    assert t.p_cb("aerodactyl") == pytest.approx(0.76)
    assert t.is_known("aerodactyl") is False


def test_species_without_prior_entry_returns_zero():
    """A species with no usage entry has prior P(CB)=0 (not a crash, not a flat fallback)."""
    t = make_tracker()
    assert obs(t, "nopriormon", events=[switch("nopriormon")]) == pytest.approx(0.0)
    assert t.is_known("nopriormon") is False


# ---------------------------------------------------------------------------
# Move-based elimination (the choice-lock tell)
# ---------------------------------------------------------------------------

def test_two_distinct_moves_one_stint_eliminates():
    """≥2 distinct selected moves in one uninterrupted stint ⇒ not a Choice item ⇒ p_cb 0."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "megahorn"), mv("heracross", "rockslide")]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.0)
    assert t.is_known("heracross") is True


def test_lone_move_not_eliminated_even_status():
    """A single selected move — even a status move (the lock CAN be status) — is NOT elimination."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "swordsdance")]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_repeat_same_move_not_eliminated():
    """Using the same move twice (a CB mon locked in, or Encore forcing a repeat) is 1 distinct."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "megahorn"), mv("heracross", "megahorn")]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_two_distinct_across_switch_not_eliminated():
    """Two distinct moves split across a switch (CB allows a different move each stint) ⇒ NOT
    eliminated — the per-stint window resets on SWITCH/DRAG."""
    t = make_tracker()
    ev = [
        switch("heracross"), mv("heracross", "megahorn"),
        switch("snorlax"), mv("snorlax", "bodyslam"),
        switch("heracross"), mv("heracross", "rockslide"),  # different move, new stint
    ]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_two_distinct_across_drag_not_eliminated():
    """Same as the switch case but the return is a phaze (DRAG) — also resets the stint."""
    t = make_tracker()
    ev = [
        switch("heracross"), mv("heracross", "megahorn"),
        switch("snorlax"),
        switch("heracross", drag=True), mv("heracross", "rockslide"),
    ]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Delegation (Sleep Talk / Metronome) and Struggle — NOT real selections
# ---------------------------------------------------------------------------

def test_sleep_talk_delegation_not_counted():
    """Sleep Talk → (different called moves) is ONE selection (Sleep Talk). The called moves carry
    ``from_move`` and must NOT count as distinct selections — else a CB user gets false-eliminated."""
    t = make_tracker()
    ev = [
        switch("snorlax"),
        mv("snorlax", "sleeptalk"),
        mv("snorlax", "earthquake", from_move="sleeptalk"),
        mv("snorlax", "sleeptalk"),
        mv("snorlax", "bodyslam", from_move="sleeptalk"),  # different called move, still Sleep Talk
    ]
    # Distinct SELECTED = {sleeptalk} → 1 → not eliminated.
    assert obs(t, "snorlax", events=ev) == pytest.approx(0.05)
    assert t.is_known("snorlax") is False


def test_sleep_talk_gen3_wire_form_not_counted():
    """The bundled gen3 sim emits the called move as ``[from] <MoveName>`` (no ``move:`` prefix), so
    ``from_move`` is None — the tracker must still recognise delegation from the raw line. Two
    DIFFERENT called moves under one Sleep Talk must NOT read as 2 selections (the regression the
    fuzz surfaced)."""
    t = make_tracker()
    ev = [
        switch("heracross"),
        mv("heracross", "sleeptalk"),
        mv_gen3_called("heracross", "megahorn", "Sleep Talk"),
        mv("heracross", "sleeptalk"),
        mv_gen3_called("heracross", "rockslide", "Sleep Talk"),  # different called move
    ]
    # Distinct SELECTED = {sleeptalk} → 1 → not eliminated (would be 2 if the gen3 form leaked).
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_pursuit_on_switch_counts_as_a_selection():
    """Pursuit hitting a switching target tags its OWN line ``[from] Pursuit`` (the move's
    sourceEffect is itself), NOT a delegation. It is a real free selection and must count — so a
    non-CB mon that used another move then Pursuit-on-switch IS eliminated. (Regression: treating
    every ``[from]`` as delegation silently under-counts and never eliminates a Pursuit user.)"""
    t = make_tracker()
    ev = [
        switch("heracross"),
        mv("heracross", "megahorn"),
        mv_gen3_called("heracross", "pursuit", "Pursuit"),  # [from] Pursuit names itself
    ]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.0)  # 2 real distinct → eliminated
    assert t.is_known("heracross") is True


def test_lockedmove_continuation_not_a_second_selection():
    """An Outrage / Solar-Beam release continuation is marked ``[from] lockedmove`` — the SAME move
    already counted from its selection turn. It must never inflate the distinct-selection count."""
    t = make_tracker()
    ev = [
        switch("heracross"),
        mv("heracross", "outrage"),                          # the selection turn
        mv_gen3_called("heracross", "outrage", "lockedmove"),  # forced continuation
    ]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)  # still 1 distinct → not eliminated
    assert t.is_known("heracross") is False


def test_snatch_stolen_move_not_a_selection():
    """A move executed via Snatch (``[from] Snatch`` names a DIFFERENT move) is the snatcher running
    the stolen move, not its own free selection — it must not count toward the choice-lock."""
    t = make_tracker()
    ev = [
        switch("heracross"),
        mv("heracross", "swordsdance"),
        mv_gen3_called("heracross", "calmmind", "Snatch"),  # stolen via Snatch
    ]
    # Free selections = {swordsdance} (calmmind came via Snatch) → 1 → not eliminated.
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_struggle_not_counted():
    """Struggle (forced, 0 PP) is not a free selection — A then Struggle is 1 distinct, not 2."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "megahorn"), mv("heracross", "struggle")]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_struggle_does_not_block_a_real_double():
    """Two real distinct moves still eliminate even with a Struggle mixed in."""
    t = make_tracker()
    ev = [
        switch("heracross"), mv("heracross", "megahorn"),
        mv("heracross", "rockslide"), mv("heracross", "struggle"),
    ]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Item-based elimination (held / consumed)
# ---------------------------------------------------------------------------

def test_held_non_cb_item_eliminates():
    """A revealed held non-CB item (e.g. Leftovers passive heal reveal) ⇒ p_cb 0."""
    t = make_tracker()
    assert obs(t, "snorlax", held="leftovers", events=[switch("snorlax")]) == pytest.approx(0.0)
    assert t.is_known("snorlax") is True


def test_consumed_item_eliminates():
    """A consumed item (Berry eaten — item now None, consumed set) ⇒ p_cb 0. CB is never consumed."""
    t = make_tracker()
    assert obs(t, "heracross", held=None, consumed="salacberry",
               events=[switch("heracross")]) == pytest.approx(0.0)
    assert t.is_known("heracross") is True


def test_knocked_off_itemless_eliminates():
    """Knock Off / Thief leaves the victim itemless (item None, consumed set in poke-env) ⇒ 0."""
    t = make_tracker()
    ev = [switch("snorlax"), item_ev("snorlax", "leftovers", end=True)]
    assert obs(t, "snorlax", held=None, consumed="leftovers", events=ev) == pytest.approx(0.0)
    assert t.is_known("snorlax") is True


def test_held_item_checked_before_moves():
    """A revealed non-CB item eliminates even if only one move has been seen."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "megahorn")]
    assert obs(t, "heracross", held="leftovers", events=ev) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Item changes hands — the bidirectional (non-monotone) cases
# ---------------------------------------------------------------------------

def test_trick_hands_cb_raises_to_one():
    """Trick / Switcheroo handing the mon Choice Band ⇒ item now KNOWN to be CB ⇒ p_cb rises to 1."""
    t = make_tracker()
    ev = [switch("snorlax"), item_ev("snorlax", CHOICE_BAND)]
    assert obs(t, "snorlax", held=CHOICE_BAND, consumed=None, events=ev) == pytest.approx(1.0)
    assert t.is_known("snorlax") is True


def test_trick_swaps_cb_away_eliminates():
    """A mon that Tricked its Choice Band away now holds the swapped-in (non-CB) item ⇒ p_cb 0."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "trick"), item_ev("heracross", "leftovers")]
    assert obs(t, "heracross", held="leftovers", consumed=None, events=ev) == pytest.approx(0.0)


def test_belief_can_rise_then_re_derive_idempotent():
    """p_cb is idempotent under re-observation even across a raising event (not a one-way ratchet)."""
    t = make_tracker()
    ev = [switch("snorlax"), item_ev("snorlax", CHOICE_BAND)]
    state = {"snorlax": (CHOICE_BAND, None)}
    t.observe(state, ev)
    first = t.p_cb("snorlax")
    t.observe(state, ev)
    assert t.p_cb("snorlax") == first == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The window-reset on item change (prevents a FALSE contradiction after Trick→CB)
# ---------------------------------------------------------------------------

def test_trick_to_cb_after_two_moves_no_false_contradiction():
    """A mon that used 2 distinct moves, THEN received Choice Band via Trick (an ITEM event that
    resets the move-window), then locked into one move — must read p_cb 1.0, NOT raise. The
    pre-Trick moves don't pertain to the newly-acquired item."""
    t = make_tracker()
    ev = [
        switch("snorlax"),
        mv("snorlax", "bodyslam"),
        mv("snorlax", "earthquake"),            # 2 distinct BEFORE the trick
        item_ev("snorlax", CHOICE_BAND),        # receives CB → window resets here
        mv("snorlax", "bodyslam"),              # now locked, 1 distinct in the new window
    ]
    assert obs(t, "snorlax", held=CHOICE_BAND, consumed=None, events=ev) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Side isolation
# ---------------------------------------------------------------------------

def test_our_side_moves_ignored():
    """Only opponent-side events drive the belief — a same-named mon on our side is irrelevant."""
    t = make_tracker()
    ev = [
        switch("heracross", side=OURS),
        mv("heracross", "megahorn", side=OURS),
        mv("heracross", "rockslide", side=OURS),  # 2 distinct, but OUR side
        switch("heracross", side=OPP),
    ]
    assert obs(t, "heracross", events=ev) == pytest.approx(0.20)  # not eliminated by our moves


# ---------------------------------------------------------------------------
# The bug-vs-data-gap ValueError self-check
# ---------------------------------------------------------------------------

def test_known_cb_with_two_distinct_moves_raises():
    """A mon known to hold Choice Band that nonetheless shows ≥2 distinct selected moves in the
    same item-window is impossible under the choice-lock — the tracker raises (mirrors the HP
    tracker's all-candidates-eliminated self-check). This requires NO intervening item/switch
    event to reset the window, so it can only mean a tracker bug or an unrecorded item event."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "megahorn"), mv("heracross", "rockslide")]
    with pytest.raises(ValueError, match="choice-lock"):
        t.observe({"heracross": (CHOICE_BAND, None)}, ev)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_idempotent():
    """Re-observing the same (item_state, events) yields the same belief."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "megahorn"), mv("heracross", "rockslide")]
    state = {"heracross": (None, None)}
    t.observe(state, ev)
    first = t.p_cb("heracross")
    t.observe(state, ev)
    assert t.p_cb("heracross") == first == pytest.approx(0.0)


def test_reset_clears_state():
    """After reset(), a previously-eliminated species reverts to its prior and is no longer known."""
    t = make_tracker()
    ev = [switch("heracross"), mv("heracross", "megahorn"), mv("heracross", "rockslide")]
    t.observe({"heracross": (None, None)}, ev)
    assert t.is_known("heracross") is True
    t.reset()
    assert t.p_cb("heracross") == pytest.approx(0.20)
    assert t.is_known("heracross") is False


def test_multiple_species_tracked_independently():
    """Each species derives from its own stint + item state in one observe() call."""
    t = make_tracker()
    ev = [
        switch("heracross"), mv("heracross", "megahorn"), mv("heracross", "rockslide"),  # → 0
        switch("aerodactyl"), mv("aerodactyl", "rockslide"),  # 1 move → prior
    ]
    t.observe({"heracross": (None, None), "aerodactyl": (None, None)}, ev)
    assert t.p_cb("heracross") == pytest.approx(0.0)
    assert t.p_cb("aerodactyl") == pytest.approx(0.76)
