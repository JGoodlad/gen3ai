"""``Gen3Battle.live_view()``'s one-slot memo — the INVALIDATION contract.

``gen3_live_view_memo_v1``. The memo is a pure speed change (a production decision built
the same immutable view five times); its whole risk is serving a view of a state that has
since moved. Every test here is a NAMED regression for one door of the invalidation proof
in :meth:`Gen3Battle.live_view` — each must FAIL if its bump is removed.

Two of them are not "does the cache work" tests at all:

* :func:`test_a_view_built_across_a_mutation_is_never_served` pins the *store* discipline
  (capture the epoch BEFORE building, store under it), which is what makes the memo safe
  even if a writer ran mid-build.
* :func:`test_a_deep_copied_battle_does_not_serve_its_twins_view` pins the clone-aliasing
  hazard the offline materializer's per-arm ``deepcopy`` restore lives on: a cache keyed by
  battle identity rather than carried ON the battle would serve arm-1's forward state to a
  rewound arm-2.
"""

import copy
import logging

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LiveView

LOG = logging.getLogger("live-view-memo-test")

_SETUP = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gen", "3"],
    ["", "start"],
    ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
    ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
    ["", "turn", "1"],
]


def _battle(extra=()):
    b = Gen3Battle("battle-gen3ou-memo", "p1user", LOG, gen=3)
    for line in list(_SETUP) + list(extra):
        b.parse_message(line)
    return b


def _fresh(battle) -> LiveView:
    """The oracle: a full rebuild, bypassing the memo entirely."""
    return LiveView.from_battle(battle)


# ──────────────────────────── the memo itself ──────────────────────────────
def test_two_reads_with_no_mutation_share_one_view():
    b = _battle()
    first = b.live_view()
    assert b.live_view() is first, "an unchanged battle must not rebuild the view"


def test_a_served_view_always_equals_a_fresh_rebuild():
    """The contract is byte-identity, not merely 'a view'. Checked at every step of a
    small game so a memo that survives one mutation too long is caught."""
    b = Gen3Battle("battle-gen3ou-memo", "p1user", LOG, gen=3)
    for line in _SETUP + [
        ["", "-weather", "Sandstorm"],
        ["", "-sidestart", "p2: p2user", "Spikes"],
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "52/100"],
        ["", "-status", "p2a: Tyra", "par"],
        ["", "-start", "p2a: Tyra", "Leech Seed"],
        ["", "-boost", "p1a: Zappy", "spa", "2"],
        ["", "turn", "2"],
        ["", "-enditem", "p2a: Tyra", "Leftovers"],
        ["", "turn", "3"],
    ]:
        b.parse_message(line)
        assert b.live_view() == _fresh(b), f"stale view after {line!r}"


# ───────────────────────── door 1: parse_message ───────────────────────────
def test_an_event_line_invalidates():
    b = _battle()
    before = b.live_view()
    b.parse_message(["", "-damage", "p2a: Tyra", "40/100"])
    after = b.live_view()
    assert after is not before
    assert after.opp.active.hp_fraction == 0.4
    assert after == _fresh(b)


def test_a_CONTROL_line_invalidates_even_though_it_is_not_an_event():
    """``|turn|`` mutates ``battle.turn`` — which LiveView carries, and which the weather
    fold's ``turns_active`` is computed from — while being ``Policy.CONTROL``. A key over
    ``len(events)`` alone would miss it."""
    b = _battle([["", "-weather", "RainDance"]])
    before = b.live_view()
    n_events = len(b.events)
    b.parse_message(["", "turn", "2"])
    assert len(b.events) == n_events, "the |turn| line must not have appended an event"
    after = b.live_view()
    assert after is not before
    assert after.turn == 2 and before.turn == 1
    assert after.weather.turns_active == 1
    assert after == _fresh(b)


def test_a_teamsize_line_invalidates():
    """``|teamsize|`` is CONTROL too, and it writes ``LiveSide.team_size``."""
    b = _battle()
    before = b.live_view()
    b.parse_message(["", "teamsize", "p2", "3"])
    after = b.live_view()
    assert after is not before
    assert after.opp.team_size == 3 and before.opp.team_size == 6
    assert after == _fresh(b)


# ───────────────────────── door 2: parse_request ───────────────────────────
def test_a_request_invalidates_although_it_is_never_an_event():
    """The named GIGO of this memo. ``_update_team_from_request`` writes HP / status /
    item / PP onto our mons and the request is not on the event log at all, so a
    ``(len(events), turn)`` key would serve a pre-request board to the encoder — the
    request-misalignment class ``gen3_op_move_align_v1`` exists to prevent."""
    b = _battle()
    before = b.live_view()
    assert before.ours.active.hp_fraction == 1.0
    n_events, turn = len(b.events), b.turn
    b.parse_request({
        "side": {"name": "p1user", "id": "p1", "pokemon": [{
            "ident": "p1: Zappy",
            "details": "Zapdos, L100",
            "condition": "37/100",
            "active": True,
            "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
            "moves": ["thunderbolt"],
            "baseAbility": "pressure",
            "item": "leftovers",
            "pokeball": "pokeball",
        }]},
        "active": [{"moves": [
            {"move": "Thunderbolt", "id": "thunderbolt", "pp": 24, "maxpp": 24,
             "target": "normal", "disabled": False},
        ]}],
    })
    assert (len(b.events), b.turn) == (n_events, turn), (
        "the request moved neither the event log nor the turn — the two signals a "
        "non-request-aware key would have keyed on"
    )
    after = b.live_view()
    assert after is not before
    assert after.ours.active.hp_fraction == 0.37
    assert after == _fresh(b)


# ───────────────────── door 3: won_by / tied (off-protocol) ────────────────
def test_won_by_invalidates():
    """``|win|`` is intercepted by ``Player._handle_battle_message`` and never reaches
    ``parse_message``, so ``finished`` / ``won`` / ``lost`` move behind door 1's back."""
    b = _battle()
    before = b.live_view()
    assert before.finished is False and before.won is None
    b.won_by("p1user")
    after = b.live_view()
    assert after is not before
    assert after.finished is True and after.won is True and after.lost is False
    assert after == _fresh(b)


def test_tied_invalidates():
    b = _battle()
    before = b.live_view()
    b.tied()
    after = b.live_view()
    assert after is not before
    assert after.finished is True
    assert after == _fresh(b)


# ──────────────── door 4: out-of-band event appends ────────────────────────
def test_record_choice_rejected_invalidates():
    """The one event append that does not come from ``parse_message``."""
    b = _battle()
    before = b.live_view()
    b.record_choice_rejected(["", "error", "[Unavailable choice]"])
    assert b.live_view() is not before
    assert b.live_view() == _fresh(b)


# ─────────────────── the store discipline (concurrency) ────────────────────
def test_a_view_built_across_a_mutation_is_never_served(monkeypatch):
    """``live_view`` reads the epoch BEFORE building and stores the view under THAT epoch.

    So a build that raced a writer lands under a key that is already dead and can never be
    served to a later reader — the memo adds no staleness window of its own. Simulated
    deterministically by mutating the battle from inside ``from_battle``.
    """
    b = _battle()
    original = LiveView.from_battle.__func__

    def racing_from_battle(cls, battle):
        view = original(cls, battle)
        # A writer lands while we were building: the board moves under us.
        battle.parse_message(["", "-damage", "p2a: Tyra", "10/100"])
        return view

    monkeypatch.setattr(LiveView, "from_battle", classmethod(racing_from_battle))
    torn = b.live_view()
    assert torn.opp.active.hp_fraction == 1.0, "sanity: the torn view predates the write"
    monkeypatch.undo()

    served = b.live_view()
    assert served is not torn, "the torn view was stored under a dead epoch and re-served"
    assert served.opp.active.hp_fraction == 0.1
    assert served == _fresh(b)


# ────────────── clone / rollback: the materializer's ground ────────────────
def test_a_deep_copied_battle_does_not_serve_its_twins_view():
    """The clone-aliasing hazard, as the offline materializer meets it.

    ``_PlayerSnapshot`` deep-copies the battle graph and restores a FRESH copy per arm.
    Arm 1 runs and advances its battle; arm 2 is restored from the pre-arm-1 copy and must
    see its OWN board. The memo rides the battle, so this holds by construction — but a
    cache keyed by ``battle_tag`` (the tags are identical across arms!) would serve arm-1's
    forward state here, silently and byte-wrongly.
    """
    root = _battle()
    root.live_view()                      # prime the memo before the snapshot is taken
    snapshot = copy.deepcopy(root)

    arm1 = copy.deepcopy(snapshot)
    arm1.parse_message(["", "-damage", "p2a: Tyra", "5/100"])
    arm1.parse_message(["", "turn", "2"])
    assert arm1.live_view().opp.active.hp_fraction == 0.05

    arm2 = copy.deepcopy(snapshot)
    arm2.parse_message(["", "-boost", "p1a: Zappy", "spa", "2"])
    view2 = arm2.live_view()

    assert arm2.battle_tag == arm1.battle_tag, "the arms are indistinguishable by tag"
    assert view2 == _fresh(arm2), "arm 2 was served a view of a board it never had"
    assert view2.opp.active.hp_fraction == 1.0
    assert view2.turn == 1
    assert dict(view2.ours.active.boosts) == {"spa": 2}


def test_a_deep_copy_does_not_share_its_source_memo_slot():
    """Writing one battle's memo must not be visible through the other. (``deepcopy``
    gives this; the test is here so a future ``__deepcopy__`` that tries to share the
    frozen view — it IS immutable, so it looks safe — cannot also share the SLOT.)"""
    root = _battle()
    root.live_view()
    clone = copy.deepcopy(root)
    clone.parse_message(["", "-damage", "p2a: Tyra", "20/100"])
    clone.live_view()
    assert root.live_view().opp.active.hp_fraction == 1.0
    assert root.live_view() == _fresh(root)


def test_a_plain_battle_without_the_memo_still_masks():
    """``mask_generator`` now prefers ``battle.live_view()`` and falls back to
    ``LiveView.from_battle`` for a battle that has no such accessor — the fallback the
    docstring promises, exercised."""
    from agents.action.mask_generator import Gen3ActionMasker

    b = _battle()

    class _NoAccessor:
        """Everything ``LiveView.from_battle`` / ``LegalActions.from_battle`` read,
        without a ``live_view`` method."""
        def __getattr__(self, name):
            if name == "live_view":
                raise AttributeError(name)
            return getattr(b, name)

    proxy = _NoAccessor()
    assert not hasattr(proxy, "live_view")
    assert LiveView.from_battle(proxy) == _fresh(b)
    # And the masker takes that path without touching the memo.
    b._live_view_memo = None
    b.parse_request({
        "side": {"name": "p1user", "id": "p1", "pokemon": [{
            "ident": "p1: Zappy", "details": "Zapdos, L100", "condition": "100/100",
            "active": True,
            "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
            "moves": ["thunderbolt"], "baseAbility": "pressure", "item": "leftovers",
            "pokeball": "pokeball",
        }]},
        "active": [{"moves": [
            {"move": "Thunderbolt", "id": "thunderbolt", "pp": 24, "maxpp": 24,
             "target": "normal", "disabled": False},
        ]}],
    })
    b._live_view_memo = None
    Gen3ActionMasker.get_mask(proxy)
    assert b._live_view_memo is None, "the no-accessor path must not populate the memo"
