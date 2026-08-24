"""Schema + completeness-registry tests for the event log (design §4.4, §8.1).

Pure unit tests — no server, no Node bridge.
"""

import re
from pathlib import Path

import pytest

import poke_env.battle.abstract_battle as ab_mod
from poke_env.battle.abstract_battle import AbstractBattle

from agents.battle.battle_event import (
    EVENT_KIND,
    MESSAGE_POLICY,
    BattleEvent,
    EventKind,
    Policy,
    UnknownMessageType,
    UnsupportedMessageType,
    classify,
    from_clause_move_source,
)


# ---------------------------------------------------------------------------
# from_clause_move_source — the shared move-call wire parser (one source of truth for
# both Gen3Battle._delegated_from and ChoiceBandTracker._is_delegated)
# ---------------------------------------------------------------------------

def _move_line(*tokens):
    return ("", "move", "p1a: Mon", *tokens)


def test_from_clause_move_source_gen3_bare_form():
    """Bundled-gen3 BARE form: `[from] Sleep Talk` (no `move:` prefix) — the form the live sim
    actually emits, and the one the old `[from]move:`-only parser silently dropped."""
    assert from_clause_move_source(_move_line("Earthquake", "p2a: X", "[from] Sleep Talk")) == "sleeptalk"


def test_from_clause_move_source_modern_form():
    """Modern Showdown form `[from]move: Sleep Talk` is also accepted (forward-compatible)."""
    assert from_clause_move_source(_move_line("Earthquake", "p2a: X", "[from]move: Sleep Talk")) == "sleeptalk"
    assert from_clause_move_source(_move_line("Earthquake", "p2a: X", "[from] move: Sleep Talk")) == "sleeptalk"


def test_from_clause_move_source_self_tag_returns_self():
    """Pursuit hitting a switching target tags its OWN line `[from] Pursuit`; the source is the
    move itself (callers treat same-move as a free selection, not a delegation)."""
    assert from_clause_move_source(_move_line("Pursuit", "p2a: X", "[from] Pursuit")) == "pursuit"


def test_from_clause_move_source_lockedmove_marker():
    """A two-turn / Outrage continuation marker is returned verbatim (a non-move id the caller
    recognises), not silently dropped."""
    assert from_clause_move_source(_move_line("Solar Beam", "p2a: X", "[from] lockedmove")) == "lockedmove"


def test_from_clause_move_source_item_ability_are_not_move_calls():
    """`[from] item:` / `[from] ability:` are causes, NOT move calls → None."""
    assert from_clause_move_source(("", "move", "p1a: Mon", "Tackle", "p2a: X", "[from] item: Leftovers")) is None
    assert from_clause_move_source(("", "move", "p1a: Mon", "Tackle", "p2a: X", "[from] ability: Sand Stream")) is None


def test_from_clause_move_source_none_when_no_from():
    """A plain move line (a free selection) has no `[from]` → None."""
    assert from_clause_move_source(_move_line("Earthquake", "p2a: X")) is None


# --------------------------------------------------------------------------- #
# Registry audit: every keyword poke-env can emit must be classified.          #
# --------------------------------------------------------------------------- #
def _dispatch_keywords_from_source() -> set[str]:
    """Extract every literal protocol keyword handled in AbstractBattle.parse_message.

    Parsing the source (rather than hardcoding) makes this audit auto-track poke-env:
    if a future merge adds an ``elif event[1] == "newthing"`` branch, this set grows
    and the coverage assertion below fails until ``newthing`` is classified.
    """
    src = Path(ab_mod.__file__).read_text()
    # isolate the parse_message body
    start = src.index("def parse_message")
    end = src.index("def parse_request", start)
    body = src[start:end]
    keywords: set[str] = set()
    # `event[1] == "x"` / `event[1] in ["a", "b"]` / `in ("a","b")` / `in {"a","b"}`
    for m in re.finditer(r"event\[1\]\s*(?:==|in)\s*([^\n:]+):", body):
        rhs = m.group(1)
        for tok in re.findall(r"""["']([^"']*)["']""", rhs):
            keywords.add(tok)
    return keywords


def test_registry_covers_poke_env_dispatch():
    """Every keyword explicitly handled in parse_message is in MESSAGE_POLICY."""
    handled = _dispatch_keywords_from_source()
    assert handled, "failed to extract any dispatch keywords from poke-env source"
    missing = sorted(k for k in handled if k not in MESSAGE_POLICY)
    assert not missing, (
        f"poke-env parse_message handles these keywords but MESSAGE_POLICY does not "
        f"classify them: {missing}"
    )


def test_registry_covers_messages_to_ignore():
    """Every keyword poke-env silently ignores is classified (COSMETIC/CONTROL/...)."""
    missing = sorted(k for k in AbstractBattle.MESSAGES_TO_IGNORE if k not in MESSAGE_POLICY)
    assert not missing, f"MESSAGES_TO_IGNORE keywords not in MESSAGE_POLICY: {missing}"


def test_every_event_policy_keyword_has_a_kind():
    """An EVENT-policy keyword must map to an EventKind (so it can be emitted)."""
    for kw, (policy, _reason) in MESSAGE_POLICY.items():
        if policy is Policy.EVENT:
            assert kw in EVENT_KIND, f"EVENT keyword {kw!r} has no EVENT_KIND mapping"


def test_no_event_kind_for_non_event_keyword():
    """EVENT_KIND should only contain EVENT-policy keywords (no dead entries)."""
    for kw in EVENT_KIND:
        assert MESSAGE_POLICY[kw][0] is Policy.EVENT, (
            f"{kw!r} is in EVENT_KIND but classified {MESSAGE_POLICY[kw][0].name}"
        )


def test_every_policy_entry_has_nonempty_reason():
    for kw, (_policy, reason) in MESSAGE_POLICY.items():
        assert reason and isinstance(reason, str), f"{kw!r} has no reason string"


# --------------------------------------------------------------------------- #
# classify() behaviour                                                          #
# --------------------------------------------------------------------------- #
def test_classify_event_returns_policy_and_reason():
    policy, reason = classify("move")
    assert policy is Policy.EVENT
    assert "move" in reason


@pytest.mark.parametrize("kw", ["-mega", "-zpower", "-terastallize", "-primal", "-burst"])
def test_classify_raises_on_unsupported(kw):
    with pytest.raises(UnsupportedMessageType):
        classify(kw)


@pytest.mark.parametrize("kw", ["totallybogus", "-newgen10mechanic", "xyzzy"])
def test_classify_raises_on_unknown(kw):
    with pytest.raises(UnknownMessageType):
        classify(kw)


# --------------------------------------------------------------------------- #
# BattleEvent schema                                                            #
# --------------------------------------------------------------------------- #
def test_battle_event_is_frozen():
    ev = BattleEvent(seq=0, turn=1, kind=EventKind.MOVE)
    with pytest.raises(Exception):
        ev.seq = 5  # frozen dataclass


def test_battle_event_accessors_read_value():
    ev = BattleEvent(
        seq=3,
        turn=2,
        kind=EventKind.DAMAGE,
        side="opp",
        actor_species="tyranitar",
        value={"amount": -0.45, "move_id": None, "reason": "Spikes"},
    )
    assert ev.amount == pytest.approx(-0.45)
    assert ev.reason == "Spikes"
    assert ev.move_id is None


def test_event_kind_values_are_stable():
    # Serialised logs/datasets depend on these ints; guard a few anchors.
    assert int(EventKind.MOVE) == 1
    assert int(EventKind.DAMAGE) == 5
    assert int(EventKind.SUPEREFFECTIVE) == 19
    assert int(EventKind.UNKNOWN) == 99


# --------------------------------------------------------------------------- #
# LIVE-SERVER room chrome (the ladder-readiness gate)                           #
# --------------------------------------------------------------------------- #
# A `--no-security` local sim emits a CLOSED set of battle-room lines, so no fuzz
# corpus and no eval battle has ever exercised the room-layer chrome the public
# server adds. `classify` raises on anything unclassified BY DESIGN, and every such
# raise wedges the battle for good (the parse task dies, no choice is ever sent).
# These are the keywords measured or sourced from the live protocol; see
# designs/research_state/ladder_readiness.md.
_LIVE_ROOM_KEYWORDS = [
    "c",             # battle-room chat (battle rooms set noLogTimes ⇒ untimestamped)
    "c:",            # the timestamped chat form every OTHER room uses
    "html",          # `|html|<div class="message-error">…` — MEASURED on :9017
    "uhtml",
    "uhtmlchange",
    "raw",           # the rated-ladder rating-change block — MEASURED on :9017
    "rated",         # MEASURED on :9017 (a real rated local-ladder game)
    "askreg",        # MEASURED on :9017
    "J", "L", "j", "l", "n",
    "inactive", "inactiveoff",   # the ladder TIMER
    "noinit", "popup", "notify", "tempnotify", "tempnotifyoff",
    "-message", "-hint", ":", "t:",
]


@pytest.mark.parametrize("kw", _LIVE_ROOM_KEYWORDS)
def test_live_server_room_chrome_never_raises(kw):
    """None of it is battle CONTENT, so none of it may be a tripwire."""
    policy, _reason = classify(kw)
    assert policy in (Policy.COSMETIC, Policy.CONTROL), (
        f"{kw!r} is live-server room chrome but classified {policy.name}"
    )


# The subset poke-env must IGNORE outright. The complement is handled some other way —
# `raw`/`inactive`/`-message`/`-hint` get real parse branches, and `noinit`/`popup`/
# `notify`/`tempnotify*`/`uhtmlchange` are pre-filtered before the battle ever sees them
# — so demanding ignore-set membership for those would assert a false contract.
_LIVE_ROOM_MUST_BE_IGNORED = [
    kw for kw in _LIVE_ROOM_KEYWORDS
    if kw not in {"raw", "inactive", "-message", "-hint", "notify", "tempnotify",
                  "tempnotifyoff", "noinit", "popup", "uhtmlchange"}
]


@pytest.mark.parametrize("kw", _LIVE_ROOM_MUST_BE_IGNORED)
def test_live_server_room_chrome_is_also_ignored_by_poke_env(kw):
    """`Gen3Battle.parse_message` delegates every non-EVENT line to poke-env, which
    raises `NotImplementedError` on anything outside ITS own ignore set. Classifying a
    keyword here without adding it there therefore fixes nothing — the raise just moves
    one frame down. `c:` is the one that was in neither."""
    assert kw in AbstractBattle.MESSAGES_TO_IGNORE, (
        f"{kw!r} is classified here but poke-env would still raise on it"
    )
