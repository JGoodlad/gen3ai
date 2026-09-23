"""The committed Rust event-schema table equals what the Python schema renders today.

``src/rust_sim/src/core_events/schema.rs`` is GENERATED from ``battle_event.py``'s MESSAGE_POLICY /
EVENT_KIND / EventKind / value-key schema and ``Player.MESSAGES_TO_IGNORE``
(``gen3_core_event_schema_codegen_v1``). A Python-side schema change that is not regenerated fails
HERE, the day it lands — the Rust core must never classify a keyword differently from poke-env.
"""

from agents.battle.battle_event import MESSAGE_POLICY
from agents.battle.rust_core_schema import OUT, keyword_rows, render


def test_the_committed_rust_schema_is_current():
    assert OUT.read_text() == render(), (
        "src/rust_sim/src/core_events/schema.rs is STALE — run "
        "`python -m agents.battle.rust_core_schema --write` and rebuild the rust core")


def test_every_policy_keyword_has_a_rust_variant():
    kws = {kw for kw, _, _ in keyword_rows()}
    assert set(MESSAGE_POLICY) <= kws
    # the player-intercepted ones route before parse_message, exactly as the Player does
    routes = {kw: r for kw, _, r in keyword_rows()}
    assert routes["request"] == "Route::Intercept(Intercept::Request)"
    assert routes["t:"] == "Route::Intercept(Intercept::Ignored)"
    assert routes["-damage"] == "Route::Event(EventKind::Damage)"
