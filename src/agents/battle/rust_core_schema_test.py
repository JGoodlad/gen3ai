"""The Rust core's event-schema table agrees with the Python classifier (``gen3_core_event_schema_codegen_v1``,
FROZEN as Rust-owned source in P1 of the poke-env retirement, ``T27``).

``src/rust_sim/src/core_events/schema.rs`` used to be GENERATED from ``battle_event.py`` and
``Player.MESSAGES_TO_IGNORE`` by ``agents/battle/rust_core_schema.py``, and this file held it EQUAL to a fresh render.
The generator imported poke-env, so it is deleted and the Rust file is now OWNED source (a new keyword is added THERE).
While the Python battle layer still classifies protocol lines (until P6), the contract "the Rust core must never
classify a keyword differently from the Python reader" is kept by PARSING the Rust file and comparing it with
``battle_event`` — the same four facts the render covered:

* every ``EventKind`` member: its variant name, its stable integer, its Python name;
* every kind's REQUIRED and OPTIONAL value keys (``EVENT_VALUE_KEYS`` / ``EVENT_OPTIONAL_KEYS``);
* every ``MESSAGE_POLICY`` keyword has a Rust variant, and its Rust route is the one its policy maps to — except a
  keyword Rust routes ``Intercept::Ignored`` (the ``Player.MESSAGES_TO_IGNORE`` set, frozen with the fork);
* a keyword Rust knows that Python's policy does not is a player-intercepted one (request, win, error, …) or ignored.
"""
import re
from typing import Dict, List, Tuple

from agents.battle.battle_event import (EVENT_KIND, EVENT_OPTIONAL_KEYS, EVENT_VALUE_KEYS, MESSAGE_POLICY,
                                        EventKind, Policy)
from utils.paths import src_path

SCHEMA = src_path("rust_sim", "src", "core_events", "schema.rs")
IGNORED = "Route::Intercept(Intercept::Ignored)"


def _text() -> str:
    return SCHEMA.read_text()


def _between(text: str, start: str, end: str) -> str:
    a = text.index(start)
    return text[a:text.index(end, a)]


def _variant(kind: EventKind) -> str:
    return kind.name.title().replace("_", "")


def rust_kinds() -> List[Tuple[str, str]]:
    body = _between(_text(), "pub enum EventKind {", "\n}\n")
    return re.findall(r"^\s+(\w+) = (\d+),$", body, flags=re.M)


def rust_kind_table(fn: str, nxt: str) -> Dict[str, str]:
    body = _between(_text(), f"pub fn {fn}(self)", nxt)
    return dict(re.findall(r"EventKind::(\w+) => (.+),$", body, flags=re.M))


def rust_keywords() -> Dict[str, str]:
    """``keyword -> Rust variant`` from ``Kw::as_str``."""
    body = _between(_text(), "pub fn as_str(self)", "pub fn from_protocol")
    return {kw: name for name, kw in re.findall(r'Kw::(\w+) => "(.*)",$', body, flags=re.M) if name != "Plain"}


def rust_routes() -> Dict[str, str]:
    """``Rust variant -> route`` from ``Kw::route``."""
    body = _between(_text(), "pub fn route(self)", "\n}\n")
    return {name: route for name, route in re.findall(r"Kw::(\w+) => (Route::.+),$", body, flags=re.M)
            if name != "Plain"}


def test_the_event_kinds_match_the_python_enum_name_value_and_all():
    kinds = sorted(EventKind, key=lambda k: k.value)
    assert rust_kinds() == [(_variant(k), str(k.value)) for k in kinds], \
        "schema.rs's EventKind enum differs from battle_event.EventKind — change BOTH"
    names = rust_kind_table("name", "pub fn required_keys")
    assert names == {_variant(k): f'"{k.name}"' for k in kinds}


def test_the_declared_value_keys_match_both_halves():
    for fn, nxt, table in (("required_keys", "pub fn optional_keys", EVENT_VALUE_KEYS),
                           ("optional_keys", "\n}\n", EVENT_OPTIONAL_KEYS)):
        rust = rust_kind_table(fn, nxt)
        for k in EventKind:
            keys = re.findall(r'"([^"]*)"', rust[_variant(k)])
            assert keys == sorted(table.get(k, frozenset())), (fn, k.name, keys)


def test_every_policy_keyword_has_a_rust_variant_with_the_route_its_policy_maps_to():
    kw_to_variant, routes = rust_keywords(), rust_routes()
    assert set(MESSAGE_POLICY) <= set(kw_to_variant), sorted(set(MESSAGE_POLICY) - set(kw_to_variant))
    for kw, (policy, _) in MESSAGE_POLICY.items():
        route = routes[kw_to_variant[kw]]
        if route == IGNORED:
            continue                       # `Player.MESSAGES_TO_IGNORE`: routed before the classifier, as the Player does
        if policy is Policy.EVENT:
            want = f"Route::Event(EventKind::{_variant(EVENT_KIND[kw])})"
        else:
            want = {Policy.CONTROL: "Route::Control", Policy.COSMETIC: "Route::Cosmetic",
                    Policy.STATE_ONLY: "Route::StateOnly", Policy.UNSUPPORTED: "Route::Unsupported"}[policy]
        assert route == want, (kw, policy.name, route, want)


def test_a_rust_only_keyword_is_player_intercepted_or_ignored():
    kw_to_variant, routes = rust_keywords(), rust_routes()
    for kw in sorted(set(kw_to_variant) - set(MESSAGE_POLICY)):
        assert routes[kw_to_variant[kw]].startswith("Route::Intercept("), (kw, routes[kw_to_variant[kw]])
    assert routes[kw_to_variant["request"]] == "Route::Intercept(Intercept::Request)"
    assert routes[kw_to_variant["t:"]] == IGNORED
    assert routes[kw_to_variant["-damage"]] == "Route::Event(EventKind::Damage)"


def test_the_keyword_count_constant_is_the_table_length():
    n = int(re.search(r"pub const N_KEYWORDS: usize = (\d+);", _text()).group(1))
    assert n == len(rust_keywords()) == len(rust_routes())
