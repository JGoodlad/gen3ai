"""``core_view`` — the read-models from the Rust core's ``present()`` view (``gen3_core_present_v1``).

The Rust Core Program's M2 (``designs/endstate/program_rust_core.md``): the core folds ONE side's
stream into poke-env's reading of the board and renders it in ``LiveView``'s own shape
(``src/rust_sim/src/present/``). **Every poke-env rule is applied on the Rust side** — the reading
rules V1–V17 are that module's, each named and pinned there — so this module is a TRANSPORT and
nothing else: it copies each field into the same frozen dataclass ``battle.strict_view()`` hands
out, and computes no value, applies no rule, and defaults nothing. (Its predecessor, the
``view_adapter`` over the port's ``one_sided_view`` payload, which re-applied poke-env's
presentation rules in Python, is deleted — Rust Core deletion pass, program §4 M2.)

Equality with the ``LiveView`` training builds is the Rust Core parity harness's slice V
(``rust_core_parity_views.py``), at every decision of every recorded battle, both viewers.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Mapping, Optional

from agents.battle.live_view import (LegalActions, LegalMove, LegalSwitch, LiveMove, LivePokemon,
                                     LiveSide, LiveView, LiveWeather)


def _mon(r: Mapping[str, Any]) -> LivePokemon:
    ivs, evs = r["ivs"], r["evs"]
    return LivePokemon(
        species=r["species"],
        active=r["active"],
        fainted=r["fainted"],
        revealed=r["revealed"],
        hp_fraction=float(r["hp_fraction"]),
        status=r["status"],
        types=tuple(r["types"]),
        moves=tuple(LiveMove(id=m["id"], current_pp=m["current_pp"], max_pp=m["max_pp"])
                    for m in r["moves"]),
        item=r["item"],
        ability=r["ability"],
        boosts=dict(r["boosts"]),
        volatiles=dict(r["volatiles"]),
        base_stats=dict(r["base_stats"]),
        ivs=tuple(ivs) if ivs is not None else None,
        evs=tuple(evs) if evs is not None else None,
        nature=r["nature"],
        spread_known=r["spread_known"],
        consumed_item=r["consumed_item"],
        status_counter=r["status_counter"],
        protect_counter=r["protect_counter"],
        stats=dict(r["stats"]),
        current_hp=r["current_hp"],
        max_hp=r["max_hp"],
    )


def _side(s: Mapping[str, Any]) -> LiveSide:
    mons = tuple(_mon(r) for r in s["mons"])
    active = next((m for m in mons if m.active), None)
    return LiveSide(team_size=s["team_size"], active=active, mons=mons,
                    side_conditions=dict(s["side_conditions"]))


def live_view_from_core(view: Mapping[str, Any], *, battle_tag: str = "") -> LiveView:
    """The ``LiveView`` of a core ``present()`` view (its ``OneSidedView::json``)."""
    w = view["weather"]
    return LiveView(
        turn=view["turn"],
        weather=LiveWeather(weather=w["weather"], is_permanent=w["is_permanent"],
                            turns_active=w["turns_active"]),
        ours=_side(view["ours"]),
        opp=_side(view["opp"]),
        battle_tag=battle_tag,
        finished=view["finished"],
        won=view["won"],
        lost=view["lost"],
    )


def legal_actions_from_core(legal: Optional[Mapping[str, Any]],
                            request: Optional[str]) -> Optional[LegalActions]:
    """The ``LegalActions`` of a core ``legal_actions()`` (its ``LegalActions::json``), with the
    raw ``|request|`` payload the core parsed as the read-only ``last_request`` mirror. ``None``
    when the side holds no request."""
    if legal is None:
        return None
    return LegalActions(
        move_slots=tuple(LegalMove(id=m["id"], current_pp=m["current_pp"], max_pp=m["max_pp"],
                                   disabled=m["disabled"], target=m["target"])
                         for m in legal["move_slots"]),
        switches=tuple(LegalSwitch(species=s["species"], slot=s["slot"]) for s in legal["switches"]),
        force_switch=legal["force_switch"],
        trapped=legal["trapped"],
        maybe_trapped=legal["maybe_trapped"],
        wait=legal["wait"],
        struggle=legal["struggle"],
        last_request=MappingProxyType(json.loads(request)) if request else None,
        own_hp_typed_id=legal["own_hp_typed_id"],
    )
