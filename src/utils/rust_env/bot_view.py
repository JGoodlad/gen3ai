"""THE BOT VIEW — every battle fact a scripted bot reads, as ONE canonical string (M5 Lane F).

The scripted bots (`utils.rust_env.bot_inventory`) read a poke-env ``Battle`` — in training, the
opponent's ``env.battle2``. The Rust port (`src/rust_env/src/bots/`) reads the env core's own
per-side ``BoardReading``, the port of exactly that object (program §2 M5, Lane F). This module is
the Python half of the proof that the two are the SAME input: it renders, from a poke-env battle,
every attribute any bot in the inventory reads, in a fixed order, as compact JSON; the Rust twin
(`bots::view::render`) renders the same string from the ``BoardReading``. The banked corpus stores
its FNV-1a-64 (`fnv64`), and the gate compares it at every decision BEFORE comparing the action — so
an action that agrees is known to agree on equal inputs, not by accident.

Floats are rendered as the hex of their IEEE-754 bits (``f:3fe0000000000000``), so the comparison is
bit-exact and no float formatting has to agree across languages.

What is IN the view (and why — each is read by at least one bot):

* ``wait`` / ``trapped`` / ``force_switch`` — ``valid_orders`` (RandomPlayer + every fallback);
* ``team`` — every own mon, dict order (``battle.team``): switch scoring reads a bench mon's types,
  base stats, boosts, HP, status, ability and full moveset; ``n_remaining`` reads ``fainted``;
* ``active`` — the index into ``team`` of ``battle.active_pokemon`` (the FIRST mon flagged active);
* ``opp_active`` — ``battle.opponent_active_pokemon``, the same mon fields (its REVEALED moves);
* ``opp_fainted`` — ``sum(fainted)`` over ``battle.opponent_team``;
* ``moves`` — ``battle.available_moves`` (request order, the resolved ``Move`` objects);
* ``switches`` — ``battle.available_switches`` as indices into ``team``;
* ``side`` / ``opp_side`` — the ``SideCondition`` NAMES of ``side_conditions`` /
  ``opponent_side_conditions`` (dict order);
* ``valid`` — ``[o.message for o in battle.valid_orders]`` (the random fallback's list, in order).

A mon: species, ``type_1`` / ``type_2`` / ``types`` (``PokemonType`` names), ``base_stats`` (hp…spe),
``boosts`` (poke-env's key order), ``current_hp_fraction``, ``status`` (``Status`` name), ``ability``,
``fainted``, ``stats`` (own only; ``None`` → null), ``moves`` (``moves.values()``).
A move: id, ``base_power``, ``type``, ``category`` (gen-3 by type), ``accuracy``, ``expected_hits``,
``target`` (``Target`` name), ``boosts`` (dict order) and ``is_last_used`` (``Pokemon.last_move``).
"""
from __future__ import annotations

import struct

BOOST_KEYS = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def fbits(x) -> str:
    """A float as the hex of its IEEE-754 bits (an int is converted first, as Python arithmetic does)."""
    return "f:" + format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "016x")


def _s(x) -> str:
    if x is None:
        return "null"
    if not isinstance(x, str):
        raise TypeError(f"bot view: {x!r} is not a string")
    # the ONE escaping rule both renderers share: `"` and `\` backslashed, a control char as
    # \u00XX, everything else (non-ASCII included) raw UTF-8
    out = []
    for c in x:
        if c in '"\\':
            out.append("\\" + c)
        elif ord(c) < 0x20:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def _b(x) -> str:
    return "true" if x else "false"


def _i(x) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"bot view: {x!r} is not an int")
    return str(x)


def _name(e) -> str:
    return "null" if e is None else _s(e.name)


def move_json(m) -> str:
    boosts = m.boosts
    bj = "null" if boosts is None else "[" + ",".join(f"[{_s(k)},{_i(v)}]" for k, v in boosts.items()) + "]"
    return ("{" + f'"id":{_s(m.id)},"bp":{_i(m.base_power)},"type":{_name(m.type)},"cat":{_name(m.category)},'
            f'"acc":{_s(fbits(m.accuracy))},"hits":{_s(fbits(m.expected_hits))},"target":{_name(m.target)},'
            f'"boosts":{bj},"last":{_b(m.is_last_used)}' + "}")


def mon_json(p) -> str:
    stats = p.stats
    return ("{" + f'"species":{_s(p.species)},"t1":{_name(p.type_1)},"t2":{_name(p.type_2)},'
            f'"types":[{",".join(_name(t) for t in p.types)}],'
            f'"base":[{",".join(_i(p.base_stats[k]) for k in STAT_KEYS)}],'
            f'"boosts":[{",".join(_i(p.boosts[k]) for k in BOOST_KEYS)}],'
            f'"hp":{_s(fbits(p.current_hp_fraction))},"status":{_name(p.status)},"ability":{_s(p.ability)},'
            f'"fainted":{_b(p.fainted)},'
            f'"stats":[{",".join(_i(stats.get(k)) for k in STAT_KEYS)}],'
            f'"moves":[{",".join(move_json(m) for m in p.moves.values())}]' + "}")


def view_json(battle) -> str:
    """The canonical view string of ``battle`` (a poke-env singles ``Battle``)."""
    team = list(battle.team.values())
    active = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    def index_of(mon):
        for i, t in enumerate(team):
            if t is mon:
                return i
        raise ValueError(f"bot view: {mon!r} is not in battle.team")

    return ("{" + f'"wait":{_b(battle.wait)},"trapped":{_b(battle.trapped)},"force_switch":{_b(battle.force_switch)},'
            f'"team":[{",".join(mon_json(m) for m in team)}],'
            f'"active":{"null" if active is None else index_of(active)},'
            f'"opp_active":{"null" if opp is None else mon_json(opp)},'
            f'"opp_fainted":{sum(1 for m in battle.opponent_team.values() if m.fainted)},'
            f'"moves":[{",".join(move_json(m) for m in battle.available_moves)}],'
            f'"switches":[{",".join(str(index_of(s)) for s in battle.available_switches)}],'
            f'"side":[{",".join(_s(c.name) for c in battle.side_conditions)}],'
            f'"opp_side":[{",".join(_s(c.name) for c in battle.opponent_side_conditions)}],'
            f'"valid":[{",".join(_s(o.message) for o in battle.valid_orders)}]' + "}")


def fnv64(s: str) -> str:
    """FNV-1a-64 of the UTF-8 bytes, as 16 hex digits (the Rust twin is `bots::view::fnv64`)."""
    h = 0xCBF29CE484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")
