#!/usr/bin/env python3
"""Feed the SAME captured protocol stream to OUR reader, and dump what it sees.

Runs under ``gen3ai_stable`` with ``PYTHONPATH=…:src``, where ``import poke_env`` resolves to the
**vendored fork**. Its counterpart ``replay_metamon_state.py`` runs under the Metamon interpreter
against upstream poke-env; the two must never share a process (root ``CLAUDE.md``, anchors H11),
which is why they are two scripts over one JSONL schema rather than one script with a switch.

THE SLOT IS METAMON'S, DELIBERATELY. We replay the chunk list Metamon's client received, through
our ``Gen3Battle``, as if we had been sitting in Metamon's seat. Same bytes, two parsers: any
difference in the resulting picture is a parser difference and nothing else.

WHAT IT DUMPS
-------------
Both of our real read models, not the raw poke-env object:
* ``live`` — :class:`agents.battle.live_view.LiveView`, the current board every non-``battle/``
  consumer reads (``battle.strict_view()``);
* ``legal`` — :class:`agents.battle.live_view.LegalActions`, the server-authoritative decision
  surface;
* ``raw`` — the poke-env-level properties, in the SAME schema the Metamon-side script emits, so a
  field Metamon reads and a field only we read can be told apart.
"""

import argparse
import json
import logging
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_common as rc  # noqa: E402


def dispatch(battle, line, ignore):
    """One protocol line through OUR fork's own dispatch (``Player._handle_battle_message``)."""
    split = line.split("|")
    if len(split) == 1:
        return None
    keyword = split[1]
    if keyword == "":
        battle.parse_message(split)
        return None
    if keyword in ignore:
        return None
    if keyword == "request":
        if not split[2]:
            return None
        request = json.loads(split[2])
        battle.parse_request(request)
        return request
    if keyword == "win":
        battle.won_by(split[2])
        return None
    if keyword == "tie":
        battle.tied()
        return None
    if keyword in ("error", "bigerror", "popup"):
        return None
    battle.parse_message(split)
    return None


def mon_row(mon):
    return {
        "species": rc.canon(getattr(mon, "species", None)),
        "hp": rc.hp(getattr(mon, "current_hp_fraction", None)),
        "status": rc.canon(getattr(getattr(mon, "status", None), "name", None)),
        "fainted": bool(getattr(mon, "fainted", False)),
        "active": bool(getattr(mon, "active", False)),
        "boosts": rc.boosts(getattr(mon, "boosts", None)),
        "moves": rc.moveset((getattr(mon, "moves", None) or {}).keys()),
        "volatiles": sorted(rc.canon(getattr(e, "name", e)) for e in (getattr(mon, "effects", None) or {})),
        "item": rc.canon(getattr(mon, "item", None)),
        "ability": rc.canon(getattr(mon, "ability", None)),
        "types": sorted(rc.canon(getattr(t, "name", t)) for t in (getattr(mon, "types", None) or ()) if t),
        "level": int(getattr(mon, "level", 0) or 0),
    }


def raw_block(battle):
    return {
        "turn": int(battle.turn),
        "weather": sorted(rc.canon(getattr(w, "name", w)) for w in (battle.weather or {})),
        "fields": sorted(rc.canon(getattr(f, "name", f)) for f in (battle.fields or {})),
        "our_conditions": rc.conditions({getattr(k, "name", k): v for k, v in (battle.side_conditions or {}).items()}),
        "opp_conditions": rc.conditions({getattr(k, "name", k): v for k, v in (battle.opponent_side_conditions or {}).items()}),
        "our_team": {rc.canon(k): mon_row(v) for k, v in battle.team.items()},
        "opp_team": {rc.canon(k): mon_row(v) for k, v in battle.opponent_team.items()},
        "our_active": rc.canon(getattr(battle.active_pokemon, "species", None)),
        "opp_active": rc.canon(getattr(battle.opponent_active_pokemon, "species", None)),
        "force_switch": bool(battle.force_switch[0] if isinstance(battle.force_switch, list) else battle.force_switch),
        "trapped": bool(battle.trapped),
        "available_moves": rc.moveset(m.id for m in (battle.available_moves or ())),
        "available_switches": sorted(rc.canon(p.species) for p in (battle.available_switches or ())),
        "finished": bool(battle.finished),
    }


def live_block(view):
    """LiveView, projected onto the SAME quantities ``UniversalState`` carries."""

    def lp(mon):
        if mon is None:
            return None
        return {
            "species": rc.canon(mon.species),
            "hp": rc.hp(mon.hp_fraction),
            "status": rc.canon(mon.status),
            "types": sorted(rc.canon(t) for t in (mon.types or ()) if t),
            "item": rc.canon(mon.item),
            "ability": rc.canon(mon.ability),
            "boosts": rc.boosts(dict(mon.boosts or {})),
            "moves": rc.moveset(m.id for m in (mon.moves or ())),
            "volatiles": sorted(rc.canon(v) for v in (mon.volatiles or {})),
            "level": 100,
        }

    return {
        "turn": int(view.turn),
        "active": lp(view.ours.active),
        "opp_active": lp(view.opp.active),
        "bench": sorted(rc.canon(m.species) for m in view.ours.mons if not m.active and not m.fainted),
        "bench_hp": {rc.canon(m.species): rc.hp(m.hp_fraction)
                     for m in view.ours.mons if not m.active and not m.fainted},
        "weather": rc.canon(view.weather.weather if view.weather else None),
        "our_conditions": rc.conditions(dict(view.ours.side_conditions or {})),
        "opp_conditions": rc.conditions(dict(view.opp.side_conditions or {})),
        "opp_known_remaining": int(view.opp.remaining),
        "opp_revealed": sorted(rc.canon(m.species) for m in view.opp.mons),
        "finished": bool(view.finished),
        "won": bool(view.won) if view.won is not None else False,
        "lost": bool(view.lost) if view.lost is not None else False,
    }


def legal_block(legal):
    return {
        # Every request slot (this is `move_ids`, wire-truth, disabled slots included) AND the
        # ENABLED subset. Upstream's `available_moves` is the enabled subset, so only the latter
        # is an apples-to-apples comparison; the former is kept so the difference stays visible.
        "move_ids": rc.moveset(legal.move_ids),
        "enabled_move_ids": rc.moveset(m.id for m in legal.move_slots if not m.disabled),
        "switches": sorted(rc.canon(s.species) for s in legal.switches),
        "force_switch": bool(legal.force_switch),
        "trapped": bool(legal.trapped),
        "struggle": bool(legal.struggle),
        "wait": bool(legal.wait),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--slot", required=True, choices=("p1", "p2"))
    ap.add_argument("--username", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from poke_env.player import Player
    from agents.battle.gen3_battle import Gen3Battle
    from agents.battle.live_view import LegalActions

    logger = logging.getLogger("replay")
    logger.addHandler(logging.NullHandler())
    ignore = set(Player.MESSAGES_TO_IGNORE)

    capture = json.load(open(args.capture))
    battle = Gen3Battle(battle_tag=capture["tag"], username=args.username, logger=logger, gen=3)

    rows = []
    seq = 0
    for slot, text in capture["chunks"]:
        if slot != args.slot:
            continue
        request = None
        for line in text.split("\n"):
            got = dispatch(battle, line, ignore)
            if got is not None:
                request = got
        if not rc.is_decision_request(request):
            continue
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            rows.append({"tag": capture["tag"], "seq": seq, "error": "no_active_pokemon"})
            seq += 1
            continue
        view = battle.live_view()
        rows.append({
            "tag": capture["tag"], "seq": seq, "turn": int(battle.turn),
            "live": live_block(view), "legal": legal_block(LegalActions.from_battle(battle)),
            "raw": raw_block(battle),
        })
        seq += 1

    # 🚨 Same refusal as the Metamon side: an empty dump is not a passing check.
    if not rows:
        raise SystemExit(f"{args.capture}: 0 decision points replayed — refusing to write a "
                         "vacuous dump")
    rc.dump(args.out, rows)
    print(json.dumps({"tag": capture["tag"], "n_decisions": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
