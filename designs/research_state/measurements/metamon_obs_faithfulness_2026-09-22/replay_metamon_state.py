#!/usr/bin/env python3
"""Feed OUR captured protocol stream to METAMON's own reader, and dump what it sees.

🚨 **RUNS UNDER THE METAMON INTERPRETER**, where ``import poke_env`` resolves to **upstream
poke-env 0.8.3.3** — not our vendored fork. Nothing in this repo imports it; it is invoked as a
subprocess with ``PYTHONPATH=""`` exactly like ``src/main/anchors/peer_scripts/metamon_side.py``.

WHAT IT REPLAYS
---------------
``--capture`` is one ``ws_frontend`` capture (``utils.bridge.ws_frontend_replay.BattleCapture``):
``chunks`` is every per-side chunk the front end RELAYED, as ``(slot, text)``. We take the chunks
for ONE slot — the one Metamon's client actually received — and hand them to an upstream poke-env
``Battle`` through the same dispatch ``poke_env.player.Player._handle_battle_message`` uses. The
bytes are therefore identical to the ones Metamon's own client consumed in the live game.

WHERE IT SNAPSHOTS
------------------
``PokeEnvWrapper.embed_battle`` (``metamon/env/wrappers.py:390``) calls
``UniversalState.from_Battle(battle)`` and ``UniversalAction.definitely_valid_actions(...)`` once
per decision. poke-env asks the player for a move once per non-``wait`` ``|request|``, so a chunk
carrying such a request is a decision point and we snapshot immediately after folding it.

WHAT IT DUMPS, and why in two blocks
------------------------------------
* ``us`` — the ``UniversalState`` fields Metamon's observation is BUILT from. A disagreement here
  is a field Metamon READS, i.e. a GIGO candidate (class **b**).
* ``raw`` — the underlying ``Battle`` state, including the parts ``UniversalState`` never looks at
  (the opponent's revealed team, per-mon items/abilities). A disagreement confined to ``raw`` is
  not something Metamon reads, so it cannot be a Metamon GIGO — it is either presentation (class
  **a**) or our own bug (class **c**).
* ``tokens`` — the actual observation Metamon would feed its network, and the count of
  ``UNKNOWN_TOKEN`` (-1) in it. ``PokemonTokenizer.tokenize`` maps an out-of-vocabulary word to -1
  and **prints nothing**, so no stderr grep can find this; only a census can.
"""

import argparse
import json
import logging
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_common as rc  # noqa: E402


def build_battle(tag, username, gen=3):
    from poke_env.environment import Battle

    logger = logging.getLogger("replay")
    logger.addHandler(logging.NullHandler())
    return Battle(battle_tag=tag, username=username, logger=logger, gen=gen)


def dispatch(battle, line, ignore):
    """One protocol line through UPSTREAM poke-env's own dispatch (``Player._handle_battle_message``)."""
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


def universal_block(state):
    """The ``UniversalState`` fields, canonicalised — exactly what Metamon's obs is built from."""
    active = state.player_active_pokemon
    opp = state.opponent_active_pokemon

    def up(pokemon):
        return {
            "species": rc.canon(pokemon.name),
            "hp": rc.hp(pokemon.hp_pct),
            "status": rc.canon(pokemon.status),
            "types": sorted(rc.canon(t) for t in pokemon.types.split(" ") if t),
            "item": rc.canon(pokemon.item),
            "ability": rc.canon(pokemon.ability),
            "boosts": {
                "atk": pokemon.atk_boost, "def": pokemon.def_boost, "spa": pokemon.spa_boost,
                "spd": pokemon.spd_boost, "spe": pokemon.spe_boost,
                "accuracy": pokemon.accuracy_boost, "evasion": pokemon.evasion_boost,
            },
            "moves": rc.moveset(m.name for m in pokemon.moves),
            # `universal_effects` keeps only the MOST RECENT volatile, so this is compared by
            # CONTAINMENT against our full volatile map, never by equality.
            "effect": rc.canon(pokemon.effect),
            "level": int(pokemon.lvl),
        }

    return {
        "active": up(active),
        "opp_active": up(opp),
        "switches": sorted(rc.canon(p.name) for p in state.available_switches),
        "switch_hp": {rc.canon(p.name): rc.hp(p.hp_pct) for p in state.available_switches},
        "weather": rc.canon(state.weather),
        "field": rc.canon(state.battle_field),
        "our_conditions": rc.canon(state.player_conditions),
        "opp_conditions": rc.canon(state.opponent_conditions),
        "forced_switch": bool(state.forced_switch),
        "opponents_remaining": int(state.opponents_remaining),
        "won": bool(state.battle_won),
        "lost": bool(state.battle_lost),
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="one ws_frontend capture json")
    ap.add_argument("--slot", required=True, choices=("p1", "p2"), help="Metamon's slot")
    ap.add_argument("--username", required=True, help="Metamon's username in that battle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="SmallRL",
                    help="the obs space + tokenizer are taken from THIS pretrained model rather "
                         "than named separately — a hand-named pair is a guess, and a tokenizer "
                         "that is not the model's own would fabricate unknown tokens")
    args = ap.parse_args(argv)

    from poke_env.player import Player
    from metamon.interface import UniversalState, UniversalAction
    from metamon.rl.pretrained import get_pretrained_model
    from metamon.tokenizer import UNKNOWN_TOKEN

    ignore = set(Player.MESSAGES_TO_IGNORE)
    # The model's OWN observation space + tokenizer (a `TokenizedObservationSpace`), so the
    # census below is over the tokens this policy actually consumed in the captured games.
    obs_space = get_pretrained_model(args.agent).observation_space
    tokenizer = obs_space.tokenizer

    capture = json.load(open(args.capture))
    battle = build_battle(capture["tag"], args.username)
    obs_space.reset()

    rows = []
    unknown_words = {}
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
            # Metamon's own `from_Battle` would raise here too; record it rather than skip.
            rows.append({"tag": capture["tag"], "seq": seq, "error": "no_active_pokemon"})
            seq += 1
            continue
        state = UniversalState.from_Battle(battle)
        legal = sorted(a.action_idx for a in UniversalAction.definitely_valid_actions(state, battle))
        # 🚨 THE ACTUAL NETWORK INPUT, counted in the array itself — not re-derived.
        # `state_to_obs` is Metamon's own composition (base space -> tokenizer), so the -1s
        # counted here are the -1s the policy consumed. The word names are recovered from the
        # base space's text for the report; the COUNT never depends on that recovery.
        obs = obs_space.state_to_obs(state)
        base = obs_space.base_obs_space.state_to_obs(state)
        n_unknown = 0
        for key in obs_space.base_obs_space.tokenizable:
            tokens = obs[f"{key}_tokens"]
            n_unknown += int((tokens == UNKNOWN_TOKEN).sum())
            text = base[key].tolist()
            if isinstance(text, list):
                text = " ".join(text)
            for word in str(text).split(" "):
                if word and tokenizer[word] == UNKNOWN_TOKEN:
                    unknown_words[word] = unknown_words.get(word, 0) + 1
        rows.append({
            "tag": capture["tag"], "seq": seq, "turn": int(battle.turn),
            "us": universal_block(state), "raw": raw_block(battle),
            "legal_action_idx": legal, "n_unknown_tokens": n_unknown,
        })
        seq += 1

    # 🚨 A comparator that compared nothing must FAIL, not pass. See PREDICTION.md's last bar.
    if not rows:
        raise SystemExit(f"{args.capture}: 0 decision points replayed — refusing to write a "
                         "vacuous dump (a check that never ran reads exactly like one that passed)")
    rc.dump(args.out, rows)
    print(json.dumps({"tag": capture["tag"], "n_decisions": len(rows),
                      "unknown_tokens": sum(r.get("n_unknown_tokens", 0) for r in rows),
                      "unknown_words": unknown_words}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
