"""THE STORY BATTLE of the synthetic fixture run (`fixture_run.build`): a six-turn gen-3 battle written
as the PROTOCOL a Rust-core trace carries (our side's HP in exact points, theirs in hundredths) plus
a reconstruction record, so the battle viewer's turn story, its perspective board and its
ground-truth overlay are gated on every shape the owner asked to be readable
(`designs/prober/battle_viewer_ux_2026-10-09.md` §4–§5):

  T1  a voluntary switch (ours) + their move + residual Leftovers
  T2  move ORDER (ours first) + a hazard set + a critical hit
  T3  their voluntary switch, Spikes chip on the way in, a super-effective hit, sand + Leftovers
  T4  our KO → THEIR forced replacement mid-turn (+ Spikes on it)
  T5  their KO of our mon before it moved → OUR forced replacement (decision 5) + "never got to move"
  T6  both move, the last KO, a forfeit, the win

The reconstruction record PLANTS facts the protocol never reveals, so the perspective guard has
something to catch (`perspective_guard_test.py`): theirs — Tyranitar's Choice Band and Focus Punch,
Starmie's Rapid Spin, and three mons that never appear (Smeargle @ Lax Incense with Spore among them);
ours — Blissey (never sent in) holding a Shell Bell and carrying Aromatherapy.
"""

from __future__ import annotations

import json
import os

import numpy as np

STORY_STEP = 6000000
STORY_OPPONENT = "heuristic2"
STORY_NAME = "win_005"
STORY_SHORT_ID = f"step_{STORY_STEP}/{STORY_OPPONENT}/win_005"
#: Facts ONLY ground truth holds (never on the protocol): absent from the model / spectator views.
STORY_HIDDEN = ("Choice Band", "Focus Punch", "Rapid Spin", "Smeargle", "Lax Incense", "Spore")
#: Our own private facts (the model knew them; a spectator never saw them): absent from spectator.
STORY_OURS = ("Shell Bell", "Aromatherapy")

_OBS_LEN = 256

STORY_LOG = """|j|☆trainee
|player|p1|trainee|1|
|player|p2|foe|2|
|teamsize|p1|6
|teamsize|p2|6
|gametype|singles
|gen|3
|tier|[Gen 3] OU
|
|start
|switch|p1a: Swampert|Swampert, M|404/404
|switch|p2a: Tyranitar|Tyranitar, M|100/100
|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Tyranitar
|turn|1
|
|t:|1
|switch|p1a: Skarmory|Skarmory, F|334/334
|move|p2a: Tyranitar|Rock Slide|p1a: Skarmory
|-resisted|p1a: Skarmory
|-damage|p1a: Skarmory|287/334
|
|-weather|Sandstorm|[upkeep]
|-heal|p1a: Skarmory|308/334|[from] item: Leftovers
|upkeep
|turn|2
|
|t:|1
|move|p1a: Skarmory|Spikes|p2a: Tyranitar
|-sidestart|p2: foe|Spikes
|move|p2a: Tyranitar|Rock Slide|p1a: Skarmory
|-crit|p1a: Skarmory
|-resisted|p1a: Skarmory
|-damage|p1a: Skarmory|215/334
|
|-weather|Sandstorm|[upkeep]
|-heal|p1a: Skarmory|236/334|[from] item: Leftovers
|upkeep
|turn|3
|
|t:|1
|switch|p2a: Celebi|Celebi|100/100
|-damage|p2a: Celebi|88/100|[from] Spikes
|move|p1a: Skarmory|Drill Peck|p2a: Celebi
|-supereffective|p2a: Celebi
|-damage|p2a: Celebi|41/100
|
|-weather|Sandstorm|[upkeep]
|-damage|p2a: Celebi|35/100|[from] Sandstorm
|-heal|p2a: Celebi|41/100|[from] item: Leftovers
|-heal|p1a: Skarmory|257/334|[from] item: Leftovers
|upkeep
|turn|4
|
|t:|1
|move|p1a: Skarmory|Drill Peck|p2a: Celebi
|-supereffective|p2a: Celebi
|-damage|p2a: Celebi|0 fnt
|faint|p2a: Celebi
|
|switch|p2a: Starmie|Starmie|100/100
|-damage|p2a: Starmie|88/100|[from] Spikes
|
|-weather|Sandstorm|[upkeep]
|-damage|p2a: Starmie|82/100|[from] Sandstorm
|-heal|p2a: Starmie|88/100|[from] item: Leftovers
|-heal|p1a: Skarmory|278/334|[from] item: Leftovers
|upkeep
|turn|5
|
|t:|1
|move|p2a: Starmie|Thunderbolt|p1a: Skarmory
|-supereffective|p1a: Skarmory
|-damage|p1a: Skarmory|0 fnt
|faint|p1a: Skarmory
|
|switch|p1a: Swampert|Swampert, M|404/404
|
|-weather|Sandstorm|[upkeep]
|-damage|p2a: Starmie|82/100|[from] Sandstorm
|-heal|p2a: Starmie|88/100|[from] item: Leftovers
|upkeep
|turn|6
|
|t:|1
|move|p2a: Starmie|Surf|p1a: Swampert
|-damage|p1a: Swampert|303/404
|move|p1a: Swampert|Earthquake|p2a: Starmie
|-damage|p2a: Starmie|0 fnt
|faint|p2a: Starmie
|
|-message|foe forfeited.
|
|win|trainee
"""

_OUR_TEAM = ("Swampert||leftovers|torrent|earthquake,surf,icebeam,protect|Relaxed|252,,252,,4,|||||]"
             "Skarmory||leftovers|keeneye|spikes,drillpeck,roar,toxic|Impish|252,,252,,4,|||||]"
             "Blissey||shellbell|naturalcure|softboiled,seismictoss,aromatherapy,icebeam|Bold|252,,252,,4,|||||]"
             "Gengar||leftovers|levitate|thunderbolt,icepunch,firepunch,explosion|Timid|,,,252,4,252|||||]"
             "Metagross||lumberry|clearbody|meteormash,earthquake,rockslide,explosion|Adamant|252,252,,,4,|||||]"
             "Jolteon||leftovers|voltabsorb|thunderbolt,batonpass,substitute,hiddenpowergrass|Timid|,,,252,4,252|||||")
_OPP_TEAM = ("Tyranitar||choiceband|sandstream|rockslide,earthquake,focuspunch,pursuit|Adamant|,252,,,4,252|||||]"
             "Celebi||leftovers|naturalcure|gigadrain,psychic,recover,leechseed|Bold|252,,252,,4,|||||]"
             "Starmie||leftovers|naturalcure|surf,thunderbolt,recover,rapidspin|Timid|,,,252,4,252|||||]"
             "Smeargle||laxincense|owntempo|spore,spikes,batonpass,substitute|Jolly|252,,4,,,252|||||]"
             "Dugtrio||choiceband|arenatrap|earthquake,rockslide,aerialace,substitute|Jolly|,252,,,4,252|||||]"
             "Zapdos||leftovers|pressure|thunderbolt,hiddenpowerice,thunderwave,roar|Modest|252,,,252,4,|||||")

_TEAM = ("swampert", "skarmory", "blissey", "gengar", "metagross", "jolteon")


def _actions(active_moves, probs, unavailable=()):
    """The recorded action distribution in ACTION-INDEX order (6 switches, 4 moves, Struggle)."""
    labels = [f"switch:{s}" for s in _TEAM] + list(active_moves) + ["struggle"]
    return {lab: {"prob": f"{probs.get(lab, 0.0) * 100:.1f}%",
                  "valid": lab != "struggle" and lab not in unavailable}
            for lab in labels}


def _inv(i, turn, chosen, our, opp, acts, opp_action, *, phase="move_selection", our_hp="100%",
         opp_hp="100%"):
    assert chosen in acts
    return {"i": i, "turn": turn, "phase": phase, "chosen": chosen,
            "our": {"species": our, "hp": our_hp}, "opp": {"species": opp, "hp": opp_hp},
            "actions": acts,
            "outcome": {"reward": {"total": 0.0}, "events": [], "opp": {"action": opp_action}}}


def write_story_battle(run: str) -> str:
    """Write the story battle under ``run`` (summary + states + protocol + reconstruction); returns
    its directory."""
    bd = os.path.join(run, "eval_traces", f"step_{STORY_STEP}", STORY_OPPONENT)
    os.makedirs(bd, exist_ok=True)
    sw = ("earthquake", "surf", "icebeam", "protect")
    sk = ("spikes", "drillpeck", "roar", "toxic")
    bench = {"switch:blissey": 0.03, "switch:gengar": 0.02, "switch:metagross": 0.02, "switch:jolteon": 0.01}
    invs = [
        _inv(0, 1, "switch:skarmory", "swampert", "tyranitar",
             _actions(sw, {"switch:skarmory": 0.55, "earthquake": 0.2, "surf": 0.1, "icebeam": 0.03,
                           "protect": 0.02, **bench}, ("switch:swampert",)), "rockslide"),
        _inv(1, 2, "spikes", "skarmory", "tyranitar",
             _actions(sk, {"spikes": 0.6, "drillpeck": 0.2, "roar": 0.08, "toxic": 0.02,
                           "switch:swampert": 0.02, **bench}, ("switch:skarmory",)), "rockslide",
             our_hp="92%"),
        _inv(2, 3, "drillpeck", "skarmory", "tyranitar",
             _actions(sk, {"drillpeck": 0.5, "spikes": 0.1, "roar": 0.2, "toxic": 0.05,
                           "switch:swampert": 0.07, **bench}, ("switch:skarmory",)), "switched_to:celebi",
             our_hp="71%"),
        _inv(3, 4, "drillpeck", "skarmory", "celebi",
             _actions(sk, {"drillpeck": 0.7, "spikes": 0.05, "roar": 0.1, "toxic": 0.07,
                           "switch:swampert": 0.02, **bench}, ("switch:skarmory",)), "none",
             our_hp="77%", opp_hp="41%"),
        _inv(4, 5, "drillpeck", "skarmory", "starmie",
             _actions(sk, {"drillpeck": 0.4, "spikes": 0.05, "roar": 0.25, "toxic": 0.05,
                           "switch:swampert": 0.17, **bench}, ("switch:skarmory",)), "thunderbolt",
             our_hp="83%", opp_hp="88%"),
        _inv(5, 5, "switch:swampert", "skarmory", "starmie",
             _actions(("move0", "move1", "move2", "move3"),
                      {"switch:swampert": 0.6, "switch:blissey": 0.2, "switch:gengar": 0.1,
                       "switch:metagross": 0.06, "switch:jolteon": 0.04},
                      ("switch:skarmory", "move0", "move1", "move2", "move3")), "none",
             phase="forced_switch", our_hp="0%", opp_hp="88%"),
        _inv(6, 6, "earthquake", "swampert", "starmie",
             _actions(sw, {"earthquake": 0.8, "surf": 0.1, "icebeam": 0.05, "protect": 0.05},
                      ("switch:swampert", "switch:skarmory")), "surf", opp_hp="88%"),
    ]
    summary = {"meta": {"step": STORY_STEP, "result": "WIN", "turns": 6, "invocations": len(invs)},
               "invocations": invs}
    with open(os.path.join(bd, f"{STORY_NAME}_summary.json"), "w") as f:
        json.dump(summary, f)
    np.savez(os.path.join(bd, f"{STORY_NAME}_states.npz"),
             obs=np.zeros((len(invs), _OBS_LEN), dtype=np.float32),
             has_state=np.ones(len(invs), dtype=np.int8),
             values=np.array([1.0, 1.4, 2.0, 3.5, 0.5, -0.4, 4.0, 9.0], dtype=np.float32),
             win_probs=np.array([0.55, 0.58, 0.63, 0.74, 0.52, 0.47, 0.81], dtype=np.float32))
    with open(os.path.join(bd, f"{STORY_NAME}_replay.html"), "w", encoding="utf-8") as f:
        f.write('<!DOCTYPE html><html><body><script type="text/plain" class="battle-log-data">\n'
                + STORY_LOG + "</script></body></html>\n")
    rec = {"v": 1, "format_id": "gen3ou", "prng_seed": "1,2,3,4",
           "input_log": ['>start {"formatid":"gen3ou","seed":"1,2,3,4"}',
                         ">player p1 " + json.dumps({"name": "trainee", "team": _OUR_TEAM}),
                         ">player p2 " + json.dumps({"name": "foe", "team": _OPP_TEAM})],
           "commands": [], "battle_tag": "fixture-story", "trainee_username": "trainee"}
    with open(os.path.join(bd, f"{STORY_NAME}_reconstruction.json"), "w") as f:
        json.dump(rec, f)
    manifest = os.path.join(os.path.dirname(bd), "eval_manifest.json")
    if not os.path.exists(manifest):
        with open(manifest, "w") as f:
            json.dump({"step": STORY_STEP, "git_hash": f"deadbeef{STORY_STEP}",
                       "arch_signature": "gen3_fixture_v1", "snapshot": None}, f)
    return bd
