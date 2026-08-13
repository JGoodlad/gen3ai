"""A synthetic run directory, good enough for every read-only view.

Shared by `app_test.py` (in-process) and `render_integration_test.py` (which must build the same
run in a subprocess the browser can reach), so the two cannot test different data. Deliberately
NOT a pytest fixture: the render test spawns `python -m main.prober.web` and needs to build the
run from a plain function first.

It writes only what the model-free views read — `*_summary.json` + `*_states.npz` + an
`eval_manifest.json` + `metadata.json` — mirroring `session_test.py`'s builders. No checkpoint,
no `reconstruction.json`: the heavy re-roll probes are expected to fail loudly on this run, and
`app_test.py` asserts that failure renders as a message rather than a 500.
"""

from __future__ import annotations

import json
import os

import numpy as np

_OBS_LEN = 256
OPPONENTS = ("aggressive_v2", "heuristic2")
STEPS = (2000000, 4000000)


def _inv(turn: int, chosen: str, reward_total: float, *, faint: bool = False,
         species=("zapdos", "jynx"), hp=None, outcome=None, opp_intent=None) -> dict:
    """One recorded decision. `hp` is `(our, opp)` display strings and `outcome` the recorded
    per-side `{action, hp_delta}` pair — both optional, because most views only need the board and
    the reward, and only the turn-by-turn replay reads them. `opp_intent` is the v67 α/β block,
    present on only SOME decisions here on purpose: it is absent from every pre-v67 trace, so the
    views have to render both cases.
    """
    our, opp = {"species": species[0]}, {"species": species[1]}
    if hp:
        our["hp"], opp["hp"] = hp
    out = {"reward": {"total": reward_total},
           "events": (["our:zapdos:fainted"] if faint else [])}
    out.update(outcome or {})
    return {"i": turn, "turn": turn, "phase": "move_selection", "chosen": chosen,
            "our": our, "opp": opp,
            **({"opp_intent": opp_intent} if opp_intent else {}),
            "actions": {chosen: {"prob": "80.0%", "valid": True}},
            "outcome": out}


# Two shapes of the opponent-intent block, because they render differently: one where α expects an
# ATTACK (β is irrelevant and stays in the drop-down) and one where α expects a SWITCH (β's named
# mon is promoted onto the card). A fixture carrying only the first would leave the β path ungated.
_INTENT_ATTACK = {
    "alpha": [{"name": "earthquake", "p": 0.52}, {"name": "SWITCH", "p": 0.21},
              {"name": "icebeam", "p": 0.18}, {"name": "toxic", "p": 0.09}],
    "beta": [{"slot": 3, "p": 0.44, "species": "blissey"},
             {"slot": 4, "p": 0.31, "species": "skarmory"}],
}
_INTENT_SWITCH = {
    "alpha": [{"name": "SWITCH", "p": 0.61}, {"name": "earthquake", "p": 0.24},
              {"name": "toxic", "p": 0.15}],
    "beta": [{"slot": 2, "p": 0.58, "species": "blissey"},
             {"slot": 5, "p": 0.22, "species": None}],
}


def _write_battle(run: str, step: int, opponent: str, name: str, invs, values) -> None:
    bd = os.path.join(run, "eval_traces", f"step_{step}", opponent)
    os.makedirs(bd, exist_ok=True)
    outcome = name.split("_")[0]
    summary = {"meta": {"step": step, "result": outcome.upper(), "turns": len(invs),
                        "invocations": len(invs)}, "invocations": invs}
    with open(os.path.join(bd, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f)
    np.savez(os.path.join(bd, f"{name}_states.npz"),
             obs=np.zeros((len(invs), _OBS_LEN), dtype=np.float32),
             has_state=np.ones(len(invs), dtype=np.int8),
             values=np.array(values, dtype=np.float32))
    manifest = os.path.join(os.path.dirname(bd), "eval_manifest.json")
    if not os.path.exists(manifest):
        with open(manifest, "w") as f:
            json.dump({"step": step, "git_hash": f"deadbeef{step}",
                       "arch_signature": "gen3_fixture_v1", "snapshot": None}, f)


def build(root: str) -> str:
    """Write the fixture run under `root` and return its path.

    Shapes chosen so the views have something to rank: two steps × two opponents, losses whose
    worst ΔV differs (so `scan`'s global ranking is observable), a faint in the deepest crater
    (so `triage` has a category to attribute), and a couple of wins so the outcome filter has
    something to exclude.
    """
    run = os.path.join(root, "run_fixture")
    os.makedirs(run, exist_ok=True)

    # step 1 — the deepest crater lives here (ΔV −8 at inv 0), with a faint.
    _write_battle(run, STEPS[0], OPPONENTS[0], "loss_001",
                  [_inv(1, "earthquake", -1.0), _inv(2, "switch:m0", 0.5, faint=True)],
                  [10.0, 2.0, 6.0])
    _write_battle(run, STEPS[0], OPPONENTS[0], "loss_002",
                  [_inv(1, "fireblast", -10.0, faint=True)], [5.0, 1.0])
    _write_battle(run, STEPS[0], OPPONENTS[1], "win_001",
                  [_inv(1, "thunderbolt", 0.5), _inv(2, "thunderbolt", 1.0)], [3.0, 8.0])

    # step 2 — a shallower crater, so ordering across steps is visible. This one also carries the
    # BOARD (hp) and the recorded per-side ACTIONS, which is what the turn-by-turn replay folds into
    # a battle log — without them every view still works but the log is empty, and a render test
    # over an empty log proves nothing about the feature it is meant to gate.
    _write_battle(run, STEPS[1], OPPONENTS[1], "loss_003",
                  [_inv(1, "icebeam", -0.5, hp=("100%", "100%"), opp_intent=_INTENT_ATTACK,
                        outcome={"our": {"action": "icebeam", "hp_delta": "-22%"},
                                 "opp": {"action": "earthquake", "hp_delta": "-31%"}}),
                   _inv(2, "roost", -2.0, faint=True, hp=("78%", "69%"), opp_intent=_INTENT_SWITCH,
                        outcome={"our": {"action": "roost", "hp_delta": "-100%"},
                                 "opp": {"action": "earthquake", "hp_delta": "0%"}})],
                  [7.0, 5.0, 1.0])
    _write_battle(run, STEPS[1], OPPONENTS[1], "win_002",
                  [_inv(1, "thunderbolt", 2.0)], [4.0, 9.0])

    # a checkpoint path gives the model-resolution ladder something to name (never loaded here).
    open(os.path.join(run, "checkpoint_3200000_steps.zip"), "w").close()
    with open(os.path.join(run, "metadata.json"), "w") as f:
        json.dump({"gamma": 0.95}, f)
    return run
