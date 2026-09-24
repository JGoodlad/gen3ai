"""Tactics probe library: fixed teams, scripted prefixes, one line to a terminal, model readouts.

Run with the PINNED tree only (the three models trained on 6eb9c776):

    cd <pin worktree> && PYTHONPATH=<pin>/src POKESIM_SIM_BRIDGE_BIN=<pin>/src/rust_sim/target/release/sim_bridge \
        python <this dir>/<script>.py ...

Nothing here writes under models/ (checkpoints are READ). No server: every battle runs on the in-process
Rust bridge (``impl="rust"``) unless a caller asks for node (the rule-verification script does, to compare).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

torch.set_num_threads(1)

from poke_env import AccountConfiguration, LocalhostServerConfiguration  # noqa: E402
from poke_env.player import Player  # noqa: E402
from poke_env.player.battle_order import ForfeitBattleOrder, SingleBattleOrder  # noqa: E402
from poke_env.teambuilder import Teambuilder  # noqa: E402
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder  # noqa: E402
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon  # noqa: E402

from agents.battle.gen3_battle import Gen3Battle  # noqa: E402
from utils.bridge.counterfactual import _battle_outcome, install_scripted_prefix, turn_cap_of  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402
from utils.bridge.reconstruction import ReconstructionRecord  # noqa: E402

FMT = "gen3ou"
MODELS_ROOT = "/home/goodlad/dev/gen3ai/models"
MODELS = {
    "G0": "ai_v13_12_plateau",      # the round-1 parent (also the fixed CONTINUATION policy)
    "B": "ai_v13_22_popr1_loop",    # the population-loop generalist
    "C": "ai_v13_23_popr1_ctrl",    # the control
}
CKPT = "final_model.zip"
OTHER = {"p1": "p2", "p2": "p1"}


# ----------------------------------------------------------------------------------------------
# teams
# ----------------------------------------------------------------------------------------------

def parse_set(text: str) -> TeambuilderPokemon:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return TeambuilderPokemon.from_showdown("\n".join(lines))


def pack(sets: List[str]) -> str:
    """Showdown-text sets (lead FIRST) -> packed team."""
    return Teambuilder.join_team([parse_set(s) for s in sets])


def mint_seed(key: str) -> str:
    return "sodium," + hashlib.sha256(key.encode()).hexdigest()[:32]


def make_record(p1_team: str, p2_team: str, seed: str, commands: List[tuple], our_side: str,
                names=("TPp1", "TPp2")) -> ReconstructionRecord:
    """A hand-built reconstruction record: the scripted prefix IS its command list."""
    input_log = (
        ">start " + json.dumps({"formatid": FMT, "seed": seed}, separators=(",", ":")),
        ">player p1 " + json.dumps({"name": names[0], "team": p1_team}, separators=(",", ":")),
        ">player p2 " + json.dumps({"name": names[1], "team": p2_team}, separators=(",", ":")),
    )
    return ReconstructionRecord(format_id=FMT, prng_seed=seed, input_log=input_log,
                                commands=tuple(tuple(c) for c in commands),
                                trainee_username=names[0] if our_side == "p1" else names[1])


# ----------------------------------------------------------------------------------------------
# a plain scripted player (rule verification only): pops its script, forfeits when it runs out
# ----------------------------------------------------------------------------------------------

class ScriptPlayer(Player):
    def __init__(self, script: List[str], **kw):
        super().__init__(battle_class=Gen3Battle, **kw)
        self.script = list(script)

    def choose_move(self, battle):
        if not self.script:
            return ForfeitBattleOrder()
        return SingleBattleOrder("/choose " + self.script.pop(0))


_NAME_CTR = [0]


def _acct(prefix: str) -> AccountConfiguration:
    _NAME_CTR[0] += 1
    return AccountConfiguration(f"{prefix}{os.getpid() % 10000}x{_NAME_CTR[0]}", None)


def run_scripted(p1_team: str, p2_team: str, s1: List[str], s2: List[str], seed: str,
                 impl: str = "rust") -> Dict[str, Any]:
    """Play a fully scripted battle (both sides), then forfeit. Returns per-side protocol chunks."""
    a = ScriptPlayer(s1, battle_format=FMT, team=ConstantTeambuilder(p1_team),
                     server_configuration=LocalhostServerConfiguration,
                     account_configuration=_acct("SPa"), start_listening=False, max_concurrent_battles=1)
    b = ScriptPlayer(s2, battle_format=FMT, team=ConstantTeambuilder(p2_team),
                     server_configuration=LocalhostServerConfiguration,
                     account_configuration=_acct("SPb"), start_listening=False, max_concurrent_battles=1)
    sink: list = []
    asyncio.run(run_local_battles(a, b, 1, battle_format=FMT, seed=seed, chunk_sink=sink, impl=impl))
    return {"sink": sink, "p1": a, "p2": b,
            "left": {"p1": list(a.script), "p2": list(b.script)}}


def side_lines(sink: list, side: str) -> List[str]:
    out: List[str] = []
    for s, chunk in sink:
        if s == side:
            out.extend(ln for ln in chunk.split("\n") if ln.startswith("|"))
    return out


def turn_block(lines: List[str], turn: int) -> List[str]:
    """Protocol lines of turn ``turn`` (between ``|turn|turn`` and ``|turn|turn+1``/end)."""
    out, on = [], False
    for ln in lines:
        if ln.startswith("|turn|"):
            t = int(ln.split("|")[2])
            if t == turn:
                on = True
                continue
            if on:
                break
        if on:
            out.append(ln)
    return out


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}


def model_path(tag: str) -> str:
    return os.path.join(MODELS_ROOT, MODELS[tag], CKPT)


def load_model(tag: str):
    if tag not in _MODEL_CACHE:
        from main.capacity import load_policy
        m, _, _ = load_policy(model_path(tag), os.path.join(MODELS_ROOT, MODELS[tag]), "cpu")
        m.policy.set_training_mode(False)
        m.policy.features_extractor.disable_observation_debugger()
        _MODEL_CACHE[tag] = m
    return _MODEL_CACHE[tag]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


@torch.no_grad()
def readout(model, obs: np.ndarray, mask: np.ndarray, opp_move_nums: Optional[List[int]] = None) -> dict:
    """Policy probs (T=1, masked), critic V, win-prob head, and (optionally) the opp-ACTIVE move belief."""
    pin = {"observation": torch.as_tensor(obs[None]).float(),
           "action_mask": torch.as_tensor(mask[None]).float()}
    dist = model.policy.get_distribution(pin)
    logits = dist.distribution.logits
    masked = logits + (pin["action_mask"] - 1.0) * 1e9
    probs = torch.softmax(masked, dim=1)[0].numpy()
    fe = model.policy.features_extractor
    wp_l = fe.last_win_prob_logits
    wp = float(torch.sigmoid(wp_l[0, 0]).item()) if wp_l is not None else None
    mb = None
    if opp_move_nums is not None:
        ml = fe.stash.move_belief_logits
        act = fe.last_opp_active_local
        a = int(act[0].item()) if act is not None else 0
        row = torch.sigmoid(ml[0, a]).numpy()
        mb = {int(n): float(row[int(n)]) for n in opp_move_nums}
    v = float(model.policy.predict_values(pin)[0].item())
    return {"probs": [float(x) for x in probs], "v": v, "win_prob": wp, "move_belief": mb}


# ----------------------------------------------------------------------------------------------
# one line: scripted prefix -> substitute at T -> both sides live (continuation policy) to terminal
# ----------------------------------------------------------------------------------------------

def run_line(record: ReconstructionRecord, *, trainee, opponent, T: int,
             substitute: "str | Callable[[Any, Any], str]", post_t_seed: Optional[str] = None,
             hook: Optional[Callable] = None, impl: str = "rust", keep_sink: bool = True,
             timeout: Optional[float] = None) -> dict:
    """``replay_counterfactual`` minus its trajectory summariser: keeps the RAW per-side protocol.

    The opponent's turn-T command is replayed if the record holds one (the PREMISE regime) and is
    chosen live by the opponent's policy if it does not (the LIVE regime: its script is exhausted)."""
    our_side = record.side_of(record.trainee_username)
    opp_side = OTHER[our_side]
    trainee._team = ConstantTeambuilder(record.packed_team(our_side))
    opponent._team = ConstantTeambuilder(record.packed_team(opp_side))
    our_state = install_scripted_prefix(trainee, side=our_side, record=record, divergence_turn=T,
                                        substitute_choice=substitute, is_our_side=True,
                                        on_scripted_decision=hook)
    opp_state = install_scripted_prefix(opponent, side=opp_side, record=record, divergence_turn=T,
                                        substitute_choice=None, is_our_side=False)
    start_extra = ({"resumeReseed": {"turn": int(T), "seed": post_t_seed}} if post_t_seed else None)
    sink: list = []
    p1, p2 = (trainee, opponent) if our_side == "p1" else (opponent, trainee)
    coro = run_local_battles(p1, p2, 1, battle_format=FMT, seed=record.start_options()["seed"],
                             start_extra=start_extra, chunk_sink=sink, impl=impl)
    asyncio.run(asyncio.wait_for(coro, timeout) if timeout else coro)
    res = _battle_outcome(trainee, record.username(our_side), turn_cap=turn_cap_of(trainee, opponent))
    res["substitute"] = our_state.get("substitute_resolved")
    res["our_exhausted"] = bool(our_state["exhausted"])
    res["opp_exhausted"] = bool(opp_state["exhausted"])
    if keep_sink:
        res["sink"] = sink
    return res


def rollout_score(res: dict) -> float:
    """win 1 / loss 0 / tie 0.5; a line that hit the stall CAP is 0.5 whatever it says
    (the cap's outcome is decided by forfeit ORDER, not by the position; see counterfactual._battle_outcome)."""
    if res.get("capped"):
        return 0.5
    return {"win": 1.0, "loss": 0.0, "tie": 0.5}.get(res.get("outcome"), float("nan"))


def make_rl(model, *, prefix: str, stochastic: bool = True, temperature: float = 1.0,
            policy_seed: Optional[int] = None, cls=None):
    from agents.inference.player import RLPlayer
    from agents.observation.state_encoder import load_mappings
    global _MAPPINGS
    try:
        _MAPPINGS
    except NameError:
        _MAPPINGS = load_mappings()
    cls = cls or RLPlayer
    return cls(model=model, team=ConstantTeambuilder(""), battle_format=FMT,
               server_configuration=LocalhostServerConfiguration, mappings=_MAPPINGS,
               account_configuration=_acct(prefix), start_listening=False, max_concurrent_battles=1,
               stochastic=stochastic, temperature=temperature, policy_seed=policy_seed)
