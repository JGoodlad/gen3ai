"""SLICE N for EPISODES AND REWARD (M5 Lane D): the Rust env core's ``reward`` / ``terminated`` /
``truncated`` equal ``Gen3Env``'s on the same battles — the terminal reward (indicator and signed),
the stall forfeit at the threshold, natural endings and ties — through BOTH front ends. No allowlist.

RECORD in Rust, REPLAY in Python (Lane 0's direction, Lane C's harness shape):

1. the env core plays each episode under a seeded random policy on both sides, once through the FFI
   front end (``FfiCore``) and once through the process front end (``ProcCore``); the two recordings
   (every action, every p1 row, every op's ``reward`` / ``done`` / ``terminated`` / ``truncated``)
   must be IDENTICAL;
2. one ``Gen3Env`` built as training builds the trainee's (``trainee_env_kwargs(production_args())``,
   the rust bridge, ``--obs-source core``) replays each episode — the same teams and seed, p1's
   recorded indices, and an opponent playing p2's recorded indices through the real mapper — with the
   SAME ``StallConfig`` threshold and ``RewardConfig`` the core's spec declares;
3. at every trainee decision the ``observation`` row must equal the core's (alignment), every
   non-final Python reward must be 0, and the episode's final ``(reward, terminated, truncated)``
   must equal the core's ``done`` op.

Per-STEP alignment is deliberately NOT compared: the core has no phantom (wait) steps, so a per-op
sequence is Lane G's rollout-level gate; here the reward is terminal-only, so the episode's end IS
the comparison.

COMMIT tier (routine): natural endings (production terminal), stall forfeits (production terminal
at a low threshold), a signed-terminal mix (forfeit = ``draw_penalty`` at the cap, natural ±), and a
TIE (Explosion, last mon vs last mon — gen 3 has no self-KO rule, so both sides wipe). MILESTONE
(``slow``): pool and ladder at scale, production terminal and threshold.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from utils.paths import src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")
NAMES = ("ldpone", "ldptwo")
SIGNED = {"victory_value": 30.0, "terminal_indicator": False, "draw_penalty": -35.0}

#: A deterministic TIE: p1's only move is Explosion (Metagross, slower), p2's Pikachu attacks first
#: and cannot KO; Explosion KOs it. Both last mons faint ⇒ ``|tie|`` in gen 3.
TIE_TEAMS = ("Metagross||Leftovers|ClearBody|explosion|Adamant|4,252,,,,252|N||||",
             "Pikachu||Leftovers|Static|thundershock|Timid|4,,,252,,252|N||||")


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the episode parity gate cannot run (install rustup)")
    return cargo


@pytest.fixture(scope="module")
def built():
    from utils.rust_env import ffi, proc

    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run([_cargo(), "build", "--lib", "--bin", proc.BIN_NAME, *FEATURES, "--manifest-path",
                        str(crate / "Cargo.toml")], env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib + child failed:\n{r.stderr[-4000:]}"
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True), proc.default_path("selfcheck")


def _seed(key: int):
    return [(11 + key) % 65536, (22 + key) % 65536, (33 + key) % 65536, (44 + key) % 65536]


def _terminal(signed_cap):
    from utils.rust_env import episode as EP

    if signed_cap is None:
        return dict(EP.PRODUCTION_TERMINAL)
    return dict(SIGNED, timeout_turn_cap=signed_cap)


# --------------------------------------------------------------------------------- 1. record

def record(core_factory, teams, n_episodes: int, *, key_base: int, threshold: int, signed_cap=None):
    """Play ``n_episodes`` (episode e: team 2e vs 2e + 1, seed ``_seed(key_base + e)``) in a core
    built by ``core_factory(spec_json)``; return per episode ``{"p1", "p2", "rows", "end"}``."""
    from utils.rust_env import protocol as P

    spec = P.spec_json(n=1, threads=1, teams=list(teams[:2 * n_episodes]), names=NAMES, decision_tense=False,
                       switch_freeze=False, turn_limit=threshold, refusal_budget=0, bank_dir=None,
                       terminal=_terminal(signed_cap))
    rng = np.random.default_rng(key_base)
    out = {e: {"p1": [], "p2": [], "rows": [], "end": None} for e in range(n_episodes)}
    with core_factory(spec) as core:
        c = core.cols

        def stage(e):
            e = min(e, n_episodes - 1)
            c["ep_team"][0] = [2 * e, 2 * e + 1]
            c["ep_seed"][0] = _seed(key_base + e)

        stage(0)
        core.reset()
        stage(1)
        done = 0
        for _op in range(200_000):
            if done == n_episodes:
                break
            ep = int(c["episode"][0])
            need = c["need"][0].copy()
            if need[0]:
                out[ep]["rows"].append((int(c["dec_n"][0, 0]), c["obs"][0, 0].copy()))
            for s in range(2):
                c["action"][0, s] = -1
                if need[s]:
                    a = int(rng.choice(np.flatnonzero(c["mask"][0, s])))
                    c["action"][0, s] = a
                    out[ep]["p1" if s == 0 else "p2"].append(a)
            core.step()
            assert not c["refused"][0], f"episode {ep} was quarantined: {core.bank()[-1:]}"
            end = (float(c["reward"][0]), int(c["terminated"][0]), int(c["truncated"][0]))
            if c["done"][0]:
                out[ep]["end"] = end
                done += 1
                stage(done + 1)
            else:
                assert end == (0.0, 0, 0), f"episode {ep}: an outcome {end} on an op that ended nothing"
        assert done == n_episodes
        assert all(v == 0 for v in core.after_freeze().values()), core.after_freeze()
    return out


def record_both(built, teams, n_episodes, **kw):
    """Record through the FFI and the process front end; the recordings must be identical."""
    from utils.rust_env import ffi, proc

    lib, binary = built
    a = record(lambda s: ffi.FfiCore(s, lib=lib), teams, n_episodes, **kw)
    b = record(lambda s: proc.ProcCore(s, binary=binary, nan_poison=True), teams, n_episodes, **kw)
    for e in a:
        for k in ("p1", "p2", "end"):
            assert a[e][k] == b[e][k], (e, k, a[e][k], b[e][k])
        assert len(a[e]["rows"]) == len(b[e]["rows"])
        for (na, ra), (nb, rb) in zip(a[e]["rows"], b[e]["rows"]):
            assert na == nb and ra.tobytes() == rb.tobytes(), (e, na)
    return a


# --------------------------------------------------------------------------------- 2. replay

def _index_script_player():
    from poke_env.player.battle_order import DefaultBattleOrder
    from poke_env.player.player import Player

    from agents.action.mapper import Gen3ActionMapper
    from agents.action.mask_generator import Gen3ActionMasker
    from agents.battle.live_view import LegalActions

    class IndexScriptPlayer(Player):
        """Plays recorded action INDICES through the real mapper, only when agent2 really moves."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.actions, self.j, self.env = [], 0, None

        def choose_move(self, battle):
            if self.env is None or not self.env.agent2_to_move:
                return DefaultBattleOrder()
            legal = LegalActions.from_battle(battle)
            mask = Gen3ActionMasker.get_mask(battle, legal=legal)
            assert self.j < len(self.actions), f"p2 asked for decision {self.j}; the core recorded {len(self.actions)}"
            a = self.actions[self.j]
            self.j += 1
            assert mask[a], f"p2 decision {self.j - 1}: recorded index {a} is illegal in Python (mask {mask})"
            return Gen3ActionMapper.action_to_order(a, battle, legal=legal, mask=mask)

    return IndexScriptPlayer


def replay(recs, teams, *, key_base: int, threshold: int, signed_cap=None, tag: str = "LD", monkeypatch=None):
    """Replay every recorded episode through a production-surface ``Gen3Env`` with the same stall
    threshold and terminal; return ``(divergences {kind: count}, examples, counts)``."""
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    from agents.observation.state_encoder import load_mappings
    from agents.training import reward_manager
    from agents.training.gen3_env import Gen3Env
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from agents.training.stall import StallConfig
    from main.rust_core_cutover.envs import SequenceTeambuilder, production_args
    from main.train.env_factory import trainee_env_kwargs
    from utils.bridge.bridge_session import attach_bridge_transport

    args = production_args()
    rc = RewardConfig.from_args(args)
    if signed_cap is not None:
        rc.victory_value, rc.terminal_indicator, rc.draw_penalty = (
            SIGNED["victory_value"], SIGNED["terminal_indicator"], SIGNED["draw_penalty"])
        # The reward's timeout cap is a module constant (== StallConfig().threshold); the signed arm
        # lowers it with the stall threshold so the cap-forfeit branch is reachable in a short battle.
        monkeypatch.setattr(reward_manager, "_TIMEOUT_TURN_CAP", signed_cap)
    kw = trainee_env_kwargs(args)
    kw["obs_source"] = "core"
    kw["stall_config"] = StallConfig(threshold=threshold)
    n = len(recs)
    env = Gen3Env(load_mappings(), battle_format="gen3ou",
                  team=SequenceTeambuilder([teams[2 * e] for e in range(n)]),
                  opponent_team=SequenceTeambuilder([teams[2 * e + 1] for e in range(n)]),
                  reward_fn=Gen3RewardManager(config=rc),
                  account_configuration1=AccountConfiguration(f"{tag}e"[:18], None),
                  start_listening=False, **kw)
    session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True, impl="rust", core_obs=True)
    opp = _index_script_player()(battle_format="gen3ou", account_configuration=AccountConfiguration(f"{tag}o"[:18], None),
                                 start_listening=False)
    opp.env = env
    w = SingleAgentWrapper(env, opp)
    w.action_space, w.observation_space = env.action_space, env.observation_space
    div, ex = {}, {}
    counts = {"episodes": 0, "decisions": 0, "terminated": 0, "truncated": 0, "forfeits": 0, "ties": 0,
              "wins": 0, "rewards": {}}

    def diverge(k, where, detail):
        div[k] = div.get(k, 0) + 1
        ex.setdefault(k, repr((where, detail))[:1500])

    try:
        for e in range(n):
            rec = recs[e]
            session.seed = _seed(key_base + e)
            opp.actions, opp.j = list(rec["p2"]), 0
            obs, _ = w.reset()
            k, final = 0, None
            for _step in range(5000):
                if env.agent1_to_move:
                    where = (e, k, int(getattr(env.battle1, "turn", -1)))
                    if k >= len(rec["rows"]):
                        diverge("[decisions]", where, f"Python decision {k} past the core's {len(rec['rows'])}")
                        break
                    dec_n, row = rec["rows"][k]
                    if dec_n != k or np.asarray(obs["observation"], dtype=np.float32).tobytes() != row.tobytes():
                        diverge("observation", where, "row differs — alignment lost")
                        break
                    act = rec["p1"][k]
                    k += 1
                    counts["decisions"] += 1
                else:
                    act = 0
                obs, r, term, trunc, _i = w.step(act)
                if term or trunc:
                    final = (float(np.float32(r)), int(bool(term)), int(bool(trunc)))
                    break
                if r != 0:
                    diverge("[nonterminal reward]", (e, k), r)
            else:
                raise AssertionError(f"episode {e} did not end in 5000 steps")
            if final is None:
                continue
            if final != rec["end"]:
                diverge("end", (e, int(env.battle1.turn), env.battle1.won), {"python": final, "core": rec["end"]})
            if k != len(rec["rows"]):
                diverge("[decisions]", (e, k), f"Python took {k} p1 decisions, the core {len(rec['rows'])}")
            if opp.j != len(rec["p2"]):
                diverge("[p2 decisions]", (e, opp.j), f"Python p2 took {opp.j}, the core {len(rec['p2'])}")
            b1 = env.battle1
            counts["episodes"] += 1
            counts["terminated"] += final[1]
            counts["truncated"] += final[2]
            counts["forfeits"] += int(b1.turn >= threshold and b1.won is False)
            counts["ties"] += int(b1.won is None)
            counts["wins"] += int(b1.won is True)
            counts["rewards"][final[0]] = counts["rewards"].get(final[0], 0) + 1
    finally:
        w.close()
    return div, ex, counts


def run_slice(built, teams, n_episodes, *, key_base, threshold=None, signed_cap=None, monkeypatch=None):
    from utils.rust_env import episode as EP

    threshold = EP.stall_threshold() if threshold is None else threshold
    recs = record_both(built, teams, n_episodes, key_base=key_base, threshold=threshold, signed_cap=signed_cap)
    div, ex, counts = replay(recs, teams, key_base=key_base, threshold=threshold, signed_cap=signed_cap,
                             tag=f"LD{key_base % 10000}", monkeypatch=monkeypatch)
    print({"threshold": threshold, "signed_cap": signed_cap, **counts, "divergences": div})
    assert not div, (div, ex)
    assert counts["episodes"] == n_episodes, counts
    return counts


def _teams(source, n_episodes, key_base):
    from main.rust_core_cutover.envs import packed_teams

    pool = packed_teams(source)
    step = 7919
    teams = [pool[(key_base + i * step) % len(pool)] for i in range(2 * n_episodes)]
    for e in range(n_episodes):
        if teams[2 * e] == teams[2 * e + 1]:
            teams[2 * e + 1] = pool[(key_base + (2 * e + 1) * step + 1) % len(pool)]
    return teams


# --------------------------------------------------------------------------------- the gates

def test_the_production_terminal_default_is_the_production_surface():
    """``spec_json``'s default terminal IS what training runs (so a default can never drift)."""
    from agents.training.baselines import production_config
    from agents.training.reward_weights import _TIMEOUT_TURN_CAP
    from agents.training.stall import StallConfig
    from utils.rust_env import episode as EP

    pc = production_config()
    assert EP.PRODUCTION_TERMINAL == {"victory_value": float(pc["victory_value"]),
                                      "terminal_indicator": bool(pc["terminal_indicator"]),
                                      "draw_penalty": float(pc["draw_penalty"]),
                                      "timeout_turn_cap": int(_TIMEOUT_TURN_CAP)}
    assert EP.stall_threshold() == StallConfig().threshold == _TIMEOUT_TURN_CAP


def test_commit_natural_endings_equal_gen3env(built):
    c = run_slice(built, _teams("pool", 6, 63_000), 6, key_base=63_000)
    assert c["terminated"] >= 4, c


def test_commit_the_stall_forfeit_equals_gen3env(built):
    c = run_slice(built, _teams("pool", 4, 63_100), 4, key_base=63_100, threshold=5)
    assert c["forfeits"] == c["truncated"] == 4, c


def test_commit_the_signed_terminal_equals_gen3env(built, monkeypatch):
    c = run_slice(built, _teams("pool", 8, 63_200), 8, key_base=63_200, threshold=45, signed_cap=45,
                  monkeypatch=monkeypatch)
    assert -35.0 in c["rewards"] and len(c["rewards"]) >= 2, c


def test_commit_a_tie_equals_gen3env(built):
    c = run_slice(built, list(TIE_TEAMS), 1, key_base=63_300)
    assert c["ties"] == 1 and c["truncated"] == 1, c


def test_the_episode_slice_has_teeth(built, monkeypatch):
    """A Python terminal that differs from the core's must fail the slice."""
    from agents.training import reward_manager

    real = reward_manager.Gen3RewardManager.process_turn_reward

    def shifted(self, battle, delta):
        r = real(self, battle, delta)
        return r + 0.5 if battle.finished else r

    monkeypatch.setattr(reward_manager.Gen3RewardManager, "process_turn_reward", shifted)
    with pytest.raises(AssertionError, match="'end'"):
        run_slice(built, _teams("pool", 1, 63_400), 1, key_base=63_400)


@pytest.mark.slow
@pytest.mark.parametrize("source,n", [("pool", 60), ("ladder", 60)], ids=["pool", "ladder"])
def test_milestone_episodes_equal_gen3env(built, source, n):
    run_slice(built, _teams(source, n, 64_000), n, key_base=64_000)
