"""SLICE N for the TRAINING LABELS (M5 Lane C): the Rust env core's label columns equal the keys
``Gen3Env`` emits under the PRODUCTION surface, per decision, byte for byte — no allowlist.

RECORD in Rust, REPLAY in Python (the direction Lane 0's gate ① runs):

1. the env core (through the FFI front end, ``labels`` = every BUILT family) plays episodes between
   pool / ladder teams under a seeded random policy on BOTH sides; at every p1 decision it records
   the row, the mask and every label column, and it records each side's action INDICES;
2. one ``Gen3Env`` built exactly as training builds the trainee's (``trainee_env_kwargs(
   production_args())``, the rust bridge, ``--obs-source core``) replays each episode: the same
   packed teams, the same Showdown seed, the trainee's recorded indices, and an opponent that
   plays p2's recorded indices through the REAL mapper on ``battle2`` (only on steps where agent2
   actually moves — the wrapper also asks it on steps whose action is never sent);
3. at every trainee decision: the ``observation`` row must equal the core's (the alignment check —
   a label compared on a different decision would be noise), and every label key of a built
   family must equal its column (dtype, shape, bytes).

COMMIT tier (routine): a handful of pool episodes. MILESTONE (``slow``; verdict in
``designs/ops/slow_tier_status.json``): pool and ladder episodes at scale.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from utils.paths import src_path
from utils.rust_env import columns as C
from utils.rust_env import label_inventory as LI

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")
#: The families the core builds today (mirrors `labels::BUILT`; a family missing here is simply
#: not compared yet — the Rust unit test pins BUILT, and the spec refuses an unbuilt one).
BUILT = ("belief", "hp_type", "item", "spread", "intent", "margin")
NAMES = ("lcpone", "lcptwo")


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the label parity gate cannot run (install rustup)")
    return cargo


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env import ffi

    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run([_cargo(), "build", "--lib", *FEATURES, "--manifest-path", str(crate / "Cargo.toml")],
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib failed:\n{r.stderr[-4000:]}"
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _label_keys(families=BUILT):
    return [k for f in families for k in C.LABEL_FAMILIES[f]]


def _seed(key: int):
    return [(11 + key) % 65536, (22 + key) % 65536, (33 + key) % 65536, (44 + key) % 65536]


# --------------------------------------------------------------------------------- 1. record

def record(lib, teams, n_episodes: int, *, key_base: int, families=BUILT, turn_limit=None):
    """Play ``n_episodes`` in the core (episode e: team ``2e`` vs ``2e + 1``, seed ``_seed(key_base
    + e)``); return per episode ``{"p1": [idx…], "p2": [idx…], "rows": [(dec_n, obs, mask,
    {key: array})…]}``."""
    from utils.rust_env import episode as EP
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P

    if turn_limit is None:
        turn_limit = EP.stall_threshold()   # the production stall forfeit (Lane D), as Gen3Env's
    assert len(teams) >= 2 * n_episodes
    spec = P.spec_json(n=1, threads=1, teams=list(teams[:2 * n_episodes]), names=NAMES, decision_tense=False,
                       switch_freeze=False, turn_limit=turn_limit, refusal_budget=0, bank_dir=None,
                       labels=tuple(families))
    keys = _label_keys(families)
    rng = np.random.default_rng(key_base)
    out = {e: {"p1": [], "p2": [], "rows": []} for e in range(n_episodes)}
    with ffi.FfiCore(spec, lib=lib) as core:
        c = core.cols

        def stage(e):
            e = min(e, n_episodes - 1)          # past the last one: any valid team (never recorded)
            c["ep_team"][0] = [2 * e, 2 * e + 1]
            c["ep_seed"][0] = _seed(key_base + e)

        stage(0)
        core.reset()
        stage(1)
        done = 0
        while done < n_episodes:
            ep = int(c["episode"][0])
            need = c["need"][0].copy()
            if need[0]:
                out[ep]["rows"].append((int(c["dec_n"][0, 0]), c["obs"][0, 0].copy(), c["mask"][0, 0].copy(),
                                        {k: c[k][0, 0].copy() for k in keys}))
            for s in range(2):
                c["action"][0, s] = -1
                if need[s]:
                    legal = np.flatnonzero(c["mask"][0, s])
                    a = int(rng.choice(legal))
                    c["action"][0, s] = a
                    out[ep]["p1" if s == 0 else "p2"].append(a)
            core.step()
            if c["done"][0]:
                assert not c["refused"][0], f"episode {ep} was quarantined: {core.bank()[-1:]}"
                done += 1
                stage(done + 1)
        assert all(v == 0 for v in core.after_freeze().values()), core.after_freeze()
    return out


# --------------------------------------------------------------------------------- 2. replay

def _index_script_player():
    from poke_env.player.battle_order import DefaultBattleOrder
    from poke_env.player.player import Player

    from agents.action.mapper import Gen3ActionMapper
    from agents.action.mask_generator import Gen3ActionMasker
    from agents.battle.live_view import LegalActions

    class IndexScriptPlayer(Player):
        """Plays a recorded list of action INDICES through the real mapper — consumed only on a step
        whose opponent action the env actually sends (``agent2_to_move``)."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.actions, self.j, self.env = [], 0, None

        def choose_move(self, battle):
            if self.env is None or not self.env.agent2_to_move:
                return DefaultBattleOrder()
            legal = LegalActions.from_battle(battle)
            mask = Gen3ActionMasker.get_mask(battle, legal=legal)
            if self.j >= len(self.actions):
                raise AssertionError(f"p2 asked for decision {self.j}; the core recorded {len(self.actions)}")
            a = self.actions[self.j]
            self.j += 1
            if not mask[a]:
                raise AssertionError(f"p2 decision {self.j - 1}: recorded index {a} is illegal in Python (mask {mask})")
            return Gen3ActionMapper.action_to_order(a, battle, legal=legal, mask=mask)

    return IndexScriptPlayer


def replay(recs, teams, *, key_base: int, families=BUILT, tag: str = "LC"):
    """Replay every recorded episode through a production-surface ``Gen3Env``; return
    ``(divergences {key: count}, examples {key: repr}, counts)``."""
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from agents.training.stall import StallConfig
    from main.rust_core_cutover.envs import SequenceTeambuilder, production_args
    from main.train.env_factory import trainee_env_kwargs
    from utils.bridge.bridge_session import attach_bridge_transport

    args = production_args()
    kw = trainee_env_kwargs(args)
    kw["obs_source"] = "core"
    n = len(recs)
    env = Gen3Env(load_mappings(), battle_format="gen3ou",
                  team=SequenceTeambuilder([teams[2 * e] for e in range(n)]),
                  opponent_team=SequenceTeambuilder([teams[2 * e + 1] for e in range(n)]),
                  reward_fn=Gen3RewardManager(config=RewardConfig.from_args(args)),
                  account_configuration1=AccountConfiguration(f"{tag}e"[:18], None),
                  start_listening=False, **kw)
    session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True, impl="rust", core_obs=True)
    opp = _index_script_player()(battle_format="gen3ou", account_configuration=AccountConfiguration(f"{tag}o"[:18], None),
                                 start_listening=False)
    opp.env = env
    w = SingleAgentWrapper(env, opp)
    w.action_space, w.observation_space = env.action_space, env.observation_space
    keys = _label_keys(families)
    missing = set(keys) - set(env.observation_space.spaces)
    assert not missing, f"the production surface does not emit {missing} — the gate would compare nothing"
    threshold = StallConfig().threshold
    div, ex = {}, {}
    counts = {"episodes": 0, "decisions": 0, "compared": 0, "forfeited": 0}

    def diverge(k, where, detail):
        div[k] = div.get(k, 0) + 1
        ex.setdefault(k, repr((where, detail))[:1500])

    try:
        for e in range(n):
            rec = recs[e]
            session.seed = _seed(key_base + e)
            opp.actions, opp.j = list(rec["p2"]), 0
            obs, _ = w.reset()
            k = 0
            for _step in range(5000):
                a1 = env.agent1_to_move
                if a1:
                    where = (e, k, int(getattr(env.battle1, "turn", -1)))
                    if k >= len(rec["rows"]):
                        diverge("[decisions]", where, f"Python decision {k} past the core's {len(rec['rows'])}")
                        break
                    dec_n, row, mask, labels = rec["rows"][k]
                    if dec_n != k:
                        diverge("[dec_n]", where, dec_n)
                    if np.asarray(obs["observation"], dtype=np.float32).tobytes() != row.tobytes():
                        diverge("observation", where, "row differs — alignment lost, labels not compared")
                        break
                    for key in keys:
                        x, y = np.asarray(obs[key]), labels[key]
                        counts["compared"] += 1
                        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                            diverge(key, where, {"python": x.tolist(), "core": y.tolist()})
                    act = rec["p1"][k]
                    k += 1
                    counts["decisions"] += 1
                else:
                    act = 0
                obs, _r, term, trunc, _i = w.step(act)
                if term or trunc:
                    break
            else:
                raise AssertionError(f"episode {e} did not end in 5000 steps")
            counts["forfeited"] += int(getattr(env.battle1, "turn", 0)) >= threshold
            # the core forfeits at the same stall threshold (Lane D), so the counts match EXACTLY
            if k != len(rec["rows"]) and "observation" not in div:
                diverge("[decisions]", (e, k), f"Python took {k} decisions, the core {len(rec['rows'])}")
            if opp.j != len(rec["p2"]) and "observation" not in div:
                diverge("[p2 decisions]", (e, opp.j), f"Python p2 took {opp.j}, the core {len(rec['p2'])}")
            counts["episodes"] += 1
    finally:
        w.close()
    return div, ex, counts


def run_slice(lib, source: str, n_episodes: int, key_base: int, families=BUILT):
    from main.rust_core_cutover.envs import packed_teams

    if source == "procedural":
        from utils.team_sources import procedural_teams

        teams = list(procedural_teams(2 * n_episodes, 20260929 + key_base))
    else:
        pool = packed_teams(source)
        step = 7919
        teams = [pool[(key_base + i * step) % len(pool)] for i in range(2 * n_episodes)]
        for e in range(n_episodes):            # two distinct teams per battle
            if teams[2 * e] == teams[2 * e + 1]:
                teams[2 * e + 1] = pool[(key_base + (2 * e + 1) * step + 1) % len(pool)]
    recs = record(lib, teams, n_episodes, key_base=key_base, families=families)
    div, ex, counts = replay(recs, teams, key_base=key_base, families=families, tag=f"LC{source[:2]}{key_base % 1000}")
    print({"source": source, **counts, "divergences": div})
    return div, ex, counts


def _assert_clean(div, ex, counts, n_episodes):
    assert not div, (div, ex)
    assert counts["episodes"] == n_episodes
    assert counts["decisions"] >= 10 * n_episodes, counts
    assert counts["compared"] == counts["decisions"] * len(_label_keys()), counts


def test_every_built_family_is_a_core_family_of_the_production_surface():
    for f in BUILT:
        rows = LI.families("core")[f]
        assert all(r.production for r in rows), f


def test_commit_tier_the_core_labels_equal_gen3env(lib):
    div, ex, counts = run_slice(lib, "pool", 8, key_base=61_000)
    _assert_clean(div, ex, counts, 8)


@pytest.mark.parametrize("method,key", [("_belief_labels", "belief_moves"), ("_hp_type_labels", "hp_type_label"),
                                        ("_item_labels", "item_label"),
                                        ("_spread_labels", "belief_ev"), ("_opp_intent_labels", "opp_action_num")])
def test_the_label_slice_has_teeth(lib, monkeypatch, method, key):
    """A Python label that differs in ONE cell from the core's must fail the slice — per family."""
    from agents.training import gen3_env

    real = getattr(gen3_env.Gen3Env, method)

    def perturbed(self, *a):
        out = real(self, *a)
        out[key] = out[key].copy()
        out[key].reshape(-1)[-1] += 1
        return out

    monkeypatch.setattr(gen3_env.Gen3Env, method, perturbed)
    div, _ex, counts = run_slice(lib, "pool", 1, key_base=61_500)
    assert div.get(key) == counts["decisions"] > 0, (div, counts)


def test_the_margin_slice_has_teeth(lib, monkeypatch):
    """`win_margin` comes from the reward manager, not a `Gen3Env` label method: move its source."""
    from agents.training import reward_manager

    real = reward_manager._material_margin
    monkeypatch.setattr(reward_manager, "_material_margin", lambda live: real(live) + 0.5)
    div, _ex, counts = run_slice(lib, "pool", 1, key_base=61_500)
    # every decision but the RESET one (0.0 there on both paths, the reward manager's reset value)
    assert div.get("win_margin") == counts["decisions"] - 1 > 0, (div, counts)


@pytest.mark.slow
@pytest.mark.parametrize("source,n", [("pool", 200), ("ladder", 200), ("procedural", 100)],
                         ids=["pool", "ladder", "procedural"])
def test_milestone_the_core_labels_equal_gen3env(lib, source, n):
    div, ex, counts = run_slice(lib, source, n, key_base=62_000 + 5_000 * ["pool", "ladder", "procedural"].index(source))
    _assert_clean(div, ex, counts, n)
