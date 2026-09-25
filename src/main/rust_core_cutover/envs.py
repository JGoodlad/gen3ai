"""The training-transport envs the SOAK (and, with `--obs-source core`, slice N) drive.

The soak env is the bridge integration test's shape (`utils/bridge/bridge_session_integration_test`):
a real ``Gen3Env`` over ONE persistent Rust ``sim_bridge`` child (``attach_bridge_transport(impl=
"rust")`` — the self-check build under ``POKESIM_EMISSION_SELFCHECK=1``), a ``RandomPlayer``
opponent, pool teams; ``recycle_every`` above the unit's episode count, so a child REPLACEMENT inside
a unit is a crash, never routine recycling.
"""
from __future__ import annotations

from typing import Tuple


def build_soak_env(tag: str, obs_source: str = "python", recycle_every: int = 100_000) -> Tuple[object, object]:
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
    from poke_env.player import RandomPlayer

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from utils.bridge.bridge_session import attach_bridge_transport
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder

    teams = team_list("pool")
    kw = {}
    if obs_source != "python":
        kw["obs_source"] = obs_source
    env = Gen3Env(load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
                  account_configuration1=AccountConfiguration(f"{tag}e"[:18], None),
                  start_listening=False, **kw)
    bkw = {"core_obs": True} if obs_source != "python" else {}
    session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True,
                                      recycle_every=recycle_every, impl="rust", **bkw)
    opponent = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                            account_configuration=AccountConfiguration(f"{tag}o"[:18], None),
                            start_listening=False)
    wrapped = SingleAgentWrapper(env, opponent)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped, session


# ---------------------------------------------------------------------------------------------
# slice N — two training-shaped envs in LOCKSTEP (gen3_core_parity_env_v1)
# ---------------------------------------------------------------------------------------------

def production_args():
    """The resolved training namespace under the PRODUCTION surface: the parser's defaults, the
    ARCH surface (``--arch production``), every ``designs/production_config.json`` key that is a
    parser dest, then the umbrella desugars — what a production launch's env factory reads."""
    from agents.training.baselines import production_config
    from main.train.arch_surface import apply_production_arch
    from main.train.config import desugar_umbrella_flags
    from main.train.parser import build_parser

    a = build_parser().parse_args(["--steps", "1"])
    apply_production_arch(a)
    for k, v in production_config().items():
        if hasattr(a, k):
            setattr(a, k, v)
    desugar_umbrella_flags(a)
    a.use_bridge, a.bridge_impl, a.use_showdown_bridge = "rust", "rust", True
    return a


def _teambuilder_base():
    from poke_env.teambuilder.teambuilder import Teambuilder

    return Teambuilder


class SequenceTeambuilder(_teambuilder_base()):
    """Yields a fixed list of PACKED teams in order (one per ``yield_team``), so two envs built
    from the same list play the same teams on the same episodes."""

    def __init__(self, packed):
        self._teams = list(packed)
        self._i = 0

    def yield_team(self) -> str:
        t = self._teams[self._i % len(self._teams)]
        self._i += 1
        return t


def packed_teams(source: str, tier: str = "full"):
    """Every team of ``source`` as PACKED strings — the pool through the TRAINING teambuilder's own
    packing (``Gen3Teambuilder``: validated, the gen-3 Hidden Power IV fix), refused unless every
    pool team survived it (an index would otherwise name a different team)."""
    from utils import ladder_corpus
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder

    if source == "pool":
        raw = team_list("pool")
        packed = Gen3Teambuilder(raw).packed_teams
        if len(packed) != len(raw):
            raise RuntimeError(f"{len(raw) - len(packed)} pool teams failed validation — team indices would shift")
        return packed
    if source == "ladder":
        return ladder_corpus.teams(tier)
    raise ValueError(source)


def build_lockstep_env(tag: str, obs_source: str, args, trainee_teams, opp_teams, opp_seed: int):
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from agents.training.obs_roundtrip_fuzz_test import SeededRandomPlayer
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from main.train.env_factory import trainee_env_kwargs
    from utils.bridge.bridge_session import attach_bridge_transport

    kw = trainee_env_kwargs(args)
    kw["obs_source"] = obs_source
    env = Gen3Env(load_mappings(), battle_format="gen3ou", team=SequenceTeambuilder(trainee_teams),
                  opponent_team=SequenceTeambuilder(opp_teams),
                  reward_fn=Gen3RewardManager(config=RewardConfig.from_args(args)),
                  account_configuration1=AccountConfiguration(f"{tag}e"[:18], None),
                  start_listening=False, **kw)
    session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True, impl="rust",
                                      core_obs=(obs_source == "core"))
    opp = SeededRandomPlayer(rng_seed=opp_seed, battle_format="gen3ou",
                             account_configuration=AccountConfiguration(f"{tag}o"[:18], None),
                             start_listening=False)
    wrapped = SingleAgentWrapper(env, opp)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped, session


def compare_obs(a: dict, b: dict, census: dict, examples: dict, where, tag: str = "") -> int:
    """Every key of the two obs dicts, TYPE-strict: the same key set, dtype, shape and BYTES."""
    import numpy as np

    bad = 0

    def diverge(k, ex):
        nonlocal bad
        k = k + tag
        census[k] = census.get(k, 0) + 1
        examples.setdefault(k, repr(ex)[:1500])
        bad += 1

    if set(a) != set(b):
        diverge("[keys]", (where, sorted(set(a) ^ set(b))))
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            cells = []
            if x.shape == y.shape:
                diff = np.flatnonzero(x.reshape(-1).view(np.uint8) != y.reshape(-1).view(np.uint8)) if \
                    x.dtype == y.dtype else []
                cells = sorted({int(i) // max(1, x.dtype.itemsize) for i in diff})[:6]
            if k == "observation" and cells:
                from agents.battle.rust_core_parity_obs import block_of

                for c in cells[:3]:
                    diverge(f"observation {block_of(c)}", (where, c, float(x.reshape(-1)[c]), float(y.reshape(-1)[c])))
            else:
                diverge(k, (where, str(x.dtype), str(y.dtype), x.shape, y.shape, cells))
    return bad


def _policy_action(model, obs, mask, rng):
    import numpy as np
    import torch

    spaces = model.observation_space.spaces
    d = {k: (np.asarray(obs[k]) if k in obs else np.zeros(sp.shape, dtype=sp.dtype))[None]
         for k, sp in spaces.items()}
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(d)
        dist = model.policy.get_distribution(t, action_masks=np.asarray(mask, dtype=bool)[None])
        p = dist.distribution.probs[0].cpu().numpy().astype(np.float64)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def run_envn(stream, i, out):
    """Slice N: ``per_unit`` episodes, each played by a PYTHON-obs env and a CORE-obs env in
    LOCKSTEP — the same seed, the same trainee / opponent teams, the same opponent RNG, and the
    trainee's action chosen ONCE (from the Python env's obs and mask) and fed to both. At reset and
    every step: every obs key (bytes), the reward, ``terminated``, ``truncated`` equal."""
    import json
    import time
    import traceback

    import numpy as np

    from main.rust_core_cutover import plan as PL

    p = stream.params
    args = production_args()
    rng_range = stream.unit_range(i)
    if p["source"] == "procedural":
        from utils.team_sources import procedural_teams

        teams = procedural_teams(2 * len(rng_range), 20260924 + 9_000_000 + i)
        pairs = [(2 * m, 2 * m + 1) for m in range(len(rng_range))]
    else:
        teams = packed_teams(p["source"])
        n = len(teams)
        # episode b: trainee team (b * 7919) % n vs ((b * 7919) + 1 + b % 97) % n — spread, both slots
        pairs = [((b * 7919) % n, ((b * 7919) + 1 + b % 97) % n) for b in rng_range]
    trainee = [teams[a] for a, _ in pairs]
    opps = [teams[c] for _, c in pairs]
    model = None
    if p["policy"]:
        import torch

        torch.set_num_threads(1)
        from agents.battle.rust_core_parity import load_production_policy

        model = load_production_policy()
    uid = PL.unit_id(stream.name, i)
    envs = {}
    census, examples, divergent, errors = {}, {}, [], []
    counts = {"episodes": 0, "steps": 0, "compared_keys": 0}
    core_counts = {}
    label_keys = None
    t0 = time.monotonic()
    try:
        for src in ("python", "core"):
            envs[src] = build_lockstep_env(f"N{src[0]}{i}{stream.name[-3:]}", src, args, trainee, opps,
                                           opp_seed=77_000 + i)
        for e, b in enumerate(rng_range):
            key = p["key_base"] + b
            seed = [11 + key, 22 + key, 33 + key, 44 + key]
            for w, sess in envs.values():
                sess.seed = seed
            rng = np.random.default_rng(key)
            if model is not None:
                import torch

                torch.manual_seed(key)
            before = dict(census)
            where = (uid, e, key)
            obs_p, _ = envs["python"][0].reset()
            obs_c, _ = envs["core"][0].reset()
            label_keys = sorted(obs_p)
            compare_obs(obs_p, obs_c, census, examples, (*where, "reset"))
            actions = []
            for step in range(2000):
                mask = np.asarray(obs_p["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                if not envs["python"][0].env.agent1_to_move:
                    # the TRAINING wrapper's shape (MaskableAgentWrapper.step): a step on which the
                    # trainee is not asked to move is driven with action 0 inside the wrapper, never
                    # sampled (and, since gen3_no_phantom_decision_v1, never recorded as a decision)
                    act = 0
                    counts["phantom_steps"] = counts.get("phantom_steps", 0) + 1
                elif model is not None:
                    act = _policy_action(model, obs_p, mask, rng) if legal.size else 0
                else:
                    act = int(rng.choice(legal)) if legal.size else 0
                actions.append(act)
                rp = envs["python"][0].step(act)
                rc = envs["core"][0].step(act)
                counts["steps"] += 1
                compare_obs(rp[0], rc[0], census, examples, (*where, step))
                for j, name in ((1, "reward"), (2, "terminated"), (3, "truncated")):
                    if type(rp[j]) is not type(rc[j]) or rp[j] != rc[j]:
                        census[name] = census.get(name, 0) + 1
                        examples.setdefault(name, repr((*where, step, rp[j], rc[j])))
                obs_p = rp[0]
                if (rp[2] or rp[3]) or (rc[2] or rc[3]):
                    break
            else:
                raise AssertionError(f"episode {e} did not end in 2000 steps")
            counts["episodes"] += 1
            delta = {k: v - before.get(k, 0) for k, v in census.items() if v != before.get(k, 0)}
            if delta:
                path = out / "divergences" / f"{uid}__ep{e}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"stream": stream.name, "unit": i, "episode": e, "key": key,
                                            "seed": seed, "trainee_team": trainee[e], "opp_team": opps[e],
                                            "opp_seed": 77_000 + i, "actions": actions,
                                            "policy": bool(model is not None)}, indent=1))
                divergent.append({"label": f"{stream.name}_{e}", "key": key, "repro": str(path),
                                  "classes": {"N": delta}})
        for src, (w, _s) in envs.items():
            core_counts[src] = dict(w.env.core_obs_counts)
        if core_counts.get("core", {}).get("core", 0) == 0:
            errors.append({"error": "the core env took no core row — the gate would be vacuous"})
        if core_counts.get("python", {}).get("core", 0) != 0:
            errors.append({"error": "the python env took a core row"})
    except Exception as ex:
        errors.append({"error": f"{type(ex).__name__}: {ex}"[:2000], "traceback": traceback.format_exc()[-6000:]})
    finally:
        for w, _s in envs.values():
            try:
                w.close()
            except Exception:
                pass
    return {"battles": counts["episodes"], "steps": counts["steps"], "label_keys": label_keys,
            "phantom_steps": counts.get("phantom_steps", 0),
            "core_obs_counts": core_counts, "totals": {"N": {"divergences": census, "examples": examples,
                                                           "episodes": counts["episodes"],
                                                           "decisions": core_counts.get("core", {}).get("core", 0)}},
            "divergent": divergent, "known": [], "errors": errors, "wall_envn_s": round(time.monotonic() - t0, 1)}
