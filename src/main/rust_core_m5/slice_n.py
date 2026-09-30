"""SLICE N at the ENV LEVEL (M5's gate, program §2 M5 "Gate"; M5 Lane J): N envs on T worker threads
in the Rust env core, through BOTH front ends, against the production-surface ``Gen3Env`` — the
observation row, the action mask, every built label key, the reward and ``terminated`` /
``truncated`` equal at every trainee decision and every episode end. No allowlist.

It is Lane C's and Lane D's record-in-Rust / replay-in-Python machinery JOINED and run at N (their
gates run N = 1, T = 1 and each compares its own columns):

1. RECORD. One core of N envs (the spec as production declares it: every label family, the
   production terminal, ``turn_limit`` = ``StallConfig().threshold``) plays ``n_episodes`` under a
   seeded random policy on BOTH sides. Episode ``e`` is teams ``(2e, 2e + 1)`` and Showdown seed
   ``_seed(key_base + e)``; each env's NEXT episode is staged in ``ep_team`` / ``ep_seed`` and
   re-staged whenever that env's ``episode`` column moves (F-LE-4), so the AUTO-RESET path is the
   one exercised — every env plays several episodes back to back. Recorded: each trainee
   decision's ``dec_n``, row, mask and label columns; both sides' action INDICES; each episode's
   ``(reward, terminated, truncated)`` from its ``done`` op. Every other op's outcome must be
   ``(0, 0, 0)``; no episode may be quarantined; every ``*_AFTER_FREEZE`` counter stays 0.
2. The same corpus is recorded through the FFI front end at ``threads`` and through the PROCESS
   front end at ``threads_b`` (a different thread count): the two recordings must be IDENTICAL —
   the two front ends and thread-count invariance, at N.
3. REPLAY. One ``Gen3Env`` built as training builds the trainee's (``trainee_env_kwargs(
   production_args())``, the rust bridge, ``--obs-source core``, the same ``StallConfig`` and
   ``RewardConfig``) replays every episode: the same teams and seed, p1's recorded indices, and an
   opponent that plays p2's recorded indices through the REAL mapper only when agent2 really moves.
   At every trainee decision: ``dec_n``, the row bytes (the alignment check — nothing else is
   compared on a misaligned decision), ``obs["action_mask"]`` AND ``env.action_masks()`` (the two
   masks the learner reads: the async and the stock rollout) against the core's mask, and every
   label key (dtype, shape, bytes). Every non-final reward must be 0 and the episode's final
   ``(reward, terminated, truncated)`` equal the core's; the decision counts must match exactly.

What it does not cover (declared): the host-filled label keys (``win_target`` / ``win_mask`` /
``opp_class`` — Lane G), an IN-CORE opponent (a bot's or a policy route's p2 decisions are never
exposed, so they cannot be replayed from the columns — Lane F's and Lane E's gates), and the ROLLOUT
level (the learner's buffer — Lane G's slice N).

    python -m main.rust_core_m5 slice-n --tier commit|milestone [--out <json>]
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

NAMES = ("mjone", "mjtwo")
FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")


def families() -> Tuple[str, ...]:
    """Every label family the core builds (all of ``columns.LABEL_FAMILIES``: each is a core family
    of the production surface — Lane C's inventory pins that)."""
    from utils.rust_env import columns as C

    return tuple(C.LABEL_FAMILIES)


def label_keys(fams: Sequence[str]) -> List[str]:
    from utils.rust_env import columns as C

    return [k for f in fams for k in C.LABEL_FAMILIES[f]]


def _seed(key: int) -> List[int]:
    return [(11 + key) % 65536, (22 + key) % 65536, (33 + key) % 65536, (44 + key) % 65536]


@dataclass
class Episode:
    p1: List[int] = field(default_factory=list)
    p2: List[int] = field(default_factory=list)
    rows: List[tuple] = field(default_factory=list)   # (dec_n, obs, mask, {key: array})
    end: Optional[Tuple[float, int, int]] = None
    env: int = -1


# ------------------------------------------------------------------------------------ 1. record

def record(core_factory: Callable[[str], object], teams: Sequence[str], n_episodes: int, *, n_envs: int,
           threads: int, key_base: int, turn_limit: int, fams: Sequence[str]) -> Tuple[Dict[int, Episode], dict]:
    """Play ``n_episodes`` in ONE core of ``n_envs`` envs on ``threads`` workers; return the per-episode
    recording and the op census."""
    from utils.rust_env import protocol as P

    assert len(teams) >= 2 * n_episodes
    spec = P.spec_json(n=n_envs, threads=threads, teams=list(teams[:2 * n_episodes]), names=NAMES,
                       decision_tense=False, switch_freeze=False, turn_limit=turn_limit, refusal_budget=0,
                       bank_dir=None, labels=tuple(fams))
    keys = label_keys(fams)
    rng = np.random.default_rng(key_base)
    out = {e: Episode() for e in range(n_episodes)}
    census = {"ops": 0, "decisions_p1": 0, "decisions_p2": 0, "auto_resets": 0, "filler_episodes": 0}
    nxt = [0]
    filler = [0]

    with core_factory(spec) as core:
        c = core.cols

        def stage(i: int) -> int:
            """Stage env ``i``'s next episode; return its id (-1 = a filler past the corpus, never
            recorded — the env keeps playing so the others can finish)."""
            if nxt[0] < n_episodes:
                e = nxt[0]
                nxt[0] += 1
                c["ep_team"][i] = [2 * e, 2 * e + 1]
                c["ep_seed"][i] = _seed(key_base + e)
                return e
            filler[0] += 1
            c["ep_team"][i] = [0, 1]
            c["ep_seed"][i] = _seed(key_base + 1_000_003 + filler[0])
            return -1

        staged = [stage(i) for i in range(n_envs)]
        core.reset()
        cur = list(staged)
        for e in cur:
            if e >= 0:
                out[e].env = cur.index(e)
        staged = [stage(i) for i in range(n_envs)]
        last_ep = c["episode"].copy()
        ended = 0
        for _op in range(2_000_000):
            if ended == n_episodes:
                break
            need = c["need"].copy()
            for i in range(n_envs):
                e = cur[i]
                if need[i, 0] and e >= 0:
                    out[e].rows.append((int(c["dec_n"][i, 0]), c["obs"][i, 0].copy(), c["mask"][i, 0].copy(),
                                        {k: c[k][i, 0].copy() for k in keys}))
                for s in range(2):
                    c["action"][i, s] = -1
                    if need[i, s]:
                        a = int(rng.choice(np.flatnonzero(c["mask"][i, s])))
                        c["action"][i, s] = a
                        if e >= 0:
                            (out[e].p1 if s == 0 else out[e].p2).append(a)
                            census["decisions_p1" if s == 0 else "decisions_p2"] += 1
            core.step()
            census["ops"] += 1
            done = c["done"].copy()
            refused = c["refused"].copy()
            moved = c["episode"] != last_ep
            for i in range(n_envs):
                e = cur[i]
                if refused[i]:
                    raise AssertionError(f"env {i} episode {e} was quarantined: {core.bank()[-1:]}")
                end = (float(c["reward"][i]), int(c["terminated"][i]), int(c["truncated"][i]))
                if done[i]:
                    if e >= 0:
                        out[e].end = end
                        ended += 1
                    else:
                        census["filler_episodes"] += 1
                elif end != (0.0, 0, 0):
                    raise AssertionError(f"env {i} episode {e}: an outcome {end} on an op that ended nothing")
                if moved[i]:
                    if not done[i]:
                        raise AssertionError(f"env {i}: its episode moved without a done (a restart?)")
                    census["auto_resets"] += 1
                    cur[i] = staged[i]
                    if cur[i] >= 0:
                        out[cur[i]].env = i
                    staged[i] = stage(i)
                elif done[i]:
                    raise AssertionError(f"env {i}: done without an auto-reset (a parked env?)")
            last_ep = c["episode"].copy()
        else:
            raise AssertionError(f"{n_episodes - ended} episodes did not end")
        after = core.after_freeze()
        if any(after.values()):
            raise AssertionError(f"a *_AFTER_FREEZE counter moved: {after}")
    return out, census


def recordings_equal(a: Dict[int, Episode], b: Dict[int, Episode]) -> List[str]:
    """Every difference between two recordings of the same corpus (empty = identical)."""
    diffs = []
    for e in sorted(set(a) | set(b)):
        x, y = a.get(e), b.get(e)
        if x is None or y is None:
            diffs.append(f"episode {e}: recorded by one front end only")
            continue
        for k in ("p1", "p2", "end"):
            if getattr(x, k) != getattr(y, k):
                diffs.append(f"episode {e}: {k} differs")
        if len(x.rows) != len(y.rows):
            diffs.append(f"episode {e}: {len(x.rows)} vs {len(y.rows)} rows")
            continue
        for j, (rx, ry) in enumerate(zip(x.rows, y.rows)):
            if rx[0] != ry[0] or rx[1].tobytes() != ry[1].tobytes() or rx[2].tobytes() != ry[2].tobytes() or any(
                    rx[3][k].tobytes() != ry[3][k].tobytes() for k in rx[3]):
                diffs.append(f"episode {e} decision {j}: a column differs")
                break
    return diffs


# ------------------------------------------------------------------------------------ 2. replay

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
            self.actions: List[int] = []
            self.j = 0
            self.env = None

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
                raise AssertionError(f"p2 decision {self.j - 1}: recorded index {a} is illegal in Python ({mask})")
            return Gen3ActionMapper.action_to_order(a, battle, legal=legal, mask=mask)

    return IndexScriptPlayer


def replay(recs: Dict[int, Episode], teams: Sequence[str], *, key_base: int, turn_limit: int,
           fams: Sequence[str], tag: str = "MJ") -> Tuple[Dict[str, int], Dict[str, str], dict]:
    """Replay every recorded episode through a production-surface ``Gen3Env``; return
    ``(divergences {kind: count}, examples {kind: repr}, counts)``."""
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
    kw["stall_config"] = StallConfig(threshold=turn_limit)
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
    keys = label_keys(fams)
    missing = set(keys) - set(env.observation_space.spaces)
    if missing:
        raise AssertionError(f"the production surface does not emit {sorted(missing)} — the slice would compare nothing")
    div: Dict[str, int] = {}
    ex: Dict[str, str] = {}
    counts = {"episodes": 0, "decisions": 0, "label_compares": 0, "mask_compares": 0, "terminated": 0,
              "truncated": 0, "wins": 0, "ties": 0, "forfeits": 0, "envs": len({r.env for r in recs.values()})}

    def diverge(k: str, where, detail) -> None:
        div[k] = div.get(k, 0) + 1
        ex.setdefault(k, repr((where, detail))[:1500])

    try:
        for e in range(n):
            rec = recs[e]
            session.seed = _seed(key_base + e)
            opp.actions, opp.j = list(rec.p2), 0
            obs, _ = w.reset()
            k, final, aligned = 0, None, True
            for _step in range(5000):
                if env.agent1_to_move:
                    where = (e, k, int(getattr(env.battle1, "turn", -1)))
                    if k >= len(rec.rows):
                        diverge("[decisions]", where, f"Python decision {k} past the core's {len(rec.rows)}")
                        aligned = False
                        break
                    dec_n, row, mask, labels = rec.rows[k]
                    if dec_n != k:
                        diverge("[dec_n]", where, dec_n)
                    if np.asarray(obs["observation"], dtype=np.float32).tobytes() != row.tobytes():
                        diverge("observation", where, "row differs — alignment lost, nothing else compared")
                        aligned = False
                        break
                    want = mask.astype(bool)
                    for name, got in (("action_mask", obs["action_mask"]), ("action_masks()", env.action_masks())):
                        counts["mask_compares"] += 1
                        if not np.array_equal(np.asarray(got).astype(bool), want):
                            diverge(name, where, {"python": np.asarray(got).tolist(), "core": mask.tolist()})
                    for key in keys:
                        x, y = np.asarray(obs[key]), labels[key]
                        counts["label_compares"] += 1
                        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                            diverge(key, where, {"python": x.tolist(), "core": y.tolist()})
                    act = rec.p1[k]
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
            if not aligned:
                continue
            if final != rec.end:
                diverge("end", (e, int(env.battle1.turn), env.battle1.won), {"python": final, "core": rec.end})
            if k != len(rec.rows):
                diverge("[decisions]", (e, k), f"Python took {k} p1 decisions, the core {len(rec.rows)}")
            if opp.j != len(rec.p2):
                diverge("[p2 decisions]", (e, opp.j), f"Python p2 took {opp.j}, the core {len(rec.p2)}")
            b1 = env.battle1
            counts["episodes"] += 1
            counts["terminated"] += int(final[1]) if final else 0
            counts["truncated"] += int(final[2]) if final else 0
            counts["wins"] += int(b1.won is True)
            counts["ties"] += int(b1.won is None)
            counts["forfeits"] += int(b1.turn >= turn_limit and b1.won is False)
    finally:
        w.close()
    return div, ex, counts


# -------------------------------------------------------------------------------------- the slice

@dataclass(frozen=True)
class Tier:
    name: str
    parts: Tuple[Tuple[str, int, Optional[int]], ...]   # (team source, episodes, stall threshold; None = production)
    n_envs: int
    threads: int
    threads_b: int
    key_base: int


#: COMMIT (routine) and MILESTONE (``slow``; M5's gate: N = 48, the production thread count 8).
TIERS = {
    "commit": Tier("commit", (("pool", 8, None), ("pool", 8, 6)), n_envs=4, threads=3, threads_b=2, key_base=66_000),
    "milestone": Tier("milestone", (("pool", 240, None), ("ladder", 240, None), ("procedural", 96, None), ("ladder", 48, 12)),
                      n_envs=48, threads=8, threads_b=5, key_base=67_000),
}


def teams_for(source: str, n_episodes: int, key_base: int) -> List[str]:
    """Two DISTINCT teams per episode from ``source`` (``pool`` / ``ladder`` / ``procedural``)."""
    if source == "procedural":
        from utils.team_sources import procedural_teams

        return list(procedural_teams(2 * n_episodes, 20260930 + key_base))
    from main.rust_core_cutover.envs import packed_teams

    pool = packed_teams(source)
    step = 7919
    teams = [pool[(key_base + i * step) % len(pool)] for i in range(2 * n_episodes)]
    for e in range(n_episodes):
        if teams[2 * e] == teams[2 * e + 1]:
            teams[2 * e + 1] = pool[(key_base + (2 * e + 1) * step + 1) % len(pool)]
    return teams


def build(profile: str = "selfcheck"):
    """Build THIS checkout's cdylib + process child (its own target, never main's); return
    ``(lib, child binary path)`` — the stamp is checked at load."""
    import os
    import shutil
    import subprocess

    from utils.paths import src_path
    from utils.rust_env import ffi, proc

    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    crate = src_path("rust_env")
    feats = FEATURES if profile == "selfcheck" else ("--release",)
    r = subprocess.run([cargo, "build", "--lib", "--bin", proc.BIN_NAME, *feats, "--manifest-path",
                        str(crate / "Cargo.toml")], env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")),
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"building the cdylib + child failed:\n{r.stderr[-4000:]}")
    nan = profile == "selfcheck"
    return ffi.load(ffi.default_path(profile), nan_poison=nan), proc.default_path(profile)


def run_part(built, source: str, n_episodes: int, *, n_envs: int, threads: int, threads_b: int, key_base: int,
             turn_limit: Optional[int] = None, fams: Optional[Sequence[str]] = None,
             mutate: Optional[Callable[[Dict[int, Episode]], None]] = None) -> dict:
    """One source's slice: record through both front ends, compare, replay. ``mutate`` (teeth only)
    edits the recording before the replay."""
    from utils.rust_env import episode as EP
    from utils.rust_env import ffi, proc

    lib, binary = built
    turn_limit = EP.stall_threshold() if turn_limit is None else turn_limit
    fams = families() if fams is None else tuple(fams)
    teams = teams_for(source, n_episodes, key_base)
    t0 = time.time()
    kw = dict(n_envs=n_envs, key_base=key_base, turn_limit=turn_limit, fams=fams)
    a, census = record(lambda s: ffi.FfiCore(s, lib=lib), teams, n_episodes, threads=threads, **kw)
    b, _ = record(lambda s: proc.ProcCore(s, binary=binary, nan_poison=True), teams, n_episodes, threads=threads_b, **kw)
    front = recordings_equal(a, b)
    t_rec = time.time() - t0
    if mutate is not None:
        mutate(a)
    div, ex, counts = replay(a, teams, key_base=key_base, turn_limit=turn_limit, fams=fams,
                             tag=f"MJ{source[:2]}{key_base % 1000}")
    return {"source": source, "n_episodes": n_episodes, "n_envs": n_envs, "threads": [threads, threads_b],
            "turn_limit": turn_limit, "families": list(fams), "record_s": round(t_rec, 1),
            "replay_s": round(time.time() - t0 - t_rec, 1), "census": census, "counts": counts,
            "front_end_diffs": front[:20], "divergences": div, "examples": ex,
            "ok": not front and not div and counts["episodes"] == n_episodes}


def run_tier(tier: str, built=None) -> dict:
    t = TIERS[tier]
    built = built or build("selfcheck")
    parts = [run_part(built, src, n, n_envs=t.n_envs, threads=t.threads, threads_b=t.threads_b, turn_limit=tl,
                      key_base=t.key_base + 1000 * j) for j, (src, n, tl) in enumerate(t.parts)]
    return {"component": "slice_n", "tier": tier, "parts": parts, "ok": all(p["ok"] for p in parts)}


def render(res: dict) -> str:
    lines = [f"SLICE N (env level), tier {res['tier']}: {'PASS' if res['ok'] else 'FAIL'}"]
    for p in res["parts"]:
        c = p["counts"]
        lines.append(f"  {p['source']:<10} stall {p['turn_limit']:<3} N={p['n_envs']} T={p['threads']}: {c['episodes']}/{p['n_episodes']} episodes, "
                     f"{c['decisions']} decisions, {c['label_compares']} label + {c['mask_compares']} mask compares, "
                     f"auto-resets {p['census']['auto_resets']}, ends term/trunc/ties/forfeits "
                     f"{c['terminated']}/{c['truncated']}/{c['ties']}/{c['forfeits']}; "
                     f"front ends {'identical' if not p['front_end_diffs'] else p['front_end_diffs'][:3]}; "
                     f"divergences {p['divergences'] or 0} (record {p['record_s']} s, replay {p['replay_s']} s)")
    return "\n".join(lines)
