"""THE BANKED BOT DECISION CORPUS — the Python bots' own decisions, recorded as RE-PLAYABLE battle
inputs (M5 Lane F, `designs/endstate/program_rust_core.md` §2 M5).

RECORD in Python, REPLAY in Rust (the direction a bot gate has to run — only Python can say what the
Python bot chooses):

1. A ``Gen3Env`` built exactly as training builds the trainee's (``trainee_env_kwargs(
   production_args())``, the rust bridge, ``--obs-source core``) plays episodes between pool /
   ladder / procedural teams; p1 is a seeded random policy over the mask; p2 is the REAL bot class,
   driven by ``SingleAgentWrapper`` exactly as training drives it (``choose_move(env.battle2)``).
2. Every bot RNG stream (``bot_inventory`` rows: ``choice`` / ``protect`` / ``bait``) is a
   :class:`CountingRandom` seeded from the corpus key, so each decision records the stream OFFSET (in
   MT19937 32-bit words) before and after it — the Rust bot draws the same stream from the same
   offset (`bots::rng::PyRandom`).
3. At every REAL p2 decision (``env.agent2_to_move``) the record holds: the bot view's hash
   (`bot_view.fnv64(view_json(battle2))`), the order the bot returned, the order the env actually
   SENT (after poke-env's ``SinglesEnv`` action round trip) as a choice token, and that order's
   11-dim action index (`agents.action.serialize.order_to_action`) — the index the Rust env core
   takes. A call on a step whose p2 action is never sent (a PHANTOM poll) is counted, with the draws
   it consumed; it is not a decision.

An episode is ``{"seed", "names", "teams", "p1": [idx | -1 forfeit], "p2": [decision…], "end",
"phantom"}`` — the env core's input log in index form (`search::game::Log` + `Game::feed`).
"""
from __future__ import annotations

import gzip
import json
import random
from typing import Optional

import numpy as np

from utils.rust_env import bot_view as BV
from utils.rust_env import bot_inventory as BI

SCHEMA = "gen3_bot_corpus_v1"
NAMES = ("bfpone", "bfptwo")


class CountingRandom(random.Random):
    """``random.Random`` that counts the MT19937 32-bit WORDS it consumes. Overriding BOTH
    ``random`` and ``getrandbits`` keeps ``_randbelow`` on its getrandbits path, i.e. the stream is
    exactly a plain ``Random``'s."""

    def __init__(self, seed):
        self.words = 0
        super().__init__(seed)

    def random(self):
        self.words += 2
        return super().random()

    def getrandbits(self, k):
        if k > 0:
            self.words += (k - 1) // 32 + 1
        return super().getrandbits(k)


def stream_seed(corpus_key: int, bot: str, stream: str) -> int:
    """The seed of one bot stream — distinct per stream, so the two coins of a staller are independent."""
    return (corpus_key * 1_000_003 + sum(map(ord, bot)) * 7919 + {"choice": 1, "protect": 2, "bait": 3}[stream]) % (2 ** 62)


def make_bot(name: str, corpus_key: int, tag: str):
    """The bot the pools construct for display name ``name``, with every stream a seeded
    :class:`CountingRandom`. Returns ``(player, {stream: rng}, {stream: seed})``."""
    from poke_env import AccountConfiguration

    row = BI.by_name()[name]
    mod, _, cls_name = row.cls.rpartition(".")
    cls = getattr(__import__(mod, fromlist=[cls_name]), cls_name)
    if name == "baitbot":
        from agents.baitbot import make_baitbot_class
        from main.train.parser import build_parser

        cls = make_baitbot_class(build_parser().parse_args(["--steps", "1"]).bait_bot_p)   # the roster's class
    p = cls(battle_format="gen3ou", account_configuration=AccountConfiguration(f"{tag}o"[:18], None),
            start_listening=False)
    seeds = {s: stream_seed(corpus_key, name, s) for s in row.rng}
    rngs = {s: CountingRandom(seeds[s]) for s in row.rng}
    attr = {"choice": "_choice_rng", "protect": "_protect_rng", "bait": "_rng"}
    for s, r in rngs.items():
        assert hasattr(p, attr[s]), f"{name}: no {attr[s]} — the inventory's rng column is stale"
        setattr(p, attr[s], r)
    return p, rngs, seeds


def _seed(key: int):
    return [(11 + key) % 65536, (22 + key) % 65536, (33 + key) % 65536, (44 + key) % 65536]


def team_list(source: str, n: int, key_base: int):
    """``2n`` packed teams (two distinct per episode) — the same selection rule as the label slice."""
    from main.rust_core_cutover.envs import packed_teams

    if source == "procedural":
        from utils.team_sources import procedural_teams

        return list(procedural_teams(2 * n, 20260929 + key_base))
    pool = packed_teams(source)
    step = 7919
    teams = [pool[(key_base + i * step) % len(pool)] for i in range(2 * n)]
    for e in range(n):
        if teams[2 * e] == teams[2 * e + 1]:
            teams[2 * e + 1] = pool[(key_base + (2 * e + 1) * step + 1) % len(pool)]
    return teams


def record(bot: str, teams, *, key_base: int, keep_views: bool = False, tag: Optional[str] = None):
    """Play ``len(teams) // 2`` episodes of p1 (seeded random) vs ``bot`` (p2); return the episodes."""
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
    from poke_env.player.battle_order import ForfeitBattleOrder

    from agents.action.serialize import order_to_action
    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from main.rust_core_cutover.envs import SequenceTeambuilder, production_args
    from main.train.env_factory import trainee_env_kwargs
    from utils.bridge.bridge_session import attach_bridge_transport

    n = len(teams) // 2
    tag = tag or f"BF{bot[:6]}{key_base % 1000}"
    args = production_args()
    kw = trainee_env_kwargs(args)
    kw["obs_source"] = "core"
    env = Gen3Env(load_mappings(), battle_format="gen3ou",
                  team=SequenceTeambuilder([teams[2 * e] for e in range(n)]),
                  opponent_team=SequenceTeambuilder([teams[2 * e + 1] for e in range(n)]),
                  reward_fn=Gen3RewardManager(config=RewardConfig.from_args(args)),
                  account_configuration1=AccountConfiguration(f"{tag}e"[:18], None),
                  # pinned: agent2's default name is random, and the bank must re-record byte for byte
                  account_configuration2=AccountConfiguration(f"{tag}f"[:18], None),
                  start_listening=False, **kw)
    session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True, impl="rust", core_obs=True)
    player, rngs, seeds = make_bot(bot, key_base, tag)
    w = SingleAgentWrapper(env, player)
    w.action_space, w.observation_space = env.action_space, env.observation_space

    state = {"calls": [], "sent": []}
    real_choose = player.choose_move

    def spy_choose(battle):
        real = bool(env.agent2_to_move) and battle is env.battle2
        before = {s: r.words for s, r in rngs.items()}
        view = BV.view_json(battle) if real else None
        order = real_choose(battle)
        after = {s: r.words for s, r in rngs.items()}
        state["calls"].append({"real": real, "before": before, "after": after, "view": view,
                               "order": order.message})
        return order

    player.choose_move = spy_choose
    real_a2o = env.action_to_order

    def spy_a2o(action, battle, **k):
        order = real_a2o(action, battle, **k)
        if battle is env.battle2:
            state["sent"].append((order.message, int(order_to_action(order, battle))))
        elif battle is env.battle1 and isinstance(order, ForfeitBattleOrder):
            state["forfeit"] = True
        return order

    env.action_to_order = spy_a2o
    rng = np.random.default_rng(key_base)
    names = (env.agent1.username, env.agent2.username)
    episodes = []
    try:
        for e in range(n):
            session.seed = _seed(key_base + e)
            state.update(calls=[], sent=[], forfeit=False)
            obs, _ = w.reset()
            p1 = []
            for _step in range(5000):
                if env.agent1_to_move:
                    legal = np.flatnonzero(np.asarray(obs["action_mask"]))
                    act = int(rng.choice(legal))
                    p1.append(act)
                else:
                    act = 0
                obs, _r, term, trunc, _i = w.step(act)
                if state["forfeit"] and p1 and p1[-1] != -1:
                    p1[-1] = -1                      # the stall forfeit replaced that decision
                if term or trunc:
                    break
            else:
                raise AssertionError(f"{bot} episode {e} did not end in 5000 steps")
            real = [c for c in state["calls"] if c["real"]]
            phantom = [c for c in state["calls"] if not c["real"]]
            sent = list(state["sent"])
            if state["forfeit"] and len(real) == len(sent) + 1:
                # the stall forfeit: p1 forfeits at a decision p2 was ALSO asked at, and p2's order is
                # never sent (`PokeEnv.step` skips agent2 after a forfeit) — banked with no index
                sent.append((None, None))
            if len(real) != len(sent):
                raise AssertionError(f"{bot} episode {e}: {len(real)} real bot decisions but {len(sent)} "
                                     "p2 orders sent — the decision pairing is lost")
            p2 = []
            for c, (msg, idx) in zip(real, sent):
                d = {"idx": idx, "tok": None if msg is None else (msg[len("/choose "):] if msg.startswith("/choose ") else msg),
                     "bot": c["order"][len("/choose "):] if c["order"].startswith("/choose ") else c["order"],
                     "view": BV.fnv64(c["view"]), "before": c["before"], "after": c["after"]}
                if keep_views:
                    d["view_json"] = c["view"]
                p2.append(d)
            b1 = env.battle1
            episodes.append({
                "bot": bot, "seed": ",".join(map(str, _seed(key_base + e))), "names": list(names),
                "teams": [teams[2 * e], teams[2 * e + 1]], "rng_seeds": seeds,
                "p1": p1, "p2": p2,
                "phantom": {"n": len(phantom), "words": sum(sum(c["after"][s] - c["before"][s] for s in c["after"])
                                                            for c in phantom)},
                "end": {"won": b1.won, "turn": int(b1.turn), "forfeit": bool(state["forfeit"])},
            })
    finally:
        w.close()
    return episodes


def build(bots, sources, n_per, key_base: int, keep_views: bool = False) -> dict:
    """A corpus: ``n_per`` episodes per (bot, source) — an int, or ``{source: n}``."""
    eps = []
    for bi, bot in enumerate(bots):
        for si, src in enumerate(sources):
            n = n_per[src] if isinstance(n_per, dict) else n_per
            k = key_base + 1000 * bi + 100 * si
            eps += record(bot, team_list(src, n, k), key_base=k, keep_views=keep_views)
    return {"schema": SCHEMA, "episodes": eps}


#: THE COMMIT TIER — banked at `src/rust_env/tests/fixtures/bots/commit_corpus.json.gz`, rebuilt
#: by `python -m utils.rust_env.bot_corpus --commit-tier --write` (and re-recorded by the routine
#: test, so a Python bot that changes cannot leave the bank stale).
COMMIT_BOTS = tuple(r.name for r in BI.ROWS if r.used_by)
COMMIT_N = {"pool": 3, "ladder": 1, "procedural": 1}
COMMIT_KEY = 80_000


def commit_path():
    from utils.paths import src_path

    return src_path("rust_env", "tests", "fixtures", "bots", "commit_corpus.json.gz")


#: CHOSEN batches ``(bot, source, n, key)``: `staller_v2` draws its Protect coin only once the foe is
#: badly poisoned and no heal / status comes first — rare against a random p1, so the tier includes
#: a battle where it fires (3 draws), picked from a probe run rather than hoped for. `heuristic2`'s
#: SETUP step (5th in its order, F-LF-1) fires in none of its five base battles, so the tier also
#: carries one where it does (18 Calm Minds; key found by a probe over pool keys 75,000 + 100k).
COMMIT_EXTRA = (("staller_v2", "pool", 1, 74_000), ("heuristic2", "pool", 1, 75_800))


def build_commit_tier() -> dict:
    c = build(COMMIT_BOTS, tuple(COMMIT_N), COMMIT_N, COMMIT_KEY)
    for bot, src, n, key in COMMIT_EXTRA:
        c["episodes"] += record(bot, team_list(src, n, key), key_base=key)
    return c


def summary(corpus: dict) -> dict:
    out = {}
    for ep in corpus["episodes"]:
        s = out.setdefault(ep["bot"], {"episodes": 0, "decisions": 0, "draw_decisions": 0, "bot_ne_sent": 0,
                                        "phantom": 0, "phantom_words": 0, "forfeits": 0})
        s["episodes"] += 1
        s["decisions"] += len(ep["p2"])
        s["draw_decisions"] += sum(1 for d in ep["p2"] if d["after"] != d["before"])
        s["bot_ne_sent"] += sum(1 for d in ep["p2"] if d["tok"] is not None and d["bot"] != d["tok"])
        s["phantom"] += ep["phantom"]["n"]
        s["phantom_words"] += ep["phantom"]["words"]
        s["forfeits"] += int(ep["end"]["forfeit"])
    return out


def write(corpus: dict, path) -> None:
    data = json.dumps(corpus, separators=(",", ":"), sort_keys=True).encode()
    with open(path, "wb") as raw:
        if str(path).endswith(".gz"):
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:   # byte-reproducible
                f.write(data)
        else:
            raw.write(data)


def read(path) -> dict:
    with (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")) as f:
        c = json.loads(f.read())
    assert c.get("schema") == SCHEMA, f"{path}: schema {c.get('schema')!r}, expected {SCHEMA}"
    return c


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bots", default=",".join(r.name for r in BI.ROWS if r.used_by))
    ap.add_argument("--sources", default="pool")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--key", type=int, default=70_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--views", action="store_true")
    ap.add_argument("--commit-tier", action="store_true", help="build the COMMIT tier (--write banks it)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.commit_tier:
        c = build_commit_tier()
        print(json.dumps(summary(c), indent=1))
        if a.write:
            write(c, commit_path())
            print(f"banked {commit_path()}")
        raise SystemExit(0)
    c = build(a.bots.split(","), a.sources.split(","), a.n, a.key, keep_views=a.views)
    print(json.dumps(summary(c), indent=1))
    if a.out:
        write(c, a.out)
