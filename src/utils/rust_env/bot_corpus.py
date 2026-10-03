"""THE BANKED BOT DECISION CORPUS — the Python bots' own decisions, recorded as RE-PLAYABLE battle
inputs (M5 Lane F, `designs/endstate/program_rust_core.md` §2 M5). **A FROZEN FIXTURE since deletion
pass U3 (R10, owner D3 2026-10-02): the bank is READ here and replayed by the Rust gate
(`rust_env/tests/bots_gate_test.rs`); the Python RE-RECORD — a `Gen3Env` over the rust bridge with every
bot RNG stream counted — was the Python env core's oracle and was deleted with it, so nothing re-derives
the bank from the Python bots any more.** A Python bot change is now caught by the bots' own unit tests;
the bank pins the Rust port to the Python bots as they were when it was last recorded (git history of
this file holds the recorder).

An episode is ``{"seed", "names", "teams", "p1": [idx | -1 forfeit], "p2": [decision…], "end",
"phantom"}`` — the env core's input log in index form (`search::game::Log` + `Game::feed`). Every
decision holds the bot view's hash (`bot_view.fnv64(view_json(battle2))`), the order the bot returned,
the order the env SENT as a choice token, that order's 11-dim action index, and the bot RNG stream's
MT19937 word offsets before and after it (the Rust bot draws the same stream from the same offset,
`bots::rng::PyRandom`).
"""
from __future__ import annotations

import gzip
import json

from utils.rust_env import bot_inventory as BI

SCHEMA = "gen3_bot_corpus_v1"
NAMES = ("bfpone", "bfptwo")


#: THE COMMIT TIER — banked at `src/rust_env/tests/fixtures/bots/commit_corpus.json.gz` (frozen; the
#: plan below is the record of which (bot, source, n, key) batches it holds).
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
#: CURSE (owner 2026-09-29: a non-Ghost's Curse is a setup move): `heuristic2` already picks it in
#: the base battles; the three batches after it are the first pool keys from 76,000 whose p2 team
#: carries Curse and where that bot picks it (heuristic 1, setup_sweep 5, setup_sweep_v2 1).
COMMIT_EXTRA = (("staller_v2", "pool", 1, 74_000), ("heuristic2", "pool", 1, 75_800),
                ("heuristic", "pool", 1, 76_011), ("setup_sweep", "pool", 1, 76_011),
                ("setup_sweep_v2", "pool", 1, 76_012))


def commit_tier_plan() -> list:
    """The COMMIT tier as its ordered ``(bot, source, n, key)`` batches — `build`'s keys for the base
    tier, then `COMMIT_EXTRA`. Each batch banks exactly ``n`` episodes, in this order."""
    plan = []
    for bi, bot in enumerate(COMMIT_BOTS):
        for si, src in enumerate(COMMIT_N):
            plan.append((bot, src, COMMIT_N[src], COMMIT_KEY + 1000 * bi + 100 * si))
    return plan + list(COMMIT_EXTRA)


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
