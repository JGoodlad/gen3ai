"""THE BOT GATE (M5 Lane F): every banked Python bot decision is equal in the Rust port — the view
it read, the action, the token and the RNG stream offset — and the banked battle replays.

* COMMIT (routine): the banked corpus `src/rust_env/tests/fixtures/bots/commit_corpus.json.gz`
  (every pooled bot × pool / ladder / procedural teams) through `tests/bots_gate_test.rs`, plus a
  RE-RECORD: the Python bots, run again on the same keys, must produce the bank byte for byte — so
  a Python bot change cannot leave the bank (and the port) stale in silence.
* TEETH (routine): a moved action, a moved view hash, a moved stream offset, and a bot relabelled
  as another bot each FAIL, on the counter they should.
* MILESTONE (``slow``; verdict in `designs/ops/slow_tier_status.json`): a freshly recorded corpus
  at scale over the three team sources.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess

import pytest

from utils.paths import src_path
from utils.rust_env import bot_corpus as BC
from utils.rust_env import bot_inventory as BI

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")
DRAWING = ("random", "staller", "staller_v2", "baitbot")   # the bots whose streams must be exercised


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the bot gate cannot run (install rustup)")
    return cargo


@pytest.fixture(scope="module")
def built():
    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run([_cargo(), "build", "--tests", *FEATURES, "--manifest-path", str(crate / "Cargo.toml")],
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the gate failed:\n{r.stderr[-4000:]}"
    return env


def gate(built, corpus: dict, tmp_path, *, expect_fail=False, name="corpus.json") -> dict:
    """Run the Rust gate on ``corpus``; return its report (the ``BOTS_GATE`` line)."""
    path = tmp_path / name
    path.write_text(json.dumps(corpus, separators=(",", ":")))
    env = dict(built, POKESIM_BOTS_CORPUS=str(path), POKESIM_BOTS_EXPECT_FAIL="1" if expect_fail else "0")
    crate = src_path("rust_env")
    r = subprocess.run([_cargo(), "test", *FEATURES, "--manifest-path", str(crate / "Cargo.toml"),
                        "--test", "bots_gate_test", "--", "--ignored", "--nocapture"],
                       env=env, capture_output=True, text=True, timeout=1800)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("BOTS_GATE ")), None)
    assert line is not None, f"the gate printed no report:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    rep = json.loads(line[len("BOTS_GATE "):])
    assert r.returncode == 0, f"the gate failed (expect_fail={expect_fail}):\n{json.dumps(rep, indent=1)[:6000]}"
    return rep


def _total(rep, key):
    return sum(t[key] for t in rep["bots"].values())


def _assert_clean(rep, corpus):
    bad = {b: t for b, t in rep["bots"].items() if t["view"] + t["action"] + t["rng"] + t["error"] + t["replay"]}
    assert not bad, (bad, rep["examples"])
    banked = sum(len(ep["p2"]) for ep in corpus["episodes"])
    assert _total(rep, "decisions") == banked, "the gate compared fewer decisions than the corpus banks"


@pytest.fixture(scope="module")
def bank():
    return BC.read(BC.commit_path())


def test_commit_tier_every_banked_decision_is_equal(built, bank, tmp_path):
    rep = gate(built, bank, tmp_path)
    _assert_clean(rep, bank)
    ported = {r.name for r in BI.ported()}
    assert ported == set(BC.COMMIT_BOTS), (ported, BC.COMMIT_BOTS)
    for bot in BC.COMMIT_BOTS:
        assert rep["bots"][bot]["decisions"] >= 40, (bot, rep["bots"][bot])
    for bot in DRAWING:
        assert rep["bots"][bot]["draw_decisions"] > 0, f"{bot}: no decision drew — its stream is untested"


def test_commit_tier_the_bank_is_what_the_python_bots_do_today(bank):
    """Re-record on the bank's keys: the Python bots must reproduce it exactly."""
    fresh = BC.build_commit_tier()
    assert len(fresh["episodes"]) == len(bank["episodes"])
    for e, (a, b) in enumerate(zip(fresh["episodes"], bank["episodes"])):
        assert a == b, (f"episode {e} ({b['bot']}) re-records differently — a Python bot, the env or the "
                        "view changed; rebuild the bank with `python -m utils.rust_env.bot_corpus --commit-tier "
                        "--write` and re-run this gate")


def _first(corpus, bot, pred=lambda d: True):
    for ep in corpus["episodes"]:
        if ep["bot"] == bot:
            for d in ep["p2"]:
                if pred(d):
                    return d
    raise AssertionError(f"no {bot} decision matches")


@pytest.mark.parametrize("what", ["action", "view", "rng", "relabel"])
def test_the_gate_has_teeth(built, bank, tmp_path, what):
    c = copy.deepcopy(bank)
    if what == "action":
        d = _first(c, "heuristic2", lambda d: d["tok"] is not None and d["tok"].startswith("move"))
        d["tok"], d["idx"] = "move struggle", 10
    elif what == "view":
        d = _first(c, "setup_sweep")
        d["view"] = "0" * 16
    elif what == "rng":
        d = _first(c, "staller_v2", lambda d: d["after"] != d["before"])
        d["after"]["protect"] += 2
    else:   # the same battles, attributed to a different bot: the ACTIONS must diverge
        for ep in c["episodes"]:
            if ep["bot"] == "aggressive":
                ep["bot"] = "setup_sweep_v2"
    rep = gate(built, c, tmp_path, expect_fail=True, name=f"teeth_{what}.json")
    key = "action" if what == "relabel" else what
    assert _total(rep, key) >= 1, (what, rep)


@pytest.mark.slow
def test_milestone_every_decision_is_equal_at_scale(built, tmp_path):
    bots = BC.COMMIT_BOTS
    corpus = BC.build(bots, ("pool", "ladder", "procedural"), {"pool": 20, "ladder": 20, "procedural": 12}, 90_000)
    rep = gate(built, corpus, tmp_path)
    print(json.dumps({b: {k: t[k] for k in ("episodes", "decisions", "draw_decisions", "branches")}
                      for b, t in rep["bots"].items()}))
    _assert_clean(rep, corpus)
    for bot in DRAWING:
        assert rep["bots"][bot]["draw_decisions"] > 0, bot
