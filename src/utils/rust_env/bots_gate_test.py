"""THE BOT GATE (M5 Lane F): every banked Python bot decision is equal in the Rust port — the view
it read, the action, the token and the RNG stream offset — and the banked battle replays.

* COMMIT (routine): the banked corpus `src/rust_env/tests/fixtures/bots/commit_corpus.json.gz`
  (every pooled bot × pool / ladder / procedural teams) through `tests/bots_gate_test.rs`.
  **The bank is a FROZEN FIXTURE** (deletion pass U3, owner D3): the re-record that proved the Python
  bots still produce it byte for byte ran through the Python env core and was deleted with it, as was
  the fresh-corpus MILESTONE.
* TEETH (routine): a moved action, a moved view hash, a moved stream offset, and a bot relabelled
  as another bot each FAIL, on the counter they should.

Since P6 slice 6d-1 (2026-10-08) the Python bots (`agents/opponents.py`) and the bot VIEW renderer
(`utils/rust_env/bot_view.py`) are deleted: this gate holds the Rust bots to the BANK alone. The bank-content
check that decided "is this banked token a setup move" through the fork's `Move`/`Target` is retired with them
(the F-LF-1 setup-site coverage stays pinned by `test_commit_tier_every_banked_decision_is_equal`'s branch
counters, and the Curse check below needs no fork).
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
DRAWING = ("random", "staller", "staller_v2")   # the bots whose streams must be exercised


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
    # F-LF-1: each setup step's Rust return site is REACHED by the bank (it was dead before the fix).
    sites = _setup_sites()
    for bot, line in sites.items():
        assert rep["bots"][bot]["branches"].get(str(line), 0) >= 1, \
            (f"{bot}: the setup site logic.rs:{line} is never reached", rep["bots"][bot]["branches"])


SETUP_BOTS = ("heuristic", "setup_sweep", "setup_sweep_v2", "heuristic2")   # logic.rs order


def _setup_sites() -> dict:
    """The line of each `// F-LF-1 SETUP SITE` return in `logic.rs` (the gate keys a branch by it)."""
    lines = src_path("rust_env", "src", "bots", "logic.rs").read_text().splitlines()
    found = [i + 2 for i, ln in enumerate(lines) if "F-LF-1 SETUP SITE" in ln]
    assert len(found) == len(SETUP_BOTS), found
    return dict(zip(SETUP_BOTS, found))


def test_every_setup_bot_sets_up_with_curse(bank):
    """Owner 2026-09-29, "allow Curse": a non-Ghost's Curse is a setup move for all four setup bots
    (``baselines.self_setup_boosts``). Before, poke-env's Curse (target NORMAL, no boosts) could not
    pass any setup step, so the bank held ZERO Curse choices. Reverting only the Rust port fails the
    COMMIT gate's action counter."""
    counts = {b: 0 for b in SETUP_BOTS}
    for ep in bank["episodes"]:
        if ep["bot"] in counts:
            counts[ep["bot"]] += sum(d["tok"] == "move curse" for d in ep["p2"])
    assert all(n >= 1 for n in counts.values()), counts


def test_the_bank_holds_exactly_the_commit_tier_plan(bank):
    assert sum(n for _, _, n, _ in BC.commit_tier_plan()) == len(bank["episodes"])


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
