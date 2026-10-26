"""THE REFEREE's in-core bot toggle (``playout.referee``, E1-0) on real banked turns (the core, a stub forward).

The teeth: the self-referee (the default) reproduces the single-open play of every action that the banked truth rows
used; with ``opp_bot`` the opponent's decisions are answered INSIDE the core (never forwarded), deterministically; and
the search's own leaves (``LeafEngine``) take no bot.
"""
from __future__ import annotations

import inspect
import os
import shutil
import subprocess

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum.bank_test import BANK_V1
from main.policy_spectrum.truth import cmd_index, select_subset, to_log, turn_seeds
from main.value_search.playout import ACT, LeafEngine, OppBot, referee
from utils.paths import src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env import ffi

    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed")
    crate = src_path("rust_env")
    r = subprocess.run([cargo, "build", "--lib", "--profile", "selfcheck", "--features", "emission-selfcheck",
                        "--manifest-path", str(crate / "Cargo.toml")],
                       env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")),
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _stub(rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    proj = rng.standard_normal((rows.shape[1], 11)).astype(np.float32) * 0.05
    out = np.zeros((len(rows), 12), dtype=np.float32)
    out[:, :11] = rows @ proj
    out[:, 11] = 0.5
    return out


@pytest.fixture(scope="module")
def turns():
    """Two free turns (turn >= 4, >= 3 legal) of the committed Lane S bank."""
    bank = B.load_bank(BANK_V1)
    dd = {d["id"]: d for d in bank.decisions}
    ids = [i for i in select_subset(bank) if dd[i]["kind"] == "free" and dd[i]["turn"] >= 4
           and dd[i]["n_legal"] >= 3][:2]
    assert len(ids) == 2
    out = []
    for i in ids:
        d = dd[i]
        b = bank.battles[bank.battle_index[d["battle"]]]
        out.append({"id": i, "log": to_log(b), "at": cmd_index(b, d["side"], d["n"]), "side": d["side"],
                    "seeds": turn_seeds(i, 4)})
    return out


def _single_open(lib, t) -> dict:
    """The banked truth rows' play: ONE open of every action x the seeds (``successors.play_out``)."""
    from utils.rust_env.successors import SearchCore, play_out

    me = 0 if t["side"] == "p1" else 1

    def pol(rows, masks, who):
        lg = np.where(masks.astype(bool), _stub(rows, masks)[:, :ACT].astype(np.float64), -np.inf)
        return lg.argmax(axis=1)

    with SearchCore(lib=lib) as core:
        res = play_out(t["log"], t["at"], t["side"], policy=pol, seeds=t["seeds"], core=core)
    out: dict = {}
    for br in res.branches:
        e = br["end"]
        c = "0" if (e["truncated"] or e["winner"] is None) else "+" if int(e["winner"]) == me else "-"
        out.setdefault(int(br["action"]), [None] * len(t["seeds"]))[int(br["seed"])] = c
    return {a: "".join(v) for a, v in out.items()}


def test_the_self_referee_is_the_default_and_reproduces_the_single_open_play(lib, turns):
    for t in turns:
        r = referee(_stub, t["log"], t["at"], t["side"], t["seeds"], lib=lib)
        assert r.out == _single_open(lib, t)
        assert sorted(r.out) == sorted(r.tokens) and len(r.out) >= 3
        other = 1 if t["side"] == "p1" else 0
        assert r.fwd_rows[other] > 0          # the self-referee forwards the opponent's decisions


def test_the_bot_referee_never_forwards_the_bot_side_and_is_deterministic(lib, turns):
    for t in turns:
        other = 1 if t["side"] == "p1" else 0
        a = referee(_stub, t["log"], t["at"], t["side"], t["seeds"], opp_bot=OppBot("heuristic", 5), lib=lib)
        b = referee(_stub, t["log"], t["at"], t["side"], t["seeds"], opp_bot=OppBot("heuristic", 5), lib=lib)
        assert a.fwd_rows[other] == 0, "the bot side reached the forward: the passthrough is not wired"
        assert a.fwd_rows[1 - other] > 0
        assert a.out == b.out and sorted(a.out) == sorted(a.tokens)
        assert all(len(v) == len(t["seeds"]) and set(v) <= set("+-0") for v in a.out.values())


def test_an_unknown_bot_is_refused(lib, turns):
    t = turns[0]
    with pytest.raises(Exception, match="not one this core plays"):
        referee(_stub, t["log"], t["at"], t["side"], t["seeds"], opp_bot=OppBot("no_such_bot", 1), lib=lib)


def test_the_search_leaves_take_no_bot():
    for fn in (LeafEngine.__init__, LeafEngine.leaves, LeafEngine.root):
        assert not any("bot" in p for p in inspect.signature(fn).parameters)
