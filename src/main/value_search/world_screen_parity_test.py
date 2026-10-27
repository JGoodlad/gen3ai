"""THE LEAN WORLD SCREEN'S PARITY GATE (``worlds.WorldSampler(lean_screen=...)``, ``core_events`` ``SCREEN``).

On banked S-U decisions (``testdata/su_stage0_decisions.json``: stage 0's first searched game, its turns 10, 32 and 43
searched and its turn 29 an EXCLUSION), the lean screen builds exactly the worlds the reference path builds (every
candidate replayed in full): the teams, tries, facts, failure counts, leak block, every sampler's draws and records, and
an exclusion's error, word for word. The only field that may differ is the count of replays run (``Worlds.screens``),
which the lean screen exists to cut; it must be smaller, so the lean path is known to have run. The search on those
worlds picks the same action with the same values. A deliberately wrong screen (one reference line changed) is caught.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from agents.battle import core_replay as CR
from main.value_search import worlds as W
from utils.paths import src_path

FIX = Path(__file__).parent / "testdata" / "su_stage0_decisions.json"
J = 8          # R0 §4's (S-U's) worlds per decision


def _decision(turn: int) -> dict:
    got = [d for d in json.loads(FIX.read_text())["decisions"] if d["turn"] == turn]
    assert len(got) == 1, turn
    return got[0]


def _build(d: dict, lean: bool, monkeypatch, dice: str = "seed") -> "tuple[dict, int]":
    """The decision's worlds (``Worlds`` as a dict, or its exclusion's error) and the replays the build ran, counted at
    ``run_core`` (an exclusion's included)."""
    n = [0]
    orig = CR.run_core

    def counting(battles, **kw):
        n[0] += len(battles)
        return orig(battles, **kw)

    s = W.WorldSampler(reveal="off", max_tries=60, dice=dice, lean_screen=lean)
    with monkeypatch.context() as m:
        m.setattr(CR, "run_core", counting)
        try:
            rec = dataclasses.asdict(s.sample(d["log"], d["at"], d["side"], J, d["key"], root_choice_in_log=False))
        except W.WorldError as exc:
            rec = {"error": str(exc)}
    return rec, n[0]


def _canon(rec: dict) -> str:
    return json.dumps({k: v for k, v in rec.items() if k != "screens"}, sort_keys=True, default=str)


@pytest.mark.slow
@pytest.mark.sim
@pytest.mark.integration
@pytest.mark.parametrize("turn,dice", [(10, "seed"), (29, "seed"), (32, "seed"), (43, "seed"), (10, "rolls")])
def test_the_lean_screen_builds_exactly_the_reference_worlds(turn, dice, monkeypatch):
    """S-U's preset (``dice="seed"``) on all four decisions; F1 tier A's ``dice="rolls"`` (the damage-roll control
    rides the screened replay) on the cheap one."""
    d = _decision(turn)
    (full, n_full), (lean, n_lean) = _build(d, False, monkeypatch, dice), _build(d, True, monkeypatch, dice)
    assert _canon(lean) == _canon(full)
    if dice == "seed":
        assert ("error" in full) == (d["live"]["why"] == "excluded"), "the fixture's outcome is the live game's"
        assert "error" in full or full["tries"] == d["live"]["world_tries"], "the reference is the live game's builder"
    assert n_lean < n_full, "the lean path ran (fewer replays: the prior's impossible proposals are not replayed)"


@pytest.mark.slow
@pytest.mark.sim
@pytest.mark.integration
def test_a_wrong_screen_is_caught(monkeypatch):
    """The negative control: a screen whose reference has one line changed (half way) stops every candidate that
    matches the truth there and scores it as diverged, so the build no longer equals the reference path's. The chain
    is capped at 25 steps to keep it cheap; the cap does not bind on this decision's true build (asserted)."""
    from main.value_search import world_screen as WS

    d = _decision(10)
    full, _ = _build(d, False, monkeypatch)
    monkeypatch.setattr(WS, "MAX_STEPS", 25)
    assert _canon(_build(d, False, monkeypatch)[0]) == _canon(full), "the cap binds: pick another decision"
    orig = CR.run_core

    def short_ref(battles, **kw):
        if kw.get("screen") is not None:
            lines = list(kw["screen"]["lines"])
            lines[len(lines) // 2] += "|perturbed"
            kw["screen"] = dict(kw["screen"], lines=lines)
        return orig(battles, **kw)

    monkeypatch.setattr(CR, "run_core", short_ref)
    assert _canon(_build(d, True, monkeypatch)[0]) != _canon(full)


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
    """Logits from a fixed projection of the row, V a fixed function of it (``search_integration_test``'s stub)."""
    rng = np.random.default_rng(7)
    proj = rng.standard_normal((rows.shape[1], 11)).astype(np.float32) * 0.05
    out = np.zeros((len(rows), 12), dtype=np.float32)
    out[:, :11] = rows @ proj
    out[:, 11] = 1.0 / (1.0 + np.exp(-rows.astype(np.float64).mean(axis=1) * 10.0))
    return out


@pytest.mark.slow
@pytest.mark.sim
@pytest.mark.integration
def test_the_search_on_the_lean_screens_worlds_is_the_reference_search(lib):
    """The whole decision, S-U's preset (8 worlds, 8 playouts an action, 4 turns, piKL): the same action, values,
    targets, tries, leak and world teams; only the CPU and wall fields differ."""
    from main.anchors.search_presets import PRESETS
    from main.value_search.playout import LeafEngine
    from main.value_search.search import SearchConfig, search_decision

    p = PRESETS["r0s4"]
    d = _decision(10)
    row = np.frombuffer(base64.b64decode(d["row_b64"]), dtype=np.float32).copy()
    mask = np.asarray(d["mask"], dtype=np.uint8)
    cfg = SearchConfig(per_action=p.per_action, n_worlds=p.n_worlds, depth=p.depth, allocation="uniform", rule=p.rule,
                       pikl_lambda=p.pikl_lambda, weights="uniform")
    out = {}
    for lean in (False, True):
        eng = LeafEngine(_stub, n_cores=p.n_worlds, lib=lib)
        try:
            r = search_decision(eng, d["log"], d["at"], d["side"], row, mask, cfg, d["key"],
                                sampler=W.WorldSampler(reveal=p.reveal, max_tries=p.max_tries, dice=p.dice,
                                                       lean_screen=lean),
                                root_choice_in_log=False)
        finally:
            eng.close()
        rec = dataclasses.asdict(r)
        assert rec["world_cpu_s"] > 0 and rec["playout_cpu_s"] > 0
        out[lean] = json.dumps({k: v for k, v in rec.items()
                                if k not in ("cpu_s", "wall_s", "world_cpu_s", "playout_cpu_s")},
                               sort_keys=True, default=lambda x: np.asarray(x).tolist())
    assert out[True] == out[False]


@pytest.mark.sim
@pytest.mark.integration
def test_a_screened_replay_returns_the_viewers_part_and_stops_after_the_first_difference():
    """``core_events`` ``SCREEN`` directly, on the truth's own replay: with the truth's lines as the reference nothing
    stops and the viewer's chunks and rows are the full replay's, every other chunk an empty placeholder in its place;
    with one line changed the replay stops after the command that shipped it."""
    from main.value_search.worlds import log_commands

    d = _decision(10)
    log, viewer = d["log"], 0 if d["side"] == "p1" else 1
    rb = CR.RecordedBattle(label="t", format_id=log["format_id"], seed=log["seed"],
                           p1={"name": log["names"][0], "team": log["teams"][0]},
                           p2={"name": log["names"][1], "team": log["teams"][1]},
                           commands=[c + ["if_open"] for c in log_commands(log, d["at"])])
    full = CR.run_core([rb], trackers=True, obs=True)[0]
    ref = [ln for s, lines in full["chunks"] if s == viewer for ln in lines]
    same = CR.run_core([rb], trackers=True, obs=True, screen={"viewer": viewer, "lines": ref})[0]
    assert full["ok"] and same["ok"] and same["stopped"] is False
    assert [s for s, _ in same["chunks"]] == [s for s, _ in full["chunks"]]
    assert all(lines == (f if s == viewer else []) for (s, lines), (_, f) in zip(same["chunks"], full["chunks"]))
    rows = [(e["after"], e["obs"]) for e in full["trackers"][viewer] if "obs" in e]
    assert [(e["after"], e["obs"]) for e in same["trackers"][viewer]] == rows and same["trackers"][1 - viewer] == []
    i = len(ref) // 2
    bad = ref[:i] + ["|bogus|line"] + ref[i + 1:]
    st = CR.run_core([rb], trackers=True, obs=True, screen={"viewer": viewer, "lines": bad})[0]
    got = [ln for s, lines in st["chunks"] if s == viewer for ln in lines]
    assert st["ok"] and st["stopped"] is True
    assert got[:i] == ref[:i] and i < len(got) < len(ref)
