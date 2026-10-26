"""``main.reads.runner``: plan writes the state once per spec and plans a model's anchor units only when it SETTLES;
the playing code is pinned; the anchor lane works the queue through an injected unit runner; the S lane's argv is
one the real ``main.h2h`` parser accepts. No game is played here."""
from __future__ import annotations

import json
import os
import time

import pytest

from main.reads import layout as LY
from main.reads import runner as RN
from main.x35.kakuna_test import install_fake_play

CODE = {"commit": "e05c45eeb" + "0" * 31, "dirty": False, "root": "/tmp/pin"}


def _zip(d, name="final_model.zip", data=b"zip", age=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_config.json").write_text("{}")
    z = d / name
    z.write_bytes(data)
    if age is not None:
        t = time.time() - age
        os.utime(z, (t, t))
    return z


def _spec(tmp_path, models_dir, *, pin=None):
    from main.reads.models import sha256_file

    pz = _zip(models_dir / "rb_old", "checkpoint_1_steps.zip", b"old")
    doc = {"schema": "gen3_read_spec_v1", "read": "demo_read_2026-10-25",
           "models": {"old": {"arm": "old", "seed": 1, "ref": {"kind": "pinned", "path": str(pz),
                                                               "sha256": sha256_file(pz)}},
                      "new": {"arm": "new", "seed": 1, "ref": {"kind": "final", "run": "models/rb_new",
                                                               "min_step": 100, "quiet_s": 60}}},
           "anchor": {"u": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix"], "unit_seeds": "0-1",
                            "models": ["old", "new"]}},
           "decision": {"rule": "anchor_trajectory", "params": {"on": "new", "off": "old", "anchor_group": "u",
                                                                "goal_bar": 0.5, "min_unit_frac": 0.9}}}
    if pin:
        doc["pin"] = pin
    p = tmp_path / "read_spec.json"
    p.write_text(json.dumps(doc))
    return p


def _step(_p):
    return 150


def test_plan_settles_models_one_at_a_time_and_never_plays_an_unsettled_final(tmp_path, run_archive_env):
    spec = _spec(tmp_path, run_archive_env)
    state = tmp_path / "state"
    out = RN.plan(spec, state, code=CODE, step_of=_step)
    assert set(out["waiting"]) == {"new"} and "no final" in out["waiting"]["new"]
    assert (LY.anchor_state(state, "u", "old") / "plan.json").is_file()
    assert not LY.anchor_state(state, "u", "new").exists()
    _zip(run_archive_env / "rb_new", age=10)                      # being written: not quiet
    assert "quiet" in RN.plan(spec, state, code=CODE, step_of=_step)["waiting"]["new"]
    _zip(run_archive_env / "rb_new", age=3600)
    assert RN.plan(spec, state, code=CODE, step_of=_step)["waiting"] == {}
    units = json.loads((LY.anchor_state(state, "u", "new") / "plan.json").read_text())["units"]
    m = json.loads(LY.models_file(state).read_text())["new"]
    assert [u["name"] for u in units] == ["new__heldout72_hpfix__00", "new__heldout72_hpfix__01"]
    assert {u["model"] for u in units} == {m["copy"]} and {u["model_sha256"] for u in units} == {m["sha256"]}
    assert m["copy"].startswith(str(state)), "a unit plays the sha-checked COPY, never the run dir"


def test_a_state_is_one_spec_and_never_inside_the_repo(tmp_path, run_archive_env):
    from utils.paths import repo_path

    spec = _spec(tmp_path, run_archive_env)
    RN.plan(spec, tmp_path / "state", code=CODE, step_of=_step)
    other = tmp_path / "other.json"
    other.write_text(spec.read_text().replace("demo_read", "demo_other"))
    with pytest.raises(RN.RunError, match="another spec"):
        RN.plan(other, tmp_path / "state", code=CODE, step_of=_step)
    with pytest.raises(LY.StateError, match="repo checkout"):
        RN.plan(spec, repo_path("designs", "x_state"), code=CODE, step_of=_step)
    with pytest.raises(LY.StateError, match="run archive"):
        RN.plan(spec, run_archive_env / "x_state", code=CODE, step_of=_step)


def test_the_playing_code_is_pinned(tmp_path, run_archive_env):
    spec = _spec(tmp_path, run_archive_env, pin="e05c45eeb")
    with pytest.raises(RN.RunError, match="not the spec's pin"):
        RN.plan(spec, tmp_path / "s1", code=dict(CODE, commit="f" * 40), step_of=_step)
    with pytest.raises(RN.RunError, match="tracked changes"):
        RN.plan(spec, tmp_path / "s2", code=dict(CODE, dirty=True), step_of=_step)
    RN.plan(spec, tmp_path / "s3", code=CODE, step_of=_step)
    with pytest.raises(RN.RunError, match="changed under the read"):
        RN._code_holds(tmp_path / "s3", dict(CODE, commit="e05c45eeb" + "1" * 31))
    with pytest.raises(RN.RunError, match="changed under the read"):  # a dirty tree edited again after plan
        RN._code_holds(tmp_path / "s3", dict(CODE, diff_sha="0123456789abcdef"))


def test_the_anchor_lane_works_every_unit_once_then_stops(tmp_path, run_archive_env, monkeypatch):
    spec = _spec(tmp_path, run_archive_env)
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    played = []

    def fake_unit(st, u):
        played.append(u.name)
        d = st / "cells" / u.name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "status": "OK", "n": 20, "wins": 10, "ties": 0, "regime_verified_decisions": True, "peer_clean": True,
            "team_source_asymmetry": False}))
        return 0

    rc = RN.run(state, "anchor", time.time() + 3600, run_unit=fake_unit, sleep=lambda s: None)
    assert rc == 0
    assert played == ["old__heldout72_hpfix__00", "new__heldout72_hpfix__00", "old__heldout72_hpfix__01",
                      "new__heldout72_hpfix__01"], "unit seeds outermost: a stopped queue stays seed-balanced"
    assert RN.run(state, "anchor", time.time() + 3600, run_unit=fake_unit, sleep=lambda s: None) == 0
    assert len(played) == 4, "a counted unit is never replayed"
    st = RN.status(state)
    assert st["anchor"]["u"] == {"old": {"counted": 2, "planned": 2}, "new": {"counted": 2, "planned": 2}}


def test_the_anchor_lane_counts_a_unit_that_overran_the_short_lane_by_playing_it_once_more_long(
        tmp_path, run_archive_env, monkeypatch):
    """N1-P unit (2026-10-26): through the lane's DEFAULT unit runner, a unit killed past the short lane (exit 88) was an
    uncounted attempt, and one that overran twice was never counted. Now the same attempt plays it again as a normal
    heavy-slot job, and the lane counts it; nothing else is replayed."""
    fake_play = install_fake_play(monkeypatch)
    spec = _spec(tmp_path, run_archive_env)
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    fake_play.script = [(88, False), (0, True), (0, True), (0, True), (0, True)]
    assert RN.run(state, "anchor", time.time() + 3600, sleep=lambda s: None) == 0
    assert len(fake_play.argvs) == 5, "four units, one of them played twice"
    assert "--max-s" in fake_play.argvs[0] and "--max-s" not in fake_play.argvs[1] and "timeout" in fake_play.argvs[1]
    assert all("--max-s" in a for a in fake_play.argvs[2:])
    assert RN.status(state)["anchor"]["u"] == {"old": {"counted": 2, "planned": 2}, "new": {"counted": 2, "planned": 2}}
    jl = [json.loads(x) for x in (LY.anchor_state(state, "u", "old") / "units.jsonl").read_text().splitlines()]
    assert [e["unit"] for e in jl if e["event"] == "short_lane_overrun"] == ["old__heldout72_hpfix__00"]
    assert not any(e.get("event") == "done" and e["not_counted"] for e in
                   (json.loads(x) for x in (state / "anchor_supervisor.jsonl").read_text().splitlines()))


def test_a_lane_has_one_supervisor_and_a_registered_stop(tmp_path, run_archive_env, monkeypatch):
    spec = _spec(tmp_path, run_archive_env)
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    held = RN._lock(state, "anchor")
    try:
        with pytest.raises(RN.RunError, match="another anchor supervisor"):
            RN._lock(state, "anchor")
    finally:
        held.close()
    from main.x35.kakuna import KakunaError

    with pytest.raises(KakunaError, match="stop-at-epoch"):
        RN.run(state, "anchor", time.time() - 1)
    with pytest.raises(RN.RunError, match="no s group"):
        RN.run(state, "s", time.time() + 60)


def test_s_engines_split_cells_over_the_workers():
    assert RN.s_engines([("a", ["m1", "m2", "m3"])], 2) == ["a:m1,m3", "a:m2"]
    assert RN.s_engines([("a", ["m1"]), ("b", ["m1", "m2"])], 2) == ["a:m1", "b:m1,m2"]


def test_the_s_unit_and_play_many_argv_are_what_the_real_parsers_accept(tmp_path):
    from types import SimpleNamespace as NS

    from main.h2h.cli import _parser as h2h_parser
    from main.reads.cli import _parser
    from main.reads.spec import SGroup

    argv = RN.s_unit_argv(tmp_path, "pool", ["a:m1,m2"], 2400, deadline_epoch=2_000_000_000)
    assert argv[1:5] == ["--slot", "--name", "reads_s", "12"] and argv[5:11] == ["nice", "-n", "19", "taskset", "-c",
                                                                                   RN.CORES]
    i = argv.index("main.reads")
    a = _parser().parse_args(argv[i + 1:])
    assert (a.cmd, a.group, a.box_s, a.deadline_epoch, a.engine) == ("s-unit", "pool", 2400, 2_000_000_000,
                                                                     ["a:m1,m2"])
    g = SGroup("pool", "r.json", "c" * 64, "home72", 500, 100, "fam", ("a",))
    spec = NS(s={"pool": g}, read="demo")
    pm = RN.play_many_argv(spec, tmp_path, "pool", "a", ["m1", "m2"], {"a": {"copy": "/s/a.zip"}},
                           {"m1": {"copy": "/s/m1.zip"}, "m2": {"copy": "/s/m2.zip"}}, threads=2)
    b = h2h_parser().parse_args(pm[pm.index("main.h2h") + 1:])
    assert (b.cmd, b.players, b.opponents, b.pairs, b.batch_pairs, b.family, b.request, b.panel, b.purpose) == (
        "play-many", ["/s/a.zip"], ["/s/m1.zip", "/s/m2.zip"], 500, 100, "fam", "pool_a_b100", "home72", "ab")
    assert (b.device, b.torch_threads) == ("cpu", 1)


def test_a_non_kakuna_opponent_is_planned_with_its_derived_config_in_the_anchor_state(tmp_path, run_archive_env):
    """be60f24f FINDING 2: a group may face another Metamon anchor; its units carry the opponent and, for an agent
    the committed cell config lacks, the derived config written into the group's kakuna state."""
    spec = _spec(tmp_path, run_archive_env)
    doc = json.loads(spec.read_text())
    doc["anchor"]["u"]["opponent"] = "metamon:LargeIL"
    spec.write_text(json.dumps(doc))
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    units = json.loads((LY.anchor_state(state, "u", "old") / "plan.json").read_text())["units"]
    assert {u["opponent"] for u in units} == {"metamon:LargeIL"}
    assert {u["config"] for u in units} == {str(LY.anchor_state(state, "u", "old") / "configs"
                                                / "anchors_heldout72_hpfix__LargeIL.json")}
