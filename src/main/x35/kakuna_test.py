"""X35's Kakuna unit runner (``main.x35.kakuna``) without playing: the committed home72 config and manifest agree with
the committed team files, the plan's unit grid, the counting rule, the argv's rule-5b wrapping, the manifest's
refusals on synthetic team dirs, and the G-K pooling. No server, no peer, no game."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.x35 import kakuna as KA


def test_the_committed_config_is_the_x26_reads_and_our_side_matches_the_manifest():
    assert KA._sha(KA.CONFIG) == KA.CONFIG_SHA256
    m = json.loads(KA.MANIFEST.read_text())
    assert m["config_sha256"] == KA.CONFIG_SHA256
    assert KA._team_hashes(KA.OUR_TEAMS, ".txt") == m["our_teams"]
    assert len(m["our_teams"]) == 72 and m["our_teams"] == m["metamon_teams"]


def _teams(d: Path, suffix: str, n: int = 3, edit: str = "") -> Path:
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"t{i}{suffix}").write_text(f"team {i}{edit if i == 1 else ''}")
    return d


def test_a_moved_team_on_either_side_is_refused(tmp_path):
    ours, theirs = _teams(tmp_path / "o", ".txt"), _teams(tmp_path / "m", ".gen3ou_team")
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    man = tmp_path / "man.json"
    h = KA._team_hashes(ours, ".txt")
    man.write_text(json.dumps({"schema": KA.MANIFEST_SCHEMA, "config_sha256": KA.CONFIG_SHA256,
                               "our_team_dir": str(ours), "our_teams": h, "metamon_team_dir": str(theirs),
                               "metamon_teams": h}))
    probs = KA.check_manifest(man, cfg)
    assert probs == [f"{cfg}: sha256 is not the X26 read's {KA.CONFIG_SHA256[:12]}"]
    (theirs / "t1.gen3ou_team").write_text("team 1 EDITED")
    assert any("Metamon's home72 teams differ" in p for p in KA.check_manifest(man, cfg))


def test_the_plan_is_every_final_by_team_set_by_shared_unit_seed():
    finals = [{"label": "a1_s3501", "arm": "A1", "seed": 3501, "path": "/x/a.zip"},
              {"label": "c0_s3501", "arm": "C0", "seed": 3501, "path": "/x/c.zip"}]
    units = KA.plan_units(finals, hash_models=False)
    assert len(units) == 2 * 2 * 7
    assert units[0].name == "a1_s3501__home72__00" and units[3].name == "c0_s3501__away__00"
    assert {u.unit_seed for u in units} == set(range(7)) and {u.opponent for u in units} == {"metamon:Kakuna"}
    with pytest.raises(KA.KakunaError, match="twice"):
        KA.plan_units(finals + finals[:1], hash_models=False)


GOOD = {"status": "OK", "n": 20, "regime_verified_decisions": True, "team_source_asymmetry": False,
        "peer_clean": True, "wins": 11, "ties": 1}


@pytest.mark.parametrize("bad", [{"status": "PARTIAL"}, {"n": 19}, {"regime_verified_decisions": False},
                                 {"team_source_asymmetry": True}, {"peer_clean": False}, {"peer_clean": None}])
def test_a_unit_counts_only_under_every_condition_of_the_rule(bad):
    assert KA.counts(GOOD, 20)
    assert not KA.counts({**GOOD, **bad}, 20)


def test_the_unit_argv_is_rule_5b_wrapped_and_greedy_from_the_root():
    u = KA.Unit(name="n", label="l", arm="A1", seed=3501, teamset="away", unit_seed=4, model="/m.zip",
                model_sha256="")
    a = KA.unit_argv(u, Path("/o"), 9541, py="PY")
    assert a[:6] == ["nice", "-n", "19", "taskset", "-c", "5-7,13-15"] and a[6].endswith("scripts/ops/mem_cap.sh")
    i = a.index("PY")
    tail = a[i:]
    for flag, val in (("--regime", "greedy"), ("--teamset", "away"), ("--team-seed", "4"), ("--seed-base", "4"),
                      ("--device", "cpu"), ("--port", "9541"), ("--opponent", "metamon:Kakuna")):
        assert tail[tail.index(flag) + 1] == val
    env = KA.unit_env({})
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["GEN3AI_ANCHORS_CONFIG"] == str(KA.CONFIG)


def test_the_port_is_a_free_95xx_one():
    p = KA.choose_port()
    assert KA.PORT_RANGE[0] <= p <= KA.PORT_RANGE[1]


def test_the_stop_epoch_must_be_absolute_and_bounded():
    KA.check_stop_epoch(None)
    KA.check_stop_epoch(1_000_100, now=1_000_000)
    for bad in (999_999, 1_000_000, 1_000_000 + 8 * 86400):
        with pytest.raises(KA.KakunaError):
            KA.check_stop_epoch(bad, now=1_000_000)


def test_pooling_counts_only_counted_units_and_feeds_g_k(tmp_path):
    finals = [{"label": "a1_s3501", "arm": "A1", "seed": 3501, "path": "/a"},
              {"label": "a1_s3502", "arm": "A1", "seed": 3502, "path": "/b"},
              {"label": "c0_s3501", "arm": "C0", "seed": 3501, "path": "/c"}]
    units = KA.plan_units(finals, hash_models=False)
    for u in units:
        d = tmp_path / "cells" / u.name
        d.mkdir(parents=True)
        s = dict(GOOD, wins=12 if u.arm == "A1" else 10)
        if u.name == "a1_s3502__away__03":
            s["peer_clean"] = False
        (d / "summary.json").write_text(json.dumps(s))
    pooled = KA.pool(tmp_path, units)
    assert pooled["a1_s3502"]["away"]["units"] == 6 and pooled["a1_s3502"]["away"]["skipped"] == [
        "a1_s3502__away__03"]
    gk = KA.gk_input(pooled, "A1", "C0")
    assert gk["home72"] == {"on": [2 * 7 * 12, 2 * 7 * 20], "off": [7 * 10, 7 * 20]}
    assert gk["away"]["on"] == [13 * 12, 13 * 20]
