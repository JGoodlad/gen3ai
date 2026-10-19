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


def test_equal_counts_of_DIFFERENT_team_content_no_longer_count():
    """F-DV-5: the rule is `main.anchors.results.unit_counts` (undeclared) — a 72-vs-72 unit whose two sides' content
    digests differ used to count on its equal file counts alone."""
    assert KA.counts({**GOOD, "team_content_asymmetry": False}, 20)
    assert not KA.counts({**GOOD, "team_content_asymmetry": True}, 20)


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


# --------------------------------------------------------------------------- D1's unfamiliar panels (X38's PRIMARY H)
def test_the_committed_panel_configs_are_the_pinned_ones_and_pilot_their_panel_on_both_sides():
    from main import heldout_panels as HP

    assert set(KA.PANEL_CELLS) == set(KA.UNFAMILIAR) == set(HP.panel_names(held_out_only=True)) - {"heldout72"}
    for ts, cell in KA.PANEL_CELLS.items():
        assert KA._sha(cell.path) == cell.sha256, ts
        cfg = json.loads(cell.path.read_text())
        assert cfg["opponents"]["metamon"]["agents"]["Kakuna"]["checkpoint"] == 34
        assert cfg["opponents"]["metamon"]["team_sets"][cell.teamset] == f"gen3ai_{ts}"


def _panel_cell(tmp_path: Path, ts: str = "forum72") -> Path:
    """A config naming tmp byte copies of panel ``ts`` on both sides (ours .txt, Metamon's .gen3ou_team)."""
    from main import heldout_panels as HP

    ours, cache = tmp_path / "ours", tmp_path / "cache"
    theirs = cache / "teams" / f"gen3ai_{ts}" / "gen3ou"
    ours.mkdir()
    theirs.mkdir(parents=True)
    for lab, text in HP.panel_team_texts(ts):
        stem = Path(lab).stem
        (ours / f"{stem}.txt").write_text(text + "\n")
        (theirs / f"{stem}.gen3ou_team").write_text(text + "\n\n\n")
    cfg = json.loads(KA.PANEL_CELLS[ts].path.read_text())
    cfg["our_team_sources"]["home"]["path"] = str(ours)
    cfg["opponents"]["metamon"]["cache_dir"] = str(cache)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_a_panel_cell_holds_on_byte_copies_and_refuses_a_moved_team_a_missing_one_or_another_config(tmp_path):
    cfg = _panel_cell(tmp_path)
    sha = KA._sha(cfg)
    assert KA.check_panel_cell("forum72", config=cfg, want_sha256=sha) == []
    assert any("sha256 is not the registered" in p for p in KA.check_panel_cell("forum72", config=cfg))
    theirs = tmp_path / "cache" / "teams" / "gen3ai_forum72" / "gen3ou"
    first = sorted(theirs.iterdir())[0]
    first.write_text("MOVED\n" + first.read_text())
    probs = KA.check_panel_cell("forum72", config=cfg, want_sha256=sha)
    assert len(probs) == 1 and "Metamon's forum72" in probs[0] and "content digest" in probs[0]
    sorted((tmp_path / "ours").iterdir())[0].unlink()
    probs = KA.check_panel_cell("forum72", config=cfg, want_sha256=sha)
    assert any("our forum72" in p and "71 teams, registered 72" in p for p in probs)


def test_the_unfamiliar_plan_is_every_final_by_panel_by_shared_unit_seed():
    finals = [{"label": "x38_s3501", "arm": "X38", "seed": 3501, "path": "/x/a.zip"},
              {"label": "c0_s3501", "arm": "C0", "seed": 3501, "path": "/x/c.zip"}]
    units = KA.plan_units(finals, hash_models=False, teamsets=KA.parse_teamsets("unfamiliar"))
    assert len(units) == 2 * 3 * 7
    assert [u.name for u in units[:6]] == [f"{f}__{ts}__00" for ts in KA.UNFAMILIAR for f in ("x38_s3501",
                                                                                                 "c0_s3501")]
    assert {u.teamset for u in units} == set(KA.UNFAMILIAR) and {u.unit_seed for u in units} == set(range(7))
    assert KA.parse_teamsets("home72,away") == KA.DEFAULT_TEAMSETS
    for bad in ("home72,home72", "home73", ""):
        with pytest.raises(KA.KakunaError):
            KA.parse_teamsets(bad)


def test_a_panel_unit_runs_on_its_panels_config_and_side():
    u = KA.Unit(name="n", label="l", arm="X38", seed=3501, teamset="p72", unit_seed=2, model="/m.zip",
                model_sha256="")
    a = KA.unit_argv(u, Path("/o"), 9541, py="PY")
    assert a[a.index("--teamset") + 1] == "home" and a[a.index("--team-seed") + 1] == "2"
    assert a[a.index("--capture-dir") + 1] == "/o/capture"
    g = KA.Unit(name="n", label="l", arm="A1", seed=3501, teamset="home72", unit_seed=2, model="/m.zip",
                model_sha256="")
    assert "--capture-dir" not in KA.unit_argv(g, Path("/o"), 9541, py="PY")      # G-K's units are unchanged
    assert KA.config_of("p72") == KA.PANEL_CELLS["p72"].path and KA.config_of("away") == KA.CONFIG
    assert KA.unit_env({}, config=KA.config_of("p72"))["GEN3AI_ANCHORS_CONFIG"] == str(KA.PANEL_CELLS["p72"].path)


def test_preflight_checks_a_panel_unit_against_its_panel_and_a_home72_unit_against_the_manifest(monkeypatch):
    from main.live import halt

    monkeypatch.setattr(halt, "read_halt", lambda: None)
    seen = []
    monkeypatch.setattr(KA, "check_panel_cell", lambda ts: seen.append(("panel", ts)) or ["moved"])
    monkeypatch.setattr(KA, "check_manifest", lambda: seen.append(("manifest",)) or [])
    mk = lambda ts: KA.Unit(name="n", label="l", arm="X38", seed=1, teamset=ts, unit_seed=0, model="/m",  # noqa: E731
                            model_sha256="")
    with pytest.raises(KA.KakunaError, match="forum72 panel cell does not hold"):
        KA.preflight(mk("forum72"))
    KA.preflight(mk("home72"))
    assert seen == [("panel", "forum72"), ("manifest",)]


def test_d1h_pools_counted_units_per_seed_final_and_panel(tmp_path):
    finals = [{"label": "x38_s3501", "arm": "X38", "seed": 3501, "path": "/a"},
              {"label": "x38_s3502", "arm": "X38", "seed": 3502, "path": "/b"},
              {"label": "c0_s3501", "arm": "C0", "seed": 3501, "path": "/c"}]
    units = KA.plan_units(finals, hash_models=False, teamsets=KA.UNFAMILIAR)
    for u in units:
        d = tmp_path / "cells" / u.name
        d.mkdir(parents=True)
        s = dict(GOOD, wins=9 if u.arm == "X38" else 8)
        if u.name == "x38_s3502__p72__05":
            s["regime_verified_decisions"] = False
        (d / "summary.json").write_text(json.dumps(s))
    h = KA.d1h_input(KA.pool(tmp_path, units), "X38", "C0")
    assert h["on"]["3501"] == {ts: [63, 140] for ts in KA.UNFAMILIAR}
    assert h["on"]["3502"]["p72"] == [54, 120] and h["units"]["on"]["3502"]["p72"]["skipped"] == [
        "x38_s3502__p72__05"]
    assert h["off"] == {"3501": {ts: [56, 140] for ts in KA.UNFAMILIAR}}
    assert h["ties"]["on"]["3501"]["forum72"] == 7
    pooled = KA.pool(tmp_path, units)
    pooled["x38_dup"] = pooled["x38_s3501"]
    with pytest.raises(KA.KakunaError, match="share seed 3501"):
        KA.d1h_input(pooled, "X38", "C0")
