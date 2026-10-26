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


def test_the_plan_takes_named_unit_seeds_and_refuses_bad_specs():
    finals = [{"label": "x26c_480m", "arm": "X26C", "seed": 2601, "path": "/x/a.zip"}]
    assert KA.parse_unit_seeds("0-19") == tuple(range(20))
    assert KA.parse_unit_seeds("3, 0-1") == (0, 1, 3)
    units = KA.plan_units(finals, hash_models=False, teamsets=KA.UNFAMILIAR,
                          unit_seeds=KA.parse_unit_seeds("0-19"))
    assert len(units) == 20 * 3 and {u.unit_seed for u in units} == set(range(20))
    assert units[-1].name == "x26c_480m__p72__19"
    assert KA.plan_units(finals, hash_models=False) == KA.plan_units(finals, hash_models=False,
                                                                     unit_seeds=KA.UNIT_SEEDS)
    for bad in ("", "0-3,2", "5-1", "x", "-1"):
        with pytest.raises(KA.KakunaError, match="unit seeds"):
            KA.parse_unit_seeds(bad)


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
    assert a[a.index("--max-s") + 1] == str(KA.UNIT_MAX_S) and a.index("--max-s") < i   # the SHORT lane (mem_cap's)
    assert 0 < KA.UNIT_MAX_S <= 600
    tail = a[i:]
    for flag, val in (("--regime", "greedy"), ("--teamset", "away"), ("--team-seed", "4"), ("--seed-base", "4"),
                      ("--device", "cpu"), ("--port", "9541"), ("--opponent", "metamon:Kakuna")):
        assert tail[tail.index(flag) + 1] == val
    env = KA.unit_env({})
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["GEN3AI_ANCHORS_CONFIG"] == str(KA.CONFIG)


def test_the_port_is_a_free_95xx_one():
    p = KA.choose_port()
    assert KA.PORT_RANGE[0] <= p <= KA.PORT_RANGE[1]


def test_two_supervisors_never_claim_the_same_port(tmp_path):
    """D1 look 1's collision: two supervisors both bind-and-released 9540. A held claim excludes the port for every
    other claimant until it is closed; a closed claim frees it again."""
    p1, f1 = KA.claim_port(lock_dir=tmp_path)
    p2, f2 = KA.claim_port(lock_dir=tmp_path)
    try:
        assert p1 != p2 and KA.PORT_RANGE[0] <= min(p1, p2) and max(p1, p2) <= KA.PORT_RANGE[1]
    finally:
        f2.close()
    f1.close()
    import fcntl
    with open(tmp_path / f"{p1}.lock", "w") as fh:   # the closed claim is free again
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)


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

    # every held-out panel but the retired heldout72 and the SEALED ones (forum216b: X54's, never read before X54)
    assert set(KA.PANEL_CELLS) == set(KA.UNFAMILIAR) == \
        set(HP.panel_names(held_out_only=True)) - {"heldout72"} - set(HP.SEALED)
    assert not set(KA.PANEL_CELLS) & set(HP.SEALED)
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


def test_a_load_option_rides_every_unit_of_its_final_and_only_its_own():
    finals = [{"label": "mix", "arm": "pstar", "seed": 2601, "path": "/m.zip", "belief_prior_mix": 0.25},
              {"label": "plain", "arm": "pstar", "seed": 2601, "path": "/m.zip"}]
    units = KA.plan_units(finals, hash_models=False, teamsets=("home72",), unit_seeds=(0, 1))
    by = {u.label: u for u in units}
    assert {u.belief_prior_mix for u in units if u.label == "mix"} == {0.25}
    assert {u.belief_prior_mix for u in units if u.label == "plain"} == {None}
    a = KA.unit_argv(by["mix"], Path("/o"), 9541, py="PY")
    assert a[a.index("--our-belief-prior-mix") + 1] == "0.25"
    assert "--our-belief-prior-mix" not in KA.unit_argv(by["plain"], Path("/o"), 9541, py="PY")
    # a plan written before the field existed still loads (the default is the unchanged model)
    old = {k: v for k, v in by["plain"].__dict__.items() if k != "belief_prior_mix"}
    assert KA.Unit(**old).belief_prior_mix is None


# ------------------------------------------------------------------------------- other Metamon opponents (FINDING 2)
NEW_OPPONENTS = ("metamon:PastaMittens", "metamon:LargeRL", "metamon:LargeIL")


@pytest.mark.parametrize("opponent", NEW_OPPONENTS)
def test_an_opponent_a_cell_lacks_gets_a_derived_config_with_the_registry_pin(tmp_path, opponent):
    """be60f24f FINDING 2: ``main.reads`` could only play Kakuna. An agent the committed cell config lacks is added
    from the registry, its pin validated, and the unit carries the derived file's sha256."""
    from main.anchors.config import load_config
    from main.reads import assemble as AS

    agent = opponent.split(":")[1]
    finals = [{"label": "p", "arm": "pstar", "seed": 2601, "path": "/m.zip"}]
    units = KA.plan_units(finals, hash_models=False, teamsets=("home72", "forum72"), unit_seeds=(0,),
                          opponent=opponent, config_dir=tmp_path / "configs")
    assert {u.opponent for u in units} == {opponent}
    for u in units:
        assert u.config is not None and Path(u.config).parent == tmp_path / "configs"
        cfg = json.loads(Path(u.config).read_text())
        base = json.loads(KA.config_of(u.teamset).read_text())
        assert cfg["opponents"]["metamon"]["agents"][agent] == KA._registry_agent(agent)
        del cfg["opponents"]["metamon"]["agents"][agent]
        assert cfg == base, "the committed cell config, unchanged but for the one agent"
        pin = load_config(Path(u.config)).metamon.pin(agent)
        assert pin.digest == AS.LEGACY_WEIGHTS[agent], "the registry's bytes: the banked weights, the same regime"
        assert KA.check_derived_config(u) == []
        a = KA.unit_argv(u, Path("/o"), 9541, py="PY")
        assert a[a.index("--opponent") + 1] == opponent
    with pytest.raises(KA.KakunaError, match="no config dir"):
        KA.plan_units(finals, hash_models=False, teamsets=("p72",), unit_seeds=(0,), opponent=opponent)


def test_kakuna_and_smallrl_play_the_committed_config_unchanged_and_others_are_refused(tmp_path):
    finals = [{"label": "p", "arm": "pstar", "seed": 2601, "path": "/m.zip"}]
    for opp in ("metamon:Kakuna", "metamon:SmallRL"):
        units = KA.plan_units(finals, hash_models=False, teamsets=("home72", "p72"), unit_seeds=(0,), opponent=opp,
                              config_dir=tmp_path / "configs")
        assert {(u.config, u.config_sha256) for u in units} == {(None, None)}
    assert not (tmp_path / "configs").exists()
    with pytest.raises(KA.KakunaError, match="is none of"):
        KA.plan_units(finals, hash_models=False, teamsets=("p72",), opponent="metamon:Elsewhere")


def test_preflight_refuses_a_derived_config_that_moved_since_the_plan(tmp_path, monkeypatch):
    from main.live import halt

    monkeypatch.setattr(halt, "read_halt", lambda: None)
    monkeypatch.setattr(KA, "check_panel_cell", lambda ts: [])
    finals = [{"label": "p", "arm": "pstar", "seed": 2601, "path": "/m.zip"}]
    (u,) = KA.plan_units(finals, hash_models=False, teamsets=("forum72",), unit_seeds=(0,),
                         opponent="metamon:LargeRL", config_dir=tmp_path)
    KA.preflight(u)
    p = Path(u.config)
    good = p.read_bytes()
    cfg = json.loads(good)
    cfg["opponents"]["metamon"]["agents"]["LargeRL"]["weights"] = {}
    p.write_text(json.dumps(cfg))
    with pytest.raises(KA.KakunaError, match="derived metamon:LargeRL config does not hold.*sha256.*no weight pin"):
        KA.preflight(u)
    cfg = json.loads(good)
    cfg["opponents"]["metamon"]["agents"]["Kakuna"]["checkpoint"] = 28      # the committed part edited
    b = KA._config_bytes(cfg)
    p.write_bytes(b)
    import hashlib

    u2 = KA.Unit(**dict(u.__dict__, config_sha256=hashlib.sha256(b).hexdigest()))
    with pytest.raises(KA.KakunaError, match="not the committed"):
        KA.preflight(u2)


@pytest.mark.integration
@pytest.mark.parametrize("opponent", NEW_OPPONENTS)
def test_a_real_main_anchors_dry_run_accepts_each_derived_config(tmp_path, opponent):
    """``main.anchors`` as a unit launches it (the derived config via ``$GEN3AI_ANCHORS_CONFIG``), dry-run."""
    import subprocess
    import sys

    (u,) = KA.plan_units([{"label": "p", "arm": "pstar", "seed": 2601, "path": "/m.zip"}], hash_models=False,
                         teamsets=("forum72",), unit_seeds=(0,), opponent=opponent, config_dir=tmp_path)
    out = subprocess.run([sys.executable, "-m", "main.anchors", "--model", "", "--opponent", opponent, "--regime",
                          "greedy", "--teamset", KA.config_side(u.teamset), "--games", "2", "--dry-run"],
                         env=KA.unit_env(config=Path(u.config)), capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "--weights-json" in out.stdout and "not pinned" not in out.stdout + out.stderr


@pytest.mark.integration
@pytest.mark.slow  # ship gate: starts the real Metamon peer (torch + metamon import) from a derived config, ~15 s
def test_the_real_peer_started_from_a_derived_config_verifies_the_registry_weights(tmp_path):
    """The peer process itself on a CLOSED local port, from a unit's DERIVED LargeRL config: it must get through the
    pinned-weights step (offline, the registry's revision and sha256s) before it fails to connect. Stopped by its PID."""
    import subprocess
    import time

    from main.anchors.config import load_config
    from main.anchors.peers import MetamonPeer
    from main.anchors.windowed_attention_integration_test import _metamon_or_skip

    _metamon_or_skip()
    (u,) = KA.plan_units([{"label": "p", "arm": "pstar", "seed": 2601, "path": "/m.zip"}], hash_models=False,
                         teamsets=("forum72",), unit_seeds=(0,), opponent="metamon:LargeRL",
                         config_dir=tmp_path / "configs")
    cfg = load_config(Path(u.config))
    plan = MetamonPeer().plan(cfg=cfg.metamon, agent="LargeRL", regime="greedy", teamset=KA.config_side(u.teamset),
                              battle_format="gen3ou", server_uri="ws://localhost:9597/showdown/websocket",
                              username="PinSmokeA", opponent_username="PinSmokeB", role="acceptor", n_games=1,
                              team_seed=1, out_dir=tmp_path)
    log = tmp_path / "peer.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(plan.argv, cwd=plan.cwd, env=plan.env, stdout=fh, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 240
        while time.time() < deadline and proc.poll() is None and "weights pinned" not in log.read_text():
            time.sleep(2)
        text = log.read_text()
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    assert "weights pinned" in text and "sha256-verified" in text, text[-2500:]
    assert "REFUSING" not in text


# ---------------------------------------------------------------- the long-lane fallback (N1-P unit, 2026-10-26)
class FakePopen:
    """A stand-in for ``subprocess.Popen`` in :mod:`main.x35.kakuna`: records each unit command, and ends with the
    next scripted ``(rc, counted)``. A counted end writes the unit's ``summary.json`` (``--out`` of its argv)."""

    script: list = []
    argvs: list = []

    def __init__(self, argv, **_kw):
        self.argv = list(argv)
        self.pid = 4242
        FakePopen.argvs.append(self.argv)

    def wait(self):
        rc, counted = FakePopen.script.pop(0)
        if counted:
            out = Path(self.argv[self.argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps(GOOD))
        return rc


class _Claim:
    def close(self) -> None:
        pass


def install_fake_play(monkeypatch):
    """No game: preflight passes, the port claim is a no-op, ``Popen`` is :class:`FakePopen` (script it via
    ``FakePopen.script``). Shared with ``main.reads.runner_test``."""
    FakePopen.script, FakePopen.argvs = [], []
    monkeypatch.setattr(KA, "preflight", lambda u: None)
    monkeypatch.setattr(KA, "claim_port", lambda *a, **k: (9555, _Claim()))
    monkeypatch.setattr(KA.subprocess, "Popen", FakePopen)
    return FakePopen


@pytest.fixture
def fake_play(monkeypatch):
    return install_fake_play(monkeypatch)


def _unit(name="a__away__00"):
    return KA.Unit(name=name, label="a", arm="A", seed=1, teamset="away", unit_seed=0, model="/m.zip", model_sha256="")


def _events(state):
    evs = (json.loads(x) for x in (state / "units.jsonl").read_text().splitlines())
    return [(e["event"], e.get("short"), e.get("rc")) for e in evs if e["event"] != "pid"]


def test_the_long_unit_argv_is_a_normal_heavy_slot_job_bounded_by_timeout():
    u = _unit()
    short, long_ = KA.unit_argv(u, Path("/o"), 9541, py="PY"), KA.unit_argv(u, Path("/o"), 9541, py="PY", short=False)
    assert "--max-s" in short and "--max-s" not in long_, "a long job declares no short lane"
    i = long_.index("PY")
    assert long_[:7] == short[:7], "rule 5b's nice / taskset / mem_cap.sh wrapping is the same"
    assert long_[7:9] == ["--name", "x35kakuna"] and long_[9:i] == [KA.MEM_GB, "timeout", str(KA.UNIT_LONG_S)]
    assert long_[i:] == short[short.index("PY"):], "the same command, only the lane differs"
    assert 0 < KA.UNIT_LONG_S <= 45 * 60


def test_a_unit_that_overruns_the_short_lane_is_played_once_more_as_a_long_job(tmp_path, fake_play):
    """mem_cap's exit 88 = the short lane killed it past its bound. Before: an uncounted attempt, and a unit that
    overran twice was never counted; now the SAME attempt plays it once as a normal heavy-slot job (FR-3's queue)."""
    u = _unit()
    fake_play.script = [(KA.SHORT_LANE_OVERRUN, False), (0, True)]
    assert KA.run_unit(tmp_path, u) == 0 and KA.is_done(tmp_path, u)
    first, second = fake_play.argvs
    assert "--max-s" in first and "--max-s" not in second and "timeout" in second
    assert _events(tmp_path) == [("start", True, None), ("end", True, 88), ("short_lane_overrun", None, 88),
                                 ("start", False, None), ("end", False, 0)]


def test_only_an_overrun_is_retried_long_and_a_short_success_is_not_played_twice(tmp_path, fake_play):
    ok, hard_fail, over_but_counted = _unit("a__away__01"), _unit("a__away__02"), _unit("a__away__03")
    fake_play.script = [(0, True)]
    assert KA.run_unit(tmp_path, ok) == 0 and len(fake_play.argvs) == 1
    fake_play.script = [(1, False)]                     # a crash, not an overrun: no long retry (the supervisor's
    assert KA.run_unit(tmp_path, hard_fail) == 1        # MAX_ATTEMPTS counts it)
    assert len(fake_play.argvs) == 2 and not KA.is_done(tmp_path, hard_fail)
    fake_play.script = [(KA.SHORT_LANE_OVERRUN, True)]  # killed after its summary landed: counted, nothing to replay
    assert KA.run_unit(tmp_path, over_but_counted) == KA.SHORT_LANE_OVERRUN and len(fake_play.argvs) == 3
    assert not any(e[0] == "short_lane_overrun" for e in _events(tmp_path))


def test_a_unit_that_overran_goes_straight_to_the_long_job_on_its_next_attempt(tmp_path, fake_play):
    """The long retry itself failed (timeout 124): the supervisor's second attempt must not queue it short again."""
    u, other = _unit(), _unit("a__away__01")
    fake_play.script = [(KA.SHORT_LANE_OVERRUN, False), (124, False)]
    assert KA.run_unit(tmp_path, u) == 124 and not KA.is_done(tmp_path, u)
    fake_play.script = [(0, True)]
    assert KA.run_unit(tmp_path, u) == 0 and KA.is_done(tmp_path, u)
    assert len(fake_play.argvs) == 3 and "--max-s" not in fake_play.argvs[2]
    assert sum(e[0] == "short_lane_overrun" for e in _events(tmp_path)) == 1, "one overrun, one record"
    fake_play.script = [(0, True)]
    KA.run_unit(tmp_path, other)
    assert "--max-s" in fake_play.argvs[3], "another unit still starts on the short lane"
