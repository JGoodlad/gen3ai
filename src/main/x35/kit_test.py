"""X35's launch kit (``main.x35.kit``): every argv it builds from a call record EQUALS the staging's validated
stand-in (``checkargs`` / ``launcher --dry-run`` at 9ebda190) flag for flag, the two-phase records refuse what they
must, and the disk pre-check's arithmetic. Synthetic files only; nothing launches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import pytest

from main.x35 import kit as K
from utils.paths import repo_path

STAGE = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")
C = "models/rb_x26_s2601/checkpoints"
ARCHIVE_STEPS = {20: 20000043, 40: 40000045, 60: 60000135, 80: 80000111, 100: 100000037, 120: 120000055,
                 140: 140000062, 160: 160000085}


def flags(tokens: List[str]) -> Dict[str, Union[str, bool]]:
    """argparse's view of an argv: flag -> value (a bare flag -> True; a repeated flag -> its last value)."""
    out: Dict[str, Union[str, bool]] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        assert t.startswith("--"), tokens
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[t], i = tokens[i + 1], i + 2
        else:
            out[t], i = True, i + 1
    return out


def standin_call() -> Dict:
    """The stand-in P* (180,000,055) and its archive, in the frozen R file's shape (refpop.freeze)."""
    arch = [{"k": k, "label": f"x26_{n:03d}m", "path": f"{C}/checkpoint_{ARCHIVE_STEPS[n]}_steps.zip",
             "nominal": n * 1_000_000, "step": ARCHIVE_STEPS[n], "sha256": f"sha{n}"}
            for k, n in enumerate(sorted(ARCHIVE_STEPS), start=1)]
    return {"side": "production", "archive": arch,
            "pstar": {"label": "x26_pstar", "path": f"{C}/checkpoint_180000055_steps.zip", "step": 180000055,
                      "sha256": "shaP"}}


def standin(name: str) -> List[str]:
    return (STAGE / f"argv_{name}_standin.txt").read_text().split()


@pytest.mark.parametrize("arm,seed,name", [("c0", 3501, "c0_s3501"), ("a1", 3501, "a1_s3501"),
                                           ("a2", 3501, "a2_s3501"), ("a3_mx", 3511, "a3_main_exploiter"),
                                           ("a6_lx", 3521, "a6_league_exploiter")])
def test_every_production_argv_equals_its_validated_stand_in(arm, seed, name):
    assert flags(K.build_argv(arm, seed, standin_call())) == flags(standin(name))


def test_the_toy_argv_equals_its_validated_stand_in():
    call = {"side": "toy", "archive": [],
            "pstar": {"label": "toy_pstar", "path": "models/rb_toy_ent_s4002/final_model.zip", "step": 100081428,
                      "sha256": "shaT"}}
    assert flags(K.build_argv("toy_c0", 3601, call)) == flags(standin("toy_c0"))


def test_a_and_c_share_everything_but_the_lever():
    call = standin_call()
    on, off = flags(K.build_argv("a1", 3502, call)), flags(K.build_argv("c0", 3502, call))
    lever = {"--stable-opponents", "--stable-opponent-selfplay-share"}
    assert set(on) - set(off) == lever
    assert {k: v for k, v in on.items() if k not in lever | {"--run-name"}} == \
        {k: v for k, v in off.items() if k != "--run-name"}


def test_unregistered_seeds_arms_and_sides_are_refused():
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv("a1", 3511, standin_call())
    with pytest.raises(K.KitError, match="not one of"):
        K.build_argv("a7", 3501, standin_call())
    with pytest.raises(K.KitError, match="reads the toy call record"):
        K.build_argv("toy_a1", 3601, standin_call())
    with pytest.raises(K.KitError, match="needs the a3_mx phase-1 record"):
        K.build_argv("a3", 3501, standin_call())


def _exploiter_runs(tmp_path: Path, arm: str, step: int) -> Path:
    md = tmp_path / "models"
    for s in K.ARMS[arm].seeds:
        d = md / K.run_name(arm, s)
        d.mkdir(parents=True)
        (d / "final_model.zip").write_bytes(f"{arm}{s}".encode())
    return md


def test_the_phase_1_record_feeds_phase_2_and_refuses_a_short_final(tmp_path, monkeypatch):
    call = standin_call()
    md = _exploiter_runs(tmp_path, "a3_mx", 0)
    monkeypatch.setattr(K, "_final_step", lambda z: 185000055)
    rec = K.phase1_record("a3_mx", call, str(md), str(tmp_path / "p1.json"))
    assert [e["label"] for e in rec["exploiters"]] == ["mx_s3511", "mx_s3512", "mx_s3513"]
    assert K.verify_phase1(str(tmp_path / "p1.json")) == []
    f = flags(K.build_argv("a3", 3501, call, rec))
    assert f["--stable-opponent-selfplay-share"] == "0.4" and f["--stable-opponent-pfsp"] is True
    assert str(f["--stable-opponents"]).count(":mx_s35") == 3 and f["--run-name"] == "rb_x35_a3mx_s3501"
    with pytest.raises(K.KitError, match="phase-1 record is for a3_mx"):
        K.build_argv("a6", 3501, call, rec)
    with pytest.raises(K.KitError, match="written ONCE"):
        K.phase1_record("a3_mx", call, str(md), str(tmp_path / "p1.json"))
    monkeypatch.setattr(K, "_final_step", lambda z: 185000054)
    with pytest.raises(K.KitError, match="< the registered"):
        K.phase1_record("a3_mx", call, str(md), str(tmp_path / "p1b.json"))
    (md / "rb_x35_mx_s3512" / "final_model.zip").write_bytes(b"swapped")
    assert any("sha256 changed" in p for p in K.verify_phase1(str(tmp_path / "p1.json")))


def test_a_missing_exploiter_final_is_refused(tmp_path):
    md = _exploiter_runs(tmp_path, "a6_lx", 0)
    (md / "rb_x35_lx_s3523" / "final_model.zip").unlink()
    with pytest.raises(K.KitError, match="no final_model.zip"):
        K.phase1_record("a6_lx", standin_call(), str(md), str(tmp_path / "p1.json"))


def test_the_disk_check_counts_the_whole_remaining_batch():
    ok, msg = K.disk_check("/nonexistent", 20.68, thinning=False, existing=[], free_gib=318.0)
    assert not ok and "18 batch-1 fork(s)" in msg
    ok, _ = K.disk_check("/nonexistent", 20.68, thinning=True, existing=[], free_gib=318.0)
    assert ok
    done = [K.run_name(a, s) for a, s in K.BATCH1[:4]]
    ok, msg = K.disk_check("/nonexistent", 20.68, thinning=False, existing=done, free_gib=318.0)
    assert ok and "14 batch-1" in msg
    ok, msg = K.disk_check("/nonexistent", 20.68, thinning=True, existing=[], free_gib=10.0)
    assert not ok and "not even this launch" in msg


def test_describe_names_the_dry_run_expectations():
    d = K.describe("a6_lx", 3522, standin_call())
    assert d["parent_run_dir"] == "models/rb_x26_s2601" and d["delta_text"] == "+5,000,000 steps"
    assert d["run_name"] == "rb_x35_lx_s3522" and json.dumps(d)


# ------------------------------------------------------------------------------- T54's post-spawn pool-seed check
def _seeded(tmp_path: Path, **over) -> Path:
    run = tmp_path / "rb_x35_a1arch_s3501"
    (run / "snapshots").mkdir(parents=True)
    rec = {"schema": 2, "fork_step": 180000055, "snapshots": ["snapshot_176000000.zip", "snapshot_180000055.zip"],
           "dropped_steps": [184000013, 188000021], **over}
    (run / "snapshots" / "pool_seed.json").write_text(json.dumps(rec))
    return run


LINE = ("🌱 [SELFPLAY] [pool] seeded 2 snapshots + metadata from rb_x26_s2601 (win_rate_vs_bots=81.00%) "
        "[files: summary.json] [step filter <= 180000055: DROPPED 2 later snapshot(s) at steps "
        "[184000013, 188000021], pruned ['pfsp_win_rates']]")


def test_the_pool_seed_check_passes_a_filtered_seed_matching_its_startup_line(tmp_path):
    run = _seeded(tmp_path)
    assert K.check_pool_seed("a1", str(run), standin_call(), LINE, 180000055) == ("PASS", [])
    st, _ = K.check_pool_seed("a1", str(run), standin_call(), "", 180000055)
    assert st == K.PENDING
    assert K.check_pool_seed("a1", str(tmp_path / "nothing"), standin_call(), LINE, 180000055)[0] == K.PENDING
    assert K.check_pool_seed("a3_mx", str(tmp_path / "nothing"), standin_call(), "", None) == ("PASS", [])


@pytest.mark.parametrize("over,line,why", [
    ({"fork_step": None}, LINE.replace("step filter <= 180000055: DROPPED 2 later snapshot(s) at steps "
                                       "[184000013, 188000021], pruned ['pfsp_win_rates']",
                                       "step filter OFF: the fork checkpoint's num_timesteps is UNREADABLE"),
     "fork_step is NULL"),
    ({"schema": 1}, LINE, "schema 1"),
    ({"fork_step": 180000054}, LINE, "!= P*'s num_timesteps"),
    ({"snapshots": ["snapshot_176000000.zip", "snapshot_184000013.zip"]}, LINE, "later than fork_step"),
    ({}, LINE.replace("DROPPED 2 later snapshot(s) at steps [184000013, 188000021]",
                      "DROPPED 1 later snapshot(s) at steps [184000013]"), "dropped steps"),
    ({}, LINE.replace("<= 180000055", "<= 170000000"), "step filter 170000000"),
])
def test_the_pool_seed_check_fails_every_t54_hazard(tmp_path, over, line, why):
    st, probs = K.check_pool_seed("c0", str(_seeded(tmp_path, **over)), standin_call(), line, 180000055)
    assert st == "FAIL" and any(why in p for p in probs), probs


def test_a_none_later_seed_passes(tmp_path):
    run = _seeded(tmp_path, dropped_steps=[])
    line = "🌱 [SELFPLAY] [pool] seeded 2 snapshots + metadata from x [files: ] [step filter <= 180000055: none later]"
    assert K.check_pool_seed("a2", str(run), standin_call(), line, 180000055) == ("PASS", [])
