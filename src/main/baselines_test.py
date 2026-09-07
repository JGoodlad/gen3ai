"""THE BASELINE REGISTRY's gate — `designs/baselines.json` + `agents.training.baselines`.

Unmarked and fast, deliberately: the whole point of `gen3_baselines_registry_v1` is that a
baseline file groomed away, a re-pointed checkpoint, or a `designs/production_config.json` that
stopped describing its declared construction FAILS THE SUITE rather than being discovered by
whoever next reads a number against it.

The archive-backed half SKIPS through `utils.paths.main_models_dir()` — never a hardcoded path,
which is how four tests in this tree once skipped forever on every machine but one.
"""
from __future__ import annotations

import json
import os

import pytest

from agents.training import baselines
from utils.paths import main_models_dir, models_skip_reason


# --------------------------------------------------------------------------------------------
# The committed registry
# --------------------------------------------------------------------------------------------

def test_the_committed_registry_validates_end_to_end():
    """`python -m main.baselines check`, as a test. Errors fail; warnings are printed."""
    findings = baselines.validate()
    bad = [f.line() for f in findings if f.level == "error"]
    assert not bad, "designs/baselines.json is not valid:\n  " + "\n  ".join(bad)


def test_every_baseline_names_a_ledger_entry_and_a_purpose():
    for name in baselines.names():
        b = baselines.get(name)
        assert b.set_by.strip(), f"{name}: set_by is empty"
        assert b.purpose.strip(), f"{name}: purpose is empty"
        assert b.set_on.strip(), f"{name}: set_on is empty"


def test_every_checkpoint_is_EXPLICIT_never_a_bare_run_dir():
    """The property the whole registry rests on: `gen3_last_snapshot_resolution_v1` cannot move
    what a name points at, because no entry names a run DIRECTORY."""
    for name in baselines.names():
        b = baselines.get(name)
        assert b.checkpoint, f"{name}: empty checkpoint"
        explicit = (b.checkpoint.startswith("@")
                    or b.checkpoint.endswith(".zip")
                    or b.checkpoint.endswith(".json"))
        assert explicit, (f"{name}: checkpoint {b.checkpoint!r} is not explicit — a bare run dir "
                          "resolves through the last-snapshot rungs and would move")


def test_the_seeded_names_are_all_present():
    """A regression on the SET, so a rename or a deletion is caught rather than absorbed."""
    assert set(baselines.names()) >= {
        "production", "v9_long_baseline", "v9_fold_parent", "v8_line", "v8_parent",
        "famine_comparator", "untaught_meter_opponent", "untaught_meter_config"}
    assert "tb_curated" in baselines.list_names()


def test_the_famine_floor_travels_with_its_comparator():
    b = baselines.get("famine_comparator")
    assert b.floor_elo == 38.0
    assert "38" in b.notes, "the machine-readable floor and its prose must not drift"


def test_a_missing_name_names_the_registry_and_every_available_name():
    with pytest.raises(baselines.BaselineError) as exc:
        baselines.get("not_a_baseline")
    msg = str(exc.value)
    assert baselines.REGISTRY_PATH in msg
    for name in baselines.names():
        assert name in msg


def test_a_list_name_is_refused_by_get_and_points_at_get_list():
    with pytest.raises(baselines.BaselineError) as exc:
        baselines.get("tb_curated")
    assert "get_list" in str(exc.value)


def test_a_list_names_baselines_never_run_strings():
    li = baselines.get_list("tb_curated")
    assert li.members
    for m in li.members:
        baselines.get(m)          # raises if it is not a name


def test_describe_works_with_no_archive_at_all(monkeypatch):
    """The provenance line every consumer prints must not need `models/` — it is what a reader
    sees on a fresh clone, in CI, and in a worktree."""
    monkeypatch.setenv("GEN3AI_MODELS_DIR", "/nonexistent-baselines-probe")
    line = baselines.describe("v9_long_baseline")
    assert "v9_long_baseline" in line and "ai_v9_29_rev1_0823" in line and "set 2026-09-06" in line


def test_is_name_separates_a_NAME_from_a_REF():
    assert baselines.is_name("production")
    assert not baselines.is_name("ai_v9_29_rev1_0823")
    assert not baselines.is_name("models/ai_v9_29_rev1_0823/final_model.zip")


# --------------------------------------------------------------------------------------------
# The `production` CONSTRUCTION check — the replacement for the newest-run heuristic
# --------------------------------------------------------------------------------------------

def _prod() -> baselines.Baseline:
    return baselines.get("production")


def _fake(overrides, *, config_version=97, mirror_version=109) -> baselines.Baseline:
    """A minimal `production`-shaped entry, so the unit tests exercise `compare_production` on the
    keys under test rather than on the committed 13-key block (whose absence from a two-key
    synthetic mirror would correctly fire the `does not carry that key` finding every time)."""
    return baselines.Baseline(
        name="production", kind="checkpoint", run="r", checkpoint="final_model.zip", commit="c",
        config_version=config_version, arch_signature="s", purpose="p", set_on="d", set_by="l",
        sha256="h", config_overrides=dict(overrides), config_mirror_version=mirror_version)


def test_production_declares_a_constructed_mirror():
    b = _prod()
    assert b.config_mirror_version == 109
    assert b.config_version == 97, "the SURFACE run's own recorded version"
    assert len(b.config_overrides) == 13, "the 13-key critic block (CHANGELOG 2026-09-06)"
    assert b.config_overrides["critic"] == "winprob"
    assert b.pending.get("candidate") == "ai_v12_02_winprob_critic"


def test_a_surface_key_that_drifts_is_an_ERROR():
    """The half the 2026-09-06 incident destroyed: a mirror built from an argv that reverted 31
    architecture fields to their OFF defaults."""
    b = _fake({})
    run = {"projection_dim": 512, "damage_op": True, "config_version": 97}
    mirror = {"projection_dim": 512, "damage_op": False, "config_version": 109}
    msgs = baselines.compare_production(run, mirror, b)
    assert any("damage_op" in m for m in msgs)
    assert not any("projection_dim" in m for m in msgs)


def test_a_declared_override_is_NOT_reported_as_drift():
    b = _fake({"critic": "winprob", "use_popart": False})
    run = {"critic": "shaped", "use_popart": True}
    mirror = {"critic": "winprob", "use_popart": False}
    assert baselines.compare_production(run, mirror, b) == []


def test_a_STALE_override_is_reported():
    """An override that no longer differs exempts a key from the only check that guards it."""
    b = _fake({"use_popart": False})
    run = {"use_popart": False}
    mirror = {"use_popart": False}
    msgs = baselines.compare_production(run, mirror, b)
    assert any("stale" in m.lower() and "use_popart" in m for m in msgs)


def test_an_override_the_mirror_does_not_carry_is_reported():
    b = _fake({"critic": "winprob", "use_popart": False})
    msgs = baselines.compare_production({}, {"critic": "winprob"}, b)
    assert any("use_popart" in m and "does not carry" in m for m in msgs)


def test_an_override_at_the_WRONG_value_is_reported():
    b = _fake({"victory_value": 1.0})
    run = {"victory_value": 30.0}
    mirror = {"victory_value": 3.0}
    msgs = baselines.compare_production(run, mirror, b)
    assert any("victory_value" in m and "3.0" in m for m in msgs)


def test_a_key_set_delta_is_ALLOWED_under_a_declared_migration_and_DRIFT_without_one():
    b = _fake({})
    run = {"threat_prob_outspeed": False}
    mirror = {"q_winprob_mode": "none"}
    assert not any("KEYS" in m for m in baselines.compare_production(run, mirror, b))

    msgs = baselines.compare_production(run, mirror, _fake({}, mirror_version=None))
    assert any("KEYS" in m for m in msgs)


def test_config_version_is_never_compared():
    """It is exactly what the declared migration MOVES."""
    b = _fake({})
    assert baselines.compare_production({"config_version": 97}, {"config_version": 109}, b) == []


# --------------------------------------------------------------------------------------------
# Archive-backed — SKIPS with no models/
# --------------------------------------------------------------------------------------------

def _models_or_skip():
    d = main_models_dir()
    if d is None:
        pytest.skip(models_skip_reason())
    return str(d)


def test_every_named_file_exists_and_its_sha_matches():
    models = _models_or_skip()
    for name in baselines.names():
        b = baselines.get(name)
        path = os.path.join(models, b.run, b.checkpoint)
        if not os.path.exists(path):
            assert b.era_checkout_only, f"{name}: {path} is gone and the entry is not era-only"
            continue
        assert baselines.sha256_file(path) == b.sha256, f"{name}: sha256 drift on {path}"


def test_the_committed_mirror_matches_the_declared_construction():
    """DRIFT GATE, registry edition — the replacement for
    `arch_tables_test.test_production_config_matches_newest_run`."""
    models = _models_or_skip()
    b = _prod()
    with open(os.path.join(models, b.run, "model_config.json")) as fh:
        run_cfg = json.load(fh)
    from utils.paths import repo_path
    with open(repo_path("designs", "production_config.json")) as fh:
        mirror = json.load(fh)
    msgs = baselines.compare_production(run_cfg, mirror, b)
    assert not msgs, "\n".join(msgs)


def test_a_checkpoint_entry_resolves_through_the_ONE_choke_point_at_an_EXPLICIT_rung():
    """A registry spec must never reach the last-snapshot rungs — that is what makes a name
    stable while its run keeps training."""
    _models_or_skip()
    for name in baselines.names():
        if baselines.get(name).kind != "checkpoint":
            continue
        try:
            r = baselines.resolve(name)
        except baselines.BaselineError:
            assert baselines.get(name).era_checkout_only, f"{name}: does not resolve"
            continue
        assert r.rung in ("explicit_zip", "explicit_step"), f"{name}: rung {r.rung}"


def test_protected_files_names_every_entrys_file():
    prot = baselines.protected_files()
    for name in baselines.names():
        b = baselines.get(name)
        assert b.rel_path in prot[b.run], f"{name}: {b.rel_path} missing from protected_files()"


# --------------------------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------------------------

def test_the_cli_lists_describes_and_checks(capsys):
    from main import baselines as cli
    assert cli.main(["list"]) == 0
    assert "production" in capsys.readouterr().out
    assert cli.main(["describe", "v9_fold_parent"]) == 0
    assert "ai_v9_59_R2ACTION_0827" in capsys.readouterr().out
    assert cli.main(["spec", "untaught_meter_opponent"]) == 0
    assert capsys.readouterr().out.strip().endswith("snapshots/snapshot_000024000000.zip")


def test_the_cli_check_exits_zero_on_the_committed_registry(capsys):
    from main import baselines as cli
    rc = cli.main(["check", "--no-sha", "--quiet"])
    capsys.readouterr()
    assert rc == 0


def test_an_unknown_name_exits_two_rather_than_raising(capsys):
    from main import baselines as cli
    assert cli.main(["show", "nope"]) == 2
    assert "no baseline named" in capsys.readouterr().err


def test_set_REFUSES_a_bare_run_directory():
    from main import baselines as cli
    with pytest.raises(SystemExit) as exc:
        cli.split_ref("ai_v9_29_rev1_0823")
    assert "bare run directory" in str(exc.value)


def test_split_ref_takes_both_explicit_forms_and_strips_a_models_prefix():
    from main import baselines as cli
    assert cli.split_ref("r/final_model.zip") == ("r", "final_model.zip")
    assert cli.split_ref("r@1234") == ("r", "@1234")
    assert cli.split_ref("models/r/checkpoints/c.zip") == ("r", "checkpoints/c.zip")


def test_the_ledger_line_names_the_reason_and_what_it_replaced():
    from main import baselines as cli
    old = baselines.get("v9_long_baseline")
    entry = {"run": "r", "checkpoint": "final_model.zip", "num_timesteps": 7,
             "commit": "abcdef1234", "config_version": 109, "arch_signature": "sig",
             "sha256": "0" * 64, "set_on": "2026-09-07", "set_by": "THE LEDGER TITLE"}
    line = cli.ledger_line("v9_long_baseline", entry, old)
    assert "THE LEDGER TITLE" in line and old.run in line and "v9_long_baseline" in line


def test_every_help_string_renders():
    """The `%o` class: an unescaped `%` in a help string raises only when help is FORMATTED."""
    from main import baselines as cli
    cli.build_parser().format_help()
