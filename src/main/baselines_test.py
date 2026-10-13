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
    """The END STATE's adoption (2026-10-12): `production` names a closing-test end-state final, the mirror is the
    pre-adoption mirror + the `--arch endstate` overlay, and the ONLY declared overrides are the five detached
    ride-along READOUT keys the closing test typed on (production keeps them off). Pinned by NAME so the block
    cannot grow or shrink silently; the pre-adoption entry survives as `production_pre_endstate`."""
    b = _prod()
    assert b.run.startswith("rb_es_end2_s") and b.checkpoint == "final_model.zip", (b.run, b.checkpoint)
    assert not b.era_checkout_only, "the adopted production loads at HEAD (math-equal closed-form KO only)"
    assert b.config_overrides == {"ridealong_adv": 0, "ridealong_ensemble": 0, "ridealong_opp": 0,
                                  "ridealong_rnd": False, "ridealong_rnd_variants": "off"}, b.config_overrides
    assert not b.pending, "the old `pending` switch (ai_v12_02) was resolved by the adoption"
    from main.train.arch_arms import arm_overlay
    mirror = baselines.production_config()
    for k, v in arm_overlay("endstate").items():
        assert mirror[k] == v, (k, mirror[k], v)
    legacy = baselines.get("production_pre_endstate")
    assert legacy.run == "ai_v9_21_gen17_pfspoff_0820" and legacy.era_checkout_only


def test_run_level_record_keys_are_not_a_key_set_delta_but_other_run_only_keys_are():
    """`_RUN_RECORD_ONLY_KEYS` (the adoption, 2026-10-12): a run's launch provenance / eval / fork records never sit
    in the mirror, so they are not drift — but ONLY those: any other run-only key still fails without a declared
    migration, and a record key the mirror DOES carry is still value-compared."""
    b = _fake({}, mirror_version=None)
    run = {"arch_source": "endstate@x", "policy_gae_lambda": 0.8, "fork_fraction": 0.0, "trunk_layers": 3}
    mirror = {"trunk_layers": 3}
    assert baselines.compare_production(run, mirror, b) == []
    msgs = baselines.compare_production({**run, "token_encoding": "static"}, mirror, b)
    assert any("KEYS" in m and "token_encoding" in m for m in msgs), msgs
    msgs = baselines.compare_production(run, {**mirror, "policy_gae_lambda": 0.95}, b)
    assert any("policy_gae_lambda" in m for m in msgs), msgs


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


def test_set_can_replace_overrides_drop_pending_and_set_the_era_flag(tmp_path, monkeypatch):
    """`set --config-overrides / --drop-pending / --era-checkout-only` (the adoption's baseline move, 2026-10-12): the
    carried-over construction fields can be REPLACED when the entry moves to a run that records them natively
    (else the validator rightly calls them stale), and an entry can be marked loadable again."""
    from main import baselines as cli
    run = tmp_path / "r"
    run.mkdir()
    (run / "final_model.zip").write_bytes(b"x")
    (run / "model_config.json").write_text(json.dumps({"config_version": 154, "arch_signature": "sig"}))
    (run / "metadata.json").write_text(json.dumps({"git_hash": "abc"}))
    monkeypatch.setattr(cli, "main_models_dir", lambda: tmp_path)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"schema": "gen3_baselines_registry_v1", "baselines": {"p": {
        "kind": "checkpoint", "run": "r", "checkpoint": "final_model.zip", "commit": "abc", "config_version": 154,
        "arch_signature": "sig", "sha256": "0" * 64, "purpose": "p", "set_on": "2026-10-12", "set_by": "t",
        "era_checkout_only": True, "config_overrides": {"critic": "winprob"}, "pending": {"candidate": "x"}}}}))
    kept = cli.build_entry("p", "r/final_model.zip", reason="R", purpose=None, notes=None, kind=None, set_on=None,
                           registry=str(reg))
    assert kept["config_overrides"] == {"critic": "winprob"} and kept["pending"] and kept["era_checkout_only"]
    moved = cli.build_entry("p", "r/final_model.zip", reason="R", purpose=None, notes=None, kind=None, set_on=None,
                            registry=str(reg), era_checkout_only=False,
                            config_overrides={"ridealong_ensemble": 0}, drop_pending=True)
    assert moved["config_overrides"] == {"ridealong_ensemble": 0}
    assert "pending" not in moved and "era_checkout_only" not in moved
    cleared = cli.build_entry("p", "r/final_model.zip", reason="R", purpose=None, notes=None, kind=None,
                              set_on=None, registry=str(reg), config_overrides={})
    assert "config_overrides" not in cleared


def test_every_help_string_renders():
    """The `%o` class: an unescaped `%` in a help string raises only when help is FORMATTED."""
    from main import baselines as cli
    cli.build_parser().format_help()


def test_check_accepts_load_and_reports_a_pre_generation_entry_without_failing(capsys,
                                                                               monkeypatch):
    """`check --load` must SAY that an era-walled entry does not load, and still exit 0.

    A declared `era_checkout_only` node is the era wall, not drift — turning it into an error
    would make `check` permanently red and train everyone to ignore it. An entry that is NOT
    declared is a different thing, and `validate()` already fails on that.
    """
    from main import baselines as cli

    def _fake_load(name, registry=None, **kw):
        if name == "v8_line":
            raise baselines.BaselineLoadError("pre-gen … FIX: use the era checkout",
                                              name=name, reason="pre_generation",
                                              era="v45/old", commit="b13b30b2")
        return object()

    monkeypatch.setattr(baselines, "load", _fake_load)
    monkeypatch.setattr("main.baselines.main_models_dir", lambda: "/tmp/models")
    findings = cli.loadability_findings(None)
    by_name = {f.name: f for f in findings}
    assert by_name["v8_line"].level == "warn" and "pre_generation" in by_name["v8_line"].message
    assert by_name["production"].level == "ok" and "LOADS at HEAD" in by_name["production"].message
    assert cli.build_parser().parse_args(["check", "--load"]).load is True
