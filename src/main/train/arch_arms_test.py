"""NAMED ARMS (`main.train.arch_arms`, gen3_static_recovery_v1): `--arch <arm>` is production's ARCH surface plus a
DECLARED overlay, on production's recipe, checked by the same guard as `--arch production`.

Since the END STATE's adoption (2026-10-12, the closing test's PASS) production IS the end state, and `--arch endstate`
is an ALIAS (its overlay equals the mirror, so it differs from plain production in NO key). `--arch static_recovery`
(the closing test's bisection rung C1, defined by OFF values) was DELETED by the legacy deletion's U0
(`designs/ops/legacy_deletion_manifest_2026-10-12.md`, D-LD3) and is refused WITH its reason.

What a revert breaks:

* every arm's overlay key is an ARCH-surface registry row the guard compares (fails if an arm sets a key the guard
  never reads);
* `--arch endstate` applies production + the overlay as if typed and passes `main.checkargs` WITHOUT consent;
* a typed flag that drifts from the ARM is refused, naming the key and the ARM as the declared surface (fails if a
  named arm switched the guard off, or the label reads `production:`);
* the RECIPE is production's, a restart of an arm launch restores it, and the provenance tag names the arm;
* `--arch static_recovery` is refused at PARSE time with the deletion reason, never a bare "invalid choice" (fails if
  the arm is re-added or the `arch_value` converter is dropped).

Run: python -m pytest src/main/train/arch_arms_test.py -q
"""
import types

import pytest

from agents.model.flag_registry import BY_NAME, Tier, arch_surface_flags
from main.train import arch_surface
from main.train.arch_arms import DELETED_ARMS, NAMED_ARMS, arm_overlay
from main.train.arch_surface_test import _namespace, _report
from main.train.parser import build_parser

ARGV = ["--steps", "15000000", "--device", "cuda"]
STATIC_RECOVERY = {"token_encoding": "static", "mon_hazard_cost": "on", "move_actor_state": "on", "trunk_layers": 3,
                   "switch_hazard_cost": "on", "eot_residual": "on"}


def test_every_arms_overlay_is_an_arch_surface_row_and_a_deleted_arm_is_named():
    surface = {f.name for f in arch_surface_flags()}
    for name in NAMED_ARMS:
        for key in arm_overlay(name):
            assert key in surface and BY_NAME[key].tier is Tier.CLI, (name, key)
    with pytest.raises(KeyError, match="endstate"):
        arm_overlay("no_such_arm")
    with pytest.raises(KeyError, match="DELETED"):
        arm_overlay("static_recovery")


def test_the_static_recovery_arm_is_refused_at_parse_time_with_its_reason(capsys):
    """Legacy deletion U0 (G1, D-LD3): the arm was defined by OFF values the deletion removes."""
    assert "static_recovery" in DELETED_ARMS and "static_recovery" not in NAMED_ARMS
    with pytest.raises(SystemExit):
        build_parser().parse_args(ARGV + ["--arch", "static_recovery"])
    err = capsys.readouterr().err
    assert "--arch static_recovery was DELETED" in err and "pre-legacy-deletion" in err, err


def test_a_drift_from_the_arm_is_refused_naming_the_key():
    _ns, rep = _report(ARGV + ["--arch", "endstate", "--trunk-layers", "2"])
    assert rep.refuses
    assert [(d.name, d.resolved, d.production) for d in rep.diffs] == [("trunk_layers", 2, 3)]
    _ns, ok = _report(ARGV + ["--arch", "endstate", "--trunk-layers", "2", arch_surface.ALLOW_FLAG])
    assert not ok.refuses


def test_the_drift_lines_name_the_arm_not_production_as_the_declared_surface():
    """Each per-key line prints the value the argv is judged against BESIDE a label naming whose it is. Under an arm
    that value is the ARM's (`trunk_layers` 3), which the production mirror (2) does not say, so the line must not
    read `production: 3`. Fails on revert of the label (the old line was hardcoded `production:`)."""
    _ns, rep = _report(ARGV + ["--arch", "endstate", "--trunk-layers", "2"])
    lines = arch_surface.report_lines(rep)
    key_lines = [ln for ln in lines if ln.lstrip().startswith("trunk_layers ")]
    assert len(key_lines) == 1, lines
    assert "arm 'endstate': 3" in key_lines[0] and "production:" not in key_lines[0], key_lines[0]
    assert [d.against for d in rep.diffs] == ["arm 'endstate'"]
    text = "\n".join(lines)
    assert "not the arm 'endstate''s" in text and "pass `--arch endstate`" in text, text
    assert "not production's" not in text
    # the same key judged against PLAIN production keeps its label (production's value there is 3)
    ns_prod, rep_prod = _report(ARGV + ["--arch", "production", "--trunk-layers", "2"])
    prod_line = [ln for ln in arch_surface.report_lines(rep_prod) if ln.lstrip().startswith("trunk_layers ")]
    assert len(prod_line) == 1 and "production: 3" in prod_line[0], prod_line
    assert "arm" not in prod_line[0].split("production:")[0].split("(this argv")[-1], prod_line


def test_a_fork_note_names_the_arm_it_diffed_against():
    _ns, rep = _report(ARGV + ["--arch", "endstate", "--trunk-layers", "2"])
    forked = rep._replace(fresh=False)
    text = "\n".join(arch_surface.report_lines(forked))
    assert "The diff above is against the ARM 'endstate', not against the parent." in text, text


def test_checkargs_accepts_the_arm(capsys):
    from main.checkargs import main as checkargs_main
    rc = checkargs_main(["--argv", " ".join(ARGV + ["--arch", "endstate"])])
    out = capsys.readouterr().out
    assert rc == 0, out[-2000:]
    assert "every ARCH-surface key matches the production mirror + the arm 'endstate'" in out
    assert "every RECIPE knob matches recipe.fresh" in out


def test_the_arm_trains_on_the_production_recipe_and_a_restart_restores_it(tmp_path):
    import json

    from main.train import recipe_surface
    arm = _namespace(ARGV + ["--arch", "endstate"])
    prod = _namespace(ARGV + ["--arch", "production"])
    rows = [r.dest for r in recipe_surface.ROWS]
    assert rows and all(getattr(arm, d, None) == getattr(prod, d, None) for d in rows), \
        [(d, getattr(arm, d, None), getattr(prod, d, None)) for d in rows if getattr(arm, d, None) != getattr(prod, d, None)]
    assert arm.recipe_source == prod.recipe_source
    # A restart restores the arm's recipe from the run's own record (every --model launch does, since 2026-10-14;
    # it no longer keys on the umbrella in `original_command`).
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "original_command": "train.py --steps 1 --arch endstate",
        "cli_args": {d: getattr(arm, d, None) for d in rows}}))
    restart = build_parser().parse_args(["--steps", "1", "--model", str(run / "checkpoints" / "c.zip"),
                                         "--run-dir", str(run)])
    got = {d: v for d, v, _s in recipe_surface.inherit_recorded_recipe(
        restart, str(run), types.SimpleNamespace(vf_coef=arm.vf_coef))}
    assert got and all(getattr(restart, d) == getattr(arm, d) for d in got), got
    assert restart.n_envs == arm.n_envs and restart.self_play == arm.self_play


# gen3_endstate_facts_v1: the END-STATE arm (owner 2026-10-09: "run all of them speculatively together, then bisect").
ENDSTATE = {**STATIC_RECOVERY, "move_resolution": "on", "speed_physics": "on", "value_threat_inject": False,
            "op_reduction": "principled", "obs_facts": "v1", "move_resolution_facts": "full", "status_facts": "exact",
            "ko_ramp": "exact", "drop_progress_clock": "on", "g_ledger": "eot",
            # gen3_probe_facts_v1 (v153): the probe battery's two facts
            "effective_stats": "on", "move_target_state": "on",
            # gen3_move_set_closure_v1 (v154; owner 2026-10-10 "Do A", a pre-data amendment of the closing test)
            "move_set_closure": "on"}


def test_the_endstate_arm_is_the_six_recovery_levers_plus_the_bundle_plus_every_fact_lever():
    """Fails if a lever is dropped from the arm, or if it sets a key the guard never reads."""
    assert arm_overlay("endstate") == ENDSTATE
    assert set(STATIC_RECOVERY.items()) <= set(arm_overlay("endstate").items())


def test_the_endstate_arm_is_an_alias_of_the_adopted_production(capsys):
    """Since the 2026-10-12 adoption the mirror carries every endstate key at the arm's value: the arm launches
    without consent and differs from PLAIN production in NO key (fails if the mirror and the arm drift apart, or if
    the adoption is reverted in one of the two places only)."""
    ns, rep = _report(ARGV + ["--arch", "endstate"])
    assert not rep.refuses and not rep.diffs, "\n".join(arch_surface.report_lines(rep))
    for key, value in ENDSTATE.items():
        assert getattr(ns, BY_NAME[key].arg) == value, key
    plain = arch_surface.diff_against_production(ns)
    assert plain == [], [d.line() for d in plain]
    ns_prod, rep_prod = _report(ARGV + ["--arch", "production"])
    assert not rep_prod.refuses and not rep_prod.diffs
    for key, value in ENDSTATE.items():
        assert getattr(ns_prod, BY_NAME[key].arg) == value, key
    assert rep.source_tag.startswith("endstate@production_config@")
    _ns, drift = _report(ARGV + ["--arch", "endstate", "--ko-ramp", "ramp"])
    assert drift.refuses and [(d.name, d.resolved, d.production) for d in drift.diffs] == [("ko_ramp", "ramp", "exact")]
    from main.checkargs import main as checkargs_main
    rc = checkargs_main(["--argv", " ".join(ARGV + ["--arch", "endstate"])])
    out = capsys.readouterr().out
    assert rc == 0, out[-2000:]
    assert "every ARCH-surface key matches the production mirror + the arm 'endstate'" in out
