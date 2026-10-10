"""NAMED ARMS (`main.train.arch_arms`, gen3_static_recovery_v1): `--arch static_recovery` is production's ARCH surface
plus a DECLARED overlay, on production's recipe, checked by the same guard as `--arch production`.

What a revert breaks:

* the overlay is the static encoding + every static-recovery lever, each key an ARCH-surface registry row the guard
  compares (fails if a lever is dropped from the arm, or the arm sets a key the guard never reads);
* `--arch static_recovery` applies production + the overlay as if typed, passes `main.checkargs` WITHOUT consent, and
  differs from plain production in EXACTLY the overlay keys (fails if the guard still compares against production, or
  the umbrella forgets the overlay);
* a typed flag that drifts from the ARM is refused, naming the key (fails if a named arm switched the guard off);
* the RECIPE is production's (both arms of a comparison train on one recipe), a restart of an arm launch restores it,
  and the provenance tag names the arm and both contents by hash.

Run: python -m pytest src/main/train/arch_arms_test.py -q
"""
import pytest

from agents.model.flag_registry import BY_NAME, Tier, arch_surface_flags
from main.train import arch_surface
from main.train.arch_arms import NAMED_ARMS, arm_overlay
from main.train.arch_surface_test import _namespace, _report

ARGV = ["--steps", "15000000", "--device", "cuda"]
STATIC_RECOVERY = {"token_encoding": "static", "mon_hazard_cost": "on", "move_actor_state": "on", "trunk_layers": 3,
                   "switch_hazard_cost": "on", "eot_residual": "on"}


def test_the_static_recovery_arm_is_static_plus_every_recovery_lever():
    assert arm_overlay("static_recovery") == STATIC_RECOVERY
    surface = {f.name for f in arch_surface_flags()}
    for name in NAMED_ARMS:
        for key in arm_overlay(name):
            assert key in surface and BY_NAME[key].tier is Tier.CLI, (name, key)
    with pytest.raises(KeyError, match="static_recovery"):
        arm_overlay("no_such_arm")


def test_the_arm_applies_production_plus_its_overlay_and_passes_without_consent():
    ns, rep = _report(ARGV + ["--arch", "static_recovery"])
    assert not rep.refuses and not rep.diffs, "\n".join(arch_surface.report_lines(rep))
    for key, value in STATIC_RECOVERY.items():
        assert getattr(ns, BY_NAME[key].arg) == value, key
    # judged against PLAIN production, the same namespace differs in exactly the overlay
    plain = arch_surface.diff_against_production(ns)
    assert {d.name for d in plain} == set(STATIC_RECOVERY), [d.line() for d in plain]
    assert rep.source_tag.startswith("static_recovery@production_config@") and "+overlay@" in rep.source_tag
    assert ns.arch_source == rep.source_tag
    lines = "\n".join(arch_surface.report_lines(rep))
    assert "arm 'static_recovery'" in lines and "trunk_layers=3" in lines


def test_a_drift_from_the_arm_is_refused_naming_the_key():
    _ns, rep = _report(ARGV + ["--arch", "static_recovery", "--trunk-layers", "2"])
    assert rep.refuses
    assert [(d.name, d.resolved, d.production) for d in rep.diffs] == [("trunk_layers", 2, 3)]
    _ns, ok = _report(ARGV + ["--arch", "static_recovery", "--trunk-layers", "2", arch_surface.ALLOW_FLAG])
    assert not ok.refuses


def test_checkargs_accepts_the_arm(capsys):
    from main.checkargs import main as checkargs_main
    rc = checkargs_main(["--argv", " ".join(ARGV + ["--arch", "static_recovery"])])
    out = capsys.readouterr().out
    assert rc == 0, out[-2000:]
    assert "every ARCH-surface key matches the production mirror + the arm 'static_recovery'" in out
    assert "every RECIPE knob matches recipe.fresh" in out


def test_the_arm_trains_on_the_production_recipe_and_a_restart_restores_it():
    from main.train import recipe_surface
    arm = _namespace(ARGV + ["--arch", "static_recovery"])
    prod = _namespace(ARGV + ["--arch", "production"])
    rows = [r.dest for r in recipe_surface.ROWS]
    assert rows and all(getattr(arm, d, None) == getattr(prod, d, None) for d in rows), \
        [(d, getattr(arm, d, None), getattr(prod, d, None)) for d in rows if getattr(arm, d, None) != getattr(prod, d, None)]
    assert arm.recipe_source == prod.recipe_source
    assert recipe_surface._is_production_launch("train.py --steps 1 --arch static_recovery")
    assert not recipe_surface._is_production_launch("train.py --steps 1")
