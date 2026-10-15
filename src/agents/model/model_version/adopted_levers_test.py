"""THE LEGACY DELETION's refusal rule (`model_version.adopted_levers`, `designs/ops/legacy_deletion_manifest_2026-10-12.md`
U0) — tested with EVERY unit treated as landed, so the rule the units U1-U4 rely on is proven before the first lands.

What a revert breaks:

* the VACUITY rule (`design_x26_readiness.md` §3.5): a TOY config (`rb_toy_ent_*`: no damage operator, no move
  resolution, every op lever recorded at its OFF value) must stay loadable — fails if `lever_active` stops checking
  the prerequisites;
* the seven per-action blocks' CONDITIONAL rule (F-LD-4): an adopted config records them True (inert under move
  resolution) and must load — fails if they become a plain "True is refused" row;
* the adopted surface (the mirror, the E arm, the bf16 runs, X26) loads; the pre-adoption production surface is
  refused on every lever its network built, naming the pin tag;
* the config side (`retired_levers.refuse_structural`) and the zip side (`snapshot.sanitize_dead_extractor_kwargs` ->
  `judge_pickled_kwargs`) answer the same predicate;
* the table agrees with `designs/production_config.json`, the `endstate` overlay and the flag registry, and a unit is
  LANDED exactly when its fields have left `ModelVersion`.

Run: python -m pytest src/agents/model/model_version/adopted_levers_test.py -q
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import io
import json
from typing import Any, Dict

import pytest

from agents.model.model_version import adopted_levers as AL
from agents.model.model_version import retired_levers as RL

ALL_UNITS = frozenset({"U1", "U2", "U3", "U4"})
SEVEN = ("intent_move_cell", "intent_threshold", "intent_conditional", "pair_outcome_cell", "pair_outcome_switch",
         "switch_branch_cell", "conditional_threat_cell")

#: The T-ENT toy's ablation set (`design_x26_readiness.md` §3.2, `rb_toy_ent_s4001`), on CPU for a test build.
TOY_ARGV = (
    "--arch production --allow-nonproduction-arch --steps 1 --device cpu "
    "--unified-moves off --no-damage-op --no-damage-outgoing --damage-matrices off --damage-topk 0 "
    "--move-belief-mode off --no-move-prior-fusion --no-move-latent --opp-belief-aux-coef 0 --opp-belief-cls-k 0 "
    "--opp-intent-coef 0 --no-item-belief --no-spread-belief --no-spread-belief-nature --no-t0-species-prior "
    "--no-species-prior-fusion --entity-topk-seats 0 --no-entity-tail-seats --edge-bias-families off "
    "--no-history-events --no-value-entity-pool --no-value-entity-pool-full --move-resolution off --speed-physics off "
    "--op-reduction max --mon-hazard-cost off --trunk-layers 2 --switch-hazard-cost off --eot-residual off "
    "--move-resolution-facts off --status-facts off --ko-ramp ramp --g-ledger coarse --effective-stats off "
    "--move-set-closure off --no-intent-move-cell --no-intent-threshold --no-intent-conditional "
    "--no-pair-outcome-cell --no-pair-outcome-switch --no-switch-branch-cell --no-conditional-threat-cell "
    "--no-op-drop-renders --no-op-believed-lean --move-belief-coef 0 --move-belief-latent-coef 0 "
    "--spread-belief-coef 0 --item-belief-coef 0 --hp-type-belief-coef 0").split()


def _mirror() -> Dict[str, Any]:
    from main.train.arch_surface import load_production_config
    return dict(load_production_config())


def _legacy_raw() -> Dict[str, Any]:
    """The PRE-ADOPTION production surface (the closing test's P arm) as a recorded config."""
    from main.train.arch_arms import legacy_production_config
    return legacy_production_config(_mirror())


@functools.lru_cache(maxsize=None)
def _toy_args() -> Any:
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    a = parser.parse_args(TOY_ARGV)
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, parser)
    return a


# ------------------------------------------------------------------------------------------------ the table
def test_the_table_is_the_manifests_25_fields_at_the_mirrors_values():
    from main.train.arch_arms import arm_overlay

    fields = [r.field for r in AL.PLANNED]
    assert len(fields) == len(set(fields)) == 25
    overlay = arm_overlay("endstate")
    assert set(overlay) - set(fields) == {"trunk_layers"}, "D-LD6: trunk depth is KEPT, every other key is deleted"
    assert set(fields) - set(overlay) == set(SEVEN)
    mirror = _mirror()
    for r in AL.PLANNED:
        if r.field in mirror:          # a landed unit's field leaves the mirror
            want = mirror[r.field] if r.field not in SEVEN else False
            assert AL._same(want, r.adopted) or r.field in SEVEN, (r.field, mirror[r.field], r.adopted)
        if r.field in overlay:
            assert overlay[r.field] == r.adopted, r.field
    assert {r.unit for r in AL.PLANNED} == ALL_UNITS
    assert {r.field for r in AL.PLANNED if r.unit == "U1"} == {
        "status_facts", "ko_ramp", "g_ledger", "op_reduction", "speed_physics", "eot_residual",
        "switch_hazard_cost", "effective_stats"}


def test_the_frozen_requires_and_off_defaults_match_the_registry_while_the_row_lives():
    from agents.model.flag_registry import BY_NAME

    for r in AL.PLANNED:
        row = BY_NAME.get(r.field)
        if row is None:
            assert r.unit in AL.LANDED_UNITS, f"{r.field}: no registry row, but {r.unit} has not landed"
            continue
        assert tuple(row.requires) == r.requires, r.field
        assert AL._same(row.default, r.off_default) and row.default == r.off_default, r.field


def test_a_unit_is_landed_exactly_when_its_fields_left_model_version():
    from agents.model.model_version.fields import ModelVersionFields

    live = {f.name for f in dataclasses.fields(ModelVersionFields)}
    for r in AL.PLANNED:
        assert (r.field not in live) == (r.unit in AL.LANDED_UNITS), (r.field, r.unit)
    assert set(AL.UNIT_VERSIONS) == set(AL.LANDED_UNITS)
    landed_fields = {r.field for r in AL.landed()}
    assert landed_fields <= set(RL.RETIRED_FIELDS)
    assert {r.field for r in RL.RETIRED if r.last_commit == AL.LAST_COMMIT_PRE_LEGACY_DELETION} == landed_fields


def test_the_pin_is_a_real_commit_the_tag_names():
    assert len(AL.LAST_COMMIT_PRE_LEGACY_DELETION) == 40
    assert AL.PRE_LEGACY_DELETION_TAG == "pre-legacy-deletion"
    text = AL.refusal_text([AL.PLANNED[0]], "x")
    assert "pre-legacy-deletion" in text and AL.LAST_COMMIT_PRE_LEGACY_DELETION[:8] in text and AL.MANIFEST in text


# ------------------------------------------------------------------------------------------------ the predicate
def test_the_adopted_surface_is_inactive_on_every_lever():
    assert AL.active_levers(_mirror(), units=ALL_UNITS) == []


def test_the_seven_blocks_recorded_true_are_inert_under_move_resolution_and_refused_without_it():
    raw = _mirror()
    for f in SEVEN:
        raw[f] = True                          # every adopted checkpoint records them True (F-LD-4)
    assert AL.active_levers(raw, units=ALL_UNITS) == []
    off = dict(raw, move_resolution="off")
    got = {r.field for r in AL.active_levers(off, units=ALL_UNITS)}
    assert set(SEVEN) <= got and "move_resolution" in got
    assert "move_resolution_facts" not in got, "vacuous once move resolution is off (its prerequisite)"


def test_the_pre_adoption_surface_is_refused_on_every_lever_its_network_built():
    got = {r.field for r in AL.active_levers(_legacy_raw(), units=ALL_UNITS)}
    # every lever OFF in the legacy surface, minus the vacuous ones: `move_resolution_facts` (move resolution off) and
    # the two levers that require the static encoding (`token_encoding legacy` is OFF)
    assert got == {r.field for r in AL.PLANNED} - {"move_resolution_facts", "mon_hazard_cost", "move_actor_state"} - (
        set(SEVEN) - {f for f in SEVEN if _legacy_raw().get(f)}), got


def test_an_absent_field_reads_as_the_pre_version_off_value():
    raw = _mirror()
    del raw["status_facts"]                    # a config older than the lever
    assert [r.field for r in AL.active_levers(raw, units=ALL_UNITS)] == ["status_facts"]
    raw = _mirror()
    del raw["value_threat_inject"]             # absent = False = adopted: nothing to refuse
    assert AL.active_levers(raw, units=ALL_UNITS) == []


def test_a_lever_without_its_prerequisites_is_vacuous_whatever_it_recorded():
    raw = dict(_legacy_raw(), damage_op=False)
    got = {r.field for r in AL.active_levers(raw, units=ALL_UNITS)}
    assert not got & {r.field for r in AL.PLANNED if "damage_op" in r.requires}, got
    assert {"obs_facts", "drop_progress_clock", "move_target_state", "token_encoding"} <= got   # prerequisite-free


# ------------------------------------------------------------------------------------------------ the toy (R-T2)
def test_the_toy_t_ent_resolved_config_is_inactive_on_every_lever():
    """The readiness doc's U0 requirement: a pre-deletion T-ENT checkpoint loads at post-deletion HEAD."""
    a = _toy_args()
    assert a.damage_op is False and a.move_resolution == "off" and a.speed_physics == "off"    # the toy as declared
    raw = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    assert AL.active_levers(raw, units=ALL_UNITS) == []


@pytest.fixture(scope="module")
def toy_snapshot(tmp_path_factory):
    from agents.training.rust_eval.recorded_version_gate_test import write_snapshot

    dst = tmp_path_factory.mktemp("toy") / "run_toy"
    model, zp = write_snapshot(dst, copy.deepcopy(_toy_args()), seed=4001)
    return model, zp, dst / "model_config.json"


def test_a_toy_checkpoint_loads_with_every_unit_judged(toy_snapshot, monkeypatch):
    """A REAL toy build: its model_config.json passes the config refusal and its pickled extractor kwargs pass the zip
    judge, with every unit's rows in force; the same judge refuses the pre-adoption surface."""
    model, _zp, cfg = toy_snapshot
    monkeypatch.setattr(AL, "LANDED_UNITS", ALL_UNITS)
    monkeypatch.setattr(AL, "UNIT_VERSIONS", {u: 999 for u in ALL_UNITS})
    monkeypatch.setattr(RL, "RETIRED", RL.RETIRED + RL._adopted_rows())
    raw = json.loads(cfg.read_text())
    assert raw["damage_op"] is False and raw["speed_physics"] == "off"
    RL.refuse_structural(raw)                                    # no raise
    RL.check_no_retired_levers(str(cfg))                         # no raise: a resume / fork is not refused either
    fek = dict(model.policy_kwargs["features_extractor_kwargs"])
    assert fek.get("op_reduction") == "max"
    assert AL.judge_pickled_kwargs(fek) is True and "op_reduction" not in fek and "speed_physics" not in fek
    from agents.model.model_version import ModelVersionError
    with pytest.raises(ModelVersionError, match="pre-legacy-deletion"):
        RL.refuse_structural(_legacy_raw())
    with pytest.raises(ModelVersionError, match="pre-legacy-deletion"):
        AL.judge_pickled_kwargs(_legacy_raw())


def test_the_toy_checkpoint_loads_on_the_h2h_path(toy_snapshot):
    """R-T2's own route (`main.h2h`): the toy loads at HEAD as a player."""
    from main.h2h.arch import _load_host
    from main.h2h.play import resolve_player

    _model, zp, _cfg = toy_snapshot
    host = _load_host(resolve_player(zp))
    assert host.policy.features_extractor.trunk_layers == 2


# ------------------------------------------------------------------------------------------------ real runs
@pytest.mark.parametrize("run,expect_refused", [("rb_es_end2_s2001", False), ("rb_bf_eb_s2001", False),
                                                ("rb_es_prod_s2001", True)])
def test_the_archived_runs_answer_as_the_manifest_says(run, expect_refused):
    """Exit criterion 2: the E arm (v154) and the bf16 A/B's EB arm record the adopted surface and load; the P arm
    (pre-adoption production, v153) is refused. Read-only on `models/`; skipped where the archive is absent."""
    from utils.paths import main_models_dir

    root = main_models_dir()
    path = root / run / "model_config.json" if root is not None else None
    if path is None or not path.exists():
        pytest.skip(f"no archived {run}/model_config.json on this box")
    raw = json.loads(path.read_text())
    assert bool(AL.active_levers(raw, units=ALL_UNITS)) is expect_refused
