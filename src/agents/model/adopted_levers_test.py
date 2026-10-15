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

#: The T-ENT toy's ablation set (`design_x26_readiness.md` §3.2, `rb_toy_ent_s4001`), on CPU for a test build — as it
#: parses at HEAD: U1 deleted eight of its flags (`--speed-physics off --op-reduction max --switch-hazard-cost off
#: --eot-residual off --status-facts off --ko-ramp ramp --g-ledger coarse --effective-stats off`) and U3 ten more
#: (`--move-resolution off --move-resolution-facts off` and the seven `--no-<block>` spellings), which the PINNED toy
#: still types; `PRE_U1_TOY_OFF` / `PRE_U3_TOY_OFF` re-add their recorded values to simulate a checkpoint the pinned toy
#: wrote.
TOY_ARGV = (
    "--arch production --allow-nonproduction-arch --steps 1 --device cpu "
    "--unified-moves off --no-damage-op --no-damage-outgoing --damage-matrices off --damage-topk 0 "
    "--move-belief-mode off --no-move-prior-fusion --no-move-latent --opp-belief-aux-coef 0 --opp-belief-cls-k 0 "
    "--opp-intent-coef 0 --no-item-belief --no-spread-belief --no-spread-belief-nature --no-t0-species-prior "
    "--no-species-prior-fusion --entity-topk-seats 0 --no-entity-tail-seats --edge-bias-families off "
    "--no-history-events --no-value-entity-pool --no-value-entity-pool-full "
    "--mon-hazard-cost off --trunk-layers 2 "
    "--move-set-closure off "
    "--no-op-drop-renders --no-op-believed-lean --move-belief-coef 0 --move-belief-latent-coef 0 "
    "--spread-belief-coef 0 --item-belief-coef 0 --hp-type-belief-coef 0").split()


#: The U1 fields as every PRE-U1 non-adopted config recorded them (the pre-adoption production surface and the pinned
#: toy alike): their OFF values. Frozen here because U1 deleted them from the mirror, the registry and
#: `PRE_ADOPTION_PRODUCTION`.
PRE_U1_TOY_OFF: Dict[str, Any] = {
    "speed_physics": "off", "op_reduction": "max", "switch_hazard_cost": "off", "eot_residual": "off",
    "status_facts": "off", "ko_ramp": "ramp", "g_ledger": "coarse", "effective_stats": "off"}


#: The U3 fields as the PINNED toy recorded them (its argv typed every one OFF; `value_threat_inject` False was the
#: mirror's value after the adoption). Frozen here because U3 deleted them.
PRE_U3_TOY_OFF: Dict[str, Any] = {
    "move_resolution": "off", "move_resolution_facts": "off", "value_threat_inject": False,
    **{f: False for f in SEVEN}}

#: The U3 fields as the PRE-ADOPTION production surface recorded them (the closing test's P arm, v153): no move
#: resolution, the critic's threat injection ON, the seven per-action blocks ON and live.
PRE_U3_LEGACY: Dict[str, Any] = {
    "move_resolution": "off", "move_resolution_facts": "off", "value_threat_inject": True,
    **{f: True for f in SEVEN}}

#: The U3 fields as an ADOPTED pre-U3 config recorded them (the E arm, v154; the bf16 runs, v155; X26's parent, v156):
#: move resolution on with its restored facts, no threat injection, the seven blocks recorded True and INERT.
PRE_U3_ADOPTED: Dict[str, Any] = {
    "move_resolution": "on", "move_resolution_facts": "full", "value_threat_inject": False,
    **{f: True for f in SEVEN}}


def _mirror() -> Dict[str, Any]:
    from main.train.arch_surface import load_production_config
    return dict(load_production_config())


def _legacy_raw() -> Dict[str, Any]:
    """The PRE-ADOPTION production surface (the closing test's P arm, config v153) as a recorded config: the mirror at
    `PRE_ADOPTION_PRODUCTION` + the U1 and U3 fields at their pre-adoption values, stamped v153."""
    from main.train.arch_arms import legacy_production_config
    return dict(legacy_production_config(_mirror()), **PRE_U1_TOY_OFF, **PRE_U3_LEGACY, config_version=153)


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
    landed = {r.field for r in AL.landed()}
    assert not landed & set(overlay), "a landed unit's keys leave the `endstate` overlay (D-LD3)"
    assert set(fields) - set(overlay) == set(SEVEN) | landed
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
    assert "move_resolution" not in raw, "U3 deleted the field from the mirror"
    for f in SEVEN:
        raw[f] = True                          # every adopted checkpoint records them True (F-LD-4)
    assert AL.active_levers(raw, units=ALL_UNITS) == [], "absent move_resolution in a post-U3 config reads ON"
    assert AL.active_levers(dict(raw, move_resolution="on"), units=ALL_UNITS) == []
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
    raw = dict(_mirror(), **PRE_U1_TOY_OFF, **PRE_U3_ADOPTED, config_version=155)
    for f in PRE_U1_TOY_OFF:
        raw[f] = next(r.adopted for r in AL.PLANNED if r.field == f)   # a pre-U1 ADOPTED config (the E arm, v154)
    assert AL.active_levers(raw, units=ALL_UNITS) == []
    del raw["status_facts"]                    # a config older than the lever
    assert [r.field for r in AL.active_levers(raw, units=ALL_UNITS)] == ["status_facts"]
    raw = dict(raw, **PRE_U3_ADOPTED, status_facts="exact")         # the E arm again, every field recorded
    del raw["value_threat_inject"]             # absent = False = adopted: nothing to refuse
    assert AL.active_levers(raw, units=ALL_UNITS) == []
    del raw["move_resolution"]                 # a pre-U3 config without the field read OFF: the seven are LIVE again
    assert {r.field for r in AL.active_levers(raw, units=ALL_UNITS)} == {"move_resolution", *SEVEN}


def test_an_absent_field_in_a_config_written_after_its_unit_reads_as_adopted():
    """U1's landing: a v156 config cannot record a U1 field (that code builds only the adopted value), so an absent one
    reads ADOPTED — reading it as the pre-version OFF value refused every post-U1 checkpoint that built the op (the
    production mirror itself). Fails if `absent_value` stops consulting `UNIT_VERSIONS`."""
    raw = _mirror()
    assert raw["config_version"] >= AL.UNIT_VERSIONS["U1"] and raw["damage_op"] is True
    assert not set(PRE_U1_TOY_OFF) & set(raw), "the mirror no longer records U1's fields"
    assert AL.active_levers(raw) == []
    old = dict(raw, config_version=AL.UNIT_VERSIONS["U1"] - 1)       # the same record claiming to be pre-U1
    # (its absent U3 fields read OFF too: move resolution is refused; its facts are vacuous without it, and the threat
    # injection's / the seven blocks' OFF value is False, which is also adopted)
    assert {r.field for r in AL.active_levers(old)} == set(PRE_U1_TOY_OFF) | {"move_resolution"}
    # U3's landing, the same rule: a v157 config cannot record a U3 field; claiming to be pre-U3 refuses move resolution
    assert raw["config_version"] >= AL.UNIT_VERSIONS["U3"] and not set(PRE_U3_TOY_OFF) & set(raw)
    pre_u3 = dict(raw, config_version=AL.UNIT_VERSIONS["U3"] - 1)
    assert {r.field for r in AL.active_levers(pre_u3)} == {"move_resolution"}
    # the ZIP side carries no version: an absent kwarg is left to the config judge (no refusal, nothing to pop)
    fek = {k: v for k, v in raw.items() if k != "config_version"}
    assert AL.judge_pickled_kwargs(fek) is False


def test_the_production_mirror_loads_through_the_migration():
    """The full config path (`_migrate_config` -> `ModelVersion(**data)`, what `from_json_file` runs) on the post-U1
    mirror: no refusal."""
    mv = _load(_mirror())
    assert mv.config_version >= AL.UNIT_VERSIONS["U1"]


def _load(raw: Dict[str, Any]) -> Any:
    """`ModelVersion.from_json_file`'s body on a dict (the mirror's hand-edited `recipe` block is not a field)."""
    from agents.model.model_version import ModelVersion
    from agents.model.model_version.migrations import _migrate_config
    data = copy.deepcopy(raw)
    data.pop("recipe", None)
    return ModelVersion(**_migrate_config(data))


def test_a_lever_without_its_prerequisites_is_vacuous_whatever_it_recorded():
    raw = dict(_legacy_raw(), damage_op=False)
    got = {r.field for r in AL.active_levers(raw, units=ALL_UNITS)}
    assert not got & {r.field for r in AL.PLANNED if "damage_op" in r.requires}, got
    assert {"obs_facts", "drop_progress_clock", "move_target_state", "token_encoding"} <= got   # prerequisite-free


# ------------------------------------------------------------------------------------------------ the toy (R-T2)
def test_the_toy_t_ent_resolved_config_is_inactive_on_every_lever():
    """The readiness doc's U0 requirement: a pre-deletion T-ENT checkpoint loads at post-deletion HEAD."""
    a = _toy_args()
    assert a.damage_op is False and a.opp_intent_coef == 0.0                                  # the toy as declared
    raw = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    assert AL.active_levers(raw, units=ALL_UNITS) == []
    # the PINNED toy's own record (it still types the U1 and U3 flags at their OFF values): vacuous, never refused
    assert AL.active_levers(dict(raw, **PRE_U1_TOY_OFF, **PRE_U3_TOY_OFF, config_version=155), units=ALL_UNITS) == []


@pytest.fixture(scope="module")
def toy_snapshot(tmp_path_factory):
    from agents.training.rust_eval.recorded_version_gate_test import write_snapshot

    dst = tmp_path_factory.mktemp("toy") / "run_toy"
    model, zp = write_snapshot(dst, copy.deepcopy(_toy_args()), seed=4001)
    return model, zp, dst / "model_config.json"


def test_a_toy_checkpoint_loads_with_every_unit_judged(toy_snapshot, monkeypatch):
    """A REAL toy build: its model_config.json passes the config refusal and its pickled extractor kwargs pass the zip
    judge, with every unit's rows in force — both as HEAD wrote them and as the PINNED toy (`rb_toy_ent_s4001` @
    `3899277a`, which records U1's eight fields at their OFF values) writes them (the VACUITY rule: no damage op, so
    none of U1's levers was built); the same judge refuses the pre-adoption surface."""
    model, _zp, cfg = toy_snapshot
    monkeypatch.setattr(AL, "LANDED_UNITS", ALL_UNITS)
    monkeypatch.setattr(AL, "UNIT_VERSIONS", {u: 999 for u in ALL_UNITS})
    monkeypatch.setattr(RL, "RETIRED", RL.RETIRED + RL._adopted_rows())
    raw = json.loads(cfg.read_text())
    assert raw["damage_op"] is False and "speed_physics" not in raw
    RL.refuse_structural(raw)                                    # no raise
    RL.check_no_retired_levers(str(cfg))                         # no raise: a resume / fork is not refused either
    pinned = dict(raw, **PRE_U1_TOY_OFF, **PRE_U3_TOY_OFF, config_version=155)     # the pinned toy's record
    RL.refuse_structural(pinned)                                 # no raise
    pinned_cfg = cfg.parent / "pinned_model_config.json"
    pinned_cfg.write_text(json.dumps(pinned))
    RL.check_no_retired_levers(str(pinned_cfg))                  # no raise
    _load(pinned)                                                # the full migration: the eight fields pop
    fek = dict(model.policy_kwargs["features_extractor_kwargs"], **PRE_U1_TOY_OFF, **PRE_U3_TOY_OFF)
    assert AL.judge_pickled_kwargs(fek) is True and "op_reduction" not in fek and "speed_physics" not in fek
    assert not set(PRE_U3_TOY_OFF) & set(fek), "the U3 kwargs pop (vacuous: no damage op, no intent)"
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
