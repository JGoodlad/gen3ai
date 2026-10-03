"""The v70/v71 dead extractor kwargs must be stripped from the ZIP, not just from the config.

Why this test exists: `_migrate_config` pops the deleted fields from `model_config.json`, but SB3
rebuilds the extractor from `policy_kwargs["features_extractor_kwargs"]` pickled inside the
CHECKPOINT ZIP. Migrating only the config left every saved checkpoint carrying arguments
`Gen3FeaturesExtractor.__init__` no longer accepts, so the whole read surface (prober, ELO, offline
probes, frozen pool opponents, eval workers) AND the training-resume path failed with
`TypeError: got an unexpected keyword argument 'damage_reattend'`.

The load-bearing case is `test_judged_field_with_unsupported_value_refuses`: popping a
`move_belief_prefuse=False` would let a POST-ordering checkpoint load into the PRE-ordering forward
and be quietly wrong forever — the state_dict is byte-identical either way, so no shape check
anywhere would catch it. Sanitising must inherit `_migrate_config`'s refusal, not soften it.

Every test here is parametrized OFF the curated lists, so a new entry is exercised the moment it is
added — which is how the five PRE-FLOOR names measured on 2026-08-17 (the v48 `mask_*_obs` trio, v52
`hp_type_belief_mode`, v66 `spread_belief_nature_marginalize`) got their coverage. Whether the
curated lists are COMPLETE is a question these tests structurally cannot ask; that is
`ctor_kwarg_snapshot_test.py`'s job.
"""
import pytest

from agents.model.model_version import MIGRATION_FLOOR, ModelVersionError, _migrate_config
from agents.model.snapshot import (
    _DEAD_FEK_INERT,
    _DEAD_FEK_JUDGED,
    sanitize_dead_extractor_kwargs,
)

# The names deleted BELOW MIGRATION_FLOOR. They are the reason `_migrate_config` needs no matching
# per-field entry: no config that can still carry them is loadable at all.
_PRE_FLOOR_JUDGED = ("mask_incoming_damage_obs", "mask_active_move_scalars_obs",
                     "mask_move_effects_obs", "hp_type_belief_mode",
                     "spread_belief_nature_marginalize")


def _supported_fek() -> dict:
    """A `features_extractor_kwargs` as a real pre-v70 checkpoint records it."""
    fek = {"d_model": 128, "move_candidate_floor": 0.02}
    fek.update({k: 0 if k == "damage_refine_rounds" else False for k in _DEAD_FEK_INERT})
    fek["move_belief_single_compute"] = True
    fek.update(dict(_DEAD_FEK_JUDGED))
    return fek


def test_every_dead_key_is_stripped():
    fek = _supported_fek()
    assert sanitize_dead_extractor_kwargs(fek) is True
    for dead in list(_DEAD_FEK_INERT) + [k for k, _ in _DEAD_FEK_JUDGED]:
        assert dead not in fek, f"{dead} survived sanitization"
    # Live keys must be untouched — this is a strip, not a filter-to-allowlist.
    assert fek == {"d_model": 128, "move_candidate_floor": 0.02}


def test_clean_kwargs_are_unchanged_and_report_no_change():
    fek = {"d_model": 128}
    assert sanitize_dead_extractor_kwargs(fek) is False
    assert fek == {"d_model": 128}


@pytest.mark.parametrize("field,supported", _DEAD_FEK_JUDGED)
def test_judged_field_with_unsupported_value_refuses(field, supported):
    """A value the surviving forward cannot reproduce must RAISE, never be silently popped."""
    fek = _supported_fek()
    fek[field] = not supported
    with pytest.raises(ModelVersionError, match=field):
        sanitize_dead_extractor_kwargs(fek)


@pytest.mark.parametrize("field,supported", _DEAD_FEK_JUDGED)
def test_zip_sanitizer_agrees_with_migrate_config(field, supported):
    """Pin the two code paths together — they hold the same rule in two places and could drift.

    `_migrate_config` owns the JSON config; `sanitize_dead_extractor_kwargs` owns the zip. Since
    gen3_ctx_dedup_v1 raised MIGRATION_FLOOR past the v70/v71 branches, migration's half of the
    rule is the blanket floor refusal (any config old enough to carry these keys is
    pre-generation); the sanitizer keeps the per-field judgment for the pickled kwargs, which
    carry no config_version of their own.
    """
    bad = {"config_version": 69, field: not supported}
    with pytest.raises(ModelVersionError):
        _migrate_config(dict(bad))
    with pytest.raises(ModelVersionError):
        sanitize_dead_extractor_kwargs({field: not supported})

    # The supported value: migration refuses on AGE (pre-floor), the sanitizer pops cleanly.
    good = {"config_version": 69, field: supported}
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config(dict(good))
    fek = {field: supported}
    assert sanitize_dead_extractor_kwargs(fek) is True
    assert field not in fek


@pytest.mark.parametrize("field", _PRE_FLOOR_JUDGED)
def test_pre_floor_fields_need_no_migrate_config_entry(field):
    """The asymmetry the 2026-08-17 measurement turned up, pinned so nobody re-derives it.

    These five left the constructor at v48/v52/v66 — all below MIGRATION_FLOOR. A `model_config.json`
    old enough to still name one is refused on AGE before any per-field rule could run, so adding
    `_migrate_config` branches for them would be dead code (that is exactly why the executable v2–v66
    branches were deleted when the floor landed). The ZIP is the asymmetric half: its pickled
    `features_extractor_kwargs` carries no `config_version`, so no floor covers it and the curated
    per-field judgment is the ONLY thing standing between it and a bare TypeError.
    """
    assert field in dict(_DEAD_FEK_JUDGED), f"{field} lost its sanitizer entry"
    # Whatever the recorded value, a config carrying it is pre-floor and refused on age alone.
    for version in (1, MIGRATION_FLOOR - 1):
        with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
            _migrate_config({"config_version": version, field: dict(_DEAD_FEK_JUDGED)[field]})
    # And a config AT the floor cannot carry it, because it was deleted long before — so the
    # sanitizer's judgment is unreachable from the config path in either direction.
    assert _migrate_config({"config_version": MIGRATION_FLOOR}).get(field) is None


def test_inert_fields_are_popped_regardless_of_value():
    """The v70 fields selected nothing in production, so no value of them is a refusal."""
    for dead in _DEAD_FEK_INERT:
        for value in (True, False, 0, 3):
            fek = {dead: value}
            assert sanitize_dead_extractor_kwargs(fek) is True
            assert fek == {}


# ---- deletion pass L4 (config v134): the four cf / Q head toggles ------------------------------------
# Every v98+ zip pickles `cf_evidential` / `cf_twin_heads` / `cf_shadow_critic` and every v107+ one
# `q_winprob_mode` into `features_extractor_kwargs`. Each ON value built a module in the state_dict the
# surviving extractor has no home for, so ON is REFUSED and OFF pops. The generic parametrized tests above
# cover whatever `_DEAD_FEK_JUDGED` holds; these name the four with REALISTIC recorded values (a string
# mode's ON value is `"read_only"`, not `not "none"`) and pin that the entries exist at all, so deleting
# one cannot make the generic tests quietly shrink.
_L4_JUDGED = (("cf_evidential", False, True), ("cf_twin_heads", False, True),
              ("cf_shadow_critic", False, True), ("q_winprob_mode", "none", "read_only"))


def test_the_L4_names_are_judged_entries_with_the_OFF_value_as_the_supported_one():
    judged = dict(_DEAD_FEK_JUDGED)
    for name, off, _on in _L4_JUDGED:
        assert name in judged, f"{name} lost its _DEAD_FEK_JUDGED entry — old zips TypeError again"
        assert judged[name] == off, name
        assert name not in _DEAD_FEK_INERT, f"{name} is INERT-listed: its ON value would pop silently"


@pytest.mark.parametrize("name,off,on", _L4_JUDGED)
def test_an_L4_toggle_recorded_ON_in_a_zip_is_refused_and_OFF_is_popped(name, off, on):
    fek = {"d_model": 128, name: on}
    with pytest.raises(ModelVersionError, match=name):
        sanitize_dead_extractor_kwargs(fek)
    assert fek[name] == on, "a refusal must not have consumed the evidence"
    fek = {"d_model": 128, name: off}
    assert sanitize_dead_extractor_kwargs(fek) is True
    assert fek == {"d_model": 128}


@pytest.mark.parametrize("name,off,on", _L4_JUDGED)
def test_an_L4_toggle_in_a_config_at_the_floor_is_refused_ON_and_popped_OFF(name, off, on):
    """Unlike the pre-floor names, these CAN appear in a config at or above MIGRATION_FLOOR (v121+ writers
    recorded them OFF), so `_migrate_config` owns the config half through `retired_levers`."""
    with pytest.raises(ModelVersionError, match=name.replace("_", "[-_]")):
        _migrate_config({"config_version": MIGRATION_FLOOR, name: on})
    out = _migrate_config({"config_version": MIGRATION_FLOOR, name: off})
    assert name not in out
