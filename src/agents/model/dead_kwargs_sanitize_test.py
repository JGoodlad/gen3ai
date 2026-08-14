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
"""
import pytest

from agents.model.model_version import ModelVersionError, _migrate_config
from agents.model.snapshot import (
    _DEAD_FEK_INERT,
    _DEAD_FEK_JUDGED,
    sanitize_dead_extractor_kwargs,
)


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


def test_inert_fields_are_popped_regardless_of_value():
    """The v70 fields selected nothing in production, so no value of them is a refusal."""
    for dead in _DEAD_FEK_INERT:
        for value in (True, False, 0, 3):
            fek = {dead: value}
            assert sanitize_dead_extractor_kwargs(fek) is True
            assert fek == {}
