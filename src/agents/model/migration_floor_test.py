"""The migration floor (MIGRATION_FLOOR): pre-generation configs are REFUSED, not migrated.

Every `if version < N` branch in `_migrate_config` with N <= 67 used to produce a config that the
arch_signature gate (run by every consumer immediately after migration) rejected anyway — dead
code. The floor replaces those branches with a loud, well-diagnosed refusal that points at the
checkpoint's own metadata.json git_hash as the way to re-probe it.
"""
import json
import os
import tempfile

import pytest

from agents.model.model_version import (
    ARCH_SIGNATURE,
    MIGRATION_FLOOR,
    MODEL_CONFIG_VERSION,
    SIGNATURE_FIRST_VERSION,
    ModelVersion,
    ModelVersionError,
    _migrate_config,
)


# --------------------------------------------------------------------- the floor refuses pre-v67


@pytest.mark.parametrize("old_version", [2, 66])
def test_pre_floor_config_is_refused_with_the_diagnosis(old_version):
    """A pre-generation config raises ModelVersionError naming the floor, the signature, and the
    metadata.json git_hash re-probe path — never a silent walk of deleted branches."""
    with pytest.raises(ModelVersionError) as exc_info:
        _migrate_config({"config_version": old_version})
    msg = str(exc_info.value)
    assert "PRE-GENERATION" in msg
    assert str(MIGRATION_FLOOR) in msg
    assert ARCH_SIGNATURE in msg
    assert "git_hash" in msg and "metadata.json" in msg


def test_missing_config_version_defaults_to_1_and_is_refused():
    """A config with no config_version at all is v1 (the pre-flag era) — refused at the floor."""
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({})


def test_from_json_file_surfaces_the_floor_error():
    """The public loader path (ModelVersion.from_json_file) must raise the same clear error, not a
    TypeError from cls(**data) on a half-migrated dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model_config.json")
        with open(path, "w") as f:
            json.dump({"config_version": 66, "arch_signature": "gen3_ancient_v1"}, f)
        with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
            ModelVersion.from_json_file(path)


# --------------------------------------------------------------------- v67 exactly does NOT floor


def test_a_minimal_config_at_the_floor_passes_and_reaches_current():
    """config_version == MIGRATION_FLOOR is the first accepted version: whatever post-floor
    branches exist run and stamp the config up to MODEL_CONFIG_VERSION (with the floor at the
    current version there are none, and the config passes through untouched)."""
    out = _migrate_config({"config_version": MIGRATION_FLOOR})
    assert out["config_version"] == MODEL_CONFIG_VERSION


def test_a_full_v67_config_migrates_and_constructs():
    """A CURRENT config wound back to config_version 67 must migrate to current and still build a
    valid ModelVersion — the floor rejects only strictly-older configs."""
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings

    version = current_model_version(load_mappings())
    data = json.loads(version.to_json())
    data["config_version"] = MIGRATION_FLOOR
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    ModelVersion(**result)


# --------------------------------------------------------------- the floor is TIED to the signature


def test_floor_matches_the_signatures_first_stamped_version():
    """MIGRATION_FLOOR must equal the first MODEL_CONFIG_VERSION stamped with the live
    ARCH_SIGNATURE. Bumping ARCH_SIGNATURE without appending its row to SIGNATURE_FIRST_VERSION
    (and raising the floor to that row) fails HERE — the pairing is maintained by hand in
    model_version.py, in the same commit as the signature change."""
    assert ARCH_SIGNATURE in SIGNATURE_FIRST_VERSION, (
        f"ARCH_SIGNATURE {ARCH_SIGNATURE!r} has no SIGNATURE_FIRST_VERSION row — a signature "
        "bump must add its first stamped version and raise MIGRATION_FLOOR in the same commit."
    )
    assert MIGRATION_FLOOR == SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE]


def test_floor_is_within_the_live_version_range():
    assert MIGRATION_FLOOR <= MODEL_CONFIG_VERSION
