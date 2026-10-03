"""ai_v12 ADVERSARIAL REVIEW (R1) — the INTERSECTIONS between the five build-wave landings.

Every flag below was individually tested by the wave that shipped it. What this file pins is what
no single wave could see: the compositions a launch actually types. It exists because of the M2
lesson (`value_from_dist`) — every flag in a tail had its own test, the INTERSECTION had none, and
an entire critic chain was orphaned for four generations while every suite stayed green.

Of the original five groups only the migration STACK (group 4, below) remains: groups 1 and 3 (the
value-dist atom support, the `train/pbrs_reward_share` rule) left with the distributional head and the
win-prob PBRS (deletion pass L1), group 5 (the Q head x the cf-twin fold) with the counterfactual training
half (L4), and group 2 (CLEAN WORLD x the TIMEOUT terminal — a shaped-composition scale guard whose
`--critic shaped` argv can no longer be typed) with the Python env core (deletion pass U3).

4. **v105 × v106 × v107 STACKED.** The three migrations landed as three commits an hour apart.
   This runs the whole chain on a fabricated v104 config AND on every REAL archived config in the
   models root, and asserts the result is a fully-populated, constructible `ModelVersion`.

Run:
    python -m pytest src/agents/training/ai_v12_intersection_test.py -q
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import dataclasses
import json
import os

import pytest

from agents.model.model_version import ModelVersion
from agents.model.model_version.constants import MODEL_CONFIG_VERSION
from agents.model.model_version.migrations import _migrate_config

# ──────────────────────────────────────────────────────────────────────────────────────────────
# 4. v105 × v106 × v107 STACKED — three migrations, one chain, on configs that really exist
# ──────────────────────────────────────────────────────────────────────────────────────────────

_WAVE_KEYS = {
    # wave A (v105) — its three shaped keys (hand_shaping / pbrs_material / pbrs_belief) left the
    # config at v122 (gen3_shaped_reward_deletion_v1)
    "victory_value": 30.0,
    # wave D (v106)
    "progress_decision_tense": False, "progress_switch_freeze": False,
    # wave C (v107) — its three q_winprob_* keys left the config at v134 (deletion pass L4)
}


def _fabricated_v104() -> dict:
    """A v104 config: every `ModelVersion` field at its default, MINUS this wave's keys,
    stamped one version back. The last config shape that existed before the wave, so it is the
    exact input the chain must handle.

    The required (default-less) fields are the weight-shape block; they are filled with the
    current `ARCH_SIGNATURE`'s own values so the dict is a plausible checkpoint record rather than
    a bag of zeros."""
    from agents.model.model_version.constants import ARCH_SIGNATURE
    filled = {}
    for f in dataclasses.fields(ModelVersion):
        if f.default is not dataclasses.MISSING:
            filled[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:      # pragma: no cover - none today
            filled[f.name] = f.default_factory()
        elif f.name == "arch_signature":
            filled[f.name] = ARCH_SIGNATURE
        elif "hidden" in f.name or f.name == "net_arch":
            filled[f.name] = [128, 128]
        else:
            filled[f.name] = 64
    for k in _WAVE_KEYS:
        filled.pop(k, None)
    filled["config_version"] = 104
    return filled


def _current_generation_config() -> dict:
    """`_fabricated_v104`'s all-defaults record, but WHOLE and stamped at the live version — the
    shape every config at or above the floor has (the floor rise at gen3_event_record_v2 means
    every such config RECORDS each wave key explicitly)."""
    filled = _fabricated_v104()
    filled.update(_WAVE_KEYS)
    filled["config_version"] = MODEL_CONFIG_VERSION
    return filled


def test_the_three_migrations_stack_on_a_v104_config():
    """The v105 / v106 / v107 branches are FLOORED AWAY: gen3_event_record_v2 raised
    MIGRATION_FLOOR to 121 (all three archived verbatim in `_migrate_config`'s v97–v120 history),
    so the fabricated v104 config is pre-generation and is REFUSED with the diagnosis — a test may
    not claim to cover a chain the floor makes unreachable. The surviving property: the wave's
    keys are recorded, at exactly those values, on a FRESH current config built through the
    project's own constructor, and the current chain passes them through."""
    from agents.model.model_version import ModelVersionError
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config(_fabricated_v104())
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    fresh = json.loads(ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}).to_json())
    out = _migrate_config(dict(fresh))
    assert out["config_version"] == MODEL_CONFIG_VERSION >= 121
    for k, want in _WAVE_KEYS.items():
        assert k in fresh, f"a current config does not record {k}"
        assert out[k] == want, (k, out[k], want)


def test_the_chain_leaves_a_v104_config_CONSTRUCTIBLE():
    """A migration that fills a dict but produces something `ModelVersion(**d)` refuses is a
    migration that only looks complete. A v104 config now never reaches `cls(**d)` (the floor
    refuses it first); the property that survives is that a whole current-generation record goes
    through the live chain and constructs."""
    from agents.model.model_version import ModelVersionError
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        ModelVersion(**_migrate_config(_fabricated_v104()))
    mv = ModelVersion(**_migrate_config(_current_generation_config()))
    assert mv.config_version == MODEL_CONFIG_VERSION
    for k, want in _WAVE_KEYS.items():
        assert getattr(mv, k) == want, k


def test_the_chain_runs_on_EVERY_real_archived_config_of_this_generation():
    """The fabricated case shares its author's assumptions; the archive does not. Every
    current-generation `model_config.json` on this box goes through the whole chain and must come
    out with every field present and constructible — and every PRE-floor one must be REFUSED with
    the diagnosis, never half-migrated (the floor moved to 121 at gen3_event_record_v2, so until a
    new-lineage run exists the refusal leg is where the archive's teeth are).

    SKIPS when there is no models archive (another machine, CI) — `main_models_dir` returns None
    there, and a test that silently passes on an empty set is the failure this docstring names."""
    from agents.model.model_version import MIGRATION_FLOOR, ModelVersionError
    from utils.paths import main_models_dir
    root = main_models_dir()
    if root is None:
        pytest.skip("no models archive on this box")
    seen = refused = 0
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "model_config.json")
        if not os.path.isfile(path):
            continue
        try:
            cfg = json.load(open(path))
        except Exception:                                          # noqa: BLE001 - a truncated file
            continue
        if int(cfg.get("config_version", 0)) < MIGRATION_FLOOR:    # pre-generation: refused by design
            with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
                _migrate_config(dict(cfg))
            refused += 1
            continue
        out = _migrate_config(dict(cfg))
        assert out["config_version"] == MODEL_CONFIG_VERSION, name
        missing = [f.name for f in dataclasses.fields(ModelVersion) if f.name not in out]
        assert not missing, f"{name}: {missing}"
        ModelVersion(**out)
        seen += 1
    if seen == 0 and refused == 0:
        pytest.skip("no readable configs in the archive")
