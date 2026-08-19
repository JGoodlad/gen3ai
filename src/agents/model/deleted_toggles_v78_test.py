"""The v78 deletion contract: POP the inert keys, REFUSE the ones that named parameters.

`gen3_flag_surface_p1_v1` deletes eight `ModelVersion` fields with the modules behind them (the
zarch family + the seed-pressure pair). Two mechanisms have to agree, and both are pinned here
because they are read at different moments by different code:

  * `_migrate_config` handles the **config JSON**, which drives the version GATE.
  * `sanitize_dead_extractor_kwargs` handles the **zip's pickled `features_extractor_kwargs`**,
    which is what SB3 splats into the extractor constructor when it rebuilds the policy.

A key dropped from one and not the other either fails the gate for the wrong reason or `TypeError`s
inside the constructor. The MIGRATION FLOOR is 76 and these were live at 76/77, so unlike a
pre-floor deletion these branches are genuinely reachable and cannot be left implicit.

**The judged/inert split is the load-bearing part.** `zarch_film != 'off'` and `seed_quantile=True`
named PARAMETERS — a checkpoint recording them holds `state_dict` keys the live extractor has no
home for — so they are REFUSED with a diagnosis (the v75 `opp_belief_latent` precedent). Everything
else only ever sized, initialised or weighted those modules, so it pops silently. Popping a judged
key instead would turn a clear "this checkpoint is from a closed research arm" into an opaque
unexpected-key error deep inside SB3's load.
"""
from __future__ import annotations

import dataclasses

import pytest

from agents.model.model_version import (
    MIGRATION_FLOOR, MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.model.snapshot import sanitize_dead_extractor_kwargs

# Every field the v78 batch removed, and the value a production (OFF) config recorded for it.
_DELETED_OFF = {
    "zarch_film": "off",
    "zarch_dim": 0,
    "zarch_lut": "off",
    "zarch_lut_teams": 0,
    "zarch_recon_coef": 1.0,
    "zarch_vicreg_coef": 0.1,
    "seed_quantile": False,
    "value_seed_vicreg_coef": 0.0,
}

# The two that carried parameters, with a value only a research arm ever recorded.
_JUDGED_ON = [("zarch_film", "heads"), ("seed_quantile", True)]


def _v77_config(**extra) -> dict:
    """A minimal v77 config carrying the OFF-valued deleted keys, as production actually wrote it."""
    data = {
        "config_version": 77, "arch_signature": "gen3_ctx_dedup_v1",
        "species_embedding_dim": 32, "max_species": 400, "move_embedding_dim": 16,
        "max_moves": 400, "item_embedding_dim": 16, "max_items": 600,
        "ability_embedding_dim": 16, "max_abilities": 100, "type_embedding_dim": 16,
        "max_types": 20, "total_dim": 2669, "active_context_dim": 58,
        "role_token_size": 128, "projection_dim": 512, "move_net_hidden": [96, 32],
        "role_encoder_hidden": [256, 128], "net_arch": [512, 512],
    }
    data.update(_DELETED_OFF)
    data.update(extra)
    return data


# --------------------------------------------------------------------------- the config JSON side
def test_an_off_v77_config_is_refused_below_the_floor():
    """gen3_frame_deletion_v1 raised MIGRATION_FLOOR to 90, so a v77 config no longer migrates at
    all — it is refused as pre-generation. The ORIGINAL claim (every deleted key pops, then
    `cls(**data)` succeeds) is now unreachable through this path, and the surviving guarantee is
    the stronger one: such a checkpoint cannot be loaded into the current architecture by any
    route. The dataclass-side assertion below still pins that the fields are gone."""
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config(_v77_config())


def test_the_deleted_fields_are_gone_from_the_dataclass():
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    assert not (fields & set(_DELETED_OFF)), f"still declared: {sorted(fields & set(_DELETED_OFF))}"


@pytest.mark.parametrize("key,on_value", _JUDGED_ON, ids=[k for k, _ in _JUDGED_ON])
def test_a_config_that_recorded_a_deleted_MODULE_is_refused(key, on_value):
    """These built parameters, so a silent pop would load a state_dict nothing can place."""
    with pytest.raises(ModelVersionError) as exc:
        _migrate_config(_v77_config(**{key: on_value}))
    msg = str(exc.value)
    # gen3_frame_deletion_v1: the FLOOR now refuses first, so the diagnosis names the version
    # boundary rather than the individual module. Both messages must still carry `git_hash` —
    # that is the part which makes a refusal a code-version boundary instead of data loss, and
    # it is the assertion worth keeping whichever guard fires.
    assert "git_hash" in msg, (
        "a refusal must say how to READ the checkpoint anyway (its own metadata.json git_hash) — "
        "otherwise it reads as data loss rather than a code-version boundary.")


def test_the_migration_floor_did_not_move():
    """This tripwire fired exactly as designed, and the note is the point.

    It was written when v78 was a POST-floor deletion (OFF byte-identical ⇒ no ARCH_SIGNATURE bump,
    no floor raise) and it said: "If the floor ever reaches 78 these branches become dead code and
    should be deleted." gen3_frame_deletion_v1 bumped the signature, which the floor contract
    requires be matched by a floor raise IN THE SAME COMMIT — so the floor went 76 → 90,
    gen3_event_semantics_v1 took it → 91 the same way, and gen3_critic_route_wave_v1 → 96. Every
    v77–v95 migration branch was thereby unreachable.

    FOLLOW-UP: DISCHARGED. Those dead branches are DELETED. `_migrate_config` now holds only the
    version-INDEPENDENT sanitizers (which run at EVERY version, because a stale key TypeErrors in
    `cls(**data)` whatever vintage wrote it — so they are not migration branches and the floor
    does not reach them) plus the genuinely post-floor `if version < 97`. Each deleted branch's
    story survives verbatim in that function's PRE-FLOOR MIGRATION HISTORY comment, so the record
    is not what was traded away. The tests that exercised the branches assert the refusal, so
    nothing claims to cover a branch it cannot reach."""
    assert MIGRATION_FLOOR == 96
    assert MODEL_CONFIG_VERSION >= 91


# ------------------------------------------------------------------------ the pickled-kwargs side
def test_off_extractor_kwargs_are_stripped():
    fek = {"damage_op": True, "zarch_film": "off", "zarch_dim": 0, "zarch_lut": "off",
           "zarch_lut_init_std": 1.0, "zarch_lut_rosters": None, "seed_quantile": False}
    assert sanitize_dead_extractor_kwargs(fek) is True
    assert fek == {"damage_op": True}, f"leftovers: {sorted(set(fek) - {'damage_op'})}"


def test_a_clean_kwargs_dict_is_untouched():
    fek = {"damage_op": True, "move_latent": True}
    assert sanitize_dead_extractor_kwargs(fek) is False
    assert fek == {"damage_op": True, "move_latent": True}


@pytest.mark.parametrize("key,on_value", _JUDGED_ON, ids=[k for k, _ in _JUDGED_ON])
def test_extractor_kwargs_that_built_a_deleted_module_are_refused(key, on_value):
    with pytest.raises(ModelVersionError) as exc:
        sanitize_dead_extractor_kwargs({"damage_op": True, key: on_value})
    assert key in str(exc.value)


def test_a_string_mode_at_its_dead_value_is_not_refused_by_truthiness():
    """THE trap this pass had to avoid: `bool("off")` is True.

    The judged loop predates v78 and compared with `bool(recorded) is not supported`, which is right
    for a bool and catastrophically wrong for a mode STRING — it would have refused every OFF
    production config, i.e. every checkpoint that exists. Pinned rather than trusted.
    """
    fek = {"zarch_film": "off", "zarch_lut": "off"}
    assert sanitize_dead_extractor_kwargs(fek) is True
    assert fek == {}
