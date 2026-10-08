"""THE X5 VERSION BREAK (config v144, ``gen3_x5_version_break_v1``): the one planned checkpoint break
after X5's adoption (``designs/endstate/design_x5_belief_tokens.md`` Decision record 2026-10-07, §3.8).

WHAT IT MEANS FOR A CHECKPOINT. The break raised ``MIGRATION_FLOOR`` to 144, so EVERY earlier checkpoint is
refused by ``_migrate_config``'s pre-floor gate — the blob arm because its belief path is DELETED (part 1),
and a pre-break ``fixed_mass`` one because the break reshapes its weights (part 2 deleted the dead value tower
and the flat pointer's dead scorer bias; part 4 tied the op's per-slot ``out_gain`` to one scalar per (block
region, channel)) and changes its forward (part 5: ``intent_conditional`` reads the op's PRE-gain values), and part 3
appended the OBS-FACTS block to the observation (2761 -> 2845, the ``total_dim`` break). Neither can be reproduced
at HEAD; both RUN PINNED to their own commit.

This module is the ONE home of:

* ``LAST_BLOB_COMMIT`` — the last commit that builds ``belief_tokens='blob'`` (and the last pre-break commit:
  the break's first commit follows it). Every refusal names it; nothing else spells the hash.
* ``pre_break_diagnosis`` — the belief-specific sentence ``_migrate_config``'s pre-floor refusal appends, keyed
  on what the config RECORDS (``belief_tokens`` ``'blob'``, or the key absent below v136 — blob was the only
  past; ``'fixed_mass'`` — a pre-break X5 checkpoint).
* ``refuse_pickled_belief_tokens`` — the ZIP-side twin (``snapshot._DEAD_FEK_JUDGED``'s ``belief_tokens`` row):
  a pickled ``features_extractor_kwargs['belief_tokens']`` other than ``'fixed_mass'`` is refused with the
  same reason, never handed to the constructor (a bare ``TypeError``) and never silently popped.
"""
from __future__ import annotations

from typing import Any, Optional

#: The break's identity (``SIGNATURE_FIRST_VERSION[VERSION_BREAK_SIGNATURE] == VERSION_BREAK_CONFIG``).
VERSION_BREAK_SIGNATURE = "gen3_x5_version_break_v1"
VERSION_BREAK_CONFIG = 144

#: The last commit that builds the blob belief path (`--belief-tokens blob`) — and the last commit BEFORE the
#: version break, so every pre-break checkpoint (blob or fixed_mass) runs pinned at or before it.
LAST_BLOB_COMMIT = "26131c0ce62b2de7c9913d9c84e8c318a0200b19"

#: The first config version that recorded `belief_tokens` (X5 U2). Below it the key is ABSENT and the only
#: possible past is blob (the hypothesis builder did not exist).
BELIEF_TOKENS_FIRST_CONFIG = 136


def recorded_belief_tokens(data: dict) -> Optional[str]:
    """What a RAW ``model_config.json`` dict says its belief representation was: ``'blob'`` /
    ``'fixed_mass'`` / another recorded string, ``'blob'`` when the key is absent below v136 (the only
    possible past), ``None`` when it is absent at or above v136 (no claim can be made)."""
    v = data.get("belief_tokens")
    if v is not None:
        return str(v)
    if int(data.get("config_version", 1)) < BELIEF_TOKENS_FIRST_CONFIG:
        return "blob"
    return None


def blob_deleted_reason() -> str:
    """The one sentence that names the deletion and the fix."""
    return (f"belief_tokens='blob' (the constant hidden-slot belief tokens + the alpha / beta intent "
            f"heads) was DELETED at the X5 version break (config v{VERSION_BREAK_CONFIG}, "
            f"{VERSION_BREAK_SIGNATURE}): X5's hypothesis tokens are the only belief representation this code "
            f"builds. Run it PINNED to its own commit (the git_hash in its metadata.json), or to "
            f"{LAST_BLOB_COMMIT[:12]} ({LAST_BLOB_COMMIT}), the last commit that builds blob; or start a "
            "fresh run.")


def pre_break_fixed_mass_reason() -> str:
    return (f"It records belief_tokens='fixed_mass': a PRE-BREAK X5 checkpoint. The X5 version break "
            f"(config v{VERSION_BREAK_CONFIG}, {VERSION_BREAK_SIGNATURE}) reshaped the weights it trained — part 2 "
            "DELETED the dead SB3 value tower (the extractor's value_pre_norm / value_projection, "
            "mlp_extractor.value_net and value_net: 592,129 parameters no loss read under the win-prob critic) "
            "and the flat opponent pointer's shared scorer bias (flat_intent_head.out.bias), so its state_dict "
            "holds keys this code refuses — and parts 4 and 5 re-wired the op (its per-slot out_gain tied to one "
            "scalar per block region and channel, 138 -> 99 on the production op, so damage_op.out_gain changed "
            "shape; intent_conditional reads the op's PRE-gain values) — and part 3 APPENDED the OBS-FACTS block to the "
            "observation (2761 -> 2845, so its total_dim no longer matches) — so it cannot be reproduced at HEAD. Run it "
            f"PINNED to its own commit (the git_hash in its metadata.json; at the latest {LAST_BLOB_COMMIT[:12]}, "
            "the last pre-break commit), or start a fresh run.")


def pre_break_diagnosis(data: dict) -> Optional[str]:
    """The belief-specific paragraph for a PRE-FLOOR config (``None`` for one older than the X5 era, whose
    pre-generation diagnosis needs nothing more)."""
    version = int(data.get("config_version", 1))
    if version >= VERSION_BREAK_CONFIG:
        return None
    rec = recorded_belief_tokens(data)
    if rec == "fixed_mass":
        return pre_break_fixed_mass_reason()
    if rec == "blob" and version >= 121:      # the generation the break floors (gen3_event_record_v2, v121+)
        how = ("records belief_tokens='blob'" if "belief_tokens" in data
               else f"predates belief_tokens (config v{BELIEF_TOKENS_FIRST_CONFIG}), so it is a blob checkpoint")
        return f"It {how}: " + blob_deleted_reason()
    return None


def refuse_pickled_belief_tokens(fek: dict) -> bool:
    """POP a pickled ``belief_tokens`` kwarg (``'fixed_mass'``, the one value whose forward survives) or
    REFUSE any other value with the deletion reason. True when it changed ``fek``."""
    if "belief_tokens" not in fek:
        return False
    rec: Any = fek["belief_tokens"]
    if rec != "fixed_mass":
        from agents.model.model_version.constants import ModelVersionError
        raise ModelVersionError(f"this checkpoint's pickled extractor kwargs record belief_tokens={rec!r}. "
                                + blob_deleted_reason())
    fek.pop("belief_tokens")
    return True


class PreBreakCheckpointError(ValueError):
    """A RESUME / FORK of a checkpoint whose ``model_config.json`` predates ``MIGRATION_FLOOR`` (every
    pre-break checkpoint — blob or fixed_mass — and anything older). Carries the migration chain's own
    refusal text (with the belief-specific reason) and the commit a pinned run must not be later than."""

    def __init__(self, message: str, *, config_path: str, config_version: int,
                 last_commit: Optional[str]) -> None:
        super().__init__(message)
        self.config_path = config_path
        self.config_version = config_version
        self.last_commit = last_commit


def check_post_break(config_path: Optional[str]) -> None:
    """Raise :class:`PreBreakCheckpointError` when the RAW config at ``config_path`` is below the migration
    floor — read BEFORE any loader runs, so the trainer and ``main.checkargs`` refuse with the reason (the
    pinned fix) instead of falling back to OFF defaults and failing somewhere unrelated. A missing or
    unparseable file is not a verdict (the caller's own handling applies)."""
    import json
    import os

    if not config_path or not os.path.exists(config_path):
        return
    try:
        with open(config_path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    from agents.model.model_version.constants import ModelVersionError
    from agents.model.model_version.migrations import MIGRATION_FLOOR, _migrate_config
    version = int(raw.get("config_version", 1))
    if version >= MIGRATION_FLOOR:
        return
    try:
        _migrate_config(dict(raw))
    except ModelVersionError as e:
        raise PreBreakCheckpointError(
            f"{config_path}: {e}", config_path=config_path, config_version=version,
            last_commit=LAST_BLOB_COMMIT if version >= 121 else None) from e
