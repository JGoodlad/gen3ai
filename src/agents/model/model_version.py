from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

# Bump this whenever the ModelVersion schema changes (fields added/renamed/removed).
# Also add a migration case in _migrate_config().
MODEL_CONFIG_VERSION = 2

# Change this when the neural architecture changes structurally in a way that makes
# weights from a different signature incompatible (e.g. adding LSTM, replacing attention).
# Same-family dim changes (role_token_size 128→256) don't need a new signature —
# check_compatible() catches those via the dim fields.
ARCH_SIGNATURE = "gen3_spread_v1"


class ModelVersionError(Exception):
    pass


@dataclass
class ModelVersion:
    # Schema and architecture identity
    config_version: int
    arch_signature: str

    # From state_encoder.get_layout()
    species_embedding_dim: int
    max_species: int
    move_embedding_dim: int
    max_moves: int
    item_embedding_dim: int
    max_items: int
    ability_embedding_dim: int
    max_abilities: int
    type_embedding_dim: int
    max_types: int
    total_dim: int
    active_context_dim: int

    # From features_extractor.py module constants
    role_token_size: int
    projection_dim: int
    move_net_hidden: List[int]
    role_encoder_hidden: List[int]
    active_ctx_hidden: List[int]
    n_history_turns: int

    # From policy_kwargs in train_rl_agent.py
    net_arch: List[int]

    @classmethod
    def from_layout_and_policy_kwargs(
        cls,
        layout: Dict[str, Any],
        policy_kwargs: Dict[str, Any],
    ) -> ModelVersion:
        from agents.model.features_extractor import (
            ROLE_TOKEN_SIZE,
            PROJECTION_DIM,
            MOVE_NET_HIDDEN,
            ROLE_ENCODER_HIDDEN,
            ACTIVE_CTX_HIDDEN,
            NET_ARCH,
            N_HISTORY_TURNS,
        )
        return cls(
            config_version=MODEL_CONFIG_VERSION,
            arch_signature=ARCH_SIGNATURE,
            species_embedding_dim=layout["species_embedding_dim"],
            max_species=layout["max_species"],
            move_embedding_dim=layout["move_embedding_dim"],
            max_moves=layout["max_moves"],
            item_embedding_dim=layout["item_embedding_dim"],
            max_items=layout["max_items"],
            ability_embedding_dim=layout["ability_embedding_dim"],
            max_abilities=layout["max_abilities"],
            type_embedding_dim=layout["type_embedding_dim"],
            max_types=layout["max_types"],
            total_dim=layout["total_dim"],
            active_context_dim=layout["active_context_dim"],
            role_token_size=ROLE_TOKEN_SIZE,
            projection_dim=PROJECTION_DIM,
            move_net_hidden=list(MOVE_NET_HIDDEN),
            role_encoder_hidden=list(ROLE_ENCODER_HIDDEN),
            active_ctx_hidden=list(ACTIVE_CTX_HIDDEN),
            n_history_turns=N_HISTORY_TURNS,
            net_arch=list(policy_kwargs.get("net_arch", NET_ARCH)),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json_file(cls, path: str) -> ModelVersion:
        with open(path) as f:
            data = json.load(f)
        data = _migrate_config(data)
        return cls(**data)

    def check_compatible(self, saved: ModelVersion) -> None:
        """Raises ModelVersionError if saved is incompatible with self (current code).
        Call as: current_version.check_compatible(saved_version).
        """
        # Architecture family — hard stop if different
        if self.arch_signature != saved.arch_signature:
            raise ModelVersionError(
                f"Architecture family mismatch: saved='{saved.arch_signature}', "
                f"current='{self.arch_signature}'.\n"
                "These models use structurally different networks and cannot be loaded interchangeably.\n"
                "Start a fresh training run, or use subprocess isolation for league play."
            )

        # Weight-relevant fields — all must match exactly
        _WEIGHT_FIELDS = {
            "total_dim", "active_context_dim",
            "species_embedding_dim", "max_species",
            "move_embedding_dim", "max_moves",
            "item_embedding_dim", "max_items",
            "ability_embedding_dim", "max_abilities",
            "type_embedding_dim", "max_types",
            "role_token_size", "projection_dim",
            "move_net_hidden", "role_encoder_hidden", "active_ctx_hidden",
            "n_history_turns",
            "net_arch",
        }
        current = asdict(self)
        saved_d = asdict(saved)
        mismatches = [
            f"  {k}: saved={saved_d[k]!r}, current={current[k]!r}"
            for k in sorted(_WEIGHT_FIELDS)
            if current[k] != saved_d.get(k)
        ]
        if mismatches:
            raise ModelVersionError(
                "Model weight-shape mismatch — cannot load saved model with current architecture.\n"
                "Mismatched fields:\n" + "\n".join(mismatches) + "\n\n"
                "Fix: restore matching constants, or start a fresh training run."
            )


def _migrate_config(data: dict) -> dict:
    """Apply incremental forward-migrations to bring an old config up to the current schema."""
    version = data.get("config_version", 1)
    if version < 2:
        # v2: added n_history_turns. Old models used a single TurnDelta (N=1).
        data.setdefault("n_history_turns", 1)
        data["config_version"] = 2
    return data
