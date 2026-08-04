"""Architecture constants — the single source of truth for every weight-shape-relevant dim.

Relocated out of `features_extractor.py` (2026-08-01) so that `damage_op.py` can read them without
importing the extractor, which would be circular: the extractor imports `DamageOperator`. Nothing
else changed — `features_extractor` re-exports the whole block, so `from
agents.model.features_extractor import D_MODEL` (and every other historical import path) still
resolves, and `model_version.py` keeps reading them from there.

Change a dim HERE and nowhere else. See the root `CLAUDE.md` -> Model Versioning: a change to any of
these is weight-shape-relevant and `check_compatible` will (correctly) reject old checkpoints.
"""
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [96, 32]        # [hidden, output] of shared move processor
# gen3_unified_move_system_v1: context-free per-move LATENT (MoveLatentEncoder) — a mechanics-grounded
# move identity (move/type embeddings ⊕ MOVE_ATTR), routed into the move network AND used as the
# similarity-grading target so Rock Slide ≈ Hidden Power Rock. Flag-gated (`move_latent`); OFF leaves the
# move network byte-identical.
MOVE_LATENT_HIDDEN = 64           # hidden width of the MoveLatentEncoder MLP
MOVE_LATENT_DIM = 32              # output dim of the per-move latent (grading is cosine in this space)
ROLE_ENCODER_HIDDEN = [256, 128]  # [hidden, output] of per-Pokémon role encoder
ACTIVE_CTX_HIDDEN = [64, 32]      # [hidden, output] of active context encoder
NET_ARCH = [512, 512]             # MLP policy layers (SB3 policy_kwargs["net_arch"])
N_HISTORY_TURNS = 7               # number of consecutive TurnDeltas in the observation
# gen3_zarch_film_v1 (v44): the team-archetype latent z_arch + head FiLM. ZARCH_DIM is the DEFAULT
# latent width (the CLI `--zarch-dim` records the run's actual value in model_config.json — the FiLM
# modulation is rank-z_dim by construction, so this is the conditioning-capacity knob). Flag-gated
# (`zarch_film != off`); OFF builds no modules (byte-identical baseline).
ZARCH_DIM = 32                    # default z_arch latent dim (= the FiLM conditioning rank)
ZARCH_ATOM_HIDDEN = 64            # hidden width of the per-mon static-atom MLP
# gen3_pointer_native_v1: the pointer action head's shared scorer hidden width (the ONLY action head —
# no flat action_net exists in this generation; see Gen3DualHeadMaskablePolicy._build).
POINTER_HIDDEN = 64               # hidden width of the pointer move/switch/struggle scorers

# Unified transformer hyperparameters. d_model matches ROLE_TOKEN_SIZE so team
# role tokens enter the transformer without a projection step.
D_MODEL = ROLE_TOKEN_SIZE         # 128
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_N_HEADS = 4
TRANSFORMER_FFN_DIM = 256
