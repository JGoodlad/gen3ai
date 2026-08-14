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

# gen3_no_concat_v1 (v61, the gen-5 world): the multi-seed CRITIC readout that replaces the op
# head-concat's value window — k learned queries cross-attend over the op's per-our-mon rows
# (the `our_mon` arity-1 tensor), giving the critic readout MULTIPLICITY (P3 refuted width,
# never multiplicity). k*dim rides vf_parts only. Ships WITH the seeds/* TB collapse monitors
# (seed_diagnostics.py) and the pre-registered VICReg trigger.
VALUE_SEED_K = 4
VALUE_SEED_DIM = 64

# gen3_intent_value_reduce_v1 (step 6): the alpha-weighted threat term appended to the CRITIC's
# pre-projection features. `_INTENT_CELL_FEATURES` is the F axis of the operator's un-reduced
# `cells_pr` stack (low/high/crit/ko_ramp/acc/phys_mask) — change one and the other must follow.
INTENT_VALUE_REDUCE_DIM = 64
_INTENT_CELL_FEATURES = 6
