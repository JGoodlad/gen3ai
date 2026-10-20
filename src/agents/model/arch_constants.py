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
# gen3_pointer_native_v1: the pointer action head's shared scorer hidden width (the ONLY action head —
# no flat action_net exists in this generation; see Gen3DualHeadMaskablePolicy._build).
POINTER_HIDDEN = 64               # hidden width of the pointer move/switch/struggle scorers
# gen3_policy_readout_trunk_v1 (config v138, architecture audit F2; `--policy-readout trunk`): the
# pointer scorers' hidden width when the flat policy tower is gone. The decision context is then the
# trunk's own D_MODEL-wide state query (`pools.PolicyStateQuery`), so the scorer keeps that width
# rather than squeezing the context to POINTER_HIDDEN. `tower` (production) keeps POINTER_HIDDEN.
TRUNK_POINTER_HIDDEN = 128
#: The state query's PRIVATE init seed: built inside `torch.random.fork_rng` from it, out of
#: `IsolatedLinear`s SB3's orthogonal re-init skips, so building it draws nothing from the global
#: stream (every surviving non-lever parameter's initial bytes equal the `tower` build's).
POLICY_QUERY_INIT_SEED = 20261006

# Unified transformer hyperparameters. d_model matches ROLE_TOKEN_SIZE so team
# role tokens enter the transformer without a projection step.
D_MODEL = ROLE_TOKEN_SIZE         # 128
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_N_HEADS = 4
TRANSFORMER_FFN_DIM = 256

# v61's `VALUE_SEED_K` / `VALUE_SEED_DIM` (the multi-seed critic readout's shape) and v74's
# `_INTENT_CELL_FEATURES` (the α-weighted vf term's cell axis) are DELETED in the critic-route
# deletion wave along with the modules they sized. The op's un-reduced pair-cell F axis
# [low, high, crit, ko_ramp, acc, is_phys] is still named — once, as `_PAIR_OUTCOME_DMG` below.

# gen3_unified_value_readout_v1 (v80, Stage-3 T3-DELIVER, design_unified_belief.md §3): ONE
# attention pool over the critic's ENTITY-ROW SET (the 12 post-transformer team tokens + the op's
# 6 per-our-mon incoming rows) — the designed successor of the bolt-on vf routes, and the one the
# critic-route audit picked (dV 5.490 = 97% of all_off on gen-14). K learned queries over rows
# projected to a shared UVR_DIM (with a per-SOURCE type embedding), zero-init output projection to
# D_MODEL. `_UVR_N_SOURCES` axes: 0=our mon token, 1=their mon token, 2=op incoming row.
# gen3_value_pooled_routes_v1 (v89): a value route INJECTS additively into value_pooled, so its
# output width is D_MODEL by definition — there are no per-route width constants.
UVR_K = 4
UVR_DIM = 64
_UVR_N_SOURCES = 3
# gen3_unified_value_readout_v2 (v82, `value_entity_pool_full`): +source 3 = the refined
# GLOBAL token, +source 4 = the hidden-opp belief queries — the pool's COMPLETE row set (the
# one successor for every condemnable vf route). A separate flag/shape so v80-table
# checkpoints keep loading.
_UVR_N_SOURCES_FULL = 5

# gen3_pair_outcome_v1 (design_opponent_intent.md §5.1 + design_pair_reduction.md §2.1/§9a): the UNIFIED PER-PAIR
# OUTCOME GRID — one `pair_in[j, k, :]` per (our mon j, their believed move k) carrying damage AND status in the SAME
# vector, which the move-resolution family reads (`move_resolution.gather_ops`).
#
# `_PAIR_OUTCOME_DMG` is the op's pair-cell width — the F axis of `last_pair_cells`:
# [low, high, crit, ko_ramp, acc, is_phys]; `_PAIR_OUTCOME_NEW` is the eight coordinates after it (6 per-identity
# status landing probabilities + the two exact status facts; the table is `pair_outcome.PAIR_OUTCOME_COORDS`).
# `_PAIR_OUTCOME_RAW` is their concatenation = the grid's width. (The seven per-action blocks that once delivered
# this grid and its siblings to the pointer cells, with their width constants, were deleted in legacy deletion U3.)
_PAIR_OUTCOME_DMG = 6
_PAIR_OUTCOME_NEW = 8
_PAIR_OUTCOME_RAW = _PAIR_OUTCOME_DMG + _PAIR_OUTCOME_NEW      # 14

# gen3_move_resolution_v1 (config v141; architecture audit F11 §9): the MOVE-RESOLUTION family that REPLACED the seven
# per-action blocks (deleted, legacy deletion U3) — per legal action, P(it resolves as stated) and its factors, plus
# the seven blocks' 55 FACT coordinates consolidated (their judgments dropped). The coordinate tables are
# `move_resolution_rules.MOVE_RESOLUTION_{MOVE,SWITCH}_COORDS`; each block rides through a zero-init projection
# of the same width.
_MOVE_RESOLUTION_MOVE_RAW = 38
MOVE_RESOLUTION_MOVE_DIM = _MOVE_RESOLUTION_MOVE_RAW
_MOVE_RESOLUTION_SWITCH_RAW = 18
MOVE_RESOLUTION_SWITCH_DIM = _MOVE_RESOLUTION_SWITCH_RAW

# gen3_value_direct_routes_v1 (v87): two direct CRITIC routes appended at the vf tail, both
# zero-init. VALUE_CLOCK_DIM — the deadline clock's 3 raw scalars projected for the critic (the
# v67 clock fix was validated for exactly this reader, and the audit read its surviving
# indirect route as dead). VALUE_INTENT_DIM — the published α (K seats + SWITCH, belief-sorted
# canonical order) and β (6 team slots) posteriors as probabilities; the critic finally reads
# WHAT WE EXPECT THEM TO DO directly rather than only through the α-weighted physics cells.


# gen3_hidden_slot_move_mixture_v1 (E10, the parameter-free hidden-slot move prior). An opponent
# slot whose MON is unrevealed used to fuse its move delta with `move_prior_logits[0]` — the flat
# floor row of the UNKNOWN-species sentinel — so its move posterior was a state-INDEPENDENT constant
# (max deviation 0.0 over 57k decisions, belief-calibration read 2026-09-24). It now fuses with the
# Smogon MIXTURE `P(m | hidden) = Σ_s P_T0(s | revealed) · P(m | s)` (the T0 team-composition species
# prior times the same per-species move-prior buffer), which has NO learned parameters by design: it
# cannot memorise the pool. The mixture probability is clamped into [EPS, 1 − EPS] before the logit.
HIDDEN_SLOT_MIX_EPS = 1e-6

# gen3_ridealong_heads_v1 (v126, `--ridealong-ensemble` / `--ridealong-rnd` / `--ridealong-adv` /
# `--ridealong-opp`): the DETACHED RIDE-ALONG heads (`agents.model.ridealong_heads`). Every one
# reads the trunk through `.detach()` and trains on its own optimizer, so none of these widths can
# move what the policy or V learn. RIDEALONG_HIDDEN is the V-ensemble member width (64, the width the retired dense-aux head used,
# for the same "differ in what they see, not in capacity" reason). The PRIOR scales are the
# randomized-prior functions' (Osband et al. 2018) output scales: V members add β·p(x) in LOGIT
# units, A/B members in PROBABILITY units (an advantage under the win-prob critic is a probability
# difference, typically |A| < 0.1). RND (Burda et al. 2018): a frozen random target
# obs → RND_HIDDEN → RND_OUT and a trained predictor one layer deeper.
RIDEALONG_HIDDEN = 64
RIDEALONG_PRIOR_SCALE_V = 1.0
RIDEALONG_PRIOR_SCALE_Q = 0.05
RIDEALONG_OPP_MOVE_EMB = 32
RIDEALONG_RND_HIDDEN = 256
RIDEALONG_RND_OUT = 64
# gen3_ridealong_rnd_variants_v1 (v127, `--ridealong-rnd-variants`): the `small` variant's predictor,
# obs → RND_SMALL_HIDDEN → RND_OUT, one hidden layer. It has 93,184 parameters, against the base
# predictor's 810,816 (11.5 %; the 2845-dim observation). Every observation is squeezed through 32 linear
# features before the one nonlinearity, so it cannot hold a per-battle fingerprint of the observation. The
# target it must match is 256 hidden units wide, so it cannot identify the target exactly either: its
# novelty should track coarse density. The other variants reuse the widths above (`fast` / `decay`
# are base's predictor; `feat` is base's shapes over the D_MODEL-dim value_pooled).
RIDEALONG_RND_SMALL_HIDDEN = 32
#: The PRIVATE init seed. Every ride-along module is built inside `torch.random.fork_rng` from this
#: seed, so building the heads draws nothing from the global stream (heads ON and OFF leave the
#: policy's own init, and every later draw, bit-identical).
RIDEALONG_INIT_SEED = 20260930

# gen3_x5_hypothesis_set_v1 (X5 build unit U2, X5;
# `designs/endstate/design_x5_belief_tokens.md` §3.2-§3.3). HYPOTHESIS_DELTA_HIDDEN is δ_θ's MLP
# width: the Deep-Sets pool over the revealed opponent role tokens ⊕ the global projection
# (2 × D_MODEL) → this → the species logits. (OTHER_species' embedding averages the WHOLE tail since
# U3 — the "next 32" cut and its constant are gone, F-X5-23.) HYPOTHESIS_INIT_SEED is the PRIVATE init seed: every X5 module is built inside
# `torch.random.fork_rng` from it (and from `IsolatedLinear`, which SB3's orthogonal re-init skips),
# so the fixed_mass arm leaves every non-X5 parameter's initial bytes equal to the blob arm's.
HYPOTHESIS_DELTA_HIDDEN = 96
HYPOTHESIS_INIT_SEED = 20261003

# gen3_x5_flat_pointer_v1 (X5 build unit U4, X5 only; design §3.7): the FLAT
# opponent pointer that replaces the α / β heads in that arm. FLAT_INTENT_HIDDEN is its one shared
# scorer's width (α's and β's scorers were 64 wide); FLAT_INTENT_INIT_SEED its PRIVATE init seed (built
# inside `torch.random.fork_rng` from `IsolatedLinear`s, so no non-X5 initial byte moves).
FLAT_INTENT_HIDDEN = 64
FLAT_INTENT_INIT_SEED = 20261004

# gen3_static_tokens_v1 (`--token-encoding static`; `designs/endstate/design_static_tokens.md`). Under
# `static` a Pokémon's token is S (its static identity: species + set + actual stats) + D (its own battle
# state), with NO board fact in either. STATIC_STAT_SCALE divides the level-100 actual stats (and the
# opponent prior's std) S reads; STATIC_DYN_HIDDEN is D's MLP width (input → this → ROLE_TOKEN_SIZE);
# STATIC_MOVE_POOL_DIM is the width of D's per-move dynamic pool Σ_k ReLU(W[m_k; pp_k; legal_k]).
STATIC_STAT_SCALE = 500.0
STATIC_DYN_HIDDEN = 128
STATIC_MOVE_POOL_DIM = 32
# gen3_static_port_v1: the per-MOVE hidden width of the static op content's OUTGOING set function (`board_tokens.OpContent`):
# each of our active's moves' d1 cell on one of their mons goes through ONE shared bias-free Linear(6 → this) + ReLU,
# summed over the moves (Deep Sets), then a zero-init bias-free Linear(this → D_MODEL). Replaces the request-ordered
# Linear(24 → D_MODEL), which gave each request slot its own weights (the v145 sweep's last arbitrary static weight).
STATIC_OPC_OUT_HIDDEN = 32
# gen3_static_recovery_v1 (`--trunk-layers N`, `trunk_depth.py`): the PRIVATE init seed of the extra trunk rounds. They
# are built inside `torch.random.fork_rng` from it, out of `IsolatedLinear`s SB3's orthogonal re-init skips, so a deeper
# trunk draws nothing from the global stream (every other parameter's initial bytes equal the 2-round build's).
TRUNK_EXTRA_INIT_SEED = 20261009

# gen3_trunk_ffn_by_kind_v1 (B-KFFN, `--trunk-ffn-by-kind on`, `trunk_kffn.py`): the PRIVATE init seed of the per-kind FFN
# experts. Round i's experts are built inside `torch.random.fork_rng` from KFFN_INIT_SEED + i, out of `IsolatedLinear`s
# SB3's orthogonal re-init skips, so the experts draw nothing from the global stream (every other parameter's initial bytes
# equal the off build's) and a round's experts are the same bytes whether it was built fresh or appended by a growth.
KFFN_INIT_SEED = 20261020
