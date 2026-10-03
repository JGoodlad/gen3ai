# Readouts off `value_pooled`, and the critic's delivery routes

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## The critic's delivery routes

**gen3_no_concat_v1 (v61): its flat block no longer enters either
projection** — the op reaches the policy via the pointer cells + prefuse injection + edge cells, and
the critic through `UnifiedValueReadout` (`--value-entity-pool`, `gen3_unified_value_readout_v1` —
Stage-3 T3-DELIVER of `design_unified_belief.md` §3): ONE attention pool over the critic's entity
rows (the 12 team tokens + the op's per-our-mon incoming rows, per-source type embeddings, UVR_K=4
queries, zero-init out projection, injected into `value_pooled` so pi is untouched at any weight).
v61's stopgap — the `MultiSeedValueReadout`, k=4×64 seed queries over the same rows — was the
critic's window in between, and the **critic-route deletion wave DELETED it** along with its
`value_seeds/*` TB collapse contract: dV **0.0000 bit-exact** on two consecutive end-of-run audits,
against the entity pool's 5.490 (97% of the whole critic route joint). The succession the v80 flag
was built for is complete; `value_entity_pool_test.py` pins the contract.
**TWO PRESSURES WERE APPLIED TO THOSE SEEDS AND BOTH WERE DELETED (v78)** — the record is kept
because the finding is what closed the line, not the code. `--value-seed-vicreg-coef` (v62,
`seed_vicreg.py`) was the repulsive one: gen-6 satisfied every VICReg term while
`out_effective_rank` stayed 1.05, because the deviations occupied <1 direction (three seeds
identical, one breakaway). `--seed-quantile-coef` (v63, `seed_quantile.py`) was the positive
counterpart — seed k predicts quantile τ_k of the return through ONE SHARED Linear, so k different
predictions require k different seed reads — and gen-7 drove `crossing_rate` to 0.000 with
`quantile_spread` 1.016 (the seeds genuinely predict four ordered quantiles) while
`out_effective_rank` reached only **1.157 of k=4**, matching gen-6's centered PR 0.846 from the
opposite direction. **A SHARED readout can only constrain each seed's component along its own
weight vector; every orthogonal direction stays free**, so multiplicity is not the missing axis and
no coefficient reaches it — which is why both flags were deleted rather than retuned. The response is
**`--value-threat-inject` (v64, `value_threat_inject.py`)** — magnitude as TOKEN CONTENT rather
than as another readout seat: one shared zero-init `Linear(13, D_MODEL)` adds the op's α-weighted
incoming row for our mon j (α = the R1 `belief_mean` rung, which the flag forces on since R0
`hard_max` builds no reducer) to that mon's token on **the value pool's copy only**, inside
`CLSPool`. Keeping the augmented tensor a local is what makes "vf-only" structural rather than a
convention: `our_cls`, `our_active_refined` and the pointer head cannot reach it, so `pi` is
bit-identical at ANY weight — gated against a large random projection, not just at init. Equivariant
in both axes (α has no defender index by Contract W; the row rides mon j's token; attention pooling
is permutation-invariant). `W_inj` sits in the `restore_identity_init()` capture set (M1) and that
is gated on a REAL `MaskablePPO` build. Structural + version-checked, off = no module).
**v95 adds its SIBLING on the same local copy — `--pair-value-route` (PV,
`design_opponent_intent.md` §7a(2)) — carrying Phase A's UNIFIED 14-coordinate `pair_in` row
instead of v64's 13-wide damage summary, so the critic gets the six status identities,
`neutralization` and `tempo_cost` per entity for the first time (they otherwise reach vf only as
the `s3` edge family's softmax-normalised RATIO). The two stack additively and independently.
⚠️ Its ENABLING owes the C4-style offline gate first (ledger C6); BUILDING it is free.

## The PRIVILEGED route — DELETED (deletion pass L2)

`--value-true-team` (v114, `gen3_value_true_team_v1`) was the critic ladder's arm-5 CEILING PROBE: a
`TrueTeamValueReadout` over the opponent's ACTUAL party read off the training-and-eval-only Dict key
`opp_true_team`, injected zero-init into `value_pooled`. It was the only route that added information
rather than re-reading the shared observation; the probe was never promoted and the route was deleted
with the rest of the critic-ladder levers (ledger arms-ladder verdicts; recoverable at pin
<= 475bd817). The seam (`_value_pooled_routes`) now has ONE member, `value_entity_pool`, and takes no
`obs` argument; `agents/model/extra_obs_keys.py`'s registry is EMPTY, but the mechanism (declare an
`(extractor attribute -> obs key, shape, zero block)` row; every training-path synthetic-obs site
builds from `zero_extra_obs` / `synthetic_obs`; an AST drift gate fails on an undeclared key) stays,
because the failure it prevents (`ai_v12_14_ladder_truevalue`: a (since-deleted) forkserver preload traced on a
one-key dict, killed at env init) is re-armed by any future obs-key-adding route. **A new route that
reads a new Dict key needs a row there and nothing else.** Adoption is partial by design: the offline
audit / probe CLIs still hand-build (`designs/ops/TECH_DEBT_BACKLOG.md`).

## `WinProbHead`

A separate `WinProbHead` (`win_prob_mode != none`) reads `value_pooled` *after* the pools and stashes
a `last_win_prob_logits` [B,1]. It never enters the pi/vf CONCAT, so projection dims are unchanged
either way — but **what consumes it depends on the recorded `critic`**: under `shaped` it is a side readout fed
to the win-prob AUX loss and the prober, and under **`winprob` it IS the critic** (`_critic_value`
returns `sigmoid` of these logits and the head's BCE is the value loss at `vf_coef`). `read_only`
feeds it a STOP-GRAD `value_pooled` (head trains its own params only); `shaping` feeds it live (the
win objective also shapes the trunk), and `winprob` implies `shaping`.

`WinProbHead` is the ONE readout off `value_pooled` that remains in the extractor: the evidential Beta
head, the twin win-prob heads B/C and the passive shadow critic (v98–v99) were deleted with the
counterfactual training half (deletion pass L4; a checkpoint that recorded one ON is refused on every
load, `model_version/retired_levers.py`), and the distributional `value_dist_head` was deleted in
deletion pass L1. `win_head` is the only one of them the forward calls.

## `DenseAuxHead` — DELETED (deletion pass L2)

`--win-prob-dense-aux` (v117, `gen3_dense_aux_v1`, the critic ladder's arm 9) was a 25-output head on
`value_pooled` (survival and final HP of the twelve slots plus turns-left) with a LIVE, un-detached
input, so its gradient reached the shared trunk; it was the SEVENTH readout off `value_pooled` and
the one exception to the (since deleted) cf readouts' detached-input pattern. It was deleted with the critic-ladder
levers (the arm read no gain on the one-bit terminal target, ledger *THE ARMS AT 400 GAMES*);
recoverable at pin <= 475bd817. Its structural `dense_aux` / `value_true_team` flags recorded ON are
refused on every load (`model_version/retired_levers.py`). `WinProbHead` is the one readout off `value_pooled` that remains.

## `QWinProbHead` — the per-action Q head is DELETED; the CLASS survives as the A head's scorer

The extractor-built per-action win-prob head (`q-winprob-mode`, v107; it scored each of the eleven action
slots from the entity token the slot selects, with `value_pooled` as context, and stashed
`last_q_winprob_logits`) was deleted with the counterfactual training half (deletion pass L4): the extractor
no longer builds a `q_winprob_head` or stashes those logits, and a checkpoint that recorded the mode ON is
refused on every load. `QWinProbHead` (`agents/model/q_winprob_head.py`) survives ONLY as the shared-scorer
class the detached ride-along **A head** (`ridealong_heads.AdvantageEnsemble`, `--ridealong-adv`; next section)
is built from — ONE scorer shared across the eleven slots, a zero-init scorer, action-space column order.

## The DETACHED RIDE-ALONG heads — on the POLICY, not off the extractor (`gen3_ridealong_heads_v1`, v126)

Owner, 2026-09-30: a baseline of epistemic confidence and of Q = V + A + B **before** any arm that
changes learning. `agents/model/ridealong_heads.py`; flags `--ridealong-ensemble K`,
`--ridealong-rnd`, `--ridealong-adv K`, `--ridealong-opp K` (all OFF by default; STRUCTURAL,
`family=CRITIC`).

| head | input (all `.detach()`ed) | label | loss |
|---|---|---|---|
| V ensemble (K) | LayerNorm(`value_pooled`) | the win bit (`win_target`, `win_mask`), V's own target | BCE per member, masked by its bootstrap bits |
| RND | the RAW observation, per-dimension running mean / variance, clipped ±5 | a frozen random net's output | MSE (one visit per rollout row) |
| A (K) | `stash.pointer_inputs` + `value_pooled` (`QWinProbHead`'s shape) | the GAE advantage of the action TAKEN | MSE of the π-centred member at the taken action |
| B (K) | α's seat move IDs (own embedding) + `value_pooled`; SWITCH from the context | the same advantage | MSE of the α-centred member at the opponent's actual action (α's own `match_seats_to_move_num` target; misses masked) |

**All four heads train on PPO's FIRST epoch only** (`RIDEALONG_EPOCHS` = 1): each rollout row once,
scored before the step that trains on it. On the GPU learner benchmark, training them on all ten
epochs cost +8.9 s on a 67.0 s update (+13 %), far past the ~2 % instrument budget.

**Why the heads are NOT extractor modules, unlike every readout above.** The owner's rule is that the
baseline learns EXACTLY what production learns, so this is tested bit-for-bit. Three things
in the extractor would break it. SB3's ortho-init `apply` re-draws every `nn.Linear` in the extractor
from the GLOBAL RNG, so any new Linear there shifts `mlp_extractor` / `value_net` init and every later
sample. `policy.optimizer` is built from `self.parameters()`. And PPO's `clip_grad_norm_` runs over
all policy parameters. So the extractor RECORDS the four kwargs and builds nothing, and the policy builds
`policy.ridealong` after `_build` inside `fork_rng` from `RIDEALONG_INIT_SEED`. The heads' params
are in no PPO param group. The learner steps them with its own Adam before PPO's loss exists and
returns their grads to None.

**Diversity is forced, because members sharing a detached trunk collapse otherwise.** There are three
sources. Each member has its own init seed. Each has a per-STATE bootstrap mask: bit j of an exact-integer hash of the
observation row, so it is identical in every epoch and minibatch and consumes no RNG
(bootstrapped DQN, Osband et al. 2016). And each has a RANDOMIZED PRIOR, a frozen random copy
added to its output (Osband et al. 2018; scale `RIDEALONG_PRIOR_SCALE_V` = 1 logit for V, `_Q` = 0.05
probability for A / B). An action the policy never plays keeps its A members at their prior spread,
which is the per-action "starved move" uncertainty.

**RND reads the observation, not the features.** Trunk features drift as the trunk trains, so
feature-space novelty confounds "rarely seen" with "the representation moved". The raw observation
carries embedding IDs as scalars. After standardisation a random net is still a fixed function of them, so
a rare id stays novel; what is lost is id-to-id semantics. MEASURED offline (fixed bank, the heads' own
shapes, 95 % battle-clustered CIs; `designs/research_state/measurements/ridealong_baseline/`):
- obs-RND is at least as good as feature-RND on every read. Novelty falls with a coarse state key's
  training count (Spearman −0.34 vs −0.31). It flags off-distribution states: a bot-trained predictor
  scores self-play states at AUROC 0.75 vs 0.64, and late game at 0.83.
- Feature-RND is dominated by representation drift: 2.5–8× novelty inflation on the same states across
  30M steps of one lineage.
- Two limits:
  - obs-RND memorises whole battles (held-out battle vs train rows 0.85 in a 10-epoch offline fit).
  - Neither variant tells the successors of the moves the policy STARVES from its other unplayed moves
    (starved near-best minus starved far +0.001 [−0.037, +0.040]). 87 % of RND variance lies between
    turns, not between one turn's actions. RND is a state- and trajectory-level allocator, not an
    action-level one.

**B is the simple pre-X5 parameterisation.** Its columns are α's support, and it is to be re-based onto X5's flat
opponent pointer (seats + switch targets + OTHER). A and B are identified by two MARGINAL regressions
on the ONE advantage label. The two actions are simultaneous and each head is centred under its
side's policy (π for A, α for B, both stop-grad), so E[adv | s, a] = A(s, a) and E[adv | s, b] = B(s, b).
B's declared limit: a move outside the believed seats is not a label, so B is conditional on the
opponent choosing a listed option (α's mask rate). There is no I term. Q = V + A + B is a readout
(`ridealong/q_out_of_range`).

**The RND VARIANT ENSEMBLE (`gen3_ridealong_rnd_variants_v1`, v127, `--ridealong-rnd-variants`).**
Plain RND counts cumulative visitation and can saturate. "Recently seen" needs forgetting, and the
lever for forgetting is the PREDICTOR's memory, not the target. So four declared variants ride beside
the unchanged RND head (`base`), each with its own predictor, Adam, clip, error z-score and
fail-closed switch (`RND_VARIANT_DECLS`; the table is in `designs/ARCHITECTURE.md` §3.4):
`fast` (10× the rate), `decay` (pulled toward its own init, half-life 10 PPO updates), `small`
(obs→32→64, deliberately less expressive than the target) and `feat` (over the detached
`value_pooled`, with its own target and normalisation; it measures the drift above LIVE). The
observation variants share base's target and normaliser, so each comparison with base is paired, and
`fast` / `decay` start from base's exact weights. Two monitors exist for the questions the variants
pose. Saturation is the raw-error IQR ÷ median of every fresh row in an update. Identification is
`*_ident_ratio`: the error on the minibatch's block chimeras (each observation block copied from a
different row, deterministic, no RNG) over the error on the real rows. A predictor that learns the
target everywhere, rather than on the states it visits, drives that ratio down. The verdicts are the
reader's, on fixed probes, per the X26 amendment.

**Lifecycle.** Every ride-along optimizer, the four heads' and each variant's, is built at the
trainer's `_setup_model` with its Adam state pre-allocated. There is no lazy build: a step that
finds one missing raises `RideAlongLifecycleViolation`. The heads' step is the K8 inventory's candidate compile region R-ride; it stays eager
for now.
