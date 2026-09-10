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

## The PRIVILEGED route — `--value-true-team` (v114, `gen3_value_true_team_v1`)

Every route above re-reads, re-pools or re-weights the SAME 2501-dim observation both heads
consume; none of them adds information. **This one does.** `TrueTeamValueReadout`
(`src/agents/model/true_team_value.py`) reads the opponent's ACTUAL party off a
training-and-eval-only Dict key `opp_true_team` — `[6, POKEMON_FULL_DIM]`, the obs's OWN per-mon
layout, built by `agents.observation.true_team.build_true_team_block` through the SAME
`PokemonEncoder.encode` called with `is_own=True` against the OPPONENT's own battle view, because
from that side every one of its mons IS a fully-known own mon. Six rows go through a shared per-mon
MLP (permutation-equivariant; the block's order is `species num ascending` and means nothing), then
`TTV_K`=4 learned queries over `TTV_DIM`=64 and a zero-init projection into `value_pooled`.

**It is a CEILING PROBE, arm 5 of the critic ladder** (`designs/research_state/winprob_critic_ladder_2026-09-08.md`
§L1): how much of the win-prob critic's residual error is irreducible uncertainty about the
opponent's team? A critic that needs privileged inputs is not the critic that ships, so it is
`family=CRITIC`, OFF by default, and never in `designs/production_config.json`.

**Four properties, each a constraint rather than a style choice.**

1. **vf-ONLY is structural.** The route injects into `value_pooled`, and `ProjectionAssembler`
   gives `value_pooled` to the value head ALONE (`vf_combined IS value_pooled`); `pi_combined` is a
   concat that does not contain it. So `pi` is bit-identical for an ARBITRARY weight in this
   module, not merely at init — `true_team_value_test.py` asserts it by perturbing the key at a
   large random weight, and asserts the backward direction too (a policy-only loss leaves the
   route's projection with no gradient).
2. **It AUGMENTS rather than REPLACES the belief-keyed opp view on the value side.** Replacing
   would be the cleaner ceiling in the abstract and would confound two changes in practice: a null
   could then mean "privilege does not help" OR "the belief route was carrying the signal".
   Additive injection leaves every existing value route bit-identical at init, and it is the only
   form the seam admits — additive injection changes no width, so route availability can never
   mis-size `value_pre_norm`.
3. **Presence follows the LOCAL sim, and nothing else.** `LocalBattleRunner` sets an `_opp_player`
   back-reference on both sides at attach time — one place that knows both sides, so "the sim is
   local" and "the privileged key is available" are the same fact. That covers bridge TRAINING,
   bridge EVAL (`eval_callback`: workers play in-process via `run_local_battles`) and the
   counterfactual replay driver. `Gen3Env` does not need it: it reads `battle2.team` directly. At
   ladder play (`src/main/play.py`, a real server) there is no runner, `RLPlayer` supplies the
   all-zero "unknown" block, and the policy — which never reads the key — runs unchanged.
4. **The route RAISES on a missing key rather than skipping.** A silent skip reads exactly like a
   route that learned nothing, which is the gen-12 dead-tail bug the seam exists to prevent. Every
   caller that builds its own obs dict therefore owes the key, and `ProbeModel._pin` — the
   prober's one offline-forward seam — REFUSES on such a checkpoint instead, because a V computed
   from the recorded observation vector alone is V stripped of the privilege, a different
   quantity. The arm's V is the one the eval traces RECORDED (computed with the key, on the
   bridge), which is what `cf_audit` and `main.critic_gate` already read — both take V and P(win)
   from the npz, never from a re-forward.

🚨 **SYNTHETIC-OBS COMPATIBILITY, and the launch it cost.** "Every caller owes the key" was
enforced by nobody, at four sites that each hand-built `{"observation": zeros(1, D)}`. The first
launch to combine `--value-true-team` with them — `ai_v12_14_ladder_truevalue` at 377a5aa1 — died
two minutes in at env init, exit 1: the forkserver compile preload traced on its one-key dict, the
route raised, and the raise killed the forkserver bootstrap so `SubprocVecEnv` construction failed
in the parent. The same argv's `--compile-trainer`, `--compile-opponents` and `--warmstart-battles`
carried the identical dict, so the crash would simply have moved.

The mapping is now DECLARED once — `agents.model.extra_obs_keys` — as
`(extractor attribute -> obs key, shape, canonical zero block)`, and the five training-run
synthetic-obs sites build from it (`compile_preload`, `lifecycle._run_roundtrip_test`,
`compile_trainer`, `compile_opponents`, `warmstart`). The enable condition is the ATTRIBUTE
(`true_team_value is not None`), the same expression the seam tests, so the two cannot drift; an
AST gate over `extractor_forward` fails on any obs key the table does not declare. **A new route
that reads a new Dict key needs a row there and nothing else.** The all-zero block those callers
supply is the same "no privileged view" encoding a real emitter uses at ladder play, so the traced
graph is the workers' graph. Adoption is partial by design: the offline audit / probe CLIs still
hand-build, and fail in the first second at a terminal rather than costing a GPU-hour
(`designs/ops/TECH_DEBT_BACKLOG.md`).

## `WinProbHead`, `CfEvidentialHead` and the three v99 additions

A separate `WinProbHead` (`win_prob_mode != none`) reads `value_pooled` *after* the pools and stashes
a `last_win_prob_logits` [B,1]. It never enters the pi/vf CONCAT, so projection dims are unchanged
either way — but **what consumes it depends on `--critic`**: under `shaped` it is a side readout fed
to the win-prob AUX loss and the prober, and under **`winprob` it IS the critic** (`_critic_value`
returns `sigmoid` of these logits and the head's BCE is the value loss at `vf_coef`). `read_only`
feeds it a STOP-GRAD `value_pooled` (head trains its own params only); `shaping` feeds it live (the
win objective also shapes the trunk), and `winprob` implies `shaping`.

`CfEvidentialHead` (`--cf-evidential`, v98) is a third readout off the same `value_pooled`, and
v99 adds THREE more there (`gen3_cf_twin_heads_v1`): the two `WinProbHead` TWINS
(`--cf-twin-heads` — heads B and C, the within-run paired R1 comparison) and the passive
`ShadowValueHead` (`--cf-shadow-critic`, an MC-grounded value twin that never computes an
advantage). All four share the evidential head's three properties — built LAST, never called by
the forward, input detached unconditionally — so the count off `value_pooled` is now SIX heads
and only `win_head` / `value_dist_head` are in the forward at all (`value_dist_head` when it is
built — `--critic winprob` refuses it, leaving `win_head` alone, as the critic). It is
the one that breaks the pattern in two ways worth knowing about. It emits a **Beta posterior** (α, β)
over P(win|state) rather than a point estimate — the counterfactual factory's uncertainty confession,
since G0 convicted the scalar head of RESOLUTION, not of an optimism offset. And it is **not called by
the forward at all**: the training-side term (`instrumented_ppo._cf_evidential_term`) applies it to the
STASHED `value_pooled`, always detached, so there is no `read_only`/`shaping` split to make and the
rollout pays nothing. Built LAST in `__init__`, so ON-at-coefficient-0 is BIT-identical to OFF in pi/vf
— not merely equal in shape, which is all the two heads above claim. Training half + the pre-registered
read: `designs/training/cf_grounding.md` → *The EVIDENTIAL Beta head*.

## `DenseAuxHead` — the one readout with a LIVE input (`--win-prob-dense-aux`, v117)

`gen3_dense_aux_v1`, the critic ladder's **arm 9**. `Linear(D_MODEL, 64) → ReLU → Linear(64, 25)`
over the same `value_pooled` the win head reads, zero-init output, built LAST, **not called by the
forward** — so the count off `value_pooled` is now SEVEN heads and the forward still calls only
`win_head` / `value_dist_head` / (per-action) `QWinProbHead`. What it predicts, for every state, is
the episode's **END-OF-BATTLE** facts, back-filled the way the win bit is: survival of each of the
12 slots (our 6 then theirs, in the observation's own team order), each slot's final HP fraction,
and the scaled turns-left. 25 sigmoid outputs, three masked-mean BCE terms averaged.

**It exists because four 10M levers on the win bit itself moved nothing** at ±0.01 on bot resolution
(ledger *THE ARMS AT 400 GAMES*), while only ~10 % of that bit's variance lies BETWEEN opponents.
KataGo's (Wu 2019 §3) answer to a one-bit terminal signal is not to fix the bit but to add auxiliary
targets that share its CAUSE — ownership of every point, the final score — for a large reported gain
in learning efficiency. Per-Pokémon end-of-battle outcomes are our analogue, and each is a fact
about a NAMED ENTITY, so the gradient runs along the axes a pooled bit cannot separate.

**It breaks the cf readouts' pattern in exactly one place, and that place is the arm.** Every other
head here that is absent from the forward (`CfEvidentialHead`, the twins, the shadow critic) takes
an UNCONDITIONALLY DETACHED input and reads `grad/*_share` 0.0 by construction. This one takes the
live stashed `value_pooled`, so its gradient reaches the shared trunk exactly as the win-prob loss
does under `shaping` — and `grad/dense_aux_share` must NOT read 0. "Never touches `pi`" still holds
in the sense that matters for a readout: the head's OUTPUT never enters the pi path, at any weight,
because the forward does not call it — a strictly stronger statement than `WinProbHead`'s, which is
in the forward and relies on the assembler.

`dense_aux` is a STRUCTURAL `flag_registry` row, `derived=True` off the coefficient
`win_prob_dense_aux` (the `opp_belief_slots` pattern: coef > 0 builds the head, the coefficient
doses the loss), `family=CRITIC`, `requires=("win_prob_mode",)`, gated by a bool compare in
`check_compatible` — so a resume may re-dose the arm but may not add or drop its parameters. No
`ARCH_SIGNATURE` bump: the observation vector is unchanged (the three LABEL keys ride separate Dict
keys and are deliberately NOT `extra_obs_keys` rows, because the extractor never reads them), no
module moves, the head is built last. Targets, the two masks, the λ precedence and every TB tag:
[`../training/critic_and_value_losses.md`](../training/critic_and_value_losses.md).

## `QWinProbHead` — the one readout that is NOT off `value_pooled`

### `QWinProbHead` (`--q-winprob-mode`, v107) — the one readout that is NOT off `value_pooled`

`gen3_q_winprob_head_v1`. Every other readout in this package evaluates a STATE, which is why "what
is my win probability if I click Rock Slide?" costs eleven simulator re-rolls rather than a read.
This head scores each of the eleven actions **from the token of the entity that action selects** —
the SAME per-action tokens `PointerNativeActionHead` scores, taken off `stash.pointer_inputs` — with
`value_pooled` as the board CONTEXT, and stashes `last_q_winprob_logits [B, 11]` in action-space
order. One forward, eleven `P(win|s,a)`: the amortized one-ply search leaf (ledger 229e9f1 /
5edbd05).

**Four properties, and each one is a constraint rather than a style choice:**

1. **ONE shared scorer** over all eleven slots. Three input projections exist only because the three
   families carry different WIDTHS; everything after them is shared, so the readout is
   permutation-equivariant within a family — permute our team and the six switch Q values permute
   with it. The pointer head's own lesson applies verbatim: a flat `Linear(ctx, 11)` learns "slot 0
   is usually right" from an ordering that means nothing, and a Q head that did so would be useless
   as a search leaf. (The pointer head's THREE scorers are correct there, where each family's logit
   has its own semantics; here every slot answers the same question.)
2. **ZERO-INIT scorer** (weight and bias) ⇒ every logit exactly 0 ⇒ `P = 0.5` everywhere ⇒ the
   untrained ranking is a total tie, which is the honest state of knowledge for a head that has seen
   no label. It is covered by `restore_identity_init`'s by-observation capture set automatically —
   and `q_winprob_head_test` proves the automatic coverage actually reached it on a REAL
   `MaskablePPO` build, because "it should be picked up" is what the M1 bug was made of.
3. **NO `shaping` mode.** Every input is detached INSIDE the forward, so `pi`/`vf` are bit-identical
   whenever the head is built and `grad/q_winprob_share` reads exactly 0.0 by construction. That is
   the `CfEvidentialHead` contract rather than `WinProbHead`'s tri-state, deliberately: a per-action
   readout carrying a COUNTERFACTUAL label is a strictly larger leak surface than a per-state one,
   so trunk exposure is a later decision that owes its own gate.
4. **The forward DOES call it** — the one place it departs from the four cf readouts. Eleven Q
   values are only useful if the forward that chose the action publishes them, so the contract is
   not "never runs" but "runs and publishes only".

**Built LAST in `__init__` for TWO reasons, and the second is specific to it.** The usual one is the
append-never-insert rule (SB3 restores optimizer state positionally; appending also leaves every
earlier module's init RNG draw untouched, which is what makes OFF byte-identical rather than merely
equal in shape). The specific one: it sizes its projections from `pointer_move_cell_dim` /
`pointer_switch_cell_dim`, so it must be constructed after every module that widens a pointer cell —
the op, the intent cells, the pair-outcome cells, the switch branch, the conditional threat.

Its two coefficients (`--q-winprob-coef`, `--q-winprob-onpolicy-coef`) are TRAINING-only and appear
nowhere in this package; the fold and the starvation caveat that governs the second one live in
`designs/training/cf_grounding.md` → *The PER-ACTION Q WIN-PROB HEAD*.
