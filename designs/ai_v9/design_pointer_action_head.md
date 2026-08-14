# Full pointer action head — making the pointer WORK for mons (switches) and moves

**Status:** design, REVISED for the fresh-generation reset (2026-08-03). The v49 delta head
(`gen3_pointer_head_v1`) is BUILT and shipped; §0 below is the operative plan — **§0 SHIPPED
2026-08-03 as v51 `gen3_pointer_native_v1`** (the flat `action_net` is deleted; the pointer
head is THE action head, with no off state). Companion
concept doc: `designs/learning/entity_tokens_biases_pointers.md`; the end-state inventory is
`design_entity_graph.md` §3 (Readout heads); the deprecation motive is
`designs/ai_v8/next_run_plan.md` → "Op head-concat deprecation".

---

## 0. Fresh-generation reset (OWNER, 2026-08-03 — supersedes the staging below)

**The premise changed:** the next architecture is a NEW GENERATION — fresh run, fresh pools,
no old checkpoint loads, many refactors landing together. Position-EQUIVARIANCE is adopted as
a first-class architectural goal (one shared scoring function per entity token; slot identity
is content, never a memorized weight row — the complexity-of-reasoning win as much as the
sample-efficiency win). Under that premise the staged plan splits into two parts with
opposite fates:

- **Migration machinery — DELETED.** The delta-on-flat form, the λ-anneal, the strip
  migration, the `flat_action_head` toggle, the Level-A/Level-B split, identity-at-init as an
  A/B guarantee, and every `_migrate_config` shim for crossing the boundary. All of it existed
  to protect a trained lineage whose biggest measured dependency (ledger P1) rides the flat
  path. A fresh generation never builds that dependency — the P1 caution is about surgery on
  a live patient, and there is no patient. **`ARCH_SIGNATURE` bumps** (e.g.
  `gen3_pointer_native_v1`): every pre-generation checkpoint fails loud with a family error;
  no compat gating, no flags — pointer scoring is UNCONDITIONAL in the new arch.
- **The gap analysis (§2) — PROMOTED from experiment plan to day-0 REQUIREMENTS.** The one
  wrong move left is shipping the bare v49-shaped head (board-blind move tokens, bare 128-dim
  ctx) as the only head — information-starved by construction. G1–G5 are closed at build
  time, not discovered by A/B.

**The pointer-native spec (one shot):**
- No `action_net`. `_get_action_dist_from_latent` builds the distribution from pointer logits
  directly (`proba_distribution(action_logits=…)`); the policy overrides `_build` /
  post-build to never carry the dead module.
- Move logit k = shared scorer over (ctx, request-slot-k move token ⊕ its outgoing +
  status-landing + secondary op cells). Switch logit j = shared scorer over (ctx, team token
  j ⊕ incoming row j ⊕ OAX row j). Struggle from ctx. All op cells sliced at named offsets
  (`decode_damage_block` is the SoT mirror); the request-order alignment is pinned by a
  throwing guard + the scrambled-moveset test (§7 — the tests survive unchanged).
- Ctx = `latent_pi` — the pi tower (projection → mlp_extractor) survives as the CONTEXT
  ENCODER, delivering the op concat / beliefs / FiLM to action scoring (closes G4/G5; FiLM
  keeps its application point).
- Scorer output layers zero/small-init — no longer for A/B byte-identity, but because
  uniform-over-legal is the correct cold-start policy (and the M1 guard covers them for free).
- ModelVersion: new fields recorded for within-generation resume/pool loads; no cross-era
  compatibility.

**What "no A/B needed" means, precisely.** Mechanical correctness (alignment, the single
logit funnel, masking/log-prob/entropy consistency, gradient flow) is PROVABLE — deterministic
tests, no experiment. Strength adequacy remains empirical, but its instrument is not an A/B:
**anchored ELO + the fixed bot suite are cross-run comparable by design** — the new generation
is judged against the old generation's curve. Attribution caveat, stated once: many
simultaneous refactors mean an underperforming generation is hard to bisect; the mitigation is
that mechanically-provable changes (this one) carry near-zero risk mass, so suspicion
concentrates on the speculative ones.

**Scope fence for this generation:** the op **vf-side** concat is also position-sensitive
(per-slot blocks at fixed projection offsets); full equivariance eventually re-homes it
through invariant routes (tokens → attention → value CLS). Policy-side pointer-native is THIS
generation's commitment; the vf re-home is a separate decision (a critic-input question, not
an action-head one — don't couple the blast radii).

**What survives from the staged plan below:** §1 (what v49 built — the permutation, the stash,
the funnel — all reused verbatim), §2 as the requirements list, §5 observability (now plain
head diagnostics), §7 tests (minus the anneal/delta cases, plus "no action_net params exist /
none receive gradient"), and §8's scope fences (ordering_integrity stays — it guards the
obs side; the op head-concat deprecation is its own step). §3/§4/§6 staging, versioning
shims, and gates are RETAINED BELOW AS THE REASONING RECORD ONLY — they document why the
enrichment inputs are requirements and what the migration would have cost on a live lineage.

---

## 1. What exists today (v49, `gen3_pointer_head_v1`)

`PointerActionHead` (`src/agents/model/features_extractor.py:1600`) scores each action from the
token of the entity it selects and emits a **[B,11] DELTA** that
`Gen3DualHeadMaskablePolicy._get_action_dist_from_latent` (`src/agents/model/policy.py:101`)
adds to the flat head's logits — the single funnel all three logit sites pass through, so
rollout/epoch-recompute/eval always agree.

Per action block:

| Action | Scored from | Where the token comes from |
|---|---|---|
| move k (6..9) | the move at **REQUEST slot k** | `PokemonEncoder.last_move_tokens` `[B,12,4,32]` (stashed pre-flatten at `features_extractor.py:715`), permuted by `_request_order_move_tokens` (`:1575`) — by MOVE-NUM IDENTITY, never position |
| switch j (0..5) | our-team token j | `our_team_out` `[B,6,128]`, post-transformer (and post-reattend when on) |
| struggle (10) | the context vector | `tanh(ctx_proj(our_active_refined))` |

Context = `our_active_refined` (128-dim). The three scoring Linears are **zero-init** (delta ==
0 at step 0, ON byte-identical to baseline; protected from SB3's ortho-init clobber by the M1
guard `restore_identity_init()` — the protected set is captured by observation, so these are
covered automatically). Structural bool `pointer_head` gated in `check_compatible` (v49);
tests in `pointer_head_test.py` + the real-policy identity pin in `identity_init_test.py`.

**What v49 fixed structurally:** F2 (switch logits read from a permutation-invariant CLS pool —
a bench mon's token could never reach its own switch logit) and the `ordering_integrity.py` bug
class (extractor reads moves sorted-by-id, actions use request order — the permutation now
happens once, by identity, and a misaligned logit is unrepresentable).

## 2. Gap analysis — why the delta head as-built may not WORK

The delta head competes against a flat head with a strictly richer input. Ledger **P1**
measured the op's head-concat as the policy's *largest single dependency* (zeroing the block:
masked KL 0.9385 all-actions / 0.4948 moves) — and that block reaches ONLY the flat head. Each
gap below is a piece of information the flat path has that the pointer path lacks; while any of
them stands, gradient descent has every reason to keep routing the decision through the flat
head and leave the delta near 0 ("didn't pay" would be the *predictable* outcome, not evidence
against pointer heads).

- **G1 — move tokens are board-blind.** The 32-dim move token is built PRE-transformer from:
  move ⊕ type embeddings, the obs remnants (BP/secondary/recoil/category/PP/accuracy), known,
  a small context slice (hp/turn/weather/fainted/spikes), per-move matchup ×6 + validity, HP
  probs, and (when on) the `MoveLatentEncoder` latent — then within-MON move self-attention
  only. It has *type-chart* awareness but has never seen the refined board: no believed opp
  spread, no boosts/burn/screen-adjusted damage, no P(KO). The flat head sees all of that
  through the op concat.
- **G2 — the op's per-move physics never reaches the move scorer.** `_outgoing_block`
  (`damage_op.py:750`) computes exactly the per-action cells the move logits need —
  `[low,high,crit,pko]` per REQUEST slot k (it reads `ctx.our_active_req_move_ids`, the same id
  source the pointer permutation matches against), plus the v27 status-landing block and the
  v24 outgoing secondaries — but they are delivered as a flat post-pool concat into the
  projection, where slot-k physics reaches logit 6+k only through two Linears and a ReLU that
  must *learn* the alignment. The pointer head is the lossless per-action route
  (`next_run_plan`: "without it the concat is the only lossless path from move-k physics to
  logit 6+k") — v49 built the route but did not connect the physics to it.
- **G3 — switch tokens carry incoming physics only incidentally.** Per-candidate switch
  quality = "what hits ME on the way in / after I'm in" (the op's per-defender incoming rows,
  `_DMG_PER_MON` incl. the crit and CB tails) + "what do I threaten back" (the v39
  `_outgoing_attacker_matrix` per-attacker rows — built precisely for the forced-switch
  blind-offense defect). Today those reach our-team token j only via the refine-loop /
  `prefuse_proj` trunk injections (production-config dependent, and the K10 family measured
  trunk injection ≈ NULL for the *pooled* head) or not at all (v39 rides the flat concat
  only). Note the K10-null caveat cuts the other way here: with a pooled head there was no
  per-mon consumer for a per-mon injection — the pointer switch scorer is the first consumer
  that can actually exploit per-token physics. That is a hypothesis to test, not a result.
- **G4 — the context vector is poorer than the flat head's latent.** The flat logits come from
  `latent_pi` = mlp_extractor([512,512]) over the full pi projection — which includes the op
  block, the hidden-opp belief, and the FiLM/z_arch modulation. The pointer ctx is the bare
  128-dim `our_active_refined`. Whole families of decision context (team archetype, believed
  hidden mons, the global token) are invisible to the pointer scorers.
- **G5 — z_arch/FiLM conditioning bypasses the pointer path entirely.** FiLM modulates
  post-projection head features; pointer scores are computed upstream of that. If the pointer
  becomes the head, per-team conditioning (the whole v44/v46 line) silently stops applying to
  action selection.
- **G6 — no observability.** Nothing currently records how much of the logit mass the delta
  carries, so "is it paying?" cannot even be answered from a run's metrics.

## 3. Design — three stages, each independently gated

The staging follows the house deprecation playbook: **build the replacement home → prove it
carries the load → only then remove the old path.** Never delete the flat head before the
pointer path demonstrably owns the decision.

### Stage 1 — connect the physics to the scorers (`pointer_enrich`, STRUCTURAL)

Feed each scorer the op cells for *its own* action, as direct input concats (no trunk detour):

- **Move scorer k:** `move_proj([token_k ⊕ outgoing_cell_k ⊕ status_cell_k ⊕ out_sec_cell_k])`
  where the cells are sliced from `damage_block` at named offsets (the `decode_damage_block`
  layout is the SoT mirror — never hardcode indices). All three are already REQUEST-ordered
  (`gen3_op_move_align_v1`), i.e. the same order the permuted tokens are in — **alignment by
  construction, but pin it**: a throwing guard asserting both sides read
  `ctx.our_active_req_move_ids` (a unit test that scrambles the moveset and checks cell k and
  token k name the same move-num, the `move_alignment_fuzz_test` pattern at the model level).
- **Switch scorer j:** `switch_proj([team_token_j ⊕ incoming_row_j ⊕ oax_row_j])` — the
  per-defender incoming row (12 dims + the 2 CB-tail dims for mon j) and, when
  `damage_matrices_outgoing_all` is on, the v39 per-attacker row (16 cells + p_outspeed_j +
  alive_j). Zeros when `damage_op` is off (widths fixed by the build-time toggle set, the op's
  own convention).
- **Struggle:** unchanged (ctx only).

Properties: purely wider scorer inputs → **STRUCTURAL bool** `pointer_enrich` in
`check_compatible` (requires `pointer_head`; the op cells additionally require `damage_op` —
without it the enrich concat is all-zeros and the flag should `parser.error`, matching the
house "no silent no-op" rule). Scorers stay zero-init ⇒ OFF/ON byte-identical at step 0, warm-
startable, M1-guard-covered automatically. Cost: a handful of narrow slices + the same three
Linears — order ~10 extra aten calls at B=1 (~4 µs against the 4.6–6.5 ms forward; noise).

### Stage 2 — upgrade the decision context (`pointer_ctx`, STRUCTURAL, string)

Options for closing G4/G5, in increasing order of coupling:

- **`active` (today):** `our_active_refined`. Baseline.
- **`assembled`:** ctx = a small Linear over the *pre-projection* `pi_combined` (the assembler
  output: pools + active ctx + op block + belief). Gets everything the flat head sees before
  the mlp; independent of SB3 internals. **Recommended.**
- **`latent`:** ctx = `latent_pi` itself (the policy-level pointer — score in
  `_get_action_dist_from_latent` from (latent_pi, stashed tokens)). Richest (post-FiLM,
  post-mlp) and the natural end-state shape, but it moves scorer forward passes into the
  policy class and makes the extractor's stash a hard interface — more surgery, defer until
  Stage 3 decides the flat head's fate.

For G5 specifically, `assembled` is *not* sufficient (FiLM applies after the projection). Two
cheap fixes, pick by A/B: (a) concat `last_zarch` (detached, the ZArchEncoder convention) into
ctx; (b) a third zero-init FiLM generator over the pointer hidden layer (`film_ptr`, the exact
v44 pattern). (b) is the principled one — per-team strategy plausibly lives exactly at "which
action do I prefer", the thing FiLM-on-heads was built for.

### Stage 3 — replace the flat head (`pointer_head_full`, FORWARD-BEHAVIOR)

When Stages 1–2 measurably pay (gates in §6), make the pointer the head:

```
logits[0:6]  = switch_scores      (pointer only)
logits[6:10] = move_scores        (pointer only)
logits[10]   = struggle_score     (pointer only)
```

Mechanics: the interception point stays `_get_action_dist_from_latent` — in `full` mode it
*replaces* `dist.distribution.logits` with the pointer logits instead of adding to them.
`action_net` keeps existing (SB3 `_build` creates it; deleting it fights the framework) but is
detached from the loss — freeze its params (`requires_grad_(False)`) so the optimizer skips
dead weights. Same state_dict either way ⇒ this is a **forward-behavior toggle** (the
`attend_unrevealed_opponents` class): gated in `check_compatible` (a frozen opponent's forward
differs under it, so it must gate every load, not resume-only).

**Warm-start problem (the real cost of `full`):** flipping delta→full on a trained checkpoint
is a policy discontinuity — the flat logits vanish and the policy collapses to whatever the
pointer path alone encodes. Three migration options:

1. **Fresh run** (rapid-iteration default). Cold start is well-behaved: zero-init scorers ⇒
   all-zero logits ⇒ uniform-over-legal, exactly a fresh policy's natural state.
2. **Anneal:** `logits = pointer + λ·flat`, λ: 1→0 over `--pointer-flat-anneal-steps` (a
   training-only schedule, `vf_coef`-class resume hparam). Continuous, but λ is a moving
   objective for PPO's ratio — keep λ FROZEN within each rollout+epoch cycle (update at
   rollout boundaries only), or the ratio numerator/denominator see different policies.
3. **Self-distill:** run the delta-mode checkpoint as teacher, KL the full-mode student on
   rollout states (the exploiter-distill machinery exists). Cleanest, costs a distill phase.

Recommend (1) for the first measurement, (2) as the migration tool if `full` must land on the
production lineage.

**What `full` buys beyond hygiene:** it is the prerequisite for the op head-concat
deprecation (the anti-accretion endgame) — once per-action physics reaches per-action logits
through the pointer path, the flat op concat can go through the mask-A/B → delete playbook,
and with it the projection-width treadmill. It also makes the action head *sharable* — one
scoring function instead of 11 learned rows — which is the entity-graph (`design_entity_graph.md`)
readout shape.

## 4. Versioning & threading (the house checklist)

- `pointer_enrich` (bool, STRUCTURAL — widens `move_proj`/`switch_proj` in_features) and
  `pointer_ctx` (string {active, assembled, latent}, STRUCTURAL — changes `ctx_proj`
  in_features) and `pointer_head_full` (bool, FORWARD-BEHAVIOR): all three gated in
  `check_compatible`; `MODEL_CONFIG_VERSION` +1 each (or one bump if landed together) with
  `_migrate_config` `setdefault`s (`False`/`"active"`/`False`). NO `ARCH_SIGNATURE` bump —
  every OFF combination reproduces v49/v50 byte-for-byte.
- Requires-chains enforced at CLI (`parser.error`) + extractor (`ValueError`):
  `pointer_enrich` ⇒ `pointer_head` + `damage_op`; `pointer_ctx != active` ⇒ `pointer_head`;
  `pointer_head_full` ⇒ `pointer_head` (and in practice `pointer_enrich` — refuse `full`
  without it, since a physics-blind full head is a strictly worse policy than the flat one).
- Thread through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` +
  BOTH `extractor_kwargs` sites (trainer + opponent-load) — the v24 lesson (a toggle that
  misses the opponent path FATALs self-play on its own sentinels).
- Optimizer: new Linears are appended params — the name-keyed momentum remap
  (`_validate_or_reset_optimizer_state`) handles resume either way, but keep append-last
  hygiene.
- `--compile-extractor` covers frozen OPPONENTS: opponents load with their own saved toggles,
  so a pointer-on opponent must compile — add the pointer path to
  `species_posterior_compiles_test.py`'s production-arch config once it enters a production
  lineage (slices + Linears + tanh: no exotic ops expected, but the M-series lesson is to
  *measure*, not assume).

## 5. Observability (fix G6 before judging anything)

Add per-cycle metrics (instrumented_ppo, TB + launcher):

- `pointer/delta_l1_moves`, `pointer/delta_l1_switch`, `pointer/delta_l1_struggle` — mean |delta|
  per block (is the head alive, and where).
- `pointer/logit_share` — |delta| / (|flat|+|delta|) over legal actions: the "who owns the
  decision" dial; the Stage-3 gate reads this.
- `pointer/argmax_flip_frac` — fraction of states where adding the delta changes the masked
  argmax (the behavioral read; 0 forever = the delta is decorative).
- Prober: surface the per-action delta in `analyze` (decode which entity's token drove a
  disagreement — the F2 forensic finally becomes visible per-decision).

## 6. Gates (each stage needs its own kill criterion)

1. **v49 delta as-is (already runnable):** train with `--pointer-head`; gate =
   `pointer/logit_share` rising off 0 AND `argmax_flip_frac` > noise by mid-run. If flat-lined,
   that is *consistent with the G1–G5 deficit prediction* — proceed to Stage 1 rather than
   killing the lever (but say so honestly in the ledger).
2. **Stage 1 enrich A/B:** same seeds ± `pointer_enrich`. Primary: forced-switch crater share
   (`falsify-scan`, the v39 defect this most directly targets) + eval ELO; secondary:
   `pointer/logit_share`. Kill if enrich moves neither the share nor the crater bracket.
3. **Stage 2 ctx A/B:** `active` vs `assembled` (+FiLM arm). Cheap — decide on logit_share +
   ELO.
4. **Stage 3 full:** only after 1–2 pay. Gate = full-mode fresh run reaches the delta-mode
   run's ELO band (non-inferiority) — the *win* at this stage is structural (deprecation
   unblocked), not strength; then run the op head-concat mask A/B that P1 says is currently
   un-passable.

## 7. Test plan

Extend `pointer_head_test.py`:
- Enrich alignment (the load-bearing one): scrambled moveset — assert outgoing cell k and
  permuted token k resolve to the same move-num; assert the incoming row j feeding switch
  scorer j is defender j's row (perturb one defender's HP, only logit j's input moves).
- Enrich zero-safety: `damage_op` block zeroed (no opp active) ⇒ enrich contributes zeros, no
  NaN.
- ctx modes: shape checks + `assembled` sees the op block (perturb block ⇒ ctx changes).
- Full mode: logits == pointer scores exactly; flat `action_net` gradient is None; masking /
  log_prob / entropy consistency through the funnel (all three logit sites agree).
- Identity-at-init THROUGH a real MaskablePPO policy for every new zero-init (M1 rule —
  `identity_init_test.py` pattern, not bare-extractor-only).
- Anneal: λ frozen within a rollout+epoch cycle (assert the ratio's two forwards use one λ).
- Version gates + migration defaults (the v49 test shape).

## 8. Stage 4 — removing the legacy flat path outright (no flat approach at all)

**OWNER DECISION (2026-08-03): this is the destination, not an option.** The end-state is a
position-EQUIVARIANT action pathway — one shared scoring function per entity token, no
positional logit rows anywhere — so slot identity is content the token carries, never
something a weight row memorizes (the same weight-sharing argument as the shared
`PokemonEncoder`: 6 switch rows that each rediscover "a team slot" become 1 scorer with 6×
the effective data). Stages 1–3 remain the *sequenced route* there (each gate protects
against deleting the strong path while the pointer path is still information-starved), but
their pass/fail decides pacing, not whether Stage 4 happens. Note the equivariance claim
needs BOTH this stage AND the op head-concat deprecation (§scope fences): the op's per-slot
blocks at fixed projection offsets are the other position-sensitive path, and the prev-turn
action-mask obs block re-homes as token legal-now/recency features (entity graph E3/E7).
Deliberate symmetry breaks that STAY: `BeliefSlots`' distinct per-slot unknown-mon tokens
(slots with no content need injected identity or they collapse under attention) and request
order on the active's 4 moves (action slots are DEFINED by it; the pointer permutation pins
it by move-num identity).

Stage 3 keeps `action_net` frozen-but-present. Full removal has **two distinct levels**, and
conflating them is the trap:

**Level A — remove it from the MODEL (new runs carry no flat params).**

- **Policy surgery** (`policy.py`): SB3's `ActorCriticPolicy._build()` creates
  `action_net = proba_distribution_net(latent_dim_pi)` and then builds the optimizer over
  `self.parameters()`. Overriding `_build` wholesale means copying SB3 internals (fragile);
  instead: call `super()._build()`, replace `self.action_net` with a **raising stub** (never
  Identity — a silent fallback to `action_net(latent_pi)` would be a garbage policy, not an
  error), and **rebuild `self.optimizer`** so the dead params don't ride in a param group.
  `_get_action_dist_from_latent` builds the distribution directly:
  `self.action_dist.proba_distribution(action_logits=pointer_logits)`.
- **The pi tower stays.** `pre_proj_norm → projection → ReLU → mlp_extractor.forward_actor`
  is NOT the flat approach — the flat approach is the 11 positional rows at the end. The tower
  is repurposed as the pointer **context encoder** (`pointer_ctx=latent`, the natural
  end-state): it is the only path that delivers the op concat / beliefs / FiLM to action
  scoring, and it keeps `film_pi`'s application point alive. Deleting it too would re-open
  G4/G5.
- **Consumers audit (done):** no non-test code reads `action_net` directly — exploiter
  distill (`_last_pi_distribution`), OPD, the search teacher, the prober, and RLPlayer all
  consume the distribution, which is built at the one funnel. The surgery is contained to
  `policy.py`.
- **Versioning:** a structural bool `flat_action_head` (default **True**; `_migrate_config`
  `setdefault(True)` so every existing checkpoint reads as flat). `False` = the pointer-only
  lineage; gated in `check_compatible` (the state_dict differs — `action_net.*` keys absent).
  **Do NOT bump `ARCH_SIGNATURE`**: the stable cross-run-opponent compat gate is
  arch_signature-only, and bumping it would orphan the entire opponent ecosystem in one move.
- **Crossing the boundary:** a resume/warm-fork across `flat_action_head` True→False is a
  state_dict mismatch. Either (a) a fresh run (zero-init scorers ⇒ uniform-over-legal start,
  well-behaved), or (b) a one-time **strip migration**: take a Stage-3 annealed checkpoint
  (λ→0, flat functionally dead), drop the `action_net.*` keys, rewrite `model_config.json`
  with `flat_action_head=False`, save as the new lineage root. (b) is safe *only* post-anneal
  — stripping a live flat head is a policy lobotomy.
- **Sequencing consequence:** the warm-fork exploiter recipe (`--model=distilled-generalist`)
  breaks across the boundary until a pointer-only distilled generalist exists. Plan one
  generalist retrain/distill BEFORE expecting exploiter forks on the new lineage.

**Level B — remove the flat CODE from the codebase.**

Blocked on the checkpoint ecosystem, not on engineering: `arch_toggles_from_model` rebuilds
every frozen opponent / pool snapshot / distill teacher / prober target with **its own saved
toggles**, so the flat build path must remain constructible for as long as ANY
`flat_action_head=True` checkpoint is still loaded by anything. Deletion criterion: a scan of
`models/` (+ the self-play pools and the stable-opponent registry) finds no pre-pointer
checkpoint still referenced. Until then the flat path lives behind the toggle — dead for new
runs, buildable for old ones. This mirrors how every deprecation here works (build the
replacement → starve the old path → delete when no consumer remains).

**What does NOT go away with the flat head (scope fences):**

- `ordering_integrity.py` — an **obs-side** guard (it reorders/validates the prev-turn
  move-validity bits the extractor consumes in sorted-slot order). The pointer head dissolves
  the *head-level* alignment class; the obs-level one retires only when the per-move obs
  blocks re-home onto entity tokens (`design_entity_graph.md` E3).
- The request-ordered obs blocks (`gen3_op_move_align_v1`) — the op's outgoing kernels AND the
  pointer permutation both key off `ctx.our_active_req_move_ids`.
- The op head-concat — its own deprecation (next_run_plan playbook: mask-A/B → delete at a
  fresh run). Stage 4 is its *prerequisite*, not its execution.

**Preconditions and cost, honestly:** the engineering is small (policy surgery + stub + strip
script + tests: days). The real cost is that Level A is only safe AFTER the Stage 1–3 gates
pass — P1 says the flat path currently carries the policy's largest single dependency, so
deleting it before the pointer path demonstrably owns the decision is a guaranteed regression
dressed up as cleanup — plus one fresh-run (or anneal+strip) retrain and a non-inferiority
eval on the new lineage.

## 9. Open questions

1. **Bench-mon move awareness for switches** — switch scorer j sees token j (which folds j's
   processed moves) + physics rows, but not j's moves *as addressable entities*. The
   entity-graph answer is E3 bench bags / D2 edges; out of scope here (flat enrich first).
2. **Does the move token need the transformer at all** (G1 fully) — Form-A cross-attention
   (move tokens as queries over the 12 mon tokens) is the escalation if Stage-1 concat
   enrichment under-delivers; it is deliberately NOT in this design's scope (next_run_plan
   stages it separately).
3. **Struggle/context asymmetry** — struggle is the only logit with no entity; if `full` mode
   ever misbehaves specifically on struggle turns (rare), give it the move-scorer form over a
   learned "struggle token" instead of the bare ctx.
4. **Where the pointer hidden width sits** (64 today) — probably irrelevant at these scales,
   but it becomes an arch constant (`arch_constants.py`) the moment Stage 1 lands, per the
   single-source rule.
5. **Interaction with `--damage-candidate-k`** — the enrich cells ride the op's existing
   outputs, so candidate-axis capping changes nothing here; noted only to preempt the
   question.

## 10. Summary of the changes required

| Change | Kind | Flag | Files |
|---|---|---|---|
| Op-cell enrich into scorers | STRUCTURAL | `--pointer-enrich` | `features_extractor.py` (PointerActionHead widths + forward slices), `damage_op.py` (named offsets exported), `model_version.py`, `train_rl_agent.py` (CLI + both kwargs sites), `extractor_arch.py` |
| Context upgrade | STRUCTURAL (string) | `--pointer-ctx {active,assembled,latent}` | same set; assembler stash for `assembled` |
| z_arch into pointer | STRUCTURAL | fold into `--pointer-ctx` arm or `film_ptr` | `features_extractor.py` |
| Full replacement | FORWARD-BEHAVIOR | `--pointer-head-full` | `policy.py` (`_get_action_dist_from_latent` replace + action_net freeze), `model_version.py` |
| Anneal migration | training-only hparam | `--pointer-flat-anneal-steps` | `policy.py`, `instrumented_ppo.py` (λ schedule at rollout boundaries) |
| Metrics | — | none | `instrumented_ppo.py`, launcher metric map, prober `analyze` |
| Tests | — | — | `pointer_head_test.py`, `identity_init_test.py`, compile test config |
| Flat-head removal (Level A) | STRUCTURAL (state_dict: `action_net.*` absent) | `flat_action_head=False` | `policy.py` (`_build` post-surgery: raising stub + optimizer rebuild), `model_version.py`, one-time strip-migration script |
| Flat-code deletion (Level B) | — | — | gated on a `models/` + pool/opponent-registry scan finding no `flat_action_head=True` consumer |
