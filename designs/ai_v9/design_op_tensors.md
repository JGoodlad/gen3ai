# design — `OpTensors`: one home per fact, arity as the type

**Status:** forward design, not built. Written 2026-08-08 off the 40M gen-3 measurements below.
**Owner decision:** ✅ ADOPTED 2026-08-08 (late) — "no more concat" is the next major goal,
execution starts when gen-4 (`run_20260808_212910`) completes; the §9.1 discriminating arms run
on gen-4's final checkpoint before step 3+. Recorded beside the superseded two-route
precondition in `design_generation_roadmap.md` §3.8.

**Written to be evaluated by someone without the originating conversation.** Every measured
number carries its provenance and its caveats inline; every inference is labelled as one.

> ### ⚠️ Version provenance — read first
> | | |
> |---|---|
> | Code at time of writing | **v60**, `ARCH_SIGNATURE = gen3_entity_rehome_v1` |
> | `designs/production_config.json` | **v60**, `gen3_entity_rehome_v1` |
> | **Checkpoint every measurement below was taken on** | `run_20260807_135637_gen3/final_model.zip` — **v59**, `gen3_edge_bias_trunk_v1` |
>
> The measurements are therefore **one generation behind the code**: Stage-3's entity re-home
> (`f9d09ad`, obs 2925 → 2667, derived obs blocks deleted) and the E2 active-context injection
> (`427656e`, each active mon's token owns its boosts+volatiles) both landed *after* that
> checkpoint was trained. Neither is expected to change the conclusions — see below — but a
> reviewer should treat every number as measured on the *previous* trunk.
>
> **What was re-verified against v60 code on 2026-08-08 and is UNCHANGED:** the op block's
> `out_dim` (**660**) and every sub-block offset; the header decomposition
> (`_DMG_IMX_HEADER` = 51 = latent 32 + 3 + effect 6 + secondary 10); `_PTR_MOVE_CELL` = 13 and
> `_PTR_SWITCH_CELL_IN` = 15; the E4-seat/header identity (§1); and `_incoming_rolls`'s two call
> sites (§1). **Stage 3 re-homed the OBSERVATION, not the op's output** — which is precisely the
> gap this design addresses, and is evidence the two are separable work.

---

## 0. The goal

> **Give every fact the `DamageOperator` computes exactly ONE structural home, typed by its
> arity, and make every consumer read a VIEW of that home rather than a slice of a flat vector.**

The concrete objectives, in priority order:

1. **Eliminate duplication without deleting information.** Today several facts are delivered
   twice (once as flat concat dims, once as entity content) and some physics is *computed*
   twice per forward. Remove the second copy, not the fact.
2. **Make the sorting-rule violation unrepresentable**, the way the pointer-native head made
   sorted-vs-request ordering bugs unrepresentable — by making the axis the type rather than a
   convention.
3. **Collapse the hard-max / OA1 / PV design space into one parameterised call site**, so those
   stop being three features and become three settings of one operator.
4. **Dissolve the op head-concat** rather than deciding to delete it: in the target design it
   has no view, so there is nowhere for it to be.

**Non-goal:** deciding which fields are expendable. This design drops **no** game facts. That is
deliberate — see §7.

---

## 1. The problem

`DamageOperator.forward` emits ONE flat block (`out_dim` = **660** under the production config)
and has **five** consumers. A flat vector cannot serve five different shapes, so three of the five
**recompute** rather than slice:

| consumer | how it gets its data today | |
|---|---|---|
| `ProjectionAssembler` — "the concat" | slices the flat 660 into BOTH projection heads (`features_extractor.py:2130-31`) | |
| `prefuse_proj` | `Linear(_DMG_PER_MON → D_MODEL)` of the per-mon row → our 6 role tokens (`:2709`) | |
| `pointer_cells` | slices move cell (13) + switch cell (15) → pointer head | |
| `reattend_layer` | slices the per-mon rows (`features_extractor.py:3607`) — **`damage_reattend = False` in production, so INERT** | |
| **E4 threat seats** | **RECOMPUTED** — `refine_candidates` + `latent_table` | ⚠️ |
| **d3 / s3 edge biases** | **RECOMPUTED** — `pairwise_incoming` → `_incoming_rolls` | ⚠️ |

Outside the forward, two more read the flat form and are unaffected by this design because it
keeps `flatten(OpTensors)` as a serialization: the prober (`main/prober/model.py` →
`decode_damage_block`) and `edge_ablation_audit.py`.

**Code-verified duplications** (2026-08-08, against `src/agents/model/`):

- `TeamTransformer`'s E4 seat is
  `threat_seat_proj(cat([latent_table[idx], w, MOVE_ACCURACY[idx], MOVE_PHYS[idx]]))` =
  `Linear(MOVE_LATENT_DIM + 3 → D_MODEL)` (`features_extractor.py`). That input is
  **bit-for-bit the first 35 dims of the incoming matrix's 51-dim per-move header**, from the same
  `refine_candidates(k=K)` selection — the extractor stashes `last_cand` precisely so seat *c* and
  D3's bias row *c* name the same move.
- `_incoming_rolls` is called from **two** sites per forward — `discrete_incoming`
  (`damage_op.py:1473`) and `pairwise_incoming` (`damage_op.py:2301`) — re-deriving
  `[high, pko, eff]` that the flat block's `_incoming_matrix` already carries. Its own docstring
  says "the same `_incoming_rolls` physics … as `discrete_incoming`".

**The flat layout is what causes the recomputation.** Each consumer needed a shape the flat
vector could not express, so it built its own.

---

## 2. The evidence base

### 2.1 The concat's dimensional accounting (code-derived, exact)

`out_dim` = 660. Offsets computed from the live constants, 2026-08-08:

| # | sub-block | offset | width |
|---|---|---|---|
| ① | incoming per-mon — 6 × `[phys(low,high,crit,pko,acc), spec(…), p_outspeed, provenance]` | `[0..71]` | 72 |
| ② | Choice-Band tail — `phys_high_cb ×6`, `phys_pko_cb ×6`, `p_cb` | `[72..84]` | 13 |
| ③ | outgoing single-active — 4 moves × `[low,high,crit,pko]`, `p_outspeed`, 4×7 secondary | `[85..129]` | 45 |
| ④ | status-landing — `P(lands) ×4`, `known ×4` | `[130..137]` | 8 |
| ⑥ | `incoming_matrix` K=6 — headers `[138..443]` (6×51) + cells `[444..659]` (6 mons × 6 moves × 6) | `[138..659]` | 522 |

**523 of the 660 dims are concat-exclusive** — the other 137 are also delivered by
`pointer_cells` / `prefuse_proj`. Of those 523, **522 are ⑥** (the odd one out is ③'s
`p_outspeed` at offset 101).

### 2.2 Where the dependence actually lives (NEW measurement)

`src/agents/model/op_block_split_audit.py`, results at
`designs/research_state/measurements/gen3_op_block_split_40M.json`. gen-3 `final_model.zip`
(40M), 6000 states, CPU, idle box (load 0.64/16).

**Which statistic — read this before the table.** Each arm is run two ways. **Zero** replaces the
columns with 0: off-manifold, so it measures reliance *plus* the shock of an impossible input.
**Shuffle** replaces them with another state's values — same width, same marginals, state
specificity destroyed. Shuffle is group **permutation importance**, it stays on-manifold, and it
is the estimator the rest of this project quotes ("shuffle-controlled flips"):
`shortcut_learning_and_feature_delivery.md` Part 6 records the concat as *"24.15% zeroed but
16.27% shuffled, i.e. a third of the naive number was the mean shift."* **The dependence column
below is therefore the shuffle arm.** The zero arm is shown beside it for reference; their
*difference* is the zero-ablation artifact, and is not a dependence measure.

| arm | width | **shuffle = importance** | zero | \|dV\| shuf |
|---|---|---|---|---|
| `FULL_CONCAT` (anchor) | 660 | **20.18%** | 24.98% | 2.45 |
| `in_matrix` (anchor) | 522 | **16.32%** | 22.00% | 1.55 |
| **`imx_HEADERS`** | 306 | **14.88%** | 21.22% | 1.36 |
| **`imx_CELLS`** | 216 | **5.77%** | 5.73% | 0.83 |
| `hdr_latent` (= E4 content) | 192 | **15.07%** | 20.63% | 1.34 |
| `hdr_w_acc_phys` (= E4 content) | 18 | 1.67% | 4.07% | 0.23 |
| `hdr_effect` | 36 | 2.28% | 1.87% | 0.24 |
| `hdr_secondary` | 60 | **0.18%** | 0.15% | 0.04 |
| `cell_LOW_CRIT` | 72 | 2.80% | 3.00% | 0.43 |
| `cell_high_pko` | 72 | 3.17% | 2.63% | 0.38 |
| `cell_mult_status` | 72 | 2.88% | 2.42% | 0.45 |

The three cell arms are **width-matched at exactly 72 dims**, so they are directly comparable —
a control the block-level arms never had.

**Reading:**

1. **⑥'s dependence is in the per-move HEADER, not the pair cells** — 14.88% vs 5.77%, a factor
   of 2.6.
2. **The header dependence IS the move latent**, which is *already E4 seat content*:
   `hdr_latent` alone reads 15.07%, statistically the whole of `imx_HEADERS` (14.88%). So the
   concat's largest single sub-block is re-delivering something the entity stream already carries.
3. **`hdr_secondary` (60 dims) is dead** at 0.18%, and `hdr_effect` (36 dims) is near-dead at
   2.28% — both below every 72-dim cell arm despite comparable width.
4. **The pair cells are secondary, but they are NOT noise.** 5.77% across 216 dims is the same
   order as `in_permon` (4.52% @9.6M, 6.60% @40M) — a block this project treats as real but
   subordinate. Among the three width-matched channels the spread is 2.80 / 3.17 / 2.88, i.e.
   **flat**: no cell channel stands out, which is the useful negative result.

Per-dim, `[w, acc, is_phys]` remains the densest header field (0.093 %/dim vs the latent's
0.078 %/dim), but only by ~1.2× — not the 4.6× an earlier draft of this table reported.

### 2.3 The pre-registered coverage probe (pre-existing, 40M)

`models/run_20260807_135637_gen3/coverage_probe_gen3_40M.json`, 5743 states. The gate recorded in
`design_conditional_opponent_cells.md` §2b.4 was: *decodable at good r² ⇒ cross-pair reasoning
already happens and PV buys little; at chance ⇒ that is the gap.*

| quantity | π | **vf** | shuffled control (vf) |
|---|---|---|---|
| `n_threatened` | r² 0.685 | **r² 0.798** | −0.034 |
| `safe_pivot_exists` | AUC 0.974 | **AUC 0.970** | 0.525 |
| `best_move_breadth` | r² 0.645 | **r² 0.752** | −0.030 |
| `act_threat` (magnitude to our active) | r² 0.690 | **r² 0.330** | −0.044 |

**The gate fired against PV.** Every cross-pair quantity is decodable from *both* heads, controls
at chance. The critic is not missing board structure.

The one genuine critic deficiency is `act_threat` — a **magnitude**, not a cross-pair quantity —
and the routing explains it: threat magnitude reaches the policy through the pointer switch cell
(lossless, per-action) and the critic only through pooling and the concat.

### 2.4 The end-of-run edge audit — the edges nearly caught up

`designs/research_state/measurements/gen3_edge_family_audit_40M.json` (promoted into
`measurements/` 2026-08-08; it had only ever lived in the run directory, so the viewer's
gen-3@40M overlay had no per-family rows). Same checkpoint, same 6000 states.

| gen-3 | all-edges-off | concat | gap | \|dV\| edges | \|dV\| concat |
|---|---|---|---|---|---|
| @9.6M | 13.9% | 23.6% | 9.7 pp | — | — |
| **@40M** | **24.65%** | **27.45%** | **2.8 pp** | **2.268** | **5.355** |

Over training the edge system gained **+10.75 pp** of dependence while the concat gained
**+3.85 pp**. The concat still wins — a fourth replication — but by 2.8 pp, not the 9.7 pp the
mid-curve read implied. **Any argument resting on the 9.6M gap is quoting a race in progress.**

The critic gap does **not** close on the same schedule: `|dV|` 5.355 concat vs 2.268 all-edges
(≈2.4×). That asymmetry — policy dependence converging, critic dependence not — is the same
readout story §3.3 tells, seen from the edge side, and it is *inferential* support for
interpretation (a) below (the edges absorbed the pair-cell role) over interpretation (b).

### 2.5 ⚠️ Caveats on the above — read before relying on any of it

1. **State distribution.** `collect()` takes trace files in sorted glob order and stops at the
   cap, so all 6000 states came from `eval_traces/step_10000032/` — i.e. the **40M model was
   evaluated on states drawn from a 10M-era policy's trajectories**. The committed 9.6M/40M
   probes share this helper and therefore this defect. It spans 8 opponent buckets
   (`sentinel_0/1/2`, `random`, `aggressive`, `aggressive_v2`, `heuristic`, `heuristic2`), so it
   is a pool average, not one opponent. **FIXED 2026-08-09 (`gen3_audit_state_sampler_v1`,
   `agents/model/audit_states.py`)**: both probes now sample one file per (step, opponent)
   bucket per pass with per-step row quotas, seeded/deterministic, and write per-step /
   per-opponent sampled counts into their provenance (verified on gen-4's tree: 125 states ×
   12 step dirs, all 12 opponent buckets). The numbers ABOVE still carry the old defect —
   they were not re-measured; every NEW measurement (the §9.1 arms, the step-3 acceptance
   read) uses the fixed sampler.
2. **Pooled means hide rare-but-decisive fields.** A mechanic live in ~3% of states that flips 15%
   of actions *within* them contributes **0.45%** to the pooled mean — inside the noise of these
   arms. `hdr_effect` at 2.28% pooled is fully consistent with `hazard` being decisive in every
   Spikes matchup. **The field-live stratification has not been run.**
3. **Generalist only.** There is no gen-3 exploiter checkpoint, so specialist dependence on these
   fields is untested. A field a generalist ignores may be load-bearing for an archetype
   specialist.
4. **Ablation-KL is marginal, at fixed weights**, measured on a model *trained with* the concat
   present. "The head does not lean on it" is not "a model trained without it would be as strong."
5. **A retraction — of this document's own first draft.** The 2026-08-08 draft claimed the
   `in_matrix` 16.27% / `in_permon` 4.52% figures quoted across the other docs were "the controls,
   not the dependence," and substituted a `net = zero − shuffle` statistic. **That was wrong, and
   it is withdrawn.** The shuffle arm *is* the dependence estimate — group permutation importance,
   on-manifold, exactly what the probe's own docstring describes ("same width, same marginals,
   state-specificity destroyed"). `zero − shuffle` measures the *zero-ablation artifact*, not
   reliance, so it is not a "width-fair statistic" and should not be quoted as one. §2.2 is
   restated on the shuffle arm; the qualitative conclusions (headers ≫ cells, the header
   dependence *is* the E4 latent, `hdr_secondary` dead) **all survive the correction**, and one
   does not: the ranking *among* the three cell channels was an artifact — under permutation
   importance they are flat (2.80 / 3.17 / 2.88), and `cell_high_pko` is the largest of the three
   rather than the smallest. The other docs' numbers stand as written; nothing needed fixing
   there. What *is* worth recording is the drift over training on the correct statistic:
   `in_permon` 4.52% → 6.60% and `in_matrix` 16.27% → 18.32% (9.6M → 40M), i.e. the gap narrowed
   from 3.6× to **2.8×** but did not close.

**Caveats 1–3 are exactly why this design drops no fields** (§7).

---

## 3. The design

Replace the flat emission with a typed one whose axes encode arity.

```
OpTensors — named tensors, explicit axes:

  board     [B,       n0 ]   arity 0            p_cb, board-level scalars
  opp_move  [B, K,   n1m ]   arity 1 (opp move) latent(32), w, acc, is_phys, effect(6), secondary(10)
  our_move  [B, 4,   n1o ]   arity 1 (our move) our move's own facts, request-slot order
  our_mon   [B, 6,   n1d ]   arity 1 (our mon)  ← the REDUCTION of pair_in over the move axis
  pair_in   [B, K, 6, np ]   arity 2            low, high, crit, pko, type_mult, status_lands
  pair_out  [B, 4, 6, np ]   arity 2            outgoing counterpart
```

Consumers become **views**, not slices:

| consumer | view | today |
|---|---|---|
| E4 threat seat *k* | `opp_move[:, k]` | recomputed |
| E3 move seat *k* | `our_move[:, k]` | separate path |
| d3 / s3 edge bias | `pair_in` → `Linear(np, 2·n_heads)` | recomputed |
| `prefuse_proj` token injection | `our_mon` | flat slice |
| pointer switch cell *j* | `our_mon[:, j]` | flat slice |
| pointer move cell *k* | `our_move[:, k]` | flat slice |
| `reattend_layer` (INERT today) | `our_mon` | flat slice |
| **the flat concat** | **no view exists** | 660 dims into both heads |
| prober / `decode_damage_block` | `flatten(OpTensors)` | the same flat vector, now a *serialization* |

### 3.1 What this makes unrepresentable

A pair fact cannot be placed on a token, because `[B,K,6,np]` does not fit a `[B,6,d]` seat
without an **explicitly named reduction**. The axis is the type. This is the same class of move as
the pointer-native head, which made the sorted-vs-request ordering bug unrepresentable rather than
merely tested-against.

Corollary: the "which facts live where" sorting rule stops being a documented convention that
review must enforce and becomes a shape error.

### 3.2 The reduction site — the unification

Today the arity-2 → arity-1 collapse is a hard-coded `amax` over the candidate axis, buried in the
forward: per (defender *j*, channel *c*), `max_c (w_c · value_c)`. In the target design it is one
declared line:

```
our_mon.threat = REDUCE(pair_in, over=MOVE_AXIS, how=…)
```

and `how` is where three separately-designed roadmap items actually live:

| `how` | is |
|---|---|
| `hard_max` | today's behaviour (`amax` of the belief-weighted value) |
| `belief_weighted_mean` | the un-maxed marginal |
| `conditional(λ)` | **OA1** (`design_conditional_opponent_cells.md` Part 1) |
| `learned_attention(k seeds)` | **PV** (ibid. Part 2b) |

They are not three features; they are **three settings of one operator at one call site**, each an
A/B flag with no new plumbing. "Is PV worth building?" becomes "which reduction wins?", measured at
one place with one harness.

Two properties of today's `amax` worth recording, because they are easy to misread:
- It maxes the **belief-weighted** value, `max_c(w_c · v_c)` — neither a max damage nor an
  expectation. A 0.30-belief 90% move (0.27) loses to a 0.90-belief 40% move (0.36).
- `acc` and `provenance` are gathered **at that argmax**, so any soft reduction must decide what
  they mean. The natural answer is to fold accuracy into the cell and let the reduction handle it —
  which is what `pair_in` already does.
- The historical justification for the hard max was avoiding soft-max dilution over a ~400-wide
  candidate sweep. **That sweep is now K=6**, so the dilution objection is ~6-way, not 400-way. It
  is much weaker than when it was written. *(Inference, not measurement.)*

### 3.3 What it does NOT solve — the readout

Giving each fact one home does **not** give the heads more bandwidth, and the concat *was* the
critic's only wide un-pooled window. §2.3's `act_threat` split (π 0.690 / vf 0.330) is the measured
shape of that. So the design must also declare a **head-input contract**:

- **policy** — pools + pointer cells (per-action, lossless; already entity-native)
- **critic** — pools + *k* seed reads over `our_mon`, which is only expressible *because*
  `our_mon` is a named arity-1 tensor a seed query can attend

Relevant prior: ledger **P3** refuted *widening* `value_pooled` (effective rank 3–4; outcome AUC
0.833 vs the policy's 0.835 on 384 dims). P3 tested **width**; it never tested **multiplicity** —
`value_pooled` is *one* learned query, and *k* seeds is a different axis, linear in *k*, no new
seats. Failure mode is seed collapse, with the `z_arch` precedent (≈2/3 of energy in one shared
direction, VICReg covariance never wired); monitor pairwise seed-query cosine **and** per-dim
variance of the k outputs, floor with a VICReg-style variance term if it collapses.

⚠️ If a per-candidate value read is ever wanted, it must be an **auxiliary head supervised on
realized returns**, never the PPO baseline — a `b(s,a)` baseline is biased without a correction
term and Tucker et al. 2018 (*The Mirage of Action-Dependent Baselines*) found the corrected
versions' reported gains largely failed to replicate.

**This must land WITH the concat's removal, not after it.**

---

## 4. Migration

| step | what | provable? |
|---|---|---|
| 1 | `OpTensors` becomes the op's internal return; `damage_block = flatten(OpTensors)` kept as a derived serialization. Nothing downstream changes. | **byte-identical** — same gate the op-split refactor used (pi/vf + raw block byte-identity, unchanged `state_dict` keys, the constructed-scenario physics oracle, full suite) |
| 2 | Re-point consumers to views one at a time. **E4 first** — biggest duplication and provably the same tensor. Then d3/s3 (`pair_in` computed once). | byte-identical, or explicitly not, per consumer |
| 3 | Drop the flat serialization from the **forward**; keep it only for `decode_damage_block` / prober. | the concat is gone *by construction*, not by decision |
| 4 | Land the head-input contract (§3.3 critic seed reads). | same generation as 3 |
| 5 | Turn the reduction knob — A/B `how` (§3.2). | one flag, one site |

Steps 1–2 are **provable refactors**, so every speculative risk is isolated to 3–5. That is the
attribution discipline `design_generation_roadmap.md` §1 asks for, and it is the opposite of
today's situation where the concat decision is entangled with OA1, PV and Stage 3 at once.

Step 2 also yields a real compute refund: `_incoming_rolls` stops running twice per forward, and
⑥'s cell block stops being *duplicated* into a consumer that already receives the same physics
through the edges — the cells keep exactly one home, they are not discarded (§7).

---

## 5. Relationship to the existing docs

| doc | relationship |
|---|---|
| `design_generation_roadmap.md` §3.8 (owner amendment) | **Amended by this.** See §8. |
| `design_conditional_opponent_cells.md` | OA1/OA2 survive as **reduction settings** (§3.2), not as separate cell families. Its §2b PV motivation is refuted by §2.2 + §2.3. |
| `design_entity_graph.md` | Unchanged — this is about *how the op's facts reach* the entities, not which entities exist. |
| Stage 3 (schema + obs re-home) | **SHIPPED** (`f9d09ad`, v60 `gen3_entity_rehome_v1`) — it made the *observation* declarative and deleted its derived blocks. This design is the **same move applied to the op's OUTPUT**, which Stage 3 did not touch: `out_dim` is still 660 and every consumer still slices a flat vector. The precedent is directly usable — the schema/validated-slice-map machinery (`cb48958`, Stage-3a) is the pattern `OpTensors` should follow. |
| `ARCHITECTURE.md` §4 | Holds the current block layout + the concat-exclusive accounting. Update in the same pass as any step above. |

---

## 6. Gates

Pre-register before building, per the project's ablation discipline:

- **Step 1/2 gate — byte-identity.** pi/vf outputs and the raw op block identical; `state_dict`
  keys unchanged; `damage_op_probe_fuzz_test.py` (constructed-scenario physics oracle) green; full
  unit suite green. A refactor claiming to change nothing proves it, it does not assert it.
- **Step 3 gate — the deletion counterfactual.** Concat arm below all-edges-off on **flips AND
  `|dV|`** (the acceptance clause from the owner amendment, unchanged), measured on the
  readout-upgraded arm at 40M, with **stratified** state sampling (caveat §2.5.1 fixed).
- **Step 4 gate — seed collapse.** Pairwise seed-query cosine stays low AND per-dim variance of
  the k outputs stays non-degenerate. Report both; a collapsed seed set means k reads were paid
  for and one was received.
- **Step 5 gate — reduction A/B.** Anchored ELO generation-vs-generation, plus `act_threat`
  vf r² as the targeted diagnostic (§2.3 baseline 0.330).
- **Strength gate, overall.** Anchored ELO non-inferiority vs gen-3 at matched tranches; fix the
  margin before the run.

---

## 7. Why this design drops no fields

An earlier version of this analysis proposed deleting 312 measurably-dead dims (the 216 pair
cells, `hdr_effect` 36, `hdr_secondary` 60). That framing asked *"what can we afford to lose?"* —
a question §2.5's caveats make hard to answer honestly, because a pooled generalist mean cannot
see a field that is decisive in a narrow archetype slice.

The fields at issue are exactly the ones that define gen3ou archetypes:

- `MOVE_EFFECT_COLS = [recovery, status, phaze, boost, hazard, protect]` — near 1:1 onto the tag
  vocabulary in `data/teams/gen3_team_archetypes.json` (stall recovery, the `phaze` tag,
  Spikes/`spikes`, every setup sweeper, Protect/Wish stall).
- `SECONDARY_COLS = [par, brn, frz, slp, psn, tox, confusion, flinch, foe_statdrop, self_boost]` —
  Ice Beam freeze, Rock Slide flinch (Tyranitar/Aerodactyl), Crunch SpD drop (how Tyranitar breaks
  Suicune).

**This design keeps all of them.** They move to `opp_move` and ride the E4 seat, which the
measurement says is the most-used input in the block. What disappears is a *second flat copy* and
a *redundant physics computation* — neither of which is a fact about the game.

Note also the structural argument, independent of the measurement: `effect` and `secondary` are
**deterministic lookups keyed by move id** (from `gen3_moves.json`, identical in every state for a
given move), and the retained `latent` is a 32-dim learned embedding of move identity over ~400
moves. Encoding 6 booleans and 10 probabilities in that embedding is trivially within capacity.
The pair cells are **not** so derivable — they are `f(move, attacker stats, defender stats, boosts,
item, weather, HP)` — which is precisely why they get their own arity-2 home here rather than being
folded anywhere.

---

## 8. How this alters the existing path

The owner amendment (`design_generation_roadmap.md` §3.8, 2026-08-08) requires, before the concat
may be deleted: **OA1 (policy) + a CRITIC route (PV *or* generalized token-content injection),
both landed and audited**, accepted only if the concat arm falls below all-edges-off on flips AND
`|dV|`, with the concat surviving Stage 3 and dying last.

What this design changes:

1. **The premise is wrong.** The precondition assumes the concat carries **magnitude** that the
   entity routes structurally cannot. §2.2 says its *dominant* dependence is the move **LATENT**
   (identity) — 15.07% against the pair cells' 5.77% — and that latent is already E4 seat content.
   The largest sub-block of the concat is therefore re-delivering something the entity stream
   already carries. The problem is **readout bandwidth**, not delivery. This
   promotes confound #1 in `designs/learning/shortcut_learning_and_feature_delivery.md` Part 6
   ("a capacity-of-the-readout problem, not a delivery problem") from footnote to leading
   hypothesis.
2. **PV as specified should not be built — on ONE gate, not two.** The pre-registered coverage
   probe (§2.3) fired against it cleanly and is untouched by the §2.5.5 retraction: the cross-pair
   reasoning PV uniquely buys is already decodable at good r²/AUC from *both* heads, controls at
   chance. The second argument — "the pair cells it would re-home are noise" — **does not
   survive** the corrected statistic: on permutation importance those cells read 5.77%, secondary
   to the headers' 14.88% but the same order as `in_permon`, a block this project treats as real.
   So PV is deprioritised by the coverage probe alone; it survives as a *reduction setting*
   (§3.2), evaluated at one site against three alternatives, and the case against building it is
   correspondingly weaker than the first draft asserted.
3. **The critic route is still required, aimed differently.** Not pair-cell cross-attention —
   a **magnitude readout** targeting the measured `act_threat` gap (§3.3).
4. **The acceptance clause is unchanged and still correct** (flips AND `|dV|`), but should be
   measured with stratified sampling.
5. **Deletion stops being a decision.** In the target design the concat has no view, so steps 1–3
   remove it by construction. The pre-registered branch A/B rule is superseded by a refactor with a
   byte-identity gate plus one deletion counterfactual.
6. **Three roadmap items merge.** OA1, OA2 and PV become settings of `REDUCE(...how)` rather than
   independently-built features.
7. **Sequencing.** This is a natural companion to **Stage 3** (same declarative move, other side
   of the obs/model boundary) rather than a post-entity add-on.

---

## 9. Open questions — what a reviewer should attack

1. **Is the readout hypothesis actually right?** It is currently an inference from (a) all
   dependence being on already-duplicated content and (b) the π/vf `act_threat` split. It has not
   been tested directly. The cheapest discriminating arms, none yet run:
   - **per-head concat** (zero in `pi` projection only, then `vf` only) — specified in the
     learning note, never executed; sizes the two halves of the precondition directly;
   - **redundancy arm** (zero E4 seats alone / flat latent alone / both) — if flat-alone ≈ both,
     the entity route is going unused and readout is confirmed;
   - **conditional coverage** (re-run the coverage probe with the concat zeroed) — does vf
     `act_threat` collapse from 0.330?
2. **Does the stratified re-measurement hold?** Caveats §2.5.1–2.5.3 are unaddressed. If a
   field-live gate shows `hdr_effect` is decisive in `hazard`/`phaze` states, §2.2's conclusion
   narrows — though §7 means the *design* is unaffected, since it drops nothing.
3. **First-mover.** The concat is full-width from step 0 while every entity route is zero-init and
   must grow. Four replications of "the concat wins" may be measuring a race, not a capacity limit.
   Only a masked-from-step-0 retrain distinguishes them. Does this design make that experiment
   cheaper (it should — step 3 *is* that arm)?
4. **Is `OpTensors` the right factorisation?** Alternatives: a declarative schema over the flat
   block (weaker — documents the mapping but does not prevent recomputation); or promoting pairs to
   real tokens (measured at +0.253 ms compiled B=1 for +36 seats, but §2.2 says the pair content is
   unused, so this looks premature).
5. **Compile safety.** The trunk's 6.5× `--compile-extractor` lever is shipped infrastructure. A
   named-tensor container must not introduce graph breaks. Unverified; needs a fullgraph check at
   step 1.
6. ~~**Does anything actually need the flat block in the forward?**~~ **CLOSED 2026-08-08.** An
   exhaustive grep over `src/` for `damage_block` / `last_damage_block` / `decode_damage_block`
   returns exactly the consumers in §1 plus two non-forward readers (prober, `edge_ablation_audit`)
   that the retained serialization satisfies. The only one not in the original inventory was
   `reattend_layer`, which slices the per-mon rows and is **INERT in production**
   (`damage_reattend = False` in `designs/production_config.json`). No forward consumer requires
   the flat form.
7. **Does the arity partition hold for the OUTGOING direction?** `pair_out` is sketched by symmetry
   with `pair_in`, but `outgoing_matrix` and the OAX attacker matrix are both **absent** in the
   production config (`damage_matrices_outgoing = false`, `damage_matrices_outgoing_all = false`),
   so that half of the type is untested against a live layout.

---

## 10. Provenance and reproduction

| artifact | path |
|---|---|
| split-audit script | `src/agents/model/op_block_split_audit.py` (this branch) |
| split-audit result (40M) | `designs/research_state/measurements/gen3_op_block_split_40M.json` |
| coverage probe result (40M) | `models/run_20260807_135637_gen3/coverage_probe_gen3_40M.json` |
| prior sub-block dependence | `designs/research_state/measurements/gen3_op_block_dependence_6k.json` |
| edge-family audits | `designs/research_state/measurements/gen*_edge_family_audit_*.json` |

```bash
export PYTHONPATH=$PYTHONPATH:src
python src/agents/model/op_block_split_audit.py \
    models/run_20260807_135637_gen3/final_model.zip \
    --states 'models/run_20260807_135637_gen3/eval_traces/**/*states*.npz' \
    --max-states 6000 --out /tmp/split.json
```

⚠️ The command above originally reproduced caveat §2.5.1 (all states from the first step
directory). Since 2026-08-09 the shared sampler (`agents/model/audit_states.py`) is stratified
and deterministic, and the output's `provenance.sampling` block records exactly what was drawn.

## See also

- `designs/ai_v9/design_generation_roadmap.md` §3.8 — the concat-deletion precondition this amends
- `designs/ai_v9/design_conditional_opponent_cells.md` — OA1/OA2/PV as originally specified
- `designs/learning/entity_tokens_biases_pointers.md` — the output-slot ladder, Shaw et al. 2018's
  value term, the head funnel, and why an edge bias carries a ratio rather than an absolute
- `designs/learning/shortcut_learning_and_feature_delivery.md` Part 6 — the concat end-state rule
  and its four confounds (confound #1 is now the leading hypothesis)
- `designs/ARCHITECTURE.md` §4 — the live op block layout and the concat-exclusive accounting
