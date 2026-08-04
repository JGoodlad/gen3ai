# ai_v9 — Generation roadmap: pointer-native → the entity graph (the alignment doc)

**Status:** OPERATIVE planning doc (2026-08-03, owner + assistant session). This is the doc
that ALIGNS the fresh-generation reset, the shipped v51 pointer-native head, the entity-graph
inventory, and the history-representation decision into one sequenced plan. It does not
duplicate the inventory — `design_entity_graph.md` owns *what* the entities and edges are;
this doc owns *what order they land in, what each stage deletes, and what gates it*.

**Doc map (who owns what):**
| Doc | Owns |
|---|---|
| `design_entity_graph.md` | The entity/edge INVENTORY (E1–E9, D/S/C/V/T/X), the sorting rule, the nothing-lost audit (§6), open questions (§7) |
| `design_pointer_action_head.md` §0 | The fresh-generation reset decision + the pointer-native head spec (SHIPPED, v51) |
| `designs/learning/entity_tokens_biases_pointers.md` | The concept vocabulary (tokens vs biases vs pointers, equivariance) |
| **this doc** | The staged sequence, per-stage deletions/risks/gates, the E9 history decision |
| `designs/ai_v8/next_run_plan.md` | PREDATES the reset — its staged on-ramp (Form-A cross-attention as an ai_v8 experiment, delta-head migration) is SUPERSEDED for generation-crossing items; re-triage its non-arch items individually |

---

## 1. The premise (the owner decision this doc executes)

The next architecture is a **new generation** — fresh run, fresh pools/anchors, no old
checkpoint resumed or warm-forked across the boundary. Two consequences govern everything
below:

- **Position-equivariance is a first-class goal.** One shared scoring/attention function per
  entity token; slot identity is content, never a memorized weight row. Adopted as much for
  reasoning-complexity reduction (one scorer to prove correct, ordering bugs unrepresentable)
  as for sample efficiency.
- **No migration machinery for generation-crossing changes.** No delta/anneal forms, no
  compat toggles, no `_migrate_config` chains across the boundary. `ARCH_SIGNATURE` bumps;
  old checkpoints fail loud. Mechanically-provable changes (alignment, funnel consistency,
  masking, gradient flow) are proven by TESTS, not A/Bs. **Strength adequacy is judged
  generation-vs-generation via anchored ELO + the fixed bot suite** (cross-run comparable by
  design), not intra-run A/B arms.
- **Attribution discipline:** many simultaneous refactors make an underperforming generation
  hard to bisect. Concentrate the speculative-risk changes (Stage 2's edge biases), keep the
  provable ones provable, and land the stages as separately-smokeable increments even inside
  one generation.

## 2. Where we are: Stage 0 SHIPPED (v51, `gen3_pointer_native_v1`)

The pointer-NATIVE action head landed 2026-08-03 (`f25e708`): the flat `action_net` is
deleted (raising stub; optimizer rebuilt), and every action is scored from the token of the
entity it selects — move logit k ← the REQUEST-slot-k move token ⊕ its op cells, switch
logit j ← our-team token j ⊕ its incoming/CB/OAX cells, struggle ← the context; ctx =
`latent_pi` so the op block / beliefs / FiLM condition every score. What Stage 0 proved
structurally: the `ordering_integrity.py` sorted-vs-request bug class is unrepresentable at
the logits, F2 (bench token never reaching its own switch logit) is dissolved, and the
op-owned cell slicing (`DamageOperator.pointer_cells`, pinned vs `decode_damage_block`)
established the pattern every later stage reuses: **the physics owner slices its own layout;
consumers never hardcode offsets.**

What Stage 0 did NOT change: moves are still not attention citizens (the trunk's tokens are
still 12 mons + 2 CLS), the op is still delivered by head-concat + trunk residuals, the obs
is still the flat 2889-vector, and the history is still 7×159 positional TurnDelta frames.
Those are Stages 1–3 + E9 below.

## 2.5 The feasibility spike (FIRST — before any Stage-1 code)

Two unknowns actually threaten the generation's viability, and both are answerable in ~a day
of benchmarking with NO training and no real implementation. Run these before Stage 1:

1. **The token-budget benchmark.** The generation lives or dies at B=1 on CPU — the PFSP
   frozen-opponent forward is DISPATCH-bound (~4.6 ms / ~14k aten calls at v50), and token
   count 14 → ~35–50 grows attention ~(n/14)². Build a SHAPE-ONLY dummy (target seat counts,
   the biased MHA, no real encoders) and measure the trunk's B=1 CPU forward + a B=256
   learner proxy, eager AND compiled, across n ∈ {14, 36, 50, 64}. This also SIZES bench-K
   empirically (inventory §7.3) instead of guessing. If it fails the budget, the plan changes
   shape (smaller K / hierarchy sooner) before any code is written.
2. **The biased-MHA kernel proof.** Stage 2 needs per-pair per-head additive float biases.
   `F.scaled_dot_product_attention` accepts an additive mask, so the mechanism exists; what
   must be proven on OUR stack: numerical correctness vs a reference, and that the custom
   layer survives `torch.compile` fullgraph (the 6.5× compiled-opponent lever is shipped
   infrastructure — a layer that breaks it is an invisible ~6.5× regression), plus
   eager-vs-compiled timing.

**RESULTS (2026-08-03, `src/agents/model/entity_spike_benchmark.py`, threads=1, idle box) —
BOTH PASS; the token budget is supported.**

- **Spike 2 PASS:** SDPA-with-additive-float-mask matches the hand-rolled float64
  softmax(logits+bias) reference at max|Δ| 1.2e-7; bias=0 reproduces bias=None EXACTLY; the
  biased layer compiles `torch.compile(fullgraph=True)` with ZERO graph breaks (3.2 s) and
  compiled matches eager at 9.5e-7. The Stage-2 delivery kernel is proven on our stack.
- **Spike 1 PASS — growth is dramatically SUB-quadratic (dispatch-bound confirmed):** the
  2-layer production-shape trunk WITH the bias map, B=1 CPU per forward: n=14 → 0.183 ms,
  n=36 → 0.288 ms (1.57×; quadratic predicts 6.6×), n=50 → 0.374 ms (2.05× vs 12.8×),
  n=64 → 0.453 ms (2.48× vs 20.9×). **Absolute verdict: the full ~50-seat entity trunk costs
  +0.19 ms on a ~4.6 ms B=1 opponent forward (~4%)** — and Stage 2 deletes the ~2.4 ms op
  flat-sweep, so the net is a large REFUND. Even the 64-seat ceiling is +0.27 ms. Honest
  caveat: the B=256 proxy grows faster (3.5× at n=50 — tensor-size-bound at batch), but the
  learner runs on GPU where this class of width is noise; the B=1 CPU path was the real risk
  and it clears with margin. Bench-K sizing is therefore NOT compute-constrained in this
  range — choose K on belief-quality grounds (inventory §7.3), not budget.

Then, since both passed: the **minimal Stage-1 vertical slice** — our active's 4 move tokens (they
already exist, request-ordered, via the v51 stash) + the opp active's top-K believed moves
into the body with type embeddings, NO edges, short bridge training run. Gate: learning
doesn't collapse, CPU holds at the benchmark's prediction, the pointer tests still pin
alignment. What NOT to do first: the schema module (Stage 3) — pure work, no feasibility
risk, and building it early couples it to a token layout the spike may still change.

**SLICE STATUS (2026-08-03): BUILT — v54 `gen3_entity_move_seats_v1` (renumbered from v52 over the concurrently-shipped typed-HP-belief v52/v53).** E3 (unconditional,
request-ordered, pointer head reads the REFINED seats) + E4 (`--entity-topk-seats K`, the op's
`refine_candidates(k=K)` single-source, requires the prefuse stack) via `TeamTransformer`'s
generic `extra` seat path; token-type table 4 → 6. Gates hit: full unit suite 3916 green (incl.
`entity_seats_test.py` — seat-position stability, masked-seat bit-identity no-leak, candidate
single-source), bridge smokes pass on BOTH the E3-only default and the prefuse+E4 K=5 stack
(round-trip serialization included), and B=1 CPU measured **+0.18 ms for E4 K=5** on a ~3.1 ms
prefuse forward — on the spike's +0.19 ms prediction. The "learning doesn't collapse" leg of the
gate is a SHORT REAL TRAINING RUN still to be scheduled (the debug smokes only prove the loop
executes); run it before committing the first long v52 run.

## 3. The staged sequence

Read each stage as: WHAT lands / what it DELETES / the RISK to bound / the GATE.

### Stage 1 — moves become first-class tokens (E3/E4/E5 into the body)

The big structural lift, and the prerequisite for edges-as-biases (an edge needs both
endpoints to be seats in attention).

- **What:** per-type input projections + a token-type embedding onto one shared d_model; OUR
  active's 4 move tokens (E3, request-ordered — the same tokens the pointer head already
  reads) and the opponent's top-K believed threat-move tokens (E4: move latent ⊕ belief
  weight w ⊕ provenance ⊕ accuracy/is-phys) + the per-opp-mon tail-threat token (E5, the
  truncation insurance) enter the main attention. Token count ~14 → ~35–50; attention cost
  grows ~(n/14)² — acceptable only WITH the Stage-2 sweep deletions, which is why Stages 1+2
  land in the same generation even though they're separately smokeable.
- **Deletes:** nothing yet (the op concat stays until Stage 2 — do not delete a delivery
  route before its replacement exists; the deprecation playbook's build-home-first rule).
- **Risk:** compute at B=1 (the PFSP frozen-opponent regime is DISPATCH-bound — more tokens
  = more aten calls; the v49/v50 lesson says measure B=1 CPU, not just learner throughput).
- **Gate:** `tmp/pfsp_opponent_sweep.py`-style B=1 forward cost within budget; the pointer
  head's move logits now read tokens that were refined IN attention (a shape change to the
  stash source, pinned by the existing pointer tests); bench-K for E4 sized by the top-K
  probe template (§8 of the inventory), not guessed.

### Stage 2 — the op re-delivered as attention EDGES; the concat dies

The speculative core of the generation — concentrate the risk here, on purpose.

- **What:** the already-fuzz-validated kernels (v26 physics, v27 landing, speed, trapping,
  entry/exit) emit per-pair cells; a small learned map turns each cell into per-head additive
  attention-bias scalars on the (move → defender) / (mon ↔ mon) pair, with the full cell
  available as edge features where marginals need it. Requires a custom MHA (stock
  `TransformerEncoderLayer` takes no per-pair float bias). Adds **D4** — their BENCH's
  believed moves × our mons ("after I KO, what comes in and what does it threaten"), the
  missing quadrant, affordable only under top-K truncation.
- **Deletes:** the op's head-concat blocks and the v33/v36/v37 between-layers trunk residuals
  (the refine loop) — per the deprecation playbook: build the edge home → mask → A/B at the
  generation gate → delete.
- **Risk (state it honestly):** physics-into-the-TRUNK measured NULL 3-for-3 (ledger
  K9/K10), while the HEAD-concat route carried the policy's largest measured dependency (P1).
  Edge biases are a bet that those nulls were about delivery-as-residual-injection, not about
  attention-over-physics per se. Two mitigations: (a) the pointer head's per-action cells
  (Stage 0) KEEP a direct lossless physics→logit route regardless, so the policy's P1-class
  dependency never rides the unproven path alone; (b) the bias-ablation-per-family audit
  (inventory §8) measures which edge families trained attention actually uses — the value
  audit that decides D4/T/X retention.
- **Gate:** generation-vs-generation anchored ELO (the only strength verdict); per-family
  bias ablation for the keep/cut decisions; every kernel's numbers already pinned by the
  existing probe/fuzz gates (the new surface is delivery only).

### Stage 3 — one declarative schema; the flat vector re-homes

- **What:** ONE schema module from which the env packer, the model unpacker, the gym spaces,
  and the dimension tests are all generated (the anti-drift decision — no IDL in the hot
  path; fixed-layout arrays + a parity harness at any future Rust boundary). Every current
  obs block moves to its §6-audited home (per-mon slots → E1/E2, matchups → D/V edges,
  reactive → C4/E7, active-req-moves → E3 ordered tokens where alignment holds by
  construction).
- **Deletes:** the `OFFSET_*` arithmetic, the golden-fixture offset pins, the 288-dim matchup
  block, the reactive block — and a large slice of CPU obs-build (today ~88% of trainer CPU;
  matchup/reactive math becomes GPU-side edge computation from raw facts). The obs-build
  benchmark gate applies in reverse: confirm the REFUND materializes.
- **Gate:** the nothing-lost audit enforced by the schema's generated tests; obs-build
  benchmark before/after; the obs-roundtrip fuzz (offline == live, bit-for-bit) ported to
  the new packer.

**STAGE-2 FIRST SLICE STATUS (2026-08-04): BUILT — v55 `gen3_edge_bias_trunk_v1`.** The
biased-attention trunk landed (spike-proven layer, stock-parity + fullgraph pinned) with the
D-family both quadrants that have validated kernels AND both endpoints seated: D1 (E3 move seats →
opp mons, the v34 outgoing-matrix kernel) + D3 (E4 threat seats → our mons, the pre-collapse
incoming kernel at the seats' own candidate selection). Zero-init maps ⇒ ON bitwise-identical at
init. `--edge-bias-families {off,d,d1,d3}` is the per-family ablation surface the §3 verdict
mechanism needs. SLICE 2 (same day): **D2** (our bench's offense vs the opp active, the v39 switch-in kernel,
one-hot mon↔mon delivery) + **S1/S3** (the status-landing kernels' per-pair branches at the E3/E4
pairs) landed behind the same gate. SLICE 3: **V** (the full mon↔mon P(outspeed) block —
`pairwise_speed`, real-vs-believed spreads, public para, no stage boosts in v1) — SLICE 4: **D4** — the
missing quadrant landed as EDGES (mon↔mon, per-opp-mon top-K_bench=4 candidates collapsed over
moves), which SIDESTEPS the bench-K seat probe entirely (the probe question was about SEATS; the
edge form needs no new seats). Seven families total, each independently ablatable. SLICE 5: **E5 tail-threat seats** landed (v57 — 6 per-opp-mon truncation-insurance tokens,
no new token-type row so in-generation checkpoints stay loadable). SLICE 6: **T** (trapping —
new `build_trap_tables` physics, fail-loud; both directions at the mon↔mon block). SLICE 7: **X** (entry/exit — Spikes chip + Pursuit exposure at the (mon, global-seat)
pairs). **FIRST AUDIT READ (2026-08-04, gen-1 @9.6M of 40M, 2048 eval-trace states — PRELIMINARY,
mid-training):** the OUTGOING damage families already dominate — d1 kl 0.059 / 8.2% flips, d2 kl
0.057 / 10.3% flips / |dV| 1.62 (the critic's largest edge dependence) — while the incoming
families are near-decorative so far (d3 0.0009, s3 0.00007; s1 0.002, v 0.002/3.1% flips); ALL
families off = kl 0.124 / 16.6% flips. This REPLICATES the ledger's P1 shape (outgoing = the
policy's biggest op dependency) and rhymes with the K10 incoming-trunk nulls — on a fresh
generation, via a different delivery route. Re-audit at end-of-run before any keep/cut decision
(report: `<run>/edge_audit_9p6M.json`; the run predates t/x/E5, which need their own trained run).
SLICE 8: **G** (the per-mon end-of-turn schedule ledger at the (mon, global) route — the
inventory's §2.G, previously slated as E2/E7 attributes; edge delivery reuses X's mechanism).
NOT yet: the C family (consequence deltas — the one remaining, biggest design lift),
the op-concat deletion (deprecation playbook: home first — the per-family ablation audit runs on
the gen-1 training run now underway). B=1: +0.63 ms both families (~3.5 → 4.16 ms, under the v50 anchor; the
concat deletion is the eventual refund). Suite 3945 green; bridge smoke passes with edges on.
Verdict remains gen-vs-gen ELO + the per-family bias ablation.

## 4. E9 decided: history follows the same sorting rule as everything else

The 7×159 TurnDelta block has exactly the two defects v51 fixed for actions: it is
**positional in time** (turn t−3 is a fixed weight range; the same event at a different lag
is different weights) and **entity-blind** (raw embedded ids, disconnected from the tokens
representing those same entities in the trunk). The principled frame: in a POMDP, history's
job is to complete the belief state. Most of its content is Markovian residue better
delivered as STATE ON THE ENTITY — which this project has been doing piecemeal for a year
(sleep-wake belief, protect counter, pending Wish, choice-lock, revealed movesets are all
"history compiled into state"). The irreducibly sequential residue (opponent tendencies,
PP-war pacing, momentum) is small.

**The decided direction, in landing order:**

1. **Per-entity recency features first** (inventory option b): last-move-used,
   damage-taken-last-appearance, last-seen-turn, folded onto E1/E2. Cheap, entity-native,
   the sufficient-statistic view; the event-sourced fold already computes all of it.
2. **A short window of turn/event TOKENS for the residue** (option a): history tokens with a
   recency embedding entering attention, so the policy can query patterns ("they Protect on
   Toxic turns"). Equivariant in the token dimension; trivially variable-length.
3. **End-state, only if the tokens earn their seats** (attention-usage audit, same method as
   the edge-family ablation): **entity-LINKED event tokens** — encode the event log directly,
   each event a token whose actor/move fields ARE the same embeddings as the live entity
   tokens, so history becomes edges between past events and present entities. The
   `BattleEvent` log is already the validated source of truth; this is a re-encoding of
   trusted data, not new state-tracking.

**Ruled OUT: recurrence** (LSTM / transformer-XL memory). Beyond recurrent-PPO buffer
plumbing, it breaks the invariant the entire forensic stack depends on — **observations are
a pure function of the replayed event log**. Reconstruction, reroll-parity,
clone-search, and the obs-roundtrip fuzz all assume the obs can be rebuilt offline
bit-for-bit; a hidden state threaded across decisions forfeits that. This invariant is a
standing constraint on EVERY history representation in this generation: whatever E9 becomes,
it must remain derivable from the event log per decision window.

**Deferral (owner decision, 2026-08-03): history is DEFERRED past Stages 1–2.** The 7×159
block is a self-contained obs slice with its own encoder path; Stages 1–2 change the MODEL
side only, so the old history rides along unchanged — one fewer simultaneous change in the
generation's riskiest stages (the attribution discipline). The forcing point is Stage 3
(the flat vector dies); the minimal port there — project the 159-dim frames through a
per-type input projection as 7 opaque "history tokens" — preserves every signal without
deciding E9. The real E9 steps (recency features → turn tokens → entity-linked event
tokens) land after, with the usage-audit evidence in hand. The §4 recurrence rule applies
throughout the deferral: don't add recurrence in the meantime, and every candidate stays
derivable from the event log per decision window.

## 5. What deliberately stays OUT of this generation

- **The vf-side op concat re-home** (critic-input equivariance) — a separate decision,
  explicitly out of the first generation's blast radius (owner scope fence, 2026-08-03).
- **Full our-bench × their-bench damage closure** (D2/D4's closure) — cost/value unproven;
  the bias-ablation audit decides whether even D4 stays.
- **Local/global two-level attention (hierarchy)** — sequence is flat+biases first; the
  inventory's §7.5 expectation is that most value lands before hierarchy is needed.
- **What is never tabled** (inventory §7.7): tempo, scouting/information value, PP-war
  strategy, win-condition identification, multi-turn lines, Sleep-Clause *strategy*. These
  stay attention's job; the edges deliver mechanics, not plans.

## 6. Summary sequence

| Stage | Lands | Deletes | Verdict mechanism |
|---|---|---|---|
| 0 (SHIPPED, v51) | Pointer-native head; op-owned cell slicing | flat `action_net`, v49 delta machinery | tests (mechanical) + smoke |
| spike (§2.5) | token-budget benchmark; biased-MHA proof; then the minimal Stage-1 slice | — | B=1/B=256 numbers vs budget; fullgraph compile; short-run training sanity |
| 1 | E3/E4/E5 move tokens, per-type projections | — | B=1 CPU gate, pointer tests, probe-sized K |
| 2 | D/S/C/V/T/X edge biases, D4, custom MHA | op head-concat, refine-loop residuals | gen-vs-gen ELO + per-family bias ablation |
| 3 | declarative schema, obs re-home, history's MINIMAL port (7 opaque history tokens) | flat 2889 vector, OFFSET arithmetic, matchup/reactive blocks | schema-generated tests, obs benchmark (refund), roundtrip fuzz |
| later | E9 proper: recency features → turn tokens → entity-linked event tokens iff usage audit pays | 7×159 TurnDelta frames | attention-usage audit |

Each stage is retrain-class; the generation's checkpoints stay compatible WITHIN a stage via
normal `ModelVersion` fields, and the pre-generation lineage stays behind the
`gen3_pointer_native_v1` signature wall.
