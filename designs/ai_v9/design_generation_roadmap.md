# ai_v9 — Generation roadmap: pointer-native → the entity graph (the alignment doc)

**Status:** OPERATIVE planning doc (2026-08-03, owner + assistant session). This is the doc
that ALIGNS the fresh-generation reset, the shipped v51 pointer-native head, the entity-graph
inventory, and the history-representation decision into one sequenced plan.

> **[STATE 2026-08-14 — read before the staged narrative below; `designs/CHANGELOG.md` is the
> authoritative sequence.]** Since this doc's last owner amendment: the op head-concat **died
> 2026-08-09 (v61, on gen-4's stratified evidence — the two-route precondition in §3.8 was
> RESOLVED, not satisfied; `designs/CLAUDE.md`'s ai_v9 row records how)**; the tiered pipeline
> landed (v70/71 — prefuse unconditional, refine loop + placement toggles deleted, tier contract
> asserted); α/β shipped (v68) and α is CONSUMED on the critic side (v74 `intent_value_reduce`);
> the SimSiam latent belief is deleted (v75); the active-ctx head concat is deleted + the
> migration floor landed (v76 `gen3_ctx_dedup_v1`); and `design_op_tensors.md` steps 1–2 shipped
> byte-identical (`gen3_op_tensors_views_v1`). The seed-multiplicity line closed after gen-6/7
> measured both pressures capping at ~1-D. Live run: gen-9 (intent + distributional critic). It does not
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

### Stage 2 — the op re-delivered as attention EDGES (⚠️ the concat SURVIVES — see §3.8)

The speculative core of the generation — concentrate the risk here, on purpose.

- **What:** the already-fuzz-validated kernels (v26 physics, v27 landing, speed, trapping,
  entry/exit) emit per-pair cells; a small learned map turns each cell into per-head additive
  attention-bias scalars on the (move → defender) / (mon ↔ mon) pair, with the full cell
  available as edge features where marginals need it. Requires a custom MHA (stock
  `TransformerEncoderLayer` takes no per-pair float bias). Adds **D4** — their BENCH's
  believed moves × our mons ("after I KO, what comes in and what does it threaten"), the
  missing quadrant, affordable only under top-K truncation.
- **Deletes:** the v33/v36/v37 between-layers trunk residuals (the refine loop) — per the
  deprecation playbook: build the edge home → mask → A/B at the generation gate → delete.
  ⚠️ **The op's head-concat is NO LONGER deleted here** (OWNER AMENDMENT 2026-08-08, §3.8): it
  survives Stage 2 *and* Stage 3 and dies LAST, behind a two-route precondition. Four
  replications have now measured the concat arm flipping MORE actions than the entire edge
  system, so Stage 2's edges were never its substitute — they do a different job (ratio vs
  absolute).
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
SLICE 9: **C4** — the first CONSEQUENCE edge (Protect: success odds × the banked G-ledger
turn, at the (Protect E3 seat, global) pair; composes G). SLICE 10: **C1** — the first
HYPOTHETICAL-WORLD damage consequence (`pairwise_boost`, 2026-08-05): post-setup-move DELTA
cells `[is_boost, d_best_high, d_best_pko, d_outspeed]` at the (E3 setup seat, opp-mon) pairs,
from RE-RUNNING the validated outgoing-matrix kernel under the `MOVE_SELF_BOOSTS` stage deltas
(a `boost_delta` threaded into the kernel's stage read, None byte-identical — the C4-over-G
composition pattern scaled to the damage kernel). The ~17 declarative pure-setup moves only
(`MoveData.self_boosts`; Belly Drum/Curse unpriced by the same gates that keep them fail-loud
in the rust engine); defensive halves (Iron Defense/Amnesia vs INCOMING) = the declared C1b
follow-up. +2.1 ms B=1 EAGER (4 extra kernel runs, dispatch-bound) — acceptable because the
production PFSP path is COMPILED (fullgraph-pinned with all 12 families, numerics 5e-7) and c1
is opt-in. **SLICE 10b (2026-08-05, owner-prioritized): the NON-GHOST CURSE branch is priced**
— CurseLax/Curse-Registeel are gen3ou-defining, so the type-conditional move gets its runtime
branch (`CURSE_BOOSTS` [+1 atk/+1 def/−1 spe] from `gen3_mechanics.CURSE_NON_GHOST_BOOSTS`,
gated by the user's live types; −1 spe → NEGATIVE d_outspeed; Ghost user's Curse = zero row;
the type-blind `MOVE_SELF_BOOSTS` table is guarded against ever growing a Curse row since it
doubles as the rust engine's draw-free contract). **RECORDED TODO — Belly Drum (niche, owner-
deferred 2026-08-05):** pricing it needs an `hp_cost` cell channel (cell 4 → 5, structural on
the zero-init c1 map), a fails-below-half-HP gate on is_boost, and the C1b
incoming-at-halved-HP re-run so the +6 atk is never shown as free; the delta-+12-clamps-to-max
trick implements "maximize" in the existing kernel. Remaining C pieces (C1b defensive/incoming
halves incl. Curse's +1 def / C2 status consequences / C3 recovery flips / C5 Baton Pass)
stay open — same hypothetical-kernel pattern, now with a worked example. NOT yet:
the op-concat deletion (deprecation playbook: home first — the per-family ablation audit runs on
the gen-1 training run now underway). B=1: +0.63 ms both families (~3.5 → 4.16 ms, under the v50 anchor; the
concat deletion is the eventual refund). Suite 3997 green; bridge smoke passes with edges on.
Verdict remains gen-vs-gen ELO + the per-family bias ablation.
**END-OF-RUN AUDIT READ (2026-08-05, gen-1 COMPLETE at 40M — Bots 90.9% / Pool 76.0% final;
4000 states from the last two eval cycles, report `<run>/edge_audit_40M.json`):** the edges
became LOAD-BEARING with training — d1 kl 0.059→**0.145** (13.6% flips), d2 0.057→**0.187**
(19.1% flips, |dV| 1.66, still the critic's largest edge dependence), ALL-off 0.124→**0.330**
(**26.9% flips**, |dV| 2.51). The incoming families stayed near-decorative in absolute terms
but grew relatively (d3 0.0009→0.0021, s3 0.00007→0.0005; s1 0.0061; v 0.0096/5.0% flips,
|dV| 0.53 — the value head reads speed). Same outgoing-dominant P1 shape, now amplified 2-3×.
DECISIONS: all six families KEEP for gen-2 (and the five untrained ones ride along for their
own trained audit); and the op-concat deletion is now REFUTED BY MEASUREMENT, not deferred —
the audit's new `concat` arm (zero the 807-dim block at the ProjectionAssembler only; edges +
prefuse injection + pointer cells all stay = the exact deletion counterfactual) reads kl
**0.482 / 35.5% flips / |dV| 7.45** on the same 4000 states — LARGER than the entire edge
system off (0.330/26.9%/2.51), with the critic hit 3× harder than by all edges combined.
`concat_cells` (op fully out of the heads) = 0.650/40.4%. The edges did NOT absorb the
concat's role — they added on top of it (the P1 head-route-works history, extended: the edge
route is real AND complementary). The concat stays for this generation; re-measure on gen-2's
trained full-stack checkpoint (report: `<run>/edge_audit_40M_with_concat.json`). **GEN-2 LAUNCHED (2026-08-05, worktree `gen2-run-0805` @ ffa851e,
run `run_20260805_060807`):** gen-1's exact config + all ELEVEN families
(d1,d2,d3,d4,s1,s3,v,t,x,g,c4) + `--entity-tail-seats`, 40M steps, fresh lineage; judged
gen-2-vs-gen-1 by anchored ELO.

**GEN-2 COMPLETE + THE GENERATION VERDICT (2026-08-06).** Clean 40M finish (final eval Bots
89.9% / Pool 78.4%; zero errors; ~547 fps — ~4% under gen-1's 563 despite 5 more families +
tail seats). **Anchored ELO: gen-2 2130±31 vs gen-1 2108±31 at 40M, and ahead at EVERY matched
tranche** (10M: 1914 vs 1891; 20M: 2004 vs 1981; 30M: 2067 vs 2052; 40M: +22) — consistent
direction ×4, CIs overlapped; a DIRECT head-to-head (both finals, pool teams, in-process
bridge; `tmp/gen2_vs_gen1_h2h.py`) reads 50.7% greedy (600 games) / 51.6% stochastic (1000)
= **51.3% ± 1.25% pooled over 1600** — consistent with the ELO-predicted 53%, not
independently significant. Read: THREE independent measurements all point the same
small-positive way (4/4 ELO tranches, both H2H arms) — the full stack is ≥ gen-1,
decisively NOT a regression at +5 families/tail-seats and ~zero throughput cost, but not a
decisive strength win; the strength case for the edge system rests on the audit (the policy
USES it — 31.5% flips) more than on the endpoint delta. **GEN-2's FULL-STACK AUDIT (edge_audit_40M.json, 4000 states,
all 11 families TRAINED):** d2 is the dominant family (kl 0.312 / 23.8% flips / |dV| 2.18),
then d1 (0.108/10.1%), v (0.012/6.3%, |dV| 0.76), d4 (0.004/2.1%, |dV| 0.49), t
(0.0018/1.8%, |dV| 0.15), s1 (0.0096/1.9%); NEAR-DECORATIVE at 40M: d3, s3, x, g, and c4
(≈0.00002 — the Protect-consequence edge never got used). All-off = 0.491/31.5% (bigger than
gen-1's 0.330/26.9%). The concat arms REPLICATE on the full stack: concat 0.537/33.1%/|dV|
7.44 — still larger than the whole edge system — so the concat stays, twice-measured.
Family evidence for gen-3: d1/d2/v/d4/t/s1 load-bearing; d3/s3/x/g/c4 near-decorative at
40M. **OWNER DECISION (2026-08-06): KEEP ALL FAMILIES.** The decorative ones encode
strategy-critical mechanics that a 40M self-play run may simply not have discovered yet —
Protect×Toxic timing (c4/g), Explosion consequence play, entry/exit costs — and they are the
substrate exploiters and长 longer runs need; cut A/Bs can come later. The migration to the
FULL entity design continues: C1b incoming halves → remaining C pieces → Stage-3 generator +
entity re-home → E9.
**GIGO FOUND + FIXED while building C1b (2026-08-06, v58 stamp):** `pairwise_speed` (V) and
`pairwise_boost`'s outspeed read stat index 4 — SPECIAL DEFENSE — as "speed" (bare-integer
indexing across the two stat layouts; the main op's index-5 paths were always right). BOTH
trained generations' V edge priced bulk as speed, so the audits' v rows (gen-1 5.0%, gen-2
6.3% flips) measured the model exploiting a systematically WRONG feature — treat them as
"the v ROUTE carries signal", not as validated speed physics; gen-3 trains on true physics.
Fixed with named `_BS_*`/`_NAT_*` indices + a discriminating regression test
(Aerodactyl-vs-Snorlax: P(outspeed)≈1 correct vs ≈0 buggy; Agility saturates to ~0 on an
already-faster mon), PROVEN to fail on the pre-fix kernel.
**SLICES 11-13 (2026-08-06): the C-piece sweep** — **C1b** `pairwise_boost_incoming` (the
incoming setup halves: believed attackers vs OUR ACTIVE at current-vs-post-boost def/spd, 5
worlds on one axis; Iron Defense/Amnesia channel-exact, Curse's +1 Def prices; c1 cell → 6
wide), **C3** `pairwise_recovery` (the heal-vs-KO FLIP: damage once, the `_rolls` KO ramp
re-evaluated at post-heal HP; `MOVE_HEAL_FRACTION` — Rest 1.0 sleep-unpriced, weather heals
flat 0.5 v1, Wish excluded; family "c3"), **C2** `pairwise_status_consequence` (what LANDING
does behind S1: para → Δoutspeed at spe×0.25, burn → worst-physical at Atk×0.5, brn/psn →
flat −1/8 tick; deltas RAW/decorrelated from `land`; family "c2"). Shared kernels factored
(`_setup_deltas` / `_believed_attackers` / `_active_defender`) so consequence families can
never disagree on their common physics. **SLICE 14 (2026-08-06, owner-prioritized — "toxic and sleep are core mechanics"):** the Toxic
RAMP + the sleep-tempo consequence land. G's Toxic leg = −(ticks+1)/16 from the PUBLIC obs
toxic counter both sides (C4's banked nets inherit it); C2 splits Toxic from plain psn by move
num (shared immunity cat 5) and lands it at the TRUE first tick −1/16; C2 gains two SLEEP
channels — `d_in_all` (their whole believed threat suspended, any category) +
`e_slp_free_turns` (E[free turns] DERIVED from the verified hazard tables via
`sleep_belief.expected_free_turns`: 2.5 no-EB / exactly 1.0 revealed Early Bird,
Smogon-prior-marginalised per mon; c2 cell 5 → 7). Obs-benchmark gate re-run for the
sleep_belief addition: 6,446 calls/encode == the canonical baseline (the new function is
never called by encode). **SLICE 15 (2026-08-06): THE C GAP IS CLOSED.** **C5 Baton Pass** ships the RECEIVER axis —
`pairwise_baton` at the NEW (E3 BP seat, OUR-mon) route: the v39 switch-in kernel re-run under
`inherit_stages=True` (the post-pass world is one flag away from D2's world) minus the neutral
baseline, `[is_bp, d_best_high, d_best_pko, d_outspeed]`; zero with no stages up; active
column zeroed. **Belly Drum** priced (curated +12-clamps-to-maximize model-side row — the
selfBoosts JSON stays the rust draw-free contract — + the half-max-HP `hp_cost` channel, c1
cell 7, + the fails-below-half gate). **Weather heals** fold LIVE weather (2/3 sun / 1/4
other / 1/2 clear). Rest's self-sleep cost shipped earlier same day (c3 cell 3, exactly 2
turns / 1 EB). Every consequence family in the inventory now has a shipped edge home:
c1 (+BD/Curse), c2 (status incl. tox tick + sleep tempo), c3 (heal flip + Rest cost),
c4 (Protect×ledger), c5 (Baton Pass). Residuals: volatile/Sub passing, the BP receiver's
incoming world, Yawn. **GEN-2.5 GATE RUN launches on this tree** — 25M, gen-2's config +
c1,c3,c2,c5 — the first TRAINED read of the whole consequence system. **K=6 EVERYWHERE
(owner 2026-08-06, v59):** all candidate K's default to 6 for future runs — `--damage-topk`
(the `--unified-moves` auto-K 5 → 6), `--entity-topk-seats` (pass 6 in future run commands),
and the NEW `--consequence-topk` (one knob for C1b/C2/C3's k_cand + D4's k_bench, default 6,
forward-behavior version-gated; pre-v59 checkpoints migrate to their trained 4). The
IN-FLIGHT gen-2.5 run rides at 5/5/4/4 (owner: "let this one ride") — gen-3 is the first
K=6 run.
**GEN-2.5 COMPLETE + VERDICT (2026-08-07, run_20260806_160611, clean 25M, zero errors).**
**The consequence stack is the strongest gen-over-gen signal yet:** anchored ELO ahead of BOTH
prior gens at EVERY matched tranche — 4M: 1788 (vs 1731/1732), 10M: 1927 (vs 1891/1914),
20M: 2023 (vs 1981/2004), 24M: **2069±30 (vs 2008/2029 — +61/+40)** — with the early-training
acceleration (+56 at 4M) the standout: consequence edges + true speed physics speed up
LEARNING, not just the endpoint. **First trained audit of the consequence families
(edge_audit_25M.json):** alive but small at 25M — c2 leads (1.0% flips, the status edge),
c1 0.6%, c5 0.7%, c3 0.3%, c4 ~0 (still unused); d1/d2 remain the top families
(6.1%/8.1%) and v holds (4.0%, |dV| 0.95). NOTE the audit-scale caveat: total edge
dependence reads LOWER than gen-2@40M (all-off 14.3% vs 31.5% flips) while ELO reads
HIGHER — dependence grows with training (gen-1 grew 3× from 9.6M→40M), so the 25M audit
is mid-curve, not a keep/cut verdict. The CONCAT arm replicates a THIRD time (31.4% flips
≫ all-edges-off) — deletion stays refuted. **GEN-3 RECOMMENDATION: the 40M K=6 reference
run on this exact stack** (all 16 families incl. c1-c5, --consequence-topk 6 --damage-topk 6
--entity-topk-seats 6).

**THE CONCAT END-STATE DECISION RULE (recorded 2026-08-07, pre-gen-3-audit).** The concat's
grip is partly STRUCTURAL, not gradient laziness: attention weights are softmax-normalized
(relative), so edges can route "who attends to whom" but cannot transmit ABSOLUTE magnitudes
("this hit is 53% of max HP"); the two entity-native channels that CAN are token CONTENT
(the prefuse injection — today only the per-our-mon incoming rows via `prefuse_proj`) and the
pointer head's per-action cells (already lossless per-logit). So gen-3's audit picks between
TWO responses, neither of which is "wait longer":
  * **Concat flips < the all-edges-off arm** on gen-3's trained checkpoint ⇒ the entity paths
    absorbed the role ⇒ mask → A/B → DELETE (the playbook's original ending; the Stage-3 CPU
    refund follows).
  * **Concat holds ≥ all-edges-off (a FOURTH replication)** ⇒ that is a measurement that
    magnitude still has no entity home of equal fidelity — the response is to WIDEN
    token-content delivery (inject the full per-mon op rows onto BOTH sides' tokens, the
    prefuse pattern generalized; audit which concat sub-blocks carry the residual via
    per-block concat arms) and re-audit, i.e. make the lazy path BE the entity path (the v51
    pointer-head reframe applied to the trunk). Deletion only after the re-homed form
    matches the concat's measured contribution.
The re-home (Stage 3) then removes the flat obs vector regardless of which branch fires —
starvation has nothing left to feed on once every fact's ONLY delivery is entity-attached.

**OWNER AMENDMENT (2026-08-08) — the deletion PRECONDITION, corrected.** Sequencing
accepted: the concat SURVIVES Stage 3 and dies LAST. Three corrections to the rule above:
  1. **The pointer route is POLICY-ONLY.** OA1/OA2 are pointer cells; the pointer head never
     reaches the critic — and the concat is the CRITIC's largest dependency (|dV| 5.67 vs
     1.86 all-edges, 3× replicated). OA re-homes none of that; deleting after OA alone
     strands the value head. The precondition is therefore TWO-ROUTE: **OA1 (policy) + PV or
     token-content injection (critic), BOTH landed and audited.** PV is hereby PROMOTED out
     of the optional tail (it was Phase-3-conditional in
     `design_conditional_opponent_cells.md`; it is now a required component of the deletion
     precondition — build still post-entity).
  2. **OA1 is a soft CONTRACTION of in_matrix, not its re-home** — it drops the per-move
     headers (latent / belief / effect / secondary). OA2 re-homes NOTHING currently in the
     concat (new content from `pairwise_outgoing`). Do not count either as "the concat's
     content now has an entity home" on its own.
  3. **The acceptance clause (was missing):** deletion requires the concat arm to fall below
     all-edges-off on **flips AND |dV|** — the |dV| clause is the critic guard.
**READING NOTE (added 2026-08-08, reconciliation pass).** Point 1 above states two things that are
not the same claim — *"PV **or** token-content injection"* (an OR over implementations) and *"PV is
a required component"*. **The OR is operative**: what the precondition requires is a **CRITIC
ROUTE**, and there are exactly two admissible implementations of it —
  * **PV** (pair-value attention, Shaw's value term) — equivariant in BOTH axes, no new seats, and
    the only option that also buys **cross-pair reasoning**; costs a new unfused side module and a
    rank-`h` reduction of the row.
  * **generalized token-content injection** (the `prefuse_proj` pattern extended to inject the full
    per-mon op rows onto BOTH sides' tokens) — cheaper (shipped mechanism, no new module) and keeps
    the full row, but pays for that by leaving the within-seat axis POSITIONAL (the cells inside a
    widened seat stay ordered by team slot), in the generation whose premise is that positions are
    not identities.

"PROMOTED" therefore means **out of the optional tail and onto the critical path as one of the two
candidates** — not that PV specifically must be built. **This changes the coverage probe's job**
(§2b.4 of `design_conditional_opponent_cells.md`): it no longer *vetoes* PV, it *chooses between the
two routes* — cross-pair quantities decodable at good r² ⇒ token-content injection suffices;
at chance ⇒ only PV/promotion buy what is missing. ⚠️ **Residual owner call:** if the intent was in
fact "PV specifically, regardless of the probe," say so here and the OR above is struck.

Localization: branch B's sub-block arm already ran at 9.6M — the residual is in_matrix
(16.27 of 18.58 shuffle-controlled flips). The gen-3 40M verdict RE-RUNS it; on
confirmation, "re-home in_matrix, BOTH directions (policy + critic)" is the settled target.

**OWNER DECISION (2026-08-08, late — supersedes the two-route precondition above): the
concat's removal proceeds by the `design_op_tensors.md` path ("no more concat" is the next
major goal, starting when gen-4 completes).** The 40M split audit showed the concat's
measured dependence is the per-move HEADER — content already carried bit-for-bit by the E4
seats — so the premise "magnitude has no entity home" fell; the residual is the CRITIC's
READOUT (act_threat vf r² 0.33 vs π 0.69). Consequences: the concat DISSOLVES by typed-view
refactor (steps 1-2 byte-identical; step 3 doubles as the masked-from-birth arm that settles
the first-mover confound); OA1/OA2/PV demote to settings of one `REDUCE(pair_in, how=…)`
call site; the critic route is k seed reads over `our_mon` (multiplicity, not width — P3
scope), landing WITH the removal. The acceptance clause is UNCHANGED (concat arm <
all-edges-off on flips AND `|dV|`), measured with stratified state sampling. Precondition
before step 3+: the op_tensors §9.1 discriminating arms run on gen-4's final checkpoint.
**The k-seed module ships WITH its TB collapse monitors** (owner, 2026-08-09):
`agents/model/seed_diagnostics.py` is the contract — `seeds/query_cos` / `seeds/out_cos` /
`seeds/out_effective_rank` (uncentered PR ≈ how many distinct readout directions) /
`seeds/out_var`, logged once per `train()` like `popart/*` — with the pre-registered VICReg
trigger (wire the variance+covariance floor iff query_cos sustains > 0.6 or effective rank
sustains < k/2 after ~2M steps; the z_arch collapse read ~1.6 effective of 32). A seed module
that lands without these monitors repeats the z_arch post-hoc-discovery failure.

## 3.9 E9 STEP 1 — per-entity RECENCY features (designed 2026-08-07, pre-gen-3)

The first "history attaches to entities" increment, chosen as the gen-3 launch gate's final
item (retrain-class ⇒ must land before the 24h run to pay off). THREE per-mon scalars appended
to the per-Pokémon slot (POKEMON_VECTOR_DIM +3, both sides, log-saturated like
`turns_since_progress` — `log(1+min(n,10))/log(11)`):

  * `turns_since_seen`   — turns since the mon was last ON FIELD (0 while active; the staleness
    of everything the slot asserts — a mon benched 20 turns ago may have been statused/damaged
    on info the history frames have already rotated out of).
  * `turns_since_acted`  — turns since it last EXECUTED a move (captures sleep/para/flinch
    lockouts + long bench stints; distinct from seen — a Protect-stalling active is seen but
    the belief about its moveset decays differently).
  * `turns_since_was_hit` — turns since it last TOOK damage (the safety/pressure recency: a
    wall that hasn't been hit in 8 turns is walling; composes with the G ledger).

Source: the EVENT LOG via EpisodeTracker (the wish/progress-clock convention — cross-turn
state owned by the tracker, threaded into encode(); NOT LiveView). Both sides PUBLIC (all
three derive from observed protocol events — no leak). Fuzz gate: a poke_env_gaps fuzz
asserting the encoded scalars == counters reconstructed from the raw protocol stream.
Obs-gate: the mandatory benchmark before/after. Versioning: retrain-class obs-dim change
(caught by the total_dim weight-field check, NO ARCH_SIGNATURE bump needed) — lands
immediately before gen-3 launches so the reference run trains on it.

WHY these three and not more: each is entity-invariant (a per-mon fact → token feature per
the sorting rule), cheap (3 counters ticked per turn), and un-derivable from the 7 history
frames (which cover only the last 7 turns positionally). The richer E9 acts — recency-weighted
event embeddings, entity-linked event tokens — ride the Stage-3 re-home (post-gen-3).

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
| 2 | D/S/C/V/T/X edge biases, D4, custom MHA | refine-loop residuals (**NOT** the op head-concat — §3.8) | gen-vs-gen ELO + per-family bias ablation |
| 3 | declarative schema, obs re-home, history's MINIMAL port (7 opaque history tokens) — ◐ **HALF-LANDED as v60 `gen3_entity_rehome_v1`** (declarative schema + matchup/reactive deletion + per-entity re-home + E2 injection; the flat vector, the OFFSET expressions and the positional history block remain) | flat 2889 vector, OFFSET arithmetic, matchup/reactive blocks | schema-generated tests, obs benchmark (refund), roundtrip fuzz |
| post-3 | ~~**the two-route concat precondition**~~ **RESOLVED 2026-08-09 without either route** — the flips half was met by training alone on gen-4, net policy dependence was +0.00%, and the critic route that shipped was `MultiSeedValueReadout` (neither candidate); OA1/OA2/PV survive as forward designs on their own merits (`designs/CLAUDE.md` ai_v9 row records the resolution) | — | gen-4 stratified end-of-run audits |
| last | ~~op head-concat deletion~~ ✅ **DONE 2026-08-09 (v61 `gen3_no_concat_v1`)** — mask→A/B→delete collapsed to delete-on-evidence; the Stage-3 CPU refund's remaining piece is `design_op_tensors.md` step 3 (drop the flat render + trim `out_gain`, gen-10) | op head-concat (660 dims off both projections) | gen-5 trained at ELO parity with gen-4 (the deletion cost nothing) |
| later | E9 proper: recency features → turn tokens → entity-linked event tokens iff usage audit pays | 7×159 TurnDelta frames | attention-usage audit |

Each stage is retrain-class; the generation's checkpoints stay compatible WITHIN a stage via
normal `ModelVersion` fields, and the pre-generation lineage stays behind the
`gen3_pointer_native_v1` signature wall.
