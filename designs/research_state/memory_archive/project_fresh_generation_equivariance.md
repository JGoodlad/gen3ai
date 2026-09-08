---
name: fresh-generation-equivariance
description: "NEW ERA (2026-08-03) — next arch is a fresh generation (no old checkpoints, fresh pools); position-EQUIVARIANCE is a first-class goal; migration/back-compat machinery is unnecessary for generation-crossing changes"
metadata: 
  node_type: memory
  type: project
  originSessionId: 922c7daf-47dd-4cec-89ca-e24e5db58f2c
  modified: 2026-08-10T03:53:32.327Z
---

> **Archived 2026-09-08** — the ai_v9 gen-1..gen-6 build narrative; the resulting architecture is stated in designs/ARCHITECTURE.md and its ELO rule survives as feedback_elo_reading_rules. Preserved verbatim; nothing below is current.

Owner decision (2026-08-03): the project is entering a **new generation of model
architecture** — a fresh start ("new era/epoch, major refactorings, start fresh with
everything"). No old checkpoint will be resumed or warm-forked across the boundary; pools /
opponents / anchors are re-seeded from the new lineage. **Position-equivariance is a
first-class architectural goal** (one shared scoring function per entity token; slot identity
as content, not memorized weight rows) — chosen for reasoning-complexity reduction as much as
sample efficiency. First commitment: the pointer-NATIVE action head (no flat `action_net` at
all) — `designs/ai_v9/design_pointer_action_head.md` §0 is the operative spec.

**Status (2026-08-04): the v54 GATE RUN PASSED and Stage 2's first slice SHIPPED.** Gate run
(15M steps, bridge, self-play + --compile-extractor, prefuse+E4 K=5, grad-accum 4×4096): bots
win rate **61.4% (2M) → 87.0% (4M) → 88.8% (6M) → 90.9% (14M)**, entropy/value stable, zero
errors — "learning doesn't collapse" decisively cleared; final model at
`models/run_20260803_231207/` (in worktree `bridge-cse_01NASVy92Pf1MGTCkSCJZbvP`). **v56
`gen3_edge_bias_trunk_v1` SHIPPED (`fa344ce`)**: BiasedEncoderLayer trunk (stock-parity +
fullgraph pinned) + `--edge-bias-families {off,d,d1,d3}` — D1 (E3 seats→opp mons, outgoing-matrix
kernel) + D3 (E4 seats→our mons, pre-collapse `_incoming_rolls`, seats' own candidates); zero-init
maps ⇒ ON bitwise==OFF at init; op head-concat KEPT pending the per-family bias-ablation audit.
+0.63 ms B=1 both families. **SIX edge families now live** (`926f5b5` + `a54b471`): d1/d2/d3
(damage quadrants incl. the v39 switch-in kernel one-hot-delivered), s1/s3 (status-landing
per_pair branches), v (full mon↔mon P(outspeed), no stage boosts in v1) — each independently
ablatable behind `--edge-bias-families`; "d" is the FROZEN d1,d3 alias (saved configs never
silently grow maps); growing the valid family set is NOT a version bump. FIVE concurrent-ship
renumberings/rebases this generation (typed-HP took v52/53, op-trim took v55, plus fixture +
bridge-fix rebases) — always `git fetch` before commit and expect to renumber. **Stage 2 essentially COMPLETE (2026-08-04)**: EIGHT families live (`5081ed9` D4 as mon↔mon
edges — sidestepped the bench-K SEAT probe, `2212bdf` v57 E5 tail seats — NO new token-type row
so in-gen checkpoints stay loadable, `ea9400d` T trapping — new fail-loud `build_trap_tables`)
+ the ablation-audit instrument (`4c715d8` `edge_ablation_audit.py` — per-family zero-ablation
KL/flip/|dV| over eval-trace states via load_foreign_opponent). **GEN-1 COMPLETE (2026-08-05,
40M): Bots 90.9% / Pool 76.0% final** (run `run_20260804_090512` in worktree `gen1-run-0804`;
6 launch families d1,d2,d3,s1,s3,v). **END-OF-RUN AUDIT (edge_audit_40M.json, 4000 states):
the edges became LOAD-BEARING with training** — d1 kl 0.059→0.145 (13.6% flips), d2
0.057→0.187 (19.1% flips, |dV| 1.66), ALL-off 0.124→0.330 (**26.9% flips**); incoming stayed
near-decorative (d3 0.002, s3 0.0005), v modest-but-real (5.0% flips, |dV| 0.53). Same
outgoing-dominant P1 shape, amplified 2-3×. **Concat deletion REFUTED by measurement** (the
new `concat`/`concat_cells` audit arms, shipped `ec8f9cb`): zeroing the 807-dim block at the
projections ONLY (edges/pointer-cells/prefuse stay = the deletion counterfactual) = kl 0.482 /
35.5% flips / |dV| 7.45 — BIGGER than the whole edge system off (0.330/26.9%/2.51); the edges
complement the concat, they don't replace it. Concat stays; re-measure on gen-2's trained
full stack. **GEN-2 COMPLETE (2026-08-06): the FULL stack** — worktree `gen2-run-0805` @
ffa851e, run `run_20260805_060807`, gen-1's config + all 11 families + tail seats, clean 40M
(Bots 89.9%/Pool 78.4%, ~547 fps ≈ gen-1's throughput despite +5 families). **VERDICT: gen-2
≥ gen-1, small consistent positive, not decisive** — anchored ELO **2130±31 vs 2108±31**,
ahead at every matched tranche (+23/+23/+15/+22); direct H2H (`tmp/gen2_vs_gen1_h2h.py`)
50.7% greedy / 51.6% stochastic = **51.3%±1.25% pooled (1600 games)**. NOT a regression at
+5 families — the entity/edge machinery is at worst free and trends positive on 3
independent reads; the strength case rests on the audit (policy USES the edges, 31.5%
flips), not the endpoint delta. **Gen-1's ELO CURVE answers "why 40M"**: 1891@10M → 1981@20M →
2052@30M → 2108@40M (+217 after bots saturated at ~10M; rule: ~20-25M for gate runs, 40M for
reference models). **GEN-2 FULL-STACK AUDIT** (all 11 trained): d2 DOMINANT (23.8% flips,
|dV| 2.18) > d1 (10.1%) > v (6.3%, |dV| 0.76) > d4/t/s1 (small but real) ≫ d3/s3/x/g/c4
near-decorative (c4 ≈ 0 — never used). Concat arm REPLICATES (33.1% flips > all-edges-off
31.5%) — concat stays, twice-measured. Gen-3 family evidence: keep d1/d2/v/d4/t/s1; the
decorative five are cut/redesign candidates. **C1 SHIPPED (SLICE 10)**: the first
hypothetical-world consequence edge — `pairwise_boost` re-runs the outgoing-matrix kernel
under `MOVE_SELF_BOOSTS` post-setup stages (`boost_delta` param, None byte-identical), DELTA
cells at the (E3 setup seat, opp-mon) pairs; ~17 pure-setup moves via the new
`MoveData.self_boosts`; +2.1 ms B=1 eager (4 kernel re-runs) but compiled-path-fused + opt-in
(NOT in gen-2's config). **Curse NON-GHOST branch priced (owner-prioritized 2026-08-05** —
CurseLax/Curse-Registeel are gen3ou-defining): runtime type branch, `CURSE_BOOSTS` from
`gen3_mechanics.CURSE_NON_GHOST_BOOSTS` (+1 atk/+1 def/−1 spe; −1 spe → NEGATIVE d_outspeed;
Ghost user = zero row); the type-blind table is GUARDED against a Curse row (it doubles as the
rust draw-free contract). **Belly Drum = recorded TODO (niche, owner-deferred)**: hp_cost cell
channel + fails-below-half gate + C1b incoming-at-halved-HP (delta-+12-clamps trick =
maximize). C1b (defensive/incoming halves incl. Curse's +1 def) declared follow-up.
Since then: **ELEVEN families** (X entry/exit `b7a2601`, G end-of-turn ledger `91281cd`, C4
Protect-consequence `a33a5dc` — the (mon/E3-seat, GLOBAL seat) delivery route), the declarative
obs SCHEMA view+tiling proof (`f0a2ddb`, Stage-3 first half), and the **FIRST REAL AUDIT**
(`62a6df4`, gen-1 @9.6M): OUTGOING families dominate (d1 8.2%/d2 10.3% flips, d2 |dV| 1.62) vs
near-decorative incoming (d3 0.0009, s3 0.00007) — REPLICATES the P1/K10 ledger shape on the
fresh generation; all-off = 16.6% flips. Re-audit at end-of-run. **THE NEXT-STEPS PLAN (owner-confirmed 2026-08-04):** (1) LET GEN-1 FINISH 40M untouched (six
launch families — a clean audit substrate; ~563 fps, compiled opponents verified 6.3× ON, zero
eager fallbacks; a background watcher alerts at completion/death). (2) END-OF-RUN AUDIT
(`edge_ablation_audit` on the final ckpt over its eval traces) → family keep/cut evidence +
the op-concat deletion decision. (3) LAUNCH GEN-2 with the FULL stack (11 families
d1-d4,s1,s3,v,t,x,g,c4 + --entity-tail-seats) — judged gen-2 vs gen-1 by anchored ELO.
(4) Build during/after: C1/C2/C3/C5 consequence edges (each = re-run the validated damage
kernel under a HYPOTHETICAL — post-boost stats / post-status / post-heal / post-BatonPass —
the C4-over-G composition pattern scaled up; C1 first: SD/Agility → per-opp-mon delta cells at
the (E3 boost-move seat, opp-mon) pairs). (5) Stage-3 GENERATOR half: gym spaces + packer +
unpacker generated FROM agents/observation/schema.py (the shipped view), then the entity
re-home (retrain-class). (6) E9 history: minimal port (7 opaque history tokens) rides the
re-home; recency-features + event tokens after. Key teaching point recorded: edge biases are
ROUTING priors, not payloads — the damage NUMBER still reaches the model via the head-concat +
pointer cells (lossless) and token content; the bias steers WHICH pairs exchange it (see
designs/learning/entity_tokens_biases_pointers.md).
**🚨 GIGO FOUND+FIXED (2026-08-06, v58 stamp): the V/C1 kernels read stat index 4 = SPECIAL
DEFENSE as "speed"** (bare-integer indexing across the two stat layouts — base/iv/ev
[hp,atk,def,spa,spd,spe] vs nature [atk,def,spa,spd,spe]; the main op's index-5 paths were
always right). BOTH generations' V edge trained on bulk-as-speed → the audits' v rows measure
"the route carries signal", NOT speed physics. Fixed with named `_BS_*`/`_NAT_*` indices + a
proven-to-fail-on-buggy regression test (Aerodactyl-vs-Snorlax). LESSON: never bare-integer a
stat index; a new kernel copying a SIBLING kernel's recipe inherits its bugs — copy from the
VALIDATED path (the main op), and cross-check any index against the layout comment. **OWNER
DECISIONS (2026-08-06): keep ALL edge families** (strategy-critical long-term — Protect×Toxic,
Explosion, exploiter surface; A/B later) + **continue the full entity-design migration**
(C1b → C pieces → Stage-3 generator + re-home → E9). **C-PIECE SWEEP SHIPPED (2026-08-06,
1179e1d/44498c0/4777a37): C1b + C3 + C2 all live** — 14 edge families total
(d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2); the consequence kernels share factored physics
(`_setup_deltas`/`_believed_attackers`/`_active_defender`); suite 4004 green. **TOXIC RAMP + SLEEP
TEMPO SHIPPED (5f2f461, owner-prioritized)**: G charges −(ticks+1)/16 from the public toxic
counter both sides; C2 lands Toxic at −1/16 (split from psn by move num — shared cat 5) and
prices sleep (threat suspended + E[free turns] from the verified hazard tables, EB-marginalised;
c2 cell 7). REMAINING:
**C5 Baton Pass** (receiver axis — (BP seat, OUR mon) delivery, new route) → **Stage-3
generator half** (spaces/packer/unpacker FROM `agents/observation/schema.py`) → the entity
re-home (retrain-class) → E9 history. Deferred: Belly Drum, weather-heal fold, Rest's sleep
self-cost. **THE C GAP IS CLOSED (5db9731, owner-directed)**: C5 Baton Pass (the receiver axis — the
first (E3, OUR-mon) route family, `inherit_stages=True` on the v39 kernel), Belly Drum
(curated +12-clamps-to-maximize + hp_cost channel c1 cell 7 + fails-below-half gate),
weather heals fold live weather (2/3 sun / 1/4 other / 1/2 clear). 16 families total.
**GEN-2.5 GATE RUN LAUNCHED (2026-08-06, owner-directed)**: worktree `gen25-run-0806` @
5db9731, PID 1714637, log gen25_run.log — 25M steps, gen-2's config + ALL consequence
families (d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5) — the FIRST trained read of the
consequence system + V on true (post-GIGO-fix) speed physics. At completion: edge audit
(concat arms incl.) + anchored ELO vs gen-1/gen-2 at matched tranches (20M/25M). **K=6
EVERYWHERE for future runs (owner 2026-08-06, v59, 9f1d526)**: `--consequence-topk` (new, one
knob for C1b/C2/C3 k_cand + D4 k_bench, default 6, version-gated; pre-v59 migrates to 4) +
`--damage-topk` auto-K 5→6 + pass `--entity-topk-seats 6` in future commands. Gen-2.5 rides
at 5/5/4/4 (owner: let it ride) — gen-3 is the first K=6 run. **GEN-2.5 COMPLETE + VERDICT
(2026-08-07, b294ba1): the consequence stack is the STRONGEST gen-over-gen signal yet** —
ELO ahead of BOTH gens at EVERY tranche (4M 1788 vs 1731/1732; 20M 2023 vs 1981/2004; 24M
**2069±30 vs 2008/2029 = +61/+40**), early-learning acceleration the standout (+56@4M).
First trained consequence audit: alive-but-small at 25M (c2 1.0% flips leads; c4 ~0), d1/d2
still top (6-8%), v holds; CAVEAT — total edge dependence 14.3% flips < gen-2@40M's 31.5%
while ELO is higher (dependence grows with training; 25M is mid-curve, not a cut verdict).
CONCAT refuted a 3rd time (31.4% flips). **OWNER MANDATE (2026-08-07): I am FREE TO LAUNCH gen-3 myself** once "additional progress
toward the entity end state" lands — my call where that bar is, BIASED TOWARD MORE (the run
costs 24+h); **standing /gen3ai-ship permission whenever the tree is clean**. MY CHOSEN
LAUNCH GATE: (1) Stage-3 generator adoptions (env gym_space + ObsUnpack reading
schema.slices() — value-neutral, obs-gate + golden-parity enforced), then (2) **E9 STEP 1:
per-entity RECENCY features** (entity-linked turns-since-last-seen/acted/was-hit on the mon
tokens — retrain-class, so it must land BEFORE the 24h run to pay off; the first real
history-attaches-to-entities step). **GEN-3 IS LIVE (2026-08-07)**: the launch gate was met and the reference run launched —
worktree `gen3-run-0807` @ e60a1e1, PID 2045648, run `run_20260807_135637` (TB symlink
`_gen3`), 40M, ALL 16 edge families, K=6 everywhere (v59 defaults + --entity-topk-seats 6),
**and the E9 RECENCY obs** (5adf3bc: per-mon turns_since_seen/acted/was_hit, turn-anchored,
obs 2889→2925, protocol-truth fuzz 48k checks 0 fail, fixture regenerated). E9 slice A+B both
shipped (RecencyTracker turn-anchored — the fuzz caught the tick-then-reset forced-switch
skew AND the fuzz oracle needed a decision-time active log since on-field mons generate no
events). **First launch attempt CRASHED** (TypeError: current_model_version missing
consequence_topk — the v59 threading missed the signature) → fixed + a CONTRACT PIN (every
ARCH_ARG_KEYS value must be a current_model_version param) which immediately caught a SECOND
latent twin (zarch_lut_init_std) — both shipped e60a1e1. Completion watcher armed. At 40M:
edge audit (concat arms) + anchored ELO vs gen-1 (2108) / gen-2 (2130) / gen-2.5 (2069@24M).
The RE-HOME + ObsUnpack slices adoption stay post-gen-3.
**GEN-3 COMPLETE + 40M GATE RUN (2026-08-08): clean 40M, final aggregate 91.4%; ELO 2131±32 ≈
gen-2's 2130 (K=6 + E9 recency + all-15-families = ELO-NEUTRAL vs gen-2 at 40M; @24M matched
tranche gen-2.5's 2069 still leads — gen-3 2023).** Full gate results in
`designs/research_state/ledger.md` §"Gen-3 40M gate" + JSONs archived in the run dir: concat
5th replication (27.45% flips / |ΔV| 5.36 ≥ all-edges-off 24.65% / 2.27 — deletion precondition
still unmet, but the flip GAP collapsed to 1.11× while |ΔV| stays 2.4× ⇒ the residual is
CRITIC-side, vindicating the two-route amendment); in_matrix sub-block dominance CONFIRMED
(18.32 of 22.37 shuffle-controlled flips ≈ 82%); **coverage probe (NEW, tmp/coverage_probe.py):
joint cross-pair aggregates ALREADY decode at the head boundary (n_threatened vf r² 0.80,
safe_pivot AUC 0.97, shuffled ≈ 0) ⇒ pair-token promotion UNMOTIVATED, PV survives only as the
owner-required critic route.** Next per owner order: Stage-3 re-home (#1), gen-4 = re-homed
gate run. OPEN BUG flagged: the final eval logged repeated `[bridge] failed to capture
reconstruction ... Invalid base64/Incorrect padding` — those battles lack `*_reconstruction.json`
siblings (falsify/counterfactual tooling starved, audit unaffected); differential CONFIRMED:
final eval runs --eval-concurrency 100 vs periodic workers' 1.
**STAGE-3 RE-HOME SHIPPED (2026-08-08, `f9d09ad`, v60 `gen3_entity_rehome_v1`) — owner mandate
"fully entity based before gen-4", concat deletion explicitly OUT of scope.** Obs 2925→2667:
288-dim matchup matrices DELETED (D/V edges the trained superset), active_status +
forced_struggle deleted (derivable), protect_odds per-mon (every entity owns its stall state) +
trapped/maybe_trapped on OUR active's slot (active flag stays LAST — hp_and_active[:,:,-1] is
load-bearing), lean 17-dim board block (fainted×2, progress, wish×2, req-moves). Model:
move_feature_blocks −matchups/−validity, non_matchup_rest 29→23 (pi_proj 1131, vf_proj 875), c4
protect re-routed per-mon, ObsUnpack fully schema-sliced (Stage-3a `cb48958`). **Refund
measured**: obs build −34% wall / −45% calls-per-encode (6332→3462), trainer-turn controllable
CPU 1.144→0.894 ms/decision. Gates all green: suite 4092, obs-roundtrip bit-for-bit 627
decisions, trapping fuzz on per-mon bits, golden fixture regen (2667/991) + parity, bridge
smoke, viewer/delivery-graph regen. **GEN-4 IS LIVE (v3, 2026-08-08 21:29)**: worktree `gen4-run-0808` @ **`d9af909`**, trainer
PID **2721075** (⚠️ pgrep the python — TWICE a wrapper PID was killed while the trainer
survived), run **`run_20260808_212910`** (TB symlink `_gen4`), **25M gate run**, gen-3's exact
config. In the pin: re-home `f9d09ad` + E2 injection `427656e` + RECON drain fix `db960ca` +
stream-limit part-2 `33a756b` + **item-4** `0368f00` (owner moved it INTO gen-4:
`gen3_unrevealed_outgoing_prior_v1` — unrevealed defenders priced via Species-Clause usage
prior; pko stays nulled, revealed channel stays 0) + compile fix `d9af909`. Two aborted gen-4
launches preceded it (v1 pre-item-4 @427656e; v2 with the [B,6,S]-expand Inductor break —
CompilePrewarm failed, would have run eager opponents). **CompilePrewarm confirmed 42.9s
warm on v3.** Gate: anchored-ELO vs gen-3 at matched tranches (10M≈1924, 20M≈1987, 24M≈2023).
CUDA fine but nvidia-smi blind (NVML mismatch, reboot pending). 🔑 LESSONS BANKED:
(1) the default-on compile gate had a HAND-PINNED arch dict frozen at ai_v8 — three
generations of green while compiling a dead graph; now derives from production_config.json +
an arch_signature staleness assertion (d9af909). (2) [B,6,S] expand of a per-battle marginal
mis-vectorizes on Inductor CPU — keep per-battle quantities at [B,·] rank and broadcast
RESULTS (2nd instance of the spelling-precedent class). (3) A "flaky under load" test verdict
must survive an idle-box rerun — the tag-test failures were MY stream-limit bug, not
contention.
**✅ "NO MORE CONCAT" SHIPPED + GEN-5 LIVE (2026-08-09, `6aac795`, v61 `gen3_no_concat_v1`):**
ProjectionAssembler drops the 660-dim block from BOTH heads (pi 1131→471, vf 875→471);
`MultiSeedValueReadout` (k=4×64 over the op's per-mon rows, vf-only, +256) is the critic's
window, logging the `seeds/*` TB contract every train() (VICReg trigger pre-registered in
seed_diagnostics.py — WATCH `seeds/query_cos` & `seeds/out_effective_rank` on gen-5's TB);
`_chan_max(how)` is the named REDUCE site (OA1/PV = future settings). OpTensors steps 1-2
(typed views, E4/d3/s3 dedup) DEFERRED to background work during gen-5 — order inverted under
the owner's priority, justified by §9.1 (net policy dependence was +0.00%). Audit semantics:
the edge audit's `concat` arm now measures the seed route (pi effect structurally 0).
**GEN-5 IS LIVE**: worktree `gen5-run-0809` @ 6aac795, trainer PID **3125035** (pgrep the
python!), run `models/ai_v9_06_gen5_no_concat_0809` (in-worktree; symlinked into main models/
under the same name), 25M, gen-4's command + `--run-name`. CompilePrewarm 36.5s ✓. Gates at
completion: anchored ELO vs gen-4 matched tranches (gen-4 FINAL fit: 6M 1865 / 10M 1949 /
14M 2020 / 18M 2051 / 24M 2096) + the acceptance-clause re-read is MOOT (the concat is gone
by construction — the verdict is pure strength non-inferiority) + seeds/* collapse check.
**NAMING CONVENTION (owner 2026-08-09, DO WITHOUT PROMPTING):** every new run launches with
`--run-name ai_v9_NN_genX_<slug>_<MMDD>` (next NN in models/; the models/ fleet was renamed:
ai_v9_01_gen1_edges6_40m_0804 … ai_v9_06_gen5_no_concat_0809; old run_<ts>_genN names are
GONE — provenance JSONs citing them are historical records, translate mentally).

**superseded — NEXT MAJOR GOAL (owner, 2026-08-08 late): "NO MORE CONCAT"** — via the ADOPTED
`design_op_tensors.md` path (see [[oa-cells-path-forward]]'s supersession block for the full
delta: dissolution-by-typed-views; OA1/PV → `REDUCE(how)` settings; critic route = k seed
reads over `our_mon` — readout MULTIPLICITY, not more content; acceptance clause unchanged +
stratified sampling). START when gen-4 completes: gate reads → the §9.1 discriminating arms
on gen-4's final checkpoint → OpTensors steps 1-2 (byte-identical) → steps 3-5 ride gen-5.
Recorded in roadmap §3.8, the op_tensors status header, and the ledger gate section.
Coordination: the sibling session authored the design; THIS line executes it post-gen-4.
⚠️ LESSON: a `cd` in a compound Bash command leaked cwd to the main checkout — several
relative-path doc edits + a smoke/benchmark ran against the WRONG TREE before being caught
(ported + reverted; the "before/after" discipline caught the wrong-tree trainer_turn numbers).
Also: a PARALLEL session ships to main concurrently (delivery-digraph viewer, model.g5d.io) —
always `git fetch` + expect rebases, and never assume main's dirty files are yours.
NOTE: subagents stalled repeatedly under model-availability blips (3× watchdog) — inline work
was the reliable path.
Earlier: **v51 pointer-native head SHIPPED** (`f25e708`) — `action_net` deleted
(raising stub, optimizer rebuilt); `PointerNativeActionHead` scores from entity tokens ⊕ op
cells (`DamageOperator.pointer_cells` owns slicing, pinned vs `decode_damage_block`); ctx =
`latent_pi`; uniform-over-legal cold start. **Feasibility spike SHIPPED** (`4b6a27a`,
`entity_spike_benchmark.py`): biased-MHA == float64 reference + fullgraph-compiles; 50-seat
trunk = +0.19 ms B=1 (2.05× where quadratic predicts 12.8× — dispatch-bound). **Roadmap
SHIPPED** (`3527de2`, `designs/ai_v9/design_generation_roadmap.md` = the operative staged plan;
history DEFERRED past Stages 1–2, recurrence ruled out — obs must stay a pure function of the
event log). **v52 Stage-1 slice BUILT (worktree `bridge-cse_01NASVy92Pf1MGTCkSCJZbvP`, not yet
shipped)**: `gen3_entity_move_seats_v1` — E3 our-active move tokens as trunk attention seats
(unconditional; pointer head reads the REFINED d_model seats) + E4 opp top-K threat seats
(`--entity-topk-seats`, requires prefuse+move_latent, `refine_candidates(k=K)` single-source),
via `TeamTransformer`'s generic `extra` seat path; token-type table 4→6. 3916 tests green, both
bridge smokes pass, E4 K=5 = +0.18 ms B=1 (on prediction). REMAINING GATE: a short REAL
training run ("learning doesn't collapse") before the first long v52 run. Lesson worth keeping:
never probe a LayerNorm-output path with `.sum().backward()` — the all-ones cotangent sits in
LN's backward null space and reads ~0 on a live path; use a random cotangent.

**Why:** the staged-migration caution (delta head, anneal, strip, compat toggles) protected a
trained lineage whose biggest measured dependency (ledger P1) rode the flat path. A fresh
generation never builds that dependency, so the machinery is deleted, not deferred.

**How to apply:**
- For generation-crossing changes: NO back-compat shims, NO `_migrate_config` chains, NO
  delta/anneal migration forms. Bump `ARCH_SIGNATURE` and fail old checkpoints loud.
  ModelVersion fields still needed for WITHIN-generation resume/pool loads.
- "If we know it's correct, just make the change": mechanically-provable changes (alignment,
  funnel consistency, masking, gradient flow) need tests, not A/Bs. Strength adequacy is
  judged **generation-vs-generation via anchored ELO + the fixed bot suite** (cross-run
  comparable by design), not by intra-run A/B arms.
- The G1–G5 information-deficit analysis in the pointer design is a day-0 REQUIREMENTS list
  (never ship a bare information-starved pointer head), not an experiment ladder.
- Attribution caveat to keep raising once per plan: many simultaneous refactors make an
  underperforming generation hard to bisect; concentrate speculative-risk changes, keep
  provable ones provable.
- vf-side op concat re-home (critic-input equivariance) is deliberately OUT of the first
  generation's scope — separate decision.

Related: [[next-run-plan]] (pre-flight list predating this reset — items may need
re-triage under the fresh-generation premise), `designs/ai_v9/design_entity_graph.md` (the
end-state this generation moves toward).

## Gen-5 verdict (2026-08-10, offline fit)
- `ai_v9_06_gen5_no_concat_0809` COMPLETE: 24M steps, final win rate 90.8%, offline anchored ELO
  **2065 ±30 @24M (best 2071 @20M)** vs gen-4's 2096 @24M — **overlapping CIs = parity**. The
  no-concat milestone goal was "delete without regression": ACHIEVED; no gain either.
- Known deficiency all run: the k=4 MultiSeedValueReadout was COLLAPSED (eff-rank 1.0 start to
  finish) → the critic's magnitude window was one pooled read. Fix built same day:
  `--value-seed-vicreg-coef` (v62, resume-immutable, cov-term gate-tested) — enable at gen-6.
- TB rename (owner request): the seed family is `value_seeds/*` from v62 on (gen-5's board shows
  the old `seeds/*` tags). "seed" alone was ambiguous with RNG seeds.
- Gen-6 pre-launch checklist: step-0 split audit on gen-5 final ckpt (pair-reduction §8.1
  downscope rule), G1/G2 rung gates offline, G7 exploiter A/B (GPU now free), VICReg ON.

## ⚠️ 2026-08-11 — verdicts must use `ladder_elo`, and gen-5 was NOT parity
`snapshot_ladder/ladder.json` (dense frozen-vs-frozen, bot-anchored, complete for both runs) gives
**±10-11** where the sparse `eval/elo` gives ±30: **gen-4 2081±11 vs gen-5 2037±11 — DISJOINT.**
So the concat deletion likely COST ~44 Elo (consistent with gen-5 running at seed eff-rank 1.0 all
run), and the "parity" I recorded was an artifact of a saturated-bot instrument. **Read
`ladder_elo` for every generation verdict from now on**; re-read old calls before quoting them.
Caveat: dense pairs pin ORDERING within a run; the run's LEVEL still rides saturated bot edges.
