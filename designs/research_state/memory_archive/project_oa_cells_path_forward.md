---
name: oa-cells-path-forward
description: "The OA1/OA2/PV plan (conditional opponent-action POINTER CELLS + pair-value critic route) — the agreed sequence, gates, and caveats for the concat branch-2 execution after gen-3"
metadata: 
  node_type: memory
  type: project
  originSessionId: d13f0acd-d833-4473-bf24-f2eaae21a6cd
  modified: 2026-08-09T22:08:55.181Z
---

> **Archived 2026-09-08** — SUPERSEDED 2026-08-08 by the OpTensors path (its own banner says so). Preserved verbatim; nothing below is current.

**⚠️ SUPERSEDED 2026-08-08 (late) — OWNER ADOPTED THE OpTensors PATH.** "Our next major goal is
no more concat" (owner, after the reconciliation of `designs/ai_v9/design_op_tensors.md` with
this plan). What changed:
- **Concat removal = DISSOLUTION by refactor** (OpTensors: typed arity op output, consumers
  become views, the concat has no view), not deletion-by-decision. Steps 1-2 byte-identical;
  step 3 IS the masked-from-birth arm that settles the first-mover confound.
- **OA1/OA2/PV demote to settings of ONE `REDUCE(pair_in, how=...)` knob** at one declared
  call site (hard_max = today, belief_mean, conditional(λ)=OA1, learned_attention(k)=PV) —
  A/B'd there, no longer separately-built features.
- **PV-as-specified is dead on two independent gates** (coverage probe decodable + the split
  audit's pair cells at noise floor, net −0.03%). **My "7a token-content injection" critic
  route is ALSO superseded**: the split audit shows the concat's dependence is the per-move
  HEADER (≡ E4 seat content, already entity-attached) — the critic's gap is READOUT
  (act_threat vf r² 0.33 vs π 0.69) → k seed reads over `our_mon` (P3 refuted WIDTH, never
  MULTIPLICITY). Seed-collapse monitors + VICReg floor per the z_arch precedent.
- **Acceptance clause UNCHANGED** (concat arm < all-edges-off on flips AND |dV|), now with
  stratified state sampling (the split audit's §2.5.1 defect: its states were all
  step_10000032 = off-policy for the 40M model).
- **SEQUENCE (owner)**: gen-4 finishes → gen-4 gate reads → the §9.1 DISCRIMINATING ARMS on
  gen-4's final checkpoint (per-head concat zeroing / E4-vs-flat redundancy / conditional
  coverage — the readout hypothesis is still an INFERENCE until these run) → OpTensors
  steps 1-2 (provable) → steps 3-5 ride the next retrain (gen-5) with the head-input
  contract landing WITH the removal, not after.
- **§9.1 ARMS EXECUTED (2026-08-09, gen-4 final, stratified, `53ef270`
  gen4_concat_readout_probe.json): the dissolution path is CLEARED.** (a) Per-head split:
  concat-off-pi = 22.1% flips / dV 0; concat-off-vf = 0 flips / |dV| 4.75 — clean separation.
  (b) Redundancy: flat-headers-off 20.6% / E4-off 22.5% / both-off 29.1% ≪ additive 43% ⇒ the
  two routes are PARTIAL SUBSTITUTES at fixed weights (strict "entity route unused" refuted —
  but substitutability is what step 3 needs). (c) Conditional coverage: **act_threat vf r²
  0.400→0.418 with the concat ZEROED — no collapse**; the critic's magnitude content arrives
  via the prefuse rows through value_cls; |dV| 4.75 = trained reliance, NOT structural
  necessity. PLUS the gen-4 gate audits (gen4_edge_family_audit_25M / gen4_op_block_split_25M):
  **all-edges-off 29.22% flips > concat 22.70% — the FLIPS half of the acceptance clause is
  met by training alone, first time in the lineage**; FULL_CONCAT net flips +0.00%; residual
  = imx_HEADERS net_dV +2.32 (critic window, headers only, cells noise). k-seed readout
  (step 4) demotes to INSURANCE. Baseline act_threat vf rose 0.33→0.40 gen-3→gen-4 (E2/item-4
  already helped).
- ⚠️ COORDINATION: the sibling session AUTHORED design_op_tensors.md — the owner assigned the
  post-gen-4 execution to THIS session's line; don't both start step 1. ⚠️ Background agents
  stalled 2× on 2026-08-09 (600s watchdog during long quiet commands — submodule init, audits);
  nudge-resume works; instruct agents to emit periodic output on long steps.
- Caveat kept honest: the shuffle-arm-vs-net semantics dispute (their §2.5.5) — the numbers
  this file quotes below as "shuffle-controlled" are shuffle ARMS; their width-fair "net"
  statistic reads in_matrix/in_permon at 1.26× (not 3.6×) at 40M, weakening the "never widen
  the per-mon row" anti-pattern note.

The original plan below is kept as the record it was; read it through the supersession above.

**The concat end-state branch-2 execution plan** (design:
`designs/ai_v9/design_conditional_opponent_cells.md`, forward design 2026-08-08; the decision
rule it executes is in the roadmap, recorded 2026-08-07). Evidence: 4th concat replication on
gen-3 @9.6M (23.7% flips vs all-15-edges 13.9%), sub-block localized (in_matrix 16.27% ≫
out_active 6.25%), |ΔV| 5.67 vs 1.86 (the CRITIC leans hardest), dilution AND belief-noise
REFUTED. Surviving hypothesis: the concat carries CONDITIONAL/JOINT structure (damage vs THIS
pivot given what's actually coming) that edges cannot express — ⚠️ OA1/OA2 are pointer CELLS
(`--opp-action-cells`, via `pointer_cells`), NOT edge families; building them as edges would
re-run the K9/K10 trunk-null.

**⚠️ OWNER RE-PRIORITIZATION (2026-08-08, supersedes the phase order below): the ENTITY
END-STATE comes FIRST.** #1 priority = the full entity migration + legacy removal (Stage-3
re-home: flat obs vector, OFFSET arithmetic, matchup/reactive blocks die) — anything serving
that leads; OA1/OA2/PV are an ADD-ON afterward (owner: "happy to add on the opponent action
logic once we've achieved that objective"). Motivation: an entity world is far easier to
ablation-test, aligns with the end state, and the conceptually clean ideas map into it.
**The one honest dependency:** Stage-3 does NOT need OA (the flat obs vector's blocks all
have audited entity homes; the op concat is GPU-side and orthogonal) — but the op
HEAD-CONCAT's deletion still does: it stays through the migration and dies only after OA
re-homes its measured content (in_matrix 16.27%). So "legacy fully removed" =
Stage-3 (obs) first, concat LAST, with OA in between.

**OWNER CORRECTIONS (2026-08-08, accepted — the deletion precondition is TWO-ROUTE):**
(a) the pointer head is POLICY-ONLY, and the concat is the CRITIC's largest dependency
(|dV| 5.67 vs 1.86 all-edges, 3× replicated) — OA re-homes none of the critic's share, so
deleting after OA alone STRANDS THE VALUE HEAD; the precondition = **OA1 (policy) + PV or
token-content injection (critic), both landed and audited**, and **PV is PROMOTED out of
the optional tail** (no longer probe-conditional; build still post-entity). (b) OA1 is a
soft CONTRACTION of in_matrix, not its re-home (drops the per-move headers:
latent/belief/effect/secondary); OA2 re-homes NOTHING in the concat (new content from
pairwise_outgoing) — neither counts alone as "the content has an entity home". (c) The
ACCEPTANCE CLAUSE: deletion requires the concat arm < all-edges-off on **flips AND |dV|**
(the |dV| clause is the critic guard). The gen-3 40M verdict RE-RUNS the sub-block
localization (9.6M: in_matrix 16.27 of 18.58 shuffle-controlled flips); on confirmation the
settled target is "re-home in_matrix, BOTH directions (policy + critic)". Recorded in the
roadmap's OWNER AMENDMENT block beside the original decision rule. The item-4 GIGO fix (inert
expected-latent defender) stays near-term — a standalone correctness fix. The phase plan
below is otherwise intact, just RE-BASED to run after the re-home (where its ablation
gates are cleaner anyway).

**PHASE 1 (the gen-3 40M gate) EXECUTED 2026-08-08 — all three reads landed:**
(1) concat arm 5th replication: flips 27.45% / |ΔV| 5.36 vs all-edges-off 24.65% / 2.27 —
precondition still unmet; flip gap collapsed (1.7×→1.11×) while |ΔV| holds ~2.4× ⇒ the residual
dependency is increasingly CRITIC-side (strengthens the PV-required amendment). (2) in_matrix
dominance CONFIRMED at 40M: 18.32 of 22.37 shuffle-controlled flips (~82%; 9.6M: 16.27/18.58) —
"re-home in_matrix, BOTH directions" is twice-measured and settled. (3) coverage probe
(§2b.4, built as `tmp/coverage_probe.py`, labels exact from the op's own in_matrix cells):
joint aggregates DECODE — n_threatened vf r² 0.80 / pi 0.69, best_move_breadth 0.75,
safe_pivot_exists AUC 0.97, shuffled controls ≈ 0 ⇒ **pair-token promotion (item 8) is
unmotivated; PV's coverage rationale is dead; PV stands ONLY on the critic-route requirement.**
vf > pi on every joint target, pi > vf on the per-action control — critic reads board structure,
policy reads per-action magnitude. Reports in the run dir + ledger §"Gen-3 40M gate".

**AGREED SEQUENCE (2026-08-08, pre-re-prioritization — see above for current order):**
- **Phase 0 (build now, OFF-default; VERIFIED claims: gen-3 has `damage_matrices_outgoing_all:
  False` and `threat_refine_outgoing: False` + refine_rounds 0 ⇒ the v36 expected-latent
  defender is INERT):** (a) item 4 — re-home the species-marginalized unrevealed defender onto
  the prefuse path (a latent dead-path GIGO of the seedless-seed class, prereq for OA2 but
  worth fixing standalone); (b) margin channel + OA1 (inputs all in `_incoming_matrix`, convex
  contraction, targets the measured-dominant incoming direction); (c) OAX flips ON in the next
  run command.
- **Phase 1 (gen-3 40M gate):** final audit w/ concat + sub-block arms (does in_matrix
  dominance HOLD at 40M? edges grow faster than concat — gen-1 tripled 9.6→40M, so 9.6M
  numbers are provisional), final ELO, AND the coverage probe (item 6, offline/free) on the
  final checkpoint — pre-answers whether PV/pair-tokens are needed before anything is built.
- **Phase 2:** OA2 (needs item 4; p_switch is ONE per-turn scalar, branches DECORRELATED,
  gives T/X edges their first consumer) → **gen-4 ~25M GATE run** with `--opp-action-cells
  both` + margin + OAX ON. THREE pre-registered gates, judged separately: in_matrix sub-block
  dependence FALLS; anchored-ELO non-inferiority at matched tranches (margin fixed before
  launch); wasted-KO rate falls (pko_stay ≥ 0.8 picks on opponent-switch turns).
- **Phase 3 (doubly conditional):** PV (pair-value cross-attention, k=2-4 PMA seeds, explicit
  α, zero-init W_p in restore_identity_init) ONLY IF probe reads chance AND gen-4 leaves the
  critic residual — **AND with explicit owner sign-off: PV brushes the vf-side scope fence
  (owner, 2026-08-03)**. Pair-token promotion (in_matrix, n 29→65) stays last resort.
- Concat DELETION re-decides on gen-4's audit (only when the re-homed form matches the
  concat's measured contribution). Stage-3 re-home + E9 proceed BEHIND the concat endgame.

Ordering logic: certain before speculative; prereq-GIGO before its dependent; free offline
evidence before expensive builds; no launch before the 40M re-read. Versioning: one
STRUCTURAL string `--opp-action-cells {off,defensive,offensive,both}` in check_compatible,
OFF byte-identical, zero-init new input columns of switch_proj/move_proj pinned on a REAL
policy (the M1 rule). See [[fresh-generation-equivariance]].
