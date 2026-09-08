---
name: project_damage_op_block_audit
description: "Per-block DamageOperator throughput/value audit — what to remove, empirical cProfile, and the near-inert-machinery finding under detached belief"
metadata:
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
  modified: 2026-08-04T05:13:43.436Z
---

> **Archived 2026-09-08** — ai_v7-era op audit; the refine loop it audits was DELETED at v50 and the era is closed. Preserved verbatim; nothing below is current.

**DAMAGE-OP BLOCK AUDIT (2026-06-30, 6-agent workflow: map + cProfile + value + adversarial verify; NOT
acted on).** Answers "what can we REMOVE from the model for throughput?" for ai_v7_02_critic_shape_0627.

**EMPIRICAL PROFILE (cProfile, CPU B=1 opponent forward = the exact self-play rollout regime, idle box):**
`get_distribution` = **7.05 ms**; DamageOperator = **3.09 ms = 43.9%** (confirms the ~45% profile). Largest
line in the WHOLE forward = `torch._nn.linear` (transformer + projection GEMMs, 10.5%) — OUTSIDE the op,
irreducible. NB the cProfile `tottime` field the agent reported is a per-FORWARD TOTAL (across all calls),
NOT per-call (proven by `_boost_mult` 22 calls/0.14 ms). Top removable blocks (per-forward tottime):
`discrete_outgoing` 0.428 (2 calls, refine-only), `discrete_incoming` 0.377, `_outgoing_attacker_matrix`
0.278 (v39 outgoing-all), `_outgoing_block` 0.234, `_outgoing_matrix` 0.228 (v34), `discrete_outgoing_status`
0.226, `_damage_rolls` 0.170, `_nature_marg_ko` 0.088 (2 calls). `_topk_block` = 0 calls (the v35 incoming
MATRIX supersedes the lean top-K at K=5, so top-K is dead code in this build).

**RANKED REMOVALS (measured ms):** (1) **`--damage-refine-rounds 0` = ~1.3 ms (~19%)** — the `discrete_*`
kernels exist ONLY in the between-layers refine loop, run 2×/forward; dropping it ALSO kills all 4 v36/v37
threat channels (they ride the loop). (2) `--damage-matrices-outgoing-all` off = 0.278. (3) outgoing side of
`--damage-matrices both` = 0.228. (4) `--spread-belief-nature-marginalize` off = 0.088. Combined A/B ≈ **1.9
ms → ~27% off the opponent forward** + ~234 fewer projection-input dims. Op blocks all `runs_at_inference`
(concatenated into BOTH pi+vf projections, no is_grad_enabled gate) → cuts hit rollout (48 CPU opponents,
68% of worker time) AND the learner forward.

**UPGRADED 2026-07-24 — TRUNK ENRICHMENT IS KILLED 3-FOR-3 (ledger K9/K10, `designs/research_state/`).**
Owner verdicts closed two dangling experiments: **`--damage-reattend` (v31) "did nothing"** (a DIFFERENT
delivery form — attention over projected physics, not between-layer node injection — production for the whole
ai_v6_09→_13 lineage, then silently dropped) and **`--pubval-mode` (v43) "did nothing"** (a different KIND —
an exogenous human-calibrated dense target, held-out AUC 0.734, against an explicit `ai_v7_11_..._nopubval`
control). With the near-inert refine/threat finding below, that is **better ROUTING, better TARGET, and
fresher PHYSICS all null.** So the near-inert result below is NOT a detached-belief artifact — it is one
instance of a general pattern. **⚠️ MECHANISM CORRECTION (owner, same day): the first explanation —
"the trunk runs in 3-5 of 128 effective dims so it can't use more" — was a SCOPE ERROR.**
`rank_metrics.py:9-11` reports FOUR separate series (`trunk`/`value_cls`/`policy`/`vf_feat`); 3-5 is
**`value_cls`**. MEASURED: `value_cls` **3-4** vs `policy` **30-40** on the SAME body; `trunk` NOT YET
MEASURED. So the trunk plainly CARRIES 30-40 dims — **the critic isn't starved, it's failing to READ what's
there.** Never repeat "the trunk uses 3-5 dims". Better architecture-derived candidate for K9: every
value-side aux (`WinProbHead`/`ValueDistHead`/`PubValHead`) reads **`value_pooled`** — so pubval was piped
through the ~4-dim bottleneck it was meant to fix; D1/FitNets is the one that targeted `value_pooled` ITSELF
(128-dim hint) and it is the one that moved rank. **STANDING RULE: any new "give the trunk X" proposal
carries a HIGH build bar** —
say why it is not a fourth instance (a difference of FORM, not content) + bring a cheap pre-build probe. This
**ARCHITECTURE VERDICT after 8 offline probes (2026-07-25): NO obvious representational hole.**
Trunk HEALTHY (`rank/trunk_pr` 24→35 of 128 and RISING; raw-obs outcome AUC 0.623 → any trunk readout
~0.83). Value readout rank ~3 is APPROPRIATE not pathological (critic emits ONE scalar; value_pooled
AUC 0.833 vs the policy's 384 dims 0.835) ⇒ "widen/delete the value pool" DEAD, and v45's 51-bin
target correctly did NOT move rank (T1: +0.01). The 3-vs-50 comparison was apples-to-oranges. Ledger
P3. **P4:** the "degenerate secondary dims" are NOT a bug — the v24 table+math are CORRECT (19/19
canonical, flinch max 0.270 = 0.30×0.90 acc, par 0.600 = 0.30×2 Serene) but the block is
opp-active-level with NO defender axis, so it is SUPERSEDED by v35's per-defender `status_lands`
(KL 0.0005 vs 0.1446). DELETE ~56 dims (incoming secondary 10 + incoming effect 6 + outgoing
slp/psn/tox 12) for 1.3% of dependence. Remaining real defects: compute allocated INVERSELY to
dependence (P1 → top-K) and per-slot addressability (P2ii, NOT binding — owner: exploiters switch
human-like on the SAME architecture, so switching is a SIGNAL issue handled elsewhere).

**✅ THE P1 DELETE LIST IS ACTED ON (2026-08-03, v54 `gen3_op_block_trim_v1`, shipped).** Deleted from the
op: incoming per-status SECONDARY (10 dims, 0.1%), incoming believed-EFFECT (6 dims, 1.2%), the OUTGOING
slp/psn/tox secondary columns (12 dims), and `_topk_block` (the v30 lean top-K, 0 calls/forward). Net −28
dims off both projections (`out_dim` 835→807, `incoming_dim` 101→85); the unmasked-belief `w=sigmoid(...)`
read leaves `forward` entirely. NEW MEASUREMENT backing the outgoing-column claim: over the 773
`data/teams/` teams, **slp 0 teams / tox 0 teams / psn 1 team (0.1%)**, and gen3 has NO damaging move that
inflicts sleep at all — structural zeros, not merely low-dependence. `damage_topk_k` now means "the
incoming matrix's K" and RAISES without `--damage-matrices incoming`. **HONESTY GATE: this bought ZERO
CPU** — B=1 forward 4.276→4.289 ms, 3534→3525 calls/forward (≈9 aten calls on a dispatch-bound forward,
and the lean block already never ran). It is a dims/complexity change; do NOT cite it as a throughput win.
ARCH_SIGNATURE bumped (projection widths change). NOT deleted per P1's correction: v39
`--damage-matrices-outgoing-all` (#2 most-used) and the v34 outgoing matrix (6.3%) — the old `next_run_plan`
trim list is still backwards. Two PRE-EXISTING v48 stragglers surfaced while verifying (fail identically on
HEAD, untouched): `gen3_data_obs_parity_integration_test` (golden fixture pinned at obs 2992 vs the live
2889) and `poke_env_gaps/move_alignment_fuzz_test` (asserts the base-power obs scalars v48 deleted).

**P1 — the op's HEAD CONCAT is HEAVILY USED ✅ (ablation probe, 2026-07-25) — the counterweight that
REFRAMES K9/K10.** Those kills are all TRUNK; the post-pool concat into the HEADS was never tested.
`tmp/op_block_ablation_probe.py`, 4000 real eval states, exact producing snapshot, per-block zero →
masked-KL. Ceiling (zero whole op) = 0.9385 all / 0.4948 moves. **OUTGOING block = 0.6165 = 65.7% of
ceiling (75% of the moves ceiling)** — the single largest dependency; then OUTGOING-attacker-mx(v39)
21.4%, incoming-mx(v35) 15.4%, incoming per-mon 12.7%, status-land 8.8%, OUTGOING-mx(v34) 6.3%, CB
2.9%, effect 1.2%, **incoming secondary 0.1% = INERT**. SHUFFLE control > zero everywhere ⇒ the head
reads STATE-SPECIFIC content. **SYNTHESIS: the op WORKS; the working route is the HEAD CONCAT, not
the trunk — which EXPLAINS the trunk nulls (redundant with a head path already at full strength) and
vindicates providing multiplicative/threshold facts.** ACTIONS: (a) compute is allocated INVERSELY to
dependence — the INCOMING family is ~⅓ of dependence but ~64% of batch compute (`_damage_rolls` = 49%
of the op at B=256) → STRENGTHENS top-K; (b) `next_run_plan` trim list is BACKWARDS: v39
`--damage-matrices-outgoing-all` is the #2 most-used block, do NOT drop; drop `incoming secondary`
(0.1%), `incoming effect` (1.2%), v34 outgoing matrix (6.3%) instead. LIMITS: KL = USE not VALUE
("learns ≠ helps" still applies); zeroing is partly OOD; marginal not Shapley.

**K10a — the "it was a FIDELITY problem" escape hatch is REFUTED (probe, 2026-07-25).**
`tmp/refine_fidelity_probe.py`: 6000 real eval-trace states across 19 opponents under the EXACT
producing snapshot; lean `discrete_incoming` vs the full op, masked to ALIVE defenders in opp-active
states. Same-belief argmax agreement on the most-threatened mon = **91.8% damage / 80.6% P(KO)** vs
29% chance (as-deployed 83.4 / 76.9). **The coarse kernel was already a GOOD proxy — the trunk got a
fine signal and still did nothing** ⇒ a "belief → full-fidelity damage → attention" reorder is a
FOURTH K10 instance; NO-GO as a learning lever. Nuances: P(KO) RANK corr weak (spearman ~0.5 —
threshold saturation); the lean kernel OVER-estimates threat ~10-27%; belief staleness alone costs
~8 pts. Scope: INCOMING channel only (v36/v37 outgoing+status unprobed). The COMPUTE case for
deleting the refine loop (~19%) is UNAFFECTED. This
also DEMOTED the ai_v9 edge-bias experiment and PARKED the entity refactor
([[project_model_frontier_roadmap]], `designs/learning/entity_tokens_biases_pointers.md` F3).

**Lone survivor, HEAVILY SCOPED (owner correction, same day):** FitNets `--distill-value-feat-coef`
marginally RAISES value-fn dim (ledger D1) — tempting to read as "input enrichment fails, TARGET
enrichment works." **DO NOT lean on that.** D1 was measured on ai_v7_21, which ran `value_dist_mode=shaping`
only as an AUX (coef 0.2) with **`value_from_dist` ABSENT** — the scalar `value_net` was still the critic and
scalar MSE the primary loss. So FitNets was counteracting exactly the collapse pressure v45
(`--value-from-dist`, live on ai_v8_03) REMOVES → likely **SUBSUMED, not additive**, and D1 is NOT
independent evidence for v45 (competing treatments of one deficit). The tidy story is also post-hoc-fitted
to 4 points (noticed late: pubval was ALSO a 1-dim target, which makes it "fit"). **FREE DECISIVE TEST,
not yet run:** did `--value-from-dist` RAISE the critic's effective rank on ai_v8_03 vs its pre-v45 lineage
(`rank_metrics.py` / `tmp/value_rank_compare.py`, offline, zero training cost)? UP = mechanism real, expect
FitNets subsumed; FLAT = D1 was a scalar-MSE artifact and target-richness dies too.

LESSON: three built+run experiments sat un-adjudicated for months; the verdicts cost one sentence each and
closed a whole direction — AND the first synthesis drawn from them was wrong on a config detail the owner
caught immediately. Check the A/B's actual `cli_args` before generalizing from it.

**THE ORIGINAL FINDING (value > throughput):** under `--belief-grad-mode detached` (ai_v7_02's config) the refine
loop + 4 threat channels + belief heads are measurably NEAR-INERT — detach CUTS the gradient they exist to
provide, belief heads collapsed to chance (latent cosine 0.004-0.013, species-acc→baseline) across the whole
winning run, YET WR hit best-ever 0.92/1998 ELO driven by CRITIC-SHAPING (`--win-prob-mode shaping` +
`--value-dist-mode shaping`, the ONLY proven-to-help blocks, ~0 ms each). So ~19%+ of every forward is dead
weight in THIS regime. See [[project_plateau_research_2026_06_25]].

**DO NOT REMOVE:** `--damage-op` (spine — belief system feeds it), win-prob/value-dist shaping heads (proven
win, ~0 ms), the hard-require chain `--move-latent`→`--move-belief`→`--hp-type-belief learned` (last fixed the
opp-HP-immune GIGO [[project_opp_hp_immune_bug]] under `--unified-obs`). Gotcha: dropping `--damage-outgoing`
alone FAILS to start — `--unified-obs` desugars to masks that hard-require it.

**TOP-K+TAIL CANDIDATE PRUNING (2026-07-20 owner idea, designed not built).** Replace the full
C≈400-candidate sweep with exact physics for the top-K posterior candidates (K≈16-24, renormalized)
+ ONE aggregate tail channel `[P(tail), tail_worst_phys, tail_worst_spec]` from a tiny precomputed
per-(type,category) max-BP bound (damage monotone in BP×eff → the tail max needs no sweep; keeps
surprise-OHKO honesty; expectation error ≤ tail mass). HONEST YIELD SPLIT: on the CPU opponent
forward the C-sweep is NOT the top line (audit: `_damage_rolls` 0.17 ms vs refine loop 1.3 ms) —
the big CPU win stays the RANKED REMOVALS above; the C-axis cost lives on the LEARNER side, where
[B,n,C] activations dominate op memory (the 2026-07-20 OOM crashed inside `_nature_marg_ko/_ramp`
allocating [B,n,C]) — top-K shrinks those ~15×, possibly enough to drop `--grad-checkpointing`
(recompute is doubling the op's learner cost). DEPLOYMENT CLASS DIFFERS: the removals are
STRUCTURAL (projection dims shrink → fresh-run-only); top-K+tail is same-out-dim VALUES-only → a
version-gated forward toggle that COULD apply to the current weights IF an offline fidelity probe
passes (current ckpt, cheap-op vs full-op on the same eval battles: feature error + head-to-head
WR). Compile stays dead ([[project_throughput_compile]]: CPU-opp 0.70×).

**VERDICT: do NOT retrain ai_v7_02 for throughput.** (a) it forfeits the 0.92/1998 best-ever checkpoint;
(b) ~27% forward-cut ≠ ~27% FPS — CPU-saturated SubprocVecEnv barrier crushed a 2.1× transport win to ~5%
end-to-end, realized gain likely <20%; (c) throughput isn't blocking (plateau = skill/regime, GPU duty 56%).
Compile is ALSO dead (learner flat / GPU-opp OOM / CPU-opp 0.70× — [[project_throughput_compile]]). ACTION:
nothing now, but the NEXT fresh run (from-scratch anyway → no forfeit) should DROP `--damage-refine-rounds` +
the 4 threat channels (leaner/faster at low value risk); keep `--belief-grad-mode` fixed across that A/B and
watch forced-switch crater-share if also dropping outgoing-all (it targets a concrete forced-switch-blind-
offense defect = medium risk). Run-root model_config.json VERIFIED accurate (agent's "stale config" surprise
was a misread).
