# Ledger — hypothesis status table

The at-a-glance dashboard. Full Known/Not-known/Pros/Cons for OPEN levers live in
[levers/](levers/); the [README](README.md) holds the protocol + the frontier.

## The method (how a hypothesis earns a verdict)

Every behavioural hypothesis runs the **forensic template** (this confirmed self-KO and killed five of
six blunder categories), reusable via `python -m main.prober.query decision-table <run>`:

1. **Frequency** — how often does the pattern fire?
2. **Confidence** — `softmax(logits)[chosen]`. >0.3 = a *learned* preference; <0.1 = exploration tail (drop it).
3. **Reward** — already *punished* (reward < 0)? If yes, the reward isn't the bug.
4. **Critic dV** — `V(s')-V(s)` *positive* on the bad turn (critic over-values, neutralizing the reward
   in the PPO advantage)? = the self-KO mechanism.
5. **Recoverability** — does the **falsifier** confirm a materially-better action existed (MISTAKE/MIXED),
   or is it a symptom of a lost position (LUCK/NEUTRAL)? **Size ≠ recoverability.**

**Honesty gates** (every claim clears these before reaching "Known"): outcome-conditioning · falsifier
myopia (over-flags setup) · legitimate-in-context · learned-vs-exploration · **always adversarially
verify a confirming measurement** (overturned 3-for-3 this session).

## Status

✅ CONFIRMED · ❌ KILLED · 🔬 OPEN · 🛠 FIX BUILT

| # | Hypothesis | Status | Mechanism / evidence | Re-verify |
|---|---|---|---|---|
| H1 | Policy explodes **healthy** mons (self-KO over-valuation) | ✅🛠 | ~38% of Explosions at ≥80% HP, conf~0.5 (learned). Reward correct (−2.7), exploration ruled out, ① exonerated. CRITIC over-values the trade (dV +2.9 → PPO advantage +1.5). Fix `--self-ko-hp-penalty` built+tested, NOT shipped. | `decision-table <run> --cat selfko` (selfko dV_med) |
| H2 | **Attack type-mismatch** (resisted/immune move while a SE move sits in the same kit) | ✅ | Falsifier+type-chart confirmed; SMALL (~3–5 confident loss decisions). REPRESENTATION gap (reward+critic correct). Fix = obs effectiveness feature, not reward. Not built. | falsify MISTAKE corpus, best_label = a different attack |
| H3 | **Surprise-OHKO / unrevealed-threat** obs-coverage gap | 🔬 GO (gated) | Belief UNDER-FIRES on 52% of lethal healthy-stay deaths (pko<0.3). Provenance: 36% just-switched, 42% new-move (priors gap), 22% fully-known (calibration bug). CAVEAT: ~same rate in wins (56%) → pervasive. **Recoverability (n=381, stable from n≈60): 42% AVOIDABLE** (64% of those a SWITCH-to-a-wall — Swampert/Jirachi/Snorlax/Gyarados, ~1 mon), **33% LUCK** (belief was RIGHT, died to crit/roll — NOT coverage), **25% NEUTRAL** (committed). 42% > 20% kill-bar → **GO**, but half-belief/half-under-switching → must pair with `--switch-bias-weight`. | `models/saved_work/surprise_death_recoverability.py` |
| B1 | **Hidden-team belief** (in-place belief slots + supervised aux head) | 🛠 **BUILT (not run)** | Gen3 has **no team preview** → the ~3 hidden opp slots are ABSENT from the obs (a probe CAN'T recover them — the exact opposite of the FALSIFIED opp-ACTION head). Pre-build learnability probe: conditioning on revealed mons beats the usage prior by **+7pp recall / +8–10pp top-1**. BUILT (`claude/belief-head`, config 16): in-place unknown-mon slot tokens refined in-lineup + **Hungarian** (order-invariant) species+moves aux head; learns immediately (acc 0.08–0.16 vs ~0.003 chance). Subsumes the bench/switch-in half of H3. **UNMEASURED** whether it HELPS the policy (the honesty gate). | [levers/hidden_team_belief.md](levers/hidden_team_belief.md); `belief_labels_fuzz_test.py` |
| K1 | **Distributional value critic** lifts the strong-opp ceiling | ❌ | Strong-opp residuals SUB-Gaussian (tail-dom 0.33, exkurt −0.89) — no tail to re-weight. The "fat tail" was outcome-conditioning + the PP-stall reward artifact. | value-calib: V vs return-to-go residual shape |
| K2 | **Damage-magnitude** obs feature is the gap | ❌ | r²≈0.08 partly a marginal-vs-conditional artifact, but belief-conditioned r²=0.012; belief saturated above its floor; residual is *which-move*, not magnitude. | `probe damage_taken` (marginal vs belief-conditioned) |
| K3 | **① `value_active_readout`** causes the self-KO over-valuation | ❌ | ai_v5_9 (no ①) explodes 16.5% when-available vs ai_v5_11 (with ①) 13.7% — same. ("0 explosions" was a script bug.) | `decision-table` per-run explosion-when-available |
| K4 | **Recovery-move misuse** (full-HP heals) | ❌ | Low-conf (0.18) exploration tail, already punished; confident cases split 9W/9L (legit Recover-stalls). Direction FLIPS — falsifier flags a heal as the *better* move 17×. | `decision-table <run> --cat recovery` |
| K5 | **Setup-move misuse** (setup-into-death) | ❌ | Myopic-falsifier over-flag; clean core <0.12%; critic already prices it NEGATIVE (dV ~−4). Blind-setup-death is the H3 obs gap. | `decision-table <run> --cat setup` |
| K6 | **Stall/utility misuse** (Protect/Leech/status-on-dying) | ❌ | Reward-punished, NO critic over-value (dV~0), falsifier never craters on the dying-mon turn (lost earlier). | `decision-table <run> --cat stall` |
| K7 | **Switch pathologies** | ❌ | UNDER-represented among crater mistakes (24% vs 28%); ~2 confirmed voluntary switch mistakes in 350 losses (254/291 craters were FORCED replacements). | `falsify-scan <run>` (switch crater base-rate) |
| K8 | **No-op status-spam** (Spikes-at-cap loops) | ❌ | Real+learned, but no-progress clock already punishes it, critic prices the state as lost; ~74 dead turns in already-lost games. | `decision-table <run> --cat status` |
| **M1** | **MEASUREMENT CONFOUND — every zero-init inside the feature extractor was destroyed at policy build** | ✅🛠 **FIXED 2026-08-01** | SB3 `ActorCriticPolicy._build()` runs `features_extractor.apply(init_weights, gain=√2)` (`stable_baselines3/common/policies.py:617-631`), which ORTHOGONALLY re-initialises **every `nn.Linear` in the extractor**. `ortho_init` defaults True; nothing in this repo overrode it. So **13 Linears** documented as zero-init were random from step 0 in EVERY real run: `refine_proj` (v31/v33), `outgoing_proj` (v36), `status_in/out_proj` (v37), `film_pi`/`film_vf` (v44) — all documented "zero-init ⇒ identity-at-init ⇒ ON starts byte-identical" — and the belief heads `MoveBelief.move_head` / `SpreadBelief.*` / `HPTypeBelief.type_head`, whose zero-init is what makes the **cold-start posterior EQUAL the Smogon prior**. Measured before fix: max\|W\| 0.19–0.47 on every one. **Invisible to every test** because they all build the module or a bare extractor DIRECTLY, where the zero-init survives — only SB3-wrapped construction destroys it. Fix: `Gen3FeaturesExtractor.restore_identity_init()` called from `Gen3DualHeadMaskablePolicy.__init__`; the protected set is captured BY OBSERVATION at the end of `__init__` (not a hand-kept list) so a future zero-init module is covered automatically. | build a real `MaskablePPO` policy, assert every name in `fe._identity_init_zeroed` still has `max\|W\|==0` |

## Programme-level (the exploiter → distill loop, ai_v8)

The levers above are *behavioural* hypotheses about one policy. These are facts about the **training
programme** — what the exploiter/distill flywheel can and cannot do. Same honesty bar.

| # | Claim | Status | Mechanism / evidence | Re-verify |
|---|---|---|---|---|
| D1 | A multi-team exploiter can be **distilled into the generalist** and pay off | ✅ | ai_v8_14: 3 teachers / 23 teams → **ELO 1986±26 → 2055±29 (CIs disjoint)**, per-team piloting on the taught teams **0.438 → 0.710** (≈ the def-10 specialist's own 0.72), far-z FiLM range +70%, head-to-head 0.228→0.36. Bot-anchored ELO ⇒ not a self-play bubble. | `python -m main.elo <run>`; `tmp/pool10_perteam_eval.py` |
| D2 | Distilled per-team skill **washes out** once the teachers are removed (⇒ O(N) scaffolding forever) | ❌ | ai_v8_15 arm A′ (no distill, no teachers, no team-PFSP, **frozen pool** ⇒ forgetting not obsolescence): 0.710 → 0.6875 → 0.645 → 0.6425 → **0.6475**. Decay early, decelerating, **STOPS** — 3 flat points / 9M steps. **Equilibrium ≈0.645 = ~76% of the gain retained, unaided** (floor 0.438). ⇒ **teachers can be RETIRED**; distillation is bootstrapping. Corollary: the plateau was **optimization difficulty**, not objective indifference. | `tmp/retention_probe.sh <run> 40` vs the FIXED ai_v8_04 ref |
| D3 | Retention is a **leaky-bucket EQUILIBRIUM**, not binary keep-or-forget | ✅ | The D2 curve's shape *is* the evidence: decay stops where restoring force (∝ P(team)×value) balances erosion (interference + gradient-noise diffusion). Predicts arm B (`--team-pfsp onesided`, P(team) ~2–3× cap-bounded) **raises the level, does not zero the decay**. | arm B: does the plateau land above 0.645? |
| D4 | The N=20 exploiter ceiling is **conditioning-signal starvation** (not capacity) | ❌ **KILLED — it is a COUNT problem** | **5-arm factorial, 2026-07-28** (`designs/ai_v8/design_conditioning_ceiling_arms.md`). Plateau WR vs the frozen target, ≥4 pooled cycles, every arm byte-identical to the `ai_v8_12` baseline but one field. **COUNT (20→10 teams) +0.077, CI [+0.046,+0.108] SIGNIFICANT** — on its own ≈ the entire 0.076 gap, and near-identical on clustered (+0.076) and random (+0.078) sets. **CONDITIONING (per-team LUT) +0.028, CI [+0.001,+0.055]** — real but MARGINAL and 2.8× smaller. DIVERSITY −0.022 [−0.053,+0.008] n.s. (consistent sign at both N). No interaction: the effects simply add. **MECHANISM:** zero-init vs random-init codes give the SAME result (+0.004 n.s.) from OPPOSITE geometries (`lut_code_dist` 0.15 vs 1.0) ⇒ FiLM's gain is a **SHARED modulation, not per-team specialisation** — which is why every per-team-signal improvement failed, and why higher-RANK conditioning (LoRA/MoE) would not help either. **N=10 GENERALIZES off the clustered set** (0.700 random vs 0.725 clustered, n.s.), so the "two N=10 exploiters" plan HOLDS — that assumption was previously untested. **Corrections on the record:** conditioning read n.s. until the 5th arm landed (so the claim is "small but real", NOT "does nothing"); and the first pooling double-counted a shared baseline, overstating precision. **Practical: stop raising N; run N≤10 exploiters and distil.** Open: where the count cliff sits (N=5/3), and whether the diversity cost COMPOUNDS across batches. | `tmp/lut_verdict.py`, `tmp/z_spread_compare.py` |

**Retired follow-ups (re-priced by D2):** arm B `--team-pfsp onesided` = *optional* (recover the last
~24%), arm C teacher-as-opponent = lower priority, arm D always-on distillation = **not needed**.

## GO-TO-BUILD queue (the next FRESH run stacks these)

Posture shift (owner, 2026-06-12): stop the kill-treadmill, start **committing**. No single big lever
remains → stack the independent, cheap, amortizable levers into one fresh run and measure the aggregate
(ELO + bot-wr). The build bar (README → Decision posture) is LOWER than "Known": positive EV +
falsifiable-after-build is enough. Every entry must clear the **amortizability gate** (L1–L4; no search
on the model). Candidate stack (pending the amortizability pass + owner confirm):

| Lever | Bucket | State | Falsify-after-build (the metric that must move) |
|---|---|---|---|
| H1 self-KO HP penalty | reward (L2) | SHIPPED (`--self-ko-hp-penalty`) | healthy-mon self-KO rate falls; selfko dV no longer +; wr non-regress |
| H3 surprise-OHKO coverage + under-switching | obs (L1) + reward (L3-ish) | GO (gated) | death-turn pko rises AND switch-to-wall rate rises (BOTH) |
| H2 attack type-mismatch obs feature | obs (L1) | confirmed-small, cheap | resisted/immune-into-available-SE picks fall |
| B1 hidden-team belief aux head | belief (L3) | **BUILT** (`--opp-belief-aux-coef`), not run | `belief_species_acc_above_chance` climbs AND surprise-OHKO/hidden-mon crater share falls AND wr non-regress |
| _amortizability-pass output_ | TBD | pending | the grind/upstream decomposition's greenlit lever(s) |

## Active runs / context

- **ai_v5_11_tail2_53m_0611** — done at 53M. Rebalanced (vf_coef 0.5→0.25); `value_share` landed ~0.6
  (partial); ELO ahead of ai_v5_10. The forensic corpus for this session.
- **ai_v5_12_bias_05_N_0612** — LIVE (bias-redesign; orthogonal to H1/H3; trained on pre-H1 code).
- **Next run** = fresh `--self-ko-hp-penalty 2.5` (H1 A/B; resume-immutable → fresh, not a resume of ai_v5_12).

## Re-verify tooling

Consolidated in `src/main/prober/forensics.py` — re-verification is one command on ANY run:
```bash
python -m main.prober.query decision-table <run> --outcome loss             # per-category forensic digest
python -m main.prober.query falsify-scan   <run> --outcome loss --max-battles 200   # luck-vs-mistake bracket
```
`ProbeSession.decision_table()` / `.falsify_scan()` for scripts. The `outcome='?'` discovery bug
(eval-shard `s<N>_` filenames) is fixed, so `--outcome` filters work.

## Cross-refs

- **[`measurements/`](measurements/)** — the raw audit outputs behind the numbers
  quoted here and in `designs/ARCHITECTURE.md`. Each file carries its own checkpoint, step,
  state count and date. **A measurement is scoped to the model AND config that produced it**:
  the 2026-07-25 P1 per-block table was quoted as current for two weeks after the config it
  measured stopped existing, and its headline is reversed by the gen-3 replacement.


Deep notes in agent memory: `project_floor_leak_critic_selfko`, `project_distributional_critic_verdict`,
`project_incoming_damage_outcome`, `project_model_frontier_roadmap`. The prober (`src/main/prober/`) is
the forensic engine; this folder is its conclusions.

Programme-level (D1–D4): `project_multiteam_distill_payoff`, `project_distill_retention_ablation`,
`project_exploiter_fork_vs_scratch`, `project_double_sided_recipe`. Reports:
`designs/ai_v8/impl_step_retention_ablation.md`, `designs/ai_v8/exploiter_batch_strategy.md`,
`designs/learning/conditioning_architectures.md` (§5b — the FiLM/SNR diagnosis behind D4).

## ⚠️ Standing caveat on the identity-at-init experiments (M1, 2026-08-01)

Until 2026-08-01 **no `zero-init ⇒ identity-at-init` claim in this codebase was true in a real run**
(M1). Two families of result were produced on top of that:

* **K10 "physics into the trunk is null, 3-for-3."** `refine_proj` / `outgoing_proj` /
  `status_{in,out}_proj` were the injection paths, and none of them started at identity — each began
  as a random orthogonal projection onto the token stream. The nulls may still be right, but "we
  eased physics in from zero and it did nothing" is **not** what was actually run.
* **D4 / the conditioning line (N=20 ceiling, both LUT arms).** v44 FiLM is documented as
  identity-at-init; in fact `film_pi`/`film_vf` injected random modulation from step 0.

**This does NOT invalidate those conclusions** — the arms were internally controlled and the confound
applied to both sides of each A/B. It does mean the mechanism stories ("started at identity and never
moved", "the generator's gradient is proportional to a tiny residual") were reasoning about a model
that did not exist. Any RE-RUN of a K10- or D4-family experiment post-fix is measuring something
different from the original and must not be compared across the boundary.

Also affected, and worth a thought before the next belief experiment: the belief heads' cold-start no
longer equals the Smogon prior in any historical run, so "cold-start == prior" baselines in the
v20/v25/v38/v40 notes describe the intended design, not the executed one.

## Gen-3 40M gate (2026-08-08) — the concat re-read + coverage probe

Run `run_20260807_135637` (gen-3: all 15 edge families, K=6, E9 recency) completed 40M clean;
final aggregate win rate 91.4%. Reports archived IN THE RUN DIR
(`edge_audit_gen3_40M.json`, `incoming_cond_gen3_40M_6k.json`, `coverage_probe_gen3_40M.json`);
all at 6000 final-step eval-trace states on `final_model.zip`.

* **ELO (anchored, complete-run fits):** gen-3 **2131 ± 32** ≈ gen-2 2130 ± 31 > gen-1 2108 ± 31;
  matched-depth @24M: gen-2.5 2069 > gen-2 2029 ≈ gen-3 2023 > gen-1 2008. The gen-3 additions
  (K=6 everywhere + E9 recency + full families) are ELO-neutral vs gen-2 at 40M.
* **Concat arm (5th replication):** flips 27.45% / |ΔV| 5.36 — still ≥ all-15-edges-off
  (24.65% / 2.27) on BOTH axes ⇒ the deletion precondition stays unmet; branch B (re-home) holds.
  NOTE the flip GAP collapsed (9.6M: 23.7% vs 13.9% = 1.7×; 40M: 1.11×) — the edges kept growing
  (all-edges flips 13.9→24.65%, d2 7.6→16.3%) while the concat plateaued — but the |ΔV| ratio
  stayed ~2.4× (9.6M: 3.1×): **the residual concat dependency is increasingly CRITIC-side**,
  exactly the owner-amendment's two-route reading (PV/token-injection required for the critic).
* **Sub-block localization CONFIRMED at 40M** (shuffle-controlled flips): `in_matrix` **18.32%**
  of FULL_CONCAT 22.37% (≈82% of the concat's state-specific flips; 9.6M: 16.27 of 18.58) ·
  out_active 9.47 · in_permon 6.60 · in_cb 2.27 · out_status 1.08. "Re-home in_matrix, BOTH
  directions (policy + critic)" is now the settled, twice-measured target.
* **Coverage probe (§2b.4, NEW — labels exact from the op's own in_matrix):** joint cross-pair
  aggregates are ALREADY linearly decodable at the head boundary: `n_threatened` r² 0.80 (vf) /
  0.69 (pi), `best_move_breadth` 0.75 / 0.64, `safe_pivot_exists` AUC 0.97, shuffled controls ≈ 0
  (positive control `act_threat` pi 0.69). Per the reconciled §2b.4 read (the probe is a ROUTE
  CHOOSER — 42c9c69) the "decodable" branch fired against PV. **SUPERSEDED SAME DAY by the
  OpTensors split audit + owner adoption (design_op_tensors.md §8; owner: "no more concat"):
  the 7a token-content route is ALSO dead — the concat's dependence is the per-move HEADER,
  already E4 seat content, so the critic's gap is READOUT not delivery (act_threat vf r² 0.33
  vs pi 0.69) → k seed reads over `our_mon`; OA1/PV survive only as `REDUCE(how)` settings.**
  (vf > pi on every joint
  target while pi > vf on the per-action control — the critic reads board-level structure, the
  policy per-action magnitude; consistent with |ΔV| being the concat's stickiest axis.)

_Last updated: 2026-08-08._

## 2026-08-10 — STEP 0 of the pair-reduction plan (gen-5 final, 24M, stratified n=6000)

* **The audit's legacy site went structurally BLIND at v61 — and that blindness is itself the
  measurement.** `gen5_op_block_split_24M.json` (site=assembler): every arm 0.00% flips; only
  FULL_CONCAT moves |dV| (4.44 zero / 1.97 shuf) — under `gen3_no_concat_v1` the assembler's
  damage_block argument feeds ONLY the `MultiSeedValueReadout` per-mon rows. The tool gained
  `--site op` (perturb the op's RETURN before any consumer binds it; recorded in provenance).
* **`gen5_op_block_split_24M_site_op.json` (site=op) — the honest post-concat read:**
  FULL_CONCAT **65.07% shuffle flips / kl 1.82 / |dV| 5.54 zero (3.30 shuf)**; every sub-arm
  (in_matrix 522, imx_HEADERS/CELLS, hdr_*, cell_*) **exactly 0.00%**. Interpretation:
  **dims 85–660 of the block are write-only at every site** (probes only) — the block's live
  content is the REDUCED per-defender statistics + CB tail (dims 0–85), consumed by the pointer
  switch cells (pi) and the seed rows (vf). The E4 seats / d3-s3 edges / prefuse consume the
  op's INTERNAL tensors and are measured by `edge_ablation_audit`, not by any block site.
* **Step-0 verdict for §8.1:** the downscope trigger ("unsuppressed `imx_CELLS` ≲7% ⇒ cheap rungs
  only") was keyed to a quantity that is structurally dead post-concat; the rule's INTENT is
  answered decisively the other way — the hard-max's reduced output is now the policy's dominant
  op route (65% flips). **FULL LADDER PROCEEDS: G1 bake-off → delivery wiring → G7.** Also: the
  seed route carries |dV| 4.4–5.5 while rank-COLLAPSED (eff-rank 1.0 all run) — the VICReg
  un-collapse (v62, `--value-seed-vicreg-coef`) is upside on an already-load-bearing route.
* **Refund note (OpTensors territory):** the op need not materialize the in_matrix region into
  the flat block at all (522/660 dims, write-only); `out_dim` could drop to 138 once the probes
  read the internal tensors instead.

* **G1 v1 (n=101 beam-switch targets, 5-seed linear probes, chance 0.167):** R0 0.457±0.093 ·
  R1 0.467±0.056 · **R1+R0 0.505±0.065** · SKYLINE(2800d) 0.486±0.082. Nothing beats R0 beyond
  seed spread at this n; R1+R0's +0.048 SUGGESTS complementarity (matches the add-beside design)
  but is within noise; the SKYLINE is overfit-limited (2800 dims / 80 train rows) and cannot
  support a "no headroom" claim. NOT a kill, NOT an endorsement — expanding to ~600 scanned
  targets for tighter CIs (`tmp/g1_bakeoff.py`, resumable); G7 remains the capability gate.

* **G1 FINAL (n=299, 5 seeds): the reduction ladder FAILS its pre-registered bar.** R0 0.403±0.034
  · R1 0.423±0.063 · R1+R0 0.423±0.031 · SKYLINE(2800d) 0.413±0.037 (chance 0.167). No rung beats
  R0 beyond seed spread; the skyline shows no linear headroom beyond single-α mixtures. Read with
  step-0: the hard-max route is DOMINANT (65% flips) but its CONTENT already summarizes the pair
  grid about as well as any mixture, linearly. Per the doc's own framing (G1 = "THE decisive early
  gate": a rung that cannot beat R0 from ground-truth cells will not learn to in the loop) this is
  a measured NULL for the W-rung line → **delivery wiring + G7 are NOT justified for gen-6 on this
  evidence**; the rungs stay in the codebase inert (tested, byte-identical, ~free). Caveats, per
  §10.2/§10.3: the target is the beam (our own critic's preference, not ground truth), and a
  linear probe may be blind to the BEHAVIORAL hedging capability G7 was designed to detect —
  reviving G7 is an owner override, not a data conclusion. Lesson pattern-match: "gate a lever on
  whether the quantity PREDICTS performance" (the code-rank lesson) — applied here BEFORE spending
  2×2M G7 forks.

## 2026-08-11 — the op `in_matrix` REFUND is a measured NO (out_dim 660→138 not worth doing for perf)

I proposed shrinking the op's returned block 660→138 on the strength of the step-0 finding that
dims 138–659 (`in_matrix`) are write-only. Measure-first killed it. All numbers CPU-only, box
busy (load **18.0–26.2 on 16 cores**, factor ~1.1–1.6); ratios are the load-stable signal, and the
conclusion sits 2–3 orders of magnitude from the decision boundary so contention cannot flip it.
GPU-side is **UNMEASURED** (the host's `nvidia-smi` is broken by a driver/library mismatch).

* **The materialization is ~0.1% of the op forward** (B=1024): final `cat` 0.014 ms + `out_gain`
  over the 522 extra columns 0.008 ms + stash clone 0.011 ms ≈ **0.033 ms of 29.2 ms**. At B=1 it
  is ~2 µs of 1.44 ms. The op is itself only 7–12% of a policy forward, so the lean block is
  ~0.01% of a forward and even DELETING the imx computation outright caps at **0.7–1.3%**.
* **Bytes are a rounding error**: 8.55 MB at B=4096 against the op's own `[B,6,370]` candidate
  intermediates at ~146 MB EACH at B=16384.
* **No backward saving**: imx under `no_grad` measured +2.2% at B=256 and −1.1% at B=1024 —
  sign-inconsistent, noise.
* **The structural catch:** `last_raw_block = block.detach()` (`damage_op.py:2910`), so keeping the
  probes' 660-wide view REQUIRES still computing and concatenating the 522 dims. A lean *return*
  can only ever refund the gain-multiply and the slice. Refunding the COMPUTE means making
  `_incoming_matrix` opt-in, which changes `last_raw_block` semantics for every probe and strands
  `last_topk_idx`/`last_topk_w` (set inside it, read by the prober for exact move names).

**Premise correction that matters for any future attempt:** `pointer_cells` does NOT read only the
first 85 dims — it reaches through **138** (`ob = incoming_dim` = 85 for the move cells,
`st0 = ob + _DMG_OUTGOING` = 130 for status; `damage_op.py:1251-1259`). So the outgoing(45) and
status(8) regions are LIVE and 138 is the correct lean width. Independent confirmation of the
write-only result: with imx removed from the graph entirely, the returned live prefix is
**bit-identical**.

**Disposition:** not doing it for throughput. If it is ever done, argue it as an API/clarity change
(a 138-wide contract that cannot be mis-sliced) and judge it on that. Harnesses kept:
`tmp/op_phase1_{measure,context,backward}.py`, write-up `tmp/op_refund_measurements.md`.

## 2026-08-11 — BOTH seed mechanisms produce ONE-DIMENSIONAL differentiation (structural, predictable)

Measured on matched 4.8M checkpoints, same 512 stratified states:

| run | mechanism | uncentered PR | **centered PR** (deviation dimensionality) |
|---|---|---|---|
| gen-6 | VICReg repulsion | 1.037 | **0.846** |
| gen-7 | per-seed quantile | 1.048 | **0.835** |

**Identical.** The quantile objective delivers everything it promised at the readout — spread
0.007 → 0.93, crossing_rate pinned at 0.000 (perfect τ-ordering), out_cos 1.00 → 0.93 — yet the
seed deviations still occupy ONE direction, exactly like the repulsion arm.

**The cause is structural and should have been predicted:** the per-seed prediction is
`p_k = w · out_k` through ONE SHARED readout `w`. Four different scalars require the seeds to
differ **along w** and constrain NOTHING orthogonal to it. A scalar-per-seed target — even k
distinct ones — can only ever demand a 1-D differentiation. The shared readout was chosen to stop
the HEAD faking the spread (it does), but it simultaneously caps the achievable dimensionality at
one. `out_effective_rank` 1.10 is therefore the objective working as specified, not failing.

**If directional multiplicity is wanted, the target must stop being a scalar-through-a-shared-line.**
Two candidates, both preserving the no-faking property: (a) **FROZEN orthogonal per-seed readout
directions** — seed k read through a fixed random unit `w_k`, so hitting τ_k requires moving along
`w_k` specifically and the k constrained directions are orthogonal by construction (the head cannot
adapt because the directions are frozen); (b) **vector targets per seed** — each seed owns a CHUNK
of the distributional head's atoms, so its job is multi-dimensional.

**Open question this raises and does NOT answer:** whether directional multiplicity is worth
anything. Two mechanisms now agree the readout wants ~1 direction; that may be the model telling us
one direction is all this readout needs.

## 2026-08-11 — G2a SEAT COVERAGE measured (the two ceilings on design_opponent_intent.md)

Run on gen-5's traces (400 battles, 16410 decisions; `tmp/g2a_seat_coverage.py`). **Neither
ceiling blocks the design; one materially bounds `β` v1.**

**α's ceiling — 89.3%.** Of 1233 scoreable decisions (the opponent used a NAMEABLE move), the
op's K=6 threat seats CONTAINED the move they actually clicked **1101 times (89.3%)**; 10.7%
masked. Seat-set size was 6 in every case. Read: `α`'s discrete-support constraint costs ~11% of
move-decisions, NOT the "thin, possibly-unrepresentative slice" §8 feared — **the discrete
constraint is affordable and `move_belief_coef` is not a hard blocker for building `α`.**
*Caveat:* measured on the model that PRODUCED the traces; and 38% of decisions were unscoreable
(1487 `none`, 1248 `unknown`, 3822 explicit switches), so 89.3% is conditional on a nameable move.

**β's ceiling — 53.6%, and this one bites.** Switches are **24.7%** of all decisions (4056 of
16410) — a large share of the action space, confirming §5.2's "SWITCH is the highest-value single
slot". But only **53.6% of switches are to an ALREADY-REVEALED mon**; **46.4% bring a previously
unseen mon**, which v1 MASKS. So `β` v1 trains on barely half its target and is blind on the half
that is hardest. **This promotes B1 (hidden-team belief, BUILT 2026-06-13, never run) from
"optional upgrade if the mask rate warrants" to the thing β v1 is actually waiting on** — §4.3
predicted the shape (early-game switches go to unrevealed mons) but not that it would be ~half.

**Consequences for the build order:**
1. `α` (move axis) is GO on coverage — 89.3% is a healthy ceiling.
2. `β`'s unrevealed half is a real limitation; either accept a half-blind `β` v1, or promote B1
   into the same generation (it is structural, so it must ride a fresh run either way).
3. The **{ATTACK, SWITCH} fallback (§7a.4) is independently attractive**, not just a fallback:
   SWITCH is 24.7% of decisions, belief-free, zero mask, and is the one slot both ceilings agree on.

Cross-check: the summary's `outcome.opp.action` encodes switches explicitly (`switched_to:X` /
`X_sent_in`, 3822) — consistent with the species-change detector used for the rates above.
