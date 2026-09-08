---
name: project_code_rank_ceiling
description: "SUPERSEDED at N=20 — the LUT arm falsified the conditioning-signal theory; the code-rank/geometry findings here stand only as GENERALIST (719-team) measurements"
metadata: 
  node_type: memory
  type: project
  originSessionId: 57d18012-052b-4e57-91af-eab5d8b13241
---

> **Archived 2026-09-08** — SUPERSEDED by project_lut_conditioning_ceiling_result (its own banner says so) and project_count_dominates_conditioning. Preserved verbatim; nothing below is current.

> ⚠️ **STATUS (2026-07-27): the headline hypothesis was TESTED AND FAILED at N=20.** `ai_v8_16` gave a
> 20-team exploiter a FREE per-team code (`--zarch-lut add`, maximal ADDRESSING): **+0.024, CI
> [−0.016,+0.064], decisive positive ruled out** ([[project_lut_conditioning_ceiling_result]], ledger D4).
> Three corrections to everything below:
> 1. **The rank-COUNT argument never applied at N=20** — 20 teams < `zarch_dim` 32, so codes were never
>    crowded BY COUNT; only crowding by REALIZATION was live, and the LUT fixed exactly that. Count-crowding
>    is a **719-team GENERALIST** argument that was wrongly carried to a 20-team problem, where it remains
>    UNTESTED.
> 2. **The LUT tested ADDRESSING, not EXPRESSION** — FiLM is still DIAGONAL, the modulation family is still
>    `span(W)`. "Conditioning rank ≤ code rank" is untouched; LoRA/MoE stay formally untested. Not building
>    them is a COST call, not an entailment of this arm.
> 3. **The fix order below is demoted** — per-team LUT is the intervention that just failed; the VICReg
>    covariance term and behavioural-z supervision inherit the same premise and need re-justification.
>
> **Better-supported reframe:** ledger D1 holds **23 distilled teams at 0.710**, ABOVE the N=20 exploiter's
> 0.6725 (VERIFY the yardsticks — D1 is per-team piloting vs a fixed opponent, D4 is exploiter WR vs its
> target). If comparable, one head STORES ~23 teams' skill when handed a supervised target ⇒ the N=20 stall
> is a **DISCOVERY/optimization** problem, not storage. That also explains why converged gradient-cosine
> probes looked healthy: exploration interference lives in the learning TRAJECTORY, not the fixed point.
>
> **ARM 2 (2026-07-27) — a SECOND independent null, and it REFUTES "stop clustering."** `ai_v8_17`
> random-20 no-LUT (×1.52 code rank, PR 9.59 vs 6.30, MATCHED team strength 0.548/0.547): plateau
> **0.6230 [0.589,0.656]** vs the z-clustered baseline 0.6488 ⇒ **−0.026, ns**, where the crowding
> argument predicted BETTER. Arm 1 changed the code PARAMETERIZATION, arm 2 the TEAM SET — two
> orthogonal manipulations, both null. **So the "A RANDOM 20-team set beats a z-similar 10-team set
> on directions/team ⇒ stop clustering" recommendation below is DEAD.** The statistic was measured
> correctly (PR / Welch / permutation null / split-half all check out) — it is just not on the causal
> path to win rate.
> **⇒ GENERAL LESSON: gate a lever on "does this quantity PREDICT performance?", never on "is this
> quantity low?"** Everything in this memory passed the measurement-quality bar and failed the
> causal-relevance bar.
>
> **RETRACTED (was in this banner): "the random-init LUT was the CORRECT choice; a zero-init LUT would
> reproduce the ill-conditioned geometry."** Both halves are wrong. (a) The THEORY: a free per-team
> code's gradient is `(Wᵀδ_i)·J_LN` — the full `∂L/∂z` on team i's samples, never scaled by the
> compositional residual ε — and `W` is trained at the fork, so zero-init merely STARTS neutral. (b)
> The EMPIRICS: the claimed −0.04 fork handicap does NOT replicate — arm 3 uses the same random init
> and its `+0M` LUT delta is **+0.070** vs arm 1's **−0.040** (one n=200 cycle each, ±0.069).
> Process note: I flagged the init as a confound, then deferred to the design doc's counter-argument
> instead of doing the one-line derivation. **Derive, don't defer.**
>
> The geometry measurements below remain VALID as descriptions of the 719-team generalist's code space.

Measured 2026-07-26 via `tmp/team_capacity_probe.py` (live ai_v8_15 @300M, 94 teams, no training
run). Write-up: `designs/learning/self_discovered_archetype_latent.md` -> "MEASURED (2026-07-26):
the CODE-RANK ceiling".

**Why 10 z-similar teams reach single-team parity but 20 capture only ~60%:**
- **Local code rank.** A z-similar cluster spans FAR fewer dims than the global code: k=5 -> PR
  2.99, k=10 -> **5.37**, k=20 -> **8.55**, k=30 -> 10.66, vs a GLOBAL PR of 16.18/32. Each extra
  team adds only ~0.3 new directions. Directions-per-team 0.49 (k=10) -> 0.41 (k=20).
- **FiLM is LINEAR in z**, so the modulation family available to separate a cluster has rank
  EXACTLY the cluster's local code dim. 10 teams share 4.9 knobs; 20 share 8.2.
- **The design tautology:** z_arch is a COMPOSITION encoder and the clusters were chosen to be
  z-SIMILAR — you pick teams the encoder thinks are identical, then ask it to tell them apart.
  With r=+0.125 (code vs behaviour similarity) the selection is adversarial to the conditioner.
- **Honest accounting:** capacity predicts 84% capture at k=20 vs ~60% observed, so code rank owns
  only ~40% of the shortfall. The leading remainder is REVISIT LOCALITY — equal per-team BUDGET
  fixes density but NOT freshness (19 other teams between visits vs 9). That confound was not
  controlled in the original experiment.
- **Gradient conflict is dead EVERYWHERE**, trunk included: shared-trunk per-team |cos| = 0.054
  with frac<0 = 0.46 (symmetric about 0 = noise, not opposition); film_pi 0.037. Global packing is
  also not binding (pairwise |cos| flat at 0.178 in k, ABOVE the Welch floor at every k).

**ROOT CAUSE — nothing is shaping z's geometry right now:** `zarch/recon_bce` = 1.2e-4 with
`recon_topk_acc` = 1.0 (anchor SATURATED, ~zero gradient); `zarch/vicreg` = 0.0 because the floor
`relu(1-std)` is slack at std 1.87 (INACTIVE); and the VICReg impl is **variance-only — the
covariance/decorrelation term that actually spreads a latent was never wired**.

**ORDERING CONSTRAINT (kills the tempting fix): conditioning rank <= code rank.** A richer map
(conditional LoRA / hypernetwork / MoE) on a rank-5 input is still rank-5. Do NOT climb the
conditioning ladder before fixing the code.

**Ranked fixes:** (1) **free per-team embedding (LUT)** for a fixed team set, or `z = f(roster) +
residual_t` — codes orthogonal BY CONSTRUCTION, bottleneck vanishes, smallest change; (2) wire the
VICReg covariance term / raise the variance floor above 1.87 so it binds; (3) behavioural
supervision of z (specialists as labels — also fixes r=0.125); (4) control interleaving in the
k=20 arm; (5) per-team distillation.

**Falsifiers:** recompute local PR on the EXACT clusters used (probe takes a team list; predict
8-9 dims for the 20-cluster); and re-run k=20 with LUT codes — 60% -> ~100% validates the whole
encoder-side program.

**SELECTION RULE IS FREE CAPACITY (measured):** directions/team at k=20 — kNN(z-similar) 0.41,
RANDOM 0.52, farthest-point 0.56. **A RANDOM 20-team set beats a z-similar 10-team set (0.49).**
Extrapolating, ~25-27 z-DISTANT teams per specialist at today's per-team rank ⇒ ~2.5× the teams for
zero compute. (Regime-dependent: if z ever becomes behaviour-aligned, clustering is right again.)

**POLICY-SIMILARITY z_arch — PROBED 2026-07-26, mostly NO** (`tmp/zarch_behavior_alignment_probe.py`,
`tmp/task_affinity_reliability_probe.py`; full write-up in the z_arch learning note):
- Grounding z in OBSERVED PLAY is a TRAP: the behaviour-fingerprint kernel has PR 3.28/21 vs the
  code's 10.33 — it would CUT rank 3×. And that low rank IS the amortization gap (the generalist
  plays every team alike), so it would lock the disease in as the target. "Rich demand, thin supply."
- The ROSTER barely predicts behaviour: `z→fingerprint` CV R² 0.054, `species multi-hot→fingerprint`
  0.031, against a split-half ceiling of 0.78 (7% / 4% of reliable variance). So `z=f(roster)` has a
  LOW CEILING on behaviour alignment however it is supervised. Corroborates r=+0.125 independently.
- **The gradient-affinity kernel does NOT REPLICATE**: split-half r = +0.097 (reliability 0.177), and
  `PR(K_ū)` = 13.37/15 vs a NOISE NULL of 14.79 — BELOW the null. z is already at 93% of the
  achievable alignment. ⚠️ **RETRACTED: "PR(K_ū)=17/21 ⇒ real headroom" and "~17 distillation anchors
  instead of 719" were NOISE ARTIFACTS — never cite them.** (Random 1024-dim vectors give PR ≈ k.)
- It is a POWER result, not impossibility: estimating a DIRECTION in 1024 dims from ~86 decisions
  needs n~O(d). De-risk cheaply by maintaining a ū EMA bank in training (2.9MB, free via one hook at
  the FiLM site) and just LOGGING `K_ū` split-half reliability — byte-identical, no loss. >0.4 ⇒ live.
- **The failure of the kernel route ARGUES FOR the free-code route**: a per-team LUT never has to
  ESTIMATE affinity — it receives gradient ∝ ū_t every sample and accumulates it implicitly.
- Lit names for the kernel route: kernel target alignment / CKA (loss form), task affinity / Task2Vec
  (target), Multi-Task Relationship Learning (classical framing), FiLM/hypernet context (consumer).

**CAVEAT (load-bearing):** all numbers are from the 719-team POOL GENERALIST, not from a 10-/20-team
specialist where the encoder was trained on the cluster; and n is 15-21 teams with ~86-173 decisions
each, so everything here is power-limited. Run the falsifiers before acting.

Related: [[project_sampling_snr_analysis]] (the companion probe that refuted interference +
starvation), [[project_multiteam_distill_payoff]].
