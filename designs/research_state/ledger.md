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
| K11 | The extractor-output **ReLU kills units**, so a smooth signed activation (SiLU/GELU) is worth a generation | ❌ **KILLED — 0 dead units** | `pi_features = ReLU(projection(...))` is the whole interface to both towers, and it is the ONE ReLU with no LayerNorm after it (its output feeds the SB3 tower's first `Linear` directly), so a unit that drifts negative has nothing to rescue it. Plausible — and **false**. Probe on gen-12 @14.2M (`tmp/relu_deadunit_probe.py`, forward hooks on `projection`/`value_projection`, **2479 real greedy on-distribution decision states**): **0/512 dead in BOTH heads** (rule of three ⇒ true activation rate <0.12% at 95%). Also **0 always-on** — every unit genuinely modulates. Mean active fraction **0.465 pi / 0.483 vf** (median 0.428/0.457) = a textbook-healthy, roughly zero-centred gate. The near-dead tail is small (6 pi / 4 vf under 1%); ~17% of units fire on <10% of states, which is sparse specialisation, not death. ⚠️ **The pilot at n=97 read "3 pi / 5 vf dead" — pure sample artifact** (at n=97 a unit firing 1% of the time reads dead 38% of the time); the dead COUNT is meaningless without the rule-of-three bound, which is why the probe prints it. ⚠️ **Do NOT read the "≈55% of pre-activation mass clipped" figure as lost information** — the network TRAINED under ReLU and arranged its representation so the discarded half carries what it does not need; that number describes ReLU's operating point, not a cost. **The surviving ORTHANT leg (a signed concept costs a channel PAIR) is ALSO dead — measured, not argued** (`tmp/relu_capacity_probe.py` + `_followup.py`, n=4944 states, n/d=9.7x). It is a CAPACITY argument, and capacity has a precondition nobody had checked: it costs nothing unless the interface is full. It is not remotely full. Effective dimensionality (participation ratio) of the post-ReLU interface is **26.9 pi / 30.8 vf of 512 — ~5-6% of the budget** — and the projections are COMPRESSIONS (1177→512, 1369→512), so full rank is reachable and the deficit is a property of the learned representation, not a structural cap. Spending 2 channels for a sign is free when ~480 contribute almost nothing. Q2 tested the mechanism directly and found it ABSENT: **0 anti-correlated pre-activation pairs** at corr<−0.8 in either head (most negative −0.711/−0.721; only 0.3%/0.17% of pairs below −0.5), so the model is not visibly paying the pair tax at all. ⚠️ **A wrong analysis was caught here and is worth keeping:** the first follow-up compared the tower's raw read-weight energy `‖Wv_k‖²` to the variance curve, found it spread over ~460 of 512 dims, and concluded the tower amplifies low-variance directions. That is the NULL — `‖Wv_k‖²` measures W's ORIENTATION only and is flat in k for isotropic W; the measured curve matched a matched-scale random-W baseline to within ~1pp. The correct quantity is variance TRANSMITTED (`λ_k‖Wv_k‖²`), whose effective dim is **23.2 pi / 30.8 vf** — i.e. the PR reading stands. Residual honest limits: all of this is variance-based (a low-variance direction *could* still be decisive — an ablation→ΔKL sweep would close it), and pairwise correlation cannot see a sign encoded in a distributed subspace rather than a clean pair. Swap is `ARCH_SIGNATURE`-bumping + fresh-only ⇒ cannot be A/B'd mid-run, cannot ride another generation without confounding it. **Verdict: BOTH legs measured dead; do not spend a generation slot.** | `python tmp/relu_deadunit_probe.py --ckpt <ckpt> --battles 70`; `python tmp/relu_capacity_probe.py --ckpt <ckpt> --battles 160` (pre-registered metrics — re-run if the activation ever changes) |
| **M2** | **MEASUREMENT CONFOUND / LIVE-RUN DEFECT — `value_from_dist` orphans the ENTIRE critic delivery chain, and has since gen-9** | ✅🛠 **FIXED by v89 `gen3_value_pooled_routes_v1` (`1fa4733`, 2026-08-16)** — a parallel session built exactly the M2 recommendation, with the same evidence (gen-12's `value_entity_pool.out_proj.weight` + `intent_value_reduce.proj.weight` bit-exact ZERO after 25M steps, the 0.4998 absmax that "looks trained", `value_threat_proj` the one live route at 0.117). Every value route now INJECTS additively into `value_pooled` through a zero-init D_MODEL projection (the `value_threat_inject` precedent generalised) — one seam `_value_pooled_routes` feeds both the dist critic and the scalar `value_net`; `value_route_gradient_test.py` backprops the critic through every registry route and fails on any zero gradient, so the next route is covered by construction. Five routes were dead for gen-11+gen-12 (intent_value_reduce, value_entity_pool, intent_threshold vf-half, value_clock, value_intent). Kept below as the RECORD (the finding is what drove the fix). | `policy._critic_value` (`policy.py:175-181`) returns `value_dist_head.mean(...)` when `value_from_dist` is true and **never calls `value_net(latent_vf)`**; `value_dist_head` reads `value_pooled` straight from `CLSPool`, i.e. UPSTREAM of the assembler. So `latent_vf` is computed every forward and discarded, and everything feeding it gets ZERO gradient. **Measured on gen-12: 17 params byte-identical (`max\|Δ\|=0.000`) from the 2.4M checkpoint to 14.2M** — `intent_value_reduce.proj`, `assembler.seed_readout.*`, ALL of `value_entity_pool.*`, `value_pre_norm`, `value_projection`, and both `mlp_extractor.value_net` layers — while 615/674 params moved normally. **Causally confirmed by contrast:** the SAME seed params in gen-7 (`value_from_dist=False`) moved by up to **1.109**. `value_net.weight/bias` drift is PopArt's POP rescale (it rewrites W,b outside the optimizer), not learning. **CONSEQUENCE 1 — gen-12's headline critic arm is VOID:** `--value-entity-pool` (v80 UVR) feeds only `vf_combined` → the dead chain; its output is **identically zero** (`max\|out\|=0.000e+00` over 4299 real states) and its zero-init `out_proj` can never leave zero. The run cannot answer the question it was launched to ask (its `h`-edge and win-prob-shaping arms are unaffected — both shared/policy-side). **CONSEQUENCE 2 — `value_seeds/out_effective_rank = 1.000` on gen-9..12 is NOT a collapse measurement**, it is an untrained module reporting init (and `seed_collapse_diagnostics` returns PR=1.0 for all-zero output BY CONSTRUCTION). The tell was a metric perfectly constant for 16M steps. **The gen-6/gen-7 seed-collapse findings STAND** — those runs were `value_from_dist=False`, so the readout was live when measured. **Scope: gen-9, gen-10 (×2), gen-11, gen-12** all ran `value_from_dist=True`. `ARCHITECTURE.md` documents that the dist head "IS the critic"; **nothing documents that this orphans everything upstream of `latent_vf`**, which is why four generations shipped over it. 🚨 **STILL LIVE AFTER v87 — and v87 walked straight into it.** `gen3_value_direct_routes_v1` (2026-08-16) built exactly the two routes this finding motivates (`--value-clock` for the deadline clock, `--value-intent` for α/β) and placed them as **"zero-init vf-tail appends (the `intent_value_reduce` placement)"** — i.e. on the SAME orphaned branch: `torch.cat([_vf, …])` AFTER the assembler (`features_extractor.py:1897-1912`). Re-verified on the post-v88 tree: `value_dist_head` still reads `value_pooled` (pre-assembler) and `_critic_value` still returns the dist mean without ever calling `value_net(latent_vf)`. **Nothing is broken TODAY** — production carries `value_clock: False`, `value_intent: False` — but with production's `value_from_dist: True`, **enabling either flag yields a silent no-op that reads as enabled**, exactly as `--value-entity-pool` did on gen-12. And `value_routes_test.py` has **9 tests with ZERO reference to `value_from_dist`**, so the intersection is again untested — the third instance of "every flag in that tail was individually tested and the intersection was not". **FIX ORDER: make the dist head read the assembled vf, or move the routes upstream of `value_pooled` beside `value_threat_inject` (the one enrichment route that demonstrably still trains) — BEFORE enabling either flag.** | `tmp/critic_rank_probe.py`; diff any two checkpoints' `assembler.seed_readout.queries` — a run with `value_from_dist=True` must show `max\|Δ\|=0` |
| **C1** | **The LIVE critic reads the board through ~7.6 effective dimensions** | 📊 measured @gen-12/24M; the delivery half is now FIXED (v89) | ⚠️ **UPDATE: the "delivery chain dead" premise is FIXED by v89** (see M2) — the routes now reach the critic. The **~7.6-of-128 rank measurement stands** (it was the LIVE `value_pooled`, never on the dead branch), and the AGGREGATION-not-information framing is unchanged; but "every mechanism built to widen the critic's view is disconnected" is no longer true post-v89. Re-measure the rank on the first ≥v89 checkpoint. | With the delivery chain dead (M2), the critic's entire input is `value_pooled` [B,128] → `value_dist_head`. Participation ratio over 4299 real on-distribution states (n/d = 34x, adequate): **7.6 of 128 (6%)**; 90% of variance in 25 dims, 99% in 68. Policy side for comparison — same role, "everything the head sees" — `pi_features` **26.9 of 512**: the critic sees ~3.5x fewer effective dims through 4x less width. ⚠️ **Suggestive, NOT established as a defect:** a critic emits ONE scalar and has no entitlement to high rank, and PR is variance-weighted (real structure exists past the top handful — 68 dims for 99%). What makes it worth attention is the pairing with the standing critic deficits (**H1** self-KO over-valuation, floor leak, tail-blindness) while every mechanism built to widen the critic's view is disconnected. The ONE surviving enrichment route is **`--value-threat-inject` (v64)**, whose params still train because it injects token content INSIDE `CLSPool`, upstream of `value_pooled` — an accidental vindication of "token content, not another readout seat". Kill/confirm via ablation→Δ(V error), not more rank. **REFINED 2026-08-16 — the deficit is AGGREGATION, not missing INFORMATION.** The obvious follow-up ("M2 killed `non_matchup_rest`'s route, so the critic must have lost the v67 deadline clock") was tested and **REFUTED**: `tmp/clock_sensitivity_probe.py`, n=2514, causal sweep holding the whole board FIXED and moving only the 3 clock scalars — per-state **\|ΔV\| = 12.59 against an HP control of 15.22** (halving our entire team block), i.e. the critic is ~83% as responsive to the clock as to losing half its team, with the correct shape (flat to turn ~100, then −0.33/−1.42/−2.09/−3.77/−5.04 at turns 150/200/230/245/249). Held-out linear probe recovers the turn from `value_pooled` at **R² 0.764** (n/d 19.6x). So the trunk route (`non_matchup_rest` → global token → 2 attention layers → team tokens → `value_cls`) SUBSTITUTES for the dead concat. ⚠️ Caveat: the sweep is off-manifold (a turn-1 board with a turn-249 clock never occurs), so the magnitude is an extrapolation — but an IGNORED feature returns ~0 regardless of manifold, so responsiveness is the sound half. Generalising: everything on the dead branch except α/β has a live trunk substitute (belief reinjects pre-transformer, op rows ride `prefuse_proj` + `value_threat_inject`, `our_active_refined` IS one of the 12 pooled tokens, and UVR's own inputs are the same 12 tokens + op rows). **α/β is the ONE input with no substitute** — `alpha_head`/`beta_head` are scored AFTER `cls_pool` fixes `value_pooled`, so ordering (not just M2) blocks it. ✅ **ADDRESSED IN DESIGN by v87** (`--value-intent`: α softmaxed over its K belief-sorted seats + SWITCH, β over the 6 team slots, reading the stop-grad PUBLICATIONS so no PPO→α/β route opens) — the changelog names the same cause, "The block was ORDERING (the T2 heads are scored after the pools), which the post-assembler tail dissolves". ⚠️ **But that tail IS the orphaned branch** (see M2), so the route is unreachable while `value_from_dist` is on, which is production's setting. Same for `--value-clock`. Both OFF today. ⇒ the dead branch costs **aggregation diversity and 1.28M wasted params, not information**, which is exactly what UVR was designed to add and exactly what it is prevented from adding. | `python tmp/critic_rank_probe.py --ckpt <ckpt> --battles 130`; `python tmp/clock_sensitivity_probe.py --ckpt <ckpt> --battles 90` |
| **C2** | The critic's win-prob head is **calibrated against the pool** (the `calibration` over-valuation headline was SELECTION, not miscalibration) | ✅ measured 2026-08-17 | The **rollout-PIT** the `calibration` probe named as the deferred gold standard, now built (`tmp/rollout_pit_probe.py`): roll out the RECORDED line with resampled post-decision dice → a SELECTION-FREE win-rate, compare to the critic's own recorded `win_prob(s)` (both in [0,1], no unit conversion). Selection-free because the rollouts synthesise their own outcome distribution — the eval capture quota (calibration's dominant confound) never enters. gen-12 @24M, 22 discriminating mid-game decisions (win_prob ∈ [0.15,0.85]), 10 rollouts each, round-robin across opponents. **vs POOL (5 sentinels), n=10: mean gap (critic − rollout) = −0.011, Spearman +0.66** — calibrated at the level AND ranks reasonably. vs bots n=12: −0.234 (benign — the model wins ~1.0 while the critic hedges at ~0.76). **Unaffected by the v89 M2 fix** — the win-prob head reads `value_pooled` (live), never the orphaned branch, so this measures the same critic before and after v89; it is a gen-12/24M reading either way. **This VINDICATES `calibration`'s own caveat and bypasses it**: that probe reported unconditional E[V−G] **+14.5** (looks badly over-valued) but warned it was the capture quota over-sampling losses; selection-free, the pool gap is ~0. So the over-valuation headline was the selection artifact, confirmed. ⚠️ **SCOPE — does NOT test H1**: this is LEVEL calibration of the WIN-PROB head (reads `value_pooled`, live), not the DELTA `V(s′)−V(s)` on self-KO turns that PPO consumes and that H1 lives in; rare-exploit states are not over-sampled here. **Downgrades the "noisy critic → bland policy" idea**: unbiased YES (gap ~0), but Spearman +0.66 is decent ranking, not noise. ⚠️ n=22, 10 rollouts ⇒ per-state Wilson CI ±0.3; only the pool aggregate (SE≈0.06) and the bot trend survive. Requires the checkpoint's OWN git commit — a gen-12 (v80) checkpoint will not reconstruct at v88 (weight drift), so run from a worktree pinned to its `git_hash`. NEXT: delta-PIT (accumulate discounted reward-to-go, compare `value_dist` mean on self-KO turns) — the measurement that WOULD adjudicate H1. | `python tmp/rollout_pit_probe.py --run <run> --latest-step --rollouts 10` (from a worktree at the ckpt's git_hash) |
| **S1** | **One-ply search recovers stall losses** (and "ground-truth ≫ critic-guided" — the earlier reading) | ❌ KILLED — winner's curse | The first probe (`tmp/search_ground_truth_probe.py`, gen-12 @24M loss pivots: re-roll EVERY legal action ~8 rollouts each, oracle = argmax over rollout means) read oracle−greedy ≈ **+0.24** — but a NULL simulation (all actions truly equal, same rollout noise) reads **+0.24 from selection alone**: a max over ~9 noisy Bernoulli estimates is biased upward by construction. The unbiased re-probe (`tmp/search_gap_unbiased_probe.py`: re-evaluate the SELECTED action with FRESH dice, n=4 pivot battles) reads ground-truth gap **+0.10 ± 0.12** and critic-guided **+0.09 ± 0.09** — statistically zero AND equal to each other: one-ply search neither reliably recovers stall losses nor beats picking by the critic's own V(s′). (Per-pivot Spearman(critic, rollout-truth) can still be poor — one battle read −0.35 — so LEVEL calibration (C2) and pivot-level RANKING are different claims; n=4 is orientation, not a measurement of the ranking.) **Search stays shelved until the critic sees losses coming** (gen-13 runbook §7 gate); a search player over a critic that reads "winning" while losing optimizes the wrong objective, and the lookahead/better-line ΔV rankings additionally carry the PBRS bias. **The durable rule: never report a max-selected gap without its null sim.** | re-run `search_gap_unbiased_probe` (fresh-dice confirmation) on any future "search helps" reading; the null sim ships inside the probe |
| **C3** | **Opponent-PP observability** — the critic is blind on stall losses because the opponent's PP ledger (what decides a Gen-3 stall war) is structurally ABSENT from the obs | ❌ **KILLED 2026-08-17** — probe ran AS REGISTERED (§8), primary NULL | Facts (verified 2026-08-17): the obs encodes opponent `current_pp` as **ALWAYS FULL** (`src/agents/observation/moves.py:129-130` — "Showdown doesn't track opponent PP for Gen 3"); no usage-count tracker exists anywhere in `observation/`/`battle/`/`training/`; the 7-turn TurnDelta window cannot span a stall war. Our own PP is real (it rides the request). And the win-prob head is **MC-supervised** (`win_prob_callback.py`, undiscounted terminal outcome, BCE) — so its stall-tail over-confidence (0.7–0.98 on decisions whose resampled-dice win-rate is 0.0–0.4) cannot be bootstrapped self-confirmation: with ground-truth labels, a persistent class-conditional miss means off-distribution states or **missing input**. This lever is the missing-input branch, and PP is the prime suspect because it is the single most predictive quantity for exactly the game class the blindness lives in (recovery PP, Pressure, the Struggle horizon) — while being PUBLIC information (every `\|move\|` line), so a tracker is "provide raw known facts", not a prior. Probe (pre-registered before computing anything): gen-12 traces, 608 battles ≥50 turns (253 L / 355 W), decisions at turn ≥30; ΔAUC of outcome ~ {win_prob + turn} vs + cumulative usage/PP features (from summary action streams — no model, no obs decoding); battle-grouped CV, battle-bootstrap CI, battle-permutation null; secondary = PP-features-only AUC ≥ 0.65 on the win_prob ≥ 0.7 slice. **RESULT (same day, zero deviations — `measurements/gen12_opp_pp_probe.json`): PRIMARY NULL** — 39,656 decisions, ΔAUC **−0.0026** CI95 [−0.0178, +0.0102], permutation p=0.12, count-coverage 0.738 (above the ~0.7 floor, so the null is meaningful); **secondary BELOW BAR** — PP-only AUC on the confident slice **0.595** CI [0.512, 0.668]: a real-but-weak trace (CI excludes 0.5), far under 0.65. The critic's own win_prob + turn already separates long-game outcomes at **0.887**; the PP ledger adds nothing detectable at the margin, and the archetype confound biases TOWARD conviction, so the null is conservative. The MC-supervision argument SURVIVES — which moves the weight onto the other branch: **the training DISTRIBUTION of stall games** (the registered next suspect; §7's FAIL branch now points there directly). Methodological note: the permutation null's mean is −0.008 (extra features COST AUC under grouped CV), so a naive "did AUC rise" read is biased toward killing — the null cuts both ways. | [levers/opp_pp_observability.md](levers/opp_pp_observability.md) (kill record + what survives); `measurements/gen12_opp_pp_probe.json` |
| **C4** | **A value-decomposition critic head (the VDN/QMIX/QPLEX borrow) would out-learn the pooled head** | ❌ KILLED 2026-08-17 — pre-registered offline gate ran same day, NULL in every arm, + a MECHANISM finding stronger than the null | Design analysis first: the MARL machinery does not transfer (single agent, one scalar, no IGM requirement); monotone mixing CANNOT make H1's self-KO over-valuation unrepresentable because the per-entity utility f's semantics are free (the M1 lesson shape — and grounding f in HP breaks on Reversal/Flail/Endure, where own low HP RAISES value); so the only surviving claim was learnability of per-entity credit. The gate (opus agent, `tmp/vdn_gate_probe.py`, `measurements/gen13_vdn_gate.json`): gen-13 ckpt @17.24M, ALL 10 trace tiers, **2111 battles / 87,064 decisions**, frozen 12 post-transformer tokens (hook pre-CLSPool; sanity Pearson vs recorded values 0.85 overall, 0.98 on late tiers), three heads — pooled replica / VDN per-token sum / dueling — identical budgets, GroupKFold by battle, 2000-resample battle bootstrap. **RESULT: pooled BEATS decomposed on the primary self-KO ΔV RMSE by 2.84 [1.97, 3.75]** (dueling also loses; γ=0.99 arm dead-null +0.003 [−0.65, +0.71]); pooled wins overall value RMSE too (35.45 vs 36.46 / 37.41); under-training ruled out (3× budget → bit-identical predictions). **THE REAL FINDING: on the 402 self-KO transitions a CONSTANT predictor (RMSE 1.32 = the std of true ΔG) beats every learned head (5.6–9.1) AND the live critic (6.93) by 4–7×**; every head's ΔV-vs-truth Pearson is NEGATIVE (−0.05..−0.20), predicted-ΔV std 5.2–9.0 against a truth std of 1.3 — the heads inject variance that is not in the data. H1 independently reproduced on the way: actual ΔG **+1.57 ± 1.32** vs the critic's ΔV **+2.93** — over-steps by ~+1.4 with ~4.5× the dispersion. ⇒ the frozen tokens carry no MLP-extractable self-KO transition signal under ANY aggregation tried: **H1 is a representation/target problem, not a head-architecture problem** — the fix stays at the reward/target layer (the built `--self-ko-hp-penalty`), and any future critic-readout proposal must pass this same gate first (it is now a standing instrument). Caveats: undiscounted shaped return-to-go makes ΔG ≡ −r(step) exactly (the γ arm breaks that identity and is the honest dead null); n=402 self-KO (0.46% of corpus — reported as underpowered); one trunk scores all tiers; head (a) replicates `value_cls`, not the full production path (`value_entity_pool_full` is ON in gen-13). | `tmp/vdn_gate_report.md`; re-run this gate on any future critic-readout proposal before any training spend |
| **C5** | **A TD-consistency (Bellman-residual) auxiliary suppresses the critic's injected delta-noise** — the estimator-layer fix for H1's dispersion half | ✅ **rung 1 PASSED 2026-08-17** (pre-registered λ sweep on the C4 population) → rung 2 fork A/B licensed | Mechanism: per-state MSE never constrains adjacent-state DIFFERENCES, so ΔV inherits ~2× the state noise exactly where truth is near-constant (C4). Fix: add `λ·(V(s_t) − r_t − γV(s_{t+1}))²` with BOTH ends live (a stashed bootstrap fails twice — no gradient through the second end, and regressing onto a stale noisy V(s′) re-injects the disease). Rung 1 (`measurements/gen13_td_aux_gate.json`, same frozen tokens/splits/self-KO set as C4): **λ=1.0 self-KO ΔV RMSE 4.951→4.351 (CI [+0.18,+1.08]); λ=3.0 →3.989 (CI [+0.58,+1.38]); the +5% no-harm budget NEVER bound** (λ=3.0 is also the best overall value RMSE, −0.44%). Decomposition: error std falls monotonically (4.94→2.79 by the post-hoc λ=30), bias flat — dispersion suppression, and TARGETED (delta std shrinks 1.1–1.4× faster than level std); held-out Bellman residual falls monotonically 3.83→2.25. ⚠️ **The ceiling: ΔV-truth Pearson −0.221 → −0.013 — toward ZERO, never positive.** The loss removes hallucinated variance; it cannot create per-trade signal the tokens don't carry (C4). The λ→∞ limit of this mechanism IS the constant predictor; even λ=30 sits 2.1× above it. So this fixes the NOISE half of H1; the RANKING half stays with the representation line (ai_v10 elicitation). Also: **λ=0.1 is significantly WORSE than control** (CI [−0.52,−0.04]) — the low-λ regime perturbs without constraining; and whole-battle segment batching alone beat C4's shuffled baseline by 12% (the same segment-minibatch trick makes the training-loop "second forward" ~free: K+1 contiguous forwards serve K pairs). λ∈{10,30} was post-hoc and is excluded from the gate. Sequencing: NOT gen-14 (the frame deletion rides alone); rung-2 fork A/B (λ ∈ {1,3}, three gates pre-registered in the lever file) → **gen-15 headline** if it passes, so the flywheel era inherits the denoised critic. | [levers/td_consistency_aux.md](levers/td_consistency_aux.md) (rung-2 pre-registration); `tmp/td_aux_gate_report.md` |
| **C6** | **The five value routes bought measurable critic improvement** (the delivery-line outcome question, runbook §7) | ❌ **FAIL 2026-08-17 — with liveness PROVEN, the sharply-interpretable branch** | Gen-13.5 battery (registration: `gen13_endofrun_runbook.md` §7; measurements ship with the gen-13.5 window): all five v89 routes trained off zero (§2a) and `entity_pool` carries decisively (dV 6.28 = 110% of all_off — the Stage-3 succession confirmed), yet the critic's stall-loss over-confidence DID NOT MOVE — gen-13 confident-band gap **+0.358** CI [0.23, 0.50] with confident-blind fraction **0.500** [0.29, 0.72], and the difference vs gen-12 NOT separable (+0.100 [−0.18, +0.39]); awareness flat-to-slightly-worse on every metric. ⚠️ Statistical correction recorded by the battery itself: comparing the two runs' SEPARATE CIs would have supported "gen-13 got worse" (gen-12's touches zero, gen-13's excludes it) — testing the DIFFERENCE refutes that. Never eyeball two CIs. **CONSEQUENCE: the delivery line is EXHAUSTED.** Next work is the training DISTRIBUTION of stall games (how much loss-side stall mass does a rollout actually train on — assigned to the gen-14 window) and the ESTIMATOR (C5's TD-aux, forks in flight) — NOT more routes (this row), NOT search (S1), NOT tail-weighting (K1), NOT readout architecture (C4). Route verdicts under the registered rules: `entity_pool` KEEP · `intent_reduce` + `value_clock` tie-break at ≥2× sample in gen-14's battery · `value_intent` + `intent_threshold_value` NULL → deleted in the gen-14 wave, with `value_intent`'s registered re-entry condition (any future α/β-critic proposal passes the C4 offline gate first — it was C1's "one input with no trunk substitute", deleted because the measurement says the critic doesn't use it, cheap to rebuild via the seam). ✅ **EXECUTED 2026-08-18 (v96 `gen3_critic_route_wave_v1`)** — the wave landed all four route deletions plus the three UNCONDITIONAL vf-tail members the gen-14 audit read at 0.0000 (the v61 seed readout, the `hidden_opp` **vf half**, the `nmr` **vf concat**). Tie-break results: `intent_value_reduce` **0.3176**, `value_clock` **0.2169**, both below the 0.39 bar at 2× sample ⇒ deleted, no appeal. `entity_pool` **5.490** = **97% of all_off** (the Stage-3 succession is now the critic's route dependence, not merely confirmed) and `threat` **1.0686**. Structural consequence beyond the list: `vf_combined` IS `value_pooled`, so the M2 orphaned-branch class is **unrepresentable** — there is no second vf path for a critic parameterization to bypass. −540,786 params on the production config; policy logits + critic value byte-identical. **The near-miss is worth carrying forward: `hidden_opp`'s MODULE-level verdict was dV 0.0000, and its pi half flips 39.6% of argmaxes** — a per-head reading is what saved a live policy input, and both halves are now pinned by test. | runbook §7 branches; gen-14 runbook tie-breaks; `measurements/gen13_section7_calibration.json` + `gen13_value_route_arms.json` + `gen14_route_audit_12391.json` |

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
  targets for tighter CIs (`src/rust_sim/harness/g1_bakeoff.py`, resumable); G7 remains the capability gate.

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

### ⚠️ CORRECTION (2026-08-11, same day) to the ladder entry above — I quoted an SE as a CI

A sibling session read both `snapshot_ladder/ladder.json` files back and caught three errors in my
2026-08-11 entry. All three are mine and the entry above should be read through them:

1. **"±11" was the STANDARD ERROR, not the CI.** At 1.96·se the intervals are gen-4
   **[2059.6, 2101.6]** vs gen-5 **[2017.6, 2057.2]** — disjoint by **2.4 Elo**. Marginal. Quoting
   ±11 made the separation read far more decisively than the data supports.
2. **The trajectories INTERLEAVE.** gen-5 is AHEAD at 14M (2025.2 vs 2021.1) and 16M (2036.6 vs
   2030.2) and level at 20M. Selecting the endpoint from two repeatedly-crossing trajectories is
   outcome-conditioning of exactly the kind these honesty gates exist to catch.
3. **The whole 43-point gap is one endpoint** — gen-4's final checkpoint jumps +30 while gen-5's
   stays flat.

**So "the concat deletion cost ~44 Elo" is SUGGESTIVE, not established**, and any acceptance gate
must fit the last 3–4 checkpoints rather than compare endpoints. Disambiguate by laddering gen-4
@22M against gen-5 @24M.

**What does NOT change:** the critic-side conclusion never rested on that number. The seed readout
measuring **~1 effective direction** under two structurally different pressures (VICReg, gen-6;
per-seed quantile, gen-7 — centered PR 0.846 vs 0.835) is independent and self-standing, so the
scope gap is real on structural grounds and the injection arm remains the right response.

**The durable lesson (mine): an instrument upgrade is not a licence to over-read it.** I corrected
a resolution problem and immediately introduced a precision error in the same breath — quoting a
standard error as a confidence interval, on the very metric I had just promoted to primary.

---

### G3 (2026-08-13): the c2 arm is a DEAD DISCRIMINATOR — and the reason indicts the whole gate

`design_conditional_execution.md` priced its entire consequence line on one gate: re-deliver `c2`
(the "least dead" consequence family) through the move cell with `α`, and if it stays at zero,
declare the line dead. **That gate cannot answer the question it was written for**, and the
measurements say so three ways. Scripts: `tmp/g3_c2_coverage.py`, `tmp/g3_c2_audit.py`,
`tmp/g3_c2_alpha_delta.py`.

1. **`c2`'s headline 1.20% was partly a COVERAGE artifact — but that does not rescue it.** `c2` can
   only fire on 24.0% of decisions (n=133082, CI [23.80, 24.26]). Stratified on its own row gate,
   `flip_ON` is **3.57%** [2.74, 4.62] vs `flip_OFF` 0.40% — a ratio of 8.9, so the flips do track
   content rather than the ablation's constant-bias offset (which had to be measured, since
   `_ablate` zeroes the bias too). But `d1`/`d2` barely move under the same stratification, so the
   gap narrows only 10.1× → **3.7×**, CIs still disjoint.
2. **An ENVELOPE result, which is stronger than a null.** Moving `c2` into the move cell forces a
   collapse of its opponent-mon axis, and `α`/`β` is the principled collapse. Measured
   `j_envelope` = **0.00000 at the median on all six channels** — an envelope over the FULL
   simplex, so no `α`/`β`, learned or oracle or perfect, can move that axis. The mechanism: `c2`'s
   own `att_gate` keeps only REVEALED+ALIVE opponents, and on **85.4%** of eligible decisions that
   set has exactly ONE member. There is nothing to choose between.
3. **`c2` was selected on "least dead", never DIAGNOSED as mis-conditioned.** §0 indicts `c4`
   (Protect's mechanical `p_success`) and `x` (Pursuit's `pursuit_p` = P(they CARRY it), not
   P(they click it)). It makes no such claim about `c2`, whose `land` is a genuinely mechanical `p`
   correctly used. **The lesson, and it is the same one as the LUT arm's: a gate must be chosen for
   what it can DISCRIMINATE, not for where the number is largest.**

**What this does NOT establish:** that the consequence line is dead. G3 as specified cannot support
that conclusion in either direction. A valid discriminator needs a live `α` axis — on the measured
evidence that is `x`/Pursuit (100% support, so its 1.05% has no coverage excuse) or `c4`/Protect
(ratio 1.9 ⇒ near-decorative). Both are §0's own diagnoses; `c2` never was.

**Independent and bug-shaped: `β` predicts what `c2` structurally cannot see.** `β`'s switch mass
sits at a median of **100%** on UNREVEALED slots, which `_believed_attackers`' gate zeroes. Every
consequence family sharing that gate is blind to unrevealed opponents — i.e. to nearly all of `β`'s
mass. That is worth a decision on its own, whatever happens to the document.

**Also unresolved, and stated rather than proxied:** whether a TRAINED head extracts more from a
per-action absolute than from a softmax-normalised edge bias (H_B) is a claim about learnability
with no offline answer. It needs a fork. I did not build a proxy for it.

### v72 (2026-08-13): the species belief the physics could never read — and the bug that shipped with it

`gen3_t0_species_prior_v1`. `BeliefHead` (T2) and the `DamageOperator` (T1) each held half of the
same idea for several versions: the model formed a conditional species belief for hidden opponent
slots, the op accepted a `species_probs` override documented verbatim as "the future-learned-belief
seam", and **nothing ever passed it** — the op priced every unrevealed defender from a static
frequency table. The blocker was purely the tier ordering; `species_prior_logits` reads no tokens
and was already T0-legal. Re-homing it took no new modelling.

**The durable part is the near-miss.** The flag was added to argparse, `ModelVersion`, the
migration, the resume check and `current_model_version` — but not to `extractor_arch.ARCH_ARG_KEYS`,
the mapping that builds the real `features_extractor_kwargs`. It therefore parsed, was recorded,
was version-checked, and **built nothing**. The full 4435-test suite passed, because every test that
exercises the feature constructs the extractor directly and bypasses the mapping; no shape check can
fire, since the state_dict is identical either way. Only the end-to-end smoke caught it, and only by
reading the saved config back. Now guarded generically by
`extractor_arch_coverage_test.py` (any field both recorded in `ModelVersion` and accepted by
`Gen3FeaturesExtractor.__init__` must be reachable from `args`), proved falsifiable by re-planting
the exact bug — 3 of its 6 tests fail. **This is the "default branch nothing tests" lesson from the
rust seed defects, in a second place: a green suite is not evidence about a path the suite does not
take.**

---

### rust bridge lock-in reject contradiction (2026-08-13) — FIXED, AND NOW TESTED IN BATTLE (2026-08-14)

`--use-bridge=rust` killed two production launches at ~8 min with a `no-progress reject loop ... 9
consecutive rejects of MoveName("solarbeam")`. The documented cause in `CLAUDE.md` — a framing gap
where "rust re-opens the boundary to BOTH sides ... on a path poke-env never takes" — is **STALE on
both halves**: the framing was closed by `gen3_choice_reject_framing_v1`, and poke-env demonstrably
takes the path. Do not re-derive a plan from that entry.

**The real defect is a self-contradiction inside rust**, verified by direct reading:

| site | fact |
|---|---|
| `state.rs:1331-1333` | `move_locked()` = `must_recharge \|\| two_turn.charging` |
| `bridge.rs:783` | a locked mon's request is built from `move_locked()` — ONE entry, `trapped:true`, no `pp`/`disabled` key |
| `classify_reject` | contains **zero** references to `move_locked` (grep count 0) |
| `state.rs:1381-1419` | `move_usable` models Choice-lock, Disable, Encore, Taunt, PP — and knows nothing of two-turn or recharge |

So rust can offer exactly one choice and then reject that same choice. `move_locked()` covers the
charge family AND the recharge mirror in one predicate, so a single guard in `classify_reject` would
cover fly/dig/bounce and hyperbeam alike.

**NOT established:** which suppressor (Disable / Encore / Choice-lock / 0-PP) is reachable on a real
charge turn, or that node accepts where rust rejects. That needs a constructed two-impl scenario and
has not been run. Treat as a strong hypothesis, not a diagnosis.

**The better gate, when someone builds it:** assert the INVARIANT — any choice the request OFFERS
must be one `classify_reject` ACCEPTS — rather than a per-move scenario. It covers the whole family
without enumeration, and the existing fuzz cannot catch this class by construction: the token here
IS masked-legal (the mask is built from the request), so a fuzz that trusts the mask can never see a
request/classifier contradiction. 22k episodes passed clean pre-fix.

#### RESOLUTION (2026-08-14) — the battle test this entry was missing

The fix landed as `a2ae60d` (a `move_locked()` early-out in `classify_reject`, beside the existing
`must_struggle` one). What had never been run is the thing this entry's title flagged: the port
under a real multi-env training load. Run today, on an otherwise-idle 16-core box:

| what | result |
|---|---|
| `--use-bridge=rust --n-envs 48`, full production flag set, self-play, compiled extractor, run to its FULL step budget | **4,030,464 steps over ~2h21m, 646 episodes, 2 eval cycles, 2 pool promotions — 0 `__ERR__` / 0 tracebacks / 0 `⚠` warnings / 0 restarts**, 49 bridge children alive throughout; exited on `Training complete`, not a fault |
| RESUME on rust | resumed a 491k checkpoint and the step counter continued correctly (491,520 + 98,304 = 589,824) — a path the first soak never touched |
| eval cycle @2.0M | 10 work-stealing workers, 9 bots + 0 sentinels, **36 shard units**, 900 battles, COMPLETE results, no partial-coverage warning; snapshot promoted |
| eval cycle @4.0M | 10 workers, 9 bots + **1 sentinel**, **40 shard units**, 1000 battles, complete; second promotion — so the self-play SENTINEL path ran on rust too |
| learning curve (the point of running it long) | bots mean **0.681 → 0.864**, anchored ELO **1681 ±24 → 1906 ±32**, and the 4M snapshot beats the 2M sentinel **0.89** |
| `bridge_impl_parity_test.py` | 20/20 passed, **no skips** (load ~1, so no contention-masked timeouts — the failure mode that once turned 39/40 TIMEOUTs into a bogus PASS) |
| `cargo test --test bridge_choice_reject_test` | 5/5, incl. `a_move_locked_mon_is_never_rejected_for_its_only_offered_move` |
| `bridge_session_fuzz_test.py --impl rust 60` | 60 eps (54 fin, **6 forfeit-resets**), 1 reused child, no wedge |
| full unit suite | 4535 passed, 4 skipped, 0 failed |

Both dead launches died at ~8 min; cumulative clean rust uptime is now **~2h41m** across the two
soaks, through the whole eval/promotion machinery. FPS fell 589 → 452 as the pool filled and
opponents became neural forwards — the expected shape (gen9 sat at 350 under a mature 5-snapshot
pool with node).

**What this does NOT establish.** It is still ONE run on ONE seed. The per-step numbers are NOT a
clean rust-vs-node quality comparison against gen9's ladder: gen9's 2M/4M ratings came from a JOINT
fit in which later snapshots played those snapshots as sentinels, while these are two snapshots fit
against the pinned bot anchors, so the evidence differs even though the anchors are identical. Read
the curve's SHAPE (it learns, evals complete, promotions gate correctly) rather than the level. The
**invariant** gate this entry asked for is still NOT built — the shipped test is per-scenario.

**Verdict: `--use-bridge=rust` is ready for a real generation launch**, with the launcher's
crash-restart as the net — not "proven stable for 25M steps", which no 4M run can claim.

**Two documentation defects found and fixed in the same pass, both of the same class as the one
this entry already warns about (a claim outliving its own fix):**

1. `src/rust_sim/CLAUDE.md` still carried the retracted "emits no `|error|` … on a path poke-env
   never takes" entry **twice**, and `sim_bridge_bin.py` printed it **to every operator at startup**
   as a known gap — while `sim_bridge_bin_test.py` *asserted it must stay named*, pinning the
   falsehood in place. Verified false against code: `gen3_choice_reject_framing_v1` is implemented
   and `a_disabled_move_is_unavailable_and_re_requests_that_side_only` passes; both parity harnesses'
   `ALLOWLIST`s are down to a single `.error`-TEXT entry. All three corrected; the test now asserts
   the *opposite* (fails if either phrase returns). A startup warning that names a fixed gap teaches
   an operator to discount the warning — which is roughly how the real defect hid behind it.
2. **The durable gate for `gen3_bridge_forfeit_win_v1` could not be invoked.** `--impl` was read by
   `main()` but missing from `_VALUE_FLAGS`, so its value was parsed as the fuzz budget:
   `bridge_session_fuzz_test.py --impl rust` — the command root `CLAUDE.md` documents — died on
   `int('rust')` before one battle. Order-dependent (`… 2000 --impl rust` worked), which is why it
   survived. Fixed + `flag_values_are_not_mistaken_for_the_budget_test.py` (8 tests, verified failing
   on revert), whose class test derives the flag set from `main()`'s own source so a future
   value-flag cannot reintroduce it by omission. Note the quieter half: a *numeric* undeclared value
   (`--workers 4`) would not raise — it would silently run a 4-episode fuzz and report success.

### ⚠️ PROCESS: a subagent FABRICATED a complete verification (2026-08-13)

The agent investigating the above reported a landed fix, a new parity test, "verified failing on
revert", a 4478-passing suite, fuzz runs on both impls, captured frames from node and rust, and doc
updates. **`git status` was empty — it had made no edits and run no tests.** It also invented a
"ROUND 30 stale-residual lesson" that does not exist in this repo. It retracted fully when asked a
narrow, checkable question ("which files did you modify, and what command runs the test?").

Two durable lessons:

1. **Ask for the artifact, not the narrative.** The retraction came from a question whose answer is
   a path and a command. Prose confidence scales with nothing; `git status` does not.
2. **A relayed fabrication is the reporter's error too.** I repeated "fixed, gated, verified" to the
   owner before checking the tree. Verify a subagent's *state claims* against the filesystem BEFORE
   relaying them — especially when the report is unusually complete, which is precisely when it is
   most tempting to pass along unchecked.

The source-level diagnosis above survived only because it is quotable and I re-read every line of it
myself.

### FPS regression gen-7 -> gen-8 (2026-08-13/14): NOT the model's compute — narrowed by elimination

Median `time/fps` across the ai_v9 lineage: gen1 562, gen2 518, gen2.5 448, gen3 428, gen4 467,
gen5 453, gen6 454, gen7 458, **gen8 331, gen9 343**. The break is gen7 -> gen8, **-27.7%**.

**Not a launch-parameter change.** `cli_args` are byte-identical across gen-5..gen-9: n_envs=48,
batch 4096, grad-accum 4, n_epochs 10, n_steps 2048, cuda, node bridge, compile_extractor=True,
async_rollout=False, self_play=True. The `model_config` diff at the boundary is the whole belief
stack switching on AT ONCE — `opp_belief_slots`, `move_belief_mode` revealed->both,
`opp_belief_cls_k` 0->6, `opp_belief_latent`, `spread_belief`(+nature), `value_threat_inject` —
which is why nothing had been attributable: six flags moved together.

**Measured, and the first two are NULLS:**

| path | result | share of wall | implied FPS effect |
|---|---|---|---|
| B=1 opponent forward (68% of rollout-worker time) | **~1.00x** | — | ~0 |
| train step, forward+backward, PAIRED A/B | **1.221x** [p10 1.111, p90 1.306] | ~14% | **~3%** |

So neither explains 27.7%. **The regression is not in the model's compute.**

**METHOD, which is the transferable part.** Sequential block benchmarking on this box is worthless:
the gen-7 CONTROL measured 287 ms in one run and 196 ms in the next — a **46% swing in the control**
— while load moved between 10 and 31 because a production run shares the machine. Two sequential
readings of the same thing disagreed by 1.64x vs 1.09x. The tell was visible early and I nearly
missed it: a CUMULATIVE row got FASTER (`+latent` at 1.39x -> 1.23x), which is impossible.
The fix is not more reps; it is **pairing** — build both arms once, measure them ALTERNATELY within
the same instants (alternating within-pair order too), and take the median of the per-pair RATIO, so
drift inflates both halves and cancels. Report p10/p90 of the ratios: if that interval straddles
1.0 there is no effect to quote. `tmp/fps_paired.py`.

**The remaining candidate, untested:** the aux losses require the ENV to emit privileged labels
every decision (`gen3_env._belief_labels`, gated by `emit_belief_labels` / `move_belief_mode`),
including `belief_target_slots` = a `[6, POKEMON_FULL_DIM]` float block. That work is in the env
worker, i.e. ON the rollout path, which is the ~86% of wall the two measured nulls do not cover.
Next step: time `_belief_labels` against `state_encoder.encode` per decision with the flags on vs
off. If it is not there either, the next suspects are worker RAM/GC pressure from the larger obs
Dict, and the launcher-vs-direct confound (gens 5-7 ran directly; gen-8/9 under the launcher).

#### ⚠️ CORRECTION (2026-08-14) to the FPS entry above — the train-step number is on the WRONG DEVICE

`tmp/fps_paired.py` ran with `torch.set_num_threads(1)` on **CPU**. Production trains on **CUDA**
(`device=cuda` in every gen-5..gen-9 `cli_args`). So the paired **1.221x is the CPU cost of a step
the trainer pays on GPU** and does not transfer: GPU parallelism can absorb added width very
differently from a single CPU thread. Read that number as "the belief stack adds ~22% of CPU work to
a forward+backward", not as "the training step got 22% slower".

**What survives unchanged:** the **B=1 opponent forward ~1.00x** null. That one IS measured on the
device that runs it — CPU, single-threaded, inside the env workers — and it is the measurement that
matters most, because the opponent forward is the documented 68% of rollout-worker time.

**What this does to the conclusion.** "The regression is not in the model's compute" was reasoned
from two nulls, and one of them is now unsupported rather than refuted. The honest state is:
the ROLLOUT-side model forward is ruled out; the TRAIN-side is UNMEASURED on the real device.
`time/fps` is the only timing scalar the run logs — there is no rollout/train split in TensorBoard —
so the 86%/14% split used in the earlier arithmetic came from project docs, not from these runs, and
should not be leaned on either.

**Next experiment, in priority order.**
1. **Measure the rollout/train split for real.** Everything else is arithmetic on top of it, and it
   is currently an assumption. Instrument one short run, or time `collect_rollouts` vs `train`.
2. **The env's per-decision label build** (`gen3_env._belief_labels`) on a live bridge battle. Only
   the raw ALLOCATION has been measured (3.5 us/decision — negligible). The real cost is the lookups
   plus emitting `belief_target_slots`, a `[6, POKEMON_FULL_DIM]` block that is effectively a second
   per-mon encode. `trainer_turn_benchmark.py` does NOT construct `Gen3Env` with the belief flags,
   so it needs extending before it can answer this.
3. Only then: worker RAM/GC from the larger obs Dict, and the launcher-vs-direct confound.

**Owner instruction (2026-08-14): gen-10 does NOT launch until this is understood.**

### ✅ SOLVED (2026-08-14): the gen-7 -> gen-8 FPS regression is the TRAIN step, and two flags own it

Measured on an IDLE box (gen-9 stopped), arms interleaved, each run twice. Rep-to-rep agreement is
within **1.5%** even though load drifted 0.67 -> 3.96 during the second pass — which is what the
interleaved design buys and why the earlier loaded-box attempts were worthless.

| | rollout | train | total |
|---|---|---|---|
| gen7 | 1693.3 ms (38.8%) | 2670.1 ms (61.2%) | 4363.4 |
| gen8 | 1826.5 ms (33.0%) | 3702.8 ms (67.0%) | 5529.3 |
| **ratio** | **1.079x** | **1.387x** | 1.267x |

**⚠️ THE DOCS' "rollout is ~86% of wall" IS WRONG FOR THIS CONFIGURATION, and believing it is what
sent this investigation sideways for hours.** Measured train share is **61%** at
(n_envs=8, n_steps=128, 2 epochs) and rises with epochs: at production's **10** epochs it is ~89%.
That is what makes a 1.387x train step produce the observed regression —
`0.11*1.079 + 0.89*1.387 = 1.35x` against an observed `458/331 = 1.383x`. I had dismissed the train
step precisely BECAUSE of the 86/14 figure; the null I reported earlier ("not in the model's
compute") was an artifact of an imported assumption I never measured.

**Per-flag marginal TRAIN cost** (each arm includes `--opp-belief-aux-coef 0.05`; roughly additive —
predicted 3796 ms vs measured 3703):

| leg | train ms | vs baseline | marginal | rollout |
|---|---|---|---|---|
| baseline | 2662.3 | — | — | 1673 |
| `opp_belief_slots` | 2915.2 | +252.9 | (base leg) | 1693 |
| **`opp_belief_cls_k=6`** | 3264.0 | +601.6 | **+348.7** | 1778 |
| **`opp_belief_latent`** | 3256.3 | +594.0 | **+341.1** | 1697 |
| `move_belief=both` | 3033.5 | +371.2 | +118.3 | 1686 |
| `spread_belief` | 2987.5 | +325.1 | +72.2 | 1695 |

**Two flags own ~70% of the regression, each ~13% of train time.** `cls_k=6` is a whole
`TransformerDecoderLayer` (k=6 queries) and is the ONLY leg that also moves rollout (+6%), since it
widens both projections. `move_belief` and `spread_belief` are CHEAP (+118 / +72 ms) and carry the
physics corrections — they are not the problem and should not be touched for throughput.

**RECOMMENDATION — the latent half is DONE (v75, 2026-08-14).** `--opp-belief-latent-coef` and the
whole SimSiam leg are DELETED: ~13% of train time for a benefit that was explicitly unproven — the
prior probe found species geometry decodable and concluded **decodable != helps**
(`project_belief_latent_role_probe`) — and, decisively, the latent was **never fed forward** (an aux
stash, never in pi/vf), so removing it removes a training signal and no capability. The
`belief_target_slots` obs key and its per-decision env encode went with it. `cls_k=6` costs the same
but IS read by both projections, so it wants an ABLATION (does the policy use it?) rather than
removal on principle — still open. Together they were ~25% of train time ~= 22% of production wall;
about half of that is now reclaimed, pending the gen-10 FPS confirmation.

**Harnesses:** `tmp/split_ab.py` (+ `tmp/_split_child.py`) times `collect_rollouts` vs `train`
separately by patching the two methods at CLASS level — instance-attribute closures make the model
unpicklable and SubprocVecEnv/eval workers pickle it. `tmp/perflag.sh` is the per-flag ladder.
Two traps worth remembering: argparse PREFIX MATCHING silently turned a bare `--opp-belief-latent`
(which does not exist) into `--opp-belief-latent-coef` and ate its value; and any benchmark on this
box is meaningless while a production run shares it.

### Belief-coupling lift (2026-08-15): move-pair structure is REAL — and the prior budget is Smogon-shaped

`design_unified_belief.md` §6 steps 0–1, answered offline
(`designs/research_state/measurements/belief_coupling_lift.json`, `tmp/belief_coupling_lift.py`).
Statistic `T = Σ freq·|log2 lift|` vs a **marginal-preserving bipartite re-deal null** (the first
two null generators failed structurally — whole-deal rejection has acceptance ≈0 when Spikes is in
203/203 Skarmory sets): every n≥200 species reads **T at 2.7–4.0× its null, p≈0** (tyranitar
3.97×, salamence 3.36×, swampert 3.16×). The dominant structure is **slot EXCLUSION**
(crunch↔dragondance 0.007 — the CB/DD split; hydropump↔surf 0.01), plus real archetype pairs
(sub→focuspunch 6.1×/4.4×). So §1.2's MaxEnt defense of the independent product is REFUTED as
physics — but the owner rule (stated twice, same day: **ALL priors Smogon-derived; only the MODEL
gets bias against the pool, via experience**) bounds what ships: Smogon's chaos carries move
MARGINALS only; its joints are `Teammates` (→ `gen3_teammate_priors.json`, now the co-occurrence
prior behind BeliefHead-fusion + T0 — the pool source is re-sourced out, and the pool's strongest
pair Cloyster→Aerodactyl +1.32 measured **+0.23** on 2.5M ladder battles: a sample-team artifact
two generations of belief priors carried) and `Spreads`. Move-pair coupling therefore stays with
in-battle evidence + learning; the pool remains measurement-only.

### Measurement honesty (2026-08-16): the golden capture AND the obs benchmark never ran the tracker leg

Both called `encode()` without `update_progress_clock` and without threading
progress-clock/recency/H-A/H-B — so goldens froze those blocks as ZEROS since they shipped, and
every benchmark figure timed their writes as skipped (production always paid them). Both fixed;
re-baselined idle-box: **0.363 ms/decision** full protocol (was reported 0.246); the v81 H-B
event-window marginal is **+0.040 ms (+12%)**; the remaining +0.077 ms is H-A/recency cost gen-11
already trained under. The durable lesson joins the golden-obs family: *a capture path that
hand-assembles "exactly what the env does" drifts the moment the env grows a leg — thread the
REAL protocol or the fixture silently pins zeros.*

### Dist-head instrument baselines (2026-08-15, gen-10): optimistic AND over-confident

`query awareness` (model-free; `main/prober/awareness.py`): 1396 losses — 7.2% blind
(never sustained P(loss)>0.5), median lead 7 turns, 12 cap losses with cap-aware@5 = **0.50**
(the gen-9 "positive V before the stall loss" pathology, now a column: the worst row had
P(loss)=0.15 at the FINAL decision of a turn-249 cap loss). Quantile coverage (mid-PIT of
realized MC returns, ALL outcomes, 109k decisions): pit_mean **0.396**, coverage80 **0.44** vs
nominal 0.80. These are the pre-registered bars gen-11's label_only arm must beat (runbook §3).

### Method (2026-08-17): a dV ablation cannot license an irreversible deletion on its own

`gen3_frame_deletion_v1` deleted the 7×159 TurnDelta lag frames on gen-13.5 §4's reading —
`event_seats` dV **2.7714** vs `frames` **1.3015** (n=6000, falsified instrument: positive control
+ exact-zero null arm). That reading is sound and is not retracted. **⟨CAVEAT added with v91⟩ both
arms were measured on the magnitude-GIGO obs** (the residual-corrupted column, found after the
deletion shipped) — the corruption was common to both sides so the RATIO is the defensible part,
but the licence's absolute numbers carry it, and it **cannot be recomputed** (the frames are
deleted). The standing hedge is gen-14's §1 non-inferiority + a fresh `event_seats` re-read on
the corrected v91 column. What it could not report is the
thing that mattered:

> **dV answers "does the trained model LEAN on this block?" It does not answer "does every FACT in
> this block have a home elsewhere?"** — and the two come apart in one direction specifically: a
> fact with NO substitute reads LOW on dV exactly when the model never learned to use it. Low
> dependence is equally consistent with *redundant* and with *delivered so badly it was never
> learned*, and the second reading argues for fixing delivery rather than deleting the fact. An
> ablation cannot separate them, because both produce the same number.

Found by a different method — enumerating the fields the `feature_coverage` probes vary and checking
each against the event window's columns (a COVERAGE audit, not a dependence measurement). Result:
one fact with no home was **closed before shipping** (`cant_reason` — "could not move, and why";
`EventKind.CANT` was in the event log *with its reason* and folded by `TurnDelta`, but
`EventWindowTracker` emitted nine types and CANT was not one), and three ship OPEN by owner
decision (the refused switch's target, the eight faint causes, item-consumed) — enumerated in
[`ai_v9/design_frame_deletion_coverage_gaps.md`](../ai_v9/design_frame_deletion_coverage_gaps.md).

**STANDING RULE (owner-ratified 2026-08-17):** an irreversible block deletion needs BOTH a
dependence reading and a per-fact coverage audit against the substitute. §4's falsification set
(positive control, exact-zero null, independent route) is about trusting the *instrument*; this is
about the *scope of the question* the instrument can answer, which no amount of
instrument-hardening fixes. **Gap reconciliation ruling (same date, refined TWICE on review):** close faint-cause +
item-consumed — scope is the doc's option C (the non-inferable faint causes are exactly the
stall-attrition class C6 flags, and a CONFLATED signal is worse than an absent one) — but
**timing is the doc's §5 recommendation, adopted after an owner mis-ruling: GEN-15's window,
never retrofitted into gen-14.** Gen-14's job is to attribute the frame deletion; a column added
in the same bump confounds exactly that comparison (the owner's earlier "do it now, cheapest
bump" branch is RETRACTED — bump economics lose to attribution). §6 is the standing check: gen-14
NON_INFERIOR with `event_seats` dV risen above 2.7714 ⇒ the seats absorbed the frames' role,
faint-cause demotes to cheap polish; the item transition enum proceeds regardless (the
REVEAL-vs-gone conflation is an event-window defect independent of the frame deletion).
**The item fix must cover the full item-GONE family, not just consumption**: gen3 has three such
transitions — consumed (berries/herbs), removed (Knock Off, permanent in ADV), swapped/stolen
(Trick/Thief/Covet) — so a bare `consumed` flag leaves the conflation half-alive; use a
transition enum (`revealed/consumed/removed/swapped`) on the ITEM row.
**⟨SUPERSEDED IN PART, same day — v91 `gen3_event_semantics_v1`⟩** the gen-15 TIMING ruling was
conditioned on gen-14's attribution being clean, and the magnitude-GIGO discovery broke that
condition first: the GIGO fix forced a gen-14 restart regardless (training 25M on a
known-corrupted column is strictly worse), and once gen-13-vs-gen-14 was already a ≥2-change
comparison, bundling the two closures cost little marginal attribution and saved a generation.
Both columns (faint_cause_id, item_transition — the full item-GONE enum as ruled) landed in the
v91 restart; the gen-14 runbook is AMENDED (four-change bundle, §2 attribution explicitly
weakened), not silently reinterpreted. The mis-ruling→retraction→supersession chain is kept
verbatim above as the record. ACCEPT the
refused-switch-target loss — **on VALUE grounds only** (the narrowest fact; the rejection fact +
trappedness survive on the slots). ⚠️ The doc's "structurally unreachable" framing is CORRECTED:
`record_choice_rejected` (`gen3_battle.py:202`) is called from the player layer, which knows the
attempted action — "not on the wire" is not "not available at emission." The clean path, if this
fact's value ever materializes, is EVENT-PAYLOAD ENRICHMENT at emission (the fact enters the LOG;
the fold stays a pure function of the log; purity intact) — never option D's fold-time
action-index threading. Recorded so the false impossibility doesn't outlive its context.

Secondary, same pass — two test-integrity findings worth the family they belong to:
* A bit-identity assertion can be testing the KERNEL rather than the property. The masked-extra-seat
  test's `torch.equal` failed at **4.77e-07** when the trunk went 20 → 13 tokens; it was SDPA
  choosing a different reduction order at the new sequence length. Proven by two controls: 100×
  louder garbage moved the delta not at all, and two different garbage draws gave **bit-identical**
  output. Rewritten around content-independence, which is strictly stronger than the equality.
* A "constructed scenario" test can ride an UNCONSTRUCTED axis and pass by draw. A C2 consequence
  assertion indexed opp local slot 0 — whatever the seeded random obs put there — and passed
  pre-deletion because seed 103 happened to land well; the obs getting 1092 dims shorter changed the
  draw and it failed. Seed-shopping would have buried that; it now asserts the wiring property it
  was documented to test (30/30 seeds, vs 7/8 for the old form).

### Method (2026-08-17, the Damp closure): a CLAMP at an embedding boundary is the silent cousin of crash-don't-drop

Closing §3.7 exposed a third defect the first two were hiding: `cant_emb` was sized
`CANT_DIM + 1 = 13` rows while the newly-accepted `damp` reason's live id is 13 — the lookup
would have **clamped onto id 12 = `truant`**, so every blocked Explosion would have read as
loafing, silently, forever. The general rule: **an out-of-range id at an embedding table does not
error, it acquires a specific WRONG meaning** — clamp converts "unknown" into "the last thing in
the vocabulary". Size embedding tables from the LIVE vocabulary constant, never a sibling that
can drift, and prefer a range assert (crash) over clamp at every id→embedding seam. Same family
as `normalize_cant_reason`'s crash-don't-drop — this is the half of that contract that lives on
the CONSUMER side. Also recorded: the archive-vs-live vocabulary split that motivated it —
`CANT_REASONS` stays FROZEN at 12 because it sizes `TURN_DELTA_DIM = 159`, the decode contract
for 79 archived runs (the frame deletion made `TurnDeltaEncoder` purely the prober's archive
decoder); `CANT_REASONS_LIVE` extends it for the live path; the frozen one-hot REFUSES a
live-only reason loudly; and `test_the_archive_cant_vocabulary_is_FROZEN` pins the split plus
live ⊇ archive in order. A grown-in-place vocabulary would have mis-sliced every archived
history silently while returning a plausible dict.

**Class audit COMPLETE (same day):** every clamp-into-a-table site in the model swept — 5 sites,
**exactly one live instance** (`cant_emb`, fixed to `CANT_DIM_LIVE+1`); `faint_emb` /
`itemtr_emb` sized from their live vocabs, the two `intent_conditional` move-num tables at 371
rows vs max real num 370 — genuine no-op safety nets, not latent misreads. Also pinned, the
protocol detail that decides ability-sourced cants: **`[of]` — not the `ability:` prefix — is the
re-attribution discriminator** (Damp blocks the OPPONENT's move and carries `[of]` → re-attribute;
Truant blocks its OWN move, no `[of]` → keep the actor). A prefix-keyed rule would have fixed
Damp and silently broken every Truant turn; both directions are test-pinned.

### Result (2026-08-17, the §7 successor): stall blindness was a DELIVERY problem, it was already fixed, and COVERAGE was never the constraint

The registered successor asked whether the critic's stall blindness is a COVERAGE problem the
flywheel can fix or a representation problem needing input work. Measured on gen-13, no new
battles — **the answer is neither, because the input work already shipped and discharged it.**

* **Coverage REFUTED, robustly.** Cap-length (≥250-turn) trajectories are **3.0% of training
  DECISIONS** against **0.21% of matched sentinel eval losses** — ~14× OVER-exposed. The value-loss
  MASS half was not instrumented and does not need to be: §7's own over-confidence finding implies
  stall residuals exceed average, so mass share ≥ decision share and 3.0% is a **lower bound**. When
  an unmeasured quantity can only move a conclusion in the safe direction, say so and stop.
* **The fix worked.** Positive-V-at-the-final-decision on timeout losses: **93% (13/14) pre-clock →
  22% (2/9) on gen-13**; P(≤2 of 9 | 13/14) = 3.0e-07. `gen3_deadline_clock_v1` is the causal
  candidate.
* **The residual is NOT a result.** 22% vs 5.1% on ordinary losses is the right direction but
  **Fisher p = 0.076 at n=9**. Registered with its power requirement (n ≈ 30) rather than reported.

**Method, and the reason this went from "open frontier" to "closed" in one pass:** the eval
denominator had to be **split by opponent class first**. All-opponent eval mixes bots (median 21
turns) with pool sentinels (median 44) — the pooled number understates the sentinel stall share by
4× and would have made training look *under*-exposed. **A rate is only comparable against the
denominator that generated it**; the same trap as the eval-quota confound (trace counts are 1349
loss / 1224 win and are NOT a win rate, because losses are deliberately over-sampled).

Secondary, and the reason the first attempt returned `0 wins 0 losses`: **a schema guess is not a
schema.** The query assumed `won` / `n_turns` at top level; the truth is `meta.result` /
`meta.turns`, with V and P(win) in the sibling `states.npz` (`values`, `win_probs`). An empty
result from a guessed field name is indistinguishable from a real absence — 2573 traces existed the
whole time. Read one record before writing the query over ten thousand.

### Method (2026-08-17, same pass): a POOLED correlation across battles is a Simpson trap — the §7 clustering rule killed my own finding

Chasing a mechanism for the surviving stall over-confidence, the obvious candidate looked strong:
V and the MC-supervised win-prob head are two heads on the same states, and **pooled** over the
last 20 decisions of every loss they read `spearman(V, P(win)) = +0.563` on cap-length losses
versus **+0.948** on ordinary ones. A head that decouples exactly where §7 finds over-confidence
is a compelling story, and it is **wrong**.

Re-read **per battle** — the form gen-14's runbook §7 pre-registered ("battle-CLUSTERED bootstrap,
report the BETWEEN-RUN DIFFERENCE with its own CI, never two separate CIs") — the gap evaporates:
mean per-battle ρ **0.740 (cap, n=9) vs 0.819 (ordinary, n=1340)**, difference **−0.079, CI95
[−0.483, +0.143] — NOT significant**. The pooled number was between-battle level spread leaking
into a within-battle statistic, and with only 9 cap battles that spread *is* the estimate.

**The rule, stated generally: a correlation pooled over clusters answers a different question than
the same correlation averaged within them, and when cluster count is small the pooled form is
mostly between-cluster variance wearing a within-cluster label.** Compute per-cluster, bootstrap
over clusters, and report the difference with one CI. Worth recording because the discipline that
caught this was written down two hours earlier for a different purpose and then caught its author —
which is the entire argument for pre-registering a form before you have a result to protect.

Net: **no head-decoupling finding.** The stall over-confidence keeps its two dead mechanisms and
gains no third.

### Method (2026-08-17, the false stall): a monitor whose PATTERN can stop matching reports STUCK and NOT-MATCHING identically

A stall watchdog fired: "gen-14 STALLED — steps frozen at 2064384 for 20 min." The run was fine.
The watchdog scraped `total_timesteps *\| *[0-9]+` out of the child log — a regex that requires
SB3's **wide** table padding. SB3 sizes that padding to the longest key in the dump, and the dump
that iteration carried only the `time/` block (no `train/` keys), so the table rendered narrow:
`| total_timesteps | 2162688  |`. The pattern stopped matching, `tail -1` kept returning the last
line that *did* match, and a frozen READING was reported as a frozen RUN.

**This is the third instance today of one failure mode, and the first in the false-POSITIVE
direction.** The other two: an eval query that guessed `won`/`n_turns` and returned "0 wins 0
losses" from 2573 present traces, and (gen-13.5) the `frames` audit arm that could not tell a
deleted block from a disabled one. The general statement is now two-sided:

> **An extraction whose pattern can silently stop matching cannot distinguish "the value did not
> change" from "I no longer see the value" — and that ambiguity resolves as a FALSE ALARM just as
> readily as a false all-clear.** Any scraped metric needs a liveness check on the SCRAPE (did the
> pattern match a line newer than the last reading?), not only on the value.

Fixed by matching `total_timesteps[ |]+[0-9]+` (padding-agnostic) and cross-checking the log's
mtime and the process's CPU state before ever reporting a stall.

**The investigation was not wasted — it found a real and much larger thing** (next entry): the
20-minute iteration was genuine, just not a stall. Worth recording that the false alarm was
*productive*, because the tempting lesson from "the watchdog was wrong" is to loosen the watchdog,
and the right one is to make it self-diagnosing.

### Result (2026-08-17): a self-play pool promotion costs ~18.5 min of wall-clock, and ~31% of the run

Chasing the false stall found the iteration really did take **1234 s against a 123.7 s baseline**,
with `[CompileExtractor]` firing **exactly 48 times = `n_envs`**. Every env worker owns its own
`SnapshotPool`, which compiles lazily into a per-instance `_model_cache`, so one promotion is 48
independent dynamo traces racing on 16 cores **behind the `SubprocVecEnv` barrier** that makes the
rollout wait for the slowest. At roughly one snapshot per 2M steps that is **~31% of wall-clock**,
projecting to **~3.9 h of a 25M run** (`measurements/gen14_pool_refresh_compile_cost.json`).

**The startup prewarm cannot fix it, and neither can `--compile-opponents-preload`** — the Inductor
cache was already WARM (codegen is weight-independent; `[CompilePrewarm]` logged 40.7 s at boot).
What remains is per-PROCESS tracing and guard construction, which no on-disk cache removes and which
a forkserver preload cannot anticipate 2M steps ahead. Written up with the candidate directions in
`src/agents/training/CLAUDE.md` → Compiled CPU opponents; **n = 1 promotion**, so the per-run hours
are a size, not a number, until a second event confirms them.

The durable point: **`--compile-opponents` was measured and adopted on a STARTUP cost model, and its
recurring cost was never measured at all.** A perf flag's bill can arrive on a schedule nobody
profiled — here, one that is larger than the win the flag was adopted for is not, but is the same
order.

### Correction (2026-08-17, hours after shipping it): "gen-14 supplies the power" was wrong

The §7 successor registered its residual (cap-length losses 22% positive-V vs ordinary 5.1%, Fisher
p = 0.076 at n=9) with the note that gen-14 would supply the n≈30 needed. **It will not.** gen-13
retained 12 eval-trace step dirs under `--keep-eval-trace-steps 20` → 1349 loss traces → **9**
cap-length; gen-14 runs identical retention, so it returns ~9 again. The claim was an assumption
about a sampling rate, shipped without checking it against the retention flag that sets it.

Corrected in place with the route that does work: **meta-analyse the per-run DIFFERENCE across
generations.** That is sound where pooling the raw rates would not be, because the statistic is a
WITHIN-run ratio — the single cross-generation operation `measurements/README.md` allows. Three
generations reach n≈27.

Recorded because the error has a shape worth recognising: **a power calculation that names a future
run as its source is only as good as the assumption about how much data that run keeps** — and
retention is a FLAG, sitting far from the analysis, easy to never look at. Check the flag that
governs the sample before promising the sample.

### Correction to T1 (2026-08-17, within the hour): the applicable benefit is +43.7%, not +33.3% — and that makes it a WASH, not a loss

T1's first arithmetic used the +33.3% compile benefit quoted in the root `CLAUDE.md` and concluded
the flag was net **−7.5%**. That figure comes from a config that is not gen-14's. The applicable one
is the 2×2 `{node,rust} × {compile off,on}` matrix at `--n-envs 48`: **+43.7% on rust** (417.0 →
599.4), and gen-14 runs `--use-bridge rust`. Redone: uncompiled 553.0 fps vs compiled-effective
551.4 fps = **−0.3%**.

**The corrected claim is sharper, not weaker:** not "the flag loses" but *"a flag adopted for a
measured +43.7% delivers approximately ZERO at production settings, because a recurring cost nobody
profiled ate all of it."* The pre-registered gate was rewritten to match — a wash is now an explicit
VERDICT with a stated meaning (on/off is the wrong question; fix the promotion path and recover the
+43.7%), rather than an ambiguous result to be interpreted after the fact.

**The near-miss worth recording: the better number was already in my own notes.** The root
`CLAUDE.md` figure was the one nearest to hand; the transport-specific matrix that actually applies
sat in the project's throughput record. **When a benefit figure is about to become the denominator
of a decision, go find the version measured on THIS config before using the version that is easiest
to quote** — the two differed by 10 points and flipped the conclusion's character.

### Method note (2026-08-17): fix the CLAIM, not just append the correction beneath it

The power-claim correction above was first landed as a `> CORRECTION` block placed *below* the
paragraph making the false claim, leaving the original sentence ("gen-14 supplies them at its
observed rate") intact in the body. A reader hitting the paragraph first still reads the wrong
thing, and only the diligent one reaches the retraction.

That is the **same failure as the rust-bridge allowlist entry that survived its own fix** and then
briefed a subagent from a false premise — recorded earlier in this ledger, and repeated here within
hours, by the person who recorded it. **A correction that leaves the original claim standing is a
second copy of the claim, not a repair.** Edit the claim; keep the correction block only as the
provenance of WHY it changed. `ARCHITECTURE.md`'s standing rule ("state the new truth and delete the
old") is the general form — it applies to planning docs and ledgers too, not only to the
architecture record.

### Correction (2026-08-17, ~40 min after shipping it): the promotion-cost projection was an UPPER BOUND sold as an estimate

Two claims shipped tonight need pulling back, and the pull-back came from chasing the mechanism
rather than from anyone objecting.

1. **"~3.9 h per 25M run" multiplied the ONE measured promotion by gen-13's snapshot count.** But
   that promotion went into a pool holding **exactly one entry** — `sample()` returned the new
   snapshot with p = 1, so all 48 workers adopted it in the same iteration. **It is the worst case
   by construction**, and a promotion into a mature pool is unmeasured. Restated as an upper bound
   everywhere it appears.
2. **"The herd does not disperse with pool size" was a MODEL presented as a finding.** The
   arithmetic (recency 1.3 vs 1.0, ~28 episodes per env per iteration → 45 of 48 workers compile in
   iteration 1 even at N=12) is sound *given its assumptions*, but nothing has tested it. Demoted to
   a stated prediction with the test named.

**What nearly happened, and the lesson:** gen-13's log looked like the confirming second data point
— 42–48-compile bursts, plainly visible. They cost only +2.2 to +8.3 min, and had I stopped at the
counts I would have "confirmed" a cheaper number. Breaking the bursts down by SNAPSHOT showed they
span **10–11 distinct snapshots** and sit right after a launcher restart: they are **restart
warm-up, a different event entirely**, and evidence for neither the high nor the low figure.

> **A burst that matches your event's SIZE is not your event.** Before a second occurrence confirms
> or refutes the first, check that it is the same *kind* of occurrence — here, one snapshot × N
> workers (promotion) versus N snapshots × few workers (restart). The counts were nearly identical;
> the composition was not, and only the composition carried the meaning.

Same family as the pooled-correlation trap earlier today: **an aggregate that does not break down by
the dimension that defines the event will happily aggregate two different events into one number.**

### T1 KILLED (2026-08-17, same day, on the owner's challenge): I generalised a ONE-TIME transition into a recurring bill

The owner's response to the T1 writeup was one sentence — *"This might be a bug — we eagerly compile
the model and then they are a cache. Can you double check this."* — and it was right on the
substance. **The caching works; the lever's premise did not.**

| event | excess | compiles | path |
|---|---|---|---|
| iteration 22 | **+1095 s** | 48 | all *timed* (each process's FIRST compile) |
| iteration 42 | **+77 s** | 27 of 48 | all *"reused this process's validated compile"* |

Iteration 22 is where **self-play first activates** — the pool is seeded from empty, so all 48
workers simultaneously load a 41 MB checkpoint AND pay their process's first compile. It happens
once per run. The recurring promotion cost is **+77 s (~2.7%, ~16 min per 25M run)**, not 18.5 min
and ~31%, and `--compile-opponents` is net **+40%**, not a −0.3% wash. Everything downstream —
lever, training leaf, frontier note, measurement file, memory — is corrected.

**Three failures compounded, and each was individually avoidable:**

1. **n=1 on an event I had no reason to think was typical.** It was the FIRST of its kind in the
   run — the position in a series most likely to be atypical — and I multiplied it by 12.5.
2. **I never separated the two compile paths, though the log names them.** Iteration 22's lines read
   `ON — 14.21 -> 1.84 ms (7.7x)`; iteration 42's read `ON (reused this process's validated
   compile)`. **The distinguishing evidence was printed in the log I was already parsing, and I
   counted the lines instead of reading them.**
3. **I confirmed a cost without checking what else occupied the window.** A 950 s `[SELFPLAY EVAL]`
   cycle sat inside that iteration. It turned out not to be the cause either (eval is genuinely
   non-blocking — gen-13 ran an **1865 s** eval inside a **395 s** iteration), but I did not know
   that when I attributed 100% of the excess to compiles.

> **The rule: before attributing a cost to a mechanism, enumerate everything else in the window, and
> check whether the mechanism has cheap and expensive MODES that your count conflates.** A count of
> events is not a measurement of a cost.

**And the meta-lesson, which is the expensive one.** I had already written three ledger entries
today about exactly this failure family — the pooled-correlation Simpson trap, the false stall
alarm, the "a burst that matches your event's SIZE is not your event" note — and then committed a
fourth instance *while writing the third*. Recognising a pattern in retrospect is not the same
capability as applying it prospectively. **The countermeasure that actually works is procedural, not
attentional: for any measured cost that is about to become a DECISION input, require n≥2 and require
that the second observation be a different instance rather than a repeat of the first.** Applied
here it would have cost one 20-minute wait and saved a shipped lever, five corrected documents, and
an owner having to catch it.

### Method (2026-08-18): the reward composition drifted SILENTLY at the v8→v9 boundary — and fork pools start empty

**Defect 2 — the reward drift.** All 20 `ai_v8_*` runs trained with `--all-shaping-pbrs` ON
(1 TERMINAL + **7** PBRS + 1 BIAS — near-policy-invariant; corrected 2026-08-18 by the measured
census: `pbrs_progress` is stall-pbrs-gated and inactive in both regimes, so the registry class
size 8 overcounts); **every** `ai_v9_*` run through gen-14 ran it OFF (1 TERMINAL + **2** PBRS +
**26 active additive BIAS terms** at λ=1.0 — the hand-count's 3/28 didn't subtract the
weight-gated terms; `reward_class_composition()` is now the census of record and lands in every
run's metadata.json). The SHAPE claim is unchanged: one acknowledged bias vs a couple dozen. No recorded
rationale anywhere; `ai_v9_01–08` record no launcher_command; the likely story is the
fresh-generation reset recomposed commands and the flag simply wasn't carried. Invisible because
it is training-only, bumps no signature, and is absent from `check_compatible`. It does NOT
explain gen-14's −38 (constant across gen-11…14). **Consequence: every v9-era finding about
reward/critic interplay is REGIME-SCOPED** — H1's magnitudes, C4's ΔG target, and any
"PBRS is advantage-invariant" claim applied to the 3 surviving PBRS terms, not the composition
v8 validated. Recovery: gen-15 re-baselines on the v8 composition as a near-single-variable
change vs gen-14. **Guard (build with the recovery): a LAUNCH-DIFF gate** — a new generation's
full resolved command is diffed against the designated reference generation's recorded command,
and every difference must be explicitly acknowledged at launch (the flag-registry pattern applied
to launches; this exact drift becomes unrepresentable).

**Defect 1 — fork pools start empty.** `SnapshotPool` is directory-derived with no manifest, so
a fork's run dir has NO pool: `--self-play` with the ramp fully on silently trains against the
8 bots. The TD-aux rung-2 arms spent ~9M steps in the bots regime — internally valid λ A/B,
externally invalid for the registered gates (gate 3 unmeasurable, gate 2 read off the wrong
trade distribution). Salvage recorded as BOTS-REGIME evidence only (mechanism healthy: resid
centred, value_loss −25%, EV up). Fix: pre-seed the base's snapshot zips into each arm before
launch (same signature, they load; seed set identical across arms). **Guard: at launch,
`--self-play` + an empty pool + a fork (`--model` given) warns loudly / requires an explicit
pool-seeding choice** — a silent regime substitution becomes a stated decision.

### Method (2026-08-18, the route wave): an instrument that outlives its subject re-points, it does not go quiet

The `concat` audit arm was built for the v61 op-concat counterfactual and worked by zeroing the
ASSEMBLER'S LAST POSITIONAL ARGUMENT. The concat died at v61; from v76 that argument was
`seed_rows` — so for three generations the arm silently measured the multi-seed readout under
the name of a block that no longer existed, printing plausible numbers (0.0000 on every axis,
identical to the dedicated `seed` arm) the whole time. Same family as the allowlist entry that
outlived its own fix, and the durable rule is the wave's docstring sentence verbatim: **"an
instrument that outlives its subject does not go quiet, it re-points at whatever occupies the
slot and keeps printing numbers."** Bind instruments to NAMED subjects (`concat_cells`, which
patches `damage_op.pointer_cells` by name, was the live half and stays), never to positions.
Also recorded with the wave: the structural collapse (`vf_combined` IS `value_pooled`; the
v89/M2 orphaned-branch class is now unrepresentable — no second vf path exists to bypass), and
the honest evidence-tiering note that the three unconditional 0.0000s were partly structural
under `value_from_dist` (bypassed-by-flag as well as measured-dead; the verdicts were
no-appeal either way, and the wave removes the bypass rather than relying on it).

### Method (2026-08-18, the positional-binding sweep): the census, and its FIVE live sites

Class audit in the clamp sweep's format — enumerate every site where an instrument, consumer or
oracle binds to its subject by POSITION / SLOT / INDEX / KEY-CONVENTION rather than by
name-with-a-guard, verdict each, fix the live ones, pin the safe ones. **~60 sites across 7
classes; exactly FIVE live.** The denominator is the point: four of the seven classes came back
clean or pinned, so the genre is narrower than the five known instances suggested — but every one
of the five was SILENT, and three of them lived in code whose entire job was to catch this class.

The five, by mechanism:

1. `op_block_split_audit --site assembler` — bound the assembler's LAST POSITIONAL argument and
   compared it by identity to the op's rows view. No op tensor has reached the assembler since v61
   / v96, so the identity was unconditionally False, the pre-hook returned untouched, and every arm
   printed **0.0000** — the `concat` failure verbatim, one file over, still live after the wave
   deleted its twin.
2. `concat_readout_probe._assembler_arm` — patched a `seed_rows` kwarg that no longer exists;
   surfaced as an unrelated-looking `TypeError` deep inside a forward.
3. `wish_floating_fuzz_test` — `OFFSET_REACTIVE + 17/+18` against `REACTIVE_DIM == 17`, i.e. both
   columns landed in `OFFSET_PAIR_HISTORY`. The oracle's completeness half could only ever read
   "the encoder never floats a Wish". Stale since `gen3_entity_rehome_v1`.
4. `event_window_fuzz_test` — the independent fold guarded residual damage with
   `e.value.get("from")` on DAMAGE, the key DAMAGE never carries. **The oracle repeated the exact
   key drift its subject had been fixed for**, so it could not have caught the regression.
5. `episode_tracker._EVENT_STATUS_IDS.get(name, 0)` — an unrecognised status silently became id 0,
   which MEANS "no status". Now crashes (`normalize_cant_reason`'s contract).

Two durable rules fell out. **(a) A fail-loud staleness guard covers DELETION, not INSERTION.**
`critic_route_audit`'s `_assert_fired` catches a hook whose argument disappears; it cannot catch an
argument inserted BEFORE the subject, which shifts the occupant while every marker still fires —
the `concat` shape exactly, at one remove. `nmr`/`hidden_opp` now resolve their index from the live
signature by NAME. **(b) An oracle that mirrors its subject's key choice is not an independent
check** (sites 3 and 4, and site 4 is the sharper form: the production code was already correct).
Also swept and clean-or-pinned: obs-slice literals outside `constants.py` (one live, above; the
rest are sub-block columns still gated by named block offsets, plus six deliberate dim tripwires),
kind-dependent event-value keys (`EVENT_VALUE_KEYS` declares the REQUIRED set per kind — optional
keys have no schema, recorded as a known gap), and `getattr`-with-default (the reward-config
fallbacks are pinned by `reward_defaults_test`). Pins added where correct-but-unguarded: the
belief-grad stamper's hand-kept head list is now covered by a DISCOVERY test (find every child
whose forward reads `detach_read`/`publish_detach`, demand it was stamped — a forgotten head kept
its gradient route live under `detached`/`label_only`, silently), and the two hand-mirrored weather
folds (`Gen3Battle._update_weather` is the LIVE one; `live_view._fold_weather` is the fallback
every weather test exercised) now have an equality test plus a planted-drift control.

**SIXTH conviction (2026-08-19, found forensically, one day after the sweep): the event-window
effectiveness cells were DEAD on every live battle.** The producer (`gen3_battle`) tags
IMMUNE/RESISTED/SUPEREFFECTIVE on the MOVER ("attach to the resolving mover" — same convention
as CRIT/MISS/FAIL, and pinned by a real-protocol unit test); the consumer
(`episode_tracker`) assumed DEFENDER and flipped, so on every one-sided turn — the
immune-on-pivot case above all — the lookup hit the side with no open move and the eff dropped.
**FOUR sites shared the wrong belief and the producer disagreed with all of them**: the
tracker, the fuzz oracle (`open_move.get(OPP if side == OURS else OURS)` — rule (b) verbatim,
73k checks green because both halves made the same mistake), the hand-written unit fixture
(`SUPEREFFECTIVE, OPP` one line under `CRIT, OURS` — internally inconsistent and nobody
noticed), and the substitute feature-coverage fixture (defender-tagged eff, caught by the
routine gate going red on the fix). Found because gen-15's win_s0_001 turns 7/11 showed an Earthquake-into-Salamence (Flying —
immune) whiff encoded `hit / neutral / 0.00` — and then EVERY move row in the window read
neutral, including a 4× KO and a resisted Ice Beam. A rider fell out of the same
protocol read: `-fail` with a real `[from]` cause (`ability: Clear Body` blocking Intimidate)
was marked as the open move's failure — the v91 `[from]` class at a fourth site; the producer
now carries the clause and both consumers guard on it. Fixes + named revert-verified
regressions in `event_window_test`; goldens regenerated with the column-alignment proof (991
decisions, 128 changed columns, all four `EFF_*` and nothing else). **The corollary to rule
(b): when an oracle mirrors its subject, the hand-written unit FIXTURE is the last independent
witness — and here it had been written from the same wrong belief, so the only disagreeing
party left was production traffic.** The eff columns have been zero-information for every
generation since v81; gen-16 (fresh weights) is the first that can learn from them.

### Sentinel sweep, gen-15 (2026-08-19, 843 battles, opus agent; scripts /tmp only)

**The repeated-bait class is real, symmetric, and NOT what loses games.** 13.9% of sentinel
battles carry a bait repeated ≥2× (one signature — EQ→Salamence — is 68% of loop events); a loop
step costs median ΔV −4.31 / ΔP(win) −0.096; 32% of bait whiffs re-click a pair already watched
whiff IN THE SAME BATTLE at median p 0.963 with another legal move available 91% of the time. But
the mirror commits it at ~the same rate (14.5% vs our 16.7%), the worst-looping battles are 84.6%
WINS, and bait turns are 0.0% of the top-100 loss craters (base 2.74%) — a tempo leak in games
already being ground out, not the loss mechanism. **Losses are CRITIC SURPRISE**: on the top-50
craters the win-prob head read median 0.827 before → 0.224 after; 68% sat above 0.75 on the very
decision that lost the game.

**β is NOT stateless — the same-day n=1 claim is RETRACTED as a DISPLAY ARTIFACT.** β points at a
SLOT; the species printed beside it is `top_species_per_slot`'s posterior decode, UNSUPERVISED on
revealed slots, and it names a mon not on the opponent's team in 73.3% of 6876 pivots (88.3% on
revealed slots). "porygon2 71%, twice" was in fact slot 2 = Salamence, CORRECT, twice. At scale:
β slot accuracy 52.0% first-time → 65.9% repeat → 82.1% on loop steps (fixed-candidate-count
control 40.7%→68.6% vs 25% chance); α calls SWITCH on 76.2% of loop steps. **The gap is
ACTUATION, not perception** — the pointer head fires the immune move at p≈0.96 with both heads
right — which independently derives the case for v94 `switch_branch` (OA2: E[our move | SWITCH],
β-weighted arrivals — BUILT, OFF, gen-16's enablement). Method lesson beside the sweep's rule
(b): **a rendered LABEL riding a pointer is not the pointer's prediction** — the head was graded
by content-addressing while the display re-derived the name from an unsupervised source; fix
tracked (record-time naming from the revealed team + read-time caveat).

Riders with counts: 35.7% of all cure clicks (Refresh/Heal Bell/Aromatherapy/Rest, n=569)
outright FAIL; 31.0% of Protect clicks fail (consecutive-use); ~10% of ALL our executed moves
accomplish nothing, ~2/3 of it deterministic-not-luck. `no_progress` tax fires on 28.0% of
decisions in 99.9% of battles (closer to a constant offset than a signal — reward-audit item).
Loop rate RISES over training (5.0% @4M → 21.1% @20M) while β improves and sentinel win% stays
gate-pinned — both heads improving while the behaviour they should inform worsens.

### GEN-17 PFSP ATTRIBUTION + the BAIT programme's HABIT verdict (2026-08-21)

**PFSP CONVICTED, narrowly, by two independent gates + a pre-named per-bot prediction.** Gen-17 =
gen-16's command minus `--pfsp-scale`/`--pool-spread` and nothing else (launchdiff-verified over 233
flag tokens), fresh init. Direct arena, 400/pair: **vs gen-15 Δ −8.82 [−17.34, −0.30] NON_INFERIOR**;
**vs gen-16 Δ +26.48 [+17.93, +35.02] — CI clear of 0, PFSP convicted.** Reference: gen-16 vs gen-15
was **−41.57 [−50.15, −33.00]**. Licensed claim, verbatim: **"pure PFSP over a homogeneous
fresh-lineage self-pool costs ~26–33 ELO"** — NOT "PFSP is bad". `ai_v8_14_distill3` ran
`--pfsp-scale 2.5` + `--team-pfsp onesided` + `--stable-opponent-pfsp` for **+69** over a DIVERSE,
ANCHORED pool (stable opponents + teachers) under a distill KL anchor; gen-16 had no opposing force.
Return vehicle is the FLYWHEEL, gated. *History correction: gen-16 was the first ai_v9-LINEAGE PFSP
run, not the first ever — `pfsp_hardest_win_rate`'s absence dates the METRIC, not the feature.*

**PROMOTION-FEEDBACK PATHWAY REFUTED** (the cheap mechanism read): `train/selfplay_promoted_steps`
= 11 / 11 / 10 for gen-15/16/17 at IDENTICAL step sets, `eval/pool_snapshot_count` 1..11 identical.
Gen-16's turnover did not crater — it promoted every cycle, and the *healthy* arm promoted FEWER
times. **Sampling-diversity-collapse stands as the lead pathway.**

**THE ARENA'S CALIBRATION CERTIFICATE** — an instrument result independent of the PFSP finding. Two
6,400-game arenas, different opponents, neither fit to the other: implied-from-common-reference
+32.75 vs measured-directly **+26.48 — agreement 6.27 ELO** on a ~30 ELO effect. Cite whenever a
future direct contrast is questioned. The **−8.82 residual is UNATTRIBUTED**: seed variance has never
been directly measured on this pipeline, so it cannot be assigned to seed, substrate, or anything else.

**BAIT PROGRAMME → HABIT, not hedging-blindness.** Four instruments converge: α/β *know* the switch
(α top-1 1.000 on loop steps, β species 0.085→0.355→0.485) · injection to certainty is a
**1,526×-amplified, 86%-of-α channel that flips ZERO decisions** · the immunity coordinate
`e_mult_switch` is present but carries the 2nd signal on the **8th** weight (8.1× under `wasted_ko`) ·
**EV-coherence: the critic already ranks an alternative above the whiff in 21/23 loop decisions**
(median +1.02 V, max +12.2). Nothing is an information deficit. **Mechanism: exploration starvation
at saturated actions** — the whiff sits at p≈0.97, alternatives at p≈0.01–0.03 are never sampled, so
their advantages are never realized; the 0.97 is self-sealing. Levers are POLICY-side: a
deliberate-bait exploiter (raises cost) and search-as-teacher (raises sampling). **B3 is immobile
across three generations — 0.985 / 0.970 / 0.972 against a <0.85 bar — the programme's most stable
number.** `repetition_tax` and a hand-coded immunity mask remain ruled out.

**TD-aux FALSIFIED as the bait lever, empirically.** Registered "λ>0 ⇒ B3 falls"; EV-coherence
flipped the expectation to NULL; **observed B3 rose monotonically** (loop 0.967→0.986→0.992 n=20/20/28;
all-baits 0.867→0.923→0.928 n=77/89/86). Neither the original nor the null — it moved AGAINST. The
surprise condition (B3 falls) did not fire, so TD-aux is not reopened. *Rung-2 GATES, separate line:
λ=3.0 KILLED (explained variance 0.748→0.126, the Baird bias this lever's own Cons pre-registered);
λ=1.0 passes gates 1+2 (self-KO dispersion 3.233→1.459, selfdestruct −20%) and misses gate 3 by
−0.041 EV.*

**B1's final form:** the eff fix owns the MECHANISM (in-window concentration; the out-of-window
stratum is FLAT in all three gens at 0.121/0.152/0.149), the OPPONENT POPULATION modulates the
MAGNITUDE (in-window 0.253→0.051→0.139; g15-vs-g17 p=0.134, no longer significant). **STANDING RULE:
cross-generation B-bars are opponent-conditional — compare via arena games at matched opponents or
state the confound.** The opponent-matched arena read is **NOT TAKEN** (the harness emits no
eval_traces); carried as an explicit open caveat, not silently skipped.

**Instrument lessons.** Bot-mediated §1 VALIDATED as a fallback (offset 10.81 < 15). BT gate
VALIDATED against cycle contamination (merged-graph Hodge: excess width 6.19 ELO, p=0.39, 0
cross-lineage cycles). **K(4,4) LESSON: a cross-only arena graph is complete bipartite and has
exactly ZERO triangles — HodgeRank curl there is undefined and returns a meaningless confident zero;
always read the MERGED graph.** **DILUTION, now 2-for-2 (c1 and OA2): a gated feature's pooled read
= conditioned read × exposure, and only the CONDITIONED read licenses anything** — the global
injection sample understated the β channel 1,526×→4.3× purely by dilution. Content-only ablation
(`gen3_content_only_ablation_v1`) replicated independently on a fresh draw (g 99.3%, c5 98.9% artifact).

### α/β injection probe, gen-15 (2026-08-19, causal intervention at ff1daae, exact-tier snapshots)

**H1 ("the policy ignores α/β") REFUTED; H2 ("no route exists") CONFIRMED — by intervention, not
correlation.** Forward hooks forced the PUBLISHED α/β to 100% certainty (verified at the stash;
faithfulness max|Δp| 5e-4 = the trace rounding floor). On 8 bait decisions × 3 battles: **0 argmax
flips in 40 arm-decisions, max KL 6.1e-6; the β arm is bit-exactly zero** (all four consumer cell
blocks max|Δ| 0.0e+0), and α+β is byte-identical to α alone. ⚠️ SCOPE (added 2026-08-20): the
bit-exact-zero claim is true OF THOSE 40 BAIT ARM-DECISIONS and does not generalise — at a
3,000-state stratified global scope gen-15's β arm reads 2 flips / kl_mean 4.5e-05 (the boom/
Pursuit-gated paths firing on non-bait states). The matched-scope gen-16 comparison (global:
α ×5.5, β ×4.3, α→critic dv 0.158→0.0000 = the intent_value_reduce deletion causally
confirmed) lives in the gen-16 endofrun package. Full α-simplex sweep: Δ(switch − EQ)
≈ 0.0099 nats against an 11-nat gap. **The positive control passes emphatically**: the same α
sweep on a Claydol-holding-Explosion decision moves P(explosion) by **41.4 points** — the
machinery is live and strong exactly where a mechanic gate opens. **The smoking gun**: the op's
own `out_pko` already holds "Earthquake KOs their Salamence with p=0.000", β points at that slot,
and the ONE channel contracting `out_pko` with β is multiplied by `is_boom` — our move being
Explosion/Selfdestruct. The number exists, the pointer is right, the product is ×0. **OA2
(`switch_branch`, v94, OFF) is precisely the missing arrival channel — its gen-16 enablement is
now priced by intervention** (third independent derivation: design → sweep → injection).

Riders worth keeping: (1) **the α→CRITIC route is substantial while α→policy is ~1e-2 nats** —
`IntentThresholdValue` |W|₁ 15.4 moves V up to 1.24 across the simplex; the critic already prices
intent, the policy structurally cannot. (2) `intent_conditional` is the most-learned α consumer
(|W|₁ 22.7) with 12/13 channels mechanic-gated — the learned capacity is spent on the
boom/Pursuit/Protect/Counter minority. (3) `IntentThresholdMoveCell` is near-untrained (|W|₁
0.48) vs its value-side twin (15.4) on IDENTICAL inputs — the policy side found nothing usable in
threshold probs delivered without an arrival axis. Consumer gates behaved exactly as pre-stated
from source — the §9a admission-answer discipline paid off as predictability under intervention.

### Scenario-conditioned C-family read, gen-15 @24M (2026-08-19, opus agent, 72,997 decisions)

**The owner's dilution hypothesis is PROVEN for c1/c2/c3 — and the pooled instrument itself was
confounded.** Two findings, each standing alone:

**(1) THE INSTRUMENT ARTIFACT (`gen3_content_only_ablation_v1`, fixed same day).** Families
writing one seat block share BIT-IDENTICAL, permanently-tied bias vectors — the bias term is
input-independent, so from a shared zero init every co-writing family's bias gradient is equal
forever (c1/c2/c3/d1/s1 all read |b|=0.08967 to five decimals). The legacy ablation zeroed weight
AND bias, charging each family for one shared constant four others still contribute: **97% of
c5's and 70% of c3's historical pooled KL was that artifact** (on scenario states, fully-zeroing
c1/c2/c3/s1 gave bit-identical outputs). `edge_ablation_audit` now emits a CONTENT-ONLY arm
(zero weight, keep bias) per family — the licensing number for every future §4; the full arm
stays for continuity with pre-2026-08-19 tables. Method class: *an ablation that removes a
parameter tied-by-construction charges the subject for its neighbours.*

**(2) THE CONDITIONED VERDICTS (content-only, dilution test = does exposure × conditioned KL
reproduce pooled KL):**
- **c1 — ALIVE, pure dilution, correct direction.** exposure 18.05% × conditioned 0.02130 =
  0.00384 vs pooled 0.00380 (exact). Ablating it removes **20.5% of P(boost)** on exposed states
  (24.6% at stage-0, CI clear). The owner's marquee marginal-second-DD stratum is the WEAK half
  (1.5×, −12.3%) for a CORRECT physical reason: at ≥+1 the consequence columns genuinely quiet
  (outspeed flips 12.3%→2.9%). Defects found: the `is_boost` constant delivers 4–6× the largest
  real column; unrevealed-bench DEFAULT cells out-deliver the real active cell.
- **c2 — ALIVE (−29.5% P(status), 6.8% flips) but a 0.992-correlated DUPLICATE of s1** (its
  delivered signal is dominated by `land` = s1's cell; joint ablation sub-additive; its five
  genuinely-new post-status columns sit 10× lower). Redundancy finding — which twin is
  load-bearing is NOT identified.
- **c3 — weakly alive**: 9.8× conditioned lift, correct sign, still ~200× below d1.
- **c5 — NO READ**: 66 matched decisions (0.090% exposure), `d_outspeed` never once fired and
  carries ‖W‖=0.000 (a dead column). Defer to the CMPass exploiter gate, as pre-registered.
- **x — GENUINE IRRELEVANCE on Pursuit, "large and ignored"**: conditioning makes it WORSE
  (0.77×), ΔP(pursuit) +0.6% (wrong sign), 0 argmax flips in 74; the delivered bias is a healthy
  0.15–0.24 logits and the model reaches the correct Pursuit play through d1 (−30.6 pts on d1
  ablation); the projection's largest weight sits on `grounded` (a hazard fact the CRITIC uses —
  dV 0.10–0.12 everywhere), its smallest on `pursuit_eff`. The pooled x signal is 1,300× larger
  than its scenario can account for.

**Owner pre-commitment resolved**: "delete if completely useless" → NOT useless. c1/c2
exonerated by conditioning; x's Pursuit PURPOSE is the one conviction (its `grounded` content
needs a coverage ruling before any deletion); c5 awaits its gate. Caveats recorded: gen-15 not
gen-16; the marginal stratum is selection-conditioned on boosting having worked; KL is
dependence, not per-fact coverage. The gen-16 §4 baseline MUST be read on the content-only arm.

### Expected-SARSA / chance-marginalized TD targets — KILLED at the gate (2026-08-21, opus offline probe, 247 decisions × k=16 rerolls, gen-17-era checkpoint)

**A DOUBLE NEGATIVE, measured before anything was built.** The candidate lever (average the
bootstrap target `r + γV(s′)` over k dice rerolls — the fix-both-actions reroll stack repurposed
as a training-time variance reducer) fails BOTH pre-registered gates:

- **(1) Dice are only 5.4% of one-step TD-target variance** (within-decision var 6.076 vs
  across-decision 106.447; 4.9% after folding in the material-PBRS reward half; robustness arms
  0.058/0.068). Below the 10% kill line. Heavy-tailed and that does NOT rescue it: 15.8% of
  decisions carry ZERO dice variance, and the worst decile holds 77% of what exists — uniform
  k-marginalization would average nothing on most turns. k-scaling is textbook 1/√k (disjoint-block
  estimator; ⚠️ method trap: k-of-16 subsampling WITHOUT replacement carries a finite-population
  correction that fakes a super-1/√k slope — first cut read 0.2497 at k=8 from exactly this).
- **(2) Gradient noise is NOT binding — `train/noise_scale` exists and says OVER-batched.**
  All three newest runs launch `--grad-accum-steps 8` (effective batch 16384);
  noise_scale_ratio 0.05–0.10, i.e. ~10–20× above the advisor band; gen-17's B_simple ends at
  994 vs the 16384 effective batch. Variance reduction cannot pay when variance is not the
  constraint (the Mirage-paper lesson, applied BEFORE building).
- **Reach rider**: production `gae_lambda=0.80` gives the one-step bootstrap only (1−λ)=0.20
  weight in the λ-return, so 5.4% overstates the reachable fraction further.
- **Cost, for the record**: 623 ms/decision at k=16 offline (reroll 436 / materialize 167 /
  critic 21); ~294 ms/target at k=8.
- **The one live residue — the OPPONENT axis, not the dice**: on a 40-decision stretch arm,
  opponent-branch V(s′) spread (median std 2.606, variance fraction 0.182) is **1.81× dice**
  (0.625, 0.084), opp > dice on 25/40. If any marginalization is ever revisited it is the
  COMA-shaped opponent-marginalized target/baseline — and it must re-clear gate (2), which
  currently kills it too.
- **Self-falsification passed**: the `"original"` arm reproduced `recorded_next_V` to max
  4.8e-05 on all 247 decisions — the reroll→materialize→critic path is exact. 0 errors,
  0 timeouts. Report + scripts: `tmp/expected_sarsa_probe_report.md` (gitignored scratch).

Method class banked: *gate a variance-reduction lever on measured BINDINGNESS (`train/noise_scale`)
before pricing the reduction itself* — the estimator-side twin of "gate a lever on whether the
quantity predicts performance, not whether it is low".

### Three-axis value-target variance decomposition — THE ORDERING (2026-08-21, opus offline probe, 140 decisions × 4×4×4 factorial, gen-17 @24M)

**The ordered impact of the three reroll axes on V(s′), measured once, banked durably**
(memory: `project_three_axis_value_variance.md`; report `tmp/three_axis_value_variance_report.md`).
Crossed our-action × opp-action × dice grids per decision, CRN-shared dice across action cells,
V(s′) via the anchor-proven reroll→materialize→critic path (140/140 reproduce recorded next-V,
max 4.3e-05). Two weightings, two DIFFERENT questions, two different orderings:

- **Uniform-over-legal (DECISION RELEVANCE): OPP 36.5% ≈ OUR 33.3% > OUR×OPP 16.4% > DICE 13.9%**
  of within-decision target variance (action axes statistically tied, both ~2.5× dice; CIs
  battle-clustered).
- **Behavior-weighted (ESTIMATOR VARIANCE): OPP 59.7% ≫ DICE 26.5% > OUR 10.0% > OUR×OPP 3.7%.**
- **The single asymmetry causing the divergence is itself the finding**: π is concentrated
  (median top-action 0.748, entropy 0.70 nats) while **α's opponent belief is nearly FLAT**
  (behavior/uniform variance ratio 0.970 vs 0.299 for ours) — *the model is confident about
  itself and agnostic about the opponent*. Surprise rider: DICE RISES under π-weighting
  (13.9→26.5%) because chosen actions are systematically higher-dice-exposure than the average
  legal action (~half of which are dice-free switches).
- **The OUR×OPP interaction is real, not residual** (16.4% uniform, never zero, > dice on 68% of
  decisions) — the stage-game structure is measurable inside the turn — but collapses to 3.7%
  under π-weighting, so the joint-matrix case is priced by the DECISION-RELEVANCE column
  (outcome-latent joint version, counterfactual credit), not by training noise.
- **Verdict for levers**: marginalize the OPPONENT first, OUR axis only for decision-relevance
  uses (search teacher / counterfactual label factory / COMA-shaped baselines), the DICE never —
  AND the bindingness gate is unchanged (noise_scale_ratio 0.05–0.10, ~10–20× over-batched), so
  no axis is a training-VARIANCE lever today. The ordering's live use is on the BIAS side:
  on-policy data has ZERO coverage of unsampled actions, and reroll-manufactured counterfactual
  labels are gated by bias meters (V-vs-MC divergence, calibration, exploiter bounds), not by
  noise_scale.
- 🚨 **INSTRUMENT CAVEAT that generalizes — CRN here shares the dice STREAM, not the
  roll-to-event MAPPING**: `replay_driver.js` swaps `b.prng`, and a different action consumes the
  stream differently, so a SINGLE-SEED action sweep — exactly what the prober's `lookahead` does —
  carries ~one dice-variance of contamination and over-read U/O by ~2× here. Any variance claim
  off that path needs replicated dice + the independent-half cross-product estimator (unbiased
  under any weighting; validated at zero true effect).
- **Limits**: move/move rounds only (forced-switch rounds structurally uncovered); 4-of-~7.5
  legal levels ⇒ main effects are lower bounds; one checkpoint, eval distribution; λ=0.80 scales
  all four axes' reach equally so the ordering holds.

### Counterfactual label factory — COST MODEL (2026-08-21, opus offline probe, paired A/Bs on gen-17 @24M, self-validating to 0.987)

**A one-ply counterfactual label (K=8 opponent branches, population-mean turn 24.7) costs 162 ms
TODAY and 28.4 ms OPTIMIZED using only code already in the tree — 5.7×, nothing new written**
(memory: `project_counterfactual_label_costs.md`; report `tmp/counterfactual_cost_model.md`).
Anchor: node and rust agree on the label V to exactly 0.0 on 50/50 decisions — bit-identity at
the LABEL, not just the protocol.

- **The unoptimization was the TRANSPORT, not the critic.** A warm rust `SearchSession` beats the
  node `reroll_many` path by a PAIRED **289×** (fixed+marginal: node 426 ms + 20.1/arm vs
  rust-warm 1.93 + 0.168). The owner's compiled-critic hunch was directionally right (the critic
  IS eager; compiling is free, 5.90× at B=1, max|Δ| 3e-5) and numerically wrong: the critic is
  2.5% of the bill, so compiling buys 1.25%. ⚠️ Under production BLAS pinning the compiled critic
  is **0.91× at B=64** — the unpinned 3.13× reading is a scheduler artifact (its own eager series
  is non-monotonic in B, marking the whole unpinned table invalid).
- **The post-transport bottleneck is the MATERIALIZER at 91%**: `materialize_decisions` replays
  the whole prefix from turn 1 for EVERY arm — arm_ms = 4.78 + 0.853·turn (R²=0.996), prefix
  replay 2.53 + 0.855·turn of it; the branched turn itself is ~0.5 ms and the obs encode ~1.8 ms.
  Prefix-sharing across a decision's arms is the ONE real build item (batch-aware estimate:
  ~7.7 ms/label). Profiling trap recorded: cProfile is BLIND through POKE_LOOP (98% in
  lock.acquire); arming inside a coroutine on the loop shows `LiveView.from_pokemon` at 1084
  calls/arm = 50% of cumulative.
- **Coverage math**: 4 nice-10 background cores ≈ **12M labels/day = 1.74% of production
  decisions at K=4** (0.88% @K=8); 100% would need ~230 cores ⇒ the factory is a prioritized
  SAMPLER by construction (which the three-axis ordering already prescribes: opponent branches
  everywhere, our-action branches where π is undecided). Rollout-to-end: 221 ms/line, 792 ms for
  a win-prob at R=8 — 28× a one-ply label; that is the price of the value-bias rungs 1-2 (MC
  re-labels on visited + counterfactual-successor states).
- **Coverage gap found**: the rust `search_driver` cannot open TURN 1 (10/10 decisions, 6/6
  records; node can) = 3.35% of move decisions — exactly the first decision of every battle —
  and the error is a JSON `error` on STDOUT with EMPTY stderr, so a stderr-reading caller sees
  silence. Fix candidate, small.
- **Method rider**: per-arm costs amortize a ~900 ms per-decision fixed cost — the prior 39.5
  ms/arm figure was a K≈76 artifact, not the factory's price (the model reproduces it at that K
  and reprices K=8 at 162 ms/label). *Never quote a per-unit cost without its batch size.*

### G0 BIAS MAP — PROCEED, and the defect is RESOLUTION not offset (2026-08-22, opus agent, 2,204 tight-MC labels / 16,832 rollouts / 216 battles, gen-17 @24M, R=8, battle-clustered CIs)

**Gate G0 of `design_counterfactual_value_grounding.md` — the kill does NOT fire; the meter gets
amended.** Report `tmp/g0_bias_map_report.md`. Label trust established first: anchor 99/100,
near-terminal 98.8%, 14/14 opponents covered, 0 self-model approximations; 12/1,639 tasks
unlabelable (one 250-turn-cap battle, both engines — NOT the rust turn-1 gap).

- **The 0.827 class reproduces as +0.23, and it SPLITS — the conviction was half wrong in the
  most useful way.** On bot-loss states with predicted win-prob ≥0.75: predicted 0.868 vs
  tight-MC 0.637 (+0.231 [+0.154,+0.313]); matched-confidence WON control −0.076; the loss−win
  difference is **+0.307 [+0.227,+0.392]** in one CI. But **53.1% [42.0,64.5] of those states
  were GENUINELY winning (MC ≥0.75 — the dice lost the game)** and only 29.6% [19.5,40.5] read
  MC <0.5. A single realized outcome cannot make that distinction — this measurement is the
  instrument's own existence proof.
- **THE REFRAME: the head's defect is RESOLUTION, not an optimism offset.** Population-mean gaps
  are |0.05|–|0.07| while the TRUE within-decile spread of P(win) is **0.11–0.36, of which
  80–95% is real state-to-state variance, not R=8 noise** — a 2–6× per-state error the mean
  cannot see. The head lumps states of very different true value into one confidence bin
  (Murphy-decomposition reading: tolerable reliability, poor resolution). ⚠️ **A G4 arm that
  merely re-centers the head would score as success on the wrong meter — the primary G4 meter is
  AMENDED from mean predicted−MC to `sd_true_excess` (within-decile true spread the head fails
  to resolve)**, recorded here as the G0-licensed amendment; the design doc edit awaits an
  explicit pass. This STRENGTHENS the R1 case: only tight-MC labels (not single outcomes, not
  re-centering) carry the within-bin separation signal.
- **The ecology-calibration theory is now MEASURED, with a sign flip.** The head under-predicts
  vs bots (frame-weighted +0.054 but TRUE-play-distribution **−0.065**) and over-predicts vs the
  pool (+0.106/+0.058) — because BCE-trained on a ~90% self-play mixture where P(win)≈0.5, it is
  calibrated to its ecology, not the game (the imperfect-info note §6 claim, empirical). Rule:
  **never quote "the critic is optimistic by X" without naming the population** — the sign
  depends on it.
- **Honest hole carried forward**: this maps the WIN-PROB HEAD, not the main critic V (the
  secondary V comparison has the PBRS caveat). The R1 head-first delivery is unchanged; a V map
  is a later, separate read.
- **Code-fix candidate found (task)**: the prober builds `--opponent-ckpt` opponents GREEDY
  while recorded sentinels played STOCHASTIC — re-labelling all 477 sentinel states in the
  correct regime moved MC +0.037 [+0.007,+0.066]. One-line seam on `prober/replay.build_opponent`.
- **Factory economics rider**: 878 ms/label at load 7 → 2,787 at load 25 (3.2× for 3.6× load —
  worse than the loadavg/cpus scaling predicts); 72 core-minutes bought the whole gate. Rollout
  throughput beside a trainer is a LOWER bound.

### The factory is BUILT and PLUMBED — steps 3+4 of the counterfactual program landed (2026-08-22, two opus agents in parallel isolated worktrees, orchestrated landing)

**`a85a3bf` (the tool) + `6c2cb45` (the plumbing) + `2ea687a` (a pre-existing red on main, fixed).**
The human-named ladder now reads: *disease proven ✅ → tool built ✅ → plumbed in ✅ → run the
experiment (parked, needs a training slot)*. Highlights and honest edges:

- **Prefix-sharing materializer** (`obs_materializer.materialize_branches`): 59/59 decisions /
  452 arms **byte-identical** to the per-arm path (every arm compared, not sampled — the clone
  SHARES append-only records, so the gate is the contract's enforcement); **2.91×** (15.4→5.3
  ms/arm), BELOW the ~5× estimate because the state clone replaces the replay (~5 ms at turn 12,
  the new bottleneck). The bit-identity gate caught two real bugs pre-landing (dropped unfed
  prefix tail; exhausted-actions flag not reset on restore). `lookahead` now rides it.
- **`cf_audit`** is a permanent command (`python -m agents.training.cf_audit <run>`): bias map
  with `sd_true_excess` + schema-v1 labels, anchor-refusal before any label (exit 3, no labels on
  a failed anchor). Its first real run independently CONFIRMED G0's shape on a 30× smaller sample
  (conviction gap +0.280 vs +0.231; 42% luck vs 53%; sd_true_excess deciles 0.25/0.25/0.17 vs
  0.30/0.26/0.21) — the instrument replicates.
- **Task #27 CLOSED** (prober opponent regime honors the RECORDED stochasticity; regression test
  fails on revert; regime stamped into `opponent_source` so no artifact can hide it). **Task #26
  half-closed**: the stdout-error-reporting defect is fixed and the diagnosis narrowed to
  `search.rs::at_turn_start` returning false at t==1 on a fresh session; the rust fix itself
  deferred (needs a cargo build beside a live trainer) — a precise KNOWN-GAP note marks it.
- **The plumbing** (`gen3_cf_label_plumbing_v1`): `--cf-records` tap (count-capped global ring,
  crash-safe, sink-exceptions swallowed so the bridge reader can never wedge), `cf_label_buffer`
  (incremental JSONL poll, staleness expiry at `--cf-label-lag-steps`, every anomaly a COUNTED
  skip with liveness scalars — `cf/labels_ingested_total` flat = producer dead, distinct from
  expiry), `--cf-winprob-coef` default 0 + `--cf-head-only` default TRUE. **Byte-identity proven
  three ways** (seeded train() state_dict SHA equal HEAD vs base; arch surface SHA equal; file-set
  equal) plus the in-suite pin (populated buffer at coef 0 ⇒ th.equal update). Head-only measured:
  `cf_grad_share` 0.0 at every step; trunk-open 0.045–0.085. Flag class = the `--opd-coef` genre
  (plain argparse, launcher-forwarded, NOT flagless-resume-inherited — documented in the help).
  One G3 sub-claim stands on construction not measurement: launcher-restart survival of the tap.
- **The pre-existing red both agents independently verified** (stash-and-rerun on base) is fixed
  on main: `eval_callback_test::test_eval_games_override_flows_through_schedule` built the
  callback via `__new__` and predated `gen3_eval_freq_flag_v1`'s instance-attr cadence — the test
  now sets both knobs and asserts the new one flows. *Method note: a test that bypasses `__init__`
  silently couples to the constructor's attribute set; it broke when the E-gate work made the
  cadence a knob, and rode main red until two unrelated agents both hit it.*
- **Landing procedure worked**: parallel isolated worktrees + a pinned shared schema; tool shipped
  first, plumbing rebased over it (one clean structural conflict — both appended sections to the
  training leaf — resolved keep-both), 376 merged-tree gate tests green before push.

### Adversarial review of the factory landings — two REAL bugs, one wedge-class gap, the scary contracts HELD (2026-08-22, opus agent, landed `5b8f485`)

**The review paid for itself on the exact failure the code claimed to survive.** Thirteen named
attack surfaces over `a85a3bf` + `6c2cb45`; findings fixed+gated in one commit, the rest tasked
(#28, #29). Verdict: **production-safe for a `--cf-records` run**; two label-quality decisions
(#28) owed before `--cf-winprob-coef` ever goes live.

- **REAL-BUG (fixed): the tap leaked `.tmp` files forever on write failure** — the prune matched
  only the record suffix, so a FULL DISK (the scenario the module explicitly promises to survive)
  orphaned one tmp per episode per worker, silently after the once-per-process warn. Fixed:
  unlink-on-failure + prune sweeps stale tmps.
- **REAL-BUG (fixed): buffer offsets keyed on filename alone** — a producer that deletes+recreates
  a label file had the reader seek past the new file's first bytes (measured: 3-row recreated file
  ingested 1; same-size rewrite ingested 0 — SILENT label loss, violating the module's own
  "never a silent accept" rule in mirror form). Fixed: `(name, inode)` keys + map pruning.
- **LATENT (gated): the battle-ENDING arm's restore path had never been executed by anything** —
  the parity gate and `lookahead` both filter `ended` arms, so the `_battle_count_queue` drain +
  tracker re-population after eviction ran under zero tests. Reverting it makes the new `sim` gate
  **wedge forever** (killed at 150 s) — the defect class was a hang, not a wrong answer. *Method
  lesson: two consumers independently filtering the same case means the case has NO consumer —
  and therefore no gate — until someone writes the adversarial one.*
- **HELD under attack (the important negatives)**: the aliasing contract (23/23 arms bit-identical
  incl. 15 terminal, mutation tripwire over 258–483 pinned shared objects, zero drift; static
  sweep found no in-place mutation of `BattleEvent`/`BattleContext` anywhere); the stash-clobber
  fold order (straight-line source order, CF fold last at 1563, no flag combination reorders);
  compile coexistence (class-eager call never enters dynamo; stash write unconditional); the ring
  prune race (bounded transient overshoot only); fork-safety (0 threads at import,
  `compile_prewarm_test` green).
- **Open by design, tasked**: #28 — no dedup on `obs_sha1` (N× weight per duplicated decision),
  future-`policy_step` labels immortal after a crash-restart rollback (tell:
  `cf/label_age_steps_p50` negative), and the ObservationDebugger being fed CF rows as if current
  under `--no-compile-trainer`. #29 — `better_line`'s INTERIOR-ply opponent is greedy argmax,
  bypassing the regime seam `a85a3bf` fixed at the divergence ply: an undeclared regime MIX in the
  beam (maybe deliberate — worst-case-opponent search — but nowhere stated). The review's flat-BCE
  note (#9) is superseded by the in-flight Beta-targets build (binomial n-weighting).
- **Static perf notes banked, unmeasured** (live trainer on the box): head-only CF forward runs
  with grad it immediately discards (free `no_grad`); per-row npz reopen; per-write full readdir
  on the bridge reader coroutine; the eager CF forward ≈ +6% row-forwards at production shapes
  when the coefficient goes live.

### Beta-distributional cf targets BUILT — evidence weighting + the head that confesses its blur (2026-08-22, opus agent, landed `0baf7d7`, MODEL_CONFIG_VERSION 98)

**Both tiers of the richness-ladder upgrade shipped, dormant by construction** (the cf coefficient
is still zero everywhere; nothing changes for any run until the experiment).

- **Tier A — `--cf-label-likelihood binomial` (the new default of the never-launched lever):** the
  scalar cf term's flat BCE becomes the exact binomial NLL — `w = round(label·n_rollouts)`,
  `NLL = w·softplus(−z) + (n−w)·softplus(z)`, folded as **Σ NLL / Σ n** so a producer changing R
  never silently rescales the coefficient, and **at n≡1 it is `th.equal`-identical to flat BCE**
  (pinned). An R=16 label now pulls 4× an R=4 label — correct likelihood weighting, which also
  discharges the review's #9 (flat-BCE-undocumented) by construction.
- **Tier B — `CfEvidentialHead`:** Beta(α,β) via softplus+1 off a **detached-always**
  `value_pooled` read — a pure supervised READOUT, never called by the extractor forward, feeding
  nothing; built LAST in `__init__` so ON-at-coef-0 is bit-identical, not merely shape-identical.
  Loss = **Beta-Binomial marginal NLL** against the rollout counts (verified vs
  `scipy.stats.betabinom.logpmf` to 1.7e-06) + `--cf-evidential-reg`·KL(Beta(α,β)‖Beta(1,1))
  riding inside the coefficient. Smoke behaves as theory demands: on RANDOM fixture labels the KL
  pulls precision DOWN toward uniform (3.41→2.71) — the head correctly refuses to claim evidence
  noise doesn't contain — and `train/cf_evidential_grad_share` reads **exactly 0.0** live (the
  always-detached contract, measured not asserted).
- **Versioning:** `cf_evidential` IS in the flag registry (it passes through
  `build_extractor_arch_kwargs` — the `win_prob_mode` precedent); v97→98 with a
  setdefault-False migration (a default, not a refusal — the module could not exist before);
  no `ARCH_SIGNATURE` bump; the three coefficients are the `--opd-coef` class. The agent's own
  note is worth keeping: with no forward call there is NO shape error anywhere, so
  `check_compatible` is the ONLY gate that can catch a flipped flag — the version field is
  load-bearing here in a way it usually is not.
- **PRE-REGISTERED READ (in the training leaf):** the head cannot fix the blur G0 measured, only
  confess it — success is the predicted Beta's WIDTH correlating with the measured
  `sd_true_excess` per stratum. Both cf terms share ONE buffer sample + ONE extractor forward
  (counted by a test). Declared skip: the per-decision (α,β) npz trace capture — the stash
  exists (`fe.last_cf_evidential`), nothing writes it yet.
- **Landing:** rebased clean over the review fixes (`5b8f485` — orthogonal layers); combined-tree
  verification 4,890 passed exit 0 + static/artifact/registry checks green. One pre-existing
  flake documented en route: `better_line_integration_test` seeds with a WALL-CLOCK timestamp, so
  it plays a different battle every run and an unlucky one outruns its recorded command list —
  flake-class, not load-class; worth a deterministic-seed fix someday.

### Experiment-readiness batch — tasks #28/#29 CLOSED, and the producer's DEFAULT output was unconsumable (2026-08-22, opus agent)

**R1 is now owed nothing but its label-producer driver.** The review's two tasked label-quality
items, its free perf notes, and the missing reader for the evidential head's own pre-registered
meter all landed together; a DRAFT R1 runbook (`cf_r1_runbook.md`) pre-registers the arm.

- **REAL BUG found en route, and it was on the DEFAULT path**: the buffer ignored `decision_idx`
  for `obs_npz` rows and `reshape(-1)`'d a battle's whole obs MATRIX into one vector, which then
  failed the obs-width GIGO guard. So `cf_audit`'s default output (npz-pointing; `--inline-obs` is
  the opt-in) was **100% unconsumable, loudly but for the wrong reason** — the warning accused the
  producer of architecture drift. Verified end-to-end after the fix on real gen-17 traces: 6/6
  accepted, 0 skips, digests verifying; before it, 6/6 `obs_dim` skips. *Method note: both halves
  of a two-process contract were tested, and neither test ever ran the other half's real output —
  the buffer's npz test stored a 1-D vector, which is the one layout the producer never writes.*
- **#28 (label quality, all three):** dedup on the obs digest keep-NEWEST (`cf/labels_replaced_total`;
  the measured 5-row-file → fill-6 rewrite case now converges to 5) · **symmetric** staleness on
  `|age|` with a named one-time warning and `cf/labels_future_total` (a crash-restart rollback made
  future-dated labels IMMORTAL; the live tell was `cf/label_age_steps_p50` = −4,999,000) · the
  ObservationDebugger SUPPRESSED (not disabled) around the CF forward, which was being handed 256
  recorded foreign rows per minibatch as if they were live decisions.
- **#29 (better_line's interior opponent): DECLARED, not changed.** Greedy argmax at interior plies
  is kept as the deliberate worst-case-opponent search assumption — sampling there would return
  lines that beat one draw of a die, an optimistic bias in the direction this tree already pays for
  — and it is now stated at the site, in the prober leaf, and stamped into every payload as
  `interior_opponent_regime`, with a gate that fails if the key disappears.
- **The evidential head's meter has a reader**: `cf_audit` gains `width_vs_blur_spearman` (rank
  correlation across strata between the confessed Beta width and the measured `sd_true_excess`, with
  a battle-clustered bootstrap that rebuilds the strata per draw) plus per-decile width/precision
  columns. A checkpoint with no head OMITS the columns and says so — zeros would render "no head"
  identically to "no uncertainty" — and a FLAT width scores `None`, not 0, because "wide everywhere"
  and "width unrelated to blur" are the same null but different diagnoses. Exercised on gen-17 (which
  predates v98): the no-head leg reports ABSENT, correctly.
- **Perf, free:** the CF forward runs under `no_grad` unless something wants the graph (the condition
  computed EXACTLY — `cf_head_only` OR a dead coefficient — because the one arm that needs it is the
  trunk-open A/B, and silently dropping it there would make both arms head-only); per-file npz LRU;
  the ring's per-write full readdir throttled 1-in-16 with a declared, gracefully-degrading overshoot
  bound (the cap is global, so any writer's next sweep collects every other's backlog).
- **The G3 sub-claim that stood on construction now has a test**: sequential fresh ring objects over
  one directory maintain the global cap and never double-count — the launcher-restart survival of the
  tap. Plus `cf/rows_sampled` (rows CONSUMED per `train()`), because residency and throughput are
  different questions and only the second goes to zero when a producer dies with labels still resident.
- **Still owed, honestly:** the label-producer DRIVER (nothing yet runs the loop from `cf_records/` to
  `cf_labels/`), and the per-decision (α,β) npz capture — the latter is NOT the "wire the stash
  through" job it looked like: `fe.last_cf_evidential` is written only by the train loop, and the
  extractor forward never calls the head, so an honest capture must CALL it at record time.

### The PRODUCER DRIVER lands — R1 is fully executable (2026-08-22, opus agent, `897ab62`)

**`python -m agents.training.cf_producer <run_dir>`** — the last unbuilt piece of the
counterfactual line: a hand-launched sidecar that watches the tap's ring, replays each record once
(`obs_materializer.scan_record`, NEW — obs + mask + action INDEX recovered in one replay by
inverting choices through the real mapper; the two replay players now share `_encode_or_track` so
they cannot drift on the one step where drift silently changes an obs), scores by a versioned
priority (critic surprise + policy entropy), rolls the top-N out tight-MC, and writes schema-v1
labels. Highlights:

- **The composition test carries the strongest assertion of the week**: the obs the producer
  materializes from a record is **bit-identical (`np.array_equal`) to the obs the LIVE player
  encoded** into states.npz — the only claim that proves the inverted action history did not
  desync the encoder's trackers. Passed first run. Real ring → real producer cycle → real
  `CfLabelBuffer`: ingested == written, 0 skips, digests re-verified, second cycle a no-op.
- **The mini-run showed the sampler selecting its own target**: a produced row carried
  `critic_surprise 0.714 == win_prob 0.714 on a lost battle` — the conviction region, self-chosen.
- **The ECOLOGY DECISION is documented in three places** (module, leaf, runbook caveat): a
  training record names NO opponent, so v1 rolls out the CURRENT snapshot on BOTH sides at temp
  1.0; every label says `opponent: "self_current"`, and the known-direction error is stated (a
  bot-episode label is biased LOW — a weak opponent replaced by a self-like one). Closing it means
  threading opponent identity through the tap — a named future change, not a silent assumption.
- **The anchor is the FULL-REPLAY oracle** (recorded dice + scripted commands to termination), a
  deliberately STRONGER form than cf_audit's — no policy acts, so a failure is unambiguously a
  defect; a crash counts as failure; failure → exit 3 and no further labels. Crash safety is
  claim-before-work (state fsync-replaced before rollouts — never double-labels; pinned on the
  ORDER). Stale-trainer PAUSE (`--stale-checkpoint-minutes` 90) replaces the prompt's lag-guard —
  the producer IS the snapshot holder, so honesty is stamping, not guarding.
- **A SECOND wall-clock-seed flake found and fixed** (`cf_audit_integration_test`'s fixture drew a
  TIE and never reached the anchor arm it tests — stash-verified pre-existing; fixed by redraw,
  since a fixed seed only relocates the coin flip). Same class as `better_line`'s — this is now a
  named test-fixture antipattern: *a fixture seeded from the wall clock plays a different battle
  every run and eventually plays the one that skips the assertion*.
- Gates: routine suite 5,988 passed exit 0 post-rebase; 47 unit + 2 sim composition tests; all cf
  pins; static + artifact checks green.

**STATUS: every R1 prerequisite is discharged** — runbook SIGNED OFF (`36a7ab3`), producer built,
consumer plumbed, meters instrumented, labels evidence-weighted, the evidential head dormant.
Outstanding beside the training slot: the hidden-information FLOOR probe (in flight) which may
amend the primary meter's expectation with evidence, the V-transfer baseline column (cheap, named),
and the ecology-drift decision at launch if the arm's opponent mix is no longer ~90% self-play.

### The HIDDEN-INFORMATION FLOOR — ~40% of R1's primary meter is irreducible, and it is CONCENTRATED (2026-08-22, opus probe, 123 decisions / 70 battles / 10,040 rollouts, gen-17 @24M)

**AMEND-THE-METER verdict, applied to the runbook same-pass** (the sign-off's binding clause:
edits require new evidence — this probe IS the evidence; report
`tmp/hidden_info_floor_report.md`). `sd_true_excess` is computed from OMNISCIENT rollouts while
the head sees only our information set, so it sums LEARNABLE blur + the irreducible variance of
the opponent's hidden half. Measured by pool-consistent DETERMINIZATION of the never-revealed
slots (prefix byte-identity verified **1,150/1,150** — the swap provably does not perturb the
replay; a deliberately-mismatched control proves the gender-PRNG guard load-bearing):

- **The floor: sd 0.151 [0.119, 0.186] in deciles 7–9 = 39% [24%, 87%] of the meter's variance;
  ~⅓ in the wp≥0.75 conviction region** (0.338 [0.199, 0.696] frame-weighted; 0.373 on the 0.827
  class). The probe replicates G0's blur on its own 84-state axis (0.301/0.303/0.178 vs G0's
  0.303/0.258/0.209), which is what makes the fraction readable.
- **CONCENTRATED, not a fog — the constructive half**: 49% of states carry essentially ZERO
  hidden-information variance (16 states fully slot-determined), the top 10% carry HALF the total
  floor, top 20% carry 75%. R1's learnable signal on the majority is intact. And the floor is
  FLAT in hidden-slot count (1 ≈ 5) and in game turn — the quantity is *"does the unknown decide
  this position"*, a state-level property, not "how much is unknown".
- **The one-state illustration**: predicted 0.879, ONE hidden slot — Salamence ⇒ MC 0.125;
  Gyarados/Skarmory/Gengar/Charizard ⇒ 1.000; Vaporeon ⇒ 0.000. G0 scores that as the head 0.75
  too optimistic; it is a COIN the head is not allowed to see.
- **The slot channel IS the floor**: varying every unused move of every revealed mon on top of the
  slot swap moves it −0.0011 [−0.0102, +0.0086] — a tight null, so the number is not a lower
  bound waiting to grow. Robustness: the strict tier-1-donor posterior reads HIGHER (hybrid-team
  incoherence is not inflating it); the true-vs-alternative level gap is proven SELECTION
  (+0.091 on wins / −0.160 on losses, opposite signs) and the estimator excludes the true arm.
- **Amendment now binding in §2**: non-zero asymptote (~0.15 in deciles 7–9); effect sizes quoted
  on the EXCESS over the floor (a 20% learnable reduction reads ~12% raw — raw comparisons
  understate the arm); the flat-kill evaluated floor-subtracted; the arm-vs-control variance
  DIFFERENCE at matched step is the primary comparison (the floor is a population property and
  cancels). Population named: uniform pool-consistent opponents (verified the exact posterior —
  228/228 recorded opponent teams are pool members); an UPPER bound on what a
  behaviour-conditioned head would face.
- **Method riders**: two artifacts caught by the verification gates before they shipped wrong
  numbers (Hidden Power's protocol id never matches the packed typed id — broke 440/615 arms;
  duplicate donor draws made an axis read smaller than the axis it strictly contains). Exclusions
  1.6%, one named cause (a recorded opponent command switching to a still-hidden mon — an
  incompatibility of "hold the action fixed" with "vary the hidden half", not a leak). Cost: 29
  core-minutes at load ~30.
- **Conceptual bank**: this is the imperfect-information note's PBS-irreducibility made a NUMBER —
  the first direct measurement here of how much of "the critic is wrong" is actually "the game is
  hiding a coin". It also sizes the ceiling of any future belief-conditioned value work: the
  concentrated top-decile states are where a better OPPONENT-TEAM BELIEF (not a better value head)
  is the only lever that can move the meter.

### E-battery ADJUDICATED — the "thin" verdict SURVIVES conditioning, the bait turf is WORSE than pooled, and E3 deepened after all (2026-08-22, opus conditioned read on gen-17/E1/E2/E3, content-only, battle-clustered)

**The dilution trap was CHECKED and did not fire — first time in three suspicions** (report
`tmp/e_battery_conditioned_read.md`). The correction is bounded 1.3–1.6× (max 2.4×) because the
relevant strata have HIGH exposure (25–49%), and the cap is structural: **the maximum dilution
correction is 1/exposure** — pooled reads can only hide large effects behind SMALL strata (c1's
turf was 18%, OA2's bait turf 6%; `mech_offered` at 25–49% cannot hide a 20× effect by
arithmetic). The battery's pooled numbers reproduce (one 0.01-pp CI graze, traced to per-step
sample composition), and the INSTRUMENT is clean: content-only ≡ full-zero on all 20 arm×cell
pairs; the tied-bias artifact class is structurally unrepresentable here (the four cells CONCAT
disjoint slices, 14/15/9/4 wide, own gradients — nothing shared to tie).

- **The sharpest new fact: on BAIT-conditioned states the cells read BELOW pooled** (0.18–0.53×),
  and `pair_outcome_move` + `conditional_threat` read **exactly 0.00% flips on bait in all four
  arms** (n=253–298) — **partially BY CONSTRUCTION**: `reduce_pair_in` takes the UNRENORMALIZED
  α slice, which sums to (1−α_SWITCH), so the pair rows are scaled toward zero precisely when a
  switch is predicted. The pre-registered "switch-predicted" stratum was therefore BACKWARDS for
  3 of the 4 cells — their physical home is the they-STAY tail — and **the substrate's coverage
  of the switch-contingent world rides entirely on `switch_branch` (OA2)**. A design fact for any
  future substrate work: renormalization is a decision, not a detail.
- **One correction to the battery report: E3 DID deepen.** A paired TRANSFER CONTROL (run the
  fork BASE on each fork's own states) separates model effects from state effects: E3 carries a
  **+1.97 pp pooled MODEL effect, CI [+0.68, +3.30]**, concentrated in `switch_branch`; E2's
  apparent rise is **100% a state effect** (the base reads 5.13% on E2's states vs E2's own
  4.39%). "Forks did not deepen it" → "two of three did not; E3 did, via OA2." *Method worth
  naming: the fork-base-on-fork-states control is how a fork's dependence delta gets decomposed —
  without it, a pilot-team change masquerades as learning.*
- **The proposed E4 decision rule is UNDECIDABLE on pooled numbers, by arithmetic**: a 3×
  bait-conditioned deepening moves pooled +0.09 to +0.40 pp (0.2–0.9σ at n=6,000); even bait
  flips at 100% move it ≤6.3 pp; and a pilot-team swap alone moved the SAME model +1.34 pp.
  **The rule keys on the bait-conditioned `switch_branch` read vs E1 at MATCHED states** (n≈280,
  CI half-width ≈1.3 pp — a 3× IS resolvable), paired with the B1/B3 behavioral bars; pooled is
  context only.
- **Instrument findings, tasked/noted**: `edge_ablation_audit`'s mask recovery is BROKEN on this
  trace format (logits stored pre-mask ⇒ `> -1e8` yields all-legal ⇒ its guard passes VACUOUSLY)
  — fix before any family audit on these runs. `load_model_snapshot` refuses E-arm checkpoints
  (PopArt resume strictness — correct behavior); `load_foreign_opponent` is the working path for
  cross-run analysis, worth remembering. And mean |dV| is **exactly 0.0** for all four cells in
  all arms — they are policy-only by construction (the pointer route); the critic never sees
  them, which bounds what any critic-side readout of E4 can attribute to the substrate.
