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
| **C6** | **The five value routes bought measurable critic improvement** (the delivery-line outcome question, runbook §7) | ❌ **FAIL 2026-08-17 — with liveness PROVEN, the sharply-interpretable branch** | Gen-13.5 battery (registration: `gen13_endofrun_runbook.md` §7; measurements ship with the gen-13.5 window): all five v89 routes trained off zero (§2a) and `entity_pool` carries decisively (dV 6.28 = 110% of all_off — the Stage-3 succession confirmed), yet the critic's stall-loss over-confidence DID NOT MOVE — gen-13 confident-band gap **+0.358** CI [0.23, 0.50] with confident-blind fraction **0.500** [0.29, 0.72], and the difference vs gen-12 NOT separable (+0.100 [−0.18, +0.39]); awareness flat-to-slightly-worse on every metric. ⚠️ Statistical correction recorded by the battery itself: comparing the two runs' SEPARATE CIs would have supported "gen-13 got worse" (gen-12's touches zero, gen-13's excludes it) — testing the DIFFERENCE refutes that. Never eyeball two CIs. **CONSEQUENCE: the delivery line is EXHAUSTED.** Next work is the training DISTRIBUTION of stall games (how much loss-side stall mass does a rollout actually train on — assigned to the gen-14 window) and the ESTIMATOR (C5's TD-aux, forks in flight) — NOT more routes (this row), NOT search (S1), NOT tail-weighting (K1), NOT readout architecture (C4). Route verdicts under the registered rules: `entity_pool` KEEP · `intent_reduce` + `value_clock` tie-break at ≥2× sample in gen-14's battery · `value_intent` + `intent_threshold_value` NULL → deleted in the gen-14 wave, with `value_intent`'s registered re-entry condition (any future α/β-critic proposal passes the C4 offline gate first — it was C1's "one input with no trunk substitute", deleted because the measurement says the critic doesn't use it, cheap to rebuild via the seam). | runbook §7 branches; gen-14 runbook tie-breaks; `measurements/gen13_section7_calibration.json` + `gen13_value_route_arms.json` (ship with the gen-13.5 window) |

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
