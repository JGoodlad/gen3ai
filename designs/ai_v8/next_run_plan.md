# Next-run plan — the pre-flight list for the next fresh run / major fork (post-ai_v8_03)

Living decision doc (2026-07-20, owner + assistant session). "Next run" = the next from-scratch
run or retrain-class fork after ai_v8_03. Items are grouped by decision state; each carries its
gate. Context: ai_v8_03 (zarch/FiLM + booster stack, full LR since 217M) is oscillating in a
~1990–2020 ELO band; the staged in-run interventions (belief-grad flip, mining pipeline) may
resolve some contingencies below before the fork happens.

## Locked-in changes (do these)

1. **Full-info (privileged) critic as the GAE baseline + the public value head kept as an aux
   readout.** FiLM the TRUE hidden opponent team (training-only labels, already emitted) onto the
   vf head — in the SCOUTING-SAFE form: condition on the public obs/history AND the privilege,
   never the privilege alone (a naive oracle critic assigns negative TD advantage to
   information-gathering — the Swampert-Protect-scout failure; Baisero-Amato). Keep the
   one-sided/public V as an auxiliary head — it is the deployed readout (prober, search,
   lookahead) and the reveal-jump probe target. **Anti-atrophy parameterization (owner
   2026-07-21): V(h,s) = V(h) + Δ(h,s)** — the public head is the BASE, privilege is an
   explicit residual correction with a magnitude penalty on Δ, so laziness routes variance
   into the public pathway by construction (the shortcut is structurally unavailable) and the
   co-trained V(h) keeps the trunk's belief/history features alive (the inverse of the
   detached-belief collapse). Optional: privilege-blackout augmentation on the value input
   (never touches the policy ratio). Gates: falsify-scan `unattributed` bucket shrinks; V(h)
   still jumps on reveal events with Δ ABLATED; **|Δ| decays with reveal count within games**
   (Δ = the value of remaining hidden info — flat |Δ| = shortcutting). **TB metrics to
   implement with the head (owner: capture these):** `privval/delta_abs` (mean |Δ| per
   batch), `privval/delta_reveal_corr` (corr(|Δ|, n-revealed-opp-attributes) — must be
   NEGATIVE and strengthen), `privval/public_reveal_jump` (mean V(h) change on reveal-event
   decisions), `privval/pub_priv_gap` (E[|V(h,s) − V(h)|] — the live shortcut monitor).
2. **Distributional value function promoted (categorical-critic Phase B).** `--value-dist-mode`
   from side-readout to the PRIMARY value loss (categorical CE; drop/heavily-downweight the
   scalar MSE — the "Stop Regressing" recipe; the weak aux version did not prevent
   crystallization). Widen vmax first (the v29 finding). Keep `--win-prob-mode shaping` (proven,
   ~0 ms). **WARM-STARTABLE — MEASURED, all three gates resolved (2026-07-21,
   `tmp/dist_vs_scalar_probe.py` @233.8M on the 1534 frozen churn-probe states):** E[Z] vs the
   scalar V in the SAME (PopArt-normalized) units — pearson 0.988 / slope 1.033 / bias −0.01σ /
   MAE 0.09σ; the TAIL (top-10% |V|) agrees BEST (0.993, slope 0.987, ZERO sign flips) — so
   swapping the GAE source is near-seamless. (First pass compared across units and looked
   broken — slope 0.066 = exactly 1/σ_PopArt(15.76); the head trains on NORMALIZED returns per
   the `_value_dist_loss` docstring.) Resolutions: (i) KEEP PopArt — GAE reads
   denorm(E[Z]) = E[Z]·σ+μ, identical plumbing to the scalar; (ii) the support must be
   **NARROWED to ~±4, not widened** — the head occupies only ±2.2σ of the ±12 support (~9 of 51
   bins live, 0.48σ/bin coarse → ±4 gives 0.16σ/bin, 3× finer; max observed |z| 2.44; zero
   edge mass; the old "widen vmax" note predates normalized-support awareness); (iii)
   advantage continuity PASSED. Phase B is a restart-class intervention on the current
   lineage: swap loss source + narrow support, expect a brief support-recalibration transient.
   **PopArt split (owner confirm 2026-07-21):** the dist head keeps PopArt's TARGET
   NORMALIZATION (the "Art" — load-bearing: fixed-support viability + gradient scale; already
   wired) but has NO exact output-preservation (the "Pop" is a linear-head identity; a
   categorical head rides μ/σ updates via continuous CE refit — negligible at converged σ,
   watch `pit_mean` early on a fresh run). IMPLEMENTATION TRAP: when the scalar MSE is
   dropped, the PopArt μ/σ UPDATE must stay alive — it is the currency peg for both the dist
   targets and the denorm(E[Z]) window; one shared normalizer, never two.
   **CLEANUP END-STATE (owner 2026-07-21): one categorical critic, no scalar head.** On the
   LINEAGE swap: keep the scalar as a frozen/low-weight aux (instant fallback + the E[Z]-vs-V
   divergence health monitor) — producer stays until the new consumer carries the load. At the
   FRESH run: the scalar head is never built; the "Pop" output-preserving machinery (which
   exists only to serve a linear head in drifting units) deletes with it; PopArt collapses to
   an EMA return-stats peg + one denorm function. The deployed readout (prober/search/
   lookahead) reads denorm(E[Z]) everywhere. COMPOSITION with item 1 (privileged critic) —
   **BOTH distributional (owner correction 2026-07-21; the scalar-Δ idea was WRONG):** Z(h) is
   a MIXTURE over possible hidden states; knowing s SELECTS a component (bimodal "Counter or
   not" → unimodal) — a scalar mean-shift cannot collapse a mode, so under the CE loss the
   unfittable SHAPE residual would leak into the shared Z(h) base, corrupting the deployed
   public head's calibration (defeats the anti-atrophy design). Correct form: a LOGIT-SPACE
   residual `logits(h,s) = logits_base(h) + Δlogits(h,s)` (zero-init, penalized — one 51-dim
   Linear, no real added complexity); GAE reads denorm(E[Z_priv]). Bonus: penalty and
   diagnostic unify as **KL(Z(h,s) ‖ Z(h))** = per-state value-of-remaining-hidden-information
   in nats (averaged ≈ I(s; return | h)) — must decay with reveals; flat = shortcutting; the
   `privval/*` metrics upgrade to this form.
3. **`--damage-refine-rounds 1`** (prefuse-style: ONE pre-layer-1 lean damage injection, no
   between-layer recompute). The recompute only pays if the belief sharpens through layers —
   measured near-zero under detached. CONTINGENT COUPLING: if the belief-grad flip (below) shows
   beliefs sharpening in-layer under shaping, re-A/B 1 vs 2; if detached stays, consider 0 + drop
   the threat channels that ride the loop (the June audit's ~19% CPU refund).
4. **Skip env-side CPU compute for GPU-subsumed obs blocks.** Under `--unified-obs` the model
   masks the incoming-damage / move-effect / active-move CPU blocks — VERIFY whether the encoder
   still COMPUTES them; if yes, skip building them (pure CPU refund on the dominant cost center;
   obs-build benchmark is the gate). Do the verification cheaply BEFORE the run.
5. **Belief-grad-mode: decide from the staged in-run flip.** The Q1 record (ai_v7_03): gradients
   measured ORTHOGONAL (H1 refuted), beliefs 3× better under shaping, performance a statistical
   wash at 30M with a small passivity signature. The staged before/after flip on ai_v8_03 (at
   plateau confirmation) decides: shaping revives the belief→physics stack → next run ships
   shaping; wash again → detached forever AND strip refine/threat/matrices (stop paying ~19% for
   inert machinery). Either way, stop straddling.
6. **Top-K + tail-risk op candidates — MEASURED: K=16, tail MANDATORY**
   (`tmp/topk_candidate_probe.py` @220M ckpt, 60 battles / 21,325 forwards / 224,218 live
   channel evals, hooking the real `_opp_candidate_weights`/`_chan_max`): top-16-by-belief owns
   94.2% of channels (coverage 95.4%, mean rel-err 4.9%); the KNEE is at 16 (12→16 halves the
   miss rate; 24→48 buys ~1pt). Misses are BIMODAL — err>5% ≈ err>50% at every K (truncation
   doesn't shave a channel, it LOSES the owning candidate: the surprise-OHKO mode) — and the
   tail never closes (even K=48 leaves 2.2% owned deeper: floor-weight nukes where w·value lets
   a 0.02-belief Explosion/4× typed-HP outscore the favored moves). So: exact physics for
   top-16 + ONE precomputed tail-worst-case bound channel covering the ~6% residual; sweep 25×
   smaller. Remaining gate before ship: the behavioral fidelity check (head-to-head WR cheap-op
   vs full-op — feature error is now known, behavior sensitivity isn't). Payoff is LEARNER-side:
   [B,n,C≈400] activations ~25× smaller (the 2026-07-20 OOM lived there) — possibly enough to
   drop `--grad-checkpointing`.
7. **Booster stack carries forward:** `--team-block-episodes 64`, `--grad-accum-steps 16`
   (global NSR ~1.07), `--team-pfsp onesided`, `--zarch-film heads --zarch-dim 32` (the
   plateau-breaker; the grid probe: +9.2pp on PFSP-emphasized teams). **Bump
   `--film-grad-accum-steps 4 → 6`** (applied ratio crept 0.80 → 1.5 as signal was consumed).
8. **Keep (proven/settled):** PopArt, async-rollout, bridge transport, draw-penalty,
   `--hp-type-belief learned` (the opp-HP GIGO fix), the unified-move stack, N_HISTORY_TURNS=7,
   full LR with NO anneal flags (anneal only as a deliberate late-run decision, not a default).
9. **Ship before the fork:** the fork eval-anchor guard (ignore inherited `latest_eval` when
   `resume_meta.step > num_timesteps` — the ai_v8_03 24M-steps-without-eval bug). The
   `_film_grad_accumulator` save-exclusion (8903a1c) is already on main — any post-fork launcher
   picks it up; do NOT resurrect model-attached CUDA state without extending
   `_excluded_save_params` (treat the exclusion list as part of any such feature).

10. **The missing all-way quadrant: THEIR bench's believed moves × our mons.** The op prices our
    active×their 6 (v34), our 6×their active (v39), their active×our 6 (v35) — but NOT their
    BENCH switch-ins' threats vs our team ("after I KO, what comes in and what does it do to
    me" — the table THEIR forced switches need, symmetric to v39; flagged as a TODO in the v34
    notes). Now affordable under top-K=16 truncation (6 attackers × 16 candidates × 6
    defenders). Owner-derived spec (2026-07-21): both sides need all-way damage + speed
    whenever a mon must come forward.

## Contingent trims (decide at fork time, from the June audit + shaping outcome)

- Threat channels (v36/v37) + refine loop: see item 3/5 coupling.
- `--damage-matrices` outgoing side / `--damage-matrices-outgoing-all`: ~0.5 ms of opponent
  forward; outgoing-all targets the forced-switch-blind-offense defect → dropping is MEDIUM risk;
  keep unless CPU-strapped, and watch forced-switch crater-share if dropped.
- `--spread-belief-nature-marginalize`: cheap (0.088 ms) and correct — keep.

## Staged experiments (not defaults; each needs its own gate)

- **Search-as-teacher at low coef from mid-run** — the highest signal-per-CPU creator once
  rollout redundancy sets in; built, never run at scale. Stage the mining pipeline
  (triage/falsify-scan → better-line verify → AWR distill) on cycle-tail CPU.
- **z_opp / matchup FiLM** — the opponent half of the conditioning space (~5 archetype clusters =
  dense targets). Gate: per-archetype win-rate spread narrows.
- **Move-tokens cross-attention (Form A)** — K move-tokens for the opp active's top-K believed
  moves (latent ⊕ belief ⊕ per-defender damage), one cross-attn layer (mon tokens = queries)
  with a ZERO-INIT output projection → identity-at-init, OFF-byte-identical, version-gated. The
  low-risk probe of moves-as-entities-in-attention; sized by the top-K probe (K=16). The
  fresh-run escalation (Form B: tokens in the main body; physics-as-attention-BIAS on the
  move↔defender edges; pointer-style action head reading per-move tokens — kills the
  action-alignment bug class by construction) is the ai_v9 hierarchical skeleton.
- **Op head-concat deprecation (the anti-accretion endgame, owner 2026-07-21).** The flat op
  concat into both projections is an accretion treadmill (each physics fact = new block + wider
  projections + version gate; audit: chunks near-inert). End-state: pairwise facts → edge
  BIASES, per-entity marginals → tokens, per-action precision → the POINTER head (the
  prerequisite — without it the concat is the only lossless path from move-k physics to logit
  6+k). Migration = the `--unified-obs` playbook: build the token/bias/pointer homes →
  granular `--mask-op-head-*` flags → A/B with concat masked → delete at a fresh run. Never
  delete the proven path before the replacement demonstrably carries the load.
- **FiLM output centering** (subtract running team-mean modulation) — RETRACTED for a converged
  run (the shared tilt was the escape hatch); on a FRESH run the trunk is not converged, so
  centering from step 0 could force differentiation early. Speculative — experiment arm only.
- **Exploiter distillation (Phase-2)** — when a probe shows an archetype-level hole; the
  double-sided recipe (teacher in the opponent pool) is validated; FiLM provides per-team storage.

## Engineering prerequisites (rank-0: multiply the whole CPU budget)

- **Rust / vectorized `state_encoder` hot path** — obs-build ≈ 80–88% of per-decision CPU on BOTH
  sides (trainee + CPU opponents); the Rust bridge + byte-parity harness pattern already exists.
  A 2–3× encoder speedup multiplies every signal source, including search.
- torch.compile is DEAD (measured: learner flat, GPU-opp OOM, CPU-opp 0.70×) — do not revisit
  without new evidence.

## Where the supporting analysis lives

- `designs/learning/amortization_gap_and_conditioning.md` — FiLM family, signal-per-compute,
  scouting caveat, conditioning ladder.
- `designs/learning/regularization_and_noise_in_ppo.md` — the unmodeled-randomness rule.
- Memory: `project_damage_op_block_audit` (per-block ms + top-K+tail), `project_belief_shaping_experiment`
  (the Q1 shaping record), `project_next_run_plan` (pointer to this doc).
