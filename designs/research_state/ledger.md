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
  CHOOSER, not a veto — 42c9c69): **the "decodable" branch fired ⇒ the critic route is 7a
  generalized TOKEN-CONTENT INJECTION; PV (7b, k-seed pair-values) and pair-token promotion
  (item 8) are both disfavored — cross-pair reasoning already happens.** (vf > pi on every joint
  target while pi > vf on the per-action control — the critic reads board-level structure, the
  policy per-action magnitude; consistent with |ΔV| being the concat's stickiest axis.)

_Last updated: 2026-08-08._
