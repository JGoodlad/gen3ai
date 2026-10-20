# The collapse battery on X26's plateau, 180M–380M (2026-10-20)

**Tag: MEASURED · DESCRIPTIVE · PROVISIONAL.** One run (`rb_x26_s2601`, pin `9c033489`, ended at 380M; Tier 1 flat
from 270–280M), one seed, nothing registered before the data. Under the owner's plateau-claims rule
(`feedback_strong_claims_need_plateau`) every trend below is a description of ONE plateau, not a replicated finding.
All reads are CPU forward passes or record reads: no games, no GPU, read-only on `models/`, `data/` untouched.

**Why.** X26 went flat on Tier 1 at 280M while the cycle monitor (the battery's B6) reads MATERIAL cycling of ≈ 3 pp
throughout (`../cycle_monitor_retro_2026-10-19/`). The battery's other items (`../../../endstate/design_collapse_and_counterfactuals.md`
§4) say which other kinds of collapse sit beside that plateau. The pilot (`../collapse_pilot_2026-10-17/`) read
10M–160M, before the plateau.

## The answer

**X26's plateau is a TREADMILL, not a collapse: the policy keeps moving but stops getting anywhere.** It is
PROVISIONAL (one seed).

- **It keeps moving.** About 30 % of greedy choices flip every 20M, at a steady rate.
- **It goes nowhere.**
  - The 380M policy differs from the 180M one on only 36 % of choices, not much more than the 28–32 % that 20M apart
    already gives.
  - The pool-relative edge stays at 0.52–0.54.
  - The cyclic part of the ladder stays MATERIAL (B6).
- **Three things coincide with it, all from records:**
  1. **The self-play pool went STALE.**
     - Promotion needs an eval-cycle win rate ≥ 0.55 against the pool. That fired 9 times in the 50M before 180M,
       and 3 times in the 50M before 380M.
     - So the pool's median member age doubled, 63M → 135M steps.
     - At 380M the pool still holds snapshots from 164M.
  2. **The adaptive KL controller halved the step size.**
     - The learning rate went 1.74e-4 → 8.4e-5, while the gradient norm rose 0.74 → 1.27.
     - The dose per env step halved.
  3. **Starvation is creeping UP a little while dominated mass and regret stop falling:** +0.03 to +0.06 of decisive
     turns, detected on both referees.
     - That is the ai_v14 direction at about a third of its size. It is not yet its size (+0.14).
     - 10M → 160M was flat.
- **What it is NOT.**
  - **Not entropy collapse.** Entropy gave back the post-30M broadening, −0.04 to −0.07 nats from 160M, then held flat
    from 280M. On X26's own bank it ends where it was at 30M.
  - **Not representation-rank collapse.** Trunk effective rank still rises, 38.5 → 40.4.
  - **Not opponent-model collapse.** α's intent log loss still falls slowly, 1.824 → 1.772, flat after 280M.
  - **Not coverage collapse.** The RND variants still separate rare late-game states (AUROC 0.85–0.91) and never
    read SATURATED; the state stream did not homogenise.
  - **Not plasticity loss of the dormant-unit kind.** The dormant share is flat (≈ 0.06 at τ = 0.025) from 30M.
  - The ride-along A head knows no more than the policy's logits and cannot find the starved moves (B5).

## 1. The battery's non-cycling items: definitions and data sources

From `design_collapse_and_counterfactuals.md` §1 and §4. B6 (the cycle monitor) is excluded by the brief.

| # | type | exact definition (the design's) | data source | offline from X26's records? |
|---|---|---|---|---|
| B1 | (b) entropy | rank-mass spectrum + mean entropy per stratum on a FIXED bank, paired against the previous read (battle-clustered 95 %). G-C's bar: an ON − OFF interval wholly below −0.05 nats is the first sign of type-(b) collapse | `main.policy_spectrum read/report` on X26's bank (`../collapse_pilot_2026-10-17/bank_x26`, 7,028 decisions from X26's 30 / 90 / 150M eval traces); `train/entropy_loss` (TB) as the on-distribution second read | **YES** (forward passes; done, also on `bank_v1`) |
| B2 | (a) starvation, on-distribution | on decisive turns (≥ 1 action dominated by > ε = 0.1): the share with a near-best action (paired gap ≤ ε) under 1 % probability; its TIE-COLLAPSE (top choice near-best) / MISRANK split; dominated mass; held-out gain; IMP; greedy regret | `starvation_readout.py` on the 306-turn truth: every legal action × 32 seeds to the end, TRUE world, X26@150M greedy continuation | **YES** (forward passes on the existing truth; done). The truth is 230M OLD at 380M (B4) |
| B3 | (a) starvation, cross-era referee | the same on Lane S `truth_v2` (1,600 turns, 64 seeds, three pre-era continuations K2 / N0 / C_fix) | `starvation_readout.py` on `bank_v1` reads | **YES** (done) |
| B4 | (a) truth refresh | re-branch B2's 306 turns under the NEWEST checkpoint's greedy continuation once per ~50M | `main.policy_spectrum.truth run` | **NO**: new rollouts (≈ 3.7 CPU-h). SCOPED in §4, not run |
| B5 | (a) + ride-along A | R3 (A vs ground truth) and R4 (A's member spread on starved near-best moves) | `main.ridealong_read` meters (ii)/(iii) | **PARTLY**: on `truth_v2` YES (done, below); on the X26 truth NO (`--truth-dir` reads only `truth_v2`'s file names, F-CC-4) |
| B7 | (c) exploitability | a fresh best responder's gap at P\* | `main.best_response_gap` after training an exploiter (X35's R0) | **NO**: GPU training. SCOPED in §4 |
| B8 | (d) coverage | the RND variants' coverage / saturation series | `main.ridealong_read` (Amendment 1 (b), (c)); TB `ridealong/rnd*` | **YES** (done: offline on a `bank_v1` slice, plus TB) |
| B9 | (e) opponent model | α's intent NLL / OTHER rate by class; R5 (B's outcome information: Brier(V) − Brier(V + A + B)) | `main.belief_roles read`; TB `opp_intent/*`; R5 has NO reader code (A2-F2) | **α YES** (done); **R5 NO** (not built) |
| B10 | (f) plasticity | dormant-unit share; a fixed-target refit probe on frozen checkpoints | not built (F-CC-6) | **dormant share + weight norms YES** (`dormant.py`, built here, descriptive); **refit probe NO** (needs a probe design and training; SCOPED in §4) |

**Added here (not battery items; the brief's "opponent starvation", and the design's §1 (f) rows):**
- **B-pool, the self-play pool's composition.** From TB `train/selfplay_promoted_steps` and the run's final
  `snapshots/`: the pool is the last 20 promotions, FIFO, sampled uniformly (`--pfsp-scale 0`). The 20 files in
  `snapshots/` (164M … 376M) equal the last 20 promotions.
- **(f) policy churn:** greedy flips between checkpoints, from the B1 reads.
- **(f) feature rank:** TB `rank/*`.
- **The update size:** TB `train/learning_rate`, `approx_kl`, `clip_fraction`, `grad_norm`.
- **The value head:** TB `win_prob/critic_resolution`, `explained_variance`.

## 2. The per-item table

Values at 180 / 230 / 280 / 330 / 380M. "Detected" = the battle-clustered 95 % interval excludes 0. The TB rows are
trailing-10M means of the per-update values (1M-block SE in `results/tb_series.md`).

| item | read | 180M | 230M | 280M | 330M | 380M | trend (paired where possible) | reading |
|---|---|---|---|---|---|---|---|---|
| **B1** entropy | own bank, all 7,028 (nats) | 1.089 | 1.063 | 1.046 | 1.079 | 1.063 | 180→380 **−0.026 [−0.045, −0.000]**; 160→380 **−0.049 [−0.063, −0.035]**; 280→380 +0.017 [+0.002, +0.033]; 30→380 +0.013 [−0.014, +0.039] | the 30→160M broadening (+0.06) given back; back at the 30M level; no collapse |
| B1 | `bank_v1`, all 20,712 | 1.134 | 1.107 | 1.089 | 1.097 | 1.095 | 180→380 **−0.039 [−0.049, −0.028]**; 160→380 **−0.065 [−0.077, −0.053]**; 280→380 +0.006 [−0.004, +0.017] | the same shape: down 160→280, flat after |
| B1 | rank-1 mass, own bank | 0.586 | 0.594 | 0.603 | 0.586 | 0.591 | 180→380 +0.004 [−0.007, +0.014] | flat |
| B1 | TB on-policy entropy | 1.038 | 1.028 | 1.009 | 1.015 | 0.995 | −0.026 / 100M [−0.032, −0.021] | the state distribution moves too; a slow sharpening |
| **B2** starvation | starved share of 179 decisive turns, X26@150M truth | 0.318 | 0.324 | 0.307 | 0.335 | **0.369** [0.298, 0.433] | 180→380 +0.050 [−0.011, +0.112]; **280→380 +0.061 [+0.017, +0.109]**; **160→380 +0.084 [+0.022, +0.150]**; **10→380 +0.073 [+0.011, +0.126]** | a RISE late (280→380), mostly MISRANK (+0.045 [+0.000, +0.092]); the truth is 230M old (F2) |
| B2 | tie-collapse / misrank | 0.089 / 0.229 | 0.089 / 0.235 | 0.084 / 0.224 | 0.095 / 0.240 | 0.101 / 0.268 | — | ~3/4 misrank, as in the pilot |
| B2 | greedy regret / dominated mass | 0.351 / 0.320 | 0.344 / 0.317 | 0.349 / 0.326 | 0.359 / 0.328 | 0.334 / 0.314 | 180→380 regret −0.017 [−0.049, +0.022], dom −0.006 [−0.031, +0.021] | no longer falling (10→160M regret fell −0.046) |
| **B3** starvation | `truth_v2`, 948–950 decisive turns, range over K2 / N0 / C_fix | 0.314–0.353 | 0.355–0.386 | 0.367–0.399 | 0.377–0.399 | 0.347–0.396 | 180→380 **+0.033 [+0.008, +0.057] / +0.048 [+0.020, +0.076] / +0.043 [+0.017, +0.072]** (C_fix / K2 / N0: detected 3/3); the step is 180→230 (+0.024 to +0.041); 280→380 −0.029 to +0.000 | a small RISE 180→280, then flat; still −0.09 to −0.13 below K3 (3/3) |
| B3 | dominated mass / greedy regret, 180→380 | | | | | | dom −0.009 / −0.002 / −0.001, regret −0.003 / +0.005 / **+0.018** | NOT falling while starvation rises: the unhealthy direction (10→160M both fell, 3/3) |
| B3 | decisive-turn entropy (K2 truth) | 1.057 | 1.001 | 1.002 | 1.003 | 0.999 | — | the B1 drop, on decisive turns |
| **B5** A head | A's within-turn Spearman vs `truth_v2`'s values, K2 / N0 / C_fix (10M: 0.166 / 0.082 / 0.123) | 0.185 / 0.126 / 0.187 | 0.204 / 0.131 / 0.200 | 0.182 / 0.133 / 0.174 | 0.192 / 0.134 / 0.193 | 0.188 / 0.131 / 0.203 | A − logits: −0.012 [−0.033, +0.006] / −0.004 / +0.001 at 380M (10M: −0.022 to −0.034, 2/3 detected below 0) | A knows NO MORE than the policy's own logits about the values; it caught up from slightly worse at 10M |
| B5 | A ranks a starved near-best action in its top 2 (chance 0.25) | 0.016–0.050 | 0.054–0.058 | 0.027–0.042 | 0.020–0.036 | 0.033–0.056 | flat, ~1/6 of chance; the logits read the same | A cannot find the starved moves (the pilot's §3.5 conclusion, on X26's plateau) |
| B5 | A's member spread, AUROC starved near-best vs other legal | 0.494–0.554 | 0.552–0.579 | 0.550–0.581 | 0.517–0.583 | 0.559–0.606 | slowly up from ≈ 0.53 | the members' disagreement points weakly at starved moves (0.55–0.60); not a usable flag |
| B5 | TB `adv_corr_logit` (A vs logits) | 0.856 | 0.858 | 0.863 | 0.868 | 0.876 | +0.012 / 100M [+0.010, +0.015] | A tracks the logits more and more |
| B5 | TB A spread on STARVED / FED actions (×1e-3) | 5.7 / 4.6 | 6.1 / 4.8 | 6.2 / 4.9 | 6.3 / 4.9 | 6.4 / 4.8 | starved +0.3e-3 / 100M | the members disagree more on starved actions, slowly more so |
| **B8** coverage | (b) late-game rows vs rest, AUROC: `decay` / `feat` / base (10M: 0.808 / 0.933 / 0.854) | 0.889 / 0.926 / 0.749 | 0.834 / 0.899 / 0.701 | 0.858 / 0.853 / 0.654 | 0.850 / 0.840 / 0.782 | 0.861 / 0.914 / 0.687 | `decay` and `feat` flat; base noisier and lower than at 10M | the variants still tell rare (late-game) states apart: no coverage narrowing visible |
| B8 | (b) exploiter rows vs rest, AUROC: base / `decay` | 0.614 / 0.527 | 0.650 / 0.541 | 0.645 / 0.542 | 0.558 / 0.529 | 0.578 / 0.536 | flat to slightly down | off-distribution opponents stay weakly novel |
| B8 | (c) saturation: checkpoint-to-checkpoint SATURATED verdicts (rule: Holm, r ≤ 0.5) | 10→180: none | 180→230: none | 230→280: none | 280→330: none | 330→380: none | `stream_homogenised` false at every step | the state stream did NOT homogenise on the plateau |
| B8 | (d) trunk-feature drift since 10M (renormalised inflation) | 5.22 | 5.66 | 5.85 | 5.79 | 5.95 | DRIFT_CONFIRMED at each; +14 % over 200M | the representation still moves, slowly |
| (B1/B8 extra) | ensemble spread's AUROC for V's large errors | 0.661 | 0.649 | 0.610 | 0.632 | 0.579 | down from 0.65 | the V ensemble's uncertainty explains V's errors less well on the plateau |
| B8 | TB base RND error (mean) | 2.97e-6 | 2.52e-6 | 2.35e-6 | 2.79e-6 | 2.24e-6 | −2.9e-7 / 100M | SATURATED since ~90M (it was 2.9e-4 at 30M): base RND cannot read coverage on the plateau (F5) |
| B8 | TB `feat` variant error | 2.0e-5 | 1.3e-5 | 1.07e-5 | 9.8e-6 | 8.3e-6 | −4.3e-6 / 100M [−5.2e-6, −2.9e-6] | still falling: the trunk's feature distribution is getting MORE familiar to its own RND, i.e. narrower or more stable |
| B8 | TB `fast` (recency) variant error | 1.29e-4 | 1.22e-4 | 1.19e-4 | (1.8e-4, restart) | 1.21e-4 | −3e-6 / 100M [−1.4e-5, +1.0e-5] | flat: what is seen recently does not get more familiar |
| **B9** α (opponent model) | intent log loss of the realised next opponent action, `bank_v1` (12,274 labelled rows; unpaired 95 %) | 1.824 [1.789, 1.860] | 1.793 | 1.778 | 1.773 | 1.772 [1.736, 1.812] | −0.052 from 180M (160M 1.825); flat after 280M | improving slowly, then flat; no collapse |
| B9 | vs pool snapshots / bots / exploiters | 1.804 / 1.848 / 1.746 | 1.770 / 1.815 / 1.744 | 1.743 / 1.813 / 1.717 | 1.743 / 1.802 / 1.734 | 1.724 / 1.817 / 1.731 | — | the gain is on POOL-like opponents; flat against bots |
| B9 | TB α info gain vs the pool (nats) | 0.605 | 0.621 | 0.613 | 0.609 | 0.612 | −0.001 / 100M [−0.005, +0.001] | flat |
| B9 | R5 (B's outcome information) | — | — | — | — | — | — | NOT BUILT (A2-F2) |
| **B10** plasticity | τ-dormant unit share at τ = 0.025 / exactly dead (τ = 0), 1,952 units outside the ride-along heads, own bank (10M 0.124 / 0.017; 30M 0.063; 160M 0.058 / 0.033) | 0.060 / 0.029 | 0.059 / 0.029 | 0.057 / 0.029 | 0.059 / 0.033 | 0.056 / 0.030 | flat from 30M on | NO dormant-unit growth: no plasticity-loss signature of the Sokar kind |
| B10 | total weight norm outside the ride-along heads (10M 221.6; 160M 236.7) | 237.7 | 239.7 | 241.1 | 242.0 | 242.7 | +2 % over 200M | slow growth (weight decay 1e-5); no blow-up |
| B10 | TB trunk / policy feature effective rank | 38.5 / 57.0 | 39.0 / 59.7 | 40.4 / 63.0 | 39.8 / 58.0 | 40.4 / 61.2 | trunk +0.67 / 100M [+0.20, +0.95] | no rank collapse |
| B10 | TB value CLS effective rank | 5.24 | 5.81 | 6.09 | 6.21 | 6.42 | +0.62 / 100M [+0.52, +0.73] | the value side is still widening |
| **B-pool** | median member age (M steps) | 63 | 80 | 83 | 104 | **135** | — | the pool AGES: it refreshes slower than the learner moves |
| B-pool | promotions in the last 50M / oldest member | 9 / 80M | 5 / 98M | 6 / 134M | 3 / 148M | **3 / 164M** | 13 per 50M at 30M, 18 at 90M | promotion (≥ 0.55 vs the pool in one eval cycle) has nearly stopped |
| B-pool | training win rate vs the pool / EVAL greedy vs the 5 sentinels | 0.538 / 0.525 | 0.540 / 0.558 | 0.535 / 0.514 | 0.524 / 0.512 | 0.542 / 0.523 | +0.005 / 100M [−0.000, +0.009] | a constant small edge over an ageing pool |
| **(f)** churn | greedy flips per 20M (consecutive reads, own bank) | 0.319 (180→200) | 0.316 (220→240) | 0.307 (280→300) | 0.297 (320→340) | 0.279 (360→380) | vs 380M: from 180M 0.358, from 280M 0.341, from 340M 0.315 | constant churn whose distance does not grow with lag: a WANDER around a fixed region |
| (f) update size | learning rate (adaptive KL) | 1.74e-4 | 1.45e-4 | 1.21e-4 | 1.00e-4 | **8.37e-5** | −5.2e-5 / 100M | halved; the step per env step halved with it |
| (f) | grad norm / clip fraction / approx KL | 0.744 / 0.179 / 0.0180 | 0.785 / 0.171 / 0.0175 | 1.039 / 0.178 / 0.0193 | 1.123 / 0.173 / 0.0184 | 1.267 / 0.162 / 0.0183 | grad norm +0.27 / 100M [+0.26, +0.30] | gradients grow as the step shrinks; KL held near 0.018 |
| (f) value head | critic resolution / explained variance | 0.0781 / 0.835 | 0.0778 / 0.835 | 0.0816 / 0.840 | 0.0833 / 0.843 | 0.0820 / 0.842 | resolution +0.003 / 100M [+0.0025, +0.0040] | still improving slowly; no under-dispersion |

**B6 for context (excluded by the brief, read elsewhere):** the cycle monitor finds X26's ladder non-transitive at
every check from 10M to 370M. The window size is MATERIAL (3–6 pp) at 28 of 37 checks and never negligible.

## 3. What kind of plateau X26 is (DESCRIPTIVE · PROVISIONAL)

1. **Type (c), strategic, is where the plateau shows. The pool is part of the mechanism.**
   - X26 holds a constant ≈ 0.53 edge over its own pool.
   - Its rule for refreshing the pool needs 0.55 in one 2M eval cycle. That now fires about once per 17M, against
     once per 2.8M at 90M.
   - So the opponents it trains against age (median 135M at the end), while the policy churns about 30 % of its
     greedy choices per 20M.
   - That is the shape of a treadmill against a slowly moving target: a material cyclic component (B6) and a policy
     whose distance from 380M barely grows with lag.
   - **UNVERIFIED as a cause.** Nothing here separates "the pool stopped refreshing because progress stopped" from
     "progress stopped because the pool stopped refreshing". X35's A1 (archive pool) and A2 (PFSP) are the
     intervention test.
2. **The step size fell by half.**
   - The adaptive KL controller (target 0.01, factor 2) lowered the learning rate 1.74e-4 → 8.4e-5 while the gradient
     norm rose 70 % and the clip fraction fell.
   - A plateau read on Tier 1 is partly a plateau at a halved dose. That confound is not in the battery.
3. **Mild re-sharpening that starts to starve: (a) is turning, (b) is not.**
   - Entropy fell 0.04–0.07 nats after 160M (bounded: back at the 30M level, flat from 280M).
   - In the same window starvation rose +0.03 to +0.06 on both referees.
   - Dominated mass and greedy regret stopped falling: they fell 10→160M, and are flat or slightly up 180→380M.
   - Before 160M the sharpening was healthy. After it, a small part of it is the ai_v14 kind. The size is about a
     third of ai_v14's.
   - B4's refreshed truth would say whether the rise on B2 is real or an artefact of the 150M continuation ageing (F2).
4. **Not (d), (e) or (f) collapse** on what could be read:
   - feature ranks rise;
   - α improves, then flattens;
   - the critic keeps improving;
   - draws fall;
   - the dormant share is flat;
   - the RND variants show no saturation step and still flag rare states.
   - Base RND alone is saturated (F5).
   - The trunk's representation keeps drifting slowly from 10M (renormalised inflation 5.2 → 6.0).

## 4. Items that need new compute: SCOPED, not run

| item | what it needs | scope | cost |
|---|---|---|---|
| **B4** truth refresh | re-branch the 306 turns under X26@380M's (or P\* = 280M's) greedy continuation: every legal action × 32 seeds, true world | `main.policy_spectrum.truth run --bank bank_x26 --subset subset_x26.json --continuation <380M zip>=X26@380M --seeds 32 --threads 1 --workers 3` (the pilot's command, a new continuation) | ≈ 3.7 CPU-h (pilot: 3.5 h wall on 3 cores); minutes on the GPU. Then B2 re-reads in forward passes. **The most useful next read**: it decides F2 |
| **B7** exploitability | a fresh best responder trained against P\* (X35's R0), then `main.best_response_gap` | X35's registered reader argv with P\* as the target (`main.x35.ge`; R0 itself is unregistered, F27 / ruling C4) | ≈ 1.1 GPU-h (`design_compute_efficiency.md` §5.6) + the gap read's games in a GPU window |
| **B10** refit probe | a fixed-target refit on frozen checkpoints (fine-tune each checkpoint's trunk on a fixed supervised target, e.g. the 10M policy's logits on `bank_x26`, and compare the loss after N steps) | not designed or registered; a design first | ≈ minutes per checkpoint on CPU once built; ≈ 1 agent-day to build |
| **B9** R5 | Brier(V) − Brier(V + A + B) at the played pair | reader code (A2-F2) | ≈ 0.5 agent-day; forward passes after |
| **B5** on X26's truth | `main.ridealong_read --truth-dir` to take a generic rows argument (F-CC-4) | a small reader change | ≈ 0.25 agent-day; minutes after |

No item needs new GAMES between policies except B7 (the exploiter's training and its gap read).

## 5. FINDINGS

- **F1 (scope).** The battery's B7 (exploitability), B4 (truth refresh) and B10's refit probe need new compute. They
  are scoped in §4, not run. R5 (B9) and B5-on-X26-truth need reader code that does not exist.
- **F2 (the B2 rise may be a referee artefact).**
  - B2's truth is the X26@150M greedy continuation, 230M older than 380M.
  - "Near-best" there means near-best when X26@150M plays on, so a later policy that has moved away from 150M's
    choices can read as "starving" moves that 150M's continuation values.
  - B3 (independent states, pre-era continuations) also rises, but with a different timing: 180→230M on B3, 280→380M
    on B2.
  - Treat the starvation rise as SUGGESTED until B4 re-branches the truth. The design's own cadence (≥ 50M past ⇒
    refresh) was due at 200M.
- **F3 (pool staleness is not a battery item).**
  - The design's battery has no read of the self-play pool's composition.
  - Here it is the plainest change across the plateau: promotions per 50M 18 → 3, pool median age 35M → 135M from 90M
    to 380M.
  - Proposed: add it to the battery as a (c) record read. Not done (out of scope, rule 9).
- **F4 (the learning rate halved across the plateau).**
  - The adaptive KL controller (`dose.kl_controller`: target 0.01, kl_factor 2, lr_factor 1.2, min 1e-5) lowered the
    learning rate from 1.74e-4 at 180M to 8.37e-5 at 380M. The approx KL sat at 0.018–0.019, near its 0.02 band edge.
  - Tier 1's "flat" is therefore measured at a falling dose. This confound belongs beside every X26 plateau claim, and
    X35's breakers fork from P\* = 280M at the LR then in force.
- **F5 (base RND is saturated).**
  - `ridealong/rnd_err_mean` fell ~100× by 90M (2.9e-4 → 5e-6) and sits at 2–3e-6 on the plateau.
  - It cannot read coverage there; the `feat` and `fast` variants still move. Amendment 1's premise ("plain RND
    saturates") is confirmed on X26.
- **F6 (reader drift: checked, negligible).**
  - These reads ran at HEAD `e5ab15f4`, where X26's pre-v160 config migrates to the FP32_ALT τ solve. X26 trained
    with the naive fp32 sum.
  - The overlap check (X26@160M read by the pilot at `63f97a57` and here): max |Δp| 1.4e-5, 0 greedy flips of 7,028.
  - Gate ① held: the bank re-encodes byte-equal 7,028 / 7,028.
- **F7 (the B9 intervals are unpaired).** `main.belief_roles` reports a per-checkpoint interval only. The −0.052 log
  loss change from 180M is outside both intervals' overlap only at its ends, so it is SUGGESTED, not detected.
- **F8 (CPU contention).** Each heavy chunk waited in a 9–13-deep slot queue:
  - chunk 1: 2 h 0 min waiting, ran 22 min, peak 5.4 GB;
  - chunk 2: 1 h 53 min waiting, ran 11 min, peak 1.9 GB;
  - chunk 3: 50 min waiting, ran 8 min, peak 2.0 GB.
  - The TB extraction (2.5 min), a one-checkpoint smoke of `dormant.py` and the summaries ran outside a slot as light
    reads, at nice 19 on the agent cores.
- **F9 (the plateau window has a restart inside it).**
  - X26 resumed at 320.14M after its Amendment-3 stop (`../x26_a3_stop_316M_2026-10-18/`).
  - Every ride-along head's Adam state resets at a restart; the `fast` RND's 330M window carries that spike.
  - The 320M checkpoint is pre-stop and the 330M one post-resume.
- **F10 (the ride-along read is on a bank SLICE).** B5 / B8's offline read takes the first 200 of `bank_v1`'s 580
  battles, so six checkpoints fit one 45-min chunk. It is descriptive and is NOT the registered FINAL read of the
  ride-along registration (`../ridealong_baseline/PREREGISTRATION.md` Amendment 2), which is now due on X26's final
  (380M) with the fresh-heads floor. That read is not done here.

## Files

| file | what |
|---|---|
| `run.sh` | one step at this checkout under rule 5b (nice 19, cores 5-7 / 13-15, one OpenMP thread, no GPU) |
| `chunk1_policy_reads.sh` | heavy chunk 1: policy reads on `bank_x26` (15 checkpoints 150M–380M) and `bank_v1` (5), `main.belief_roles` (6) |
| `chunk2_heads_dormant.sh` | heavy chunks 2 (the ride-along read) and 3 (`dormant.py`) |
| `tb_extract.py`, `tb_series.py` → `results/tb_series.{md,json}` | every TB scalar of X26 (archived, 31 MB) → the battery's training-time rows and the pool's composition |
| `summarize.sh` → `results/b1_spectrum_x26bank.*`, `b2_readout_x26truth.json`, `b3_readout_oldtruth.json`, `f_churn.json` | B1 / B2 / B3 / churn from chunk 1's reads (+ the pilot's 10 / 30 / 90M reads, symlinked) |
| `results/b1_spectrum_bankv1.*` | B1 on `bank_v1` (`policy_spectrum report`, command below) |
| `churn.py` | (f) churn and the read-overlap check |
| `b9_summary.py` → `results/b9_alpha.{md,json}` | B9 α from the `belief_roles` JSONs |
| `b5_b8_summary.py` → `results/b5_b8_ridealong.{md,json}` | B5 / B8 from the ride-along JSONs (+ `rnd_variants_series.json` in the archive: (c) saturation, (d) drift) |
| `dormant.py` → `results/b10_dormant.json` | B10: τ-dormant share (Sokar et al. 2023's definition, UNVERIFIED citation) and weight norms, per layer |

Archive (not committed): `~/gen3ai_archive/x26_collapse_battery_2026-10-20/` (reads, `belief_roles/`, ride-along
JSONs, `dormant.json`, `tb_scalars.json.gz`, logs).

```bash
# from the worktree root, CPU, rule 5b; each chunk in one heavy slot
M=designs/research_state/measurements/x26_collapse_battery_2026-10-20; A=~/gen3ai_archive/x26_collapse_battery_2026-10-20
scripts/ops/mem_cap.sh --name x26cb_c1 10 timeout 45m $M/chunk1_policy_reads.sh
scripts/ops/mem_cap.sh --name x26cb_c2 10 timeout 45m $M/chunk2_heads_dormant.sh 2
scripts/ops/mem_cap.sh --name x26cb_c3 10 timeout 45m $M/chunk2_heads_dormant.sh 3
$M/run.sh $M/tb_extract.py /home/goodlad/dev/gen3ai/models/rb_x26_s2601/tb $A/tb_scalars.json.gz
python3 $M/tb_series.py $A/tb_scalars.json.gz $M/results/tb_series.json $M/results/tb_series.md
$M/summarize.sh
$M/run.sh -m main.policy_spectrum report --bank designs/research_state/measurements/m5_laneS/bank_v1 \
  --reads $A/reads_bankv1_all --labels X26@10M,X26@160M,X26@180M,X26@230M,X26@280M,X26@330M,X26@380M \
  --pair X26@180M:X26@380M --pair X26@160M:X26@380M --pair X26@280M:X26@380M \
  --out $M/results/b1_spectrum_bankv1.md --json $M/results/b1_spectrum_bankv1.json
python3 $M/b5_b8_summary.py $A/ridealong X26@10M,X26@180M,X26@230M,X26@280M,X26@330M,X26@380M \
  $M/results/b5_b8_ridealong.json $M/results/b5_b8_ridealong.md
python3 $M/b9_summary.py $A/belief_roles X26@160M,X26@180M,X26@230M,X26@280M,X26@330M,X26@380M \
  $M/results/b9_alpha.json $M/results/b9_alpha.md
```
