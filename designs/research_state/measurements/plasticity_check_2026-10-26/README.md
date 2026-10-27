# The plasticity check on X26's lineage (2026-10-26/27)

**Status: MEASURED · DESCRIPTIVE (one lineage, one bank, CPU only, no games, no GPU).** It is the "plasticity probe" of
[`../../../endstate/strategy_if_levers_null_2026-10-26.md`](../../../endstate/strategy_if_levers_null_2026-10-26.md)
§2 (tree item T1 d), read under that doc's bars.

**The question.** Is part of X26's plateau a LOSS OF PLASTICITY: a trunk that has stopped bending to new data? If it
were, it would explain why several levers applied as plateau forks (D1, D3, N1-P, A1, SPRT, LR-F) read small or null.

## 0. One screen

- **Verdict: PLASTICITY-LOSS-UNLIKELY.** By the strategy doc's registered bars the class is NO-LOSS (§2):
  - **The registered fit:** fit P\*'s own greedy labels end-to-end from each checkpoint. Its normalised area under
    the loss curve is **0.986× X26 @25M's**; every seed pair falls in 0.984–0.987, and the NO-LOSS bar is < 1.10.
  - **The frozen-trunk version:** 0.947×.
  - **The near-dormant leg:** P\*'s share of near-dormant units is **0.78× @25M's**, against the ≥ 2× that LOSS
    needs. LOSS was therefore unreachable on that leg alone.
- **The plateau adds no further loss.** On an architecture-neutral new target (T_net: random projections of a fresh
  network's features), the loss after 400 end-to-end steps is the seed mean of 2–3 seeds per checkpoint:

  | checkpoint | loss at step 400 |
  |---|---|
  | 100M | 0.576 |
  | 175M | 0.633 (one seed) |
  | P\* (280M) | 0.598 |
  | 380M | 0.611 |
  | 480M (X26c) | 0.602 |
  | C0 (P\* + 20M) | 0.596 |

  The X26c continuation and the C0 fork are no worse than P\*.
- **A modest trainability loss does exist, but it came BEFORE the plateau.** On T_net the end-to-end loss rises
  from 0.527 at 25M (one seed) and 0.474 fresh to about 0.60 by 175M. On the doc's AUC statistic that is 1.11× @25M,
  exactly on the NO-LOSS / UNCLEAR line: seed pairs give 1.101–1.125, so this target alone could not separate them.
- **The structure says the same, with one exception.**
  - The dormant share is flat from 100M: 0.059 → 0.057 at τ = 0.025.
  - The feature rank falls only slightly: latent_pi srank 493 → 458 (−7 %); vf flat at about 105.
  - The weight norm grows +5 % from 100M to 480M.
  - Gradient interference is flat: mean |cos| 0.044 → 0.039.
  - **The exception is the policy MLP's tanh units, which keep saturating across the plateau.** In layer 1 the
    fraction of activations with |a| > 0.99 goes 0.84 (100M) → 0.90 (P\*) → 0.92 (480M), and the units saturated
    on ≥ 99 % of rows go 6 % → 14 % → 18 %. In layer 2 the fraction goes 0.18 → 0.24 → 0.32. This is a known route
    to lost gradient. Here it does not show up as slower refitting (FINDING F3).
- **What it routes (§5):** T3 of the strategy tree takes the **NO-LOSS** branch: the capacity side, the N4 toy
  screen. The fresh-breadth funding that a plasticity LOSS would have bought (N5-S first, X53-N1 behind its σ gate)
  loses this probe as its reason. No remedy test is proposed, because the verdict is not LIKELY (§6).

## 1. What was measured

**Checkpoints.** All are read-only, loaded on CPU through `main.policy_spectrum.reader.load_checkpoint` (the
arch signature is verified; every one is `gen3_mon_tied_gain_v1`).

| label | file |
|---|---|
| fresh0, fresh1 | X26 @280M's architecture rebuilt through the production init path (SB3 `_build` + `restore_identity_init`), seeds 0 / 1; checked to differ from the checkpoint |
| x26_010M … x26_380M | `rb_x26_s2601/checkpoints/checkpoint_{10000089,25000027,100000037,175000114,280000170,380000195}_steps.zip` (280M = P\*) |
| x26c_480M | `rb_x26c_s2601/checkpoints/checkpoint_480000085_steps.zip` (X26 forked at 380M, at the production pin) |
| c0_s3501_300M | `rb_x35_c0_s3501/final_model.zip` (X35's plain-continuation control: P\* + 20M) |

**Banks.** All are seeded and deterministic (`plasticity_probe.py bank`).

| bank | rows | what |
|---|---:|---|
| `unfam` | 11,511 (8,509 train / 3,002 held-out, split 3 : 1 BY GAME, 180 / 60 games) | 240 self-play games on 400 ladder-corpus teams (0 shared with the 719-team pool the lineage trained on); both sides are a REFERENCE policy outside the compared set (X26 @50M, sampled at T = 1); p1's rows, its sampled action, the winner and both teams; p1 won 0.50 |
| `fixture` | 3,725 | `fixture_battles.play_rows(120, seed=20261026, n_teams=None)`: p1 uniform over the mask against the episode-seeded heuristic bot, pool teams |
| `golden` | 991 | the obs golden's core rows (`golden_obs_core.core_rows()`), with an all-legal mask (the taps read no mask) |

**Probes** (`plasticity_probe.py`; the probes and their literature in §7):
- **Structure** (`structure`), on each bank:
  - **τ-dormant share** (Sokar et al. 2023's score): 1,952 units, i.e. every ReLU output and every transformer FFN
    hidden unit outside the ride-along heads.
  - **The policy MLP's two tanh layers** (512 units each): their dormant share, their **saturated** share, and the
    mean fraction of rows with |a| > 0.99.
  - **Feature rank:** Kumar et al.'s **srank** (δ = 0.01, uncentred singular values) and the centred participation
    ratio, at latent_pi (512, the pointer head's context), vf (128), pi_features and value_pooled.
  - **Weight norms.**
  - **Per-example policy gradients** of log π(a|s) at the bank's action, on 64 fixed held-out rows: their norm, the
    mean off-diagonal cosine, and the Gram matrix's participation ratio.
- **The frozen-trunk head** (`heads`): a ZERO-initialised linear head on the cached [latent_pi ++ vf] (640 dims).
  Adam at 1e-3, batch 256, 1,000 steps, 2 seeds. Five targets:
  - win: p1 won the game;
  - species: whether the opponent's team holds each of the 8 commonest species, partly revealed in the row;
  - T_net;
  - T_lin: the capacity battery's random function of the raw row;
  - pstar: P\*'s greedy action, a masked cross-entropy.
- **The end-to-end refit** (`refit`): the trunk is UNFROZEN (extractor + policy MLP, 2.55M parameters) under a
  ZERO-initialised linear head, so every network starts at the same loss.
  - Adam at **1e-4**, the training scale (P\*'s own rate was 1.2e-4), batch 64, **400 steps**.
  - The held-out loss is read at 0, 25, 50, 100, 200, 300 and 400 steps on 768 fixed held-out rows.
  - The rate was chosen on a two-network calibration (fresh0 and P\* at 1e-4 and 3e-4); both rates rank the two
    alike (`results/refit_calib_*.json`).

**Cost.** Eight heavy-slot jobs (≈ 2.0 slot-hours in all, about 6 CPU-h on 3 agent cores at nice 19; one was a
26-s crash fixed and re-run), peak 3.9 GB. The slot queue added ≈ 1.6 h of waiting.

## 2. The criteria, and when each was fixed

1. **The registered bars** (the strategy doc §2, at `7574743d3`, written before any number here):
   - **The fit:** fit a FIXED new target, P\*'s greedy labels on a bank of held-out-team states, through a fresh
     linear head on the frozen trunk and then end-to-end.
   - **The statistic:** AUC is the area under the loss curve divided by the curve's own start (trapezoid over steps
     0..budget, divided by the budget; 1.0 means it never moved).
   - **LOSS** iff AUC(P\*) ≥ 1.25 × AUC(@25M) **and** the near-dormant share (τ = 0.025) of P\* ≥ 2 × @25M's.
   - **NO-LOSS** iff the AUC ratio is < 1.10.
   - **UNCLEAR** otherwise.
2. **This unit's own criteria** (fixed 2026-10-26 before any probe ran, kept in the work log):
   - **The read and its noise:** the primary read R is the end-to-end T_net loss at the budget. Its noise band s is
     the larger seed-0 / seed-1 gap at P\* and at fresh0, which came out at 0.047.
   - **LIKELY:** the plateau mean of R exceeds R(100M) by > 2s, and one structural metric degrades from 100M to
     480M. The structural thresholds:
     - dormant share +0.02 or more;
     - srank −10 % or more;
     - extractor weight norm +25 % or more;
     - mean |grad cos| +50 % or more;
     - tanh saturated share +0.05 or more.
   - **UNLIKELY:** the plateau R ≤ R(100M) + s on T_net and on the secondary refits, with no structural degradation.
   - **INCONCLUSIVE:** anything else.
3. **The verdict rule** (an addendum written BEFORE the pstar target ran):
   - The verdict is the registered class on the END-TO-END pstar fit: LOSS → LIKELY, NO-LOSS → UNLIKELY,
     UNCLEAR → INCONCLUSIVE.
   - It is downgraded to INCONCLUSIVE if the frozen fit's class, or criterion set 2, points to the OPPOSITE extreme.
   - **Disclosed (FINDING F1):** this unit's brief did not cite the bars, which arrived in a strategy-doc revision
     during the unit. The structural numbers and the T_net / T_lin / win reads had been seen before the addendum.
     The pstar numbers had not.

## 3. Results

### 3.1 Structure (unfam bank; the fixture and golden banks agree in direction, `results/structure.json`)

| net | dormant τ = 0 | τ = 0.025 | τ = 0.1 | srank latent_pi /512 | srank vf /128 | PR latent_pi | PR vf | tanh-1 sat. units / mean | tanh-2 sat. units / mean | ‖W‖ (excl. ride-along) | ‖W‖ policy MLP | grad norm mean / median | mean \|grad cos\| | grad Gram PR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh0 | 0.067 | 0.194 | 0.271 | 438 | 99 | 18.9 | 8.21 | 0.010 / 0.049 | 0.002 / 0.011 | 219.3 | 45.3 | 2.5 / 2.8 | 0.670 | 1.2 |
| fresh1 | 0.057 | 0.190 | 0.267 | 435 | 99 | 22.3 | 13.30 | 0.020 / 0.062 | 0.000 / 0.006 | 219.7 | 45.3 | 2.4 / 2.7 | 0.653 | 1.3 |
| 10M | 0.010 | 0.106 | 0.170 | 469 | 110 | 8.9 | 6.48 | 0.002 / 0.356 | 0.000 / 0.107 | 221.6 | 45.5 | 42.7 / 39.7 | 0.055 | 20.3 |
| 25M | 0.016 | 0.072 | 0.108 | 491 | 109 | 12.8 | 3.42 | 0.002 / 0.525 | 0.000 / 0.141 | 224.9 | 46.6 | 47.0 / 39.1 | 0.052 | 10.7 |
| 100M | 0.013 | 0.059 | 0.091 | 493 | 105 | 15.8 | 2.14 | 0.059 / 0.835 | 0.000 / 0.181 | 233.0 | 51.6 | 67.9 / 47.2 | 0.044 | 15.5 |
| 175M | 0.027 | 0.058 | 0.091 | 484 | 103 | 15.8 | 2.19 | 0.119 / 0.880 | 0.000 / 0.191 | 237.4 | 54.6 | 95.4 / 76.2 | 0.038 | 12.0 |
| **P\* 280M** | 0.025 | 0.056 | 0.088 | 476 | 105 | 16.3 | 2.25 | 0.135 / 0.898 | 0.002 / 0.236 | 241.1 | 57.2 | 160.5 / 81.7 | 0.037 | 5.6 |
| 380M | 0.028 | 0.055 | 0.087 | 462 | 105 | 13.8 | 2.38 | 0.164 / 0.912 | 0.004 / 0.295 | 242.7 | 58.3 | 163.0 / 125.8 | 0.039 | 13.6 |
| X26c 480M | 0.029 | 0.057 | 0.087 | 458 | 106 | 12.7 | 2.36 | 0.176 / 0.918 | 0.014 / 0.316 | 244.0 | 59.0 | 141.7 / 86.1 | 0.039 | 12.0 |
| C0 300M | 0.030 | 0.057 | 0.093 | 472 | 104 | 15.4 | 2.16 | 0.160 / 0.903 | 0.002 / 0.240 | 241.5 | 57.4 | 137.9 / 107.3 | 0.038 | 16.7 |

- **Dormancy falls with training and stays flat from 100M.** On the other banks at τ = 0.025: fixture 0.065 →
  0.058 and golden 0.057 → 0.068 from 100M to 480M. That replicates the X26 collapse battery's B10 on X26's own
  bank (P\* 0.057 there, 0.056 here).
- **srank of latent_pi falls 7–9 % from 100M to 480M on all three banks:** unfam 493 → 458, fixture 487 → 442,
  golden 460 → 419. All are below the 10 % threshold.
- **The vf features are about two-dimensional from 100M on** (PR ≈ 2.1–2.4), at srank ≈ 105.
- **Per-example gradient norms grow 2.4× from 100M to P\*, then stop.** This matches the TensorBoard grad-norm rise
  the collapse battery read; the gradients do not become more aligned.

### 3.2 End-to-end refits: held-out loss at step 400 (seed values) · normalised AUC (`results/refit_*.json`)

| net | T_net (MSE, start 1.023) | pstar (CE, start 1.864) | T_lin (MSE, start 0.998) | win (BCE, start 0.693) |
|---|---|---|---|---|
| fresh0 | 0.474 (0.484, 0.463) · 0.606 | 1.811 · 0.955 | 0.807 · 0.790 | 1.880 · 1.696 |
| fresh1 | 0.484 · 0.616 | 1.782 · 0.951 | — | 2.267 · 2.169 |
| 25M | 0.527 · 0.618 | 1.664 (1.655, 1.673) · 0.896 | — | 1.403 · 1.355 |
| 100M | 0.576 (0.572, 0.573, 0.582) · 0.666 | 1.683 · 0.890 | 0.810 · 0.812 | 1.251 · 1.272 |
| 175M | 0.633 · 0.703 | 1.669 · 0.893 | — | 0.908 · 1.039 |
| **P\* 280M** | 0.598 (0.580, 0.627, 0.588) · 0.685 | 1.631 (1.628, 1.633) · 0.883 | 0.787 · 0.799 | 1.511 (1.399, 1.623) · 1.373 |
| 380M | 0.611 · 0.693 | 1.638 · 0.878 | — | 1.374 · 1.298 |
| X26c 480M | 0.602 (0.610, 0.590, 0.605) · 0.686 | 1.659 · 0.885 | 0.766 · 0.794 | 1.354 · 1.277 |
| C0 300M | 0.596 · 0.682 | 1.659 · 0.883 | — | 1.320 · 1.263 |

**AUC relative to @25M (the registered statistic):**

| target | 100M | 175M | P\* | 380M | 480M | C0 | fresh0 |
|---|---|---|---|---|---|---|---|
| pstar | 0.993 | 0.996 | **0.986** (seed pairs 0.984–0.987) | 0.980 | 0.988 | 0.985 | 1.065 |
| T_net | 1.078 | 1.139 | **1.110** (seed pairs 1.101–1.125) | 1.123 | 1.111 | 1.105 | 0.981 |

### 3.3 Frozen-trunk heads (seed means; held-out loss at 300 steps / train loss at 1,000; AUC for the binary targets)

| net | win | species | T_net | T_lin | pstar |
|---|---|---|---|---|---|
| fresh0 | 0.531 / 0.529 · AUC 0.807 | 0.584 / 0.483 · 0.711 | 0.646 / 0.460 | 0.858 / 0.750 | 1.763 / 1.652 |
| 10M | 0.459 / 0.434 · 0.853 | 0.531 / 0.381 · **0.787** | 0.745 / 0.516 | 0.880 / 0.745 | 1.728 / 1.520 |
| 25M | 0.448 / 0.433 · 0.859 | 0.553 / 0.402 · 0.759 | 0.767 / 0.548 | 0.888 / 0.744 | 1.705 / 1.435 |
| 100M | 0.447 / 0.428 · 0.849 | 0.571 / 0.452 · 0.716 | 0.785 / 0.605 | 0.915 / 0.781 | 1.681 / 1.390 |
| 175M | 0.434 / 0.445 · 0.863 | 0.589 / 0.462 · 0.686 | 0.819 / 0.629 | 0.921 / 0.788 | 1.682 / 1.387 |
| **P\*** | 0.445 / 0.433 · 0.862 | 0.590 / 0.462 · **0.692** | 0.832 / 0.641 | 0.925 / 0.783 | 1.625 / 1.348 |
| 380M | 0.435 / 0.439 · 0.868 | 0.612 / 0.465 · 0.689 | 0.810 / 0.628 | 0.917 / 0.788 | 1.652 / 1.397 |
| X26c 480M | 0.429 / 0.438 · 0.872 | 0.591 / 0.463 · 0.694 | 0.813 / 0.631 | 0.897 / 0.779 | 1.673 / 1.408 |
| C0 | 0.419 / 0.434 · 0.879 | 0.587 / 0.465 · 0.698 | 0.831 / 0.638 | 0.919 / 0.784 | 1.665 / 1.366 |

The frozen-head pstar AUC ratio (train curve 0 → 1,000) is 0.947. The frozen-head T_net ratio is 1.119.

## 4. The verdict, applied

| criterion | value | class |
|---|---|---|
| registered, end-to-end pstar (PRIMARY) | AUC ratio 0.986 (all seed pairs 0.984–0.987); near-dormant ratio 0.78 | **NO-LOSS** |
| registered, frozen pstar | 0.947 | NO-LOSS |
| registered bars on T_net, end-to-end / frozen | 1.110 / 1.119 | UNCLEAR, inside the seed noise of the 1.10 line |
| this unit's T_net criteria | plateau R (P\* 0.598, 380M 0.611, 480M 0.602; mean 0.604) − R(100M) 0.576 = +0.028 < 2s = 0.094; the tanh saturated share +0.12 | neither LIKELY nor UNLIKELY |

Neither secondary points to LOSS, so the downgrade does not fire. **Verdict: PLASTICITY-LOSS-UNLIKELY.**

**How it reads, in words.**
- Measured on a target no checkpoint was trained toward, the trunk does fit new things more slowly than a young one.
  Each new target was fit in 400 steps with the trunk free to move.
- That slow-down was paid between 25M and 175M, while X26 was still climbing, not across the plateau.
- P\*, its 100M-longer continuation (480M) and its fork C0 refit alike. On P\*'s own policy labels the plateau
  checkpoints refit fastest of all.
- So the plateau forks did not start from a network losing plasticity as it trained on. Whatever kept their deltas
  small acts on a trunk that refits a new target about as fast at 480M as at 175M.

## 5. What it routes

- **The strategy tree, T3 d = NO-LOSS:** the capacity branch is next, the N4 toy screen (rank 10).
- **The END-OF-T1 escalation:** if PR-1 reads KILL, it fires at once, because this probe is not LOSS.
- **Fresh breadth (N5-S, X53-N1)** keeps its other reasons: N1-P's outcome, X53's σ gate. It no longer has plasticity
  as a reason.
- **The behavioural complement** (N1-P's argv forked from @100M against from P\*) was the UNCLEAR route. It is not
  needed by this verdict. It would still be the decisive test if a later read reopened the question, because this
  probe measures supervised refitting, not PPO under a moving target (FINDING F5).

## 6. Remedy test

None is proposed: the verdict is not LIKELY. For the record, the one degradation this probe saw is the policy MLP's
tanh saturation (§3.1). If a later read argued for plasticity, the cheapest targeted test would perturb only that
512 × 512 MLP: ReDo-style recycling of its saturated units, or a shrink-and-perturb of it alone. It would be one fork
in D1's frame against the banked C0, ≈ 12.3 GPU-h at look 1. Nothing here licenses it.

## 7. Literature

Venues and one-line claims were checked by web search on 2026-10-26. Any claim about a paper's detail beyond its
abstract is from memory and marked UNVERIFIED.

| paper | used for |
|---|---|
| Sokar, Agarwal, Castro, Evci, *The Dormant Neuron Phenomenon in Deep RL*, ICML 2023 (PMLR 202) | the τ-dormant score and ReDo (recycling dormant units). That the score is mean \|activation\| normalised by the layer mean is UNVERIFIED (from memory; the same definition as the collapse battery's `dormant.py`) |
| Kumar, Agarwal, Ghosh, Levine, *Implicit Under-Parameterization Inhibits Data-Efficient Deep RL*, ICLR 2021 | srank: rank collapse of value features under bootstrapping. That δ = 0.01 on the singular-value sum is the paper's setting is UNVERIFIED |
| Lyle, Rowland, Dabney, *Understanding and Preventing Capacity Loss in RL*, ICLR 2022 | capacity loss: networks trained on a sequence of targets lose the ability to fit new ones; the target-fitting probe (here T_net / T_lin) |
| Lyle, Zheng, Nikishin, Avila Pires, Pascanu, Dabney, *Understanding Plasticity in Neural Networks*, ICML 2023 | plasticity loss is tied to loss-landscape curvature and often occurs WITHOUT saturated units, so the tanh saturation here is not by itself evidence |
| Lyle, Zheng, Khetarpal, van Hasselt, Pascanu, Martens, Dabney, *Disentangling the Causes of Plasticity Loss in Neural Networks*, CoLLAs 2024 (PMLR 274) | several independent mechanisms; layer normalisation plus weight decay preserves plasticity (X26's trunk has LayerNorm and weight decay 1e-5: UNVERIFIED as to whether that dose matters) |
| Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton, *Loss of Plasticity in Deep Continual Learning*, Nature 632 (2024) | continual backprop: re-initialise a small fraction of little-used units |
| Juliani, Ash, *A Study of Plasticity Loss in On-Policy Deep RL*, NeurIPS 2024 | on-policy (PPO-like) plasticity loss is pervasive under domain shift, and "regenerative" methods mitigate it: the closest setting to ours (FINDING F5) |
| Nikishin et al., *The Primacy Bias in Deep RL*, ICML 2022 | partial resets of the last layers |
| Ash, Adams, *On Warm-Starting Neural Network Training*, NeurIPS 2020 | shrink-and-perturb: θ ← λθ + γθ_init |

## 8. FINDINGS (standing rule 7)

- **F1 (criteria timing).** The registered bars reached this unit through a revision of the strategy doc made during
  the unit; the brief named an earlier version.
  - The pstar target was added, and the verdict rule written, after the structure, T_net, T_lin and win reads had
    been seen, but before any pstar number existed (§2).
  - Under the bars, the dormant leg alone already made LOSS unreachable (0.78 against ≥ 2).
- **F2 (the registered target is weakly sensitive).** pstar's held-out cross-entropy moves only 1.86 → 1.63 in 400
  steps from every checkpoint.
  - A linear head on the state latent cannot see the pointer head's per-action move / switch features, so most of
    P\*'s choice is out of the probe's reach.
  - The target is also P\*'s OWN function, which favours P\* by construction. Its NO-LOSS is therefore weak evidence
    alone. The architecture-neutral T_net read carries the trend, and it is the reason the verdict is UNLIKELY
    rather than "no effect".
- **F3 (tanh saturation keeps rising across the plateau).** The policy MLP's layer 1 has 92 % of its activations at
  |a| > 0.99 by 480M, and 18 % of its units saturate on ≥ 99 % of rows.
  - This is the one structural metric still degrading after 100M.
  - It did not slow any refit measured here, and the literature does not equate it with plasticity loss.
  - It belongs on the collapse battery as a watched metric. Not done here (rule 9).
- **F4 (the win refit is not a plasticity read).** The label is game-level, and only 180 training games exist. Every
  network overfits within 50–100 steps, and the held-out BCE then explodes, so the win AUCs measure memorisation.
  They are reported, not used.
- **F5 (scope).** This is supervised refitting on CPU, with fixed Adam, on 11,511 states.
  - It is not PPO under a moving target, and it does not measure how a fork's breadth gain scales with the fork's
    starting checkpoint.
  - That would take the strategy doc's behavioural complement (≈ 8–16 GPU-h).
  - The literature's on-policy result (Juliani and Ash 2024) is that plasticity loss under DOMAIN SHIFT is common,
    and a new team distribution is a domain shift. This probe does not exclude that form.
- **F6 (single-seed cells).** 25M on T_net, 175M, 380M, C0 and T_lin have one seed each. The T_net seed band at P\*
  is 0.047 (0.580–0.627). Cells within that band of each other are not ordered.
- **F7 (an off-axis observation).** Frozen-feature decodability of the opponent's team composition FALLS with
  training: species AUC 0.787 at 10M → 0.69 from 175M on. It does not recover on the plateau. This is not a
  plasticity criterion; it is recorded for the piloting / FACING questions, unread.
- **F8 (reference policy).** The unfam bank's states come from X26 @50M sampling. They are nearer to the early
  checkpoints' own distribution than to the plateau's. Every comparison is between plateau checkpoints, which share
  that handicap.
- **F9 (the work).** No GPU, no game against an outside opponent, `models/` read-only. The 282 MB feature cache and
  the bank stay out of the repo, in `/tmp`. They regenerate deterministically from `plasticity_probe.py bank` /
  `structure`.

## 9. Files and reproduction

- `plasticity_probe.py`: the frozen measurement script, with the subcommands `bank`, `tnet`, `structure`, `heads`
  and `refit`. Its docstring holds every definition.
- `summarize.py`: the tables and the bar arithmetic, from `results/`.
- `results/`: `structure.json`, `heads.json`, `refit_{t_net,t_lin,win,pstar}.json`, `refit_calib_lr{1e-4,3e-4}.json`
  and `bank_meta.json`.

```bash
# in the worktree: export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""; every step under nice -n 19 taskset -c 5-7,13-15
# and scripts/ops/mem_cap.sh (bank: --max-s 600; the rest in <= 45-min chunks)
python plasticity_probe.py bank --out /tmp/plast/bank
python plasticity_probe.py tnet --bank /tmp/plast/bank --spec fresh:12345:<X26 @280M zip>
python plasticity_probe.py structure --bank /tmp/plast/bank --out results/structure.json --feats /tmp/plast/feats --net <label>=<zip | fresh:SEED:zip> ...
python plasticity_probe.py heads --bank /tmp/plast/bank --feats /tmp/plast/feats --out results/heads.json
python plasticity_probe.py refit --bank /tmp/plast/bank --out results/refit_<t>.json --lr 1e-4 --target <t> [--seed N ...] --net ...
python summarize.py results
```
