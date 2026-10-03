# Design — the learner recipe, re-grounded knob by knob

**Status: REVIEW + PROPOSAL (2026-09-29).** Nothing here is built or adopted unless its Decision-record
row says so. Rows marked **PROPOSED** are recommendations for the owner and orchestrator to accept or
reject. Experiments this doc proposes go to [`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md)
once they are accepted; build tasks go to [`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md).

🚨 **ALWAYS-CURRENT (owner, 2026-09-27).** This doc is the spec of record for the TRAINING RECIPE: every
PPO and optimizer knob, the rollout and update shapes, and the opponent mix. A change that differs from
it updates this doc in the same commit, saying what changed and why. [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
still owns the MODEL. [`program_rust_core.md`](program_rust_core.md) §2 M5 owns the INFRASTRUCTURE the
recipe runs on, including order constraint 5 (the SIZING study). Per-flag mechanics stay in
`designs/training/<topic>.md`.

**Where the numbers come from.**
- **The live recipe** is read from `models/ai_v14_07_g0p_k2/metadata.json` (`original_command`,
  `cli_args`, the `dose` block) and `model_config.json`. That run is K2, also called G0′ (G0-prime),
  the plateau parent. Its sibling `ai_v14_08_g0p_k3` carries the same recipe. Both files were read,
  never written.
- **Defaults** are read from `src/main/train/parser/` and `src/main/train/config.py`.
- **Provenance** comes from `git log -S` and blame.
- **`L<line>`** is a ledger entry as [`../research_state/ledger_index.md`](../research_state/ledger_index.md)
  lists it.

---

## 0. Why this doc exists

The owner, 2026-09-29: *"when a lot of this was written it was my first week — needed just to run any
testing. I've learned a lot … there are much more sophisticated, grounded approaches out there, and I
want us to always bias towards those."*

Most recipe values were set in May 2026, in the project's first weeks. Many are SB3 (Stable-Baselines3)
defaults nobody chose. Others were set on reasoning alone, with no measurement. Since then the ledger
has measured several of them, and M5 changes the shape of the problem: the Rust environment core makes
environment samples cheap, and the learner becomes the scarce resource.

This doc applies three rules:

1. **Start from the grounded literature.** Every knob names its published default and the paper that
   supports it.
2. **The ledger wins for OUR setting.** Where our measurement contradicts a literature default, the
   ledger wins, and the doc says why the two settings differ (§4).
3. **One lever per arm, judged by the standing meters.** A change that could move learning either
   lands as a SAFE DEFAULT (no learning effect, or a bit-identity or equivalence gate) or runs as a
   one-lever ARM (§5).

**The meters every arm reads** (all defined in `research_state/measurements/learner_battery_2026-09-26/`
and `new_lineage_2026-09-26/`):
- **Untaught-8 non-inferiority.** The win rate on the 8 untaught pool teams against the fixed opponent
  `untaught_meter_opponent_v14`, 600 games per team. An arm passes when the lower end of its CI is above
  the −3.69 pp replicate floor.
- **SmallRL guard.** Metamon's `SmallRL` agent, greedy vs greedy, 1,200 games. It fires only on a gross
  loss, beyond its 11.0 pp floor.
- **Stalls.** The rate of `[STALL LOGGED]` lines. The guard fires when an arm's rate exceeds 1.5× the
  control's.
- **The Lane S spectrum.** M5 Lane S's policy-spectrum instrument, read on its fixed turn bank
  ([`program_rust_core.md`](program_rust_core.md) Lane S). It reports rank-1 mass, mass on dominated
  actions, and STARVATION, the share of turns where a near-best action gets less than 1 %.
- **KL and clip descriptors.** `train/approx_kl`, `clip_fraction` and `train/dose_rate`.

---

## 1. The one-page summary

**Class key:**
- **KEEP**: stays as it is.
- **SAFE**: a safe default that lands with no arm. It either has no learning effect or passes an
  equivalence gate.
- **ADAPTIVE**: becomes a controller, adopted through an arm.
- **ARM**: needs a one-lever registered arm.
- **WAITS**: gated on the architecture being stable, the Q head, or search.

| # | knob | live value (K2) | grounded literature | recommendation | class |
|---|---|---|---|---|---|
| 1 | `n_envs` (N) | 256 | Rudin 2021: 2k–4k envs at ~100k-sample batches; performance drops when the per-env horizon gets too short | Decided by the registered SIZING study, not picked. **Read 2026-10-02 (§3.1): N\* = 256** — throughput rule on the serial rate; no learning loss detected beyond the bar (not equivalence; n = 1 flagged); fits at steady state on `07eebe13` | KEEP (study) |
| 2 | rollout size N × `n_steps` | 256 × 384 = 98,304, fixed | Andrychowicz 2021: samples per iteration matter a lot; McCandlish 2018: batch ∝ √B_noise | **Size it dynamically from the policy noise scale**, inside a declared maximum. First land Lane G's complete-game collector (program_rust_core.md order constraint 6), because fixed short windows drop critic rows (§3.2); with it, the controller sets the SAMPLES-PER-UPDATE trigger rather than n_steps | ADAPTIVE (after a SAFE prerequisite) |
| 3 | micro-batch | 2048 | — (a memory and compile lever) | Keep it fixed: one compiled learner graph | KEEP |
| 4 | accumulation K / effective batch | 32 → 65,536, **plus a half-size second step every epoch** | Exact accumulation is standard; a ragged step is not | Rollout = a whole multiple of micro × K (no ragged step), then move K with `--adaptive-batch policy` | SAFE, then ADAPTIVE |
| 5 | noise-scale estimator | two-point, raw gradient, epoch 0, first group, EMA 0.99 | McCandlish 2018 (raw B_simple predicted B_crit for Adam) | Add a split-half-by-GAME estimate: it is unbiased under within-game correlation and exposes the rollout's own noise | SAFE (telemetry) |
| 6 | `n_epochs` | 10 in the argv; **5 adopted** for generalists (L21451) | PPG 2021: 1 policy epoch near-optimal; OpenAI Five: reuse ~1; Rudin: 5 | Keep 5 for forks. A FRESH run keeps 10: 5 epochs at a KL-controlled 3e-4 lost 10.6 pp of untaught at matched samples (SIZING, 2026-10-02, §3.5). Test E2 at matched dose. Decoupled policy/value epochs wait for the Q head | KEEP → ARM |
| 7 | `clip_range` | 0.15, no schedule; clip fraction 0.065–0.077 | Andrychowicz: 0.25 start; Huang: 0.1–0.2; Hilton 2022: decouple the proximal policy | Keep 0.15, no schedule. PPO-EWMA if the batch starts moving | KEEP; WAITS (EWMA) |
| 8 | KL → LR controller | target 0.01, ×1.2 per rollout, FROZEN in this lineage | Rudin/rsl_rl: desired KL 0.01, ×1.5 | Make it **guard-only** (it may LOWER the LR, never raise it above the declared rate) | SAFE |
| 9 | learning rate | frozen 2.8e-5; **5.6e-5 with E5** | OpenAI Five 5e-5 → 5e-6; DeepNash 5e-5; Andrychowicz 3e-4 (small nets) | Keep frozen at the declared dose. Hold LR fixed when K moves, and record the dose | KEEP |
| 10 | DOSE bookkeeping | `lr·E/B_eff` | Hilton 2022: Adam needs a √-batch LR rule | **Fix the step count** (the live shape takes 4/3 the recorded steps); keep dose as the lineage's control variable | SAFE (code task) |
| 11 | `ent_coef` | 0.05 | 0 to 0.01 (Huang; Andrychowicz: no regularizer helps); MMD 2022: an entropy bonus IS a KL to a UNIFORM magnet | Replace the uniform magnet with a trailing magnet (R-NaD/MMD), measured by Lane S. Owner: no direct adoption until the architecture is stable | WAITS → ARM |
| 12 | γ (discount) | 1.0, terminal-only reward | AlphaZero/MuZero board games: γ = 1; V = P(win) | Keep | KEEP |
| 13 | policy GAE λ | 0.80 | 0.95 (OpenAI Five, Rudin); 0.9 (Andrychowicz) | Keep 0.80; **L95 lost −10.1 pp** (L21451). Sweep {0.6, 0.7, 0.8, 0.9} in the M5 era; state-dependent λ is PROPOSED | KEEP → ARM |
| 14 | critic: λ = 1 BCE (cross-entropy on the win head), `vf_coef` 0.5, no value clip | same | Andrychowicz and Huang: no value clipping | Keep; λ < 1 never replicated (L17453, L17578); `vf_coef` 1.5 harmful (L18010) | KEEP |
| 15 | advantage normalization | per MICRO-batch (2048) | per minibatch (Huang #7, minimal effect); Hilton: its sample size should scale | Normalize once per ROLLOUT, which makes accumulation exact and K-independent | SAFE (equivalence gate) |
| 16 | aux-loss weights into the shared trunk | eight belief/intent terms at 0.05, `belief_grad_mode shaping` | PPG 2021: policy/value/aux interference; Andrychowicz: separate networks | No blind sweep. Route by the Q-head's gradient-cosine guards; a PPG auxiliary phase waits for the Q head | WAITS |
| 17 | grad-norm clip | 0.5 (SB3), median norm 0.42 | 0.5 (Huang #8); small effect (Andrychowicz) | Keep. Log the share of steps it binds | SAFE (telemetry) |
| 18 | optimizer | AdamW β = (0.9, 0.999), ε = 1e-5, wd 1e-5 on every param | ε 1e-5 (Huang #1); β1 0.9 (Andrychowicz) | Keep. Weight decay is inert at this LR (lr·wd ≈ 5e-10 per step) | KEEP |
| 19 | observation normalization | none | "always" (Andrychowicz) | Keep none: running statistics break the declared lifecycle and search/inference parity. Run a one-time offline scale audit | KEEP + SAFE audit |
| 20 | init | pointer scorers zero-init ⇒ uniform first policy | small last policy layer (Andrychowicz) | Keep | KEEP |
| 21 | self-play opponent temperature | 1.0 | OpenAI Five/AlphaStar sample stochastic policies; our model is +23 pp stronger greedy (L21447) | Arm X23(b): opponent T 0.5. This is exploration in the DATA | ARM |
| 22 | opponent pool | 20 snapshots, newest weighted 1.3×; stable share 0.2; bots ≥ 10 %; PFSP off | OpenAI Five 80/20 with quality scores; AlphaStar PFSP | Keep. PFSP only over a diverse, anchored pool (L01151); the arm-C read was never done (L15610) | KEEP; ARM later |
| 23 | eval regime | greedy vs greedy sentinels | — | Keep (L13654, L21447) | KEEP |
| 24 | recipe defaults vs the argv | mirrored: `recipe.fresh` (N0) + `recipe.fork` (E5) (§3.22) | — | **BUILT** (K10(a)): `--arch production` applies `recipe.fresh`, `checkargs` diffs and refuses untyped drift | SAFE (BUILT) |

**Headline.**
- **Changes that land as safe defaults:** a divisible rollout shape and a correct step count (4, 10);
  per-rollout advantage normalization (15); a guard-only KL controller (8); a recipe mirror (24); and
  three telemetry additions (5, 17, the critic masked-row share). The delayed-label buffer is the
  prerequisite build.
- **Becomes adaptive:** the effective batch through K, then the rollout length (2, 4).
- **Needs a one-lever arm:** adaptive batch, dynamic rollout, E2, the λ sweep, and opponent
  temperature, in that order (§5).
- **Waits:** the magnet in place of entropy, PPO-EWMA and PFSP wait for the architecture to be stable.
  The PPG phase, state-dependent λ and search targets wait for the Q head and search.
- **Stays:** everything else, most of it now with a citation it never had.

---

## 2. The batch and rollout question: what "dynamic" should mean, and where the owner's asks conflict

**The owner's asks (2026-09-29):**
1. Make rollout length and batch DYNAMIC.
2. Don't waste samples early.
3. Never let the update be noise-dominated.
4. Aim long-term at about 1.1× the critical batch.

### 2.1 The arithmetic (McCandlish et al. 2018)

McCandlish et al. model the cost of training at batch B through the gradient noise scale
B_noise ≈ B_simple = tr(Σ)/‖G‖². Here tr(Σ) is the total per-example gradient variance and ‖G‖² is the
squared length of the true gradient. The two costs are:

- optimizer steps: **S = S_min (1 + B_noise/B)**
- samples processed: **E = E_min (1 + B/B_noise)**

So (S/S_min − 1)(E/E_min − 1) = 1 (their eq. 2.11), and the critical batch is B_crit ≈ B_noise.

| B / B_noise | noise ratio (B_noise/B, our `noise_scale_ratio`) | steps vs minimum | samples vs minimum | regime |
|---|---|---|---|---|
| 0.5 | 2.0 | 3.0× | 1.5× | noise-dominated; sample-frugal |
| 1.0 | 1.0 | 2.0× | 2.0× | the knee |
| **1.1** (the owner's target) | **0.91** | **1.91×** | **2.1×** | just past the knee |
| 2.0 | 0.5 | 1.5× | 3.0× | step-frugal; sample-wasteful |

**The pushback.** Asks 2 and 3 pull in opposite directions at every point on this curve.
- *Not wasting samples* means B ≤ B_noise.
- *Never noise-dominated* means B ≥ B_noise.

The only batch that satisfies both is the knee, where each resource costs twice its minimum. So the
owner's 1.1× is a consistent choice: it sits just on the "not noise-dominated" side. But it spends about
2× the minimum samples, and nothing can make it spend fewer while also staying out of the noise.

Two grounded facts sharpen this:

1. **The Pareto-optimal ADAPTIVE schedule is not a fixed ratio.** McCandlish App. D, eq. D.3: when
   samples can be traded between phases of training, the optimal batch is **B(s) ∝ √B_noise(s)**, not
   ∝ B_noise(s). A constant-ratio controller (which is what `--adaptive-batch` is) is therefore not the
   optimum. It is close, however, because the gain from adapting depends on how much √B_noise varies
   over the run (their γ, eq. D.5), and B_noise must vary "quite a bit" before the difference matters.
   **PROPOSED: keep the constant-ratio controller with the existing wide band.** Its target is a ratio
   of 1.0 and its band is ×/÷ 2, so it holds B between 0.5× and 2× B_noise. Record the √ schedule as
   the refinement, to adopt only if the sizing study measures B_noise moving by more than ~10× over a
   run.
2. **Which resource is scarce decides the target, and M5 changes the answer.** On the process-per-env
   path, samples were the scarce resource. On the Rust core, env samples become cheap and the learner's
   wall-clock becomes scarce, which argues for B ≥ B_noise (a ratio ≤ 1).
   **PROPOSED: target a ratio of 0.9–1.0 (the owner's 1.1×) for the M5 era.** Revisit if the sizing
   study shows env throughput is still the bottleneck.

### 2.2 The PPO-specific point the owner's framing misses: two noises, two levers

McCandlish's model assumes fresh samples every step. PPO instead takes E epochs over ONE rollout of D
samples. So an update carries two separate noises:

- **Minibatch noise.** This is the gap between a minibatch gradient and the ROLLOUT's gradient G_D.
  K controls it, and reshuffling across epochs averages it further.
- **Rollout noise.** This is the gap between G_D and the true gradient. Every epoch re-fits it, so
  epoch reuse cannot average it away. Only a bigger rollout reduces it.

The update is therefore **not noise-dominated only if D_eff ≳ B_noise**, where D_eff = D / c. The
factor **c** is the rollout's *design effect*: how much within-game correlation inflates the variance.
Decisions of one game share an outcome and, at λ = 0.8, advantages that overlap over ~5 steps, so
c > 1. How much above 1 is **UNMEASURED**.

**This splits the owner's "dynamic" ask into TWO controllers:**

| controller | moves | holds | signal |
|---|---|---|---|
| **K (the batch)** | K accumulated micro-batches (micro-batch shape fixed: one graph) | the per-STEP noise ratio B_noise/(micro·K) in band | `noise_scale_ratio_policy` (exists: `--adaptive-batch policy`) |
| **n_steps (the rollout)** | the rollout length, inside a DECLARED maximum (buffers allocated at the max; M5's declared lifecycle) | D_eff ≥ k·B_noise (update SNR), and D ≥ m·micro·K_max (at least m optimizer steps per epoch at the largest batch) | the split-half estimate below, plus the design effect c |

**PROPOSED rollout rule** (the constants come from the sizing study):

    n_steps(t) = clamp( ceil( max(m·micro·K(t), k·c·B_noise(t)) / N ), n_min, n_max )
    (2026-09-29, owner: under Lane G's complete-game collector this quantity is the SAMPLES-PER-UPDATE trigger D(t) = N·n_steps(t); n_steps itself is retired as a knob — program_rust_core.md order constraint 6.)

Suggested starting constants are m = 4 and k = 2. Andrychowicz and Rudin both use about 4 minibatches
per epoch.

- **Early** in training B_noise is small, so rollouts are short and updates frequent. That serves
  "don't waste samples early".
- **Later**, as B_noise grows (McCandlish: it rises as the loss falls), rollouts lengthen. That serves
  "never noise-dominated".

**The floor n_min is set by the critic, not by PPO** (§3.2), until the delayed-label buffer lands.

### 2.3 Is the estimator robust enough to steer by?

The live estimator (`instrumented_ppo/noise_scale.py`, detail in
[`../training/step_size_and_batch.md`](../training/step_size_and_batch.md)) works as follows:
- It takes two gradient norms: one micro-batch (2,048 rows) and the first accumulated group (65,536).
- It uses epoch 0 only, so the data is fresh.
- The gradients are raw (not Adam-preconditioned) and taken before clipping.
- Per term: the policy term is what steers.
- It is smoothed by an EMA with decay 0.99, after 20 warm-up folds.

| concern | verdict |
|---|---|
| **Adam** | McCandlish tried Adam-preconditioned gradients and found "mixed results". Raw B_simple predicted B_crit "very accurately" under Adam on their SVHN test. **The raw estimator is the grounded choice**; that it transfers to our setting is **UNVERIFIED**. |
| **Epoch reuse** | Measured on epoch 0 only, which is correct: after one pass the gradient on reused data is no longer an unbiased sample. Keep. |
| **Sampling without replacement from a finite rollout** | The finite-population terms cancel in the two-point solve. With iid rows, the ‖G‖² estimate is unbiased for the POPULATION gradient. (Derivation: E‖Ĝ_b‖² = ‖G_D‖² + s(1/b − 1/D). Solving the pair gives tr(Σ) = s exactly and ‖G‖² = ‖G_D‖² − s/D = ‖G_pop‖².) No correction is needed. |
| **Within-game correlation** | **The real bias.** When rows are clustered by game, ‖G_D‖² = ‖G_pop‖² + c·s/D, so the ‖G‖² estimate comes out too large by (c−1)·s/D and **B_simple reads too LOW**, i.e. the run looks over-batched. At the live shape with c = 3, the estimate would be about half the truth. The same c also makes the rollout itself noisier (§2.2), so **the live `noise_scale_ratio_policy` of about 0.81 may in truth be noise-limited.** **UNVERIFIED** until c is measured. The 0.81 is a TensorBoard read of 2026-09-29 and is not in the ledger. `program_rust_core.md`'s N0 figure of 0.44 is also not in the ledger. |
| **Aux deflation** | Already solved: the per-term policy scale is read, not the total (L08486). |
| **Responsiveness** | An EMA at 0.99 with one fold per rollout has a ~100-rollout time constant (~10M samples at today's shape), plus a 20-fold warm-up (~2M). That is too slow to "not waste samples early". |

**PROPOSED telemetry (SAFE, no learning effect):**
1. **A split-half-by-GAME estimate.** Split epoch 0's rows into two halves by GAME. Games are close to
   independent, so the dot product G_A·G_B is an unbiased estimate of ‖G_pop‖² even under within-game
   correlation. Its ratio to the two-point ‖G‖² estimate gives the design effect c.
2. **Fold more than one point per rollout.** Use every epoch-0 group, not just the first; the group
   norms are already computed for clipping. More points per rollout allow a faster EMA (about 0.9)
   without sign flips.

These two land before the adaptive-batch arm, so that the arm steers on a measured signal.

### 2.4 The Adam LR-scaling question when B moves

For SGD, keeping training equivalent across batch sizes means scaling the LR linearly with B. For Adam
the equivalent rule is the **square-root rule**: LR ∝ √B (Malladi et al. 2022; Hilton, Cobbe &
Schulman 2022 §3.2). Our DOSE, `lr·E/B_eff`, is the linear, large-batch form. It predicted fold
collateral in OUR data (L09065), but the ledger only ever varied LR at a FIXED batch shape. **Whether
dose or √-scaling is the right invariant when B moves is UNTESTED.**

**PROPOSED:** in the adaptive-batch arm, hold LR fixed when K moves, and record `train/dose_rate`
(which then moves as 1/B). The guard-only KL controller (knob 8) catches overshoot. Whether the arm
matches its control at the SAME dose is its secondary read.

---

## 3. Per-knob sections

### 3.1 `n_envs` — 256, decided by the SIZING study (2026-10-02)

- **Provenance.** The parser default is 32 (`operational.py:34`, `25fdb04f`, 2026-05-11, "optimize
  default hyperparameters for GPU"). The value 48 is typed per run in the argv.
- **Literature.**
  - Andrychowicz et al. 2021 §3.5: more parallel envs at a fixed batch "decreases sharply" on some
    tasks, because of shorter experience chunks and earlier bootstrapping.
  - Rudin et al. 2021 (legged_gym) found the same at a fixed batch and chose 2,048–4,096 envs at
    ~100k–200k samples. Their production shape is **4,096 × 24 = 98,304**, the same rollout size as ours.
- **Ledger.** No learning evidence on N. L21421 measured transport throughput only.
- **Recommendation.** KEEP the registered SIZING study (`program_rust_core.md` order constraint 5):
  sweep N at a fixed rollout size, and read samples/s and learning per sample. **Add to the study**:
  the critic masked-row share (§3.2) and the design effect c (§2.3), both as functions of N.
- **Outcome (the SIZING study, read 2026-10-02;** `designs/research_state/measurements/m5_sizing/`**):
  `recipe.sizing.n_envs` = 256, `n_steps` = 384.**
  - **Throughput.** The registered end-to-end rule, applied to the SERIAL rate the training path can
    achieve (orchestrator D-10), picks 256: E2E 1,945 against a maximum of 1,986. Past 256, more envs
    barely move E2E, because the UPDATE dominates: ~41 s at 10 epochs, against ~10 s of collection per
    98,304 rows. At 10 epochs, 256 envs shorten the update cycle from 55.2 s to 51.6 s (~7 %).
  - **Learning guard** (fresh 8M arms, fp32, matched samples):
    - U(256) − U(48) = −2.71 pp [−4.50, −0.94]. The SmallRL guard (G-A) reads +0.2 pp [−3.8, +4.1].
    - So **no loss is detected beyond the −3.69 bar. That is not equivalence.**
    - The seed replicate at 48 moved U by 4.90 pp, above the bar, so the read carries "run floor exceeds
      bar — n = 1 is not decisive".
  - **Memory.**
    - At 256 the arms' CUDA reserved memory CLIMBED from 8,012 to 9,674 MiB over the first ~33 updates.
      At steady state that left 341.5 MiB of headroom, below D-6's 512.
    - It was a bug: a CUDA stream per update in the staged batch. `07eebe13` FIXED it. Its acceptance run
      at 256 with the X26 heads stayed flat at 7,798 MiB, with steady-state D-6 headroom of 2,218 MiB.
    - The arms' learning is unaffected: the numerics are identical and the K9 golden is unchanged.
  - **Staleness at 256:** 5.5 % of rows are one version old (predicted 6.1 %).
  - **Future option:** 1024 envs with the overlapped collector. It needs that collector on the training
    path, a staleness correction and memory.

### 3.2 `n_steps` and the rollout size — 2048 fixed; make it dynamic, AFTER the critic stops dropping rows

- **Provenance.** 2048 is SB3's default (`8885aeee`, 2026-05-10).
- **Literature.**
  - Andrychowicz: the number of transitions per iteration influences performance "quite significantly".
  - OpenAI Five sends 256-step segments and targets a staleness of 0–1 parameter versions (Berner et
    al. 2019 §4.4). Staleness of about 8 versions caused "significant slowdowns".
- **🚨 HAZARD found by this review.** Under the win-prob critic (the only critic) at λ = 1, rows of an episode still
  running when the buffer fills get `win_mask = 0`. They are excluded from the critic loss, and no
  label is back-filled (`win_prob_callback.py:22-24`; `critic_and_value_losses.md` "The BUFFER
  BOUNDARY").
  - **Size.** Per env column, the excluded rows are the unfinished tail: to first order about L/2
    rows, where L is the game length in decisions, plus a length-variance term. That is roughly
    **L/(2·n_steps)** of all rows.
    - At n_steps = 2048: under 1.1 %.
    - At the SIZING study's n_steps = 64 (N = 2048): **about 18–35 %** for L = 23–45.
  - **Bias.** The dropped rows are biased toward LONG games' early states. That is the "biases toward
    short games" failure `design_q_head.md` already rejected, when it chose a **DELAYED-LABEL BUFFER**
    that back-fills outcomes so every row is used once.
  - The policy advantage (λ = 0.8) bootstraps from V at the window edge. That is standard and
    harmless: its horizon is 1/(1 − 0.8) = 5 steps.
- **Recommendation.**
  1. Land the delayed-label buffer (the Q-head doc's decision) **before** any rollout shorter than
     about 512 steps per env.
  2. Log the masked-row share. At λ = 1 no `win_prob/lambda_*` tag exists, so this needs a new tag.
  3. Then run the dynamic-rollout arm (§2.2).
  - Until the buffer lands, **n_min = 512**. The masked share there is about 2–4 %.
  - **On the Rust core this is SUBSUMED.** The complete-game collector labels every row with its own
    game's outcome, so `win_mask` = 1 everywhere (`designs/training/rust_collector.md`, K10(b)), and
    production's n_steps 384 drops no critic row.
- **The n_steps MAXIMUM (the SIZING study, 2026-10-02, REGISTRATION §6).** This is the adaptive-batch
  arm's ceiling, NOT applied: production's `n_steps` stays 384.
  - D_max = 4 optimizer steps per epoch at K_max = 4 · 2,048 · 32 = **262,144 rows**, so
    **n_steps_max = 1,024 at N = 256**.
  - The host memory at D_max is an ANALYTIC estimate: +1.8–3.6 GB over a 16 GB peak, which fits the
    56 GB cap. It is UNVERIFIED by a run. With `--device-batch staged` the device holds one
    micro-batch, so D does not move GPU memory.

### 3.3 Micro-batch 2048, accumulation K 32 — and the ragged step

- **Provenance.**
  - The micro-batch of 2048 came from an out-of-memory correction for the full architecture (ledger
    2026-09-06, ops).
  - Accumulation: `bb6b6835`, 2026-06-16. It gives the exact large-batch gradient at the memory cost
    of one micro-batch.
- **🚨 FINDING.** 98,304 / 2,048 = 48 micro-batches, and K = 32. So each epoch takes one step on 65,536
  rows **and a second, full-size step on the remaining 32,768** (`ppo.py:1430-1448`: the partial group
  is rescaled to the mean, then stepped at the same LR).
  - One step in two runs at half the batch, which doubles that step's noise ratio.
  - The rollout gets **20 optimizer steps (at 10 epochs), not the 15** the DOSE block records.
    `dose.py:12` counts n_epochs/(batch·K) as if steps were fractional, so the dose undercounts by
    **4/3** at this shape.
  - Arms compared at ONE shape are unaffected, because the factor is common to both. **Doses across
    DIFFERENT shapes in the ledger's history carry this error.**
  - `step_size_and_batch.md`'s claim that production configurations divide cleanly was false for this
    shape. It is corrected in the same commit as this doc.
- **Recommendation (SAFE).**
  - Declare rollout shapes that are a whole multiple of micro × K_max. For example, N = 2048 × 64 =
    131,072 = 64 micro-batches, and the controller's K choices are the powers of 2 up to 64.
  - Count the dose in actual steps: ceil(D/B_eff) per epoch. This is a code task (TASK_BACKLOG); the
    doc only records it.

### 3.4 The adaptive-batch controller (`--adaptive-batch policy`, `gen3_adaptive_batch_v1`)

- **Provenance.** `f60242ba`, 2026-09-01. It has never run in an arm (the ledger has no evidence).
- **Design as built.**
  - It moves K only, never the batch shape. So there is one compiled graph and no VRAM cost.
  - It doubles or halves K when the smoothed policy ratio leaves [target/band, target·band], with
    defaults of 1.0 and 2.0.
  - Its floor is K = 2, so the estimator is never blinded.
  - It is separated by construction from the KL-LR loop's timescale.
- **Literature.** McCandlish App. D: an adaptive batch is sound and "manag[es] the proportion of
  gradient noise".
- **Recommendation.** ADAPTIVE through an arm (§5, A1), after the §2.3 telemetry. Use a target of
  1.0 and a band of 2.0 (the existing defaults: they cover the owner's 0.91 and never go
  noise-dominated by more than 2×). Keep `max_accum` under the rollout (§2.2, m = 4).
- **Bounds (the SIZING study, 2026-10-02, REGISTRATION §6; not adopted, arm A1 adopts).**
  - **K_min = 2.**
  - **K_max = 32**: the smallest power of two with 2,048 · K ≥ 1.25 × the largest measured policy
    B_noise (49.4k at K2 @91M). The fresh 8M arms read 2.2k–13.6k.
  - Every B_noise may read LOW by up to the unmeasured design effect c (§2.3).

### 3.5 `n_epochs` — 10 → 5 (adopted 2026-09-29)

- **Provenance.** The parser default is 5 (`41502618`, 2026-05-26, "to match the 400M run"). Where 10
  started in the argv is **UNTRACED**. L08418 flagged "10 epochs REVISIT" and it was never tested
  until the battery.
- **Ledger.** L21451: **E5 ADOPT.** Five epochs at twice the LR (5.6e-5, the same dose) saves
  13.59 % [13.38, 13.71] of GPU time per step. Untaught Δ +0.75 pp [−1.21, +2.65] (non-inferior),
  SmallRL −1.33 [−5.18, +2.52], stall ratio 1.10. It is trained on the miscompiled learner, so its
  transfer to the fixed learner is UNVERIFIED (L21435).
- **Literature.**
  - PPG (Cobbe et al. 2021 §3.2): "training with a single policy epoch is almost always optimal or
    near-optimal", **once value training is decoupled**. They add: "if we use an artificially low
    learning rate … it will become advantageous to increase policy sample reuse". That describes our
    frozen 2.8e-5 exactly, and it explains why epochs and LR trade at a matched dose.
  - OpenAI Five targets a sample reuse of about 1 and saw that reusing data 2–3 times "can cause a
    factor of two slowdown".
  - Rudin uses 5 epochs.
- **Recommendation.** Keep 5 for generalist FORKS (`recipe.fork`, at the frozen 5.6e-5). A FRESH launch keeps N0's measured 10 at a KL-controlled 3e-4 (`recipe.fresh`, §3.22). **The SIZING study measured 5 epochs at a fresh LR (2026-10-02), and E10 STAYS.** At N = 256, matched samples and the same seed, E5 lost **10.56 pp [−13.12, −7.94]** of untaught (G-A −9.58 pp [−13.44, −5.68]) for an update 2.0× faster (20.4 s against 41.0 s). Its KL controller raised the LR further (max 5.18e-4 against 4.32e-4) at a lower approx-KL. E5 at MATCHED WALL (~1.9× the samples) is UNMEASURED. **ARM A3:** E2 at the matched dose (LR 1.4e-4). The literature favours
  it, and it saves more GPU time. The risk the ledger names is Adam overshoot at a higher per-step LR
  (L05595, L07591), so the arm reads KL and clip fraction as well as the meters. Decoupled policy and
  value epochs (PPG) WAIT for the Q head, which splits the value side anyway.

### 3.6 `clip_range` — 0.15, no schedule

- **Provenance.** `e5335b89`, 2026-05-17: 0.2 → 0.15 because the clip fraction was 0.23–0.26. It was
  re-set as a flag in `0ba4119a`, 2026-05-26. There is no clip schedule.
- **Ledger.**
  - The clip fraction is 0.077 (C) and 0.065 (E5), so clipping rarely binds at D_g (L21451).
  - The 0.10 → 0.15 change moved with the era and was never isolated (L17929, L18067: "the one
    untested regime leg").
- **Literature.**
  - Andrychowicz: "start with 0.25".
  - OpenAI Five, Rudin: 0.2.
  - Hilton, Cobbe & Schulman 2022: what the clip controls is how fast the policy moves, and that
    depends on how old the PROXIMAL policy is, not the behaviour policy. Decoupling the two with an
    EWMA of the weights (PPO-EWMA) makes PPO batch-size-invariant.
- **Recommendation.** KEEP 0.15. At a clip fraction of about 7 % it is a backstop, not the active
  constraint. Add no schedule. If the adaptive batch moves B by more than 4× in a run, adopt PPO-EWMA
  so the trust region does not move with it (WAITS, arm).

### 3.7 The KL → LR controller — make it a GUARD

- **Provenance.** It went through four commits:
  - `1d91dc99`, 2026-05-18: first version.
  - `9150c061`, 2026-05-23: target 0.01.
  - `f813b903`, 2026-05-27: ×1.2 and a cooldown, because the LR had fallen from 3e-4 to 1.2e-5 in 8
    moves.
  - `e4c305d9`, 2026-05-30: the band [0.005, 0.02].
- **How it acts.** Once per rollout, on an EMA of the last epoch's `approx_kl`. It multiplies the LR
  and never stops epochs early. The SB3 `target_kl` early-stop is not set.
- **Literature.** This is the rsl_rl / legged_gym controller (Rudin et al. 2021 Alg. 1: desired KL
  0.01, ×/÷ 1.5).
- **Ledger.**
  - The lineage runs it FROZEN (`--fork-lr-freeze`).
  - The live `approx_kl` is 0.0039, below the band's 0.005, so an unfrozen controller would RAISE the
    LR, up to `max_lr` = 6e-4 (2 × `--lr`, not 2 × the fork LR).
  - The ledger says a higher dose hurts: at 1.78× the teacher is not admitted (L21165), and Adam
    overshoot accounts for ~79 % of fold collateral (L05595).
- **Recommendation (SAFE).** **Guard-only mode.** The controller may LOWER the LR when KL exceeds
  2 × target, and may restore it back up to the declared rate, but never beyond it. That keeps the
  literature's safety half and drops the half our ledger convicted. It would not have fired once in
  this lineage (KL 0.0039).

### 3.8 Learning rate, `--fork-lr-freeze`, and the DOSE

- **Provenance.** `--lr` 3e-4 is the SB3 / Andrychowicz default. `--fork-lr`/freeze came in
  `e9b2a352`, 2026-09-01, because on resume SB3 restores the optimizer's own rate, so a fork inherited
  an annealed LR. `min_lr` 1e-5.
- **Literature.**
  - OpenAI Five: 5e-5 → 5e-6.
  - DeepNash: 5e-5.
  - Andrychowicz: 3e-4 is a "safe default" for small MLPs, and linear decay is "of secondary
    importance".
  - Our 2.8e-5–5.6e-5 sits in the large-model, large-batch range those systems used.
- **Ledger.** The DOSE (`lr × n_epochs / B_eff`) predicts fold collateral (L09065). More dose does not
  buy more (L09015, L21165).
- **Recommendation.**
  - KEEP the frozen, declared LR for the first M5 era: 5.6e-5 at E5, or whatever dose A3 settles.
  - KEEP the freeze as the lineage's control variable. It is what makes arms comparable.
  - No decay schedule: self-play is non-stationary and open-ended.
  - Fix the step count (§3.3).

### 3.9 `ent_coef` — 0.05; the owner's direction is a magnet, not a bonus

- **Provenance.**
  - The parser default is 0.02 (`4c57ffd7`, 2026-05-11, "increase switching incentives").
  - The live 0.05 is the 2026-09-12 `ent05` verdict (L18067), which restored v8's entropy regime:
    end-of-run entropy 1.086, inside v8's 1.07–1.11.
- **Ledger.**
  - `ent05`'s strength effect was NOT DETECTED (L18073).
  - Lane S: sharpness is flat after 9.5M. On the fixed learner, sharpening lowers the mass on
    dominated actions (−0.039 [−0.064, −0.022]) but **raises starvation 0.21 → 0.47** (L21471,
    PRELIMINARY).
  - Owner, 2026-09-29: no direct adoption of a lower bonus. X23 runs after the architecture is stable,
    judged by Lane S.
  - **Inconsistency:** X23's backlog row says "0.02 → 0.005", but the lineage runs 0.05.
- **Literature.**
  - Huang et al. 2022: 0.01 on Atari, 0 on MuJoCo, "no evidence that entropy term improves
    performance".
  - Andrychowicz §3.8: no regularizer helped, because careful initialization already provides
    exploration.
  - OpenAI Five anneals 0.01 → 0.001.
  - **Magnetic Mirror Descent** (Sokota et al. 2023) regularizes toward a MAGNET policy ρ with
    KL(π‖ρ). With a uniform ρ, that term is −H(π) + const, so **our entropy bonus is exactly MMD with a
    uniform magnet at α = 0.05.** The paper's deep-RL MMD is RLlib's PPO with the adaptive forward-KL
    penalty replaced by a reverse-KL one. The authors report "favorable performance" for it as a
    self-play algorithm in 3x3 Dark Hex and Phantom Tic-Tac-Toe.
  - **R-NaD / DeepNash** (Perolat et al. 2022) transforms the reward with −η log(π/π_reg), where
    η = 0.2 and π_reg is replaced by the last fixed point every Δm steps. A TRAILING magnet converges
    to Nash in two-player zero-sum imperfect-information games.
  - DeepNash also filters actions below 3 % probability at test time. Our greedy eval does a cruder
    version of that.
- **Recommendation (WAITS → ARM).** The owner's "exploration belongs in the data, not the weights"
  has a grounded form: **replace the uniform magnet with a trailing one.** The candidates, to be
  pre-registered:
  1. KL(π‖π_reg) with π_reg an EWMA or periodic snapshot of the policy (R-NaD's schedule).
  2. KL to a Smogon-prior behaviour-cloned policy. That fits the Smogon-only priors rule, because the
     magnet is a prior the network reads through its loss.

  Either keeps probability on moves the magnet thinks plausible, instead of on every legal move
  uniformly. That is the starvation mechanism Lane S measures, and the literature does not settle it
  for us; Lane S decides. The arm is X23 re-scoped: an entropy anneal, a trailing magnet, and a
  control. It runs after M5 plus the discrete-token boundary, per the owner.
  - **FINDING:** the "exploration in the data / KL-to-magnet" direction was not recorded anywhere in
    the repo before this doc. This doc now records it as the owner's direction of 2026-09-29, as
    relayed by the orchestrator.

### 3.10 γ = 1.0 with a terminal-only reward

- **Provenance.** `cbcb0bfb`, 2026-09-06: at γ = 1 with a terminal-only reward, V(s) is exactly
  P(win | s). The shaped reward path was deleted on 2026-09-26 (L21417).
- **Literature.** Andrychowicz calls γ "one of the most important hyperparameters" and starts at 0.99
  for continuing-control tasks. Episodic zero-sum board games use γ = 1 (AlphaZero, MuZero), and
  training's stall forfeit guarantees termination.
- **Ledger.** Late optimism was traced to the γ = 1 terminal target (L15491). The shaped-vs-sparse
  strength comparison was NOT DETECTED at 75M (L18659).
- **Recommendation.** KEEP. It is the definition of the win-probability value the owner's goal names.

### 3.11 Policy GAE λ — 0.80; why 0.95 lost, and what adaptive λ would look like

- **Provenance.** It moved several times in May, all on reasoning, never measured:
  - 0.95 → 0.85 (`f29251b4`, 2026-05-16).
  - 0.95 again (`8ce35fe2`), then reverted (`c608ea87`).
  - 0.85 → 0.80 (`eff7ddee`, 2026-05-22): "Gen3 RNG variance better absorbed by critic than
    advantages".
  - It became a flag in `2cc83080`, 2026-09-26.
- **Ledger.** L21451: **L95 (λ 0.95) −10.10 pp [−12.77, −7.65], OUTSIDE BELOW, 0/8 teams up**. It
  learned nothing untaught in 8M (−0.22 vs N0) while its SmallRL rate held.
- **Literature.** 0.95 (OpenAI Five, Rudin, the original GAE paper's range); 0.9 (Andrychowicz §3.4).
- **Why the ledger wins here** (the orchestrator's analysis, verified here; mechanism **PROPOSED**,
  not measured).
  - With γ = 1 and a terminal-only reward, each advantage A^λ = Σ_k λ^k δ_{t+k} sums TD errors that
    carry the noise of every later random event, of which there are about 45:
    - our own T = 1 sampling (the policy plays its own argmax only 62.8 % of the time, X23);
    - the opponent's sampling;
    - damage rolls, crits, accuracy and speed ties.
  - Under independent per-step noise, Var(A^λ) ∝ 1/(1 − λ²): **2.78 at 0.80 vs 10.26 at 0.95, a
    ratio of 3.7×.**
  - L95's scalar `train/value_loss` was 3.7× C's. That head's returns follow the policy λ, so this is
    consistent but only suggestive, and the ledger says it is not evidence about the win-prob critic.
  - Per-batch advantage normalization then rescales each update to unit advantage variance, so a
    3.7× noisier advantage cuts the signal share of every fixed-size, fixed-LR update.
  - The bias that λ < 1 introduces is V's error, and our critic is trained at λ = 1 to real outcomes,
    so it is small.
  - Our game is far noisier per step than the MuJoCo and Dota settings the 0.9–0.95 defaults came
    from.
- **Recommendation.**
  - (a) KEEP 0.80.
  - (b) **ARM A4, a SWEEP {0.6, 0.7, 0.8, 0.9} in the M5 era**, re-run after the sharpness and critic
    changes. λ's optimum is coupled to policy entropy, opponent temperature and critic quality, and
    each of those is scheduled to move.
  - (c) **STATE-DEPENDENT λ as the grounded adaptive option (PROPOSED, WAITS for X25):**
    - Variable-λ traces are standard (Sutton & Barto 2018 §12.8).
    - White & White 2016 adapt λ per state with a greedy bias–variance objective.
    - Meta-gradient RL (Xu, van Hasselt & Silver 2018) learns λ online.
    - Tie it to X25's epistemic confidence: **trust the critic (lower λ) where it is confident, lean
      on the real outcome (higher λ) where it is not.** That needs X25's uncertainty validated on
      Lane S's ground truth first.

### 3.12 The critic — λ = 1 BCE on the win head, `vf_coef` 0.5, no value clipping, no PopArt

- **Provenance.**
  - `vf_coef` 0.5 is SB3's default, made a flag in `5afc9a73` (2026-06-06).
  - `win_prob_lambda` no longer exists (the λ-return target was deleted, deletion pass L2; the BCE target is always the terminal outcome).
  - Under winprob the value loss is `vf_coef` × the masked-mean BCE on the win logit (`ppo.py:767-769`).
  - `clip_range_vf` is inert under winprob.
  - PopArt, the distributional value head, `value_from_dist`, the value-tail weight and `--win-prob-coef` were DELETED (deletion pass L1, config v131; `designs/deleted_flags.md`) — there is nothing left to refuse.
- **Ledger.**
  - λ 0.9 targets failed their replicate (L17453).
  - λ 0.95 failed its directional claim (L17578).
  - `vf_coef` 1.5 **removes opponent-class information from V** (L18010).
  - `vf_coef` 0.25 was NOT DETECTED (L18206).
  - V is under-dispersed by 25–40 % (L16929; X21).
- **Literature.**
  - Andrychowicz §3.4: "neither Huber loss nor PPO-style value loss clipping".
  - Huang #6: value clipping "hurts".
  - PopArt / value normalization exists for unbounded returns (van Hasselt et al. 2016). A
    probability target is bounded, so it does not apply.
- **Recommendation.** KEEP all of it. The open critic questions are X21 (native calibration) and the
  Q head, not recipe knobs.

### 3.13 Advantage normalization — per micro-batch; move it to per rollout

- **Provenance.** SB3's `normalize_advantage = True`, never overridden. Under accumulation it is
  computed per 2,048-row MICRO-batch (`ppo.py:392`), not per 65,536-row effective batch.
- **Literature.**
  - Huang #4 and Andrychowicz §3.3 (C67): per-minibatch normalization "does not affect performance
    too much".
  - Hilton et al. 2022 §4: its statistics' sample size should scale with the batch to stay
    batch-invariant.
- **Our interaction.** Masked-mean loss terms (the win BCE, the belief losses) are also averaged per
  micro-batch before the ÷K, so their effective weight varies with mask coverage per micro-batch.
- **Recommendation (SAFE, with an equivalence gate).** Compute the advantage mean and std ONCE per
  rollout. That makes K a pure batching choice, exactly as `step_size_and_batch.md` claims. Gate it
  with a learner-benchmark equivalence read: gradient cosine and the loss trajectory over one rollout.
  It needs no strength arm, because at 2,048 rows the statistics differ at the third digit.

### 3.14 Aux-loss weights and the shared trunk — the PPG question

- **Live values.** The belief, intent and item heads each at 0.05: opp-belief, move, move-latent,
  spread, hp-type, item, opp-intent. The set-valued β term is at 0.05 × 0.05.
  - `belief_grad_mode shaping` means nothing is cut: the belief gradients reach the trunk, and PPO
    trains the belief heads.
  - The opponent-intent head reads a DETACHED input.
  - These are loss weights, so `--arch production` does not set them; they ride the argv.
- **Ledger.**
  - No arm has varied the 0.05s (the ledger has no evidence).
  - Belief memorization is DETECTED on-pool (L21301), with a win-rate DiD of +4.8 pp [+1.9, +7.8]
    (L21381).
  - Half-batch trunk cosine was −0.030 in distill arms (L04637).
- **Literature.**
  - PPG (Cobbe et al. 2021): "Interference between policy and value function optimization can
    negatively impact performance when parameters are shared between the policy and the value
    function networks". PPG separates the networks and distils
    value and aux features into the policy in a periodic AUXILIARY PHASE with a behaviour-cloning KL
    (β_clone).
  - Andrychowicz §3.2: separate policy and value networks on 4 of 5 tasks.
- **Recommendation (WAITS).** The Q-head design already decided gradient routing: Q heads train into
  the shared trunk, with a detached probe as the baseline and gradient-cosine guards
  (`design_q_head.md` §5; X20 for α). The 0.05 weights should be set by those guards
  (`grad/*_share`, trunk cosine), not by a blind sweep. A PPG-style auxiliary phase is the grounded
  fallback if the guards show interference, and it is an arm after the Q head.

### 3.15 Grad-norm clipping — 0.5

- **Provenance.** SB3's `max_grad_norm = 0.5`, never passed. It is applied once per optimizer step,
  before Adam.
- **Live.** Median `train/grad_norm` is 0.42. How often it binds is **UNMEASURED**, and the half-batch
  step is the likelier one to clip.
- **Literature.** Huang #8 uses 0.5. Andrychowicz §3.3 found a "small performance boost with the
  exact clipping threshold making little difference". DeepNash clips at 10,000, i.e. effectively
  never.
- **Recommendation.** KEEP. **Log the fraction of steps clipped** (SAFE). If it exceeds about 20 %,
  the clip is acting as a hidden LR modifier and belongs in the dose.

### 3.16 The optimizer — AdamW, β = (0.9, 0.999), ε = 1e-5, weight decay 1e-5

- **Provenance.** `6bdc096f`, 2026-05-23: switched to AdamW and kept SB3's ε = 1e-5. There is one
  parameter group, so decay also hits biases and norms.
- **Literature.**
  - Huang #1: ε 1e-5, not PyTorch's 1e-8.
  - Andrychowicz §3.7: Adam, β1 = 0.9.
  - DeepNash uses β1 = 0.
- **Ledger.** Plasticity loss measured NULL (L01905, L05191). There is no evidence on β, ε or weight
  decay.
- **Recommendation.** KEEP. Note that weight decay is **effectively inert**: AdamW's per-step decay
  is lr × wd ≈ 5.6e-10. Excluding biases and norms is hygiene with no expected effect, so it gets no
  priority.

### 3.17 Observation normalization — none

- **Literature.** Andrychowicz §3.3: "Always use observation normalization." That is measured on raw
  MuJoCo state vectors.
- **Ours.** The observation is engineered: embeddings plus bounded features from
  `Gen3ObservationEncoder`. A RUNNING normalizer would:
  - make the network's input depend on training history;
  - break inference-slot and search parity;
  - acquire state after freeze, which M5's declared lifecycle forbids.
- **Recommendation.** KEEP none. **SAFE audit:** a one-time offline scan of per-feature scale over a
  banked corpus. Any feature with a scale far from its block's goes to the obs-enrichment backlog as a
  fixed, declared rescale.

### 3.18 Initialization — the first policy is uniform

- **Provenance.** SB3 orthogonal initialization, then `restore_identity_init` puts back the deliberate
  zero-inits (`d560b00f`, 2026-08-01; ledger L00118). The pointer head's scorers start at zero, so the
  first policy is uniform over legal actions.
- **Literature.** Andrychowicz §3.2: "Initialize the last policy layer with 100× smaller weights".
  Ours is the limit case.
- **Recommendation.** KEEP.

### 3.19 Self-play opponent temperature — 1.0; the exploration that DOES belong in the data

- **Provenance.** `f9ac97ec`, 2026-06-01: stochastic opponents by default.
- **Ledger.**
  - Our model is +23 pp [+9.5, +35.4] stronger GREEDY than at T = 1 against greedy Kakuna (L21447).
  - Self-play opponents at T = 1 are off their own plan on ~37 % of moves (X23).
  - So V learns the win rate of SAMPLED play, which is biased low for greedy deployment and ties to
    X21.
- **Literature.** OpenAI Five and AlphaStar sample their opponents stochastically. None of them sets
  a temperature for a model that is sharply better greedy.
- **Recommendation.** **ARM A5 = X23(b):** opponent T 1.0 → 0.5, with the trainee's own rollout left
  at T = 1. This changes the DATA the trainee sees, not its weights' regularizer, so it matches the
  owner's direction. The P2 peek is V calibration on greedy games.

### 3.20 The opponent pool — window, recency, stable share, bots, PFSP

- **Live values** (`snapshot_pool.py:166-167,309`, `ddf36ea4`, 2026-05-22):
  - 20 snapshots, sliding.
  - Recency weight 1 + 0.3 × relative age, so the newest is weighted 1.3×.
  - Snapshots enter by promotion.
  - Bots get at least 10 % of games.
  - `stable_opponent_selfplay_share` 0.2.
  - `pfsp_scale` 0 (and `team_pfsp` was off — team-PFSP was deleted in deletion pass L4).
  - `exploiter_bot_fraction` is 0.5 but INERT (it acts only with `--exploiter-keep-bots`).
  - The 20 and the 0.3 are hardcoded and undocumented in `designs/training/`.
- **Ledger.**
  - Pure PFSP over a homogeneous fresh self-pool cost 26–33 Elo. v8's +69 came with PFSP over a
    diverse, anchored pool (L01151).
  - Arm C (`--pfsp-scale 2.5`) was paused at 30.46M and **never read** (L15610).
  - Population loop rounds 1 and 2: NOT DETECTED (L21329, L21387).
- **Literature.**
  - OpenAI Five: 80 % latest self and 20 % past, with past opponents sampled by a softmax over quality
    scores that fall when the agent beats them (Berner et al. App. N).
  - AlphaStar's league: PFSP with f_hard(x) = (1 − x)^p over past players plus exploiters (Vinyals et
    al. 2019).
- **Recommendation.** KEEP for the first M5 era. PFSP is an arm only over a DIVERSE, anchored pool,
  which is `design_ladder_campaign.md`'s training ecology. OpenAI Five's quality-score sampler is the
  grounded self-tuning alternative to a fixed recency weight. Record the 20 and the 0.3 in
  `self_play_and_pool.md` (a doc task).

### 3.21 The eval regime — greedy vs greedy sentinels

- **Ledger.** Greedy became the default on 2026-09-07 (L13654). The old asymmetric regime was worth
  +8.9 pp to the trainee.
- **Recommendation.** KEEP. `designs/training/eval_and_rating.md` owns it.

### 3.22 The recipe surface — BUILT (K10(a), 2026-09-30)

- **The finding that ordered it.** Parser defaults differed from the live recipe on `n_envs`
  (32 vs 48), `batch_size` (4096 vs 2048), `n_epochs` (5 vs 10), `ent_coef` (0.02 vs 0.05) and
  `clip_range_vf` (0.5 vs none). The build's survey found more: `grad_accum_steps` (1 vs 32, an
  effective batch of 4,096 instead of 65,536), `self_play` (off vs on), `beta_setvalued_coef`
  (0 vs 0.05), the critic (`shaped` vs `winprob`) with its three required reward values, and the
  supervision doses. `--arch production` applied none of them, so a fresh argv typed without them
  trained a different recipe: the recipe-side twin of the 2026-09-06 stripped-architecture
  incident. The old parser's 5-epoch default was one of those silent divergences, not evidence for
  5 epochs.
- **What is built** (`src/main/train/recipe_surface.py`; per-flag mechanics in
  `src/agents/training/CLAUDE.md` "The recipe surface"):
  - a `recipe` block in `designs/production_config.json` with two parts. `recipe.fresh` is N0's
    MEASURED fresh recipe: every training knob `models/ai_v14_01_base` launched with, including
    the doses and the KL controller's constants. `recipe.fork` is
    what a generalist fork changes: E5. A `recipe.fresh` key that is also a recorded mirror field
    must equal it.
  - `--arch production` writes every `recipe.fresh` knob the argv did not TYPE, as if typed. A
    typed token still wins; the parser records which tokens were typed. `recipe.fork` is never
    applied: `--fork-lr` is refused on a fresh run.
  - `checkargs`, `--dry-run` and the launcher print a RECIPE SURFACE block beside the ARCH SURFACE.
    A fresh argv that differs on an UNTYPED knob is REFUSED; a TYPED difference is the arm's lever
    (INFO); `--allow-nonproduction-recipe` consents. A fork is compared with `recipe.fork`, as INFO.
  - **Restarts.** A launcher restart strips `--arch`. On a same-run restart of an
    `--arch production` run, each untyped knob comes from exactly one place, announced:
    - `--lr`, `--batch-size` and `--n-steps` are INERT on a resume (SB3 restores
      them, and a checkpoint's own gamma) and are never re-applied;
    - a recorded tri-state field (critic, doses incl. `opp_intent_coef`, `policy_gae_lambda`) is inherited by the resume's
      own `_resolve`;
    - a recorded value-checked field (`vf_coef`) comes from the checkpoint's
      `model_config.json`;
    - a knob recorded nowhere else (`n_envs`, `n_epochs`, `ent_coef`, …) comes from the run's
      `metadata.json` `cli_args`.

    A value missing from its source REFUSES by name (`FATAL_CONFIG`), never a default. The
    provenance tags survive too: `recipe_source` from `cli_args`, `arch_source` from the
    checkpoint's `model_config.json` (the first restart used to record `arch_source: null`). This follows
    the general restart rule (`68850f27`: the surface is inherited from `model_config.json`,
    `opp_intent_coef` a recorded field from config v125) and covers only what it cannot supply.
  - `src/recipe_doc_gate_test.py` holds the table below and §1's live-value column to the block.
  - **The production namespace carries the recipe (closed 2026-09-30):** `main.train.production_args.
    production_args()` (re-homed from the deleted cutover harness, deletion pass U3) is the resolved namespace of a real fresh `--arch production` launch — the
    trainer's own `resolve_config`, not a second copy of the surface. It used to build its namespace
    from the ARCH surface plus a `hasattr` copy of the mirror's top-level fields, which skipped the
    nested `recipe` block (`n_envs` 32, `ent_coef` 0.02, self-play off, …).
    `src/main/train/production_args_test.py` compares it with a real resolved fresh
    launch on every mirror key, the recipe included, and fails if either side drops a key. The K9
    learner golden was re-recorded for it (its learner now carries `beta_setvalued_coef` 0.05).

**The production recipe, and where each value comes from.** "N0" is `models/ai_v14_01_base`, the
lineage's FRESH launch (`metadata.json` `original_command`, `cli_args`, `dose`; `model_config.json`).
"E5" is `models/ai_v14_03_lbat_e5`. Every file was read, never written.

| knob | production | block | source |
|---|---|---|---|
| `n_envs` | 256 | `recipe.sizing` | The SIZING study's N\* (2026-10-02, §3.1; `program_rust_core.md` Decision record): arms B / C. Was N0's `--n-envs 48` |
| `n_steps` | 384 | `recipe.sizing` | The SIZING study (2026-10-02): D / N\* = 98,304 / 256, the shape arms B / C trained (was N0's 2048); on the Rust core the n_steps MAXIMUM the buffer is allocated at. The adaptive-batch ceiling n_steps_max = 1,024 is NOT applied (§3.2, arm A1) |
| `rollout_target_samples` | 98304 | `recipe.sizing` | the complete-game collector's update size, set EXPLICITLY by the SIZING study (2026-10-02) to the 98,304 every arm trained at (null would mean N × n_steps, the same number today) |
| `trainee_slots` | null | `recipe.sizing` | T2's trainee slots; null = derived (1; 3 under per-game pinning). The SIZING study's |
| `t2_buckets` | null | `recipe.sizing` | T2's buckets; null = derived at startup: (8, N), plus the opponent cap 64 when N > 64 (`gen3_slot_bucket_caps_v1`, `--t2-opponent-bucket-cap`) — (8, 48) at N = 48, (8, 64, 256) at N = 256. The SIZING study's |
| `t2_lanes` | null | `recipe.sizing` | T2's lanes; null = derived (min(slots, 8)). The SIZING study's |
| `batch_size` | 2048 | `recipe.fresh` | N0 `--batch-size 2048`, `dose.batch_size` |
| `grad_accum_steps` | 32 | `recipe.fresh` | N0 `--grad-accum-steps 32`, `dose.grad_accum_steps` (effective 65,536) |
| `n_epochs` | 10 | `recipe.fresh` | N0 `--n-epochs 10`, `dose.n_epochs` |
| `lr` | 0.0003 | `recipe.fresh` | N0 `--lr 0.0003`, `dose.lr_flag`: the KL controller's seed |
| `min_lr` | 1e-05 | `recipe.fresh` | N0 `--min-lr 1e-05`, `dose.kl_controller.min_lr` |
| `max_lr` | none | `recipe.fresh` | N0 `cli_args.max_lr` null, so 2 × lr = `dose.kl_controller.max_lr` 0.0006 |
| `anneal_lr_start_steps` | none | `recipe.fresh` | N0 `cli_args.anneal_lr_start_steps` null: no cosine phase (`dose.kl_controller.phase` adaptive) |
| `target_kl` | 0.01 | `recipe.fresh.kl_controller` | N0 `dose.kl_controller.target_kl`; the callback's constructor default |
| `kl_factor` | 2.0 | `recipe.fresh.kl_controller` | N0 `dose.kl_controller.kl_factor`; the callback's constructor default |
| `lr_factor` | 1.2 | `recipe.fresh.kl_controller` | N0 `dose.kl_controller.lr_factor`; the callback's constructor default |
| `weight_decay` | 1e-05 | `recipe.fresh` | N0 `--weight-decay 1e-05`, metadata `weight_decay` |
| `clip_range` | 0.15 | `recipe.fresh` | N0 `--clip-range 0.15`, metadata `clip_range` |
| `clip_range_vf` | none | `recipe.fresh` | N0 `--clip-range-vf none`, metadata `clip_range_vf` −1.0 (disabled). INERT under the win-prob critic (the only critic) |
| `ent_coef` | 0.05 | `recipe.fresh` | N0 `--ent-coef 0.05`, metadata `ent_coef` |
| `policy_gae_lambda` | 0.8 | `recipe.fresh` | N0 metadata `gae_lambda` 0.8 (N0 predates the flag; the value was hardcoded) |
| `self_play` | true | `recipe.fresh` | N0 `--self-play`, `cli_args.self_play` |
| `vf_coef` | 0.5 | `recipe.fresh` | N0 `--vf-coef 0.5`, `model_config.json` |
| `opp_belief_aux_coef` | 0.05 | `recipe.fresh` | N0 `--opp-belief-aux-coef 0.05`, `model_config.json` |
| `opp_intent_coef` | 0.05 | `recipe.fresh` | N0 `--opp-intent-coef 0.05`, `cli_args.opp_intent_coef` |
| `move_belief_coef` | 0.05 | `recipe.fresh` | N0 `--move-belief-coef 0.05`, `model_config.json` |
| `move_belief_latent_coef` | 0.05 | `recipe.fresh` | N0 `--move-belief-latent-coef 0.05`, `model_config.json` |
| `spread_belief_coef` | 0.05 | `recipe.fresh` | N0 `--spread-belief-coef 0.05`, `model_config.json` |
| `hp_type_belief_coef` | 0.05 | `recipe.fresh` | N0 `--hp-type-belief-coef 0.05`, `model_config.json` |
| `item_belief_coef` | 0.05 | `recipe.fresh` | N0 `cli_args.item_belief_coef`, `model_config.json` |
| `beta_setvalued_coef` | 0.05 | `recipe.fresh` | N0 `--beta-setvalued-coef 0.05`, `cli_args` |
| `intent_label_bot_weight` | 0.25 | `recipe.fresh` | N0 `--intent-label-bot-weight 0.25`, `model_config.json` |
| `n_epochs` | 5 | `recipe.fork` | E5 `--n-epochs 5`, `dose.n_epochs` (adopted L21451 for generalist forks) |
| `fork_lr` | 5.6e-05 | `recipe.fork` | E5 `--fork-lr 5.6e-05`, `dose.lr_now` (2 × K2's 2.8e-05: the same dose at half the epochs) |
| `fork_lr_freeze` | true | `recipe.fork` | E5 `--fork-lr-freeze`, `dose.lr_frozen` |

- **Left out, and why.**
  - `seed`: an arm's identity, not a recipe choice.
  - `matmul_precision`: fp32 (`highest`) is the only precision. TF32 was RETIRED by the owner on 2026-10-01 (Decision record) and `--matmul-precision` is DELETED (deletion pass K2, 2026-10-02); the realized value is still recorded in `metadata.json`.
  - `adaptive_batch`: off, which is the default, and arm A1's lever.
  - The eval regime, opponent-pool shares and temperatures, and `obs_source`:
    their resolved defaults already equal N0's (`value_true_team` was a recorded field here until deletion pass L2 removed it).
  - The critic readout the win-prob critic (the only critic) implies (`win_prob_mode`): implied. (`value_dist_*`, PopArt and
    `--win-prob-coef` were deleted, L1, config v131 — they no longer exist to imply or refuse.)
- 🚨 **5 epochs at a FRESH learning rate: MEASURED, and it LOSES (the SIZING study, 2026-10-02).** E5
  was adopted as a FORK at a frozen 5.6e-5. On a fresh launch at a KL-controlled 3e-4 it lost
  10.56 pp [−13.12, −7.94] of untaught at matched samples (arm C against arm B, §3.5). `recipe.fresh`
  keeps N0's measured 10 epochs, and E5 lives only in `recipe.fork`.

---

## 4. Where our ledger contradicts a literature default, and why the ledger wins here

| # | literature default | our evidence | why our setting differs |
|---|---|---|---|
| 1 | GAE λ 0.95 (OpenAI Five, Rudin) or 0.9 (Andrychowicz) | λ 0.95 −10.1 pp vs 0.80 (L21451) | γ = 1 with one terminal outcome and ~45 random events after each decision: a high λ carries 3.7× the advantage variance (§3.11). The λ = 1 critic keeps the bias of λ < 1 small |
| 2 | TD(λ < 1) value targets lower variance | λ 0.9 / 0.95 critic targets did not replicate (L17453, L17578) | The critic is a calibrated probability trained on outcomes; a bootstrapped target imports V's own under-dispersion (L16929) |
| 3 | entropy 0–0.01 | 0.05 restores v8's regime (L18067), strength NOT DETECTED (L18073) | Not a contradiction but a regime choice. It is recorded here because Lane S shows the uniform magnet starving good moves (L21471) |
| 4 | 3–10 epochs | E5 ≡ E10 at a matched dose (L21451) | Consistent with PPG: at a low, frozen LR, epochs and LR trade. The dose, not the epoch count, is the control variable |
| 5 | PFSP (AlphaStar) | pure PFSP on a homogeneous self-pool cost 26–33 Elo (L01151) | PFSP assumes a DIVERSE league. On near-copies it concentrates games on noise |
| 6 | read the gradient noise scale of the loss (McCandlish) | the total reads "over-batched" while the policy term reads noise-limited, 35–40× apart (L08486) | Our loss is mostly dense supervised aux heads, whose gradients agree. McCandlish measured single-objective losses |
| 7 | raise the value weight for a better critic | `vf_coef` 1.5 strips opponent information from V (L18010) | A shared trunk: PPG's interference, shown in the other direction |
| 8 | "always" normalize observations | no normalizer | Not measured. It is a design constraint: an engineered observation, search/inference parity, and the declared lifecycle (§3.17) |
| 9 | stochastic self-play opponents | greedy is +23 pp stronger than T = 1 for our model (L21447) | A diffuse policy (62.8 % own-argmax), so sampled opponents are off-plan. This is what arm A5 tests |

---

## 5. Migration order for the first serious training era on M5

**Standing rules.**
- Every arm is a one-lever fork on the M5 infrastructure (about 8M steps, the battery's size) against
  a matched control.
- Every arm reads untaught-8 non-inferiority (CI low > −3.69), the SmallRL guard, the stall ratio
  (≤ 1.5× control), the Lane S spectrum and starvation, and KL, clip fraction and `train/dose_rate`.
- Before any adoption, the E5 transfer check on the fixed learner must read clean (L21451, L21435).

### Stage 0 — SAFE defaults and prerequisites (land before the SIZING study; no arm)

| # | change | gate |
|---|---|---|
| 0.1 | **Delayed-label buffer for the critic** (`design_q_head.md` §5, already decided). Until it lands, n_steps ≥ 512 | the masked-row share tag reads ≈ 0 at short windows; critic loss unchanged at n_steps 2048 |
| 0.2 | **Divisible rollout shapes** (D = a whole multiple of micro × K_max) and **dose counted in real steps** | bit-identical at divisible shapes; `main.dose` reports ceil steps |
| 0.3 | **Per-rollout advantage normalization** | learner-benchmark equivalence (gradient cosine, one-rollout loss trajectory) |
| 0.4 | **KL controller guard-only** (may lower the LR, never above the declared rate) | never fires on the K2 recipe (KL 0.0039); a unit test of the down-only rule |
| 0.5 | **Telemetry:** the split-half-by-game ‖G‖² and design effect c; multi-point noise folding; the grad-clip binding share; the critic masked-row share | read-only, bit-identical learning (the K2 diagnostics-cadence test pattern) |
| 0.6 | **The RECIPE mirror** in `production_config.json` + `checkargs` RECIPE SURFACE — **BUILT** (K10(a), §3.22) | a fresh argv missing a recipe token is refused with its name (`recipe_surface_test`) |

### Stage 1 — the SIZING study (`program_rust_core.md` order constraint 5, unchanged)

The study runs with Stage 0 in place, so its short-window arms are not dropping critic rows. It adds c
and the masked-row share to its reads.

### Stage 2 — one-lever arms, in this order

| arm | lever | control | what decides it |
|---|---|---|---|
| **A1** | `--adaptive-batch policy`, target 1.0, band 2.0, LR held, dose recorded | fixed K at the study's shape | the standing meters. Secondary: the arm's time-to-equal-untaught at matched samples |
| **A2** | dynamic n_steps (§2.2 rule; m = 4, k = 2) inside a declared maximum | A1's winner at fixed n_steps | the same, plus samples-to-matched-untaught (the "don't waste samples early" claim) |
| **A3** | E2 at the matched dose | E5 | the meters, plus KL and clip fraction (Adam-overshoot risk, L05595) |
| **A4** | policy λ sweep {0.6, 0.7, 0.9} | 0.80 | the meters. Re-run after any sharpness or critic change (§3.11) |
| **A5** | self-play opponent temperature 0.5 (X23b) | T = 1.0 | the meters, plus V calibration on greedy games (P2) and the Lane S own-argmax rate |

A1 goes before A2 because A2's rule reads K(t). A3–A5 are independent of each other and may run in
any order once A1 and A2 settle the shape.

### Stage 3 — after the architecture is stable (M5 + the discrete-token boundary; the owner's gate for X23)

- **X23 re-scoped:** an entropy anneal vs a trailing magnet (R-NaD/MMD; §3.9) vs a control, judged by
  Lane S starvation and the meters.
- **PPO-EWMA**, if A1 moves B by more than 4× in a run (§3.6).
- **PFSP over a diverse, anchored pool**, or OpenAI Five's quality-score sampler (§3.20).

### Stage 4 — after the Q head and search

- **PPG-style auxiliary phase**, if the Q-head's gradient-cosine guards show interference (§3.14).
- **Decoupled policy and value epochs** (PPG), which the Q head's split value side makes natural.
- **State-dependent λ from X25's confidence** (§3.11c).
- **Search targets:** Gumbel MuZero's policy-improvement target (Danihelka et al. 2022) and expert
  iteration (X15). These replace much of the entropy and exploration question with a planner. (Distillation was the only BUILT route to X15 and is DELETED, deletion pass L3, 2026-10-02; a Rust port is ~1-2 agent-days if X15 is scheduled.)

---

## 6. Open questions (answer by measurement, record the answer here)

- The design effect c of our rollouts (§2.3). If c ≳ 2 at today's shape, the ROLLOUT is
  noise-limited now.
- How much B_noise moves over a run (it decides whether the √ schedule is worth building, §2.1).
- Dose vs √-scaling as the invariant when B moves (§2.4).
- Whether E5's non-inferiority transfers to the fixed learner (L21451).
- The binding share of the 0.5 grad clip (§3.15).

---

## Sources

**Literature.**
- Andrychowicz et al. 2021, *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical
  Study* — https://arxiv.org/abs/2006.05990
- Huang, Dossa, Raffin, Kanervisto, Wang 2022, *The 37 Implementation Details of Proximal Policy
  Optimization* (ICLR blog track) — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- McCandlish, Kaplan, Amodei et al. 2018, *An Empirical Model of Large-Batch Training* —
  https://arxiv.org/abs/1812.06162
- Cobbe, Hilton, Klimov, Schulman 2021, *Phasic Policy Gradient* — https://arxiv.org/abs/2009.04416
- Hilton, Cobbe, Schulman 2022, *Batch size-invariance for policy optimization* —
  https://arxiv.org/abs/2110.00641
- Malladi, Lyu, Panigrahi, Arora 2022, *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*
  — https://arxiv.org/abs/2205.10287
- Berner et al. (OpenAI) 2019, *Dota 2 with Large Scale Deep Reinforcement Learning* —
  https://arxiv.org/abs/1912.06680
- Rudin, Hoeller, Reist, Hutter 2021, *Learning to Walk in Minutes Using Massively Parallel Deep RL* —
  https://arxiv.org/abs/2109.11978
- Espeholt et al. 2018, *IMPALA* (V-trace) — https://arxiv.org/abs/1802.01561
- Sokota et al. 2023, *A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and
  Two-Player Zero-Sum Games* (Magnetic Mirror Descent) — https://arxiv.org/abs/2206.05825
- Perolat et al. 2022, *Mastering the Game of Stratego with Model-Free Multiagent RL* (R-NaD /
  DeepNash) — https://arxiv.org/abs/2206.15378
- Vinyals et al. 2019, *Grandmaster level in StarCraft II using multi-agent RL* (AlphaStar, PFSP) —
  https://www.nature.com/articles/s41586-019-1724-z
- Schrittwieser et al. 2020, *MuZero* — https://arxiv.org/abs/1911.08265
- Danihelka et al. 2022, *Policy improvement by planning with Gumbel* —
  https://openreview.net/forum?id=bERaNdoegnO
- Schulman et al. 2017, *PPO* — https://arxiv.org/abs/1707.06347
- Schulman et al. 2016, *GAE* — https://arxiv.org/abs/1506.02438
- van Hasselt et al. 2016, *Learning values across many orders of magnitude* (PopArt) —
  https://arxiv.org/abs/1602.07714
- Sutton & Barto 2018, *Reinforcement Learning: An Introduction*, 2nd ed., §12.8 —
  http://incompleteideas.net/book/the-book-2nd.html
- White & White 2016, *A Greedy Approach to Adapting the Trace Parameter for Temporal Difference
  Learning* — https://arxiv.org/abs/1607.00446
- Xu, van Hasselt, Silver 2018, *Meta-Gradient Reinforcement Learning* —
  https://arxiv.org/abs/1805.09801

**Ours.**
- [`../training/step_size_and_batch.md`](../training/step_size_and_batch.md)
- [`../training/ppo_step.md`](../training/ppo_step.md)
- [`../training/critic_and_value_losses.md`](../training/critic_and_value_losses.md)
- [`../training/self_play_and_pool.md`](../training/self_play_and_pool.md)
- [`../training/eval_and_rating.md`](../training/eval_and_rating.md)
- [`program_rust_core.md`](program_rust_core.md) §2 M5
- [`design_q_head.md`](design_q_head.md)
- `research_state/measurements/learner_battery_2026-09-26/`

---

## Decision record

Owner decisions are marked **(owner)**. `L…` is the ledger line as `ledger_index.md` lists it.
**PROPOSED** rows are this review's recommendations, awaiting the owner or the orchestrator.

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-09-29 | Recipe principle **(owner)** | Bias every knob toward grounded published practice; our ledger wins where it measured our setting, and the doc says why | Keeping first-week values because they run | owner 2026-09-29; §0 |
| 2026-09-29 | `n_epochs` | 5 at 2 × D_g (5.6e-5, same dose) for generalists; exploiters and readers keep arm A's recipe | 10 epochs | L21451 (E5 ADOPT) |
| 2026-09-29 | Policy GAE λ | Stays 0.80 | 0.95 (−10.10 pp, OUTSIDE BELOW) | L21451 |
| 2026-09-29 | Entropy **(owner)** | No direct adoption of a lower bonus; X23 after the architecture is stable, judged by Lane S | Adopting a lower `ent_coef` on a strength read | `program_rust_core.md` Decision record |
| 2026-09-29 | Update batch **(owner)** | Dynamic via K at a fixed micro-batch; long-term target ≈ 1.1 × critical batch | A static, swept batch | `program_rust_core.md` order constraint 5 |
| 2026-09-29 | Batch target and schedule | **PROPOSED:** constant-ratio controller, target 1.0, band 2.0 (covers the owner's 0.91); the √B_noise schedule is recorded as the refinement if B_noise varies by more than 10× | A fixed ratio presented as optimal (McCandlish App. D says B ∝ √B_noise is) | §2.1 |
| 2026-09-29 | Rollout length | **PROPOSED:** a second controller moves n_steps inside a declared maximum so that D_eff ≥ k·B_noise and there are at least m steps per epoch | Sizing the update's noise by K alone (K cannot remove rollout noise under epoch reuse) | §2.2 |
| 2026-09-29 | Short windows | **PROPOSED:** delayed-label buffer before n_steps < 512 | Shrinking n_steps while the λ = 1 critic drops unfinished-episode rows (~18–35 % at 64) | §3.2; `design_q_head.md` §5 |
| 2026-09-29 | Ragged step and dose count | **PROPOSED:** divisible shapes; count the dose in real steps | The live 2-steps-per-epoch shape with a recorded 1.5 | §3.3; `ppo.py:1430-1448`, `dose.py:12` |
| 2026-09-29 | KL controller | **PROPOSED:** guard-only (down only, capped at the declared rate) | Bidirectional (would raise LR at KL 0.0039, against L21165) or none | §3.7 |
| 2026-09-29 | Advantage normalization | **PROPOSED:** once per rollout | Per micro-batch (K-dependent) | §3.13 |
| 2026-09-29 | Entropy replacement | **PROPOSED:** a trailing magnet (R-NaD/MMD) or a Smogon-prior magnet as X23's arms | A uniform magnet (today's bonus) as the end state | §3.9 |
| 2026-09-29 | Recipe mirror | **BUILT 2026-09-30 (next row); was PROPOSED:** a RECIPE block in `production_config.json` plus a `checkargs` RECIPE SURFACE diff | Recipe values living only in argvs | §3.22; L18192 |
| 2026-09-30 | Recipe mirror — **BUILT** (K10(a)) | A `recipe` block in `production_config.json`: `recipe.fresh` = N0's measured fresh recipe (every training knob `models/ai_v14_01_base` launched with, including the critic, reward values, doses and the KL controller's constants; a key that is also a recorded mirror field must equal it), and `recipe.fork` = E5. `--arch production` applies `recipe.fresh`; `checkargs` / `--dry-run` / the launcher REFUSE untyped drift on a fresh argv (a TYPED difference is the arm's lever; `--allow-nonproduction-recipe` consents); a fork is compared with `recipe.fork` as INFO. A same-run restart of an `--arch production` run takes each untyped knob from exactly one source, announced (INERT `--lr` / `--batch-size` / `--n-steps` / `--gamma` never re-applied; recorded fields from the checkpoint; the rest from `metadata.json:cli_args`), and a MISSING value REFUSES by name. **Value sources:** N0 `metadata.json` (`original_command`, `cli_args`, `dose`) and `model_config.json`; E5 `models/ai_v14_03_lbat_e5`; the §3.22 table cites each key | Five knobs only (misses `grad_accum_steps`, `self_play`, the critic and the doses, each a silent change); pairing E5's 5 epochs with a fresh LR (next row); refusing every typed deviation (every one-lever arm would need a consent flag); falling back to a default when a restart cannot find a value | §3.22; `recipe_surface_test`, `recipe_doc_gate_test` |
| 2026-09-30 | Fresh-run epochs **(orchestrator)** | `recipe.fresh.n_epochs` = N0's measured **10** at a KL-controlled 3e-4; E5's 5 epochs (at a frozen 5.6e-5) live ONLY in `recipe.fork`. **Pairing 5 epochs with a fresh LR was never measured.** The first fresh run on the M5 infrastructure re-reads the epoch count inside the SIZING study. The old parser's 5-epoch default is NOT evidence for 5: it was one of the five silent divergences §3.22 closes | 5 epochs at 3e-4 on a fresh launch (half N0's initial dose, never run) | L21451 (E5 measured on a fork); N0 `dose.n_epochs` 10 |
| 2026-09-30 | `--gamma` under a TYPED `--critic shaped` (cutover loose end) | A BUG, fixed: the discount is PAIRED with the critic, declared once in `agents.model.critic_mode.critic_gamma` (winprob 1.0 — the identity V = P(win); shaped 0.9999 = `PBRS_GAMMA`). `--arch production --critic shaped` now resolves 0.9999 (was recipe.fresh's 1.0); the recipe report calls it `paired`, not silent drift. `--critic winprob` REFUSES a typed gamma other than 1.0 (`combination_checks.winprob_critic_needs_unit_gamma` — `resolve_critic_mode` handed a typed value to the checks, which had no row); an UNTYPED gamma that is not its critic's is a launch `FATAL_CONFIG`; the launch prints `[Critic] gamma=… — <source>`. **Evidence (read-only):** every shaped / pre-critic run in `models/` trained at 0.9999 (205 pre-critic + 3 shaped, incl. the ladder controls `ai_v12_26/27` that took the winprob reward triple), every winprob run (64) at 1.0 — no shaped run ever at 1.0 | intended 1.0 + document it (no run supports the pairing); refusing a typed shaped gamma (it was always that critic's tunable); extending the pairing to the reward triple (`ai_v12_26/27` ran shaped with the indicator triple, so both shaped compositions have precedent) | `critic_gamma_pairing_test.py`; `combination_checks_test` |
| 2026-09-29 | Migration order | **PROPOSED:** Stage 0 safe defaults → SIZING → A1 adaptive K → A2 dynamic n_steps → A3 E2, A4 λ sweep, A5 opponent T → magnet / EWMA / PFSP → PPG / state-dependent λ / search targets | Adopting several levers in one arm | §5 |
| 2026-10-01 | Matmul precision **(owner)** | fp32 only; TF32 RETIRED, its knob and gate paths DELETED (deletion pass K2, 2026-10-02) | TF32 as an opt-in speed knob | T32b FUTILE at +2.48 % [+2.24, +2.91] GPU time per step vs C_fix (L21451); the K9(b) exclusion sweep (2026-10-01, `designs/research_state/measurements/k9_behaviour_exclusion/`): TF32's margin rounding scale is 9.6e-3 relative (~700x fp32), so no tie margin makes the behaviour check deterministic, and the existing TF32 rule false-FATALs on the rust core with trained weights (p99 over its bar in 36% of probes); `program_rust_core.md` Decision record |
| 2026-10-01 | The env core and the run's SIZES live in ONE block, `recipe.sizing` (the M5 switch; SHIPPED 2026-10-02 at the pre-sizing N = 48 shape, orchestrator: the switch does not wait for the sizing verdict, which fills N\* and the verdict-dependent rows in its own commit) | `env_core` rust + N, the n_steps maximum, the collector's update size and T2's slots / buckets / lanes in `recipe.sizing`, applied by `--arch production`; `verdict` null until the SIZING study fills N*; an untyped `--env-core` on a `--model` launch (restart or fork) inherits the checkpoint's core | the sizes left in `recipe.fresh` beside the learning knobs (the verdict would edit two places); a parser-default flip to rust for every argv (a bare argv defaults to `--critic shaped`, which the Rust core refuses) | `program_rust_core.md` order constraint 5; `main/train/env_core_switch_test.py` |
| 2026-10-02 | The SIZING verdict fills `recipe.sizing`; fresh-run epochs (the SIZING study's Part L; registered rules; orchestrator rulings) | **`recipe.sizing`: `n_envs` 256, `n_steps` 384, `rollout_target_samples` 98,304, `verdict` set; `recipe.fresh.n_epochs` stays 10.** (1) N\* = 256: the registered throughput rule on the serial rate; untaught −2.71 pp [−4.50, −0.94] vs N = 48, G-A +0.2 [−3.8, +4.1] ⇒ NO LOSS DETECTED beyond the bar (not equivalence); the replicate floor of 4.90 pp exceeds the 3.69 bar ⇒ flagged "run floor exceeds bar — n = 1 is not decisive". Memory: the arms (pin `7ef99979`) climbed to a steady-state D-6 failure (341.5 MiB), the staged batch's per-update stream; `07eebe13` fixed it (256 + X26 heads flat at 7,798 MiB, D-6 2,218 MiB), so 256 fits the production configuration (orchestrator ruling); learning unaffected. (2) **E10 STAYS:** E5 at a KL-controlled 3e-4 vs E10, N = 256, same seed, matched samples: untaught −10.56 pp [−13.12, −7.94], G-A −9.58 pp [−13.44, −5.68]; E5's update is 2.0× faster. (3) K bounds 2–32 and D_max 262,144 (n_steps_max 1,024 at 256) recorded, NOT applied (arm A1) | E5 on fresh launches; N = 48 kept; the adaptive ceiling applied without arm A1; reading the N guard as equivalence | `designs/research_state/measurements/m5_sizing/` (REGISTRATION §5–6, PROGRESS); `program_rust_core.md` Decision record 2026-10-02; ledger 2026-10-02 |
| 2026-10-02 | The BARE-ARGV default **(owner, deletion pass D2)** | A fresh argv without `--arch production` defaults to `--critic winprob` + `--terminal-indicator --victory-value 1.0 --draw-penalty 0` + `--env-core rust` (`gen3_bare_argv_winprob_v1`, config v130): the bare parser and the production surface agree on the critic, its terminal and the core, so the `--debug` smoke runs the Rust core. What an ABSENT record means did NOT move (`critic_mode.CRITIC_UNRECORDED` = shaped; `_REWARD_IMMUTABLE_FIELDS`), so no checkpoint loads differently and a flagless resume reads its recorded critic and terminal; no `ARCH_SIGNATURE` bump | Keep `python` / `shaped` as the bare default until the Python core is deleted (the 2026-10-01 rows' interim — a bare argv with nowhere to run once it goes); make the Rust core accept `shaped` (D1 deletes it instead) | `designs/ops/deletion_pass_manifest.md` §0 D2, §2.1; `main/critic_mode_config_test.py`, `main/reward_defaults_test.py`, `main/train/env_core_switch_test.py`; the `--debug --steps 10000` smoke on the Rust core (Training complete, K9(b) excluded share 1.6–4.5 % < 0.15) |
| 2026-10-03 | `--env-core` is DELETED; `env_core` leaves `recipe.sizing` **(deletion pass P11b, batch (a); orchestrator, owner asleep)** | The Rust env core is the only core, so the flag had one legal value and the recipe a row that could not vary: `recipe.sizing` now holds only the SIZES (`n_envs`, `n_steps`, `rollout_target_samples`, `trainee_slots`, `t2_buckets`, `t2_lanes`, `verdict`); `recipe_blocks` refuses a stray `env_core` key by name; the restart route no longer restores a core from `cli_args`. What stays keyed on the RECORD: `metadata.json`'s `env_core` stamp, the CORE SWITCH announcement for a python-era winprob checkpoint, and the D4 refusal of a shaped-critic checkpoint (`rust_env_setup.refuse_python_era_checkpoint`). No `ARCH_SIGNATURE` / `MODEL_CONFIG_VERSION` change (it was never a recorded `ModelVersion` field) | Keeping a one-valued row "for the record" (a row the gate can only ever confirm); a hidden tolerated spelling of `--env-core rust` (a flag that stays is a flag the census counts) | `designs/ops/flag_census.md` (P11b (a)); `src/main/train/env_core_switch_test.py`, `census_deleted_flags_test.py` |
| 2026-10-03 | `--critic` is DELETED; `critic` leaves `recipe.fresh` **(deletion pass P11b, batch (b))** | The win-prob critic is a constant of the trainer namespace (`src/main/train/parser/objective.py`, `parser.set_defaults`); `recipe.fresh` holds no `critic` row; the typed `--critic` is refused with its reason (`designs/deleted_flags.md`); a recorded shaped checkpoint is refused on a resume or fork (D4) and an absent record still means shaped (`CRITIC_UNRECORDED`); no `MODEL_CONFIG_VERSION` change | A hidden tolerated `--critic winprob` spelling (a flag that stays is a flag the census counts) | `designs/ops/flag_census.md` (P11b (b)); `src/main/train/retired_flags_test.py`, `src/main/critic_mode_config_test.py` |
| 2026-10-03 | `--gamma`, `--victory-value`, `--draw-penalty` and `--terminal-indicator` are DELETED; their rows leave `recipe.fresh` (deletion pass P11b batch (c)) | They are constants of the namespace (`parser/objective.py`), `recipe.fresh` holds none of them, a typed one is refused with its reason, `check_reward_config` refuses a recorded non-production reward with no flag to re-pass, SB3 still restores a checkpoint's own gamma, no `MODEL_CONFIG_VERSION` change | Keeping gamma as a research lever (no registered experiment varies it; `winprob_critic_needs_unit_gamma` refused every value but 1.0); a tolerated production-value spelling | `designs/ops/flag_census.md` (P11b (c)), `src/main/train/objective_constants_test.py`, `src/main/reward_defaults_test.py` |
