# Design: Self-Play Opponent Distillation

Self-play training is **rollout-bound**, and the dominant per-decision cost is the **opponent's
neural-network forward pass run on CPU inside every env worker**. This document proposes spending
the otherwise-idle GPU (offline, asynchronously) to produce a **cheap distilled opponent** that
runs ~5–10× faster on the CPU critical path — buying training throughput **without touching the
trainee** and without changing PPO. It recommends a concrete path (soft policy distillation into a
small policy-head-only network), the data and pipeline, and — the part that matters most — **how we
measure whether the distillation is any good**.

This is purely a *training-speed* optimization. The trainee's architecture, observations, reward,
and learning algorithm are unchanged. The only thing that changes is the opponent it spars against,
and the central risk the metrics below guard against is **degrading that opponent** (making it
weaker, less diverse, or more exploitable) in a way that hurts the curriculum.

---

## 1. Why — the cost we're attacking

Live py-spy profiling of the self-play run (`run_20260601_193826`, 64 envs) established:

- **~86% of wall-clock is rollout collection; ~14% is the PPO update.** The GPU is idle ~86% of
  the time (2–4% util during rollout, 100% only in the update).
- Per env worker: **~37% busy on-CPU, ~63% blocked** (≈47% at the SB3 per-step barrier, ≈16% on
  the Showdown sim round-trip). Of the busy CPU, **~70% is the self-play opponent's forward pass**
  (`SingleAgentWrapper.step → opponent.choose_move → RLPlayer._predict_best_action →
  policy.get_distribution → Gen3FeaturesExtractor.forward`) — **~26% of total wall per worker**.
- obs build is only ~2–4% of worker wall here; the opponent forward dwarfs it. (This contradicts
  the obs-build benchmark gate, which uses a random action and *no opponent net*.)
- The workload is **latency-bound**: the per-step barrier makes each rollout step run at the speed
  of the slowest of N workers, so cutting per-decision latency on the critical path raises FPS even
  though the CPU is not fully saturated. n-envs is *not* an FPS lever (48 ≈ 64; the box is
  latency-bound, not throughput-bound).

**The opponent is a frozen `SnapshotPool` snapshot** (up to `max_snapshots` ≈ 14 distinct), each
worker loading one per `pool_generation` (LRU-cached, reused across episodes — per-episode loading
was a known FPS killer). Crucially, **the opponent only ever uses the policy head** — it picks
moves via masked logits and never calls the value head.

The lever: the opponent forward is on the **per-decision critical path** of every worker, and the
GPU that could compute a cheaper substitute sits idle 86% of the time. After the gradient-
checkpointing change (Phase: VRAM 10.2 GB → 7 GB, ~5 GB free), there is now headroom for an async
GPU distillation job to run alongside training.

**Target:** a distilled opponent whose forward is **≥5× cheaper** on CPU → opponent forward falls
from ~26% to ~5% of worker wall → frees ~20% of rollout wall-clock, which (because rollout is the
critical path) should translate to a meaningful FPS gain. The exact gain is an empirical question
the Phase-3 A/B answers; **≥5× cheaper-per-decision** is the engineering target, **net FPS up with
no measurable learning regression** is the success criterion.

---

## 2. The decision: distill the *policy* or distill *behaviors*?

Two independent axes. Naming them apart is what makes the choice clear.

### Axis A — training signal: soft distribution vs hard actions

| | **Soft (policy distillation)** | **Hard (behavioral cloning)** |
|---|---|---|
| Target | teacher's full masked action **distribution** (logits) | the teacher's **chosen action** (argmax/sample) |
| Loss | `KL(teacher ‖ student)` over legal actions, at temp | cross-entropy on the one-hot action |
| Preserves | calibration, entropy, the *shape* of the distribution | only the mode; entropy is implicit, learned from frequencies |
| Data efficiency | high (each state carries a full soft target) | low (1 bit of signal per state) |
| Risk for a *sampled* opponent | low | **high — a BC student trained on argmax is near-deterministic → exploitable** |

**The opponent is sampled stochastically at `--self-play-temp` (default 1.0).** Its *distribution*
is the product — a rich, varied, hard-to-exploit sparring partner. Behavioral cloning throws that
away and tends toward a deterministic opponent the trainee can overfit to beating. **→ Use soft
policy distillation (KL on the full masked distribution).**

### Axis B — student form: small network vs non-network

| | **Small NN (slimmed `Gen3FeaturesExtractor`)** | **Non-NN (GBDT / tree / linear over features)** |
|---|---|---|
| Speed-up | ~5–10× | ~50–100× (no torch overhead) |
| Fidelity on 3357-dim structured obs + embeddings + masking | high (right inductive bias) | **risky** — categorical IDs, per-slot structure, and masking are awkward for trees/linear |
| Integration | **drop-in**: still produces masked logits via `get_distribution`; `RLPlayer` swaps it for the snapshot unchanged | bespoke decision fn returning masked logits/action; breaks the torch path |
| Controllability | size the net to a CPU budget; smooth fidelity/speed trade | coarse |

A non-NN opponent is tempting for raw speed but high-risk for fidelity on this structured input,
and it breaks the clean `_predict_best_action` seam. **→ Use a small NN student**, sized to a CPU
budget, as a drop-in.

### Recommendation

**Soft policy distillation into a small, policy-head-only NN student.** "Distill the policy"
(match the distribution) realized as "a smaller network." **Behavior distillation** (hard targets
and/or a non-NN form) is the lossier fallback, considered only if a small NN can't hit the CPU
budget while passing the fidelity gate (Phase 0 answers this before we commit).

> A deeper reason soft-policy + on-distribution data wins here: a sparring opponent must react
> *correctly to whatever board the trainee creates*, including weird states the teacher never
> visited in its own games. Distilling the **state→distribution mapping** (with states drawn from
> trainee-vs-opponent play) covers that; cloning the teacher's own **trajectories** would only
> cover states the teacher itself reached. This shapes the data strategy (§4).

---

## 3. The student architecture

Start from `Gen3FeaturesExtractor` and cut to a CPU budget, keeping the **input obs (3357-dim) and
the masked-logits output interface identical** so it drops into `RLPlayer` unchanged:

- **Drop the value head entirely** — the `value_cls` pool, value projection, and value MLP. The
  opponent never calls `predict_values`. (Free ~⅓ of the head compute and the value CLS attention.)
- **Slim the body** — the profiled hot frames were the per-slot `PokemonEncoder` (12 Pokémon ×
  batch) and the `TeamTransformer`. Candidate cuts, searched smallest-first under the fidelity gate:
  - `TRANSFORMER_N_LAYERS` 2 → 1, or replace team attention with mean/attention-pool.
  - smaller `D_MODEL` (128 → 64) / `TRANSFORMER_FFN_DIM`.
  - thinner `ROLE_ENCODER_HIDDEN` / move processor; possibly fold the within-Pokémon move
    self-attention into a pool.
- **Keep the embedding tables** — they are cheap lookups, not the cost, and the categorical IDs
  (species/move/item) need them. Consider sharing/initialising from the teacher's tables.
- **Own `ARCH_SIGNATURE`** (e.g. `gen3_distilled_v1`) so `load_model_snapshot`'s version check is
  honest; the student is a *separate small arch*, not a change to the trainee's arch (which stays
  full — this doc never touches the learner's network).

The student is sized empirically: **the smallest architecture that still passes the fidelity +
behavioral gates (§5)**. Phase 0 sweeps 2–3 sizes to find the knee.

---

## 4. Data — on-distribution states

The student must match the teacher on the states the opponent **actually faces during training**
(the trainee-vs-opponent distribution), not on some generic distribution. Two ways to get them:

- **(A) Live logging (DAgger-like) — recommended primary.** The opponent forward already runs every
  step in the worker, and there is already a `need_aux` hook (`RLPlayer._last_prediction` =
  `{obs, logits}`) used by the prober. A slim variant dumps `(obs, masked_logits, mask)` to a
  sharded file during rollout. **Teacher labels come for free** (the forward already happened), the
  states are exactly on-distribution, and because the *improving trainee* drives them, the dataset
  **tracks distribution shift automatically**. Cost: a little I/O on the worker hot path (mitigated
  by ring-buffer + async flush, and sampling, e.g. log 1-in-k decisions).
- **(B) Offline bridge generation — cold-start / augmentation.** A separate process plays
  teacher-vs-{trainee, pool, bots} via the **server-free in-process bridge** (`utils/bridge/`, no
  `:8001` contention), collects opponent-perspective states, and labels them on the GPU. More
  control over coverage; fully decoupled. Useful to seed a student before any live logs exist, or to
  deliberately broaden coverage. (The prober's `eval_traces/*/states.npz` is a third ready-made
  source for a quick start.)

**Distribution shift is the central data risk.** A student distilled once on a fixed dataset can
diverge from the teacher on states the *now-stronger* trainee later drives the opponent into. The
sliding pool bounds this (each snapshot is an opponent only briefly), and live-logging refreshes the
dataset continuously; if drift still shows up in the gates, re-distill on a cadence.

---

## 5. How we know we're doing a good job

Three tiers, cheapest first. **A distilled opponent ships only if it passes all three.** Every
distill run writes a `distill_report.json` with these numbers next to the `*.distilled.zip`.

### Tier 1 — Fidelity (cheap, offline, held-out on-distribution states)
Computed in the labeling pass; basically free. "Does the student make the same decisions, with the
same confidence?"

| metric | meaning | target |
|---|---|---|
| **top-1 agreement** | `mean[ argmax(student) == argmax(teacher) ]` over legal actions | **≥ 0.92** |
| **`KL(teacher ‖ student)`** | distribution match at temp `--self-play-temp` | **≤ 0.05 nats** (median) |
| **top-3 overlap** | the move it *considers*, not just picks | ≥ 0.95 |
| **entropy ratio** | `H(student)/H(teacher)` — guards against a peaky, exploitable student | **0.9–1.1** |
| **calibration (ECE)** | the student isn't over/under-confident | ≤ teacher + small ε |

Fidelity is necessary but **not sufficient** — it can't tell you whether the residual gap *matters*.

### Tier 2 — Behavioral (the real signal, via the existing `eval_worker` + sentinels)
The student is just another snapshot to evaluate. "Same strength, same playstyle?"

| metric | how | target |
|---|---|---|
| **head-to-head vs teacher** | distilled (greedy) vs full teacher (greedy), K bridge battles | **win rate ∈ [0.45, 0.55]** (~0.5 = faithful; deviation = strength delta, *signed*) |
| **gauntlet Δ win-rate** | student vs bot roster + pool sentinels, compared to the teacher vs the same set | **|Δ `win_rate_vs_bots`| < 0.03**, **|Δ `win_rate_vs_pool`| < 0.03** — in the curriculum's own units |

The head-to-head being *signed* matters: a student that is *weaker* than its teacher makes the
opponent easier (curriculum regression); *stronger* is also wrong (it's no longer that snapshot).

### Tier 3 — Does it actually help training? (the ultimate A/B)
The only test that closes the loop. Train the same trainee for a fixed budget, distilled opponents
vs full opponents, and compare:

- **the win (throughput):** opponent-forward CPU cost (py-spy) and **FPS** — distilled should be
  materially higher. This is the whole point.
- **the cost (learning):** `eval/win_rate_vs_bots` learning curve and final value, and
  `eval/sentinel_monotonicity`. Distilled must be **statistically indistinguishable** from full.

**Verdict:** distillation is "good" iff **FPS up AND learning not measurably hurt**. If FPS gain is
below some floor (say < 15%), it isn't worth the moving parts — say so and stop (see Risks).

### Acceptance gate (per snapshot)
Deploy the distilled student as the training opponent **only if** Tier 1 + Tier 2 pass; otherwise
keep the full snapshot for that pool entry and log why. Either way the report accumulates a
"how distillable is each snapshot" record — itself a useful signal (a snapshot that won't distill
faithfully is telling you something about its policy's complexity).

---

## 6. The async pipeline (trade GPU for CPU)

Hang a non-blocking GPU job off **snapshot promotion** (`SnapshotPool.add_from_path`), mirroring the
existing non-blocking eval/self-play subprocess pattern:

```
promote(snapshot)  ──▶  distill_job(snapshot)            [separate process, GPU]
                          1. gather states     (live logs §4A, or bridge §4B)
                          2. teacher labels    (GPU, batched)   → cache (obs, soft_labels, mask) to disk
                          3. train student     (GPU, batched SGD over cached labels; KL loss at temp)
                          4. Tier-1 fidelity   (free, same pass)
                          5. Tier-2 behavioral (existing eval_worker, CPU, bridge)
                          6. gate + write      snapshot_XXXX.distilled.zip + distill_report.json
```

- **VRAM:** ~5 GB free now. **Cache-labels-then-train** keeps peak low — only the labeling phase
  holds the teacher on-GPU (shrink its batch to fit), then it's freed and student training is
  VRAM-trivial. Alternatively run the whole job in a **launcher-restart window** when the training
  child has released all GPU memory.
- **Propagation:** the worker's `RLPlayer` loads `*.distilled.zip` (when present and gate-passed) in
  place of the full snapshot, on the next `pool_generation` bump — the *existing* mechanism that
  already propagates promotions. No new plumbing in the hot path; the student exposes the same
  `get_distribution`/masked-logits interface, so `_predict_best_action` is unchanged.
- **Per-snapshot students** (≈14 tiny nets) to preserve the pool diversity that makes self-play
  work. A single net distilled across the whole pool would collapse that diversity — see the future
  extension (a *style-conditioned* student) below.

---

## 7. Guardrails — doing this robustly (fail-closed)

The distillation path must **never be able to silently degrade training.** It is strictly
*additive*: its only job is to replace a verified-equivalent full opponent with a cheaper one, and
on any doubt it falls back to the full model. A distilled opponent can only ever be *as good as or
better-monitored than* the full model it replaces — never worse, because rejection/uncertainty
keeps the full model. The mechanisms that enforce that:

1. **Fail-closed gate — default is the full model.** A snapshot's training opponent is the full
   model *unless* a distilled student has **passed** the acceptance gate (§5). Distillation can only
   *upgrade* a passing snapshot to cheaper; it can never make an opponent worse than full, because
   the failure mode is "keep full." No distilled student is ever used un-gated.
2. **Anchor opponents — never 100% distilled.** A fixed floor (e.g. **≥ 20%** of training
   opponents) always draws the *full* model, so even a subtly-degraded distilled pool can't fully
   define the curriculum. The league/exploitability literature is explicit that robustness needs
   ground-truth opponents in the mix.
3. **Canary rollout, not a flip.** Never switch the whole pool at once. A newly-distilled student
   enters as a *minority* of opponents for its snapshot; it earns a larger share only after the
   downstream guards (below) stay green for K eval cycles. New ≠ trusted.
4. **Continuous re-verification + auto-revert (the distribution-shift guard).** The gate is **not
   one-shot.** Each `pool_generation`, re-run Tier-1 fidelity on *fresh* live-logged states for every
   active distilled student; if fidelity has decayed past threshold (the trainee has moved on),
   that snapshot **auto-reverts to its full model** and a re-distill is queued. This is the
   operational form of the Ross–Bagnell on-distribution requirement — stale ≈ off-distribution ≈
   compounding error.
5. **Downstream kill-switch.** Wire the *existing* bot-regression guard (`⚠️ BOT_REGRESSION`),
   `sentinel_monotonicity`, and `train/entropy` to an **automatic global revert**: if any regresses
   past its floor *after* a distilled rollout, revert the whole pool to full opponents and alert.
   The downstream signal is the principled gate (per the distillation literature — performance
   parity, not fidelity), so it must drive *action*, not just a chart.
6. **Fail-safe pipeline — never fatal.** The distill job is a non-blocking subprocess (like eval).
   A crash / timeout / OOM is **logged-and-continued**; training proceeds on full opponents.
   Distillation failing is a perf non-event, never a training outage. Mirrors the eval-worker
   contract.
7. **Provenance on every opponent.** Each `*.distilled.zip` carries a manifest (teacher snapshot,
   distill step, #data-states, fidelity + behavioral numbers, `ARCH_SIGNATURE`, git hash), and the
   worker records *which* opponent variant (full vs `distilled-vN`) each episode actually used — so
   any anomaly is traceable to the exact opponent. A stale-arch student fails its version check
   rather than loading silently.
8. **Deterministic, fixed gate.** The acceptance eval uses a **fixed held-out state set + fixed
   gauntlet seed** so the gate is comparable across snapshots and runs. A noisy gate would both leak
   bad students and reject good ones.
9. **Exploitability probe (stretch).** Beyond head-to-head, periodically train a cheap best-response
   exploiter against the distilled student *and* against the teacher for a few hundred steps, and
   compare how easily each is beaten. A distilled opponent that is **more exploitable than its
   teacher is rejected** even if top-1/KL look fine — directly targeting the failure the league/PBT
   work warns about.
10. **Bounded blast radius.** Per-snapshot students mean a bad distill affects only opponents drawn
    from that *one* snapshot (a minority at any moment), never the whole pool.

**Failure modes these address:**

| risk | covered by |
|---|---|
| **Distribution shift** — student diverges as the trainee improves | (4) continuous re-verify + auto-revert; live-logging data (§4A); sliding pool bounds exposure |
| **Diversity collapse** — distilled pool less varied than the real pool | (2) anchors; (10) per-snapshot students; soft KL preserves each distribution (vs hard BC) |
| **Exploitability** — a weaker/peakier opponent the trainee overfits to | (9) exploitability probe; entropy-ratio gate; (2) full-model anchors |
| **Silent downstream regression** | (5) kill-switch on `win_rate_vs_bots` / monotonicity / entropy |
| **It isn't worth it** — FPS gain too small | Phase-3 A/B with a hard FPS-gain floor; if below floor, ship nothing |
| **Staleness across generations** | (1)(7) gate re-runs per promotion; stale `*.distilled.zip` fails its version check |
| **Pipeline failure** | (6) non-blocking, logged-and-continued, full-opponent fallback |
| **Hot-path I/O from live logging** | ring buffer + async flush + 1-in-k sampling; or offline bridge generation (§4B) |

---

## 8. Phased plan

- **Phase 0 — Measurement harness (no pipeline, learner untouched).** A standalone script: load one
  pool snapshot as teacher, build 2–3 candidate small students, distill on bridge-generated states,
  print Tier-1 fidelity + a Tier-2 head-to-head. **Answers the make-or-break question — can a small
  arch reproduce a snapshot faithfully, and at what size? — before building anything.** Also profiles
  the student forward (py-spy) to confirm the ≥5× CPU win.
- **Phase 1 — Async pipeline + gate.** Trigger on promotion; label→train→evaluate→gate→write;
  per-snapshot students; `RLPlayer` loads gate-passed `*.distilled.zip`; offline (bridge) data.
- **Phase 2 — Live-logging data path (DAgger).** Slim `need_aux` logging in the workers; on-
  distribution datasets; re-distill cadence.
- **Phase 3 — The A/B and the verdict.** Distilled-vs-full training A/B (Tier 3); ship as default
  only if FPS up and learning unhurt.
- **Future — Single style-conditioned student.** One net taking a snapshot/style embedding,
  reproducing any pool member — avoids N loads while (with the conditioning) preserving diversity.
  Bigger build; only if per-snapshot students become the limiting overhead.

---

## 9. Observability — TensorBoard metrics & alerts

Distillation is invisible unless instrumented, and every guardrail in §7 (the gate, the canary, the
auto-revert, the kill-switch) needs a **signal to act on**. Everything below lives under a
`distill/` namespace, logged **each eval cycle**, per-snapshot where it makes sense **plus a pool
aggregate** (so one bad student shows up even when the mean looks fine — always log the *worst*
active student, not just the mean). These piggyback on `SelfPlayCallback`'s existing TensorBoard
writer.

**Fidelity — re-measured every generation on FRESH live-logged states (not the distill-time set):**
- `distill/top1_agreement` (pool mean) · `distill/top1_agreement_min` (worst active student)
- `distill/kl_teacher_student` (mean) · `distill/kl_p90`
- `distill/entropy_ratio` — student/teacher entropy; the **exploitability leading indicator** (a
  peaky student is the danger, §7-9)
- `distill/top3_overlap`
- `distill/fidelity_decay` — current top-1 **minus** top-1 at distill time; the drift detector that
  *drives* the §7-4 auto-revert

**Behavioral — the principled gate (from the eval workers):**
- `distill/head_to_head_winrate` (vs teacher; **target ≈ 0.50**) — per snapshot + mean
- `distill/gauntlet_winrate_delta` — (student vs bots) − (teacher vs bots)
- `distill/exploitability_gap` — best-response WR vs student − vs teacher (if the §7-9 probe is built)

**Speed — the actual payoff:**
- `distill/opponent_forward_ms_full` vs `distill/opponent_forward_ms_distilled` · `distill/speedup`
- `train/fps` (existing) — the headline; the whole point is this **rises**. Emit a TB *event marker*
  on each rollout/revert so the FPS step-change is unmistakable.

**Coverage / deployment state (so the canary + anchors are auditable):**
- `distill/frac_opponents_distilled` — live share of opponents using a distilled net (vs the §7-2
  anchor floor)
- `distill/snapshots_distilled` · `distill/snapshots_rejected` · `distill/auto_reverts_total`
- `distill/rejections_by_reason` — histogram: fidelity / head-to-head / exploitability

**Pipeline health:**
- `distill/job_duration_s` · `distill/job_failures_total` · `distill/data_states_collected` ·
  `distill/vram_peak_mb` · `distill/last_distill_step`

**Downstream safety — existing metrics, now wired to the §7-5 kill-switch:**
- `eval/win_rate_vs_bots` · `eval/sentinel_monotonicity` · `train/entropy` — watched for regression
  *after* a rollout; a breach auto-reverts the pool to full opponents.

### Alerts & actions (the watchlist, with triggers)

| Signal | Healthy | Trigger → action |
|---|---|---|
| `distill/head_to_head_winrate` | 0.45–0.55 | outside band → **gate rejects** (or auto-revert if live); keep full snapshot |
| `distill/entropy_ratio` | 0.9–1.1 | < 0.9 (peaky/exploitable) → reject; re-distill at higher temp / more data |
| `distill/top1_agreement_min` | ≥ 0.92 | below → grow student / more data; if persistent, arch too small |
| `distill/fidelity_decay` | ≥ −0.03 | drop past −0.05 → **auto-revert that snapshot** + queue re-distill (drift) |
| `distill/gauntlet_winrate_delta` | \|Δ\| < 0.03 | exceed → opponent too weak/strong → reject |
| `distill/speedup` / `train/fps` | ≥ 5× / FPS materially up | FPS gain < ~15% → **not worth it, ship nothing** (§8 Phase-3) |
| `distill/frac_opponents_distilled` | ≤ (1 − anchor floor) | at ceiling with guards green → widen canary; any guard red → freeze/roll back |
| `eval/win_rate_vs_bots` (post-rollout) | no regression vs full baseline | drop > X → **kill-switch: global revert to full** + alert |
| `eval/sentinel_monotonicity` | ≥ 0.6 | sustained drop → diversity collapse → revert / lower distilled share |
| `train/entropy` | stable | collapsing under distilled opponents → revert + inspect |
| `distill/job_failures_total` | flat | rising → pipeline broken, but **training is unaffected** (full-opponent fallback) — fix offline |

---

## 10. What this is *not*

- **Not a change to the trainee.** The learner's architecture, obs, reward, and PPO are untouched;
  only the sparring opponent's network is substituted. No `ARCH_SIGNATURE` change to the trainee.
- **Not the GPU-batched-inference-server approach.** Hoisting the opponent forward into one central
  batched GPU call is a *different* lever, and the profiling found it marginal at this scale (tiny
  batches split ≤14 ways by snapshot, plus IPC and a new barrier). Distillation keeps the forward in
  the worker but makes it cheap — no gather/scatter barrier.
- **Not a correctness change to learning.** If the opponent stays faithful (the gates enforce this),
  the trainee's experience distribution is preserved; distillation only changes *how fast* the
  opponent decides, not *what* a faithful opponent would decide.

---

*Relationship to the rest of ai_v5:* the **league tooling** (`design_league_tooling.md`) payoff-
matrix / Nash metrics are a natural Tier-2+ measure of whether a distilled pool preserves the
strategic structure of the real pool; **reward annealing** (`design_reward_annealing.md`) is
independent. This optimization is orthogonal to Step 2 league play and can land whenever opponent-
forward CPU is the throughput limiter.
