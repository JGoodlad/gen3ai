# Distillation — Integration & Deployment Plan

Companion to `design_opponent_distillation.md` (the forward design) and the empirical record
(`distill_findings.md`). This doc is the **concrete "how to wire it in"** — the validated recipe,
where it runs, how the cheap opponent loads into the pool, and how to calibrate the gate given the
*measured* ~0.44 fidelity ceiling. It assumes the exploration's headline result:

> **`cheaper_encoder`: ~4.67× cheaper opponent forward (0.41 ms vs teacher ~1.93 ms), head-to-head
> 0.443 ± 0.056 (N=300) — faithful-*ish* (slightly weaker than the teacher), not a perfect 0.50 draw.**

The payoff: the opponent forward is ~26% of env-worker wall (~70% of its on-CPU work), so a 4.67×
opponent is an estimated **~+15–25% rollout throughput at zero quality cost** while the gate holds.

---

## 1. The validated recipe (what to build)

Two-stage, both stages **offline on the GPU** (free; we're CPU-bound, so spending an hour of GPU to
shave CPU inference is a great trade):

1. **Cheap encoder via behavioral distillation.** Train a small (~131K-param) shared per-slot MLP
   encoder to reproduce the teacher's frozen `pokemon_encoder` output — the **12×128 role tokens** —
   by MSE over a state dataset (reached cosine 0.956). This is the load-bearing step: direct
   token-target supervision broke the cheap-encoder ceiling (top-1 0.757 → 0.858) that logit-KL
   alone could not.
2. **Matchup-aware policy head via soft-KL.** Freeze the cheap encoder; train a head that reads the
   role tokens **plus** the opp-active role token and the per-move type-matchup slices
   (`ctx.matchups_all`), distilled to the teacher's masked logits at **temperature 0.7 + label
   smoothing 0.02**.

> Reuse-encoder reuse sets the fidelity ceiling; the matchup signal moves *live play* (h2h) more
> than static top-1; the cheap encoder is where the *speed* comes from (the transformer is **not**
> the bottleneck — cutting it bought ~0; the encoder/unpack is ~63% of the forward).

**Inputs:** a state dataset of (obs, teacher masked-logits, teacher role tokens) for the snapshot
being distilled. Prefer **on-distribution + DAgger** states (see §4) over a generic set.

---

## 2. Where it runs — the async pipeline (no training-loop coupling)

Hang a non-blocking GPU job off **snapshot promotion** (`SnapshotPool.add_from_path`), exactly
mirroring the existing non-blocking eval/self-play subprocess pattern (`spawn_eval_workers` /
`merge_eval_results`). The training thread's only added work is firing the job.

```
promote(snapshot S)
  └─▶ distill_job(S)              [detached process, GPU]
        1. gather states          (live-logged opponent decisions for S, or bridge-generated)
        2. cache teacher targets  (role tokens + masked logits) on GPU
        3. stage-1: distill cheap encoder (MSE on role tokens)     ← GPU, thorough
        4. stage-2: distill matchup head (soft-KL T=0.7)           ← GPU
        5. GATE eval              (eval_worker: h2h vs the teacher, N≥300 battles)
        6. if pass → write models/<run>/distilled/S.distilled.pt + S.distilled.json (manifest)
```

- **VRAM:** trivial (small student + cached targets). Runs comfortably alongside training, or in a
  launcher-restart window. The offline cost is irrelevant — only the resulting CPU inference ms
  matters.
- **Failure is a non-event:** crash/timeout → logged-and-continued, the pool keeps the full teacher
  for S. Distillation is strictly additive.

---

## 3. How the cheap opponent loads into the pool (the seam)

The opponent path today: `SnapshotPool.load_model(entry)` → a `MaskablePPO`; the worker's
`RLPlayer` calls `self.model.policy.get_distribution({obs, action_mask}).distribution.logits`;
`SingleAgentWrapper.step` polls `opponent.choose_move`. The distilled student is **not** a
`MaskablePPO`, so wrap it to satisfy that one interface:

```python
# a thin adapter so RLPlayer is unchanged — same shape it already expects
class _DistilledModel:
    def __init__(self, student, device="cpu"):
        self.policy = _StudentPolicy(student)   # .get_distribution(obs)->obj with .distribution.logits
        self.device = torch.device(device)
```

Extend the pool's loader: **`SnapshotPool.load_model(entry)` returns the distilled student
(adapter-wrapped) iff `S.distilled.pt` exists AND its manifest passed the gate; else the full
`MaskablePPO`.** Keep the existing per-`pool_generation` LRU caching — distilled students are ~10×
smaller, so caching more of them is cheap. Workers pick up the distilled variant on the next
`pool_generation` bump, the *same* mechanism that already propagates promotions. **Nothing in the
hot path changes** except the forward got cheap; `_predict_best_action`, masking, the stale-decision
re-decide, and `set_self_play_target` are all untouched.

**Provenance:** `S.distilled.json` carries teacher snapshot, distill step, #states, the gate metrics
(top-1, KL, ent_ratio, h2h ± CI, N), `ARCH_SIGNATURE` for the student, and git hash — so the worker
can log exactly which opponent variant (full vs `distilled-vN`) each episode used.

---

## 4. Calibrating the gate — the honest part (h2h sits at ~0.44, *below* 0.45)

The measured faithful ceiling is **~0.44**, just under the design's [0.45,0.55] band: the cheap
opponent is *slightly weaker* than the teacher. A strict band would **reject every** distilled
student. That's the wrong call here — the alternative is paying **4.67×** the CPU for the full
teacher, and a 0.44 sparring partner is only marginally easier (it wins ~44% of mirror matches).
So calibrate for **"faithful-ish and safe"**, not "perfect draw":

**Accept a distilled student iff ALL hold (fail-closed otherwise):**
- **h2h CI overlaps [0.45,0.55]** at **N ≥ 300** battles (the 0.443 ± 0.056 CI = [0.387, 0.500]
  qualifies; a student at 0.35 ± 0.05 would not), **and**
- **ent_ratio ∈ [0.9, 1.1]** (not collapsed/peaky → not exploitable), **and**
- **not more exploitable than the teacher** — a fixed bot beats the student no more than ~+5 pts
  more than it beats the teacher (the league-play failure mode).

**Auto-revert (the live kill-switch):** during training, track the distilled opponent's live
win-rate-vs-teacher (or vs the bot roster). If it **drifts below ~0.40** for a snapshot, revert that
snapshot to the full teacher and queue a re-distill. The point estimate at 0.44 means the *revert*
threshold (0.40), not the *accept* band edge (0.45), is the operative safety line.

**Anchor:** keep ≥20% of training opponents on the **full** model regardless, so even a fleet of
slightly-weak distilled opponents can't fully define the curriculum.

The full metric set + alert→action wiring is in `design_opponent_distillation.md` §9; this section
just sets the **numeric thresholds** to the measured reality.

---

## 5. Distribution shift & re-distill cadence (the amortized cost)

Each pool snapshot is a distinct policy → **each needs its own distilled student** (a 72M-distilled
student won't match a 70M snapshot). The cost is therefore (distill-job time) × (promotion rate).
Two shift sources, two mitigations:

- **Trainee drives the opponent off-distribution as it improves** → use **DAgger** data (states the
  distilled student actually visits when playing, labeled by the teacher). The ~0.44 ceiling's
  signature is compounding error, so on-distribution data is the principled lever (under test).
- **Pool advances** → re-distill per promotion, **opportunistically**: attempt every generation,
  *accept* only if the gate passes at N≥300, else fall back to the full teacher for that generation.
  Self-play throughput then degrades **gracefully** (some generations cheap, some full) instead of
  silently sparring a weak ghost.

A drift-vs-generation measurement (how fast a frozen distilled student decays across teacher
checkpoints) sizes this — queued as an experiment.

---

## 6. Build order

1. **Land the offline distill harness** (the two-stage recipe + the gate eval via `eval_worker`) as
   a standalone tool — no training-loop coupling yet. Validate it reproduces the 4.67× / ~0.44 result.
2. **`SnapshotPool` adapter + loader change** (§3) behind a `--distill-opponents` flag, default off.
3. **Trigger + manifest + gate** (§2, §4) — fail-closed, anchored, auto-revert.
4. **Observability + restart resilience** (§7 + `design_opponent_distillation.md` §9) — the `distill/*`
   TensorBoard series, the rollout/revert event markers, and the resume re-publish + `summary.json`
   state. Wire this *with* the gate, not after — a gate with no telemetry is unshippable.
5. **Measure the real FPS delta** in the integrated loop (the analytical ~+15–25% needs confirming)
   and the live h2h drift — these decide whether to keep it on.
6. **DAgger data path** (§5) only if the ~0.44 needs pushing toward 0.50.

Everything before step 5 is reversible and flag-gated; the live measurement decides adoption.

---

## 7. Restart resilience (the launcher restarts every N hours for fragmentation)

The launcher SIGTERMs the child every `--restart-interval-hours` (pymalloc-fragmentation reclaim;
the child checkpoints + relaunches) and auto-restarts on crash. The distillation layer must survive
that transparently. It **mirrors how the existing eval/self-play pipeline already handles restarts**
(graceful drain, file-reconstructed pool, `replay_last_eval_to_tui` re-publish) — so this is wiring
into a proven pattern, not new machinery:

1. **State is files → survives for free.** Distilled students (`models/<run>/distilled/S.distilled.pt`)
   and manifests (`.json`, carrying the gate verdict) live in the run dir, exactly like the pool's
   `snapshots/`. On restart the layer **rescans `distilled/`** and reconstructs "which snapshot is
   distilled-deployed vs full" from the manifest pass-flags — no in-memory state to lose (same
   pattern as `SnapshotPool` reconstructing from `snapshots/`).
2. **In-flight distill job at SIGTERM → abandon, don't block.** A distill job is a detached
   subprocess; on graceful shutdown it is **not waited on** (unlike a gate-eval drain): an
   interrupted distill simply leaves no `S.distilled.pt`, which the loader reads as "not distilled
   yet → full teacher for S," and it **re-triggers idempotently** after resume (a pool snapshot
   lacking a distilled file *is* the trigger). Interrupted distill = harmless and self-healing;
   never a reason to extend the restart deadline.
3. **Metrics re-publish on resume.** The `distill/*` series piggyback on `SelfPlayCallback` and log
   at `num_timesteps` (continuous curves across restarts under `reset_num_timesteps=False`). On
   restart, **extend `replay_last_eval_to_tui`** to also re-emit the last distill state — per-snapshot
   top1/h2h/speedup + the aggregates (`frac_opponents_distilled`, `snapshots_distilled`,
   `auto_reverts_total`) read from the manifests/summary — so the distill TUI panel and TB markers
   aren't blank until the next cycle (the same fix the eval + pool blocks already use).
4. **Gate / canary / revert state persists in `summary.json`.** The canary share, the per-snapshot
   deploy decision, and any auto-reverts are written alongside `win_rate_vs_bots` / `pool_generation`,
   so a restart resumes the **canary at its current width and remembers reverts** — it does not reset
   the rollout to 0% or re-deploy a snapshot it already reverted, and the kill-switch reads its
   baseline from there.
5. **Distillation does not worsen the fragmentation the restart exists to fix.** The restart targets
   the *trainer's* pymalloc heap; the distill job runs in a **separate subprocess** that returns all
   its memory to the OS on exit, adding nothing to the trainer's heap (a point in favor of the
   subprocess design, like the eval workers).

Net: distilled opponents and their gate state survive a restart **as files**, an interrupted distill
self-heals, metrics re-publish on resume, and the canary/revert state persists — **no new failure
mode is introduced by the launcher's restart cadence.**
