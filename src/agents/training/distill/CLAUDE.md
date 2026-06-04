# CLAUDE.md — Opponent Distillation (`src/agents/training/distill/`)

Distil frozen self-play opponents into a **cheaper network** so rollouts run faster. The opponent
forward is ~70% of env-worker CPU; a faithful ~4.7–6.4× cheaper opponent is an estimated ~+15–25%
rollout throughput at ~zero quality cost. **Off by default**; enable with `--distill-opponents`.

> **Design + the empirical record live in `designs/ai_v5/`**: `design_opponent_distillation.md`
> (forward design + guardrails + observability), `distill_integration.md` (the integration contract —
> recipe, async pipeline, load seam, gate calibration, **§7 restart resilience**, **§8 all-or-nothing**),
> `distill_results.md` + `distill_component_map.md` (what was tried and why). This file is the
> as-built module map.

## The one constraint that shapes everything: all-or-nothing (`distill_integration.md` §8)

`SubprocVecEnv.step_wait` is a barrier — every rollout step runs at the speed of the **slowest**
worker. A worker on the full teacher is ~26% slower than one on a distilled opponent, so **a single
full-opponent worker gates the whole batch and erases the speedup**. Therefore the pool is only ever
**100% distilled or 100% full**, never mixed. Consequences baked into the design:

- **No live full-model anchors** (they'd straggle). Safety = the pre-deployment **gate** + drift
  **auto-revert** (drops a snapshot, doesn't swap in the slow teacher) + the independent full-model
  **bot-eval** (`win_rate_vs_bots`) as ground truth.
- A snapshot is **sampleable only once its distilled variant passed the gate**; the full model is
  never a live opponent in steady state.
- Enabling mid-run = **backfill** (distil the whole pool, incl. the ≤5 sentinels) → then an **atomic
  per-generation switch** to distilled. Partial progress buys nothing, by design.

## Module map

| file | role |
|---|---|
| `student.py` | `DistilledStudent` — self-contained cheap policy (own `ObsUnpack`+`Embeddings`, no teacher at inference): `CheapEncoder` (per-slot MLP → 12×128 role tokens) → 1-layer transformer → matchup pool → MLP head → raw [B,11] logits. `DistilledOpponentModel` adapter is duck-typed like `MaskablePPO` so **`RLPlayer` is unchanged**. Capacity is parameterized (the escalation ladder). |
| `recipe.py` | `distill_snapshot(layout, teacher, obs, mask, config)` — the **validated two-stage** recipe: (1) distil the cheap encoder onto the teacher's **frozen role tokens** (MSE), (2) freeze it + distil the head on the teacher's masked logits (soft-KL T=0.7, eps=0.02). The student copies the teacher's embeddings (then frozen). |
| `manager.py` | `DistilledOpponentManager.reconcile(active_steps)` — **one idempotent loop** (below). |
| `worker.py` | the distil subprocess (the manager's `run_distill_fn`): bridge state-gen → recipe → **gate** (fidelity + greedy head-to-head vs the teacher) → atomic `.pt` + `.json` manifest. Exit 0 always; `passed` is in the manifest. |

Integrations: `snapshot_pool.py` (distilled-variant disk hooks + `load_distilled_opponent`), the env
`MaskableAgentWrapper.set_distill_active` (the atomic full↔distilled switch), `selfplay_callback.py`
(builds the manager + reconciles + pushes + logs `distill/*`), and the `--distill-opponents` flag.

## The reconcile loop (the heart)

`reconcile(active_steps)` makes the on-disk distilled set match the pool's sampleable snapshots —
**backfill and steady-state are the same call; no-op when nothing's missing**:

```
harvest finished jobs (pass→ready / fail→escalate a ladder rung / exhaust) →
  spawn for whatever's still missing (≤ max_concurrent) →
  clean up evicted artifacts →
  return all_distilled (every deployable snapshot is gate-passed) + frac + sampleable
```

Called each eval cycle (post-promote) **and** on a ~100k-step throttle (so backfill flips promptly).
Restart-safe: state is files on disk; a fresh manager rebuilds from the manifests (`recover_fn`
restores ladder progress so it doesn't re-distil known-unfit snapshots). Pure logic via injected
hooks → fully unit-tested (`manager_test.py`).

## Capacity escalation

A gate **failure** means the student is too small for that snapshot → re-distil up a fixed
`DEFAULT_LADDER` (bigger `enc_hidden`/`head_hidden`/ffn). Exhausting the ladder (or dropping below
`min_speedup`) → that snapshot is **not sampled** (the rest stay distilled); if too few remain for a
valid pool (`min_pool`), fall back **pool-wide to full**. The one-glance health metric is
**`distill/frac_active_opponents_distilled`** — it must be 1.0 for any speedup.

## On-disk layout (`models/<run>/distilled/`, sibling of `snapshots/`)

`snapshot_<step:012d>.distilled.pt` (the cheap student) + `.distilled.json` (the **manifest** =
the per-snapshot source of truth: `step, passed, top1, kl, ent_ratio, h2h, speedup, config`) +
`distill_<step>.log`. **The manifest is the record** — `summary.json` carries only a small
`distill` re-publish block. **Cleanup is automatic**: the reconcile evicts a snapshot's `.pt` +
manifest + log + `.tmp` when it slides out of the pool window, so the distilled set is bounded by
the pool size with no separate GC.

## Gate (calibrated to the measured ~0.44 ceiling — `distill_integration.md` §4)

A distilled student deploys iff: **h2h CI overlaps [0.45,0.55]** at N≥300, **ent_ratio ∈ [0.9,1.1]**,
**not more exploitable than the teacher**, and **speedup ≥ min_speedup**. Live drift below ~0.40 →
drop the snapshot. Fail-closed: a crash/no-manifest reads as failed and the reconcile re-triggers.

## Enable / smoke

```
... train_rl_agent.py --self-play --distill-opponents ...   # distil runs on --eval-device (CPU; no GPU contention)
```
Unit tests: `pytest src/agents/training/distill -q` (+ `snapshot_pool_distill_test.py`). The worker
is bridge-backed (run it directly for an e2e smoke against a teacher snapshot).
