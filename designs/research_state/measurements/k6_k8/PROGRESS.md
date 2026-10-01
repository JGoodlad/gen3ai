# M5 Lane K6 + K8 — progress and RESUME POINT

**Brief (orchestrator, 2026-09-30):** K6 = the learner's DECLARED LIFECYCLE (freeze guard, static
gate, declared compile signatures frozen BEFORE the first real iteration, in-run parity canary);
K8 = DECLARED COMPILE REGIONS (`fullgraph=True`, the inventory's fix plan, `compile_inventory` as a
routine test, the acceptance read). Worktree `/home/goodlad/dev/gen3ai-wt/k6-k8`, branch `k6-k8`.
Owner rule: correctness over churn — an undeclared break or recompile is an ERROR.

Orchestrator course corrections (2026-09-30, binding):
- HARD single-shot FATALs only for the DETERMINISTIC cases: a new optimizer / `nn.Parameter` /
  module after freeze, and an undeclared compile signature.
- CUDA memory is a LEAK DETECTOR, WARN by default: fit the trend of reserved/allocated bytes over
  update windows; a one-off step-up or a fragmentation plateau never stops the run; stop
  deliberately (checkpoint, then a typed FATAL the launcher does not restart) only when a
  SUSTAINED trend PROJECTS an OOM within a declared horizon; log the projection every window;
  calibrate windows/K from a measured healthy run and document the false-trip estimate; measure
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (segment stability + update time).

## Unit plan (ship order; one logical unit per commit, each after a green routine gate)

| # | unit | status | commit |
|---|---|---|---|
| K6.1 | freeze guard: learner object-graph snapshot (params/modules/optimizers/param groups) + registration-site recording; `LazyAcquisitionError` (FATAL_CONFIG); optimizer state declared at startup | built, gate pending | — |
| K6.2 | CUDA memory LEAK detector (warn + projected-OOM clean stop) + calibration + expandable_segments read | dispatched (subagent, branch `k6-mem`) | — |
| K6.3 | AST static gate, EMPTY allowlist: no optimizer / Parameter / Module construction in training-step paths outside declared startup builders | built on branch `k6-astgate` (`37f19456`); RED on main by design (5 ride-along sites) — lands after the RND lane's ride-along move + `@startup_builder` on `ridealong_heads._mlp` / `build_ridealong` | — |
| K6.4 | declared compiled-signature table, compiled at startup, LOCK before the first real iteration; the `8fc297a2` iteration-1 signature found and declared | planned | — |
| K6.5 | in-run parity canary (every N updates; fixture; startup gate's bars) | planned | — |
| K8.1 | functional masking (no sb3 `__dict__.pop`), bitwise vs sb3 | planned | — |
| K8.2 | `forward_guard` hoisted (2.5.1 `fullgraph=True`); MHA fast path off in eval region | planned | — |
| K8.3 | fold rewrite: steps 1–4 as a functional micro-step returning tensors; one hoisted host read; static bincount | planned | — |
| K8.4 | R1 compiled `fullgraph=True` + the region declaration table | planned | — |
| K8.5 | rank probe: reps from R1's own forward, spectrum on the device | planned | — |
| K8.6 | device-resident micro-batch build | planned | — |
| K8.7 | `compile_inventory` routine test (graphs == regions × signatures, 0 undeclared breaks, both torches) | planned | — |
| K8.8 | acceptance read (idle-box baseline first) | planned | — |

## Resume point

(updated at every unit boundary)

- 2026-09-30 22:05 PT — branches: `k6-k8` (K6.1 committed, gate green modulo 2 env-flaky BIG-RSS
  watchdog tests that pass on rerun; WAITING for the RND lane's ride-along startup-optimizer change, then
  rebase + one `--ridealong-*` path under the guard, per the orchestrator); `k6-b1` (the batch-1 fix,
  rebased onto origin/main WITHOUT K6.1 — ship first; CUDA tests pass on 2.8; the 2.8 final-eval smoke
  `~/gen3ai_archive/k6_k8/smoke28_b1.log` and its routine gate `gate_b1.log` pending); `k6-k8-dev`
  (K8.1 functional masking, K6.4 hook-free rank probe + lock-before-iteration-1 + guard-naming FATAL,
  K6.5 canary, the static belief bank — all WIP commits; to be rebased and split per unit). Subagents:
  `k6-mem` (memory leak detector, running), `k8-intent` (static opp-intent fold, running),
  `k6-astgate` (`37f19456`, done; lands after the ride-along change). Follow-up queued (orchestrator):
  `learner_golden.RECIPE` read from `main.train.recipe_surface` (own commit).
- FOUND: the `8fc297a2` iteration-1 signature = the rank probe's forward hooks
  (`~/gen3ai_archive/k6_k8/sigprobe/run1.log`). FINDING: on torch 2.8 those hooks were silently
  skipped inside the compiled frame, so `rank/trunk_*` / `rank/value_cls_*` vanished on compiled 2.8 runs.

- 2026-09-30 20:45 PT — worktree created, bootstrap green, plan written. Next: K6.1 core + K8.1.
