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
| K6.1 | freeze guard + optimizer state declared at startup | SHIPPED | `01183e39` |
| — | batch 1 never reaches the compiled learner forward (`gen3_batch1_eager_v1`) | SHIPPED | `ddc8017a` |
| — | the learner golden's recipe read from K10(a)'s production block | SHIPPED | `055f636d` |
| K6.2 | CUDA memory LEAK detector, wired (`CudaMemoryWatch`), calibrated; expandable_segments not recommended | SHIPPED | `32bc32ec` |
| K6.3 | AST static gate, EMPTY allowlist | SHIPPED | `6985afc9` |
| K6.4 + K6.5 | lock at the END OF STARTUP; FATAL names the guard; the `8fc297a2` signature found (rank-probe hooks); the in-run canary; CUDA optimizer-state test | SHIPPED | `990851de` |
| K8.1 | functional masking, bitwise vs sb3 | SHIPPED | `65ee90c9` |
| K8.2 | `forward_guard` on 2.5.1 / MHA fast path | DROPPED (orchestrator: 2.5.1 is legacy; R0 traces on 2.8) | — |
| K8.3 | fold rewrite: the micro-step (R1), static belief bank + intent fold, one host read; golden re-recorded as a pure refactor | SHIPPED | `6f10b877` |
| K8.4 | declared regions R0 + R1 (`fullgraph=True`), ragged/batch-1 eager routes, the inventory test | built, branch `k8s`, gate pending | — |
| K8.5 | rank probe from R1's stashes, spectra on the device | built, branch `k8s`, gate pending | — |
| K8.6 | device-resident micro-batch | built, branch `k8s`, gate pending | — |
| K8.7 | `compile_regions_test` IS the routine inventory test (graphs == regions × signatures, 0 compiles after the lock; 2.5.1 a declared exception) | inside K8.4 | — |
| K8.8 | acceptance read (the compile inventory's time stage, TF32, interleaved A = main pre-regions / B = K8, n = 2) | ACCEPTED: 46.7 → 36.3 s (0.778x), compiled share 63.4% → 90.0% (`acceptance/`) | branch `k8s` |

## Resume point

(updated at every unit boundary)

- 2026-10-01 03:40 PT — K8 ACCEPTED (`acceptance/README.md`). Branch `k8s` (on main `ab207bf6`):
  K8.4 regions, K8.5 rank probe, K8.6 device batch, the bench fix, the acceptance read — gate and the
  2.8 compile slow tier (`~/gen3ai_archive/k6_k8/gpu_queue5.status`) pending, then ship. Branch
  `k6leak` (worktree `/home/goodlad/dev/gen3ai-wt/k6leak`): the orchestrator's FATAL_CUDA_LEAK (6)
  follow-up, built and tested, ships after K8.

- 2026-10-01 01:55 PT — shipped: K6.1–K6.5, K8.1, K8.3 (+ the batch-1 fix, the golden recipe).
  Branch `k8s` (worktree `/home/goodlad/dev/gen3ai-wt/k8s`, on main `32bc32ec`) holds K8.4 / K8.5 /
  K8.6 as three commits, docs per commit; `k84` is the SAME code, the tree the acceptance read's B
  units run from (do not rebase it while `accept/run.sh` runs). Next: the acceptance verdict (A1 B1
  A2 B2, `accept/status` is the resumable row file; rerun `accept/run.sh` to finish missing units),
  then the routine gate on `k8s` and ship the three, then the final report. FINDINGS so far are in
  the K6/K8 commit bodies and `compile_flags.md` "K8 — DECLARED COMPILE REGIONS".

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
