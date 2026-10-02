# Training — search-as-teacher (DELETED — tombstone)

**Deleted 2026-10-02, deletion pass L3** (owner: "delete all, port none"; config v133,
`gen3_retired_levers_l3_v1`). This file used to hold the operating detail; it is kept as a tombstone
because other documents link to it. Nothing below is operable.

**What it was.** Selective Expert Iteration, the offline-teacher plateau-breaker: each cycle, search
and rollout-confirm the worst loss craters of recent eval traces (the prober's `better_line` beam plus a
Wilson-gated rollout-confirm tier) and distil the VERIFIED-better action into the policy through an
advantage-weighted CE auxiliary loss on a bounded correction ring (the `agents/training/teacher/`
package, `--search-teacher` and its `-mode` / `-coef` / `-value-coef` / `-beta` / `-batch-size` /
`-buffer-size` knobs, the `--teacher-*` search-budget / worker / freq / refresh knobs, a second
`winprob_oneply` supply with `--winprob-teacher-band` / `--winprob-teacher-margin`, and the OPD
`--opd-coef` / `--opd-beta` term). It shared the learner's AWR / distill machinery with the exploiter
fold ([`exploiter_and_distillation.md`](exploiter_and_distillation.md) → HISTORY).

**Why it is gone.** Search was wound down on 2026-09-12 (objective = PPO effectiveness); the
composition gate's verdict was never recorded and a search label is not reproducible by construction
(ledger 2026-09-22, *the search-teacher composition gate's verdict was never RECORDED*). Nothing had run
`winprob_oneply`. Flag-by-flag citations: `designs/deleted_flags.md`; the deletion's scope:
`designs/ops/deletion_pass_manifest.md` §2.

**What stays.** The Rust search driver and the prober's search / replay tools
([`../rust_sim/search_and_replay_drivers.md`](../rust_sim/search_and_replay_drivers.md)) are untouched:
only the TRAINING-side consumer (the teacher callback, its shard workers and the learner term) is
deleted. A resume of an old run that used it fails argparse unless pinned: use
`LAST_COMMIT_L3 = 615a764fdb7e05abfcc1575797e2c83eb61c3ea1` (`model_version.retired_levers`), the last
commit whose tree still has it.

**If it is ever wanted again.** Port it onto the Rust search driver if X15 (expert iteration) is
scheduled — a Rust port of the distillation route is estimated at ~1-2 agent-days
(`distill_mask` becomes a per-episode host key from `TeamStager`; teachers are frozen T2 slots or a
learner-side forward). Distillation was the only BUILT route to X15. Under the one-ply search scope
(owner 2026-09-30) in-loop search additions are limited to ONE ply and must be subsample-eligible.
