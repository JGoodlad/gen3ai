# `hidose_teacher_2026-09-21/` — the HIGH-DOSE offense teacher, PREPARED not launched

**Status: PRE-REGISTRATION ONLY. The arm does not exist.** This directory holds the argv, its
executed validation, and the read — all committed BEFORE any battle of
`ai_v13_18_teach5_offense_hidose` is played.

| file | what |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the registration: the question, the argv, the dose arithmetic, the read, the bar, the prior, the branches, the disclosed facts, and the ready-to-append ledger paragraph |
| [`argv_hidose.txt`](argv_hidose.txt) | the argv VERBATIM (230 tokens, everything after the launcher entry point) |
| [`launch.sh`](launch.sh) | runnable **by the Training Run session**, verbatim. Re-validates, then launches. `--dry-run` stops before launching |
| [`out/checkargs.txt`](out/checkargs.txt) | `main.checkargs` on this argv **and** on the template `ai_v13_13_exploit5_offense`, for the ARCH-surface comparison |
| [`out/dry_run.txt`](out/dry_run.txt) | `main.launcher --dry-run` — FORK, +8,000,000, `--fork-lr 0.00025`, `--fork-lr-freeze True`, nothing created |
| [`out/dose.txt`](out/dose.txt) | `main.dose` over the three era-1 exploiters, the three era-2 exploiters, stage A and the plateau parent — the table the `2.5e-4` is derived from |
| [`scripts/harness/run_admission_hidose.sh`](scripts/harness/run_admission_hidose.sh) | the admission cell, reused verbatim from the 2026-09-21 gate; only the teacher ref changed |
| [`scripts/harness/run_untaught_hidose.sh`](scripts/harness/run_untaught_hidose.sh) | the collateral off-slice cell, reused verbatim from `split_lossoff_read_2026-09-20/` |
| [`scripts/harness/teams_offense.json`](scripts/harness/teams_offense.json) | the offense five in REGISTERED order — the order IS the seed offset; do not re-sort |
| [`scripts/hidose_delta.py`](scripts/hidose_delta.py) | adjudicates both cells; the paired bootstrap is VERBATIM from `admission_delta.py` (20,000 draws, seed 20260915, the TEAM as the unit) |

**The one-line question.** Four teachers manufactured on `models/ai_v13_12_plateau/final_model.zip`
have failed the same admission gate, all four at **0.39× the v8 dose**. Two accounts survive: the
dose was too small, or the parent is at a **team-level ceiling** on these teams. This arm moves the
dose to **1.78×** — era-1's, re-derived from `main.dose`'s formula — and moves nothing else.

**This session ran nothing on the GPU and wrote nothing under `models/`.** Everything above ran
`CUDA_VISIBLE_DEVICES="" nice -n 15` while `ai_v13_17_fold_k1` held the GPU.
