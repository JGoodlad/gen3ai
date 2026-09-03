# Fleet-admission artifacts — the committed copies

**Why this directory exists.** These three files are the entire evidential basis of the
exploitability curve, of the teacher-ceiling reframe, and of the taught/untaught decomposition —
and until 2026-08-31 they existed **only** in `~/.claude/jobs/1046b1d6/tmp/probes/`, a
session-scoped job directory outside the repo. Every exploitability claim in the ledger was one
`rm -rf` away from being un-reproducible. They are copied here **byte-identically** (verified by
sha256 against the job directory at copy time) with no schema change, so the tools that read them
need no adaptation.

| file | fleet | shape | teachers × teams | per team | games |
|---|---|---|---|---|---|
| `fleet_admission.json` | **rev-2** (F5a–F5e) | 5 × 2 | 10 cells / 9 teams | 1.5M | 800/arm, 400/team |
| `r3_admission.json` | **rev-3** (F6a–F6f) | 6 × 2 | 12 cells / 12 teams | 2.5M | 800/arm, 400/team |
| `r4_admission.json` | **rev-4** (R4S3a–c) | 3 × 8 | 24 cells / 24 teams | 1.25M | 800/arm, 400/team |

## Provenance

Each artifact is the output of a **fleet-admission battery**: for every (arm, team) cell it plays a
fixed number of games with the arm's TEACHER piloting the team, plus seed-paired games with a fixed
REFERENCE model (`rev1final`) and with the arm's TARGET — the agent being best-responded to. That
pairing is what makes `net` (teacher − reference) and `ordered` (teacher − target) meaningful; the
two differ by the target's own gain against the fixed anchor (`seniority`).

Team identities are **not** carried in the artifacts — cells are keyed by symbolic name
(`ZapDug`, `COV_82e97fe2`, `N_a04c29cf`). The name → team-file mapping is reconstructed from each
arm's run `metadata.json` (`cli_args.trainee_teams`) by the decomposition script, never hand-copied.
So these files are self-contained for `main.exploitability`, and need `models/` only for the
taught/untaught split.

**rev-3 and rev-4 share a target** (`ai_v9_59_R2ACTION_0827`, from every arm's recorded
`--exploiter`). There are two curve points here, not three; rev-4 is a re-measurement of rev-3's
subgame by a differently-shaped fleet.

## Schema

They satisfy the `fleet_admission` schema that `main.exploitability.load_artifact` validates.
**Schema drift REFUSES (exit 2, naming the key)** — that refusal is a feature and these copies must
not be edited to satisfy a future reader. The loader also RECOMPUTES the
`net = teacher − reference = ordered − seniority` identity from each artifact's own per-team cells
rather than trusting the recorded aggregate; all three pass.

## Reproduce

```bash
export PYTHONPATH=$PYTHONPATH:src

# the published curve (rev-2 +0.1165 / rev-3 +0.1350 / rev-4 +0.1252)
nice -n 15 python -m main.exploitability \
  rev-2=designs/research_state/measurements/admission_artifacts/fleet_admission.json \
  rev-3=designs/research_state/measurements/admission_artifacts/r3_admission.json \
  rev-4=designs/research_state/measurements/admission_artifacts/r4_admission.json

# the taught/untaught decomposition — this directory is now its DEFAULT --probe-dir
nice -n 15 python designs/research_state/measurements/exploitability_taught_untaught.py \
  --out /tmp/tu_check
```

Verified 2026-08-31: the first command's output is **byte-identical** to the same command run in the
job directory; the second reproduces
`exploitability_taught_untaught_2026-08-31.json` exactly, the only difference being the recorded
`artifact` paths.

## What is NOT here

The job directory also holds the piloting/coverage probes `team_slate_40.md` §8 cites
(`pilot_R2ACTION_n300.json`, `cov_R2ACTION.json`, `cov_rev1fin.json`, `headroom_screen.json`,
`coverage_sample.json`, `coverage_sweep.json`). Those are already distilled into the committed
`designs/ai_v12/team_slate_40.json`, which is what makes the slate survive that directory's
deletion. The three files here are the ones with a live TOOL reading them.

## rev-5 (2026-09-02/03): `r5_admission.json` (COMPLETE, 136/136 cells) and `r5_admission_PARTIAL_funding.json`

The 40-team fleet (20 teachers × 2 teams × 1.5M/team, target `ai_v9_59_R2ACTION_0827/final_model.zip`,
reference `ai_v9_29_rev1_0823/final_model.zip`) plus the 8 funded forks (`R5FUND00…14`, 2.5M/team).
28 arms, 40 teams, 800 games/arm (400/team), 54,400 games, `shard_fail=0`, `UNCOVERED []`. Produced
by the training session's sharded battery (4 BLAS-pinned shards, cell-round-robin, priority-first on
the funding block; the merge REFUSES a shard with no artifact and carries its completeness stamp
through the verdict pass — both protections added after they bit on 2026-09-02). Team identities are
each run's RECORDED `--trainee-teams`. `team_class_split` is deliberately NOT CLASSIFIED (these are
uniform random draws — no `COV_` slate); the archetype join lives in
`../r5_team_classes_2026-09-02.tsv`. `FUNDING_MATCHED_SAME_TEAMS` (funded fork − its own parent on
the same two teams) is the funding verdict's primary read; `FUNDING_PAIRS` (draw-position pairing) is
archetype-imbalanced and demoted to a sensitivity read. The PARTIAL file is the 66-cell funding-block
merge adjudicated first (status stamped "PARTIAL — funding block complete"); the full file supersedes
it for every cell and agrees on all 64 shared ones.
