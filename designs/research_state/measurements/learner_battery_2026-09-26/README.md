# Learner battery — launch kit + validation evidence (2026-09-26)

The registration (the arms, the dose, the rules, the power, the schedule amendment) is
[`../../learner_battery_2026-09-26.md`](../../learner_battery_2026-09-26.md). **Nothing here was executed
for real by the session that wrote it: no arm was launched, nothing was written under `models/`.**
`launch_arm.sh` is **PREPARED BUT NEVER EXECUTED**, for the Training Run session after the orchestrator's go.

**Codes:** **N0** = `ai_v14_01_base`, the new lineage's fresh base run (the fork parent, at its
`final_model.zip`, 75,005,952). **C** = `ai_v14_02_lbat_ctrl`, the control (N0's recipe continued = the
lineage's K1). **E5** = `ai_v14_03_lbat_e5`, 5 epochs at 2·D_g (same dose). **T32** = `ai_v14_04_lbat_t32`,
TF32 matmuls. **L95** = `ai_v14_05_lbat_l95`, policy GAE λ 0.95. **D_g** = the lineage's frozen generalist lr.

## Contents

| path | what |
|---|---|
| `argv_{C,E5,T32,L95}.txt` | the FINAL argvs with the `__DG__` / `__2DG__` placeholder — **they REFUSE to launch** until D_g is decided (registration F-1) |
| `launch_arm.sh <ARM> [--dry-run]` | checkargs + launcher `--dry-run` every time; refuses the placeholder, an unfinished N0, a D_g that differs across arms, any arm before C has finished, a clobber, and a concurrent trainer; prints the STOP-list. `STANDIN=1 … --dry-run` validates the stand-in argv (refused without `--dry-run`) |
| `scripts/build_argvs.py` | builds all four from the lineage kit's `argv_base.txt` and ASSERTS the design: the deleted flags and `--arch` dropped, each arm differs from C in exactly its registered value(s), the dose lr × n_epochs identical on every arm. `--final --dg <D_g>` writes the launchable argvs; `--standin …` the validation ones |
| `scripts/argv_*_STANDIN.txt` | the validated stand-ins (parent N0's `checkpoint_4800000_steps.zip`, D_g 4.3e-04 = N0's lr at 4.8M) |
| `scripts/speed_read.py` | the speed endpoint S (projected from `train/train_ms`, bootstrap CI), the per-epoch KL / clip descriptor, stalls per 1M, and the T32 futility look |
| `scripts/battery_rule.py` | the decision rule (`rule`) and the power table (`--power`) |
| `scripts/validate_all.sh` | everything below, and FAILS if `models/` gained an entry |
| `validation/` | every output, verbatim (the worktree path shortened to `<wt>`) |

## To launch (Training Run, after the go)

```bash
K=designs/research_state/measurements/learner_battery_2026-09-26
python3 $K/scripts/build_argvs.py --final --dg <D_g>     # D_g at 2 s.f., the owner's O1
bash $K/launch_arm.sh C            # then E5, then T32 (futility look at rollout 11), then L95
```

## VALIDATED BY EXECUTING (2026-09-26 11:37 PT, this worktree's `src/`, cwd the main checkout)

`bash scripts/validate_all.sh` → every check rc 0, the two refusals rc 2 as designed, `models/ untouched
(282 entries; no ai_v14_0[2-5]_lbat_* dir)` (`validation/validate_all_run.txt`).

- **checkargs, all four** (`validation/checkargs_*.txt`): parser = the PINNED `2cc83080`'s, 0
  unrecognized; resolved against the fork parent N0's `model_config.json` (56 flags inherited);
  `[ForkLR] ✓ this fork names its own dose (--fork-lr 0.00043 --fork-lr-freeze)` (E5: 0.00086); `✓ every
  ARCH-surface key matches the production mirror`; `✓ this command still launches`. (The first attempt,
  with `--arch production` kept, was REFUSED by `combination_checks` — registration F-4.)
- **launcher `--dry-run`, all four** (`validation/dryrun_*.txt`): `role FORK of …/ai_v14_01_base`, pin
  `2cc83080…`, `+8,000,000 steps`, transport rust, obs source core, `✓ DRY RUN — this command would launch.
  Nothing was created or modified.`
- **`launch_arm.sh`**: `STANDIN=1 … --dry-run` rc 0 on all four; the committed placeholder argv REFUSES
  (rc 2); `STANDIN=1` without `--dry-run` REFUSES (rc 2).
- **`speed_read.py` on N0** (mechanics, an identity pair): bots-only window median iteration 106.0 s,
  update 55.7 s (share 0.525); self-play window 182.0 s / 56.2 s (share 0.309); S = +0.00 % [−1.32, +1.33].
  The stall line in the self-play window over-counts (F-6).
- **`battery_rule.py`**: `--power` → `validation/power_table.txt`; five synthetic cases exercise every
  branch (`validation/rule_synthetic_cases.txt`).
- **Read-only**: N0's `model_config.json` migrates 121 → 123 (`policy_gae_lambda` 0.8), no shaped-reward
  evidence.

**Not validated here (child-only):** the v121 zip's full load under v123, the pool seeding, the per-epoch
scalars / TF32 stamp on a real launch, the `--debug` smoke (creates a run dir). The first two minutes of
each arm are the test.
