# New lineage — launch kit + validation evidence (2026-09-26)

The registration (the recipe, the meters, the pooled loop design, its power, the rules, the watch list) is
[`../../new_lineage_2026-09-26.md`](../../new_lineage_2026-09-26.md). **Nothing here was executed for real by
the session that wrote it: no run was launched, and nothing was written under `models/`.** `launch_base.sh` is
**PREPARED BUT NEVER EXECUTED**, for the Training Run session to run VERBATIM after the orchestrator's go.

**The code, each time:** **N0** = `ai_v14_01_base`, the new lineage's FRESH base run (the last production
lineage's fresh-root recipe, `ai_v13_02_flywheel_winprob`, on the clean-input boundary `b0a28b5b`, pinned
`8d07051a`).

## Contents

| path | what |
|---|---|
| `argv_base.txt` | the exact argv `launch_base.sh` launches (235 tokens) |
| `launch_base.sh` | the launch: checkargs + launcher `--dry-run` every time; refuses to clobber `models/ai_v14_01_base`; refuses to share the GPU with a live trainer (`ALLOW_CONCURRENT=1` overrides); prints the STOP-list. `--dry-run` resolves and creates nothing; no argument LAUNCHES |
| `scripts/template_ai_v13_02_flywheel_winprob.txt` | VERBATIM copy of the recipe's recorded `original_command` (launcher path dropped) |
| `scripts/build_argv.py` | regenerates `argv_base.txt` from the template and ASSERTS that only the four registered values move (`--run-name`, `--pin-commit`, `+--arch production`, `+--obs-source core`), that there is no `--model`, and that `--steps` is 75,000,000 |
| `scripts/validate_all.sh` | checkargs + `--dry-run` + `launch_base.sh --dry-run`, and FAILS if `models/` gained an entry |
| `scripts/pooled_rule.py` | the registration's §5.3 pooled read, mechanised, and `--power` (the §5.4 table) |
| `validation/` | every output quoted below, verbatim (the worktree path is shortened to `<wt>`) |

## The argv diff (`validation/build_argv.txt`)

```
== base  ai_v14_01_base  (231 -> 235 tokens vs ai_v13_02_flywheel_winprob)
     --arch: None -> ['production']
     --obs-source: None -> ['core']
     --pin-commit: ['6eb9c776940ed6040ddb90acfbc28b038457fc42'] -> ['8d07051aa767331cfa1ca2292029d5217cbf6d40']
     --run-name: ['ai_v13_02_flywheel_winprob'] -> ['ai_v14_01_base']
```

## VALIDATED BY EXECUTING (2026-09-26 06:04 PT, this worktree's `src/`, cwd the main checkout)

`bash scripts/validate_all.sh` → all three `rc=0`, `models/ untouched (281 entries; no models/ai_v14_01_base)`
(`validation/validate_all_run.txt`).

- **checkargs** (`validation/checkargs_base.txt`): 134 flags; parser = the current tree's (the pin names HEAD);
  132 accepted, 2 launcher-owned, **0 unrecognized**; `[ForkLR] no --model: a fresh run`; **`✓ every
  ARCH-surface key matches the production mirror (39 of 51 registry toggles …)`**; `✓ this command still
  launches`. The ⚠️ "NOT applied" line lists the 16 critic-readout / supervision-dose flags `--arch` never
  writes; each is typed in the argv or at a default equal to production (registration §1).
- **launcher `--dry-run`** (`validation/dryrun_base.txt`): `role FRESH`, `run dir models/ai_v14_01_base
  [would be created]`, `pin 8d07051a… (source: pin_commit)`, `steps 75,000,000 (fresh run, from 0)`,
  `transport in-process bridge [rust]`, **`obs source core (the Rust core row)`**, restarts every 3.0 h,
  `--arch production — 39 ARCH-surface key(s) applied`, `✓ DRY RUN — this command would launch. Nothing was
  created or modified.`
- **`launch_base.sh --dry-run`** (`validation/launchsh_base.txt`): the same two outputs, then `--dry-run
  requested: stopping here. Nothing was created.`
- **`pooled_rule.py`**: `--power` → `validation/power_table.txt`; the STAND-IN on the old lineage's rounds 1–2
  (`validation/pooled_rule_standin_old_r1r2.txt`) reproduces −10.00 [−16.60, −3.27] and −8.25 [−15.01, −1.38]
  exactly and pools them to −9.13 [−13.91, −4.34], NOT DETECTED — a mechanics check, not a read.

**Not validated here (child-only or out of scope):** the resolved `model_config.json` (written by the child;
its diff against `production_config.json` is the launch entry's job), the 60-s `--debug` smoke (it creates a
run directory under `models/`; registration S-3), and the first two minutes of the real launch — the only
test of the preload layer.
