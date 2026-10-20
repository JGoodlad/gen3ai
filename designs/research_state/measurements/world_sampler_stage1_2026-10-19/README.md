# The WORLD SAMPLER's STAGE 1: the sampler ablation with playouts (2026-10-19)

Stage 1 of [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §6.2, registered in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any arm's output existed. The numbers and their reading go to the
design doc (§6.2's read, §7, the Decision record) once read; this folder holds the harness, the registration and the
read's outputs.

**Status: REGISTERED; the runner crawls detached (§3).** No number yet.

| file | what |
|---|---|
| `PREREGISTRATION.md` | the arms, the metrics, the rules, the sample and the registered stop |
| `s1_common.py` | paths, the 287 states and their registered order, the arms, the archive layout |
| `fit_alpha.py` → `alpha_fit.json` | the corpus backoff's α, fitted on the LADDER corpus only (leave-one-team-out), before the registration |
| `corpus_s1.py` | the two corpora, normalised and validator-checked (`<archive>/corpus/` caches them) |
| `samplers_s1.py` | the arms as samplers: F's builder with the arm's departures swapped in for one `sample` call (`hooks`) |
| `build_s1.py` | BUILDER side: every arm's worlds per state on one shared screen cache |
| `play_s1.py` | PLAY side (X26's pin + the roll control): every arm played by the pilot's own driver, a rolls world with its force table; `--identity` = the identity tooth |
| `run_s1.sh`, `queue_s1.py`, `launch_s1.sh` | the checkout selector (CPU rules), the sequential runner (chunks in the registered order; the stop), the detached launcher |
| `score_s1.py` | the table: per state every source's IMP at every scorer, F_rolls weighted, F_legal's dose curves, the world meters |
| `gigo_s1.py` | the GIGO audit before the read: coverage, provenance, defects, the rolls, two CRN teeth, the leak tooth |
| `rules_s1.py`, `read_s1.py` | the registered rules as code, and the read (`result_s1.json`) |
| `read_s1_test.py` | planted-effect tests of the rules and the read, and of the samplers' pure parts (run by hand: `python -m pytest read_s1_test.py`) |

## 1. Checkouts and archive

| | |
|---|---|
| DRIVER = BUILDER | `~/.cache/gen3ai/ws_stage1/builder`, detached at the launch commit (on `origin/main`); every world row's `head`, every row's `driver`. **Keep it until the read** |
| PLAY | `~/.cache/gen3ai/ws_stage1/pin_rolls`: X26's pin `9c033489` + the cherry-pick of `fc010d7d3` (the damage-roll control), local commit `13ffd55c` (recreate: `git worktree add --detach <dir> 9c033489 && git cherry-pick fc010d7d3`, taking the pin's side of the one `src/rust_sim/CLAUDE.md` conflict; `play_s1.check_pin` verifies the Rust diff equals the roll-control commit's). Its own `src/rust_env/target/release` build. **Keep it until the read** |
| archive | `~/gen3ai_archive/world_sampler_stage1/` (`worlds/<arm>/`, `rows/<arm>/`, `identity/`, `corpus/`, `score/`, `logs/`, `STOP_AT`, `QUEUE_DONE` / `STOPPED`, `PARKED`) |
| inputs (read only) | `~/gen3ai_archive/hidden_info_maturity/` (the interim bank, the 225M checkpoint copy, roots, truth, banked F / FN rows and worlds, the 225M belief rows); its plateau step appends there: torn lines are skipped and nothing is written there |
| engineering smoke | `~/gen3ai_archive/world_sampler_stage1_smoke/` (2 states outside the 287; never read as output) |

## 2. Operate

```bash
F=~/.cache/gen3ai/ws_stage1/builder/designs/research_state/measurements/world_sampler_stage1_2026-10-19
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
$PY -I $F/queue_s1.py --status                 # PROGRESS: complete / built / played per arm, the stop (no outcome)
tail -f ~/gen3ai_archive/world_sampler_stage1/logs/queue.log
bash $F/launch_s1.sh                           # (re)start, from the DRIVER checkout; every unit resumes; STOP_AT kept
# THE READ, once QUEUE_DONE or STOPPED exists (PREREGISTRATION §5-§7), with PYTHONPATH = the driver's src:
cd $F && PYTHONPATH=~/.cache/gen3ai/ws_stage1/builder/src $PY -m pytest read_s1_test.py -q -p no:cacheprovider
PYTHONPATH=~/.cache/gen3ai/ws_stage1/builder/src $PY score_s1.py && PYTHONPATH=... $PY gigo_s1.py && PYTHONPATH=... $PY read_s1.py
```

CPU (root rule 5b): ONE single-thread worker at a time, `chrt --idle` + `nice -n 19` + `taskset -c 5-7,13-15` + its
own `mem_cap` scope; `CUDA_VISIBLE_DEVICES=""`. No GPU.

## 3. The run

Filled in at launch: the launch commit, the runner's pid, `STOP_AT`, the identity tooth's result and the smoke's cost.
