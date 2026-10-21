# The WORLD SAMPLER's STAGE 1: the sampler ablation with playouts (2026-10-19)

Stage 1 of [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §6.2, registered in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any arm's output existed. The numbers and their reading go to the
design doc (§6.2's read, §7, the Decision record) once read; this folder holds the harness, the registration and the
read's outputs.

**Status: READ 2026-10-21 (§4).** Only team identification (C_pool) pays. Every open-population sampler is NOT DETECTED,
and none is SEARCH-ENABLED. The reading lives in the design doc (§6.2).

| file | what |
|---|---|
| `PREREGISTRATION.md` | the arms, the metrics, the rules, the sample and the registered stop |
| `s1_common.py` | paths, the 287 states and their registered order, the arms, the archive layout |
| `fit_alpha.py` → `alpha_fit.json` | the corpus backoff's α, fitted on the LADDER corpus only (leave-one-team-out), before the registration |
| `corpus_s1.py` | the two corpora, normalised and validator-checked (`<archive>/corpus/` caches them) |
| `samplers_s1.py` | the arms as samplers: F's builder with the arm's departures swapped in for one `sample` call (`hooks`) |
| `build_s1.py` | BUILDER side: every arm's worlds per state on one shared screen cache |
| `play_s1.py` | PLAY side (X26's pin + the roll control): every arm played by the pilot's own driver, a rolls world with its force table; `--identity` = the identity tooth |
| `run_s1.sh`, `queue_s1.py`, `launch_s1.sh`, `loop_s1.sh` | the checkout selector (CPU rules), the sequential runner (chunks in the registered order; the stop; `--one-chunk`), the detached launcher, the slot-per-chunk loop |
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

- **Launched** 2026-10-19 18:27 PDT from the builder checkout at `8d304953`. `STOP_AT` = 2026-10-20 18:27 PDT.
  Identity tooth 12 / 12; the smoke's cost is in PREREGISTRATION §5.
- **Relaunched** 2026-10-19 21:13 PDT (30 states complete, the registered order and stop unchanged). The heavy-job
  slots (`24099876a`) had landed beside the run, and its every build and play waited for a slot among 12 random
  pollers (chunk 6 waited 47 min). The runner itself now holds ONE heavy slot for its life (`mem_cap.sh --slot`
  around `queue_s1.py`), so its children take none. It is one heavy job, as rule 5b and the brief allow. Relaunch
  the same way: the env of `launch_s1.sh`, then
  `nohup setsid nice -n 19 <builder>/scripts/ops/mem_cap.sh --name ws1_runner --slot 2 timeout 28h python -I queue_s1.py`.
- **Switched 2026-10-20 at a chunk boundary to ONE heavy slot per BOUNDED chunk** (the orchestrator's policy: a slot is
  held per chunk of ≤ ~45 min, never for a multi-hour run). `loop_s1.sh 4` runs `queue_s1.py --one-chunk --chunk 4`
  in a loop, each call inside its own `mem_cap.sh --slot`, released between chunks. It runs from the RUNNER checkout
  `~/.cache/gen3ai/ws_stage1/runner` (the commit carrying `--one-chunk`), with `WS1_RUN_S1` = the BUILDER checkout's
  `run_s1.sh`, so every row's builder and drivers stay the launch commit `8d304953`. The chunk size (6 → 4) is
  operational: any chunk size keeps the complete states a prefix of the registered order. The order and the stop are
  unchanged.
- **2026-10-20 08:28**: the RUNNER checkout moved to current main (`855bbc8b`; the measurement folder unchanged), so the
  loop's own `mem_cap.sh` / `heavy_slots` take FIFO tickets (`96f011dc`; the old copy polled at random). Every
  per-chunk slot hold so far was 8–20 min (≤ 45 min).
- **RULING (orchestrator, 2026-10-20 ~08:50): stage 1's chunk jobs run BLOCKING**, `GEN3AI_BLOCKING="WS stage 1
  (P0.1)"`: the P0.1 family, as stage 1c. They bypass the heavy slots, still at `nice -n 19` + `taskset -c
  5-7,13-15` under `mem_cap`, one chunk at a time, chunks ≤ 45 min, with the same registered stop and design. A
  chunk whose peak memory passes ~10 GB is reported to the orchestrator. The loop was relaunched with the mark: its
  waiting slot request had started no work and was stopped by pid. Relaunch the same way: `loop_s1.sh 4` with the
  env of §3's switch, plus `GEN3AI_BLOCKING`.
- **Amendment A1 (L2)** registered before any outcome was read (PREREGISTRATION §9).
- **Amendment A2 (2026-10-20, the orchestrator): the 24 h stop is LIFTED**, so the run goes to all 287 states and the
  read is made once at `QUEUE_DONE` (PREREGISTRATION §10). `STOP_AT` holds a safety bound; the original is in
  `STOP_AT.registered`. **BLOCKING ends 2026-10-20 18:27**: after it the loop is relaunched at a chunk boundary
  WITHOUT `GEN3AI_BLOCKING` (FIFO heavy slots, nice 19, the agent cores, 3-state chunks). The read must run from a
  checkout that holds it (`l2_s1.py`).

## 4. The read (2026-10-21)

- **QUEUE_DONE** 2026-10-21 05:40: 287 / 287 complete, 0 parked.
- The table is `<archive>/score/table.pkl` (sha256 `5b53d306…`).
- The GIGO audit PASSES (`gigo_s1.json`), with the leak tooth corrected by A3 before any outcome was read.
- The planted tests pass (14), then `read_s1.py` → `result_s1.json`.
- The verdicts, the tables and the reading are `design_world_sampler.md` §6.2. In one line: C_pool PAYS and is
  MATERIAL (Δ +0.038, r 0.72); B, C_ladder, C_forum_strict and NS are NOT DETECTED; NOT CALIBRATED; L2 EXACT-ONLY
  between arms; every secondary NOT DETECTED; the 1b trigger fires on C_pool.
- **Closed 2026-10-21.** The orchestrator ruled that stage 1b is not run (`design_world_sampler.md` §6.2). The checkouts
  `~/.cache/gen3ai/ws_stage1/{builder,pin_rolls,runner}` were removed. The archive
  (`~/gen3ai_archive/world_sampler_stage1/`) is kept. To re-run or re-read, recreate them:
  - the BUILDER and RUNNER: detached at `8d304953` and `855bbc8b`;
  - the PLAY checkout: §1's recipe (`9c033489` + the cherry-pick of `fc010d7d3`).
