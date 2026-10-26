# The arm chain: one tested module that operates production arms end to end

> Always-current (`designs/CLAUDE.md`). Code: [`src/main/ops/arm_chain/`](../../src/main/ops/arm_chain/), entry point
> `python -m main.ops.arm_chain`. Tests: `src/main/ops/arm_chain/*_test.py` (180; fake `/proc`, a fake
> launcher, temp git repos; nothing touches the GPU). **Status: BUILT; CUT-OVER PENDING.** The training session
> still runs its job-directory scripts until the checklist in §7 is done.

## 1. Why

The training session operated every production arm through scripts in its job directory, outside the repo:
`arm_end_v13.sh`, `poll_launch_v5.sh`, `ship_record_v2.sh`, `ship_dose_v2.sh`, `canary_log_v12.sh`, `live_runs.py`,
`aw_files.sh`, and one hand-written window or hand-off script per occasion (`x26c_after_v3.sh`, `x26c_win2.sh`,
`x59s_window.sh`, `d3_redirect.sh`). None of them was tested or reviewed. Every version bump was a class bug found
in production:

- the K9(b) record written into main's tree;
- a trigger that matched its own line;
- a name glob that missed a run family;
- an empty `CUDA_VISIBLE_DEVICES` that blinded a GO;
- a stopped run read as finished;
- a ship worktree missing the submodule a gate reads.

§5 lists all of them and the regression test for each.

The module owns the chain BETWEEN launches: what the launcher cannot see. The launcher still owns everything inside
a run, including a GPU window inside a live run (`--gpu-window`, [`../launcher/gpu_window.md`](../launcher/gpu_window.md)).

## 2. The interface

### The chain file (`arm_chain_v1`)

The chain file is one JSON file. It lists the queue in order. Every value an old script took from a default, an
environment variable or a run-name glob is a required field here, checked when the file is loaded.

```json
{"schema": "arm_chain_v1", "name": "x35-oct26", "job_dir": "/home/goodlad/.claude/jobs/x35",
 "repo": "/home/goodlad/dev/gen3ai", "python": "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3",
 "backend": {"kind": "x35", "launch_script": "designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh",
             "stage": "designs/research_state/measurements/x35_breakers_stage_2026-10-17",
             "lease_owner": "Training Oct", "lease_token_file": "/home/goodlad/.claude/jobs/gpu_lease_training.token"},
 "records": {"stage": "designs/research_state/measurements/x35_breakers_stage_2026-10-17",
             "submodules": ["deps/pokemon-showdown"]},
 "items": [
  {"kind": "arm", "arm": "d3", "seed": 3501, "pin": "<40-hex>", "run": "rb_x35_d3mut_s3501", "go_by": "orchestrator 10-25 13:00",
   "thin": "planner", "flags": ["--thinning"], "disk_need_gib": 12.49,
   "banner": {"expect": "[TEAM-MIX] d3_setmut_v1@ALL", "void": ["[TEAM-MIX] FATAL", "does NOT inherit"], "within_min": 20},
   "throughput": {"min_steps_per_s": 1000, "within_min": 60}},
  {"kind": "window", "name": "x59s", "worktree": "/home/goodlad/dev/gen3ai-wt-x59s", "require_commit": "a1370436",
   "budget_min": 480, "on_fail": "continue", "preempt_marker": "NO_GO_X59",
   "steps": [{"name": "preflight", "argv": ["python3", "-m", "main.human_data.cuda_preflight", "…"],
              "gpu": true, "mem_gb": 16, "timeout_s": 3600}]},
  {"kind": "arm", "arm": "r0", "seed": 3531, "pin": "<40-hex>", "run": "rb_x35_r0_s3531", "go_by": "…",
   "thin": "none", "thin_reason": "AM-R0: main.x35.r0 reads every 2M point", "disk_need_gib": 15.11}]}
```

The field rules:

- **Each arm has its OWN full 40-hex `pin`.** There is no chain-wide or default pin.
- **`run` must agree with the kit's describe.** Every later match is on that exact name.
- **`thin` is `planner` or `none` plus a `thin_reason`.** It is never a `case` on the run name.
- **`go_by` records whose GO the arm runs under.**
- **`status_file` must be `<job_dir>/status.txt`**, where the kit writes its lines; anything else is refused.
- **`models_dir`** is passed to every CPU tool (`GEN3AI_MODELS_DIR`, the thin planner's `--models-dir`), so the
  chain and its tools never read two different archives.
- **`verify` is an optional list of commands.** They run before the dry run and again just before the GO. The X52
  arms used this for `python -m main.x35.x52 verify`.
- **`banner` and `throughput` are the post-spawn guards.**
- **A window's `on_fail` says what a failed window means.** `hold` stops the chain. `continue` hands off to the
  next arm.

`x_*` keys are free extras; `x_dose_comparator` is copied into the dose record.

### The verbs

| verb | what it does |
|---|---|
| `plan CHAIN [--check-pins] [--describe]` | validates the file. `--check-pins` checks that each pin is a commit. `--describe` asks the kit for each arm's describe on CPU and checks the run name. A wrong arm name or run is refused when the chain is PLANNED, not at the hand-off |
| `run CHAIN` | the SUPERVISOR: walks the items from the saved state. Exit 0 = advanced or complete, 3 = HOLD, 4 = another supervisor holds this chain |
| `status CHAIN` | each item's phase and hold |
| `clear-hold CHAIN --item KEY --by WHO --why WHY` | the orchestrator's decision to RETRY the held step |
| `abandon CHAIN --item KEY --by WHO --why WHY` | the orchestrator's decision that a held item will not finish: it becomes `DONE_FAILED` and the chain hands off past it. Refused while the item's launch wrapper or launcher is alive (stop it first) |
| `live [--models-dir D]` | live runs, read from launcher argv (§4) |
| `watch-files --status-file F` | the alert watch's file list: `F` plus each live launcher's stdout log |
| `canary --out-dir D --status-file F` | the canary logger loop. It is bounded, writes a heartbeat file and an `EXPIRING` line |
| `cpu-gate` | one GO CPU-gate sample |
| `wait-pid PID --timeout-s N` | a `/proc` wait on `(pid, start ticks)` |
| `signal-if PID ERE [--signal S]` | the identity-gated signal (`kill_if.sh`'s replacement) |

**What the supervisor writes:**
- **State:** `<job_dir>/arm_chain/<name>.state.json`, rewritten atomically.
- **Lock:** `<name>.state.json.lock`, held for the supervisor's whole life.
- **Status lines:** `[ts] ARMCHAIN[<name>]: …`, appended to the job's `status.txt`.
- **Transcripts, scratch stages and record sources:** `<job_dir>/arm_chain/<name>/`.

**The Monitor filter:** `ARMCHAIN\[<name>\]: (HOLD|chain COMPLETE)|CANARYLOG: HOLD`.

## 3. The state machine

```
ARM     PENDING ─▶ PRECHECKED ─▶ GOING ─▶ LAUNCHED ─▶ SPAWNED ─▶ BANNER_OK ─▶ RUNNING ─▶ ENDED
        ─▶ COMPLETE ─▶ K9B_PASS ─▶ THINNED ─▶ RECORDED ─▶ DONE ─────────▶ (the next item, at ITS pin)
WINDOW  PENDING ─▶ STARTED ─▶ DONE | DONE_FAILED (on_fail continue)   any step ─▶ HOLD (exit 3)
```

| from | the step | a HOLD when |
|---|---|---|
| PENDING | disk ≥ the arm's need + headroom; the `verify` commands; the CPU gate (≤ 13 polls, 4 min apart); the kit's dry run at the arm's pin with the card visible; the describe re-read at its fixed path; the launcher transcript re-verified | any check fails, a file is stale, the describe names another run, or the transcript is not `FORK of <parent>` / `pin : <the pin>` / the step delta / `cuda devices : ✓` / `would launch` |
| PRECHECKED | a dry run older than 10 min (only a supervisor restart can age one) goes back to PENDING; the GPU must be idle (≤ 5 min wait); the END / START / pool-seed / HOLD triggers are armed and persisted, `GOING` is saved, then the kit's GO is spawned detached | the GPU stays busy; `verify` fails at the GO; `models/<run>` exists; the supervisor runs at a nice other than 0 (the trainer would inherit it for life) |
| GOING | (only after a supervisor died mid-GO) ADOPT the launch if there is evidence: a live `launch_x35.sh <arm> <seed> … GO`, a START line, or the run dir | no evidence. Clearing that hold resets the arm to PENDING; there is **never a blind second GO** |
| LAUNCHED | the wrapper's `post-spawn pool-seed check PASS` for this run (≤ 75 min after the SPAWN: the kit's GO re-runs its pre-checks, then the pool-seed check takes up to 30 min) | the wrapper wrote `HOLD:`, or it is gone without a PASS |
| SPAWNED | the banner guard, run over the launcher's stdout log AND `launcher_child.log` | a VOID line; or no expected banner by `within_min` |
| BANNER_OK | the first-hour throughput gate: marginal steps/s from the mtimes of the run's OWN checkpoints | no consecutive pair reaches the bar by `within_min` after the GO |
| RUNNING | the wrapper's `END <run> launcher_rc=N`. If the wrapper is gone but its launcher (pid file, `main.launcher` cmdline) lives, keep waiting | wrapper and launcher both gone without END |
| ENDED | completeness: rc 0, launcher `Last step` ≥ `--steps`, `final_model.zip`, newest checkpoint ≥ `--steps − 98,304` | any of the four fails: "stopped, not finished" |
| COMPLETE | `kit k9b --stage <scratch copy>`; main's records stage checked clean before AND after; the verdict read from the record's JSON | not PASS, or main's stage changed |
| K9B_PASS | the thin (repo planner `--thin-run`). It retries ONLY the "run dir open" refusal (10 × 30 s). The plan must keep the end-step checkpoint and every checkpoint a later item names; `final_model.zip` must survive | any other refusal; the plan drops a kept file; exit 4 (a hash mismatch) |
| THINNED | the dose record (the START→END segments of THIS GO only, summed) is written beside the K9(b) record. BOTH ship in one records-only commit (`ship.py`) | the dose cannot be sourced, the launcher log names another pin, or the ship HOLDs |
| RECORDED | DONE; the next item starts | — |

A window saves `STARTED` before its first step; a supervisor that restarts in `STARTED` HOLDs (a step may still hold the GPU) instead of re-running the window. Any exception other than a HOLD is turned into a HOLD line (`UNEXPECTED <type>`), so the Monitor sees it. A HOLD stops the supervisor at once. Nothing after the held step runs: no thin, no record, no next GO. Exactly one
status line says why.

**The ship is synchronous, on purpose.** The next arm's dry run runs `kit k9b-batch --stage <main's stage> --out-dir
<scratch>`: the stage is only READ, and a MISSING arm-end record is read and written to the scratch dir
(`$X35_K9B_OUT`, default `<job dir>/k9b_scratch`), never into a checkout. A next GO started before the previous record
landed therefore re-reads the arm and cannot dirty main; the synchronous ship keeps the record in the stage.

## 4. Who owns what

| concern | owner |
|---|---|
| the kit pre-checks (live run, lease, pin / required / corpus / stage / pool / K9(b)-batch checks, `checkargs --pin`, the launcher `--dry-run`, disk), the GO, the post-spawn pool-seed check, `END … launcher_rc=` | `launch_x35.sh` (repo, unchanged). The chain RE-VERIFIES its dry run (§3) but never re-implements a check |
| inside a run: restarts, `--gpu-window`, the deferred-abort save | the launcher |
| live-run discovery | `arm_chain.live`: each live `python -m main.launcher` (not `--dry-run`). The source is its trainer child's `--run-dir` (authoritative), else its own `--run-dir` / `--run-name` / a checkpoint-shaped `--model` (labelled: a fork with no `--run-name` would resolve to its PARENT), else UNRESOLVED with the reason. A trainer with no launcher is reported as an orphan |
| waits and signals | `arm_chain.procwatch`: identity is `(pid, start ticks)`, read from `/proc/<pid>/stat`; never `kill -0`; every wait is bounded; every signal checks the cmdline |
| the GO's environment | an ALLOW-list of the supervisor's environment (`runner.GO_ENV_KEEP`) plus `PYTHONPATH=<repo>/src` and the lease token file. A CPU step's `CUDA_VISIBLE_DEVICES=""`, `GEN3AI_TEST_ALLOW_GPU` or a worktree `PYTHONPATH` cannot reach a launch |

## 5. The class bugs, where each is closed, and its test

| # | the bug (first incident) | closed by | regression test |
|---|---|---|---|
| CB-01 | next-arm name accepted only at hand-off (`x47` vs `toy_x47`, 10-20 06:46; the GPU idled) | `plan --describe` | `launch_test::test_plan_describe_refuses_an_arm_the_kit_does_not_know`, `…_a_run_name_the_kit_does_not_give`; the real kit refused `x47` in the 10-26 probe |
| CB-02 | empty `CUDA_VISIBLE_DEVICES` reached the GO (toy_x47 3602) | `runner.go_env` allow-list, `assert_card_visible`, transcript `cuda devices : ✓` | `runner_test::test_go_env_drops_cuda_empty_and_allow_gpu_and_pins_pythonpath`, `chain_test::test_cross_pin_chain_hands_each_arm_its_own_pin_and_a_visible_card`, `launch_test::test_the_go_env_is_an_allow_list` |
| CB-03 | `load1` as the CPU meter | `cpugate` (trainer-core idle + named hogs; a hog born mid-sample counts) | `cpugate_test::test_agent_cores_busy_trainer_cores_idle_passes`, `…hog_born_during_the_window_is_named` |
| CB-05 | one ambient pin per chain (`X35_PIN`, toy default) | the pin is an item field; there is no default | `spec_test::test_abbreviated_pin_refused`, `test_two_arms_keep_their_own_pins`; `chain_test::test_cross_pin_chain…` |
| CB-06 | the VOID guard watched `launcher_child.log`; the banner is in the launcher's stdout | the banner guard reads both | `postspawn_test::test_banner_found_when_only_in_launcher_stdout_log`, `chain_test::test_the_banner_in_the_launcher_stdout_passes` |
| CB-08 | a stopped run read as finished (d1wide_s3501) | `armend.completeness` | `armend_test::test_the_d1wide_s3501_shape_is_stopped_not_finished`, `chain_test::test_a_stopped_run_is_never_k9b_read_thinned_or_chained` |
| CB-09 | thin HELD while the run's own exiting processes held the dir | retry only that refusal | `armend_test::test_the_open_run_dir_refusal_is_retried_and_others_are_not` |
| CB-10 | end-step checkpoint at risk; keep-all arms thinned by a name `case` | the plan must keep the newest checkpoint; `thin: none` is a field | `armend_test::test_a_plan_that_drops_the_end_step_checkpoint_holds`, `test_a_keep_all_arm_is_never_thinned`, `spec_test::test_thin_none_without_reason_refused` |
| CB-11 | K9(b) record written into main's tree (arm_end v3-v12) | `armend.k9b_scratch` + the ship from a copy | `armend_test::test_k9b_writes_only_the_scratch_stage_and_main_stays_clean`, `test_a_k9b_read_that_writes_into_mains_stage_holds`, `ship_test::test_a_source_inside_main_is_refused` |
| CB-13 | a bounded logger lapsed silently (canary v9) | `canary.watch` heartbeat + `EXPIRING` | `canary_test::test_watch_emits_expiring_on_last_pass_and_writes_heartbeat` |
| CB-14 / 15 | ship worktree lacked `deps/pokemon-showdown`; submodule blocked the remove; remove unchecked | `ship.Shipper`: submodules are checked out, the remove is guarded with `--force`, and main must be clean after | `ship_test::test_a_ship_lands_byte_identical_records_and_leaves_main_clean` |
| CB-16 | an END written while the chain was swapped was missed | triggers and their offsets are persisted at GO | `statuslog_test::test_persisted_trigger_fires_on_line_written_between_supervisors`, `chain_test::test_a_resumed_supervisor_sees_an_end_written_while_nobody_watched` |
| CB-17 | the END writer died with its parent | wait on the launcher's identity while it lives | `armend_test::test_a_dead_wrapper_with_a_live_launcher_keeps_the_wait_going` |
| CB-19 | one pytest for 33 GPU ids | one process per window step | `window_test::test_gpu_step_argv_and_env_one_process_per_step` |
| CB-21 | a thin deleted checkpoints a later replay hard-coded | `armend.later_evidence` | `armend_test::test_a_checkpoint_a_later_window_reads_blocks_the_thin` |
| CB-22 | name globs missed three run families | `live.discover` (launcher argv + trainer children) | `live_test::test_run_name_of_a_family_no_glob_knew_resolves`, `canary_test::test_watch_logs_a_live_run_discovered_without_a_glob` |
| CB-23 | a pid file held the wrapper's pid; pid reuse | `ProcRef` = pid + start ticks | `procwatch_test::test_wrong_start_ticks_reads_not_alive` |
| CB-24 | triggers matched their own "armed" line | anchored `^[ts] <foreign tag>: <body>$`; `append` refuses a line any armed trigger would match | `statuslog_test::test_quoting_armed_line_does_not_fire_trigger_and_trigger_tag_append_refused`, `test_trigger_on_writers_own_tag_refused` |
| CB-25 | triggers matched stale lines | offset at arming; truncation and inode change are errors | `statuslog_test::test_stale_line_before_arming_never_fires`, `test_truncated_file_raises`, `test_replaced_file_raises` |
| CB-26 | `kill -0` read a live supervisor as dead (acf2193fa) | `/proc` reads only | `procwatch_test::test_module_never_probes_liveness_with_kill_zero`, `armend_test::test_wait_end_never_uses_kill_zero` |
| CB-27 | a signal sent by name or by guess | `signal_if` | `procwatch_test::test_signal_if_refuses_cmdline_mismatch` |
| H-03 / 04 | unterminated run names (`…_s350` matched `…_s3501`; `|X35END: HOLD` matched any chain's HOLD) | every pattern escapes the name and ends it with a fixed delimiter | `statuslog_test::test_run_name_prefix_does_not_fire_longer_run*` |
| H-06 / 07 | describe found by `grep -l … \| head -1`; K9(b) judged by `grep -q PASS` | a fixed describe path, checked fresh; the verdict read from the JSON | `launch_test::test_a_describe_naming_another_run_holds`, `armend_test::test_k9b_reads_the_verdict_from_the_record_not_from_grep` |
| H-08 | a failing `nvidia-smi` read as an idle GPU | `window.nvidia_idle` | `window_test::test_nvidia_smi_failing_is_not_an_idle_gpu` |
| H-10 / 11 | canary seen-file keyed by run name; family bars by name prefix | keyed by `(path, inode)`; no family bar (§6 F8) | `canary_test::test_recreated_run_dir_with_fewer_rows_is_logged_from_row_one` |
| H-15 | a held ship left its worktree and branch, so the retry died | `Shipper._discard` before the push | `ship_test::test_a_failed_static_tier_pushes_nothing_and_removes_its_worktree` |
| H-17 / dup | one shared `arm_end.pid`; a duplicate supervisor (10-26) | per-chain state + an exclusive lock | `chain_test::test_one_supervisor_per_chain` |
| new | a supervisor dying mid-GO could re-GO | the `GOING` phase + adoption | `chain_test::test_a_supervisor_that_died_mid_go_never_issues_a_second_go`, `test_mid_go_death_without_evidence_holds_and_clearing_resets_to_pending` |

Two classes stay with their owners:
- **CB-04** (the launcher log reused on re-launch) is fixed in `launch_x35.sh`, which rotates the log at GO. The
  chain refuses a GO onto an existing run dir.
- **CB-07** (`--model` resume dropping typed flags) is fixed in the launcher.

**Validated against the real artifacts** (10-26, read-only):
- `plan --describe` against the real kit agrees for `n1p 3502` and `toy_x47 3601`, and refuses `x47` (CB-01).
- The transcript verifier passes the real dry runs of `rb_x35_n1p_s3501` and `rb_x35_d3mut_s3503`.
- `completeness` passes both runs.
- `dose_record` reproduces both shipped dose records field for field.

## 6. Hazards that remain (each a FINDING)

- **F1 (fixed). `kit k9b-batch` wrote a missing record into MAIN's stage.** It now writes to `--out-dir` (default
  `$GEN3AI_X35_K9B_SCRATCH` or `~/.cache/gen3ai/x35_k9b`, outside every checkout) and `batch.k9b_record` refuses any
  directory inside the main checkout (`batch_test.py::test_a_k9b_record_inside_the_main_checkout_is_refused`).
- **F2 (fixed in `launch_x35.sh`). It used `kill -0` on the launcher it spawned** (3 sites there, not 5). It now reads
  `/proc` (pid + start ticks, a zombie is gone) and SIGTERMs only a launcher that is still the same process
  (`launch_x35_test.py`). **Open:** `scripts/ops/watch_run.sh` (the launch script's watcher) has two `kill -0` sites.
- **F3. The supervisor's niceness is inherited by the GO.** A `nice -n 19` supervisor (rule 5b's default while a
  run is live) would start the trainer at nice 19 for life (RLIMIT_NICE). The GO now HOLDs at any nice but 0, so
  **start the supervisor at nice 0**; its own CPU is negligible, and its CPU children set their own `nice` /
  `taskset` / `GEN3AI_BLOCKING`.
- **F4. Only the fakes have exercised the GO path.** The real `launch_x35.sh … GO`, the real `kit k9b` and the real
  thin planner have not been run under the chain. Its first live transition is that test, and the cut-over watches
  it by hand (§7).
- **F5. The GPU lease is held by a `sleep 604800`** that ends ~10-30 08:05. The chain does not renew it. A window
  checks the owner; the kit checks it at the GO.
- **F6. The dose record's FPS and last step come from the NEWEST launcher log.** Wall time is summed over the
  START→END segments. A by-hand resume with a rotated log still needs a hand read.
- **F7. The CPU gate cannot see a process that exits inside its 60 s sample.** Its time moves to the parent's
  `cutime`.
- **F8. The canary logger is a behaviour change.**
  - It applies NO family bars, so the old `[descriptive: ratio>2.64]` tags are gone.
  - It alerts only on a non-PASS record above 0.54, and on liveness.
  - The no-first-record alert is at 60 min, not 40: a production cold start reached its first record at ~34 min.
  - Its log, seen file and alert keys are NEW files under `--out-dir`.
- **F9. Window GPU steps run with `GEN3AI_TEST_ALLOW_GPU=1` under the chain's lease.** This is correct only because
  the supervisor runs in the lease holder's session (root rule 5).
- **F10. A HOLD idles the GPU until a human decides.** That is by design: SOP §3 rule 3 says a window or chain that
  cannot finish cleanly is the orchestrator's decision. `abandon` is the way past an item that will never finish.
- **F11. The chain never signals a run.** A VOID banner or a failed throughput gate HOLDs and pages, and the run
  keeps training until the orchestrator stops it. The old `d3_redirect.sh` behaved the same way. `launch_x35.sh`'s
  own pool-seed SIGTERM is unchanged.

## 7. Cut-over checklist (the training session, at a quiet hand-off; NOT done yet)

1. **Preconditions.** This module is on main. The training session has read this doc. The orchestrator's standing
   delegation of the records-only ship covers the chain's combined K9(b) + dose commit; confirm it names that
   commit.
2. **Choose the hand-off.** It must be after an arm-end has written `X35END: DONE <run>` and both of its ships have
   written `shipped`, and before the next GO. No `arm_end*`, `poll_launch*` or window script may be alive: check by
   argv in `/proc`, for example `python -m main.ops.arm_chain live` and `ps -o pid,args`.
3. **Write the chain file** as `~/.claude/jobs/x35/chains/<name>.json`, holding the remaining queue in order. For
   each arm:
   - its own pin (from the orchestrator's queue or `batch.ARM_PINS`);
   - its canonical arm name and its run;
   - `thin: none` with the AM reason for R0, X47 and X52;
   - its `banner` (D3 and N1-P carry `--team-mix`);
   - a `throughput` bar (C0's 1,384-1,424 steps/s means about 1,000 as a floor);
   - `disk_need_gib`, taken from the last dry run's REQUIRED GiB.

   For each window: its steps, `on_fail` and its preempt marker.
4. **`python -m main.ops.arm_chain plan <file> --check-pins --describe` must print `PLAN OK`.**
5. **Retire the old chain without signalling it mid-step.** Let the last old arm-end finish with next `none -`, or
   do not arm one. Confirm that no old chain script is alive.
6. **Start the canary logger.** Run `python -m main.ops.arm_chain canary --out-dir ~/.claude/jobs/x35/canary
   --status-file ~/.claude/jobs/x35/status.txt` under `nohup setsid`. After its first pass writes the heartbeat,
   stop `canary_log_v12.sh` by PID through the identity gate (`signal-if <pid> 'canary_log_v12'`).
7. **Start the supervisor** from the MAIN checkout at nice 0 (F3), with the env's interpreter and
   `PYTHONPATH=$PWD/src`:
   `nohup setsid python -m main.ops.arm_chain run <file> >> ~/.claude/jobs/x35/arm_chain/<name>.out 2>&1 < /dev/null &`
   - Point the Monitor at the filter in §2.
   - Keep the 55-min fallback cron. Its repair commands are `status`, then `clear-hold` or `abandon` on the
     orchestrator's word, then `run`.
8. **Watch the first GO by hand** (F4). Check:
   - the kit's `START` line and its pin;
   - the `GOING → LAUNCHED → SPAWNED` lines;
   - `ps` showing exactly one launcher.

   At the first arm end, check:
   - the scratch K9(b) record;
   - that `git status` on main is clean;
   - the combined records commit.
9. **Rollback.** Nothing in the job directory was changed, so the old scripts still work. After a module HOLD:
   - `abandon` the item;
   - stop the supervisor (it holds nothing but its lock);
   - re-arm `arm_end_v13.sh` / `poll_launch_v5.sh` by hand.
10. **After two clean arms,** update this doc's status line and TRAINING_RUN_SOP §2 to remove "cut-over pending".
    Then move the job scripts aside as history; never delete them before then.

## 8. Decision record

- **2026-10-26: built.**
  - One owner per check: the kit's pre-checks stay in `launch_x35.sh`, and the chain re-verifies the dry run
    instead of copying them.
  - The ship is synchronous (F1).
  - The K9(b) and dose records ship as ONE commit, where there used to be two ships in sequence.
  - The canary logger drops the family bars (F8).
  - `GOING` is a phase so that a supervisor death can never cause a second GO.
  - An independent review (10-26) found 13 issues before landing; all are fixed with a regression test each: the
    GO HOLDs at a nice other than 0; a window has a `STARTED` phase; the GPU must be idle before a GO; a restart
    never GOes on a dry run older than 10 min; the dose reads only THIS GO's segments; the post-spawn budget starts
    at the spawn; `abandon` refuses a live launch; a mid-GO reset takes the orchestrator's edit; any exception
    becomes a HOLD line; the ship is idempotent after a death past the push and retries a beaten push once; the GO
    environment pins `GEN3AI_PYTHON` and drops `GEN3AI_BLOCKING`; the canary logger never reads a half-written row.
  - A window's failure semantics are a field (`on_fail`), never a script's choice.
