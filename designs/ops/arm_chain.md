# The arm chain: one tested module that operates production arms end to end

> Always-current (`designs/CLAUDE.md`). Code: [`src/main/ops/arm_chain/`](../../src/main/ops/arm_chain/), entry point
> `python -m main.ops.arm_chain`. Tests: `src/main/ops/arm_chain/*_test.py` (254; fake `/proc`, a fake
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
   "throughput": {"min_steps_per_s": 1000, "within_min": 60, "first_line_within_min": 90},
   "on_void": "stop", "on_throughput_fail": "page"},
  {"kind": "window", "name": "x59s", "worktree": "/home/goodlad/dev/gen3ai-wt-x59s", "require_commit": "a1370436",
   "budget_min": 480, "on_fail": "continue", "preempt_marker": "NO_GO_X59",
   "steps": [{"name": "preflight", "argv": ["python3", "-m", "main.human_data.cuda_preflight", "…"],
              "gpu": true, "mem_gb": 16, "timeout_s": 3600}]},
  {"kind": "arm", "arm": "r0", "seed": 3531, "pin": "<40-hex>", "run": "rb_x35_r0_s3531", "go_by": "…",
   "thin": "none", "thin_reason": "AM-R0: main.x35.r0 reads every 2M point", "disk_need_gib": 15.11,
   "on_void": "page", "on_throughput_fail": "page"}]}
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
- **`banner` and `throughput` are the post-spawn guards.** The throughput gate counts `within_min` (default 60) from
  the FIRST TRAINER LINE in `launcher_child.log` (the launcher's `child attached` header is not one), or from the
  first own checkpoint if that comes first, never from the GO: a cold start took ~22-23 min from the GO to its first
  line. No trainer line `first_line_within_min` (default 90) after the GO fails the gate too.
- **`on_void` and `on_throughput_fail` are REQUIRED on every arm, `stop` or `page`, with no default.** `stop` is the
  SOP stop path: SIGTERM the launcher through `signal_if` (its pid file's process, and only if its cmdline is THIS
  run's `main.launcher … --run-name <run>`), so it saves `final_model_interrupted.zip` at its next safe point; wait
  up to 300 s for it to go; record the stop in the state (`data.stopped`) and a status line; HOLD. `page` HOLDs
  only, and the run keeps training until the orchestrator decides.
- **A window's `on_fail` says what a failed window means.** `hold` stops the chain. `continue` hands off to the
  next arm.

`x_*` keys are free extras; `x_dose_comparator` is copied into the dose record.

### The verbs

| verb | what it does |
|---|---|
| `plan CHAIN [--check-pins] [--describe]` | validates the file. `--check-pins` checks that each pin is a commit. `--describe` asks the kit for each arm's describe on CPU and checks the run name. A wrong arm name or run is refused when the chain is PLANNED, not at the hand-off |
| `run CHAIN` | the SUPERVISOR: walks the items from the saved state. Exit 0 = advanced or complete, 3 = HOLD, 4 = another supervisor holds this chain |
| `status CHAIN` | first the supervisor's liveness, `supervisor: ALIVE pid N` / `STALE` / `DEAD` (`!!!` in front when an item is mid-flight with no supervisor, or the heartbeat is past its own deadline), then each item's phase and hold. It never takes the lock (§3, "Liveness") |
| `clear-hold CHAIN --item KEY --by WHO --why WHY [--resolution R]` | the orchestrator's decision on a held item, recorded in the history and a status line. `retry` (default) re-runs the held step; `relaunch` and `accept` are below. Refused while a supervisor holds the lock |
| `abandon CHAIN --item KEY --by WHO --why WHY` | the orchestrator's decision that a held item will not finish: it becomes `DONE_FAILED` and the chain hands off past it. Refused while the item's launch wrapper or launcher is alive (stop it first) |
| `ship-records CHAIN --item KEY` | ships one arm's K9(b) + dose records from its request file: the async worker the supervisor spawns, and the by-hand retry after a `SHIP-HOLD` (idempotent: records already on the remote are a success) |
| `live [--models-dir D]` | live runs, read from launcher argv (§4) |
| `watch-files --status-file F` \| `--chain CHAIN` | the alert watch's file list: the status file, each live launcher's stdout log, and its run's `launcher_child.full.log` (the trainer's own lines; it rotates by rename, which `tail -F` follows). `--chain` adds every unfinished arm's two fixed log paths, existing or not (`tail -F` waits for them) |
| `monitor-filter --chain NAME` | the one ERE the alert Monitor greps over the watch-files (below) |
| `monitor --chain CHAIN [--max-hours 24]` | the alert FOLLOWER (§7 step 7): every 10 s it re-reads the file set (`watch-files --chain`'s), follows each file `tail -F` style, prints each line the filter matches as `[<file>] <line>`, and pages a dead or stale supervisor once (`SUPERVISOR DEAD\|STALE:` in the status file). No re-arm at a GO; bounded, with an `EXPIRING` line |
| `canary --out-dir D --status-file F [--archive-root R]` | the canary logger loop. It is bounded, writes a heartbeat file and an `EXPIRING` line, alerts on EVERY non-PASS record, and archives each canary weight dump to `R/<run>_canary_u<u>/` (default `~/gen3ai_archive`) |
| `cpu-gate` | one GO CPU-gate sample |
| `wait-pid PID --timeout-s N` | a `/proc` wait on `(pid, start ticks)` |
| `signal-if PID ERE [--signal S]` | the identity-gated signal (`kill_if.sh`'s replacement) |

**What the supervisor writes:**
- **State:** `<job_dir>/arm_chain/<name>.state.json`, rewritten atomically.
- **Lock:** `<name>.state.json.lock`, held for the supervisor's whole life (its pid is written in it).
- **Heartbeat:** `state["supervisor"]`: pid, start ticks, the commit its code came from, `last_beat`, what it is doing,
  and `stale_after_epoch`. The supervisor beats before every blocking step, declaring that step's bound plus 10 min,
  and records `exit` / `exited_at` when it stops.
- **Status lines:** `[ts] ARMCHAIN[<name>]: …`, appended to the job's `status.txt`.
- **Transcripts, scratch stages and record sources:** `<job_dir>/arm_chain/<name>/`; each async ship's request,
  result and worker output under its `ships/`.

**The Monitor filter** (`python -m main.ops.arm_chain monitor-filter --chain <name>` prints it; a test pins this
line to the code): `ARMCHAIN\[<name>\]: (HOLD|SHIP-HOLD|chain COMPLETE|MONITOR-REARM|SUPERVISOR (DEAD|STALE))|CANARYLOG: HOLD|AWFILES: HOLD|Traceback|Child crashed|WEDGE|UpdateWontFit|BEHAVIOUR-POLICY MISMATCH|StampMismatch|DISAGREES`.
`MONITOR-REARM <item>: <launcher log> <launcher_child.full.log>` is written at each GO: the cue for a `tail -F`
Monitor to re-arm on the new run's logs. `SUPERVISOR DEAD|STALE` is written by `monitor`.
The part after `AWFILES: HOLD` is the launcher-log alert set (`canary.LAUNCHER_ALERT_ERE`): a crash, a wedge, an
update that cannot fit, a behaviour-policy mismatch, a Rust stamp mismatch, a compiled-vs-eager disagreement.

## 3. The state machine

```
ARM     PENDING ─▶ PRECHECKED ─▶ GOING ─▶ LAUNCHED ─▶ SPAWNED ─▶ BANNER_OK ─▶ RUNNING ─▶ ENDED
        ─▶ COMPLETE ─▶ K9B_PASS ─▶ THINNED ─▶ RECORDED ─▶ DONE ─────────▶ (the next item, at ITS pin)
WINDOW  PENDING ─▶ STARTED ─▶ DONE | DONE_FAILED (on_fail continue)   any step ─▶ HOLD (exit 3)
```

| from | the step | a HOLD when |
|---|---|---|
| PENDING | disk ≥ the arm's need + headroom; main clean; the `verify` commands; the CPU gate (≤ 13 polls, 4 min apart); the kit's dry run at the arm's pin with the card visible; the describe re-read at its fixed path; the launcher transcript re-verified; main clean again | any check fails, a file is stale, the describe names another run, the transcript is not `FORK of <parent>` / `pin : <the pin>` / the step delta / `cuda devices : ✓` / `would launch`, or main is dirty before or after the dry run (re-read 3 times, 5 s apart, so an async ship's own fast-forward is never read as dirt) |
| PRECHECKED | a dry run older than 10 min (only a supervisor restart can age one) goes back to PENDING; main clean; the GPU must be idle (≤ 5 min wait); no other launch is live; the END / START / pool-seed / HOLD triggers are armed and persisted, `GOING` is saved, then the kit's GO is spawned detached and `MONITOR-REARM` is written | main is dirty; the GPU stays busy; a live GO wrapper of ANY arm or a live launcher / trainer (`live.discover`) exists; `verify` fails at the GO; `models/<run>` exists; the supervisor runs at a nice other than 0 (the trainer would inherit it for life) |
| GOING | (only after a supervisor died mid-GO) ADOPT the launch if there is evidence: a live `launch_x35.sh <arm> <seed> … GO`, a START line, or the run dir | no evidence. Clearing that hold resets the arm to PENDING; there is **never a blind second GO** |
| LAUNCHED | the wrapper's `post-spawn pool-seed check PASS` for this run (≤ 75 min after the SPAWN: the kit's GO re-runs its pre-checks, then the pool-seed check takes up to 30 min) | the wrapper wrote `HOLD:`, or it is gone without a PASS. A `retry` re-reads that same line: clear it with `relaunch` or `accept` |
| SPAWNED | the banner guard, run over the launcher's stdout log AND `launcher_child.log` | a VOID line; or no expected banner by `within_min`, or by the wrapper's END line if the run ends first: the arm's `on_void` (`stop`: the stop is persisted, then SIGTERM + record, then HOLD, and a restart never sends a second SIGTERM; `page`: HOLD) |
| BANNER_OK | the throughput gate: the first trainer line (or first own checkpoint) is found and its time persisted, then marginal steps/s from the mtimes of the run's OWN checkpoints | no trainer line by `first_line_within_min` after the GO, or no consecutive pair reaches the bar by `within_min` after the first trainer line, or by the END line if the run ends first: the arm's `on_throughput_fail` |
| RUNNING | the wrapper's `END <run> launcher_rc=N`. If the wrapper is gone but its launcher (pid file, `main.launcher` cmdline) lives, keep waiting | wrapper and launcher both gone without END |
| ENDED | completeness: rc 0, launcher `Last step` ≥ `--steps`, `final_model.zip`, newest checkpoint ≥ `--steps − 98,304` | any of the four fails: "stopped, not finished" |
| COMPLETE | `kit k9b --stage <scratch copy>`; main's records stage checked clean before AND after; the verdict read from the record's JSON | not PASS, or main's stage changed |
| K9B_PASS | the thin (repo planner `--thin-run`). It retries ONLY the "run dir open" refusal (10 × 30 s). The plan must keep the end-step checkpoint and every checkpoint a later item names; `final_model.zip` must survive | any other refusal; the plan drops a kept file; exit 4 (a hash mismatch) |
| THINNED | the dose record (the START→END segments of THIS GO only, summed) is written beside the K9(b) record. BOTH ship in one records-only commit (`ship.py`): ASYNCHRONOUSLY (a detached `ship-records` worker) once `ship.kit_k9b_out_of_main` passes, else inline | the dose cannot be sourced, the launcher log names another pin, or (inline only) the ship HOLDs |
| RECORDED | DONE; the next item starts | — |
| (chain end) | wait ≤ 45 min for in-flight ships; `chain COMPLETE` only when every record landed | a ship failed or is still in flight: `SHIP-HOLD:` and exit 3 |

A window saves `STARTED` before its first step; a supervisor that restarts in `STARTED` HOLDs (a step may still hold the GPU) instead of re-running the window. Any exception other than a HOLD is turned into a HOLD line (`UNEXPECTED <type>`), so the Monitor sees it; one that escapes the supervisor itself (a full disk under its own save, a SIGTERM or SIGHUP) leaves `HOLD: supervisor died: <type>: …`. A HOLD stops the supervisor at once. Nothing after the held step runs: no thin, no record, no next GO. Exactly one
status line says why. A restarted supervisor on a held chain writes `HOLD: <item> is still HELD …` and exits 3.

**Clearing a hold** (`clear-hold --resolution`; the minimal sound set, each recorded with its resolution):
- **`retry`** re-runs the held step. GOING and STARTED reset to PENDING. Elsewhere the step re-reads its evidence, which
  is right for a cause outside the persisted evidence (a dirty main, a busy GPU, a ship), and useless at LAUNCHED after
  a wrapper HOLD (review finding 3: it held again forever).
- **`relaunch`** (LAUNCHED only) resets the arm to PENDING for a fresh dry run and GO; the triggers are re-armed past
  the old lines, and the old attempt's data is kept under `data.attempts`. It is refused unless the kit provably never
  launched: the GO wrapper is gone, no live GO wrapper for the arm, no live launcher in the pid file, and no
  `models/<run>`. A run dir means the kit launched and refuses a GO onto it: a human decides (accept, abandon, or move
  the run dir aside first). A "fresh run name" is not a resolution: the kit names the run from the arm and seed, so a
  new name is a new item with a new seed, appended after `abandon`.
- **`accept`** advances an arm past the post-spawn gate that held (LAUNCHED → SPAWNED, SPAWNED → BANNER_OK,
  BANNER_OK → RUNNING; finding 9's paged arm) on the orchestrator's evidence, recorded in `data.accepted`; every
  later gate still runs. It is refused for a run its registered `stop` signalled, and at LAUNCHED without
  `models/<run>` (nothing launched to accept).

**Liveness** (`liveness.py`). `status` reads three things and takes no lock: a `status` that took the lock even for an
instant could make a supervisor starting at that moment exit 4.
- The **lock holder**: the `FLOCK` entry in `/proc/locks` on the lock file's device and inode, whose pid still has
  that file open. Every supervisor version takes this lock, so this also sees a supervisor older than the heartbeat.
- The **heartbeat** (`state["supervisor"]`), written by the supervisor's main thread, so a wedged loop goes STALE where
  a liveness thread would keep beating.
- **Mid-flight**: an item started, neither terminal nor held.

The verdicts:
- `ALIVE`: the lock is held.
- `STALE`: the lock is held, but now is past the heartbeat's own `stale_after_epoch`.
- `DEAD`: nobody holds the lock. It is loud with an item mid-flight, naming any live launcher of that item.
- `UNKNOWN`: `/proc/locks` cannot be read.

`clear-hold` and `abandon` take the lock non-blocking: a live supervisor means nothing is held, and its next save would
overwrite theirs. Every state save goes through a per-process tmp file, fsync'd, then `os.replace`d, then the directory
is fsync'd.

**The ship is asynchronous.** A synchronous ship idled the GPU for the whole ship, and a ship HOLD idled it until a
human came. The worker writes its own status line (`… records shipped as <sha> (async)`, or `SHIP-HOLD: <item>:
<why>; retry: python -m main.ops.arm_chain ship-records <chain> --item <item>`), because the supervisor may be deep in
the next arm's wait. A worker that dies without a result is a `SHIP-HOLD` the supervisor writes. The next arm's dry
run runs `kit k9b-batch --stage <main's stage> --out-dir <scratch>` while the previous records may still be in flight:
the stage is only READ, and a MISSING record is read and written to the scratch dir (`$X35_K9B_OUT`, default
`<job dir>/k9b_scratch`), never into a checkout (F1, fixed at `7f9ff6236`). The async path is taken only while
`ship.kit_k9b_out_of_main` reads BOTH halves of that fix in MAIN (every `kit k9b-batch` in the launch script passes
`--out-dir`; `main.x35.batch.k9b_batch` takes `out_dir` and `refuse_inside_main_checkout` refuses a write inside the
main checkout); otherwise the ship goes inline and says `SYNCHRONOUS (<why>)`. The main-clean checks around every
dry run and GO stand either way.

## 4. Who owns what

| concern | owner |
|---|---|
| the kit pre-checks (live run, lease, pin / required / corpus / stage / pool / K9(b)-batch checks, `checkargs --pin`, the launcher `--dry-run`, disk), the GO, the post-spawn pool-seed check, `END … launcher_rc=` | `launch_x35.sh` (repo, unchanged). The chain RE-VERIFIES its dry run (§3) but never re-implements a check |
| inside a run: restarts, `--gpu-window`, the deferred-abort save | the launcher |
| live-run discovery | `arm_chain.live`: each live `python -m main.launcher` (not `--dry-run`). The source is its trainer child's `--run-dir` (authoritative), else its own `--run-dir` / `--run-name` / a checkpoint-shaped `--model` (labelled: a fork with no `--run-name` would resolve to its PARENT), else UNRESOLVED with the reason. A trainer with no launcher is reported as an orphan |
| waits and signals | `arm_chain.procwatch`: identity is `(pid, start ticks)`, read from `/proc/<pid>/stat`; never `kill -0`; every wait is bounded; every signal checks the cmdline |
| the GO's environment | an ALLOW-list of the supervisor's environment (`runner.GO_ENV_KEEP`) plus `PYTHONPATH=<repo>/src` and the lease token file. A CPU step's `CUDA_VISIBLE_DEVICES=""`, `GEN3AI_TEST_ALLOW_GPU` or a worktree `PYTHONPATH` cannot reach a launch |
| what a VOID / a failed throughput gate does | the chain file (`on_void`, `on_throughput_fail`, per arm, required); the chain only executes it |
| the GPU lease token inside a window | `window.lease_token_valid` before every `gpu_lock` step (read-only: `utils.gpu_lease.read_lease` + `token_matches`); an invalid token HOLDs whatever `on_fail` says |
| deleting anything | `fsguard.remove_under` only: a path strictly inside the caller's root, never through a symlink (CB-28) |
| the daily K9(b) hand read of a live arm, and LR-F's VOID-KL read | **the training session**, by hand, as today. The chain reads K9(b) only at an arm's END (AM-6) and reads no KL |

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
| CB-11 | K9(b) record written into main's tree (arm_end v3-v12; and by the kit's `k9b-batch`, F1, fixed at `7f9ff6236`) | `armend.k9b_scratch` + the ship from a copy; main clean before and after every dry run and before every GO; async ship only while `kit_k9b_out_of_main` passes | `armend_test::test_k9b_writes_only_the_scratch_stage_and_main_stays_clean`, `test_a_k9b_read_that_writes_into_mains_stage_holds`, `ship_test::test_a_source_inside_main_is_refused`, `chain_test::test_a_dry_run_that_dirties_main_holds_and_never_goes`, `test_a_dirty_main_holds_before_the_dry_run`, `test_an_unfixed_kit_keeps_the_ship_synchronous`, `test_the_kit_check_refuses_until_the_launch_script_and_main_code_write_outside_main` |
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
| H-10 / 11 | canary seen-file keyed by run name; family bars by name prefix silently applied C0's bar to new families | keyed by `(path, inode)`; the family bar is `canary_decide`'s registered one, a DESCRIPTIVE tag only, never an alert (§6 F8) | `canary_test::test_recreated_run_dir_with_fewer_rows_is_logged_from_row_one`, `test_a_ratio_above_the_family_bar_is_a_descriptive_tag_that_never_alerts` |
| H-15 | a held ship left its worktree and branch, so the retry died | `Shipper._discard` before the push | `ship_test::test_a_failed_static_tier_pushes_nothing_and_removes_its_worktree` |
| H-17 / dup | one shared `arm_end.pid`; a duplicate supervisor (10-26) | per-chain state + an exclusive lock | `chain_test::test_one_supervisor_per_chain` |
| new | a supervisor dying mid-GO could re-GO | the `GOING` phase + adoption | `chain_test::test_a_supervisor_that_died_mid_go_never_issues_a_second_go`, `test_mid_go_death_without_evidence_holds_and_clearing_resets_to_pending` |
| CB-28 | an `rm` / `shutil.rmtree` on a variable path (one bad join from a run dir) | `fsguard.remove_under`, the module's only delete | `fsguard_test::test_the_module_never_removes_a_variable_path_outside_the_guarded_helper` (an AST scan of every module), `test_remove_under_removes_only_strictly_inside_its_root`, `…never_follows_a_symlink_out_of_its_root` |
| CB-29 | a non-PASS canary below a value bar paged nobody (N1-P s3501 u200 UNCONFIRMED, compiled 0.1675 < 0.54; 10-27) | every non-PASS verdict alerts, whatever its value | `canary_test::test_format_row_readings_and_every_non_pass_alerts`, `test_scan_alerts_on_every_non_pass_row_only` |
| CB-30 | the canary weight dumps lived only in the run dir (near-margin dumps rotate to the newest 3; a thin or move loses the rest) | `canary.archive_dumps`: a sha256-verified copy to `~/gen3ai_archive/<run>_canary_u<u>/`, never over a copy | `canary_test::test_every_unconfirmed_fatal_and_near_dump_is_archived_with_its_sha256`, `test_the_watch_loop_archives_a_live_runs_dump` |
| CB-31 | launcher-log failures (a Traceback, `Child crashed`, a WEDGE, `UpdateWontFit`, a BEHAVIOUR-POLICY MISMATCH, a `StampMismatch`, `DISAGREES`) paged nobody | `canary.LAUNCHER_ALERT_ERE` in `monitor_filter`; `watch-files` adds each run's `launcher_child.full.log` | `canary_test::test_the_monitor_filter_pages_every_launcher_alert_and_the_chain_and_canary_holds`, `test_the_design_doc_carries_the_monitor_filter_the_code_prints` |
| CB-32 | what a VOID / a slow start DOES was a script's choice (the chain only paged) | `on_void` / `on_throughput_fail` required per arm; `stop` through `signal_if` | `spec_test::test_each_arm_must_register_its_action_with_no_default`, `chain_test::test_a_void_arm_registered_stop_is_sigtermed_through_the_identity_gate_recorded_and_held`, `…registered_page_is_held_and_never_signalled`, `test_a_failed_throughput_gate_registered_stop_is_sigtermed`, `test_a_stop_signals_nothing_without_this_runs_live_launcher`, `test_the_stop_identity_never_matches_another_run` |
| CB-33 | the throughput gate counted from the GO, so a healthy cold start (~22-23 min to its first line) failed it | the gate counts from the first trainer line (or first own checkpoint) | `chain_test::test_the_throughput_gate_counts_from_the_first_trainer_line_not_the_go`, `postspawn_test::test_the_gate_counts_from_the_first_trainer_line_not_the_go`, `test_the_launchers_child_attached_header_is_not_a_trainer_line` |
| CB-34 | a window checked the lease OWNER's name; an expired token passed it and `gpu_lock` refused the step | `window.lease_token_valid` before every `gpu_lock` step; HOLD | `window_test::test_an_invalid_lease_token_holds_before_any_gpu_step`, `test_the_token_is_checked_again_before_each_gpu_step`, `test_lease_token_valid_reads_the_lease_without_changing_it` |
| CB-35 | a ship HOLD idled the GPU until a human came | the async ship (`ship-records` worker, `SHIP-HOLD:`), gated on the kit fix | `chain_test::test_an_async_ship_lets_the_next_arm_go_while_the_records_are_in_flight`, `test_a_failed_async_ship_pages_ship_hold_without_idling_the_gpu`, `test_a_ship_worker_that_dies_without_a_result_pages`, `test_the_ship_worker_writes_its_result_and_one_line_and_is_the_retry` |
| CB-36 | a dead supervisor was invisible: `status` read `RUNNING` while nobody would hand off (review 10-26, finding 2) | the heartbeat, the lock-holder probe and `status`'s liveness line; `monitor` pages `SUPERVISOR DEAD\|STALE`; an exception or signal that escapes the supervisor leaves `HOLD: supervisor died` | `chain_review_test::test_a_live_supervisor_reads_alive_with_its_heartbeat_and_what_it_is_doing`, `…a_beat_before_a_blocking_step_declares_its_bound`, `…a_dead_supervisor_with_a_live_run_is_shown_loudly`, `…a_wedged_supervisor_reads_stale_and_a_held_chain_reads_dead_quietly`, `…an_exception_that_escapes_the_supervisor_leaves_one_hold_line`, `…the_supervisor_turns_a_sigterm_into_its_hold_line`, `…the_monitor_pages_a_dead_supervisor_once` |
| CB-37 | an arm held at LAUNCHED could not be recovered: `clear-hold` re-read the same wrapper HOLD forever (finding 3); a paged arm had no path forward (finding 9) | `clear-hold --resolution relaunch \| accept` | `chain_review_test::test_a_kit_hold_at_the_go_is_relaunched_fresh_never_retried_forever`, `…relaunch_is_refused_once_the_kit_launched`, `…accept_at_launched_advances_and_every_later_gate_still_runs`, `…a_paged_banner_is_accepted_past_its_gate`, `…accept_is_refused_for_a_stopped_run` |
| CB-38 | a Monitor armed before a GO followed none of the new run's logs (finding 4) | `monitor` (re-reads the file set each pass); `MONITOR-REARM` at each GO; `watch-files --chain` | `chain_review_test::test_the_monitor_follows_logs_that_appear_after_it_started_with_no_rearm`, `…a_go_writes_a_monitor_rearm_line_the_filter_matches`, `…watch_files_for_a_chain_lists_each_unfinished_arms_fixed_paths` |
| CB-39 | a timed-out step's python children (a dry run holding a CUDA context) survived it (finding 5) | each step in its own process group; a timeout SIGTERMs the group, SIGKILLs it 10 s later | `runner_test::test_a_timeout_ends_the_whole_process_group_not_just_the_child` |
| CB-40 | a run that died in its first hour was noticed only at its gate's deadline (finding 6) | the banner and throughput gates watch the END line and judge the complete logs at once | `postspawn_test::test_a_run_that_ended_fails_its_gates_at_once`, `chain_review_test::test_a_run_that_ends_during_the_banner_guard_fails_at_once` |
| CB-41 | a GO beside another live GO wrapper or launcher was not refused (finding 7) | `_refuse_a_second_go` (`launch.live_go_wrappers`, `live.discover`) | `chain_review_test::test_a_go_beside_a_live_launch_is_refused` |
| CB-42 | a restart between the `stop`'s SIGTERM and its record sent a second SIGTERM (finding 8) | `data.stopping` persisted before the signal | `chain_review_test::test_a_supervisor_that_died_after_its_sigterm_never_sends_a_second` |
| CB-43 | the CLI wrote the state without the lock, through a shared tmp file, unsynced (finding 10) | `_cli_lock`; a per-process tmp file; fsync | `chain_review_test::test_clear_hold_and_abandon_refuse_under_a_live_supervisor_and_saves_are_fsynced` |
| CB-44 | `ship-records` loaded the state, so a chain-file edit to a started item killed it with no result (finding 11) | the worker builds its shipper from the chain FILE (`default_shipper`); a missing request writes a result and pages | `chain_review_test::test_ship_records_survives_a_chain_file_edit_to_a_started_item`, `…a_ship_worker_with_no_request_writes_a_result_and_pages` |
| CB-45 | a failure after the push (main's `index.lock`) left the ship's worktree and branch behind (finding 12) | main's fast-forward is retried, then the worktree is removed whatever happens; the retry lands through `_already_shipped` | `ship_test::test_a_failure_after_the_push_cleans_up_and_the_retry_lands` |
| CB-46 | a chain file without `x_dose_comparator` silently dropped `c0_comparator` (finding 13) | `chain.dose_comparator`: the chain file's (`null` omits it), else the block every dose record in main's stage agrees on, else none; a status line says which | `chain_review_test::test_the_dose_record_takes_the_comparator_the_stages_records_agree_on` |
| restart | the 10-26 supervisor (`93a05fe91`) is replaced at a restart by this code, onto the state it wrote | the state schema stays `arm_chain_state_v1`: every addition (`supervisor`, `data.stopping` / `attempts` / `accepted`) is optional, and each version keeps the other's keys | `chain_review_test::test_a_93a05fe9_*` on `testdata/state_93a05fe9_*.json` (written by `93a05fe91`'s code on these fakes: held at LAUNCHED twice, held at SPAWNED, mid-RUNNING, an async ship) |

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
- **F12. The async-ship check keys on the kit fix's interface** as it landed at `7f9ff6236`: `k9b_batch(…, out_dir=…)`,
  `refuse_inside_main_checkout(path, roots=…)` and `--out-dir` on the launch script's `k9b-batch` line. A later rename
  fails the check closed (the ship goes synchronous, and says so) until the check is updated; the test
  `chain_test::test_the_kit_check_passes_on_this_checkouts_real_kit` fails first.
- **F13. An async ship moves main while the next arm pre-checks.** Its `merge --ff-only` lands records (and anyone
  else's commits) under a running dry run, exactly as any other ship to main does. The launcher runs at the arm's
  pin; the kit (`launch_x35.sh`, `main.x35.kit`) runs main's code, so a kit change landing mid-pre-check is seen.
- **F14. The first trainer line's time is when the gate SAW it** (≤ 30 s late), persisted once. A supervisor that
  restarts in `BANNER_OK` before persisting it starts the clock at its own first read (later, so more lenient).
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
  - It alerts on EVERY non-PASS record (UNCONFIRMED, FATAL, an unparsable row), whatever its value, and on
    liveness. A PASS never alerts.
  - A ratio above the run family's REGISTERED bar (`main.ops.canary_decide.FAMILY_BARS`: C0 2.64, X47 2.40, family
    read from the run name) is a descriptive tag on the log line, `[descriptive: ratio>2.64 (c0 bar)]`; a run of no
    registered family gets no tag.
  - The no-first-record alert is at 60 min, not 40: a production cold start reached its first record at ~34 min.
  - Its log, seen file and alert keys are NEW files under `--out-dir`; its dump copies go to `--archive-root`.
- **F9. Window GPU steps run with `GEN3AI_TEST_ALLOW_GPU=1` under the chain's lease.** This is correct only because
  the supervisor runs in the lease holder's session (root rule 5).
- **F10. A HOLD idles the GPU until a human decides.** That is by design: SOP §3 rule 3 says a window or chain that
  cannot finish cleanly is the orchestrator's decision. `abandon` is the way past an item that will never finish.
- **F15. The heartbeat is a DECLARED bound, not a ticker.** The supervisor beats before each runner step (its own
  timeout) and each sleep, plus 10 min of slack. A blocking call outside the runner (git, the kit probe ≤ 120 s, a
  60 s CPU-gate sample) relies on that slack. STALE therefore means "past its own declared deadline", not "slow".
- **F16. `accept` skips a gate on the orchestrator's word.** It is recorded (`data.accepted`, the history, a status
  line), and every later gate still runs.
- **F17. `monitor` writes `SUPERVISOR DEAD|STALE` under the chain's own tag.** It is a second writer of that tag,
  and its once-only keys live in `<work_dir>/monitor_alerted.txt`.
- **F11. The chain signals a run in ONE case: the arm's registered `stop`** (`on_void` / `on_throughput_fail`),
  SIGTERM only, through `signal_if` on this run's launcher identity. Only the fakes have exercised it (F4); the
  first real `stop` is watched by hand. `page` behaves as the old `d3_redirect.sh` did. `launch_x35.sh`'s own
  pool-seed SIGTERM is unchanged.

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
   - a `throughput` bar (C0's 1,384-1,424 steps/s means about 1,000 as a floor), counted from the first trainer line;
   - `on_void` and `on_throughput_fail`, each `stop` or `page`: the orchestrator's call per arm, written down;
   - `disk_need_gib`, taken from the last dry run's REQUIRED GiB.

   For each window: its steps, `on_fail` and its preempt marker.
4. **`python -m main.ops.arm_chain plan <file> --check-pins --describe` must print `PLAN OK`.**
5. **Retire the old chain without signalling it mid-step.** Let the last old arm-end finish with next `none -`, or
   do not arm one. Confirm that no old chain script is alive.
6. **Start the canary logger.** Run `python -m main.ops.arm_chain canary --out-dir ~/.claude/jobs/x35/canary
   --status-file ~/.claude/jobs/x35/status.txt` under `nohup setsid` (its dumps go to `~/gen3ai_archive/`). After its first pass writes the heartbeat,
   stop `canary_log_v12.sh` by PID through the identity gate (`signal-if <pid> 'canary_log_v12'`).
7. **Start the supervisor** from the MAIN checkout at nice 0 (F3), with the env's interpreter and
   `PYTHONPATH=$PWD/src`:
   `nohup setsid python -m main.ops.arm_chain run <file> >> ~/.claude/jobs/x35/arm_chain/<name>.out 2>&1 < /dev/null &`
   - **The alert Monitor: the operator's choice.** Prefer ONE follower that never needs a re-arm, and that also pages
     a dead supervisor: `python -m main.ops.arm_chain monitor --chain <file>` (bounded at 24 h; re-arm it at its
     `EXPIRING` line). The fallback is the `tail -F` pipeline, which must be RE-ARMED at every `MONITOR-REARM` line
     the filter passes, because the new run's logs did not exist when it started:
     `tail -F -n0 $(python -m main.ops.arm_chain watch-files --chain <file>) | grep --line-buffered -E
     "$(python -m main.ops.arm_chain monitor-filter --chain <name>)"`. It cannot see a dead supervisor.
   - Keep the 55-min fallback cron. It reads `status` (a `!!!` liveness line pages). Its repair commands are
     `clear-hold --resolution …` or `abandon` on the orchestrator's word, then `run`.
8. **Watch the first GO by hand** (F4). Check:
   - the kit's `START` line and its pin;
   - the `GOING → LAUNCHED → SPAWNED` lines;
   - `ps` showing exactly one launcher.

   At the first arm end, check:
   - the scratch K9(b) record;
   - that `git status` on main is clean;
   - the combined records commit, and which way it shipped (`records ship ASYNC` or `SYNCHRONOUS (<why>)`).
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
- **2026-10-27: the operator's review (Training Oct).**
  - The canary logger alerts on every non-PASS; the family bar is back as a non-alerting tag from `canary_decide`;
    canary weight dumps are archived with a sha256; the launcher-log alerts joined the Monitor filter (CB-29-31).
  - `on_void` / `on_throughput_fail` (`stop` | `page`) are required per arm; the throughput gate counts from the
    first trainer line (CB-32, CB-33).
  - A window validates the lease TOKEN before every `gpu_lock` step (CB-34).
  - The record ship is asynchronous while the kit's `k9b-batch` writes outside main (`7f9ff6236`), inline otherwise
    (CB-35, F1, F12); main is checked clean around every dry run and before every GO.
  - The module deletes only through `fsguard.remove_under` (CB-28).
- **2026-10-26 (evening): the adversarial review's findings 2-13** ([`arm_chain_review_2026-10-26.md`](arm_chain_review_2026-10-26.md);
  finding 1 was fixed at `93a05fe91`). The changes are CB-36-46.
  - The supervisor's liveness became readable: a heartbeat plus a lock probe, and `status` prints ALIVE / STALE / DEAD.
  - `clear-hold` takes a resolution: `retry`, `relaunch`, or `accept`.
  - The `monitor` follower replaces the re-armed `tail -F`.
  - The state schema stays v1: a supervisor restarted on this code resumes the `93a05fe91` supervisor's state as it
    is, and the reverse also holds.
