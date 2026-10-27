# Adversarial review of `main.ops.arm_chain` before the 10-26 cut-over

**What was reviewed:** `src/main/ops/arm_chain/` at d4387711 + 92f0f45a (unchanged at 71c76dfbb) and `designs/ops/arm_chain.md`.

**The case it was reviewed for:** the chain file `~/.claude/jobs/x35/chains/x35-lrf-oct26.json`, which runs D3 s3504 and s3505 at pin 9e289ba4, then LR-F s3502 and s3503 at pin f4b25238. The supervisor is started by `x26r/x59s_window_v2.sh`'s `cutover()`, which `exec`s it from main at nice 0 after the X59-S window.

**Severity scale:**
- **BLOCKER:** fix before the cut-over.
- **MAJOR:** fix soon; the run is not wrong, but the GPU can idle unpaged or a held item cannot be recovered.
- **MINOR:** a hardening item.

## What was traced, and what held

Everything below was run read-only and CPU-only, at nice 19 on the agent cores.

**`plan --check-pins --describe` on the real chain file prints `PLAN OK`.** The kit agrees on every arm: each is a FORK of `rb_x26_s2601` at checkpoint 280,000,170, adding 20,000,000 steps, with `--steps 300,000,170`.

**`kit stage-check` reads ALLOWED for all four arms:**
- d3 3504 and 3505: "look 1 is CONTINUE";
- lrfloor 3502 and 3503: "B2-LRF, no unmet condition".

**The checks after the GO match the kit's real output.** This was checked against the D3 s3501-3503 and N1-P s3502-3503 artifacts.
- **The dry run.** The chain's dry-run regexes (`role : FORK of`, `pin : <40 hex>`, `cuda devices : ✓`, `this command would launch`, the step delta) match `rb_x35_d3mut_s3503.dry.dryrun.txt`. The kit writes its describe and transcript to the same fixed paths the chain reads (`describe.<arm>.<seed>.json`, `<run>.dry.dryrun.txt`).
- **The banner.** The chain file's D3 `expect` string is byte-identical to the banner in `rb_x35_d3mut_s350{1,2,3}.launcher.log`. That banner appears about 23 s after START, so it is in the log before SPAWNED, which comes about 3 min after START. Neither VOID string occurs in those logs.
  - The only emitter of `does NOT inherit` is `matchup_setup.py:185`, and it fires only when a fork drops its parent's `--team-mix`. `rb_x26_s2601` has none.
- **The throughput gate.** A fork's step counter continues from the parent, so its first own checkpoint is `checkpoint_281000337_steps.zip`, which is above `fork_step`. The gate therefore sees the run's own checkpoints.
  - In the reference runs the first two own checkpoints land 17 and 28 min after the child attaches. Their marginal rate is about 1,450 steps/s against a bar of 1,000. The gate passes at about 30 min, well inside 60.
  - The "first trainer line" is `[Belief] resume: …`, seconds after `child attached`. The 60-min window therefore effectively starts at the spawn. That is harmless at these timings.
- **The stop path.** The pid file `<job>/<run>.pid` names `$!` of a subshell that `exec`s `taskset`, which `exec`s python. The pid is therefore the launcher's, and its cmdline ends in `--run-name <run>`, so `launcher_ere` is anchored correctly. `signal_if` gates on liveness, the cmdline and an exact `( |$)` terminator. I found no wrong-process path.
- **The thin.** The planner prints each kept file as `^ +checkpoint_N_steps.zip `, and its open-dir refusal reads `THIN REFUSED: …have the run dir open`. Both match the chain's regexes.

**Async ship is ON tonight.** `ship.kit_k9b_out_of_main` returns True: one `k9b-batch` call passes `--out-dir`, and the probe is refused with `BatchError`. `.claude/worktrees/` is git-ignored, so a ship worktree is never read as dirt in main. A K9(b) read takes about 1 s and a ship about 35 s (old chain, 10-26). The next arm's dry run is about 15 s.

**Never a second GO.** Traced:
- `GOING` is persisted before the spawn.
- A restart while in GOING adopts an existing GO by its wrapper argv, its START line or its run dir; with no evidence it HOLDs.
- The kit refuses an existing `models/<run>`, and `kit live` refuses while any launcher is live.

**State writes are atomic.** Each is a tmp file plus `os.replace`. A restart at every arm-end phase re-runs that phase idempotently. A worker spawned twice ships once (`_already_shipped`).

**The GO environment.** The allow-list drops nothing the launch needs. I compared it with the environment of the process whose image the supervisor will be: no torch or CUDA knob is set there today. The PENDING dry run runs under the same environment, so a gap would show as a HOLD before any GO.

**Not verified (FINDINGS):**
- I did not run the kit's dry run or GO under the chain. The launcher's dry run checks GPU visibility (rule 5), and the kit writes to the job dir and `status.txt` (out of scope).
- The first real `stop` is still unexercised (F11).

## Findings

### 1. BLOCKER (FIXED at 93a05fe91): a multi-line HOLD reason kills the supervisor and pages nobody

**Status:** `StatusLog.append` now folds CR/LF into ` | ` at the single sink, which covers every caller below. I
re-ran the repro on 93a05fe91: exit 3, the hold recorded, and its HOLD line in `status.txt`. The outer `except
BaseException` in the CLI (the last bullet of the fix) is still open; it belongs with finding 2.

**Where:**
- `chain.py:281`: the Hold carries `dirt[:200]`, the raw multi-line `git status --porcelain`.
- `chain.py:168`: `_hold` then calls `StatusLog.append`.
- `statuslog.py:141`: `append` raises `StatusError` on any `\n`.

`_hold` runs inside `_run_locked`'s `except Hold` handler, outside the try. The `StatusError` therefore escapes `run()`, and the CLI catches only `SpecError` / `ChainError`.

**Failure scenario:**
1. Main has two or more dirty paths at a dry run or GO boundary. Two stray files are enough, which is exactly what the guard exists for.
2. The hold is saved in the state.
3. No `HOLD:` line reaches `status.txt`; the traceback goes only to `arm_chain/<name>.out`.
4. The supervisor is dead, the Monitor never fires, and the GPU idles.
5. A restarted `run` prints `… is HELD at …`, which the Monitor filter does not match either.

**Reproduced:** with the module's own fakes, using `make_world` and the `chain_test.deps` stand-ins in a temp dir, two untracked files in the fake main give:
- `SUPERVISOR DIED: StatusError a status line may not contain a newline`;
- the hold recorded in the state;
- no HOLD line in `status.txt`.

**Same shape elsewhere:**
- `armend.py:123`, `armend.py:136`: the K9(b) stage dirt.
- `ship.py:173`, `ship.py:204`: inside the async worker. `ship_worker` writes its result, then dies on its own `SHIP-HOLD` line. The supervisor writes a `SHIP-HOLD` only for a *synthetic* outcome, so this ship failure is silent until the chain's end.

**Fix:**
- Sanitise the message at the single choke point. Either `StatusLog.append` replaces `\r\n`, `\n` and `\r` with ` | ` instead of raising, or `Chain._hold` and `ship_worker` pass `reason.replace("\n", " | ")`.
- Add a regression test: two stray files in main lead to exit 3 and exactly one `HOLD:` line.
- Also wrap the CLI `run` in an outer `except BaseException` that appends one single-line `HOLD: supervisor died: <type>`; see 2.

### 2. MAJOR: a dead supervisor is invisible

**Where:** `chain.py:217-260` and `chain.py:583`. Nothing reports whether a supervisor is alive:
- `status` shows phases only;
- there is no heartbeat;
- an exception outside the per-item try is not turned into a HOLD line. `_reconcile_ships()` at `chain.py:240`, and `_hold`/`_save` themselves on a full disk, raise outside it.

**Failure scenario:** an OOM kill, a session teardown, finding 1, or a disk-full `_save` during RUNNING. The current run trains to its END, but nobody hands off. `status` still reads `RUNNING`, and the GPU idles until a human looks.

**Fix:**
- `status` probes `<state>.lock` with a non-blocking flock and prints `supervisor: ALIVE pid N` or `NOT RUNNING`. It also prints a loud line when NOT RUNNING while an item is mid-flight (not terminal, not held).
- The supervisor touches a heartbeat file every poll.
- The training session's 55-min cron checks both.

### 3. MAJOR: `clear-hold` cannot recover an arm held at LAUNCHED

**Where:** `chain.py:179` resets only GOING and STARTED. Re-running `_arm_launched` (`chain.py:352`) re-reads the same persisted evidence through `postspawn.wait_spawned` (`postspawn.py:53`): the wrapper's old `HOLD:` line past the offset, or a wrapper that is already dead.

**Failure scenario:**
1. The kit's GO-time pre-checks HOLD transiently (`kit live`, the lease or pool status), or the pool-seed check never passes.
2. The orchestrator says retry, and the session runs `clear-hold` then `run`.
3. It HOLDs again at once, with the same reason, forever.
4. `abandon` makes the arm `DONE_FAILED`. The same run cannot be re-added, because item keys are `arm:<run>` and duplicates are refused. The only way out is to hand-edit the state or write a new chain file.

**Reproduced:** with the fakes and `FAKE_NO_PASS=1`, run 1 HOLDs at LAUNCHED; `clear-hold`, then run 2, HOLDs at LAUNCHED with the same reason; there is one GO call.

**Fix:** in `clear_hold`, reset LAUNCHED to PENDING when all of these hold:
- the wrapper is not alive;
- the launcher pid file names no live `main.launcher`;
- `models/<run>` does not exist.

The last condition means the kit never launched. Also re-arm the triggers at a fresh offset on the next GO, which PRECHECKED already does. Otherwise keep the HOLD and say why (a run dir exists: a human decides).

### 4. MAJOR: the Monitor follows only the logs that existed when it was armed

**Where:**
- `canary.py:57`: the filter passes only `HOLD`, `SHIP-HOLD` and `chain COMPLETE`.
- `canary.py:216`: `watch_files` lists the live launchers at the moment it is called.
- `arm_chain.md` §7 says "re-armed at each GO", but no filtered line is written at a GO. Nothing wakes the training session to re-arm.

**Failure scenario:**
1. The Monitor is armed at the cut-over, before s3504's GO, so it follows `status.txt` only.
2. The `Traceback` / `Child crashed` / `WEDGE` / `UpdateWontFit` / `BEHAVIOUR-POLICY MISMATCH` alerts of all four runs are never seen.
3. A wedge after the first hour is caught only by the launcher's own watchdog, or by the canary logger's 3.5 h staleness alert.

**Fix:** the launch paths are deterministic per item: `<job>/<run>.launcher.log` and `models/<run>/launcher_child.full.log`. Add `watch-files --chain <file>`, which prints them for every item up front; `tail -F` waits for a file to appear. Alternatively, add `ARMCHAIN\[<name>\]: arm:\S+: LAUNCHED -> SPAWNED` to the filter as an explicit re-arm cue.

### 5. MINOR: a timed-out step leaves orphans

**Where:** `runner.py:77-84`. `subprocess.run(..., timeout=)` SIGKILLs only the direct child (bash). The kit's python children, including a launcher `--dry-run` that touched CUDA, survive.

**Consequence:** an orphan holding a CUDA context then fails the next GO's GPU-idle check, which shows up as a HOLD.

**Fix:** spawn with `start_new_session=True` and `os.killpg` the group on timeout.

### 6. MINOR: the post-spawn gates do not watch for END

**Where:** `postspawn.py:78` (banner) and `postspawn.py:160` (throughput).

**Failure scenario:** a run that dies in its first hour is noticed only at the throughput deadline: 60 min after the first trainer line, or 90 min after the GO. Then `stop` finds no launcher and HOLDs.

**Fix:** pass the END trigger and fail fast when it matches.

### 7. MINOR: a double GO by another launcher is not refused

**Where:** `chain.py:298`. The GO never checks for a live `launch_x35.sh <arm> <seed> … GO` (`find_go_wrapper`) or for a live launcher. Two GO wrappers whose pre-checks overlap within seconds could both pass `kit live`.

**Tonight:** `x59s_window_v2.sh` refuses to start the supervisor while an old `arm_end`/`poll_launch` is alive, so tonight is covered.

**Fix:** refuse the GO if `find_go_wrapper` or `live.discover` returns anything.

### 8. MINOR: a crash between the SIGTERM and `_note` sends a second SIGTERM

**Where:** `chain.py:376`. A restart re-runs the banner or throughput gate, reads VOID again and SIGTERMs again. A second SIGTERM may escalate the launcher's shutdown, so the interrupted save could be skipped.

**Fix:** persist `stopping` before the kill, and skip the kill when it is set.

### 9. MINOR: a `page` action has no path forward

`clear-hold` re-runs the gate, which fails again. `abandon` is refused while the launcher lives. A paged arm the orchestrator accepts gets no K9(b), no thin and no records.

None of tonight's arms uses `page`.

**Fix:** add `clear-hold --accept`, which advances one phase.

### 10. MINOR: the CLI writes the state without the supervisor's lock

**Where:** `clear-hold` and `abandon` call `_save` without the flock, through the fixed `state.json.tmp`. That is harmless today, because a HOLD always means the supervisor has exited.

**Also:** `_save` does not `fsync`.

### 11. MINOR: a chain-file edit to a started item kills `ship-records` silently

**Where:** `cli.py:137`. `ship-records` builds a `Chain`, whose state load raises `ChainError` when a started item's entry was edited. The worker then exits with `REFUSED` and writes no result. The supervisor reports only a synthetic "died without a result".

### 12. MINOR: the ship can fail after the push, leaving its worktree behind

**Where:** `ship.py:192`. The `merge --ff-only` into main after the push sits outside the try. A transient `index.lock`, for example from a concurrent `git status`, fails the ship after the push and leaves the worktree and branch behind. A retry succeeds through `_already_shipped`.

### 13. MINOR: the new dose records lose `c0_comparator`

**Where:** `chain.py:460`. The chain file carries no `x_dose_comparator`, so tonight's dose records omit the `c0_comparator` block that the D3 s3501-3503 records carry. Nothing in `src/` reads it today.

## Verdict

**Findings 2-13 are fixed** (the commit that adds `src/main/ops/arm_chain/liveness.py`; `arm_chain.md` §5 CB-36-46,
each with a test that fails on revert). The fixed code is picked up only when the supervisor restarts. It resumes the
`93a05fe91` supervisor's state as it is (§5, "restart").

**Finding 1 is fixed at 93a05fe91, so it no longer holds the cut-over.** The `cutover()` fast-forwards main before it
`exec`s the supervisor, so tonight's supervisor runs that fix.

Findings 2-4 are the GPU-idles-unpaged and unrecoverable-HOLD classes. Fix them before the chain runs unattended for a second night.

Every path that launches, signals, deletes or writes into main that I traced is correct for this chain file.
