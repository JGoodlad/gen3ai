# The GPU window: pause a live run for a guest GPU job (`--gpu-window`, T46)

> Always-current (`README.md`). Code: `src/main/launcher/gpu_window/` (`protocol.py`, `controller.py`, `cli.py`),
> `src/utils/procgroup.py` (the guest's session) and its two hooks in `run._supervise`. Tests:
> `gpu_window/window_cycle_test.py` (a real supervisor over a tiny CPU child, `integration`),
> `gpu_window/guest_group_test.py` (real guest processes: session, claim record, `preempt`; `integration`) and
> `gpu_window/protocol_cli_test.py` (unit, fake GPU probes).

## Why

GPU work between a run's safe points (a battery, an A/B, a K9(b) read, a canary diagnostic) used to need a
hand-written chain script: stop the launcher, wait for the GPU, run the job, relaunch. The X34 end chain took
four rewrites plus two continuations (2026-10-19), and each version hit a parser or state bug. The worst was a
whole-file grep that matched a second table and held the chain mid-window. Stopping the launcher also dropped
any lease whose `--watch-pid` was the launcher. The window is now a launcher verb, and the launcher stays alive
throughout.

## The mechanism

```
arm (preconditions) ─▶ request.json ─▶ launcher, at the trigger epoch: SIGTERM ─▶ child saves at its
safe point, exits INTERRUPTED ─▶ checkpoint + empty GPU checked ─▶ GO ─▶ guest claims, runs ─▶ DONE rc 0
─▶ the ordinary same-run RESTART (same pin, same command, RESUME role, --model = the pause's checkpoint)
```

- **Off unless asked for.** Only a launch with `--gpu-window` reads requests: it writes
  `<run_dir>/gpu_window/launcher.json`, naming itself by pid and start ticks. A running launcher started
  without the flag can never be paused this way. A live run pinned to older code gets the window only by a
  deliberate re-launch, because the window is launcher-side and works with any pinned trainer: it uses only
  SIGTERM and the deferred-abort save.
- **Trigger = an absolute epoch, never a message.** `arm --at-epoch E` (or `--in-minutes M`, made absolute at
  arming) is refused if it is more than 60 s in the past or more than `--horizon-hours` ahead (default 48).
  The launcher re-checks the bounds against the request's own arming time and horizon (at most 168 h).
- **Never during the child's startup.** A due request waits until the CURRENT child has pushed its first
  rollout metrics (`LauncherState.child_has_reported`). The trainer installs its stop handlers only after
  load, compile and the dry update (`main.train.model_build`, minutes); a SIGTERM before them kills it unsaved
  (rc -15 → HOLD; on a fresh run, nothing to resume). So a window due just after a restart starts late.
- **A request is honoured only by the launcher it names.** A request whose launcher identity is not this
  launcher's is moved to `refused_<id>.json` with an event and never acted on. Typical sources: a leftover
  from a dead session, or a hand-written file. The launcher CONSUMES a due request by an atomic rename into
  `w_<id>/`, so a racing `cancel` either wins whole or loses whole.
- **The pause is the existing stop.** The launcher SIGTERMs the child as for a quit. The child's deferred
  abort saves `final_model_interrupted.zip` at its next safe point, and `KILL_GRACE_SECONDS` (150 s) still
  escalates to SIGKILL. The resume is the existing restart path, so nothing about the pin, the recipe or the
  argv is re-derived.
- **GO only on a clean pause.** All three must hold: the child exited `INTERRUPTED` (15); a checkpoint at least
  as new as the SIGTERM exists (2 s mtime slack; the trainer's save is atomic and moves `latest.txt` last, so a save that
  failed leaves the PREVIOUS file with its old mtime and the window HOLDS, never a torn zip with a fresh one —
  `training/checkpoint_writes.md`); and `nvidia-smi` lists no compute process within 600 s. The
  GPU check covers the whole box, because under the lease only its holder uses the GPU. Only an ABSENT
  `nvidia-smi` (a CPU box) skips it, with a warning; one that fails or hangs is not a free GPU (it HOLDS).
- **The run comes back only to an empty GPU.** After DONE rc 0 the same check runs again, so a worker the
  guest spawned and left behind holds the window instead of OOMing the trainer's first update. An operator's
  `resume` is checked the same way (see HOLD).
- **One claimant.** `w_<id>/claim.json` is created `O_EXCL`, by the guest (which then runs) or by the launcher
  when no guest claimed within `--guest-start-timeout-s` (default 180). A launcher claim makes the window
  MISSED, and the run resumes, because nothing ran on the GPU. A late guest then finds the claim taken and
  never runs its command.
- **The guest's job is a SESSION, not a pid.** The guest runs its command with `setsid` (`utils.procgroup`), so the
  command leads a session and a process group of its own (`pgid == sid ==` its pid), and records them in the claim
  (`job: {pgid, sid, leader: {pid, start_ticks}}`, atomically REPLACED right after the start; the exclusive create
  already decided who runs). The claim still names the guest wrapper (`guest`), but the launcher judges the job by
  the session: every HOLD that follows a dead wrapper or an overrun names the job's live processes (pids, `STILL
  RUNS`) or says it has ended, so an operator sees whether the card is still held. A stop signal sent to the guest
  (SIGINT / SIGTERM / SIGHUP) is forwarded to every group of the job's session and the guest still writes DONE (a
  signal death is `128+N`, so a SIGTERM reads `rc 143` and HOLDs); SIGKILL of the guest cannot be forwarded, and
  leaves the job running (the HOLD then names it). The command has no controlling terminal.
- **The lease never moves.** `arm` and `guest` require the caller to HOLD the GPU lease: its token must be in
  the environment. `--no-lease-check REASON` is for a box with NO lease (a CPU run, a test) and is refused
  while anyone holds one. The guest's GPU commands go through
  `scripts/ops/gpu_lock.sh` on the holder's token. The launcher stays alive throughout, so a lease that
  watches it is kept.

## HOLD: every step that cannot finish cleanly

The run stays PAUSED (no child, launcher alive, lease kept) whenever one of these happens:
- the child exits other than `INTERRUPTED`;
- the pause saves no checkpoint;
- the GPU is not released within 600 s, or `nvidia-smi` keeps failing (before GO, and again after DONE rc 0);
- the guest exits non-zero;
- the guest dies without writing DONE (its pid and start ticks are in its claim); the reason names what is left
  of its job (the session's live pids, or "no live process");
- the guest is still running `--max-window-min` after GO (default 60). It may still hold the GPU; the reason
  names the job's live pids.

The reason goes to `current.json`, to the events (`🛑 GPU WINDOW HOLD …`) and to the red status badge.
Only an operator ends a HOLD: `python -m main.launcher.gpu_window resume <run_dir> --reason "…"`, or a quit.
The launcher never auto-resumes over a failure (`TRAINING_RUN_SOP.md` §3 rule 3). A window left OPEN by a launcher
that died is closed as `abandoned` by the next `--gpu-window` launcher of the run, so it cannot block `arm` forever.

**A `resume` is an intent, not a go.** The launcher re-checks the GPU first, with the check the window made at
its entry: it re-launches the run only when `nvidia-smi` lists no compute process (or is absent: warned). While
a pid is listed, or `nvidia-smi` fails, the HOLD stays, its reason reads `… | resume requested, waiting up to 600 s:
the GPU still lists compute pid(s) […]` (also an event), and the launcher waits up to `GPU_RELEASE_TIMEOUT_S`
(600 s) for the card to empty. Past that the resume is REFUSED: `RESUME.json` is set aside as
`RESUME.refused_<epoch>.json`, the HOLD stays (`resume REFUSED: …`), and the operator frees the card (the usual
tool: `preempt`) and resumes again. A re-launch onto a busy card OOMs at the trainer's first update and burns the
3 rapid-crash restarts.

**`preempt` stops the guest's job.** `python -m main.launcher.gpu_window preempt <run_dir> [--kill]` (window in
CLAIMED or HOLD, a job on record) sends SIGTERM (`--kill`: SIGKILL) to every process group of the job's recorded
SESSION, re-derived from the live process table: the command's group and any group inside it (a `gpu_lock.sh`
makes one), not the guest wrapper, not a pid, nothing outside the session. It refuses when the record names the
caller's or the guest's own group or session. The guest then writes DONE with `128+N` and the window HOLDs; a
SIGTERM-ignoring job needs `--kill`.

## Files (`<run_dir>/gpu_window/`) and phases

| file | written by | meaning |
|---|---|---|
| `launcher.json` | launcher (start) | this launcher (pid + start ticks) honours windows |
| `request.json` | `arm` | the armed window: id, launcher, trigger epoch, bounds, precheck record, lease line |
| `current.json` | launcher | the window in flight: `phase`, `reason`, `since_epoch` (the status line) |
| `w_<id>/request.json` · `GO.json` · `claim.json` · `DONE.json` · `RESUME.json` | launcher · launcher · guest or launcher · guest · `resume` | the window's record, every file carrying the window id. `claim.json` carries the guest's `job` (pgid, sid, leader identity); `DONE.json` its `pgid`; a refused resume is kept as `RESUME.refused_<epoch>.json` |

Phases: `pausing` → `go` → `claimed` → `done` → `resumed`. The other endings are `missed` → `resumed`,
`hold` → `resumed`, `abandoned` (a quit, or the run completed first) and `refused`. The phases `pausing`, `go`,
`claimed` and `hold` are OPEN, and `arm` refuses while one is open.

## The CLI (`python -m main.launcher.gpu_window`)

```bash
# the lease holder, while the run is live — every precondition is checked BEFORE request.json exists:
# a live --gpu-window launcher for the run · no open window · epoch and bounds · the lease · the precheck
GEN3AI_GPU_LEASE_TOKEN_FILE=<holder token file> python -m main.launcher.gpu_window arm models/<run> \
    --precheck "<the guest's CPU dry-check, e.g. run_battery.sh --dry-run>" --at-epoch "$(date -d '…' +%s)" \
    --max-window-min 130 --note "X35 battery"
# the guest, started any time after arm, BOUNDED; runs once on GO, writes DONE with the command's rc
GEN3AI_GPU_LEASE_TOKEN_FILE=<holder token file> nohup python -m main.launcher.gpu_window guest models/<run> \
    --wait-max-min 240 -- scripts/ops/gpu_lock.sh <battery …> > guest.log 2>&1 < /dev/null &
python -m main.launcher.gpu_window status models/<run>
python -m main.launcher.gpu_window resume models/<run> --reason "…"   # ends a HOLD, once the GPU is empty
python -m main.launcher.gpu_window preempt models/<run> [--kill]      # stops the guest's job (its session), nothing else
python -m main.launcher.gpu_window cancel models/<run>                # withdraws an armed, untriggered request
```

`--precheck` runs `bash -c` with `CUDA_VISIBLE_DEVICES=""` under `--precheck-timeout-s` (default 900). A
non-zero exit or a timeout refuses the arm and writes nothing. `--no-precheck REASON` records why there is no
precheck. The guest is ONE command: a multi-step guest is a script that exits non-zero on its first failed
step, which makes the window HOLD. Exit codes: 0 ok (for `guest`, its command's rc); 2 usage; 3 refused
(nothing written); 4 `guest` did not run (MISSED, abandoned, or held before GO); 5 `guest` timed out waiting
for GO.

## What it does not do

- It does not stop a run for good, and it does not chain across runs (a stop at a plateau, then the next arm).
  Those stay operator scripts.
- It does not run the guest itself.
- It does not watch the guest's GPU use beyond its job's processes and the time bound, and it never kills the job on
  its own (`preempt` is an operator's act).

## Hazards

- **A stop with no safe point turns a pause into a HOLD.** If the child reaches no safe point within 120 s
  (TASK_BACKLOG T45: a stop deferred across an eval cycle), the trainer's own watchdog exits 15 WITHOUT a save
  (`deferred_abort.SAFE_POINT_DEADLINE_SEC`; `latest.txt` stays on the last periodic checkpoint). The window
  HOLDS on "no checkpoint saved by the pause"; at 150 s the launcher's SIGKILL (rc -9) also HOLDS. Until T45
  lands, trigger a window away from an eval cycle.
- **A HOLD waits indefinitely.** It is a decision for the orchestrator, not a timeout. Watch for the HOLD line
  with the run's alert watcher.
- **The layer-1 watcher knows windows.** `scripts/ops/watch_run.sh` does not call an open window a wedge, and
  it writes one `ALERT: GPU WINDOW HOLD` line per held window (`watch_run_gpu_window_test.py`).
- **A guest that `setsid`s a worker out of the job's session is outside `preempt`** (and outside the HOLD's list of
  live pids); the empty-GPU re-check on `resume` is what still catches it.
- **SIGKILL of the guest wrapper leaves its job running** in its own session (nothing can forward a SIGKILL): the
  HOLD says so, `preempt` stops it. This is why a preempt names the job, never the wrapper's pid.
- **Quitting the launcher during a window ends its lease watch.** A lease whose `--watch-pid` is the launcher
  ends when the launcher exits, even if a guest is still running.

## The first real trial (not yet run; every test so far is a fake CPU child)

The cheapest real-trainer check, for the training session in a GPU gap, ~25 min of GPU:
1. **Preconditions:** no live run on the GPU; `scripts/ops/gpu_lease.sh status` FREE; the box quiet enough
   for a `--arch production` startup. Acquire: `scripts/ops/gpu_lease.sh acquire --owner t46-trial
   --token-file /tmp/t46.tok --max-hours 2`; every command below carries `GEN3AI_GPU_LEASE_TOKEN_FILE=/tmp/t46.tok`.
2. **Launch** a throwaway run with the window on, the real path (not `--debug`), into a scratch archive:
   `GEN3AI_MODELS_DIR=<scratch> scripts/ops/gpu_lock.sh "$PY" -m main.launcher --steps 2000000 --device cuda
   --arch production --restart-interval-hours 0 --gpu-window --run-name t46_trial` (detached: headless). Check
   it with `--dry-run` first.
3. **Arm before the first rollout** (it must WAIT for the first report, not kill the startup):
   `"$PY" -m main.launcher.gpu_window arm <run> --no-precheck "trial" --in-minutes 0 --guest-start-timeout-s 600`,
   then `"$PY" -m main.launcher.gpu_window guest <run> --wait-max-min 30 -- scripts/ops/gpu_lock.sh nvidia-smi`.
4. **PASS** = the child log's `[ABORT] … Checkpoint saved → …final_model_interrupted.zip`, the launcher's
   `GO`, the guest's DONE rc 0 with the trainer absent from its `nvidia-smi` listing, the re-launch with
   `--model …/final_model_interrupted.zip` and the SAME `--run-dir`, and that child's first rollout at a step
   ≥ the pause's. Then a second window whose guest is `false`: it must HOLD (no re-launch) until `resume`.
5. **Teardown:** quit the launcher by its PID (SIGTERM; it saves), `gpu_lease.sh release`, delete the scratch
   archive. Any other outcome is a FINDING against this doc before the first use on a live run.
