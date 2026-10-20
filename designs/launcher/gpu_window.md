# The GPU window: pause a live run for a guest GPU job (`--gpu-window`, T46)

> Always-current (`README.md`). Code: `src/main/launcher/gpu_window/` (`protocol.py`, `controller.py`, `cli.py`)
> and its two hooks in `run._supervise`. Tests: `gpu_window/window_cycle_test.py` (a real supervisor over a
> tiny CPU child, `integration`) and `gpu_window/protocol_cli_test.py` (unit).

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
- **A request is honoured only by the launcher it names.** A request whose launcher identity is not this
  launcher's is moved to `refused_<id>.json` with an event and never acted on. Typical sources: a leftover
  from a dead session, or a hand-written file. The launcher CONSUMES a due request by an atomic rename into
  `w_<id>/`, so a racing `cancel` either wins whole or loses whole.
- **The pause is the existing stop.** The launcher SIGTERMs the child as for a quit. The child's deferred
  abort saves `final_model_interrupted.zip` at its next safe point, and `KILL_GRACE_SECONDS` (150 s) still
  escalates to SIGKILL. The resume is the existing restart path, so nothing about the pin, the recipe or the
  argv is re-derived.
- **GO only on a clean pause.** All three must hold: the child exited `INTERRUPTED` (15); a checkpoint at least
  as new as the SIGTERM exists (2 s mtime slack); and `nvidia-smi` lists no compute process within 600 s. The
  GPU check covers the whole box, because under the lease only its holder uses the GPU. If `nvidia-smi` is
  absent, the launcher prints a warning and skips the check.
- **One claimant.** `w_<id>/claim.json` is created `O_EXCL`, by the guest (which then runs) or by the launcher
  when no guest claimed within `--guest-start-timeout-s` (default 180). A launcher claim makes the window
  MISSED, and the run resumes, because nothing ran on the GPU. A late guest then finds the claim taken and
  never runs its command.
- **The lease never moves.** `arm` and `guest` require the caller to HOLD the GPU lease: its token must be in
  the environment, or the caller passes `--no-lease-check REASON`. The guest's GPU commands go through
  `scripts/ops/gpu_lock.sh` on the holder's token. The launcher stays alive throughout, so a lease that
  watches it is kept.

## HOLD: every step that cannot finish cleanly

The run stays PAUSED (no child, launcher alive, lease kept) whenever one of these happens:
- the child exits other than `INTERRUPTED`;
- the pause saves no checkpoint;
- the GPU is not released within 600 s;
- the guest exits non-zero;
- the guest dies without writing DONE (its pid and start ticks are in its claim);
- the guest is still running `--max-window-min` after GO (default 60). It may still hold the GPU.

The reason goes to `current.json`, to the events (`🛑 GPU WINDOW HOLD …`) and to the red status badge.
Only an operator ends a HOLD: `python -m main.launcher.gpu_window resume <run_dir> --reason "…"`, or a quit.
The launcher never auto-resumes over a failure (`TRAINING_RUN_SOP.md` §3 rule 3).

## Files (`<run_dir>/gpu_window/`) and phases

| file | written by | meaning |
|---|---|---|
| `launcher.json` | launcher (start) | this launcher (pid + start ticks) honours windows |
| `request.json` | `arm` | the armed window: id, launcher, trigger epoch, bounds, precheck record, lease line |
| `current.json` | launcher | the window in flight: `phase`, `reason`, `since_epoch` (the status line) |
| `w_<id>/request.json` · `GO.json` · `claim.json` · `DONE.json` · `RESUME.json` | launcher · launcher · guest or launcher · guest · `resume` | the window's record, every file carrying the window id |

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
python -m main.launcher.gpu_window resume models/<run> --reason "…"   # ends a HOLD
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
- It does not watch the guest's GPU use beyond the pid and the time bound.

## Hazards

- **An unfixed T45 overrun turns a pause into a HOLD.** If the child does not reach a safe point within 150 s
  (TASK_BACKLOG T45: a stop deferred across an eval cycle), the existing SIGKILL escalation kills it. The child
  exits `-9` with no fresh save, so the window HOLDS rather than resuming. Until T45 lands, trigger a window
  away from an eval cycle.
- **A HOLD waits indefinitely.** It is a decision for the orchestrator, not a timeout. Watch for the HOLD line
  with the run's alert watcher.
- **Quitting the launcher during a window ends its lease watch.** A lease whose `--watch-pid` is the launcher
  ends when the launcher exits, even if a guest is still running.
