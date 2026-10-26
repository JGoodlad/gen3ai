"""The arm end: is it FINISHED, does K9(b) PASS, what may be thinned, and the dose record — in that order.

Every step is a HOLD on failure, and nothing after a HOLD runs (no thin, no record, no hand-off). The class bugs:

* **A stopped run treated as finished** (arm_end v11: ``rb_x35_d1wide_s3501`` ended ``launcher_rc=0`` after a
  SIGTERM): :func:`completeness` needs rc 0 AND the launcher's ``Last step`` >= the run's own ``--steps`` (from the
  describe) AND ``final_model.zip`` AND a newest checkpoint within one rollout of ``--steps``.
* **The K9(b) record written into MAIN's tree** (arm_end v13): :func:`k9b_scratch` reads into a SCRATCH copy of the
  records stage under the chain's work dir, and verifies main's stage is byte-for-byte untouched before and after.
* **The thin deleted the end-step checkpoint** (v2), **thinned arms whose reads need every checkpoint** (v4 / v7 /
  v10: a run-name ``case`` each time), and **HELD on the run's own exiting processes** (v9): :func:`thin` takes the
  policy from the spec, retries ONLY the planner's "run dir open" refusal (bounded), and verifies the plan KEEPS the
  newest checkpoint and that ``final_model.zip`` survives the apply.
* **The dose record read the wrong describe** (``grep -l run_name | head -1``): :func:`dose_record` reads the
  deterministic describe path and refuses a field it cannot source.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from main.ops.arm_chain.fsguard import remove_under
from main.ops.arm_chain.launch import Describe, Hold
from main.ops.arm_chain.procwatch import ProcRef, alive, ref_from_pidfile
from main.ops.arm_chain.runner import Runner, agent_prefix, cpu_env, tail
from main.ops.arm_chain.spec import ArmItem, ChainSpec, WindowItem
from utils import procfs

from main.ops.arm_chain.statuslog import StatusLog, Trigger, launch_tag, line_pattern, start_body

#: One rollout (256 envs x 384 steps): a finished run's newest periodic checkpoint is within this of ``--steps``.
ONE_ROLLOUT = 98_304
_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip")
THIN_PLANNER = "designs/research_state/measurements/archive_grooming_dryrun.py"


def wait_end(log: StatusLog, end: Trigger, wrapper: ProcRef, launcher_pidfile: str, timeout_s: float,
             poll_s: float = 30.0, grace_s: float = 20.0, proc: str = "/proc",
             sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic) -> int:
    """The launcher rc from the wrapper's ``END <run> launcher_rc=N`` line.

    The wrapper writes END only while its own shell lives (``wait $LP``). If the wrapper is gone but the LAUNCHER its
    pid file names is still alive (identity: a ``main.launcher`` cmdline), the run is still going: keep waiting on
    the launcher. Only when both are gone without an END line is it a HOLD (the rc is unreadable)."""
    t0 = monotonic()
    while True:
        m = log.find(end)
        if m:
            return int(m.group(1))
        if not alive(wrapper, proc):
            lref = ref_from_pidfile(launcher_pidfile, proc)
            if lref is None or "main.launcher" not in procfs.cmdline(lref.pid, proc):
                sleep(grace_s)
                m = log.find(end)
                if m:
                    return int(m.group(1))
                raise Hold(f"the launch wrapper (pid {wrapper.pid}) and its launcher are gone without an END line "
                           "(the launcher rc is unreadable): nothing K9(b)'d / thinned, next NOT launched")
        if monotonic() - t0 > timeout_s:
            raise Hold(f"no END line within {timeout_s / 3600:.1f} h (the run may still be live: nothing done)")
        sleep(poll_s)


@dataclass(frozen=True)
class LauncherSummary:
    last_step: Optional[int]
    fps: Optional[int]
    restarts: Optional[int]
    pins: List[str]


def parse_launcher_log(text: str) -> LauncherSummary:
    def last_int(rx: str) -> Optional[int]:
        v = re.findall(rx, text, re.M)
        return int(v[-1].replace(",", "")) if v else None
    return LauncherSummary(last_step=last_int(r"^\s+Last step\s*:\s*([\d,]+)"), fps=last_int(r"^\s+FPS\s*:\s*([\d,]+)"),
                           restarts=last_int(r"^\s+Restarts\s*:\s*(\d+)"),
                           pins=sorted(set(re.findall(r"--pin-commit ([0-9a-f]{40})", text))))


def checkpoint_steps(run_dir: str) -> List[int]:
    try:
        names = os.listdir(os.path.join(run_dir, "checkpoints"))
    except OSError:
        return []
    return sorted(int(m.group(1)) for n in names if (m := _CKPT.fullmatch(n)))


def completeness(run_dir: str, target_steps: int, rc: int, summary: LauncherSummary) -> str:
    if rc != 0:
        raise Hold(f"launcher_rc={rc} (not a clean end) — nothing K9(b)'d / thinned, next NOT launched")
    newest = (checkpoint_steps(run_dir) or [0])[-1]
    final = os.path.isfile(os.path.join(run_dir, "final_model.zip"))
    last = summary.last_step or 0
    if not (last >= target_steps and final and newest >= target_steps - ONE_ROLLOUT):
        raise Hold(f"stopped, not finished: last step {summary.last_step} vs its --steps {target_steps}, newest "
                   f"checkpoint {newest or 'none'}, final_model.zip {'present' if final else 'ABSENT'} — nothing "
                   "K9(b)'d / thinned, next NOT launched")
    return (f"complete: last step {last:,} >= --steps {target_steps:,}, final_model.zip present, newest checkpoint "
            f"{newest:,} >= --steps - {ONE_ROLLOUT:,}")


def _git_status(repo: str, rel: str) -> str:
    p = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=all", "--", rel],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise Hold(f"git status of {rel} in {repo} failed: {p.stderr.strip()[:160]}")
    return p.stdout


def k9b_scratch(spec: ChainSpec, it: ArmItem, runner: Runner, base_env: Mapping[str, str]) -> str:
    """AM-6 K9(b) into a scratch copy of the records stage; the record's path (the ship's SOURCE)."""
    main_stage = os.path.join(spec.repo, spec.records_stage)
    before = _git_status(spec.repo, spec.records_stage)
    if before:
        raise Hold(f"main's records stage is dirty BEFORE the K9(b) read (main is never dirty): {before.strip()[:200]}")
    scratch = os.path.join(spec.work_dir, "k9b_stage", it.run)
    os.makedirs(spec.work_dir, exist_ok=True)
    remove_under(scratch, spec.work_dir, tree=True)   # a previous read's scratch copy, strictly inside the work dir
    shutil.copytree(main_stage, scratch, symlinks=True)
    out = os.path.join(spec.work_dir, f"{it.run}.k9b.txt")
    argv = [*agent_prefix(spec.agent_cores, 0), spec.python, "-m", "main.x35.kit", "k9b",
            "--run-dir", os.path.join(spec.models_dir, it.run), "--stage", scratch]
    rc = runner.run(argv, cwd=spec.repo, env=cpu_env(base_env, spec.repo, blocking=f"arm_chain arm end ({it.run})",
                                                     models_dir=spec.models_dir),
                    timeout_s=3600, out_path=out)
    after = _git_status(spec.repo, spec.records_stage)
    if after:
        raise Hold(f"the K9(b) read changed MAIN's records stage (it must write only the scratch copy): {after.strip()[:200]}")
    rec = os.path.join(scratch, "k9b", f"{it.run}.json")
    if rc != 0:
        raise Hold(f"K9(b) not PASS (rc {rc}): {tail(out, 2, 200)} — nothing thinned, next NOT launched")
    try:
        verdict = json.loads(open(rec).read()).get("verdict")
    except (OSError, ValueError) as e:
        raise Hold(f"K9(b) rc 0 but no readable record at {rec}: {e}") from e
    if str(verdict).upper() != "PASS":
        raise Hold(f"K9(b) record verdict {verdict!r}, not PASS — nothing thinned, next NOT launched")
    return rec


def disk_ok(models_dir: str, need_gib: float, headroom_gib: float,
            usage: Callable[[str], Any] = shutil.disk_usage) -> str:
    free = usage(models_dir).free / 2 ** 30
    if free < need_gib + headroom_gib:
        raise Hold(f"disk {free:.1f} GiB free < the next arm's {need_gib:g} + {headroom_gib:g} GiB — next NOT launched")
    return f"disk {free:.1f} GiB free (need >= {need_gib + headroom_gib:g})"


def later_evidence(spec: ChainSpec, it: ArmItem) -> List[str]:
    """Checkpoint files of ``it``'s run that a LATER item of the chain names in an argv (a window's replay reads
    them): a thin that deletes one breaks that read (10-25: the segment-1 thin removed the replay's hard-coded
    u1000 neighbours)."""
    ck = os.path.join(spec.models_dir, it.run, "checkpoints") + os.sep
    idx = spec.items.index(it)
    out = []
    for later in spec.items[idx + 1:]:
        argvs = [s.argv for s in later.steps] if isinstance(later, WindowItem) else [later.verify[i] for i in range(len(later.verify))]
        for argv in argvs:
            for tok in argv:
                for part in tok.split("="):
                    if os.path.normpath(part).startswith(ck) and _CKPT.fullmatch(os.path.basename(part)):
                        out.append(os.path.basename(part))
    return sorted(set(out))


def thin(spec: ChainSpec, it: ArmItem, runner: Runner, base_env: Mapping[str, str], target_steps: int,
         retries: int = 10, retry_s: float = 30.0, sleep: Callable[[float], None] = time.sleep) -> str:
    if it.thin == "none":
        return f"NOT thinned ({it.thin_reason})"
    run_dir = os.path.join(spec.models_dir, it.run)
    final = os.path.join(run_dir, "final_model.zip")
    newest = (checkpoint_steps(run_dir) or [0])[-1]
    if newest < target_steps - 1_000_000:
        raise Hold(f"the newest checkpoint is {newest or 'none'} (< --steps {target_steps} - 1M) — NOT thinned")
    env = cpu_env(base_env, spec.repo, blocking=f"arm_chain thin ({it.run})", models_dir=spec.models_dir)
    planner = [*agent_prefix(spec.agent_cores, 0), spec.python, os.path.join(spec.repo, THIN_PLANNER),
               "--thin-run", it.run, "--models-dir", spec.models_dir]
    dry = os.path.join(spec.work_dir, f"{it.run}.thin_dry.txt")
    rc = 1
    for _ in range(retries):
        rc = runner.run(planner, cwd=spec.repo, env=env, timeout_s=1800, out_path=dry)
        if rc == 0:
            break
        if not re.search(r"THIN REFUSED: .*have the run dir open", open(dry, errors="replace").read()):
            break
        sleep(retry_s)
    if rc != 0:
        raise Hold(f"thin dry run rc={rc}: {tail(dry, 2, 200)}")
    text = open(dry, errors="replace").read()
    if not re.search(rf"^ +checkpoint_{newest}_steps\.zip ", text, re.M):
        raise Hold(f"the thin plan does NOT keep the end-step checkpoint_{newest}_steps.zip — NOT thinned")
    lost = [n for n in later_evidence(spec, it) if not re.search(rf"^ +{re.escape(n)} ", text, re.M)]
    if lost:
        raise Hold(f"the thin plan deletes {lost}, which a later chain item reads — NOT thinned (archive them first)")
    if not os.path.isfile(final):
        raise Hold("final_model.zip is absent before the thin — NOT thinned")
    app = os.path.join(spec.work_dir, f"{it.run}.thin_apply.txt")
    rc = runner.run([*planner, "--apply"], cwd=spec.repo, env=env, timeout_s=3600, out_path=app)
    if rc != 0:
        raise Hold(f"thin --apply rc={rc} (4 = a kept file's hash mismatch): {tail(app, 2, 200)}")
    if not os.path.isfile(final) or newest not in checkpoint_steps(run_dir):
        raise Hold("after the thin, final_model.zip or the end-step checkpoint is GONE")
    return f"thinned (kept final_model.zip + checkpoint_{newest}): {tail(app, 1, 160)}"


def _ts(line: str) -> dt.datetime:
    return dt.datetime.fromisoformat(line[1:line.index("]")])


def dose_record(it: ArmItem, status_lines: List[str], desc: Describe, launcher_log: str, rc: int,
                comparator: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The descriptive dose record (``x35_dose_v1``): wall / GPU-h / steps/s / first and last step."""
    tag = launch_tag(it.arm, it.seed)
    start_p = line_pattern(tag, start_body(it.run))
    end_p = line_pattern(tag, "END " + re.escape(it.run) + r" launcher_rc=-?\d+")
    segs, open_t = [], None
    for ln in status_lines:
        if start_p.match(ln):
            if open_t is not None:
                raise Hold(f"dose: two START lines for {it.run} with no END between them")
            open_t = _ts(ln)
        elif end_p.match(ln) and open_t is not None:
            segs.append((open_t, _ts(ln)))
            open_t = None
    if not segs:
        raise Hold(f"dose: no START → END segment for {it.run} under '{tag}:'")
    if any(t1 <= t0 for t0, t1 in segs):
        raise Hold(f"dose: a segment of {it.run} ends before it starts")
    t0, t1 = segs[0][0], segs[-1][1]
    m = re.search(r"checkpoint_(\d+)_steps\.zip$", desc.argv[desc.argv.index("--model") + 1]) if "--model" in desc.argv else None
    if not m:
        raise Hold("dose: the describe's --model is not a checkpoint_<N>_steps.zip")
    s = parse_launcher_log(launcher_log)
    if not (s.pins and s.last_step is not None and s.fps is not None):
        raise Hold("dose: the launcher log lacks --pin-commit / 'Last step' / 'FPS'")
    if s.pins != [it.pin]:
        raise Hold(f"dose: the launcher log names pin(s) {s.pins}, not the arm's {it.pin} (a cross-pin launch)")
    wall = sum((b - a).total_seconds() for a, b in segs) / 60
    rec: Dict[str, Any] = {
        "schema": "x35_dose_v1", "run": it.run, "arm": it.arm, "seed": it.seed, "pin": it.pin,
        "wall_min": round(wall, 1), "gpu_h": round(wall / 60, 3), "gpu_h_note": "one card: GPU-h = wall START→END",
        "steps_per_s": s.fps, "steps_per_s_source": "launcher summary 'FPS'", "first_step": int(m.group(1)),
        "last_step": s.last_step, "target_steps": desc.steps, "restarts": s.restarts, "launcher_rc": rc,
        "start": t0.isoformat(), "end": t1.isoformat(), "segments": len(segs),
        "segments_note": "wall = the sum of every START→END segment; FPS / last step from the newest launcher log"}
    if comparator:
        rec["c0_comparator"] = dict(comparator)
    return rec
