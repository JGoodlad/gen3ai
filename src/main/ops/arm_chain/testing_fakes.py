"""Test doubles for the arm chain (imported by the ``*_test.py`` files only; nothing here runs in production).

* :data:`FAKE_LAUNCH_SH` — a bash stand-in for ``launch_x35.sh`` with the same argv, the same files (describe,
  launcher dry-run transcript, pid file, launcher log) and the same status lines (``<arm> s<seed>: START …``,
  ``post-spawn pool-seed check PASS``, ``END <run> launcher_rc=N``). Its "launcher" is a real ``sleep`` whose argv
  reads ``python -m main.launcher …`` so ``/proc`` identity checks see a launcher. Per-run knobs come from
  ``<job>/fake/<run>.env`` (``FAKE_RC``, ``FAKE_LAST``, ``FAKE_FINAL``, ``FAKE_BANNER``, ``FAKE_RUN_S``,
  ``FAKE_DRY_FAIL``, ``FAKE_NO_PASS``, ``FAKE_DIRTY_MAIN``: the dry run writes a stray record into main's stage, as
  the unfixed kit's ``k9b-batch`` did); every call appends ``<mode> <arm> <seed> <pin> CVD=<state>`` to
  ``<job>/fake/calls.log``.
* :class:`FakeRunner` — scripted handlers for ``main.x35.kit k9b`` and the thin planner; anything else runs for real.
* :func:`make_world` — a temp main checkout (git) with a records stage, the fake kit, a parent run and a job dir.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from main.ops.arm_chain.procwatch import ProcRef
from main.ops.arm_chain.runner import SubprocessRunner
from main.ops.arm_chain.spec import ChainSpec, parse

PARENT_STEP = 1000
TARGET_STEPS = 5000

FAKE_LAUNCH_SH = r'''#!/usr/bin/env bash
set -u
ARM=$1; SEED=$2; shift 2; PIN=""; GO=0
while [ $# -gt 0 ]; do case "$1" in --pin) PIN=$2; shift 2;; --lease-owner|--max-load|--t62|--at-epoch|--phase1) shift 2;;
  GO) GO=1; shift;; *) shift;; esac; done
J=$X35_JOB_DIR; ST=$J/status.txt; M=$(dirname "$J")/models; mkdir -p "$J/fake"
RUN=rb_${ARM}_s${SEED}; [ -f "$J/fake/$ARM.$SEED.run" ] && RUN=$(cat "$J/fake/$ARM.$SEED.run")
FAKE_RC=0; FAKE_LAST=5000; FAKE_FINAL=1; FAKE_BANNER=""; FAKE_RUN_S=1.2; FAKE_DRY_FAIL=0; FAKE_NO_PASS=0; FAKE_DIRTY_MAIN=0
[ -f "$J/fake/$RUN.env" ] && . "$J/fake/$RUN.env"
log() { echo "[$(date -Is)] $ARM s$SEED: $1" >> "$ST"; }
CVD=${CUDA_VISIBLE_DEVICES-unset}; [ -z "${CUDA_VISIBLE_DEVICES-x}" ] && CVD=EMPTY
echo "$([ $GO = 1 ] && echo GO || echo DRY) $ARM $SEED $PIN CVD=$CVD" >> "$J/fake/calls.log"
P=$M/parent_run
cat > "$J/describe.$ARM.$SEED.json" <<EOF
{"arm": "$ARM", "seed": $SEED, "run_name": "$RUN", "parent_run_dir": "$P", "delta_text": "+4,000 steps",
 "argv": ["--model", "$P/checkpoints/checkpoint_1000_steps.zip", "--steps", "5000", "--run-name", "$RUN"]}
EOF
if [ "$FAKE_DRY_FAIL" = 1 ]; then log "HOLD: fake dry run refusal"; echo "HOLD: fake" >&2; exit 7; fi
if [ "$FAKE_DIRTY_MAIN" = 1 ] && [ "$GO" != 1 ]; then
  echo '{"verdict": "PASS"}' > "$(dirname "$(dirname "$0")")/designs/stage/k9b/written_by_k9b_batch.json"; fi
[ -e "$M/$RUN" ] && { log "HOLD: $M/$RUN exists"; exit 7; }
CU="✓ CUDA_VISIBLE_DEVICES unset"; [ "$CVD" = EMPTY ] && CU="✗ CUDA_VISIBLE_DEVICES is EMPTY"
cat > "$J/$RUN.dry.dryrun.txt" <<EOF
  role        : FORK of $P
  pin         : $PIN  (source: pin_commit)
  steps       : --steps 5,000 vs checkpoint at 1,000 (sidecar) → +4,000 steps
  cuda devices : $CU
  ✓ DRY RUN — this command would launch. Nothing was created or modified.
EOF
if [ "$GO" != 1 ]; then echo "DRY RUN PASS — every pre-check passed"; exit 0; fi
log "START $RUN: --model $P/checkpoints/checkpoint_1000_steps.zip (pin ${PIN:0:8}, restart 24 h)"
R=$M/$RUN; mkdir -p "$R/checkpoints"
bash -c "exec -a 'python -m main.launcher --run-name $RUN' sleep $FAKE_RUN_S" &
LP=$!; echo "$LP" > "$J/$RUN.pid"
[ -n "$FAKE_BANNER" ] && echo "$FAKE_BANNER" >> "$J/$RUN.launcher.log"
echo "  launching with --pin-commit $PIN" >> "$J/$RUN.launcher.log"
[ "$FAKE_NO_PASS" = 1 ] || log "post-spawn pool-seed check PASS: fake"
T=$(date +%s); i=0
for s in 2000 3000 4000 $((FAKE_LAST - 1)); do : > "$R/checkpoints/checkpoint_${s}_steps.zip"; : > "$R/checkpoints/checkpoint_${s}_steps.json"
  touch -d "@$((T - 400 + i * 100))" "$R/checkpoints/checkpoint_${s}_steps.zip"; i=$((i + 1)); done
[ "$FAKE_FINAL" = 1 ] && : > "$R/final_model.zip"
printf '  Last step   : %s\n  FPS         : 1,400\n  Restarts    : 0\n' "$FAKE_LAST" >> "$J/$RUN.launcher.log"
wait "$LP"
log "END $RUN launcher_rc=$FAKE_RC"
'''


def _git(cwd: str, *args: str) -> str:
    return subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True).stdout


def init_repo(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    return path


def commit_all(path: str, msg: str = "c") -> str:
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", msg, "--allow-empty")
    return _git(path, "rev-parse", "HEAD").strip()


@dataclass
class World:
    root: str
    repo: str
    job: str
    models: str
    stage_rel: str
    pin_a: str
    pin_b: str

    def chain_dict(self, items: List[Dict[str, Any]], **over: Any) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "schema": "arm_chain_v1", "name": "t", "job_dir": self.job, "repo": self.repo, "models_dir": self.models,
            "python": "/usr/bin/python3",
            "backend": {"kind": "x35", "launch_script": "kit/launch_x35.sh", "stage": "stage",
                        "lease_owner": "Training Oct", "lease_token_file": os.path.join(self.root, "lease.token")},
            "records": {"stage": self.stage_rel, "submodules": []}, "disk_headroom_gib": 0, "items": items}
        d.update(over)
        return d

    def spec(self, items: List[Dict[str, Any]], **over: Any) -> ChainSpec:
        return parse(self.chain_dict(items, **over))

    def arm(self, arm: str, seed: int, pin: str, **over: Any) -> Dict[str, Any]:
        d = {"kind": "arm", "arm": arm, "seed": seed, "pin": pin, "run": f"rb_{arm}_s{seed}", "go_by": "orchestrator test",
             "thin": "planner", "disk_need_gib": 0.0, "on_void": "page", "on_throughput_fail": "page"}
        d.update(over)
        return d

    def knobs(self, run: str, **kv: Any) -> None:
        os.makedirs(os.path.join(self.job, "fake"), exist_ok=True)
        with open(os.path.join(self.job, "fake", f"{run}.env"), "w") as f:
            f.write("".join(f"{k}={shlex.quote(str(v))}\n" for k, v in kv.items()))

    def calls(self) -> List[str]:
        try:
            return open(os.path.join(self.job, "fake", "calls.log")).read().splitlines()
        except OSError:
            return []


def make_world(tmp: str) -> World:
    root = str(tmp)
    repo = init_repo(os.path.join(root, "main"))
    stage_rel = "designs/stage"
    os.makedirs(os.path.join(repo, stage_rel, "k9b"))
    with open(os.path.join(repo, stage_rel, "k9b", "old_run.json"), "w") as f:
        f.write('{"verdict": "PASS"}\n')
    os.makedirs(os.path.join(repo, "kit"))
    sh = os.path.join(repo, "kit", "launch_x35.sh")
    with open(sh, "w") as f:
        f.write(FAKE_LAUNCH_SH)
    os.chmod(sh, 0o755)
    pin_a = commit_all(repo, "a")
    pin_b = commit_all(repo, "b")
    models = os.path.join(root, "models")
    os.makedirs(os.path.join(models, "parent_run", "checkpoints"))
    job = os.path.join(root, "job")
    os.makedirs(job)
    return World(root=root, repo=repo, job=job, models=models, stage_rel=stage_rel, pin_a=pin_a, pin_b=pin_b)


Handler = Tuple[Callable[[Sequence[str]], bool], Callable[[Sequence[str], str], int]]


@dataclass
class FakeRunner:
    """Scripted ``run`` for matched argv; real subprocesses otherwise. Every call is recorded."""

    handlers: List[Handler] = field(default_factory=list)
    calls: List[Tuple[List[str], Dict[str, str]]] = field(default_factory=list)
    real: SubprocessRunner = field(default_factory=SubprocessRunner)

    def on(self, pred: Callable[[Sequence[str]], bool], fn: Callable[[Sequence[str], str], int]) -> "FakeRunner":
        self.handlers.append((pred, fn))
        return self

    def run(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout_s: float, out_path: str) -> int:
        self.calls.append((list(argv), dict(env)))
        for pred, fn in self.handlers:
            if pred(argv):
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                return fn(argv, out_path)
        return self.real.run(argv, cwd=cwd, env=env, timeout_s=timeout_s, out_path=out_path)

    def spawn(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], out_path: str) -> ProcRef:
        self.calls.append((list(argv), dict(env)))
        return self.real.spawn(argv, cwd=cwd, env=env, out_path=out_path)


def _arg(argv: Sequence[str], name: str) -> str:
    return argv[list(argv).index(name) + 1]


def k9b_handler(verdict: str = "PASS", into_main_stage: Optional[str] = None, rc: int = 0) -> Handler:
    """``main.x35.kit k9b``: write ``<stage>/k9b/<run>.json`` (or, to test the guard, into ``into_main_stage``)."""
    def pred(argv: Sequence[str]) -> bool:
        return "main.x35.kit" in argv and "k9b" in argv

    def fn(argv: Sequence[str], out: str) -> int:
        run = os.path.basename(_arg(argv, "--run-dir"))
        stage = into_main_stage or _arg(argv, "--stage")
        os.makedirs(os.path.join(stage, "k9b"), exist_ok=True)
        with open(os.path.join(stage, "k9b", f"{run}.json"), "w") as f:
            json.dump({"run": run, "verdict": verdict}, f)
        with open(out, "w") as f:
            f.write(f"[x35 kit] K9(b) {verdict}\n")
        return rc
    return pred, fn


def planner_handler(models: str, refuse_open_times: int = 0, keep_extra: Sequence[str] = (),
                    other_refusal: bool = False) -> Handler:
    """The thin planner: keep the newest checkpoint (+ ``keep_extra``); ``--apply`` deletes the rest."""
    state = {"open": refuse_open_times}

    def pred(argv: Sequence[str]) -> bool:
        return any(a.endswith("archive_grooming_dryrun.py") for a in argv)

    def fn(argv: Sequence[str], out: str) -> int:
        run = _arg(argv, "--thin-run")
        if other_refusal:
            open(out, "w").write("THIN REFUSED: a reference file names a planned file\n")
            return 3
        if state["open"] > 0:
            state["open"] -= 1
            open(out, "w").write("THIN REFUSED: 2 process(es) have the run dir open\n")
            return 3
        ck = os.path.join(models, run, "checkpoints")
        zips = sorted((int(n.split("_")[1]), n) for n in os.listdir(ck) if n.endswith("_steps.zip"))
        keep = {zips[-1][1], *keep_extra}
        lines = [f"checkpoint zips kept    {len(keep)}"] + [f"    {n}   kept" for n in sorted(keep)]
        lines.append(f"zips to DELETE          {len(zips) - len(keep)}")
        if "--apply" in argv:
            for _, n in zips:
                if n not in keep:
                    os.remove(os.path.join(ck, n))
            lines.append("APPLIED")
        open(out, "w").write("\n".join(lines) + "\n")
        return 0
    return pred, fn
