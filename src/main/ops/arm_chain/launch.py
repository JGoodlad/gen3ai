"""Pre-launch and GO for one arm, through the X35 launch kit (``launch_x35.sh``, in the repo).

ONE OWNER PER CHECK. The kit's own pre-checks (live run, lease, the pin's required commits, ``kit pin-check`` /
``required-check`` / ``corpus-check`` / ``stage-check`` / ``pool-check`` / ``k9b-batch``, ``checkargs --pin``, the
launcher ``--dry-run``, the disk) stay in ``launch_x35.sh``; this module does not re-implement them. It adds what
the job-directory scripts used to bolt on, and RE-VERIFIES the dry run independently:

1. the arm's ``verify`` commands (e.g. ``main.x35.x52 verify``) pass before the dry run AND again right before GO;
2. the CPU gate (``cpugate``) passes, polled every 4 min for at most 60 min;
3. the kit's dry run passes AT THE ARM'S OWN PIN, with the card visible (the launcher's CUDA preflight refuses a
   ``--device cuda`` dry run under an empty ``CUDA_VISIBLE_DEVICES``);
4. the kit's describe names THIS run (``describe.<arm>.<seed>.json``, by its deterministic path — never the first
   file a ``grep -l`` happens to list), and the launcher transcript reads ``this command would launch``,
   ``role : FORK of <the describe's parent>``, ``pin : <the arm's pin>``, the describe's step delta, and
   ``cuda devices : ✓``.

GO re-runs the verify commands, re-asserts the card is visible and that ``models/<run>`` does not exist, then
spawns ``launch_x35.sh … GO`` detached. The wrapper owns the launch, the post-spawn pool-seed check and the
``END <run> launcher_rc=N`` line; the chain watches it by ``/proc`` identity.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from main.ops.arm_chain import cpugate
from main.ops.arm_chain.procwatch import ProcRef
from main.ops.arm_chain.runner import Runner, assert_card_visible, cpu_env, go_env, tail
from main.ops.arm_chain.spec import ArmItem, ChainSpec


class Hold(RuntimeError):
    """A step refused: the chain stops at this item and pages (exit 3). The message is the status line."""


@dataclass(frozen=True)
class Describe:
    run_name: str
    parent_run_dir: str
    delta_text: str
    steps: int
    argv: List[str]

    @staticmethod
    def load(path: str) -> "Describe":
        try:
            text = open(path, errors="replace").read()
        except OSError as e:
            raise Hold(f"no readable kit describe at {path}: {e}") from e
        try:   # a transcript may carry stderr lines around the JSON object
            d: Dict[str, Any] = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except ValueError as e:
            raise Hold(f"no readable kit describe at {path}: {e}") from e
        a = list(d.get("argv") or [])
        if "--steps" not in a or a.index("--steps") + 1 >= len(a):
            raise Hold(f"the describe {path} carries no --steps")
        return Describe(run_name=str(d.get("run_name")), parent_run_dir=str(d.get("parent_run_dir")),
                        delta_text=str(d.get("delta_text")), steps=int(a[a.index("--steps") + 1]), argv=a)


def describe_path(spec: ChainSpec, it: ArmItem) -> str:
    return os.path.join(spec.job_dir, f"describe.{it.arm}.{it.seed}.json")


def dry_transcript_path(spec: ChainSpec, it: ArmItem) -> str:
    return os.path.join(spec.job_dir, f"{it.run}.dry.dryrun.txt")


_ROLE = re.compile(r"^\s*role\s*:\s*FORK of (\S+)\s*$", re.M)
_PIN = re.compile(r"^\s*pin\s*:\s*([0-9a-f]{40})\b", re.M)
_CUDA_OK = re.compile(r"^\s*cuda devices\s*:\s*✓", re.M)


def verify_dry_transcript(text: str, pin: str, parent: str, delta_text: str) -> List[str]:
    """The problems with a launcher ``--dry-run`` transcript (empty = it is the launch we meant)."""
    bad = []
    if "this command would launch" not in text:
        bad.append("the dry run does not say 'this command would launch'")
    roles = _ROLE.findall(text)
    if len(roles) != 1:
        bad.append(f"the dry run's role is not exactly one 'FORK of …' line (found {len(roles)})")
    elif os.path.realpath(roles[0]) != os.path.realpath(parent):
        bad.append(f"the dry run forks {roles[0]}, not the describe's parent {parent}")
    pins = _PIN.findall(text)
    if pins != [pin]:
        bad.append(f"the dry run's pin is {pins or 'absent'}, not the arm's {pin}")
    if delta_text not in text:
        bad.append(f"the dry run's step arithmetic does not read {delta_text!r}")
    if not _CUDA_OK.search(text):
        bad.append("the dry run does not read 'cuda devices : ✓' (the GO would not see the card)")
    return bad


class X35Launcher:
    def __init__(self, spec: ChainSpec, runner: Runner, base_env: Mapping[str, str],
                 gate: Optional[Callable[[], cpugate.GateResult]] = None,
                 sleep: Optional[Callable[[float], None]] = None, gate_polls: int = 13, gate_every_s: float = 240.0,
                 niceness: Optional[Callable[[], int]] = None):
        self.spec, self.runner, self.base_env = spec, runner, dict(base_env)
        self._gate = gate or (lambda: cpugate.sample(spec.trainer_cores, spec.agent_cores, me=os.getpid()))
        import time
        self._sleep = sleep or time.sleep
        self._now = time.time
        self._niceness = niceness or (lambda: os.getpriority(os.PRIO_PROCESS, 0))
        self.gate_polls, self.gate_every_s = gate_polls, gate_every_s

    @property
    def script(self) -> str:
        return os.path.join(self.spec.repo, self.spec.backend.launch_script)

    def _kit_argv(self, it: ArmItem) -> List[str]:
        b = self.spec.backend
        return [self.script, it.arm, str(it.seed), "--pin", it.pin, "--lease-owner", b.lease_owner,
                "--max-load", f"{b.max_load:g}", *it.flags]

    def _env(self) -> Dict[str, str]:
        nice = self._niceness()
        if nice != 0:
            raise Hold(f"the supervisor runs at nice {nice}: a GO would start the trainer at nice {nice} for its whole "
                       "life (RLIMIT_NICE forbids lowering it) — restart the supervisor at nice 0")
        env = go_env(self.base_env, self.spec.repo, self.spec.backend.lease_token_file, self.spec.python)
        env["X35_JOB_DIR"] = self.spec.job_dir   # the kit writes its describe / transcripts / status lines here
        return env

    def run_verify(self, it: ArmItem, when: str) -> List[str]:
        notes = []
        for i, v in enumerate(it.verify):
            out = os.path.join(self.spec.work_dir, f"{it.run}.verify{i}.{when}.txt")
            rc = self.runner.run(v, cwd=self.spec.repo, env=cpu_env(self.base_env, self.spec.repo,
                                                                     models_dir=self.spec.models_dir), timeout_s=900,
                                 out_path=out)
            if rc != 0:
                raise Hold(f"{it.run}: verify[{i}] ({' '.join(v)[:80]}) FAILED {when} (rc {rc}): {tail(out, 1, 160)}")
            notes.append(f"verify[{i}] {when}: {tail(out, 1, 120)}")
        return notes

    def cpu_gate(self, it: ArmItem) -> str:
        last = ""
        for i in range(self.gate_polls):
            g = self._gate()
            if g.ok:
                return g.line
            last = g.line
            if i + 1 < self.gate_polls:
                self._sleep(self.gate_every_s)
        raise Hold(f"{it.run}: the CPU gate never PASSED in {self.gate_polls} polls — NOT launched; last: {last}")

    def prelaunch(self, it: ArmItem) -> Describe:
        """verify → CPU gate → the kit's dry run at the arm's pin → the independent re-verification."""
        self.run_verify(it, "before the dry run")
        self.cpu_gate(it)
        env = self._env()
        assert_card_visible(env)
        out = os.path.join(self.spec.work_dir, f"{it.run}.kit_dry.txt")
        t0 = self._now()
        rc = self.runner.run(self._kit_argv(it), cwd=self.spec.repo, env=env, timeout_s=1500, out_path=out)
        if rc != 0:
            raise Hold(f"{it.run}: the kit dry run failed (rc {rc}): {tail(out, 2, 200)} — NOT launched")
        if "DRY RUN PASS" not in open(out, errors="replace").read():
            raise Hold(f"{it.run}: the kit dry run exited 0 without 'DRY RUN PASS' — NOT launched")
        def fresh(p: str) -> None:
            if not os.path.exists(p) or os.path.getmtime(p) < t0 - 1:
                raise Hold(f"{it.run}: {p} was not written by THIS dry run (a stale file is never read as this launch)")
        fresh(describe_path(self.spec, it))
        d = Describe.load(describe_path(self.spec, it))
        if d.run_name != it.run:
            raise Hold(f"the kit describe for {it.arm} s{it.seed} names run {d.run_name!r}, not the chain's {it.run!r}")
        fresh(dry_transcript_path(self.spec, it))
        try:
            text = open(dry_transcript_path(self.spec, it), errors="replace").read()
        except OSError as e:
            raise Hold(f"{it.run}: no launcher dry-run transcript: {e}") from e
        bad = verify_dry_transcript(text, it.pin, d.parent_run_dir, d.delta_text)
        if bad:
            raise Hold(f"{it.run}: the dry run is not the launch we meant: " + "; ".join(bad))
        return d

    def go(self, it: ArmItem) -> ProcRef:
        self.run_verify(it, "at the GO")
        if os.path.exists(os.path.join(self.spec.models_dir, it.run)):
            raise Hold(f"models/{it.run} already exists — a GO never re-launches a run (a restart is the launcher's)")
        env = self._env()
        assert_card_visible(env)
        out = os.path.join(self.spec.work_dir, f"{it.run}.kit_go.txt")
        return self.runner.spawn([*self._kit_argv(it), "GO"], cwd=self.spec.repo, env=env, out_path=out)

    def describe_offline(self, it: ArmItem) -> Describe:
        """The kit's describe for the arm, computed now on CPU (read-only; nothing launched): the chain file's arm
        name, seed and run are checked when the chain is PLANNED, not at the hand-off (10-20: ``x47`` was handed to
        the GO, the kit knows only ``toy_x47``, and the GPU idled until a hand relaunch)."""
        side = "toy" if it.arm.startswith("toy_") else "production"
        call = os.path.join(self.spec.repo, self.spec.backend.stage, f"reference_population_{side}.json")
        out = os.path.join(self.spec.work_dir, f"plan_describe.{it.arm}.{it.seed}.json")
        argv = [self.spec.python, "-m", "main.x35.kit", "describe", "--arm", it.arm, "--seed", str(it.seed),
                "--call", call]
        rc = self.runner.run(argv, cwd=self.spec.repo, env=cpu_env(self.base_env, self.spec.repo), timeout_s=600,
                             out_path=out)
        if rc != 0:
            raise Hold(f"{it.key}: the kit refuses arm {it.arm!r} seed {it.seed} (rc {rc}): {tail(out, 2, 200)}")
        d = Describe.load(out)
        if d.run_name != it.run:
            raise Hold(f"{it.key}: the kit names arm {it.arm} s{it.seed}'s run {d.run_name!r}, not {it.run!r}")
        return d

    def find_go_wrapper(self, it: ArmItem, proc: str = "/proc") -> Optional[ProcRef]:
        """A live ``launch_x35.sh <arm> <seed> … GO`` process for this arm (by argv), or None: the evidence that a
        GO is in flight before the kit's GO-time pre-checks (minutes) let it write its START line."""
        from utils import procfs
        from main.ops.arm_chain.procwatch import ref_of
        procs = procfs.snapshot(proc)

        def matches(pid: int) -> bool:
            argv = [a for a in procfs.read_text(pid, "cmdline", proc).split("\0") if a]
            if self.script not in argv or argv[-1:] != ["GO"]:
                return False
            i = argv.index(self.script)
            return argv[i + 1:i + 3] == [it.arm, str(it.seed)]
        hits = [pid for pid in procs if matches(pid)]
        # the wrapper's own subshells carry the same argv: take the OUTERMOST (its parent is not a match)
        for pid in sorted(hits):
            if procs[pid].ppid not in hits:
                ref = ref_of(pid, proc)
                if ref is not None:
                    return ref
        return None

    def launcher_pidfile(self, it: ArmItem) -> str:
        return os.path.join(self.spec.job_dir, f"{it.run}.pid")

    def launcher_log(self, it: ArmItem) -> str:
        return os.path.join(self.spec.job_dir, f"{it.run}.launcher.log")
