"""Ship arm-end RECORDS to main from a SOURCE COPY, through a fresh worktree: main is never dirty.

The records-only ``/gen3ai-ship`` the orchestrator DELEGATES to the training session for arm-end records. It never
commits code: every destination must be a ``.json`` under the chain's records stage. The class bugs it closes:

* **A record written into main's tree, then shipped from there** (arm_end v13): the source is a copy outside the
  checkout; main must not hold the destination (tracked or stray) before, and holds a byte-identical copy after.
* **The ship worktree lacked a submodule a gate reads** (``deps/pokemon-showdown``, read by the freshness gate since
  a8dd5bd3f; ship_record v1 HELD 10-25 18:22): the configured submodules are checked out before the static tier.
* **A checked-out submodule blocked a plain ``git worktree remove``**: the remove is ``--force``, and only after
  ``utils.worktree_guard`` finds no run data in the tree (the worktree-removal incident).
* **Two ships pushing at once** (the K9(b) ship and the dose ship were chained by hand): one ``flock`` serialises
  every ship of every chain on the box.

Any failure is a :class:`~main.ops.arm_chain.launch.Hold` naming the step; a worktree left behind is named in it.

:func:`kit_k9b_out_of_main` decides whether the chain may ship ASYNCHRONOUSLY: only once the kit's ``k9b-batch``
pre-check can no longer write a missing record into main's stage (F1), because the next arm's dry run runs it while
the previous arm's records are still in flight.
"""
from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from main.ops.arm_chain.launch import Hold
from main.ops.arm_chain.runner import Runner, agent_prefix, cpu_env
from main.ops.arm_chain.spec import ChainSpec

#: A ship worktree is removed only after this guard finds no run data in it.
StaticTier = Callable[[str], Tuple[int, str]]


def _git(cwd: str, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise Hold(f"git {' '.join(args)[:80]} failed in {cwd}: {(p.stderr or p.stdout).strip()[:200]}")
    return p.stdout


@dataclass
class ShipResult:
    commit: str
    files: List[str]
    already: bool = False     # the records were already on the remote, byte-identical (an idempotent re-ship)


class Shipper:
    def __init__(self, repo: str, records_stage: str, submodules: Sequence[str], work_dir: str, lock_path: str,
                 static_tier: StaticTier, push_guard: Optional[Callable[[str], int]] = None,
                 worktree_guard: Optional[Callable[[str, str], List[str]]] = None,
                 lock_timeout_s: float = 1800.0, remote: str = "origin", branch: str = "main"):
        self.repo, self.stage, self.subs = repo, records_stage.rstrip("/"), tuple(submodules)
        self.work_dir, self.lock_path = work_dir, lock_path
        self.static_tier = static_tier
        self.push_guard = push_guard or _push_guard
        self.worktree_guard = worktree_guard or _worktree_guard
        self.lock_timeout_s, self.remote, self.branch = lock_timeout_s, remote, branch

    def _check_dest(self, rel: str) -> None:
        norm = os.path.normpath(rel)
        if os.path.isabs(rel) or norm.startswith("..") or not norm.startswith(self.stage + os.sep) \
                or not norm.endswith(".json"):
            raise Hold(f"refusing to ship {rel!r}: a record ship writes only .json files under {self.stage}/")

    def _lock(self) -> int:
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() - t0 > self.lock_timeout_s:
                    os.close(fd)
                    raise Hold(f"another ship held {self.lock_path} for {self.lock_timeout_s / 60:.0f} min") from None
                time.sleep(2)

    def ship(self, files: Mapping[str, str], message: str, tag: str) -> ShipResult:
        """``files`` = {repo-relative destination: source path}. Returns the pushed commit."""
        if not files:
            raise Hold("nothing to ship")
        for rel, src in files.items():
            self._check_dest(rel)
            if not os.path.isfile(src):
                raise Hold(f"the record source {src} is absent")
            if os.path.realpath(src).startswith(os.path.realpath(self.repo) + os.sep):
                raise Hold(f"the record source {src} is inside the main checkout (ship from a copy, never from main)")
        fd = self._lock()
        try:
            return self._ship_locked(dict(files), message, tag)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _prepare_and_commit(self, wt: str, files: Dict[str, str], message: str, R: str) -> None:
        for sub in self.subs:
            _git(wt, "submodule", "update", "--init", "-q", sub)
        for rel, src in files.items():
            if os.path.lexists(os.path.join(wt, rel)):
                raise Hold(f"{rel} is already on {R} — not overwritten")
            os.makedirs(os.path.dirname(os.path.join(wt, rel)), exist_ok=True)
            shutil.copy2(src, os.path.join(wt, rel))
        changed = sorted(ln[3:] for ln in _git(wt, "status", "--porcelain", "--untracked-files=all").splitlines())
        if changed != sorted(files):
            raise Hold(f"the ship worktree changed {changed}, not exactly {sorted(files)}")
        rc, out = self.static_tier(wt)
        if rc != 0:
            raise Hold(f"static tier failed (rc {rc}, {out}) — not shipped")
        _git(wt, "add", "--", *files)
        _git(wt, "commit", "-q", "-m", message)
        _git(wt, "fetch", "-q", self.remote, self.branch)
        if _git(wt, "log", "--oneline", f"HEAD..{R}").strip():
            _git(wt, "rebase", "-q", R)
        if self.push_guard(wt) != 0:
            raise Hold("push_guard refused")

    def _push(self, wt: str, R: str) -> None:
        """Push; a NON-FAST-FORWARD (someone landed between the rebase and the push) is rebased over once and retried."""
        for attempt in (1, 2):
            p = subprocess.run(["git", "-C", wt, "push", "-q", self.remote, f"HEAD:{self.branch}"], capture_output=True,
                               text=True)
            if p.returncode == 0:
                return
            if attempt == 2:
                raise Hold(f"push failed twice: {p.stderr.strip()[:200]}")
            _git(wt, "fetch", "-q", self.remote, self.branch)
            _git(wt, "rebase", "-q", R)
            if self.push_guard(wt) != 0:
                raise Hold("push_guard refused after the re-rebase")

    def _fast_forward_main(self, R: str, tries: int = 3, wait_s: float = 2.0) -> None:
        """Fetch and fast-forward main, retrying a transient failure (another git process's ``index.lock``)."""
        last = ""
        for i in range(tries):
            try:
                _git(self.repo, "fetch", "-q", self.remote, self.branch)
                _git(self.repo, "merge", "--ff-only", "-q", R)
                return
            except Hold as h:
                last = str(h)
                if i + 1 < tries:
                    time.sleep(wait_s)
        raise Hold(last)

    def _discard(self, wt: str, br: str, pushed: bool = False) -> str:
        """A HOLD leaves no worktree or branch behind (a retry would die at ``worktree add``), guard permitting."""
        hazards = self.worktree_guard(self.repo, wt)
        if hazards:
            return f"worktree LEFT at {wt} (worktree_guard: {'; '.join(hazards)[:120]})"
        p1 = subprocess.run(["git", "-C", self.repo, "worktree", "remove", "--force", wt], capture_output=True)
        p2 = subprocess.run(["git", "-C", self.repo, "branch", "-q", "-D", br], capture_output=True)
        if p1.returncode or p2.returncode:
            return f"worktree / branch cleanup FAILED: {wt} {br}"
        return "the ship worktree was removed" if pushed else "nothing pushed; the ship worktree was removed"

    def _already_shipped(self, files: Dict[str, str], R: str) -> Optional[str]:
        """The commit that landed these exact records, if ``R`` already holds ALL of them byte-identical (a supervisor
        that died between the push and saving RECORDED re-ships on restart: that must succeed, not HOLD forever).
        Some but not all present, or present with other bytes, is a HOLD."""
        same = []
        for rel, src in files.items():
            p = subprocess.run(["git", "-C", self.repo, "show", f"{R}:{rel}"], capture_output=True)
            if p.returncode != 0:
                same.append(False)
                continue
            if p.stdout != open(src, "rb").read():
                raise Hold(f"{R} already holds {rel} with DIFFERENT bytes than {src} — not shipped")
            same.append(True)
        if not any(same):
            return None
        if not all(same):
            raise Hold(f"{R} holds some of {sorted(files)} but not all — a partial earlier ship; not shipped")
        return _git(self.repo, "log", "-1", "--format=%H", R, "--", *files).strip()[:9] or "?"

    def _ship_locked(self, files: Dict[str, str], message: str, tag: str) -> ShipResult:
        repo, R = self.repo, f"{self.remote}/{self.branch}"
        dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            raise Hold(f"main is dirty before the ship (never ship onto a dirty main): {dirty.strip()[:200]}")
        _git(repo, "fetch", "-q", self.remote)
        done = self._already_shipped(files, R)
        if done is not None:
            self._fast_forward_main(R)
            return ShipResult(commit=done, files=sorted(files), already=True)
        for rel in files:
            if os.path.lexists(os.path.join(repo, rel)):
                raise Hold(f"main already holds {rel} (tracked or stray) — not shipped")
        br = f"armchain-ship-{tag}"
        wt = os.path.join(repo, ".claude", "worktrees", br)
        _git(repo, "worktree", "add", "-q", "-b", br, wt, R)
        try:
            self._prepare_and_commit(wt, files, message, R)
            self._push(wt, R)
        except Hold as h:
            raise Hold(f"{h} — {self._discard(wt, br)}") from None
        commit = _git(wt, "rev-parse", "--short=9", "HEAD").strip()
        # PUSHED: from here the records are on the remote. A failure below (a transient index.lock in main, say)
        # must still clean up the ship's worktree and branch, or the by-hand retry dies at ``worktree add`` (review
        # 10-26, finding 12); the retry then lands through ``_already_shipped``.
        try:
            self._fast_forward_main(R)
        except Hold as h:
            raise Hold(f"pushed as {commit}, but main's fast-forward FAILED ({h}); {self._discard(wt, br, pushed=True)} — the "
                       "records ARE on the remote: retry ship-records (it fast-forwards main)") from None
        for rel, src in files.items():
            with open(os.path.join(repo, rel), "rb") as a, open(src, "rb") as b:
                if a.read() != b.read():
                    raise Hold(f"main's tracked {rel} differs from the source {src} (pushed as {commit}); "
                               f"{self._discard(wt, br, pushed=True)}")
        hazards = self.worktree_guard(repo, wt)
        if hazards:
            raise Hold(f"worktree_guard refused {wt}: {'; '.join(hazards)[:200]} (pushed as {commit})")
        _git(repo, "worktree", "remove", "--force", wt)
        _git(repo, "branch", "-q", "-D", br)
        left = _git(repo, "status", "--porcelain", "--untracked-files=all")
        if left:
            raise Hold(f"main is not clean after the ship (pushed as {commit}): {left.strip()[:200]}")
        return ShipResult(commit=commit, files=sorted(files))


def _push_guard(wt: str) -> int:
    from utils import push_guard
    return int(push_guard.main(["--repo", wt]))


def _worktree_guard(main_checkout: str, wt: str) -> List[str]:
    from utils import worktree_guard as g
    try:
        return [str(h) for h in g.find_hazards(main_checkout, wt)]
    except (g.GuardError, OSError) as e:
        return [f"guard error (fails closed): {e}"]


def static_tier_runner(runner: Runner, python: str, agent_cores: str, base_env: Mapping[str, str],
                       out_dir: str) -> StaticTier:
    """The real static tier (``utils.static_gates --paths``) in the ship worktree, at nice 19 on the agent cores."""
    def run(wt: str) -> Tuple[int, str]:
        env = dict(base_env)
        env.update({"PYTHONPATH": os.path.join(wt, "src"), "CUDA_VISIBLE_DEVICES": "", "GEN3AI_SKIP_DEPS_GUARD": "1"})
        env.pop("GEN3AI_TEST_ALLOW_GPU", None)
        p = subprocess.run([python, "-m", "utils.static_gates", "--paths"], cwd=wt, env=env, capture_output=True,
                           text=True)
        if p.returncode != 0:
            return p.returncode, "utils.static_gates --paths failed"
        out = os.path.join(out_dir, f"ship_static_{os.path.basename(wt)}.txt")
        rc = runner.run([*agent_prefix(agent_cores, 19), python, "-m", "pytest", *p.stdout.split(), "-q", "-n", "2",
                         "-p", "no:cacheprovider"], cwd=wt, env=env, timeout_s=1800, out_path=out)
        return rc, out
    return run


def main_dirty(repo: str) -> str:
    """``git status --porcelain`` of the main checkout (untracked included): empty = clean. A failing git is dirt
    (it cannot be shown clean)."""
    p = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=all"], capture_output=True,
                       text=True)
    if p.returncode != 0:
        return f"(git status failed rc {p.returncode}: {(p.stderr or '').strip()[:160]})"
    return p.stdout.strip()


#: The probe run in MAIN's code (the kit runs main's ``src``): ``k9b_batch`` takes ``out_dir``, and a write into the
#: records stage inside the main checkout is REFUSED. Exit 0 = fixed.
_K9B_PROBE = """
import inspect, os, sys
from main.x35 import batch as b
if "out_dir" not in inspect.signature(b.k9b_batch).parameters:
    sys.exit("k9b_batch takes no out_dir")
refuse = getattr(b, "refuse_inside_main_checkout", None)
if refuse is None:
    sys.exit("no refuse_inside_main_checkout")
try:
    refuse(sys.argv[1], roots=[os.path.realpath(sys.argv[2])])
except Exception as e:
    print("refused:", type(e).__name__)
    sys.exit(0)
sys.exit("a write into main's stage is NOT refused")
"""

_K9B_BATCH_CALL = re.compile(r"\bkit\s+k9b-batch\b")


def kit_k9b_out_of_main(spec: ChainSpec, python: Optional[str] = None,
                        base_env: Optional[Mapping[str, str]] = None) -> Tuple[bool, str]:
    """``(ok, why)``: may the record ship be ASYNCHRONOUS? Only when BOTH hold in the MAIN checkout the kit runs:

    1. every ``kit k9b-batch`` call in the backend's launch script passes ``--out-dir`` (a scratch dir), and
    2. main's ``main.x35.batch`` takes ``k9b_batch(out_dir=…)`` and REFUSES a record write inside the main checkout.

    Anything else (an unreadable script, a probe that fails or times out) is ``False``: the ship stays synchronous,
    which is always safe."""
    script = os.path.join(spec.repo, spec.backend.launch_script)
    try:
        text = open(script, errors="replace").read()
    except OSError as e:
        return False, f"the launch script is unreadable ({e})"
    calls = [ln.strip() for ln in text.splitlines() if _K9B_BATCH_CALL.search(ln) and not ln.lstrip().startswith("#")]
    bare = [c for c in calls if "--out-dir" not in c]
    if bare:
        return False, f"the kit's k9b-batch pre-check still writes into the stage it reads (no --out-dir): {bare[0][:120]}"
    env = cpu_env(base_env if base_env is not None else os.environ, spec.repo)
    try:
        p = subprocess.run([python or spec.python, "-c", _K9B_PROBE, os.path.join(spec.repo, spec.backend.stage),
                            spec.repo], cwd=spec.repo, env=env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"the k9b-batch probe did not run ({type(e).__name__})"
    if p.returncode != 0:
        return False, f"main's kit still writes k9b-batch records into the stage: {(p.stderr or p.stdout).strip()[-160:]}"
    return True, f"the kit's k9b-batch writes outside main ({len(calls)} call(s) pass --out-dir; {p.stdout.strip()})"

