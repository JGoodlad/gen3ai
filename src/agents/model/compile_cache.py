"""HERMETIC COMPILE CACHE — every compile artifact a process reads was written by ITS OWN run
(gen3_hermetic_compile_cache_v1; Lane K item K3, `designs/endstate/program_rust_core.md`).

The owner (2026-09-29): "always initialize from a fresh setup — who knows how many silent errors a
shared cache allows". Before this module the learner's Inductor/Triton caches were UNMANAGED: one
shared `/tmp/torchinductor_<user>` (12 GB on tmpfs, i.e. RAM) used by every run, pin, torch env,
benchmark and test, plus the opponents' CPU compile in a second shared dir
(`/tmp/gen3ai_inductor_cache`). The 2026-09-29 K1b fault proved a cache KEY can omit a setting that
changes the artifact (a backward compiled WITH donated buffers was served to a donation-OFF compile;
`compile_config.COMPILE_CONFIG`'s comment has the story) — so the rule is NOT "trust the key".

THE RULE (one decision, made once per process, BEFORE the first compile):

* A TRAINING RUN (`prepare_run_cache`, called by the trainer the moment its run dir exists, before
  any compile, env worker, forkserver, eval subprocess or T2 service): the cache lives at
  `<run>/compile_cache/` — `inductor/` (TORCHINDUCTOR_CACHE_DIR: FX-graph, AOTAutograd, codegen,
  the opponents' CPU compile), `triton/` (TRITON_CACHE_DIR: T2's and the learner's CUDA kernels),
  `t2_aot/` (T2's AOTInductor packages). It is created EMPTY at a FRESH launch or a FORK, and reused
  only by the run's OWN restarts (`--model` = a checkpoint this run wrote), and only when the STAMP
  matches: the code's commit (a dirty tree never reuses — a sha cannot name it), the torch version,
  and `compile_config.config_row_hash()`. Any mismatch WIPES it. Children inherit the three
  environment variables, so every process of the run's tree compiles into the run's own cache.
* ANY OTHER PROCESS that compiles (a test, a benchmark, the prober, an offline meter) gets a FRESH
  private temp dir (`ensure_hermetic_cache`, called at every compile site), deleted when the process
  that created it exits (`atexit`) — unless something above it (a run, pytest's conftest, a
  benchmark) already declared one. A value equal to torch's own shared default is never honoured.
  Every temp cache carries its creator's PID in its name, and each new declaration SWEEPS the ones
  whose creator is dead, so a SIGKILLed process leaks until the next compile, not forever.

The real-obs parity gate still runs at EVERY start (it verifies outputs, whatever was loaded); this
module decides only what may be loaded. It never touches `/tmp/torchinductor_<user>` or
`/tmp/gen3ai_inductor_cache` — it stops using them; deleting them is the owner's call.
"""
from __future__ import annotations

import atexit
import dataclasses
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

SCHEMA = "gen3_hermetic_compile_cache_v1"
ENV_INDUCTOR = "TORCHINDUCTOR_CACHE_DIR"
ENV_TRITON = "TRITON_CACHE_DIR"
#: The cache ROOT this process tree compiles into (the run's `compile_cache/`, or a private temp
#: dir). T2's AOT packages go under it; its presence says the two torch variables are MANAGED.
ENV_ROOT = "GEN3AI_COMPILE_CACHE_DIR"
CACHE_SUBDIR = "compile_cache"
STAMP_FILE = "stamp.json"
#: The two shared dirs this module retired. Named so the refusal below can recognise them.
RETIRED_SHARED_DIRS = ("/tmp/gen3ai_inductor_cache",)


class CompileCacheError(RuntimeError):
    """The run's compile cache cannot be declared hermetically (a compile already ran in this
    process against another cache, or the cache dir is not writable). Fatal at startup."""


def torch_default_cache_dir() -> str:
    """Torch's own shared default (`/tmp/torchinductor_<user>`), computed the way both supported
    torches compute it — WITHOUT calling torch's `cache_dir()`, which would create it and export it."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 — no passwd entry (containers): torch sanitises the same way
        user = "unknown"
    return os.path.join(tempfile.gettempdir(), "torchinductor_" + user.replace("\\", "_"))


def _is_shared(path: Optional[str]) -> bool:
    if not path:
        return False
    p = os.path.realpath(path)
    return p == os.path.realpath(torch_default_cache_dir()) or any(
        p == os.path.realpath(d) for d in RETIRED_SHARED_DIRS)


# --------------------------------------------------------------------------- the stamp
def _code_identity() -> Dict[str, Any]:
    """The commit the running CODE came from (`utils.git.get_git_hash`, anchored at the code's own
    checkout — a pinned child reports its pin) and whether that checkout's `src/` is dirty."""
    from utils.git import get_git_hash
    from utils.paths import repo_root
    sha = get_git_hash()
    dirty: Optional[bool]
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "src"],
                                      text=True, stderr=subprocess.DEVNULL, cwd=str(repo_root()))
        dirty = bool(out.strip())
    except Exception:  # noqa: BLE001 — no git: identity unknown, which never reuses
        dirty = None
    return {"code_sha": sha, "code_dirty": dirty}


def cache_stamp() -> Dict[str, Any]:
    """What a cache must have been written under to be reused: code commit + torch version +
    the compile-config row's hash (+ the interpreter, so two conda envs never meet).

    Read from `compile_config` (data only), NEVER from `compile_control`: every trainer stamps its
    cache, compiled or not, and the adapter's torch check belongs to the compile path alone (P10-D —
    importing it here crashed every trainer on any torch but the recorded one)."""
    import torch

    from agents.model.compile_config import config_row_hash
    return {"schema": SCHEMA, **_code_identity(), "torch": str(torch.__version__),
            "python": sys.executable, "config_row_sha": config_row_hash()}


def stamp_mismatch(recorded: Any, current: Dict[str, Any]) -> Optional[str]:
    """None when `recorded` licenses reuse under `current`; else the reason it does not."""
    if not isinstance(recorded, dict):
        return "no readable stamp"
    stamp = recorded.get("stamp", recorded)
    diffs = [f"{k}: {stamp.get(k)!r} -> {current.get(k)!r}" for k in sorted(current)
             if stamp.get(k) != current.get(k)]
    if diffs:
        return "stamp mismatch (" + "; ".join(diffs) + ")"
    if current.get("code_sha") in (None, "", "unknown") or current.get("code_dirty") is not False:
        return ("the code cannot be named by a commit (dirty or unknown tree) — a dirty tree "
                "never reuses a cache")
    return None


# --------------------------------------------------------------------------- the run's cache
@dataclasses.dataclass(frozen=True)
class CacheDecision:
    root: str
    action: str            # "created" | "wiped" | "reused"
    reason: str
    bytes_before: int
    stamp: Dict[str, Any]

    def line(self) -> str:
        return (f"🧊 [CompileCache] {self.action.upper()} {self.root} — {self.reason} "
                f"({self.bytes_before / 1e6:.1f} MB before; torch {self.stamp.get('torch')}, "
                f"code {str(self.stamp.get('code_sha'))[:10]}"
                f"{' DIRTY' if self.stamp.get('code_dirty') else ''}, "
                f"config row {str(self.stamp.get('config_row_sha'))[:10]}) [{SCHEMA}]")


def dir_bytes(path: str) -> int:
    total = 0
    for dp, _dn, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.lstat(os.path.join(dp, fn)).st_size
            except OSError:
                pass
    return total


def _export(root: str) -> None:
    os.environ[ENV_ROOT] = root
    os.environ[ENV_INDUCTOR] = os.path.join(root, "inductor")
    os.environ[ENV_TRITON] = os.path.join(root, "triton")
    for sub in ("inductor", "triton"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)


def _refuse_prior_compile() -> None:
    """A compile that already ran in THIS process wrote to whatever cache was in force, and its
    in-memory artifacts would outlive the switch — so the run's cache must be declared first."""
    cur = os.environ.get(ENV_INDUCTOR)
    if cur and _is_shared(cur):
        raise CompileCacheError(
            f"{ENV_INDUCTOR} is already {cur!r} — the shared cache torch exports on its first "
            f"compile — before the run declared its own. Something compiled before "
            f"`prepare_run_cache`; move that compile after it (or unset the variable if it was "
            f"exported by hand).")


def prepare_run_cache(run_dir: str, *, same_run_resume: bool,
                      emit: Callable[[str], None] = print) -> CacheDecision:
    """Declare `<run_dir>/compile_cache/` as this process tree's ONLY compile cache.

    `same_run_resume` = this process continues the run from a checkpoint the run itself wrote
    (`main.train.fork_lr.is_same_run_checkpoint`). Anything else — a fresh launch, a fork, a
    warm-start init — gets an EMPTY cache. Overwrites any inherited value of the three variables."""
    _refuse_prior_compile()
    root = os.path.join(os.path.abspath(run_dir), CACHE_SUBDIR)
    stamp_path = os.path.join(root, STAMP_FILE)
    current = cache_stamp()
    existed = os.path.isdir(root)
    before = dir_bytes(root) if existed else 0
    if not existed:
        action, reason = "created", ("fresh launch or fork" if not same_run_resume
                                     else "a same-run restart with no cache yet")
    elif not same_run_resume:
        action, reason = "wiped", "a fresh launch or a fork never inherits a cache"
    else:
        try:
            with open(stamp_path) as f:
                recorded: Any = json.load(f)
        except Exception:  # noqa: BLE001 — absent / truncated: not licensed
            recorded = None
        why = stamp_mismatch(recorded, current)
        action, reason = ("reused", "same-run restart, stamp matches") if why is None else ("wiped", why)
    try:
        if action == "wiped":
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)
        prior = {}
        if action == "reused":
            with open(stamp_path) as f:
                prior = json.load(f)
        record = {"stamp": current,
                  "created_at": prior.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "reuses": int(prior.get("reuses", 0)) + (1 if action == "reused" else 0)}
        tmp = stamp_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        os.replace(tmp, stamp_path)
        _export(root)
    except OSError as exc:
        raise CompileCacheError(f"cannot prepare the run's compile cache at {root!r}: {exc}") from exc
    decision = CacheDecision(root=root, action=action, reason=reason, bytes_before=before, stamp=current)
    emit(decision.line())
    return decision


# --------------------------------------------------------------------------- everyone else
_PRIVATE: Dict[str, Any] = {"root": None, "pid": None}
#: Temp-cache name prefixes; the creator's PID follows the prefix (`<prefix><pid>_<random>`), so a
#: dir whose creator died without cleaning up (SIGKILL, SIGTERM's default action, a hard crash) is
#: recognisable and swept by the next process that declares one — /tmp is tmpfs (RAM and inodes).
PRIVATE_PREFIX = "gen3ai_compile_"
PYTEST_PREFIX = "gen3ai_pytest_compile_"
#: Where every TEMP compile cache lives: `utils.scratch.scratch_root()` — `$GEN3AI_SCRATCH`, else
#: `~/.cache/gen3ai/tmp`, the REAL disk, never tmpfs `/tmp` (RAM + a 1,048,576-inode cap that ran out
#: on 2026-09-30; a compile cache is thousands of files and up to 1.5 GB on torch 2.8). ONE helper,
#: shared with the root conftest's pytest temp root. A run's own cache is already on disk.
from utils.scratch import SCRATCH_ENV as ENV_SCRATCH  # noqa: E402,F401  (re-exported)
from utils.scratch import ScratchOnRamError, fs_type  # noqa: E402,F401  (fs_type: re-exported)


def scratch_root() -> str:
    """`utils.scratch.scratch_root()`, its tmpfs refusal surfaced as this module's typed error."""
    from utils.scratch import scratch_root as _root
    try:
        return _root()
    except ScratchOnRamError as exc:
        raise CompileCacheError(str(exc)) from exc


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_temp_caches(tmpdir: Optional[str] = None) -> List[str]:
    """Delete this user's K3 temp caches whose creating PID is dead. Returns what was removed."""
    base = tmpdir or scratch_root()
    removed: List[str] = []
    try:
        names = os.listdir(base)
    except OSError:
        return removed
    uid = os.getuid()
    for name in names:
        prefix = next((p for p in (PYTEST_PREFIX, PRIVATE_PREFIX) if name.startswith(p)), None)
        if prefix is None:
            continue
        head = name[len(prefix):].split("_", 1)[0]
        if not head.isdigit() or _pid_alive(int(head)):
            continue
        path = os.path.join(base, name)
        try:
            if os.lstat(path).st_uid != uid:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def _cleanup_private() -> None:
    root, pid = _PRIVATE["root"], _PRIVATE["pid"]
    if root and pid == os.getpid():          # only the creator deletes; a forked child never does
        shutil.rmtree(root, ignore_errors=True)


def ensure_hermetic_cache(label: str = "") -> str:
    """Called at EVERY compile site before its first compile. Returns the cache ROOT in force.

    If a run (or pytest's conftest, or a benchmark) declared a cache, it is used. Otherwise this
    process gets a FRESH private temp dir, deleted at exit by the process that created it — never
    torch's shared default. Idempotent."""
    root = os.environ.get(ENV_ROOT)
    ind = os.environ.get(ENV_INDUCTOR)
    if root and ind and not _is_shared(ind):
        os.environ.setdefault(ENV_TRITON, os.path.join(root, "triton"))
        return root
    if ind and not _is_shared(ind) and not root:
        # a caller exported its own Inductor dir (a benchmark's fresh temp dir): adopt it as the root
        os.environ[ENV_ROOT] = ind
        os.environ.setdefault(ENV_TRITON, os.path.join(ind, "triton"))
        return ind
    sweep_stale_temp_caches()
    root = tempfile.mkdtemp(prefix=f"{PRIVATE_PREFIX}{os.getpid()}_", dir=scratch_root())
    _PRIVATE.update(root=root, pid=os.getpid())
    atexit.register(_cleanup_private)
    _export(root)
    print(f"🧊 [CompileCache] PRIVATE {root} for this process{f' ({label})' if label else ''} — "
          f"no run declared a cache; deleted at exit [{SCHEMA}]", file=sys.stderr, flush=True)
    return root


def t2_aot_dir() -> str:
    """A fresh directory for T2's AOTInductor packages, inside the cache root in force."""
    root = ensure_hermetic_cache("t2 aot")
    base = os.path.join(root, "t2_aot")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="pkg_", dir=base)
