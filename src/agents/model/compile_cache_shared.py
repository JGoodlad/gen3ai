"""THE SHARED COMPILE CACHE — runs with the SAME key share one Inductor/Triton cache
(gen3_shared_compile_cache_v1; T25, owner approval 2026-10-10).

The owner (2026-10-10): **"Shared cache is fine as long as we check the git hash and the workspace is
clean."** K3 (`agents.model.compile_cache`, `gen3_hermetic_compile_cache_v1`) made every run compile
into its OWN empty cache, because the K1b fault proved Inductor's key can omit a setting that changes
the artifact. This module relaxes "its own" to "one per KEY", and keeps that protection: the key holds
every field of K3's per-run stamp PLUS everything else that can change an artifact without changing
the traced graph.

THE KEY (`shared_fields`; the entry dir is the first 32 hex of its SHA256, and the FULL fields are
written into the entry's `stamp.json` and compared on every open — a mismatch is a MISS, never a hit):

* K3's per-run stamp, verbatim (`compile_cache.cache_stamp`: the code commit, `code_dirty`, torch,
  the interpreter, `compile_config.config_row_hash()` — the K1b row);
* the commit AGAIN, read by `git rev-parse HEAD` in the checkout the code was IMPORTED from
  (`utils.paths.repo_root()`: a pinned child's own worktree, so its PIN), and it must agree with
  `$LAUNCHER_GIT_HASH` when the launcher set one;
* python's full version, triton's version, torch's CUDA version, the GPU(s) — name, compute
  capability, driver — read through `nvidia-smi` (no CUDA context), and `$CUDA_VISIBLE_DEVICES`;
* every compile-relevant environment variable (`TORCHINDUCTOR_*`, `TORCHDYNAMO_*`, `TORCH_COMPILE*`,
  `TRITON_*`, `PYTORCH_*`, `TORCH_CUDA_ARCH_LIST`; the cache-dir variables excluded);
* the RUN's compile surface: the model config (`extractor_arch.arch_toggles_from_args` — every
  version-checked architecture toggle) and the trainer flags that shape what is compiled
  (`COMPILE_ARG_KEYS`: the compile switch, device, batch size, grad checkpointing, T2's backend /
  buckets / lanes / caps). The compile REGIONS themselves are code, so the commit names them.

THE CLEAN RULE (`workspace_state`): a shared entry is read or written only when the checkout the code
was imported from is CLEAN — no tracked change anywhere (staged or not) and no untracked file under
`src/`. Untracked files OUTSIDE `src/` do not matter: the trainer and every process it spawns import
from `src/` alone (`PYTHONPATH=<checkout>/src`), so nothing outside it can enter a compiled graph.
An IGNORED or untracked IMPORTABLE file under `src/` is dirt too (`untracked_importables`: `git status`
never reports an ignored file, and an ignored `src/foo.py` can change behaviour under a "pristine" key —
owner 2026-10-10); `__pycache__` and Cargo `target/` dirs are skipped, reasons at
`IMPORT_SCAN_SKIP_DIRS`. The `deps/pokemon-showdown` submodule
is EXCLUDED: no torch compile reads it (it is the JS simulator; training runs on the Rust core, and
whatever data reaches a model reaches it as tensor VALUES, which are graph inputs, never artifacts),
and the launcher's own pinned worktrees replace it with a symlink to the launching checkout's
submodule — so it is never "at its recorded commit" there, and requiring that would make every pinned
run unshareable. A dirty tree, an unknown commit, a launcher-pin disagreement, an unreadable GPU or
the opt-out falls back to K3's per-run hermetic cache, with ONE loud line saying why.

SAFETY.
* Entry creation is ATOMIC: a `.new-<key>-<pid>-*` dir gets its lock file, stamp and subdirs, then
  one `rename` publishes it; a concurrent creator that loses the rename uses the winner's entry. An
  entry is never visible without its stamp.
* Inside an entry, the writers are Inductor's and Triton's own, and torch 2.8 / triton 3.4 were READ
  (2026-10-10), not assumed: Inductor's FX-graph, AOTAutograd, autotune and generated-source writes go
  through `codecache.write_atomic` (a `.<pid>.<thread>.tmp` file + `rename`); Triton's
  `FileCacheManager.put` writes a `tmp.pid_<pid>_<uuid>` dir + `os.replace`, and a kernel's group
  file last; Inductor's C++ builds (CPU kernels, AOTInductor) run under a cross-process `FileLock`
  in the cache's `locks/` dir — SERIALISED against a concurrent writer, but the linker writes the
  `.so` in place, so a process SIGKILLed mid-link can leave a truncated library that the next process
  fails to LOAD (loud, never a silent wrong kernel; delete the entry). FINDING, recorded in
  `designs/training/compile_flags.md`.
* LIVE USE is a lease: every process that compiles into an entry holds a shared `flock` on its
  `.inuse.lock` for its lifetime (the trainer from `compile_cache.prepare_run_cache`, every child
  from `ensure_hermetic_cache`), released by the kernel when the process dies, SIGKILL included.
* PRUNING runs at startup, after this process holds its own lease: least-recently-used entries first
  (`.last_used`'s mtime, touched on every open), until the root is under its cap
  (`$GEN3AI_COMPILE_SHARED_CAP_GB`, default 20). An entry is removed only under an EXCLUSIVE
  non-blocking `flock` — which no lease holder lets anyone take — and only by renaming it to a
  `.trash-*` dir first, so a process that opened the lock file just before sees its inode vanish from
  the path and retries on a fresh entry. One pruner at a time (`.prune.lock`).

The parity gates are untouched: the learner's startup gate, T2's startup gate and the in-run canary
verify OUTPUTS at every start and at update N whatever the cache served (`compile_cache_shared_test`
pins it at the call level).
"""
from __future__ import annotations

import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA = "gen3_shared_compile_cache_v1"
#: Opt-out: `0` / `off` / `false` / `no` turns sharing off (K3's per-run cache for every run). Unset,
#: `1` / `on` / `true` / `yes` = ON. Any other value is treated as OFF, with the notice saying so.
ENV_OPT = "GEN3AI_SHARED_COMPILE_CACHE"
#: The shared root (default `~/.cache/gen3ai/compile_shared`). A tmpfs/ramfs root is refused.
ENV_DIR = "GEN3AI_COMPILE_SHARED_DIR"
#: The size cap in GiB (default 20 — ~8 entries at K3's measured 2.3-2.4 GB per run).
ENV_CAP = "GEN3AI_COMPILE_SHARED_CAP_GB"
#: Where T2's AOT packages go when the root in force is a shared entry: the RUN's own dir (they are
#: `mkdtemp`-unique, never reused, so a shared entry would only accumulate them).
ENV_T2_AOT = "GEN3AI_T2_AOT_DIR"
DEFAULT_SUBPATH = (".cache", "gen3ai", "compile_shared")
DEFAULT_CAP_GB = 20.0
LOCK_FILE = ".inuse.lock"
STAMP_FILE = "stamp.json"
LAST_USED_FILE = ".last_used"
PRUNE_LOCK = ".prune.lock"
NEW_PREFIX = ".new-"
TRASH_PREFIX = ".trash-"
KEY_HEX = 32
_KEY_RE = re.compile(r"^[0-9a-f]{%d}$" % KEY_HEX)
_OFF = ("0", "off", "false", "no")
_ON = ("", "1", "on", "true", "yes")
SUBMODULE_PATHS = ("deps/pokemon-showdown",)
#: The trainer flags that shape what is COMPILED (argparse dests). Every dest the
#: `compile_cache_shared_test` regex flags must be here or in `NOT_COMPILE_ARG_KEYS`, so a new compile
#: flag forces a decision rather than slipping past the key.
COMPILE_ARG_KEYS = ("batch_size", "compile_trainer", "device", "device_batch", "grad_checkpointing",
                    "t2_backend", "t2_buckets", "t2_lanes", "t2_opponent_bucket_cap")
NOT_COMPILE_ARG_KEYS = ("checkpoint_every_steps",)     # matched by the regex; a save cadence
_ENV_PREFIXES = ("TORCHINDUCTOR_", "TORCHDYNAMO_", "TORCH_COMPILE", "TRITON_", "PYTORCH_",
                 "TORCH_CUDA_ARCH_LIST", "INDUCTOR_")
_ENV_EXCLUDED = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
LEASE_WAIT_S = 30.0

#: Leases this process holds: entry path -> open fd (kept open for the process's lifetime).
_LEASES: Dict[str, int] = {}


class SharedMiss(RuntimeError):
    """An entry exists at the key's path but its stamp is not this key's fields (corrupt, or a key
    derivation bug). Never served; the caller falls back to the per-run cache."""


# --------------------------------------------------------------------------- switches + root
def enabled() -> Tuple[bool, str]:
    raw = os.environ.get(ENV_OPT)
    val = (raw or "").strip().lower()
    if val in _OFF:
        return False, f"${ENV_OPT}={raw} (opted out)"
    if val in _ON:
        return True, "on (default)" if raw is None else f"${ENV_OPT}={raw}"
    return False, f"${ENV_OPT}={raw!r} is not one of {_ON[1:] + _OFF} — treated as OFF"


def shared_root_path() -> str:
    """Where the shared root IS (`$GEN3AI_COMPILE_SHARED_DIR`, else `~/.cache/gen3ai/compile_shared`), absolute,
    WITHOUT creating it — for readers that must change nothing (the disk preflight, `--dry-run`)."""
    root = os.environ.get(ENV_DIR) or os.path.join(os.path.expanduser("~"), *DEFAULT_SUBPATH)
    return os.path.abspath(root)


def shared_root() -> str:
    """`$GEN3AI_COMPILE_SHARED_DIR`, else `~/.cache/gen3ai/compile_shared` (created). Raises
    `OSError` when it cannot be made and `RuntimeError` when it is on a RAM filesystem."""
    from utils.scratch import REFUSED_FS, fs_type
    root = shared_root_path()
    os.makedirs(root, exist_ok=True)
    fs = fs_type(root)
    if fs in REFUSED_FS:
        raise RuntimeError(f"the shared compile root {root!r} is on {fs} (RAM); point ${ENV_DIR} at "
                           f"a real disk")
    return root


def cap_bytes() -> int:
    raw = os.environ.get(ENV_CAP)
    try:
        gb = float(raw) if raw else DEFAULT_CAP_GB
    except ValueError:
        gb = DEFAULT_CAP_GB
    return int(max(gb, 0.0) * (1 << 30))


# --------------------------------------------------------------------------- the checkout
def _git(root: str, *args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL, cwd=root,
                                   timeout=60)


def code_root() -> str:
    """The checkout the running code was IMPORTED from — a pinned child's own worktree."""
    from utils.paths import repo_root
    return str(repo_root())


def head_sha(root: str) -> Optional[str]:
    try:
        return _git(root, "rev-parse", "HEAD").strip() or None
    except Exception:  # noqa: BLE001 — no git / not a checkout: unnameable, never shared
        return None


@dataclasses.dataclass(frozen=True)
class WorkspaceState:
    clean: Optional[bool]          # None = could not be read (never shared)
    detail: str


def _porcelain_paths(out: str) -> List[Tuple[str, str]]:
    rows = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:                         # a rename: judge the destination
            path = path.split(" -> ", 1)[1]
        rows.append((code, path.strip().strip('"')))
    return rows


def workspace_state(root: str) -> WorkspaceState:
    """CLEAN = no tracked change anywhere (staged or not) and no untracked file under `src/`.
    The `deps/pokemon-showdown` submodule is excluded (module docstring says why)."""
    try:
        tracked = _git(root, "status", "--porcelain=v1", "--untracked-files=no",
                       "--ignore-submodules=all")
        untracked = _git(root, "status", "--porcelain=v1", "--untracked-files=all",
                         "--ignore-submodules=all", "--", "src")
    except Exception as exc:  # noqa: BLE001
        return WorkspaceState(None, f"git status failed in {root}: {type(exc).__name__}")
    for code, path in _porcelain_paths(tracked):
        if any(path == s or path.startswith(s + "/") for s in SUBMODULE_PATHS):
            continue
        return WorkspaceState(False, f"tracked change {code.strip()} {path}")
    for code, path in _porcelain_paths(untracked):
        if code == "??":
            return WorkspaceState(False, f"untracked file under src/: {path}")
    try:
        stray = untracked_importables(root)
    except Exception as exc:  # noqa: BLE001
        return WorkspaceState(None, f"the importable-file scan failed in {root}: {type(exc).__name__}")
    if stray:
        return WorkspaceState(False, f"an IGNORED / untracked importable file under src/: {stray[0]}"
                              + (f" (+{len(stray) - 1} more)" if len(stray) > 1 else ""))
    return WorkspaceState(True, "clean")


#: Suffixes Python can IMPORT or that change what an import resolves to (a source module, a byte-compiled module
#: outside the cache dir — a `foo.pyc` beside `foo.py`'s place is imported when no source exists — an extension
#: module, a path-configuration file, Cython sources a pyximport could build).
IMPORTABLE_SUFFIXES = (".py", ".pyc", ".pyo", ".pyd", ".so", ".pth", ".pyx", ".pxd")
#: Directory names the importable scan SKIPS, each with its reason (owner 2026-10-10: the shared cache is fine "as
#: long as we're confident it comes from a hash with no changes"):
#:   `__pycache__` — CPython writes it from the TRACKED source and validates every `.pyc` against that source's
#:      mtime + size before using it (PEP 3147 / 552 timestamp pycs), so a stale one is recompiled, never imported
#:      over a source git already judged clean; a `.pyc` OUTSIDE `__pycache__` is a sourceless module and IS counted;
#:   `target` (a Cargo build dir, `src/rust_*/target/`) — the Rust crates' build output, rebuilt by cargo from the
#:      TRACKED crate sources, and it feeds observations to the model, never code to a torch compile (the same reason
#:      the `deps/pokemon-showdown` submodule is excluded: no input of an Inductor / Triton artifact lives there).
IMPORT_SCAN_SKIP_DIRS = ("__pycache__", "target", ".git")


def untracked_importables(root: str) -> List[str]:
    """Every file under `<root>/src/` with an `IMPORTABLE_SUFFIXES` suffix that git does NOT track — ignored or
    not (`git status` never reports an IGNORED file, so a gitignored `src/foo.py` would otherwise pass as clean).
    Skips `IMPORT_SCAN_SKIP_DIRS` (reasons there). Sorted repo-relative paths."""
    src = os.path.join(root, "src")
    tracked = set(_git(root, "ls-files", "-z", "--", "src").split("\0"))
    out: List[str] = []
    for d, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if x not in IMPORT_SCAN_SKIP_DIRS]
        for f in files:
            if f.endswith(IMPORTABLE_SUFFIXES):
                rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                if rel not in tracked:
                    out.append(rel)
    return sorted(out)


# --------------------------------------------------------------------------- the key
def _gpu_identity() -> Optional[List[str]]:
    """`name, compute_cap, driver_version` per GPU, through nvidia-smi (NVML; no CUDA context).
    `[]` when the box has no NVIDIA tooling; None when it exists but cannot be read."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def platform_fields() -> Dict[str, Any]:
    import torch
    try:
        import triton
        triton_ver: Optional[str] = str(triton.__version__)
    except Exception:  # noqa: BLE001 — a CPU-only torch has none
        triton_ver = None
    return {"python_version": sys.version, "triton": triton_ver,
            "cuda": getattr(torch.version, "cuda", None), "gpu": _gpu_identity(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def torch_env_fields(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ if environ is None else environ
    return {k: env[k] for k in sorted(env)
            if k.startswith(_ENV_PREFIXES) and k not in _ENV_EXCLUDED}


def run_fields(args: Any) -> Dict[str, Any]:
    """The RUN's compile surface: the model config (every version-checked architecture toggle) and
    the trainer flags that shape what is compiled."""
    from agents.model.extractor_arch import arch_toggles_from_args
    return {"arch": arch_toggles_from_args(args),
            "compile_args": {k: getattr(args, k, None) for k in COMPILE_ARG_KEYS}}


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))


def key_of(fields: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical(fields).encode()).hexdigest()[:KEY_HEX]


@dataclasses.dataclass(frozen=True)
class SharedPlan:
    use: bool
    reason: str                      # why not (use=False), or "clean" (use=True)
    fields: Optional[Dict[str, Any]] = None
    key: Optional[str] = None


def plan(stamp: Dict[str, Any], run: Optional[Dict[str, Any]], *, root: Optional[str] = None,
         platform: Optional[Callable[[], Dict[str, Any]]] = None) -> SharedPlan:
    """Decide whether THIS process may use a shared entry, and under which key. Never raises."""
    on, why = enabled()
    if not on:
        return SharedPlan(False, why)
    if run is None:
        return SharedPlan(False, "no run compile surface was given (only a trainer shares)")
    root = root or code_root()
    sha = head_sha(root)
    if not sha:
        return SharedPlan(False, f"the code's checkout {root} has no readable commit")
    if stamp.get("code_sha") not in (sha,):
        return SharedPlan(False, f"the stamp's commit {str(stamp.get('code_sha'))[:12]} is not the "
                                 f"checkout's HEAD {sha[:12]} ({root})")
    pin = (os.environ.get("LAUNCHER_GIT_HASH") or "").strip().lower()
    if pin and not (sha.startswith(pin) if len(pin) < len(sha) else pin == sha):
        return SharedPlan(False, f"$LAUNCHER_GIT_HASH {pin[:12]} disagrees with the imported "
                                 f"checkout's HEAD {sha[:12]}")
    ws = workspace_state(root)
    if ws.clean is not True:
        return SharedPlan(False, f"the workspace is not clean ({ws.detail}) — a dirty tree is not "
                                 f"named by its commit")
    plat = (platform or platform_fields)()
    if plat.get("gpu") is None:
        return SharedPlan(False, "the GPU identity is unreadable (nvidia-smi failed)")
    fields = {"schema": SCHEMA, "k3_stamp": dict(stamp), "code_sha": sha, "workspace": "clean",
              **plat, "torch_env": torch_env_fields(), "run": run}
    # one canonical form: what is hashed is what is written and compared
    fields = json.loads(canonical(fields))
    return SharedPlan(True, "clean", fields, key_of(fields))


# --------------------------------------------------------------------------- entries + leases
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _flock_with_deadline(fd: int, op: int, deadline: float) -> bool:
    while True:
        try:
            fcntl.flock(fd, op | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


@dataclasses.dataclass
class Entry:
    path: str
    key: str
    created: bool
    created_at: str
    fd: int


def _publish(root: str, key: str, fields: Dict[str, Any]) -> bool:
    """Create the entry ATOMICALLY: build it under a `.new-*` name, then one `rename`. Returns True
    when this call published it, False when another creator got there first."""
    path = os.path.join(root, key)
    tmp = tempfile.mkdtemp(prefix=f"{NEW_PREFIX}{key}-{os.getpid()}-", dir=root)
    try:
        open(os.path.join(tmp, LOCK_FILE), "a").close()
        for sub in ("inductor", "triton"):
            os.makedirs(os.path.join(tmp, sub))
        record = {"fields": fields, "key": key, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "created_by_pid": os.getpid()}
        _write_atomic(os.path.join(tmp, STAMP_FILE), json.dumps(record, indent=2, sort_keys=True))
        open(os.path.join(tmp, LAST_USED_FILE), "a").close()
        try:
            os.rename(tmp, path)
            return True
        except OSError as exc:
            if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
                return False
            raise
    finally:
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def acquire(root: str, key: str, fields: Dict[str, Any], *,
            wait_s: float = LEASE_WAIT_S) -> Entry:
    """Open (creating atomically if absent) the entry for `key`, take its LEASE (a shared flock held
    for this process's lifetime) and verify its stamp. Raises `SharedMiss` on a stamp that is not
    `fields`, `TimeoutError` when the lease cannot be taken."""
    path = os.path.join(root, key)
    deadline = time.monotonic() + wait_s
    while True:
        created = False
        if not os.path.isdir(path):
            created = _publish(root, key, fields)
        lock = os.path.join(path, LOCK_FILE)
        try:
            fd = os.open(lock, os.O_RDWR)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"the shared entry {path} kept vanishing (a pruner)")
            continue
        if not _flock_with_deadline(fd, fcntl.LOCK_SH, deadline):
            os.close(fd)
            raise TimeoutError(f"could not take the lease on {path} within {wait_s:.0f}s")
        try:
            same = os.fstat(fd).st_ino == os.stat(lock).st_ino
        except FileNotFoundError:
            same = False
        if not same:                       # pruned between our open and our lock: start over
            os.close(fd)
            continue
        try:
            with open(os.path.join(path, STAMP_FILE)) as f:
                rec = json.load(f)
        except Exception as exc:  # noqa: BLE001
            os.close(fd)
            raise SharedMiss(f"the entry {path} has no readable stamp ({type(exc).__name__}) — "
                             f"delete it") from exc
        if not isinstance(rec, dict) or rec.get("fields") != fields:
            os.close(fd)
            raise SharedMiss(f"the entry {path}'s stamp is not this key's fields — never served")
        try:
            os.utime(os.path.join(path, LAST_USED_FILE))
        except FileNotFoundError:
            open(os.path.join(path, LAST_USED_FILE), "a").close()
        _LEASES[path] = fd
        return Entry(path=path, key=key, created=created,
                     created_at=str(rec.get("created_at")), fd=fd)


def is_shared_entry(root: str) -> bool:
    if not root:
        return False
    parent, name = os.path.split(os.path.abspath(root))
    return bool(_KEY_RE.match(name)) and os.path.isfile(os.path.join(root, LOCK_FILE)) and \
        os.path.isfile(os.path.join(root, STAMP_FILE))


def hold_lease_if_shared(root: str) -> Optional[int]:
    """A CHILD process compiling into a shared entry (it inherited the variables) takes its own
    lease for its lifetime, so the entry outlives the trainer for as long as the child uses it."""
    if not is_shared_entry(root):
        return None
    path = os.path.abspath(root)
    if path in _LEASES:
        return _LEASES[path]
    fd = os.open(os.path.join(path, LOCK_FILE), os.O_RDWR)
    if not _flock_with_deadline(fd, fcntl.LOCK_SH, time.monotonic() + LEASE_WAIT_S):
        os.close(fd)
        raise TimeoutError(f"could not take the lease on the shared compile entry {path}")
    _LEASES[path] = fd
    return fd


def release(path: str) -> None:
    """Drop this process's lease (tests; a process otherwise holds it until it exits)."""
    fd = _LEASES.pop(os.path.abspath(path), None)
    if fd is not None:
        os.close(fd)


# --------------------------------------------------------------------------- pruning
def _dir_bytes(path: str) -> int:
    total = 0
    for dp, _dn, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.lstat(os.path.join(dp, fn)).st_size
            except OSError:
                pass
    return total


@dataclasses.dataclass
class PruneReport:
    ran: bool
    total_before: int = 0
    total_after: int = 0
    cap: int = 0
    removed: List[str] = dataclasses.field(default_factory=list)
    in_use_skipped: List[str] = dataclasses.field(default_factory=list)

    def line(self) -> str:
        if not self.ran:
            return "prune skipped (another pruner is running)"
        gb = 1 << 30
        return (f"{self.total_after / gb:.1f} / {self.cap / gb:.0f} GiB after pruning "
                f"{len(self.removed)} entr{'y' if len(self.removed) == 1 else 'ies'}"
                + (f", {len(self.in_use_skipped)} over-cap in use" if self.in_use_skipped else ""))


def _sweep_orphans(root: str) -> None:
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.startswith(TRASH_PREFIX):
            shutil.rmtree(path, ignore_errors=True)
        elif name.startswith(NEW_PREFIX):
            parts = name[len(NEW_PREFIX):].split("-")
            if len(parts) >= 2 and parts[1].isdigit() and not _pid_alive(int(parts[1])):
                shutil.rmtree(path, ignore_errors=True)


def prune(root: str, cap: int, *, keep: Tuple[str, ...] = ()) -> PruneReport:
    """Remove least-recently-used entries until the root is at or under `cap` bytes. Never an entry
    a live process leases (EXCLUSIVE non-blocking flock), never one in `keep`."""
    plock = os.open(os.path.join(root, PRUNE_LOCK), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(plock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return PruneReport(ran=False, cap=cap)
        _sweep_orphans(root)
        entries = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not _KEY_RE.match(name) or not os.path.isdir(path):
                continue
            try:
                used = os.stat(os.path.join(path, LAST_USED_FILE)).st_mtime
            except OSError:
                used = 0.0
            entries.append((used, name, path, _dir_bytes(path)))
        total = sum(e[3] for e in entries)
        rep = PruneReport(ran=True, total_before=total, cap=cap)
        for _used, name, path, size in sorted(entries):
            if total <= cap:
                break
            if name in keep:
                continue
            lock = os.path.join(path, LOCK_FILE)
            try:
                fd = os.open(lock, os.O_RDWR)
            except FileNotFoundError:          # not a well-formed entry: nobody can lease it
                fd = -1
            try:
                if fd >= 0:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        rep.in_use_skipped.append(path)
                        continue
                    if os.fstat(fd).st_ino != os.stat(lock).st_ino:
                        continue
                trash = tempfile.mkdtemp(prefix=f"{TRASH_PREFIX}{name}-{os.getpid()}-", dir=root)
                os.rmdir(trash)
                os.rename(path, trash)         # atomic: the key path is gone before anything is deleted
            finally:
                if fd >= 0:
                    os.close(fd)
            shutil.rmtree(trash, ignore_errors=True)
            total -= size
            rep.removed.append(path)
        rep.total_after = total
        return rep
    finally:
        os.close(plock)
