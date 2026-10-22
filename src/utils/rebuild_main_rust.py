"""Rebuild MAIN's Rust target after a ship that touched ``src/rust_sim`` / ``src/rust_env``, and PROVE it is current.

``scripts/ops/rebuild_main_rust.sh`` (this module) is step 9's tail in ``/gen3ai-ship``. Since the build stamp
(``utils.rust_env.stamp``, 2308f76d) every ship that touches a Rust crate leaves main's built binaries REFUSED
(``StampMismatch``) until someone rebuilds them; it happened twice on 2026-10-20/21. This does it, safely:

1. REFUSE (exit 3, naming every pid and path) if any live process maps, executes or names an artifact under main's
   ``src/rust_sim/target`` or ``src/rust_env/target`` (``/proc/<pid>/maps`` + ``exe`` + ``cmdline``): a live run's
   ``rust_env_proc`` / ``ab_replay`` must never have its file replaced under it. Also REFUSE a dirty main (4) or a
   target dir that resolves outside main (3).
2. Run the FOUR builds (rust_sim ``release`` and ``selfcheck``, rust_env ``release`` and ``selfcheck``) in MAIN's crate
   dirs with ``CARGO_TARGET_DIR`` UNSET, at ``nice -n 19`` on the agent cores, under ``mem_cap.sh 8``, with
   ``GEN3AI_BLOCKING`` set (rule 5b: a build that blocks the next ship skips the heavy slots).
3. VERIFY every artifact: its ``--build-stamp`` (the cdylib's ``rust_env_stamp()``) is judged by ``stamp.check_stamp``
   against MAIN's tree (sources, column schema, data dir = main's, build kind, self-check flag), and main is still
   git-clean (a build must not leave ``Cargo.lock`` or anything else changed).

Exit status: 0 = built and verified; 2 usage; 3 refused (live users / foreign target); 4 refused (main dirty before)
or left dirty by the build; 5 a build failed or timed out; 6 an artifact failed verification. Nothing is ever forced,
killed or signalled: on a refusal, report and rebuild later.

USAGE (the shell wrapper has the same flags)
    scripts/ops/rebuild_main_rust.sh run            # check, build the four targets, verify
    scripts/ops/rebuild_main_rust.sh --check        # only the refusal checks + verification of what is built now
    scripts/ops/rebuild_main_rust.sh --dry-run      # print the plan and the four commands, run nothing
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from utils.rust_env import columns
from utils.rust_env import stamp as S

#: Rule 5b's agent cores (CPU n and n+8 are SMT siblings): 3 whole physical cores.
AGENT_CPUS = "5-7,13-15"
NICE = 19
MEM_CAP_GB = 8
BLOCKING_WHY = "rebuild main's Rust target after a ship (rebuild_main_rust.sh)"
#: One build must finish inside this (a cold full build of both crates measured well inside it).
BUILD_TIMEOUT_S = 2400.0

EXIT_OK, EXIT_USAGE, EXIT_LIVE, EXIT_DIRTY, EXIT_BUILD, EXIT_VERIFY = 0, 2, 3, 4, 5, 6

#: profile -> the cargo flags that build it into ``target/<profile>/`` (the same strings the loaders print as the fix).
PROFILE_FLAGS: Dict[str, Tuple[str, ...]] = {
    "release": ("--release",),
    "selfcheck": ("--profile", "selfcheck", "--features", "emission-selfcheck"),
}
#: The binaries that answer ``--build-stamp`` (port ``kind=port``; env ``kind=env``). ``ab_replay`` /
#: ``scan_move_probe`` / ``bridge_replay`` carry no stamp: they are built by the port build but cannot be judged.
PORT_STAMPED = ("sim_bridge", "search_driver", "core_events", "live_reader")
ENV_STAMPED = ("rust_env_proc", "bot_reader")
CDYLIB = "libpokesim_env.so"


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str                 # "port" | "env"
    selfcheck: bool           # the profile it was built under
    is_cdylib: bool = False


@dataclass(frozen=True)
class Build:
    crate: str                # "rust_sim" | "rust_env"
    profile: str              # "release" | "selfcheck"
    cwd: Path
    cargo_args: Tuple[str, ...]
    artifacts: Tuple[Artifact, ...]

    @property
    def label(self) -> str:
        return f"{self.crate}/{self.profile}"


def target_dirs(main: Path) -> List[Path]:
    return [main / "src" / "rust_sim" / "target", main / "src" / "rust_env" / "target"]


def plan(main: Path) -> List[Build]:
    """The four builds, in dependency-warming order (the port first: the env crate path-depends on it)."""
    out: List[Build] = []
    for crate, kind in (("rust_sim", "port"), ("rust_env", "env")):
        cwd = main / "src" / crate
        for profile in ("release", "selfcheck"):
            sc = profile == "selfcheck"
            if kind == "port":
                args = ("build", *PROFILE_FLAGS[profile])           # all bins, as `src/rust_sim/CLAUDE.md` § Builds
                names = PORT_STAMPED
            else:
                # bot_reader is its own bin: `build.py`'s `--lib --bin rust_env_proc` alone would leave it stale.
                args = ("build", *PROFILE_FLAGS[profile], "--lib", "--bin", "rust_env_proc", "--bin", "bot_reader")
                names = ENV_STAMPED
            tdir = cwd / "target" / profile
            arts = [Artifact(tdir / n, kind, sc) for n in names]
            if kind == "env":
                arts.append(Artifact(tdir / CDYLIB, kind, sc, is_cdylib=True))
            out.append(Build(crate, profile, cwd, args, tuple(arts)))
    return out


# ------------------------------------------------------------------------------------------ refusals


def _real(p: str) -> str:
    try:
        return os.path.realpath(p)
    except OSError:
        return p


def find_live_users(main: Path, proc: str = "/proc", self_pids: Sequence[int] = (),
                    argv_exempt: Sequence[int] = ()) -> Tuple[List[str], int]:
    """``(lines, unreadable)``: one ``pid comm: <how> <path>`` line per live process that maps, executes or names an
    artifact under main's two ``target/`` dirs, and the count of processes whose ``maps`` could not be read (another
    user's; reported, never silently treated as clear). ``self_pids`` (this process) are never reported; ``argv_exempt`` (our ancestors, whose command line may merely
    MENTION a path) are judged by what they map or run, not by their argv."""
    prefixes = sorted({p for t in target_dirs(main) for p in (str(t), _real(str(t)))})

    def under(path: str) -> bool:
        path = path.removesuffix(" (deleted)")
        return any(path == p or path.startswith(p + "/") for p in prefixes)

    lines: List[str] = []
    unreadable = 0
    skip = set(self_pids)
    no_argv = set(argv_exempt)
    try:
        names = os.listdir(proc)
    except OSError as e:
        raise RuntimeError(f"cannot list {proc}: {e}") from e
    for n in sorted((x for x in names if x.isdigit()), key=int):
        pid = int(n)
        if pid in skip:
            continue
        try:
            with open(f"{proc}/{n}/comm") as f:
                comm = f.read().strip()
        except OSError:
            continue                                                  # gone
        hits: List[str] = []
        try:
            exe = os.readlink(f"{proc}/{n}/exe")
            if under(exe):
                hits.append(f"runs {exe}")
        except OSError:
            pass
        try:
            if pid in no_argv:
                raise OSError("argv of an ancestor is not evidence")
            with open(f"{proc}/{n}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
            for a in argv:
                if under(a) or any(p + "/" in a for p in prefixes):
                    hits.append(f"names {a}")
                    break
        except OSError:
            pass
        try:
            with open(f"{proc}/{n}/maps") as f:
                seen = set()
                for ln in f:
                    parts = ln.split(None, 5)
                    if len(parts) == 6:
                        path = parts[5].strip()
                        if path.startswith("/") and under(path) and path not in seen:
                            seen.add(path)
                            hits.append(f"maps {path}")
        except PermissionError:
            unreadable += 1
        except OSError:
            pass                                                      # gone mid-scan
        for h in hits:
            lines.append(f"{pid} {comm}: {h}")
    return lines, unreadable


def foreign_targets(main: Path) -> List[str]:
    """Target dirs that resolve OUTSIDE main (a symlink into another checkout is the 2026-09-09 incident)."""
    root = _real(str(main))
    bad = []
    for t in target_dirs(main):
        r = _real(str(t))
        if r != root and not r.startswith(root + "/"):
            bad.append(f"{t} -> {r}")
    return bad


def git_porcelain(main: Path, run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> str:
    r = run(["git", "-C", str(main), "status", "--porcelain"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"git status failed in {main}: {(r.stderr or '').strip()}")
    return r.stdout.strip()


# ------------------------------------------------------------------------------------------ building


def cargo_path() -> Optional[str]:
    c = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    return c if os.path.exists(c) else None


def build_env(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The child env: ``CARGO_TARGET_DIR`` UNSET (cargo builds into the crate's own ``target/``), BLOCKING marked."""
    env = dict(os.environ if environ is None else environ)
    env.pop("CARGO_TARGET_DIR", None)
    env["GEN3AI_BLOCKING"] = BLOCKING_WHY
    return env


def build_argv(b: Build, mem_cap: str, cargo: str) -> List[str]:
    """``nice`` and the pin OUTSIDE ``mem_cap.sh`` (a launched nice is inherited; renicing down is refused)."""
    return ["nice", "-n", str(NICE), "taskset", "-c", AGENT_CPUS, mem_cap, str(MEM_CAP_GB), cargo, *b.cargo_args]


def run_builds(builds: Sequence[Build], mem_cap: str, cargo: str, *,
               run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
               emit: Callable[[str], None] = print, environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Run the builds in order; ``None`` when all succeeded, else a one-line reason (the rest are not attempted)."""
    env = build_env(environ)
    for b in builds:
        argv = build_argv(b, mem_cap, cargo)
        emit(f"[rebuild_main_rust] {b.label}: (cd {b.cwd} && {' '.join(argv)})")
        try:
            r = run(argv, cwd=str(b.cwd), env=env, timeout=BUILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"{b.label}: the build did not finish in {BUILD_TIMEOUT_S:.0f} s"
        if r.returncode != 0:
            return f"{b.label}: cargo exited {r.returncode}"
    return None


# ------------------------------------------------------------------------------------------ verifying


def read_stamp(a: Artifact, run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> str:
    """What the artifact says about its own build: ``<bin> --build-stamp`` or the cdylib's ``rust_env_stamp()``."""
    if not a.path.is_file():
        raise FileNotFoundError(f"{a.path} is missing")
    if a.is_cdylib:
        lib = ctypes.CDLL(str(a.path))
        lib.rust_env_stamp.restype = ctypes.c_char_p
        raw = lib.rust_env_stamp()
        return (raw or b"").decode()
    r = run([str(a.path), S.BUILD_STAMP_FLAG], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("stamp=")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"{a.path} {S.BUILD_STAMP_FLAG} gave no stamp (exit {r.returncode})")
    return lines[0]


def verify_artifact(a: Artifact, main: Path, stamp_text: str) -> Optional[str]:
    """``None`` when ``stamp_text`` is main's build of this artifact, else the one-line reason."""
    try:
        st = S.check_stamp(stamp_text, str(a.path), src=main / "src", kind=a.kind, allow_stale=False,
                           nan_poison=a.selfcheck if a.kind == "env" else None,
                           fix="scripts/ops/rebuild_main_rust.sh run")
    except S.StampMismatch as e:
        return str(e)
    if a.kind == "port" and st.get("selfcheck") != str(int(a.selfcheck)):
        return f"{a.path}: selfcheck={st.get('selfcheck')} but it lives in the {'selfcheck' if a.selfcheck else 'release'} profile"
    if a.kind == "env":
        reads, how = S.runtime_data_dir(st.get("data", ""), a.path)
        want = str((main / "data" / "pokemon").resolve())
        if reads != want:
            return f"{a.path}: reads data {reads} ({how}), main's is {want}"
    return None


def verify(builds: Sequence[Build], main: Path, *,
           reader: Callable[[Artifact], str] = read_stamp) -> List[str]:
    """Every problem found (empty = every artifact of every build is main's current build)."""
    problems: List[str] = []
    for b in builds:
        for a in b.artifacts:
            try:
                text = reader(a)
            except Exception as e:                                            # noqa: BLE001 - reported, not raised
                problems.append(f"{b.label}: {e}")
                continue
            why = verify_artifact(a, main, text)
            if why:
                problems.append(f"{b.label}: {why}")
    return problems


# ------------------------------------------------------------------------------------------ CLI


def _ancestors(proc: str) -> List[int]:
    """This process's ppid chain, from ``<proc>/<pid>/stat`` (empty if unreadable)."""
    out: List[int] = []
    pid = os.getppid()
    while pid > 1 and pid not in out:
        out.append(pid)
        try:
            with open(f"{proc}/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def main(argv: Optional[Sequence[str]] = None, *, proc: str = "/proc",
         run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
         reader: Callable[[Artifact], str] = read_stamp) -> int:
    ap = argparse.ArgumentParser(prog="rebuild_main_rust", description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", nargs="?", choices=("run",), help="build the four targets, then verify")
    ap.add_argument("--check", action="store_true", help="refusal checks + verification only, no build")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--main", required=True, help="the MAIN checkout (the shell wrapper derives it)")
    ap.add_argument("--mem-cap", required=True, help="path of scripts/ops/mem_cap.sh")
    ns = ap.parse_args(argv)
    if not (ns.mode or ns.check or ns.dry_run):
        print("REFUSING: give `run`, `--check` or `--dry-run` (a rebuild is never started by accident)", file=sys.stderr)
        return EXIT_USAGE
    main_dir = Path(ns.main).resolve()
    builds = plan(main_dir)

    if ns.dry_run:
        cargo = cargo_path() or "cargo"
        print(f"main = {main_dir}; env: CARGO_TARGET_DIR unset, GEN3AI_BLOCKING={BLOCKING_WHY!r}")
        for b in builds:
            print(f"{b.label}: (cd {b.cwd} && {' '.join(build_argv(b, ns.mem_cap, cargo))})")
            for a in b.artifacts:
                print(f"    verify {a.path}")
        return EXIT_OK

    bad = foreign_targets(main_dir)
    if bad:
        print("REFUSING: a target dir resolves outside main (the 2026-09-09 rust-target incident): " + "; ".join(bad),
              file=sys.stderr)
        return EXIT_LIVE
    users, unreadable = find_live_users(main_dir, proc, self_pids=(os.getpid(),),
                                        argv_exempt=_ancestors(proc))
    if unreadable:
        print(f"[rebuild_main_rust] FINDING: {unreadable} process(es) had unreadable /proc/<pid>/maps (not ours); "
              "they are not covered by the live-user check", file=sys.stderr)
    if users:
        print("REFUSING: live processes use main's Rust target; nothing was built. Rebuild when they end "
              "(never kill them):\n  " + "\n  ".join(users), file=sys.stderr)
        return EXIT_LIVE
    try:
        dirty = git_porcelain(main_dir, run)
    except RuntimeError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return EXIT_DIRTY
    if dirty:
        print("REFUSING: main is not git-clean (a stamp over uncommitted sources proves nothing):\n" + dirty,
              file=sys.stderr)
        return EXIT_DIRTY

    if not ns.check:
        cargo = cargo_path()
        if cargo is None:
            print("REFUSING: cargo not found (PATH or ~/.cargo/bin)", file=sys.stderr)
            return EXIT_BUILD
        why = run_builds(builds, ns.mem_cap, cargo, run=run)
        if why:
            print(f"BUILD FAILED: {why}", file=sys.stderr)
            return EXIT_BUILD

    problems = verify(builds, main_dir, reader=reader)
    if problems:
        print("VERIFICATION FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return EXIT_VERIFY
    try:
        dirty = git_porcelain(main_dir, run)
    except RuntimeError as e:
        print(f"VERIFICATION FAILED: {e}", file=sys.stderr)
        return EXIT_DIRTY
    if dirty:
        print("VERIFICATION FAILED: main is not git-clean after the build:\n" + dirty, file=sys.stderr)
        return EXIT_DIRTY
    n = sum(len(b.artifacts) for b in builds)
    print(f"[rebuild_main_rust] OK: {len(builds)} builds, {n} artifacts match main's tree ({columns.schema_id()}), main is clean")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
