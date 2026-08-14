"""Checkpoint discovery and CLI argument manipulation for the launcher."""

import argparse
import glob
import os
import re


# The subdir resumable training checkpoints live in (current layout): <run>/checkpoints/.
# Legacy runs kept them directly in <run>/; both are discovered.
CHECKPOINTS_DIRNAME = "checkpoints"


# Directories that hold *.zip files which are NOT resumable training checkpoints:
#   <run>/snapshots/snapshot_*.zip        self-play pool (the step-0 seed is written
#                                         at startup, before any rollout)
#   <run>/best_model/best_model.zip       best-by-eval export
#   <run>/eval_traces/step_*/snapshot.zip retained eval snapshots
# Resumable checkpoints live in <run>/checkpoints/ (current) or <run>/ (legacy) — note
# that ``checkpoints`` is deliberately NOT in this set, so those zips ARE resumable; the
# caller derives run_dir via ``run_dir_for_checkpoint`` (which strips a trailing
# checkpoints/). Returning one of the nested ARTIFACT dirs would mis-derive run_dir — a
# naive dirname(checkpoint) would become e.g. <run>/snapshots, and the relaunched child
# would then nest its own snapshots/ and write checkpoints into the pool. It would also
# let a crash *before* the first real checkpoint silently "resume" from the step-0 seed
# instead of surfacing as the fatal no-checkpoint case it is.
_ARTIFACT_DIRS = {"snapshots", "best_model", "eval_traces"}


def _is_resumable_checkpoint(path: str, models_root: str) -> bool:
    """True unless ``path`` lives under one of the non-checkpoint artifact dirs."""
    rel = os.path.relpath(path, models_root)
    dir_parts = rel.split(os.sep)[:-1]  # directories between models_root and the file
    return _ARTIFACT_DIRS.isdisjoint(dir_parts)


def run_dir_for_checkpoint(checkpoint_path: str) -> str:
    """Return the RUN dir that owns a checkpoint .zip.

    A checkpoint lives at ``<run>/checkpoints/<name>.zip`` (current layout) or
    ``<run>/<name>.zip`` (legacy / the final-model singletons). The run dir is the
    checkpoint's parent — UNLESS that parent is the ``checkpoints/`` subdir, in which
    case the run dir is one level up. Use this everywhere run_dir is derived from a
    checkpoint path, so the TUI 🗂 badge / run-dir arg / metadata.json lookup stay at
    the run root instead of pointing inside ``checkpoints/``.
    """
    d = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(d) == CHECKPOINTS_DIRNAME:
        d = os.path.dirname(d)
    return d


def find_latest_checkpoint(
    models_root: str,
    run_dir: "str | None" = None,
    min_mtime: float = 0.0,
) -> "str | None":
    """Latest resumable checkpoint. STRICTLY RUN-SCOPED when ``run_dir`` is given: latest.txt →
    a glob under ``run_dir`` only → ``None``. It must NEVER fall back to a global ``models/``
    search for a scoped caller — a fresh run that crashes before its first checkpoint has nothing
    to resume (a fatal condition to surface), and the global step-number max used to resolve to
    whatever ancient run had the biggest step anywhere (observed: a crashed fresh run's exit
    summary pointing at a ``models/_goldens/ai_v3_*`` zip at 469M steps). ``run_dir=None`` keeps
    the global search (the legacy un-scoped callers)."""
    if run_dir:
        latest_txt = os.path.join(run_dir, "latest.txt")
        if os.path.exists(latest_txt):
            with open(latest_txt) as f:
                name = f.read().strip()
            # ``name`` is run-relative: a current-layout checkpoint is
            # "checkpoints/checkpoint_<N>_steps.zip", a legacy one or a final-model
            # singleton is a bare basename. os.path.join handles both.
            candidate = os.path.join(run_dir, name)
            if os.path.exists(candidate):
                return candidate
        # No latest.txt (or stale) → search THIS run only. The artifact-dir filter still
        # applies (snapshots/best_model/eval_traces zips are not resumable checkpoints).
        search_root = run_dir
    else:
        search_root = models_root

    zips = glob.glob(os.path.join(search_root, "**", "*.zip"), recursive=True)
    zips = [p for p in zips if _is_resumable_checkpoint(p, search_root)]
    if min_mtime:
        zips = [p for p in zips if os.path.getmtime(p) >= min_mtime]
    if not zips:
        return None

    def _step_key(path: str) -> int:
        n = os.path.basename(path)
        m = re.search(r"(\d+)_steps\.zip$", n)
        if m:
            return int(m.group(1))
        m = re.search(r"forced_(\d+)_", n)
        if m:
            return int(m.group(1))
        return 0

    return max(zips, key=lambda p: (_step_key(p), os.path.getmtime(p)))


class _SilentParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of printing to stderr / sys.exit."""

    def error(self, message):  # noqa: D102
        raise ValueError(message)

    def exit(self, status=0, message=None):  # noqa: D102
        raise ValueError(message or "")


def _peek_arg(args: list, name: str, type_=str):
    """Read a single optional arg's value, accepting both '--x v' and '--x=v'.

    Returns None when the arg is absent or its value fails type conversion.
    Uses argparse so the launcher's view of a flag matches the child's.
    """
    parser = _SilentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(name, dest="value", type=type_, default=None)
    try:
        known, _ = parser.parse_known_args(args)
    except ValueError:
        return None
    return known.value


def _set_arg(args: list, name: str, value: str) -> list:
    """Replace name's value in place; append '--name value' if absent.

    Handles both the '--name value' and '--name=value' spellings so a flag is
    never duplicated when the user passed the combined form.
    """
    out = []
    i = 0
    replaced = False
    while i < len(args):
        a = args[i]
        if a == name:
            out.extend([name, value])
            replaced = True
            i += 2
        elif a.startswith(name + "="):
            out.extend([name, value])
            replaced = True
            i += 1
        else:
            out.append(a)
            i += 1
    if not replaced:
        out.extend([name, value])
    return out


# The dedicated, stable training Showdown server. train_rl_agent.py defaults to the
# shared dev server on 8000 when --showdown-port is omitted — fine for an ad-hoc run,
# but a long launcher session pointed at 8000 dies whenever routine dev-server churn
# (a restart, `npm run stop`) drops every worker's connection at once. The launcher
# isolates training onto its own port by default.
DEFAULT_TRAINING_SHOWDOWN_PORT = 8001


def child_uses_bridge(args: list) -> bool:
    """True when the child args enable the in-process bridge transport (no Showdown server).

    Covers BOTH the deprecated ``--use-showdown-bridge`` boolean AND the current
    ``--use-bridge {node,rust}`` (value form ``--use-bridge X`` or ``--use-bridge=X``; only
    ``off`` means server). ``--use-bridge`` at its ``off`` default (or absent) is NOT a bridge run."""
    if "--use-showdown-bridge" in args:
        return True
    val = _peek_arg(args, "--use-bridge")
    return val is not None and val != "off"


def _apply_default_showdown_port(
    args: list, default_port: int = DEFAULT_TRAINING_SHOWDOWN_PORT
) -> list:
    """Inject ``--showdown-port <default_port>`` into the child args when the user
    didn't pass one. An explicit ``--showdown-port`` (any spelling) always wins.

    A bridge run (``--use-showdown-bridge`` / ``--use-bridge {node,rust}``) connects to no Showdown
    server at all (training AND eval run in-process), so no default port is injected — a phantom
    port would only mislead the TUI."""
    if child_uses_bridge(args):
        return args
    if _peek_arg(args, "--showdown-port", type_=int) is not None:
        return args
    return _set_arg(args, "--showdown-port", str(default_port))


def _find_model_arg(args: list) -> "str | None":
    return _peek_arg(args, "--model")


def _insert_or_replace_model_arg(args: list, checkpoint: str) -> list:
    return _set_arg(args, "--model", checkpoint)


def _insert_or_replace_run_dir_arg(args: list, run_dir: str) -> list:
    return _set_arg(args, "--run-dir", run_dir)


def _resolve_fresh_run_dir(args: list, timestamp: str) -> str:
    """Run dir for a fresh (no --model) launcher run.

    Honour a user-supplied ``--run-dir`` verbatim (normalized) — the folder the run should
    write into; only mint a timestamped ``models/run_<timestamp>`` when none was given. Without
    the ``--run-dir`` branch the launcher always overwrote the user's folder with a fresh
    timestamp, so the TUI 🗂 badge and every checkpoint landed in the wrong place."""
    user_run_dir = _peek_arg(args, "--run-dir")
    if user_run_dir:
        return os.path.normpath(user_run_dir)
    run_name = _peek_arg(args, "--run-name") or _peek_arg(args, "--run_name")
    if run_name:
        # A memorable name → models/<name>/ (basename-sanitized so it can't path-escape models/).
        return os.path.join("models", os.path.basename(run_name.rstrip("/")) or f"run_{timestamp}")
    return os.path.join("models", f"run_{timestamp}")


def resolve_launch_run_dir(args: list, timestamp: str) -> str:
    """The run dir for a launch, covering all three cases:

    - **fresh** (no ``--model``) → ``_resolve_fresh_run_dir`` (honours ``--run-dir``/``--run-name``).
    - **fork** (a ``--model`` resume WITH an explicit ``--run-name``, or ``--exploiter``) → a fresh
      dir: the ``--model`` is only the INIT (an exploiter trained vs a frozen target, or a named
      experiment forked off a still-running run), so its OWN checkpoints must land in a NEW dir, not
      the checkpoint's source dir (which may be a live run / the exploiter's target). Refuses to fork
      onto an EXISTING run (one with a metadata.json) → ``ValueError``.
    - **plain resume** (a ``--model`` with no fork signal) → the checkpoint's own dir (continue it).

    Pure (no makedirs / no process exit) → unit-tested; the caller does the makedirs + arg injection."""
    existing_model = _find_model_arg(args)
    is_fork = bool(_peek_arg(args, "--run-name") or _peek_arg(args, "--run_name")
                   or "--exploiter" in args)
    if not existing_model:
        return _resolve_fresh_run_dir(args, timestamp)
    if is_fork:
        run_dir = _resolve_fresh_run_dir(args, timestamp)
        # IDEMPOTENT FORK ("copy once from the source, resume in place after"): a fork COPIES the
        # source --model into a NEW named dir on its FIRST launch, but a *re-launch* of that same
        # fork (launcher-process restart / machine reboot / re-running the launch command) must
        # CONTINUE it, not re-copy the source (which would silently discard all the fork's
        # progress). So a fork target that ALREADY holds its own resumable checkpoint is a resume,
        # not a clobber — `resolve_fork_resume_model` swaps --model to that checkpoint in the
        # caller. The clobber guard now fires ONLY when the target exists WITHOUT any resumable
        # checkpoint (a genuine run-name collision, or a fork that crashed before its first save).
        if run_dir != run_dir_for_checkpoint(existing_model) \
                and find_latest_checkpoint("models", run_dir=run_dir) is None \
                and os.path.exists(os.path.join(run_dir, "metadata.json")):
            raise ValueError(f"fork target {run_dir!r} is already a run (has a metadata.json but no "
                             f"resumable checkpoint) — refusing to clobber it; pick a different "
                             f"--run-name")
        return run_dir
    return run_dir_for_checkpoint(existing_model)


def resolve_fork_resume_model(args: list, run_dir: str) -> "str | None":
    """The idempotent-fork resume target, or ``None``.

    When ``run_dir`` is a FORK dir (distinct from the source ``--model``'s own dir) that ALREADY
    holds its own resumable checkpoint — i.e. a prior launch of this same fork made progress — this
    returns that checkpoint so the launch RESUMES the fork IN PLACE instead of re-initialising from
    the source ``--model`` (which would discard the fork's progress on a launcher restart / reboot).
    Returns ``None`` on a fork's FIRST launch (no checkpoint yet → keep the source ``--model`` so it
    copies the source once) and on a plain resume (``run_dir`` IS the model's own dir → the restart
    loop already handles it). This is the one swap that makes a fork "copy once, resume after"."""
    model = _find_model_arg(args)
    if not model or run_dir == run_dir_for_checkpoint(model):
        return None
    return find_latest_checkpoint("models", run_dir=run_dir)


def _strip_launcher_args(argv: list) -> list:
    """Strip launcher-only flags so they are not forwarded to train_rl_agent.py."""
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--restart-interval-hours":
            i += 2
        elif argv[i].startswith("--restart-interval-hours="):
            i += 1
        elif argv[i] == "--restart-grace-minutes":
            i += 2
        elif argv[i].startswith("--restart-grace-minutes="):
            i += 1
        elif argv[i] == "--max-crash-restarts":
            i += 2
        elif argv[i].startswith("--max-crash-restarts="):
            i += 1
        elif argv[i] == "--nice":
            i += 2
        elif argv[i].startswith("--nice="):
            i += 1
        elif argv[i] == "--no-pin":
            i += 1
        elif argv[i] == "--sync-to-main":
            i += 1
        elif argv[i] == "--pin-to-hash":
            i += 2
        elif argv[i].startswith("--pin-to-hash="):
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out
