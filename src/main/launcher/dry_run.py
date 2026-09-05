"""``--dry-run`` — resolve a launch completely, print it, and touch NOTHING.

WHY IT EXISTS (2026-09-05 incident). To check that a same-run RESTART with a larger ``--steps``
would still launch, a session "dry launched" the real command and killed it a few seconds later,
after the startup lines. A FORK dry-launched that way is harmless — it writes a NEW directory —
but a RESTART operates on the REAL run directory, and those few seconds were enough to write
``final_model_interrupted.zip``/``.json``, repoint ``latest.txt`` at that phantom artifact,
overwrite ``metadata.json`` (whose ``steps`` became a target that never ran) and
``model_config.json``, and leave ``.compile_quorum`` files behind. **"Dry" was a property of forks,
never of the launcher.** So it is a property of the launcher now.

WHAT IT DOES. Everything the launcher resolves *before* a child exists — argv parse,
fork-vs-restart classification, the run dir the argv would write into, the pin decision, the
effective config a ``--model`` inherits, and the combination checks — then prints one
startup-shaped block and exits. Every refusal the real path makes, it makes.

WHAT IT MUST NEVER DO, and how that is enforced. No run dir is created or modified, no worktree is
created, the startup prune does not run, no child is spawned, nothing is written to
``metadata.json`` / ``latest.txt`` / ``model_config.json``, and no environment is exported. This is
enforced BY CONSTRUCTION — this module calls only pure resolvers (``resolve_launch_run_dir``,
``resolve_fork_resume_model``, ``resolve_pin``) and never reaches ``_create_run_worktree``,
``_prune_stale_launcher_worktrees`` or ``_launch_child`` — and it is PROVEN by
``dry_run_test.py``, which sha256s every file in a fake run dir before and after a same-run
restart dry-run and asserts byte-identity, plus an unchanged ``git worktree list``.

WHAT IT CANNOT KNOW. A line the child could only print after importing torch and building a model
(the architecture-compatibility verdict, the resolved compile flags, the pool seeding, the obs
dim) is printed as ``(child-only: …)`` rather than guessed at. A dry run that invented those would
be worse than one that admits the gap.

It is the EXECUTING complement to ``python -m main.checkargs``: that one answers "do these flags
still parse and cohere?" from an argv; this one answers "what would this exact command do, on this
box, right now?" — and reuses ``checkargs`` for the flag half rather than re-implementing it.
"""

from __future__ import annotations

import os
import time
from typing import Callable, List, Optional

from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import (
    child_uses_bridge,
    _find_model_arg,
    _insert_or_replace_model_arg,
    _insert_or_replace_run_dir_arg,
    _peek_arg,
    resolve_launch_run_dir,
    resolve_fork_resume_model,
    run_dir_for_checkpoint,
)
from main.launcher.child import PYTHON_ENV_VAR, resolve_child_python
from main.launcher.pinned_argv import (
    ParseReport,
    differs_from_head,
    pinned_parser_check,
    refuses,
    report_lines,
)
from main.launcher.worktree import (
    PinRefused,
    _commit_subject,
    _git_hash,
    _read_checkpoint_field,
    get_repo_root,
    resolve_pin,
)
from main.train.fork_lr import is_same_run_checkpoint


#: The flags whose value the combination checks read, plus the two step-size pins. These are the
#: ones where an INHERITED value has actually killed a launch (C1: an inherited
#: ``distill_target="action"`` beside ``--distill-coef 0``), so the block names them explicitly
#: rather than leaving the reader to diff two JSON files.
REPORTED_DESTS = (
    "distill_teacher",
    "distill_target",
    "distill_coef",
    "distill_topk",
    "grad_accum_steps",
    "fork_lr",
    "fork_lr_freeze",
)

#: Lines the dry run structurally cannot compute: each needs torch, a built model, or a spawned
#: child. Printed as gaps rather than guessed at.
CHILD_ONLY = (
    "architecture compatibility (check_compatible against the checkpoint's model_config.json)",
    "the ModelVersion round-trip smoke test",
    "the resolved --compile-trainer / --compile-opponents decisions (they read the device)",
    "self-play pool SEEDING from a fork parent (agents.training.pool_seed runs in the child)",
    "the observation dimension and arch signature",
)


def _fmt_int(n: Optional[int]) -> str:
    return f"{n:,}" if isinstance(n, int) else "unknown"


def _checkpoint_steps(model_path: str) -> "tuple[Optional[int], str]":
    """``(num_timesteps, where it came from)`` for a checkpoint — sidecar first, then the zip.

    The sidecar (``checkpoint_N.json`` beside the zip, then ``snapshot_history``, then the
    run-level value) is the launcher's own reader, so the dry run agrees with what the resume
    contract reads. A checkpoint whose sidecar was groomed away still names its step inside the
    SB3 zip's plain-JSON ``data`` member, read WITHOUT importing torch."""
    try:
        val = _read_checkpoint_field(model_path, key="num_timesteps", toplevel_key="num_timesteps")
    except Exception:                              # noqa: BLE001 — a dry run never crashes on IO
        val = None
    if isinstance(val, (int, float)):
        return int(val), "sidecar"
    try:
        from agents.training.lineage import checkpoint_num_timesteps
        val = checkpoint_num_timesteps(model_path)
    except Exception:                              # noqa: BLE001
        val = None
    return (int(val), "checkpoint zip") if isinstance(val, (int, float)) else (None, "not recorded")


def _pool_line(run_dir: str, child_args: List[str]) -> Optional[str]:
    """``N snapshots … win_rate_vs_bots …`` for the pool this launch would use, or None.

    Pool drift is invisible until a run is already training against the wrong opponents (an empty
    pool does not disable ``--self-play`` — it silently falls back to the BOT pool), so it is worth
    a line BEFORE launching. Read-only: a glob and two file reads."""
    from agents.training import pool_seed

    snapshot_dir = _peek_arg(child_args, "--snapshot-dir")
    pool_dir = str(snapshot_dir) if snapshot_dir else os.path.join(run_dir, "snapshots")
    if not os.path.isdir(pool_dir):
        return None
    zips = pool_seed.snapshot_zips(pool_dir)
    wr = pool_seed._read_win_rate(pool_dir)
    wr_txt = f"{wr:.3f}" if isinstance(wr, float) else "NO metadata (reads as self_play_fraction=0%)"
    return f"{len(zips)} snapshot(s) in {pool_dir}, win_rate_vs_bots {wr_txt}"


def _effective_namespace(child_args: List[str]) -> dict:
    """The effective config a launch would build, via ``main.checkargs`` — never re-implemented.

    ``checkargs.check`` parses the argv with the REAL trainer parser, overlays the parent
    checkpoint's recorded ``model_config.json`` through ``config.inherit_saved_flag`` (the launch
    path's own function), and runs both combination-check families on the result. Reusing it is the
    point: two implementations of "what does this argv actually resolve to" is exactly the drift
    that cost C1 a launch."""
    from main.checkargs import check
    res = check(child_args)
    if res.get("resolution") is None or res["resolution"].get("ns") is None:
        # No `--model` (nothing to inherit) or an argv that does not parse. Parse it here so the
        # flag report still has values to show; an unparseable argv yields ns=None and the caller
        # reports the parse failure instead.
        try:
            import contextlib
            import io

            from main.train_rl_agent import build_parser
            # argparse dumps the WHOLE usage block on a parse failure; swallow it — the refusal is
            # reported below, and 200 lines of flag table in the middle of a dry run is noise.
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                ns, _rest = build_parser().parse_known_args(child_args)
        except SystemExit:
            ns = None
        res["resolution"] = {"ns": ns, "inherited": {}, "config_path": None,
                             "tried": [], "same_run": False, "model": None}
    return res


def dry_run(
    child_args: List[str],
    *,
    interval_hours: float,
    pin: bool,
    sync_to_main: bool,
    pin_commit: Optional[str],
    grace_minutes: float,
    max_crash_restarts: int,
    nice: int,
    out: Callable[[str], None] = print,
) -> int:
    """Resolve this launch, print the block, return the exit code the launch would leave with.

    ``0`` = it would launch. ``FATAL_CONFIG`` (3) = a refusal the real path also makes (an
    unresolvable ``--pin-commit``, a restart that would move the pin, a refused flag combination, a
    flag the parser no longer knows). ``1`` = the run-dir resolution itself refused, matching
    ``_prepare_session``'s own exit for that case.

    Creates nothing, modifies nothing, spawns nothing — see the module docstring.
    """
    fatal = int(TrainExitCode.FATAL_CONFIG)
    out("")
    out("── launcher --dry-run — resolving only, nothing will be created ────────────")

    # 1. Run dir — the same pure resolver the launch path calls, WITHOUT the makedirs that follows
    #    it there. Fresh / fork / plain resume, all three.
    try:
        run_dir = resolve_launch_run_dir(child_args, time.strftime("%Y%m%d_%H%M%S"))
    except ValueError as e:
        out(f"  ✗ REFUSED: {e}")
        return 1

    # 2. The idempotent-fork swap, exactly as `_prepare_session` does it and for the same reason:
    #    once a fork has its own progress, the relaunch is a RESTART of the fork, and the pin guard
    #    below must be checked against the fork's own checkpoint.
    fork_resume = resolve_fork_resume_model(child_args, run_dir)
    if fork_resume is not None:
        child_args = _insert_or_replace_model_arg(child_args, fork_resume)

    # The child sees the run dir as an argument; the effective-config resolution below reads it to
    # tell a same-run RESTART checkpoint from a FORK PARENT, so inject it here too.
    child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)

    model_path = _find_model_arg(child_args)
    if not model_path:
        role = "FRESH"
    elif is_same_run_checkpoint(model_path, run_dir):
        role = f"RESTART of {run_dir}"
    else:
        role = f"FORK of {run_dir_for_checkpoint(model_path)}"

    out(f"  role        : {role}")
    if fork_resume is not None:
        out(f"                ♻️  this fork already has progress — --model swapped to "
            f"{os.path.basename(fork_resume)} (idempotent)")
    exists = os.path.isdir(run_dir)
    out(f"  run dir     : {run_dir}"
        f"   [{'EXISTS — a real launch WRITES INTO IT' if exists else 'would be created'}]")
    if model_path:
        out(f"  --model     : {model_path}")

    # 3. The pin decision — the real `resolve_pin`, so every refusal it makes, this makes.
    pinned: Optional[ParseReport] = None
    if pin:
        try:
            decision = resolve_pin(
                model_path=model_path,
                run_dir=run_dir,
                pin_commit=pin_commit,
                sync_to_main=sync_to_main,
                repo_root=get_repo_root(),
            )
        except PinRefused as e:
            out(f"  ✗ REFUSED (pin): {e}")
            return e.exit_code
        # `resolve_pin` fills `subject` only for a commit it resolved itself (--pin-commit);
        # look it up for the other three sources too, because "which commit" is only useful
        # to a human alongside "which change".
        subject = decision.subject or _commit_subject(decision.sha, repo_root=get_repo_root())
        out(f"  pin         : {decision.sha}  (source: {decision.source})")
        if subject:
            out(f"                ↳ {subject}")
        out("                (a real launch would create an isolated worktree at this commit — "
            "the dry run does not)")
        # 3b. AN ARGV IS VALIDATED BY THE PARSER OF THE TREE THAT WILL RUN IT. When the pin is
        #     not HEAD, the flag checks below read a parser the child will never use — which is
        #     how `--pin-commit b13b30b2` came to be refused for a flag that b13b30b2 accepts.
        if differs_from_head(decision.sha, get_repo_root()):
            pinned = pinned_parser_check(decision.sha, child_args, get_repo_root())
            for line in report_lines(pinned, child_args):
                out(f"  {line}")
    else:
        out(f"  pin         : --no-pin — would run from the current tree ({_git_hash()})")

    # 4. Steps: the total the argv asks for, beside where the checkpoint actually is, so "+X steps"
    #    is visible rather than inferred. This is the exact question the 2026-09-05 dry launch was
    #    asked to answer.
    steps = _peek_arg(child_args, "--steps", type_=int)
    if model_path:
        ckpt_steps, src = _checkpoint_steps(model_path)
        delta = (f" → +{_fmt_int(steps - ckpt_steps)} steps"
                 if isinstance(steps, int) and isinstance(ckpt_steps, int) else "")
        out(f"  steps       : --steps {_fmt_int(steps)} vs checkpoint at "
            f"{_fmt_int(ckpt_steps)} ({src}){delta}")
    else:
        out(f"  steps       : --steps {_fmt_int(steps)} (fresh run, from 0)")

    # 5. The operational lines the launcher itself announces at startup.
    py = resolve_child_python()
    py_pinned = (f" (pinned by ${PYTHON_ENV_VAR})"
                 if os.environ.get(PYTHON_ENV_VAR, "").strip() else "")
    out(f"  interpreter : {py}{py_pinned}")
    if child_uses_bridge(child_args):
        impl = _peek_arg(child_args, "--use-bridge") or "rust"
        out(f"  transport   : in-process bridge [{impl}] (no Showdown server)")
    else:
        out(f"  transport   : websocket → Showdown :{_peek_arg(child_args, '--showdown-port', int)}")
    sched = (f"every {interval_hours:.1f}h" if interval_hours > 0 else "single run (no restart)")
    out(f"  restarts    : {sched}, grace {grace_minutes:.1f} min, "
        f"max {max_crash_restarts} crash restart(s), nice {nice}")

    # 6. The effective config — argv OVERLAID on the parent's recorded model_config.json. An argv
    #    is not a config, and the flags below are where that bites.
    res = _effective_namespace(child_args)
    resolution = res["resolution"]
    ns, inherited = resolution.get("ns"), resolution.get("inherited", {})
    if resolution.get("config_path"):
        out(f"  effective   : argv overlaid on {resolution['config_path']} "
            f"({len(inherited)} unset flag(s) INHERITED)")
    elif model_path:
        out("  ⚠️  effective : could NOT read the parent's model_config.json — falling back to "
            "ARGV-ONLY, which cannot see an inherited value")
        for path in resolution.get("tried", []):
            out(f"                  tried: {path}")
    else:
        out("  effective   : the argv IS the config (no --model, nothing to inherit)")
    if ns is not None:
        for dest in REPORTED_DESTS:
            if not hasattr(ns, dest):
                continue
            where = "INHERITED" if dest in inherited else "from the argv"
            out(f"      --{dest.replace('_', '-'):<20} {getattr(ns, dest)!r:<12} ({where})")

    # 7. Pool drift, when there is a directory to look in.
    if exists:
        pool = _pool_line(run_dir, child_args)
        out(f"  pool        : {pool}" if pool else "  pool        : no snapshots/ directory yet")

    for line in CHILD_ONLY:
        out(f"  (child-only: {line})")

    # 8. The refusals. Same three families `main.checkargs` reports, on the same resolved namespace
    #    — but read against the CURRENT tree. When the pin names another commit AND we managed to
    #    ask that commit's parser (3b), these are ADVISORY: they describe rules the child will not
    #    run under, and treating them as refusals is exactly what made `--pin-commit` unusable.
    advisory = pinned is not None and pinned.available
    mark = "ℹ️  advisory (CURRENT tree, NOT the pinned one)" if advisory else "✗ REFUSED"
    failed = False
    if res["unknown"]:
        failed = failed or not advisory
        out(f"  {mark}: flags the CURRENT trainer parser does not know —")
        for flag, vals in res["unknown"]:
            out(f"      {flag}{' ' + ' '.join(vals) if vals else ''}")
    for flag, dep, token in res["unsatisfiable"]:
        failed = failed or not advisory
        out(f"  {mark} (extractor): {flag} requires {dep}, but the config says `{token}`")
    for combo, provenance in res["combinations"]:
        failed = failed or not advisory
        out(f"  {mark} (resolve_config): {combo.message}")
        for line in provenance:
            out(f"      · {line}")
    # Only the PARSER half is pinned. The extractor's `requires` graph and the value-conditional
    # refusals still come from the current tree, so say so rather than imply a full pinned check.
    if advisory and (res["unknown"] or res["unsatisfiable"] or res["combinations"]):
        out("      (only the PARSER is read from the pinned commit; the extractor + "
            "resolve_config rules above are this tree's)")
    # The pinned parser IS the child's parser, so its refusal is the launch's refusal — whether
    # it could not parse the argv, or the argv names a flag that exists only in THIS tree.
    if pinned is not None and refuses(pinned, child_args):
        failed = True

    if failed:
        out("  ✗ DRY RUN — this command would NOT launch. Nothing was created or modified.")
        return fatal
    out("  ✓ DRY RUN — this command would launch. Nothing was created or modified.")
    return 0
