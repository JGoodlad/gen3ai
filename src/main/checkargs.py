"""Does this command still launch? — validate an argv against the CURRENT parser, offline.

WHY THIS EXISTS. A run's recorded `launcher_command` outlives the flags inside it. Relaunching
gen-12's argv on v89 died on `--pubval-*`, deleted at v88 — and it died the expensive way:

* **argparse reports only the FIRST unrecognized flag.** Two stale flags mean two launches.
* **The only way to ask was to launch.** Each attempt spun up the launcher, created a run dir,
  crashed the child, and wrote a crash report — ~40 s and a directory to clean up, per flag.
* **`--help` was itself broken** (an unescaped `%` in one help string rendered as a `%o`
  conversion, `TypeError: %o format: an integer is required, not dict`), so there was no offline
  way to enumerate what the parser accepts either.

This reports EVERY problem in one pass, without importing torch, touching `models/`, or starting
anything. It is the "we have a lot of flags to delete one day" tool: after a deletion, run it over
the recorded commands of the runs you might still want to relaunch or fork.

Usage:
  python -m main.checkargs <run_dir>              # the run's recorded launcher_command
  python -m main.checkargs --argv "--steps 1 …"   # a literal argv

Exit 0 = every flag is accepted. Exit 1 = something would fail at launch.

Three ways a command fails, and it reports all three in one pass:
  * a flag the parser no longer knows (the motivating case above);
  * a combination the extractor constructor refuses — `agents.model.flag_registry`'s `requires`
    graph, e.g. `--intent-conditional` without `--damage-outgoing`. That crash is later and more
    expensive than an argparse error: the run dir exists, the child starts, and the traceback comes
    out of `Gen3FeaturesExtractor.__init__`.
  * a combination `resolve_config` refuses — `main.train.combination_checks`, the value-conditional
    rules that are not `requires`-shaped (`--distill-target action` needs `--distill-coef > 0`).

⚠️ AN ARGV IS NOT A CONFIG, and this tool believed it was for three launches. With `--model`, every
flag the argv does NOT name is INHERITED from the checkpoint's recorded `model_config.json`
(`main.train.config`'s `_resolve`). So the thing that launches is the argv OVERLAID ON THE PARENT,
and checking the argv alone is checking a document nobody executes. C1 (2026-09-01) is the third and
sharpest instance: its parent recorded `distill_target="action"`, the argv said `--distill-coef 0`
and named no target, `_resolve` inherited `action`, and the run died at launch — while this tool had
printed "✓ this command still launches".

So when the argv carries `--model`, the checks below run on the EFFECTIVE namespace: argv parsed by
the real parser, then every unset value filled from the parent's recorded config through
`config.inherit_saved_flag` — the launch path's own function, called rather than re-implemented. The
report says, per finding, whether the value came from the command line or was inherited.

WITHOUT a `--model` (or when the parent's config cannot be read — which is a WARNING naming every
path tried, never a silent pass) the dependency half stays deliberately CONSERVATIVE: it fires only
when the argv enables a flag AND explicitly names a dependency with a disabled value. An argv that
simply omits a dependency is not reported, because there is nothing to resolve it against and
absence then carries no information. Under-reporting is the right failure direction: this tool's
value is that a warning from it is worth acting on.

It REPORTS; it does not repair. A deleted flag may have a replacement, so dropping it silently can
change the run in a way nobody sees — naming the problem is the job, deciding what the command
should say is the reader's.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import sys
from typing import Any, Dict, List, Tuple

from main.train.combination_checks import failing_checks

# Flags the LAUNCHER owns and strips before forwarding — absent from the trainer's parser by
# design, so they must not be reported as stale. Sourced from the launcher rather than re-listed
# here would be better; it is a short, stable set and the launcher does not export it.
LAUNCHER_ONLY = {
    "--restart-interval-hours", "--restart-grace-minutes", "--max-crash-restarts",
    "--nice", "--no-pin", "--sync-to-main", "--pin-commit", "--pin-to-hash",
    "--dry-run",
}


def known_option_strings() -> Dict[str, str]:
    """Every option string the CURRENT trainer parser accepts → its dest.

    Reads the parser's own `_actions` rather than scraping `--help`: help text wraps, abbreviates
    and (as this module's docstring records) can fail to render at all, while the actions are the
    parser's actual contract.
    """
    from main.train_rl_agent import build_parser
    out: Dict[str, str] = {}
    for action in build_parser()._actions:
        for opt in action.option_strings:
            out[opt] = action.dest
    return out


def split_argv(argv: List[str]) -> List[Tuple[str, List[str]]]:
    """`['--a','1','--b']` → `[('--a',['1']), ('--b',[])]`. Positionals attach to the flag before
    them, so a stale flag can be reported WITH its value (`--pubval-mode none`) — which is what
    tells the reader whether it mattered."""
    out: List[Tuple[str, List[str]]] = []
    for tok in argv:
        if tok.startswith("--"):
            out.append((tok, []))
        elif out:
            out[-1][1].append(tok)
        else:
            out.append((tok, []))          # a leading positional; kept, never judged
    return out


def _enabled_state(vals: List[str]) -> bool:
    """Is `--flag <vals>` the ON state, by the registry's OFF convention?

    No value = a `store_true` that is present, so ON. One value is read as a number when it looks
    like one (the two derived toggles are set by a COEFFICIENT, where `0` means off), else as the
    mode string it is. Anything longer is not a toggle spelling and is treated as ON rather than
    guessed at.
    """
    from agents.model.flag_registry import is_enabled
    if not vals:
        return True
    if len(vals) > 1:
        return True
    tok = vals[0]
    try:
        return is_enabled(float(tok))
    except ValueError:
        return is_enabled(tok)


def unsatisfiable_pairs(argv: List[str]) -> List[Tuple[str, str, str]]:
    """`(flag, dependency, the disabling token)` for each dependency the argv explicitly negates.

    The registry import is function-local: it is pure data today (VERIFIED — calling this leaves
    `torch` out of `sys.modules`), but it lives under `agents.model`, and this module's promise to
    answer without importing torch should not depend on that staying true by luck.
    """
    from agents.model.flag_registry import BY_NAME, cli_flags

    # CLI spelling -> the registry rows it sets. `--damage-matrices` desugars into two rows, so a
    # dict of lists rather than a dict; a mode flag that grew a third row would just work.
    by_cli: Dict[str, List[str]] = {}
    for f in cli_flags():
        by_cli.setdefault(f.cli_flag, []).append(f.name)

    state: Dict[str, Tuple[bool, str]] = {}      # flag name -> (enabled?, the token that said so)
    for flag, vals in split_argv(argv):
        for name in by_cli.get(flag, ()):
            state[name] = (_enabled_state(vals), " ".join([flag] + vals))

    out: List[Tuple[str, str, str]] = []
    for name, (on, _) in state.items():
        if not on:
            continue
        for dep in BY_NAME[name].requires:
            dep_state = state.get(dep)
            if dep_state is not None and not dep_state[0]:
                out.append((name, dep, dep_state[1]))
    return sorted(out)


def effective_run_dir(ns) -> str | None:
    """Where this argv would WRITE — mirroring `train_rl_agent`'s Directory Setup, in its order.

    `--run-dir` (the launcher-managed resume) wins; else `--run-name` names `models/<name>`; else
    the run dir is freshly minted (a date stamp, or the exploiter default) and `None` says so.

    `None` costs nothing that matters. It only routes the model through the FORK label rather than
    the RESTART one, and both resolve against the SAME file — `_resolve` reads the checkpoint's
    recorded config either way. The distinction is reported, not acted on.
    """
    if getattr(ns, "run_dir", None):
        return str(ns.run_dir)
    if getattr(ns, "run_name", None):
        return os.path.join("models", str(ns.run_name))
    return None


def parent_config_path(model_path: str) -> Tuple[str | None, List[str]]:
    """The `model_config.json` a resume from `model_path` would read, and every path tried.

    Same two-directory search `load_model_snapshot` does — a checkpoint may sit in
    `<run>/checkpoints/` while the run-level config stays at the run root. `_resolve_paths` is
    imported for the zip-candidate ladder (`X`, `X.zip`, `X/final_model.zip`, `X/best_model.zip`)
    and falls back to the literal dirname when it RAISES, because this tool must still answer about
    a command whose checkpoint is not on this box.
    """
    try:
        from agents.model.snapshot import _resolve_paths
        _, cfg_dir = _resolve_paths(model_path)
    except Exception:                                # noqa: BLE001 — no zip here; keep answering
        cfg_dir = os.path.dirname(os.path.abspath(model_path))
    tried: List[str] = []
    for d in (cfg_dir, os.path.dirname(cfg_dir)):
        cand = os.path.join(d, "model_config.json")
        tried.append(cand)
        if os.path.exists(cand):
            return cand, tried
    return None, tried


def resolve_against_parent(argv: List[str]) -> dict | None:
    """The EFFECTIVE namespace a launch would build: argv, then the parent's recorded config.

    Returns None when the argv names no `--model` (nothing to inherit from — the argv IS the
    config). Otherwise a dict carrying the namespace, which flags were inherited and from where,
    and — when the parent's config could not be read — the paths tried, so the caller can WARN
    instead of passing silently.
    """
    if not any(f in ("--model",) for f, _ in split_argv(argv)):
        return None
    from main.train.config import inherit_saved_flag
    from main.train.fork_lr import is_same_run_checkpoint
    from main.train_rl_agent import build_parser

    parser = build_parser()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            ns, _rest = parser.parse_known_args(argv)
    except SystemExit:
        return {"ns": None, "parse_error": buf.getvalue().strip().splitlines()[-1:] or ["?"],
                "model": None, "config_path": None, "tried": [], "inherited": {}, "same_run": False}

    model = getattr(ns, "model", None)
    if not model:
        return None
    run_dir = effective_run_dir(ns)
    same_run = bool(run_dir) and is_same_run_checkpoint(model, run_dir)
    config_path, tried = parent_config_path(model)

    saved = None
    if config_path:
        try:
            from agents.model.model_version import ModelVersion
            saved = ModelVersion.from_json_file(config_path)
        except Exception as e:                       # noqa: BLE001 — unreadable is not a crash
            return {"ns": ns, "model": model, "config_path": config_path, "tried": tried,
                    "inherited": {}, "same_run": same_run, "read_error": str(e)}

    inherited: Dict[str, Any] = {}
    if saved is not None:
        for dest in sorted({a.dest for a in parser._actions} - {"help"}):
            if not hasattr(ns, dest):
                continue
            if inherit_saved_flag(ns, saved, dest, getattr(ns, dest)):
                inherited[dest] = getattr(ns, dest)
    return {"ns": ns, "model": model, "config_path": config_path, "tried": tried,
            "inherited": inherited, "same_run": same_run}


def unsatisfiable_from_namespace(ns) -> List[Tuple[str, str, str]]:
    """`(flag, dependency, the value that disables it)` over a RESOLVED namespace.

    Complete rather than conservative, and the difference is the whole point: on a resolved
    namespace an unset flag is not unknown, it is whatever the parent recorded, so a dependency
    that reads OFF really is off. A row whose args attribute is absent or still `None` is skipped —
    that is a value nothing determined, and unknown is not a verdict.
    """
    from agents.model.flag_registry import BY_NAME, REGISTRY, is_enabled

    missing = object()
    state: Dict[str, Tuple[bool, str]] = {}
    for f in REGISTRY:
        val = getattr(ns, f.arg, missing)
        if val is missing or val is None:
            continue
        state[f.name] = (is_enabled(val), f"{f.cli_flag} = {val!r}")
    out: List[Tuple[str, str, str]] = []
    for name, (on, _) in state.items():
        if not on:
            continue
        for dep in BY_NAME[name].requires:
            dep_state = state.get(dep)
            if dep_state is not None and not dep_state[0]:
                out.append((name, dep, dep_state[1]))
    return sorted(out)


def _provenance(dest: str, ns, inherited: Dict[str, Any]) -> str:
    """`--distill-target action (INHERITED from the parent's model_config.json)` — one finding's
    origin. The word INHERITED is what turns a puzzling refusal into an obvious one."""
    val = getattr(ns, dest, None)
    flag = "--" + dest.replace("_", "-")
    where = "INHERITED from the parent's recorded config" if dest in inherited else "from the argv"
    return f"{flag} {val!r} ({where})"


def check(argv: List[str]) -> dict:
    """Every flag in `argv` classified against the live parser. Pure — unit-testable."""
    known = known_option_strings()
    unknown, launcher, ok = [], [], []
    for flag, vals in split_argv(argv):
        if not flag.startswith("--"):
            continue
        if flag in known:
            ok.append(flag)
        elif flag in LAUNCHER_ONLY:
            launcher.append(flag)
        else:
            unknown.append((flag, vals))

    res = {"n_flags": len(ok) + len(launcher) + len(unknown),
           "accepted": ok, "launcher_only": launcher, "unknown": unknown,
           "unsatisfiable": unsatisfiable_pairs(argv),
           "resolution": None, "combinations": []}
    if unknown:
        # A stale flag makes the effective namespace unbuildable (argparse refuses the argv) and,
        # more to the point, the reader has to fix that first. Report it alone.
        return res

    resolution = resolve_against_parent(argv)
    res["resolution"] = resolution
    if not resolution or resolution.get("ns") is None:
        return res
    ns, inherited = resolution["ns"], resolution["inherited"]
    if resolution.get("config_path") and not resolution.get("read_error"):
        res["unsatisfiable"] = unsatisfiable_from_namespace(ns)
    res["combinations"] = [
        (c, [_provenance(d, ns, inherited) for d in c.dests]) for c in failing_checks(ns)]
    return res


def argv_from_run(run_dir: str) -> List[str]:
    """The run's recorded `launcher_command`, minus the script path.

    A run only grows `metadata.json` at its first save, so a run launched minutes ago legitimately
    has none yet — say that, rather than raising a traceback at someone who asked a reasonable
    question about a live run.
    """
    path = os.path.join(run_dir, "metadata.json")
    if not os.path.isdir(run_dir):
        raise SystemExit(f"no such run dir: {run_dir}")
    if not os.path.exists(path):
        raise SystemExit(
            f"{run_dir} has no metadata.json yet — a run writes it at its first save, so a "
            f"just-launched run has nothing recorded to check. Use --argv to check the command "
            f"you are about to launch instead.")
    with open(path) as fh:
        meta = json.load(fh)
    cmd = meta.get("launcher_command") or meta.get("original_command")
    if not cmd:
        raise SystemExit(f"{run_dir}/metadata.json records no launcher_command/original_command")
    parts = shlex.split(cmd)
    return parts[1:] if parts and not parts[0].startswith("--") else parts


def main(raw: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="a models/<run> dir whose command to validate")
    ap.add_argument("--argv", help="a literal argv string to validate instead")
    a = ap.parse_args(raw)
    if not a.run_dir and not a.argv:
        ap.error("give a run_dir or --argv")

    argv = shlex.split(a.argv) if a.argv else argv_from_run(a.run_dir)
    res = check(argv)

    src = a.run_dir or "--argv"
    print(f"checked {res['n_flags']} flags from {src}")
    print(f"  accepted by the trainer parser : {len(res['accepted'])}")
    print(f"  launcher-owned (not forwarded) : {len(res['launcher_only'])}")
    if res["unknown"]:
        print(f"  unrecognized                   : {len(res['unknown'])}  ✗ WOULD FAIL AT LAUNCH")
        for flag, vals in res["unknown"]:
            print(f"      {flag}{' ' + ' '.join(vals) if vals else ''}")
        print("\n  These are not in the current parser — most likely deleted by a version bump.")
        print("  Check designs/CHANGELOG.md for which version removed each one, and whether a")
        print("  replacement flag exists, before you drop it from the command.")
    else:
        print("  unrecognized                   : 0")

    _print_resolution(res["resolution"])

    if res["unsatisfiable"]:
        print(f"  unsatisfiable combinations     : {len(res['unsatisfiable'])}  "
              "✗ WOULD FAIL IN THE EXTRACTOR")
        for flag, dep, token in res["unsatisfiable"]:
            print(f"      {flag} requires {dep}, but the command says `{token}`")
        print("\n  These pass argparse and crash later, inside Gen3FeaturesExtractor.__init__.")
        print("  The dependency graph is designs/flag_registry.md (generated from")
        print("  agents.model.flag_registry, which is where the constructor's raises are declared).")

    if res["combinations"]:
        print(f"  refused combinations           : {len(res['combinations'])}  "
              "✗ WOULD FAIL IN resolve_config")
        for combo, provenance in res["combinations"]:
            print(f"      {combo.message}")
            for line in provenance:
                print(f"        · {line}")
        print("\n  These are main.train.combination_checks — the value-conditional refusals the")
        print("  launch path prints verbatim. A value marked INHERITED was never typed: it came")
        print("  from the parent checkpoint's model_config.json, which is what a fork resumes.")

    if not res["unknown"] and not res["unsatisfiable"] and not res["combinations"]:
        print("  ✓ this command still launches")
        return 0
    return 1


def _print_resolution(resolution: dict | None) -> None:
    """What the argv was resolved AGAINST — stated on every run, so a silent pass is never mute
    about whether it read the parent at all."""
    if resolution is None:
        return
    if resolution.get("parse_error"):
        print("  ⚠️  WARNING: the argv does not parse, so no inherited value could be resolved: "
              + "; ".join(resolution["parse_error"]))
        return
    label = "same-run RESTART checkpoint" if resolution["same_run"] else "FORK PARENT"
    if resolution.get("config_path") and not resolution.get("read_error"):
        n = len(resolution["inherited"])
        print(f"  resolved against the {label}: {resolution['config_path']}")
        print(f"      {n} unset flag(s) INHERITED from it — the checks below read the EFFECTIVE "
              f"config, not the argv")
    else:
        why = resolution.get("read_error", "no model_config.json")
        print(f"  ⚠️  WARNING: could not read the {label}'s recorded config ({why}) — falling back "
              f"to ARGV-ONLY checking, which cannot see an inherited value.")
        for path in resolution["tried"]:
            print(f"        tried: {path}")


if __name__ == "__main__":
    sys.exit(main())
