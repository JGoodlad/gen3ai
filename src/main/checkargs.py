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

Two ways a command fails, and it reports both in one pass:
  * a flag the parser no longer knows (the motivating case above);
  * a combination the extractor constructor refuses — `agents.model.flag_registry`'s `requires`
    graph, e.g. `--intent-conditional` without `--damage-outgoing`. That crash is later and more
    expensive than an argparse error: the run dir exists, the child starts, and the traceback comes
    out of `Gen3FeaturesExtractor.__init__`.

The dependency half is deliberately CONSERVATIVE — it fires only when the argv enables a flag AND
explicitly names a dependency with a disabled value. An argv that simply omits a dependency is not
reported, because a resume inherits every unspecified flag from the checkpoint's recorded config
(`_resolve`), so absence carries no information here. Under-reporting is the right failure
direction: this tool's value is that a warning from it is worth acting on.

It REPORTS; it does not repair. A deleted flag may have a replacement, so dropping it silently can
change the run in a way nobody sees — naming the problem is the job, deciding what the command
should say is the reader's.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Dict, List, Tuple

# Flags the LAUNCHER owns and strips before forwarding — absent from the trainer's parser by
# design, so they must not be reported as stale. Sourced from the launcher rather than re-listed
# here would be better; it is a short, stable set and the launcher does not export it.
LAUNCHER_ONLY = {
    "--restart-interval-hours", "--restart-grace-minutes", "--max-crash-restarts",
    "--nice", "--no-pin", "--sync-to-main",
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
    return {"n_flags": len(ok) + len(launcher) + len(unknown),
            "accepted": ok, "launcher_only": launcher, "unknown": unknown,
            "unsatisfiable": unsatisfiable_pairs(argv)}


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

    if res["unsatisfiable"]:
        print(f"  unsatisfiable combinations     : {len(res['unsatisfiable'])}  "
              "✗ WOULD FAIL IN THE EXTRACTOR")
        for flag, dep, token in res["unsatisfiable"]:
            print(f"      {flag} requires {dep}, but the command says `{token}`")
        print("\n  These pass argparse and crash later, inside Gen3FeaturesExtractor.__init__.")
        print("  The dependency graph is designs/flag_registry.md (generated from")
        print("  agents.model.flag_registry, which is where the constructor's raises are declared).")

    if not res["unknown"] and not res["unsatisfiable"]:
        print("  ✓ this command still launches")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
