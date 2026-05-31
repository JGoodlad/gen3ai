"""Checkpoint discovery and CLI argument manipulation for the launcher."""

import argparse
import glob
import os
import re


def find_latest_checkpoint(
    models_root: str,
    run_dir: "str | None" = None,
    min_mtime: float = 0.0,
) -> "str | None":
    if run_dir:
        latest_txt = os.path.join(run_dir, "latest.txt")
        if os.path.exists(latest_txt):
            with open(latest_txt) as f:
                name = f.read().strip()
            candidate = os.path.join(run_dir, name)
            if os.path.exists(candidate):
                return candidate

    zips = glob.glob(os.path.join(models_root, "**", "*.zip"), recursive=True)
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


def _apply_default_showdown_port(
    args: list, default_port: int = DEFAULT_TRAINING_SHOWDOWN_PORT
) -> list:
    """Inject ``--showdown-port <default_port>`` into the child args when the user
    didn't pass one. An explicit ``--showdown-port`` (any spelling) always wins."""
    if _peek_arg(args, "--showdown-port", type_=int) is not None:
        return args
    return _set_arg(args, "--showdown-port", str(default_port))


def _find_model_arg(args: list) -> "str | None":
    return _peek_arg(args, "--model")


def _insert_or_replace_model_arg(args: list, checkpoint: str) -> list:
    return _set_arg(args, "--model", checkpoint)


def _insert_or_replace_run_dir_arg(args: list, run_dir: str) -> list:
    return _set_arg(args, "--run-dir", run_dir)


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
