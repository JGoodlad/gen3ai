"""Checkpoint discovery and CLI argument manipulation for the launcher."""

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


def _find_model_arg(args: list) -> "str | None":
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            return args[i + 1]
    return None


def _insert_or_replace_model_arg(args: list, checkpoint: str) -> list:
    out = []
    i = 0
    replaced = False
    while i < len(args):
        if args[i] == "--model":
            out.extend(["--model", checkpoint])
            replaced = True
            i += 2
        else:
            out.append(args[i])
            i += 1
    if not replaced:
        out.extend(["--model", checkpoint])
    return out


def _insert_or_replace_run_dir_arg(args: list, run_dir: str) -> list:
    out = []
    i = 0
    replaced = False
    while i < len(args):
        if args[i] == "--run-dir":
            out.extend(["--run-dir", run_dir])
            replaced = True
            i += 2
        else:
            out.append(args[i])
            i += 1
    if not replaced:
        out.extend(["--run-dir", run_dir])
    return out


def _strip_launcher_args(argv: list) -> list:
    """Strip launcher-only flags so they are not forwarded to train_rl_agent.py."""
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--restart-interval-hours":
            i += 2
        elif argv[i].startswith("--restart-interval-hours="):
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
