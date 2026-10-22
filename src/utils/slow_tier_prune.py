"""Prune ``designs/ops/slow_tier_status.json`` rows whose test is no longer collected as ``slow``.

The artifact is keyed by test id and the writer only ever merges (``slow_tier_status.record_results``), so a deleted,
renamed, re-parametrised or un-marked slow test leaves its row behind forever: it ages into the "stale" report, and a
stale ``fail`` row of a test that no longer exists keeps the routine gate red. This is the one deterministic way to
remove them: ask pytest what it collects as ``slow`` (the SAME ``-m slow`` collection the conftest's
``pytest_itemcollected`` records) and drop every row outside that set.

    python -m utils.slow_tier_prune            # CHECK: list the rows that would go; exit 1 if there are any
    python -m utils.slow_tier_prune --write    # delete them (under the writer's flock + atomic rename)

Exit: 0 nothing to prune (or pruned with --write), 1 rows would be pruned (check mode), 2 the collection failed or
its output did not parse to the count pytest reported (nothing is touched). Run it with ``PYTHONPATH`` on the
checkout whose ``src/`` is being judged and ``CUDA_VISIBLE_DEVICES=""``; collection is CPU-only (~25 s).

A test collected only under some environment (a GPU, a missing optional dependency's parametrisation) is judged by the
environment this runs in: its row would be pruned and the next full slow-tier run would re-bank it, so run the prune on a
box with the same deps the tier runs on, and read the printed list before committing.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Set

from utils import slow_tier_status as ST
from utils.paths import repo_path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY = re.compile(r"(\d+)(?:/\d+)? tests? collected")


class CollectionError(RuntimeError):
    """The slow-set collection failed or its output did not add up."""


def parse_collect_output(text: str) -> Set[str]:
    """The node ids in ``pytest --collect-only -q`` output; raises if their count differs from the summary line's."""
    text = _ANSI.sub("", text)
    ids = {ln.strip() for ln in text.splitlines() if "::" in ln and not ln.startswith((" ", "\t"))}
    m = _SUMMARY.search(text)
    if not m:
        raise CollectionError("no '<N> tests collected' summary line in pytest's output")
    if int(m.group(1)) != len(ids):
        raise CollectionError(f"pytest reported {m.group(1)} collected tests but {len(ids)} node ids parsed")
    return ids


def collect_slow_ids(run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> Set[str]:
    argv = [sys.executable, "-m", "pytest", "src/", "-m", "slow", "--collect-only", "-q", "--color=no",
            "-p", "no:cacheprovider"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = run(argv, cwd=str(repo_path()), env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise CollectionError(f"`{' '.join(argv)}` exited {r.returncode}:\n{(r.stdout or '')[-1500:]}{(r.stderr or '')[-1500:]}")
    ids = parse_collect_output(r.stdout)
    if not ids:
        raise CollectionError("the slow collection is empty")
    return ids


def main(argv: Optional[Sequence[str]] = None, *, collect: Callable[[], Set[str]] = collect_slow_ids,
         path: Optional[Path] = None) -> int:
    write = "--write" in (argv if argv is not None else sys.argv[1:])
    try:
        collected = collect()
    except CollectionError as e:
        print(f"slow_tier_prune: REFUSING, nothing touched: {e}", file=sys.stderr)
        return 2
    removed = ST.prune_rows(collected, path, write=write)
    by_file: dict = {}
    for nodeid, status in removed:
        by_file.setdefault(nodeid.split("::")[0], []).append(status)
    for f, sts in sorted(by_file.items()):
        print(f"  {len(sts):3d} row(s) {f}  (statuses: {', '.join(sorted(set(sts)))})")
    verb = "pruned" if write else "WOULD prune"
    print(f"slow_tier_prune: {verb} {len(removed)} row(s) of the status file; {len(collected)} slow tests collected")
    return 1 if (removed and not write) else 0


if __name__ == "__main__":
    sys.exit(main())
