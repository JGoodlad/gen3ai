"""The BYTE-BUDGET RATCHET for the instruction docs (``designs/ops/instruction_docs_principles.md`` §3).

The files an agent loads because of WHERE it works grew append-only: two whole-corpus cuts regrew within weeks
(``designs/ops/instruction_docs_audit_2026-10-19.md`` F1). This is the mechanism that stops it. The numbers live in
ONE committed file, :data:`BUDGET_FILE`; ``src/claude_md_freshness_gate_test.py`` enforces them.

**The rules** (pure, in :func:`check`):

1. Every file has a cap: its TYPE cap (``type_caps``), or its ``grandfathered`` entry when it was already over the
   type cap. A file over its cap FAILS.
2. A grandfathered entry exists only while the file is over its type cap: a file back under it must LEAVE the list.
3. A grandfathered cap follows the file down: more than :data:`LOWER_SLACK` bytes of headroom FAILS until the cap
   is lowered to the file's size. So a sweep that shrinks a file locks the gain in, in the same commit.

A cap is never raised. An addition to an over-budget file removes more than it adds (principle 7).

**Two scopes.** The REPO files (every ``CLAUDE.md``, ``.claude/commands/*.md``) are checked on every run. The
USER files (the auto-memory index ``MEMORY.md`` and the subagent definitions ``~/.claude/agents/*.md``) are
machine-local configuration outside git: they are checked only where they exist (skipped, saying so, elsewhere),
and against rule 1 only; their grandfathered entries are keyed by the placeholder paths below. Sizes are
whole-file bytes.

    python -m utils.instruction_budget      # the census: every file, its bytes, its cap, its type target
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from utils.paths import repo_path, repo_root

BUDGET_FILE = "designs/ops/instruction_docs_budget.json"

#: Headroom a grandfathered file may keep under its recorded cap before the cap must come down. Small edits that
#: shrink a file need not touch the budget file; a sweep (kilobytes) must.
LOWER_SLACK = 1000

#: The placeholder keys of the two user-file kinds in :data:`BUDGET_FILE` (the real paths are machine-local).
MEMORY_KEY = "~/.claude/projects/<main checkout>/memory/MEMORY.md"
AGENT_KEY_PREFIX = "~/.claude/agents/"

# `deps/` is a submodule, `.claude/worktrees/` holds other agents' checkouts of this repo, and the build trees are
# output: none is a document this checkout owns.
_EXCLUDED_DIR_PARTS = ("deps", ".claude", "node_modules", "target")


class Doc(NamedTuple):
    kind: str   # a key of ``type_caps``: root / leaf / command / memory / agent
    size: int   # bytes


def load_budget(path: Optional[Path] = None) -> dict:
    return json.loads((path or repo_path(BUDGET_FILE)).read_text())


def claude_md_files(root: Optional[Path] = None) -> List[Path]:
    """Every ``CLAUDE.md`` this checkout owns, sorted."""
    root = root or repo_root()
    return sorted(p for p in root.rglob("CLAUDE.md")
                  if not any(part in _EXCLUDED_DIR_PARTS for part in p.relative_to(root).parts[:-1]))


def repo_docs(root: Optional[Path] = None) -> Dict[str, Doc]:
    """Repo-relative path -> :class:`Doc` for every in-repo instruction file."""
    root = root or repo_root()
    out = {}
    for p in claude_md_files(root):
        rel = p.relative_to(root).as_posix()
        out[rel] = Doc("root" if rel == "CLAUDE.md" else "leaf", p.stat().st_size)
    for p in sorted((root / ".claude" / "commands").glob("*.md")):
        out[p.relative_to(root).as_posix()] = Doc("command", p.stat().st_size)
    return out


def claude_config_dir() -> Path:
    """Claude Code's user config directory (``$CLAUDE_CONFIG_DIR``, else ``~/.claude``)."""
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def memory_index_path(main_checkout: Path, config_dir: Optional[Path] = None) -> Path:
    """Where Claude Code keeps this project's ``MEMORY.md``: the project dir is the MAIN checkout's absolute path with
    every non-alphanumeric character replaced by ``-`` (worktree sessions share the main checkout's memory)."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(main_checkout))
    return (config_dir or claude_config_dir()) / "projects" / slug / "memory" / "MEMORY.md"


def user_docs(main_checkout: Optional[Path] = None, config_dir: Optional[Path] = None) -> Dict[str, Doc]:
    """Placeholder key -> :class:`Doc` for each user file PRESENT on this machine (empty when none is)."""
    config_dir = config_dir or claude_config_dir()
    if main_checkout is None:
        from utils.git import get_main_repo_root
        try:
            main_checkout = Path(get_main_repo_root(cwd=str(repo_root())))
        except (subprocess.CalledProcessError, OSError):   # no git (a tarball): this checkout is the main one
            main_checkout = repo_root()
    out = {}
    memory = memory_index_path(main_checkout, config_dir)
    if memory.is_file():
        out[MEMORY_KEY] = Doc("memory", memory.stat().st_size)
    for p in sorted((config_dir / "agents").glob("*.md")):
        out[AGENT_KEY_PREFIX + p.name] = Doc("agent", p.stat().st_size)
    return out


def is_user_key(key: str) -> bool:
    return key.startswith("~/")


def check(docs: Dict[str, Doc], type_caps: Dict[str, int], grandfathered: Dict[str, int],
          *, user_scope: bool = False) -> List[str]:
    """Every rule broken by ``docs`` (one line each; empty = green). ``user_scope``: the machine-local user files,
    held to rule 1 ONLY. Rules 2-3 and a missing file's entry would make an unrelated diff edit the budget file
    for a change made outside git; the grandfathered user entries are lowered by the sweep that trims them (U9)."""
    problems = []
    for key, doc in sorted(docs.items()):
        type_cap = type_caps[doc.kind]
        cap = grandfathered.get(key, type_cap)
        if doc.size > cap:
            how = "its grandfathered cap" if key in grandfathered else f"the {doc.kind} type cap"
            problems.append(f"OVER BUDGET: {key} is {doc.size:,} bytes, over {how} of {cap:,}. Remove more than "
                            f"you add (principles 7-8); a cap is never raised.")
        elif user_scope:
            continue
        elif key in grandfathered and doc.size <= type_cap:
            problems.append(f"STALE ENTRY: {key} is {doc.size:,} bytes, within the {doc.kind} type cap of "
                            f"{type_cap:,}. Remove its `grandfathered` entry from {BUDGET_FILE}.")
        elif key in grandfathered and cap - doc.size > LOWER_SLACK:
            problems.append(f"LOWER THE CAP: {key} shrank to {doc.size:,} bytes, {cap - doc.size:,} under its "
                            f"grandfathered cap of {cap:,}. Set its entry in {BUDGET_FILE} to {doc.size}.")
    if not user_scope:
        problems += [f"MISSING FILE: {BUDGET_FILE} grandfathers {key}, which does not exist. Remove the entry."
                     for key in sorted(grandfathered) if not is_user_key(key) and key not in docs]
    return problems


def census(docs: Dict[str, Doc], type_caps: Dict[str, int], grandfathered: Dict[str, int],
           targets: Dict[str, int]) -> List[str]:
    rows = [f"{'file':62s} {'bytes':>8s} {'cap':>8s} {'type cap':>8s} {'target':>8s}"]
    for key, doc in sorted(docs.items(), key=lambda kv: -kv[1].size):
        cap = grandfathered.get(key, type_caps[doc.kind])
        rows.append(f"{key:62s} {doc.size:8,d} {cap:8,d} {type_caps[doc.kind]:8,d} {targets[doc.kind]:8,d}")
    return rows


def main() -> int:
    budget = load_budget()
    docs = {**repo_docs(), **user_docs()}
    print("\n".join(census(docs, budget["type_caps"], budget["grandfathered"], budget["type_targets"])))
    problems = (check(repo_docs(), budget["type_caps"], budget["grandfathered"])
                + check(user_docs(), budget["type_caps"], budget["grandfathered"], user_scope=True))
    print("\n".join(problems) or "within budget")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
