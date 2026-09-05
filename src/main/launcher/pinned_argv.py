"""AN ARGV IS VALIDATED BY THE PARSER OF THE TREE THAT WILL RUN IT.

WHY THIS EXISTS (2026-09-05). ``--pin-commit`` exists so an old recipe can be re-run on its own
commit. But every argv check the launcher performed — its own ``--dry-run``, and
``main.checkargs`` — read the CURRENT tree's ``train_rl_agent.build_parser()``, while the child
would run the PINNED tree's parser. So the one command ``--pin-commit`` is for was the one command
it refused::

    python -m main.launcher --pin-commit b13b30b2 <that run's own argv> --dry-run
    error: argument --hp-type-belief-coef: invalid float value: 'learned'

At b13b30b2 ``--hp-type-belief`` TOOK A VALUE (``learned``). Today that flag is deleted, so
argparse abbreviation-matches the token onto the surviving ``--hp-type-belief-coef`` and feeds it
the value. **A same-named flag whose ARITY or TYPE changed is invisible to any presence check** —
you cannot catch it by asking "does the current parser know this flag", because the current parser
thinks it does.

WHAT THIS DOES. When the resolved pin differs from the HEAD of the checkout the launcher is
running from, materialise that commit's ``src/`` with ``git archive`` (no worktree — the startup
prune has already cost this program one live run, and a validation command must create nothing),
copy the probe beside it, and run ONE subprocess with a clean environment that parses the child
argv against the pinned parser. The subprocess boundary is the point: the pinned tree's imports
(torch included) never enter the launcher's process, and neither do its exceptions.

TWO MODES, AND ONLY ONE OF THEM IS A VERDICT (see ``pinned_argv_probe``):

* ``build_parser`` — the pinned tree's own ``build_parser()``. AUTHORITATIVE: a parse failure here
  is a launch that would die ~40 s later with a stray run dir, so the launcher exits
  ``FATAL_CONFIG`` naming the offending token.
* ``ast_scan`` — a static read of the pinned ``add_argument`` calls, for commits before
  ``build_parser()`` existed (it landed 2026-08-16; b13b30b2 predates it). Best-effort, so its
  findings are a WARNING.
* ``unavailable`` — no parser could be obtained (git failure, an import the pinned tree cannot
  satisfy here, a timeout). **Never a silent pass**: the launcher says the argv is UNVALIDATED and
  proceeds. Validating with the wrong parser is the defect; refusing to launch on a check we could
  not run would be a worse one.

WHAT IT DOES NOT COVER. Only the PARSER is pinned. The extractor dependency graph
(``agents.model.flag_registry``) and the value-conditional refusals
(``main.train.combination_checks``) are still read from the CURRENT tree, so callers print those
findings as ADVISORY whenever a pinned check ran — they describe a different commit's rules.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: What of the tree the pinned parser needs. ``src/rust_sim`` is deliberately excluded — 66 MB of
#: the 86 MB a full ``src/`` archive costs, and nothing importable from it is on the parser path.
#: ``data/`` IS included (18 MB) and is NOT optional: ``utils.paths.repo_root()`` is
#: ``__file__``-relative, so a pinned tree looks for its data beside itself, and the gen3_data
#: facade raises ``FileNotFoundError`` at import time when it is missing — which silently demotes
#: every recent pin from the authoritative ``build_parser`` mode to the static scan.
_ARCHIVE_PATHS = ("src/main", "src/agents", "src/utils", "src/poke_env", "data")

#: Time box for the probe subprocess. A pinned tree may import torch (~2 s measured), and a cold
#: page cache or a loaded box stretches that; past this the answer is UNAVAILABLE, never a pass.
DEFAULT_TIMEOUT_S = 60.0

_PROBE_BASENAME = "_pinned_argv_probe.py"

#: sha -> materialised temp dir, for this process only. A launcher validates one argv per launch,
#: but ``checkargs`` over a directory of runs asks the same commit many times.
_CACHE: Dict[Tuple[str, str], str] = {}


@dataclass
class ParseReport:
    """What the pinned tree's parser said about this argv."""

    sha: str
    mode: str                                   # "build_parser" | "ast_scan" | "unavailable"
    reason: str = ""                            # why unavailable / why not build_parser
    unknown: List[str] = field(default_factory=list)
    stray: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    values: Dict[str, str] = field(default_factory=dict)
    n_options: int = 0
    seconds: float = 0.0

    @property
    def available(self) -> bool:
        """Did we manage to ask the pinned tree anything at all?"""
        return self.mode in ("build_parser", "ast_scan")

    @property
    def authoritative(self) -> bool:
        """Is a failure here a REFUSAL (the pinned parser itself) or a WARNING (a static read)?"""
        return self.mode == "build_parser"

    @property
    def ok(self) -> bool:
        return not (self.unknown or self.stray or self.errors)

    def findings(self) -> List[str]:
        """One human line per thing the pinned parser could not accept."""
        out = [f"argparse: {e}" for e in self.errors]
        out += [f"unknown flag at this commit: {f}" for f in self.unknown]
        out += [f"unconsumed value {t!r} — a flag's ARITY differs at this commit" for t in self.stray]
        return out

    def summary_line(self) -> str:
        """The one line a launch prints about the check it just ran."""
        sha8 = self.sha[:8]
        if not self.available:
            # `parser_unavailable_at_pin` is a NAME on purpose: it is what a reader greps for, and
            # naming it is the difference between "unvalidated" and a silent pass.
            return (f"⚠️  parser_unavailable_at_pin @{sha8} — argv NOT validated: {self.reason}. "
                    f"The child will parse it with code this box could not inspect.")
        how = ("the pinned build_parser()" if self.authoritative
               else "a best-effort STATIC scan of the pinned add_argument() calls (NOT "
                    "authoritative)")
        if self.ok:
            return (f"✅ argv validated against PINNED parser @{sha8} via {how} "
                    f"[{self.n_options} options, {self.seconds:.1f}s]")
        verb = "REFUSED by" if self.authoritative else "questioned by"
        return f"✗ argv {verb} the PINNED parser @{sha8} via {how}"


def _git(repo_root: str, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True,
                          timeout=timeout)


def head_sha(repo_root: Optional[str] = None) -> Optional[str]:
    """Full sha of HEAD in the checkout the launcher itself is running from."""
    root = repo_root or _repo_root()
    try:
        out = _git(root, "rev-parse", "HEAD")
    except Exception:                                # noqa: BLE001 — never crash a launch on this
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _repo_root() -> str:
    from utils.git import get_repo_root
    return get_repo_root()


def differs_from_head(sha: str, repo_root: Optional[str] = None) -> bool:
    """Does this pin name a DIFFERENT commit than the tree the launcher is running from?

    Prefix-tolerant in both directions, so a short pin that names HEAD is correctly read as "same"
    and skips the whole check (behaviour then is unchanged, which is the contract).
    """
    head = head_sha(repo_root)
    if not head or not sha:
        return False
    return not (head == sha or head.startswith(sha) or sha.startswith(head))


class _Unavailable(Exception):
    """We could not materialise or interrogate the pinned tree. Carries the reason, verbatim."""


def _tree_paths(sha: str, repo_root: str) -> List[str]:
    """Which of ``_ARCHIVE_PATHS`` actually exist at ``sha``.

    ``git archive`` errors out on a pathspec that matches nothing, and the tree's layout has
    moved over the range of commits a pin can name — so ask before extracting. A spec that names
    no commit is refused here rather than surfacing as a confusing archive error.
    """
    if _git(repo_root, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}").returncode != 0:
        raise _Unavailable(f"{sha!r} does not name a commit in {repo_root}")
    return [p for p in _ARCHIVE_PATHS
            if _git(repo_root, "rev-parse", "--verify", "--quiet", f"{sha}:{p}").returncode == 0]


def materialise(sha: str, repo_root: Optional[str] = None) -> str:
    """Extract the pinned ``src/`` into a temp dir and return that dir. Cached per process.

    ``git archive`` rather than ``git worktree add`` on purpose: a worktree is a durable,
    registered, prunable object, and this program has already lost a live run to a validation
    command that touched the worktree list. An archive is a read of the object database and leaves
    nothing registered anywhere.
    """
    root = repo_root or _repo_root()
    key = (os.path.abspath(root), sha)
    cached = _CACHE.get(key)
    if cached and os.path.isdir(os.path.join(cached, "src")):
        return cached

    paths = _tree_paths(sha, root)
    if not paths:
        raise _Unavailable(f"commit {sha[:8]} carries none of {', '.join(_ARCHIVE_PATHS)}")
    tmp = tempfile.mkdtemp(prefix=f"pinned-argv-{sha[:8]}-")
    atexit.register(shutil.rmtree, tmp, True)
    tar_path = os.path.join(tmp, "tree.tar")
    out = _git(root, "archive", "--format=tar", "-o", tar_path, sha, "--", *paths, timeout=120.0)
    if out.returncode != 0:
        raise _Unavailable(f"`git archive {sha[:8]}` failed: {out.stderr.strip() or 'unknown'}")
    ex = subprocess.run(["tar", "-xf", tar_path, "-C", tmp], capture_output=True, text=True,
                        timeout=120.0)
    if ex.returncode != 0:
        raise _Unavailable(f"extracting the {sha[:8]} archive failed: {ex.stderr.strip()}")
    os.remove(tar_path)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinned_argv_probe.py"),
                os.path.join(tmp, _PROBE_BASENAME))
    _CACHE[key] = tmp
    return tmp


def _probe_env(src_dir: str) -> Dict[str, str]:
    """A CLEAN environment for the probe: the caller's ``PYTHONPATH`` must not leak in.

    The whole point is to import the PINNED tree; inheriting a ``PYTHONPATH`` that names the
    current checkout's ``src`` would silently validate against exactly the parser we are trying
    not to use (and it is set in every worktree shell on this box, per the root CLAUDE.md).
    """
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = src_dir
    return env


def pinned_parser_check(
    sha: str,
    child_argv: List[str],
    repo_root: Optional[str] = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> ParseReport:
    """Parse ``child_argv`` with the parser of commit ``sha``. Never raises; never a silent pass.

    A failure to materialise, an import the pinned tree cannot satisfy on this box, or a timeout
    all come back as ``mode="unavailable"`` carrying the reason — the caller then says the argv is
    UNVALIDATED rather than pretending the current parser's verdict applies.
    """
    started = time.monotonic()
    try:
        tmp = materialise(sha, repo_root)
    except _Unavailable as e:
        return ParseReport(sha=sha, mode="unavailable", reason=str(e),
                           seconds=time.monotonic() - started)
    except Exception as e:                           # noqa: BLE001 — a check must not kill a launch
        return ParseReport(sha=sha, mode="unavailable",
                           reason=f"{type(e).__name__}: {e}", seconds=time.monotonic() - started)

    src_dir = os.path.join(tmp, "src")
    argv_path = os.path.join(tmp, "argv.json")
    out_path = os.path.join(tmp, "report.json")
    with open(argv_path, "w") as f:
        json.dump(list(child_argv), f)
    if os.path.exists(out_path):
        os.remove(out_path)

    cmd = [sys.executable, os.path.join(tmp, _PROBE_BASENAME), src_dir, argv_path, out_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=_probe_env(src_dir), cwd=tmp)
    except subprocess.TimeoutExpired:
        return ParseReport(sha=sha, mode="unavailable",
                           reason=f"the pinned-parser probe did not finish within {timeout:.0f}s",
                           seconds=time.monotonic() - started)
    if not os.path.exists(out_path):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
        return ParseReport(sha=sha, mode="unavailable",
                           reason=f"the pinned-parser probe wrote no report ({tail[0]})",
                           seconds=time.monotonic() - started)
    with open(out_path) as f:
        data = json.load(f)
    return ParseReport(
        sha=sha,
        mode=str(data.get("mode", "unavailable")),
        reason=str(data.get("reason", "")),
        unknown=list(data.get("unknown", [])),
        stray=list(data.get("stray", [])),
        errors=list(data.get("errors", [])),
        values=dict(data.get("values", {})),
        n_options=int(data.get("n_options", 0)),
        seconds=time.monotonic() - started,
    )


def report_lines(report: ParseReport) -> List[str]:
    """The block a caller prints: the summary, then one line per finding, then what to do."""
    lines = [report.summary_line()]
    lines += [f"    {f}" for f in report.findings()]
    if report.available and not report.ok and not report.authoritative:
        lines.append("    (a STATIC scan can be incomplete, so this is a warning, not a refusal — "
                     "the child's own parser is the authority)")
    if report.mode == "ast_scan" and report.reason:
        lines.append(f"    (why not the real parser: {report.reason})")
    return lines


def clear_cache() -> None:
    """Drop the per-process materialisation cache (tests; the temp dirs die at exit anyway)."""
    _CACHE.clear()
