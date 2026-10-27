"""Bulk, read-only git object access: the one place this package shells out to git for whole-history reads.

Every function takes the repository as an argument (never the process cwd) and only READS: no ref, no config and
no object is written, so pointing one at the real repository is safe. ``git`` is run with ``-c core.quotepath=off``
so a path comes back as bytes, not as an escaped string.
"""
from __future__ import annotations

import re
import subprocess
import threading
from typing import Iterable, Iterator, Optional

#: A full commit id.
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
NULL_SHA = "0" * 40


def run_git(repo: str, *args: str, stdin: Optional[bytes] = None, check: bool = True) -> bytes:
    """``git -C repo args``, stdout as bytes. ``check`` raises ``RuntimeError`` carrying stderr."""
    proc = subprocess.run(["git", "-c", "core.quotepath=off", "-C", repo, *args], input=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:300]}")
    return proc.stdout


def commit_oids(repo: str) -> list[str]:
    """Every COMMIT object in the object database, reachable or not (the universe an old citation can name)."""
    out = run_git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objecttype) %(objectname)")
    return sorted(line.split()[1] for line in out.decode().splitlines() if line.startswith("commit "))


def reachable_objects(repo: str, *revs: str, all_refs: bool = False) -> dict[str, str]:
    """``{oid: first path}`` of every object reachable from ``revs`` (or every ref); commits map to ``""``."""
    args = ["rev-list", "--objects"]
    args += ["--all"] if all_refs else list(revs)
    out: dict[str, str] = {}
    for line in run_git(repo, *args).decode(errors="replace").splitlines():
        oid, _, path = line.partition(" ")
        out.setdefault(oid, path)
    return out


def stream_objects(repo: str, oids: Iterable[str]) -> Iterator[tuple[str, str, bytes]]:
    """Yield ``(oid, type, data)`` for each oid through ONE ``git cat-file --batch`` process (no per-object spawn)."""
    oid_list = list(oids)
    proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None and proc.stdout is not None

    def _feed() -> None:
        try:
            for oid in oid_list:
                proc.stdin.write(oid.encode() + b"\n")  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
        except BrokenPipeError:
            pass

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()
    try:
        while True:
            header = proc.stdout.readline()
            if not header:
                break
            fields = header.split()
            if len(fields) != 3:  # "<oid> missing"
                continue
            oid, typ, size = fields[0].decode(), fields[1].decode(), int(fields[2])
            data = proc.stdout.read(size + 1)[:-1]
            yield oid, typ, data
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
        feeder.join(timeout=5)


def resolve_all(repo: str, revs: Iterable[str]) -> dict[str, str]:
    """``{rev: status}`` for rev expressions through one ``cat-file --batch-check``: the oid, ``ambiguous`` or ``missing``.

    This is git's OWN resolution (the independent check the citation rewriter's index is verified against).
    """
    rev_list = list(revs)
    if not rev_list:
        return {}
    out = run_git(repo, "cat-file", "--batch-check", stdin=("\n".join(rev_list) + "\n").encode(), check=False)
    result: dict[str, str] = {}
    for rev, line in zip(rev_list, out.decode().splitlines()):
        fields = line.split()
        result[rev] = fields[0] if len(fields) == 3 and FULL_SHA.match(fields[0]) else ("ambiguous" if "ambiguous" in line else "missing")
    return result
