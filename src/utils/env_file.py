"""Read a conda env file's pins, and check a LIVE env against them — strictly READ-ONLY.

WHY THIS EXISTS (2026-10-12). `scripts/bootstrap.sh` keys its "the shared env is current" stamp on the sha256
of `environment_torch28.yml`, so ANY edit to that file (the scipy / scikit-learn fix, 2026-10-12) makes every
worktree bootstrap REFUSE (exit 3) until someone re-stamps. The old way to re-stamp is
`--update-shared-env` = `conda env update --prune` on the env every live run and agent shares: exactly the
mutation the standing rule forbids. But when the edit only adds pins the env ALREADY has, nothing needs
installing, and the safe path is to PROVE that and write the stamp: ``python -m utils.env_file check
--env-file environment_torch28.yml`` (what `bootstrap.sh --adopt-env` runs) exits 0 only when the interpreter's
env already satisfies EVERY pin of the file — it installs, removes and updates nothing.

WHAT "SATISFIES" MEANS.
* a pip pin ``name==version`` needs that exact version (PEP 440 equality, so ``2.8.0+cu126`` is not
  ``2.8.0+cu128``); an unpinned entry (``requests[socks]``, ``python-socks[asyncio]``) needs presence only;
* a conda dependency ``name=version`` needs that version as a PREFIX (conda's ``=`` semantics: ``python=3.11``
  is met by 3.11.15), read from ``<prefix>/conda-meta`` (never by running conda);
* pip flags (``--extra-index-url ...``) are not packages and are skipped.
It is a one-way check: packages in the env that the file does not name are NOT a failure (an env update would
not remove them without ``--prune``, and the shared env carries dev extras by design).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

#: (normalised name, version or None for an unpinned entry, the entry as written)
Pin = Tuple[str, Optional[str], str]


def norm(name: str) -> str:
    """PEP 503 name normalisation (``Jinja2`` / ``typing_extensions`` / ``python.socks`` -> one spelling)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _entry_pin(entry: str, sep: str) -> Optional[Pin]:
    entry = entry.strip()
    if not entry or entry.startswith("-"):                       # pip flags: `--extra-index-url ...`
        return None
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(?:" + sep + r"\s*([^\s#,;]+))?", entry)
    if m is None:
        return None
    return norm(m.group(1)), m.group(3), entry


def _entries(env_text: str) -> Tuple[List[str], List[str]]:
    """(conda dependencies, pip entries) of an env file — comments and blanks dropped."""
    conda: List[str] = []
    pip: List[str] = []
    in_deps = in_pip = False
    pip_indent = -1
    for line in env_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if re.match(r"^dependencies:\s*$", line):
            in_deps, in_pip = True, False
            continue
        if not in_deps:
            continue
        if indent == 0:                                           # the next top-level key ends the list
            in_deps = in_pip = False
            continue
        if re.match(r"^\s*-\s*pip:\s*$", line):
            in_pip, pip_indent = True, indent
            continue
        if in_pip and indent <= pip_indent:
            in_pip = False
        m = re.match(r"^-\s+(.*?)\s*(#.*)?$", stripped)
        if m:
            (pip if in_pip else conda).append(m.group(1))
    return conda, pip


def conda_pins(env_text: str) -> List[Pin]:
    """The conda dependencies (``python=3.11.15``, ``cuda-cccl=12.6.77``) as pins."""
    return [p for p in (_entry_pin(e, "=") for e in _entries(env_text)[0]) if p is not None]


def pip_pins(env_text: str) -> List[Pin]:
    """The pip block's entries as pins (``torch==2.8.0+cu126``; unpinned entries carry version None)."""
    return [p for p in (_entry_pin(e, "==") for e in _entries(env_text)[1]) if p is not None]


def _same_version(want: str, have: str) -> bool:
    try:
        from packaging.version import InvalidVersion, Version
        try:
            return Version(want) == Version(have)
        except InvalidVersion:
            pass
    except ImportError:  # pragma: no cover - packaging ships with every env of ours
        pass
    return want == have


def unsatisfied(env_text: str, pip_installed: Dict[str, str], conda_installed: Dict[str, str]) -> List[str]:
    """Every pin of ``env_text`` that the described env does NOT meet, as one human line each ([] = satisfied).

    ``pip_installed`` / ``conda_installed``: normalised name -> installed version."""
    out: List[str] = []
    for name, want, entry in conda_pins(env_text):
        have = conda_installed.get(name)
        if have is None:
            out.append(f"conda {entry}: not installed")
        elif want is not None and not (have == want or have.startswith(want + ".")):
            out.append(f"conda {entry}: the env has {have}")
    for name, want, entry in pip_pins(env_text):
        have = pip_installed.get(name)
        if have is None:
            out.append(f"pip {entry}: not installed")
        elif want is not None and not _same_version(want, have):
            out.append(f"pip {entry}: the env has {have}")
    return out


# --------------------------------------------------------------------------- reading a live env

def conda_installed(prefix: str) -> Dict[str, str]:
    """name -> version of every conda package in ``<prefix>/conda-meta`` (no conda process is run)."""
    out: Dict[str, str] = {}
    for path in glob.glob(os.path.join(prefix, "conda-meta", "*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
            out[norm(str(meta["name"]))] = str(meta["version"])
        except (OSError, ValueError, KeyError):
            continue
    return out


_PIP_PROBE = ("import json, importlib.metadata as m; "
              "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() if d.metadata['Name']}))")


def pip_installed(python: Optional[str] = None) -> Dict[str, str]:
    """name -> version of every pip distribution of ``python`` (default: this interpreter, in-process)."""
    if python is None or os.path.realpath(python) == os.path.realpath(sys.executable):
        import importlib.metadata as md
        raw = {str(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}
    else:
        res = subprocess.run([python, "-c", _PIP_PROBE], capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError(f"could not list {python}'s packages: {res.stderr.strip()[-300:]}")
        raw = json.loads(res.stdout)
    return {norm(k): v for k, v in raw.items()}


def prefix_of(python: Optional[str]) -> str:
    """The env prefix of ``python`` (``<prefix>/bin/python3``); this interpreter's by default."""
    if python is None:
        return sys.prefix
    return os.path.dirname(os.path.dirname(os.path.abspath(python)))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="exit 0 iff the env already satisfies every pin of the file (read-only)")
    c.add_argument("--env-file", required=True)
    c.add_argument("--python", default=None, help="the env's interpreter (default: this one)")
    a = p.parse_args(argv)
    with open(a.env_file, encoding="utf-8") as fh:
        text = fh.read()
    prefix = prefix_of(a.python)
    bad = unsatisfied(text, pip_installed(a.python), conda_installed(prefix))
    n = len(conda_pins(text)) + len(pip_pins(text))
    if bad:
        print(f"{len(bad)} of {n} pins of {a.env_file} are NOT met by {prefix}:")
        for line in bad:
            print(f"  - {line}")
        return 1
    print(f"all {n} pins of {a.env_file} are already met by {prefix} (nothing to install)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
