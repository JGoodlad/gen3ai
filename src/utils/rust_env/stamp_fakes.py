"""Stamps and fake Rust binaries for TESTS of everything that judges a build by its stamp (``stamp.py``).

A resolver / spawner test that needs "a binary" without paying a cargo build writes one of these: an
executable script that answers ``--build-stamp`` with the stamp of THIS tree (or any deliberately wrong
one) and exits 0 otherwise. The check stays ACTIVE in those tests — a stub that skipped it would prove
nothing about the refusal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.rust_env import columns
from utils.rust_env import stamp as S


def make_stamp(kind: str = "port", *, tree: Optional[Path] = None, **over: str) -> str:
    """A well-formed stamp of ``kind`` for the ``src/`` tree at ``tree`` (default: this checkout), any field overridable
    (``src=`` is the hash field)."""
    h, n = S.source_hash(tree, kind)
    fields = dict(stamp=S.STAMP_VERSION, kind=kind, commit="c0ffee", src=h, nfiles=str(n))
    if kind == "env":
        fields.update(nan_poison="1", schema=columns.schema_id(), data=S.data_dir())
    else:
        fields.update(selfcheck="1")
    fields.update(over)
    return ";".join(f"{k}={v}" for k, v in fields.items())


def write_fake_binary(path: Path, stamp: Optional[str]) -> Path:
    """An executable at ``path`` that prints ``stamp`` for ``--build-stamp`` (nothing at all when ``None``: a binary
    that predates the flag) and exits 0 for any other argv. Returns ``path``."""
    answer = f"printf '%s\\n' '{stamp}'" if stamp is not None else ":"
    path.write_text(f'#!/bin/sh\nif [ "$1" = "{S.BUILD_STAMP_FLAG}" ]; then {answer}; fi\nexit 0\n')
    path.chmod(0o755)
    return path
