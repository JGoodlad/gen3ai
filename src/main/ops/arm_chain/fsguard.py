"""The ONE place the arm chain deletes anything: :func:`remove_under`, a remove guarded by a containing root.

THE CLASS (operator review 10-27): an ``rm`` / ``shutil.rmtree`` on a VARIABLE path is one bad join, one empty
variable or one symlink away from a run dir, ``models/`` or the main checkout. Here a path is removed only when its
real path is strictly INSIDE the caller's ``root`` (never ``root`` itself), ``root`` is an absolute directory that is
not ``/`` or the home directory, and a tree removal is never through a symlink. Every other module calls this;
``fsguard_test::test_the_module_never_removes_a_variable_path_outside_the_guarded_helper`` scans them (CB-28).
"""
from __future__ import annotations

import os
import shutil


class UnsafeRemove(RuntimeError):
    """A removal the guard refused; nothing was removed."""


def remove_under(path: str, root: str, *, tree: bool = False) -> None:
    """Remove ``path`` (a file; a directory tree with ``tree=True``) only if it lies strictly inside ``root``.
    A path that does not exist is a no-op."""
    if not os.path.isabs(root) or not os.path.isdir(root):
        raise UnsafeRemove(f"the guard root {root!r} is not an absolute directory")
    r = os.path.realpath(root)
    if r in ("/", os.path.realpath(os.path.expanduser("~"))):
        raise UnsafeRemove(f"the guard root {root!r} is / or the home directory")
    if not os.path.lexists(path):
        return
    parent = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    real = os.path.join(parent, os.path.basename(os.path.abspath(path)))
    if not real.startswith(r + os.sep) or real == r:
        raise UnsafeRemove(f"{path!r} is not strictly inside {root!r}: NOT removed")
    if tree:
        if os.path.islink(real):
            raise UnsafeRemove(f"{path!r} is a symlink: a tree removal never follows one; NOT removed")
        shutil.rmtree(real)
    else:
        if os.path.isdir(real) and not os.path.islink(real):
            raise UnsafeRemove(f"{path!r} is a directory: pass tree=True")
        os.unlink(real)
