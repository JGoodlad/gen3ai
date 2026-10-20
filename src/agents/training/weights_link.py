"""Store a weights file as a HARD LINK to a byte-identical twin, else as a copy.

THE WASTE. A run keeps the same ~62 MB of weights under several names: the periodic checkpoint
(`checkpoints/checkpoint_<N>_steps.zip`), the eval cycle's persisted snapshot
(`eval_traces/step_<N>/snapshot.zip`), the best-model export (`best_model/best_model.zip`) and the
self-play pool entry (`snapshots/snapshot_<N>.zip`). The periodic checkpointer is first in the callback
list and the eval callbacks fire on the SAME call whenever their cadences coincide, both naming the file
by `num_timesteps`, so at a coincident step the cycle's frozen snapshot IS the checkpoint's bytes. The
eval snapshot was the first to link (`eval_collect.store_eval_snapshot`, measured on a finished run: 416 MB
of 484 MB of eval traces were duplicated weights); the best-model copy and the pool copy are the same
waste and use the same rule.

THE RULE. `link_or_copy(src, dst, twin)`: ``dst`` becomes a second NAME for ``twin``'s inode when - and only
when - ``twin`` exists, is on the SAME filesystem as ``dst``'s directory and is BYTE-IDENTICAL to ``src``
(size first, then sha256 of both); otherwise ``dst`` is a copy of ``src``. A hard link survives deletion of
the other name (retention frees the blocks only when the LAST name goes) and is an ordinary file path to
every reader.

🚨 THE HAZARD, CLOSED HERE AND AT EVERY WRITER. ``dst`` is replaced ATOMICALLY (`utils.atomic_io`: a temp file, then
``os.replace``) and never opened for writing: an existing ``dst`` may already be a link to a checkpoint (a restart
re-persisting the same step), and a plain ``shutil.copy2`` onto it would truncate the CHECKPOINT through the shared
inode. The mirror image - an in-place ``model.save(path)`` opening a name that shares an inode - cannot happen
either: the checkpointer, the pool's live ``add`` and the eval snapshots all save through
``utils.atomic_io.atomic_save_model``, which gives the name a fresh inode, so the other names keep their bytes and a
torn write corrupts none of them.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Optional, Tuple

from utils.atomic_io import atomic_write

#: The note a copy carries when there was no twin to link to (eval and checkpoint cadences do not always
#: coincide, so this is normal and must not warn).
NO_TWIN_NOTE = "no periodic checkpoint at this step (eval and checkpoint cadences do not coincide here)"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def link_or_copy(src: str, dst: str, twin: Optional[str], *, no_twin_note: str = NO_TWIN_NOTE) -> Tuple[str, str, str]:
    """Put ``src``'s bytes at ``dst``; returns ``(mode, sha256, note)``.

    ``mode`` is ``"hardlink"`` when ``twin`` is a byte-identical file on the same filesystem (``dst`` is then a
    second name for ``twin``'s inode, not a duplicate of it), else ``"copy"`` and ``note`` says why a link was not
    made. ``sha256`` is the digest of ``src`` when it was computed (a link attempt got that far), else ``""``.
    ``dst`` is replaced atomically and never opened for writing (see the module docstring)."""
    mode, note = "copy", no_twin_note
    digest = ""
    if twin is not None:
        try:
            if os.stat(twin).st_size != os.stat(src).st_size:
                note = "the checkpoint at this step is not the same size as the snapshot"
            elif os.stat(twin).st_dev != os.stat(os.path.dirname(dst)).st_dev:
                note = "the checkpoint is on another filesystem"
            else:
                digest = sha256_file(src)
                if sha256_file(twin) != digest:
                    note = "the checkpoint at this step is not byte-identical to the snapshot"
                else:
                    atomic_write(dst, lambda tmp: os.link(twin, tmp))
                    return "hardlink", digest, "hard link to " + os.path.join("checkpoints", os.path.basename(twin))
        except OSError as e:
            note = f"linking failed ({e})"
    atomic_write(dst, lambda tmp: shutil.copy2(src, tmp))
    return mode, digest, note
