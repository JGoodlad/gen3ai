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

🚨 TWO HAZARDS, BOTH CLOSED HERE.
* ``dst`` is replaced ATOMICALLY (build ``dst.tmp``, then ``os.replace``) and never opened for writing: an
  existing ``dst`` may already be a link to a checkpoint (a restart re-persisting the same step), and a
  plain ``shutil.copy2`` onto it would truncate the CHECKPOINT through the shared inode.
* The mirror image - an IN-PLACE writer (``model.save(path)``: the checkpointer, the pool's live ``add``)
  opening a name that now shares its inode would rewrite every other name with it, and a torn write would
  corrupt all of them at once. :func:`unshare` removes THAT name first when it has more than one, so the
  writer creates a fresh inode and the other names keep their bytes.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Optional, Tuple

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
    tmp = dst + ".tmp"
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass
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
                    os.link(twin, tmp)
                    os.replace(tmp, dst)
                    # POSIX rename() of one hard link onto ANOTHER NAME OF THE SAME INODE does nothing and
                    # leaves the source: a re-persist over an existing link would strand `dst.tmp` (a third name).
                    if os.path.lexists(tmp):
                        os.remove(tmp)
                    return "hardlink", digest, "hard link to " + os.path.join("checkpoints", os.path.basename(twin))
        except OSError as e:
            note = f"linking failed ({e})"
            try:
                os.remove(tmp)
            except OSError:
                pass
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return mode, digest, note


def unshare(path: str) -> bool:
    """Remove ``path`` when it is one of several names for its inode (``st_nlink > 1``); True when it did.

    Call this before an IN-PLACE writer (``model.save(path)``) opens ``path``: the writer then creates a fresh
    inode and the other names (a checkpoint, a best-model export, a pool entry, an eval snapshot) keep the bytes
    they were linked for. A single-name file is left alone (the writer overwrites it as it always did)."""
    try:
        if os.stat(path).st_nlink > 1:
            os.remove(path)
            return True
    except FileNotFoundError:
        pass
    return False
