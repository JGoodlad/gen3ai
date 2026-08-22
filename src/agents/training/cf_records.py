"""The TRAINING-side reconstruction-record TAP — a bounded on-disk ring of recent episodes.

The rust/node bridge emits a ``__RECON__`` frame (the referee-view reconstruction record) at the
end of EVERY episode, seeded and seedless alike. Eval keeps those (they become the
``<prefix>_reconstruction.json`` sibling a trace is joined to); **training discards them** — the
per-env ``BridgeSession`` keeps a single-slot raw stash and overwrites it every episode.

That discard is what stops the counterfactual label factory
(``designs/ai_v10/design_counterfactual_value_grounding.md`` §4, component 1) from reaching
training decisions at all: a label needs a REPLAYABLE episode, and a training episode's record
lives for exactly one episode. This module is the smallest thing that fixes it — a **count-capped
ring on disk**, written by each env worker as its episodes end, pruned to the newest ``keep``
files so it can never grow.

**It is OFF unless asked for.** ``BridgeSession`` builds one only when a records dir is threaded in
(``--cf-records``), so a default run writes no file and behaves byte-for-byte as before.

Design notes worth keeping:

* **Crash-safe write** — ``<name>.tmp`` then ``os.replace``, so a reader (an out-of-process label
  producer) never sees a half-written record. ``os.replace`` is atomic within a filesystem.
* **The cap is GLOBAL and the prune is racy-but-safe.** Every env worker writes into the SAME
  directory, so a per-writer cap would multiply by ``n_envs`` (48 x 512 records), and a
  per-writer ring would leak every previous process's files across a launcher restart. Instead
  each writer prunes the whole directory to the newest ``keep`` by filename, and a delete that
  loses the race is swallowed (``FileNotFoundError``). Two workers may briefly overshoot or
  double-delete one file; neither costs anything, and the bound holds across restarts because
  nothing is keyed on the process.
* **Filenames sort by time**: ``<ns>_<pid>_<tag>_reconstruction.json``. ``time.time_ns()`` is
  19 digits for the next ~280 years, so a lexical sort IS a chronological sort — no ``stat`` per
  file during the prune.
* **The artifact shape is the one the offline stack already reads** — ``{**raw, "battle_tag":
  tag}``, identical to ``utils.bridge.reconstruction._write_artifact``, so
  ``ReconstructionRecord.load()`` works on a ring file with no special-casing.

A write failure is WARNED and swallowed, never raised: the tap is instrumentation for a
background label producer, and a full disk must not crash a training run.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

from utils.bridge.reconstruction import RECON_SUFFIX

# Newest-N records kept in the ring. 512 episodes is ~an hour of one env worker's output at
# production pace and comfortably more than a label producer consumes per cycle; the records are
# tens of KB each, so the whole ring is order tens of MB.
DEFAULT_KEEP = 512


class CfRecordRing:
    """A bounded directory of recent training-episode reconstruction records.

    ``keep`` is the GLOBAL cap on the directory, not a per-instance one — see the module
    docstring for why every writer prunes the whole dir.
    """

    def __init__(self, records_dir: str | os.PathLike, keep: int = DEFAULT_KEEP) -> None:
        self.dir = Path(records_dir)
        self.keep = max(1, int(keep))
        self._warned = False
        self.written = 0
        self.pruned = 0
        self.errors = 0
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------------------
    def write_b64(self, battle_tag: Optional[str], payload_b64: str) -> Optional[Path]:
        """Decode a raw ``__RECON__`` payload (the base64 blob AFTER the marker) and ring it.

        Returns the written path, or None if anything went wrong (already warned).
        """
        try:
            raw = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
        except Exception as exc:                                   # pragma: no cover - defensive
            self._warn(f"undecodable __RECON__ payload: {exc}")
            return None
        return self.write_record(battle_tag, raw)

    def write_record(self, battle_tag: Optional[str], raw: dict) -> Optional[Path]:
        """Ring one decoded record. Same on-disk shape as an eval trace's recon sibling."""
        try:
            tag = _safe_tag(battle_tag)
            stem = f"{time.time_ns():019d}_{os.getpid()}_{tag}"
            path = self.dir / f"{stem}{RECON_SUFFIX}"
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump({**raw, "battle_tag": battle_tag or tag}, f)
            os.replace(tmp, path)
            self.written += 1
            self.prune()
            return path
        except Exception as exc:                                   # pragma: no cover - defensive
            self.errors += 1
            self._warn(f"could not ring a reconstruction record: {exc}")
            return None

    # -- pruning ---------------------------------------------------------------------
    def prune(self) -> int:
        """Delete oldest files until at most ``keep`` remain. Returns how many were deleted."""
        try:
            names = sorted(p.name for p in self.dir.iterdir() if p.name.endswith(RECON_SUFFIX))
        except FileNotFoundError:                                  # pragma: no cover - defensive
            return 0
        excess = len(names) - self.keep
        if excess <= 0:
            return 0
        removed = 0
        for name in names[:excess]:
            try:
                (self.dir / name).unlink()
                removed += 1
            except FileNotFoundError:
                # Another env worker pruned the same file first — the whole point of the
                # racy-but-safe design. Not an error.
                pass
            except OSError as exc:                                 # pragma: no cover - defensive
                self._warn(f"could not prune {name}: {exc}")
        self.pruned += removed
        return removed

    # -- diagnostics -----------------------------------------------------------------
    def _warn(self, msg: str) -> None:
        # Once per process: a broken tap should say so, not spam a training log every episode.
        if not self._warned:
            self._warned = True
            print(f"⚠️  [cf_records] {msg} (further warnings suppressed)", flush=True)


def _safe_tag(battle_tag: Optional[str]) -> str:
    """A filename-safe battle tag. Tags look like ``battle-gen3ou-17``; be defensive anyway."""
    if not battle_tag:
        return "untagged"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(battle_tag))[:80]
