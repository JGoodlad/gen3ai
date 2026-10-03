"""The JOIN between a rollout-buffer row and a reconstruction record on disk — what is left of the
`<run>/cf_records/` ring after deletion pass L4.

The ring itself (`CfRecordRing`, `--cf-records`, the env-worker tap and the label buffer that
consumed what a producer made of it) was the counterfactual TRAINING half and is DELETED; no run
writes a ring any more. What stays, until the Python fork arm goes in L5, is the two functions that
name the join so the arm's call sites and the wrapper's decision-time handle capture keep one
spelling of the key:

* :func:`record_key` — the `<pid>_<tag>` handle an env worker publishes per decision;
* :func:`index_records` — `{record_key: path}` over a directory of `<ns>_<pid>_<tag>_reconstruction.json`
  files (the ring's filename convention: `time.time_ns()` is 19 digits, so a lexical sort IS
  chronological).

**DELETE THIS FILE WITH THE PYTHON FORK ARM (L5)** — its only readers are `fork_callback` and
`wrappers.MaskableAgentWrapper`'s handle capture, both of which go with it.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from utils.bridge.reconstruction import RECON_SUFFIX


def safe_tag(battle_tag: Optional[str]) -> str:
    """A filename-safe battle tag. Tags look like ``battle-gen3ou-17``; be defensive anyway.

    PUBLIC because the filename is a JOIN KEY, not just a name: :func:`record_key` builds the same
    ``<pid>_<tag>`` handle in the env worker so a rollout-buffer row can be matched back to its record
    here. Two spellings of the sanitiser would make that join silently miss on exactly the tags that
    needed sanitising.
    """
    if not battle_tag:
        return "untagged"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(battle_tag))[:80]


def record_key(pid: int, battle_tag: Optional[str]) -> str:
    """The handle that joins a BUFFER ROW to a reconstruction record on disk.

    A ring file is named ``<ns:019d>_<pid>_<tag>_reconstruction.json``, so ``<pid>_<tag>``
    identifies the episode uniquely: the tag counter is per BridgeSession (it repeats across env
    workers) and the pid disambiguates the workers. Captured in the env worker, where both halves
    are known, and matched by SUFFIX in :func:`index_records`.
    """
    return f"{int(pid)}_{safe_tag(battle_tag)}"


def index_records(records_dir) -> Dict[str, str]:
    """``{record_key: path}`` over a record directory, NEWEST wins.

    One ``readdir`` per rollout, not one ``stat`` per handle: the filenames sort chronologically by
    construction, so the newest record for a key is simply the last one seen in sorted order.
    """
    out: Dict[str, str] = {}
    try:
        names = sorted(os.listdir(str(records_dir)))
    except OSError:
        return out
    for name in names:
        if not name.endswith(RECON_SUFFIX):
            continue
        stem = name[: -len(RECON_SUFFIX)]
        # `<ns>_<pid>_<tag>` — drop the 19-digit timestamp, keep `<pid>_<tag>`.
        cut = stem.find("_")
        if cut < 0:
            continue
        out[stem[cut + 1:]] = os.path.join(str(records_dir), name)
    return out
