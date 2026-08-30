"""The HARVEST LABEL SCHEMA (v1) — the contract between the harvest producer and every consumer.

One gzipped-JSONL row per labelled state. The producer is ``main.harvest``; the first consumer is
``agents.training.winprob_finetune``. Neither imports the other — both import this.

Why a second schema when ``cf_audit`` already has one
-----------------------------------------------------
``cf_audit``'s v1 (``kind="mc_winprob"``) is a **single-run bias-map** schema: it labels a
stratified sample of one run's decisions with that run's own checkpoint, and its consumer is the
live trainer's label ring. The harvest is a different job with a different join:

* it spans **many runs' traces** scored by **one subject checkpoint**, so ``run`` and the
  subject's ``phi_head`` are separate facts and the row must carry both;
* it targets the **stall tail** rather than a confidence×outcome×turn stratification, so the
  priority that selected the row has to travel with it (a downstream re-weighting that cannot see
  the selection rule is a distribution-shift confound);
* its labels are ``k`` wins of ``n`` rollouts and the consumer fits a **binomial** likelihood, so
  the counts are the payload — a pre-divided rate throws away the precision that makes a k/n label
  worth more than a bit.

Keeping them separate is deliberate. ``cf_audit``'s schema is a shipped contract with a live
consumer; widening it in place to carry harvest-only fields would make every existing reader
tolerate columns that are absent on every row it has ever seen.

The row
-------
The **eleven pinned fields** are the mission contract — present on every row, never renamed::

    run             source run name (which run's traces supplied the state)
    battle_tag      the trace PREFIX relative to the models root — the battle's identity
                    and the join key back to its *_summary.json / *_states.npz / *_reconstruction.json
    decision_idx    the INVOCATION index — the row of states.npz, not the move ordinal
    turn            game turn at the decision
    n_rollouts      n — rollouts actually adjudicated to a terminal (excludes the timeout bucket)
    n_wins          k — wins among those n
    phi_head        the SUBJECT checkpoint's win-prob head reading at this state (re-scored,
                    never the recorded value: the trace's own phi came from a different policy)
    beta_evidence   the subject's CfEvidentialHead precision alpha+beta, or null if it has no head
    beta_mean       alpha/(alpha+beta), or null
    priority        the selection score this row was ranked by (see ``main.harvest``)
    provenance      a dict: opponent, opponent_source, outcome, subject checkpoint, sampler
                    version, seed, timeout count, and the wall-clock the label cost

Plus the **obs locator triple**, which is not optional — without it a consumer cannot fit
anything. It is listed apart because the mission pinned the eleven above and these three are the
mechanism that makes them usable::

    obs_npz         path to the states.npz, relative to the models root
    obs_sha1        sha1 over the obs float32 bytes AS LABELLED
    obs_inline      base64 float32, or null

``obs_sha1`` is always present, so a consumer verifies the row it loaded is the row that was
labelled. That check is not ceremonial: ``cf_audit`` shipped a real bug where ``obs_npz`` rows
ignored ``decision_idx`` and **every** default-mode label was rejected as architecture drift,
because the two halves of the contract each had a test and neither ran the other's real output.
:func:`load_obs` is therefore the ONE function that resolves a row to an array, and both sides
call it.

A timed-out rollout is its own bucket
-------------------------------------
``n_rollouts`` counts adjudicated rollouts only; ``provenance["n_timeout"]`` counts the rest. A
timeout is never scored as a loss — that is the convention probe G fixed and the contention doc
states as a rule, and folding one into ``n_wins``' denominator would make a busy box look like a
losing position.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

#: Bump when a field changes meaning. A reader that does not recognise the version REFUSES.
HARVEST_SCHEMA_VERSION = 1

#: The kind tag, so a directory of mixed label families stays sortable.
HARVEST_KIND = "harvest_winprob"

#: The eleven pinned fields, in order. :func:`validate_row` asserts every one is present, so a
#: producer that drops one fails at write time rather than at the consumer.
PINNED_FIELDS = (
    "run", "battle_tag", "decision_idx", "turn", "n_rollouts", "n_wins",
    "phi_head", "beta_evidence", "beta_mean", "priority", "provenance",
)

#: The obs locator. Not pinned by the mission, required by every consumer.
OBS_FIELDS = ("obs_npz", "obs_sha1", "obs_inline")


@dataclass
class HarvestRow:
    """One harvested state: what it was, what the subject head said, and what the dice said."""

    run: str
    battle_tag: str
    decision_idx: int
    turn: int
    n_rollouts: int
    n_wins: int
    phi_head: Optional[float]
    beta_evidence: Optional[float]
    beta_mean: Optional[float]
    priority: float
    provenance: Dict[str, Any]

    obs_npz: str
    obs_sha1: str
    obs_inline: Optional[str] = None

    schema: int = HARVEST_SCHEMA_VERSION
    kind: str = HARVEST_KIND
    created_unix: float = field(default_factory=time.time)

    @property
    def label(self) -> float:
        """``k/n`` — the Monte-Carlo win rate. ``nan`` when nothing adjudicated, which
        :func:`validate_row` forbids on a written row."""
        return (self.n_wins / self.n_rollouts) if self.n_rollouts else float("nan")

    def to_json(self) -> dict:
        d = asdict(self)
        return {"schema": d.pop("schema"), "kind": d.pop("kind"), **d}


def obs_digest(obs: np.ndarray) -> str:
    """sha1 over the float32 bytes — byte-identical to ``cf_audit.obs_digest`` on purpose, so a
    row from either factory can be checked by either reader."""
    return hashlib.sha1(np.ascontiguousarray(obs, dtype=np.float32).tobytes()).hexdigest()


def obs_b64(obs: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(obs, dtype=np.float32).tobytes()).decode()


def validate_row(row: dict) -> None:
    """Raise ``ValueError`` unless ``row`` satisfies the contract. Called on every write AND on
    every read — a schema whose producer checks it and whose consumer trusts it is a schema with
    one enforcement point and two beliefs."""
    if row.get("schema") != HARVEST_SCHEMA_VERSION:
        raise ValueError(f"unknown harvest schema {row.get('schema')!r} "
                         f"(this reader speaks {HARVEST_SCHEMA_VERSION})")
    if row.get("kind") != HARVEST_KIND:
        raise ValueError(f"not a harvest row: kind={row.get('kind')!r}")
    missing = [k for k in PINNED_FIELDS + OBS_FIELDS if k not in row]
    if missing:
        raise ValueError(f"harvest row missing pinned field(s): {missing}")
    n, k = row["n_rollouts"], row["n_wins"]
    if not isinstance(n, int) or not isinstance(k, int):
        raise ValueError(f"n_rollouts/n_wins must be ints, got {type(n)}/{type(k)}")
    if n <= 0:
        raise ValueError("n_rollouts must be > 0 — a row with nothing adjudicated is not a label")
    if not (0 <= k <= n):
        raise ValueError(f"n_wins {k} out of range for n_rollouts {n}")
    if not isinstance(row["provenance"], dict):
        raise ValueError("provenance must be a dict")


def write_rows(rows: Sequence[HarvestRow], path: str) -> str:
    """Write gzipped JSONL to ``path`` (``.jsonl.gz``), atomically.

    Atomic because a harvest is hours long and a reader that finds a half-written shard cannot
    tell it from a short one — ``cf_audit`` shipped a ``.tmp`` leak in this exact spot, so the
    temp file is removed on any failure rather than left behind on a full disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with gzip.open(tmp, "wt") as fh:
            for r in rows:
                d = r.to_json()
                validate_row(d)
                fh.write(json.dumps(d) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_rows(path: str) -> Iterator[dict]:
    """Stream validated rows out of one ``.jsonl.gz`` shard."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            validate_row(row)
            yield row


def read_dir(path: str) -> List[dict]:
    """Every row under a harvest directory (or a single shard file), in a stable shard order."""
    if os.path.isfile(path):
        return list(read_rows(path))
    shards = sorted(
        os.path.join(path, f) for f in os.listdir(path)
        if f.endswith(".jsonl.gz") or f.endswith(".jsonl"))
    out: List[dict] = []
    for s in shards:
        out.extend(read_rows(s))
    return out


def load_obs(row: dict, models_root: Optional[str] = None,
             npz_cache: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Resolve one row to its observation vector, verifying ``obs_sha1``.

    Inline first (self-contained rows need no archive at all), else ``obs_npz`` **indexed by
    ``decision_idx``** — the indexing bug ``cf_audit`` shipped is exactly here, so the digest check
    below is what makes the two paths provably the same array rather than two beliefs about it.

    Raises ``ValueError`` on a digest mismatch: a row whose obs cannot be reproduced is a row that
    must not be trained on.
    """
    inline = row.get("obs_inline")
    if inline:
        arr = np.frombuffer(base64.b64decode(inline), dtype=np.float32)
    else:
        p = row["obs_npz"]
        if not os.path.isabs(p):
            if models_root is None:
                raise ValueError(
                    f"row {row['battle_tag']} carries a RELATIVE obs_npz and no models_root was "
                    "given — pass models_root=, or harvest with --inline-obs")
            p = os.path.join(models_root, p)
        arr = None
        if npz_cache is not None and p in npz_cache:
            arr = npz_cache[p][row["decision_idx"]]
        else:
            with np.load(p) as z:
                full = np.asarray(z["obs"], dtype=np.float32)
            if npz_cache is not None:
                npz_cache[p] = full
            arr = full[row["decision_idx"]]
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    got = obs_digest(arr)
    if got != row["obs_sha1"]:
        raise ValueError(
            f"obs digest mismatch for {row['battle_tag']}#{row['decision_idx']}: "
            f"row says {row['obs_sha1'][:12]}, loaded array hashes {got[:12]} — the array on disk "
            "is not the array that was labelled (wrong row index, or the trace was rewritten)")
    return arr
