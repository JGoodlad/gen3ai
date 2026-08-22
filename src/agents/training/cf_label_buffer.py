"""The counterfactual LABEL BUFFER — the training-side consumer of an offline label producer.

``designs/ai_v10/design_counterfactual_value_grounding.md`` §4 component 4, rung **R1**: a
background producer manufactures tight Monte-Carlo win-probability labels by re-rolling recorded
training decisions through the reconstruction stack, drops them as JSONL in
``<run_dir>/cf_labels/``, and this buffer feeds them to the win-prob head's auxiliary BCE.

**Producer and consumer are separate processes that share only a file format.** That is the whole
interface — there is no IPC, no socket, no shared memory — so the producer can be restarted,
rewritten, or absent entirely without the trainer noticing anything except an empty buffer.

The schema (v1), one JSON object per line::

    {"schema": 1, "kind": "mc_winprob", "battle": str, "decision_idx": int,
     "obs_sha1": "<sha1 hex of the obs float32 bytes>",
     "obs_npz": "<path>::<key>" | null, "obs_inline": "<base64 float32 raw>" | null,
     "label": <float in [0,1]>, "n_rollouts": int, "wilson_lo": float, "wilson_hi": float,
     "policy_step": int, "opponent": str, "created_unix": float}

Obs resolution order is ``obs_inline`` > ``obs_npz`` > skip-with-a-counter. An unknown ``kind`` or
a ``schema`` != 1 is SKIPPED with a counter, never a crash — a producer that ships a v2 row must
not take a training run down with it.

**Staleness is bounded, not denied** (design §5.2). An MC label is a sample under the policy that
produced it; consumed many iterations later it teaches an ancestor's value. Rows older than
``lag_bound`` policy steps are dropped on ingest and on every sample, and the drop is COUNTED —
`cf/labels_expired_total` rising while `cf/buffer_fill` sits at zero is the signature of a
producer that is running but lagging, which reads completely differently from one that is dead.

**Producer liveness is a first-class scalar, deliberately.** The oldest failure mode in this tree
is an empty buffer that does not announce itself (the search-teacher's silent starvation), so the
buffer publishes five counters every log cycle:

===============================  ================================================================
``cf/buffer_fill``               rows currently resident (0 = starving RIGHT NOW)
``cf/label_age_steps_p50``       median staleness of the resident rows, in policy steps
``cf/labels_ingested_total``     rows accepted since process start (flat = producer is DEAD)
``cf/labels_expired_total``      rows dropped for exceeding ``lag_bound``
``cf/labels_skipped_total``      rows rejected — bad schema, unknown kind, unresolvable or
                                 malformed obs, sha1 mismatch, out-of-range label
===============================  ================================================================

``skipped`` is the GIGO meter: a producer whose obs bytes disagree with their own ``obs_sha1``, or
whose obs dimension does not match this run's, is feeding the critic garbage. The buffer refuses
those rows and says so loudly ONCE, rather than training on them or crashing the run.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 1
KIND_MC_WINPROB = "mc_winprob"
_KNOWN_KINDS = frozenset({KIND_MC_WINPROB})

# Default residency cap. Sized well above one PPO update's sample (256) so a healthy producer
# keeps the sampler saturated, and small enough that the buffer's memory is bounded by
# `capacity * obs_dim * 4` bytes (~25 MB at 2501 dims / 2048 rows).
DEFAULT_CAPACITY = 2048
# Default staleness bound in POLICY STEPS — order one PPO iteration at production shapes
# (n_envs 64 x n_steps 2048 = 131k), so a label is consumed by roughly the policy that produced it.
DEFAULT_LAG_BOUND = 150_000
# How many rows one PPO fold samples. See `instrumented_ppo._cf_winprob_term`.
CF_SAMPLE_SIZE = 256

_LABEL_GLOB = "labels_*.jsonl"


@dataclass(frozen=True)
class CfLabel:
    """One resolved counterfactual label: an observation vector and its ground-truth target."""
    obs: np.ndarray            # [obs_dim] float32
    label: float               # in [0, 1]
    policy_step: int           # the num_timesteps the producing policy was at
    battle: str
    decision_idx: int
    opponent: str
    n_rollouts: int


class CfLabelBuffer:
    """FIFO buffer over ``<labels_dir>/labels_*.jsonl`` with staleness expiry.

    ``poll(current_step)`` scans the directory and ingests any bytes appended since the last poll
    (per-file byte OFFSETS are remembered, so a producer may append to a file it already wrote and
    the buffer picks up only the new lines — and a partial final line is left for the next poll
    rather than being parsed as garbage).
    """

    def __init__(
        self,
        labels_dir: str | os.PathLike,
        *,
        obs_dim: Optional[int] = None,
        capacity: int = DEFAULT_CAPACITY,
        lag_bound: int = DEFAULT_LAG_BOUND,
    ) -> None:
        self.dir = Path(labels_dir)
        self.obs_dim = int(obs_dim) if obs_dim else None
        self.capacity = max(1, int(capacity))
        self.lag_bound = max(0, int(lag_bound))
        self._rows: "deque[CfLabel]" = deque()
        # name -> (inode, byte offset already consumed). The INODE is half the key on purpose: a
        # producer that DELETES and RECREATES `labels_x.jsonl` (a restart that rotates in place)
        # gets a new inode, and keying on the name alone would seek past the new file's first
        # `offset` bytes and drop those rows SILENTLY — no skip counter, no warning, just missing
        # labels. Measured before this was a tuple: a recreated 3-row file ingested 1 row.
        self._offsets: Dict[str, tuple] = {}
        # Counters — monotonic for the whole process lifetime, so a TB curve of
        # `labels_ingested_total` going FLAT is unambiguous evidence the producer stopped.
        self.ingested_total = 0
        self.expired_total = 0
        self.skipped_total = 0
        # Skip BREAKDOWN (not on TensorBoard; read from `stats()` in a test or a debugger). The
        # single `skipped_total` scalar answers "is the producer feeding me garbage"; this answers
        # "which garbage", which is a debugging question, not a monitoring one.
        self.skip_reasons: Dict[str, int] = {}
        self._warned: set = set()
        self._rng = np.random.default_rng(0)

    # -- ingest ----------------------------------------------------------------------
    def poll(self, current_step: int) -> int:
        """Ingest every new row on disk. Returns the number of rows ACCEPTED this poll."""
        if not self.dir.is_dir():
            return 0
        accepted = 0
        seen = set()
        for path in sorted(self.dir.glob(_LABEL_GLOB)):
            seen.add(path.name)
            accepted += self._ingest_file(path, current_step)
        # Forget files that are no longer on disk, so the offset map tracks the DIRECTORY rather
        # than growing one entry per label file ever seen for the life of a multi-day run.
        for gone in [k for k in self._offsets if k not in seen]:
            del self._offsets[gone]
        self.expire(current_step)
        return accepted

    def _ingest_file(self, path: Path, current_step: int) -> int:
        key = path.name
        try:
            st = path.stat()
        except OSError:                                            # pragma: no cover - defensive
            return 0
        size = st.st_size
        prev_ino, start = self._offsets.get(key, (st.st_ino, 0))
        if prev_ino != st.st_ino:
            # A DIFFERENT FILE now wears this name (the producer rotated/recreated it). Its bytes
            # have nothing to do with the offset we remembered — read it from the beginning.
            start = 0
        if size < start:
            # The file was TRUNCATED or replaced under us (a producer restart). Re-read it whole
            # rather than seeking past its new end and silently ingesting nothing forever.
            start = 0
        if size == start:
            return 0
        try:
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read()
        except OSError as exc:                                     # pragma: no cover - defensive
            self._warn(f"unreadable {key}: {exc}")
            return 0
        # A producer appending line-by-line may be caught mid-line; keep the tail for next poll.
        cut = data.rfind(b"\n")
        if cut < 0:
            return 0
        # Everything up to the last newline is complete; the remainder (if any) is a line the
        # producer has not finished writing, and is deliberately NOT advanced past.
        consumed = data[: cut + 1]
        self._offsets[key] = (st.st_ino, start + len(consumed))
        accepted = 0
        for line in consumed.splitlines():
            if not line.strip():
                continue
            row = self._parse(line, current_step)
            if row is None:
                continue
            self._push(row)
            accepted += 1
        return accepted

    def _parse(self, line: bytes, current_step: int) -> Optional[CfLabel]:
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:
            return self._skip("malformed_json")
        if not isinstance(obj, dict):
            return self._skip("not_an_object")
        if obj.get("schema") != SCHEMA_VERSION:
            # Tolerated by contract: a v0/v2 producer is not an error, it is a version we do not
            # consume. Counted so the operator can see it happening.
            return self._skip("schema")
        if obj.get("kind") not in _KNOWN_KINDS:
            return self._skip("kind")
        try:
            label = float(obj["label"])
            policy_step = int(obj["policy_step"])
        except Exception:
            return self._skip("missing_fields")
        if not (0.0 <= label <= 1.0) or not np.isfinite(label):
            return self._skip("label_range")
        # Staleness at INGEST as well as at sample: a producer that fell far behind should not be
        # able to flood the buffer with rows that would be dropped on the very next expire().
        if self.lag_bound and (current_step - policy_step) > self.lag_bound:
            self.expired_total += 1
            return None
        obs = self._resolve_obs(obj)
        if obs is None:
            return None
        self.ingested_total += 1
        return CfLabel(
            obs=obs, label=label, policy_step=policy_step,
            battle=str(obj.get("battle", "")), decision_idx=int(obj.get("decision_idx", -1)),
            opponent=str(obj.get("opponent", "")), n_rollouts=int(obj.get("n_rollouts", 0)),
        )

    def _resolve_obs(self, obj: dict) -> Optional[np.ndarray]:
        """obs_inline > obs_npz > skip-with-counter (the schema's declared resolution order)."""
        arr: Optional[np.ndarray] = None
        inline = obj.get("obs_inline")
        if inline:
            try:
                arr = np.frombuffer(base64.b64decode(inline), dtype=np.float32)
            except Exception:
                return self._skip("obs_inline_undecodable")
        elif obj.get("obs_npz"):
            arr = self._load_npz(str(obj["obs_npz"]))
            if arr is None:
                return None
        else:
            return self._skip("obs_unresolvable")
        if arr is None or arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            return self._skip("obs_malformed")
        if self.obs_dim is not None and arr.size != self.obs_dim:
            # A GIGO guard, not a nicety: an obs of the wrong width would be silently reshaped or
            # would crash deep inside the extractor's slicing with an unreadable error.
            self._warn(f"obs dim {arr.size} != this run's {self.obs_dim} — producer/consumer "
                       f"architecture drift; rows rejected")
            return self._skip("obs_dim")
        want = obj.get("obs_sha1")
        if want:
            got = hashlib.sha1(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()
            if got != want:
                self._warn("obs_sha1 MISMATCH — the producer's obs bytes disagree with its own "
                           "digest; rows rejected (GIGO guard)")
                return self._skip("obs_sha1")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _load_npz(self, spec: str) -> Optional[np.ndarray]:
        if "::" not in spec:
            return self._skip("obs_npz_spec")
        path, _, key = spec.partition("::")
        try:
            with np.load(path) as z:
                if key not in z:
                    return self._skip("obs_npz_key")
                return np.asarray(z[key], dtype=np.float32).reshape(-1)
        except Exception:
            return self._skip("obs_npz_unreadable")

    def _push(self, row: CfLabel) -> None:
        self._rows.append(row)
        while len(self._rows) > self.capacity:
            self._rows.popleft()               # FIFO: the OLDEST label is the one to lose

    def _skip(self, reason: str) -> None:
        self.skipped_total += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        return None

    def _warn(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            print(f"⚠️  [cf_labels] {msg}", flush=True)

    # -- expiry / sampling -----------------------------------------------------------
    def expire(self, current_step: int) -> int:
        """Drop rows staler than ``lag_bound`` policy steps. Returns how many were dropped.

        The boundary is INCLUSIVE of the bound: age == lag_bound survives, age == lag_bound + 1
        does not. ``lag_bound == 0`` disables expiry entirely (nothing is ever too old).
        """
        if not self.lag_bound:
            return 0
        dropped = 0
        keep = deque()
        for row in self._rows:
            if (current_step - row.policy_step) > self.lag_bound:
                dropped += 1
            else:
                keep.append(row)
        if dropped:
            self._rows = keep
            self.expired_total += dropped
        return dropped

    def sample(self, n: int) -> List[CfLabel]:
        """Up to ``n`` rows drawn uniformly WITHOUT replacement. Empty when starving."""
        if not self._rows or n <= 0:
            return []
        rows = list(self._rows)
        if len(rows) <= n:
            return rows
        idx = self._rng.choice(len(rows), size=n, replace=False)
        return [rows[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self._rows)

    # -- metrics ---------------------------------------------------------------------
    def stats(self, current_step: int) -> Dict[str, float]:
        """The five `cf/` TensorBoard scalars (see the module docstring)."""
        ages = [current_step - r.policy_step for r in self._rows]
        return {
            "cf/buffer_fill": float(len(self._rows)),
            "cf/label_age_steps_p50": float(np.median(ages)) if ages else 0.0,
            "cf/labels_ingested_total": float(self.ingested_total),
            "cf/labels_expired_total": float(self.expired_total),
            "cf/labels_skipped_total": float(self.skipped_total),
        }


def batch_tensors(rows: Sequence[CfLabel], device) -> "tuple":
    """``(obs [B, obs_dim], labels [B], n_rollouts [B])``, all float32 on ``device``.

    ``n_rollouts`` rides along because the label is a SUFFICIENT STATISTIC only in company: a
    ``label`` of 0.75 from 4 rollouts and one from 16 are the same number carrying four times the
    evidence, and the row's win COUNT is recoverable as ``round(label · n_rollouts)``. The binomial
    likelihood (`instrumented_ppo._cf_binomial_nll`) and the Beta-Binomial evidential loss both
    need the count, not the ratio.

    A row whose producer omitted ``n_rollouts`` parses as 0; the consumers clamp to 1, so an
    unlabelled-count row degrades to exactly one observation rather than vanishing or dividing by
    zero. Torch imported lazily.
    """
    import torch as th
    obs = np.stack([r.obs for r in rows]).astype(np.float32, copy=False)
    lab = np.asarray([r.label for r in rows], dtype=np.float32)
    n = np.asarray([r.n_rollouts for r in rows], dtype=np.float32)
    return (th.as_tensor(obs, device=device), th.as_tensor(lab, device=device),
            th.as_tensor(n, device=device))
