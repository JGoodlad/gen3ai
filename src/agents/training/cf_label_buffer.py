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
     "policy_step": int, "opponent": str, "created_unix": float,
     # -- v1 ADDITIVE-OPTIONAL (gen3_cf_twin_heads_v1); absent on any older producer's rows --
     "outcome_label": <float in [0,1]> | null,      # the RECORDED battle's realized outcome
     "mc_return": float | null, "mc_return_n": int,  # mean realized SHAPED return over R rollouts
     "reward_sha1": "<sha1 hex of the producing RewardConfig>" | null}

Obs resolution order is ``obs_inline`` > ``obs_npz`` > skip-with-a-counter. An unknown ``kind`` or
a ``schema`` != 1 is SKIPPED with a counter, never a crash — a producer that ships a v2 row must
not take a training run down with it.

**THE THREE LABEL STREAMS RIDE ONE ROW, and that is a decision, not a convenience**
(`gen3_cf_twin_heads_v1`). The twin-heads amendment needs a SINGLE-OUTCOME label and a TIGHT-MC
label for the *same state*, plus an ``mc_return`` for the shadow critic. The obvious alternative —
a second row per state carrying ``kind: "outcome_winprob"`` — is **structurally unavailable here**:
this buffer DEDUPS on the obs digest (keep-newest), so two rows describing one state would collide
on ``obs_sha1`` and one would silently replace the other. Widening the dedup key to
``(kind, obs_sha1)`` would fix the collision and break something better: the twin design's whole
premise is that heads B and C see *identical* states, which one-row-per-state makes structural
rather than hoped-for. So the extra streams are SIBLING FIELDS.

**The schema version does NOT move for them.** ``schema`` is a REFUSAL gate — a consumer skips
every row whose version it does not know — so bumping it to 2 would make a new producer's output
unreadable by an existing trainer, which is the opposite of backward compatible. Additive-optional
fields at v1 are compatible in both directions: an old consumer reads a fixed key set and ignores
them; a new consumer reads them when present and simply supervises nothing extra when they are not.

**``mc_return`` carries a REWARD DIGEST and is REFUSED on a mismatch.** A shaped return is a fact
about a board *under a reward composition*, so a label produced under a different `RewardConfig`
is not a noisier measurement of this run's value function — it is a measurement of a different one.
The buffer therefore drops the ``mc_return`` FIELD (never the row: the win-prob labels on it are
still perfectly good) when the digest disagrees with the run's, counts it as
``cf/labels_mc_return_rejected_total`` and warns once by name.

**Staleness is bounded SYMMETRICALLY, not denied** (design §5.2). An MC label is a sample under the
policy that produced it; consumed many iterations later it teaches an ancestor's value. The bound
is on ``abs(current_step - policy_step)``, and the absolute value is load-bearing: a **crash-restart
rollback** resumes from the last checkpoint, so `num_timesteps` moves BACKWARDS while the label
files on disk still carry the pre-crash steps. Those rows are FUTURE-dated relative to the running
process, and a one-sided ``current - policy > bound`` test makes them immortal — they never expire,
never refresh, and quietly become the whole buffer. The tell was measured:
``cf/label_age_steps_p50`` reading **-4,999,000**. A future label is expired like a past one AND
counted separately (`cf/labels_future_total`), with a one-time loud warning naming the cause,
because a negative age is a *diagnosis* (someone restarted from an older checkpoint), not noise.

**Rows are DEDUPED on the obs digest, keep-NEWEST.** A producer that re-labels a decision it has
already shipped (a re-run over the same trace tree, a truncate-and-rewrite, an overlapping cycle)
would otherwise give that one state N× the weight of every other — a silent, unannounced change to
the sampler's declared distribution, which design decision-of-record 3 forbids. The resident row
for a repeated ``obs_sha1`` is REPLACED by the arrival, never appended beside it. Keep-NEWEST
rather than keep-first because a fresher label is a strictly better estimate of the same state: it
was measured under a policy closer to the one now consuming it, and if the producer changed R it
carries more evidence. Counted as `cf/labels_replaced_total`.

**Producer liveness is a first-class scalar, deliberately.** The oldest failure mode in this tree
is an empty buffer that does not announce itself (the search-teacher's silent starvation), so the
buffer publishes seven counters every log cycle:

===============================  ================================================================
``cf/buffer_fill``               rows currently resident (0 = starving RIGHT NOW)
``cf/label_age_steps_p50``       median staleness of the resident rows, in policy steps
``cf/labels_ingested_total``     rows accepted since process start (flat = producer is DEAD)
``cf/labels_expired_total``      rows dropped for |age| > ``lag_bound`` (past AND future)
``cf/labels_future_total``       of those, the ones dated AHEAD of this process (restart rollback)
``cf/labels_replaced_total``     rows superseded by a newer label of the SAME state
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
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

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

# How many `obs_npz` FILES to keep decoded in memory. A label row points at one row of a battle's
# `states.npz`, and a producer emits many rows per battle — so a per-row `np.load` re-opened,
# re-inflated and re-parsed the same archive dozens of times per poll. The cache is tiny and
# deliberately so: an entry is one battle's obs matrix (~50 x 2501 x 4B ≈ 0.5 MB), and label files
# are written battle-by-battle, so a 4-deep LRU already collapses the reopens to ~one per file.
_NPZ_CACHE_FILES = 4


def _digest(arr: np.ndarray) -> str:
    """sha1 over the float32 bytes — the row identity, and the dedup key."""
    return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


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
    # sha1 of `obs`'s float32 bytes — the DEDUP key. Computed here from the bytes actually loaded,
    # never copied from the row's declared `obs_sha1`: a producer whose digest disagrees with its
    # own bytes is rejected upstream, and a key taken from an unverified field would let two
    # different states collide (or one state fail to dedup with itself).
    obs_sha1: str = ""
    # gen3_cf_twin_heads_v1: the two ADDITIVE-OPTIONAL streams (see the module docstring).
    # `outcome_label` is the RECORDED battle's realized outcome for this state (1.0 win / 0.0 loss /
    # 0.5 tie) — one Monte-Carlo sample, which is exactly what makes it head B's arm: same states as
    # C, single-outcome precision. None when the producer did not ship one.
    outcome_label: Optional[float] = None
    # The mean realized SHAPED return over the producer's rollouts, in the run's own reward units,
    # and the rollout count behind it. None when absent or when the reward digest was refused.
    mc_return: Optional[float] = None
    mc_return_n: int = 0
    # The producing `RewardConfig`'s digest — carried for the record even when it matched, so a
    # label file is auditable a year later without the run beside it.
    reward_sha1: str = ""


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
        reward_sha1: Optional[str] = None,
    ) -> None:
        self.dir = Path(labels_dir)
        self.obs_dim = int(obs_dim) if obs_dim else None
        # gen3_cf_twin_heads_v1: THIS run's reward digest. When set, an `mc_return` whose row
        # declares a different digest is refused (the field only — the row's win-prob labels are
        # untouched). None disables the check, which is what a run with no shadow critic wants.
        self.reward_sha1 = str(reward_sha1) if reward_sha1 else None
        self.capacity = max(1, int(capacity))
        self.lag_bound = max(0, int(lag_bound))
        # obs_sha1 -> row, INSERTION-ORDERED, which is what makes one mapping serve both jobs: the
        # order is the FIFO the capacity cap evicts from, and the key is the dedup identity. A
        # repeat re-inserts at the end (keep-newest), so a re-labelled state also becomes the
        # youngest resident — which is the right FIFO position for the fresher measurement.
        self._rows: "OrderedDict[str, CfLabel]" = OrderedDict()
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
        # A SUBSET of `expired_total`: rows dated AHEAD of this process. Broken out because it has
        # exactly one cause worth naming (a resume from an older checkpoint) and a completely
        # different remedy from ordinary lag.
        self.future_total = 0
        # Rows superseded by a newer label of the same state. Rising = the producer is re-labelling
        # ground it already covered, which is not an error but IS a fact about the sampler.
        self.replaced_total = 0
        self.skipped_total = 0
        # gen3_cf_twin_heads_v1: `mc_return` fields dropped for a reward-digest disagreement. NOT a
        # subset of `skipped_total`, because the ROW was accepted — only one of its label streams
        # was refused, and conflating "this row is garbage" with "this row's shadow label is for a
        # different reward" would hide a whole-arm misconfiguration inside the GIGO meter.
        self.mc_return_rejected_total = 0
        # gen3_cf_twin_heads_v1: OPTIONAL-FIELD rejections (a malformed / out-of-range
        # `outcome_label` or `mc_return`). Deliberately NOT `skipped_total`: that counter's whole
        # job is "is the producer feeding me garbage ROWS", and `ingested + skipped` must partition
        # the input. A field rejection ACCEPTS the row, so counting it there would make the GIGO
        # meter climb at the ingestion rate on a buffer that is refusing nothing — the same
        # conflation `mc_return_rejected_total` exists to avoid, one level down.
        self.field_skipped_total = 0
        # Skip BREAKDOWN (not on TensorBoard; read from `stats()` in a test or a debugger). The
        # single `skipped_total` scalar answers "is the producer feeding me garbage"; this answers
        # "which garbage", which is a debugging question, not a monitoring one.
        self.skip_reasons: Dict[str, int] = {}
        self._warned: set = set()
        self._rng = np.random.default_rng(0)
        # path -> {key: array}. See `_NPZ_CACHE_FILES`.
        self._npz_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

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
        if self.lag_bound and self._too_stale(current_step, policy_step):
            return None
        obs = self._resolve_obs(obj)
        if obs is None:
            return None
        # gen3_cf_twin_heads_v1: the two ADDITIVE-OPTIONAL streams. Both are parsed permissively in
        # the ABSENT direction (an older producer's row simply carries neither and supervises
        # nothing extra) and strictly in the PRESENT-BUT-WRONG direction — a field that is there and
        # unusable is a counted skip of that FIELD, never a silent zero and never a dropped row.
        outcome = self._parse_outcome(obj)
        mc_return, mc_return_n = self._parse_mc_return(obj)
        self.ingested_total += 1
        return CfLabel(
            obs=obs, label=label, policy_step=policy_step,
            battle=str(obj.get("battle", "")), decision_idx=int(obj.get("decision_idx", -1)),
            opponent=str(obj.get("opponent", "")), n_rollouts=int(obj.get("n_rollouts", 0)),
            obs_sha1=_digest(obs),
            outcome_label=outcome, mc_return=mc_return, mc_return_n=mc_return_n,
            reward_sha1=str(obj.get("reward_sha1") or ""),
        )

    def _parse_outcome(self, obj: dict) -> Optional[float]:
        """``outcome_label`` — head B's SINGLE-OUTCOME stream, or None when the row has none."""
        raw = obj.get("outcome_label")
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            self._skip_field("outcome_label_malformed")
            return None
        if not np.isfinite(val) or not (0.0 <= val <= 1.0):
            # The same range contract as `label`: a win/loss/tie scalar. Anything else is a producer
            # bug, and head B trained on it would be trained on garbage with no tell.
            self._skip_field("outcome_label_range")
            return None
        return val

    def _parse_mc_return(self, obj: dict) -> "tuple[Optional[float], int]":
        """``mc_return`` — the SHADOW critic's stream, gated on the reward digest.

        Returns ``(value, n)`` or ``(None, 0)``. The reward check is the load-bearing half: a shaped
        return measured under a different `RewardConfig` is a measurement of a DIFFERENT value
        function, not a noisier measurement of this one.
        """
        raw = obj.get("mc_return")
        if raw is None:
            return None, 0
        try:
            val = float(raw)
            n = int(obj.get("mc_return_n", 0) or 0)
        except (TypeError, ValueError):
            self._skip_field("mc_return_malformed")
            return None, 0
        if not np.isfinite(val):
            self._skip_field("mc_return_malformed")
            return None, 0
        if self.reward_sha1 is not None:
            got = str(obj.get("reward_sha1") or "")
            if got != self.reward_sha1:
                self.mc_return_rejected_total += 1
                # The message deliberately carries NO per-row digest: `_warn` dedups on the whole
                # string, so embedding `got` would print (and retain) one line per distinct foreign
                # digest instead of warning once. The breakdown lives in the counter.
                self._warn(
                    f"mc_return REWARD DIGEST mismatch (this run's is {self.reward_sha1[:12]}) — a "
                    f"shaped return measured under a different reward composition is a different "
                    f"value function, not a noisier sample of this one. The mc_return field is "
                    f"dropped; the row's win-prob labels are kept. Count: "
                    f"cf/labels_mc_return_rejected_total.")
                return None, 0
        return val, max(0, n)

    def _too_stale(self, current_step: int, policy_step: int) -> bool:
        """The SYMMETRIC staleness test, and the accounting that goes with it.

        Counts into `expired_total` (and `future_total` for the ahead-of-us half) as a side effect,
        so ingest-time and expire-time rejection are the same decision made in one place."""
        age = current_step - policy_step
        if abs(age) <= self.lag_bound:
            return False
        self.expired_total += 1
        if age < 0:
            self.future_total += 1
            self._warn(
                f"label from a NEWER snapshot than this process (policy_step {policy_step} > "
                f"num_timesteps {current_step}) — crash-restart rollback? Those rows are EXPIRED "
                f"like stale ones; without this they would never age out and would silently become "
                f"the whole buffer. Clear <run>/cf_labels/ or restart the producer at this step.")
        return True

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
            # `decision_idx` selects the ROW of a 2-D array — that is the schema's own wording, and
            # the producer's default output (`cf_audit` without `--inline-obs`) points every row of
            # a battle at the SAME `<states.npz>::obs` matrix. Ignoring the index flattened the whole
            # matrix into one 1-D vector, which then failed the obs-width GIGO guard: the entire
            # non-inline half of the schema was unconsumable, loudly but for the wrong reason.
            arr = self._load_npz(str(obj["obs_npz"]), int(obj.get("decision_idx", -1)))
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
            got = _digest(arr)
            if got != want:
                self._warn("obs_sha1 MISMATCH — the producer's obs bytes disagree with its own "
                           "digest; rows rejected (GIGO guard)")
                return self._skip("obs_sha1")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _load_npz(self, spec: str, decision_idx: int) -> Optional[np.ndarray]:
        """One row of ``<path>::<key>``, through a small per-FILE LRU (see `_NPZ_CACHE_FILES`)."""
        if "::" not in spec:
            return self._skip("obs_npz_spec")
        path, _, key = spec.partition("::")
        cached = self._npz_cache.get(path)
        if cached is None:
            try:
                with np.load(path) as z:
                    cached = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
            except Exception:
                return self._skip("obs_npz_unreadable")
            self._npz_cache[path] = cached
            while len(self._npz_cache) > _NPZ_CACHE_FILES:
                self._npz_cache.popitem(last=False)
        else:
            self._npz_cache.move_to_end(path)
        arr = cached.get(key)
        if arr is None:
            return self._skip("obs_npz_key")
        if arr.ndim >= 2:
            if not (0 <= decision_idx < arr.shape[0]):
                return self._skip("obs_npz_row")
            arr = arr[decision_idx]
        # `.reshape(-1)` (not `.ravel()`) so a >2-D array is a malformed-shape skip downstream
        # rather than a silently flattened one; a 1-D array is passed through untouched, which is
        # the "one vector per file" layout a hand-written producer would use.
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    def _push(self, row: CfLabel) -> None:
        # DEDUP, keep-newest. `pop` + re-insert rather than assignment: assigning to an existing key
        # keeps the OLD position, which would leave a re-labelled state sitting at the FIFO head
        # about to be evicted despite being the freshest thing in the buffer.
        if self._rows.pop(row.obs_sha1, None) is not None:
            self.replaced_total += 1
        self._rows[row.obs_sha1] = row
        while len(self._rows) > self.capacity:
            self._rows.popitem(last=False)     # FIFO: the OLDEST label is the one to lose

    def _skip(self, reason: str) -> None:
        """Reject the whole ROW. Counted into the GIGO meter."""
        self.skipped_total += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        return None

    def _skip_field(self, reason: str) -> None:
        """Reject one OPTIONAL FIELD and keep the row (see `field_skipped_total`)."""
        self.field_skipped_total += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        return None

    def _warn(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            print(f"⚠️  [cf_labels] {msg}", flush=True)

    # -- expiry / sampling -----------------------------------------------------------
    def expire(self, current_step: int) -> int:
        """Drop rows whose |age| exceeds ``lag_bound`` policy steps. Returns how many were dropped.

        The boundary is INCLUSIVE of the bound and SYMMETRIC: |age| == lag_bound survives,
        |age| == lag_bound + 1 does not, in EITHER direction. ``lag_bound == 0`` disables expiry
        entirely (nothing is ever too old — nor too new).
        """
        if not self.lag_bound:
            return 0
        doomed = [k for k, row in self._rows.items()
                  if self._too_stale(current_step, row.policy_step)]
        for k in doomed:
            del self._rows[k]
        return len(doomed)

    def sample(self, n: int) -> List[CfLabel]:
        """Up to ``n`` rows drawn uniformly WITHOUT replacement. Empty when starving."""
        if not self._rows or n <= 0:
            return []
        rows = list(self._rows.values())
        if len(rows) <= n:
            return rows
        idx = self._rng.choice(len(rows), size=n, replace=False)
        return [rows[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self._rows)

    # -- metrics ---------------------------------------------------------------------
    def stats(self, current_step: int) -> Dict[str, float]:
        """The `cf/` TensorBoard scalars (see the module docstring).

        The last three are `gen3_cf_twin_heads_v1`'s and they answer the launch-window question the
        others cannot: **is the arm's own label stream actually arriving?** A twin-heads run whose
        producer ships no ``outcome_label`` trains head B on nothing while every other counter reads
        healthy — B would then equal A, the C−B contrast would silently become C−A, and the run
        would look like a result. A COVERAGE fraction near 1.0 is the reading that says the
        factorial is live.
        """
        ages = [current_step - r.policy_step for r in self._rows.values()]
        rows = list(self._rows.values())
        n = float(len(rows)) or 1.0
        return {
            "cf/buffer_fill": float(len(self._rows)),
            "cf/label_age_steps_p50": float(np.median(ages)) if ages else 0.0,
            "cf/labels_ingested_total": float(self.ingested_total),
            "cf/labels_expired_total": float(self.expired_total),
            "cf/labels_future_total": float(self.future_total),
            "cf/labels_replaced_total": float(self.replaced_total),
            "cf/labels_skipped_total": float(self.skipped_total),
            "cf/outcome_label_coverage": float(
                sum(1 for r in rows if r.outcome_label is not None)) / n,
            "cf/mc_return_coverage": float(
                sum(1 for r in rows if r.mc_return is not None)) / n,
            "cf/labels_mc_return_rejected_total": float(self.mc_return_rejected_total),
            "cf/labels_field_skipped_total": float(self.field_skipped_total),
        }


class CfBatch(NamedTuple):
    """The tensors ONE cf fold works from — every label stream the sampled rows carry.

    A NamedTuple rather than a bare tuple because the streams grew from one to three
    (`gen3_cf_twin_heads_v1`) and a positional 7-tuple is exactly the shape of the order-mismatch
    bug class this tree treats as drop-everything. Field names are the contract.

    ``obs``          [B, obs_dim] float32
    ``label``        [B] the TIGHT-MC win ratio — head C's and head A's cf stream
    ``n_rollouts``   [B] the evidence behind ``label`` (see below)
    ``outcome``      [B] the RECORDED single outcome — head B's stream (0 where absent)
    ``outcome_mask`` [B] 1.0 where ``outcome`` is a real label, 0.0 where the row carried none
    ``mc_return``    [B] the mean realized SHAPED return — the shadow critic's stream (0 where absent)
    ``mc_return_mask`` [B] 1.0 where ``mc_return`` is real

    The MASKS are not decoration. A minibatch mixing rows from two producers (one that ships the new
    streams, one that does not) must supervise each head on exactly the rows that have its label,
    and a zero-filled absent label is indistinguishable from a confident "you lose" — the single
    most dangerous silent target this schema could produce.
    """
    obs: "th.Tensor"            # type: ignore[name-defined]  # noqa: F821
    label: "th.Tensor"          # type: ignore[name-defined]  # noqa: F821
    n_rollouts: "th.Tensor"     # type: ignore[name-defined]  # noqa: F821
    outcome: "th.Tensor"        # type: ignore[name-defined]  # noqa: F821
    outcome_mask: "th.Tensor"   # type: ignore[name-defined]  # noqa: F821
    mc_return: "th.Tensor"      # type: ignore[name-defined]  # noqa: F821
    mc_return_mask: "th.Tensor"  # type: ignore[name-defined]  # noqa: F821


def batch_tensors(rows: Sequence[CfLabel], device) -> CfBatch:
    """The sampled rows as a :class:`CfBatch`, every field float32 on ``device``.

    ``n_rollouts`` rides along because the label is a SUFFICIENT STATISTIC only in company: a
    ``label`` of 0.75 from 4 rollouts and one from 16 are the same number carrying four times the
    evidence, and the row's win COUNT is recoverable as ``round(label · n_rollouts)``. The binomial
    likelihood (`instrumented_ppo._cf_binomial_nll`) and the Beta-Binomial evidential loss both
    need the count, not the ratio.

    A row whose producer omitted ``n_rollouts`` parses as 0; the consumers clamp to 1, so an
    unlabelled-count row degrades to exactly one observation rather than vanishing or dividing by
    zero. Head B's ``outcome`` is deliberately fed at **n ≡ 1** by its caller — a single realized
    outcome IS one observation, and under the binomial term's ``Σ NLL / Σ n`` normalization that
    gives B and C the SAME per-row gradient magnitude with only the TARGET differing. That equality
    is what makes C−B a clean read of label precision rather than of effective learning rate.

    Torch imported lazily.
    """
    import torch as th
    obs = np.stack([r.obs for r in rows]).astype(np.float32, copy=False)
    lab = np.asarray([r.label for r in rows], dtype=np.float32)
    n = np.asarray([r.n_rollouts for r in rows], dtype=np.float32)
    out = np.asarray([0.0 if r.outcome_label is None else r.outcome_label for r in rows],
                     dtype=np.float32)
    out_m = np.asarray([0.0 if r.outcome_label is None else 1.0 for r in rows], dtype=np.float32)
    ret = np.asarray([0.0 if r.mc_return is None else r.mc_return for r in rows], dtype=np.float32)
    ret_m = np.asarray([0.0 if r.mc_return is None else 1.0 for r in rows], dtype=np.float32)
    t = lambda a: th.as_tensor(a, device=device)                             # noqa: E731
    return CfBatch(obs=t(obs), label=t(lab), n_rollouts=t(n),
                   outcome=t(out), outcome_mask=t(out_m),
                   mc_return=t(ret), mc_return_mask=t(ret_m))
