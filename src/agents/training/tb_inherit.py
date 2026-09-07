"""A FORK INHERITS ITS PARENT'S CURVES — `gen3_tb_inherit_v1`.

WHY THIS WORKS AT ALL, and why it is bookkeeping rather than a new mechanism. Two facts about
this tree line up exactly:

1. **TensorBoard merges every `events.out.tfevents.*` file inside ONE run directory into one
   series per tag, ordered by step.** Not a claim — it is how every run here already displays:
   `ai_v8_03_zarch_control_0718` carries **29** event files (one per launcher restart) in a single
   `tb/` and renders as one curve.
2. **A fork's global step CONTINUES from the parent.** `model.learn(reset_num_timesteps=False)`
   keeps `num_timesteps`, so the fork's first logged step is its `fork_step`. Measured on the same
   run: `ai_v8_03`'s own `tb/` opens at step **148,401,356**, which is precisely the `fork_step` its
   `lineage` block records.

So the parent's curve occupies `[0, fork_step]` and the fork's occupies `[fork_step, …]` — two
halves of one line that were only ever apart because they live in two directories. Dropping a
TRUNCATED copy of the parent's scalar events (steps ≤ `fork_step`) into `<fork>/tb/` makes the
fork's TensorBoard show the whole history from step 0. Nothing is recomputed and no curve is
invented: every point written here was written by the parent's own trainer.

WHY TRUNCATE. The parent usually trained PAST the fork point (`ai_v8_01` reached 170.6M having been
forked at 148.4M). Copying its tail would draw parent-only progress inside the fork's own step
range, where it would read as the fork's. Steps `> fork_step` are therefore dropped, always.

FORK OF A FORK falls out for free. The parent's `tb/` already holds its own inherited prefix, so
reading the parent's WHOLE directory and truncating again yields grandparent `[0, parent_fork_step]`
+ parent `[parent_fork_step, fork_step]`. One rule, applied once per link, composes down the chain.

SCALARS ONLY. Histograms, images, audio and every non-`scalars` plugin are skipped — they are
per-step blobs, they are what makes an event file large, and no plot in this programme reads them.
Measured over the whole archive (2026-09-06): **every** event value in `models/*/tb/` is a scalar,
so today the filter drops nothing; it exists so that adding a histogram later cannot silently
multiply the copy cost.

🚨 **THE VALUE FORM IS PART OF THE COPY, and getting it wrong is invisible.** A TensorBoard scalar
has two on-disk spellings — the classic `simple_value` field, and a rank-0 tensor tagged with the
`scalars` plugin — and `EventFileLoader` **migrates the first into the second as it reads**. Copying
what that loader returns therefore writes the migrated form, which `EventAccumulator` files under
`tensors` rather than `scalars`: the curve is present in the file, the provenance is correct, the
byte count is right, and **the scalars dashboard shows nothing**. Every writer here emits
`simple_value`, so this is the live case, not a corner one. `scalar_prefix` reads with
`LegacyEventFileLoader` (raw) for exactly this reason, and `tb_inherit_test` writes its synthetic
parent in `simple_value` form and reads back through `EventAccumulator.Scalars` — the accessor the
dashboard uses — because a test that writes and reads the same migrated form cannot see the bug.

IDEMPOTENT, and that is load-bearing. `<fork>/tb/INHERITED_FROM.json` is the key: if it exists, this
is a no-op. A launcher RESTART that still names the parent as `--model` (which happens before the
fork has written its own checkpoint) re-enters the fork path and would otherwise append a SECOND
copy of the prefix — the same series twice, which TensorBoard renders as a saw-tooth rather than an
error.

COST. `tb/` is 262 MB across all 217 runs on disk; a prefix is a few hundred KB. The copy is
proportional to the parent's scalar count, not to its checkpoints.

THE ONE THING TO KNOW BEFORE TURNING IT ON FLEET-WIDE: a sibling fleet (eight exploiters forked off
one target) then shares an IDENTICAL prefix. Under a curated logdir (`main.tb_curate`) that is
exactly what you want — the one arm on screen carries its history. Under an uncurated
`--logdir models/`, it is eight redundant copies of the same curve competing in every chart.
`--no-tb-inherit` opts a run out.
"""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

#: Bumped when the provenance file's SHAPE changes.
INHERIT_SCHEMA = 1

#: The provenance file. Its EXISTENCE is the idempotency key — see the module docstring.
PROVENANCE_BASENAME = "INHERITED_FROM.json"

#: The written event file's name. Deterministic on purpose (a stable sha, a detectable re-run) and
#: chosen to sort FIRST among `events.out.tfevents.<unix_seconds>.…` siblings, so a reader that
#: walks the directory in name order sees the prefix before the fork's own events. TensorBoard
#: identifies an event file by the substring `tfevents` in its basename, not by the timestamp.
INHERITED_EVENTS_BASENAME = "events.out.tfevents.0000000000.inherited.0.0"

#: The plugin whose values we carry. Everything else is skipped (see the docstring).
_SCALARS_PLUGIN = "scalars"

#: What `EventFileWriter` writes as a file's first record.
_FILE_VERSION = "brain.Event:2"


class TbInheritError(RuntimeError):
    """The copy was asked for and could not be done truthfully."""


# --------------------------------------------------------------------------------------------
# locating things
# --------------------------------------------------------------------------------------------
def tb_dir_of(run_dir: str) -> str:
    """`<run_dir>/tb` — the one place this tree keeps a run's TensorBoard data.

    Written by `main.train.run_io._attach_run_tb_logger`, which deliberately bypasses SB3's
    `tb_log_name` `_<N>` suffixing so a run's curves live inside its own model dir.
    """
    return os.path.join(run_dir, "tb")


def provenance_path(run_dir: str) -> str:
    return os.path.join(tb_dir_of(run_dir), PROVENANCE_BASENAME)


def read_provenance(run_dir: str) -> Optional[Dict[str, Any]]:
    """The recorded inheritance for `run_dir`, or None if it never inherited."""
    try:
        with open(provenance_path(run_dir), encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:  # noqa: BLE001 — absent, truncated, or not ours
        return None
    return obj if isinstance(obj, dict) else None


def event_files(tb_dir: str) -> List[str]:
    """Every `events.out.tfevents.*` under `tb_dir`, recursively, in name order.

    Name order is timestamp order for the writer's own files, and puts an inherited prefix first
    (see `INHERITED_EVENTS_BASENAME`) — the same order TensorBoard's directory walk uses.
    """
    if not os.path.isdir(tb_dir):
        return []
    found = glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents.*"), recursive=True)
    return sorted(f for f in found if os.path.isfile(f))


def _sha256_file(path: str, *, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------------------------
# reading + filtering
# --------------------------------------------------------------------------------------------
def _is_scalar_value(value: Any) -> bool:
    """Is this one `Summary.Value` a scalar we carry?

    Two spellings, because both exist in the wild: the classic `simple_value` field, and the
    TensorBoard-2.x form every writer in this tree actually emits — a rank-0 `tensor` tagged with
    the `scalars` plugin. Anything else (histograms, images, text, hparams tensors, an unknown
    plugin) is NOT a scalar and is skipped rather than guessed at.
    """
    which = value.WhichOneof("value")
    if which == "simple_value":
        return True
    if which != "tensor":
        return False
    try:
        return value.metadata.plugin_data.plugin_name == _SCALARS_PLUGIN
    except AttributeError:
        return False


def scalar_prefix(tb_dir: str, fork_step: int) -> Iterator[Tuple[Any, List[str]]]:
    """Yield `(Event, [tags])` for every SCALAR summary at step ≤ `fork_step`, in file order.

    The yielded Event is a COPY holding only the scalar values — an event that mixed a scalar and a
    histogram would otherwise smuggle the histogram through. An event left with no values after the
    filter is not yielded at all.
    """
    # 🚨 LegacyEventFileLoader, NOT EventFileLoader — and this is not a preference.
    # `EventFileLoader.Load` runs `data_compat.migrate_event`, which silently rewrites a
    # `simple_value` scalar into a rank-0 `scalars`-plugin TENSOR. Every writer in this tree emits
    # `simple_value` on disk, so reading with the migrating loader and writing what came back
    # produced a file whose records were the POST-migration form. `EventAccumulator` — the reader
    # behind the scalars dashboard — routes on the raw on-disk field, so those records landed under
    # `tensors` and the inherited curve DID NOT APPEAR in the scalars view at all: 124 tags read
    # back as 124 `tensors` tags and 0 `scalars` tags, beside the fork's own 124 `scalars`.
    # The legacy loader hands back exactly what is on disk, so the copy preserves the value FORM.
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    from tensorboard.compat.proto import event_pb2

    for path in event_files(tb_dir):
        if os.path.basename(path) == PROVENANCE_BASENAME:
            continue
        try:
            loader = LegacyEventFileLoader(path)
            for event in loader.Load():
                if event.WhichOneof("what") != "summary":
                    continue          # file_version / graph_def / session_log carry no curve
                if int(event.step) > int(fork_step):
                    continue
                keep = [v for v in event.summary.value if _is_scalar_value(v)]
                if not keep:
                    continue
                out = event_pb2.Event()
                out.wall_time = event.wall_time   # keep the real clock: the wall/relative axes
                out.step = event.step
                out.summary.value.extend(keep)
                yield out, [v.tag for v in keep]
        except Exception as exc:  # noqa: BLE001 — a truncated tail is normal on a killed run
            # A partially-written final record is the ordinary end of a crashed run's event file.
            # Everything already yielded is still valid, so stop reading THIS file and move on.
            _warn(f"stopped reading {os.path.basename(path)}: {type(exc).__name__}: {exc}")
            continue


def _warn(msg: str) -> None:
    print(f"[tb-inherit] WARNING: {msg}")


# --------------------------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class InheritResult:
    """What one `inherit_tb` call did. `written` is False for every no-op, `reason` says which."""

    written: bool
    reason: str
    events_path: Optional[str] = None
    events_copied: int = 0
    values_copied: int = 0
    tags: Tuple[str, ...] = ()
    parent_run_name: Optional[str] = None
    parent_tb_dir: Optional[str] = None
    fork_step: Optional[int] = None
    first_step: Optional[int] = None
    last_step: Optional[int] = None
    sha256: Optional[str] = None
    parent_inherited: bool = False

    def describe(self) -> str:
        if not self.written:
            return f"[tb-inherit] {self.reason}"
        return (f"[tb-inherit] {self.events_copied:,} events / {self.values_copied:,} scalar values "
                f"/ {len(self.tags)} tags from {self.parent_run_name} "
                f"(steps {self.first_step:,}..{self.last_step:,} ≤ fork_step {self.fork_step:,})"
                + (" [parent carried its own inherited prefix]" if self.parent_inherited else ""))


# --------------------------------------------------------------------------------------------
# the copy
# --------------------------------------------------------------------------------------------
def inherit_tb(fork_run_dir: str, *, parent_run_dir: str, fork_step: int,
               parent_run_name: Optional[str] = None, dry_run: bool = False,
               force: bool = False) -> InheritResult:
    """Write the parent's scalar prefix (steps ≤ `fork_step`) into `<fork_run_dir>/tb/`.

    A NO-OP — never an error — when the run already inherited (`INHERITED_FROM.json` present, the
    launcher-restart guard), when the parent has no `tb/`, or when the prefix is empty. `force`
    overwrites an existing inheritance; `dry_run` reports without touching the filesystem.
    """
    fork_tb = tb_dir_of(fork_run_dir)
    parent_tb = tb_dir_of(parent_run_dir)
    pname = parent_run_name or os.path.basename(os.path.normpath(parent_run_dir))

    if not force and read_provenance(fork_run_dir) is not None:
        return InheritResult(False, f"already inherited (see {PROVENANCE_BASENAME}) — nothing to do",
                             parent_run_name=pname)
    if not os.path.isdir(parent_tb):
        return InheritResult(False, f"parent {pname} has no tb/ — nothing to inherit",
                             parent_run_name=pname)
    if fork_step is None or int(fork_step) <= 0:
        return InheritResult(False, f"fork_step is {fork_step!r} — a fork at step 0 inherits nothing",
                             parent_run_name=pname)

    parent_inherited = read_provenance(parent_run_dir) is not None
    events: List[Any] = []
    tags: List[str] = []
    seen = set()
    n_values = 0
    for event, ev_tags in scalar_prefix(parent_tb, int(fork_step)):
        events.append(event)
        n_values += len(ev_tags)
        for t in ev_tags:
            if t not in seen:
                seen.add(t)
                tags.append(t)
    if not events:
        return InheritResult(False,
                             f"no scalar events at step <= {int(fork_step):,} in {pname}/tb",
                             parent_run_name=pname, parent_tb_dir=parent_tb,
                             fork_step=int(fork_step))

    steps = [int(e.step) for e in events]
    out_path = os.path.join(fork_tb, INHERITED_EVENTS_BASENAME)
    partial = InheritResult(
        True, "would write" if dry_run else "wrote", events_path=out_path,
        events_copied=len(events), values_copied=n_values, tags=tuple(tags),
        parent_run_name=pname, parent_tb_dir=parent_tb, fork_step=int(fork_step),
        first_step=min(steps), last_step=max(steps), parent_inherited=parent_inherited)
    if dry_run:
        return partial

    os.makedirs(fork_tb, exist_ok=True)
    _write_event_file(out_path, events)
    sha = _sha256_file(out_path)
    result = dataclasses.replace(partial, sha256=sha)
    _write_provenance(fork_run_dir, parent_run_dir, result)
    return result


def _write_event_file(path: str, events: Iterable[Any]) -> None:
    """Write `events` as a TFRecord event file at exactly `path`.

    `RecordWriter` (a file object in, framed records out) rather than `EventFileWriter`, because the
    latter invents its own `…<time>.<host>.<pid>.<uid>` name and we need a deterministic one that
    sorts first. Written to a `.tmp` and `os.replace`d, so a reader never sees a half file.
    """
    from tensorboard.compat.proto import event_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter

    header = event_pb2.Event()
    header.wall_time = 0.0
    header.file_version = _FILE_VERSION

    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        writer = RecordWriter(fh)
        writer.write(header.SerializeToString())
        for event in events:
            writer.write(event.SerializeToString())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_provenance(fork_run_dir: str, parent_run_dir: str, result: InheritResult) -> None:
    """State what was copied, from where, and under which rule — beside the file it describes.

    A curve that appears in a run's TensorBoard without that run having trained it MUST say so
    somewhere a reader will find it; this is that somewhere.
    """
    block = {
        "schema": INHERIT_SCHEMA,
        "marker": "gen3_tb_inherit_v1",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "parent_run_name": result.parent_run_name,
        "parent_run_dir": os.path.abspath(parent_run_dir),
        "parent_tb_dir": os.path.abspath(result.parent_tb_dir or ""),
        "parent_carried_inherited_prefix": result.parent_inherited,
        "fork_step": result.fork_step,
        "truncated_at_step": result.fork_step,
        "events_file": os.path.basename(result.events_path or ""),
        "events_copied": result.events_copied,
        "scalar_values_copied": result.values_copied,
        "tags_copied": list(result.tags),
        "first_step": result.first_step,
        "last_step": result.last_step,
        "sha256": result.sha256,
        "note": ("SCALARS ONLY, steps <= fork_step. These points were logged by the PARENT run, not "
                 "by this one. Delete this file and the events file it names to undo."),
    }
    path = provenance_path(fork_run_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(block, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# the training-time seam
# --------------------------------------------------------------------------------------------
def inherit_from_lineage(fork_run_dir: str, lineage_block: Optional[Dict[str, Any]], *,
                         enabled: bool = True, dry_run: bool = False) -> InheritResult:
    """THE SEAM the trainer calls, right where the `lineage` block is written.

    `lineage_block` is `main.train.run_io._run_lineage`'s return — and its being non-None is already
    the FORK decision (`agents.training.lineage.build_lineage` returns None on a same-run restart,
    via `main.train.fork_lr.is_same_run_checkpoint`). This function therefore re-derives nothing: it
    reads the parent and the `fork_step` out of the block that the run is about to record, so the
    curve a fork inherits and the parent it claims can never disagree.
    """
    if not enabled:
        return InheritResult(False, "disabled (--no-tb-inherit)")
    if not lineage_block:
        return InheritResult(False, "same-run restart (no new lineage block) — nothing to inherit")
    parent = lineage_block.get("fork_parent")
    if not isinstance(parent, dict):
        return InheritResult(False, "fresh run (no fork parent) — nothing to inherit")
    parent_dir = parent.get("run_dir")
    if not parent_dir:
        return InheritResult(False, "fork parent records no run_dir — nothing to inherit")
    fork_step = lineage_block.get("fork_step")
    if fork_step is None:
        return InheritResult(False, "lineage block records no fork_step — nothing to inherit")
    try:
        return inherit_tb(fork_run_dir, parent_run_dir=str(parent_dir), fork_step=int(fork_step),
                          parent_run_name=parent.get("run_name"), dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        # A cosmetic convenience must never be able to kill a training launch. The run trains
        # identically without its inherited prefix; it just starts its chart at fork_step.
        return InheritResult(False, f"FAILED (training unaffected): {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------------
# the backfill survey
# --------------------------------------------------------------------------------------------
def forks_missing_prefix(models_dir: str) -> List[Dict[str, Any]]:
    """Every run under `models_dir` that IS a fork, HAS a tb/, and has NOT inherited.

    The census behind `python -m main.tb_inherit --backfill`. Reads the PERSISTED `lineage` block in
    `metadata.json` and never `agents.training.lineage.fork_parent`'s derive-on-the-fly fallback —
    a backfill must act on what a run recorded, not on what a regex reconstructs at read time.

    🚨 **Each row carries `derived`, and on this archive it is `True` for every run.** Every block on
    disk was written by `main.lineage --backfill`, i.e. by REGEXing `--model` out of a recorded
    shell command — so the parent it names is an inference. It can be wrong: `ai_v8_01_zarch_film_0717`
    records `role="fresh", fork_step=0` while its own `tb/` opens at step **148,401,356**, which is
    arithmetically impossible for a fresh run (it was created by a hand-written `tmp/fork_zarch_v8.py`
    that the regex cannot read). Attaching a curve to a wrong parent is exactly the kind of silent
    provenance error this programme has paid for before, which is why the backfill is dry-run by
    default and prints this flag on every row.
    """
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(models_dir):
        return out
    for name in sorted(os.listdir(models_dir)):
        run_dir = os.path.join(models_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if not os.path.isdir(tb_dir_of(run_dir)):
            continue
        if read_provenance(run_dir) is not None:
            continue
        block = _read_lineage_block(run_dir)
        if not block:
            continue
        parent = block.get("fork_parent")
        if not isinstance(parent, dict) or not parent.get("run_dir"):
            continue
        step = block.get("fork_step")
        if not step:
            continue
        pdir = str(parent["run_dir"])
        out.append({
            "run": name, "run_dir": run_dir, "role": block.get("role"),
            "parent_run_name": parent.get("run_name"), "parent_run_dir": pdir,
            "fork_step": int(step),
            "parent_tb_exists": os.path.isdir(tb_dir_of(pdir)),
            # True = the parent claim was REGEXED out of `original_command`, not recorded at fork
            # time. See this function's docstring; it is True for every run in today's archive.
            "derived": bool(block.get("derived")),
        })
    return out


def _read_lineage_block(run_dir: str) -> Optional[Dict[str, Any]]:
    """The RECORDED `lineage` block, or None. Deliberately not `lineage.read_block`'s derived path."""
    try:
        with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    block = meta.get("lineage") if isinstance(meta, dict) else None
    return block if isinstance(block, dict) else None
