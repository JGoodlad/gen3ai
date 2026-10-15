"""The learner's LOGGER — ours (`gen3_owned_logger_v1`; `designs/endstate/design_own_ppo_loop.md` §3.3).

Until deletion pass U4 the trainer logged through sb3's ``Logger`` (``configure(<run>/tb, ["stdout",
"tensorboard"])``). This is that logger, written out operation for operation for the two outputs the
repo uses — the stdout table and TensorBoard — so the event files carry the SAME tags, steps and values
and the stdout tables the same text — except a key too long for the table's column, which sb3 truncated
into a possible COLLISION that raised mid-run; it now renders collision-proof (`truncate_label`), and
`check_metric_keys` refuses a key set that collides in any format at startup
(`gen3_metric_key_collision_v1`). Any other output format is REFUSED.

THE BUS. ``name_to_value`` is the pending scalars until the next ``dump(step)`` — a ``defaultdict(float)``
as sb3's was — and it is READ in between: the launcher's metrics pipe (``MetricsExporterCallback`` at each
``on_rollout_end``), the K9 golden (``compute``), `logger_scope.isolated_dump` (which holds it aside for
an eval cycle's own dump), ``update_fit`` (which snapshots and restores it around the startup dry
update). ``name_to_count`` / ``name_to_excluded`` travel with it.

THE STEP CONVENTION is the LOOP's, not this module's (`instrumented_ppo/loop.py`: the dump comes BEFORE
the update, so update k's ``train/*`` is stamped after rollout k+1). The readers that read the tags BY
NAME — ``main.ops.killbar`` / ``tb_read`` / ``restart_startup`` / ``stall_exhibit``, the launcher's
``format.py``, ``utils/plot_tb.py``, ``tb_curate`` / ``tb_inherit`` — read the event files, never this
class.

ONE PRODUCER PER KEY, PER DUMP INTERVAL (`gen3_metric_producer_guard_v1`). ``record`` replaces a pending key, so
two code sites that write one key between two dumps lose the earlier site's value SILENTLY (2026-10-15 survey:
`fork/*`, `train/effective_batch`, `lifecycle/cuda_streams_after_freeze`, all fixed). ``record`` /
``record_mean`` therefore remember WHICH CALL SITE (file and line of the caller) wrote each pending key, and a
different site writing it before the dump is a `DuplicateProducerError` when the logger is STRICT (tests, a
``--debug`` run, ``GEN3AI_STRICT_METRIC_PRODUCERS=1``: `producer_guard_strict`) and otherwise COUNTED and loudly
REPORTED (one stdout line per key per process, ``duplicate_producers`` on the logger, and a
``telemetry/duplicate_producer_keys`` scalar on the dumps that carried one) — a production run never stops on it.
The same site writing a key again (a loop, a per-minibatch ``record_mean``) is one producer, not two.

The writers pair ``name_to_value`` with ``name_to_excluded`` KEY FOR KEY (:func:`paired`): the two key
sets must be EQUAL, or the dump raises ``ValueError`` naming the keys that are in one and not the other.
A key that reaches ``name_to_value`` without ``record`` (an indexing READ of the defaultdict inserts 0.0)
fails the dump loudly instead of writing a phantom 0. sb3 paired the two sorted lists with
``zip(strict=True)``, which checks LENGTHS only — a phantom value key plus a stranded exclusion key
(one dict edited without the other) is the same length, pairs the wrong rows and drops or keeps a value
under another key's exclusion; here that is the same loud error (P10 follow-up F1, item 5).
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import torch as th

DEBUG, INFO, WARN, ERROR, DISABLED = 10, 20, 30, 40, 50

#: The output formats `configure` builds (sb3 also had log / json / csv — never used here).
FORMATS = ("stdout", "tensorboard")

Excluded = Tuple[str, ...]

#: The stdout table's key column: 48 (sb3's was 36), so every row the production network writes today prints IN
#: FULL (the longest, `cell/move_resolution_restored_switch_weight_norm`, is a 46-character row). A longer row is
#: shortened by `truncate_label`. A table whose rows all fit 36 is sb3's, byte for byte.
STDOUT_MAX_LENGTH = 48
#: The hex digits of the key digest a shortened label carries (`truncate_label`).
LABEL_DIGEST_HEX = 6
#: TensorBoard's text plugin writes ``add_text(key, …)`` under ``key + TB_TEXT_SUFFIX``
#: (`torch.utils.tensorboard.summary.text`); scalars and histograms are written under the key verbatim.
TB_TEXT_SUFFIX = "/text_summary"

#: Forces the one-producer guard strict ("1") or lenient ("0"). Unset: strict under pytest (it sets
#: ``PYTEST_CURRENT_TEST``) and under ``--debug`` (the trainer sets this variable), lenient otherwise.
STRICT_PRODUCERS_ENV = "GEN3AI_STRICT_METRIC_PRODUCERS"
#: The scalar a dump carries when a lenient logger counted a key with two producers in that interval
#: (ABSENT otherwise, so a clean run's tag set is unchanged).
DUPLICATE_PRODUCERS_KEY = "telemetry/duplicate_producer_keys"
#: A lenient logger prints at most this many distinct duplicate keys (the count scalar keeps counting).
MAX_REPORTED_DUPLICATE_KEYS = 20

#: One call site: the caller's ``(file, line, is it a `record_mean` call)``.
Site = Tuple[str, int, bool]


def producer_guard_strict() -> bool:
    """Is the one-producer guard STRICT for a logger built now? (`STRICT_PRODUCERS_ENV`; else pytest.)"""
    v = os.environ.get(STRICT_PRODUCERS_ENV, "").strip().lower()
    if v:
        return v not in ("0", "false", "off", "no")
    return "PYTEST_CURRENT_TEST" in os.environ


def site_str(site: Site) -> str:
    """``path/under/src.py:123`` — a call site as a reader wants it."""
    path = site[0].replace("\\", "/")
    return f"{path.split('/src/')[-1]}:{site[1]}"


class MetricKeyCollisionError(ValueError):
    """Two DISTINCT metric keys become ONE name in an output format (or one key has two producers) —
    refused at STARTUP (`check_metric_keys`), mapped to FATAL_CONFIG (`main.exit_codes`).

    THE CLASS (`gen3_metric_key_collision_v1`): sb3's stdout table truncated a key to 33 characters + "..."
    and RAISED mid-run when two keys truncated alike. It killed every fixed_mass self-play run at its
    first dump (F-XC-5, `flat_switch_target_recall_top1_{bot,pool}`), and again every production run at a
    commit >= 65b71f6a (2026-10-15: ``cell/move_resolution_restored_move_{weight,grad}_norm``, at step
    ~202k). The table's labels are now collision-proof (`truncate_label`), and a key set that still
    collides in ANY format is refused here, before the first rollout."""


class DuplicateProducerError(MetricKeyCollisionError):
    """Two DIFFERENT call sites wrote one metric key between two dumps (`Logger.record`): the later write
    replaced the earlier silently. Raised by a STRICT logger (`producer_guard_strict`: tests, ``--debug``);
    a production logger counts and reports it instead and never stops (`gen3_metric_producer_guard_v1`).
    FATAL_CONFIG like its parent. The fix is ALWAYS one producer per quantity, or two names for two
    quantities — never an exemption."""


def key_digest(key: str, n: int = LABEL_DIGEST_HEX) -> str:
    """A STABLE short digest of ``key`` (blake2s; identical in every process, unlike ``hash``)."""
    return hashlib.blake2s(key.encode("utf-8"), digest_size=8).hexdigest()[:n]


def truncate_label(string: str, max_length: int = STDOUT_MAX_LENGTH) -> str:
    """``string`` if it fits, else ``<head>~<digest>~<tail>`` of exactly ``max_length`` characters.

    The digest is of the WHOLE string, so two strings that share a head AND a tail still render apart
    (sb3's ``string[:33] + "..."`` did not: ``…restored_move_weight_norm`` / ``…restored_move_grad_norm``
    became one label and the dump raised). The label is a function of the string alone — the same row
    reads the same in every dump, whatever else is pending. Two thirds of the room go to the TAIL, because
    our keys end in their distinguishing part (``_weight_norm``, ``_bot``, ``_pool``, a run label's seed)."""
    if len(string) <= max_length:
        return string
    digest = key_digest(string)
    room = max_length - len(digest) - 2
    head = room // 3
    return f"{string[:head]}~{digest}~{string[len(string) - (room - head):]}"


def _stdout_split(key: str, tag: str) -> Tuple[str, str, bool]:
    """sb3's grouping, one key: ``(tag, row text, does this key open its tag)``. A key with a ``/`` past
    its first character opens the tag ``<prefix>/``; a key holding the current tag is shown indented
    without it. The tag CARRIES to the next key (sb3's; it matters only for keys with no ``/``)."""
    opens = key.find("/") > 0
    if opens:
        tag = key[: key.find("/") + 1]
    if len(tag) > 0 and tag in key:
        key = f"{'':3}{key[len(tag):]}"
    return tag, key, opens


def _stdout_shown(excluded: Optional[Excluded]) -> bool:
    return not (excluded is not None and ("stdout" in excluded or "log" in excluded))


def metric_key_collisions(keys: Iterable[str], *, max_length: int = STDOUT_MAX_LENGTH) -> List[str]:
    """Every way ``keys`` (a LIST: one entry per producer) fail to stay distinct, as readable lines.

    * ``identity`` — one key written by two producers (the second silently replaces the first);
    * ``stdout`` — two keys whose table rows (`_stdout_split` + `truncate_label`) are the same label
      under the same tag;
    * ``tensorboard`` — two keys written under one event-file tag (a text key ``k`` is written under
      ``k/text_summary``, so it collides with a scalar key spelled that way).

    Empty = every key stays distinct in every format."""
    listed = list(keys)
    out: List[str] = []
    dup = sorted(k for k, c in Counter(listed).items() if c > 1)
    if dup:
        out.append(f"identity: {len(dup)} key(s) have more than one producer: {dup}")
    uniq = sorted(set(listed))
    rows: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    tag = ""
    for key in uniq:
        tag, text, _ = _stdout_split(key, tag)
        rows[(tag, truncate_label(text, max_length))].append(key)
    for (tag, label), ks in sorted(rows.items()):
        if len(ks) > 1:
            out.append(f"stdout: {ks} all render as the row {label!r} under {tag!r}")
    tb: Dict[str, List[str]] = defaultdict(list)
    for key in uniq:
        tb[key].append(key)
        if not key.endswith(TB_TEXT_SUFFIX):
            tb[key + TB_TEXT_SUFFIX].append(key)
    for name, ks in sorted(tb.items()):
        if len(set(ks)) > 1:
            out.append(f"tensorboard: {sorted(set(ks))} can all write the event-file tag {name!r}")
    return out


def check_metric_keys(keys: Iterable[str], *, where: str, max_length: int = STDOUT_MAX_LENGTH) -> None:
    """`MetricKeyCollisionError` naming every collision of `metric_key_collisions`, else nothing."""
    problems = metric_key_collisions(keys, max_length=max_length)
    if problems:
        raise MetricKeyCollisionError(
            f"metric keys collide ({where}; gen3_metric_key_collision_v1) — rename the producer's key, never "
            "the formatter:\n  " + "\n  ".join(problems))


def paired(key_values: Dict[str, Any], key_excluded: Dict[str, Excluded]) -> List[Tuple[str, Any, Excluded]]:
    """``(key, value, exclusion)`` in sorted key order — and the two dicts' KEY SETS must be equal.

    ``ValueError`` otherwise, naming the keys only one side holds. (``zip(..., strict=True)`` over the two
    sorted item lists checks only that they are the same LENGTH.)"""
    only_value = sorted(set(key_values) - set(key_excluded))
    only_excluded = sorted(set(key_excluded) - set(key_values))
    if only_value or only_excluded:
        raise ValueError(
            "the logger's pending values and exclusions do not pair key for key: "
            f"{len(only_value)} key(s) hold a value with no `record` (an indexing READ of the "
            f"name_to_value defaultdict inserts one): {only_value[:5]}; {len(only_excluded)} key(s) hold an "
            f"exclusion with no value: {only_excluded[:5]}")
    return [(k, key_values[k], key_excluded[k]) for k in sorted(key_values)]


class HumanOutputFormat:
    """The stdout table: ``| key | value |`` rows grouped under their ``tag/`` — sb3's, text for text,
    except a row LONGER than 36 (sb3's ``max_length``; ours is 48): sb3 printed ``key[:33] + "..."`` and
    RAISED when two keys truncated alike (mid-run, at the first dump that held both); here a row up to 48
    prints in full and a longer one prints `truncate_label`'s ``<head>~<digest>~<tail>``, which is
    collision-proof. Should two rows still meet (a 24-bit digest clash, or sb3's tag quirk for a key with
    no ``/``), the later one prints its FULL key — a display never stops a run. `check_metric_keys`
    refuses such a key set at startup."""

    def __init__(self, file: TextIO, max_length: int = STDOUT_MAX_LENGTH):
        if max_length < 2 * LABEL_DIGEST_HEX:
            raise ValueError(f"max_length {max_length} cannot hold a shortened label")
        self.max_length = max_length
        self.file = file

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Excluded], step: int = 0) -> None:
        key2str: Dict[Tuple[str, str], str] = {}
        tag = ""
        for key, value, excluded in paired(key_values, key_excluded):
            if not _stdout_shown(excluded):
                continue
            elif isinstance(value, float):
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)
            tag, text, opens = _stdout_split(key, tag)
            if opens:
                key2str[(tag, truncate_label(tag, self.max_length))] = ""
            label = truncate_label(text, self.max_length)
            n = 1
            while (tag, label) in key2str:            # residual collision (class docstring): never raise
                label = key if n == 1 else f"{key} #{n}"
                n += 1
            key2str[(tag, label)] = self._truncate(value_str)
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        key_width = max(map(len, (k[1] for k in key2str.keys())))
        val_width = max(map(len, key2str.values()))
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (_, key), value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    def _truncate(self, string: str) -> str:
        """sb3's VALUE truncation (a value is not a key, so two alike collide with nothing)."""
        if len(string) > self.max_length:
            string = string[: self.max_length - 3] + "..."
        return string

    def write_sequence(self, sequence: Sequence[str]) -> None:
        self.file.write(" ".join(sequence) + "\n")
        self.file.flush()

    def close(self) -> None:
        return None


class TensorBoardOutputFormat:
    """``<folder>``'s event file: a scalar per numeric value, text per string, a histogram per tensor /
    array, at the dump's step; flushed every dump (sb3's, call for call)."""

    def __init__(self, folder: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=folder)
        self._is_closed = False

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Excluded], step: int = 0) -> None:
        assert not self._is_closed, "The SummaryWriter was closed, please re-create one."
        for key, value, excluded in paired(key_values, key_excluded):
            if excluded is not None and "tensorboard" in excluded:
                continue
            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    self.writer.add_text(key, value, step)
                else:
                    self.writer.add_scalar(key, value, step)
            if isinstance(value, (th.Tensor, np.ndarray)):
                self.writer.add_histogram(key, th.as_tensor(value), step)
        self.writer.flush()

    def write_sequence(self, sequence: Sequence[str]) -> None:   # not a sequence writer
        return None

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self._is_closed = True


Writer = Union[HumanOutputFormat, TensorBoardOutputFormat]


class Logger:
    """The pending scalars + their writers (module docstring)."""

    def __init__(self, folder: Optional[str], output_formats: List[Any],
                 strict_producers: Optional[bool] = None):
        self.name_to_value: Dict[str, Any] = defaultdict(float)
        self.name_to_count: Dict[str, int] = defaultdict(int)
        self.name_to_excluded: Dict[str, Excluded] = {}
        #: the call site that last wrote each PENDING key (cleared at a dump; `hold` / `restore` and
        #: `update_fit` carry it with the values, so a held key keeps its producer)
        self.name_to_site: Dict[str, Site] = {}
        #: STRICT: a second call site writing a pending key raises `DuplicateProducerError`; else it is counted
        self.strict_producers = producer_guard_strict() if strict_producers is None else bool(strict_producers)
        #: key -> (first site, second site), for the life of this logger (a lenient logger's record of them)
        self.duplicate_producers: Dict[str, Tuple[Site, Site]] = {}
        self._duplicates_pending: set = set()
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    @staticmethod
    def to_tuple(string_or_tuple: Union[str, Excluded, None]) -> Excluded:
        if string_or_tuple is None:
            return ("",)
        if isinstance(string_or_tuple, tuple):
            return string_or_tuple
        return (string_or_tuple,)

    def _claim(self, key: str, mean: bool = False) -> None:
        """The one-producer guard (module docstring): remember the CALLER of ``record`` / ``record_mean`` as
        ``key``'s producer, and a DIFFERENT site claiming a key that is still pending is a duplicate. (Two
        ``record_mean`` calls are NOT: pending the running mean of records from anywhere is that method's
        contract.) Call it BEFORE the key is written (it reads ``name_to_value`` for membership only)."""
        try:
            frame = sys._getframe(2)             # _claim <- record / record_mean <- the producer
            site: Site = (frame.f_code.co_filename, frame.f_lineno, mean)
        except ValueError:                       # called with no Python caller (a C-level thread target): a guard
            site = ("<no python caller>", 0, mean)   # never stops a run — an unknown site equals itself, so no clash
        prev = self.name_to_site.get(key)
        if (prev is not None and prev[:2] != site[:2] and not (prev[2] and mean)
                and key in self.name_to_value):
            self._duplicate(key, prev, site)
        self.name_to_site[key] = site

    def _duplicate(self, key: str, first: Site, second: Site) -> None:
        msg = (f"metric key {key!r} has TWO producers within one dump interval: {site_str(first)} and "
               f"{site_str(second)} — the later write replaces the earlier one silently "
               "(gen3_metric_producer_guard_v1). Give the quantity ONE producer, or two names if they are "
               "two quantities.")
        if self.strict_producers:
            raise DuplicateProducerError(msg)
        new = key not in self.duplicate_producers
        self.duplicate_producers.setdefault(key, (first, second))
        self._duplicates_pending.add(key)
        if new and len(self.duplicate_producers) <= MAX_REPORTED_DUPLICATE_KEYS:
            print(f"[metric-producers] WARNING: {msg} (counted in {DUPLICATE_PRODUCERS_KEY}; "
                  "reported once per key)", flush=True)

    def record(self, key: str, value: Any, exclude: Union[str, Excluded, None] = None) -> None:
        """Pend ``value`` under ``key`` until the next dump (a later record of the key replaces it — from the
        SAME call site; a different site writing a pending key is a duplicate producer, `_claim`);
        ``exclude`` names the outputs that skip it (``"tensorboard"`` / ``"stdout"``)."""
        self._claim(key)
        self.name_to_value[key] = value
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def record_mean(self, key: str, value: Optional[float], exclude: Union[str, Excluded, None] = None) -> None:
        """``record``, but repeated records of the key pend their running mean."""
        if value is None:
            return
        self._claim(key, mean=True)
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def dump(self, step: int = 0) -> None:
        """Write everything pending at ``step`` to every output, then clear it."""
        if self.level == DISABLED:
            return
        if self._duplicates_pending:             # a lenient logger's count (absent on a clean interval)
            self.name_to_value[DUPLICATE_PRODUCERS_KEY] = float(len(self._duplicates_pending))
            self.name_to_excluded[DUPLICATE_PRODUCERS_KEY] = self.to_tuple(None)
            self._duplicates_pending = set()
        for fmt in self.output_formats:
            fmt.write(self.name_to_value, self.name_to_excluded, step)
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()
        self.name_to_site.clear()

    def log(self, *args: Any, level: int = INFO) -> None:
        if self.level <= level:
            for fmt in self.output_formats:
                fmt.write_sequence(list(map(str, args)))

    def debug(self, *args: Any) -> None:
        self.log(*args, level=DEBUG)

    def info(self, *args: Any) -> None:
        self.log(*args, level=INFO)

    def warn(self, *args: Any) -> None:
        self.log(*args, level=WARN)

    def error(self, *args: Any) -> None:
        self.log(*args, level=ERROR)

    def set_level(self, level: int) -> None:
        self.level = level

    def get_dir(self) -> Optional[str]:
        return self.dir

    def close(self) -> None:
        for fmt in self.output_formats:
            fmt.close()


def configure(folder: Optional[str] = None, format_strings: Sequence[str] = ()) -> Logger:
    """A logger writing ``format_strings`` (a subset of `FORMATS`): ``stdout`` the table on ``sys.stdout``,
    ``tensorboard`` an event file in ``folder``. ``configure(None, [])`` is the NULL logger (records,
    reads and dumps; writes nothing, creates nothing — the K9 golden's and the startup dry update's)."""
    formats = [f for f in format_strings if f]
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"logger output format(s) {unknown} are not served (only {FORMATS})")
    outs: List[Any] = []
    for f in formats:
        if f == "stdout":
            outs.append(HumanOutputFormat(sys.stdout))
        else:
            if folder is None:
                raise ValueError("a tensorboard logger needs a folder")
            os.makedirs(folder, exist_ok=True)
            outs.append(TensorBoardOutputFormat(folder))
    logger = Logger(folder=folder, output_formats=outs)
    if formats and formats != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger
