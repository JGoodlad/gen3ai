"""The learner's LOGGER — ours (`gen3_owned_logger_v1`; `designs/endstate/design_own_ppo_loop.md` §3.3).

Until deletion pass U4 the trainer logged through sb3's ``Logger`` (``configure(<run>/tb, ["stdout",
"tensorboard"])``). This is that logger, written out operation for operation for the two outputs the
repo uses — the stdout table and TensorBoard — so the event files carry the SAME tags, steps and values
and the stdout tables the same text. Any other output format is REFUSED.

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

The writers pair ``name_to_value`` with ``name_to_excluded`` KEY FOR KEY (:func:`paired`): the two key
sets must be EQUAL, or the dump raises ``ValueError`` naming the keys that are in one and not the other.
A key that reaches ``name_to_value`` without ``record`` (an indexing READ of the defaultdict inserts 0.0)
fails the dump loudly instead of writing a phantom 0. sb3 paired the two sorted lists with
``zip(strict=True)``, which checks LENGTHS only — a phantom value key plus a stranded exclusion key
(one dict edited without the other) is the same length, pairs the wrong rows and drops or keeps a value
under another key's exclusion; here that is the same loud error (P10 follow-up F1, item 5).
"""
from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import torch as th

DEBUG, INFO, WARN, ERROR, DISABLED = 10, 20, 30, 40, 50

#: The output formats `configure` builds (sb3 also had log / json / csv — never used here).
FORMATS = ("stdout", "tensorboard")

Excluded = Tuple[str, ...]


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
    """The stdout table: ``| key | value |`` rows grouped under their ``tag/`` (sb3's, text for text)."""

    def __init__(self, file: TextIO, max_length: int = 36):
        self.max_length = max_length
        self.file = file

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Excluded], step: int = 0) -> None:
        key2str: Dict[Tuple[str, str], str] = {}
        tag = ""
        for key, value, excluded in paired(key_values, key_excluded):
            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue
            elif isinstance(value, float):
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)
            if key.find("/") > 0:
                tag = key[: key.find("/") + 1]
                key2str[(tag, self._truncate(tag))] = ""
            if len(tag) > 0 and tag in key:
                key = f"{'':3}{key[len(tag):]}"
            truncated_key = self._truncate(key)
            if (tag, truncated_key) in key2str:
                raise ValueError(f"Key '{key}' truncated to '{truncated_key}' that already exists. "
                                 "Consider increasing `max_length`.")
            key2str[(tag, truncated_key)] = self._truncate(value_str)
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

    def __init__(self, folder: Optional[str], output_formats: List[Any]):
        self.name_to_value: Dict[str, Any] = defaultdict(float)
        self.name_to_count: Dict[str, int] = defaultdict(int)
        self.name_to_excluded: Dict[str, Excluded] = {}
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

    def record(self, key: str, value: Any, exclude: Union[str, Excluded, None] = None) -> None:
        """Pend ``value`` under ``key`` until the next dump (a later record of the key replaces it);
        ``exclude`` names the outputs that skip it (``"tensorboard"`` / ``"stdout"``)."""
        self.name_to_value[key] = value
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def record_mean(self, key: str, value: Optional[float], exclude: Union[str, Excluded, None] = None) -> None:
        """``record``, but repeated records of the key pend their running mean."""
        if value is None:
            return
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def dump(self, step: int = 0) -> None:
        """Write everything pending at ``step`` to every output, then clear it."""
        if self.level == DISABLED:
            return
        for fmt in self.output_formats:
            fmt.write(self.name_to_value, self.name_to_excluded, step)
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

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
