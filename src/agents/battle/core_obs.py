"""The core's observation ROW on the Python side of the wire (``gen3_core_obs_wire_v1``, the Rust
Core Program's M4 transport).

The Rust encoder (``src/rust_sim/src/encoder/``) writes the 2501-dim float32 row of one side and
ships it in the reply of a pipe that already exists (``core_events --obs``, ``search_driver``'s core
road) as a self-describing frame ``{"dtype": "<f4", "shape": [2501], "b64": …}``. :func:`wrap_row`
turns it into an ``np.ndarray`` with ``np.frombuffer`` — ZERO conversion — and REFUSES, never
converts, anything that is not exactly the observation: a wrong dtype, a wrong shape, a byte count
that is not ``4 × dim``. :func:`check_row` is the same contract for an array handed in directly (a
wrong dtype, shape or a non-contiguous view is refused, not copied into shape).
"""

from __future__ import annotations

import base64
from typing import Any, Mapping

import numpy as np

#: The one dtype a row may carry: little-endian float32.
DTYPE = "<f4"


class RowRefused(ValueError):
    """A row whose dtype, shape, length or layout is not the observation's — refused, never
    converted (a converted row is a silently different observation)."""


def obs_dim() -> int:
    from agents.observation import constants as C

    return int(C.OFFSET_EVENT_WINDOW + C.EVENT_WINDOW_DIM)


def wrap_row(frame: Mapping[str, Any], dim: int | None = None) -> np.ndarray:
    """The frame's row as a read-only float32 array over its own bytes (``np.frombuffer``)."""
    dim = obs_dim() if dim is None else dim
    if frame.get("dtype") != DTYPE:
        raise RowRefused(f"row dtype {frame.get('dtype')!r} is not {DTYPE!r}")
    shape = frame.get("shape")
    if not isinstance(shape, list) or shape != [dim]:
        raise RowRefused(f"row shape {shape!r} is not [{dim}]")
    raw = base64.b64decode(frame["b64"], validate=True)
    if len(raw) != 4 * dim:
        raise RowRefused(f"row carries {len(raw)} bytes, the observation is {4 * dim}")
    return np.frombuffer(raw, dtype=DTYPE)


def check_row(row: Any, dim: int | None = None) -> np.ndarray:
    """``row`` itself when it IS an observation row (float32, shape ``(dim,)``, C-contiguous);
    :class:`RowRefused` otherwise."""
    dim = obs_dim() if dim is None else dim
    if not isinstance(row, np.ndarray):
        raise RowRefused(f"row is a {type(row).__name__}, not an ndarray")
    if row.dtype != np.dtype(np.float32):
        raise RowRefused(f"row dtype {row.dtype} is not float32")
    if row.shape != (dim,):
        raise RowRefused(f"row shape {row.shape} is not ({dim},)")
    if not row.flags.c_contiguous:
        raise RowRefused("row is not C-contiguous")
    return row
