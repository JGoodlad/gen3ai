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
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

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


# ---------------------------------------------------------------------------------------------
# The TRAINING transport's frame (`gen3_core_obs_source_v1`, program M6): `sim_bridge` in core-obs
# mode ships `__OBS__ <side> <json>` BEFORE the chunk that carries the side's request, so the row is
# stashed by the time poke-env hands the decision to the env. `Gen3Env(obs_source="core")` takes
# its row and mask from here.
# ---------------------------------------------------------------------------------------------

#: The obs sources a training env can read its trainee row from.
OBS_SOURCES = ("python", "core")


class CoreObsMismatch(RuntimeError):
    """The core's frame does not belong to the decision the env is taking (another battle, a
    stale request), is missing, or disagrees with the reading's mask — the env REFUSES (crash over
    corruption: a PPO transition built on the wrong row is worse than a restart)."""


@dataclass(frozen=True)
class CoreObsFrame:
    tag: Optional[str]
    side: str
    n: int
    row: np.ndarray
    mask: np.ndarray
    turn: int
    rqid: Optional[int]


def decode_frame(payload: str, tag: Optional[str], side: str) -> CoreObsFrame:
    """One ``__OBS__`` payload (``{"frame", "mask", "tokens", "turn", "line", "rqid", "n"}``) as a
    frame: the row wrapped by :func:`wrap_row` (refused, never converted), the mask 11 ints in
    ``{0, 1}``, ``n`` the side's decision index in the battle."""
    d = json.loads(payload)
    row = wrap_row(d["frame"])
    mask = d.get("mask")
    if not isinstance(mask, list) or len(mask) != 11 or any(m not in (0, 1) for m in mask):
        raise RowRefused(f"core mask {mask!r} is not 11 bits")
    if not isinstance(d.get("n"), int):
        raise RowRefused(f"core frame carries no decision index n: {sorted(d)}")
    rqid = d.get("rqid")
    return CoreObsFrame(tag=tag, side=side, n=d["n"], row=row, mask=np.asarray(mask, dtype=np.int8),
                        turn=int(d.get("turn") or 0), rqid=None if rqid is None else int(rqid))


def frame_for_decision(battle, raw: Optional[tuple], n: int, python_mask: np.ndarray) -> CoreObsFrame:
    """The core frame for ``battle``'s decision number ``n`` (0-based, this battle), or
    :class:`CoreObsMismatch`. ``raw`` is the bridge session's stash for the side, ``(tag it arrived
    under, the __OBS__ JSON)``.

    It must be this battle's (the bridge tag), be the ``n``-th decision the core took for the side
    (the core and the reading decide at the same requests — slices T / O's alignment), sit at the
    reading's turn, carry no NaN cell (the self-check build's poison — an unwritten cell), and its
    mask must equal the reading's (the env maps the chosen action through the reading's legality)."""
    if raw is None:
        raise CoreObsMismatch(f"{battle.battle_tag}: no core obs frame for decision {n}")
    frame = decode_frame(raw[1], raw[0], battle.player_role or "?")
    if frame.tag != battle.battle_tag:
        raise CoreObsMismatch(f"core obs frame is for {frame.tag}, the decision is in {battle.battle_tag}")
    if frame.n != n:
        raise CoreObsMismatch(f"{battle.battle_tag}: core obs frame is decision {frame.n}, the reading is "
                              f"taking decision {n} (turn {battle.turn})")
    if frame.turn != battle.turn:
        raise CoreObsMismatch(f"{battle.battle_tag} decision {n}: core turn {frame.turn} != reading turn {battle.turn}")
    if np.isnan(frame.row).any():
        bad = np.flatnonzero(np.isnan(frame.row))[:8].tolist()
        raise CoreObsMismatch(f"{battle.battle_tag}: core row has unwritten (NaN) cells {bad}")
    if not np.array_equal(frame.mask, np.asarray(python_mask, dtype=np.int8)):
        raise CoreObsMismatch(f"{battle.battle_tag} turn {battle.turn}: core mask {frame.mask.tolist()} != "
                              f"the reading's {np.asarray(python_mask).tolist()}")
    return frame


def fresh_frame_index(raw: Optional[tuple], tag: Optional[str]) -> Optional[int]:
    """The decision index ``n`` of the stashed frame if it belongs to battle ``tag``, else None
    (cheap: reads only ``"n"`` — the row is decoded once, by :func:`frame_for_decision`)."""
    if raw is None or raw[0] != tag:
        return None
    return json.loads(raw[1]).get("n")
