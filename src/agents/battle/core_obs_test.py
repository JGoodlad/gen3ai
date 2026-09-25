"""The obs row's wire contract (``gen3_core_obs_wire_v1``): a frame is wrapped with ZERO conversion,
and anything that is not exactly the observation — dtype, shape, byte length, contiguity — is
REFUSED, never converted."""

import base64

import numpy as np
import pytest

from agents.battle.core_obs import DTYPE, RowRefused, check_row, obs_dim, wrap_row


def _frame(row, dtype=DTYPE, shape=None):
    return {"dtype": dtype, "shape": [len(row)] if shape is None else shape,
            "b64": base64.b64encode(np.asarray(row).tobytes()).decode()}


def test_a_frame_wraps_its_own_bytes_without_conversion():
    dim = obs_dim()
    row = np.arange(dim, dtype="<f4") / 7
    got = wrap_row(_frame(row))
    assert got.dtype == np.float32 and got.shape == (dim,) and got.tobytes() == row.tobytes()
    assert not got.flags.writeable, "np.frombuffer over the frame's bytes — a view, not a copy"
    assert check_row(got) is got


def test_the_wire_refuses_a_wrong_dtype_shape_or_length():
    dim = obs_dim()
    row = np.zeros(dim, dtype="<f4")
    with pytest.raises(RowRefused, match="dtype"):
        wrap_row(_frame(row.astype("<f8"), dtype="<f8"))
    with pytest.raises(RowRefused, match="dtype"):
        wrap_row(_frame(row, dtype=">f4"))
    with pytest.raises(RowRefused, match="shape"):
        wrap_row(_frame(row, shape=[dim - 1]))
    with pytest.raises(RowRefused, match="shape"):
        wrap_row(_frame(row, shape=[1, dim]))
    with pytest.raises(RowRefused, match="bytes"):
        wrap_row(dict(_frame(row), b64=base64.b64encode(row[:-1].tobytes()).decode()))


def test_an_array_is_refused_not_converted():
    dim = obs_dim()
    with pytest.raises(RowRefused, match="dtype"):
        check_row(np.zeros(dim, dtype=np.float64))
    with pytest.raises(RowRefused, match="shape"):
        check_row(np.zeros((1, dim), dtype=np.float32))
    with pytest.raises(RowRefused, match="contiguous"):
        check_row(np.zeros(2 * dim, dtype=np.float32)[::2])
    with pytest.raises(RowRefused, match="ndarray"):
        check_row([0.0] * dim)
