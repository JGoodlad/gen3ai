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


# --- the training transport's frame (gen3_core_obs_source_v1) ---------------------------------

def _payload(row, n=0, turn=3, mask=None):
    import json

    return json.dumps({"frame": _frame(row), "mask": mask if mask is not None else [1] * 4 + [0] * 7,
                       "tokens": {}, "turn": turn, "line": 10, "rqid": None, "n": n})


class _Battle:
    battle_tag, turn, player_role = "battle-gen3ou-7", 3, "p1"


def _ok_row():
    return np.full(obs_dim(), 0.25, dtype="<f4")


def test_a_frame_is_taken_only_for_its_own_battle_decision_and_turn():
    from agents.battle.core_obs import CoreObsMismatch, frame_for_decision

    mask = np.array([1] * 4 + [0] * 7, dtype=np.int8)
    f = frame_for_decision(_Battle(), ("battle-gen3ou-7", _payload(_ok_row(), n=2)), 2, mask)
    assert f.n == 2 and f.row.tobytes() == _ok_row().tobytes()
    for raw, n, why in (
            (None, 0, "no core obs frame"),
            (("battle-gen3ou-6", _payload(_ok_row())), 0, "is for battle-gen3ou-6"),
            (("battle-gen3ou-7", _payload(_ok_row(), n=1)), 2, "is decision 1"),
            (("battle-gen3ou-7", _payload(_ok_row(), turn=4)), 0, "core turn 4"),
            (("battle-gen3ou-7", _payload(_ok_row(), mask=[1] * 5 + [0] * 6)), 0, "core mask")):
        with pytest.raises(CoreObsMismatch, match=why):
            frame_for_decision(_Battle(), raw, n, mask)


def test_an_unwritten_nan_cell_is_refused_not_trained_on():
    from agents.battle.core_obs import CoreObsMismatch, frame_for_decision

    row = _ok_row()
    row[853] = np.nan
    with pytest.raises(CoreObsMismatch, match="unwritten"):
        frame_for_decision(_Battle(), ("battle-gen3ou-7", _payload(row)), 0,
                           np.array([1] * 4 + [0] * 7, dtype=np.int8))


def test_a_frame_without_a_decision_index_or_with_a_bad_mask_is_refused():
    import json

    from agents.battle.core_obs import decode_frame

    d = json.loads(_payload(_ok_row()))
    d.pop("n")
    with pytest.raises(RowRefused, match="decision index"):
        decode_frame(json.dumps(d), "t", "p1")
    with pytest.raises(RowRefused, match="11 bits"):
        decode_frame(_payload(_ok_row(), mask=[2] * 11), "t", "p1")


def test_fresh_frame_index_reads_only_its_own_battle():
    from agents.battle.core_obs import fresh_frame_index

    assert fresh_frame_index(("b1", _payload(_ok_row(), n=5)), "b1") == 5
    assert fresh_frame_index(("b0", _payload(_ok_row(), n=5)), "b1") is None
    assert fresh_frame_index(None, "b1") is None
