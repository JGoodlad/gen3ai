"""THE CLOSING TEST'S CROSS SHAPE on ``main.h2h``'s engine (gen3_recorded_version_gate_v1, 2026-10-11): an
``--arch endstate`` x ``--arch production`` plan with TWO end-state checkpoints declares a two-group engine.

The failure this pins (measured at ``b132b099`` on the real ``rb_es_end_s2001`` / ``rb_es_prod_s2001`` checkpoints):
``EngineArch`` built its version from toggles read off the BUILT modules, so every end-state checkpoint "differed"
from its own group (``intent_move_cell mismatch: saved=True, current=False`` — ``--move-resolution on`` retires the
seven per-action blocks its config records True), the second end-state checkpoint matched neither group, and
``declare_engine`` raised ``CellArchMismatch``. A revert of ``arch_toggles_from_model`` / ``EngineArch.of`` to a
built-module read fails both tests here.

Run: python -m pytest src/main/h2h/recorded_version_cross_test.py -q
"""
from __future__ import annotations

import pytest

from agents.training.rust_eval.recorded_version_gate_test import write_arm_snapshot


@pytest.fixture(scope="module")
def cross(tmp_path_factory):
    from main.h2h.play import resolve_player

    root = tmp_path_factory.mktemp("h2h_cross")
    _, e1 = write_arm_snapshot(root / "run_es_end_a", "endstate", seed=11)
    _, e2 = write_arm_snapshot(root / "run_es_end_b", "endstate", seed=12)
    _, p1 = write_arm_snapshot(root / "run_es_prod_a", "production", seed=21)
    return [resolve_player(x) for x in (e1, e2, p1)]


def test_an_endstate_checkpoint_is_its_own_groups_architecture(cross):
    from main.h2h.arch import EngineArch, _load_host

    e1, e2, _p = cross
    m1, m2 = _load_host(e1), _load_host(e2)
    arch = EngineArch.of(e1, m1)
    assert arch.differs(e1, m1) is None
    assert arch.differs(e2, m2) is None              # another end-state seed: the SAME architecture


@pytest.mark.parametrize("end_side", ["player", "opponent"])
def test_the_endstate_x_production_cross_declares_two_groups(cross, end_side):
    from main.h2h.arch import _load_host, declare_engine

    e1, e2, p = cross
    cells = [(e1, p), (e2, p)] if end_side == "player" else [(p, e1), (p, e2)]
    decl = declare_engine(cells, _load_host)
    assert len(decl.archs) == 2
    assert {a.toggles["move_resolution"] for a in decl.archs} == {"on", "off"}
