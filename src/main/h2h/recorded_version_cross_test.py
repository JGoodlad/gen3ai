"""THE CLOSING TEST'S CROSS SHAPE on ``main.h2h``'s engine (gen3_recorded_version_gate_v1, 2026-10-11): an end-state
(``--arch production``) x legacy-production plan with TWO end-state checkpoints declares a two-group engine.

Since the END STATE's ADOPTION (2026-10-12) ``--arch endstate`` is an ALIAS of ``--arch production`` — the end state
IS production — so the closing test's pair "end state x production" became one architecture. Its two ARCHITECTURES
survive as the end state (``--arch production``) and the LEGACY production network (the nineteen end-state keys at
their pre-adoption values: ``main.train.production_args.legacy_production_args``, the h2h ``foreign`` fixture's own
network), which is what a cross of two DIFFERENT architectures must still declare as two groups.

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

from agents.training.rust_eval.recorded_version_gate_test import write_arm_snapshot, write_snapshot


@pytest.fixture(scope="module")
def cross(tmp_path_factory):
    from main.h2h.play import resolve_player
    from main.train.production_args import legacy_production_args

    root = tmp_path_factory.mktemp("h2h_cross")
    _, e1 = write_arm_snapshot(root / "run_es_end_a", "production", seed=11)       # the end state IS production
    _, e2 = write_arm_snapshot(root / "run_es_end_b", "production", seed=12)
    legacy = legacy_production_args()
    assert legacy.move_resolution == "off"                      # the legacy network holds none of the retired blocks
    _, p1 = write_snapshot(root / "run_legacy_prod_a", legacy, seed=21)
    return [resolve_player(x) for x in (e1, e2, p1)]


def test_an_endstate_checkpoint_is_its_own_groups_architecture(cross):
    from main.h2h.arch import EngineArch, _load_host

    e1, e2, _p = cross
    m1, m2 = _load_host(e1), _load_host(e2)
    arch = EngineArch.of(e1, m1)
    assert arch.differs(e1, m1) is None
    assert arch.differs(e2, m2) is None              # another end-state seed: the SAME architecture


@pytest.mark.parametrize("end_side", ["player", "opponent"])
def test_the_endstate_x_legacy_production_cross_declares_two_groups(cross, end_side):
    from main.h2h.arch import _load_host, declare_engine

    e1, e2, p = cross
    cells = [(e1, p), (e2, p)] if end_side == "player" else [(p, e1), (p, e2)]
    decl = declare_engine(cells, _load_host)
    assert len(decl.archs) == 2
    assert {a.toggles["move_resolution"] for a in decl.archs} == {"on", "off"}
