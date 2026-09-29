"""The Rust env's column contract: ONE table, rendered into Rust, pinned (M5 Lane 0, gate ③).

``src/rust_env/src/core/columns.rs`` is GENERATED from ``columns.py`` + ``protocol.py``. This routine
pin fails the day the committed file is stale, so a lane that edits the table and forgets the
regeneration fails here, not as a byte-misaligned column deep in a front end.
"""
import dataclasses

import numpy as np

from utils.rust_env import columns as C
from utils.rust_env import protocol as P


def test_the_committed_columns_rs_is_current():
    assert C.OUT.read_text() == C.render(), (
        "src/rust_env/src/core/columns.rs is STALE — run `python -m utils.rust_env.columns --write` "
        "and rebuild the rust env")


def test_the_schema_id_is_rendered_and_moves_with_the_table(monkeypatch):
    assert f'pub const SCHEMA_ID: &str = "{C.schema_id()}";' in C.render()
    before = C.schema_id()
    # Teeth: any change to a column's dtype moves the id (and therefore the rendering).
    changed = tuple(dataclasses.replace(c, dtype="f32") if c.name == "mask" else c for c in C.COLUMNS)
    monkeypatch.setattr(C, "COLUMNS", changed)
    assert C.schema_id() != before
    assert C.render() != C.OUT.read_text()


def test_the_rendering_is_not_vacuous():
    text = C.render()
    for c in C.COLUMNS:
        assert f'name: "{c.name}"' in text, c.name
        if c.per_env:
            assert f"pub {c.name}: &'a" in text, c.name
    for o in P.OPS:
        assert f"pub const {o.name}: u8 = b'{chr(o.code)}';" in text
    for s in P.STATUSES:
        assert f"pub const {s.name}: i32 = {s.code};" in text
    for i, k in enumerate(P.COUNTERS):
        assert f"pub const {k.name}: usize = {i};" in text
    # OBS_DIM is the port's constant, never a literal copy that could drift.
    assert "pub const OBS_DIM: usize = pokesim::encoder::OBS_DIM;" in text


def test_the_dims_agree_with_the_python_encoder_and_the_action_space():
    from agents.action.constants import ACTION_SPACE_SIZE
    from agents.observation import constants as OC

    assert C.FIXED_DIMS["ACT"] == ACTION_SPACE_SIZE
    obs_dim = OC.OFFSET_EVENT_WINDOW + OC.EVENT_WINDOW_DIM
    sh = C.shapes(3, obs_dim)
    assert sh["obs"] == (3, 2, obs_dim)
    assert sh["counters"] == (len(P.COUNTERS),)


def test_allocate_matches_the_table():
    cols = C.allocate(5, 17)
    nb = C.nbytes(5, 17)
    for c in C.COLUMNS:
        a = cols[c.name]
        assert a.flags.c_contiguous, c.name
        assert a.dtype == np.dtype(C.numpy_dtype(c)), c.name
        assert a.nbytes == nb[c.name], c.name
    assert (cols["action"] == -1).all()


def test_the_lifecycle_counters_are_declared():
    after = [k.name for k in P.COUNTERS if k.after_freeze]
    assert after and all(n.endswith("_AFTER_FREEZE") for n in after)
    assert all(k.name.endswith("_AFTER_FREEZE") == k.after_freeze for k in P.COUNTERS)


def test_every_failure_status_maps_to_a_typed_class():
    for s in P.STATUSES:
        if s.exc is None:
            continue
        e = P.error_for(s.code, "x")
        assert type(e).__name__ == s.exc and isinstance(e, P.RustEnvError) and e.status == s.code
    assert isinstance(P.error_for(99, "x"), P.CoreFault)
