"""The FFI front end's GENERATED signatures — the routine pin (M5 Lane A).

``src/rust_env/src/ffi.rs``'s generated region must be exactly what ``ffi.render`` makes from the
``FUNCTIONS`` table today (the ``columns_test.py`` pattern), and the table must be consistent with
the column contract. No build, no library: the loaded-library gates are ``ffi_integration_test.py``.
"""
import ctypes

import pytest

from utils.rust_env import columns as C
from utils.rust_env import ffi


def test_the_committed_generated_region_is_current():
    cur = ffi.FFI_RS.read_text()
    assert ffi.render(cur) == cur, "src/rust_env/src/ffi.rs is STALE — run `python -m utils.rust_env.ffi --write`"


def test_every_row_is_generated_and_carries_the_table_id():
    region = ffi.render_region()
    for f in ffi.FUNCTIONS:
        assert f"pub unsafe extern \"C\" fn {f.name}(" in region, f.name
        assert f"imp::{f.name.removeprefix('rust_env_')}(" in region, f.name
    assert f'pub const FFI_SIG_ID: &str = "{ffi.sig_id()}";' in region


def test_the_table_id_moves_with_the_column_schema_and_the_signatures(monkeypatch):
    base = ffi.sig_id()
    assert C.schema_id() in ffi.sig_text() and f"columns {len(C.COLUMNS)}" in ffi.sig_text()
    rows = list(ffi.FUNCTIONS)
    i = next(k for k, f in enumerate(rows) if f.name == "rust_env_dispatch")
    rows[i] = ffi.Fn(rows[i].name, (("h", "handle"), ("op", "i32"), ("addrs", "addrs")), "i32", rows[i].doc)
    monkeypatch.setattr(ffi, "FUNCTIONS", tuple(rows))
    assert ffi.sig_id() != base, "an argument's type change must change the id the loader compares"


def test_the_ctypes_mapping_covers_every_abi_type():
    for f in ffi.FUNCTIONS:
        for _, t in f.args:
            assert ffi.ABI[t][1] is not None
    assert ffi.ABI["usize"][1] is ctypes.c_size_t


def test_a_relative_path_is_refused_before_anything_is_loaded(tmp_path):
    with pytest.raises(ffi.FfiLoadError, match="ABSOLUTE"):
        ffi.load("target/selfcheck/libpokesim_env.so")
    with pytest.raises(ffi.FfiLoadError, match="does not exist"):
        ffi.load(tmp_path / "libpokesim_env.so")
