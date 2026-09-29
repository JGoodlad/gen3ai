"""The process front end's WIRE — the routine pin (M5 Lane B).

``src/rust_env/src/shm.rs``'s generated region must be exactly what ``proc.render`` makes from the
wire table today (the ``ffi_test.py`` pattern), the layout must be the column table's, and the
handshake check must refuse a foreign child on every field. No build, no child: the spawned-child
gates are ``proc_integration_test.py``.
"""
import pytest

from utils.rust_env import columns as C
from utils.rust_env import proc
from utils.rust_env import protocol as P
from utils.rust_env import stamp as S


def test_the_committed_generated_region_is_current():
    cur = proc.SHM_RS.read_text()
    assert proc.render(cur) == cur, "src/rust_env/src/shm.rs is STALE — run `python -m utils.rust_env.proc --write`"


def test_the_region_carries_the_wire_id_and_every_control_byte():
    region = proc.render_region()
    assert f'pub const WIRE_ID: &str = "{proc.wire_id()}";' in region
    for c in proc.CONTROL:
        assert f"pub const {c.name}: u8 = b'{chr(c.code)}';" in region, c.name


def test_the_wire_id_moves_with_the_column_schema_and_the_control_table(monkeypatch):
    base = proc.wire_id()
    assert C.schema_id() in proc.wire_text() and f"columns {len(C.COLUMNS)}" in proc.wire_text()
    monkeypatch.setattr(proc, "CONTROL", proc.CONTROL + (proc.Ctl("X", ord("X"), "x"),))
    assert proc.wire_id() != base, "a new control byte must change the id the handshake compares"


def test_no_control_byte_shadows_a_core_opcode():
    assert not {c.code for c in proc.CONTROL} & {o.code for o in P.OPS}


def test_the_layout_is_the_column_table_aligned_and_disjoint():
    for n in (1, 3, 48):
        off, total = proc.layout(n, 2761)
        sizes = C.nbytes(n, 2761)
        names = [c.name for c in C.COLUMNS]
        assert off[names[0]] == proc.HEADER_BYTES
        for i, k in enumerate(names):
            assert off[k] % proc.ALIGN == 0
            end = off[k] + sizes[k]
            assert end <= (off[names[i + 1]] if i + 1 < len(names) else total)


def _hs(**over):
    d = {"wire": proc.wire_id(), "obs_dim": "2761", "n_columns": str(len(C.COLUMNS)), "schema": C.schema_id(),
         "stamp": "stamp=v1;commit=x;src=0;nfiles=0;nan_poison=1;schema=x;data=/nowhere"}
    d.update(over)
    return d


def test_the_handshake_refuses_a_foreign_child_on_every_field():
    with pytest.raises(proc.ProcLoadError, match="not a rust env process"):
        proc.parse_handshake("hello\tworld\n")
    line = proc.HANDSHAKE_TAG + "\t" + "\t".join(f"{k}={v}" for k, v in _hs().items()) + "\n"
    assert proc.parse_handshake(line)["wire"] == proc.wire_id()
    with pytest.raises(proc.ProcLoadError, match="wire"):
        proc.check_handshake(_hs(wire="0" * 16), "child")
    with pytest.raises(proc.ProcLoadError, match="column schema"):
        proc.check_handshake(_hs(schema="0" * 16), "child")
    with pytest.raises(proc.ProcLoadError, match="column schema"):
        proc.check_handshake(_hs(n_columns="1"), "child")
    with pytest.raises(S.StampMismatch, match="sources"):
        proc.check_handshake(_hs(), "child")  # a forged stamp: this tree's sources do not hash to 0


def test_a_relative_or_missing_binary_is_refused_before_anything_is_spawned(tmp_path):
    with pytest.raises(proc.ProcLoadError, match="ABSOLUTE"):
        proc.ProcCore("{}", binary="target/selfcheck/rust_env_proc")
    with pytest.raises(proc.ProcLoadError, match="does not exist"):
        proc.ProcCore("{}", binary=tmp_path / "rust_env_proc")
