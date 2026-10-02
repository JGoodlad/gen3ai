"""The Rust env's COLUMN CONTRACT — the ONE table both languages read (M5 Lane 0).

Every array that crosses between the Python host and the Rust env core is a row of ``COLUMNS``:
its name, dtype, shape (symbolic), direction and owning lane. ``render()`` turns this table and
``protocol.py``'s ops / statuses / counters into ``src/rust_env/src/core/columns.rs``
(``python -m utils.rust_env.columns --write``); ``columns_test.py`` (routine) fails the day the
committed file is stale. Both front ends allocate / map the columns from THIS table
(``shapes`` / ``allocate``), so a front end can never disagree with the core about a layout.

Layout rules (every column):
  * caller-allocated, C-contiguous, little-endian, row-major in the order of ``dims``;
  * a PER-ENV column's first dim is ``N`` and env ``i`` owns exactly its ``i``-th row — a worker
    thread touches only its own envs' rows, so the columns need no lock;
  * a POOL column has no ``N`` dim (the ``counters``);
  * ``in`` is read by the core and never written; ``out`` is written by the core and never read.

Symbolic dims: ``N`` (envs in the pool, declared at startup), ``SIDES`` = 2, ``ACT`` = 11
(``agents.action.constants.ACTION_SPACE_SIZE``), ``SEED_WORDS`` = 4 (a Showdown seed "a,b,c,d"),
``OBS_DIM`` (the encoder's row; in Rust it IS ``pokesim::encoder::OBS_DIM`` — never a literal here —
and a front end reads it from the loaded core), ``NCOUNTERS`` (``len(protocol.COUNTERS)``).

A lane that needs a new column adds a row here (its ``owner``) and regenerates; nothing else.

The LABEL columns (Lane C) are not hand-written rows: one per ``core`` row of
``label_inventory.LABELS``, named by its ``Gen3Env`` key, with its dtype and ``(N, SIDES, *shape)``,
so the table the inventory test pins against ``Gen3Env`` is the table the columns are built from.
A label column is written iff ``need`` = 1 AND its family is declared in the spec's ``labels``;
otherwise it is stale. ``LABEL_FAMILIES`` (rendered as ``labels::FAMILIES``) maps each ``core``
family to its columns; the spec refuses a ``host_*`` family by name.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple, Union

from utils.paths import src_path
from utils.rust_env import label_inventory as LI
from utils.rust_env import protocol as P

Dim = Union[str, int]

#: The fixed symbolic dims (OBS_DIM and N are supplied at allocation time).
FIXED_DIMS: Dict[str, int] = {"SIDES": 2, "ACT": 11, "SEED_WORDS": 4, "NCOUNTERS": len(P.COUNTERS)}

_DTYPES = {  # table dtype -> (rust type, numpy dtype string, bytes)
    "f32": ("f32", "<f4", 4),
    "u8": ("u8", "u1", 1),
    "i32": ("i32", "<i4", 4),
    "u32": ("u32", "<u4", 4),
    "u64": ("u64", "<u8", 8),
    "i64": ("i64", "<i8", 8),
}


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    dims: Tuple[Dim, ...]
    direction: str  # "in" | "out"
    owner: str  # the lane that defined it
    doc: str

    @property
    def per_env(self) -> bool:
        return bool(self.dims) and self.dims[0] == "N"


COLUMNS: Tuple[Column, ...] = (
    # ---- inputs
    Column("action", "i32", ("N", "SIDES"), "in", "0",
           "STEP: the action index (0..ACT) for every (env, side) with need = 1; ignored where need = 0"),
    Column("ep_team", "u32", ("N", "SIDES"), "in", "0",
           "the NEXT episode's teams, as indices into the team table declared at startup (p1, p2); read at "
           "RESET and at every auto-reset, so the caller keeps it staged for the episode after the current one"),
    Column("ep_seed", "u32", ("N", "SEED_WORDS"), "in", "0",
           "the NEXT episode's Showdown seed \"a,b,c,d\" (each word < 65536); read with ep_team"),
    Column("ep_opp", "u32", ("N",), "in", "E",
           "the NEXT episode's OPPONENT ROUTE, an index into the spec's `opponents` table (M5 Lane E, "
           "`crate::opponents`); read with ep_team / ep_seed, so the caller keeps it staged one episode ahead"),
    # ---- outputs
    Column("obs", "f32", ("N", "SIDES", "OBS_DIM"), "out", "0",
           "the side's observation row (`BattleVersion::encode` on its PARSE chain — the row sim_bridge's "
           "__OBS__ ships); written iff need = 1, stale otherwise"),
    Column("mask", "u8", ("N", "SIDES", "ACT"), "out", "0",
           "`present::mask` — the side's 11-dim legal-action mask; written iff need = 1"),
    Column("need", "u8", ("N", "SIDES"), "out", "0",
           "1 iff the caller must supply this side's action at the next STEP (a decision is open)"),
    Column("reward", "f32", ("N",), "out", "D",
           "the TERMINAL reward from p1's (the trainee's) view of an episode that ENDED this op — "
           "`Gen3RewardManager.process_turn_reward` under the spec's `terminal` (indicator: victory_value on a "
           "win, 0 otherwise; signed: +victory_value / -victory_value, draw_penalty when the end turn >= "
           "timeout_turn_cap); 0 on every other op and for a battle QUARANTINED in progress. It SURVIVES a "
           "refused auto-reset (F-L0-2): the ended episode's reward/terminated/truncated stand"),
    Column("done", "u8", ("N",), "out", "0",
           "1 iff an episode the caller saw ended this op (terminated, truncated, or quarantined in progress); "
           "the env has auto-reset, so obs / mask / need already describe the NEXT episode's first decision "
           "(EnvPool semantics) — unless the auto-reset was refused (then the env is PARKED: need = 0 0)"),
    Column("terminated", "u8", ("N",), "out", "D",
           "`PokeEnv.calc_term_trunc` from p1's reading of the ended battle: exactly ONE side wiped (p1's "
           "team size minus each side's fainted count)"),
    Column("truncated", "u8", ("N",), "out", "D",
           "`PokeEnv.calc_term_trunc`: the battle ended with neither or both sides wiped — the stall forfeit "
           "(p1 FORCELOSEs at its first decision with turn >= turn_limit) and a tie. RAW env flags: the "
           "learner-side re-label (`wrappers.resolve_episode_end`, winprob => terminal) is the host's. NO "
           "terminal observation is produced (Lane D decision)"),
    Column("refused", "u8", ("N",), "out", "0",
           "1 iff a battle was QUARANTINED this op (its input log banked): the one in progress (done = 1, "
           "terminated = truncated = 0) or the START of the next (the env is PARKED: need = 0 0, and the next "
           "STEP starts it from the then-staged ep_team / ep_seed)"),
    Column("episode", "u32", ("N",), "out", "0",
           "the env's 0-based episode ordinal of the battle obs / need describe (it moves on every start)"),
    Column("dec_n", "u32", ("N", "SIDES"), "out", "0",
           "the index of the side's open decision among its decisions in this battle — sim_bridge's __OBS__ "
           "`n`, the alignment key; valid iff need = 1"),
    Column("turn", "u32", ("N",), "out", "0", "the battle's turn after the op"),
    Column("opp_route", "u32", ("N",), "out", "E",
           "the route (`ep_opp` as consumed at the start) of the episode obs / need describe — correct across "
           "auto-resets; after a refused start (PARKED) it names the refused episode's route"),
    Column("opp_slot", "i32", ("N",), "out", "E",
           "the T2 slot answering p2 in that episode (the route's `slot`), -1 when its route is not a policy — "
           "the host groups p2's rows by it"),
    Column("counters", "u64", ("NCOUNTERS",), "out", "0",
           "pool counters, indexed by `protocol.COUNTERS` (the `*_AFTER_FREEZE` ones must stay 0)"),
) + tuple(
    Column(r.key, r.dtype, ("N", "SIDES") + r.shape, "out", "C",
           f"label `{r.key}` (family `{r.family}`; written iff need = 1 and the family is declared): {r.derivation}")
    for r in LI.LABELS if r.rust == "core"
)

#: ``core`` family -> its label columns, in table order (``labels::FAMILIES`` in Rust).
LABEL_FAMILIES: Dict[str, Tuple[str, ...]] = {
    fam: tuple(r.key for r in rows) for fam, rows in LI.families("core").items()}
#: Families the core never writes, with why (the spec refuses each by name).
LABEL_NOT_CORE: Dict[str, str] = {
    fam: rows[0].rust for fam, rows in LI.families().items() if rows[0].rust != "core"}

OUT = src_path("rust_env", "src", "core", "columns.rs")


def _check_table() -> None:
    names = [c.name for c in COLUMNS]
    if len(set(names)) != len(names):
        raise AssertionError(f"duplicate column names: {names}")
    for c in COLUMNS:
        if c.dtype not in _DTYPES:
            raise AssertionError(f"{c.name}: unknown dtype {c.dtype}")
        if c.direction not in ("in", "out"):
            raise AssertionError(f"{c.name}: direction must be in / out")
        for d in c.dims:
            if isinstance(d, str) and d not in FIXED_DIMS and d not in ("N", "OBS_DIM"):
                raise AssertionError(f"{c.name}: unknown symbolic dim {d}")
        if "N" in c.dims[1:]:
            raise AssertionError(f"{c.name}: N may only be the FIRST dim")


_check_table()


def shapes(n: int, obs_dim: int) -> Dict[str, Tuple[int, ...]]:
    """Every column's concrete shape for a pool of ``n`` envs and the core's ``obs_dim``."""
    env = dict(FIXED_DIMS, N=n, OBS_DIM=obs_dim)
    return {c.name: tuple(d if isinstance(d, int) else env[d] for d in c.dims) for c in COLUMNS}


def nbytes(n: int, obs_dim: int) -> Dict[str, int]:
    out = {}
    for c in COLUMNS:
        k = _DTYPES[c.dtype][2]
        for d in shapes(n, obs_dim)[c.name]:
            k *= d
        out[c.name] = k
    return out


def numpy_dtype(c: Column) -> str:
    return _DTYPES[c.dtype][1]


def allocate(n: int, obs_dim: int) -> Mapping[str, "object"]:
    """Zeroed, C-contiguous NumPy arrays for every column (``action`` pre-filled with -1)."""
    import numpy as np

    sh = shapes(n, obs_dim)
    cols = {c.name: np.zeros(sh[c.name], dtype=numpy_dtype(c)) for c in COLUMNS}
    cols["action"][...] = -1
    return cols


def schema_text() -> str:
    """The canonical text of the contract — hashed into ``SCHEMA_ID`` on both sides."""
    lines = [f"col {c.name} {c.dtype} {'x'.join(map(str, c.dims))} {c.direction}" for c in COLUMNS]
    lines += [f"op {o.name} {o.code}" for o in P.OPS]
    lines += [f"status {s.name} {s.code}" for s in P.STATUSES]
    lines += [f"counter {i} {k.name}" for i, k in enumerate(P.COUNTERS)]
    lines += [f"dim {k} {v}" for k, v in sorted(FIXED_DIMS.items())]
    lines += [f"spec {k}" for k in P.SPEC_KEYS]
    lines += [f"labelfamily {f} {' '.join(ks)}" for f, ks in LABEL_FAMILIES.items()]
    lines += [f"labelnotcore {f} {why}" for f, why in LABEL_NOT_CORE.items()]
    return "\n".join(lines) + "\n"


def schema_id() -> str:
    """FNV-1a-64 of ``schema_text()`` (the same function the stamp uses; no hashlib needed)."""
    h = 0xCBF29CE484222325
    for b in schema_text().encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


# ------------------------------------------------------------------ the Rust rendering


def _rs_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rs_dim(d: Dim) -> str:
    return str(d) if isinstance(d, int) else d


def render() -> str:
    out: List[str] = []
    w = out.append
    w("// @generated by `python -m utils.rust_env.columns --write` — DO NOT EDIT.")
    w("// Source of truth: `src/utils/rust_env/columns.py` (the columns) + `protocol.py` (ops, statuses,")
    w("// counters). Pinned by `src/utils/rust_env/columns_test.py` (routine).")
    w("#![allow(dead_code)]")
    w("")
    w("/// FNV-1a-64 of `columns.schema_text()`; a front end compares it with its own table at load.")
    w(f"pub const SCHEMA_ID: &str = {_rs_str(schema_id())};")
    w("")
    w("// ---- dims")
    for k, v in FIXED_DIMS.items():
        w(f"pub const {k}: usize = {v};")
    w("/// The encoder's row length — the port's own constant, never a literal here.")
    w("pub const OBS_DIM: usize = pokesim::encoder::OBS_DIM;")
    w("")
    w("/// The startup declaration's keys (`protocol.SPEC_KEYS`): every one required, no other accepted.")
    w(f"pub const SPEC_KEYS: [&str; {len(P.SPEC_KEYS)}] = [{', '.join(_rs_str(k) for k in P.SPEC_KEYS)}];")
    w("")
    w("// ---- ops (`core::dispatch`'s first argument)")
    w("pub mod op {")
    for o in P.OPS:
        w(f"    /// {o.doc} (lane {o.owner})")
        w(f"    pub const {o.name}: u8 = b'{chr(o.code)}';" if 32 < o.code < 127 and chr(o.code) not in "'\\" else f"    pub const {o.name}: u8 = {o.code};")
    w("    /// Every op, for the front ends' tables.")
    w(f"    pub const ALL: [(u8, &str); {len(P.OPS)}] = [{', '.join(f'({o.name}, {_rs_str(o.name)})' for o in P.OPS)}];")
    w("}")
    w("")
    w("// ---- statuses (`core::dispatch`'s result code)")
    w("pub mod status {")
    for s in P.STATUSES:
        w(f"    /// {s.doc}" + (f" — Python `{s.exc}`" if s.exc else ""))
        w(f"    pub const {s.name}: i32 = {s.code};")
    w("}")
    w("")
    w("// ---- pool counters (indices into the `counters` column)")
    w("pub mod counter {")
    for i, c in enumerate(P.COUNTERS):
        w(f"    /// {c.doc}")
        w(f"    pub const {c.name}: usize = {i};")
    w(f"    pub const NAMES: [&str; {len(P.COUNTERS)}] = [{', '.join(_rs_str(c.name) for c in P.COUNTERS)}];")
    af = [c.name for c in P.COUNTERS if c.after_freeze]
    w("    /// The DECLARED-LIFECYCLE counters: each must stay 0 for the process's life.")
    w(f"    pub const AFTER_FREEZE: [usize; {len(af)}] = [{', '.join(af)}];")
    w("}")
    w("")
    w("// ---- the label families (Lane C; `label_inventory.py`)")
    w("pub mod labels {")
    w("    /// Every family the CORE writes: (name, its columns' indices).")
    fams = list(LABEL_FAMILIES.items())
    w(f"    pub const FAMILIES: [(&str, &[usize]); {len(fams)}] = [")
    for f, ks in fams:
        idx = ", ".join(f"super::col::{k.upper()}" for k in ks)
        w(f"        ({_rs_str(f)}, &[{idx}]),")
    w("    ];")
    w("    /// Every other family, with where it comes from instead (`host_const`, `host_episode`).")
    nc = list(LABEL_NOT_CORE.items())
    w(f"    pub const NOT_CORE: [(&str, &str); {len(nc)}] = [{', '.join(f'({_rs_str(f)}, {_rs_str(k)})' for f, k in nc)}];")
    w("}")
    w("")
    w("// ---- the columns")
    w("#[derive(Clone, Copy, Debug, PartialEq, Eq)]")
    w("pub enum Dtype { F32, U8, I32, U32, U64, I64 }")
    w("")
    w("impl Dtype {")
    w("    pub const fn size(self) -> usize {")
    w("        match self { Dtype::F32 | Dtype::I32 | Dtype::U32 => 4, Dtype::U8 => 1, Dtype::U64 | Dtype::I64 => 8 }")
    w("    }")
    w("}")
    w("")
    w("#[derive(Clone, Copy, Debug, PartialEq, Eq)]")
    w("pub enum Dir { In, Out }")
    w("")
    w("/// One row of the table.")
    w("#[derive(Clone, Copy, Debug)]")
    w("pub struct ColSpec {")
    w("    pub name: &'static str,")
    w("    pub dtype: Dtype,")
    w("    pub dir: Dir,")
    w("    /// Its first dim is N (env i owns row i).")
    w("    pub per_env: bool,")
    w("    /// Elements per env row (per_env) or in the whole column (a pool column).")
    w("    pub row_elems: usize,")
    w("    pub owner: &'static str,")
    w("}")
    w("")
    w("/// Column indices, in table order.")
    w("pub mod col {")
    for i, c in enumerate(COLUMNS):
        w(f"    pub const {c.name.upper()}: usize = {i};")
    w("}")
    w("")
    w(f"pub const N_COLUMNS: usize = {len(COLUMNS)};")
    w("")
    w("pub const COLUMNS: [ColSpec; N_COLUMNS] = [")
    for c in COLUMNS:
        rest = c.dims[1:] if c.per_env else c.dims
        elems = " * ".join(_rs_dim(d) for d in rest) or "1"
        w(f"    // {c.doc}")
        w(f"    ColSpec {{ name: {_rs_str(c.name)}, dtype: Dtype::{c.dtype.upper()}, "
          f"dir: Dir::{c.direction.capitalize()}, per_env: {str(c.per_env).lower()}, "
          f"row_elems: {elems}, owner: {_rs_str(c.owner)} }},")
    w("];")
    w("")
    w("/// Bytes of every column for a pool of `n` envs.")
    w("pub const fn col_bytes(n: usize) -> [usize; N_COLUMNS] {")
    w("    let mut out = [0usize; N_COLUMNS];")
    w("    let mut i = 0;")
    w("    while i < N_COLUMNS {")
    w("        let c = COLUMNS[i];")
    w("        out[i] = c.row_elems * c.dtype.size() * if c.per_env { n } else { 1 };")
    w("        i += 1;")
    w("    }")
    w("    out")
    w("}")
    w("")
    w("/// The caller's column addresses, in table order (plain integers: `Send`, `Copy`, comparable —")
    w("/// the frozen binding is one of these). Only `core::pool` turns them back into slices.")
    w("#[derive(Clone, Copy, Debug, PartialEq, Eq)]")
    w("pub struct ColAddrs(pub [usize; N_COLUMNS]);")
    w("")
    w("/// One env's view of every PER-ENV column (inputs shared, outputs exclusive).")
    w("pub struct EnvCols<'a> {")
    for c in COLUMNS:
        if not c.per_env:
            continue
        ty = _DTYPES[c.dtype][0]
        w(f"    pub {c.name}: &'a {'mut ' if c.direction == 'out' else ''}[{ty}],")
    w("}")
    w("")
    w("impl ColAddrs {")
    w("    /// Env `i`'s rows.")
    w("    ///")
    w("    /// # Safety")
    w("    /// Every address holds its column for a pool of `n > i` envs, alive, aligned and unaliased by")
    w("    /// anything but other envs' rows for the duration of the borrow; distinct `i` give disjoint")
    w("    /// output slices.")
    w("    pub unsafe fn env<'a>(&self, i: usize) -> EnvCols<'a> {")
    w("        EnvCols {")
    for idx, c in enumerate(COLUMNS):
        if not c.per_env:
            continue
        ty = _DTYPES[c.dtype][0]
        k = f"COLUMNS[{idx}].row_elems"
        fn = "from_raw_parts_mut" if c.direction == "out" else "from_raw_parts"
        ptr = f"(self.0[{idx}] as *{'mut' if c.direction == 'out' else 'const'} {ty}).add(i * {k})"
        w(f"            {c.name}: std::slice::{fn}({ptr}, {k}),")
    w("        }")
    w("    }")
    for idx, c in enumerate(COLUMNS):
        if c.per_env:
            continue
        ty = _DTYPES[c.dtype][0]
        w("")
        w(f"    /// The pool column `{c.name}`.")
        w("    ///")
        w("    /// # Safety")
        w("    /// As [`ColAddrs::env`], and no env borrow reaches this column.")
        if c.direction == "out":
            w(f"    pub unsafe fn {c.name}<'a>(&self) -> &'a mut [{ty}] {{")
            w(f"        std::slice::from_raw_parts_mut(self.0[{idx}] as *mut {ty}, COLUMNS[{idx}].row_elems)")
        else:
            w(f"    pub unsafe fn {c.name}<'a>(&self) -> &'a [{ty}] {{")
            w(f"        std::slice::from_raw_parts(self.0[{idx}] as *const {ty}, COLUMNS[{idx}].row_elems)")
        w("    }")
    w("}")
    w("")
    return "\n".join(out)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write src/rust_env/src/core/columns.rs")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file is stale")
    a = ap.parse_args(argv)
    text = render()
    if a.write:
        OUT.write_text(text)
        print(f"wrote {OUT} (schema {schema_id()})")
        return 0
    if a.check:
        ok = OUT.exists() and OUT.read_text() == text
        print("columns.rs is current" if ok else "columns.rs is STALE — run with --write")
        return 0 if ok else 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
