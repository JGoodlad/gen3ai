"""Classify a `perf report --no-children --sort dso,sym -q` text dump of the env core into WHAT the time is spent on
(self time only: the release build's DWARF unwinding did not resolve call chains, so a generic symbol — malloc, a
`String` builder, `core::fmt` — is counted as its own class, not charged to its caller).

    perf record -F 499 -o perf.data python env_core_rate.py --threads 1 --blocks 3 --steps 300
    perf report -i perf.data --no-children --sort dso,sym -q > syms.txt
    python3 classify_profile.py syms.txt

Classes (by the symbol's module path; `drop_in_place<pokesim::X::…>` goes to X's class):

* ``sim``        — the battle simulation itself: `pokesim::{turn,state,engine,damage,battle,stats,prng,team,event,
  moves,abilities,items,conditions,actions,choice}`;
* ``emit``       — protocol emission and the bridge transport: `pokesim::{protocol,bridge,json}`;
* ``parse_fold`` — re-reading the emitted protocol into each side's view: `pokesim::{core_events,present,trackers,
  version,dex}`;
* ``encode``     — the observation encoder: `pokesim::encoder`;
* ``env_glue``   — `pokesim_env::` (the env core's pool, columns, labels);
* ``alloc``      — the allocator and memory moves: libc `malloc` / `free` / `realloc` / `mem*`, Rust `alloc::` /
  `__rust_*` / `RawVec` growth;
* ``strings_fmt``— `String` / `str` / `core::fmt` / iterator glue with no module of ours in the symbol;
* ``other``      — everything else (the kernel, Python, numpy, ctypes).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

SIM = {"turn", "state", "engine", "damage", "battle", "stats", "prng", "team", "event", "moves", "abilities", "items",
       "conditions", "actions", "choice", "sim", "field", "side", "pokemon"}
EMIT = {"protocol", "bridge", "json"}
PARSE = {"core_events", "present", "trackers", "version", "dex"}


def classify(dso: str, sym: str) -> str:
    m = re.search(r"pokesim::([a-z_]+)", sym)
    if m:
        mod = m.group(1)
        if mod in SIM:
            return "sim"
        if mod in EMIT:
            return "emit"
        if mod in PARSE:
            return "parse_fold"
        if mod == "encoder":
            return "encode"
        return "sim_other_module:" + mod
    if "pokesim_env::" in sym:
        return "env_glue"
    if dso.startswith("libc") and re.search(r"malloc|free|realloc|mem(move|cpy|set|cmp)|_int_|calloc", sym):
        return "alloc"
    if re.search(r"alloc::raw_vec|__rust_|__rdl_|__rustc::|RawVec|finish_grow|do_reserve", sym):
        return "alloc"
    if re.search(r"alloc::string|core::fmt|alloc::fmt|core::str|String|memchr|core::iter|btree|core::slice", sym):
        return "strings_fmt"
    if dso.startswith("libpokesim_env"):
        return "rust_other"
    return "other"


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    tot = defaultdict(float)
    with open(path) as fh:
        lines = fh.readlines()
    for ln in lines:
        m = re.match(r"\s*([\d.]+)%\s+(\S+)\s+\[[.k]\]\s+(.*)$", ln)
        if not m:
            continue
        pct, dso, sym = float(m.group(1)), m.group(2), m.group(3).strip()
        tot[classify(dso, sym)] += pct
    s = sum(tot.values())
    print(f"| class | self time | (of {s:.1f} % classified) |\n|---|---|---|")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"| {k} | {v:.1f} % | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
