"""Attribute the A->B Rust CPU delta to components from two `gdb_sampler.py` sample files.

A component's INCLUSIVE share (fraction of samples with any frame containing its key) times the
arm's measured CPU per replay (median of `replay_rows.jsonl`) gives milliseconds per replay;
the B-minus-A column is where the delta lives. Binomial SE is printed so a small difference is
not read as a finding.

    python profile_attrib.py samples_A.jsonl samples_B.jsonl replay_rows.jsonl
"""

from __future__ import annotations

import json
import math
import statistics
import sys

COMPONENTS = [
    ("whole replay", "sim_bridge::main"),
    ("request build (build_request)", "build_request"),
    ("battle engine (FullBattleDriver::feed)", "FullBattleDriver::feed"),
    ("per-side log fold (emit_log_batch_chunk)", "emit_log_batch_chunk"),
    ("per-side fold (derive_side)", "derive_side"),
    ("chunk push (push_chunk_lines)", "push_chunk"),
    ("  one-sided view (SideObservation::observe)", "SideObservation::observe"),
    ("typed Line: ProtocolBuilder::emit (build+render)", "ProtocolBuilder::emit"),
    ("typed Line: core_events::line (any)", "core_events::line"),
    ("typed Line: Line::render", "Line::render"),
    ("retro-edit re-parse (retro_edit / Line::parse)", "retro_edit"),
    ("stdout frame write (flush_new_chunks)", "flush_new_chunks"),
]


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main(pa, pb, rows_path):
    rows = load(rows_path)
    cpu = {a: statistics.median(r["replay_cpu_s"] for r in rows if r["arm"] == a) for a in "AB"}
    samples = {"A": load(pa), "B": load(pb)}
    print(f"median CPU/replay: A {cpu['A'] * 1000:.1f} ms, B {cpu['B'] * 1000:.1f} ms "
          f"(delta {1000 * (cpu['B'] - cpu['A']):+.1f} ms); samples A {len(samples['A'])}, "
          f"B {len(samples['B'])}")
    print(f"{'component':<50}{'A share':>9}{'B share':>9}{'A ms':>8}{'B ms':>8}{'B-A ms':>9}{'±1SE':>7}")
    for label, key in COMPONENTS:
        out = {}
        for a in "AB":
            s = samples[a]
            p = sum(any(key in n for n in st) for st in s) / len(s)
            se = math.sqrt(p * (1 - p) / len(s)) * cpu[a] * 1000
            out[a] = (p, p * cpu[a] * 1000, se)
        d = out["B"][1] - out["A"][1]
        se = math.hypot(out["A"][2], out["B"][2])
        print(f"{label:<50}{out['A'][0]:>9.1%}{out['B'][0]:>9.1%}{out['A'][1]:>8.1f}"
              f"{out['B'][1]:>8.1f}{d:>+9.1f}{se:>7.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
