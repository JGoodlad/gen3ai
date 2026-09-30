"""THE M5 GATE HARNESS (M5 Lane J, ``designs/endstate/program_rust_core.md`` §2 M5, order constraint 4).

The single place the M5 milestone is judged — not a pile of per-lane tests:

* ``lanes`` / ``gates`` — every lane's parity gate as ONE declared row (delegated to the lane's own
  tests, never re-implemented) and one verdict table: PASS / FAIL / NOT BUILT / INCONCLUSIVE /
  NOT RUN. A lane that is not built reads NOT BUILT, never a pass; a lane the program doc marks
  BUILT with no row fails a routine test.
* ``slice_n`` — SLICE N at the ENV level: N envs on T worker threads in the Rust env core, through
  BOTH front ends, against the production-surface ``Gen3Env`` replaying every episode — obs, mask,
  every built label key, reward, ``terminated`` / ``truncated`` byte-equal (Lane C's and Lane D's
  record-in-Rust / replay-in-Python machinery, joined and run at N).
* ``depth3`` — the depth-3 successor slice (search's default ``max_depth``), Lane I's lockstep
  harness at the MILESTONE sources.
* ``throughput`` (+ ``hooks``) — the THROUGHPUT A/B scaffold: today's Python rollout path vs the
  Rust env at matched N, with typed hooks for Lane G's collector and Lane E's opponent routing.

    export PYTHONPATH=$PYTHONPATH:src
    python -m main.rust_core_m5 gates [--tier commit|milestone] [--gpu]
    python -m main.rust_core_m5 slice-n --tier commit|milestone
    python -m main.rust_core_m5 depth3 --tier commit|milestone
    python -m main.rust_core_m5 throughput --n-envs 48 ...
    python -m main.rust_core_m5 verdict            # the M5 gate, composed from the above

Progress and resume point: ``designs/research_state/measurements/m5_laneJ/PROGRESS.md``.
"""
