"""``main.ops`` — the OPERATIONAL instruments for a run that is CURRENTLY RUNNING.

Promoted 2026-09-07 out of the Training Run session's session-scoped temporary directory, where
they lived as loose scripts with the arm name, the model directory and the interpreter written
in as literals, and where they vanished with the session that wrote them
(``designs/ops/TECH_DEBT_BACKLOG.md`` P1).

**Why a subpackage and not ``src/main/`` beside the offline CLIs.** ``src/main/*.py`` is a
curated list — ``elo``, ``dose``, ``lineage``, ``baselines``, ``critic_gate``,
``untaught_meter``, ``exploitability``, ``scaffolding_gauge``, ``capacity``, ``checkargs``,
``sidecar_audit``, ``tb_curate`` — and the root ``CLAUDE.md`` names it as *the offline meters*:
tools that read a FINISHED run's artifacts and write nothing. These are a different tier. They
read a LIVE run's TensorBoard events, its launcher child log, its checkpoint mtimes and its
snapshot ladder while all of those are still being appended to, and several of them REFUSE
rather than report when a precondition of the live read is unmet (the ladder not caught up with
the snapshots on disk, the restart boundary not yet in the data, the child log's ring buffer
trimmed). Mixing them into that list would dilute the one thing its name promises.

**Two rules every module here keeps, because both were paid for.**

1. **ABSENCE IS NOT A ZERO.** A missing scalar, an empty window, a trimmed ring buffer and an
   unfinished reference window each print what is missing and refuse the verdict. The tools exist
   because a log-rendering read produced two confident wrong readings (an episode length
   "falling" that was oscillating, and a metric read under the wrong group prefix).
2. **A SINGLE SAMPLE IS NEVER A VERDICT.** Every statistic is a median or a series over a
   REGISTERED window, and a last single reading is printed labelled as such or not at all.

Each module is a ``python -m main.ops.<name>`` entry point and guards its ``main()`` behind
``if __name__ == "__main__"`` — importing a module must never run a read
(``src/main/entry_point_guard_test.py`` records why).
"""
