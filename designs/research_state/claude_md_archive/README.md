# `CLAUDE.md` archive — narrative lifted out of the rules files

Created **2026-09-07**, when the root `CLAUDE.md` was cut from 2,644 lines (206 KB, ~57k tokens
loaded into EVERY session) to ~1,650 (119 KB, ~33k tokens).

**The rule applied** (from `designs/research_state/claude_md_census_2026-09-06.md` §0):

> A `CLAUDE.md` earns a line only if an agent that has NOT read it would do the work WRONG.

Four classes stayed in `CLAUDE.md` — a **RULE**, a **COMMAND/GATE**, a **MAP**, and a **HAZARD**
(with just enough evidence to be believed). Everything else — how we found out, what we measured,
which arm won, what the number was in August — is **NARRATIVE**, and lives here, where it is read
**on purpose** rather than **on every turn**.

| file | lifted from |
|---|---|
| `bridge_transport_history.md` | § In-process bridge transport (225 → 44 lines) |
| `checkargs_incidents.md` | § Will this command still launch? (188 → 57) |
| `model_versioning_history.md` | § Model Versioning (104 → 50) |
| `benchmark_measurements.md` | § Benchmarks (85 → 46) |
| `contention_measurements.md` | § Running beside a live training run (72 → 39) |
| `pythonpath_archaeology.md` | § The `export PYTHONPATH` prefix (67 → 40) |
| `file_size_paydown.md` | § The FILE-SIZE ratchet (50 → 30) |

**Added 2026-09-07 (second pass), from `designs/CLAUDE.md`** — 95 KB → 20 KB, the same rule applied
to the leaf every session that touches `designs/` loads:

| file | lifted from |
|---|---|
| `designs_version_map_config_version_ladder.md` | the **Code on main** state-table cell (21.7 KB in one cell) |
| `designs_version_map_ai_v9_stages_and_forward_designs.md` | the **ai_v9** state-table cell (20.8 KB in one cell) |
| `designs_version_map_state_table_2026-09-07.md` | the remaining long cells — the active run and its generation history, and the ai_v12 / ai_v10 / ai_v11 chapter cells |

**Added 2026-09-08 (third pass), from `src/agents/training/CLAUDE.md`** — 3,183 → 739 lines, the
same rule applied to the leaf every session that touches `src/agents/training/` loads:

| file | lifted from |
|---|---|
| `training_leaf_faint_attribution_history.md` | § Faint attribution in the trace (49 → 13 lines) |
| `training_leaf_deleted_subsystems_history.md` | § Latent-belief loss — DELETED (v75) and § V_pub — DELETED (v88) (35 → 8) |

**Nothing here is current.** Every guard these incidents produced is live and described in
`CLAUDE.md`; this is the evidence behind them. Do not re-derive a plan from anything here without
checking the code first — an entry that outlives its own fix misleads every reader after it.

Related, from the same pass: `designs/training/` (the training leaf's topics),
`designs/rust_sim/port_build_log.md` (the port's closed coverage rounds), and
`designs/research_state/memory_archive/` (retired agent memories).
