# Archived Claude memories

Memories retired from the agent memory index on **2026-09-07**, preserved verbatim.

They were removed because their subject is a CLOSED era, a landed fix, or a fact the
repository now records directly (a `CLAUDE.md`, `designs/ARCHITECTURE.md`, or the ledger) —
per the memory rule that a memory must not duplicate what the repo already says.

They are kept here rather than deleted because each records what was believed at the time,
which the ledger's append-only history depends on. **Nothing here is current.** Read the
ledger and `designs/research_state/UNDERSTANDING.md` for what is true now.

| file | retired because |
|---|---|
| `project_current_run.md` | GEN-14/17 era run notes, self-labelled historical; superseded by the win-prob critic era |
| `project_gen13_launch_plan.md` | gen-13 launched and closed |
| `project_gen14_preregistered.md` | gen-14 pre-registration, since executed |
| `project_loss_analysis_run20260531.md` + `_v2.md` | May-2026 forensic passes; superseded by `falsify` / `calibration` |
| `project_stall_recovery_analysis.md` | superseded by the anti-stall fix and the contention work |
| `project_plateau_diagnosis_2026_06_09.md` | superseded by `project_plateau_research_2026_06_25` |
| `project_loss_triage_tool.md` | the `triage` tool now documented in `src/main/prober/CLAUDE.md` |
| `project_eval_item_workstealing.md` | shipped; documented in `src/agents/training/CLAUDE.md` |
| `project_stable_opponents_design.md` | shipped; documented in `src/agents/training/CLAUDE.md` |
| `project_env_worker_memory_leak.md` | fixed |
| `project_bridge_unique_battle_tags.md` | fixed |
| `project_showdown_server_memory_growth.md` | moot — the in-process bridge is the default, no server |
| `project_human_agreement_probe.md` | ai_v6 behaviour-cloning track, not a live line |
| `project_next_run_plan.md` | pointed at `designs/ai_v8/next_run_plan.md`; that era is closed |
| `project_opponent_system_parity.md` | audit whose debts now live in `designs/ops/TECH_DEBT_BACKLOG.md` |
| `project_positional_grind_decomposition.md` | the live rule survives as `feedback_no_circular_unwinnable_claims` |
| `feedback_progress_cron_notifications.md` | fully covered by `feedback_long_run_sop` (mechanism) + `feedback_notification_standing_order` (policy) |
