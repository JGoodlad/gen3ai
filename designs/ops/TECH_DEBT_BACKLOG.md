# TECH-DEBT BACKLOG

The one list of tech-debt work. Research and measurement work is NOT tracked here; it lives in
`designs/research_state/UNDERSTANDING.md` §6 and the ledger.

**Two rules (owner, 2026-09-06).** (1) During the week nothing here is dispatched on the
orchestrator's own initiative: the orchestrator ADDS rows; the owner moves a row to ACCEPTED, and an
ACCEPTED row is dispatched under [`ORCHESTRATOR_SOP.md`](ORCHESTRATOR_SOP.md) §2. (2) **Whatever
quota remains before the weekly reset is burned down on this list as a standing housekeeping
procedure** — ACCEPTED rows first, top to bottom; if ACCEPTED is empty and quota remains, the
orchestrator notifies the owner with the top PROPOSED rows and, absent a reply in 15 minutes, starts
the highest one. Read the reset time with `python3 ~/.claude/skills/usage-limits/check_usage.py`
(the weekly window resets **Tuesdays ~14:00 PDT**); the burn-down begins when the remaining window
is under ~24 h and the live arm needs nothing.

**Ordering:** by CRITICALITY tier, then by size within a tier (small first, so a partial burn-down
still closes rows). Tiers: **P0** — a GIGO class: could make a measurement or a test lie ·
**P1** — blocks or slows correct work (a gate near its bound, a tool that will vanish, a doc every
session loads) · **P2** — hygiene and cost. Each row carries its size and its acceptance test; a row
is DONE when that test is in the routine suite or its artifact is on main, and moves to §3 with the
commit.

---

## 1. ACCEPTED

| tier | item | why | size | done when |
|---|---|---|---|---|
| P0 | **Mode-flag doc gate.** A test that every MODE-flag value `designs/ARCHITECTURE.md` states in prose (`belief_grad_mode`, `opp_intent_grad_mode`, `critic`, `hp_belief_mode`, each head's on/off coefficient, `hand_shaping`, `terminal_indicator`) equals the value in `designs/production_config.json`, and that every key the mirror marks INERT is called INERT where the doc names it. | "What is the baseline?" — the architecture doc said belief heads run `label_only` while its own flag table, the mirror and the live arm say `shaping` (hand-fixed, `22e757db`). A doc read as fact must not drift from the mirror. | S (~1 day) | unmarked in the routine suite; fails on a planted contradiction; passes on main |
| P0 | **Anti-vacuity audit of test stubs.** For every `monkeypatch.setattr` / `patch(...)` target under `src/**/*_test.py`, assert the patched symbol is referenced by the code under test, so a stub that stubs nothing FAILS; then fix what it finds. | The `ppo.py` decomposition found four sites in one file family where the stub stubbed nothing and a byte-identity test compared two identical arms and PASSED (`ccd08003`). Green for the wrong reason is the GIGO class the owner named. | M (~2 days) | the audit runs in the routine suite; every finding fixed or allow-listed with a reason |

## 2. PROPOSED — the orchestrator adds rows; only the owner moves one to §1 (or the week-end burn-down reaches it)

| tier | item | why | size |
|---|---|---|---|
| P1 | Move the landing script into the repo (`scripts/land.sh`, the gates-then-push procedure `ORCHESTRATOR_SOP.md` §3 describes) | it lives in a previous session's temporary directory and disappears with it | XS (5 min) |
| P1 | Root `CLAUDE.md` restructure to ≤ ~400 lines (`designs/research_state/claude_md_census_2026-09-06.md`; the rule "a line earns its place only if an agent that has not read it would do the work wrong" is owner-accepted) | ~45k tokens loaded into every session; ~32% narrative + 6% duplicate; the leaves take the detail | L |
| P1 | Era-boundary flip (`designs/research_state/era_boundary_deprecation_2026-09-06.md`; `MIGRATION_FLOOR` 109) — **licensed by the owner once the SPARSE arm proves out; not before** | frees 6 flag families / 11 flags; retires the shaped-critic scaffolding; the untaught meter's opponent must be re-measured on the same checkpoints first | M; gated on the arm |
| P2 | TensorBoard fork-prefix backfill (`python -m main.tb_inherit --backfill --all`, 137 forks, 105 with a DERIVED parent) | a reviewed operation, not a sweep: at least one derived parent is provably wrong | S, manual review |
| P2 | `models/` retention apply (`archive_grooming_dryrun.py --policy tiered --apply`, 87.9 GB) | dry-run only so far; disk at 43% so not urgent; **owner-run** | owner |

## 3. DONE

| item | commit |
|---|---|
| `instrumented_ppo/ppo.py` decomposed around the fold sequence (1998 → 1331) + the contested-mask guard | `ccd08003` |
| Shaped-era timeout comparator — **CLOSED, no comparator** (owner, 2026-09-06); the stripped arm's curve is the only one and the kill bar is registered against it | — |
