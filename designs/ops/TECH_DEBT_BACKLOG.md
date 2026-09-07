# TECH-DEBT BACKLOG

The one list of tech-debt work. **Nothing here is dispatched automatically** (owner, 2026-09-06):
an item moves to ACCEPTED on the owner's word and is dispatched by the orchestrator under
[`ORCHESTRATOR_SOP.md`](ORCHESTRATOR_SOP.md) §2. Research and measurement work is NOT tracked here;
it lives in `designs/research_state/UNDERSTANDING.md` §6 and the ledger.

Each row: what, why (the smell or the incident it closes), size, and the acceptance test. An item is
DONE when its test is in the routine suite or its artifact is on main; the row then moves to §3 with
the commit.

---

## 1. ACCEPTED — dispatch in this order

| # | item | why | size | done when |
|---|---|---|---|---|
| 1 | **Mode-flag doc gate.** A test that every MODE-flag value `designs/ARCHITECTURE.md` states in prose (`belief_grad_mode`, `opp_intent_grad_mode`, `critic`, `hp_belief_mode`, the `*_coef` a head is on/off by, `hand_shaping`, `terminal_indicator`) equals the value in `designs/production_config.json`, and that every key the mirror marks INERT is called INERT where the doc names it. | "What is the baseline?" — the architecture doc said belief heads run `label_only` while its own flag table, the mirror and the live arm say `shaping` (fixed by hand, `22e757db`). A doc read as fact cannot be allowed to drift from the mirror. | small (~1 day) | the gate is unmarked in the routine suite, fails on a planted contradiction, and passes on main |
| 2 | **Anti-vacuity audit of test stubs.** For every `monkeypatch.setattr` / `patch(...)` target in `src/**/*_test.py`, assert the patched symbol is actually referenced by the code under test (import-graph or `getsource` check), so a stub that stubs nothing fails instead of passing vacuously. Then fix what it finds. | The `ppo.py` decomposition found four sites in one file family where the stub would have stubbed nothing and a byte-identity test would have compared two identical arms and PASSED (`ccd08003`). A green test for the wrong reason is the GIGO class the owner named. | medium (~2 days) | the audit runs in the routine suite; every finding fixed or allow-listed with a reason |

## 2. PROPOSED — not yet accepted (the orchestrator may add rows; only the owner moves one up)

| item | why | size |
|---|---|---|
| Move the landing script into the repo (`scripts/land.sh`) | it lives in a previous session's temporary directory and disappears with it | 5 min |
| Root `CLAUDE.md` restructure to ≤ ~400 lines (`designs/research_state/claude_md_census_2026-09-06.md`) | ~45k tokens loaded into every session; ~32% narrative + 6% duplicate | large |
| Era-boundary flip (`designs/research_state/era_boundary_deprecation_2026-09-06.md`; `MIGRATION_FLOOR` 109; licensed by the owner once the SPARSE arm proves out) | frees 6 flag families / 11 flags; retires the shaped-critic scaffolding | medium; gated on the arm |
| TensorBoard fork-prefix backfill (`python -m main.tb_inherit --backfill --all`, 137 forks, 105 with a DERIVED parent) | reviewed operation; at least one derived parent is provably wrong | small but manual review |
| `models/` retention apply (`archive_grooming_dryrun.py --policy tiered --apply`, 87.9 GB) | dry-run only so far; disk is at 43% so not urgent | owner-run |
| Timeout-rate comparator for the shaped era | **CLOSED by the owner (2026-09-06): no comparator** — the stripped arm's curve is the only one, and the kill bar is registered against it | — |

## 3. DONE

| item | commit |
|---|---|
| `instrumented_ppo/ppo.py` decomposed around the fold sequence (1998 → 1331) + the contested-mask guard | `ccd08003` |
