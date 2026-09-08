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
(the weekly window resets **Tuesdays ~14:00 PT**). **Target: the weekly quota is used up to 95% by
Tuesday ~08:00 PT** (owner, 2026-09-06), leaving ~5% spare for an issue — so the burn-down is PACED to
land there (it starts Monday, earlier if the backlog is large) and STOPS at 95%: burn-down agents run
under the usage gate at `--weekly-threshold 95`, never higher. The live arm's needs always come first;
the burn-down uses what the arm does not.

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

*(Empty as of 2026-09-07 — both accepted rows landed the same day: the mode-flag doc gate and
the anti-vacuity stub audit; see §3. Per the rule above, with ACCEPTED empty and quota
remaining, the orchestrator notifies the owner with the top PROPOSED rows and, absent a reply
in 15 minutes, starts the highest one.)*

## 2. PROPOSED — the orchestrator adds rows; only the owner moves one to §1 (or the week-end burn-down reaches it)

| tier | item | why | size |
|---|---|---|---|
| P0 | **A `slow`-marked test can ride RED invisibly — close the gap.** `tb_relevance_test::test_a_winprob_run_emits_no_noise_tag_and_every_live_tag` is failing on main (2/2 in isolation, 2026-09-07) and the ROUTINE GATE CANNOT SEE IT: it is `@pytest.mark.slow`, so `-m "not slow and not e2e"` deselects it. Nothing surfaces a red `slow` test between full-suite runs. Options: a scheduled full-suite run that reports, or a slow-tier last-known-status file the routine gate asserts against. | A gate that reads GREEN while a test is red is the GIGO class exactly — the same shape as the obs-golden linchpin riding main RED three times behind a marker. Unrecorded in both the ledger and this backlog until now. | S–M | a red `slow` test is surfaced without a 31-minute run; planted red fails it |
| P1 | **Six `win_prob/*` tags the win-prob era calls content-free are emitted anyway.** `acc_contested`, `brier_contested`, `brier_material`, `contested_frac`, `contested_label_mean`, `skill_vs_material` — all six are on the LIVE arm `ai_v12_02_winprob_critic` (419 scalar tags, verified 2026-09-07) and are what the P0 row's test fails on. Either suppress them under `--critic winprob` or reclassify them in `gen3_tb_relevance_v1` if they are in fact informative. | TB noise on the era's only live arm, and it is what keeps that test red; the classification came in with the test (`fc10fbb4`) and one side of it is wrong. | S | the test passes on main and the arm stops emitting them (or they are reclassified with a reason) |
| P1 | Promote the Training Run session's operational scripts into the repo (`scripts/ops/`): the OS watcher (`watch_arm2.sh`), the TensorBoard scalar reader (`tb_read.py` — reads a tag's last-N series/median from `<run>/tb` events, warns on a missing tag instead of defaulting to 0), the 10M read script (`read_10M.sh`), the restart-read script | all four live in a session-scoped temporary directory and vanish with the session; the TB reader already caught two log-rendering misreads (episode length "falling" that oscillates; a metric under the wrong group) and is the SOP's "validate at the source" tool | S (~2 h) |
| P1 | Era-boundary flip (`designs/research_state/era_boundary_deprecation_2026-09-06.md`; `MIGRATION_FLOOR` 109) — **licensed by the owner once the SPARSE arm proves out; not before** | frees 6 flag families / 11 flags; retires the shaped-critic scaffolding; the untaught meter's opponent must be re-measured on the same checkpoints first | M; gated on the arm |
| P1 | `designs/ARCHITECTURE.md` still says "a frozen-φ provenance fix is in flight" for the three PBRS keys that read `true` while INERT under the win-prob critic; either land the fix (record the RESOLVED composition beside the raw fields) or close the sentence | an always-current doc carrying an "in flight" claim of unknown status is a baseline smell | S |
| P1 | **129 git worktrees, 13 detached** — most are finished agents' or past runs' pinned checkouts; review and remove the dead ones (a run's pin is now a `/tmp/launcher-<sha>-*` tree, so the old `gen*-run-*` worktrees may be unreferenced — VERIFY against every live and resumable run before removing) | disk, submodule checkouts, and `git worktree list` is unreadable | S, reviewed |
| P1 | Lineage backfill: 105 legacy runs carry a parent DERIVED from `original_command`, and one is provably wrong (`ai_v8_01_zarch_film_0717` claims fresh while its curves start at 148M); `python -m main.lineage --backfill` reviewed run by run — a PREREQUISITE for the TensorBoard backfill row below | a wrong parent poisons every tool that walks ancestry | S, manual |
| P2 | Split the two giant leaves — `src/agents/training/CLAUDE.md` 7,503 lines and `src/rust_sim/CLAUDE.md` 8,436 lines (74% narrative, 45 "ROUND N" headings) — into sub-leaves by subsystem, narrative to `research_state/engineering_notes/` (census `claude_md_census_2026-09-06.md`) | ~400k tokens loaded when working in either area | L |
| P2 | `designs/CLAUDE.md` — 79 KB in 411 lines; two version-map cells are ~5,000 words each | densest doc in the tree; loaded when working in designs | S |
| P2 | The critic gate's G5 / G6 / G8 criteria print NOT RUNNABLE on every report (design §6 gaps) — research-INSTRUMENT debt: build the instruments or retire the criteria from the design | a gate with unrunnable criteria quietly becomes the runnable ones | M, research-gated |
| P2 | A full multi-cycle search-teacher run end-to-end on the rust bridge is not gated (every leg is, the composition is not) | the first such run will find the seam | M |
| P2 | Test-tier DURATIONS were last measured 2026-08-14 on an idle box; counts re-measured since but durations not | the routine-gate timing in CLAUDE.md is a month stale | XS, needs an idle box |
| P1 | **The eval-trace capture has NO DRAW bucket.** `meta.result` is WIN/LOSS only and filenames are `win_`/`loss_`; a 250-turn timeout is written as a LOSS (G7 survives because it keys on `meta.turns`), and a true tie that ends before 250 turns is written as a LOSS and is invisible. Add a `draw`/`tie` result + filename prefix, thread it through `trace_selection`'s quota accounting and `prober` filters, and make the summary refuse an unknown result | 2026-09-07 (c) probe: "0 draws in every eval trace" was what the instrument could express, not what happened — the absence-is-not-a-zero class, again. Until fixed, the traces can estimate a STALL rate but never a TIE rate; the tie half of any draw split must come from the training signal | S |
| P2 | `launcher_child.log` is a ~1024 KiB RING BUFFER that trims silently (by design, after the 982 MB repaint log). Keep a full ROTATING copy beside it (`launcher_child.full.log.N`, size-capped rotation, never unbounded), so a per-worker count survives a restart | 2026-09-06: the Training Run session counted compile lines across a restart and got a number that was unrecoverable once trimmed — the count had to be settled from source instead. A read that cannot be redone is a read that cannot be checked | S |
| P2 | The ledger is one 11,600-line append-only file; regex tooling now fails on it (`ugrep` complexity errors twice on 2026-09-06). A generated title index (date · title · line) beside it, never editing the file | findability of registrations | S |
| P2 | TensorBoard fork-prefix backfill (`python -m main.tb_inherit --backfill --all`, 137 forks, 105 with a DERIVED parent) | a reviewed operation, not a sweep: at least one derived parent is provably wrong | S, manual review |
| P2 | `models/` retention apply (`archive_grooming_dryrun.py --policy tiered --apply`, 87.9 GB) | dry-run only so far; disk at 43% so not urgent; **owner-run** | owner |

## 3. DONE

| item | commit |
|---|---|
| **`reward_manager.py` decomposed around the fold contract** (1,990 → **808**; the size gate's hard bound is 2,000 and the next edit anywhere in it would have tripped). Three new modules, each one responsibility: `reward_bias_terms.py` 533 (the `RewardBiasTerms` mixin — every `_compute_*` BIAS field, plus the CURRENT-BOARD accessors), `reward_config.py` 445 (`RewardClass` / `RewardConfig` / `RewardBreakdown` + `_REGISTRY` / `SWITCH_BIAS_DROP_FAMILY`), `reward_potentials.py` 357 (the `RewardPotentials` mixin — the Φ potentials, `_pbrs_step`, `_hand_pbrs_on`, the eight `_fold_*_pbrs`). **`process_turn_reward`'s fold SEQUENCE stays one straight line**, the `ccd08003` rule. Reward stream **byte-identical**: 2,802 decisions × 39 fields × 6 compositions, sha256 `9463dc24…` before and after. The three `_encode_incoming_block` patch targets were repointed at `reward_potentials`; stub-vacuity gate green, allowlist still EMPTY | `b0b3a253` (2026-09-07) |
| **Mode-flag doc gate** — `src/mode_flag_doc_gate_test.py`: every MODE-flag value `ARCHITECTURE.md`'s PROSE states equals `production_config.json` (read via the baselines registry accessor), the (pattern → key) table is DECLARED so a rename FAILS, and a planted contradiction is proven to fire. Found and fixed 4 drifts: `out_dim` 660→**138** with `op_drop_renders` (not `damage_matrices_outgoing`) as the real gate, 15→**17** edge families, and `move_belief_mode` / `hp_belief_mode` stated nowhere in prose | `a0317cdf` (2026-09-07) |
| Landing script into the repo as `scripts/land.sh` — paths derived from `git rev-parse --git-common-dir`, all `src/*_gate_test.py` statics, `--help`, re-execs out of the worktree it removes; `ORCHESTRATOR_SOP.md` §3 points at it | `a0317cdf` (2026-09-07) |
| Stale numerics sweep — `--eval-workers` default 3→**5**; the rust_sim `better_line` "STILL NEEDED" block wrong on all 3 claims (impl switch, `--impl rust`, the `input_log` blocker) rewritten to the ungated COMPOSITION; the 722-team pool re-measured at **762/762** (813 files, 51 validate-fail) with the 719-vs-762 disambiguation recorded | `a0317cdf` (2026-09-07) |
| Root `CLAUDE.md` restructure to a constitution + command card + map (206 KB → 32 KB); the two giant leaves split by topic | `206f792c` (2026-09-07) |
| `instrumented_ppo/ppo.py` decomposed around the fold sequence (1998 → 1331) + the contested-mask guard | `ccd08003` |
| **Anti-vacuity audit of test stubs** — `gen3_stub_vacuity_gate_v1`, unmarked in the routine gate (3.6 s). Census 407 patch sites / 70 files: 267 ok (44 of them cleared only by the consumer search), 139 skipped with reasons, **1 vacuous, 0 tests green for the wrong reason**; allowlist EMPTY. The finding was `dry_run_test.py`'s `_launch_child` booby trap, installed on the defining module while `run.py` calls its own module-level import — the trap whose tripping means `--dry-run` spawned a real training child. Census: `designs/research_state/measurements/stub_vacuity_audit_2026-09-07.md` | `83478fcd` |
| Shaped-era timeout comparator — **CLOSED, no comparator** (owner, 2026-09-06); the stripped arm's curve is the only one and the kill bar is registered against it | — |
