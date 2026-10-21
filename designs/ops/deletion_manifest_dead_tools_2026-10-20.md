# Deletion manifest: the five "dead tools" (owner approval 2026-10-20)

**Status: DONE for 3 of 5; STOPPED on 2 of 5 (live callers).** The owner approved deleting
`/gen3ai-update-design-docs`, `critic_gate`, `critic_read`, `policy_spectrum.bank` and `ridealong_read`
(`TECH_DEBT_BACKLOG.md` had asked "check whether `policy_spectrum.bank` and `ridealong_read` are dead tools
altogether"; `instruction_docs_audit_2026-10-19.md` F16 retired the slash command). The brief's stop rule: a tool a
LIVE path (training, eval, launcher, prober or meter) still imports or calls is not deleted, and a helper other code
imports stays. Grep scope: `src/`, `designs/`, `docs/`, `scripts/`, `.claude/`, `tools/`, root configs, at `cd8690c8a`.
History docs (`CHANGELOG`, the ledger, `claude_md_archive/`, `memory_archive/`, `measurements/`, dated log rows,
the deletion manifests and the instruction-docs audit) are left alone.

## Verdicts

| tool | verdict | why |
|---|---|---|
| `/gen3ai-update-design-docs` | **DELETED** | named only by history docs; no CLAUDE.md, no code |
| `main.ops.critic_read` | **DELETED** | no non-test importer, no non-test caller; its only live neighbours are libraries it used |
| `main.ridealong_read` | **DELETED** | no importer or caller outside its own package and its docs |
| `main.critic_gate` | **STOPPED** | called by `main.ops.g7_report` (subprocess) and `scripts/ops/famine_read.sh`, which the run SOP names as live-run instruments; listed in the prober web UI's command table |
| `main.policy_spectrum.bank` | **STOPPED** | imported by `main.belief_roles` and `main.x35.x47_report` (meters) and by the policy_spectrum reader, truth and qhat modules that those use; its `load_bank` is the only door to the committed Lane S banks |

## 1. `/gen3ai-update-design-docs` (DELETED)

- Own file: `.claude/commands/gen3ai-update-design-docs.md` (13 lines, 2,047 bytes; its description line was in
  every session's skill list).
- Importers / callers: none. Doc mentions: only history (`claude_md_archive/root_CLAUDE_2026-10-09.md`,
  `designs_CLAUDE_2026-10-10.md`, `memory_archive/project_team_pool_weighting_fix.md`,
  `instruction_docs_audit_2026-10-19.md` F16). No live CLAUDE.md names it. `designs/CLAUDE.md` and root
  `CLAUDE.md` describe `impl_step*.md` as explicit-only without the command.
- Record: `deleted_flags.md` Deleted PATHS row.

## 2. `main.ops.critic_read` (DELETED)

Own files (3,167 lines): `src/main/ops/critic_read.py` 1,179, `critic_read_render.py` 893, `critic_readouts.py` 226,
`critic_read_test.py` 869.

Importers outside those files, all TESTS of the tool:

| file | what it did with the tool | action |
|---|---|---|
| `src/main/ops/quota_match_test.py` | `CR.compute_deltas`, `critic_read_render` (9 tests + the `_delta_rows` helper) | those tests deleted; the 8 pure quota_match tests and the `artefact` frame-size test stay |
| `src/main/ops/eval_trace_gen_test.py` | 9 tests of the reader's refusals (`check_comparable`, `resolve_read_root`, `_fingerprint`, `ledger_line`, `_population_block`) | deleted |
| `src/main/ops/eval_trace_gen_integration_test.py` | `test_critic_read_picks_the_generated_cycle_and_its_gates_read_it` | deleted (the generator's own tests stay) |
| `src/main/ops/v_column_matched_test.py` | `test_the_two_paths_read_the_flag_through_ONE_accessor` | deleted (the other 5 tests exercise `conditioning_meters` / `quota_match` directly) |
| `src/main/ops/conditioning_meters_test.py` | `test_the_within_cell_resolution_reduces_to_the_ordinary_murphy_resolution` (compared to `critic_readouts`) | deleted |

Callers: `critic_read.py` shelled out to `main.critic_gate` (so deleting it removes a caller of the stopped tool).
No training, eval, launcher, prober or meter path calls it. No script, no `.claude/` file.

Code comments and docstrings naming it (reworded in place): `conditioning_meters.py`, `conditioning_meters_test.py`,
`team_conditioning.py`, `calibration_slope.py`, `quota_match.py`, `eval_trace_gen.py` (its refusals are now stated
as a reader's duty), `value_sidecar_read.py` (including two report-text sentences), `agents/training/value_sidecar.py`,
`tb_read_test.py`, the test docstrings above.

Docs rewritten to the current truth: `scripts/ops/README.md` (3 table rows + the library list),
`designs/ops/TRAINING_RUN_SOP.md` (the ladder-arm row, the high-power row, ~130 lines of critic_read guidance
replaced by a 30-line eval_trace_gen section: population, completeness, sentinel clamp, reproducibility),
`designs/ops/training_runbook.md`, `designs/training/value_sidecar.md`, `eval_and_rating.md`, `cf_grounding.md`,
`designs/endstate/design_arch_audit.md` (the screen's meter is now `main.critic_gate`'s G rows),
`designs/ops/TECH_DEBT_BACKLOG.md` (one open P3 row about `critic_read_render` removed), `designs/prober/engine_and_model.md`.
Left as history: dated MOOT / FIXED rows in `TECH_DEBT_BACKLOG.md`, `UNDERSTANDING.md` finding lines that cite
`critic_ladder_reads/*/critic_read_*.md` measurement files, `program_rust_core.md` and `design_arch_audit.md` dated log rows,
`legacy_removal_manifest.md`, `deletion_pass_manifest.md`.

Data kept: `designs/research_state/measurements/critic_ladder_reads/` (289 committed outputs).

**Left behind, unreferenced:** `conditioning_meters.py` (+ test), `quota_match.py` (+ test), `team_conditioning.py`,
`calibration_slope.py`, `v_column_matched_test.py`: after this deletion their only consumers are each other and their
tests (`quota_match.py` is also imported by `prober/core_trace_readers_test.py`). Deleting them is a separate
decision (FINDING in the commit report).

## 3. `main.ridealong_read` (DELETED)

Own files (3,964 lines of code and test): `src/main/ridealong_read/` = `__init__`, `__main__`, `boot`, `meters`,
`reader`, `rnd_choice`, `rnd_states`, `variants`, `variants_probe` and the tests `boot_test`, `meters_test`,
`reader_columns_test`, `rnd_choice_test`, `rnd_states_test`, `variants_test`, `ridealong_read_integration_test` (slow;
two rows in `slow_tier_status.json` removed).

Importers / callers outside the package: none (grep for `import` and `-m main.ridealong_read` over `src/`,
`scripts/`, `tools/`). The package itself imported `main.policy_spectrum.{bank,reader,spectrum,truth,qhat,qhat_report}`
(which stay).

Comment-only mentions fixed: `agents/model/ridealong_heads.py` (a docstring naming `rnd_choice`),
`agents/training/ridealong_host_sync_test.py`.

Docs rewritten: `designs/training/ridealong_heads.md` (the offline-reader sentence),
`designs/endstate/design_collapse_and_counterfactuals.md` (B5 / B8 / B9 and F-CC-4 now say no reader is in the tree),
`design_compute_efficiency.md` T59, `design_x5_belief_tokens.md`, `design_x26_readiness.md`,
`designs/ops/flag_census.md`, `designs/ops/TASK_BACKLOG.md` X35-6, `TECH_DEBT_BACKLOG.md` (two open rows),
`training_runbook.md`'s offline-meter index.

**Consequence the owner should know:** the registered X26 ride-along reads R1-R5 and battery items B5 / B8 / B9 have no
reader now. `design_x26_readiness.md` F20 already recorded R1xR2 and R5 as having no reader code. `ridealong_heads.py`,
`--ridealong-ensemble` (production 0) and the heads' training meters are untouched.

## 4. `main.critic_gate` (STOPPED, not deleted)

Own files (3,143 lines): `src/main/critic_gate.py` 1,462, `critic_gate_design.py` 128, `critic_gate_render.py` 390,
`critic_gate_test.py` 1,163.

Live callers (the stop condition):

- `src/main/ops/g7_report.py:82`: `python -m main.ops.g7_report` runs `main.critic_gate` as a subprocess; the run SOP
  (`TRAINING_RUN_SOP.md` section 2 table) lists it as the live-run G7 quote.
- `scripts/ops/famine_read.sh:107`: "the deciding famine read" (same table).
- `src/main/prober/web/templates/index.html:130`: the prober web UI's command table; `index_commands_test.py:76` asserts it.
- `src/main/ops/killbar.py:264-268` prints "see `python -m main.critic_gate`"; `src/main/ops/calib_trend.py` parses a gate report.
- `designs/ops/training_runbook.md` offline-meter index, `TRAINING_RUN_SOP.md` lines 274 / 587.

Test-only importers that would need work: `src/utils/era_test.py:165`, `src/agents/training/ladder_recipe_test.py:122`,
`src/main/prober/core_trace_readers_test.py:110` (imports the private `_trace_turns`), and the entry-point list in
`src/poke_env_free_entry_points_test.py:92`.

What deleting it takes (a separate approved unit): also retire `g7_report`, `famine_read.sh`, `calib_trend`, the
killbar message, the prober index row and its test, the SOP rows, and re-home or drop three test imports.

## 5. `main.policy_spectrum.bank` (STOPPED, not deleted)

Own files: `src/main/policy_spectrum/bank.py` 511, `bank_test.py` 123 (`BANK_V1` fixture).

Live importers of `bank` (module or `load_bank` / `Bank`):

- `src/main/belief_roles/bank_rows.py:140` (the X5 belief reader, an offline meter in the runbook index).
- `src/main/x35/x47_report.py:230` (`--bank bank_x26`, the registered X26 bank).
- the package's own `reader.py`, `truth.py`, `qhat.py`, `qhat_report.py`, `report.py`, `truth_report.py`, `__main__.py`,
  which `probe_battery/pin_worker.py`, `belief_roles/__main__.py`, `x35/x47_report.py` and `torch_globals_restore_test.py`
  import (`reader.load_checkpoint`, `inference_globals`).
- tests across live areas that use the committed bank as their fixture: `agents/info_levers/inject_integration_test.py`,
  `agents/action/move_order_bridge_integration_test.py`, `agents/model/speed_physics_bridge_integration_test.py`,
  `agents/model/move_resolution_bridge_integration_test.py`, `main/value_search/{search,world_rolls,world_guard}_integration_test.py`,
  `agents/battle/gender_draw_pad_core_test.py` (docstring).

What stays true: the bank BUILDER (`bank.build`, `SOURCES_V1`) names `ai_v14_*` runs that no longer load
(`TECH_DEBT_BACKLOG.md` P3 "Stale ai_v14 defaults"); the loader half is live. If the owner wants the builder gone, that is
a narrower unit (delete `build` and `SOURCES_V1`, keep `load_bank`).

## 6. Recording

- `designs/deleted_flags.md`: two 1c rows (`python -m main.ops.critic_read`, `python -m main.ridealong_read`) and one
  Deleted PATHS row (the slash command).
- `designs/ops/slow_tier_status.json`: 12 rows of removed slow tests dropped.
- No CLAUDE.md named any of the three deleted tools; none changed. `scripts/ops/README.md` and
  `TRAINING_RUN_SOP.md` shrank (about 5.5 KB and 12 KB).
