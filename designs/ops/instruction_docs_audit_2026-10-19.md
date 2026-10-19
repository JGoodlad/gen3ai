# Instruction-docs audit — 2026-10-19 (phase 1)

**A dated record, not an always-current doc.** It audits every `CLAUDE.md`, the user-memory index
`MEMORY.md`, the slash commands and the subagent definitions against
[`instruction_docs_principles.md`](instruction_docs_principles.md), and plans the phase-2 sweep.
Phase 1 edited no instruction file. **Snapshot:** `c6a14d167`. Main has moved since:
`f20526571` rewrote one `gen3_data` leaf sentence, and three build agents and a test-gate agent were
editing leaves on the audit day. **Phase 2 re-measures from the then-current main**, so read every
figure below as an estimate for that unit.

**Method.** Sizes are bytes; tokens ≈ bytes ÷ 4 (no tokenizer was run, so tokens are ±20 %). Four
read-only auditors (one per group of files) checked about 470 concrete claims against the code:
names, numbers, behaviours and the paths and flags the freshness gate already covers. They
classified every section as KEEP / TIGHTEN / MOVE / DELETE. For each MOVE they checked whether the
destination already holds the text; if it does, the class is "DELETE as duplicate". Each file also
has a "must survive" list, the checklist its phase-2 editor works against.

---

## 0. Headline

| | today | proposed | change |
|---|---:|---:|---:|
| **always loaded** (root + `MEMORY.md` + one agent definition) | 54.0 KB ≈ 13.5k tokens | 19.6 KB ≈ 4.9k tokens | **−64 %** |
| all 13 `CLAUDE.md` files | 233 KB ≈ 58k tokens | ≈ 75 KB ≈ 19k tokens | −68 % |
| editing `src/main/prober/web/` (root + both prober leaves) | 72 KB | ≈ 23.5 KB | −67 % |
| editing `src/agents/training/` (root + training leaf) | 74 KB | ≈ 22.5 KB | −70 % |

**The worst offenders:**
- **Root `CLAUDE.md`** (38.6 KB, loaded everywhere). It has 38 🚨 and about 45 % duplication; it is accurate but bloated.
- **Training leaf** (35.5 KB, 45 🚨). Every rule in it is already in `designs/training/` (49 of 49 key terms checked).
- **`gen3_data` leaf** (23.8 KB). Six stale claims.
- **rust_sim leaf** (22.9 KB, 27 🚨).
- **Prober leaf + web leaf** (33.7 KB, loaded together).

**Why two earlier cuts did not hold.** Each cut was a one-off with nothing to stop regrowth:

| date | root | training leaf |
|---|---:|---:|
| 2026-09-08 (after the first cut) | 37.5 KB | 58.8 KB |
| 2026-10-09 | 70.7 KB | 132.6 KB |
| 2026-10-10 (after the second cut) | 32.9 KB | 26.1 KB |
| 2026-10-19 | 38.6 KB | 35.5 KB |

The root regrew by about 0.6 KB a day. **The fix is a ratchet the gates enforce** (§5 U1, §6), not
a third cut.

**What we found, by kind:**
- **Duplication of the owned trees.** Almost every leaf rule already lives in its `designs/<tree>/`
  doc. The auditors checked 49 / 49 key terms for training, 30 / 30 for model and 11 / 11 for
  observation, and found the same for the prober, launcher and rust_sim trees. Most cuts are
  therefore DELETE-as-duplicate, not moves.
- **Emphasis inflation.** 🚨 counts: root 38, training 45, rust_sim 27, model 26, prober 17,
  launcher 16, tools 9, observation 8, `gen3_data` 8, `MEMORY.md` 8. The principle allows five per file.
- **Stale claims the freshness gate cannot see.** These are symbols, behaviours and section
  pointers. There are about 20 across the files (F5–F12), and two are live hazards: the missing
  `EVENT_COL` / torch.fx note (F6) and `/gen3ai-resume-run` (F14).
- **Accuracy is otherwise high.** The root, launcher, prober, web, battle and tools leaves had 0
  wrong names in about 175 checks. The freshness gate works; volume is the problem.

---

## 1. Summary table

| file | lines | bytes | ≈ tok | wrong / stale | narrative | dup | 🚨 | proposed | class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `CLAUDE.md` (root) | 302 | 38,589 | 9.6k | 0 | 20 % | 45 % | 38 | 13,000 | TIGHTEN, last |
| `src/agents/training/CLAUDE.md` | 361 | 35,505 | 8.9k | 2 | 15 % | ~90 % in tree | 45 | 9,500 | TIGHTEN |
| `src/agents/gen3_data/CLAUDE.md` | 289 | 23,773 | 5.9k | 6 | 25 % | 30 % | 8 | 4,200 (+5,000 moved) | TIGHTEN + MOVE |
| `src/rust_sim/CLAUDE.md` | 312 | 22,911 | 5.7k | 2 | 15 % | 45 % | 27 | 9,000 | TIGHTEN |
| `src/main/prober/CLAUDE.md` | 281 | 22,718 | 5.7k | 0 | 10 % | 50 % | 17 | 6,300 | TIGHTEN |
| `src/agents/model/CLAUDE.md` | 259 | 21,507 | 5.4k | 2 | 8 % | ~90 % in tree | 26 | 8,200 | TIGHTEN |
| `src/main/launcher/CLAUDE.md` | 175 | 15,550 | 3.9k | 0 | 15 % | 75 % | 16 | 5,200 | TIGHTEN |
| `designs/CLAUDE.md` | 155 | 15,302 | 3.8k | 0 | 25 % | 40 % | 1 | 5,600 | TIGHTEN + MOVE |
| `src/main/prober/web/CLAUDE.md` | 149 | 10,949 | 2.7k | 1 | 8 % | 50 % | 1 | 4,200 | TIGHTEN |
| `src/agents/observation/CLAUDE.md` | 121 | 9,291 | 2.3k | 3 | 15 % | 40 % | 8 | 3,600 | TIGHTEN |
| `tools/CLAUDE.md` | 102 | 8,260 | 2.1k | 1 | 10 % | 27 % | 9 | 3,900 | TIGHTEN |
| `src/agents/battle/CLAUDE.md` | 55 | 5,034 | 1.3k | 0 | 18 % | 16 % | 3 | 1,900 | TIGHTEN |
| `src/main/tui/CLAUDE.md` | 64 | 3,697 | 0.9k | 3 | 3 % | 40 % | 0 | 800 | TIGHTEN (or fold away) |
| **13 `CLAUDE.md`** | **2,625** | **233,086** | **58k** | **20** | | | **199** | **≈ 75,000** | |
| `MEMORY.md` | 65 | 12,631 | 3.2k | 3 | high | 35 % | 8 | 5,800 | TIGHTEN + MOVE |
| `.claude/commands/` (6) | 625 | 34,932 | 8.7k | 3 files | | | 0 | ≈ 22,000 | 1 TIGHTEN, 1 DELETE, 1 RETIRE |
| `~/.claude/agents/` (6) | — | 16,945 | 4.2k | 0 | | ≈ 75 % | 0 | ≈ 4,800 | TIGHTEN |

"dup" means the share of bytes another home already holds; "~90 % in tree" means the leaf is
essentially a summary of its `designs/` tree.

---

## 2. Per-file notes

### 2.1 Root `CLAUDE.md`

- **Loaded:** everywhere.
- **Accuracy: high.** 26 claims checked, 0 wrong. The checks covered the 17 static gates,
  `gate_lock.sh --status`, `DEFAULT_RESTART_INTERVAL_HOURS = 6.0`, `trunk_precision: bf16`,
  `torch_runtime.carries`, `fixture_battles.play_rows`, the 2845-dim observation,
  `arch_arms.legacy_production_config`, the refusal of ports 8000 / 8001, `main.live.halt clear
  --fixed-by`, `GEN3AI_SKIP_DEPS_GUARD`, the empty poke-env allowlist, 13 `python -m` modules, and
  the cargo `selfcheck` profile with its three binaries.
- **Copied numbers that will rot:** "the same 240 tests" and "~4.3 min quiet" cannot be checked
  cheaply. "**Seventeen static gates**" is pinned by a test, so it cannot go stale (F2).
- **Narrative:** about 20 % — 29 dates, 19 "owner" citations, and incident asides such as
  "unnoticed for four weeks", "−25 % … −30 %", "131 → 0 watchdog aborts" and "24.4M steps on a
  near-bare network".
- **Duplication:** about 45 %, of the following homes:
  - the runbook's "Launch hazards in full";
  - `testing.md`;
  - the launcher leaf (`gen3ai_stable`);
  - `ORCHESTRATOR_SOP.md` §2 / §7;
  - the ship command (rebase mechanics);
  - the prober, observation, model and `gen3_data` leaves;
  - `EXTERNAL_ANCHORS_SOP.md`;
  - itself: § Showdown Server repeats rule 1, and the leaf table repeats the map.
- **Enforced elsewhere:** these need a line or nothing:
  - ports 8000 / 8001;
  - poke-env (two gates);
  - the T28 halt;
  - `--mode ladder` (`ladder_guard`);
  - the deleted `--critic` / `--env-core` / `--gamma`;
  - the conftest deps guard;
  - the bootstrap shared-env refusal;
  - the torch-build switch;
  - torch global-state leaks;
  - `/home/…` literals.

| § | bytes | class → bytes | reason |
|---|---:|---|---|
| title + preamble | 525 | TIGHTEN → 250 | one sentence of purpose plus a principles pointer; drop the archive pointer |
| Where the detail is | 1,712 | KEEP → 1,600 | the map; absorb the leaf table and the Prober / Playing / Model pointers as rows |
| Development Stage | 563 | TIGHTEN → 300 | keep "no compatibility shims" and "constants live only in `arch_constants.py`" |
| Standing rules | 3,407 | TIGHTEN → 2,200 | keep all ten; drop quotations, percentages and dates; fold in § Showdown Server |
| Subagents, Workflows and waiting | 1,790 | TIGHTEN → 600 | keep named types, one bounded 55-min timer, `stallMs`; history and `parallel()` → `ORCHESTRATOR_SOP.md` §7 / §2 (already there) |
| Documentation Maintenance | 3,342 | TIGHTEN → 1,300 | keep the always-current, append-only and explicit-only lists plus the principles pointer; DELETE the 12-row leaf table |
| Git Workflow | 1,363 | TIGHTEN → 700 | worktree only, land via `/gen3ai-ship`, who ships (one line), no attribution; rebase mechanics are the ship command's |
| Python Environment | 3,663 | TIGHTEN + MOVE → 900 | keep the env, the interpreter, the worktree `PYTHONPATH` hazard and bootstrap; cu128 → runbook / `cloud_gpu_benchmark.md`; `gen3ai_stable` → launcher leaf (already there); poke-env is enforced |
| Git Worktree Setup | 1,933 | MOVE → 250 | `bootstrap.sh` does it and conftest refuses with the fix; the recipe and the ELOOP trap → `docs/DEVELOPING.md`; Node's role → rust_sim leaf |
| Running Tests | 4,979 | TIGHTEN → 2,000 | keep the command table and the silent-failure rules; the rest is in `testing.md`; the cargo self-check build → rust_sim leaf |
| Smoke Test | 903 | TIGHTEN → 400 | the command and `GEN3AI_MODELS_DIR`; "what to look for" is in the runbook |
| Training + the Launcher | 4,259 | MOVE → 700 | the runbook already holds every hazard; keep `checkargs` + `--dry-run`, never launch-and-kill, and "an argv is not a config" |
| Playing / the LADDER | 3,305 | TIGHTEN + MOVE → 600 | keep "no ladder, no human contact by agents" and "a parse panic halts all play"; commands and the halt procedure → `EXTERNAL_ANCHORS_SOP.md` / `designs/rust_sim/live_reader.md` |
| Prober | 993 | MOVE → 0 | the prober leaf owns it; a map row stays |
| Showdown Server | 591 | DELETE | repeats rule 1 |
| Repository Structure | 1,337 | DELETE → 100 | derivable with `ls` (Anthropic's exclude list) |
| Path discovery | 798 | TIGHTEN → 400 | `utils.paths`; `models/` only in main; the `run_archive` fixture |
| The Model | 2,407 | MOVE → 400 | obs layout → observation leaf; extractor pairing → model leaf; "END STATE is production" → `ARCHITECTURE.md`; keep baselines-by-name and immutable sidecars |
| Data Dependencies | 701 | TIGHTEN → 350 | Smogon-only priors and the facade; the forme `num` trap → `gen3_data` leaf |

**Must survive:**
- **Standing rules and environment:** the ten standing rules; the env and interpreter path; the worktree `PYTHONPATH` hazard.
- **Testing:** the routine-gate, static-tier, GPU and `mem_cap` commands; cut on `slow`; empty static allowlists; recorded slow-tier verdicts; a timeout is never an outcome; an obs change runs the encoder benchmark and the golden.
- **Launching:** the smoke command; `checkargs` / `--dry-run`; an argv is not a config.
- **Play:** no ladder or human contact by agents; the T28 halt; the drift gate before a public-server session.
- **Paths, data and history:** path helpers; `models/` only in main; baselines by name; Smogon-only priors; never edit the ledger.
- **Shipping:** who may ship; no attribution.

### 2.2 `src/agents/training/CLAUDE.md`

- **Accuracy:** 88 claims checked; 1 wrong, 1 imprecise.
  - Wrong: line 161 says "`--opponent <a v144+ checkpoint>`", but `MIGRATION_FLOOR = 145` and the refusal prints the floor itself.
  - Imprecise: `labels/margin.rs` is in `src/rust_env/`, not `rust_sim`.
- **Narrative:** about 15 %, plus 40 ticket codes and 10 `*_v1` tags.
- **Duplication:** the root holds about 3.5 KB of it (recipe and dose, fork pool, bare run dir,
  baselines, deleted flags, bf16 trunk, lineage); the model leaf holds about 1.5 KB
  (`belief_supervision`, the critic, ride-along heads); every rule is in `designs/training/`.
- **Enforced elsewhere:** about 30 guards (startup key checks, the K9(b) check, host-sync and
  lifecycle gates, typed fatal exits). Each shrinks to a clause.
- **Wasted text:** 1.9 KB on `--learner-cudagraphs on`, a lever that still fails on the GPU.

| § | bytes | class → bytes |
|---|---:|---|
| title + current world | 1,100 | TIGHTEN → 350 |
| topic map | 3,400 | TIGHTEN → 1,800 (bare pointers) |
| closed history | 500 | DELETE |
| TB currency | 1,200 | TIGHTEN → 400 |
| metric keys / one producer | 1,900 | TIGHTEN → 2 lines (enforced) |
| 28-tag dashboard | 1,300 | MOVE → `designs/training/telemetry_scalars.md` (**test-pinned**, F2) |
| PPO step / fold order | 3,800 | TIGHTEN → 1,300 |
| reward | 750 | TIGHTEN → 200 |
| eval / ledger / ELO | 3,400 | TIGHTEN → 1,200 |
| opponents / pool / supply | 2,500 | TIGHTEN → 600 |
| Rust collector | 2,600 | TIGHTEN → 500 |
| compile | 6,300 | TIGHTEN → 1,300 (the rest duplicates `compile_flags.md`) |
| critic / value / aux | 2,500 | TIGHTEN → 600 |
| belief losses | 830 | KEEP → 400 (the one home of `belief_supervision`) |
| recipe / dose | 2,150 | TIGHTEN → 300 (root duplicate) |
| offline meters | 840 | TIGHTEN → 150 |
| DELETED | 700 | TIGHTEN → 150 |

**Must survive:**
- **The PPO step:** fold order is fixed and a tail fold's own forward clobbers stashes; R1 is static-shape and allows no host reads; the dump comes before the update, and `isolated_dump` is used mid-rollout; no global reseed after the freeze, and everything is built at startup; the K9 golden is re-recorded only deliberately; never mask a non-finite value.
- **Metrics:** a new key goes through the real logger from one call site; tags are in probability units, and an absent source leaves a gap.
- **Opponents and eval:** `resolve_model_ref` is the only choke point, and the spec hash is a regime tag; eval regimes are inherited; nickname is not species; a recorder never kills a run; the p1 seat effect; `baselines.load`; selecting games never rate; bump `LADDER_FITTER_VERSION`; never write through a hard link; `lever_supply.loud`; team corpora never go under `data/`.
- **Collector:** a cadence counts total env steps; `single_thread_build`; one `trainee_spaces`.
- **Compile:** `compile_control.py` alone touches dynamo; `ensure_hermetic_cache` and `COMPILE_ARG_KEYS`; regenerate the parity fixture; gate forwards keep nothing; `fp32_scope`; do not launch `--learner-cudagraphs on`; nothing reads `--canary-shift-max`.
- **Value and belief:** a draw is a not-win and the cap is a terminal; `belief_supervision`, not `last_*`; the value sidecar is unrecoverable after the fact.

### 2.3 `src/agents/gen3_data/CLAUDE.md`

- **Accuracy:** 32 claims checked, 6 wrong or stale (F5):
  - The "Performance note" is obsolete since poke-env retired, and names `moves._category_val`, which no Python file has.
  - `hidden_power_tracker` does not read priors.
  - The archetypes file's "league targeting" consumer does not exist; the real consumers are `team_winrate_callback.py` and `promote_teams.py`.
  - The `GEN3_HP_IVS` 🚨 line is stale on main (`f20526571`).
  - `build_move_prior_logits` moved to `belief_tables.py`; it is still re-exported, so this one is minor.
  - The file refers to itself as "(see `src/agents/gen3_data/CLAUDE.md`)".
- **Narrative:** about 25 %.
- **Duplication:**
  - The forme rule appears four times: in the root, twice in this file, and in `tools`.
  - The weighted-count rule W is also in `tools` and in `designs/tools/smogon_priors.md`.
  - The format spec is also in `design_format_spec.md`.
  - The HP-repair text is also the `hp_repair.py` docstring.

| § | bytes | class → bytes |
|---|---:|---|
| intro + split + `num` refusal | 1,153 | TIGHTEN → 300 |
| facade API listing | 2,191 | TIGHTEN → 350 (derivable) |
| HP repair | 1,686 | DELETE as duplicate (docstring) → 1 line |
| format spec | 2,177 | TIGHTEN → 600 |
| learnset legality | 688 | TIGHTEN → 250 |
| prevo chain | 1,418 | DELETE as duplicate (`designs/tools/pokemon_data_extractor.md`) |
| formes | 1,698 | TIGHTEN → 300 |
| concept-module discipline | 1,319 | TIGHTEN → 400 |
| who reads it / performance note / "moved here" | 1,432 | DELETE (stale, obsolete, history) |
| value-neutrality | 1,113 | TIGHTEN → 300 |
| dex file schemas | 4,471 | MOVE → new `designs/tools/data_files.md` (keep ≈ 700) |
| Smogon priors | 3,044 | TIGHTEN → 400; rest → `designs/tools/smogon_priors.md` |
| archetypes | 741 | MOVE → `data_files.md` (fix the consumer) |
| closing poke-env paragraph | 627 | DELETE |

**Must survive:**
- **The facade:** the runtime reads `data/` only through `gen3_data`; `get()` tolerates a miss and `*_data()` raises; the enums are keys only.
- **Formes:** `raw()` includes formes; num-indexed consumers iterate `base_form_ids()`.
- **Moves and Hidden Power:** the dex `category` decides only Status vs damaging (gen 3 type decides physical vs special), and no designs doc holds this (F5); typed Hidden Power nums 355–370, and 237 for the unrevealed one.
- **Resolution, rules and priors:** a packed token needs `canonical_id`; a learnset `None` means no constraint; format rules come from `format_spec`; priors use W, never `Raw count`.
- **Changes:** a data VALUE change is retrain-class.

Moving the schemas also updates `designs/tools/README.md` and `tools/CLAUDE.md:9`, which both say
this leaf owns them.

### 2.4 `src/rust_sim/CLAUDE.md`

- **Accuracy:** 28 non-path claims checked; 0 hard-wrong, 2 defects.
  - "762 today" and "as of ROUND 59" contradict the section's own never-quote rule.
  - `turn.rs::display_name` is in `turn/helpers.rs`.
- **Duplication:** about 10 KB.
  - The tree's "Moved from the leaf" sections already hold the module map, the census, the gate table, the fuzzer table and the standing lessons.
  - The root holds about 1.1 KB.
- **Enforced elsewhere:** `emission_check_test.rs`, the layout test, `core_events_test.rs`,
  `handler_audit_test.rs`, `.gitignore` (`ab_fuzz_out*`), `ElidedSide`, and the parser refusals.

| § | bytes | class → bytes |
|---|---:|---|
| intro + transport | 983 | TIGHTEN → 350 |
| "this leaf is the RULES" | 372 | DELETE |
| move census | 1,873 | TIGHTEN → 350 (recount with `scan_move_probe`, never quote) |
| why bit-for-bit | 854 | TIGHTEN → 450 |
| module map | 2,716 | DELETE as duplicate (`module_map.md`) |
| Rust core rules | 2,436 | TIGHTEN → 1,400 |
| cargo commands / no build into main's `target/` | 723 | TIGHTEN → 400 |
| offline drivers | 2,544 | TIGHTEN → 900 |
| gate ladder | 1,471 | TIGHTEN → 350 |
| regression pins | 696 | KEEP → 500 |
| four fuzzers | 2,611 | TIGHTEN → 850 |
| data-driven / handler audit | 1,487 | TIGHTEN → 550 |
| protocol emission | 796 | TIGHTEN → 350 |
| standing lessons | 1,317 | DELETE as duplicate (`port_build_log.md`) |
| conventions | 393 | KEEP → 300 |
| where the rest lives | 1,624 | TIGHTEN → 900 (one pointer table) |

**Must survive:**
- **Fidelity:** bit-for-bit PRNG draw order and byte-identical lines; the census is recounted, never quoted (0 MISMODELED; MODELED ≠ gated); canonical `parse(render(l)) == l`.
- **Code rules:** `in_nested_move_scope`; `schema.rs` / `tables.rs` / `layout.rs` are Rust-owned, and `layout.rs` changes with `constants.py`; `present()` takes no board; no core recording unless asked; the encoder reproduces and never fixes; std-only and generation-generic.
- **Builds:** the self-check build goes in `target/selfcheck` and never into main's `target/`.
- **Seeds and replay:** two fresh seeds before green; allowlist claims verified; gen3ou needs `allowHiddenPower`; `volatile_names` never reach the obs layer; the gender draw pad is offline-only.
- **Gates and pins:** strict E2E, where a regen is a new golden; every fuzz bug becomes a named pin, proven by a revert; an allowlist edit needs an injection proof plus `--selftest`.
- **Mechanics data and emission:** the mod-chain law and its `--check` / `--audit` commands; emission draws no PRNG; a `MonRef` is the nickname.

### 2.5 `src/main/prober/CLAUDE.md` and `src/main/prober/web/CLAUDE.md` (they load together)

- **Accuracy:** prober 42 checks, 0 wrong; web 31 checks, 0 wrong.
  - Imprecise: `battle_row` is a closure, not a module attribute; the `run_<timestamp>` example predates the era names.
- **Duplication:**
  - Prober about 11.5 KB: the root § Prober; `designs/prober/engine_and_model.md`, which holds the module bullets almost word for word; `query --help` `_EXAMPLES`, which covers the CLI block; `session_scans.md`, `sim_impl.md`, `retention.md` and `tests.md`.
  - Web about 5.5 KB: the parent leaf, the root, and `web_service.md` §1 / §3 / §10.
- **Derivable:** both module tables (3.2 KB) restate each module's first docstring line (33 of 33
  checked); the web leaf's 13-row map does the same.
- **Hazards that cannot fire today (F20):** no trace from before the `rb_` era survives retention,
  and every `rb_` run is `critic: winprob`. The shaped-critic, `overvalue_tau`, pre-2026-09-07
  draw and `best_model`-mislabel cases can therefore go. Keep the rules: read `critic_currency`;
  name the rung.

Prober leaf, 22,718 → ≈ 6,300:
- **KEEP:**
  - the pointer table;
  - the engine / app seam;
  - the hub DAG line;
  - the timeline-vocabulary placement;
  - the "no effect" claim, raw-protocol detectors, raw + exclusive belief, β-name provenance, trace quota and move-label hazards (one line each);
  - the `ProbeSession` contract;
  - the two counterfactual must-flags.
- **DELETE as duplicate:**
  - both module tables;
  - the other-modules bullets, except three one-liners;
  - current-arch-only;
  - blocking work;
  - `models/`;
  - the CLI block (`--help`);
  - the investigation recipe, except "start with `triage`";
  - `--compile` / Rust-only;
  - tests;
  - the web pointer;
  - the retired TUI, which refuses its old flags.

Web leaf, 10,949 → ≈ 4,200:
- **KEEP** rules 1–5, 7–9, 11 and 12, and the starlette and HTMX-`Form` gotchas.
- **DELETE:**
  - the map;
  - rule 10 (`web_service.md` §3);
  - rule 6 (one home, in the parent);
  - "Commands a template prints", whose premise is false (F10);
  - deployment (`web_service.md` §1);
  - most of tests (`web_tests.md` §5);
  - the ports / `models/` / `build_trace_tree` gotchas.

**Must survive:**
- **The engine:** one engine, two surfaces, with torch only through the injected model; a turn's events come from decision T+1; `core_trace` is the sole summary reader, and a mismatch raises and is never repaired.
- **Errors and models:** `ArchDriftError` is rendered as `plain` and branched on `kind`; resolution goes exact → nearest → last, with `best_model` named; the turn cap is checked before the loss; an unknown result is refused.
- **Reading results:** "no effect" needs evidence; read `critic_currency` before any V; every β name carries `BELIEF_NAME_CAVEAT`; read `trace_selection` first; labels are in action order; offsets come from `get_layout()`.
- **Counterfactuals:** the interior opponent is flagged; one rollout is not a probability.
- **Web:** numbers come verbatim from the session; gates go only in `dependencies=`; membership, never a path join; bound anonymous growth and fail closed; secrets never in argv; banners never dropped; long work goes to jobs, and handlers are `def`; perspective only via `fv`; thresholds default to `None`; a template edit needs a restart.

### 2.6 `src/agents/model/CLAUDE.md`

- **Accuracy:** 82 claims checked, 2 wrong.
  - It says "each topic doc ends with a Moved-from-the-leaf section", but two docs do not.
  - Line 49 says re-meaning an observation block bumps `ARCH_SIGNATURE`. Line 210, `versioning.md` and `constants.py` say a meaning-only change sets `OBS_SEMANTICS_VERSION` and gates no resume. **This is F4 and needs a ruling.**
- **Duplication:** every key hazard is in `designs/model/` or `ARCHITECTURE.md` (30 of 30 checked).
  The root and the training, observation and prober leaves carry more copies.
- **Wasted bytes:** the `../../../` relative links roughly double the topic map's size.

| § | bytes | class → bytes |
|---|---:|---|
| preamble | 500 | KEEP → 300 |
| topic map | 2,400 | TIGHTEN → 1,000 (plain paths) |
| closed history | 200 | DELETE |
| arch constants | 620 | TIGHTEN → 150 |
| phase rules | 1,100 | TIGHTEN → 600 (after F4) |
| belief-grad-mode | 400 | TIGHTEN → 150 |
| dual head / critic / ride-along | 1,700 | TIGHTEN → 450 |
| hazards in the forward | 3,400 | TIGHTEN → 1,500 |
| X5 + static encoder | 2,700 | TIGHTEN → 900 |
| compile spellings | 1,950 | TIGHTEN → 800 |
| identity-at-init | 520 | KEEP → 300 |
| flag registry | 1,350 | TIGHTEN → 400 |
| versioning playbooks | 2,800 | TIGHTEN → 900 |
| opponent intent | 1,700 | TIGHTEN → 400 |
| mypy | 310 | TIGHTEN → 1 line (the one home of the mypy rule) |

**Must survive:**
- **Versioning:** a phase-structure or forward-math change bumps `ARCH_SIGNATURE`, and nothing else catches it; the `OBS_SEMANTICS_VERSION` rule; every `--arch production` value is a `ModelVersion` field; deleting an extractor kwarg follows `versioning.md`; end-state levers go through `adopted_levers.py`; `resume_immutable` is re-applied on resume.
- **The forward:** never hand-build an obs dict; `last_tensors` vs `last_raw_tensors`; `forward` is not re-entrant and stays on `Gen3FeaturesExtractor`; two move orders cross by identity only; `ability_known`; `selection_sites.py`; presence semantics with π detached; the `PokemonEncoder` stitch; S / D / board classification; the compile spellings; tables import downward only.
- **Heads and tests:** check identity-at-init on a real policy; `is_winprob`; `value_pooled` is vf-only; detach the logits, never the product.
- **Obligations:** the hand-computed-features ledger.

### 2.7 `src/main/launcher/CLAUDE.md`

- **Accuracy:** 46 checks, 0 wrong.
- **Duplication:** about 12 KB. The root holds about 2.5 KB; the runbook's § Launcher and
  "Launch hazards in full" hold more; and `designs/launcher/` holds every hazard, both tables and
  the interpreter paragraph (`restarts_and_resume.md`, `pinning_and_worktrees.md`,
  `launch_guards.md`, `crashes_and_exit_codes.md`, `flags.md`, `interpreter.md`). `--help`
  repeats the flag table.
- **Enforced elsewhere:** about 20 guards and tests.
- **Test-pinned (F2):** `disk_guard_wiring_test.py` needs "FATAL_DISK" and "--allow-low-disk" in
  this leaf, and `restart_interval_default_test.py` needs its flag-table row.

Launcher leaf, 15,550 → ≈ 5,200:
- **KEEP, one line each:**
  - `--dry-run` stays pure (the booby traps);
  - the resume role, with recipe knobs typed to the child;
  - one fork-vs-restart predicate;
  - the child gets an absolute `--run-dir`;
  - the prune keeps on any ambiguity;
  - only parser verdicts refuse;
  - never "clean up" the child's `PYTHONPATH`, and pass no `cwd=`;
  - one git-hash resolver;
  - a restart never moves the pin;
  - single-function surface guards;
  - the launcher owns no trainer default (`allow_abbrev=False`);
  - detached means headless;
  - `KILL_GRACE` exceeds deadline + budget;
  - the preflights run before the pin;
  - new fatal exits go through `exit_code_for`;
  - the interpreter is decided once per session.
- **DELETE as duplicate:** the module map, the exit-code table, the flag table, the interpreter
  section, the tests list, the torch / ports / run-dir bullets.

### 2.8 `src/agents/observation/CLAUDE.md`

- **Accuracy:** 44 checks, 3 wrong (F6):
  - `obs_with_event_row` does not exist; the function is `obs_with_event`.
  - It says model consumers import `EventCol`. They import the plain-int mirror `EVENT_COL`, because an `IntEnum` member breaks torch.fx code generation. **A consumer that follows the leaf would break the compiled forward.**
  - The poke-env warning is obsolete.
- **Duplication:** the root holds the index rule and the benchmark mandate nearly verbatim; the
  model leaf holds the two move orders; the tree has 11 of 11 hazards.
- **Derivable:** the 19-row file map (2.1 KB).

Observation leaf, 9,291 → ≈ 3,600:
- **KEEP:** "two things can regress", tightened and fixed for F4; the rules and hazards, tightened.
- **DELETE:** the benchmark mandate (the root), the map (keep three facts), poke-env.

**Must survive:**
- **Layout:** `layout.rs` and `constants.py` are edited together, and the prober pins move with them.
- **Events and volatiles:** `EventCol` order is load-bearing, and compiled code uses `EVENT_COL`; volatiles are classified, never caught-all.
- **Specific dims:** the ACTIVE flag is the last dim; `turns_since_progress` is not the deadline; embedded ids are declared.
- **History and tooling:** the archived TurnDelta / incoming-damage encoders are not revived; never loosen `mypy.ini`.

### 2.9 `tools/CLAUDE.md`, `src/agents/battle/CLAUDE.md`, `src/main/tui/CLAUDE.md`

- **tools** (8,260 → ≈ 3,900). 16 checks.
  - One omission: `build_moves` is in neither build-requirements group.
  - The forme, mod-chain and W rules duplicate the root, `gen3_data`, rust_sim and `designs/tools/`.
  - It says "No `data/` change while a pinned run is live **or queued**", while root rule 6 says only "live" (F12).
  - KEEP: the tools table; regenerate to a temp dir and diff; the team pool changes only at a retrain boundary; the `promote_teams` second writer; never fix the committed mojibake; obs VALUE fields vs mechanics fields; mechanics come from the resolved `Dex.mod('gen3')`; the `cloud/vast` keys never leave the machine.
- **battle** (5,034 → ≈ 1,900). 14 checks, 0 wrong.
  - KEEP: there is no Python battle layer; `LiveView` / `LegalActions` compute nothing; Struggle is a flag; protocol-revealed facts live on the view; the two faint-cause vocabularies (8 vs 10); `core_corpus_test`.
  - DELETE: the field and symbol enumerations; the schema-is-Rust-owned line, which duplicates rust_sim.
- **tui** (3,697 → ≈ 800). 12 checks, 3 stale.
  - The prober is no longer a consumer; the TUI was retired (F9). The same stale claim is in three code docstrings outside this audit's scope.
  - KEEP: keep the base thin; `CSS_PATH` puts the absolute theme first; an app with a live timer overrides the copy-mode freeze hooks; never shadow `super+c`.
  - Option: fold these four lines into `designs/launcher/tui_and_session.md` and delete the leaf.

### 2.10 `designs/CLAUDE.md` (loaded when touching `designs/`)

- **Accuracy:** 12 checks, 0 wrong.
- **Duplication:** about 40 %.
  - The ALWAYS-CURRENT table restates the root.
  - The "Now:" paragraph is one 1.2 KB sentence restating `endstate/README.md` and the root map.
  - The tree "holds" column restates each tree's index.
  - The `learning/` table (3.1 KB) restates the notes' titles.

| § | bytes | class → bytes |
|---|---:|---|
| title + ALWAYS-CURRENT table | 2,369 | TIGHTEN → 1,000 (keep the ledger-wins rule and "never hand-edit the generated index") |
| A run ≠ the code | 2,186 | TIGHTEN → 800 (pointer to `endstate/README.md` + the backlogs) |
| leaf-detail trees + `ops/` | 3,651 | TIGHTEN → 1,600 |
| cross-version docs | 1,232 | KEEP → 900 |
| version folders | 2,715 | TIGHTEN → 1,000 (the inline findings → `UNDERSTANDING.md`) |
| `learning/` | 3,144 | MOVE → new `designs/learning/README.md`; keep 2 lines |

### 2.11 `MEMORY.md` (user memory index; outside the repo; the orchestrator and the owner own it)

- **When measured:** 2026-10-19 08:44, just after the orchestrator's edit.
- **Stale or contradictory (F13):**
  - Line 26 (quota "until Tue 10-13 13:00") has expired.
  - Line 48 ("no tech debt auto-dispatched mid-week") contradicts line 47, the owner's 10-19 ruling.
  - Line 24 (the plateau-first funnel) is history: its last step, X26, launched 10-16.
- **Duplication with the root:** about 35 %.
  - The "Repo hygiene" section calls itself "the incidents behind" root rules.
  - Lines 41, 43 and 45 restate root § Subagents and every agent definition.
  - Lines 31, 32 and 35 restate the ladder ban, poke-env and the restart interval.
- **Wrong home:** lines 38–45 are how the orchestrator operates. `designs/CLAUDE.md` already says
  such owner rulings belong in the SOP, not in memory.

| section | ≈ bytes | class → bytes |
|---|---:|---|
| header | 450 | TIGHTEN → 250 |
| Owner and how to work with them | 2,100 | KEEP → 1,600 (memory's real job) |
| Roadmap (current) | 3,400 | TIGHTEN → 1,700 (drop expired, history and root duplicates) |
| Sessions, agents, quota | 3,400 | MOVE → `ORCHESTRATOR_SOP.md` §2 / §5 / §7; keep ≈ 900 |
| Repo hygiene | 1,600 | DELETE → 150 (incidents behind root rules; `utils.worktree_guard` enforces the removal check) |
| Measurement and engineering discipline | 1,800 | TIGHTEN → 900 (keep the methodology no doc states) |
| References | 260 | KEEP |

### 2.12 Slash commands (loaded when invoked; each description line in every session)

| file | bytes | class → bytes | why |
|---|---:|---|---|
| `gen3ai-ship.md` | 12,881 | TIGHTEN → 7,000 | stale lines (F15); rationale paragraphs → one clause each; add the replace-don't-append step (§6) |
| `gen3ai-resume-run.md` | 2,897 | DELETE or REWRITE → ≤ 1,000 | hazard (F14) |
| `gen3ai-update-design-docs.md` | 3,865 | RETIRE (owner's call) | targets `designs/ai_vN/impl_step*.md`; none has changed since 2026-07-26 (F16) |
| `gen3ai-learning.md`, `gen3ai-probe.md`, `gen3ai-review-training.md` | 15,289 | KEEP | not audited line by line; **UNVERIFIED:** `review-training`'s TensorBoard tag names |

### 2.13 Subagent definitions (`~/.claude/agents/*.md`; user configuration outside the repo)

There are six files of about 2.8 KB each. Each is a role line, the spawn rule, and the 55-minute
waiting rule. The waiting rule restates root § Subagents almost word for word, and every subagent
also loads the root, so each subagent reads it twice.

Proposed: keep about 0.8 KB of tier-specific text (role, which tiers it may spawn, "report what
needs more depth as a FINDING"), and keep the waiting rule only in the root. The owner or the
orchestrator decides (F18).

---

## 3. Findings

- **F1 — Regrowth.** Two whole-corpus cuts (2026-09-07 and 2026-10-10) each regrew within weeks (§0). Without a ratchet, a third cut will regrow too.
- **F2 — Tests pin doc text.** A sweep that moves the text without moving the test target fails the gate:
  - `src/utils/static_gates_test.py` pins the root's "**Seventeen static gates** run in every tier". It also pins `docs/DEVELOPING.md` and `docs/RUNNING.md`.
  - `src/agents/training/tb_relevance_test.py:342-346` pins "gen3_tb_relevance_v1" and "What to watch on a WIN-PROB run" in the training leaf. `designs/training/telemetry_scalars.md` (lines 13-14, 479) also says the dashboard "stays in" the leaf.
  - `src/main/train/disk_guard_wiring_test.py` pins "FATAL_DISK" and "--allow-low-disk" in the launcher leaf.
  - `src/main/launcher/restart_interval_default_test.py` pins the launcher leaf's `--restart-interval-hours` table row.
- **F3 — Section pointers dangle, and no gate checks them.** The freshness gate checks paths and flags, not "file § section" references. Already dangling:
  - `designs/ops/training_runbook.md:154` points at launcher § "Validating a launch without launching", which does not exist.
  - `training_runbook.md:504` and `designs/training/rust_collector.md:302` point at "A Rust-core run under the launcher", which is gone.
  - `designs/prober/web_pages.md:136` points at "The SPECIES-CLAUSE reading", which is gone.
  - `designs/prober/battle_view_v2.md:91` points at web "Access", which is gone.
  - `src/main/tui/app_base.py:49` points at "Copy mode"; the section is titled "Copying text".
  - `src/main/prober/web/app_test.py:1480` points at a prober gotcha that does not exist.
  - Other references are about to dangle, because the sweep moves their targets: `src/main/prober/web/app.py:106`, `src/agents/model/CLAUDE.md:240`, `training_runbook.md:31`, and three quotes of "start here for 'what next'" (`web/app.py:1069`, `web/app_test.py:675`, `designs/prober/web_pages.md:19`).
  - Every sweep unit greps for references to the sections it moves and repoints them.
- **F4 — Contradiction that needs a ruling before the model / observation unit.** Does a change in the *meaning* of an observation cell bump `ARCH_SIGNATURE`, or only `OBS_SEMANTICS_VERSION`?
  - `ARCH_SIGNATURE`: model leaf line 49, `designs/model/phase_pipeline.md:309`, observation leaf lines 28-30.
  - `OBS_SEMANTICS_VERSION`: model leaf line 210, `designs/model/versioning.md:423-428`, `model_version/constants.py:610-622`. v151 shipped this way.
  - The model owner or the orchestrator decides; an editor must not pick a side.
- **F5 — `gen3_data` leaf: six stale claims** (§2.3). Rebase first, because `f20526571` already rewrote the `GEN3_HP_IVS` line on main. The dex-`category` hazard has no designs-doc home, so it stays in the leaf.
- **F6 — Observation leaf:**
  - `obs_with_event_row` → `obs_with_event`.
  - **The missing `EVENT_COL` / torch.fx hazard.** A consumer that follows the leaf literally breaks code generation.
  - An obsolete poke-env warning.
- **F7 — Training leaf:**
  - The stale `v144+` (the floor is 145).
  - The `labels/margin.rs` location.
  - Volatile run state inside the leaf: the X26 headroom watch, "until the follow-up fix lands", "ADOPTED 2026-10-14".
- **F8 — Model leaf:**
  - The false "Moved from the leaf" claim.
  - The root calls the extractor output `(pi_features, vf_features)`; the leaf calls it `(pi_features, value_pooled)`. It is the same object, but one name should be used.
  - Five rule groups are duplicated across the training, model and observation leaves: `belief_supervision`, the critic, ride-along heads, the two move orders and mypy. Each needs one home.
- **F9 — tui leaf:**
  - It says the prober uses `main.tui` (in three places), but the prober's TUI is retired.
  - The same stale claim is in `src/main/tui/__init__.py:1`, `app_base.py:3` and `colors.py:4`. These are code docstrings; fixing them is a code-file edit, so it needs its own approval in phase 2.
- **F10 — Web leaf, false premise.** The section "Commands a template prints (kept here so the freshness gate resolves their flags)" assumes the gate checks the template. The gate reads only `CLAUDE.md` files, so `templates/index.html:130-138` is unguarded. A real guard is a web test that resolves each printed command against its parser; that is a build decision.
- **F11 — rust_sim leaf:**
  - Quoted census figures contradict its own never-quote rule.
  - The `display_name` location is wrong.
  - `src/utils/bridge/README.md:326-336` keeps a second, dated copy of the census, against principle 4.
- **F12 — Two homes for one rule.** The tools leaf says no `data/` change while a pinned run is "live **or queued**"; root rule 6 says "live". The root should hold the rule, and someone must decide whether "queued" is part of it.
- **F13 — `MEMORY.md`:**
  - The expired quota line.
  - The tech-debt contradiction.
  - Funnel history.
  - About 35 % duplicates the root.
  - Session-operation rulings sit in memory instead of the SOP (§2.11).
- **F14 — `/gen3ai-resume-run` is hazardous and obsolete.**
  - It runs `git checkout <branch>` in the **main checkout**, which must never be dirty.
  - It invokes the trainer script directly, bypassing the launcher's pin and torch checks.
  - It passes `--lr` / `--batch-size` / `--n-steps`, which are inert on a resume.
  - It suggests defaults (96 envs, 16,384 batch, …) that the recipe inheritance overrides.
  - Its description appears in every session's skill list.
  - Replace it with `python -m main.checkargs` plus `python -m main.launcher --dry-run --model <ckpt>`, or delete it.
- **F15 — `/gen3ai-ship` is stale:**
  - "PLUS the ten static gates": there are 17, and the static tier replaces hand-picking them.
  - "`-n 4` beside a live training run": the root says `nice -n 19 taskset -c 5-7,13-15 … -n 3`.
  - Step 1c runs mypy and ruff by hand, which the static tier now covers.
- **F16 — `/gen3ai-update-design-docs` is dormant.** No `impl_step` doc has changed since 2026-07-26, and current design work is in always-current `endstate/` docs.
- **F17 — `MEMORY.md` reaches subagents, contrary to the docs.** The Claude Code docs say auto memory is not loaded into (non-fork) subagents. The audit agent, an `opus-high` subagent, received the full `MEMORY.md`. The mechanism is unverified, so treat the file as loaded everywhere.
- **F18 — Agent definitions duplicate the root's waiting rule** (§2.13). They are outside the repo, so no gate sees them.
- **F19 — Emphasis.** 199 🚨 across the 13 `CLAUDE.md` files (cap: 5 per file, 65 in all).
- **F20 — Prober hazards about pre-era traces cannot fire** on any trace that still exists (see §2.5). Read-only check of `models/`: 93 `rb_` runs have traces, no pre-`rb_` run does, and all 97 `rb_` configs are `critic: winprob`.
- **F21 — Minor.** The docstrings of `src/claude_md_freshness_gate_test.py` and `src/file_size_gate_test.py` still say they sit beside the deleted `strict_api_lock_test.py`.
- **F22 — Not verified:**
  - Token counts (no tokenizer was run).
  - The rust_sim census figures; rerunning them needs a cargo build.
  - The measurement claims the tables delete anyway: "~6.5×", "73.3 %", "five of five", "96 % / 13 h".
  - The "28-tag" dashboard count; some tags are f-string-composed.
  - `review-training`'s tag names.

---

## 4. The sweep plan (phase 2)

**Preconditions:**
- The three build agents and the test-gate agent have shipped.
- Each unit rebases and re-measures before it edits.
- The F4 ruling exists before U5.
- `data/` stays frozen.
- These are doc units; the only code they touch is a test's target string (F2), and that change ships in the same unit.

**Every unit's guardrails:**
1. **Nothing load-bearing is lost.** Every deleted rule carries a disposition: `OBSOLETE (why)`,
   `MOVED → path § section`, or `ENFORCED BY gate / guard`. The file's "must survive" list (§2) is
   ticked line by line.
2. **One home.** A MOVE whose destination already holds the text is a delete. Before deleting,
   the editor confirms the destination has it.
3. **References.** Grep the repo for references to every section moved or renamed, and repoint
   them (F3). Move the targets of the tests that pin doc text in the same commit (F2).
4. **Review.** Before / after bytes in the commit body. A fresh-context reviewer (`sonnet-high`)
   reads the diff against the disposition list and the must-survive list.
5. **Gates.** The static tier is green, including the freshness gate and the budget gate, plus the
   tests that pin the unit's files. `deleted_flags.md` entries stay; a doc stops naming a deleted
   flag rather than adding an entry for it.
6. **Fix stale claims, don't carry them.** A wrong claim found in the audit is fixed or deleted,
   never moved.

**Editors:** `opus-high` (judging what is load-bearing is the hard part). **Parallelism:** U3, U4, U6
and U7 touch disjoint leaves and disjoint `designs/` trees, so two or three can run at once.

| unit | scope | saves | notes |
|---|---|---:|---|
| **U1** | the ratchet: a byte budget per instruction file, **folded into `src/claude_md_freshness_gate_test.py`** (so the static-gate count, pinned in the root, does not change), plus `python -m <module>` targets must resolve | 0 (stops regrowth) | caps from principles §3; files over cap grandfathered at today's bytes, may not grow, must leave the list when under cap (the `file_size_gate_test.py` pattern); a census prints the target band. Extend the scan to `.claude/commands/*.md`. Agent: `opus-medium` |
| **U2** | slash commands: delete or rewrite `/gen3ai-resume-run` (F14); fix `/gen3ai-ship` (F15) and add the replace-don't-append step (§6); retire `/gen3ai-update-design-docs` (F16, owner's call) | ≈ 12 KB on invocation; 2 description lines in every session | first, because F14 is a live hazard |
| **U3** | training leaf | ≈ 26 KB | move the `tb_relevance_test` anchors and the two `telemetry_scalars.md` notes together (F2); fix F7; one home for `belief_supervision` |
| **U4** | `gen3_data` + tools + battle | ≈ 27 KB (+ 5 KB moved) | rebase over `f20526571`; new `designs/tools/data_files.md`; fix F5; collapse the forme rule to one home; update `designs/tools/README.md` and `tools/CLAUDE.md:9` (who owns the schemas); F12 for the root unit |
| **U5** | model + observation | ≈ 19 KB | after the F4 ruling; fix F6 (the `EVENT_COL` hazard **stays**, now stated correctly) and F8; the move orders and mypy get one home (model) |
| **U6** | prober + web + tui | ≈ 26 KB | repoint F3's prober / web / tui references; drop F20's dead cases; F10 noted for a build decision; F9's code docstrings only with approval |
| **U7** | rust_sim + launcher | ≈ 24 KB | move the launcher test anchors (F2); repoint `training_runbook.md:31` / `:154` / `:504` and `rust_collector.md:302`; F11 (point the bridge README at the tool) |
| **U8** | `designs/CLAUDE.md` | ≈ 9.7 KB | after U3–U7, since it indexes the trees; new `designs/learning/README.md` |
| **U9** | `MEMORY.md` + the six agent definitions | ≈ 6.8 KB + ≈ 12 KB | the orchestrator's unit (it owns the memory and the agent types); first write session-operation rulings into `ORCHESTRATOR_SOP.md`; fix F13; F18 is the owner's or orchestrator's call |
| **U10** | **root `CLAUDE.md`, last** | ≈ 25.6 KB | keep the test-pinned count sentence; absorb the leaf table into the map; settle F12; one pointer to the principles; re-read every leaf's must-survive list for rules that now live only in the root |
| **U11** | tighten the ratchet | — | lower each grandfathered cap to the type cap; the grandfather list ends empty |

**Total saved:** about 158 KB of `CLAUDE.md` (≈ 40k tokens) and about 7 KB of `MEMORY.md`. The
always-loaded share falls by about 34 KB, roughly 8.6k tokens, in every session and every
subagent.

---

## 5. Keeping the docs lean afterwards

1. **The budget ratchet (U1, U11)** is the mechanism. The two earlier cuts had none, and both
   regrew.
2. **Replace, don't append, at ship time.** Add a step to `/gen3ai-ship`: if the diff grows any
   `CLAUDE.md`, slash command or `MEMORY.md`, the commit body names what each addition replaces or
   retires, or why the budget has room. The budget gate backs this up.
3. **One line in the root's § Documentation Maintenance:** apply the deletion test and point at the
   principles. This lands in U10.
4. **A section-reference check (backlog).** A `path § heading` reference into a `CLAUDE.md` or
   `designs/` doc must resolve. This prevents F3. It is a build unit, not part of the sweep.
5. **Revisit after each major model release**, as Anthropic advises (principles §6, [CC-LARGE]). Claude Code's
   `/doctor prompt-audit` (≥ 2.1.283; this box runs 2.1.292) finds outdated or conflicting
   instructions and missing references, and is a cheap owner-run second opinion.
6. **`MEMORY.md` size.** The platform drops everything past 25 KB, and the owner's ceiling is 10 KB.
   No repo gate can see the file, so the orchestrator's :07 / :37 health check could report its
   size when it is over 10 KB. A user-settings hook is the alternative; that is the owner's call.
7. **Briefs** already must not restate standing rules; the same discipline applies to their length.
