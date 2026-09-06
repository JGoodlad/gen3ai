# CLAUDE.md census — 2026-09-06

**What this is.** A section-level inventory of the five largest `CLAUDE.md` files, a proposed target
structure for the root and for the training leaf, and the context-cost arithmetic behind both. It is
a **census and a proposal, not a restructure** — six agents were editing these files while it was
taken, so nothing was moved or deleted. The only edits landed in the same pass are two corrected
paths (below) and the new freshness gate.

**Companion deliverable:** `src/claude_md_freshness_gate_test.py` — the mechanical half. Every
repo-relative path and every `--flag` a `CLAUDE.md` names must resolve, or be listed in
`designs/deleted_flags.md` with a citation. That gate is what stops the *next* two years of drift;
this document is about the accumulated *volume*.

---

## 0. The rule I used — accept it, amend it, or replace it

> **A `CLAUDE.md` earns a line only if an agent that has NOT read it would do the work WRONG.**
> Four things pass that test: a **RULE** (a standing instruction — "never `git commit` without
> `/gen3ai-ship`"), a **COMMAND/GATE** (the invocation or the check, because guessing it costs a
> cycle), a **MAP** (where a thing lives, because searching for it costs a cycle), and a **HAZARD**
> (a footgun that has already fired, *with just enough evidence to be believed* — the evidence is
> what stops the next agent re-deriving the trap). Everything else — how we found out, what we
> measured, which arm won, what the number was in August — is **NARRATIVE**. Narrative is valuable
> and must be kept; it belongs in `designs/research_state/`, `designs/CHANGELOG.md`, or a learning
> note, where it is read **on purpose** rather than **on every turn**.
>
> The corollary that does most of the work: **a HAZARD entry needs the lesson, not the
> investigation.** "The seedless bridge path replayed one dice stream because every gate on it was
> seeded — when a default branch has no test, it is untested however green the suite looks" is a
> hazard in two lines. The 40 lines of how it was found are narrative.

Two secondary rules follow from the same test:

* **A leaf owns its subject; the root owns the pointer.** Where the root and a leaf both describe
  the bridge, the compile flags, ELO or `checkargs`, the root should carry the one-sentence
  operational fact and the flag name, and the leaf everything else. Today the root carries 200-line
  versions of sections whose leaf it explicitly points to in the same paragraph.
* **A measured number in a rules file is a liability with a half-life.** It goes stale silently and
  it is quoted as current. If the number is load-bearing, it belongs beside the measurement that
  produced it (with its date and provenance), and the rules file cites that file.

---

## 1. Headline — the whole corpus

Token figures are estimates at **3.6 bytes/token** (markdown-heavy technical English with dense code
identifiers; no tokenizer was available offline). Read them as ±8%.

| file | lines | bytes | ~tokens | loaded |
|---|---:|---:|---:|---|
| **`CLAUDE.md`** (root) | **2,275** | **174 KB** | **~48,000** | **EVERY session** |
| `src/rust_sim/CLAUDE.md` | 8,436 | 831 KB | ~231,000 | working in the port |
| `src/agents/training/CLAUDE.md` | 7,503 | 616 KB | ~171,000 | working in training |
| `src/main/prober/CLAUDE.md` | 1,503 | 127 KB | ~35,000 | working in the prober |
| `src/agents/model/CLAUDE.md` | 1,239 | 106 KB | ~29,000 | working in the model |
| `designs/CLAUDE.md` | 411 | 79 KB | ~22,000 | working in designs |
| `src/agents/observation/CLAUDE.md` | 741 | 55 KB | ~15,000 | |
| `src/main/prober/web/CLAUDE.md` | 734 | 53 KB | ~15,000 | |
| `src/main/launcher/CLAUDE.md` | 663 | 57 KB | ~16,000 | |
| `src/agents/battle/CLAUDE.md` | 306 | 26 KB | ~7,000 | |
| `tools/CLAUDE.md` | 284 | 25 KB | ~7,000 | |
| `src/agents/gen3_data/CLAUDE.md` | 120 | 8 KB | ~2,000 | |
| `src/main/tui/CLAUDE.md` | 64 | 4 KB | ~1,000 | |
| **total** | **24,279** | **2.16 MB** | **~600,000** | |

**The distribution is the finding.** Two files are 66% of the corpus, and one file — the root — is
paid for on **every turn of every session** whether or not the session touches any of it. A session
that opens the root and the training leaf starts ~219,000 tokens down.

⚠️ **`designs/CLAUDE.md` deserves a line of its own: 411 lines but 79 KB — 191 bytes/line, 2.5× the
root's density.** Two table cells in its version map are ~5,000 words each (the "Code on main" cell
is one unbroken paragraph running v100 back to v51). It is the densest document in the tree and it
is not on this census's five, but it should be on the next one.

---

## 2. Root `CLAUDE.md` — 2,275 lines, 20 sections

Dominant class per section, with an estimate of the narrative *embedded inside* a section whose
dominant class is something else. That second column is where the volume actually is.

| § | section | lines | class | embedded NARR | recommendation |
|---|---|---:|---|---:|---|
| — | header | 2 | — | 0 | keep |
| 1 | Development Stage | 8 | RULE | 0 | **keep verbatim** |
| 2 | 🚨 running subagents / Workflows | 33 | HAZARD | 6 | keep; trim the binary-string-inspection detail → ~25 |
| 3 | Documentation Maintenance | 33 | RULE + MAP | 0 | **keep verbatim** — this is the file's constitution |
| 4 | Git Workflow | 16 | RULE | 0 | **keep verbatim** |
| 5 | Python Environment | 21 | RULE | 4 | trim → ~14 |
| 5a | ├ `export PYTHONPATH` is OPTIONAL… | 67 | RULE | **46** | keep the worktree rule (~14); the `.pth`-ordering archaeology and the `environment.yml` history → engineering note |
| 6 | Git Worktree Setup | 46 | COMMAND | 18 | trim → ~22; `bootstrap.sh` + the conftest guard already do this. Keep the `dist/dist` HAZARD as 4 lines |
| 7 | Running Tests | 455 | mixed | **~200** | **the second-largest block; → ~130** (see 2.1) |
| 8 | Smoke Test | 27 | COMMAND | 0 | keep |
| 9 | Launcher | 224 | MAP | **~95** | → ~55 (see 2.2) |
| 10 | Training | 533 | mixed | **~280** | **the largest block; → ~110** (see 2.3) |
| 11 | Playing / LADDER | 52 | COMMAND + RULE | 8 | trim → ~32 |
| 12 | Prober | 70 | MAP | 10 | **DUPLICATE** of the prober leaf → ~18 |
| 12a | ├ Web front end | 56 | MAP | 12 | **DUPLICATE** of `prober/web/CLAUDE.md` → ~8 |
| 13 | Showdown Server | 57 | RULE + HAZARD | 0 | **keep the :8001 rule verbatim**; trim the port prose → ~34 |
| 14 | Repository Structure | 269 | MAP | **~140** | → ~95 (see 2.4) |
| 14a | ├ Path discovery (`paths.py`) | 38 | RULE | 6 | keep → ~30 |
| 15 | Observation Vector | 52 | MAP | 27 | keep the table + never-hardcode rule; the two `gen3_*_v1` paragraphs are already verbatim in CHANGELOG → ~24 |
| 16 | Feature Extractor Architecture | 33 | MAP | 4 | keep |
| 17 | Model Versioning | 100 | RULE | 40 | 30 lines DUPLICATE the model leaf's 170-line § → ~38 |
| 18 | Data Dependencies | 68 | MAP | 20 | the per-file schema belongs in `gen3_data/CLAUDE.md` → ~28 |
| 19 | Event-Sourced Battle Layer | 15 | MAP | 0 | **keep verbatim** |

### Per-class line totals — root

| class | lines | share |
|---|---:|---:|
| **NARRATIVE** (dominant + embedded) | **~735** | **32%** |
| MAP | ~590 | 26% |
| COMMAND/GATE | ~410 | 18% |
| RULE | ~330 | 15% |
| DUPLICATE (of a leaf that says it better) | ~145 | 6% |
| HAZARD | ~65 | 3% |

**Read that as: ~38% of the root is narrative or duplication.** RULE — the only class that is
strictly irreplaceable at the root — is **15%, about 330 lines.**

### 2.1 § Running Tests (455)

| sub-section | lines | class | note |
|---|---:|---|---|
| Running beside a live training run | 72 | RULE + NARRATIVE | the RULE is ~10 lines ("a timeout is never a semantic outcome"; benchmarks warn, never stretch). The other ~60 is the incident record — the 39/40-bogus-skips story, the load-22 measurements, the 0-vs-6 scaling table |
| Test tiers — TWO AXES | 68 | RULE + NARRATIVE | the two-axis rule + the marker table is ~25; the "MEASURE BEFORE YOU TIER" evidence and the 1.24-band story are ~43 |
| Test file naming conventions | 11 | MAP | keep |
| Which command to run | 20 | COMMAND | **keep verbatim — this is the most-used table in the file** |
| The two STATIC gates | 36 | COMMAND/GATE | keep ~22; the ruff-findings archaeology is narrative |
| The FILE-SIZE ratchet | 50 | COMMAND/GATE | keep ~20; the five-entry paid-off table and the 🎉 celebration are a *closed* history |
| Unit tests only / `-n 2` | 36 | COMMAND | keep ~12; the idle-vs-loaded timing tables are narrative with a 2026-08-14 date |
| Everything incl. slow tiers | 16 | COMMAND + HAZARD | keep |
| Fuzz tests | 23 | COMMAND + MAP | keep — the per-fuzz inventory is a genuine map |
| E2E tests | 7 | COMMAND | keep |
| Benchmarks | 61 | COMMAND + NARRATIVE | the invocations are ~18; the baselines, the "reward is NOT negligible" correction and the memoized-value story are ~43 |
| What "fuzz test" means | 53 | RULE + HAZARD | **keep ~40** — the reproducibility rule and the concurrency clause are binding and have cost real arms |

### 2.2 § Launcher (224)

`### Will this command still launch? — python -m main.checkargs` is **120 lines**, of which ~25 are
the command and its contract and **~95 are four dated incident narratives** (the "EVERY IS ENFORCED"
G5 story, "AN ARGV IS NOT A CONFIG" / C1, the pinned-parser arity defect, the relative-`models/`
worktree defect). Every one is a genuine finding; none of them binds an agent's behaviour beyond
*"run `checkargs`, and `--dry-run` on a restart."* `src/main/launcher/CLAUDE.md` already owns the
detail.

### 2.3 § Training (533) — the largest block

| sub-section | lines | class | note |
|---|---:|---|---|
| intro + `--grad-accum-steps` | 45 | RULE + COMMAND | keep ~24 |
| `--critic {shaped,winprob}` | 43 | RULE | **DUPLICATE** — `model/CLAUDE.md` § *The CRITIC MODE* is 39 lines on the same flag. Root keeps ~8 |
| In-process bridge transport | **225** | NARRATIVE | ~30 lines are the live rule (the three values, `rust` is the default, why). The rest is the seed-defect history *explicitly marked FIXED and "kept as history"*, the two CHOOSE-path incidents, and the coverage census — all also in `src/utils/bridge/README.md` and the rust_sim leaf |
| The two compile flags | 26 | RULE + MAP | keep — the flag/default/opt-out table is the useful artifact |
| Compiled CPU opponents | 38 | NARRATIVE | ends with "Full detail … is in `src/agents/training/CLAUDE.md`" — so this **is** the detail, duplicated. Root keeps ~6 |
| Compiled GPU trainer | 30 | NARRATIVE | same shape, same pointer. Root keeps ~8 |
| `--async-rollout` | 15 | MAP | the training leaf's own §16 is flagged DUPLICATE *of this*; keep it here, delete there |
| Bot evaluation | 20 | MAP | keep ~10 |
| The UNTAUGHT METER | 30 | RULE + MAP | keep ~14 — the control-arm rule is binding |
| ELO / skill rating | 25 | MAP + RULE | keep the three reading rules (~10) |
| Exploitability | 36 | MAP + COMMAND | keep ~12 |

### 2.4 § Repository Structure (269)

A tree diagram that has grown per-module essays inside its comment column. `search_dividend/` is
**55 lines**, `untaught_meter.py` 20, `scaffolding_gauge.py` 18, `promote_teams.py` 11,
`instrumented_ppo/` 25. Those are leaf content living in the root's map. A tree whose comments are
one line each lands at ~95 lines and is *more* useful as a map, not less.

### 2.5 STALE in the root — verified

1. **FIXED THIS PASS.** L2178 named `src/agents/model/model_version.py`; it became the package
   `model_version/` on 2026-08-23 — as the same file says 1,700 lines earlier at L470. Corrected to
   `src/agents/model/model_version/`. (Same defect at `src/agents/observation/CLAUDE.md:11`, also
   corrected.) Both were found by the new gate on its first run.
2. **`deps/venv` "exists but is outdated — ignore it"** (L~106) is true in the main checkout and
   false in every worktree, where the directory does not exist. Not corrected — the sentence's
   *advice* is right either way, and `deps/` is deliberately outside the gate's scope. Flagging it
   because "ignore the venv that isn't there" is a sentence that costs a reader a `ls`.
3. **Test counts carry a 2026-08-23 date and durations a 2026-08-14 one**, and the file says so
   loudly and correctly. Not stale — *correctly dated*. This is the pattern the rest of the file
   should follow, and mostly does.

---

## 3. `src/agents/training/CLAUDE.md` — 7,503 lines, 50 sections

Full section table in the appendix of the sub-agent report; the headline:

| class | lines | share |
|---|---:|---:|
| RULE | 4,990 | 66.5% |
| MAP | 1,917 | 25.5% |
| NARRATIVE (whole sections) | 277 | 3.7% |
| COMMAND/GATE | 165 | 2.2% |
| DUPLICATE | 106 | 1.4% |
| HAZARD | 49 | 0.7% |

🚨 **The 66% RULE figure needs its caveat or it will be read backwards.** Classifying by *dominant*
content puts two-thirds of the file in RULE, but the narrative here is not in narrative sections —
it is **embedded inside** the RULE and MAP sections as dated measurement paragraphs. Measured proxy:
**98 lines carry an explicit ISO date** and **238 lines carry a measurement/ledger/probe anchor
word**, each typically heading a 5–8 line paragraph ⇒ **~1,200–1,500 lines (16–20%) of embedded
narrative**, concentrated in the counterfactual-grounding section (38 anchors), distillation (20),
stall-tail harvest (18), reward (16), compile (15), bot eval (14). That is roughly the difference
between a 7,500-line file and a 6,000-line one, and it is the pool a `research_state/` extraction
should come from.

The two structural outliers: **`--cf-*` counterfactual grounding is 1,063 lines** and
**exploiter distillation is 734** — each larger than the entire prober leaf's analysis section.

### Split plan — training

A `CLAUDE.md` auto-loads only for files in **its own directory**, so a sub-leaf works only where a
real subpackage exists. Existing subpackages: `eval_sharding/`, `instrumented_ppo/`,
`poke_env_gaps/`, `teacher/`, `team_completion/`.

| proposed file | real dir? | source sections | est. lines |
|---|---|---|---:|
| `src/agents/training/CLAUDE.md` (hub) | yes | preamble, MatchupSpec, faint attribution, *WHICH FILE a run spec names*, the belief-target reading rule, `stats.py`, watchdog, port threading, *Reading an ELO* lifted up, + a leaf map | ~400 |
| `training/instrumented_ppo/CLAUDE.md` | yes | fold order, grad balance, `--critic`, PopArt, tail-weight, TD-aux, grad accumulation + noise scale | ~590 |
| `training/eval_sharding/CLAUDE.md` | yes | bot evaluation | ~360 |
| `training/teacher/CLAUDE.md` | yes | search-as-teacher | ~237 |
| `training/cf/CLAUDE.md` | **create the package** — `cf_audit`/`cf_label_buffer`/`cf_mc_return`/`cf_producer`/`cf_q_labels`/`cf_records`/`cf_terms.py` are already one unit | cf tap, buffer, loss, likelihood, evidential head, twin+shadow, Q head | ~560 |
| `training/distill/CLAUDE.md` | **create the package** — `distill_spec`/`distill_anchor_callback`/`distill_stop_callback.py` | anchor, reference modes, `grad_project`, stop rule + dual, target/gate + rank tripwire | ~560 |
| `designs/training/cf_producer.md` | no | producer driver, duty cycle, throughput, forfeit class | ~440 |
| `designs/training/telemetry_scalars.md` | no | TB census, capacity telemetry, `signal/`, scaffolding gauge | ~545 |
| `designs/training/reward.md` | no | reward redesign, defensive entropy, bait entropy | ~500 |
| `designs/training/belief_and_value_heads.md` | no | the six belief losses, opp-class weight, win-prob PBRS, value-dist | ~529 |
| `designs/training/self_play_and_pool.md` | no | self-play, stable opponents | ~473 |
| `designs/training/compile_flags.md` | no | CPU opponents, GPU trainer (trimmed of dated measurement) | ~450 |
| `designs/training/exploiter_and_team_curriculum.md` | no | exploiter mode, team PFSP, per-team WR | ~378 |
| `designs/training/offline_meters.md` | no | untaught meter, cf_audit, replay imputation, prefix sharing | ~271 |
| `designs/training/elo_and_rating.md` | no | ELO minus the reading rules | ~170 |
| `designs/training/step_size_and_dose.md` | no | dose / `--fork-lr` / `--adaptive-batch` | ~150 |
| `designs/training/stall_tail_harvest.md` | no | the harvest pipeline (trimmed) | ~120 |
| `designs/CHANGELOG.md` (append) | — | *Latent-belief loss — DELETED (v75)*, *V_pub — DELETED* | −35 from the leaf |
| *(delete, 1-line pointers)* | — | async rollout (root owns it), LINEAGE (root owns it) | −106 |

Every proposed file is ≤ ~600 lines; the whole set is ~6,000 lines across 19 documents against
7,503 in one, and **the auto-loaded cost of "working in training" drops from ~171,000 tokens to
~11,000** (the hub alone), with the subsystem leaf loading only when you are in that subpackage.

**Making `cf/` and `distill/` real packages is the load-bearing half.** Without it, the two largest
clusters become `designs/` documents nobody auto-loads — the same decomposition the file-size
ratchet already applied to `instrumented_ppo/`, `main/train/` and `main/prober/`.

### STALE in the training leaf — verified

1. **`--eval-workers` default is 5, not 3.** L1174 says "default 3"; `main/train/parser/eval_subprocess.py:18`
   is `default=5`, and this leaf's *own* flag table at L1347 says 5. Self-contradicting inside one file.
2. **`train()` is 1,811 lines, not "~1,250"** (L216). AST-measured over `instrumented_ppo/ppo.py`
   (185→1995). The figure is the stated justification for not splitting `train()`.
3. **The `instrumented_ppo/` module table (L205–214) lists 8 of 14 modules** — missing
   `calibration`, `capacity_terms`, `distill_anchor`, `distill_grad_project`, `signal_metrics`, four
   of which this same file documents at length elsewhere.
4. **Four sections cite `instrumented_ppo.py`, a file that does not exist** (L3288, 4422, 4483, 4718)
   — it became a package on 2026-08-23, which §2 of the same file states correctly.
5. **`ppo.py` is 1,995 lines** against a "~1.8k under a hard 2,000 gate" claim (L6900). The real
   margin is 5 lines; anyone adding a seam on that sentence trips `file_size_gate_test`.
6. **`reward_composition.py` is absent from the reward section's own "Where the reward lives" list**
   (L253–258), though the root documents it.
7. **`training/team_completion/` (3 modules) is mentioned zero times** anywhere in the leaf.

---

## 4. `src/rust_sim/CLAUDE.md` — 8,436 lines, ~50 sections

| class | lines | share |
|---|---:|---:|
| **NARRATIVE** | **6,241** | **74.0%** |
| COMMAND/GATE | 1,515 | 18.0% |
| MAP | 570 | 6.8% |
| RULE | 87 | 1.0% |
| HAZARD | 23 | 0.3% |
| DUPLICATE | 0 | — |

**Three-quarters of the largest file in the tree is dated coverage history**, and it is structurally
identifiable: **3,265 lines sit under 45 explicit `### ROUND N` headings** (3 inside the A/B fuzzer,
32 inside the bridge/request fuzzer, 11 under *Next steps*), plus ~2,976 more of the same shape that
predates the ROUND convention (the per-mechanic gate writeups, the eight BATCH sections, the
regression bug→pin map, the data-driven Phase writeups).

**The binding content inside those 6,241 lines is ~40 lines, in five passages** (found by grepping
for imperative/lesson language, since nothing marks them):

* a new mechanic can **falsify an old "no draw here" proof** — when a class gains a second member,
  re-read every "this can never tie" argument (L5478);
* a determinism-oriented suite **systematically under-tests the nondeterministic default** — at
  least one gate must run with the reproducibility knob OFF and assert a *distributional* property
  (L6119);
* **a gate that exempts a file by basename exempts every file with that name** — compare the
  relative path (L8379);
* `ab_fuzz_out*` run dirs are gitignored: **never commit run output** (L3912);
* a gen3ou repro **must** be replayed with `{format:'gen3ou', allowHiddenPower:true}` (L4018).

Recommendation: lift those five into a ~40-line *"Standing lessons from the coverage rounds"* block
beside § Conventions, move all 45 ROUNDs to `designs/research_state/rust_sim_coverage_rounds.md`,
and keep the **A1/E1 allowlist narrowing clauses** (L3914–4132) in the leaf — those are not history,
they are the live definition of when the green gate may pass. Result: **~2,200–2,600 lines**, a 70%
cut with nothing binding lost.

**Rename `## Next steps (engine, not yet built)` in the same pass** — it is 874 lines of *completed*
work under a title claiming the opposite, and 587 of those lines are ROUNDs 45–55. It is the most
misleading heading in the corpus.

### STALE in the rust_sim leaf — verified

1. **The "STILL NEEDED before this can replace node in `better_line`" block (~L310–314) is wrong on
   all three claims.** `SearchSession` already has the impl switch (`search_session.py:126,134`);
   `search_clone_parity_fuzz_test` documents `--impl rust` in its own header; and the `input_log`
   blocker for `--search-teacher` is recorded as **FALSE** in `main/train/config.py:982–988` with
   the guard removed. This is exactly the class the file elsewhere polices — *a note that outlived
   its own fix* — and it is the single highest-value correction in the census.
2. **`pool 722/722 fully engine-playable` (~L34) is 40 teams stale** — `node harness/scan_move_coverage.js`
   returns **762/762**. The invariant holds; the count drifted as the pool grew to 813 `.txt` files.
3. **The e2e capstone's "loads all 770 `data/teams/*.txt` … → 719 valid" (~L2854)** is a
   generation-time figure in the present tense; the harness globs and the tree now holds 813 files.
   The committed 220-battle golden is fine; the prose describing what the generator *does* is not.

Verified **not** stale (do not re-audit): the 369→309/60/0 move census reproduces exactly; the
220-battle e2e golden; `writeline_test.rs` 44 battles / 2,377 writes; the 1,075-row handler audit;
the `turn/` split; and **every** `tests/*.rs`, `harness/*` and `src/**/*.rs` path named in the file
exists on disk — zero dangling references.

---

## 5. `src/main/prober/CLAUDE.md` — 1,503 lines, 15 sections

| section | lines | class | recommendation |
|---|---:|---|---|
| header + invocation | 44 | COMMAND + MAP | keep |
| Engine / app split | 83 | **RULE** | **keep verbatim** — this is the seam the whole package rests on |
| ⚠ Architecture drift | 45 | HAZARD | keep ~30; the 79-of-79 measurement is dated and belongs beside it |
| Per-battle model resolution | 22 | RULE | keep |
| What one decision's analysis CONTAINS | 141 | MAP | keep ~70 |
| ├ The SPECIES-CLAUSE reading | 58 | HAZARD | keep — a genuine "what this is NOT" |
| ├ Beliefs / Threats | 203 | MAP + NARRATIVE | → ~90; the per-panel provenance is narrative |
| **Agent API & JSON CLI** | **622** | MAP + COMMAND | **→ ~250** — a per-subcommand reference that has grown design rationale; the rationale → `designs/` |
| Obs-offset dependence | 20 | HAZARD | keep |
| Blocking work | 10 | RULE | keep |
| The counterfactual tier | 24 | MAP | keep |
| `--compile` | 17 | COMMAND | keep |
| `--impl {node,rust}` | 36 | COMMAND | keep ~20 |
| Gotchas | 79 | HAZARD | **keep verbatim** — this is the highest-density-per-line section in the file |
| Retention / grooming | 22 | COMMAND | keep |
| Tests | 52 | COMMAND/GATE | keep |
| Web front end | 25 | MAP | keep — correctly a pointer to its own leaf |

| class | lines | share |
|---|---:|---:|
| MAP | ~700 | 47% |
| COMMAND/GATE | ~330 | 22% |
| NARRATIVE | ~230 | 15% |
| HAZARD | ~140 | 9% |
| RULE | ~103 | 7% |

**Verdict: this file is basically healthy** — it is a reference manual for two surfaces over one
engine, and a reference manual is legitimately long. The one disproportion is the 622-line CLI
section, which mixes *how to invoke* with *why the analysis is shaped that way*. Target ~1,000
lines. No stale paths or flags: the gate found none here beyond the `--damage-refine-rounds`
demotion (§7).

## 6. `src/agents/model/CLAUDE.md` — 1,239 lines, 15 sections

| section | lines | class | recommendation |
|---|---:|---|---|
| Architecture constants | 22 | **RULE** | **keep verbatim** |
| Phase module structure | 140 | MAP | keep ~90 |
| ├ `QWinProbHead` | 44 | MAP | keep |
| ├ `--belief-grad-mode` | 60 | RULE + NARRATIVE | keep ~35 |
| ├ Phase-by-phase data flow | 104 | MAP | **keep verbatim** — the contract |
| File layout | 68 | MAP | keep |
| ├ The extractor CLASS is a base-class CHAIN | 69 | MAP + NARRATIVE | keep ~40 |
| The op's flat layout has ONE slicer | 13 | RULE | keep |
| The op's SIDE VALUES have ONE container | 34 | RULE | keep ~20 |
| ⚠️ One op's SPELLING is load-bearing | 50 | **HAZARD** | **keep** — unguessable, and it has fired |
| ⚠️ Identity-at-init is NOT free | 28 | HAZARD | keep |
| The flag registry | 35 + 78 in 3 subs | RULE | keep — the tier/class/requires rules are binding |
| Model versioning | 170 | RULE + NARRATIVE | → ~110; ~30 lines DUPLICATE the root's § Model Versioning |
| The CRITIC MODE | 39 | RULE | **keep — and let the ROOT shrink to a pointer at this** |
| PopArt | 36 | RULE | keep |
| Where the canonical architecture lives | 21 | MAP | keep |
| Opponent intent α/β | 31 + 144 in 2 subs | RULE + HAZARD | keep — the `_pool`-suffix reading rule and the consumer contract are both footguns |
| Static typing (mypy) | 51 | COMMAND/GATE | keep ~35 |

| class | lines | share |
|---|---:|---:|
| RULE | ~530 | 43% |
| MAP | ~380 | 31% |
| HAZARD | ~150 | 12% |
| NARRATIVE | ~120 | 10% |
| COMMAND/GATE | ~59 | 5% |

**Verdict: the healthiest of the five.** 43% RULE and 12% HAZARD is exactly the profile the rule in
§0 asks for, and almost every hazard here is unguessable from the code (the `torch.compile`
spelling, the SB3 ortho-init clobber, the `_pool`-suffix key). Target ~1,000 lines; the only real
saving is the ~30-line overlap with the root, and **the right fix is to shrink the ROOT**, not this.

---

## 7. Proposed target structure for the root — ≤ 400 lines

Ordered so a reader hits the binding material first and the maps last. Line budgets are targets, not
bounds.

```
CLAUDE.md — Gen3AI Project Guide                                        ~395 lines
├─ 1  Development Stage                                          8   RULE   verbatim
├─ 2  Git Workflow (worktree only; only /gen3ai-ship commits)   16   RULE   verbatim
├─ 3  Documentation Maintenance + the LEAF MAP table            36   RULE   verbatim (+3 new leaves)
├─ 4  🚨 STANDING HAZARDS  (one screen, one line each)          34   HAZARD  NEW — see below
├─ 5  Python Environment + worktree setup                       26   RULE/COMMAND
├─ 6  Running Tests                                            130   COMMAND/GATE
│     ├ which command to run (the table)                        20          verbatim
│     ├ the four static gates (mypy · ruff · size · CLAUDE.md)  26
│     ├ tiers: the two axes + the marker table                  25
│     ├ file naming conventions                                 11
│     ├ fuzz / e2e / benchmark invocations                      33
│     └ contention: a timeout is never a semantic outcome       15
├─ 7  Smoke Test                                                18   COMMAND
├─ 8  Launcher + checkargs + --dry-run                          55   COMMAND
├─ 9  Training — the flags that change what a run IS           110   RULE/COMMAND
│     ├ launch commands (fresh · resume · fork)                  30
│     ├ --critic · --use-bridge · the two compile flags          34   (pointers, not detail)
│     ├ --fork-lr / the DOSE, the INERT-on-resume rule           16
│     └ the offline meters: untaught · elo · exploitability      30
├─ 10 Playing / the LADDER path                                 32   COMMAND
├─ 11 Showdown Server (the :8001 rule)                          34   RULE/HAZARD
├─ 12 Repository Structure (a real tree, one line per entry)     95   MAP
├─ 13 Path discovery — utils/paths.py                           30   RULE
├─ 14 Observation Vector (the table + never-hardcode)           24   MAP/RULE
├─ 15 Feature Extractor Architecture (pointer)                  18   MAP
├─ 16 Model Versioning (pointer + the immutability rules)       38   RULE
├─ 17 Data Dependencies (pointer to gen3_data leaf)             28   MAP
└─ 18 Prober · Web · Battle layer (three pointers)              22   MAP
```

**§4 is the one new section, and it is the census's main structural proposal.** Today's hazards are
scattered — the `dist/dist` ELOOP, the subagent stall rules, the BLAS pinning cliff, `pkill -f`
matching its own argv, the fuzz-reproducibility concurrency clause, "a memoized value is billed to
whoever asks first", "an allowlist entry can outlive its own fix". Each currently arrives as a
paragraph inside whatever section it was discovered in, so a reader meets them in the order we
*found* them rather than the order they *fire*. One screen, one line each, each with a link to the
leaf or research-state file holding the evidence, is both shorter and more likely to be read.

### Where each NARRATIVE block goes

Proposed new folder **`designs/research_state/engineering_notes/`** — engineering history, as
distinct from `measurements/` (numbers with provenance) and `learning_notes/` (concepts).

| from the root | lines | to |
|---|---:|---|
| the `.pth` vs `PYTHONPATH` ordering archaeology, the `environment.yml` two-pins story | ~50 | `engineering_notes/python_env_and_packaging.md` |
| the contention incident record (39/40 bogus skips, the load-22 tables, the 1.24-band bug) | ~60 | `engineering_notes/contention_and_timeouts.md` |
| the tier-measurement story, the ruff/file-size paid-off tables, the `-n N` timing tables | ~90 | `engineering_notes/test_tiering_and_static_gates.md` |
| the benchmark baselines + the memoized-`live_view` correction | ~43 | `measurements/post_paydown_baselines_2026-08-23.md` (already exists — cite it) |
| the four `checkargs` incident narratives | ~95 | `src/main/launcher/CLAUDE.md` (already owns them) + `engineering_notes/checkargs_incidents.md` |
| the bridge seed defects, the two CHOOSE-path incidents, the coverage census | ~195 | `src/utils/bridge/README.md` (already owns most) + `engineering_notes/bridge_transport_history.md` |
| the compile-flag measurement tables | ~50 | `designs/training/compile_flags.md` (per §3's split) |
| the `gen3_frame_deletion_v1` / `gen3_deadline_clock_v1` obs paragraphs | ~27 | `designs/CHANGELOG.md` (already verbatim there — delete the copy) |
| the lineage / `pin_history` / `git_hash` incident detail | ~40 | `src/agents/model/CLAUDE.md` § Model versioning (already owns it) |
| the `Repository Structure` per-module essays | ~140 | the corresponding leaf for each module |

Note how often the destination already has the content. **Roughly half of the root's narrative is
not homeless — it is a second copy**, and the deletion is the whole edit.

---

## 8. Context cost — today vs the proposal

| | lines | bytes | ~tokens |
|---|---:|---:|---:|
| root today | 2,275 | 174 KB | **~48,000** |
| root proposed | ~395 | ~30 KB | **~8,300** |
| **saved, every session** | −1,880 | −144 KB | **−39,700** |

| | ~tokens |
|---|---:|
| a session working in **training** today (root + leaf) | ~219,000 |
| the same session after both proposals (root + training hub) | ~19,300 |
| a session working in the **port** today (root + rust_sim leaf) | ~279,000 |
| the same session after both proposals (root + trimmed leaf) | ~76,000 |

Corpus-wide, the three proposals (root → 400, training → hub + sub-leaves, rust_sim → 2,400) take
the total from ~600,000 tokens to ~330,000 — but **the number that matters is the first row**:
~40,000 tokens returned on *every turn of every session*, whether or not the session ever touches
training, the port, or the prober.

**What the saving buys, stated honestly:** not correctness — the content is still findable, one
`Read` away, and an agent that needs the bridge's seed history can get it. What it buys is
**attention**. A 2,275-line preamble is skimmed; a 400-line one is read. The `--eval-workers`
default has been wrong in the training leaf for long enough that the same file contradicts itself
two hundred lines later, and the `better_line` "STILL NEEDED" block outlived its own fix — both are
what happens when a document is too long for anyone to read end to end.

---

## 9. What the freshness gate now enforces

`src/claude_md_freshness_gate_test.py` — unmarked, **0.84 s**, opt out with
`GEN3AI_SKIP_CLAUDE_MD_GATE=1`. Scope: all 13 `CLAUDE.md` under the repo.

| check | rule | escape hatch |
|---|---|---|
| **paths** | every `src/` `designs/` `data/` `tools/` `scripts/` path a CLAUDE.md names must exist — resolved against the repo root, the doc's own directory, and (for the port's leaf, declared with a reason) the vendored Showdown tree | `designs/deleted_flags.md` § Deleted PATHS; `_PROSE_NOT_A_PATH` for tokens that are English, not paths |
| **flags** | every `--flag` must resolve in the tree's own CLI surface — 602 option strings AST-scanned from `add_argument` calls plus hand-rolled `sys.argv` literals across `src/`, `tools/`, the Rust binaries, the Node harness and `scripts/*.sh` | `designs/deleted_flags.md` (DELETED / DEMOTED / PROPOSED, each cited); `_EXTERNAL_TOOL_FLAGS` for pytest/ruff/pip/cargo/git/chrome flags, each naming its tool |
| **the history doc** | every row carries a citation (version, signature, date or sha), and a row must LEAVE when its flag returns to the live surface | none — that is the point |

The surface is an AST scan rather than `build_parser()` for three reasons recorded in the module
docstring: importing the training parser alone costs 1.15 s and this covers ~30 parsers; several
entry points live in `__main__.py`, and importing one of those is the 2026-09-06 incident
`entry_point_guard_test.py` records (an import *started a training run*); and the scan was **measured
against the live `build_parser()`** — of 588 live options, every one it missed was an auto-generated
`--no-` negation, which is why the negation rule exists rather than being a guess.

**First run: 3 genuine misses, all fixed.**

| finding | fix |
|---|---|
| `src/agents/model/model_version.py` named in the root (L2178) and in `observation/CLAUDE.md` (L11) — it became a package on 2026-08-23 | corrected to `model_version/` in both |
| `--zarch-*`, `--value-clock`, `--use-showdown-bridge`, `--pubval-*` and 15 more named as live flags | 19 rows in `designs/deleted_flags.md` with CHANGELOG citations, split DELETED / DEMOTED / PROPOSED |
| `data/gen3_pubval.json` named in two leaves | listed as a deleted path with its v88 citation |

**The DEMOTED category is a finding in its own right.** `--attend-unrevealed-opponents`,
`--damage-refine-rounds` and `--threat-unrevealed-outgoing` are still spelled as command-line flags
in three leaves, but v78's tier axis moved them to `config_only`: the field lives, the flag cannot
be typed. The gate cannot tell that apart from a deletion, so they are recorded — but the honest fix
is a prose edit in each leaf, deliberately **not** made in this pass while other agents hold the
files.

---

## 10. Suggested order of work

1. **The three STALE corrections, now** — the rust_sim `better_line` "STILL NEEDED" block (it is
   actively wrong and it is the kind of block a subagent gets briefed from), the training leaf's
   `--eval-workers` default, and the four `instrumented_ppo.py` citations. Minutes each.
2. **Rename `## Next steps (engine, not yet built)`** — 874 lines of finished work under a title
   claiming the opposite.
3. **`designs/research_state/rust_sim_coverage_rounds.md`** — one move, 45 ROUNDs, −3,265 lines,
   after lifting the five standing lessons. Biggest single win in the corpus.
4. **The root's §4 STANDING HAZARDS block**, then the narrative extraction. Do the hazards *first*:
   it is the part that makes the trimming safe, because it forces every scattered footgun to be
   named before its host paragraph moves.
5. **`cf/` and `distill/` as real packages**, then the training split. Everything else in that plan
   is a `designs/` document; these two are what keep the biggest clusters auto-loading.
6. `designs/CLAUDE.md`'s two ~5,000-word table cells — not censused here, but at 191 bytes/line it
   is the densest file in the tree and the next one to look at.
