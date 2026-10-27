# Research State

The folder where what we try, what we know, what we do not, and what is left to find is written down. It
tracks the ongoing hunt, not one `ai_vN`. A finding that is not written here is rediscovered, or re-believed,
next session: **write the kills, the not-knowns and the pros/cons, not only the wins.** The measured numbers
live in [`measurements/`](measurements/), each file carrying its own checkpoint, step, state count and date:
cite a file there instead of retyping a percentage, so a stale number shows by its date.

## Layout

**The APPEND-ONLY record and the MUTABLE views are different documents, and the difference is the point.**
`ledger.md` records what was believed *at the time* and is never edited; everything else here states what is
believed *now* and is rewritten whenever the ledger says so. When the two disagree, **the ledger wins and the
view is wrong**: fix the view.

### The record (append-only)

- **[ledger.md](ledger.md)**: every dispatch with its predictions registered BEFORE the data, every verdict
  scored against them, every kill written as a kill. It opens with an at-a-glance status table and then runs
  chronologically. **Entries are cited by their landing sha** (the headings are prose), or by date plus title.
- **[ledger_index.md](ledger_index.md)**: a GENERATED title index over the ledger, one line per heading with its
  line number and date. **Never edit it, and never edit the ledger to suit it.** Regenerate with
  `python -m main.ledger_index --write` (generator `src/main/ledger_index.py`); `src/ledger_index_gate_test.py`
  fails a landing whose index is stale.

  **The heading convention** the index formalises: an ENTRY is a heading (`##` to `####`) carrying an ISO date.
  The early era parenthesises it mid-title (`## Gen-3 40M gate (2026-08-08) - ...`), the current one leads with it
  (`### 2026-09-07 · TECH DEBT · ...`). Undated headings are front matter or a subsection inside an entry; the
  index lists them nested and dateless. **Write new entries as `### YYYY-MM-DD · TAG · title`** so an index line
  sorts and reads correctly.

  **On a rebase conflict in `ledger_index.md`, never merge it by hand: keep BOTH ledger entries, then take either
  side of the index and re-run the generator.** The index is a pure function of the ledger.
- **[measurements/](measurements/)**: the raw outputs behind every number quoted anywhere. A `.md` beside a
  `.json` is the probe's own write-up.

### The mutable views (rewritten to match the record)

- **[UNDERSTANDING.md](UNDERSTANDING.md)**: what we believe about the research now (it owns the claim-tag
  vocabulary). **[EXPERIMENT_BACKLOG.md](EXPERIMENT_BACKLOG.md)**: the one ranked queue of experiments.
- **This README**: the index of this folder, plus the standing method below (defect genres, maintenance protocol,
  the amortizability gate, the build bar).
- **[levers/](levers/)**: one file per lever, in the **Known / Not-known / Pros / Cons / Next test** shape of
  `levers/_template.md` (its first line carries the status). A killed lever normally has no file; the two that
  survive (`opp_pp_observability`, `compile_opponents_net_value`) are kept because the way they died is reusable.
  The lever files are dated views: read each file's ledger shas, and re-check its status against the ledger.

### Standing programme documents

- **[bait_loop_hunt.md](bait_loop_hunt.md)**: the pre-registered hunt for the bait/loop pathology (an immune move
  fired into a voluntary pivot, repeatedly): baselines, bars, confounds, a fork per outcome. Instrument:
  `python -m main.prober.query loops`.
- **[hodge_predictions.md](hodge_predictions.md)**: the Hodge non-transitivity baselines and the pre-registered
  spinning-top predictions, with the game-count confound that voids a naive cross-generation width comparison.
  Instrument: `python -m main.elo <run> --hodge-bootstrap N`.
- **[capacity_battery.md](capacity_battery.md)**: the saturation battery on the shared trunk (effective rank,
  trainability against a fresh net, probe decodability, parameter census) with each metric's validity notes.
  A tripwire, not a verdict. Instrument: `python -m main.capacity <run>`.
- **[ladder_readiness.md](ladder_readiness.md)**: the public-ladder gap list (works / fixed / sized), the
  protocol-drift and latency measurements, Showdown's bot policy, and a go-live checklist. Instruments:
  `python src/main/play.py`, `python src/main/ladder_drift_scan.py`. An agent never plays the ladder (root § Playing).
- **[wang_search_reconciliation.md](wang_search_reconciliation.md)**: why Wang 2024's MCTS ablation and our
  depth-1 search-dividend probe are the same phenomenon at different noise levels, section by section from the
  thesis. Wang's leaves are the same trained critic ours are: do not cite him for "rollout-to-end". Every
  search-dividend cell measured before 2026-08-29 carries the allocator-noise caveat in the ledger entry `d2a0212`.
- **[exploiter_coverage_board.md](exploiter_coverage_board.md)** (+ `.json`): GENERATED, which vetted
  `data/teams/sample/` teams have a banked exploiter teacher, and the proposed arms for the rest.
- **[substrate_exploiter_gates.md](substrate_exploiter_gates.md)**: the pre-registered kill-capable gates for the
  conditional-mechanics substrate.
- **[cf_r1_runbook.md](cf_r1_runbook.md)**: the counterfactual label factory (R1): cost model, sampler design,
  label-quality prerequisites.
- **[critic_calibration_plan.md](critic_calibration_plan.md)**, **[gigo_signal_probe.md](gigo_signal_probe.md)**:
  the critic-reliability plan and the GIGO / order-mismatch tripwire.
- **[metamon_replay_feasibility.md](metamon_replay_feasibility.md)**: the external-corpus feasibility study behind
  the human-replay chapter.
- **Pre-registrations and registered designs**: [week_plan_2026-09-07.md](week_plan_2026-09-07.md),
  [flywheel_era_pair_2026-09-12.md](flywheel_era_pair_2026-09-12.md),
  [population_loop_round1_2026-09-23.md](population_loop_round1_2026-09-23.md),
  [population_loop_round2_2026-09-24.md](population_loop_round2_2026-09-24.md),
  [learner_battery_2026-09-26.md](learner_battery_2026-09-26.md),
  [new_lineage_2026-09-26.md](new_lineage_2026-09-26.md),
  [winprob_critic_ladder_2026-09-08.md](winprob_critic_ladder_2026-09-08.md) (a design note). Each is dated and
  history once its decision closed.
- **`gen{11..16}_endofrun_runbook.md`**: the pre-registered end-of-run battery of one generation, its numbered
  gates, and what each returned. History once the generation closes.
- **Censuses and policies** (dated): [claude_md_census_2026-09-06.md](claude_md_census_2026-09-06.md),
  [flag_census_2026-09-06.md](flag_census_2026-09-06.md),
  [era_boundary_deprecation_2026-09-06.md](era_boundary_deprecation_2026-09-06.md),
  [models_retention_policy.md](models_retention_policy.md) (what `models/` keeps).
- **Data files other code reads**: `heldout_panels.json`, `diagnostic_panels.json` (panels played by name outside the
  exclusion rule: `rc72`), `plateau_panels.json`, `human_replay_snapshots.json`,
  `exploiter_teamsets_2026-09-22.json`, `exploiter_coverage_board.json`. Do not edit one by hand without
  finding its reader (`grep` the name under `src/`).
- **[literature/](literature/)**: dated external-literature surveys (every claim cited and marked primary,
  secondary or UNVERIFIED): showdown bots landscape (2026-10-18), the hidden-information search review
  (2026-10-20) and the state-of-the-art survey on generalising to UNFAMILIAR teams (2026-10-26, five ranked levers
  N1-N5). A snapshot as of its date.
- **[reviews/](reviews/)**: dated independent code reviews (each finding confirmed by a repro or marked
  plausible); currently `core_review_2026-10-20.md`. Confirmed findings become TECH_DEBT_BACKLOG rows.
- **[learning_notes/](learning_notes/)**: concept notes a probe spun off. Several carry STATUS BANNERS marking
  which claims a later verdict refuted: read the banner first.
- **[memory_archive/](memory_archive/)**: agent memories retired from the index, verbatim.
  **[claude_md_archive/](claude_md_archive/)**: instruction-file narrative lifted out, verbatim, append-only; the
  late-August 2026 programme snapshot, the frontier table and the defect-genre specimens of THIS README are
  in [research_state_readme_history.md](claude_md_archive/research_state_readme_history.md).
- **[../ops/TRAINING_RUN_SOP.md](../ops/TRAINING_RUN_SOP.md)**: the operating procedure for a training run.

## The defect genres: five named failure classes

Five ways this codebase has produced confidently wrong measurements. Read a new probe design against this list
before it runs; the specimens are in the archive file above.

| Genre | What it is |
|---|---|
| **recorded != effective** | A flag is recorded in `cli_args`, the argv looks right, and the code path never fires. The house's most common defect. |
| **composition drift** | Every piece is individually correct; the ensemble's meaning is inverted by a sibling's deletion. Invisible to every per-piece test. |
| **silent-inert** | A mechanism "ran" and left no artifact, so nobody can tell whether it engaged. Defense: verifiability by construction (the mechanism writes its own state artifact). |
| **global-random coupling** | Two arms share the process-wide `random` module, so one arm's draws move the other's. |
| **untested intersections** | Each flag tested alone, the combination never. |

A sixth kind is not about a flag: **the metric's model of the data is wrong** (a magnitude statistic read over
columns that are categorical ids). Before quoting a magnitude statistic, check that every column is consumed as a
magnitude (`capacity_battery.md`).

## Maintenance protocol

After any investigation that changes our belief, update this folder in the same pass:
1. **Update the lever file** (create it from `levers/_template.md` when a candidate becomes active): facts into
   Known, open questions into Not-known, the upside into Pros, the caveats into Cons. A confirmed Con is as
   valuable as a Pro.
2. **Update the ledger** (append) and, if the belief changed, `UNDERSTANDING.md` and the backlog row.
3. **Apply the honesty gates** (`ledger.md` § "The method") before promoting a finding to Known, and
   adversarially verify a *confirming* measurement.
4. **A measurement cited by a committed doc must be committed in the same pass.** A citation to an uncommitted
   file is a dangling pointer wearing provenance clothes.
5. **Read a new arm against the defect genres before it runs and verify EFFECT, not argv**: confirm the flag moved
   a telemetry quantity, make the mechanism write its own state artifact, and test the intersection you are about
   to run.

## The amortizability gate: route every oracle finding through L1 to L4

The falsifier is an ORACLE (hindsight plus re-rolls), so "a better action existed" is its default output and
"needs search/MCTS" is the degenerate conclusion it is built to reach. A finding is a **lever** only if the
improvement is AMORTIZABLE into the single-forward-pass network. Route it through:

- **L1 obs-info**: the deciding info was ABSENT from the obs but is known/addable, so add an obs feature.
- **L2 representation**: the info WAS in the obs and the net mis-used it, so change arch / training / value
  target. Test: linear-probe the trunk; recoverable means L2, not recoverable means L1.
- **L3 anticipation**: needed the opponent's TYPICAL reply, so an opponent-action / forward-model aux head.
  Test: was the opponent move predictable from history and priors, or idiosyncratic/hidden?
- **L4 deliberation**: genuinely needs per-state multi-ply and the opponent move is not anticipatable. The only
  "search" bucket, and it ships as an OFFLINE TEACHER distilled into the net, never as runtime compute.

**HARD CONSTRAINT (owner): no lever may put search/MCTS on the model** (inference or the training loop). Search is
a teacher or offline diagnostic. "Cheap vein mined out, so MCTS" is a falsifiable claim about the SIZE of L4 that
must beat the measured L1+L2+L3 mass. (`endstate/search_ledger.md` owns the search record.)

## Decision posture: the build bar (say YES)

Falsification is a means (do not waste a retrain on a dud), not the goal. The honesty gates govern what we CLAIM
TO KNOW; they are not the bar for what we BUILD. With no single remaining big lever, progress is a long tail of
small stacked wins. A lever is **GO-TO-BUILD** when ALL hold (deliberately lower than "Known"):
1. **Amortizable**: it passes the L1 to L4 gate.
2. **Positive EV**: a plausible mechanism and non-trivial headroom (about 1% win rate or a clear behavioural fix).
3. **Falsifiable after build**: the metric that must move in the retrain is named, so a no-op is cheap to abandon.
4. **Bounded cost**: an obs feature, reward term or aux head, not a moonshot.

**Portfolio rule:** stack the GO-TO-BUILD levers into the next FRESH run and measure the aggregate, rather than
A/B-ing each forever. Stacking confounds attribution, so stack only mechanistically independent, cheap levers,
each with its own offline pre-retrain proxy; isolate a lever only when it is expensive or risky. The GO-TO-BUILD
queue is `ledger.md` § "GO-TO-BUILD queue".

## The frontier: what else might be there

The ranked candidate levers live in [EXPERIMENT_BACKLOG.md](EXPERIMENT_BACKLOG.md) (what to run next) and
[UNDERSTANDING.md](UNDERSTANDING.md) (what is believed); the per-lever files are in [levers/](levers/). The
2026-08 frontier table (each candidate with its status as of then) is in the archive file above. When a new
candidate surfaces, add it to the backlog and, once it becomes active, give it a lever file.
