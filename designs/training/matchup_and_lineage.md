# Training — matchup, run-spec resolution and lineage

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## MatchupSpec — the declared matchup (`matchup_spec.py`)

**The ONE explicit declaration of what a run's battles look like** (design:
`designs/ai_v8/design_matchup_config.md`, P0 built). One week produced four independent failures with a
shared root — *the matchup a run plays is assembled implicitly across seams that nothing forces to
agree*: the eval worker rebuilt its own default teams (specialists measured OOD), the env's single
`team=` fed BOTH sides (the training mirror), training/eval play modes drifted (stochastic
noise-farming), and the launcher's exit summary resolved "Last model" to a global-glob golden. The spec
makes the matchup EXPLICIT: built ONCE in `train_rl_agent` (`MatchupSpec.from_args(args)`), then
CONSUMED — never re-derived — by the consumers (the `plan.json` pattern).

- **`TeamSource`** — where one side's teams come from; its `build(all_teams, sample_teams)` is the ONLY
  constructor of that side's `Gen3Teambuilder` (the env factory no longer assembles builders inline).
  Kinds: `pool` (opponent default), `default_biased` (trainee default — full pool + 10% sample-team
  bias, `DEFAULT_TRAINEE_BIAS_PROB`), `pinned` (`--trainee-team`), `pin_multi` (`--trainee-teams` — a
  SMALL FIXED SET sampled uniformly, the z-near multi-team exploiter / 1-vs-3-team A/B; `pin_str`
  mirrors `pin_strs[0]` so single-team consumers keep working, and unlike a single pin z_arch VARIES
  across the set), `pin_biased` (the future
  `--trainee-team-prob` shape — supported, no CLI yet). Each is byte-parity with the legacy
  construction (pinned by `matchup_spec_test.py`). **The two sides are independent BY CONSTRUCTION**
  (`trainee_teams` / `opponent_teams` → `Gen3Env(team=, opponent_team=)`) — the mirror-bug class is
  structurally closed.
- **`PlayMode`** — how the frozen-NN opponents select actions (greedy | stochastic@temp, schedule
  fixed | anneal | ratchet). Descriptive in P0 — the executors (RLPlayer temp, the anneal/ratchet
  callbacks) already exist; the spec records the intent so echo/provenance say what a metric was
  measured under. `eval_opponent_play` defaults greedy; `eval_trainee_teams` defaults to
  `trainee_teams` (**the eval-OOD fix made structural**: eval pilots what training pilots).
- **Provenance** — `to_dict()` (pin fingerprints via sha1, not full text) + `spec_hash()` (a 10-hex
  **measurement-regime tag**: two runs/eras with different hashes are NOT metric-comparable) are
  stamped into `metadata.json` beside `cli_args` (`_matchup_spec` / `_matchup_spec_hash`).
- **Startup echo** — `summary_lines()` emits a `🧭 [MATCHUP <hash>]` block to the launcher Events
  panel: trainee teams, opponent teams + mix, exploiter target + play mode, eval regime — one glance
  at what the run actually plays.
- **The realized-matchup fuzz** (`poke_env_gaps/matchup_realized_fuzz_test.py`, bridge, no server) is
  the permanent mirror-catcher: it drives the REAL construction path (spec → builders →
  `Gen3Env(team=, opponent_team=)` → bridge) over real battles and asserts per episode that the
  trainee fields EXACTLY the declared pin, the opponent does NOT (the mirror signature), and opponent
  rosters VARY across episodes. P1+ (not built): controllers keyed on eval play modes, per-row regime
  tags, per-opponent team pools.

### Matchup provenance (what a run trained/evaled against — the diligence layer)

Four self-describing records, all metadata-only + additive (old readers unaffected), closing the
"a row/trace/checkpoint can't say what regime produced it" gap the OOD-eval era exposed:

- **`eval_results.jsonl` rows carry `matchup_hash` + `externals`** (`append_eval_result_row`):
  each append-only ladder row is stamped with the run's CURRENT declared-matchup hash (rows from
  different regimes/eras are distinguishable IN-FILE, not by dates), and the per-cycle vs-target
  record (`{ext label: {win_rate, counts}}` — e.g. the exploiter VERDICT) now survives in the
  jsonl instead of only the overwritten `latest_eval` + TensorBoard. Externals stay OUT of `bots`
  (the ELO fit's ladder is untouched).
- **`metadata.json:matchup_history`** (append-only, maintained by `save_model_snapshot` from the
  `cli_args` stamp): one `{hash, spec, recorded_at}` entry per ERA — a resume that changes the
  declared matchup appends a new era instead of silently overwriting the old one (cli_args keeps
  only the latest). Saves without cli_args (the periodic-checkpoint path) preserve it.
- **The resume MATCHUP-DRIFT guard** (`train_rl_agent`, warn-not-fatal): a `--model` resume whose
  declared matchup hash ≠ the run's recorded one emits a loud `⚠️ [MATCHUP DRIFT]` + the
  field-level diff (`matchup_spec.describe_drift`) — a mid-run curriculum change is legitimate,
  doing it SILENTLY is not. Launcher restarts forward flags verbatim → never fire it.
- **`eval_manifest.json` records the eval REGIME**: `matchup_hash`, `trainee_team_sha` (the pin
  the trainee piloted; None = pool), `opponent_pins` ({ext label: sha} for fold-back-pinned
  opponents) — a trace dir is self-describing about HOW its numbers were measured.
- **Checkpoint sidecars + `snapshot_history` entries carry `matchup_hash`** (via
  `record_checkpoint` → `_build_snapshot_entry`, like the `latest_eval` stamp) — each checkpoint
  is self-describing about what it was training against as of its save, robust to later eras.

Readers: `snapshot._read_matchup_hash(model_dir)` (current era) /
`snapshot.read_recorded_matchup(model_path)` (the drift guard's input). Tests:
`snapshot_test.py::test_matchup_*`/`test_eval_row_*`/`test_checkpoint_sidecar_*`,
`matchup_spec_test.py::test_describe_drift_*`, `eval_callback_test.py::test_eval_manifest_records_the_regime`.

## WHICH FILE a run spec names — the ruling, and why it changed

> The rungs table and the DISAGREEMENT rule stay in `src/agents/training/CLAUDE.md`.

**A bare run directory resolves to the run's LAST SNAPSHOT, not to `best_model/best_model.zip`.**
Owner ruling, 2026-09-06, verbatim: *"I would either prefer us do best against target or just do the
last snapshot. I feel like best against target will always have a nuance that we need to keep track
of, whereas the last one is probably what our metrics would measure anyway."*

**WHY IT CHANGED.** `best_model/best_model.zip` is exported on **BOT win rate** — an opponent set
with nothing to do with what a teacher is being distilled FOR. Ledger 2026-09-06 (probe H8,
*exploiter off-slice competence*) measured the consequence: for 2 of 8 unfunded R5F teachers
(`ai_v9_94_R5F02`, `ai_v9_98_R5F06`) the exported file was a **~0.93M-step exploiter rather than the
~2.93M final**, so "the teacher" a fold distilled from was neither the last snapshot nor the best
against its target — and **nothing recorded which file was used**. It made `teacher_distance`'s
UNF budget covariate (3.07M) heterogeneous (≈2.43M mean) on the very axis it had found
rank-indistinguishable from D_off. Every meter this programme banks scores a run at its END, so the
last snapshot is what the metrics already measure.

### Every consumer goes through the ONE choke point

`agents.training.fixed_opponent_pool.resolve_model_ref(path, step=None)` → a `ResolvedModel`
(`zip_path`, `config_path`, `run_base`, `run_dir`, `rung`, `rule`, `num_timesteps`). The flags it
serves: **`--distill-teacher`** and **`--win-prob-pbrs-source`** (`main/train/model_build.py`),
**`--stable-opponents`** and **`--exploiter`** (via `resolve_stable_opponents`),
**`--exploiter-ladder`** (`exploiter_ladder.py`), **`--warmstart-consensus`** (`warmstart.py`) and
**`--distill-anchor-parent`** (`main/train/callbacks.py`). `run_spec_test.py` holds the census that
fails, naming the file and its flags, when one of them stops.

**`_resolve_zip_and_config(path, step)` is a FROZEN 3-tuple wrapper over it** — the offline probe
scripts under `designs/research_state/measurements/arch_transfer_2026-09-05/`
(`content_locality_v2`, `exploiter_competence`) import it by name to reproduce exactly the call
`model_build.py` makes for a teacher. **They measured the OLD rule's files, by design, and stay as
records of it.** New call sites that want the rung or the step should call `resolve_model_ref`.

### Provenance — a fold now records which file it loaded

* `metadata.json`'s **`lineage`** block: every model reference (`fork_parent`, each entry of
  `teachers`, `exploiter_target`) carries `resolved_file`, `resolved_num_timesteps`,
  `resolution_rung` and `resolution_rule`. `python -m main.lineage <run>` prints them.
* **Startup lines**: `🧪 [DISTILL]` emits one `teacher <k>: <spec> -> <zip> @<N> steps [rung=… rule=…]`
  per teacher; `🐴 [STABLE]` and `🥊 [EXPLOITER]` emit the same per opponent
  (`FixedOpponentEntry.provenance()`); `🧊 [WinProbPBRS]` names its frozen φ the same way.

🚨 **EVERY TEACHER LOADED BEFORE 2026-09-06 WENT THROUGH THE OLD RULE** (`best_model` first, then
`final_model.zip`, then `<run>/best_model.zip`) and recorded nothing about it. A pre-change run's
teacher identity is therefore **not recoverable from its metadata**, and `main.lineage` says
`resolved file not recorded (pre gen3_last_snapshot_resolution_v1)` rather than re-resolving it
under today's rule — a current answer presented as history is worse than no answer. A reference
this change DID try and fail to resolve records `unresolved`, so the two are distinguishable.

**NOT VERSIONED.** This changes which FILE a run loads, never a weight shape, so it is absent from
`ModelVersion.check_compatible` / `arch_signature` by design, and no checkpoint on disk becomes
incompatible with it. Gates: `agents/training/fixed_opponent_pool_test.py` (each rung, both
disagreement directions, the explicit-form passthroughs, the frozen 3-tuple, the entry's
provenance), `agents/training/run_spec_test.py` (the choke-point + consumer census),
`agents/training/lineage_test.py` (the recorded fields and the legacy message).

## LINEAGE — who forked whom (`lineage.py`, `python -m main.lineage`)

**Every exploiter, fold, funding fork and dose arm is a fork of some parent, and every comparison
the ledger makes is a claim about that graph.** Until `gen3_run_lineage_v1` that graph was
recoverable only by REGEXing `--model` out of `metadata.json`'s recorded `original_command` —
brittle in the obvious ways (a renamed flag, a quoted path, a `--model=X` spelling) and silent in
the worst one: **a failed parse reads exactly like a fresh run.** `metadata.json` now states the
answer instead of implying it.

**The block** — one top-level `lineage` key, written ONCE at fork creation:

```
lineage: {schema, role, fork_step, recorded_at,
          fork_parent: {path, resolved_path, run_dir, run_name, git_hash, arch_signature,
                        model_config_version, num_timesteps, sha256, created_at} | null,
          teachers: [ …same shape… ], exploiter_target: {…} | null,
          ancestry: [ {run_name, run_dir, git_hash, arch_signature, fork_step, role,
                       model_path, source} … ],
          ancestry_stop: {at, reason}}
```

`role` is `fresh` / `fork` / `fold` (`--distill-teacher`) / `exploiter` (`--exploiter`, which wins
— a double-sided exploiter is an exploiter that also distils, and its TARGET is what identifies
it). `ancestry` is walked through each parent's OWN block, nearest ancestor first, bounded and
cycle-safe on realpaths; `ancestry_stop` says where the chain went dark and why, because *"the
chain ends at a fresh root"* and *"the parent directory is gone"* are different facts a bare list
conflates.

🚨 **IMMUTABILITY is the whole feature, and it reuses `original_command`'s mechanism verbatim**:
`save_model_snapshot` reads the existing value first and the existing value always wins. A launcher
run restarts every few hours and an idempotent FORK has its `--model` swapped to the fork's OWN
latest checkpoint on each relaunch (`launcher.checkpoint.resolve_fork_resume_model`), so a block
re-derived on a restart would silently re-point the recorded parent at the DRIFTED student — the
exact failure `distill_anchor_callback` has a module of prose defending against. Belt and braces:
`build_lineage` also returns `None` on a same-run restart, decided by
**`main.train.fork_lr.is_same_run_checkpoint`, IMPORTED rather than re-derived** (a second
predicate for the same question is a second answer waiting to disagree; `<run>/warmstart/…` is
deliberately a FORK there, and the seam captures `args.model` BEFORE `--warmstart-consensus`
re-points it, or a warm-started exploiter would record itself as its own ancestor).

**The FRESH form is explicit** (`fork_parent: null, role: "fresh", ancestry: []`) — "no block" and
"no parent" are different facts and only one of them is a measurement.

**THE ACCESSOR is the API** — `agents.training.lineage.fork_parent(run_dir) -> ForkParent | None`
and `.ancestry(run_dir)`. It returns the recorded block's parent when there is one and otherwise
DERIVES it from `original_command`, printing
`[lineage] WARNING: derived from original_command (legacy run, pre-lineage)` to stderr.
`ForkParent.derived` says which, so a legacy guess is never mistaken for a recorded fact. Every run
on disk today is legacy, so the derive path is not a corner case — but it lives in exactly ONE
place, marked as legacy, instead of in each consumer.

⚠️ `distill_anchor_callback.resolve_anchor_parent`'s `original_command` branch is the CURRENT
consumer of that regex and should move to this accessor — same answer, one implementation, and the
recorded block preferred where a run has one.

**The CLI** is `python -m main.lineage <run>…` (torch-free and model-free, so it reads a run whose
architecture drifted past current code — which is most of `models/`). It prints the tree with each
node's `git_hash` / `arch_signature` / `role` / `fork_step`, plus the run's teachers and target, and
**flags a broken link**: a parent run directory that is gone, a parent checkpoint whose sha256 no
longer matches the file on disk, and an `arch_signature` that CHANGED across a link (a fork cannot
have loaded a differently-shaped parent, so the recorded parent is wrong). `--json` for scripts;
exit 1 when anything is flagged. `--backfill` derives a block for a LEGACY run and writes it marked
`"derived": true` — **dry-run unless `--apply`**, and it REFUSES a run that already records one,
because a backfill that overwrote a recorded parent with a re-parsed guess would defeat the point of
recording it.

🚨 **RECORDED and DERIVED are INDEPENDENT, and the CLI's header line states both.** A `--backfill`
block is on disk *and* was REGEXed out of `original_command`, so it prints
`[recorded ⚠ DERIVED from original_command]` — a recorded GUESS, not a lesser kind of recording. A
run with no block but a parseable command prints `[⚠ DERIVED from original_command]`; a DERIVED
ANCESTOR is a third fact, marked `⚠ derived` on its own node. Each row carries `derived_self` (this
run's lineage was derived), `parent_derived` (the parent REFERENCE it names carries the flag) and
`derived`, their union; more than one run also prints a summary counting each. **Until 2026-09-07
the flag was computed from the parent alone**, so a block whose derivation concluded `fresh` — no
parent, hence no parent-side signal — read `derived: false` while its metadata said the opposite:
**47 runs invisible to the marker, and an archive count of 115 against the census's 162**
(`designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md` defect 1).

**The seam is two lines.** `run_io._run_lineage(args, model_dir, model_path=…, fork_step=…)` is the
one place that knows which argparse fields carry the parent, teachers and target; `model_build`
builds the block once per path and passes it to every `save_model_snapshot` call. All the work is in
`agents/training/lineage.py`, which is torch-free and reads a checkpoint's `num_timesteps` out of
the SB3 zip's plain-JSON `data` member via `zipfile` — never by loading the model.

Tests: `lineage_test.py` (41) — the fork block incl. the sha256 and the zip/filename step read; the
same-run restart building nothing and the recorded block surviving a restart byte-for-byte against
a DIFFERENT offered parent; the fresh null form; ancestry over two recorded levels, a legacy stop,
a missing directory, a CYCLE and the depth limit; the accessor's recorded-vs-derived split and its
warning; the three integrity checks; the CLI on a synthetic tree; backfill's dry-run, apply and
refusal; and the seam pins (both build paths ask for it, every save carries it, the pre-warm-start
capture, and `dose` staying current while `lineage` stays immutable).

### A FORK INHERITS ITS PARENT'S CURVES (`tb_inherit.py`, `gen3_tb_inherit_v1`)

**Bookkeeping over two facts this tree already had, not a new mechanism.** (1) TensorBoard merges
every `events.out.tfevents.*` in ONE run directory into one series per tag, ordered by step — not a
claim: `ai_v8_03_zarch_control_0718` carries **29** event files (one per launcher restart) in a
single `tb/` and renders as one curve. (2) A fork's global step CONTINUES the parent's
(`reset_num_timesteps=False`), measured on that same run — its `tb/` opens at step **148,401,356**,
precisely the `fork_step` its `lineage` block records. So the parent's curve occupies `[0, fork_step]`
and the fork's occupies `[fork_step, …]`: two halves of one line, apart only because they live in two
directories. `<fork>/tb/` now also gets a TRUNCATED copy of the parent's scalar events at steps
**≤ `fork_step`**, and the fork's TensorBoard reads from step 0.

**TRUNCATION is not optional.** The parent usually trained PAST the fork (`ai_v8_01` reached 170.6M
having been forked at 148.4M); copying its tail would draw parent-only progress inside the fork's own
step range, where it reads as the fork's.

**FORK-OF-A-FORK composes for free.** The parent's `tb/` already holds its own inherited prefix, so
reading the parent's WHOLE directory and truncating again yields grandparent `[0, parent_fork_step]`
+ parent `[parent_fork_step, fork_step]`. One rule per link.

🚨 **THE VALUE FORM IS PART OF THE COPY, AND GETTING IT WRONG IS INVISIBLE.** A TB scalar has two
on-disk spellings — the classic `simple_value` field and a rank-0 tensor tagged with the `scalars`
plugin — and **`EventFileLoader` MIGRATES the first into the second as it reads**. Every writer in
this tree emits `simple_value`, so copying what that loader returns writes the migrated form, which
`EventAccumulator` files under `tensors` rather than `scalars`: the events are in the file, the
provenance is right, the byte count is right, and **the scalars dashboard shows nothing**. Measured
while building this — the fork's prefix read back as 124 `tensors` tags and **0** `scalars` tags,
beside its own 124 `scalars`. `scalar_prefix` therefore reads with **`LegacyEventFileLoader`** (raw),
and the test writes its synthetic parent in `simple_value` form and reads back through
`EventAccumulator.Scalars` — the accessor the dashboard uses — because a test that writes and reads
the same migrated form cannot see it. Two guards fail on revert.

**IDEMPOTENCY is load-bearing.** `<fork>/tb/INHERITED_FROM.json` is the key: if it exists, no-op. A
launcher restart that still names the parent as `--model` (which happens before the fork has written
its own checkpoint) re-enters the fork path and would otherwise append a SECOND copy of the prefix —
the same series twice, which renders as a saw-tooth rather than an error.

**The seam is the LINEAGE seam.** `inherit_from_lineage(model_dir, lineage_block, enabled=…)` is
called in `main/train/model_build.py` immediately after the block is recorded and BEFORE
`_attach_run_tb_logger`. The block being non-None **is already the fork decision**
(`build_lineage` returns None on a same-run restart, via `fork_lr.is_same_run_checkpoint`), and the
parent + `fork_step` are read out of it — so the curve a fork inherits and the parent it claims
cannot disagree, and nothing here owns a second answer to "is this a fork?" (pinned by an AST test).
It **never raises**: a cosmetic convenience must not be able to kill a launch, so a failure returns a
reason and the run simply starts its chart at `fork_step`.

**SCALARS ONLY** — histograms/images/audio and every non-`scalars` plugin are skipped. Measured
2026-09-06: every value in `models/*/tb/` is a scalar, so the filter drops nothing today; it exists
so adding a histogram later cannot silently multiply the copy cost. `tb/` is 262 MB across 217 runs;
a prefix is a few hundred KB.

**`--no-tb-inherit`** opts out. Worth it for a large sibling FLEET under an UNCURATED logdir — 8
exploiters off one target then draw 8 identical prefixes in every chart (under `main.tb_curate` that
is exactly what you want). Training-runtime class: reaches no extractor, scales no loss, changes no
weight shape ⇒ no `ARCH_SIGNATURE` bump, not on `ModelVersion`, not in `check_compatible`, not in
`flag_registry`; it lands in `cli_args` and the launcher forwards it verbatim.

**Existing forks are NOT backfilled.** `python -m main.tb_inherit --list` censuses them (**137**
missing a prefix as of 2026-09-06); `--backfill <run>… | --all` writes one, **dry-run unless
`--apply`**, `--show` prints a run's provenance. ⚠️ **105 of the 137 name a DERIVED parent** — every
`lineage` block on disk was written by `main.lineage --backfill`, i.e. by REGEXING `--model` out of a
recorded shell command — and the derivation can be wrong: `ai_v8_01_zarch_film_0717` records
`role="fresh", fork_step=0` while its own `tb/` opens at step **148,401,356**, which is arithmetically
impossible for a fresh run (it was built by a hand-written `tmp/fork_zarch_v8.py` the regex cannot
read). Every row prints that flag and the census prints the caveat.

Tests: `tb_inherit_test.py` (22) — truncation at the exact boundary; the one-continuous-series claim
read through `EventAccumulator.Scalars`; the on-disk value-form preservation (both spellings) and its
revert-catcher; the non-scalar skip; provenance content + sha; the second call being a no-op and
`force` replacing rather than appending; dry-run touching nothing; fork-of-a-fork truncating the
grandparent prefix; the seam's restart/fresh/disabled/missing-parent no-ops; the census incl. the
`derived` flag; and the torch-free + no-second-fork-predicate contracts.

