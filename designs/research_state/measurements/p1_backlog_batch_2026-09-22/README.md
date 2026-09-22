# P1 tech-debt batch — 2026-09-22

Three P1 rows from `designs/ops/TECH_DEBT_BACKLOG.md`, each landed as its own commit.

---

## 1. The `production` baseline "does not load at HEAD" — the diagnosis was wrong

**Row (2026-09-14):** `designs/baselines.json` `production` → `ai_v9_21_gen17_pfspoff_0820`
(config v97) "fails in `ExtractorBuild.__init__()` (`threat_prob_outspeed`) under current code.
Either re-point it … or make `main.baselines` report loadability."

### The measurement that decided it

Two loaders, same checkpoints, one box, CPU, 2026-09-22:

| baseline | bare `MaskablePPO.load` | `load_foreign_opponent` (the project's own) |
|---|---|---|
| `production` (v97) | `TypeError: ExtractorBuild.__init__() got an unexpected keyword argument 'threat_prob_outspeed'` | **OK** |
| `v9_long_baseline` (v101) | same `TypeError` | **OK** |
| `v9_fold_parent` (v103) | same `TypeError` | **OK** |
| `famine_comparator` (v101) | same `TypeError` | **OK** |
| `untaught_meter_opponent` (v101) | same `TypeError` | **OK** |
| `v8_line`, `v8_parent` (v45) | — | `ModelVersionError`: PRE-GENERATION, below `MIGRATION_FLOOR` v96 |

**So re-pointing `production` was never the fix.** The failure is not a property of that entry: it
is the *bare* load path rebuilding the extractor from the zip's own pickled
`features_extractor_kwargs`, which dies on any constructor flag deleted since — here
`threat_prob_outspeed`, deleted at config v108 (`gen3_dead_flag_purge_v2`). Every
current-generation entry fails it identically, so a re-point would have moved the failure to
whatever checkpoint was named next. `agents.model.snapshot._patch_historical_floor` →
`sanitize_dead_extractor_kwargs` already handles exactly this, and `load_foreign_opponent` runs it.

`production` is therefore **NOT re-pointed**: it is the architecture SURFACE the constructed mirror
(`designs/production_config.json`) is derived from, and its `pending` block already names the
owner-visible condition for the move (the win-prob arm passing its critic gate). Re-pointing it is
an owner call under `python -m main.baselines set … --reason`, not a tech-debt call — and it would
not have fixed anything. **No `main.baselines set` was run and no ledger line is owed for a
re-point.**

The 2026-09-14 Metamon de-risk (`metamon_derisk_2026-09-14/README.md` §"Why not the `production`
baseline") read the `TypeError` as "ordinary arch drift", substituted
`ai_v12_02_winprob_critic/final_model.zip`, and published under the task's original framing. That
substitution was avoidable; closing the silence is what this commit does. (That measurement's own
numbers are about the checkpoint it names and are unaffected.)

### What landed

* **`agents.training.baselines.load(name, *, device="cpu")`** — THE by-name model load. Goes
  through `load_foreign_opponent` (verifies `arch_signature`, runs the deleted-kwarg sanitizer),
  never a bare `MaskablePPO.load`.
* **`BaselineLoadError(BaselineError)`** — typed, with `.name`, `.reason`, `.era`, `.commit`.
  `.reason` ∈ `pre_generation` · `arch_drift` · `unresolvable` · `not_a_model`
  (`LOAD_FAILURE_REASONS`). Every message contains `FIX:` and the entry's commit. A
  `pre_generation` refusal names the entry's era (`v45/gen3_opp_hp_typed_candidates_v1`), this
  tree's `MIGRATION_FLOOR` / `ARCH_SIGNATURE`, the era-checkout command, and the
  current-generation names to use instead. An `arch_drift` refusal says in the message that it is
  a defect, **not a licence to substitute a stand-in checkpoint**.
  It subclasses `BaselineError`, so every existing `except BaselineError` keeps working.
* **`is_pre_generation(b)` / `era_of(b)` / `check_era(name)`** — the era verdict from REGISTRY DATA
  ALONE (`config_version < MIGRATION_FLOOR`, or a different `arch_signature`): no archive, no
  torch, same answer in a fresh clone. The `agents.model.model_version` import is lazy, so
  `baselines.py` stays torch-free at import time.
* **`era_checkout_only` is VALIDATED, not merely declared** (`_validate_era`, folded into
  `validate()`): an unmarked pre-generation entry is an **error**, and so is a stale mark on an
  entry this tree loads.
* **`python -m main.baselines check --load`** — actually loads every checkpoint entry (~2.5 s
  each) and prints the verdict. A declared `era_checkout_only` refusal is a `warn` (it is the era
  wall, not drift) so `check` does not go permanently red; an undeclared one is already an error
  from `validate()`.

### Tests that fail on revert

`src/agents/training/baselines_loadability_test.py` (10 tests, 8 structural + 2 `integration`):

* `test_EVERY_named_baseline_either_loads_or_raises_the_typed_error_naming_the_fix` — the row's
  acceptance test. Iterates the registry; every checkpoint entry either loads or raises
  `BaselineLoadError` with a reason in `LOAD_FAILURE_REASONS`, `FIX:` in the message, the entry's
  commit in the message, and `reason == "pre_generation"` **only** when the entry declares
  `era_checkout_only`.
* `test_the_bare_load_that_started_this_STILL_fails_where_baselines_load_succeeds` — pins the
  measurement above against the real bytes for `production` AND `v9_long_baseline` (the second is
  what kills the re-pointing hypothesis). Fails if `load` is ever "simplified" to a bare load.
* `test_load_goes_through_the_SANITIZING_loader_never_a_bare_MaskablePPO_load` — the same claim
  with no archive, by stubbing `agents.model.snapshot.load_foreign_opponent`.
* `test_every_registry_entry_declares_its_era_honestly` — fails red the day an entry's generation
  and its `era_checkout_only` flag disagree.
* `test_an_UNMARKED_pre_generation_entry_is_reported_as_an_ERROR` /
  `test_a_STALE_era_mark_on_a_loadable_entry_is_reported`.

Plus `test_check_accepts_load_and_reports_a_pre_generation_entry_without_failing` in
`src/main/baselines_test.py`.

### Still open

* Other call sites still bare-load by design and are correct to: `main.play.load_policy` (a ladder
  session should play the model it was handed or refuse — its docstring says so), and
  `src/main/anchors/session.py`, which already detects and reports this exact `TypeError`. Neither
  was touched (`src/main/anchors/` is another agent's scope this session).
* `designs/production_config.json` still mirrors v97 and still carries `threat_prob_outspeed`, a
  key deleted at v108 — recorded in `designs/research_state/era_boundary_deprecation_2026-09-06.md`
  item 4, unchanged here, and gated by the declared `config_mirror_version` migration.

---

## Ready-to-append ledger paragraph

> **2026-09-22 · TECH DEBT P1 ×1 — the `production` baseline was never the problem; the BARE
> LOADER was.** The 2026-09-14 row said `designs/baselines.json`'s `production` entry "does not
> load at HEAD" and offered a re-point or a loadability report. Measured 2026-09-22 on this box:
> a bare `MaskablePPO.load` raises `TypeError: ExtractorBuild.__init__() got an unexpected keyword
> argument 'threat_prob_outspeed'` on **all five** current-generation registry entries
> (`production` v97, `v9_long_baseline` v101, `v9_fold_parent` v103, `famine_comparator` v101,
> `untaught_meter_opponent` v101) and **every one of them loads** through
> `agents.model.snapshot.load_foreign_opponent`, which runs the deleted-kwarg sanitizer for the
> flag deleted at config v108. So a re-point would have moved the failure, not removed it;
> `production` is NOT re-pointed and remains the architecture SURFACE with its `pending` condition
> intact. Landed instead: `baselines.load(name)` as THE by-name load (the sanitizing loader,
> never a bare one); a typed `BaselineLoadError` carrying `.reason` ∈ {`pre_generation`,
> `arch_drift`, `unresolvable`, `not_a_model`} whose message always names the FIX, the entry's
> commit and — for a pre-generation node — the era and the current-generation alternatives, with
> `arch_drift` stating in prose that it is a defect rather than a licence to substitute a
> stand-in; `era_checkout_only` validated against each entry's recorded generation so an unmarked
> pre-generation node is an error; and `python -m main.baselines check --load`. The silence this
> closes is concrete: on 2026-09-14 the Metamon de-risk read that `TypeError` as arch drift,
> substituted a different checkpoint, and published under the task's original framing.
> `src/agents/training/baselines_loadability_test.py`;
> `measurements/p1_backlog_batch_2026-09-22/`.
