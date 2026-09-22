# P1 tech-debt batch — 2026-09-22

Three P1 rows from `designs/ops/TECH_DEBT_BACKLOG.md`, each landed as its own commit:

| # | row | commit |
|---|---|---|
| 1 | the `production` baseline registry entry does not load at HEAD | `80560c85` |
| 2 | `ladder.json` needs a RECIPE STAMP and a gate | `0f230405` |
| 3 | `critic_read --v-column` silently ignored on every quota-MATCHED row | `eda0d0a3` |

**Two of the three rows named the wrong cause**, and in both cases the row's own suggested fix would have changed no number — see §1 and §3. Neither was found by reading the row; both came out of reproducing the failure first.

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

## 2. `ladder.json` now carries a RECIPE STAMP, and cross-run readers refuse or refit

**Row (2026-09-14):** a committed `snapshot_ladder/ladder.json` fitted before `3e6875a5` reads
+73 Elo above the current recipe on the same nodes and flipped the sign of a cross-run delta.

### Why the existing signal was not enough

Rule 24 of `UNDERSTANDING.md` §7 said "quote a committed file only if `eval_sentinel_edges_dropped`
is present". That key is a **count**, and a count cannot be the stamp:

* `0` is what a run that never measured a sentinel pair writes — indistinguishable from a file
  fitted by a tree that did not drop them;
* the 2026-09-08 file that started this records the key as **`null`**, which a presence check
  waves straight through;
* and it catches exactly ONE historical recipe change. The next one would be silent again.

### What landed

`fit_ladder` writes a `recipe` BLOCK:

```json
"recipe": {"name": "gen3_ladder_recipe_v1", "fitter_version": 2,
           "eval_sentinel_edges_dropped": true, "eval_sentinel_edges_dropped_count": 37,
           "commit": "<40-char HEAD of the tree that fit it>"}
```

`eval_sentinel_edges_dropped` is the **policy** (boolean); the count beside it is a property of the
run. `LADDER_FITTER_VERSION = 2` is documented as **bump-on-meaning-change**: v1 is the implicit
pre-stamp recipe (sentinel edges folded in), v2 is `3e6875a5`.

* `snapshot_ladder.recipe_status(doc) -> (status, detail)` — `current` / `absent` / `differs`,
  pure, so every reader asks the same question and a fixture needs no run directory.
* `snapshot_ladder.check_recipe(...)` raises **`LadderRecipeError`**; `recipe_refusal(...)` words
  the fix once — `python -m agents.training.snapshot_ladder <run> --fit-only` — and carries the
  +73.1 Elo the refusal is preventing.

Readers:

| reader | behaviour |
|---|---|
| `main.critic_gate --at-snapshots` | refits both sides normally (stamp then irrelevant). On the FALLBACK path — no `games.jsonl`, so the committed numbers are what gets quoted — a stale stamp is now a **`GateRefusal`**, replacing a "⚠️ FELL BACK" label on a number nobody could convert. Each side's `recipe_status` / `recipe` is in the section payload |
| `--exploiter-ladder auto:<run>` | **refits in memory** when `games.jsonl` exists (loud line), **refuses** otherwise. Rungs are picked BY ELO and the pre-fix inflation is non-uniform (+21..+29 on the newest nodes only), so a stale file builds a different curriculum |
| `main.ops.plateau_signal` | its bias note reads the stamp instead of asserting the bias unconditionally — which was wrong for any file fitted after `3e6875a5` |
| `snapshot_ladder` CLI | prints `[ladder] recipe: …` under every table |
| `snapshot_ladder.latest_promoted_elo` | **deliberately unchecked**, with the reason in its docstring: a within-run trend scalar written moments earlier by the run's own pinned code. Refusing would stop a live run logging its own curve |

`main.elo` was checked and does **not** read `ladder.json` — it fits the sparse star from the eval
rows, a different object. `g7_ladder.py` is an ad-hoc measurement script that reads TensorBoard
scalars, not the ladder file. The prober does not read it either.

### Tests that fail on revert

`src/agents/training/ladder_recipe_test.py` — before/after fixtures (`pre_recipe_ladder()`,
`post_fix_but_unstamped_ladder()`, `current_ladder()`), identical ratings, differing only in the
stamp:

* `test_the_fitter_writes_a_recipe_block` — including the disk round trip.
* `test_the_POLICY_and_the_COUNT_are_different_facts` — a `0`-count file reads `absent`.
* `test_recipe_status_classifies_the_before_and_after_fixtures` — incl. `null` and `{}`.
* `test_a_BUMPED_fitter_version_reads_as_differs` — catches the NEXT change, not only this one.
* `test_check_recipe_refuses_with_the_refit_command` — asserts `--fit-only` and `+73.1` are in
  the message.
* `test_the_exploiter_auto_ladder_REFUSES_a_stale_file_it_cannot_refit` /
  `..._REFITS_a_stale_file_when_it_can` (and that a read never rewrites the committed file).
* `test_latest_promoted_elo_does_NOT_check_the_stamp` — pins the deliberate exemption.

`src/main/critic_gate_test.py` — `test_a_STALE_recipe_on_the_FALLBACK_path_is_a_REFUSAL_not_a_label`,
`test_a_stale_recipe_is_IRRELEVANT_when_the_side_can_be_REFIT`, `test_the_section_REPORTS_each_sides_recipe`.
`build_run(..., stale_ladder_recipe=True)` and `_write_run(..., ladder_recipe=False)` are the
fixture switches.

### Still open

* Every `ladder.json` already on disk is unstamped and will read `absent` until refit — which is
  the intended, loud outcome. `--fit-only` is cheap (it plays nothing).
* `UNDERSTANDING.md` rule 24 still describes the count-key heuristic. Not edited here (out of
  scope); the ledger paragraph below is what supersedes it.

---

## 3. `critic_read --v-column` now reaches the quota-MATCHED rows

**Row (2026-09-16):** `quota_match._scan` calls `conditioning_block(...)` without `v_column`, so
a `--v-column values` report's matched rows are byte-identical to the `win_probs` run while its
as-traced table reads the shaped column. Suggested fix: "one kwarg at two call sites".

### The named site was not the bug

`conditioning_block(run_dir, step, *, frame=(arr, meta), v_column=…)` **selects nothing** when a
frame is injected: the column was chosen when the frame was extracted, and the argument only
reaches `extract_cycle` on the `frame is None` branch. Adding the kwarg at those two call sites
would have looked like a fix and changed no number.

The flag is actually dropped one level up, in `quota_match.build_quota_match`:

```python
cache[side] = CM.extract_cycle(d["trace_dir"])      # ← no v_column, ever
```

That is the matched path's OWN read of the trace tree. `critic_read`'s as-traced path passed
`getattr(args, "v_column", "win_probs")`; the matched path had no opinion at all — two
independent reads of one flag, which is the shape of the defect.

### What landed

* **`conditioning_meters.v_column_of(args)`** — THE one accessor, plus named `V_COLUMNS` and
  `DEFAULT_V_COLUMN`. Both `critic_read` call sites and `quota_match` now use it; a test asserts
  neither module still contains its own `getattr(args, "v_column"`.
* **`build_quota_match` extracts with the requested column** and records it on the document
  (`doc["v_column"]`), on both the matched and the opted-out branch.
* **`rung` reads the column OFF THE FRAME** (`smeta["v_column"]`, preserved by `subsample`), so
  the two `conditioning_block` calls agree with the data by construction rather than by a caller
  remembering.
* **`conditioning_block` REFUSES a frame/column disagreement** — the structural guard that turns
  the silent no-op into a loud `ConditioningRefusal` naming the fix. A frame predating the
  `v_column` meta key is trusted as the default, which is what it was.

### Tests that fail on revert

`src/main/ops/v_column_matched_test.py` plants a cycle where the two npz columns carry DIFFERENT
signals (`win_probs` is a function of the opponent only; `values` adds a strong per-team term), so
the own-team decoder finds something in one and almost nothing in the other:

* `test_the_matched_rows_CHANGE_when_the_v_column_does` — drives `build_quota_match` end to end
  at each column and asserts the matched points differ. **Verified against a revert**: with the
  `v_column=` kwarg removed from `build_quota_match`'s `extract_cycle`, this fails with "every
  quota-MATCHED row came out identical".
* `test_the_matched_frame_is_extracted_with_the_requested_column` — also fails on the same
  revert (`['win_probs', 'win_probs']`).
* `test_the_two_paths_read_the_flag_through_ONE_accessor` — fails if either module reintroduces
  its own `getattr`.
* `test_a_frame_whose_column_DISAGREES_with_the_request_is_REFUSED`, and
  `test_a_pre_v_column_frame_is_trusted_as_the_DEFAULT_not_refused`.
* `test_the_subsample_carries_the_column_into_every_matched_block`.

### Still open

* Banked reads are unaffected: every one was taken at the default `win_probs`, and the
  `conditioning_meters_v5_golden.json` byte-identity pin still holds.
* The `gate.*` reliability rows are still computed by the scaffolding gauge on `win_probs`
  regardless of the flag — pre-existing, documented in `--v-column`'s help, not changed here.

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


> **2026-09-22 · TECH DEBT P1 ×2 — a `ladder.json` now STAMPS the recipe it was fitted with, and
> a cross-run reader refuses or refits without it.** Rule 24 said to quote a committed file only
> if `eval_sentinel_edges_dropped` is present; that key is a COUNT and cannot carry the claim —
> `0` is what a run with no sentinel pair writes, the 2026-09-08 file records it as `null`, and it
> catches exactly one historical change. `fit_ladder` now writes a `recipe` block (name,
> `fitter_version`, the drop POLICY beside its count, and the fitting tree's commit);
> `recipe_status()` classifies a file `current`/`absent`/`differs` and `check_recipe` raises
> `LadderRecipeError` naming the refit command and the **+73.1 Elo** it prevents.
> `main.critic_gate` REFUSES a stale committed file on its fallback path — where the committed
> numbers are the ones quoted — instead of labelling it, and reports each side's stamp;
> `--exploiter-ladder auto:` refits in memory or refuses, because rungs are picked by ELO and the
> pre-fix inflation is non-uniform; `plateau_signal`'s bias note reads the stamp instead of
> asserting the bias. `latest_promoted_elo` is deliberately exempt (a within-run scalar).
> `LADDER_FITTER_VERSION` is documented as bump-on-meaning-change, so the NEXT recipe change is
> loud too. Every ladder already on disk reads `absent` until refit, which is intended.
> `src/agents/training/ladder_recipe_test.py`; `measurements/p1_backlog_batch_2026-09-22/`.


> **2026-09-22 · TECH DEBT P1 ×3 — `critic_read --v-column` now reaches the quota-MATCHED rows,
> and the backlog row's suggested fix would have changed no number.** The row named
> `conditioning_block(...)` without `v_column` as the site. That argument SELECTS NOTHING once a
> frame is injected — the column is fixed at extraction — so the real drop was one level up, in
> `quota_match.build_quota_match`, which reads the trace tree itself with
> `CM.extract_cycle(d["trace_dir"])` and no column at all. Fixed there. Both halves of a report
> now read the flag through ONE accessor (`conditioning_meters.v_column_of`), replacing two
> independent `getattr(args, "v_column", …)` reads; `rung` takes the column off the FRAME so it
> agrees with the data by construction; and `conditioning_block` REFUSES an injected frame whose
> recorded column disagrees with the request, turning a silent no-op into a loud error naming the
> fix. `src/main/ops/v_column_matched_test.py` plants a cycle whose `values` and `win_probs` are
> different tensors and asserts the matched rows change with the column — verified to fail on a
> revert of the one-line extraction fix. Banked reads unaffected (all at the `win_probs`
> default; the v5 golden byte-identity pin still holds).
> `measurements/p1_backlog_batch_2026-09-22/`.
