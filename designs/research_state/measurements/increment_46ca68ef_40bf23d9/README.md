# INCREMENT FUNCTIONAL TEST — `46ca68ef → 40bf23d9`

Method: the same pinned-input functional test the four preceding increments used
(ledger 2026-09-09 ×3, 2026-09-10 ×1). Two worktrees at the two commits, one argv, one seed,
one set of fixture bytes; everything compared by per-tensor byte hash / `torch.equal`-equivalent
hashing, never by eyeball. No end-to-end bit comparison of a rollout — that method is VOID
(ledger 2026-09-09: two runs of the SAME commit at the SAME seed diverge at rollout 1).

## VERDICT

| configuration | verdict |
|---|---|
| a **control**-configured run (no new flags, `ai_v12_11_ladder_ctrl10M`'s argv) | **TRAINING-SEMANTICS-NEUTRAL** |
| **arm 8**'s λ-0.9 configuration (`--win-prob-lambda 0.9`, no rollout flags) | **TRAINING-SEMANTICS-NEUTRAL** |

Arm 10 (`argv_J_rollout_l09.txt`) is therefore readable against arm 8 (pin `28ece02a`) and its
replicate `lambda09_b`: the OFF-state of `--win-prob-rollout-target` / `-r` / `-mode` /
`--win-prob-rollout-weight` is bit-identical on every path those two runs exercise.

## THE SPAN

`git log --oneline 46ca68ef..40bf23d9` = **41 commits**. `git diff --stat … -- src/rust_sim/`
is **EMPTY**. 40 files under `src/` changed (4,840 insertions).

⚠️ **The brief's claim that the only src-touching commits are the five named ones is WRONG.**
Three further commits touch `src/`, all of them confined to `src/main/ops/` — the OFFLINE
meter code, not the training path:

| commit | subject | src files |
|---|---|---|
| `a0e72b38` | WIP ops(critic_read v5): the CALIBRATION SLOPE | `calibration_slope.py`, `conditioning_meters.py`(+test), `critic_read.py`, `critic_read_render.py`(+test), `quota_match_test.py` |
| `579c5ceb` | ops(critic_read): the (A)/(B) rows | `conditioning_meters.py`(+test), `critic_read*.py`, `team_conditioning.py`, `quota_match_test.py` |
| `322acaee` | ops+measurement: the (A)/(B) reads | `critic_read_render.py` |

Nothing in `src/agents/**` or `src/main/train/**` (production code) imports `main.ops` — the
only occurrences are in help text, docstrings and `*_test.py` files. So the ops tree is
outside the training-semantics question by construction; it is a **METER-COMPARABILITY**
question instead (hazard 6 below).

## EVERY `src/` FILE IN THE SPAN, AND HOW IT WAS COVERED

### A. Training path — covered by execution on pinned inputs

| file | change | covered by |
|---|---|---|
| `agents/model/arch_tables.py` | +2 `_COEF_MODULE` rows | registry key diff (§1); rows are report-only |
| `agents/model/model_version/constants.py` | `MODEL_CONFIG_VERSION` 117→119 | migration §5 |
| `agents/model/model_version/construct.py` | +4 ctor kwargs, all defaulted | `ModelVersion.from_layout_and_policy_kwargs` on arm 8's own argv; `arch_signature` unchanged |
| `agents/model/model_version/fields.py` | +4 dataclass fields | recorded-field count 139→143; none in `_WEIGHT_FIELDS` |
| `agents/model/model_version/migrations.py` | +2 hops (118, 119) | synthetic v117 → 119 with the OFF record; three real configs migrated |
| `agents/training/async_vec_env.py` | +`wp_handle` capture | guarded on `model._win_handle_keys is not None`, which only `_on_rollout_start` allocates and only when the fraction > 0. Not reachable at the default; the surrounding terminal capture is byte-unchanged |
| `agents/training/cf_records.py` | `_safe_tag` → public `safe_tag`, alias kept | pure rename + alias; no caller moved |
| `agents/training/gen3_env.py` | `emit_win_row_weight` kwarg + `win_row_w` key | REAL `Gen3Env` obs space, both configs (§3) |
| `agents/training/instrumented_ppo/hparams.py` | +4 dataclass defaults | `_model_hparams` / `apply_training_hparams` table; smoke ran |
| `agents/training/instrumented_ppo/ppo.py` | `rollout_weight_on` predicate + metric fold | predicate is `weight>1.0 AND target>0.0 AND key in buffer` — all three false at the default; TB tag set unchanged in both smokes |
| `agents/training/instrumented_ppo/value_terms.py` | `_win_prob_loss(rollout_w=…)` | §2 — loss bits identical, `rollout_w=None` and `rollout_w=ones` both equal the old value |
| `agents/training/opp_class_plumbing_test.py` | stub gains `_emit_win_row_weight` | test-only |
| `agents/training/value_sidecar.py` | schema 2→3, +3 header fields, +3 quantity fields | §6 + both smokes' real `rows.jsonl` |
| `agents/training/value_sidecar_test.py` | probe string `WinProbLabelCallback(` | test-only |
| `agents/training/win_prob_callback.py` | ctor args, `_apply_rollout`, `_apply_rollout_weight`, anchors in the recursion | §2 and §4 — the callback DRIVEN on a seeded buffer at λ ∈ {1.0, 0.95, 0.9} × {bootstrap, mask} |
| `agents/training/win_prob_rollout.py` (new) | constants + selection | imported at module scope by the callback; AST-checked to have **no module-level side effects** (numpy + 2 intra-package imports + 9 constants) |
| `agents/training/win_prob_rollout_labeller.py` (new) | continuation labeller | imported **lazily**, inside `if states:` under `_apply_rollout`; unreachable at fraction 0 |
| `agents/training/win_prob_rollout_worker.py` (new) | worker | reached only from the labeller |
| `agents/training/winprob_rollout_test.py`, `winprob_rollout_weight_test.py` (new) | tests | test-only |
| `agents/training/wrappers.py` | `+import os`; handle capture in `step()` | guarded on `getattr(self.env, "_emit_wp_rollout_handle", False)`, which `env_factory` sets only when the fraction > 0 |
| `main/train/callbacks.py` | `WinProbLabelCallback(records_dir=…, impl=…)` | **executed** — both debug smokes build the callback list for real and reach `Training complete` |
| `main/train/combination_checks.py`(+test) | +3 refusals | registry name diff (§1); all three predicates are false at the default |
| `main/train/config.py` | +4 `_resolve` lines, +3 range checks | `resolve_config` executed on both argvs; resolved values 0.0 / 8 / "replace" / 1.0 |
| `main/train/env_factory.py` | `emit_win_row_weight=` + `_emit_wp_rollout_handle` | the factory's OWN predicate captured by running `_init()` (§3) — `False` for both argvs |
| `main/train/lifecycle.py` | +4 `ModelVersion` kwargs in the round-trip | both smokes print `[ModelVersion] Round-trip smoke test PASSED` |
| `main/train/model_build.py` | +4 `_TRAINING_HPARAMS` rows, +4 kwargs ×2 sites | smoke; hparams table |
| `main/train/parser/clean_world.py` | +4 `add_argument` | §1 — 286 → 290 args, all 4 at their OFF default on both argvs |
| `main/train/run_io.py` | +4 `_model_hparams` keys | smoke's `model_config.json` |

### B. Offline meters — NOT training semantics (see hazard 6)

`main/ops/calibration_slope.py`, `conditioning_meters.py`(+test), `critic_read.py`,
`critic_read_render.py`(+test), `quota_match_test.py`, `team_conditioning.py`,
`value_sidecar_read_test.py`.

## THE NUMBERS

### §1 argv → resolved config (`build_parser → resolve_config`)

Both argvs, both trees. `_explicit_flags` is **set-equal** (179 / 180 entries; the repr order
differs only because `frozenset` iteration order is not stable) — zero flags gained or lost.

| | control | arm 8 |
|---|---|---|
| args, old → new | 286 → **290** | 286 → **290** |
| `win_prob_rollout_target` | — → `0.0` | — → `0.0` |
| `win_prob_rollout_r` | — → `8` | — → `8` |
| `win_prob_rollout_mode` | — → `'replace'` | — → `'replace'` |
| `win_prob_rollout_weight` | — → `1.0` | — → `1.0` |
| `win_prob_lambda` | `1.0` (both trees) | `0.9` (both trees) |

`arch_tables._COEF_MODULE` gains `win_prob_rollout_target`, `win_prob_rollout_weight`.
`COMBINATION_CHECKS` gains `winprob_rollout_needs_the_winprob_critic`,
`winprob_rollout_needs_cf_records`, `winprob_rollout_weight_needs_the_rollout_target`.
Every other key in both structures is unchanged.

### §2 Extractor forward + parameters (production arch, `--arch production` surface ✓)

Seeded `torch.manual_seed(42)` construction, pinned obs batch `Generator().manual_seed(1234)`,
8 × 2501.

| | control | arm 8 |
|---|---|---|
| state_dict tensors | **230 = 230** | **230 = 230** |
| named parameters | **229 = 229** | **229 = 229** |
| parameters | **1,950,003 = 1,950,003** | same |
| tensors differing by byte hash | **0** | **0** |
| `pi` hash | `a2032b7d4742df22` both | same |
| `vf` hash | `1456ec43788494d8` both | same |
| `pi` / `vf` float64 sums | 917.5859788386151 / 905.7699173854198 | identical |
| **extractor kwargs diff** | **EMPTY** — 51 keys, every value equal | **EMPTY** |

🚨 This is the first increment in the chain since `f871e79f→d11386dc` whose extractor-kwargs
diff is empty again — v117's `dense_aux` added a key, v118/v119 add none.

### §3 λ-return targets and the win-prob BCE

`lambda_return_targets` on a fixed buffer (24 × 6, `default_rng(20260910)`; fixture hashes in
`new.json`/`old.json` under `lambda_fixture`), six cells:

| cell | target hash (both trees) | `n_unmasked` |
|---|---|---|
| λ=1.0 bootstrap | `52cf57d479ff162f` | 4 |
| λ=1.0 mask | `c2db962595bdf591` | 0 |
| λ=0.95 bootstrap | `3863df38b5d007a5` | 4 |
| λ=0.95 mask | `55cd851bd347e039` | 0 |
| **λ=0.9 bootstrap** | **`49bbdfd293d72be9`** | 4 |
| λ=0.9 mask | `5f37dffbd1b20abb` | 0 |

`weight` and `new_mask` likewise identical in every cell. The NEW signature called explicitly
at its OFF defaults (`anchor_mask=None, anchor_value=None, anchor_share_out=None`) returns
`49bbdfd293d72be9` / `7d90ea7c453abef5` / `f516d58c23fed492` / 4 — **bit-identical to the old
positional call**. (Mechanically: the anchors enter only through `np.where(anchors[t], …, x)`
with `anchors` all-False, which selects `x` unchanged; `share` is computed into a local that is
dropped when `anchor_share_out is None`.)

`ValueTerms._win_prob_loss`, B=512, `manual_seed(777)`:

| call | old | new |
|---|---|---|
| `(logits, target, mask)` | `1.2312099933624268` / `a97e49f47e696982` | identical |
| `+ margin` | `1.2312099933624268` / `a97e49f47e696982` | identical |
| `+ strata_w, opp_class` | `0.3078024983406067` / `fb62a7b537a48df0` | identical |
| `rollout_w=None` | n/a (TypeError) | `1.2312099933624268` — equals the unweighted expression |
| `rollout_w=ones(B,1)` | n/a | `1.2312099933624268` — equals it too |

Metric key sets are identical wherever `row_w is None` (the control's and arm 8's case).

### §4 `WinProbLabelCallback._on_rollout_end`, driven

24 × 6 buffer, `default_rng(555)`, **26 finite terminals written AFTER `_on_rollout_start`**
(the 2026-09-10 vacuous-pass hazard), non-degeneracy asserted (mask sum > 0, target not
constant, and λ=0.9 ≠ λ=1.0). `policy._critic_mode = "winprob"` — without it `_lambda_config`
refuses and the whole λ half is vacuous; the first run of this probe hit exactly that and was
rejected, see hazard 4.

| cell | `win_target` | `win_mask` | identical |
|---|---|---|---|
| λ=1.0 bootstrap | `40b16fe55d8fda64` | `58f979deb50b358a` | ✅ |
| λ=0.9 bootstrap | `e8ca4f8cf19c7ec1` | `64e140dac713e7a8` | ✅ |
| λ=0.9 mask | `5455da0b7b41e93a` | `58f979deb50b358a` | ✅ |
| λ=0.95 bootstrap | `6bce44b6091c1b4d` | `64e140dac713e7a8` | ✅ |

`lambda_metrics` identical in every cell (10 keys, e.g. at λ=0.9: `lambda_loss`
0.9910040476109365, `lambda_target_shift` 0.45008577567963337, `lambda_weight_mean`
0.5388877123677046, `lambda_bootstrap_frac` 0.4084507042253521, `lambda_rows` 71.0).

**The ONE genuine difference on arm 8's path** — see hazard 1.

### §5 Observation space

A REAL `Gen3Env` (`start_listening=False`, no server) at each argv's emit set:

- control: **23 keys**, arm 8: **23 keys** — key sets, shapes, dtypes and bounds all identical
  across the two trees. `win_row_w` is **ABSENT** in both.
- Forced `emit_win_row_weight=True`: the old tree raises
  `TypeError: SinglesEnv.__init__() got an unexpected keyword argument 'emit_win_row_weight'`;
  the new tree adds exactly one key, `win_row_w: Box(0.0, inf, (1,), float32)`.
- `env_factory.create_training_env_random(...)._init()` executed with `Gen3Env` replaced by a
  kwarg recorder: **`emit_win_row_weight = False` for BOTH argvs**, every other emit flag
  byte-identical to the old tree's.

### §6 model_config migration + the structural gate

`MODEL_CONFIG_VERSION` **117 → 119**; `ARCH_SIGNATURE` `gen3_critic_route_wave_v1` unchanged;
recorded fields **139 → 143**.

- **synthetic v117 → 119** with `win_prob_rollout_target 0.0`, `win_prob_rollout_r 8`,
  `win_prob_rollout_mode "replace"`, `win_prob_rollout_weight 1.0`. ✅ exactly as specified.
- `ai_v12_19_ladder_lambda09` (arm 8), `ai_v12_11_ladder_ctrl10M`, `ai_v12_20_ladder_denseaux`:
  all on disk at 117, all migrate 117 → 119, all gain exactly the four OFF fields, `dense_aux`
  and both λ fields preserved.
- `check_compatible`, built from arm 8's OWN argv against arm 8's saved config:
  **PASS in both trees, both directions** (`current.check_compatible(saved)` and
  `saved.check_compatible(current)`); `current.check_compatible(ctrl10M_saved)` **PASS** in both.
  None of the four new fields is in `_WEIGHT_FIELDS`, so the gate cannot reject on them —
  unlike v117's `dense_aux`, which could.

### §7 The value sidecar

`SIDECAR_SCHEMA` **2 → 3**; `SCHEMA_EQUIVALENCE` `{1,2}` → `{1,2,3}`; `QUANTITY_FIELDS` and
`TARGET_IDENTITY_FIELDS` each gain the three rollout-TARGET fields.
`win_prob_rollout_weight` is deliberately NOT a quantity field (it prices rows, it does not
change what `target` holds) — verified present in neither tuple.

- `same_quantity(arm-8's REAL v2 header, a v3-at-target-0 header)` = **True** in both trees.
- `target_is_outcome(arm 8 v2)` = **False** in both (λ=0.9), `describe_target` string identical.
- Both debug smokes wrote a real `rows.jsonl`: header schema 2 (old) vs 3 (new), with exactly
  the three extra header fields; **row key sets identical**, `same_quantity(old, new) = True`.

### §8 End-to-end `--debug` smokes (execution, not bit comparison)

Both trees, `--run-dir` under the job tmp, `GEN3AI_MODELS_DIR` redirected, CPU/serverless:

| config | old | new | TB scalar tags |
|---|---|---|---|
| arm 8 (`--critic winprob --win-prob-mode shaping --win-prob-lambda 0.9 --value-sidecar on`) | `Training complete` | `Training complete` | **153 = 153**, none added, none removed |
| control (same without `--win-prob-lambda`) | `Training complete` | `Training complete` | **143 = 143**, none added, none removed |

Both print `[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: (1, 512))`.

### §9 `checkargs` — VERBATIM

See `checkargs_*.txt`. Summary: arm 8's recorded `original_command` validates identically in
both trees; arm 10's argv launches on `40bf23d9` and is REFUSED on `46ca68ef` — the gate bites
in both directions.

## HAZARDS (each one a FINDING)

1. **`model._win_prob_terminal_outcome` changes dtype, float32 → float64.** The old code
   published `wt[:, :, 0].copy()` (float32); the new one publishes `np.asarray(terminal_y,
   dtype=np.float64).copy()`. The VALUES are bit-identical after normalisation
   (`149d03a43ad92871` / `df97fd35718ff46b` in both trees) and the stash's only consumer,
   `ValueSidecarCallback._outcome_columns → "outcome": float(y_arr[t, e])`, casts through
   Python `float` — an exact float32→float64 widening — so the emitted JSON is unchanged
   (verified: the first 8 values match exactly, and both smokes' row key sets and header
   `same_quantity` agree). **Named, not elided**: this is a real byte-level change on arm 8's
   path (λ < 1.0 only; at λ = 1.0 the stash is never written).

2. **`_on_rollout_start` now writes `model._win_prob_rollout_metrics = None` on every rollout**,
   including at the default fraction 0.0. A default-path write with no semantic effect — the
   same shape as v116's `_win_prob_lambda_metrics = None` and v117's
   `_win_prob_terminal_outcome = None`. Third in a row; the pattern is now a habit worth a
   convention rather than three separate hazard lines.

3. **`win_prob_rollout.py` is now imported at MODULE scope by `win_prob_callback.py`** on every
   win-prob run, flag or no flag. AST-checked: no module-level side effects (numpy, two
   intra-package imports already in the graph, nine constants). The labeller and the worker are
   imported lazily inside the guarded branch. Benign, but it is a new import-graph edge on the
   default path and a forkserver-preload surface the `--debug` smoke does NOT test (per
   CLAUDE.md, the first two minutes of a REAL launch are the only test of that layer).

4. 🚨 **A VACUOUS PASS WAS CAUGHT AND FIXED MID-TEST.** The first version of the callback
   fixture set `model.policy = None`; `_lambda_config()` requires
   `is_winprob(policy._critic_mode)`, so it returned `None`, the λ recursion never ran, and
   λ=0.9 produced byte-identical output to λ=1.0 — a green result that proved nothing about the
   λ path, i.e. nothing about arm 8. The fixture now carries a `_critic_mode = "winprob"` policy
   stub and ASSERTS both that `_win_prob_lambda_metrics` is non-None and that λ=0.9's label
   differs from λ=1.0's. **This is the second consecutive increment in which the callback
   fixture passed vacuously on the first attempt** (2026-09-10's was the NaN-filled scratch).
   Any repeat of this method must assert the precondition, not observe it.

5. **A new TB series `win_prob/row_w_mean` appears whenever the composed row weight is live.**
   `strata_row_w_mean` is preserved at the same value (0.25 on the fixture), so the strata arm's
   own series is intact — but a re-read of **arm 7 (`--win-prob-strata-weight`)** on this code
   would gain a series it did not have. Not a training-semantics change (neither the control nor
   arm 8 has a non-None `row_w`; both smokes' tag sets are unchanged), but it is a metric-surface
   change for one earlier arm.

6. **Meter comparability is a SEPARATE question and this test does not answer it.** Three
   commits in the span (`a0e72b38`, `579c5ceb`, `322acaee`) change `src/main/ops/` — `critic_read`
   tool v4→v5, the conditioning meters, the new `calibration_slope` and `team_conditioning`
   modules. The ledger records that every read was regenerated on one tool version, but that is
   a fact about the reads, not about this increment. Re-ask it on every meter change.

7. **`SIDECAR_SCHEMA` 2 → 3 means arm 10's sidecar is a v3 at `win_prob_rollout_target > 0`, and
   `same_quantity` will then REFUSE to pool it with arm 8's v2 file** — correctly, because on a
   subsample the `target` column genuinely holds a different quantity. That is the intended
   behaviour, but it is a constraint on the arm-10-vs-arm-8 read: any sidecar comparison between
   them has to be done on rows the refusal permits, or with the refusal explicitly reasoned past.

8. **`checkargs` on arm 8's recorded `original_command` never reaches `40bf23d9`'s parser.** The
   command carries `--pin-commit 28ece02a`, so checkargs validated against the PINNED commit —
   and there it could not import the pinned tree (`BaselineError: no baseline registry at
   /tmp/pinned-argv-…/designs/baselines.json`) and fell back to a **best-effort static AST scan,
   explicitly labelled NOT authoritative**. `designs/baselines.json` DOES exist at `28ece02a`
   (`git cat-file -e` confirms), so the temporary checkout checkargs makes is incomplete — a
   tooling defect, not a repo one. Worked around here by also running the argv with
   `--pin-commit` stripped, which exercised each tree's own real parser (identical output).

9. **`original_command` is a COMMAND, not an argv.** Passing it verbatim to `--argv` made
   checkargs report `unconsumed value '/home/goodlad/dev/gen3ai/src/main/launcher/__main__.py'
   — a flag's ARITY differs at this commit`, which reads like a real arity regression and is
   not one. Strip `argv[0]`.

10. **Scope, stated.** Construction, the forward, the extractor parameters, the recorded config,
    the checkpoint load and its gate, the λ recursion, the BCE, the label callback, the env obs
    SPACE (not per-step obs VALUES), the argv surface, the sidecar identity, and two end-to-end
    `--debug` smokes. **No bit comparison of a real rollout** — that method is void. **No test of
    the forkserver preload / `--compile-*` / warm-start layer**, which `--debug` bypasses
    entirely.

## FILES HERE

- `probe.py` — the probe, run once per tree.
- `old.json` / `new.json` — the two blobs.
- `compare.py` / `diffs.json` — the deep diff (84 differing paths, all enumerated above; 2 of
  them are `_explicit_flags` frozenset-repr ORDER, proven set-equal).
- `checkargs_*.txt` — verbatim checkargs output.
- `smoke_*.log`, `smokectl_*.log`, `smoke_*/`, `smokectl_*/` — the four `--debug` smokes.
- `results.json` — the machine-readable summary.
