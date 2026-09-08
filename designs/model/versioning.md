# Model versioning — the package, the sanitizers, and the resume-immutable families

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## The `model_version` package, and what a save writes

**`model_version` is a PACKAGE** (2026-08-23; it was a single 2,000-line file, exactly at the size
gate's hard bound). `agents/model/model_version/__init__.py` is a pure re-export hub, so every
`from agents.model.model_version import <name>` across the ~48 import sites resolves unchanged:

| module | holds |
|---|---|
| `constants.py` | `MODEL_CONFIG_VERSION` · `ARCH_SIGNATURE` · `ModelVersionError` · the reward-immutable field table + `_reward_flag_repr` |
| `migrations.py` | `MIGRATION_FLOOR` · `SIGNATURE_FIRST_VERSION` · `_migrate_config`, **including the PRE-FLOOR HISTORY archive** (a deliberate record of what every deleted branch did — do not trim it) |
| `fields.py` | `ModelVersionFields` — the dataclass field block alone. Declaration ORDER is the constructor's positional order and `asdict()`'s key order |
| `construct.py` | `from_layout_and_policy_kwargs` |
| `compat.py` | `check_compatible` — the gate that runs on **every** load |
| `resume_checks.py` | `check_opponent_compatible` + the six resume-immutable hparam gates |
| `spec.py` | `ModelVersion` = the fields plus one mixin per family, and the JSON IO |

`ModelVersion` is assembled from MIXINS, which trades a file-size problem for a **base-list**
problem: a family can drop out of the bases without any import failing, and the class would still
construct, still round-trip through JSON, and simply stop gating. `model_version_hub_contract_test.py`
pins the base list, every gate method by name, the hub's pre-split export list (recovered by AST),
the no-submodule-imports-its-own-hub cycle guard, and that the migration archive survived.

Every model save writes the **run-level** `model_config.json` + `metadata.json` at the run root via `save_model_snapshot()`, plus a **per-checkpoint** `.json` sidecar beside each checkpoint `.zip` (`write_checkpoint_metadata`, derived from the zip path). Periodic + forced checkpoints `.zip` live in `<run>/checkpoints/` (so their sidecar lands there too); the run-level config/metadata stay one level up at the run root. Loading goes through `load_model_snapshot()`, which resolves the zip then searches **its dir AND its parent** for `model_config.json` (so the run-root config is found even when the zip is in `checkpoints/`; `load_foreign_opponent` does the same) and runs `check_compatible()` before `MaskablePPO.load()` — a mismatch fails fast with a clear error rather than silently loading bad weights. (`snapshot_history` keys + the `worktree.py` resume lookup stay BARE basenames, e.g. `checkpoint_123_steps.zip`, regardless of the subdir.) Both files carry a top-level **`num_timesteps`** — how far the run had trained at that write, "latest" at the run level (overwritten every save, unlike the immutable `original_command` / `lineage` / `pin_history`) and per-checkpoint in each sidecar, so the JSON-only offline tools (`main.lineage`, `main.sidecar_audit`, `main.dose`) can read a run's step count without opening a `.zip`; a run that predates the key is ABSENT, which reads as unknown and never as 0.

## Judging a deleted extractor kwarg, and the two sanitizers

1. **Judge it**, into exactly one of `snapshot._DEAD_FEK_*`:
   - **`_DEAD_FEK_INERT`** — no value of it selected anything in the surviving forward (it only
     SIZED or INITIALISED a deleted module, was training-only, or its branch was unreachable in
     production). Popped unconditionally.
   - **`_DEAD_FEK_JUDGED`** — some value fed a forward this code can no longer reproduce. Record the
     one value that IS still reproducible; every other value is REFUSED loudly. Two things put a
     flag here: a byte-identical state_dict across its values (nothing shape-based can catch the
     swap), or an ON value that named PARAMETERS (popping it hands SB3 an unplaceable state_dict).
2. **`_migrate_config`** needs a matching entry only if the name could still appear in a config at
   or above `MIGRATION_FLOOR`; below the floor the blanket PRE-GENERATION refusal already owns the
   config half. That asymmetry is pinned by `dead_kwargs_sanitize_test.py`.
3. **Update `ctor_kwarg_snapshot_test.CTOR_KWARGS_V96`** — last, and only after steps 1–2.

That snapshot is the tripwire, and it exists because the machinery cannot detect this failure about
itself: the sanitizer is a CURATED list, so a forgotten name produces no error at deletion time and
no failing test — only a bare `TypeError` months later. **Measured 2026-08-17 over the 89 archived
runs carrying a checkpoint:** 23 distinct rejected kwarg names, **five in neither list**
(`mask_incoming_damage_obs` / `mask_active_move_scalars_obs` / `mask_move_effects_obs` at v48,
`hp_type_belief_mode` at v52, `spread_belief_nature_marginalize` at v66), present on **70 of the 89
runs**; 7 of them (generations `ai_v9_01`–`ai_v9_07`) reached the TypeError rather than a judgment.
All five are JUDGED now and the archive is at 0 TypeErrors.

**Do not confuse this with the prober's sanitizer.** `main.prober.model.sanitized_load_custom_objects`
is pure set math over the live signature and NEVER refuses — reading an archived model may be
approximate as long as it SAYS SO (`dropped_kwargs` rides the drift banner). The curated one serves
the paths where being wrong corrupts something, so it refuses (69 of 89 runs). Both are needed;
neither delegates to the other.

## The resume-immutable hparams, and the reward-config family

**Resume-immutable training hparams (value-meaning, NOT weight-shape).** A hyperparameter can
be wrong-to-change-mid-run without changing any weight shape — `vf_coef` (`--vf-coef`) is the
first: it rescales the value head's gradient on the shared trunk, so a forgotten/typo'd flag on
resume would silently drift training. These are recorded on `ModelVersion` (→ `model_config.json`)
but **deliberately excluded from `check_compatible`** — that gates EVERY load, including the frozen
eval / self-play-pool / distill opponents, where the forward is identical regardless of the value
and a false rejection would break league play. Instead they get a dedicated check
(`ModelVersion.check_vf_coef`) invoked **only on the training-resume path** via
`load_model_snapshot(..., enforce_vf_coef=…)`; `train_rl_agent.py` FATALs on mismatch exactly like
an arch error. To add another such hparam, follow the optional-feature playbook in
`src/agents/model/CLAUDE.md` (field +
`MODEL_CONFIG_VERSION` bump + `_migrate_config` default) **plus** a dedicated `check_*` + an
`enforce_*` opt-in on `load_model_snapshot`, and leave it out of `_WEIGHT_FIELDS`.

The **reward-config** hparams are the same kind, bundled into one check: `bias_additivity`
(`--bias-additivity`), `mat_alive_weight` (`--mat-alive-weight`), `bias_redesign` (`--bias-redesign`),
`switch_bias_weight` (`--switch-bias-weight`, the belief-risk stay-into-KO BIAS lever, v5),
`draw_penalty` (`--draw-penalty`, the DRAW/250-turn-timeout terminal, v7 — **DEFAULT −35.0**, so a
stall-to-cap is strictly worse than a clean loss; `-30` restores the historical value, where a tie
scored as a decisive loss), `self_ko_hp_penalty`
(`--self-ko-hp-penalty`, the HP-scaled self-KO penalty — default 0.0 = OFF; >0 charges −w·hp when
our mon self-KOs via Explosion/Self-Destruct, since the symmetric material PBRS prices a healthy 1-for-1
trade at ~0 and the critic then over-values it), the de-bias cleanup pair `drop_redundant_bias` +
`drop_switch_bias` (`--drop-redundant-bias` / `--drop-switch-bias` — zero the audit-flagged
distorting BIAS terms: stall_tax + matchup_penalty redundant with the no-progress clock/`--draw-penalty`
and `pbrs_belief`; the hand-coded switch subsidy), and the **two end-state PBRS switches**
`all_shaping_pbrs` (**DEFAULT ON**) + `stall_pbrs` (default off) plus `no_progress_penalty`
(`--all-shaping-pbrs` / `--stall-pbrs` / `--no-progress-penalty`):
`all_shaping_pbrs` = "everything but stall" — folds
Φ_hazard/Φ_boost/Φ_opp_boosts + Φ_status and **zeros every BIAS term except the anti-stall tilt
`no_progress_tax`** (so all non-stall shaping is policy-invariant; the bad turn-ramp `stall_tax` is
zeroed); `stall_pbrs` = "stall" — folds Φ_progress and zeros `no_progress_tax`+`stall_tax`. Run BOTH ⇒
the whole BIAS class is zero (TERMINAL + PBRS only); run only `all_shaping_pbrs` ⇒ keep the
`no_progress` stall tilt as the single acknowledged BIAS. `no_progress_penalty` is recorded+checked
because it is Φ_progress's weight. (`--all-shaping-pbrs` ALSO now folds the DEDICATED phaze-out-boosts PBRS
**`pbrs_roar`** Φ_roar = −`ROAR_BOOST_WEIGHT`(0.25)·Σmax(0,opp-active-boost) — NO separate flag/field, it
rides the existing `all_shaping_pbrs` toggle, stacking with the bundled `pbrs_opp_boosts` for stronger
proportional roar-out-boosts shaping; safe since both telescope to 0.) All are recorded on
`ModelVersion` and enforced on resume by **`check_reward_config`** (FATAL on drift, since they silently
shift the reward/objective), excluded from `check_compatible`. They are reward-VALUE changes — **no
`ARCH_SIGNATURE` bump** (the network/obs are unchanged) — so a fresh run is needed to measure them but
old checkpoints don't fail an arch check — a fresh run is needed to measure them.

🚨 **Two of these defaults FLIPPED on 2026-08-18** (`all_shaping_pbrs` false→**true**,
`draw_penalty` −30.0→**−35.0**), restoring the validated ai_v8 composition after the ledger recorded
that the flag had silently stopped being passed at the v8→v9 generation boundary. Consequences that
belong to THIS file: (1) the `ModelVersion` field defaults and `_REWARD_IMMUTABLE_FIELDS`'
per-field fallbacks track `RewardConfig`'s, so a version built with `reward_config=None` records
what a default run actually trains with — pinned by `src/main/reward_defaults_test.py`; (2) every
pre-flip run now FATALs on a FLAGLESS resume, which is correct (a live run's reward must never flip
under it) and is why `check_reward_config`'s error NAMES the flags to re-pass
(`--no-all-shaping-pbrs --draw-penalty -30.0`) rather than only printing a diff; (3) frozen
eval/pool/distill opponents are untouched, because reward fields stay out of `check_compatible`.
The composition each config resolves to — and the announcer that states it at launch — is in
`designs/training/reward.md` → *The reward COMPOSITION*.

## Where the per-version entries went

**The per-version entries that used to live here have moved to `designs/CHANGELOG.md` §4**
(verbatim). They described what each of v6–v57 added, in parallel with the root `CLAUDE.md`'s own
version narrative — two records of the same history that had drifted out of agreement with each
other and with the code.

- **What the architecture IS right now** — obs layout, the phase chain under the production config,
  what each head consumes, the `DamageOperator` block, the edge families, the flag table with
  `INERT` markings: **`designs/ARCHITECTURE.md`**.
- **What each version changed**: `designs/CHANGELOG.md` (history — do not quote as current).
- **The live values**: `MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` in `model_version/constants.py`. Read them
  there. This file deliberately no longer states them: a version number written into prose is stale
  the moment the next one lands, and quoting a stale one is how a v30 description got applied to a
  v59 model.

## The `--critic` route has no fallback, its version gate, and how the mode is threaded

**The `winprob` route has NO FALLBACK, for `value_from_dist`'s exact reason** (the v89
orphaned-route class): `value_net` is in no loss graph under this critic, so quietly returning it
would be a critic the training loop believes in and nothing updates. A missing head, an un-stashed
`last_win_prob_logits`, or a batch-size disagreement with `latent_vf` all RAISE.

🚨 **The version gate matters more here than for a typical structural flag, and the reason is
worth internalising: BOTH routes return a `[B,1]` float tensor.** A flipped `critic` produces no
shape error anywhere, no load failure, and no metric that changes name — the run simply predicts a
different quantity for the rest of its life. So the string compare in `check_compatible` is the
ONLY thing standing between a resume and that, which is the same argument `win_prob_mode` and
`q_winprob_mode` make and the reason all three are gated identically.

**NO `ARCH_SIGNATURE` bump at v109, and that is the safety rule rather than a convenience.**
`shaped` is the DEFAULT, so on every run that does not type the flag no module is added or removed,
no `state_dict` key moves, the constructor's init RNG stream is untouched and the forward is
byte-identical. The signature bump belongs to the DEFAULT FLIP — where it is *forced*, because a
critic trained to predict a shaped return cannot be warm-started into predicting a probability.

`critic` is threaded as a POLICY kwarg (the `use_popart` / `value_from_dist` class), which is why
it is absent from `agents/model/flag_registry.py`: that registry's declared scope is EXTRACTOR
architecture toggles, and this one reaches no extractor — the heads it selects between were already
built by their own flags. It rides `snapshot.current_model_version(critic=…)` and
`arch_toggles_from_model` so a frozen eval / pool / sentinel opponent's load gate sees it.
