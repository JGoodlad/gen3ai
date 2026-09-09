# Training — eval and rating

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.** *THE BASELINE REGISTRY* below was added
in the **2026-09-08** second pass.

---

## THE BASELINE REGISTRY (`baselines.py` · `designs/baselines.json` · `python -m main.baselines`)

**A baseline is the thing a result is read AGAINST, and this module is the ONE accessor over the
named set.** `gen3_baselines_registry_v1` (2026-09-06) — before it, "production" was a hand-copied
JSON nothing consumes at launch, THIS package's untaught meter kept its fixed opponent as a string
literal (`DEFAULT_OPPONENT = "ai_v9_29_rev1_0823/snapshots/…"`), the famine comparator and its floor
lived in one ledger entry, and the curated TensorBoard set was decided by asking. Torch-free and
offline: it reads JSON, and only `resolve()` touches `models/`.

**Every entry is EXPLICIT** — a `.zip`, a `.json`, or an `@step`, never a bare run directory — so
`gen3_last_snapshot_resolution_v1`'s last-snapshot rule cannot move what a name points at while its
run keeps training. `resolve()` therefore always lands on the `explicit_zip` / `explicit_step` rung,
and that is asserted rather than assumed.

```python
from agents.training import baselines
baselines.get("v9_fold_parent").spec        # "ai_v9_59_R2ACTION_0827/final_model.zip"
baselines.resolve("v9_fold_parent")         # through fixed_opponent_pool.resolve_model_ref
baselines.describe("famine_comparator")     # the line every consumer prints
baselines.get("famine_comparator").floor_elo   # 38.0 — the bar travels with its comparator
baselines.protected_files()                 # {run: [rel path]} — the grooming keep-list
```

**Consumers in this package and its CLIs, all accepting a NAME wherever they accept a ref:**
`main.untaught_meter`'s `--opponent` / `--config` (their literals are GONE — the engine exposes
`default_opponent()` / `default_config()` and `resolve_ref` expands a name), `--baseline` and
`--control` through the same path; `main.critic_gate --parent` and its new
`--famine-comparator` (default the `famine_comparator` baseline, whose `floor_elo` is the kill
floor — and **an absent DEFAULT comparator is recorded as NOT READ rather than refusing the whole
read**, since `models/` is not committed and one endpoint of five must not take the other four down;
an explicit one still refuses); `main.elo`'s positional run dir; `main.tb_curate`, which unions the registry's `tb_curated`
list into the curated logdir. **Each prints `baseline <name> = <run>@<step> (set <date>, <ledger
title>)`** — a reader must never have to recognise a path.

🚨 **A NEW OPPONENT IS A RE-MEASUREMENT, NOT A RENAME.** Untaught-meter levels are not comparable
across opponents, so re-pointing `untaught_meter_opponent` invalidates every banked level measured
against the old one. That is exactly why it is a registry entry with a `set_by` ledger title rather
than a constant somebody can edit: `python -m main.baselines set <name> <run>/<file>.zip --reason
"<ledger entry title>"` rewrites the entry with a freshly computed sha/commit/version and PRINTS the
ledger line to append. It never edits the ledger — append-only, and the WHY is the one field no tool
can author.

**Validation is a test in the routine suite** (`src/main/baselines_test.py`, unmarked): every named
file exists, every sha matches, `config_version` / `arch_signature` are re-read from the run's own
`model_config.json`, and the `production` entry's declared CONSTRUCTION matches
`designs/production_config.json`. Archive-backed checks skip through `main_models_dir()`; the
structural half runs everywhere. `designs/research_state/measurements/archive_grooming_tiers.py`
reads `protected_files()` so a registry-named checkpoint survives every retention tier.

## Bot evaluation (subprocess, non-blocking)

**Flat schedule, full roster.** Eval fires every `EVAL_FREQ_STEPS` (2M steps) and plays
`EVAL_GAMES` (100) games per opponent — overridable per run with `--eval-games N` (threaded to both
callbacks via the `_schedule()` seam; n=100 → ±0.098 per-cell 95% CI, n=200 → ±0.069; the recorded
`n_games` tracks the actual cycle size) — one cadence, one game count, applied uniformly to
every bot *and* every self-play sentinel (no maturity tiers, no per-opponent caps). The
roster is the full set of eight archetype bots — both the v1 and v2 of each
(`heuristic`/`heuristic2`, `staller`/`staller_v2`, `aggressive`/`aggressive_v2`,
`setup_sweep`/`setup_sweep_v2`) — plus `random` as the eval-only "is-the-model-broken"
floor (excluded from `win_rate_vs_bots`). All nine are the single source of truth in
`_EVAL_OPPONENT_SPECS` / `eval_opponent_names()`, shared by the bot path, the self-play
path, and the worker. There is no roster flag — every bot always plays, because they play
differently and the playstyle diversity is the point. The flat numbers are safe precisely
because eval is non-blocking and **skips a cycle while the previous one is still running**
(below): a heavier roster self-throttles to a sparser cadence instead of needing tuned
ceilings.

### ⚠️ GLOBAL-RANDOM COUPLING — the five seeds a paired-arm design must set

A drawer that reaches into a **process-wide** RNG couples itself to every other drawer in the
process. Two players interleave their `choose_move` calls inside one battle; two paired **arms**
interleave them *differently* (the searched arm awaits an executor, the control runs inline). So a
decision that consumed the shared stream lands differently in the two arms **with no treatment
involved**, which is precisely what a paired design claims cannot happen.

**It is measured, not hypothetical**, and it was found by a FAILED INTEGRITY CHECK rather than by
review. The transfer-coefficient cell
(`designs/research_state/measurements/transfer_coefficient_cell_2026-08-29.md` §4) ran a paired-arm
falsifier whose zero-overrule units MUST be the same battle in both arms. It passed **exactly** on
the deterministic bots (2,693 pairs, A−B = 0.0000, **zero** divergences) and failed on **exactly the
two stallers** (755 pairs, 4 divergences), whose Protect coin (`_PROTECT_PROBABILITY = 0.6`) came
off the global module. Unbiased noise (3 favoured A, 1 favoured B), but it widens every paired
interval for free.

⚠️ **"The two stallers are the roster's only source of randomness" was WRONG, and this doc said it.**
The follow-up census (`designs/research_state/measurements/global_random_sweep_2026-08-30.md`) found
**four more** and the stallers were the *smallest*. The falsifier above could not have caught the
biggest one: it conditions on zero-overrule units, and the overrule rate against `random` is 1.00,
so that bot contributed **no units at all** — a subject a falsifier gets zero units from has not
been exonerated by it.

| what draws | when | seed kwarg | env hook |
|---|---|---|---|
| **every player's** `choose_random_move` + `DEFAULT_CHOICE_CHANCE` (`poke_env.Player`) | per decision | `rng_seed=` | **`$GEN3AI_PLAYER_SEED`** |
| the **team draw** (`Gen3Teambuilder`) | per battle | `rng_seed=` | **`$GEN3AI_TEAM_SEED`** |
| the **policy's action sample** (`RLPlayer`, torch's default generator) | per decision, when `stochastic` | `policy_seed=` | **`$GEN3AI_POLICY_SEED`** |
| the **self-play pool draw** (`SnapshotPool.sample`) | per episode | `rng_seed=` | **`$GEN3AI_POOL_SEED`** |
| the two stallers' **Protect coin** (`agents/opponents.py`) | conditional | `protect_seed=` | **`$GEN3AI_STALLER_SEED`** |

The first is the widest: `choose_random_move` is `RandomPlayer`'s *entire policy*, the fallback of
all sixteen scripted bots, and `DEFAULT_CHOICE_CHANCE` fires inside the RL players too — so even an
all-deterministic-bot roster has a shared-stream consumer in it. The third is the one a
`random`-only grep never finds: torch has its own process-wide generator, and `stochastic=True` is
the **default** for the pool and stable cross-run opponents.

**Every fix is OPT-IN and every default is unchanged, byte-for-byte.** With no seed by either route
the RNG *is* the `random` module (or torch's default generator), so the call site makes the same
call on the same stream in the same order; an unseeded instance does not even carry the attribute.
An unparseable env seed **raises** rather than falling back — a seed that was meant to be set and
silently was not would make an arm look reproducible while it is not.

**Any paired-arm design over battles should set all five:**

```bash
GEN3AI_PLAYER_SEED=1 GEN3AI_TEAM_SEED=2 GEN3AI_POLICY_SEED=3 GEN3AI_POOL_SEED=4 \
GEN3AI_STALLER_SEED=5   <harness>
```

Measured stake — 2 real bridge battles per arm under the **same fixed sim seed**, arm B burning
1234 unrelated global draws first: **unseeded the arms played different games** (84/145 turns vs
212/233, different winners); **seeded they were identical battle for battle.** A fixed sim seed
bought nothing on its own.

Caveat: a flat seed makes two instances draw the same *sequence* — reproducibility, not
independence (their decisions still differ, because their legal-order lists do). Pass distinct
`rng_seed=` values where the two sides must be independent as well.

Tests: `global_random_coupling_test.py` (47, all four new seams) and
`opponents_test.py::TestStallerProtectRng`. Each seam carries a **revert arm** — unseeded, the same
interleaving pulls the two apart — so if that ever passes, the per-instance RNG has stopped being
the difference and the rest of the suite is asserting nothing.

### The untaught meter (`untaught_meter.py` · CLI `python -m main.untaught_meter`)

**The first in-tree consumer of all five seeds, and the meter every fold verdict in the ledger rests
on.** It plays a checkpoint PILOTING a fixed team slice against ONE fixed opponent and reports a
cluster-bootstrapped win rate. Offline — no training, no launcher, no server, nothing written under
`models/`.

**THE RECIPE**, which is the banked probes' recipe with one declaration instead of five copies:

| | |
|---|---|
| teams | `--teams` a manifest JSON, **in order — the order IS the seed** (index = team seed offset). Default: the untaught 8 (`reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json`). `--taught` swaps in the taught 16 (`teacher_content_2x2_2026-09-04/taught_teams.json`). Both `pin_sha` (raw bytes, the MatchupSpec convention) and `team_sha` (strip-normalized, the archetype-artifact join key) are recorded per team — they DIFFER on a file with a trailing newline |
| refs | resolved through the **imported** `fixed_opponent_pool.resolve_model_ref` — the same call `main/train/model_build.py` makes for a `--distill-teacher`. A bare run dir therefore means the run's **LAST SNAPSHOT**, and the resolved file + `rung` + `rule` + `num_timesteps` are printed per ref and stamped in the JSON, so no reader has to infer WHICH FILE was scored |
| opponent | one fixed model piloting the **paired** pool draw — **BY NAME** out of the baseline registry (`untaught_meter_opponent`, rev-1's 24M snapshot). The string literal is GONE; `--opponent` also takes any other registry name or a raw ref |
| module tree | one `model_config.json` for every model (**BY NAME**, `untaught_meter_config` — rev-1's snapshot config, what the probes used; `--config auto` resolves each model's own), observation debugger stripped, `device="cpu"` |
| play | `stochastic=True` both sides · rust bridge · **`concurrency=1`** |
| aggregation | equal-weight cluster mean over TEAMS, and **ONE fixed resampling index set shared by every ref and every contrast** so a ref-vs-ref difference is paired on the same team draws |

**BOTH DEFAULTS ARE REGISTRY NAMES** (`gen3_baselines_registry_v1`), and the CLI prints `[baseline] --opponent default: …` naming the run before it resolves anything — see *THE BASELINE REGISTRY* above. 🚨 **A new opponent is a RE-MEASUREMENT, not a rename**: levels are not comparable across opponents, so re-pointing that entry is a `python -m main.baselines set` with a ledger title, never a module edit.

**THE SEEDS.** Per team, all five global-RNG seams above are set from `--seed` + the team index;
additionally both players' sampling generators are re-seeded **per battle** and the sim takes a
per-battle dice seed. **At `--seed 0` the dice, the pool draw and the policy seeds reproduce
`arch_transfer_2026-09-05/exploiter_competence/compete.py` exactly**, so a level here is comparable
to that banked one.

| stream | value |
|---|---|
| `$GEN3AI_{PLAYER,TEAM,POLICY,POOL,STALLER}_SEED` | `{10000,20000,30000,40000,50000} + 1e6·seed + team_index` |
| sim dice | `[seed + team_index + 1, battle_index + 1, 3, 4]` |
| pool sequence | `random.Random(61000 + 1e6·seed + team_index).randrange(n_pool)`, drawn **sequentially** — one `Random` per team, prefix-consistent, so a ref at 12 games/team plays the first 12 of another ref's 200 |
| pilot / opponent policy | `71000 / 72000 + 1e6·seed + team_index·1000 + battle_index`, re-seeded per battle |

🚨 **`concurrency > 1` is REFUSED** (`GEN3AI_UNTAUGHT_METER_ALLOW_CONCURRENCY=1` accepts unquotable
levels). Seeds pin the dice and both players' sampling, but interleaved battles consume the shared
streams in a **scheduling-dependent order** — measured 2026-09-03, seeded at concurrency 3 two runs
of the offline collateral-KL probe still gave 1193 vs 1141 states with arm levels up to +0.043
apart. **SHARDING IS OVER TEAMS**: `--workers N` splits the teams round-robin across N
single-concurrency child processes, which is safe because a cell is a pure function of (ref, team
index, battle index) — verified by `exploiter_competence` before it sharded 3200 battles across six
workers, and gated here by
`src/main/untaught_meter_reproducibility_integration_test.py` (`sim`+`slow`: two `--workers 2` runs,
byte-identical JSON).

🚨 **THE CONTINUATION CONTROL IS THE SECOND COLUMN, and it is not optional bookkeeping.** Ledger
2026-09-06 (cell 2) measured a plain +1.08M-step continuation of v8's parent — no teacher, no
distillation term, no stable opponents — moving this meter **+3.45pp [+0.46, +6.48]** on its own. A
delta against a **frozen** parent therefore credits a fold with progress the parent would have made
anyway; re-based, v8's celebrated +4.64pp becomes ≈ +1.2pp and is not significant. `--control
<arms…>` pools the continuation arms equal-weight and computes their **max-pairwise replicate
floor** (`|Δ|` is a magnitude; **one** control arm gives a re-based delta but NO floor, and the
meter says so rather than inventing a zero). `--floor PP` supplies the externally-ruled floor for
the *baseline* column — regime-specific (1.66pp frozen K=3 · 4.27pp controller-live), **never
pooled across regimes**. Verdicts, in order: `WITHIN FLOOR` (|Δ| below the floor — the CI may still
exclude zero, which says the games are consistent, not that the arm differs) → `NOT DETECTED` (CI
spans zero) → `SIGNIFICANT`.

**Timeouts are their own bucket**: an unfinished battle is never scored as a loss, win rate is over
FINISHED games, and a run whose timeouts exceed **25%** of attempted battles reports `INCONCLUSIVE`
with no verdict at all (CLI exit 3).

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.untaught_meter FOLD_A=<run> FOLD_B=<run> \
  --baseline <frozen parent> --control <cont A> <cont B> <cont C> \
  --games-per-team 200 --workers 6 --floor 1.66 --json out.json --md out.md
python -m main.untaught_meter <refs…> --baseline <ref> --check     # resolve only, plays nothing
python -m main.untaught_meter --from-rows A=<untaught_X_end.json> --baseline B=<untaught_Y_end.json>
```

`--from-rows` re-reads committed per-team artifacts (`untaught_<TAG>_<depth>.json`-shaped) with no
models and no battles — the `POOLED` row is a summary and is never counted as a ninth cluster. It is
what pins the aggregation to the record: `untaught_meter_test.py` reproduces the banked 2×2 endpoint
legs (`TCFUNDA−TCUNFA −4.50`, `TCFUNDB−TCUNFB −4.25`, mean **−4.37**, the ledger's funded−unfunded
untaught endpoint) and the `UNF/end` replicate draw that is the two-arm control floor.

### The CRITIC GATE (`python -m main.critic_gate`) — the meter's caller, and three others

The win-prob-critic arm's pre-registered read (`designs/ai_v12/design_winprob_only_critic.md` §5.5)
is four instruments. `python -m main.critic_gate <run> --parent <ref> --control <cont…>` runs all
four and emits ONE markdown report + a JSON, and its whole design rule is that **it composes and
never re-derives** — a second estimator for a number another tool already owns is how two published
values of one quantity start disagreeing.

| section | what it does | whose statistics |
|---|---|---|
| 1 ladder | reads both `snapshot_ladder/ladder.json` for the node LIST, then **REFITS BOTH SIDES** on their first n from their own `games.jsonl` with this tree's `fit_ladder(first_n=…)` (strict prefix) — *always*, not only when the counts differ, because a committed ladder written before 2026-09-07 carries the eval-**sentinel** bias (+21..+29 Elo on its newest nodes) and a run pinned to an older commit keeps writing it. Compares at **matched SNAPSHOT COUNT**, never matched step. A side that cannot be refit falls back to its committed ladder and is NAMED in `refit_fallbacks` + `fit_size_note`; a side with MORE rated nodes that cannot be refit is still **UNMATCHED FIT SIZE** and the famine pre-test still REFUSES on it. Prints `rating not final` while the run is unfinished | the BT fit's, re-fit on the prefix — both sides |
| 2 calibration | §4.3 G1–G4 per checkpoint, `bot`/`pool` **separately**, selection-reweighted | `main.scaffolding_gauge`'s `collect_slices` / `build_reliability` / `true_win_rates`, IMPORTED |
| 3 G7 kill | stall rate + mean episode length vs the era; episode length is read from TensorBoard `eval/mean_ep_len_vs_{bots,pool}` (every cycle) first, the `metadata.json` blocks second — the blocks are keyed by CHECKPOINT and miss any cycle no checkpoint captured (10M and 14M on `ai_v12_02`); a step with no episode length prints **ep_len NOT EVALUABLE**, never OK. 🚨 **The 1.25× threshold is defined on EVAL episode length, never on the train series `rollout/ep_len_mean`** — same name, same units, different population: on `ai_v12_02` at 18M the train series was +25% over its early value (self-play opponents improve WITH the trainee, so near-equal games lengthen) while eval-vs-bots was +3.7% and eval-vs-pool flat, so applying the eval threshold to the train series would have fired a kill on an arm at 0% of the real one | the run's own recorded metrics + its trace summaries |
| 4 untaught meter | `main.untaught_meter --baseline <parent> --control <cont…>` | the meter's, read back out of its own `--json` |

**Where the two halves of G7 come from is not obvious and is worth stating**: `eval_results.jsonl`
carries **no** episode-length field, so the full-cycle `mean_ep_len_vs_bots` / `pool.mean_ep_len`
come from `metadata.json`'s `latest_eval` and every `snapshot_history[*].latest_eval`; the stall
rate is computed off the per-battle trace summaries (`meta.turns >= MAX_TURNS`, which is also the
forfeit deadline, or a non-win/loss result) and is therefore a **CAPTURE-QUOTA** statistic, printed
with that label every time because the quota is loss-enriched by design.

**The one statistic the gauge does not publish is the RESOLUTION CI** that G1 needs. It is obtained
by calling the gauge's own `reliability_table` under `agents.training.scaffolding`'s own
`cluster_bootstrap_ci` — the same primitives, one extra pass over the gated strata only — never a
second estimator. The committed baseline publishes no interval for `resolution` either, so G1
compares the arm's CI against the baseline as a **FIXED bar**; that asymmetry is printed rather than
smoothed over.

Refusals (exit 2, key named, `main.exploitability`'s pattern): an absent or **unconverged** ladder,
an empty `ratings` block, `--at-snapshots` beyond the matched count, a missing baseline artifact, a
**raw (un-reweighted)** baseline artifact (§4.2 — the raw table inverts the verdict, so it is not a
bar), a missing matched stratum, and the gauge's own `SelectionWeightError` surfaced verbatim rather
than falling back to unweighted. `--check` resolves every input including the meter's own `--check`
and exits non-zero naming **every** miss, computing nothing.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.critic_gate models/<arm> --parent models/<parent> \
  --control models/<contA> models/<contB> --md gate.md --json gate.json
python -m main.critic_gate models/<arm> --parent models/<parent> --check   # resolve only
```

**Validated against the record**: run over `ai_v9_59_R2ACTION_0827` it reproduces §4.1's committed
table value for value (26M `all` resolution 0.0618 / reliability 0.0013 / ECE 0.0249 / skill +0.336;
28M `bot` 0.0215 / 0.0012 / 0.0228 / +0.222; …), and comparing that run against itself correctly
reports **G1 false everywhere** — a generation cannot out-resolve its own baseline.

🚨 **G2 and G3 are PER-STRATUM RELATIVE bars** (owner ruling 2026-09-06, §4.3): because the
committed baseline's `pool` stratum already breaches §4.3's absolutes (reliability 0.0064 / 0.0103
against ≤0.005, ECE 0.0667 / 0.0875 against ≤0.05, true on `bot` and pooled but not on the stratum
the criteria are registered "in both classes" over), the arm must be **no worse than the baseline's
SAME-stratum value at the matched checkpoint** (its own row at that step, else the artifact's steps
reduced by `--baseline-reduce`, default `last` — without the matched half a generation reads as
inferior to ITSELF) — PASS when its point estimate is at or below that value or its
cluster-bootstrap CI contains it (non-inferiority, never a direction claim), FAIL only when its
whole CI sits above — while the absolute numbers are still computed and printed as ASPIRATIONAL
targets that gate nothing, and every row names which clause decided it beside the baseline value it
was decided against. G1 and G4 are unchanged.

`PerOpponentEvalCallback` (non-self-play path) does **not** eval in-process. On each
scheduled step it snapshots the live weights (`model.save`) and spawns `--eval-workers`
(default 5) `main.eval_worker` subprocesses that **work-steal at battle granularity** from a
shared pool, load the **frozen** snapshot, and play against the shared Showdown server (or the
in-process bridge) **without pausing training**. **The trainee's eval teambuilder follows the
run's `--trainee-team` pin** (`trainee_team_str` in the worker cfg → `eval_worker._build_trainee_tb`;
threaded by BOTH callbacks): a specialist run is measured piloting ITS OWN team. The worker used to
hardcode the default full-pool builder, so every specialist eval (win rates / ELO / `vs_ext`
verdicts) measured the model piloting random teams it never trained on — pure OOD; the
"ai_v7_05–08 plateau" was this instrumentation gap, not the training (see `eval_worker_test.py`,
the fix's pin). No pin → the default pool builder, byte-identical. **The companion TRAINING-side bug
(the mirror):** PokeEnv feeds its single `team=` kwarg to BOTH internal env agents, and the
per-episode opponent Players are decision-functions over `battle2` (agent2 does the networking), so
agent2's `_team` decides the opponent's REAL team — a `--trainee-team` pin therefore also pinned the
OPPONENTS, turning every specialist run's training into a single-team MIRROR vs bot pilots
(genuinely-won ~100% training WRs, fake curriculum; a probe on the exact path measured the same
checkpoint at 1.000 mirror vs 0.483 with real opponent teams). Fixed by the `Gen3Env(opponent_team=…)`
post-init seam (the `_battle_class` injection pattern), threaded unconditionally from the env factory
(`opponent_teambuilder`); `None` = the pre-fix both-sides behavior. Pinned by `gen3_env_test.py`. Each opponent's `EVAL_GAMES` are split into
**shard units** of `--eval-shard-games` (default 25 → 4 shards/opponent); a worker claims units
(atomic `O_EXCL` lock per `unit_id`), plays them, and publishes one `shard__<unit_id>.json` of
**raw** counts; the parent pools an opponent's shards back into one **exact** result. This is the
long-tail fix — when fewer opponents remain than workers, the straggler's remaining games spread
across idle workers instead of one worker grinding a whole opponent alone (workers are capped by
unit count, not opponent count). The whole mechanism lives in the **`eval_sharding/` package**
(below); when all workers finish the parent merges → TensorBoard + TUI + best-model (the winning
snapshot is promoted by copy, not re-saved). Forensic traces land under
`<run_dir>/eval_traces/step_<N>/<opponent>/` as a per-captured-battle triple (`write_battle_record`,
`battle_recorder.py`): `<outcome>_s<shard>_NNN_summary.json` (the human-readable per-decision dump —
each invocation also carries a **`belief`** block, the model's top-`BELIEF_TOPK` (3) most-likely species
per still-HIDDEN opp slot, present ONLY when the hidden-opponent belief is on and a slot is un-revealed;
`RLPlayer._decode_belief` → `inference/belief_decode`, see `src/agents/model/CLAUDE.md` — and an
**`opp_intent`** block, the v67 `α`/`β` heads' read of what the OPPONENT was about to do: `α` a ranked
list of NAMED believed moves plus `SWITCH`, `β` the candidate switch-ins each named by the model's own
species posterior. Present only under `--opp-intent-coef>0`, so an intent-off run's trace is unchanged;
it is what the prober's `EXPECT` line and the web replay's per-turn *expect* line read) +
`<outcome>_s<shard>_NNN_states.npz` (raw obs/logits/values **+ the chosen `actions`** for the prober
and offline obs replay) +
**`<outcome>_s<shard>_NNN_replay.html`** — a self-contained, **browser-watchable** Showdown replay of
that battle (poke-env `save_replay` over the accumulated protocol stream). The first two are
prober-only; the HTML lets a human just open the game in a browser (no checkout, no prober) — the
only watchable replay for *non-stall* eval battles (stall games still get their own `stalls/*.html`).
The `s<shard>_` prefix namespaces the files so concurrent shards of one opponent never collide.
The filename stem is built by the single helper **`trace_filename_stem(outcome, trace_tag, idx)`**
(`<outcome>_<trace_tag><idx:03d>`) — the **one source of the naming contract** the prober's
`discovery._FNAME_RE` must invert. (When sharding added the `s<shard>_` infix, the prober's regex
didn't follow → every sharded trace parsed as outcome `"?"` and the **whole prober went blind**;
`eval_callback_test.test_trace_naming_contract` now pins that `discovery` parses exactly what
`trace_filename_stem` emits, so the producer↔consumer pair can't silently drift again.)
On a BRIDGE run each trace also gets a fourth sibling,
`…_NNN_reconstruction.json` — the battle's **full-information reconstruction record** (resolved
PRNG seed + both packed teams + the raw command log), captured at the bridge layer and joined to
the trace by battle tag (`utils/bridge/reconstruction.py`). It makes the battle fully replayable
and turn-re-rollable offline (`replay_battle` / `reroll_turn`), and
`agents.training.obs_materializer` can rebuild the trainee's one-sided obs from it bit-for-bit
(guarded by `obs_roundtrip_fuzz_test.py`). It is referee-view data in a **separate artifact** on
purpose — nothing in the obs/training path reads it (the one-sided/omniscient wall; see the bridge
README). Websocket eval simply doesn't produce it (degrades gracefully). 🚨 **THE RESULT VOCABULARY IS `WIN` | `LOSS` | `DRAW`** (`gen3_trace_result_v2`,
`agents/training/trace_result.py`, pure stdlib — one declaration the recorder writes and the
prober, the harvesters and the gauge read). `<outcome>` in every filename above is the lowercase
form, so a captured draw is `draw_s<shard>_NNN_*`. A `DRAW` carries **`meta.draw_kind`**:

| `draw_kind` | the battle layer's report | what it is |
|---|---|---|
| `timeout` | `lost` **and** `turn >= MAX_TURNS` (250) | the trainee FORFEITED at the deadline — a stall |
| `tie` | `won`/`lost` both falsy, `finished` | the sim emitted `\|tie\|` |

🚨 **THE DEFECT THIS CLOSED (2026-09-07).** A timeout arrives wearing a loss's flags — the trainee
forfeits at the cap (`inference/player._handle_stall`), so poke-env reports `lost=True` — and was
written as an ordinary `loss_*`. The **training reward never agreed**: `reward_manager`'s terminal
fold pays `draw_penalty` for exactly that state, detected by the TURN COUNT, which is also why the
G7 kill clause survived (it keys on `meta.turns`, never on the label). A true TIE was worse: it
matched neither quota branch, so its buffered capture was **dropped** — no file, no count, nothing
on disk to question. Measured over the whole archive: **145,173 traces, every one `win_*` or
`loss_*`, `meta.result` never anything but `WIN`/`LOSS`.** "0 draws in every eval trace" was what
the instrument could express, not what happened.

**The two seams that implement it.** `classify_result` tests the 250-turn cap **before** the
loss, on the same constant the reward uses (`reward_weights._TIMEOUT_TURN_CAP` == `MAX_TURNS`),
so the label and the terminal fold cannot disagree about a timeout; and the per-cycle manifest
states all three quotas in words under `forensic_selection_rule`, so a reader never has to
recognise the numbers.

**READING AN OLD TREE.** A trace written before this carries no `meta.result_vocabulary`, and that
ABSENCE is what dates it (`trace_result.result_era` → `gen3_trace_result_v1`). In that era a
timeout is separable only by `meta.turns >= MAX_TURNS` and a tie is not present at all, so such a
tree can estimate a **STALL rate** but a **TIE rate is NOT MEASURABLE** there —
`trace_result.era_note` is the one sentence that says so, printed by `run_summary()` and by the
prober's `/` page. Nothing back-fills the archive, and `to_summary` / the prober both **REFUSE** a
result outside the vocabulary (`UnknownTraceResult`) rather than coercing it. ⚠️ The live arm
`ai_v12_02_winprob_critic` is PINNED and keeps writing `gen3_trace_result_v1` until it ends.

🚨 **THE CAPTURE QUOTA IS OUTCOME-CONDITIONAL, and the manifest now RECORDS it**
(`gen3_trace_selection_manifest_v1`): `EvalRLPlayer` persists at most `_FORENSIC_WIN_QUOTA` (5) wins,
`_FORENSIC_LOSS_QUOTA` (10) losses and `_FORENSIC_DRAW_QUOTA` (5) draws per opponent per cycle
(scaled per shard unit), so the traces are a LOSS-ENRICHED sample by design — and every consumer that averages over them (`calibration`,
`falsify_scan`, `main.scaffolding_gauge`) used to inherit that skew with nothing on disk saying so.
**WHERE DRAWS SIT: their OWN bucket**, independent of both others. Folding them into the loss
quota is what the old code did, and it lets a stall storm evict the decisive losses the prober
exists to study — the loss slice would change meaning in exactly the cycles where a regression is
worth reading. Giving them no bucket is the other half of the old defect. The draw quota is set to
the WIN quota rather than the loss quota because a draw is a rate to notice, not a game to dissect,
and it bounds a pathological stall cycle at 5 extra traces per opponent.

`record_eval_selection` patches each cycle's manifest at COLLECT with, per opponent,
`battles_played` / `battles_won` / `battles_drawn` / `traces_written` / `traces_won` /
`traces_drawn` plus the derived `capture_rate_win` / `capture_rate_loss` / `capture_rate_draw` and
the rule in words; the counts ride the existing shard plumbing (`ShardResult.traces_{written,won,drawn}`
+ `n_drawn`, defaulted so a legacy shard still deserializes). 🚨 **A DRAW IS SUBTRACTED FROM THE
LOSSES, NOT ADDED TO THE PLAYED COUNT** — poke-env's `n_finished_battles` already contains every
draw (a tie is neither a win nor a loss to it, a timeout is our forfeit), so
`lost = played − won − drawn` is what keeps `capture_rate_loss` a statement about DECISIVE losses.
The drawn-battles-played denominator comes from `EvalRLPlayer.draws_seen` because **no other layer
counts it**. The `selection` block is **schema 2**, purely additive: `read_selection` accepts
schema 1 as well, so every cycle recorded before today stays READ rather than being demoted to
SELECTION UNKNOWN — its draw keys are simply ABSENT, which is not the same as zero.
**Absent reads as SELECTION UNKNOWN, never as uniform** — a legacy tree, or a cycle that crashed
before collecting (the block is written `null` at launch). One declaration,
`agents/training/trace_selection.py` (pure stdlib, imported by the prober and the gauge too), so
producer and consumers cannot drift. All
three sit alongside a per-cycle
**`eval_manifest.json`** (`write_eval_manifest`) recording exactly which model produced them
— `num_timesteps`, `git_hash` + `arch_signature` (read from the run's `metadata.json` /
`model_config.json`), and a `snapshot` pointer. The eval snapshot is normally ephemeral
(`model.save` → workers load → deleted in `_cleanup`) and the eval `step` rarely lines up with
a persisted `<run>/checkpoints/checkpoint_<N>_steps.zip`, so the prober can't reload the *exact* weights unless
they're retained: `--keep-eval-snapshots N` copies the snapshot into
`eval_traces/step_<N>/snapshot.zip` (keeping the N most-recent) and points the manifest at it.
The prober consumes the manifest to load the exact model, falling back to the nearest
checkpoint. **The trainer grooms the traces it writes**: after each cycle
`_prune_eval_traces` keeps only the `--keep-eval-trace-steps` most-recent eval
step dirs — **default `0` = KEEP ALL since 2026-09-08** (`gen3_keep_all_eval_traces_v1`) — and
`_prune_eval_snapshots` keeps the `--keep-eval-snapshots` (default 10)
most-recent snapshots (`python -m main.prober.groom` is the manual fallback).
🚨 **THE OLD CAP OF 20 DELETED A REGISTERED COMPARATOR.** Arm A
(`ai_v12_02_winprob_critic`) ran to 75M under it, so its 10M-step traces were groomed off disk
and the win-prob critic ladder's own **A@10M control became uncomputable** — the matched-step
read every ladder arm is scored against. A cycle is ~55 MB and a 75M run's full set ~3 GB: disk
is recoverable, a matched-step comparator is not. The weight snapshots stay capped at 10 because
those are ~27 MB *each* and the prober falls back to the nearest checkpoint; only the TRACES,
which nothing can reconstruct, are now kept forever. **The same cycle also bounds the run's
two append-only debug dirs** via `_prune_run_artifacts` (`artifact_retention.py`, a dedicated
module — not bolted onto this busy callback): keep the `--keep-stalls` (default 50) most-recent
`stalls/stall_*.html` replays and the `--keep-crashes` (default 10) most-recent
`crashes/restart_err_*.txt` launcher diagnostics, newest-by-mtime, `0` = keep all. Same
producer-grooms-its-own-data contract; `python -m agents.training.artifact_retention <run_dir |
models_dir> [--apply]` is the manual fallback (dry-run by default; sweeps every run under a
`models/` tree). The eval summary itself is
written to `metadata.json` as a **top-level `latest_eval`** block (step-labeled, NOT
nested under a checkpoint) — robust to the async timing (an eval can finish after a
newer checkpoint, or before any checkpoint exists); `save_model_snapshot` carries it
forward so a later checkpoint never erases it. That top-level block is the canonical,
timing-robust record; **additionally, `record_checkpoint` stamps a point-in-time copy
of the then-current `latest_eval` into each checkpoint's entry** (both the per-checkpoint
sidecar `.json` and the run-level `snapshot_history` entry, under a `latest_eval` key) so
each checkpoint carries the most-recent eval+pool stats as of when it was saved. The
embedded block keeps its own `step`, so storing it under a possibly-newer checkpoint never
mislabels which weights were measured (`snapshot._read_latest_eval` reads it; the union
builder `_build_snapshot_entry` keeps sidecar + history in lockstep).

The frozen snapshot makes parallel eval correct (a worker can't read mutating in-memory
weights), and the fresh process returns all eval memory to the OS on exit (no fragmentation
in the trainer). Behaviors:
- A trigger that fires while the previous cycle still runs is **skipped** (logged) — on CPU
  an eval can outlast its interval; cadence just goes sparser.
- A worker crash is **logged-and-continued**, never fatal (its opponents are just missing
  for that cycle).
- **The hung-cycle watchdog is CONTENTION-SCALED** (`eval_cycle_timeout()` =
  `scale_timeout(_EVAL_CYCLE_TIMEOUT_SEC)`, 30 min baseline, shared by BOTH callbacks;
  `gen3_contention_robust_timeouts_v1`). Eval is the path most exposed to load — it runs
  concurrently with training *by design*, so it is contended 100% of the time, and the bullet
  above already concedes "an eval can outlast its interval". Firing early does **not** merely
  lose a cycle: `_abort_pending_cycle` kills the workers and collects **PARTIAL** results, which
  feed `win_rate_vs_bots` (the curriculum ramp), `win_rate_vs_pool` (the promotion gate) and the
  ELO fit — and the survivors are whichever shards got scheduled, not a random subsample. So a
  merely-slow cycle must never be mistaken for a hung one. The partial-coverage warning no longer
  asserts "worker crash mid-opponent" as the cause either (an overrun-kill produces an identical
  shortfall); it states the fact and appends `describe_contention()` so the reader can tell which
  happened. ⚠️ **Tests must read `eval_cycle_timeout()`, never the raw constant** — the two
  hung-cycle tests built a past timestamp from `_EVAL_CYCLE_TIMEOUT_SEC` and so passed on an idle
  box while failing on a loaded one; `GEN3AI_TIMEOUT_SCALE=6 pytest src/ -m "not integration and
  not e2e"` is the check that catches that class.
- **An operator can force an off-cadence eval** from the launcher's `f` button (confirm →
  SIGUSR2). The signal handler (`train_rl_agent._setup_signal_handlers`) only flags a
  process-global `request_forced_eval()`; whichever eval callback is active CONSUMES it on its
  next `_on_step` (the shared `eval_callback._ForcedEvalMixin._maybe_force_eval`, mixed into BOTH
  callbacks so the path can't drift) and launches a cycle immediately. A request that arrives
  while a cycle is already in flight is **rejected** and reported to the launcher Events panel —
  the same skip-while-running rule as the normal cadence. The forced launch consumes the current
  cadence bucket (`_last_eval_step = num_timesteps`) so the schedule check can't double-launch the
  same step; the next boundary still fires normally. Tests: `eval_callback_test.py` /
  `selfplay_callback_test.py` (`test_force_eval_*`).
- **Graceful shutdown waits for eval to finish**: a scheduled restart is self-initiated by
  `GracefulRestartCallback` at a rollout boundary and the launcher won't force-kill until the
  child overruns the deadline by `--restart-grace-minutes` (20 min), so the drain budget is a
  full `_ABORT_EVAL_DRAIN_SEC` (10 min) AFTER the checkpoint is saved — long enough for a CPU
  eval to complete. Even the pathological forced-SIGTERM case (already overran → ~90s SIGKILL)
  is safe: the checkpoint is saved first, only the in-flight eval can be lost.
- **On resume the last eval is re-published to the TUI** from the resumed checkpoint's
  `metadata.json` (`replay_last_eval_to_tui`), so the eval panel isn't blank until the next
  cycle. This covers the **self-play `pool` block too** — the aggregate (`win_rate_vs_pool`,
  `mean_reward_vs_pool`, monotonicity, snapshot count) and every per-sentinel row are
  re-published from the saved block, with the saved step tags, so Pool/sentinel rows survive
  a restart exactly like the bot rows (no waiting a full cadence for fresh numbers). Safe
  because the pool only changes at an eval-collect — the same moment the block is persisted —
  so the saved rows match the pool reconstructed from `snapshots/`. A pre-seed eval persists an
  empty `sentinels` list, which isn't re-published (nothing to show yet).
- **The cadence ANCHOR is restored on resume — CLAMPED to the current step**
  (`_ForcedEvalMixin._restore_last_eval_step`, shared by BOTH callbacks). `_last_eval_step` is
  in-memory and resets each process, so it is restored from metadata; otherwise the resumed step
  sits far past a boundary and a fresh `0` would eval on step 1. That is right for a launcher
  RESTART and **wrong for a FORK**: `resume_eval_metadata` is the SOURCE run's run-level
  `metadata.json`, whose `latest_eval.step` is where *that* run last evaluated, not the step of the
  older checkpoint being forked from. Measured 2026-08-21 on an exploiter fork of gen-17's
  9,084,672-step checkpoint out of a run that reached 25M: the anchor restored to **24,000,000**,
  and since the cadence test is `(now // freq) > (anchor // freq)`, the fork would have launched
  **ZERO eval cycles** until it itself reached 26M — no `win_rate_vs_*`, no `eval_results.jsonl`
  row, no ELO. A gate arm whose verdict IS an eval metric silently produces nothing to read.
  Clamping to the model's `num_timesteps` restores the intended meaning and the next boundary after
  the fork point fires normally; a restart is unaffected (its recorded step is at or behind the
  loaded one, so the clamp never bites) and the clamp announces itself with an `anchor is AHEAD`
  event that states the FACT rather than asserting a cause — a crash-restart that rewound past a
  completed eval reads identically to a fork.
  ⚠️ It reads **`self.model.num_timesteps`**, not `BaseCallback.num_timesteps` — the latter is a
  mirror SB3 only syncs inside `_on_step`, so at `_init_callback` time it is still `0` even on a 9M
  resume, and reading it would clamp every restart to 0 and re-eval on step 1 (observed live before
  the fix, as `this model is at 0`). Same family as `_warn_if_fork_pool_empty`: a fork inherits the
  base's weights but none of its run-directory state, and the silent failures live in that gap.
  Test: `eval_fork_cadence_test.py` (both callbacks, parametrized).

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-workers` | `5` | Eval subprocesses per cycle; work-steal **shard units** from a shared pool. Capped at the unit count (≈ opponents × shards-per-opponent, so sharding lets the full pool help). Self-play doubles this (→ `10`) since sentinel matchups run the model for both players. |
| `--eval-games` | `None` (=`EVAL_GAMES`, 100) | Games per **opponent** per eval cycle. Raise for tighter sentinel/promotion CIs (200 → ±0.069) at proportionally more eval compute — work-stolen across the workers, off the training path. Shards/opponent = eval-games / `--eval-shard-games`. |
| `--eval-shard-games` | `25` | Games per work-steal **shard unit** (battle-level work-stealing). Each opponent's `EVAL_GAMES` split into chunks any idle worker drains → the long tail collapses to one shard (≈4-shards-per-opponent default = ~4× shorter tail). Smaller = finer tail collapse but more player builds / (on websocket) more connection churn — the bridge is preferred for fine shards. `>= EVAL_GAMES` ⇒ one shard/opponent = the original opponent-level behaviour. Aggregation is exact (Σwon/Σfinished etc.); see the package below. |
| `--eval-device` | `cpu` | Device for eval-worker inference. `cpu` decouples eval from the training GPU. |
| `--eval-concurrency-per-worker` | `1` | Battles each worker overlaps **within** its claimed opponent (single-thread asyncio latency-hiding — NOT multi-core). `1` = today's sequential play. Threaded to the constructor's `eval_concurrency` → `cfg["concurrency"]` → `run_local_battles(concurrency=)` (bridge) / the player's `max_concurrent_battles` (websocket). See the concurrency note below. |
| `--keep-eval-snapshots` | `10` | Retain the N most-recent eval weight snapshots in `eval_traces/step_<N>/snapshot.zip` (~27MB each; default ≈270MB) for bit-exact prober replay. `0` writes the identity manifest only; the prober then loads the nearest persisted checkpoint. The trainer auto-prunes to this cap each cycle. |
| `--keep-eval-trace-steps` | `0` (= **KEEP ALL**) | The trainer keeps only the N most-recent eval **step dirs** under `eval_traces/` after each cycle. 🚨 **Was `20`, and that default deleted arm A's 10M traces — the ladder's registered A@10M comparator — before anyone read them** (`gen3_keep_all_eval_traces_v1`, 2026-09-08). ~55 MB/cycle, ~3 GB for a 75M run. Pass a positive N to cap it again; `python -m main.prober.groom` is the manual fallback. |
| `--keep-stalls` | `50` | Each cycle keep only the N most-recent `stalls/stall_*.html` replays (`0` = keep all). A self-play run writes thousands (~80 KB each); this caps the dir. `artifact_retention.py`; CLI fallback `python -m agents.training.artifact_retention`. |
| `--keep-crashes` | `10` | Each cycle keep only the N most-recent `crashes/restart_err_*.txt` launcher diagnostics (`0` = keep all). Same module/CLI as `--keep-stalls`. |

**TD-residual tail metric (`eval/td_resid_tail_*`).** Each cycle also folds a **left-tail
statistic of the per-decision critic surprise** δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t) — the same
formula the prober uses (`main/prober/session/core.py::ProbeSession._td`, the single source of
truth). `BattleRecorder`
accumulates δ live (one-step delayed backfill, closing each transition at the next `record()` when
the reward is finalized and V(s′) is known; the last decision has no δ). It costs **zero extra GPU**:
δ is computed only over the battles eval already captures forensically (where `need_aux=True` already
paid for V(s)), pooled per opponent (one `EvalRLPlayer` per matchup → `td_tail()`), and folded as a
**CVaR@5%** (mean of the worst 5%, `TD_TAIL_FRAC`; single min below `TD_TAIL_MIN_SAMPLES`=20). It
rides the exact win-rate plumbing — worker `shard__<unit_id>.json` (raw δ pooled across shards) → `merge_eval_results` →
`eval/td_resid_tail_vs_<opponent>` + `eval/td_resid_tail_mean` (TB + TUI), the `metadata.json`
`latest_eval` block (per-opponent + pool aggregate), and the append-only `eval_results.jsonl`. The
run's `model.gamma` is threaded into the worker (`base_cfg["gamma"]`) so the live δ matches the
prober's offline recompute (guarded by `td_residual_parity_fuzz_test.py`). More-negative = the critic
got blindsided more often — the **leading indicator for the critic-coverage obs work** (it moves in a
cycle or two, where saturated win-rate / gate-pinned `win_rate_vs_pool` / wide-CI ELO don't).

**Intra-worker concurrency (`--eval-concurrency-per-worker`, default `1` = sequential).** Each
worker overlaps up to N battles **within** its claimed opponent. This is **single-thread asyncio
latency-hiding, NOT multi-core** — everything (the obs build + PyTorch forward in `choose_move`, the
bridge/server I/O) runs on the one `POKE_LOOP` thread with BLAS pinned (`OMP/MKL=1`), so concurrency
only overlaps the time a worker is *blocked* on the bridge subprocess / websocket round-trip with
another battle's forward. The ceiling is **one core of compute**: a single-core bridge benchmark
(`/tmp/eval_concurrency_bench.py`, NN trainee vs bot and vs NN sentinel) measured ~**2.0× decisions/sec
at conc=3** on spare cores (plateau ~3; bot eval ≈2.0×, the heavier NN-vs-NN ≈1.8×) — i.e. about half
the per-decision wall-time at conc=1 was bridge I/O wait. **The old `_EVAL_SUBPROCESS_CONCURRENCY` = 1
default and its "measured slower" note were the *saturated* regime** (eval contending with training's
64 env workers for already-full cores — there the extra event-loop overhead nets negative); on **spare
cores (idle box / the cycle tail)** it's a clean ~2×. So the live gain runs between 1× and 2×
depending on how saturated the box is during the eval window; default stays `1` (opt-in). It does
**not** use idle cores at the tail — that needs *process-level* sharding (chunk one opponent across
workers); concurrency stacks multiplicatively on top of that (≈`2 × #shards`). Cross-opponent
parallelism is still the `--eval-workers` (5) subprocesses work-stealing the pool.

### Battle-level work-stealing (`eval_sharding/` package)

The *process-level* tail fix above is the `eval_sharding/` package — a small, deeply-encapsulated
unit with a narrow interface (4 focused files, no mega-file):

- **`units.py`** — `EvalItem` (one opponent the parent declares) + `ShardUnit` (a chunk of its
  games) + `plan_units(items, shard_games)`, a **pure** partition: split each item's games into
  ≤`shard_games` chunks (Σshards == n_games exactly), ordered LPT-ish (cost-descending items, shards
  round-robined) so every opponent starts early and the expensive ones lead.
- **`results.py`** — `ShardResult` (raw additive metrics: won/finished, reward+turn sums, the raw δ
  list — never a reduced ratio) + `aggregate`, which pools an opponent's shards back **exactly**:
  win_rate=Σwon/Σfinished, reward/ep_len count-weighted, and the TD tail by **pooling raw δ then one
  `td_tail`** (a CVaR can't be averaged). `td_tail` + its constants live here (the single source of
  truth; `eval_callback` re-exports them, so the dependency is one-way `eval_callback → eval_sharding`).
- **`pool.py`** — `ShardedEvalPool`, the deep coordinator. Parent: `write_plan(run_dir)` →
  `collect(result_dir)`. Worker: `from_plan(run_dir)` → `claim_next(claim_dir)` / `publish(...)`. It
  hides every filesystem mechanic; the worker never touches a lock file, the parent never touches a
  shard file. The plan (`plan.json`, items + shard_games) is the **single source of truth** both
  sides read — neither reconstructs the universe independently, so they can't drift.
- **`merge_eval_results`** is now a thin delegate to `ShardedEvalPool.collect` returning the same
  `merged` shape every downstream consumer already reads (record_per_opponent / build_bot_eval_block
  / record_elo / pool & externals blocks are **untouched**), plus additive `counts` (exact W/L) and
  `coverage` siblings.

**Exactness caveat (documented, by design):** win_rate / reward / ep_len are exact regardless of
`shard_games`. `td_resid_tail`'s *aggregation* is exact (pool the raw δ, compute the CVaR once), but
the *captured-battle sample* it's computed over shifts slightly with the shard count — the forensic
capture quota is per-unit (scaled `max(1, ⌈quota/shards⌉)`), so which battles contribute δ depends on
the split. It's a sampled diagnostic either way. Forensic trace files are namespaced by a per-unit
`trace_tag` (`{outcome}_s{shard}_{idx}`) so concurrent shards of one opponent don't collide in the
shared `eval_traces/step_<N>/<opponent>/` dir. Per-cycle `run_dir` is wiped at cleanup (and cleared
at launch), so no lock/shard/plan ever leaks across cycles. Sentinel/fixed opponent models are cached
per worker by path (immutable within a cycle → safe; the version check rides the first load) so a
fine split doesn't pay an N× 27MB deserialize. Worker rewrite: `eval_worker._play_unit` (one fresh
trainee + opponent per unit → independent measurement) + a per-worker model cache; tests:
`eval_sharding_test.py` (partition + aggregation-exactness property + claim-once + coverage),
`eval_sharding_fuzz_test.py` (real bridge battles through the real worker → exact pooled result).

### OFFLINE generation of an eval cycle (`main.ops.eval_trace_gen`) — and the PROVENANCE marker

A live eval cycle is sized for a training run: `--eval-battles` games (100 by default) per opponent,
with a per-opponent OUTCOME QUOTA persisting only ~200 traced battles a side. That is right for a
run — the traces are a loss-forensics sample and the disk is the trainer's — and wrong for a
MEASUREMENT, because every conditioning and identity row an offline read computes is a statistic of
that frame. When a read's binding constraint turns out to be POWER rather than effect size,
`python -m main.ops.eval_trace_gen <run>@<step> --games N --sentinels K --out DIR` plays the cycle
again from the saved checkpoint, offline and on CPU, at whatever size is worth paying for.

**It reuses this chapter's machinery rather than restating it.** The generator builds the same
`EvalItem` list, the same `ShardedEvalPool` plan, and spawns the same `python -m main.eval_worker`
processes — so the sentinel construction, the `_sentinel_tb` regime, the reward built from
`model_config.json`, the forensic quota, the shard-namespaced `trace_tag` and the
`record_eval_selection` collect are all the code documented above, not a parallel copy. What it
supplies is the WHAT: which checkpoint, how many games, which pool snapshots, and where the output
goes.

**Four contracts are worth stating here, because they are what make a generated cycle safe to read.**

1. **The REGIME is read, never assumed.** `eval_sentinel_greedy` is taken from the run's
   `model_config.json`; a run that recorded none is REFUSED. That key names the 2026-09-07
   opponent-regime boundary worth **+8.9 pp** to the trainee, and generating a cycle under the
   other regime would produce numbers that look exactly like a result.
2. **Capture defaults to ALL.** `--quota` restores a live-shaped outcome quota for parity work, but
   the default writes a trace for every battle played, and the manifest's `selection_rule` says
   which was used. Full capture is why the low-variance rows gain power and not merely precision.
3. **Nothing is written under `models/`.** The output is a self-contained SHADOW RUN DIR: the run's
   `model_config.json` and `metadata.json` copied (with `latest_eval.pool.sentinels` REWRITTEN to
   this cycle's sentinels, so `cf_audit.sentinel_snapshots` pins the right networks), `snapshots/`
   symlinked back read-only, and the checkpoint copied to `eval_traces/step_<N>/snapshot.zip` —
   which is where `cf_audit` and the prober's `resolve_model_for_step` look for the network that
   played the traces. `--out` inside the run archive is refused.
4. 🚨 **The manifest carries a `generated_by` block, and a reader may not ignore it.** It records
   the tool, schema, source run, checkpoint sha, games, the bot list, `sentinels_requested` beside
   `sentinels_used`, the capture rule, the seed, the worker count, the concurrency, whether the
   cycle is REPRODUCIBLE, and the POPULATION in words. `main.ops.critic_read` **REFUSES** to form a
   delta between a generated frame and a live one — they are different populations, and the v3
   quota match corrects a difference in capture RATE between two frames of the same shape, not a
   difference in the shape itself. Two generated frames must further agree on games / opponent set
   / capture rule / sentinel regime; a differing SEED is deliberately NOT checked, being two draws
   from one population. It also records `battles_expected` beside `battles_played` and a
   `complete` flag, and `critic_read` REFUSES a side that is incomplete **or** whose completeness
   is UNRECORDED. A truncated cycle is otherwise indistinguishable from a finished one — same
   nominal games, same opponents, same `selection`, same capture rates — while being a smaller
   frame. (2026-09-09: `scripts/land.sh` removed the worktree a generation was running out of;
   all four workers died with `failed to make path absolute` and the cycle landed at 71%. Run a
   long generation from the MAIN checkout.)

**Sentinels are CLAMPED, never padded.** `--sentinels K` draws from the run's own `snapshots/`,
evenly spaced across the step range (hence the rating range, both endpoints kept), excluding any
snapshot at or above the read step — a live pool holds only snapshots older than the trainee, and a
self-mirror is a 50%-by-construction cell (`--include-current-snapshot` opts in). A run with fewer
than K keeps what it has and records the gap: repeating a snapshot to reach the requested count
would inflate the between-opponent SPREAD with a duplicated cell, which is the quantity being
measured. Unlike a live cycle, the chosen snapshot steps ARE recorded (a live cycle leaves
`opponent_pins` empty for pool sentinels).

**Reproducibility, and its honest limit.** `--seed S` pins every stream a shard unit draws from —
the process-global `random` the scripted bots use, both teambuilders' draw RNGs, and the sim PRNG
per battle via `run_local_battles(seed_base=…)`, which derives battle *i*'s own `[m,n,o,p]` from a
hash so the dice stay VARIED within a call and IDENTICAL across calls (one fixed `seed` would run N
copies of one battle, which is not a sample of N). The unit seed is keyed on
`(seed, opponent, shard index)` and **deliberately not on the worker id**, which work-stealing
decides in a race — so `--workers 1` and `--workers 8` give the same cycle. `--concurrency > 1`
does not: several battles of one unit then share the bots' global `random` stream and the order
they draw in is a timing race. The tool prints the caveat, records both numbers, and marks the
cycle NOT reproducible rather than emitting a number that wanders silently.

### Rating-model seam (`rating.py`) — extensibility for Glicko-2 / TrueSkill

The live skill rating is anchored Bradley-Terry (`elo.py`), a *global batch* fit. `rating.py` is the
**ready drop-in point** for a different model without re-plumbing: `MatchRecord` (exact counts +
draws + `period_id` + optional opponent priors — the union BT, Glicko-2 and TrueSkill all need),
`RatingResult`, a `RatingModel` **batch** protocol, and `BradleyTerryRating` — a thin adapter over
`elo.fit_pairwise` whose ratings+SE are **byte-identical** to the live fit (pinned by `rating_test.py`).
`eval_rows_to_match_records` bridges the existing `EvalRow` history. The live `record_elo` path is
**deliberately unchanged** (zero risk): the seam exists and is tested, but routing through it buys
nothing until a new model is actually wanted — and Glicko-2 is *sequential* (period-by-period RD
carry-forward), so it needs the `SequentialRatingModel` sibling sketched in the module footer, not the
batch `fit`. Data fidelity is already in place: `eval_results.jsonl` now carries exact per-opponent
`counts` (additive, backward-compatible), so a future Glicko backfill has an exact ladder even under
partial shard coverage (where `win_rate × n_games` would be ambiguous).
## ELO / skill rating (`elo.py`, `bot_elo_calibration.py`, `main.elo`)

Once training is mostly self-play **pool play**, win-rate stops being legible: the promotion
gate only promotes when `win_rate_vs_pool > promote_threshold` and the pool is a *sliding window
of recent selves*, so `win_rate_vs_pool` is a treadmill pinned near 50-65% **by construction** —
it cannot trend up however much the model improves; `win_rate_vs_bots` saturates near 100%. The
ELO subsystem gives a single **absolute** number that genuinely rises with skill, anchored to the
fixed bots.

- **No new battles.** Every eval cycle already plays the trainee (greedy) vs all 9 bots and vs
  up to `--n-sentinels` (default 5) pool sentinels, `EVAL_GAMES` each — a full tournament-matrix
  row. `record_elo`
  (`eval_callback.py`, shared by BOTH callbacks) appends that row to an **append-only
  `<run>/eval_results.jsonl`** (`snapshot.append_eval_result_row`) — the canonical, restart-safe
  source of truth, distinct from the overwritten `metadata.json:latest_eval`.
- **The model = anchored Bradley-Terry** (`elo.fit_elo`): `P(i beats j)=σ((Rᵢ−Rⱼ)·ln10/400)`,
  fit in **batch** by penalized MLE (weak Gaussian prior keeps 100-0 records finite), SE from the
  inverse Hessian. Each bot is a player `bot:<name>`, each snapshot `snap:<step>` — a snapshot is
  the SAME player whether it appears as a cycle's trainee or later as a sentinel (unified by
  step), which links the whole ladder. Batch-BT (not online K-factor Elo) is drift-free and
  re-runnable; the fit is a few Newton steps over ~tens of players. **Not Glicko-2**: its
  volatility models skill drift, but snapshots are *frozen* — the drift is the *sequence* of
  snapshots (the ELO-vs-step curve); the per-player uncertainty (Glicko's valuable part) is the
  Hessian SE.
- **Anchor = a precomputed bot-vs-bot round-robin.** `python -m agents.training.bot_elo_calibration`
  plays all 36 bot pairs toward `--target-games` (default 5000) **in-process via the bridge — no
  server** (safe alongside a live run; it does use CPU — throttle with `--concurrency`), fits BT
  (`elo.fit_pairwise`, `random` pinned at `base`=1000), and writes the anchor. **Artifact split:**
  the immutable bot anchor (ratings, SEs, the 9×9 win-matrix, a non-transitivity `fit_quality`) is
  the only runtime input, so it lives in **`data/gen3_bot_elo_anchors.json`**; the raw game-count
  **store** (resume state) and the **heatmap** PNG are calibration provenance/viz, so they live with
  the ELO design work under **`designs/ai_v5/elo_calibration/`** (override with `--games-store` /
  `--heatmap`). The
  live/offline fits then **pin all 9 bots** to those high-confidence ratings and fit only
  snapshots — so a snapshot is well-grounded from its first cycle, and because the anchor is
  identical across runs, **snapshot ELOs are comparable run-to-run**. **Regenerate when bot logic
  changes** (the json records `git_hash` + date). Graceful fallback when the file is absent:
  `random` pinned at `base`, other bots float (rank/trend preserved, scale not cross-run-stable).
  Bots build once and are reused across pairs (`reset_battles` between) — building warms the data
  singletons (~4.5 s each), so per-pair rebuilds dominated cost; the full 5000-game job is a
  many-hour, run-overnight one-time cost.
- **The RAW matrix, at higher resolution.** `python -m agents.training.bot_matchup_matrix`
  accumulates the same round-robin (same bots, same team sampling, same bridge driver — it calls
  the calibration's own `_build_bot`/`_play_chunk`) as **raw per-pair `wins_a`/`wins_b`/`draws`/`n`**
  toward 10 000 games/pair in resumable chunks → `data/gen3_bot_matchups.json`. Draws stay
  separate and it **never writes the anchor** (regenerating that is an owner decision).
- **Live (each eval cycle).** `record_elo` refits and records `eval/elo` + `eval/elo_ci` (95% CI
  half-width) to TensorBoard + the TUI dict, and stamps `elo`/`elo_ci` into `metadata.json:
  latest_eval` (so the resume-republish path shows ELO immediately after a restart — the saved
  headline is authoritative; and if a resumed checkpoint predates the `elo` field,
  `replay_last_eval_to_tui` **fits** the saved block's win rates via `elo.fit_from_block` to recover
  both the headline and each opponent's ELO, so the badge never blanks for a full cadence). The
  launcher
  surfaces a `🏅 ELO 1532 ±40` badge (`app.py::_elo_badge`) + an `elo` column in the eval panel:
  the model's rating on the `all` row, and each opponent's anchored ELO on its row
  (`_record_opponent_elos` records `eval/elo_vs_<bot>` + positional `eval/elo_vs_sentinel_<i>` to
  the TUI). The live number is the best estimate from data SO FAR (batch-BT is global → early
  points retro-adjust; the single-cycle per-sentinel ELO is rough — only the trainee is
  bot-anchored each cycle); the offline CLI re-fits canonically over the full per-snapshot history.
- **Offline (`python -m main.elo <run_dir>`).** Loads results (`--source auto|log|tb|meta` —
  `tb` **backfills an already-running run straight from TensorBoard, zero training change**), fits,
  and prints a ranked ladder + writes `elo_ratings.json` + an Elo-vs-step `elo_curve.png` (CI band
  + bot anchor lines). `--out` defaults to `<run>/elo/`; point elsewhere to analyze a LIVE run
  without writing into it.
- **The two-regime caveat is CLOSED for new runs (2026-09-07, `gen3_eval_sentinel_greedy_default_v1`).**
  It used to bite by default: the trainee played greedy but the sentinels sampled at
  `--self-play-temp`, so a snapshot's rating blended greedy strength (as a cycle's trainee) with
  stochastic strength (as a later sentinel), and the sentinel additionally drew its team from the
  flat pool builder while the trainee drew sample-biased. **Sentinels are now GREEDY and draw the
  trainee's own teams by default**, so every snapshot is scored in one regime and the ELO ladder is
  internally consistent. `--no-eval-sentinel-greedy` restores the old behaviour. 🚨 **The cost is a
  ONE-TIME DISCONTINUITY in `eval/elo` and `win_rate_vs_pool` against every v9-era run** — the
  asymmetric edge read **+8.9 pp [+7.0, +10.7]** in the trainee's favour — and the discontinuity is
  in the SENTINEL edges only: the bot-anchored scale is preserved, since trainee-vs-bot records are
  untouched. A run's regime is recorded (`model_config.json` config v112) and INHERITED on a
  flagless resume, so a windowed statistic can no longer cross the boundary unmarked; see
  [`self_play_and_pool.md`](self_play_and_pool.md) for the resolution rule and the startup line.
  Tests: `elo_test.py` (synthetic-ladder recovery, anchoring, perfect-score, loaders,
  `fit_pairwise`), `main/train/eval_sentinel_regime_test.py` (the regime + gate resolution).

### Frozen-snapshot ELO ladder — the dense, pay-once resolution (`snapshot_ladder.py`)

The live ELO above is RESOLUTION-limited at the frontier: the fixed bots have SATURATED (we sit
~400 Elo above them, out on the flat tail of the logistic — a 10-Elo trainee move shifts its
bot-WR by ~0.5% against a 1.9%/200-game noise floor), so the bots pin the absolute LEVEL but the
fine ordering rides on the sparse, near-50% sentinel edges (±15 Elo CIs). Fix from the other side:
a promoted snapshot is FROZEN, so snapshot-A-vs-snapshot-B is a STATIONARY Bernoulli — measure it
ONCE (dense round-robin) and it is permanent. On each promotion, `SelfPlayCallback._spawn_snapshot_ladder_update`
fires a **DETACHED** `python -m agents.training.snapshot_ladder <run> --promote <step>` subprocess
(bridge, off the training path) that plays the new frozen node vs the current frozen pool
(`--snapshot-ladder-games`, default 100/pair; 0 disables) and appends to
`<run>/snapshot_ladder/games.jsonl` (**forever, race-safe line appends; a measured pair is NEVER
replayed**). `fit_ladder` combines that dense frozen-vs-frozen matrix with each snapshot's
historical bot edges (from `eval_results.jsonl` — the anchor connection, **and only the BOT edges**:
see the eval-sentinel exclusion below) → an anchored BT fit
(`fit_pairwise`, bots pinned) written to `<run>/snapshot_ladder/ladder.json` (the sidecar metric);
`_record_ladder_elo` surfaces the latest promoted node's rating as `eval/ladder_elo` (+`_ci`) on
TB/TUI — the high-resolution counterpart to the saturated `eval/elo`. Snapshots load via
`load_foreign_opponent` (their own saved config → PopArt/toggles honored, `check_compatible`
skipped). `--backfill` pays the one-time back tax over the whole current pool (idempotent — skips
measured pairs); `--fit-only` refits without playing. `ladder.json.fit_quality.mean_abs_err`
QUANTIFIES non-transitivity (a scalar Elo is lossy if the pool is rock-paper-scissors — the dense
matrix at least measures it). Tests: `snapshot_ladder_test.py` (store accumulation/symmetry,
measure-once contract, fit-recovers-ordering, sidecar read, the eval-sentinel exclusion).

🚨 **THE EVAL CYCLES' SENTINEL EDGES ARE EXCLUDED FROM THE LADDER FIT** (2026-09-07).
`elo._rows_to_results` yields TWO families off one eval row — trainee-vs-bot (`bot:`) and
trainee-vs-**sentinel** (`snap:` vs `snap:`) — and `fit_ladder` used to fold in both. The second
is a **different measurement of the same frozen pair** the dense matrix already holds: an eval
cycle plays the GREEDY trainee against a **STOCHASTIC** sentinel (`eval_worker`:
`stochastic=not sentinel_greedy`, `temperature=self_play_temp`) with an **asymmetric teambuilder**
(the trainee gets the sample-team bias, the sentinel does not), while the ladder plays
greedy-vs-greedy with the same biased builder on both sides. Measured on
`ai_v12_02_winprob_critic` over the **60 pairs both sources cover**: the eval edge favours the
NEWER snapshot by **+8.9 pp [+7.0, +10.7]** — systematic, not noise. Mixing them inflated
`ladder.json`'s newest nodes by **+21 to +29 Elo**, compounding exactly the newest-node inflation
the matched-count rule exists to control. All three runs compared in the win-prob era carry
`eval_sentinel_greedy=False` / `self_play_temp=1.0`, so **every ladder written before 2026-09-07
has it**. `games.jsonl` is now the ONLY snapshot-vs-snapshot source; the excluded count is
returned and written as **`eval_sentinel_edges_dropped`** (absent ⇒ a pre-fix ladder).
Re-fit shift, per run (committed `ladder.json` → current `fit_ladder`, same rated steps): the
early nodes RISE and the late nodes FALL — `ai_v12_02_winprob_critic` 4M **+32.1** → 34M
**−33.6**; `ai_v9_29_rev1_0823` 2M **+30.5** → 24M **−45.1**.
**The per-cycle `eval/elo` star fit (`elo.fit_from_run` / `record_elo`) is UNCHANGED and still
uses the sentinel edges** — it has no dense matrix to prefer, and those near-50% edges are the
only resolution the saturated bots cannot give (removing them from one cycle moves the trainee
−110 Elo and WIDENS the CI ±30 → ±41). The exclusion is the LADDER's alone, and
`snapshot_ladder_test.py` pins that separation with a test on `fit_from_run`.

#### Reusing the pairs an eval cycle already measured — OPTION A (2026-09-07)

The exclusion above is what a *different* measurement of the same pair costs. Once the two
measurements are **the same experiment**, the opposite move becomes available: the ladder can count
an eval-measured pair as COVERED instead of replaying it. The eval cycle freezes the live model to
the very file promotion copies into the pool (`selfplay_callback.py` → `snapshot_pool.add_from_path`
— byte-identical weights, zero step gap), so the players were never in question; only the protocol
was.

**The gate is BOTH halves of the regime, read off the row and never assumed.** A pair is reusable
only when its `eval_results.jsonl` row's `sentinel_regime` says `greedy` **and** `symmetric_teams`.
Either alone is the asymmetric measurement the +8.9 pp was measured on, and a row with no stamp
(written before 2026-09-07) is UNKNOWN, which reads as not-reusable. A `--no-eval-sentinel-greedy`
run therefore reuses nothing and pays the full round-robin tax, byte-identically.

**The mechanics** (`snapshot_ladder.py`): `eval_measured_pairs` parses the reusable edges (exact
`counts` when the row has them, else `win_rate × n_games`); `ingest_eval_measured_pairs` appends the
ones among this promotion's target pairs that `games.jsonl` does not already hold, tagged
`"source": "eval_cycle"`; `_measure_missing` then finds them present and skips them.
`update_for_promotion` prints `N REUSED … M played` per promotion, and `fit_ladder` reports
`pairs_by_source` (a run that reuses nothing reads `{"ladder": N}` and its rows carry no `source`
key at all). **Landing the edge in `games.jsonl` rather than teaching the FIT a second source is
what keeps the arithmetic honest** — source (2) drops every `snap:`-vs-`snap:` eval edge
unconditionally, so an ingested pair is counted exactly ONCE; the double-count that ruled this
approach out before that filter landed cannot occur.

**The saving:** 5 sentinels per cycle = **500 battles per promotion** at the default
`--snapshot-ladder-games 100`; on a 15-snapshot pool that is 5 of 14 pairs = **36% of the
per-promotion tax**. On `ai_v12_02_winprob_critic` (the pre-fix run, measured but not applied) 60 of
105 pairs — 6,000 battles — had been measured twice. Tests: `snapshot_ladder_test.py` (the
regime read incl. the planted HALF-symmetric row that must NOT be reused, the counts fallback, the
promotion path, idempotence, exactly-once-in-the-fit, and the byte-identical stochastic path);
`elo_row_contract_test.py` (the writer→row→ladder join, so the two field names cannot drift).

### Hodge decomposition — the SPINE and the WIDTH (`hodge.py`)

A scalar rating is a **transitive** model by construction, so a BT fit cannot see a cycle: two
snapshots with identical ELO can have a lopsided head-to-head. `ladder.json`'s
`fit_quality.mean_abs_err` NOTICES the residue but reports it as one unitless number with **no
noise floor**, which cannot answer the only question that matters — *is the non-transitivity real,
or is it binomial noise on 100-game edges?* HodgeRank answers it by splitting the measured flows

```
Y_ij = logit(p_ij)   =   (r_i − r_j)   +   R_ij        w_ij = n_ij·p_ij·(1−p_ij)
                          ───────────       ────
                          TRANSITIVE        CYCLIC     (Fisher info of a logit = the weight)
```

where `r` is the weighted-least-squares (graph-Laplacian) solve — BT's quadratic cousin, reported
BESIDE the BT ratings so the estimators' disagreement is visible. The split is **exactly
w-orthogonal** (`Σw·Y² = Σw·(rᵢ−rⱼ)² + Σw·R²`, pinned by a test), so spine and width cannot be
traded against each other by refitting. Units: 1 logit = 400/ln10 ≈ **173.72 ELO**.

**The noise floor is the whole instrument.** Two nulls, both reported: an exact-mean analytic one
(`E[Σw·R²] = Σ(1 − w_e·Reff_e)` — per-edge effective resistance, i.e. Foster's `E−V+C` spread over
edges) and a **parametric bootstrap** that simulates games from the fitted transitive model and
re-runs the whole pipeline. `width_rms_excess = √(raw² − null²)` is the width that survives, with a
p-value for "width > noise".

**Width SCOPE — a pendant edge's residual is identically zero.** A player with one measured
opponent has that single edge as its whole normal equation, so counting it only inflates Σw and
deflates the RMS. Width statistics therefore default to the **triangle-supported subgraph**; the
spine is always fit over every edge. `n_triangles` + `n_width_edges` ride with every read.

- **Offline — THE instrument.** `python -m main.elo <run>` prints the block and writes it into
  `elo/elo_ratings.json` under `hodge` (flags: `--no-hodge`, `--hodge-bootstrap N`, `--hodge-seed`,
  `--hodge-with-bot-rr`). The graph is exactly `fit_ladder`'s: the dense frozen matrix
  (`snapshot_ladder/games.jsonl`) + every cycle's bot/sentinel edges. The static bot round-robin is
  **excluded by default** — its 36 edges carry ~2700 games each against a ladder edge's 100, so on
  the Fisher weighting they would carry ~99% of Σw and the "width" would become a property of the
  immutable shared anchor rather than of this run. `main.endofrun`'s §1 block carries the same read
  for the run and its `--ref`.
- **Live — two scalars beside `eval/elo`**, recorded by `record_elo` on the same cadence:
  `eval/hodge_width_elo` (excess width, ELO) and `eval/hodge_cyclic_fraction` (null-adjusted).
  Both also ride in the `eval_results.jsonl` row's `hodge` block for offline replotting, and a
  cycle whose graph had **no testable triangle** writes `recorded: false` + a reason there and
  records NOTHING to TB (never 0-as-a-stand-in, never NaN — a missing point and a suppressed one
  look identical in TensorBoard, and only one is a fact about the graph).

⚠️ **THE STAR-GRAPH SUBTLETY — read this before quoting a live width.** A cycle's own new games are
a **star** (trainee vs each opponent). A star is a tree; a tree has no cycles; so a width computed
on the cycle's games alone is *identically zero* — a fake instrument that would read "no
non-transitivity" forever. The triangles come from joining the trainee's edges to the **static
bot-vs-bot round-robin** in `data/gen3_bot_elo_anchors.json` (which does ship the raw 9×9
`win_matrix` + per-pair `pair_games`, so those are MEASURED edges; a future anchor carrying only
`ratings` falls back to edges reconstructed from them, which are transitive by construction and act
purely as a pinning prior — flagged in `caveats` when it happens). So the live metric means exactly
**"the trainee's matchup deviation from its own rating, over trainee×bot×bot triangles"** — nothing
about the pool's width. Sentinel edges are in the FIT (real spine information) but on no triangle,
so they are excluded from the width scope. And the live read is **weak by construction**: ~100-game
edges put the noise floor around 35-60 ELO, so only a gross cyclic profile clears it (measured on
gen-15's 12 cycles: p between 0.13 and 0.93, i.e. never significant on its own). **The offline
dense-ladder read is the real instrument; the ELO-reading rules below apply unchanged — never
narrate a mid-run width.**

**First reading (gen-15, `ai_v9_18_gen15_v8rewards_0818`, 21 players / 174 edges / 814 triangles,
300 bootstrap reps):** spine 939 ELO, width raw 58 → null 36 → **excess 46 ELO, p = 0.005**; cyclic
energy 6.3% raw / 3.8% null-adjusted; **3 significant 3-cycles**, all snapshot-vs-snapshot
(16M > 20M > 18M > 16M, curl +217 ELO z=4.3; 8M > 20M > 18M; 8M > 20M > 14M). gen-14 on the same
read (same 21/174/814 shape): spine 765, excess **26 ELO, p = 0.0033**, 2.2% null-adjusted, and **0
individually-significant cycles** — its width is real but diffuse. So both ladders are
overwhelmingly spine (96-98%) and both carry cycle content that is **not sampling noise** — the
first evidence here that the BT gate is a lossy projection by a *measured* amount, and that the
loss is bigger on gen-15. (p ≈ 0.003-0.005 is the bootstrap's floor `1/(B+1)`, not a coincidence:
no null replicate reached the observed width.) Tests: `hodge_test.py`.

#### 🚨 Reading an ELO: `ladder.json`, at matched SNAPSHOT COUNT, never mid-run

**Read `<run>/snapshot_ladder/ladder.json` — not `eval/elo`, not the per-cycle TB scalar.** On
gen-10's completed ladder the two agree at the end (24M: dense 2079 vs sparse 2102) but the dense
CI is **±10 vs ±29**. Precision is the reason to prefer it; it is *not* immune to the drift below.

**A snapshot's rating keeps moving until it stops gaining opponents.** Anchored BT is a GLOBAL
BATCH fit — every added player re-solves every rating — and the movement is a **systematic
downward bias on the newest node**, not noise. Measured over gen-10's 12 successive refits
(`snapshot_ladder/updater.log` records each one):

| snapshot | first fit | final fit (n=12) | drift |
|---|---|---|---|
| 2M | 1790 | 1705 | **−85** |
| 4M | 1945 | 1844 | **−101** |
| 12M | 2089 | 2021 | **−68** |
| 14M | 2088 | 2044 | **−44** |

Mechanism, and the SE is the tell (12M: 25.9 → 18.4 as it fell 2089 → 2021): a fresh snapshot's
only edges are ~90% wins over the bots. A saturated edge says *"≥380 Elo above"* with a likelihood
that is **flat upward**, so the MLE is inflated and under-constrained. The near-50% frozen-vs-frozen
edges are sharply informative, and as they accumulate they pull the chain down onto the anchor.
Dense measurement buys resolution; it does not buy an early answer.

**Therefore, two rules:**

1. **Never narrate a mid-run ELO or a mid-run delta.** The gen-10 12M delta read +108, +82, +73,
   +64 before settling at **+11** — four reported "results", all artifacts. Wait for the run to end.
2. **Cross-run comparison must be at matched snapshot count `n`, not matched step.** Both runs'
   node at n=k carries the same inflation, so it cancels; a live run's newest node against a
   finished run's *final* value does not. Worked example — gen-11 at n=7 vs gen-10's **n=7** fit
   (recoverable from `updater.log`) reads 14M: 2082 vs 2088 = **−6, tied**; against gen-10's n=12
   final it reads **+38**, which is the drift and nothing else.

`n_frozen_pairs_measured` / `n_pairs_possible` in `ladder.json` is the completeness check — a fit
at 21/21 pairs is internally complete but only 7 nodes deep, and depth is what the bias tracks.

3. **Matched COUNT means matched FIT SIZE, and the tool now REFITS to get it.** Rule 2's
   "recoverable from `updater.log`" was the only mechanism, and `main.critic_gate` did not use it:
   it took the comparator's n-th node from its FINAL fit. Measured 2026-09-07 at the win-prob
   arm's 10M read: rev-1's 8M node is **2052** in a first-4 fit and **1958** in its final 12-node
   fit, so the gate read a **30-Elo TRAIL as a 64-Elo LEAD** (94 Elo handed to the arm; ledger
   *10M DECIDING READ*). `snapshot_ladder.fit_ladder(run_dir, first_n=n, write=False, steps=…)`
   is the refit — every frozen pair AND every bot/sentinel edge whose snapshot endpoints all lie
   in the prefix, **strict**: an edge to a node outside the prefix re-imports exactly the future
   the rule excludes (the loose filter reads −12 where strict reads −30). A ladder that cannot be
   refit — no `games.jsonl`, or a FORK's ladder whose early nodes are rated only through edges to
   its own late selves (`ai_v9_59_R2ACTION_0827`: every pair touches 26M/28M) — is reported at
   **UNMATCHED FIT SIZE** with that label on the number, and the famine pre-test REFUSES on it
   rather than printing a lead.

4. **A committed `ladder.json` is not evidence — REFIT BOTH SIDES from `games.jsonl`.** A run's
   `ladder.json` was written by the code that run is PINNED to, and every ladder written before
   2026-09-07 folded the eval cycles' greedy-vs-stochastic **sentinel** edges into the fit
   (+8.9 pp to the newer snapshot, +21..+29 Elo on the newest nodes — see the ladder section
   above). A live run pinned to an older commit keeps writing biased ladders for as long as it
   runs, so trusting the committed file compares one side's corrected fit against another side's
   biased one. `main.critic_gate` therefore refits **both** sides with THIS tree's `fit_ladder`
   whenever `games.jsonl` exists — even at already-matched node counts, where the old
   `len(nodes) > n` guard refit neither — and LABELS in `fit_size_note` (and `refit_fallbacks`)
   any side that had to fall back to its committed ladder. Measured effect on the registered
   famine read (arm `ai_v12_02_winprob_critic` vs `famine_comparator` = `ai_v9_29_rev1_0823`,
   both refit on a strict prefix): the trail goes from **+29.9 Elo (se 20.7) → +13.0 (se 22.1)**
   at n=4 and from **+34.3 (se 14.7) → +20.9 (se 15.8)** at n=12. The verdict does not change —
   both are inside the registry floor of 38 — but the registered −30 / −34 numbers were
   measured through the biased fit.
