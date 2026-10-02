# The post-switch DELETION PASS + bounded tech-debt PAYDOWN — manifest

**Status: APPROVED 2026-10-02 (owner: "I am happy just to delete and we don't need to keep the goldens").** Decisions as recorded in §0's APPROVED column. The 8-day box starts 2026-10-02. Model policy (owner): the mechanical DELETION units run on opus-medium; the structural units (U1, U2, U4) on opus-high.
Scoped at `b2c09132` (THE M5 SWITCH, ledger 2026-10-02). Owner sequence (2026-10-01): validate Rust infra →
switch (DONE) → **this pass** → slow tier → T15 re-bake → bottleneck profile → the X26 baseline. Reason: *one
system to reason about.* Parent lists: `designs/endstate/program_rust_core.md` §4 (the per-milestone rows; this
doc supersedes its figures where they disagree — see §7), `design_own_ppo_loop.md` §3.3, `TECH_DEBT_BACKLOG.md`
§1, `TASK_BACKLOG.md` T10.

**The owner decides six things (§0). Everything else follows from them.**

---

## 0. The owner's decisions

| # | decision | recommendation (APPROVED 2026-10-02 unless noted) |
|---|---|---|
| D1 | The LEVER LIST (§2): per lever, PORT to Rust or DELETE with the Python core | DELETE all 16; port none now. Distillation is the one with a live future user (X15); port it when X15 is scheduled (~1–2 agent-days) |
| D2 | The BARE-ARGV default (a fresh argv without `--arch production`) | `--critic winprob` + its three reward defaults + `--env-core rust` (§2.1). The `--debug` smoke then runs on the Rust core |
| D3 | `Gen3Env` as an ORACLE. The Rust env core's env-level parity gates (episode, labels, opponents, `rust_rollout/parity`, the bot-corpus re-record) all replay through `Gen3Env` + `bridge_session`; §1 point 4's named survivors do not include it | RETIRE them: bank one full green run of every env-level gate at the deletion commit's parent (pinned, recorded in the ledger), freeze the bot-corpus bank as a fixture, then delete. A slim `Gen3Env` keeps `wrappers`, `bridge_session` and the reward manager alive — the opposite of one system | **APPROVED AS MODIFIED (owner): retire them WITHOUT banking a last-green run — "we don't need to keep the goldens". This covers ONLY the Python-env ORACLE goldens (the env-level parity gates, the bot-corpus re-record); the production goldens — the K9 learner golden, the obs golden, `reward_golden` — STAY, they guard live code.**
| D4 | A PYTHON-ERA checkpoint resumed or forked on HEAD (today it inherits `--env-core python`) | winprob checkpoint → onto the Rust core, announced as a core switch (legitimate: the ledger's switch entry carries checkpoints across). Shaped checkpoint → REFUSE loudly, run it pinned (the `e3ef16db` precedent) |
| D5 | The legacy FINAL EVAL (TECH_DEBT §2(g), OFF since `e6121412`) | delete `final_eval.py` + `--final-eval`; the last periodic Rust eval cycle is the readout (nothing reads the final eval's output) |
| D6 | The offline cf stack (`cf_producer*`, `cf_audit*`, `harvest`, `critic_read`, the prober's counterfactual) — ~3.9k lines that READ old runs' `cf_records` rings | NOT in this pass. The FLAG CENSUS decides each on its users |

---

## 1. DELETION ROWS

LOC measured 2026-10-02 at `b2c09132` (`wc -l`; `~` = estimated from a line range). "Tests" = test lines that go
or are rewritten. Paths under `src/`.

| # | row | files ≈ lines (module / tests) | live dependents outside the row → what changes first | risk | order |
|---|---|---|---|---|---|
| R1 | **The Python env core + its collect path** | `agents/training/gen3_env.py` 1,122 · `wrappers.py` 657 · `main/train/env_factory.py` 323 · `agents/training/async_vec_env.py` 287 · `utils/bridge/bridge_session.py` 667 · `train_rl_agent.py`'s python branch ~80 · `instrumented_ppo/loop.py` `_collect_python` 96 · `--env-core python` + its resolution (`rust_env_setup.py` L72–118, `parser/env_core.py`, `run_io.py:204`) · `--use-bridge node/off` for training. **≈ 3.2k / ≈ 6–8k** (26 test files import `gen3_env`, 15 `wrappers`, 13 `bridge_session`; overlapping) | 🚨 **PRODUCTION STARTUP builds a `Gen3Env`**: `rust_rollout/build.py` `trainee_spaces()` (called by `rust_env_setup.py:169`, the K9 learner golden, `rust_rollout/testkit.py`, `rust_core_m5/production.py`) → extract a standalone spaces builder. `rust_env_opponents.py`, `selfplay_callback.py`, `parser/eval_subprocess.py`, `ops/value_sidecar_read.py` import `OPP_CLASS_*` / `STABLE_CHALLENGE_SHARE` from `wrappers` → move the constants. `config.py:1472` (`resolved_obs_source`), `rust_eval/build.py:75` (`eval_worker._build_trainee_tb`) → re-home. **SURVIVES:** `battle_stream_client.py` (178; eval worker, snapshot ladder, untaught meter, warm-start, the prober) | silent loss of the env-level parity oracles (→ D3); the root `CLAUDE.md` smoke and every unpinned python-era resume run this core today (→ D2, D4) | after D2–D4, U2 (extraction) and every R-lever row; with R6 |
| R2 | **The 16 Python-only levers** (§2) + the levers that mean something only under `--critic shaped` | ≈ **8.8k / ≈ 16k** at D6's line (training half of cf only; ≈ 12.7k with the whole cf stack): rollout target 805 · dense aux 342 · true team 264 + extractor branches · distillation ≈ 2.2k · self-PBRS 395 · frozen-φ 299 · cf training half ≈ 2.1k (`cf_records`, `cf_label_buffer`, `cf_terms`, `cf_q_labels`, `cf_mc_return`, `q_winprob_terms`) · search teacher 1,690 + worker · team-PFSP 251 · exploiter ladder 429 · entropy boosts ~150 · the shaped critic's branches (PopArt, value-dist, `value_from_dist`) ~1–2k | `win_prob_callback.py` (729) is SHARED: `rust_rollout/store.py` imports `backfill_terminal_labels` → keep that half. The second refusal layer goes with them: `utils/rust_env/label_inventory.py`'s `refused` keys, `store.obs_key_sources`' unknown-key list, `rust_vec_env.REFUSED_WITH_FLAG` (or `rust_vec_env_test` fails) | 🚨 `--cf-label-supply external` passes the launch checks on Rust today and would starve in flight (FATAL_SUPPLY) — dies with the row; a deleted flag in a RECORDED argv breaks an unpinned HEAD resume of that run (it must run pinned) | after D1, D2; before R1 |
| R3 | **The Python fork arm** | `fork_callback.py` 331 · `fork_driver.py` 290 · `fork_worker.py` 290 · `fork_crn.py` 72 · `fork_buffer.py` L202–426 (225) · `fork_needs_cf_records` · **≈ 1.2k / ≈ 0.9k** (`fork_callback_test` 338, `fork_crn_sim_test` 114, `fork_crn_test` 138 — missing from program §4 — 15 of 20 `fork_buffer_test` tests, 9 `fork_flags_test` cases) | KEEP: `fork_arm.py` (18 symbols the Rust port imports; `PG_MASK_KEY` read by `micro_step` / `train_setup` / `compile_regions`), `fork_buffer.py` L1–201 (`FILL`, `unfillable_keys`, `refusal_text`). Dead with it: `gae`, `branch_rewards`, `fork_buffer_class`, `fork_arm.forks_per_battle` | forks.md §5's mask rule is pinned only through `build_branch_rows` tests → confirm `rust_rollout/fork_test.py` pins it before deleting | with R2 (it is a lever row) |
| R4 | **PPO stage 3** (`design_own_ppo_loop.md` §3.3; stages 1 + 2 BUILT 2026-10-01) | delete the `GEN3AI_PPO_LOOP` seam (`_reference_loop` / `_reference_learn` / `loop_mode`, ~60), the python branch of `rollout_probes.py`; REPLACE SB3's `MaskableDictRolloutBuffer` (owned device buffer), the SB3 logger (owned `SummaryWriter`, same tags + step convention), `BaseCallback` (declared hook protocol, 25 subclasses in 22 files); `rust_vec_env.py`'s SB3 `VecEnv` base; `device_batches.py`'s SB3 sample type | E2 (`own_ppo_loop_parity_test`) and E1 (`own_ppo_loop_test`) replay through the Python collect → RETIRE them (E2 passed when stage 1 shipped); the owned buffer's bar = the K9 golden + Rust-core E3 | the buffer must reproduce SB3's `get()` permutation or the K9 golden moves; the logger's tag names are read by `killbar` / `tb_read` / the launcher / `plot_tb` | with R1; after the GPU memory fix lands (`device_batches.py`) |
| R5 | **SB3 leftovers** (the rest) | stage 3: `Monitor` (3 sites), `DummyVecEnv` / `SubprocVecEnv` (4 + 4), SB3 `configure` (4). **Stage 4 — NOT in this pass** (separate owner decision, §3.4): `MaskablePPO` (47 import sites), `.zip` save/load, the policy base classes, the dependency; `fresh_checkpoint.py`'s `DummyVecEnv` stub stays until then | `learner_golden.py`, `update_fit.py`, `rust_rollout/parity.py` build the SB3 logger | low | with R4 |
| R6 | **Compile-opponents** (TRIMMED vs program §4) | the three flags (+`--no-` forms; `parser/clean_world.py` L264–300, `compile_flags.py`), `compile_preload.py` 153, `compile_prewarm.py` 89, the quorum code ~60–90, `snapshot_pool`'s compile wiring ~22 · **≈ 0.35k / ≈ 0.3k** | 🚨 program §4 is WRONG to delete the module: `compile_opponents.maybe_compile_extractor` is live in the prober's counterfactual view, `snapshot_ladder.py:723` (hardcoded `True`), the eval worker, `cf_producer_snapshot`, search-dividend; the pool's LRU cache feeds T2's weight loads (`rust_rollout/build.py:230`) → KEEP both | `compile_cache_test.py:358` (source-order assertion) goes red; 184 recorded argvs carry the flag (unpinned resume → argparse error; they run pinned) | with R1 |
| R7 | **torch 2.5.1-only code** | `team_transformer` split ~26 · `compile_control`'s 2.5.1 rows ~60 · `compile_regions.regions_supported` ~8 · `learner_golden.json` 2.5.1 entry · `aot.py`'s < 2.8 refusal · **≈ 0.2k / ≈ 0.6k (mostly edits)** | a PINNED resume runs its own commit's `src/` (`launcher/child.py:403-423`); HEAD supplies only the launcher's env selection. KEEP: `torch_runtime.KNOWN_ENVS["2.5.1"]` / `LEGACY_TORCH`, `gen3ai_stable`, `environment.yml` (until A′ `ai_v14_09_r0_offense_a` is finished or abandoned). ADD FIRST: a HEAD-side REFUSAL of torch < 2.8 (none exists) | without the refusal a 2.5.1 checkpoint under `--no-pin` / `--sync-to-main` / a bare `train_rl_agent.py --model` runs HEAD code without the trunk split it needs on 2.5.1 CUDA. The cache-limit LOG detector is NOT 2.5.1-only (torch 2.8 still emits it) → replace with `fail_on_recompile_limit_hit`, do not just delete | with R8 |
| R8 | **The legacy extractor-only compile gate** (the one the K8 regions replaced) | `compile_trainer.py` L743–904 + gate-only helpers ≈ 360 · `compile_canary.py` ~37 · `main/train/lifecycle.py` L65–101 · the sentinel L1033–1042 · **≈ 0.45k / ≈ 1.1k** (`compile_trainer_test` 369, `parity_probe_test` 290, `compile_gate_probe_test` 195, `compiled_train_probes_test` 172, `compile_batch1_cuda_test` 81) | selected only when `regions_supported()` is False → fully coupled to R7. 🚨 `rust_core_m5/hooks.py:179` calls `compile_trainer_extractor` without `regions_follow` — on 2.8 it measures the LEGACY compile, not production's R0 region (a live mis-measurement, fixed by this row); `compile_cache_benchmark.py:57` loses its compiled arm | the TF32 rule's second fp32 graph sits inside this gate (R9 coupling) | with R7; `lifecycle.py` waits for the GPU memory fix |
| R9 | **RETIRE TF32** (TECH_DEBT §1, owner-accepted) | `--matmul-precision` (`parser/hyperparameters.py` L225–234, `config.apply_matmul_precision` ~30) · the compile gate's TF32 rule ~90 (partly in R8) · regions / canary ~25 · `parity_probe.PRECISION_BARS` TF32 rung ~15 · K9(b)'s TF32 rule (`rust_rollout/consistency.py` ~60) · `env_core_rust_tf32_behaviour_check_fatal` · the inference service's precision rungs above fp32 · **≈ 0.25k / ≈ 0.35k** | production has no matmul key (runs `highest`); no golden depends on TF32. KEEP: `torch_state_guard`'s `allow_tf32` leak guard, the inference-service freeze check, `run_io`'s provenance stamp | `src/agents/training/CLAUDE.md` names the flag → `deleted_flags.md` row in the same commit | with/after R8 |
| R10 | **The Python-vs-Rust harnesses** whose subject is gone (follows D3) | `main/rust_core_cutover/` 1,825 / 589 · `main/rust_core_m5/` 3,263 / 721 (keep any piece the X26 / sizing reads still call — check at the unit) · `rust_env_*_parity` ≈ 1.3k · `utils/rust_env/bot_corpus.py`'s re-record | the bank of last-green runs (D3) is their record | a harness the sizing study still runs | last of the spine, with R1 |
| — | **Stays BLOCKED (T10 / M7), not in this pass** | trackers / `TurnDelta` / `episode_tracker` / the assembler / the `live_view` memo — no longer reached by training, but `agents/inference/player.py`'s `RLPlayer` runs them for `play.py`, anchors, the snapshot ladder (`ladder.json`), the untaught meter, the eval worker, the prober; `clone_pins.py`, `materialize_branches`, `_ReplayObsPlayer` (M7); node `search_driver.js` / `replay_driver.js` / `replay_kernels.js` (still diffed); `search_session.py` (393; in-process `Successors` exists but search's default + the prober / teacher / reconstruction still use the protocol); `turn_delta_legacy.py` (a test rewire). **One exception, deletable now:** `choice_band_tracker.py` (240, no non-test importer) — fold into R2 | | | |

**Before → after (estimate, ±25%; each unit re-measures at its commit).** `src/` today: Python 222,101 non-test /
189,044 test (686 / 704 files), Rust 119,970, JS/TS 75,511, vendored `poke_env/` 18,045. The pass removes ≈ **21k
non-test Python (−9.5%, → ≈ 201k)** and ≈ **27k test lines (−14%, → ≈ 162k)**; Rust, JS and `poke_env/`
unchanged. `combination_checks.py` (1,499) loses the 16-row unported table, the env-core / fork / cf / distill /
shaped families — target **< 1,000** after the census. Command (from `src/`):
`F() { find . \( -path ./poke_env -o -name target -o -name node_modules -o -name __pycache__ \) -prune -o -type f "$@" -print0; }; F -name '*.py' ! -name '*_test.py' | xargs -0 cat | wc -l`.

---

## 2. THE LEVER LIST — everything that runs ONLY on `--env-core python`

Source: `combination_checks._ENV_CORE_UNPORTED` (15) + `env_core_rust_needs_the_winprob_critic` (`--critic
shaped`) + the fork arm. **Every one is OFF in `designs/production_config.json`; the only open backlog row that
needs any of them is X15 (distillation / search-as-teacher).** Ledger line numbers are `ledger.md` at `b2c09132`.

| lever (plain words) | flag(s) | last use → result | open need | rec. |
|---|---|---|---|---|
| **Shaped critic** — V predicts a discounted, PopArt-rescaled hand-shaped return instead of P(win). **The bare-argv default** | `--critic shaped` (+ `--use-popart`, `--value-dist-*`, `--value-from-dist`, `--value-tail-weight`, `--win-prob-coef`, non-unit `--victory-value`, `--draw-penalty`, `--terminal-indicator` off) | 2026-09-16 *FLYWHEEL-ERA PAIR READ*: shaped vs winprob at 75M, strength NOT DETECTED (+17.5 Elo). Owner: win-prob IS the critic | none (production + N0 are winprob) | **DELETE** (§2.1) |
| **λ-return value target** — V's target blends multi-step bootstraps instead of the pure outcome | `--win-prob-lambda < 1` | 2026-09-11 (L17453, L17578): λ 0.9 not confirmed by its replicate; λ 0.95 failed its registered claim | none (X2's λ is `--policy-gae-lambda`, unaffected) | **DELETE** |
| **Monte-Carlo rollout target** — replay a state and play it out R times for a richer win label | `--win-prob-rollout-target` (+`-r/-mode/-weight`) | 2026-09-11 arm 10 (L17634): null on every primary row; "richer-label family CLOSED at 10M" | X4/X6 playouts DEFERRED (one-ply rule); a future one runs on Rust search | **DELETE** |
| **Dense auxiliary targets** — 25 end-of-battle facts predicted from every state | `--win-prob-dense-aux` | 2026-09-10 arm 9 (L17087): nothing on any row | none | **DELETE** |
| **Privileged true-team critic** — V sees the opponent's real team (a ceiling probe) | `--value-true-team` | 2026-09-10 arm 5 (L16879): "hidden information is NOT the limit" | none | **DELETE** |
| **Defensive entropy boost** — more exploration when a recovery / cure move is useful | `--defensive-entropy-boost` | no ledger entry names it | none | **DELETE** |
| **Bait entropy boost** — more exploration when the likely attack does zero into a revealed bench mon | `--bait-entropy-boost` | 2026-08-23 BUILT, never run; bait hunt CLOSED | none | **DELETE** |
| **Distillation** (the "fold") — a KL term pulling the generalist toward per-team teachers, + anchor / stop-rule / gradient-projection machinery (~25 `--distill-*` flags) | `--distill-coef`, `--distill-teacher`, `--distill-*` | 2026-09-23 (L21217): wider targets cost MORE; the era-1 fold trained worse than nothing, the distill LOSS carried the cost (2026-09-20). v8's +69 anchored Elo is attributed to the parent still learning (UNDERSTANDING TL;DR 3) | **X15 (expert iteration) YES**; population loop NO (specialists are opponents); ladder campaign round 1 NO | **DELETE**; port when X15 is scheduled (~1–2 agent-days: `distill_mask` becomes a per-episode host key from `TeamStager`, teachers frozen T2 slots or a learner-side forward) |
| **Self-PBRS** — adds γφ(s′)−φ(s) from our own win-prob head to the reward | `--win-prob-pbrs-coef`, `--win-prob-pbrs-source` | 2026-08-30 clean-world runbook (L7013); already refused under winprob (double counting) | none | **DELETE** (dies with shaped) |
| **Frozen-φ PBRS** — a FROZEN win-prob head reshapes the POLICY's advantages only, so V stays P(win) | `--win-prob-pbrs-frozen` | registered 2026-09-06 (famine pre-test); a confirmed run read not found | UNDERSTANDING keeps the owner's open question "do we need PBRS?" — no X row | **DELETE**; port ~0.5 agent-day if the owner wants that question kept |
| **cf_records** — a bridge reconstruction ring that makes episodes replayable for counterfactual labels; with it every cf CONSUMER (`--cf-winprob-coef`, `--cf-twin-*`, `--cf-shadow-*`, `--cf-evidential*`, `--q-winprob-*`, `--cf-label-*`, `--cf-head-only`) | `--cf-records` | 2026-09-30 (L21651): "only 3 runs ever ingested a cf label"; the cf-labels arm and the leaf battery (2026-09-12) found no lever moves leaf quality | X4/X6 deferred, and would use Rust search, not the ring | **DELETE** the training half (D6 holds the offline readers) |
| **Search-as-teacher** — a search worker finds a better action, an advantage-weighted cross-entropy pulls the policy to it | `--search-teacher`, `--teacher-*` | 2026-09-22 (L21018): the composition gate's verdict never recorded; labels not reproducible; search wound down 09-12 | X15 conceptually — would rebuild on the Rust search driver | **DELETE** |
| **Team-PFSP** — sample OUR teams by self-play win rate | `--team-pfsp` | 2026-08-30 probe P (L6742) | OFF in the recipe (§3.20); the ladder campaign replaces it with one PLR sampler (a new build) | **DELETE** |
| **Exploiter ladder** — a ratchet curriculum of exploiter rungs | `--exploiter-ladder` | built 2026-08-28 (F6); no strength read | the new-lineage loop uses a stable-set WINDOW (share 0.40) — `--exploiter`, `--stable-opponents`, `--pfsp-scale` ARE served on Rust | **DELETE** (port ~0.5 agent-day if wanted) |
| **Async rollout** — a SubprocVecEnv wave-scheduling mode | `--async-rollout` | 2026-08-29 built OFF | the Rust collector has its own trigger | **DELETE** (R1) |
| **Python fork arm** — replay an episode to turn t and branch an alternative action | `--fork-fraction` on python | 2026-09-16 (L18815) `ai_v13_03_fork` NOT DETECTED | the Rust port (`rust_rollout/fork.py`, forks.md §14) is DECLARED + OFF; its §14.11 checklist gates enabling | **DELETE** the Python half (R3), keep the port |

**What DELETE-ALL loses, plainly.** (1) **Distillation** is the only BUILT route to X15 (expert iteration /
search-as-teacher, multi-team KL distillation) and to the K-ladder / anchor / stop-rule instruments — UNDERSTANDING
still calls the flywheel "the intended engine". The port is cheap and is better built on the Rust core when X15
is scheduled than kept alive on a dead one. (2) **Frozen-φ PBRS** is the only built test of "do we need PBRS
under the win-prob critic" — never answered. (3) **Shaped-era comparability on HEAD**: every shaped / pre-gen-17
comparator runs PINNED from here (already true for the shaped reward since `e3ef16db`). Nothing else on the list
has a live user or an open question.

### 2.1 The bare-argv default (D2)

`--arch production` already applies `critic winprob`, `gamma 1.0`, `terminal_indicator true`, `victory_value 1.0`,
`draw_penalty 0.0`, `env_core rust`. The PARSER defaults still say shaped / python. The change:
`critic_mode.CRITIC_DEFAULT` → `winprob` · `--terminal-indicator` → on · `--victory-value` 30.0 → 1.0 ·
`--draw-penalty` −35.0 → 0.0 (winprob REQUIRES these and deliberately does not imply them; otherwise a bare argv
FATALs on three `winprob_critic_*` checks) · `--env-core` → `rust`. Then, once R2 deletes shaped, the critic,
indicator, victory-value and draw-penalty flags have one legal value each → the census deletes or fixes them.
`recipe_surface.py:156`'s critic row, `CRITIC_DEFAULT`'s comment, the root `CLAUDE.md` smoke text and ~28 test
files that pin the default move with it; `MODEL_CONFIG_VERSION` bump (129 → 130). Resume safety: these fields are
recorded + resume-immutable — the unit adds a test that a flagless resume of a SHAPED run still reads `shaped`
from `model_config.json` (and is then refused, D4), never the new parser default.

---

## 3. THE BOUNDED PAYDOWN LIST

| # | item | source | size (agent-days) | depends on | lane |
|---|---|---|---|---|---|
| P1 | **Opponent loads must not reseed the global RNG.** `InferenceMaskablePPO._setup_model` (`instrumented_ppo/inference.py:63`) calls `set_random_seed(self.seed)`: every pool refresh / sentinel / teacher load re-seeds Python `random`, NumPy and torch to the SNAPSHOT's saved seed mid-run. Any later global-RNG draw (team builders that draw from the global `random`, `rust_collector.md` "Teams, seeds") replays a stream | **SHIPPED 2026-10-02** (`gen3_no_global_reseed_v1`): Rust core: only the minibatch permutation replayed; python core: the team curriculum (ledger 2026-10-02). Loads isolated, a global seed after the K6 freeze is FATAL, static gate `src/global_rng_seed_gate_test.py` | 0.25 (+ a check of which Rust-core draws read the global RNG — GIGO class, so first) | — | C |
| P2 | **Opponent T2 slots compute the value forward they never use** (`inference/service/decision.py` runs `_critic_value` for every slot) — a policy-only forward for non-trainee slots | queued 2026-10-02 | 0.75 (T2 slot identity + parity tests) | GPU memory fix landed | C |
| P3 | **The last update's scalars are never dumped**: `dump_logs` runs BEFORE each update (the TB step contract), so the final update's `train/*` stay pending at `learn()`'s end (switch report #3) | switch report | 0.25 | — (folds into R4 if R4 is in flight) | A |
| P4 | **Launcher `render error: cannot convert float NaN to integer`** (`main/launcher/app.py:325`, once, headless) | switch report — ✅ **SHIPPED 2026-10-02**: the cause was `format._fmt_val`'s `int(v)` on a NaN metric value (the render guard swallowed it and the tick's whole dashboard went unpainted); a non-finite metric now renders as `nan` / `inf` (`launcher_app_test`) | 0.25 | — | C |
| P5 | **The arch-report cosmetic lines** the switch report flagged | orchestrator — ✅ **SHIPPED 2026-10-02**: (a) a same-run restart's report read `damage_matrices_outgoing/incoming False (this argv, default)` against a run that recorded True — `checkargs`' inheritance sweep covered parser dests only, and those two are desugared from `--damage-matrices`; it now also sweeps every registry row's attribute; (b) the `--arch` block's NOT-applied list printed the mirror's value for a TYPED flag (the ride-along heads) — it now prints the typed value, marked `(typed)` (`arch_surface.unapplied_for_argv`) | 0.25 | — | C (🚨 the lines are NOT named in `m5_switch/README.md` — the orchestrator names them at dispatch) |
| P6 | **Final-eval replacement** (TECH_DEBT §2(g)) — per D5, delete `final_eval.py` + `--final-eval` | §2(g) P3 | 0.5 | D5 | A (parser) |
| P7 | **`proc_integration_test` flake** — apply the row's named unscaled bounds (`op_timeout=2.0` bounding setup, `_exited(wait=10.0)`, `startup_timeout=20`) through `utils/contention`; keep the traceback hunt open | §2(b) P2 — ✅ **SHIPPED 2026-10-02** (the bounds; the TECH_DEBT row stays OPEN for the traceback) | 0.5 | — | C |
| P13 | **The PINNED-TEAM meters REFUSE `--mirrored-pairs`** (owner-directed 2026-10-02: the fixed-team meters measure TWO different things — piloting the pinned team and the response to it as an opponent — so mirroring them must not pool the two) — `main.untaught_meter --mirrored-pairs` and `main.best_response_gap --play --mirrored-pairs` refuse with a typed error (`MirroredPinnedTeamError`, cause `mirrored_pinned_team`) at the CLI and at `untaught_meter.play_cells(mirrored=True)`; the in-loop `--eval-mirrored-pairs` and `main.anchors --mirrored-pairs` (symmetric) are untouched | owner via orchestrator, 2026-10-02 — ✅ **SHIPPED 2026-10-02** | 0.25 | — | C |
| P8 | **Test-speed rows** (§2(b) P2): parity gates subprocess-startup bound — cache `fresh:N` pool zips per session (S–M, 1.0) · `Teambuilder` Node validator per construction — memoise (S, 0.5) · fixed sleeps in 3 tests (S, 0.5) · anchors smoke 22 s of process startup (M, 1.0). P3 rows (mypy cache warm, stub-scan profile) only if the box allows | §2(b) | 3.0 | AFTER the deletions (re-measure first: R1/R2/R10 delete many of the slowest parity files) | C |
| P9 | **Distillation teachers carry ride-along weights** | queued 2026-10-02 | 0.25 | **MOOT if D1 deletes distillation** | — |
| P10 | **ONE independent code review** of the newest core modules (owned loop, lifecycle, compile regions, inference-only load) | TECH_DEBT §1 | 0.75 | AFTER R4 (so it reads the owned buffer / logger / hooks, not the code R4 replaces) | C |
| P11 | **FLAG CENSUS** — every surviving flag names a live user; the rest go with code + checks + tests (`deleted_flags.md`) | TECH_DEBT §1 | 2.0 | LAST — after every R row (the deletions shrink its input from 303 flags) | A |
| P12 | **Stale docs found while scoping** (fold into the unit that touches each): the refusal text's pointer to `rust_collector.md` "'What --env-core rust refuses'" (no such section; R2 deletes the table) · `design_model_management.md:31,251` (v123 is `policy_gae_lambda`, not `--matmul-precision`) · forks.md §14.1 (the port uses `store.game_gae`, not `branch_rewards`) · program §4's LOC and survivor claims (§7) | scoping | 0 (in-unit) | — | each |

---

## 4. THE PLAN — units in order, three lanes

Each unit is one commit with the routine gate green. **Serialized shared files** (one in-flight unit at a time
edits them; the lane that holds one hands off on ship): `main/train/combination_checks.py`, `main/train/config.py`,
`main/train/parser/*` (per family file — two units may hold DIFFERENT family files), `train_rl_agent.py`,
`instrumented_ppo/loop.py`. **Live agents to rebase around:** the SIZING verdict (`production_config.json`,
`recipe_surface`, Decision records → U1 rebases after it), the GPU memory fix (`device_batches.py`, `update_fit.py`,
`cuda_memory_trend.py`, the lifecycle gate → R4, R8 and P2 wait for it).

| unit | content | agent-days | lane | holds | after |
|---|---|---|---|---|---|
| **U0** | owner signs D1–D6 | — | — | — | — |
| **U1** ✅ **SHIPPED 2026-10-02** (`gen3_bare_argv_winprob_v1`, config v130) | bare-argv default (§2.1) | 0.75 | A | parser `clean_world` / `reward` / `env_core`, `critic_mode`, `recipe_surface` | U0, sizing verdict |
| **U2** ✅ **SHIPPED 2026-10-02** (D3: no banked run) | extractions: standalone spaces builder (off `Gen3Env`), `OPP_CLASS_*` / `STABLE_CHALLENGE_SHARE` / `resolved_obs_source` / `_build_trainee_tb` re-homed; D4's python-era resume rule; D3's banked last-green run of every env-level gate | 1.0 | A | `rust_rollout/build.py`, `rust_env_setup.py`, `rust_env_opponents.py` | U1 |
| **L1** | shaped-only levers + self-PBRS + frozen-φ (R2) | 1.0 | B | `combination_checks`, `config`, parser `clean_world` / `distillation` / `value_heads` | U1 |
| **L2** | λ, rollout target, dense aux, true team, entropy boosts, `choice_band_tracker` (R2) | 1.0 | B | same + `hyperparameters`; `label_inventory` | L1 |
| **L3** | distillation + search teacher (R2) | 1.0 | B | + parser `capacity` / `distillation` / `teacher`, `matchup_setup` | L2 |
| **L4** | cf training half + team-PFSP + exploiter ladder + `REFUSED_WITH_FLAG` (R2) | 0.75 | B | + parser `cf_grounding` / `eval_subprocess`, `rust_vec_env` | L3 |
| **L5** | the Python fork arm (R3) | 0.5 | B | `combination_checks`, `fork_*` | L4 |
| **K1** ✅ **SHIPPED 2026-10-02** (`utils/torch_floor.py`; the cache-limit log detector KEPT — §6 finding 10) | torch < 2.8 refusal; 2.5.1 code + legacy compile gate (R7 + R8), the `hooks.py:179` fix | 1.25 | C | `compile_trainer`, `compile_control`, `team_transformer`, `lifecycle.py` | memory fix |
| **K2** | RETIRE TF32 (R9) | 1.0 | C | parser `hyperparameters` (after L2 hands it off), `consistency.py`, `parity_probe`, one `combination_checks` row | K1, L2 |
| **P1, P4, P5, P7** | small fixes | 1.25 | C | disjoint | — (P1 first) |
| **P2** | value-free opponent slots | 0.75 | C | `inference/service/*` | memory fix |
| **U3** | DELETE the Python env core + collect + async + compile-opponents trim + harnesses (R1 + R6 + R10) | 1.5 | A | `train_rl_agent`, `env_factory`, `rust_env_setup`, `combination_checks` | U2, L5 |
| **U4** | PPO stage 3 + SB3 leftovers (R4 + R5) + P3 | 2.5 | A | `loop.py`, `rust_vec_env`, `device_batches`, `run_io`, callbacks | U3, memory fix |
| **P6** | final-eval deletion | 0.5 | A or B | parser `operational`, `final_eval.py` | U0 (any gap in lane A/B) |
| **P10** | independent code review | 0.75 | C | read-only → fixes as units | U4 |
| **P8** | **OUT OF THE BOX (orchestrator default 2026-10-02, owner may override): stays in TECH_DEBT_BACKLOG** — test-speed rows (re-measured first) | 3.0 | C | test files, `utils/bridge/team_validator.py` | U3 |
| **P11** | FLAG CENSUS | 2.0 | A | everything | every unit above |
| **GATE** | slow tier as the MILESTONE run (`-m slow -n 2`, refreshes `slow_tier_status.json`) | ~1 wall-hour, 0.25 | — | — | P11 |

**Totals.** Lane A (spine) 6.25 · lane B (levers) 4.25 · lane C (compile / precision / paydown) 8.0 · census +
milestone 2.25 → **≈ 21 agent-days (range 17–26)**. Program §2 M6 budgeted "~2 agent-days" for this pass; the
difference is the lever list (R2, ≈ 8.8k lines + their checks and tests), stage 3, and the paydown — none of
which that figure counted. **Critical path:** U1 → L1…L5 → U3 → U4 → P11 → slow tier ≈ **11.25 agent-days**. At the
program's observed pace (own-loop stages 1 + 2, estimated 2.85 agent-days, both landed 2026-10-01) that is **≈ 5–7
calendar days with 3 lanes; 8 is the box**.

---

## 5. EXIT CRITERION

1. **The list is CLOSED** at the owner's approval. A row is added only with the owner's word; a fact found
   mid-pass that needs a new row is a FINDING to the orchestrator, not scope creep. A row may be DEFERRED only by
   the owner (it moves to `TECH_DEBT_BACKLOG.md` with its reason).
2. **DONE** = every row (R1–R10, the levers per D1, P1–P12) shipped or explicitly deferred by the owner, **AND**
   the routine gate green on `main`, **AND** the slow tier green as the milestone run (`slow_tier_status.json`
   refreshed; no recorded FAIL), **AND** the `--debug` smoke and one real-launch pre-flight (the first minutes of
   `--arch production` under the launcher, the preload layer's only test) green on the final commit.
3. **TIME BOX: 8 calendar days from approval.** Day 4 is a checkpoint (the orchestrator reports shipped vs
   remaining to the owner). At the box: a unit IN FLIGHT finishes; every unit NOT STARTED is deferred to
   `TECH_DEBT_BACKLOG.md` by default, with the orchestrator's one-line case for each; **the census and the
   milestone slow tier run regardless** (on whatever shipped); then the sequence moves on to T15. Exception that
   holds the box: if U3 has shipped and U4 has not, U4 finishes first — a deleted env core with SB3's VecEnv glue
   half-removed is not a state to start T15 on.

---

## 6. FINDINGS raised while scoping (each owned by the unit named)

1. ✅ **CLOSED by U2 (2026-10-02):** production startup no longer instantiates `Gen3Env` — the spaces come from
   `agents.training.trainee_spaces` (which `Gen3Env` also builds through), the constants from
   `agents.training.opponent_classes`, the eval trainee builder from `agents.training.eval_teams`; the trainer's and
   the rollout probes' python branches import the core lazily. `agents/training/trainee_spaces_test.py` proves it at
   RUNTIME (a fresh interpreter resolving a production argv, building its spaces, opponent plan and eval table, never
   loads the five core modules) and STATICALLY (every remaining importer is declared with its deletion unit: the
   two lazy branches → U3/U4, the R10 harnesses and the bridge benchmarks → U3). D4's rule is in
   `rust_env_setup.resolve_env_core_default`.
2. 🚨 **Deleting the Python core deletes the Rust env core's ORACLE** (D3): the env-level parity gates are not on
   program §1's survivor list.
3. ✅ **P1 was a GIGO class — CLOSED 2026-10-02** (`gen3_no_global_reseed_v1`). On the Rust core the only
   steady-state global-RNG consumer is the minibatch permutation, which replayed after every load (no effect on
   which rows train). On the python core every worker's team draws replayed, so its team curriculum was a
   seed-fixed skew (ledger 2026-10-02).
4. ✅ **CLOSED by U1 (2026-10-02):** the root `CLAUDE.md` `--debug` smoke now runs the Rust core (bare argv →
   `--critic winprob` → rust; Training complete, K9(b)'s excluded share 1.6–4.5 % on its 2,048-row updates, under
   the 0.15 ceiling — TECH_DEBT §2(b)'s P3 small-rollout row did NOT trip, so it stays open, unfixed). U1 split
   "the bare-argv default" (`CRITIC_DEFAULT` = winprob) from "what an ABSENT record means" (`CRITIC_UNRECORDED` =
   shaped, the `_REWARD_IMMUTABLE_FIELDS` signed terminal): no checkpoint loads differently, so no `ARCH_SIGNATURE`
   bump (`designs/model/versioning.md`).
5. **`--cf-label-supply external` passes launch checks on Rust and would starve in flight** — unverified at
   runtime; R2/L4 deletes it.
6. ✅ **`rust_core_m5/hooks.py:179` measured the legacy compile on 2.8, not production's R0 region** — FIXED by K1
   (`compile_regions.install_rollout_region`).
7. **Program §4 is wrong in four places** (§7). Not a deletion, a correction; this doc carries them until the
   pass edits §4 row by row.
8. **184 recorded argvs carry `--compile-opponents`** (and older runs carry other flags this pass deletes): an
   UNPINNED HEAD resume of such a run fails argparse. The launcher pins by default; the census decides whether
   `resolve_config` should strip deleted flags from inherited argvs with a printed line.
9. **Not verified:** that every pre-2026-09-30 run records no `torch_version` (inferred from the launcher rule);
   that `b2c09132` runs under 2.5.1 at all; that `rust_rollout/fork_test.py` pins forks.md §5's mask rule; the
   test-by-test fate in `extractor_compiles_test`, `train_test`, `rust_eval/parity_test`, `eval_sharding_fuzz_test`;
   whether the `--distill-anchor-*` / `--distill-stop` family can run without `--distill-coef > 0` (if so it is
   silently inert on Rust today). The P5 arch-report lines were not located.

10. **K1 (2026-10-02): R7's "replace the cache-limit LOG detector with `fail_on_recompile_limit_hit`" was NOT
    done — the detector is KEPT.** Dynamo asserts `fail_on_recompile_limit_hit` and `suppress_errors` are never
    both set, and `main/compile_inventory/capture.py` patches `suppress_errors=True` in the trainer process; the
    regions compile `fullgraph=True`, where torch 2.8 already raises `FailOnRecompileLimitHit` on a limit hit, so
    the detector's remaining job is the sticky FATAL_CONFIG. Also from K1: `compile_trainer.eager_extractor` is a
    no-op on the learner (its callers are L3/L5's to delete); no test now drives a full first `train()` through
    the compiled REGIONS with the donating default forced (the deleted `compiled_train_probes_test` did it for the
    extractor compile); the region gate's collapsed-critic ladder climb has no teeth test (the deleted
    `parity_probe_test` gate tests had one for the extractor gate).

## 7. Corrections to `program_rust_core.md` §4 (applied by the unit that executes each row)

| §4 row | says | measured 2026-10-02 |
|---|---|---|
| M5: per-env bridge child | `bridge_session.py` + `battle_stream_client.py` "287 + 810" | `async_vec_env` 287; `bridge_session` 667 goes; **`battle_stream_client` (178) SURVIVES** (eval, ladder, meters, prober) |
| M5 / T2: compile-opponents | delete `compile_opponents.py`, `compile_preload.py`, `compile_prewarm.py`, the pool cache — 673 | 709; **`compile_opponents.py`'s core and the pool's LRU cache STAY** (offline users, T2 weight loads) — R6 |
| M5: `gen3_env.py` + `wrappers.py` | "most of 1,711" | 1,779; UNBLOCKED by U2 (2026-10-02) — no production dependent left |
| SKIPPED rows (trackers, assembler, memo) | blocked on training's opponents | the blocker MOVED: training no longer reaches them; `RLPlayer` (offline players) does → still M7. Only `choice_band_tracker.py` is free now |
| M5 + fork port | the Python fork arm's tests | add `fork_crn_test.py` (138) |
