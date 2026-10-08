# P5 of the poke-env retirement — the PROBER reads the Rust core (2026-10-07)

`gen3_pokeenv_p5_identity_v1`. Backlog T27, plan
[`../pokeenv_and_hotpath_survey_2026-10-06/README.md`](../pokeenv_and_hotpath_survey_2026-10-06/README.md) §A4.5 (P5).

**Verdict: IDENTICAL.** On a fixed set of 21 banked battles from 3 runs, 283 of 284 prober JSON-CLI captures are
byte-identical before vs after. The one difference is `replay-counterfactual`, which the after side ran with poke-env
BLOCKED and which still needs it (the declared P6-blocking command). Run with poke-env available, the new code's
`replay-counterfactual` is byte-identical too. The after side's only poke-env import attempt was that one command.

## What moved

| prober path | before | after |
|---|---|---|
| a core trace's expansion (`core_trace.load_summary` / `protocol_log`) | the reconstruction replayed on the rust driver, its trainee chunks fed through a poke-env `Player` into a real `BattleRecorder` | `core_events --walk` (per decision the core's `present()` view, legality, slot registries, frozen `TurnDelta` projection, mask, choice tokens; the terminal view + delta) → `main/prober/core_walk.py` + `core_recorder.py` |
| the `*_replay.html` stand-in | `battle._build_replay_events()` of that poke-env battle | `core_walk.replay_log`: the player's dispatch over the walk's trainee stream |
| falsify / lookahead / better-line: a decision's legal choice map | the poke-env materializer's `map_actions_at` | `core_walk.decision_choices` (the core's `present::choice_tokens`) |
| lookahead / better-line: successor rows | `materialize_branches` / `materialize_decisions` (poke-env battle + Python encoder) | `core_walk.read_streams` → `core_events --obs-stream` (the parse chain with the trackers on, the chain training's rows are encoded on) |
| better-line: the interior opponent's action history | `infer_action_indices` (a poke-env replay) | each walked decision's recorded choice inverted through its core tokens |
| `engine/switch_in.py`'s type enum | `poke_env…PokemonType` (a blocked import returned `None` silently) | the owned `agents.enums.PokemonType` |
| `forensics` (`decision-table`) | `turn_delta` imported poke-env at module load | `DamagingMoveEvent` is an annotation (TYPE_CHECKING) plus one lazy run-time import in the fold |

Shared, so the two writers of the summary shape cannot drift: `agents/training/trace_labels.py` (every label,
`BattleRecorder` delegates to it) and `agents/training/reward_config.terminal_breakdown` (the reward rule,
`Gen3RewardManager` calls it). The poke-env import allowlist went **138 → 136**
(`src/main/prober/core_trace.py`, `src/main/prober/engine/switch_in.py`).

## The identity read

**Fixed set.** `capture.py`'s `BATTLES`: 9 `sizing_C_n256_e5_s1001` battles (core traces; the run is at the CURRENT
architecture, so every model-loading command runs there), 7 `rb_x5ab_blob_s1008` battles (core traces), and 5
`ai_v14_01_base` battles (poke-env-eval traces with a stored summary and `_replay.html` — the control, which P5 does not
touch). Wins, losses, ties and two 250-turn stall timeouts are included. The battles were COPIED read-only out of
`models/` into a closed fixture (`capture.py fixture`), so a run-level command reads exactly these battles.

**Commands.** Per run: `summary`, `list`, `scan` (both metrics), `awareness`, `loops`, `triage`, `switch-vs-info`,
`decision-table` (digest and rows), `calibration` and `falsify-scan` (`--impl rust`, small budgets). Per battle:
`turns`, `overview`, `find` × 8 model-free criteria, and `falsify` on every non-win. Model-loading, on `sizing_C`:
`analyze` × 3 decisions × 4 battles, `find disagree` × 4, `lookahead` × 2, `better-line` × 2, `probe`,
`history-saliency`, `replay-counterfactual`. Each capture is `query._run` on the CLI's own parser, so it is exactly
the JSON `python -m main.prober.query <argv>` prints.

**Regime (pinned, both sides).** `PYTHONHASHSEED=0` and 4 torch threads. BEFORE ran the base commit `6c6d2e09` from a
detached worktree (`--src`); AFTER ran this change with `utils.poke_env_blocker` installed first (`--blocked`). Why
both pins are needed is F-P5-1 and F-P5-6 below.

```bash
python capture.py fixture --dir /tmp/p5/fixture
PYTHONHASHSEED=0 python capture.py run --fixture /tmp/p5/fixture --out /tmp/p5/before0 --src <base-commit worktree>/src --threads 4
PYTHONHASHSEED=0 python capture.py run --fixture /tmp/p5/fixture --out /tmp/p5/after0 --blocked --threads 4
python capture.py diff /tmp/p5/before0 /tmp/p5/after0 --md identity.md
```

`before_manifest.json` / `after_manifest.json` hold each capture's argv, sha256 and wall time.
[`identity.md`](identity.md) is the table:

| command | captures | byte-identical |
|---|---|---|
| `analyze` (3 decisions × 4 battles) | 12 | 12 |
| `awareness` | 3 | 3 |
| `better-line` | 2 | 2 |
| `calibration` | 3 | 3 |
| `decision-table` (digest + rows) | 6 | 6 |
| `falsify` | 15 | 15 |
| `falsify-scan` | 3 | 3 |
| `find` (8 model-free criteria) | 168 | 168 |
| `find disagree` | 4 | 4 |
| `history-saliency` | 1 | 1 |
| `list` | 3 | 3 |
| `lookahead` | 2 | 2 |
| `loops` | 3 | 3 |
| `overview` | 21 | 21 |
| `probe` | 1 | 1 |
| `replay-counterfactual` | 1 | 0 (BLOCKED by design; identical with poke-env available) |
| `scan` (both metrics) | 6 | 6 |
| `summary` | 3 | 3 |
| `switch-vs-info` | 3 | 3 |
| `triage` | 3 | 3 |
| `turns` | 21 | 21 |

`awareness` errors identically on both sides: it reads the deleted distributional value head.

**Summary-level check (diagnostic).** Each of the 16 core battles was expanded by the base commit's
`core_trace` and by the walk, and compared field by field. `meta`, `teams` and every invocation were equal (0
differences). The protocol log first differed only by its opening `|init|battle` line, the room framing the client
receives; the walk now prepends it, and the logs are equal.

**Row-level check (diagnostic).** `core_walk.read_streams` over each battle's own trainee stream, with the stored
actions, reproduces the stored `states.npz` rows byte for byte on 19 of 21 battles (5 of them poke-env-eval traces,
whose rows the Python encoder wrote). The 2 others are F-P5-2. On those 2 the poke-env materializer and the core
reader agree with each other byte for byte.

## Standing gates (in the routine tier)

- `src/main/prober/core_trace_integration_test.py` (`sim`): the walk's summary equals a LIVE poke-env
  `BattleRecorder` field for field on real core games (a stall forfeit included). The protocol log equals the live
  `_replay.html`. The core stream's rows equal the stored rows, and its choice maps equal the poke-env
  materializer's. `PLAYER_IGNORED` equals the fork's `Player.MESSAGES_TO_IGNORE`.
- `src/poke_env_free_entry_points_test.py` (`sim`): every JSON-CLI command and the web app's views RUN with poke-env
  blocked. `PROBER_POKE_ENV_COMMANDS = ("replay-counterfactual",)` is the closed exception list. Measured 216 s at a
  contention factor of 4.26.
- `src/main/prober/core_walk_test.py` (unmarked): the script, the dispatch, the refusals, the stall rule.

## What still needs poke-env, and why

- **`replay-counterfactual`** (`replay.py`, `session/counterfactual.py`). It plays the rest of a battle LIVE from the
  divergence: the trainee as a greedy `RLPlayer`, the opponent as a reloaded `RLPlayer` or a Python bot, over the
  in-process bridge. Its port is a Rust play-out: `utils.rust_env.successors.play_out` with the in-core bot ports, the
  trainee's greedy forward over core rows, and the post-divergence reseed mapped onto play-out seeds. That port is
  P6-blocking and was not costed in P5's day.
- **The oracle tests.** `core_trace_integration_test.py` holds the core reading to poke-env and imports it on
  purpose until P6. `belief_obs_fuzz_test.py` is a model-side fuzz script that plays poke-env battles; it is P6's to
  migrate.

## FINDINGS

- **F-P5-1 — the live recorder's multi-faint order varies with `PYTHONHASHSEED`.** `BattleRecorder` names a turn's
  faints by iterating a `frozenset`, so two faints in one turn come out in hash order. That order also picks the slot
  of the outcome's HP delta. Two unpinned base-commit captures of `sizing_C` `step_8000030/sentinel_0/win_s1_001`
  disagreed: `turns` named the faints in the other order and read `damage 17%` against `9%`; `overview` disagreed too.
  The core recorder iterates in BOARD order, which is deterministic. Python-eval traces written by the live recorder
  keep the hash order they were written with.
- **F-P5-2 — HEAD re-encodes 2 of 21 banked traces differently from their stored rows.** `sizing_C`
  `step_2000128/aggressive/draw_s0_006` differs from decision 8 on, and `rb_x5ab_blob_s1008`
  `step_8000179/sentinel_0/loss_s0_004` from decision 49. Both differ at 2 opponent-team cells
  (`OFFSET_OPP_TEAM` +257/+258, stored 8.0 / 0.99, HEAD 0.0 / 1.0). The poke-env materializer at HEAD and the core
  reader agree with each other, so this is READING DRIFT since the runs' pinned commits (`7ef99979`, `706fa536`), not
  a P5 change. `analyze` reads the STORED row. lookahead / better-line successors are HEAD's reading of re-rolled
  text. **UNVERIFIED:** which commit moved those cells.
- **F-P5-3 — `materialize_branches`, `_PlayerSnapshot`, `open_branch_fork` and `clone_pins.py` have no production
  caller any more.** The prober was their last user. They are P6 deletions. P1's FINDING 3 (the clone shares the
  fork's `GenData`) no longer touches the prober.
- **F-P5-4 — better-line's interior-opponent history, refused probe.** For a refused opponent switch (a trapped
  probe), the walk's decision records the FIRST token sent, which is a legal switch, so the inversion keeps it. The
  poke-env inverter dropped a refused probe. This touches only depth ≥ 2 with a reloaded opponent, on a battle where
  the opponent probed a trap. **UNVERIFIED** on a real case: the identity set has none.
- **F-P5-5 — `--impl node` now mixes engines.** falsify / lookahead / better-line take their choice maps and rows
  from the Rust core even under `--impl node`; only the re-rolls run on Node. The two engines are byte-identical by
  the port's gates, and the identity read ran `--impl rust`.
- **F-P5-6 — `analyze`'s recomputed probabilities move with torch's thread count.** Default threads (16) against 4
  moved the `rerun` probabilities in their 7th–8th significant digit on all 12 `analyze` captures. An identity
  check of a model-loading view must pin the thread count. On a contended box, unpinned default threads also stalled
  `probe` / `history-saliency` about 70× (553 s against 7.8 s), which is how the pin was found.
- **F-P5-7 — the model-loading half of this read cannot be repeated on archived runs at today's HEAD.** The read
  ran at the base commit `6c6d2e09`, where `sizing_C` was the current architecture. The X5 version break (config
  v144, `MIGRATION_FLOOR` 144) landed while P5 was built and puts every archived checkpoint behind the floor. At
  HEAD, model-loading prober views on any archived run are an `ArchDriftError` diagnosis, P5 or not. The blocked
  test's fresh v144 checkpoint is what covers the model-loading commands from here on.
- **F-P5-8 — two Rust one-side readers now exist.** P4 (`25ea2cc6`, landed concurrently) built
  `src/rust_sim/src/side_reader.rs` + the `live_reader` binary for live play. P5 built `core_events --obs-stream`
  for the prober's counterfactual rows. Both fold one side's TEXT through the parse chain with the trackers on. They
  should converge on one reader, in P6 or before; this was not done here.
