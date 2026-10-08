# P6 slice 1 of the poke-env retirement — `replay-counterfactual` on the Rust play-out (2026-10-07)

`gen3_pokeenv_p6_replay_cf_identity_v1`. Backlog T27; plan
[`../pokeenv_and_hotpath_survey_2026-10-06/README.md`](../pokeenv_and_hotpath_survey_2026-10-06/README.md) §A4 (P5's
one deviation, P6's first slice); the P5 read this continues: [`../pokeenv_p5_prober_2026-10-07/`](../pokeenv_p5_prober_2026-10-07/README.md).

**Verdict: IDENTICAL.** On the P5 identity set (21 banked battles, 42 decisions), every regime the old road could be
made deterministic in — greedy trainee vs a greedy checkpoint, vs its own greedy self, vs a roster bot, vs a SAMPLED
checkpoint, each at `n = 1` (the battle's own dice) and `n = 4` (post-divergence reseeds) — gives the SAME outcome, the
SAME last turn and the SAME protocol for our side, line for line, on every rollout: **336 / 336 cases, 840 / 840
rollouts, 691,043 protocol lines** (+ 48 / 48 cases, 120 / 120 rollouts on a TRAINED checkpoint). The one textual
difference is the player NAME (F-P6-1). The prober's `replay-counterfactual` now runs with poke-env blocked; the
poke-env road was deleted after this read.

## What moved

| | before (the poke-env road) | after (`gen3_cf_core_playout_v1`) |
|---|---|---|
| engine | two poke-env players over the in-process bridge (`utils/bridge/counterfactual.py`: `run_local_battles`, both `choose_move`s SCRIPTED through the prefix) | ONE in-process `play_out` on the Rust core (`utils/rust_env/counterfactual.py` → `successors.play_out`) |
| divergence | each side's script popped until turn T; our side sent the substitute at its turn-T move round; the opponent its RECORDED turn-T move; live after | the playout root `at = {"turn": T, "other": "recorded"}`: replay to the START of turn T (the search tree's `build_to_turn`), feed the opponent's recorded turn-T choice (`recorded_turn_choices`), branch our side on the substitute; live after |
| our side | a greedy `RLPlayer` (the Python encoder) | `ModelPolicy`: `RLPlayer._predict_best_action`'s arithmetic on the CORE row, one B=1 forward per decision |
| a checkpoint / self opponent | an `RLPlayer` (stochastic: torch's process-wide generator) | `ModelPolicy` (stochastic: a per-rollout `torch.Generator`, seeded from the battle + decision) |
| a bot opponent | the Python bot (the process-wide unseeded `random`) | the IN-CORE Lane-F port (`bot` playout key, asked only at a real decision, never pending; rollout k's streams `bot_stream_seed(seed, k, s)`) |
| post-divergence dice | `resumeReseed`: the PRNG swapped just before turn T's FIRST choice was fed | each branch's engine reseeded at the branch point (after the opponent's recorded turn-T choice) — no draw lies between |
| stall forfeit | every `RLPlayer` forfeits at `turn >= StallConfig().threshold`, p1's processed first; a bot never | `stall.sides` = our side (+ the opponent when it is a model), p1 first when both are open |
| narrate | the trainee's chunks from `run_local_battles`' sink → `summarize_trajectory` | the core's protocol lines for our side (`text`) → the same `summarize_trajectory` (moved to `utils/rust_env/counterfactual.py`) |
| `--impl` | picked the bridge child (node / rust) | does not apply (one engine); an `--impl node` read says so in its `caveats`; `engine: rust_core` |
| reproducible | no (the bot and sampled opponents drew from process-wide streams) | yes — every draw seeded from `(battle_tag, inv)` |

Rust (`src/rust_env/src/search/`): `Game::replay_to_turn` / `feed_logged` / `feed_token` / `at_turn_start` /
`side_lines`; the playout's `at` object, `bot`, `stall.sides`, `text`. Gate: `src/rust_env/tests/search_playout_cf_test.rs`.

## The read

**Fixed set.** P5's `capture.py fixture` (the 21 battles: 9 `sizing_C_n256_e5_s1001` and 7 `rb_x5ab_blob_s1008` core
traces, 5 `ai_v14_01_base` poke-env-eval traces), copied read-only out of `models/`. Per battle up to two
`move_selection` decisions (an early and the middle one, ≥ 2 legal actions, turn ≥ 2); the substitute is the first
legal action that was NOT played.

**Models.** Two CURRENT-architecture checkpoints (`identity.py models`: seeded perturbed fresh production policies
built through the trainer's parser, CPU, 1 thread — archived runs are behind `MIGRATION_FLOOR`, F-P5-7), the trainee
and the "checkpoint" opponent. Plus a TRAINED one: `~/.cache/gen3ai/p4/models/p4_shadow_v144/final_model.zip` (copied
out; it loads at HEAD) as the trainee on the 6 `step_8000030` battles (one decision each).

**Regimes.** `ckpt_greedy` · `self_greedy` (the trainee's own checkpoint, greedy) · `bot` (the record's opponent when it
is a roster bot, else a roster bot by a hash rotation — 8 of the 9 bots drew) · `ckpt_sampled` (temp 1.0); × `n ∈ {1, 4}`.
For `bot` and `ckpt_sampled` the OLD road's streams are seeded exactly as the new road seeds them (the legacy builder's
`bot_streams` / `policy_seed` knobs — the Rust bot gate's rule), so the two roads draw the same numbers.

**Per rollout, IDENTICAL =** same outcome, same last turn, and our side's protocol lines equal (every `|`-line but
`|request|`, the bridge's JSON request frame, which the core road does not render), the poke-env account names mapped
to the recorded names (F-P6-1). Each old rollout also reports whether a script ran OUT before the divergence: none did.

**Regime (pinned).** CPU only; `PYTHONHASHSEED=0`; torch threads pinned per job (2 or 1; both roads in one process, so
one thread count per comparison — F-P5-6). Both roads at commit `6168924c` (the legacy road deleted after it; the
first three jobs started on its working tree before two import-only refactors — `replay.py`'s imports moved to
module level, `summarize_trajectory` moved to `utils/rust_env/counterfactual.py` — no behaviour between them).

```bash
python ../pokeenv_p5_prober_2026-10-07/capture.py fixture --dir /tmp/p6/fixture
python identity.py models --dir /tmp/p6/models
PYTHONHASHSEED=0 python identity.py run --fixture /tmp/p6/fixture --models /tmp/p6/models --out /tmp/p6/run.jsonl
PYTHONHASHSEED=0 python identity.py dist --fixture /tmp/p6/fixture --models /tmp/p6/models --out dist.json --n 40 \
    --battles step_8000030/sentinel_0/loss_s0_005,step_8000030/aggressive/win_s2_002
python identity.py table rows.jsonl.gz            # and rows_trained.jsonl.gz
```

(`run` / `dist` need commit `6168924c`'s `src`: the cut-over deleted the legacy road. The read ran as six sibling jobs —
`--reverse` / `--skip-from` — under `scripts/ops/mem_cap.sh`; `rows.jsonl.gz` is every job's rows, `table` keeps the
first row of a case and checks the rest.)

| regime | n | cases | cases identical | rollouts | rollouts identical |
|---|---|---|---|---|---|
| `ckpt_greedy` | 1 | 42 | 42 | 42 | 42 |
| `ckpt_greedy` | 4 | 42 | 42 | 168 | 168 |
| `self_greedy` | 1 | 42 | 42 | 42 | 42 |
| `self_greedy` | 4 | 42 | 42 | 168 | 168 |
| `bot` | 1 | 42 | 42 | 42 | 42 |
| `bot` | 4 | 42 | 42 | 168 | 168 |
| `ckpt_sampled` | 1 | 42 | 42 | 42 | 42 |
| `ckpt_sampled` | 4 | 42 | 42 | 168 | 168 |
| **all (fresh checkpoints)** | | **336** | **336** | **840** | **840** |
| **all (trained `p4_shadow_v144`, 6 decisions)** | | **48** | **48** | **120** | **120** |

What the rollouts covered (fresh checkpoints): 392 wins / 435 losses / 13 ties; median 48 turns, max 250; **14 lines
reached the stall cap** (12 `ckpt_greedy`, 1 `ckpt_sampled`, 1 `bot`) — all trainee (p1) losses on BOTH roads, so the
p1-first stall order held where both sides stall. Bots that drew: `setup_sweep_v2`, `heuristic`, `aggressive`,
`staller_v2`, `staller`, `setup_sweep`, `heuristic2`, `aggressive_v2` (not `random`). 0 errors, 0 exhausted scripts.
Trained trainee: 3 wins / 116 losses / 1 tie, 45 lines at the cap.

**Reproducibility of the new road.** 18 cases ran TWICE in separate processes (sibling jobs crossing): all 18 identical
(outcome, turn, protocol length per rollout). The `dist` read's two full prober calls per decision were identical.

**The SAMPLED regime's distribution** (`dist.json`; the old road UNSEEDED, as the CLI ran it, vs the new seeded road;
`ckpt_sampled`, n = 40 each, Wilson CIs, the difference's normal-approximation 95 % CI):

| decision | new (seeded) wins | old (unseeded) wins | new − old [95 % CI] |
|---|---|---|---|
| `sizing_C` `step_8000030/sentinel_0/loss_s0_005` inv 1 | 16 / 40 [0.26, 0.55] | 11 / 40 [0.16, 0.43] | +0.125 [−0.080, +0.330] |
| `sizing_C` `step_8000030/aggressive/win_s2_002` inv 1 | 34 / 40 [0.71, 0.93] | 35 / 40 [0.74, 0.95] | −0.025 [−0.176, +0.126] |

Both differences' CIs contain 0 — NOT DETECTED, a weak check (n = 40, half-widths ~0.15–0.2) next to the matched-seed
identity above, which is the claim: given the same draws the roads play the same games, so the distributions are equal.

**Cost.** New / old wall over the 336 cases: 4,236 / 4,555 s = **0.93** — both roads are B=1-forward-bound (F-P6-9).

## Standing gates

- `src/rust_env/tests/search_playout_cf_test.rs` (cargo): a divergence-turn root replays the recorded battle command for
  command (turn T's two choices in [other, ours] order) and ends with its winner, for either side as ours; `"other":
  "policy"` leaves the other root decision pending; every in-core bot command equals a FRESH bot seeded
  `stream_seed(seed, branch, k)` deciding at that decision of a linear replay (a phantom draw would shift its stream);
  reruns byte-identical; a different bot seed changes the play; p1 forfeits first; `prefix_text` + `text` == the side's
  whole protocol; the refusals by name.
- `src/utils/rust_env/cf_playout_test.py` (unit): `ModelPolicy` == `RLPlayer`'s greedy and seeded-sampling arithmetic,
  per-rollout draws independent of interleaving; the rollout mapping; the request; the side dispatch;
  `summarize_trajectory` (moved here).
- `src/utils/rust_env/successors_integration_test.py::test_the_counterfactual_keys_through_the_ffi` (`sim`): the keys
  through the real cdylib and the Python wrapper.
- `src/main/prober/replay_test.py` (unit): the orchestration — opponent resolution and regime, per-rollout seeds, stall
  sides, narrate, every refusal.
- `src/poke_env_free_entry_points_test.py::test_every_prober_command_and_the_web_app_run_with_poke_env_blocked` (`sim`):
  `replay-counterfactual` RUNS with poke-env blocked (a model opponent with `--narrate`, and a self-model one);
  `PROBER_POKE_ENV_COMMANDS` is EMPTY.

## What is left on the old road

`utils/bridge/counterfactual.py` stays: `agents/training/cf_producer.py` (the shadow critic's counterfactual labels) and
`main/search_dividend/playoff.py` still call its `replay_counterfactual`. Both are candidates for
`utils.rust_env.counterfactual.replay_counterfactual`; `cf_producer` additionally needs what the new API does not return
yet (F-P6-10). The poke-env import allowlist went **135 → 133** (`src/main/prober/session/counterfactual.py`,
`src/main/prober/replay.py`).

## FINDINGS

- **F-P6-1 — the player NAMES differ, nothing else.** The poke-env road's players logged in as `CfT{inv}x{k}` (trainee)
  and `CfO…` / `CfB…` (opponent), so its protocol — `|player|`, `|-sidestart|p1: CfT…`, `|win|CfT…` — carried those
  names; the core plays the RECORDED names. The read maps one onto the other; a `--narrate` trajectory's `→ X WINS` line
  now names the recorded player.
- **F-P6-2 — the old road was not reproducible.** A sampled opponent drew from torch's process-wide generator and a bot
  from the process-wide `random`, so two runs of one `replay-counterfactual` could disagree; the new road seeds every
  draw from `(battle_tag, inv)` (`replay.sampler_seeds`, `replay.bot_seed`). The identity in the `bot` / `ckpt_sampled`
  regimes is therefore "the same draws give the same games" (the legacy builder seeded to match), and the unseeded
  distribution check above is the only read of the road as the CLI ran it.
- **F-P6-3 — `--narrate` narrates turns 0–80 of the WHOLE battle, the recorded prefix included.** `summarize_trajectory`
  keeps the first 80 turns from turn 0, so a divergence past turn ~80 narrates no post-divergence turn at all. Inherited
  unchanged (identity first); the fix is to start the summary at the divergence turn.
- **F-P6-4 — the reseed point moved, without effect here.** The old road swapped the PRNG just before turn T's first
  choice was fed; the core reseeds at the branch point, after the opponent's recorded turn-T choice. All 672 reseeded
  rollouts were identical, consistent with "feeding one side's choice draws nothing". **UNVERIFIED** as an engine
  property beyond this read.
- **F-P6-5 — the trainee was p1 in every battle read.** The p2-trainee path (an in-core bot on p1; the stall order with
  the trainee second) has no old-vs-new identity; the cargo test covers the divergence root with either side as ours,
  but not a bot on p1.
- **F-P6-6 — the `random` bot never drew in the read** (8 of 9 roster bots did). Its Rust port equals the Python bot per
  decision by the bot gate; the playout's bot plumbing does not depend on the kind.
- **F-P6-7 — `capped` is now surfaced.** The old result scored a stall-capped line (decided by SEAT, `_battle_outcome`)
  as a loss silently; the new result still does (unchanged), and adds `play_out.capped` (how many of the rollouts it
  was) so a reader can see it.
- **F-P6-8 — `--impl node` is no longer honoured, and old-on-node vs new was not compared.** The CLI default is `node`,
  so the old road's live leg ran on the Node sim by default; the read compared against the old road on `--impl rust`.
  Node ≡ Rust only transitively, by the port's gates.
- **F-P6-9 — no speed-up.** Both roads spend their time in B=1 forwards (new / old 0.93). The play-out hands the policy
  every live branch's decisions in ONE batch, so a batched forward would cut a Monte-Carlo read several-fold — declined
  here because a batched forward can differ in the last bits from the B=1 forward the eval ran, and flip a greedy tie.
- **F-P6-10 — `cf_producer` needs more than the new API returns.** It hooks the trainee's DIVERGENCE-turn decision
  (`trainee_decision_hook`: the turn-T reward for the `mc_return` labels) and can capture the scripted prefix's obs
  (`capture_obs`). The play-out returns neither per-decision rows nor rewards; porting it needs the play-out to report
  our side's decisions (rows, rewards) per branch, or a host-side fold over the pending rows.
- **F-P6-11 — the blocked prober test is over its unmarked budget** (127.7 s at a contention factor of 1.25, advisory;
  P5 measured 216 s) and now runs two more commands. Pre-existing; not marked `slow` here.
- **F-P6-12 — a trained policy was read on 6 decisions only.** The bulk of the read is fresh perturbed checkpoints (the
  archive is behind `MIGRATION_FLOOR`); `p4_shadow_v144` (48 / 48 identical) is the one trained check.
