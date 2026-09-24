# The ladder-usage corpus: the full Metamon smoke and the MILESTONE gates (2026-09-24)

**Verdict.** At the registered n, **all 11,431 battles (every one of the 22,862 Metamon `hl_05_26`
gen3ou public-ladder teams, played once) finished with 0 failures**: 2,030,753 decisions, each
fully encoded by both players on the node bridge. The same seven Metronome-calling battles crash
on the pre-fix tree (7 of 7, `Unhandled move message format`) and pass on this one. **Tag:
MEASUREMENT · full smoke 11,431 / 11,431 ok · 0 crash classes.**

## 1. What ran

* **Corpus.** Metamon `hl_05_26/gen3ou`, `teams` revision v5, downloaded 2026-09-24, archived at
  `~/gen3ai_archive/metamon_cache_2026-09-24/`. All 22,862 teams are Showdown-legal. The committed
  test corpus (`src/utils/ladder_corpus/`) keeps the 22,813 the Rust ENGINE can play. The smoke
  plays all 22,862, because it runs on the node bridge.
* **Registration** (`registration.json`, written before the first unit): 22,862 team files sorted
  by name, one seeded permutation (seed 20260924), battle `k` plays teams `perm[2k]` vs
  `perm[2k+1]`, n = **11,431**. Sim seed `[11+k, 22+k, 33+k, 44+k]`, players seeded-random
  (1000+k / 2000+k), each running the full training `embed_battle` at every decision. A battle is
  `ok` or carries the exception that ended it.
* **Code.** A frozen snapshot of the change (`efd3ee78` + the working diff, snapshot sha
  `2fdf476d19ec4e57`), so no edit made during the run could reach a later unit.
* **Incremental** (`ORCHESTRATOR_SOP.md` §2): 229 units of 50 battles, each a separate process
  whose rows are written atomically to `~/gen3ai_archive/ladder_usage_smoke_2026-09-24/full/units/`;
  the driver ran detached, 4 workers, `nice 10`, 12:17–13:41. The rows are `rows.jsonl.gz` here.
* **Resume, proven on a small slice.** 6 units of 10 battles: the driver and both running unit
  processes were killed with SIGKILL after 2 units. The restarted driver launched only the 4
  missing units and left the 2 finished files byte-unchanged (md5). The resumed rows equal an
  uninterrupted run of the same slice exactly (60 / 60, `wall_s` excluded).

## 2. Totals (`report.json`)

| | |
|---|---|
| battles | 11,431 / 11,431 registered |
| ok | **11,431** |
| failed | **0** (no parse crash, no encoder crash, no hang) |
| finished | 11,431 (median 66 turns; one reached the 1,000-turn tie) |
| decisions encoded | 2,030,753 |
| battles with a Heal Bell `-activate` | 309 |
| battles with a called move | 373 (Sleep Talk 1,568 lines, Metronome 26 lines) |

**Teeth.** The seven battles in which Metronome called a move were replayed on the pre-fix tree
(`efd3ee78`): 7 of 7 crash with `ValueError: Unhandled move message format`
(`prefix_fix_metronome_battles.jsonl`). On this tree they pass.

## 3. The MILESTONE gates on the corpus

All on the `milestone` tier (800 teams), this tree, emission self-check build, 2026-09-24.

| gate | battles | result |
|---|---|---|
| slice E/V MILESTONE, ladder keys 0–149 | 150 | **0 divergences**: 174,098 events, 25,975 decisions, 14.97 M field comparisons |
| slice E/V MILESTONE, ladder keys 150–299 minus the named three | 147 | **0 divergences**: 174,844 events, 25,620 decisions |
| the NAMED known divergences (must still fire) | 3 | key 173: 3 in the Mimic-overlay classes (the view road); keys 237 / 260: 47 / 17 `[SIM-FACT] ours.moves` (the R3 residue). Slice E is clean on all three |
| the rest of the slice E/V MILESTONE tier (pool random 2 × 360, policy 2 × 50, protocol + byte-fuzz) | — | pass |
| `ab_fuzz.js --mode ladder` (state + seed), master seed 20260924 | 400 | 399 ok, **1 `kind=seed`** (§3a), 0 panic |
| `ab_fuzz.js --mode ladder --protocol --format gen3ou`, seed 20260925 | 400 | 398 ok, 2 allowlisted (A1 `turn0-construction-speed-tie-mirror-of-flip`), **0 diverged** |
| `bridge_ab_fuzz.js --mode ladder --format gen3ou`, seed 20260926 | 200 | 199 ok, 1 allowlisted (`perside-construction-speed-tie-mirror-of-flip`), **0 diverged** |
| `gen_sim_bridge_diff.js --mode ladder --format gen3ou --persistent`, seed 20260927 | 200 | **200 ok**, 0 diverged, 0 errored, `drain_timeouts` 0 |
| Python fuzz scripts, `--team-source ladder` (run before the Rollout fix) | 40 + 60 | `obs_roundtrip`: 3,376 decisions bit-identical; `event_log`: 5,297 decisions, 1 mismatch of the shape of the known stale-oracle class (backlog §2 (b) P1 (a)), not re-attributed |
| the same, on the fixed engine | 3 + 10 | `one_sided_view_parity`: 51 branch points (11 D10), 122 comparisons, 0 divergence; `live_view_memo`: 738 decisions, 1,486 view + 738 obs identity checks, PASS |
| pool byte-identity, this tree vs main `4f32a9ce` | 400 | obs + mask **byte-identical** at all 66,992 decisions (the same held against `efd3ee78` before the rebase) |

**Before these runs**, the corpus's first contact (slice V on 400 ladder battles) found two engine
bugs with zero pool exposure: lock-in continuations lacked `[from] lockedmove`, and Rollout kept
its lock across a turn that dealt no damage. Both are fixed (`gen3_lockedmove_announce_v1`,
`gen3_rollout_lock_duration_v1`) and pinned; the table above ran on the fixed engine.

### 3a. The one `ab_fuzz` state-mode seed divergence — NOT attributed (repro kept here)

`repro_ab_state_rmufzzde5_ab_12_16/` (a Dusclops Imprison team vs a Gengar that knows Will-O-Wisp).
It reproduces on the pre-change binary too, so none of this change's commits caused it. The sim's
decision 38 is a REJECTED choice: Gengar picks the imprisoned Will-O-Wisp, the sim answers
`[Unavailable choice]` and records a checkpoint with no draws. The port rejects the same choice and
records no checkpoint. `ab_replay`'s subsequence anchor assumes the port's checkpoint list is a
SUPERSET of the sim's, so it fails at exactly that duplicate (`kind=seed` at decision 38). After
that the two lists re-converge on the same seeds at several later checkpoints (48503…, 24954…,
63693…, 45914…), with different boundary placement in between. That is consistent with the
documented checkpoint-segmentation artifact, and the live-boundary gate (`gen_sim_bridge_diff`, which
cannot have that artifact) is 200/200 clean on the same tier. It is **not proven** to be an
artifact: a draw bug hidden in the non-aligned stretch is not excluded. Backlog §2 (c) P2.

## 4. Hazards

* The smoke's players are seeded-random, not the policy, so it measures the reading and the
  encoder over real teams. It says nothing about strength.
* Assist, Nature Power and Mirror Move never fired in the smoke (the corpus carries few of them).
  Their shapes are covered by the constructed pins and the node-bridge battle instead
  (`agents/battle/called_move_bridge_integration_test.py`).
