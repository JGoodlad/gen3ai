# GIGO audit: battle state → network and losses (2026-10-26)

An adversarial audit of the path from the battle state to the network's inputs and its losses, run on CPU at
`13c704e63`. It covered the observation encoder (Rust and Python), the hidden-information features and their Smogon
priors, the action mask, the reward and terminal handling, the auxiliary labels, and team / set decoding. Four
read-only investigators each took one area (A: encoder and hidden information, B: mask, reward and terminal, C:
categorical ids and team decoding, D: labels). Every CONFIRMED item below was reproduced by execution and checked
against the source (`deps/pokemon-showdown`) or the engine's own state; reproductions are under `repros/<letter>/`.

**Verdict.** Nothing invalidates a past result. **No cheat:** with the oracle reveal off, no hidden fact of the
opponent's team reached our row before it was revealed. Six input or label defects were confirmed and FIXED, each with a
regression test that fails on revert and a throwing guard for its class. One engine-fidelity defect and two low ones
are confirmed and NOT fixed (reported as FINDINGs). Live runs are pinned: none of this reaches them.

## Confirmed and fixed

| id | what was wrong | reach | since | fix (tag) | guard |
|---|---|---|---|---|---|
| **D-1** | The opponent-intent (α/β) label MASKED real opponent choices. (A) The window closing on OUR forced switch was masked (`phase_is_forced_switch`), although it holds the opponent's move of the turn before, usually the one that KOed us. (B) A move followed by their faint and replacement (or by our Roar, or Baton Pass) masked the whole window. | Every production run's α supervision (`opp_intent_coef` 0.05; α is stop-grad, so this CE is its only signal). Pool self-play, 100 games: **0 / 59 Explosions, 0 / 7 Self-Destructs, 1 / 18 Baton Passes labelled**; attacks masked on forced rows 16.7 % (pool) and 12.4 % (ladder), against 2 % of status moves. | `0661bbec4` (2026-08-13), ported rule for rule to Rust | `gen3_intent_label_move_then_switch_v1`: drop the forced-switch mask; a move BEFORE their switch-in (`SideTurn::move_before_switch`) is labelled MOVE; a switch with no move, or a switch before a move, keeps the old masks. Rust `IntentLabel::build` and its Python twin. | `tracker_semantics_test.rs`: `the_move_that_forces_our_switch_is_labelled` and `a_move_then_their_faint_and_replacement_is_labelled_the_move` (both fail on revert); the drag test now asserts that a drag is never a SWITCH label |
| **C-1** | The damage op read a CONSUMED item as HELD. After Knock Off, Thief, Covet or Trick, the row keeps the gone item's id with `consumed = 1`, and the op compared only the id. A knocked-off Choice Band gave P(their CB) = **1.0** (0.79 when unrevealed, 0 for any other revealed item) and kept our own ×1.5. | Any row whose active's item was knocked off or stolen and was a Choice Band: 30 of 719 pool teams carry an item-removal move, 306 a Band. The Quick Claw reads had the same spelling but were latent (Quick Claw is banned in gen3ou). | `82b7cfe4e` (2026-06-16) | `gen3_op_held_item_v1`, config **v162**: `extractor_ctx.held_item_ids` (0 once consumed) and `item_unrevealed` at every identity read and item-table lookup. No field, no weight shape; every past config stamps through. | `src/agents/model/held_item_gate_test.py`: an AST gate (no raw `item_ids` subscript or table index in any forward module, empty allowlist) and the production forward on a fixture row (a consumed item never moves the op block, whatever its id; P(CB) is 0 when gone, 1 when held, and the prior when unrevealed). Fails on revert, ours and theirs both. |
| **A-C1** | Single-turn and single-move volatiles never ended. **Destiny Bond** and **Endure** were executed in the repro; Grudge, Rage, Snatch and Helping Hand follow the same code path. The sim ends them with no protocol line, so the active-context volatile cells stayed 1 for the whole stint. | Training rows (active-context volatile slots). Pool: 6 Destiny Bond and 10 Endure sets among 4,314 mons. Obs golden: **36 of 991 decisions** carried a stale Destiny Bond cell. | Rust core M2 (`ba7375c46`, 2026-09-23); poke-env before it (UNVERIFIED) | `gen3_volatile_silent_ends_v1`: Endure, Snatch and Helping Hand end at `\|turn\|`. Destiny Bond, Grudge and Rage end at the holder's next `\|move\|` or `\|cant\|` (`onBeforeMove` / `onMoveAborted`; a re-used Destiny Bond keeps it). | `src/rust_sim/tests/silent_volatile_end_test.rs`, plus `destinybond` in the engine-truth audit `TRUTH_VOLATILES`, which `core_corpus_test` runs at every decision of the recorded corpus. |
| **A-C2** | Mean Look's `trapped` stayed set after the TRAPPER left the field (`removeLinkedVolatiles` ends it silently), and was NOT copied by Baton Pass (gen 4's `noCopy: false`, which gen 3 inherits). | Same as A-C1; 13 Mean Look sets in the pool. | same | same tag: `BoardReading::release_foe_trap` on a non-Baton-Pass switch-out, a drag or a faint; TRAPPED is `bp_copied` | same test file, plus `trapped` in `TRUTH_VOLATILES` |
| **C-2** | The paste-to-packed path dropped an explicit `Happiness: 0` (`self.happiness or ""`) and lacked Showdown's import rule (a set running Frustration with no Happiness line gets 0). The core unpacked 255, so **Frustration hit at base power 1**; the op priced it at 102. | 13 mons: pool 1, `w_forum_v1` 4, `d3_setmut_v1` 8. The live D3 run draws `d3_setmut_v1` at its own pin and is unaffected by the fix. | the vendored poke-env packer (present from the start) | `gen3_team_pack_happiness_v1` (`utils.team_packing`) | the pack oracle (below) and `src/utils/team_packing_happiness_test.py` |
| **C-3** | A `Hidden Power: <Type>` line on a mon WITHOUT the move left its IVs at 31. Showdown's validator fills the type's IVs (team-validator.ts, `set.hpType && maxedIVs`). | 45 mons (pool 2, `w_forum_v1` 36, `d3_setmut_v1` 7): a ±1 stat point against the authored team. Engine, observation and labels stayed self-consistent. | the same | `gen3_team_pack_hptype_ivs_v1` (`utils.gen3_utils.fix_gen3_hp_ivs`) | the pack oracle; `gen3_utils_test.py` |

**The class guard for C-2 and C-3 (`gen3_team_pack_oracle_v1`).** Validating the PACK cannot see these defects: every
such pack is a legal team, just not the authored one. `utils/bridge/pack_oracle.js` compares Showdown's own reading of
each paste (import plus the validator's mutations) with Showdown's unpack of OUR pack, field by field over everything
the engine reads. `Gen3Teambuilder` runs it on its pool, bias and mix packs at construction and RAISES
`PackIdentityError` on any difference. Cost, measured here: pool 4.1 s, `w_forum_v1` 7.5 s, `d3_setmut_v1` 10.6 s (memoised
per process). On the reverted packer it refuses the pool and names the 4 teams (`repros/me/pack_check.py`). Test:
`src/utils/bridge/pack_oracle_integration_test.py`.

### D-1: the record

**Which runs trained with the biased α mask: every production run since `0661bbec4` (2026-08-13).** That includes P\*,
the end-state closing-test runs (`rb_es_end2_*`), X26 / X26c and every X35 arm. This is a known BIAS in α's supervision,
not a void: the masked rows were never wrong labels, only missing ones. α learned with no Explosion or Self-Destruct as a
positive example and with KO turns under-represented. The fix reaches only runs launched at or after its commit.

After the fix, on the same pool self-play read (`repros/D/run_pool_all100_after_fix.out`, a nearby sample; the battles are
not byte-identical to the before run): Explosion **54 / 58**, Self-Destruct 8 / 8 and Baton Pass 21 / 24 are labelled. Moves
labelled rose from 3,178 to 3,701 and moves masked fell from 668 to 372; masks on forced rows fell from 409 to 30. The rest
are by design: a denied or `cant` turn, an Encore override, a switch before a move. Two measurement-script categories
now read non-zero and are ARTIFACTS of its window bookkeeping, traced to the same cause: the move a side submitted before
an Arena Trap / Magnet Pull re-ask executes in the next window. The label there is the move that actually ran
(`repros/D` trace of seed 1, env 0, episode 6, decision 29: Dugtrio's Rock Slide, then its KO and the Gyarados
replacement).

## Versioning and the observation

- **Config v162**, no field and no ARCH_SIGNATURE or floor change; every past config stamps through. The obs golden is
  re-written: 36 of 991 decisions moved, every one `active_context` volatile slot 6 (`destinybond`) going 1 → 0
  (`repros/me/diff_golden.py`). `obs_semantics_test` re-pins it with the marker KEPT at 151.
- 🚨 **FINDING (orchestrator ruling, 2026-10-26): `OBS_SEMANTICS_VERSION` is NOT raised.** The written rule says to
  raise it for C-1, a model-internal input the golden cannot see, and arguably for A-C1. But `ProbeModel.load`
  hard-refuses an `obs_semantics` verdict. Raising it would blind every current model (P\*, X26 / X26c, every X35 arm
  including the live D3 run) in five places:
  - the prober's model views and web app;
  - `main.harvest` and `main.harvest_meter` (the hidden-info maturity battery reads X26 checkpoints);
  - `main.probe_replay`;
  - `winprob_finetune`'s scoring path.

  So a checkpoint recorded below v162 reads as `current` in the prober even though its op (consumed items) and its
  Destiny Bond / Endure / trapped cells changed underneath it. This is a known gap. The marker's gap policy (refuse for a
  TRACE REPLAY of a model's own old traces; warn plus a badge for fresh probing at HEAD) is the TECH_DEBT row
  "obs_semantics marker policy"; raise the marker once it lands.
- **Read paths at the fixed tree** (they never read the marker): `check_compatible` and resumes, `main.h2h`,
  `main.anchors` (and so the Kakuna runner, which spawns it), the x35 kit, and every `load_checkpoint_strict` reader.
  Executed on CPU:
  - h2h P\* (`rb_es_end2_s2005`) vs X26c (`rb_x26c_s2601@478000208`): 2 games played;
  - h2h P\* vs X35 (`rb_x35_sprt_s3503`): 2 games played;
  - `load_checkpoint_strict` of P\*, X26c, `rb_x35_c0_s3501` and `rb_x35_sprt_s3503` (`repros/me/h2h_*.log`,
    `load_check.py`).
- **Encoder benchmark** (40 decisions × 300 reps, same box, contended), before → after:
  - `encode_us` median 6.0 → 6.1 µs;
  - `present_encode_us` median 9.8 → 10.0 µs: noise.

  The change is in the reading, not the encoder (`repros/me/enc_*.txt`).

## Confirmed, NOT fixed (FINDINGs)

- **B-C1 (MEDIUM for engine fidelity, LOW for training).** The Rust ENGINE keeps an Outrage / Thrash / Petal Dance lock
  that Showdown ends quietly after a cancelled repeat (full paralysis, a confusion self-hit, an immune target). The mask
  then forbids legal switches and moves, and the engine confuses the user where Showdown does not.
  - Replays diverge from Showdown in 11 of 100 targeted lock / trap games and 28 of 30 lock-in / Ghost probes; 80 of 80
    pool games are identical.
  - Cause: `turn/residuals.rs:1036-1056` and `turn/moves.rs:1791-1796` keep one countdown and never model Showdown's
    separate 2-turn lock duration (`data/conditions.ts:253-286`).
  - Zero reach today: no pool, ladder, `w_forum_v1` or `d3_setmut_v1` team carries these moves. Search and any future team
    are exposed.
  - Since `30a8a47d9`. NOT fixed here: it is an engine-parity change that needs its own lock-in mode in
    `gen_sim_bridge_diff.js` (`repros/B/`). Tracked as a TECH_DEBT (c) row.
- **B-C2 (LOW).** At a p1 decision at turn ≥ 250 the core forfeits instead of acting, but the trainer's collector still
  writes that decision's row (`rust_rollout/collector.py:213`), so PPO credits a sampled action that never ran with
  0 − V(s). One row per stall game. The core side was executed; the trainer side was confirmed by reading only.
- **B-C3 (LOW).** `turns_since_progress` increments on a same-turn re-request after a refused (Arena Trap) switch:
  `trackers/clock.rs` updates once per decision. It touches about 0.6 % of rows, and production does not read the cell
  (`--drop-progress-clock on`).
- **C-F1.** Nothing pins "every dex num < its embedding table". Headroom: species 386 / 400, moves 370 / 400, items
  499 / 600, abilities 76 / 100, types 18 / 19. `constants.py`'s unused `MAX_*_ID` disagree with the layout. The
  effective type vocabulary is 19 (`max_types` says 20; a type id of 19 raises).
- **C-F2.** `belief_bank_static._masked_ce_terms` CLAMPS a masked label ≥ its class count onto the last class, where the
  eager path raises. Unreachable today (labels in range), but it is a silent path.
- **D-F2.** The core accepts a team that breaks Species Clause. The label path then faults only by accident, and two
  same-species mons with identical spreads would be mislabelled silently. The validator stops this upstream; a duplicate
  dex num in `from_packed` should refuse.
- **A-F.** Baton Pass also copies `lockon` and `taunt` in gen 3 and the reading copies neither (PLAUSIBLE, not executed;
  no pool set uses Lock-On / Mind Reader). The last-action cell reads `was_switch = 1` for a mon dragged in or Baton-Passed
  in (a design question). `get_layout()["parts"]["reactive"]["end"]` is the row length, not `start + 17` (no reader).
- **Engine / harness stale docs (B-F1, F2).** `episode.rs:9,22`, `core/columns.rs:179` and `utils/rust_env/columns.py:107`
  cite the deleted `wrappers.resolve_episode_end` / `--critic winprob`. `fixture_battles.py` says a stall-forfeit decision
  makes "no row, as in training", which B-C2 shows false for the trainer.

- **D-1 also changes the HUMAN labels (X50).** `main.x50_r1` extracts human opponent-intent labels through the same
  `IntentLabel`, so a re-extraction at this commit labels a move made before a switch-in, and the move that forced the
  viewer's switch, where the old extraction masked them. Its golden test (`x50_r1/extract_test.py`) moved on exactly
  those two rows. The frozen human opponent model (rung 1b, sha256 `76e29a01`) was trained on the OLD masks; its
  registration is unaffected, but an extraction at HEAD is not its training data.

## Plausible (not reproduced)

- A lock-in repeat that misses on accuracy probably shares B-C1's cause (the immune case is confirmed).
- D-1's rule (B) also masks a voluntary switch whose incoming mon is KOed and replaced in the same window; a switch before
  a move stays masked by design.
- The final window of a game (often the game-ending KO) has no next decision, so it is never labelled: a second, smaller
  selection against KO moves in α's supervision. Unmeasured.

## Cleared

- **Hidden-information leak:** 9 hidden-fact swaps × 3 seeds, deterministic policies on both sides. Lead swaps (an
  unrevealed move, item, EVs, nature, IVs, ability) are byte-identical over the whole game; bench swaps first differ
  exactly at the reveal (`repros/A/run_variants.py`, `variants2.out`). Each side's row is built from its own protocol
  stream; HP privacy matches `getHealth`.
- **Row invariants** over 30 battles / 1,107 rows:
  - every opponent HP cell is k / 100, and the opponent spread block is zero;
  - at most one active flag per team, on the latest switch-in in 1,174 of 1,174 rows;
  - trapped and last-action cells appear only on the active slot.
- **Categorical ids:** 0 non-integer and 0 out-of-range values in all 25 id columns, over pool rows and crafted edge teams
  (Deoxys, Castform, Unown, Ditto, Curse `???`, typed Hidden Power). An out-of-range id RAISES in the production forward.
- **Mask vs request:** an independent mask rebuilt from each request JSON equals the core's at all 9,731 p1 decisions.
  That covers forced replacement, locked, recharge, disabled, 0 PP, Struggle-only, trapped and maybe-trapped requests.
  Switch action i names obs slot i in 718 of 718 rows. The only refused choice was the hidden-trap switch, which matches
  Showdown.
- **Reward and terminal:**
  - win 1, loss / tie / stall forfeit 0 from p1's side, only on the last row;
  - every episode end is terminal in GAE (no bootstrap through a truncation);
  - a quarantined battle's rows are discarded;
  - a new episode's first row is never paired with the previous reward.
- **Labels vs truth:** 0 mismatches on the belief species, moves, known moves, item, Hidden Power type, spread, nature and
  EV labels over 40,155 pool, 57,221 ladder, 14,732 procedural and 25,672 `d3_setmut_v1` revealed-slot decisions; the slot
  order is the reveal order. **Leak safety:** the learner forward is bit-identical with all 21 label keys real, poisoned or
  absent.
- **Priors:** the ability and Hidden Power priors are generated from Smogon chaos data
  (`tools/smogon_stats_downloader/compute_priors.py`), not the pool. The committed files were not re-derived.

## Follow-up: the open items worked (2026-10-26, after `d8d674b99`)

The FINDINGs above that were left open (B-C1, B-C2, C-F1, C-F2, D-F2 and the Baton Pass note of A-F) were worked in one
unit on CPU, each by execution. Reproductions and outputs: `repros/followup/`. Live runs (D3 at pin `9e289ba4`) are pinned
and unaffected.

| id | outcome | tag, tests |
|---|---|---|
| **B-C1** lock-in | **FIXED** (below), plus a second engine defect the re-run exposed (a never-expiring `mustrecharge`) | `gen3_lockin_cancelled_repeat_v1`, `gen3_recharge_expiry_v1`; `regression_test.rs` LI3 / RC1, `utils/rust_env/lockin_parity_integration_test.py`, `gen_sim_bridge_diff.js --mode lockin` |
| **B-C2** stall-forfeit row | **CONFIRMED by execution, FIXED** (below) | `gen3_stall_forfeit_row_v1`; `rust_rollout/store_test.py`, `stall_forfeit_row_integration_test.py` |
| **C-F1** dex num vs table | **FIXED**: one declaration, three doors that throw | `gen3_vocab_bounds_v1`; `vocab_bounds_test.py`, `encoder::data::num_tests` |
| **C-F2** static twin clamps | **FIXED**: no clamp, plus the host check that nothing called, now called | `gen3_label_vocab_raise_v1`; `belief_bank_static_test.py`, `learner_gates_test.py` |
| **D-F2** Species Clause | **FIXED** in clause formats (`gen3ou`), customgame unchanged | `gen3_species_clause_v1`; `regression_test.rs` SC1 |
| **A-F** Baton Pass copies `lockon` / `taunt` | **CLEARED as unobservable** (below); the Lock-On silent end is a new TECH_DEBT row | `repros/followup/probe_bp_taunt_lockon.js` |
| the final window never labelled | **REAL by construction, NOT fixed**: a TECH_DEBT row (needs a core column) | |

**B-C1, what Showdown does.** `lockedmove` (`data/conditions.ts`, gen 3 inherits it through gen 4: `data/mods/gen4/conditions.ts`
only deletes `onAfterMove`) has TWO counters. `duration: 2` is ticked by the residual's generic `duration--` (0 removes the
volatile and runs `onEnd`, which confuses the user iff `trueDuration <= 1`; `onResidual` does not run on that tick), and put back
to 2 only by `onRestart` (a LANDED repeat's self `volatileStatus`) or by gen 4's Protect `onTryHit`
(`data/mods/gen4/moves.ts`), both while `trueDuration >= 2`. `trueDuration` is the `random(2,4)` rolled once on the cast and
ticked by `onResidual`, whose SLEEP branch deletes the volatile. So a repeat that never lands (an immune target, full paralysis,
a confusion self-hit, a flinch, a miss) leaves `duration` unrefreshed: the lock ends at that residual, QUIETLY on a 3-turn
roll's second turn (`trueDuration` is still 2), with the fatigue confusion on the roll's last turn; and a user Spored after its
last repeat is confused although asleep (the duration tick precedes the sleep branch). The probe, `harness/probe_lockin_cancelled_repeat.js`,
prints all four cases from the real sim. The port kept ONE countdown; `state::LockedMove` now holds both, `moves.rs` restarts it on
a landed repeat and a Protect block, and the residual ticks in the sim's order.

**B-C1, what the re-run found.** The audit's "11 of 100" was not all one defect. Replayed PACED (one command at a time) on
the old build: **10 of the 11 are real, 1 is a replay-tool artifact** (a bulk feed of a script carrying refused hidden-trap switches
trips Node's reject-streak guard; `repros/followup/showdown_replay_paced.py`). The real ones include a second engine defect
unrelated to the lock: a Hyper Beam user whose recharge action a faint cancelled (gen 3: "fainting in singles skips to residuals",
`Battle.faintMessages`; here the foe's switch-in died to Spikes) kept `mustrecharge` forever, while the sim's `duration: 2`
expires at the second residual and frees the next turn. `MonState::must_recharge_duration` now ticks it
(`probe_recharge_expiry.js`). After both fixes, regenerated with the fixed engine and replayed against Showdown: seeds 1 and 2
(40 and 60 games, the audit's own shape) **100 of 100 identical** (the 2 bulk-differing games re-check clean paced); the fresh
seeds and the new standing mode:
- fresh seeds 3 and 4 (`run_targeted_set.sh`, 50 games each): **50 of 50 and 50 of 50 identical** (1 bulk-differing game each, both
  clean when re-fed paced);
- `gen_sim_bridge_diff.js --mode lockin` (`--persistent`, gen3customgame, 60 battles each): seeds 101 and 202 **60 of 60 and 60 of 60
  identical** (649 and 654 `trapped:true` frames, so the locks are reached); seeds 606 and 707, after an Uproar locker joined the
  mode, **60 of 60 and 60 of 60** (661 and 645 frames). The first Uproar run (seed 303, killed after its first error) found an
  Uproar finding of its own, not a lock-in one: a Rest under a live Uproar succeeds in the port and fails in Showdown
  (`uproar.onAnySetStatus`), a TECH_DEBT row with no training reach; the mode carries no Rest;
- `--mode ladder` and `--mode pool` at `--format gen3ou`, 60 battles each (the Species Clause on real teams): **59 ok, 0 diverged,
  1 harness safety-cap stall** (a battle that never ended, no divergence) and **60 of 60**.
The new `--mode lockin` FAILS 26 of 40 battles on the pre-fix binary (seed 101: 14 ok, 2 diverged, 24 errored, the old engine's
locked request against Showdown's free one; `POKESIM_SIMBRIDGE_PREBUILT=1` with main's old `sim_bridge`), against 0 of 60 on the fixed
one, and 3 of the 5 `lockin_parity_integration_test.py` fixtures FAIL on the old engine (the 3-turn cancel, the asleep
confusion, the recharge; the other two, a 2-turn cancel and a Protect refresh, hold on both and pin the new code).
The Rust pins are revert-verified the same way: the new `regression_test.rs` tests run against the OLD `src/` (a scratch crate
with `git show HEAD:` sources): the 3-turn cancel, the asleep confusion, the recharge expiry and both Species Clause pins FAIL; the
2-turn cancel, the Protect refresh and the recharge control hold on both engines, so each of the first two is proven by a MUTATION
of the new code instead (no Protect refresh fails the Protect pin; a fatigue confusion that never fires fails the 2-turn pin, the
Protect pin and the asleep pin).
The obs golden still reproduces all 991 hashes (`python -m agents.training.golden_obs_core --check`), so no obs-semantics change.
Reach: zero training reach for the lock (no pool, ladder, `w_forum_v1`, `d3_setmut_v1` or `n1_proc_v1` team carries Outrage /
Thrash / Petal Dance); the recharge expiry is reachable in training (2 team files of `data/teams` carry Hyper Beam, and a Spikes switch-in KO is a
normal OU event), at a rate the audit's pool games did not show (80 of 80 identical): unmeasured, and every run launched before
this commit trained on the old engine for it.

**B-C2, by execution.** `repros/followup/smoke_stall.py` is the `--debug` smoke with the production turn limit patched to 6
(`GEN3AI_MODELS_DIR` in a scratch dir, CPU, short lane), printing for every ended game its collector rows against the p1
`CHOOSE` lines the core actually fed. Before: **297 of 297 ended games** (all stall forfeits at that limit) carried
`rows_written = p1_CHOOSE_fed + 1`, i.e. a trained row whose action never ran (`stall_smoke_before.out`). After:
`rows_kept == p1_CHOOSE_fed` in **347 of 347** (`stall_smoke_after.out`). `GameLog.end` releases the forfeited game's last row and
the terminal lands on the last PLAYED decision; `staleness/forfeit_rows_dropped_total` counts them. Which runs: every run on the
Rust collector trained on one such row per stall game. The stall rate is the run's, and no instrument reads it now (the G7 chain
was retired 2026-10-25), so the size is UNMEASURED; at the production limit of 250 turns it is a small fraction of games.
`num_timesteps` still counts the presented decision.

**C-F1 / C-F2 / D-F2.** `constants.VOCAB_SPECIES` ... `VOCAB_TYPES` (400 / 400 / 600 / 100 / 20, the values the layout always
had; the unused `MAX_*_ID` that disagreed are deleted) are read by `get_layout()` and mirrored in `layout.rs`
(`rust_core_obs_layout_test` holds them equal). A dex num at or past its table now raises at `Gen3ObservationEncoder`
construction (`vocab_bounds.check_vocab_bounds`) and at the Rust tables' load (`encoder/data.rs`: every id the Rust encoder
writes is read out of those tables); the headroom is unchanged (species 386 of 400, moves 370, items 499, abilities 76, types
18 of 19 in use). `belief_bank_static.check_label_vocab`, the host check the compiled micro-step's docstring leaned on, had NO
caller: it is now called once per update from `train()` (`learner_gates.check_label_vocab_buffer`, `LabelVocabError`), covers
the item and hp-type labels at their scored slots, and the twin's CE no longer clamps its targets (a scored PAD `-1` used to
train class 0 in BOTH the legacy and the static path; both raise now) and a scored out-of-range slot's CE is poisoned with NaN, because
inductor on CPU raises for `>= n_classes` but NOT for a scored `-1` (measured), so K9(c) still fails closed in the compiled region. `BattleState::start` refuses a clause-format team with
two of one species num (Showdown's `speciesclause`, keyed on `species.num`, in gen3ou's `Standard`); 17,670 teams (the three
corpora and `data/teams/**`) hold no breach, and three test fixtures that carried a duplicate in `gen3ou` were made legal with
their draws unchanged.

**A-F, verified in Showdown (`probe_bp_taunt_lockon.js`).** Baton Pass DOES copy `lockon` and `taunt` in gen 3 (`noCopy` is
false for both in the resolved gen-3 dex), but neither copy is observable. A taunted mon cannot use Baton Pass (`|cant|…|move:
Taunt`), so a Taunt never reaches the copy. The copied `lockon` is `duration: 2` and expires at the same turn's residual, before
the entrant can act. The reading's `bp_copied` rows (`false`) therefore match what a reader can see; flipping them would leave a
stale cell on the entrant. What IS open: the reading never ends `lockon` itself (UNVERIFIED by a played game; the engine models
no Lock-On), a TECH_DEBT row. The final window of a game is never labelled (the label of row t is read from the window closing at
t+1): about one window in 41 on pool self-play, exactly the game-ending one; a TECH_DEBT row.

**A red the audit itself left (found by the slow tests of the touched Rust).** `src/rust_env/tests/oracle_reveal_test.rs`'s
three pinned digests (the OFF, `species` and `full` bytes over 8 envs x 600 ops x seeds 11-13) were STALE on main since
`d8d674b99`: its unit did not run the slow tier (`core_cargo_test.py` was last banked green on 2026-10-21). A census, every hashed
column dumped per op at the parent `e9a7a53ea` (the pins hold there) and at `d8d674b99`: 297 of 28,848 side-rows moved, ONLY in the
active-context cells `endure`, `snatch` and `trapped` (A-C1 / A-C2) and the four intent-label columns (D-1); every other column
is byte-identical. The follow-up tree against `d8d674b99` has NO moved column, so none of this unit's engine changes reaches that
corpus. The three pins are re-recorded with the census in their doc comments.

## Hazards of the audit itself

- The CPU was contended (a live training run, 8-12 queued heavy jobs). Investigator A's 250-battle invariant sweep was not
  run; its evidence is 30 battles.
- The before / after label counts come from random-vs-random self-play; rates under a trained policy differ (the
  0-of-N Explosion count does not depend on the policy).
