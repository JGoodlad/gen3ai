# Implementation: Step 6 — Next-Run Bundle

This is the consolidated record of **everything that lands in the next training
run** since the currently-running v3 model was launched. It is a bundle, not a
single feature: six independent changes — three to the observation, one to the
extractor's internal structure, one to the policy/value split, and one to the
reward — were developed in sequence and are documented together because they ship
as one batch. Each section is self-contained; the combined dimension, version, and
test tables at the end describe the tree as it stands at the end of the batch.

The headline moves, in dependency order:

| # | Change | Surface | Version effect |
|---|---|---|---|
| A | Move **accuracy + never-miss** bit | observation (move slot 9→11) | obs dim ↑ |
| B | Move **outcome** (crit / miss / fail / cant) | observation (TurnDelta) | `ARCH_SIGNATURE → gen3_move_outcome_v1` |
| C | **Modular extractor** refactor (phase `nn.Module`s) | architecture (internal) | `ARCH_SIGNATURE → gen3_modular_v1` |
| D | **Dual-head** policy/value (value-dedicated CLS) | architecture (readout) | `ARCH_SIGNATURE → gen3_dual_value_v1` |
| E | **Protocol-accurate move attribution** | observation (TurnDelta) | obs dim ↑ (no arch bump) |
| F | **Reward overhaul** (anti-spam taxes, pivot pressure) | reward only | none |

Section B is summarised here and recorded in full in
`impl_step5_move_outcome.md`; the other five are documented in full below.

---

## A — Move accuracy + never-miss bit (observation)

### What changed

The 11-dim move slot (`moves.py`) gains two trailing fields:

- **accuracy** (offset 9) — raw percentage / 100, so a 70%-accuracy move is `0.70`.
- **never_miss** (offset 10) — a categorical bit.

Never-miss moves carry `accuracy = 100` in the mapping, so they encode as
`[1.0, 1]`, while a genuine 100%-accuracy move encodes as `[1.0, 0]`: same scalar,
distinguished only by the bit. This is a *kind* difference, not a 1% one — a
100%-accuracy move can still miss into evasion (Double Team) or after Sand-Attack,
whereas a never-miss move (Swift, Aerial Ace, all status/self moves) bypasses the
accuracy/evasion check entirely — so it earns its own bit rather than the "101"
trick. `features_extractor.py`'s shared move processor takes the two new scalars as
pass-through inputs.

### Data pipeline

`gen3_moves.json` was regenerated to carry per-move `accuracy` / `never_miss`. The
old `tools/gen_abilities_by_gen.py` was relocated and generalised into
`tools/pokemon_data_extractor/sync.py`, which now extracts **both abilities and
moves (with accuracy)** from the live static data, reads abilities from the current
`src/poke_env` pokedex (not the stale `deps/venv` path), and fixes a Gen 3
ability-ceiling off-by-one (77→76) that had leaked Tangled Feet onto the Pidgey
line.

---

## B — Move outcome: crit / miss / fail / cant (observation)

*Full record: `impl_step5_move_outcome.md`. Summary for this bundle:*

Each turn-history slot now reports the **fate** of each side's move: a 3-way
outcome one-hot `[hit, miss, fail]` plus a crit bit per side, and the `|cant|`
reason one-hot widened from 5 to 11 (`recharge, taunt, disable, imprison, truant,
nopp` added, with `move:`/`ability:` prefix normalisation). Five protocol message
families (`-crit`, `-miss`, `-fail`, `-notarget`, `-nothing`) were un-ignored and
attributed to the **currently-resolving mover** rather than the Pokémon the line
names. This is what lets a value head separate luck (a KO-by-crit, a missed Hydro
Pump) from policy. `ARCH_SIGNATURE → gen3_move_outcome_v1`; the base-block layout
moved to computed `OFFSET_*` constants so later widening can't silently corrupt
decoders.

---

## C — Modular extractor refactor (architecture, internal)

### What changed

The ~390-line `Gen3FeaturesExtractor.forward_internal` was split into named
`nn.Module` phases chained by a thin orchestrator, with an `ExtractorContext`
dataclass as the inter-phase contract:

```
Embeddings → ObsUnpack → PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler
```

Each phase owns its layers, so `state_dict` keys are phase-prefixed.
`ARCH_SIGNATURE → gen3_modular_v1` so old checkpoints fail with a clean
arch-family error.

### Why it's safe

A **pure behaviour-preserving** refactor: verified **bit-identical** (`max|Δ| = 0`)
to the prior code by transplanting golden weights through a 1:1 suffix remap. It
adds precise per-phase unit tests (`phase_modules_test.py` tests `CLSPool` /
`ProjectionAssembler` in isolation on a synthetic context) and updates existing
tests to the nested paths. `visualize_arch` now groups the phases as Netron blocks
(2377 → 26 top-level nodes). This refactor is the structural groundwork that makes
the dual-head split (D) a localised change rather than surgery on a monolith.

---

## D — Dual-head policy/value: value-dedicated CLS readout (architecture)

### Motivation

Under SB3 the feature extractor is shared between actor and critic. Both losses
backprop into the same 23-token representation, but they want different things:
**policy** wants action-discriminative features anchored to our active mon;
**value** wants a *global* "who's winning" estimate with no single anchor token.
The symptom on the v4 run was `value_loss ≈ 90` with `explained_variance ≈ 0.82`
(headroom). A dedicated readout is cheap, principled groundwork — it also tightens
GAE advantages and is the direct prerequisite for the v5 MCTS value function.

> **Honest caveat (from the design):** with `gamma = 0.9999`, `HP_VALUE = 2`,
> `VICTORY_VALUE = 30`, returns span tens-to-hundreds, so the value RMSE may partly
> reflect **reward scale**, not underfit. The dedicated readout's win is largest if
> it's genuine underfit — cheap enough to just measure.

### What changed

This implements **Step 1** of `design_dedicated_value_head.md`: split *late*
(decouple the readout, not the body), keep one body.

- **`value_cls` pool** — a third learned CLS query alongside `our_cls`/`their_cls`,
  added in `CLSPool`. It cross-attends over **both teams' 12 post-transformer
  tokens** (combined fainted key-mask) → a 128-dim `value_pooled` global summary.
  `CLSPool` now returns the 4-tuple `(our_team_pooled, their_team_pooled,
  our_active_refined, value_pooled)`.
- **Dual projection heads** — `ProjectionAssembler` emits a `(pi_combined,
  vf_combined)` pair; the root extractor grows a second `value_pre_norm` /
  `value_projection` head and `forward` returns `(pi_features, vf_features)`:

  ```
  pi_combined = our_pool 128 + their_pool 128 + our_active_refined 128
                + our_ctx_enc 32 + opp_ctx_enc 32 + non_matchup_rest 25   = 473 → 512
  vf_combined = value_pooled 128 + our_ctx_enc 32 + opp_ctx_enc 32
                + non_matchup_rest 25                                      = 217 → 512
  ```

  The transformer **body is shared and runs once**; only the readout, projection,
  and critic MLP branch are independent. `features_dim` stays `PROJECTION_DIM = 512`.
- **Custom policy** (`policy.py`, new) — `Gen3DualHeadMaskablePolicy`, a
  `MaskableMultiInputActorCriticPolicy` subclass that keeps
  `share_features_extractor=True` (so SB3 builds exactly one body) and overrides the
  four feature-consuming methods (`forward` / `evaluate_actions` /
  `get_distribution` / `predict_values`) to unpack the tuple and route each half to
  `mlp_extractor.forward_actor` / `forward_critic`. `train_rl_agent.py` passes this
  class in place of `"MultiInputPolicy"`. The extractor **must** be paired with it.

`ARCH_SIGNATURE → gen3_dual_value_v1`. The observation vector is **unchanged by
this step**. Step 2 (per-head PMA readout transformer) is intentionally **not**
built — it is gated on whether Step 1 closes the value gap.

### Divergences from the design (all deliberate)

1. **Tuple, not concat.** The design recommended a single `[B, 1024]` tensor split
   inside a custom MLP-extractor; we returned the `(pi, vf)` tuple but neutralised
   its downside by keeping `share_features_extractor=True` (one body) and
   overriding the four consumers. Keeps `features_dim = 512`, no custom slicing.
2. **Value query pools the 12 team tokens, not all 23.** History and global info
   have already flowed into the team tokens, so they are a whole-board read, and a
   `cat` of the two fainted masks keeps masking symmetric. Widening back to 23 is a
   cheap first thing to try if value underfits.
3. **Value input includes the encoded active contexts** (`our/opp_ctx_enc`) — they
   are value-relevant and already computed for the policy head.

---

## E — Protocol-accurate move attribution (observation)

### The gap=0 desync

Sections B/D anchor every outcome to `our_move_id` / `opp_move_id`. Those were
inferred from the **action index** the agent pressed — wrong whenever the protocol
and action desync, which Gen 3 does around faints and forced switches: poke-env's
two-player async env emits **multiple history records per game turn** ("gap = 0"),
so the action counter runs ahead of the turn the `TurnDelta` is built for, the
post-faint `active_pokemon.last_move` reads the *replacement*, and the turn-gated
`DamagingMoveEvent` can name a *different* move than the one that fired. The
visible symptom: a turn where the model is told it used move X when the protocol
fired move Y, an Explosion reading `miss`, or a switch-into-Spikes death reported
as a missed attack.

### What changed

`TurnDelta.build` (`battle_context.py`) re-sources both sides' move id and outcome
from **protocol truth**, symmetric across our side and the opponent:

- **Stayed in and moved** → the protocol `last_move` (new `our_last_move_id` field,
  delegation-aware: Sleep Talk → the *called* move).
- **Moved, then fainted** → the promoted `DamagingMoveEvent.move_id` (the active
  slot has already shifted).
- **Opponent** → recovered from `opp_all_last_move_ids[opp_prev_active]` when the
  opp moved before dying; `None` on a voluntary opp switch.
- **KO'd before acting** → a shared `_ko_before_acting(...)` helper sets
  `move_id = None`, outcome `None`, and a new `cant_reason = "fainted"` — distinct
  from a voluntary switch and from a genuine `|cant|`. This is the explicit
  "nothing happened" signal the accuracy mandate requires.
- **Outcome** via `_derive_move_outcome(..., connected)` so a stale miss/fail flag
  can't override an event that clearly landed. `connected = damaging_event is not
  None or move_id in SELF_KO_MOVES` (`{explosion, selfdestruct}`) — a **neutral**
  Explosion emits no effectiveness line so no event promotes, but self-KO moves
  always connect when used (an immune target *does* emit). Without this a neutral
  self-faint read `miss`.
- **Effectiveness guard** — `_align_effectiveness(move_id, eff, event)` nulls
  effectiveness to the unknown sentinel when the turn-gated event names a different
  move than the one that fired (hidden-power variants count as matching).

The action index is now used only to resolve **switch targets**, never move ids.

### Ordering-integrity module (`action/ordering_integrity.py`, new)

Three alignment invariants became executable checks that raise
`OrderingMismatchError` instead of failing silently:

| Function | Invariant | Wired into |
|---|---|---|
| `reorder_move_bits_to_sorted` + `assert_sorted_validity_correct` | prev-mask MOVE bits map action-order → sorted-order before becoming an obs feature (fixes a validity bit landing on the wrong move's embedding) | `EpisodeTracker.prev_mask` |
| `check_switch_ordering_alignment` | switch bits line up with the team slot order the model consumes | `Gen3ActionMasker.action_masks` |
| `check_outcome_matches_intent` | pressed move == fired move, with principled skips for callers (Sleep Talk/Metronome/…), `\|cant\|`, switch-out, two-turn charge, stale `last_move` | `reward_tracker` |

### Forensic replay state

To make any decision replayable offline, `RLPlayer._predict_best_action` now also
computes the critic `value` and stashes `self._last_prediction = {obs, logits,
value}` (without changing its return contract). `BattleRecorder` gained a parallel
`_states` list + `states_arrays()`, `ReplayCallback` writes a sibling
`battle_N_states.npz`, and `src/main/probe_replay.py` (new) replays a recorded obs
back through the model.

### Layout

The only dimensional change is the `fainted` cant reason (one per side): cant
one-hot 11→12, **TURN_DELTA_DIM 108 → 110**, obs **2734 → 2754** (`10 × 2`). No
`ARCH_SIGNATURE` bump — the forward pass is unchanged; the obs-dim field rejects
pre-step checkpoints.

---

## F — Reward overhaul: anti-spam taxes + dead-matchup pivot pressure

### Goal

Make pivoting out of a useless matchup **strictly out-value** spamming a useless
move — attacking the established root cause (switching systematically
under-valued, not matchup-blindness). Targets four eval pathologies:
immune/futile-attack spam, switch oscillation, capped setup/no-op spam, and
non-terminating stalls.

### Changes (`reward_manager.py`)

| Term | Before | After |
|---|---|---|
| Repetition tax | plateaus after the 4th repeat | **linear, uncapped** `max(-STEP·n, floor)` — steep zero-effect step for no-op/immune repeats, gentle step for productive ones |
| "had effect" definition | damage only | **"did something productive"** — damage OR boost gained OR status landed OR hazard added (so capped Calm Mind past +6, Spikes at 3, redundant status all route through the steep zero-effect path) |
| Dead-matchup tax | — | **new**: every damaging move 0× vs opp active and we don't switch → `-0.10·n` (floor −2.0), reset on switch |
| Switch-bounce tax | flat | **escalates** `-0.15·n` (floor −2.0) to kill A→B→A oscillation that dodges the move-repetition tax |
| Immune-attack penalty | −0.25 | **−0.5** |
| Stall tax | starts turn 125 | **starts turn 60**, ramps with turns-past-start, clamped at −0.5/turn |

`reward_manager_test.py` gained/updated coverage for every changed term;
`reward_invariants_e2e_test.py` was fixed for the new escalating/ramped model.

---

## Combined dimensions & versions (end of batch)

### Observation

| Field | Value | Set by |
|---|---|---|
| Move slot | 11 dims (`+accuracy`, `+never_miss`) | A |
| `TURN_DELTA_DIM` | **110** (108 → +2 `fainted`) | E (on top of B's 88→108) |
| **Obs dimension** | **2754** | E (`2734 → 2754`) |

### `ARCH_SIGNATURE` chain since the last run

```
gen3_move_outcome_v1   (B)  →  gen3_modular_v1   (C)  →  gen3_dual_value_v1   (D, current)
```

E changed the obs dim without a signature bump (caught by the obs-dim weight
field); A and F are not weight-relevant to the signature. Pre-batch checkpoints
fail `check_compatible()` at load with a clear error — correct for this
rapid-iteration project.

---

## Test status (end of batch)

- **Unit suite: 924 passed, 2 skipped** (+ integration green). Includes the new
  `phase_modules_test.py` (per-phase isolation + value-pool masking),
  `move_attribution_test.py` (27 tests pinning the attribution/outcome decision
  table), `ordering_integrity_test.py`, `moves_test.py` (accuracy/never-miss),
  dual-head `snapshot_test.py` round-trip, and reward-term coverage.
- **E2E fuzz green.** `move_outcome_fuzz_e2e_test.py` with the `FaintEdge` team:
  **0 edge mismatches** — Explosion self-faints all `hit` (incl. neutral, via the
  self-KO rule), switch-in deaths stay switches, KO-before-acting all
  `cant_reason = "fainted"`, all four layers consistent. `transition_fuzz` intent↔
  outcome: **0 real mismatches** over ~30K turns.
- **Startup smoke** (`train_rl_agent.py --debug`) prints
  `[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: (1, 512))`, completes
  episodes, fires the replay callback, survives a 250-turn stall, prints eval win
  rates — full pipeline runs on the dual-head policy.

---

## What this enables

- A value head that **reads the board through its own lens** (global pool + global
  scalars), with decoupled critic gradients — groundwork for the v5 MCTS `V(s)`.
- An observation that, in the turn-history window, reports the **actual** move each
  side used (even on faint turns), a clean **"nothing happened"** (`fainted`)
  signal, correct **outcome** for self-KO moves, and **accuracy/never-miss** so the
  policy can weigh a 70% Hydro Pump against a sure-hit Surf.
- A reward that makes **pivoting out-value spamming**, directly attacking the four
  observed eval pathologies.
- **Three hard integrity assertions** so a future ordering regression raises loudly
  instead of silently feeding the model misaligned validity bits — plus recorded
  `*_states.npz` snapshots that make any decision replayable offline.

---

## Files Changed

### A — accuracy
| File | Change |
|---|---|
| `data/pokemon/gen3_moves.json` | Regenerated with per-move `accuracy` / `never_miss` |
| `src/agents/observation/moves.py` | Move slot +accuracy (off 9) +never_miss (off 10) |
| `src/agents/observation/constants.py` | Move-slot layout offsets |
| `src/agents/model/features_extractor.py` | Move processor takes the 2 new scalars |
| `tools/pokemon_data_extractor/sync.py` (new), `tools/pokemon_data_extractor/pokemon_data_extractor_test.py` (new) | Replaces `gen_abilities_by_gen.py`; extracts moves+abilities, fixes ability ceiling 77→76 |
| `src/agents/observation/{moves_test,pokemon_test,state_encoder_test}.py` | Coverage + dim |

### C — modular refactor
| File | Change |
|---|---|
| `src/agents/model/features_extractor.py` | `forward_internal` → phase `nn.Module`s + `ExtractorContext`; phase-prefixed `state_dict` |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE → gen3_modular_v1` |
| `src/agents/model/phase_modules_test.py` (new), `features_extractor_test.py`, `model_embedding_test.py` | Per-phase tests; nested paths |
| `src/main/visualize_arch.py`, `CLAUDE.md`, `src/agents/model/CLAUDE.md` | Netron block grouping; refreshed arch docs |

### D — dual-head policy/value
| File | Change |
|---|---|
| `src/agents/model/features_extractor.py` | `value_cls` pool (4-tuple); `ProjectionAssembler` `(pi,vf)`; second projection; tuple `forward` |
| `src/agents/model/policy.py` (new) | `Gen3DualHeadMaskablePolicy` — one body, routes each tuple half |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE → gen3_dual_value_v1` |
| `src/main/train_rl_agent.py` | Pass the dual-head policy; round-trip unpacks the tuple |
| `src/agents/model/{phase_modules_test,features_extractor_test,features_extractor_hp_test,snapshot_test}.py`, `state_encoder_test.py` | Value-pool tests; unpack `(pi,vf)`; both heads reproduced |
| `CLAUDE.md`, `src/agents/model/CLAUDE.md`, `README.md` | Document the dual-head readout |

### E — protocol-accurate attribution
| File | Change |
|---|---|
| `src/agents/training/battle_context.py` | `SELF_KO_MOVES`; `our_last_move_id`; `_ko_before_acting` / `_align_effectiveness` / `_derive_move_outcome(connected)` / `_moves_match`; protocol-truth `our/opp_move_id` + `cant_reason="fainted"`, symmetric |
| `src/agents/action/ordering_integrity.py` (new) + `_test.py` (new) | Reorder/assert, switch & move-validity alignment, outcome↔intent |
| `src/agents/training/move_attribution_test.py` (new) | 27 decision-table tests |
| `src/agents/observation/turn_delta_encoder.py` | `"fainted"` cant reason (11→12) |
| `src/agents/action/mask_generator.py` | `check_switch_ordering_alignment` |
| `src/agents/training/episode_tracker.py` | `prev_mask` reorders MOVE bits + asserts |
| `src/agents/training/reward_tracker.py` | Per-turn `check_outcome_matches_intent` |
| `src/agents/inference/player.py` | Compute `value`; stash `_last_prediction` |
| `src/agents/training/{battle_recorder,replay_recorder}.py` | `_states` / `states_arrays()`; save `battle_N_states.npz` |
| `src/main/probe_replay.py` (new) | Replay a recorded obs through the model |
| `src/agents/training/poke_env_gaps/{move_outcome_fuzz_e2e_test,transition_fuzz_e2e_test}.py`, `README.md` | FaintEdge team + per-side edge validation; intent↔outcome; Fuzz Coverage Map |
| `src/agents/observation/{turn_delta_encoder_test,state_encoder_test}.py` | `TURN_DELTA_DIM == 110`; `EXPECTED_OBS_DIM = 2754` |
| `designs/ai_v3/README.md` | prev_mask / move-validity routing nodes |

### F — reward overhaul
| File | Change |
|---|---|
| `src/agents/training/reward_manager.py` | Linear-uncapped repetition tax; "productive" effect; dead-matchup tax; escalating switch-bounce; immune −0.5; stall from turn 60, ramped |
| `src/agents/training/reward_manager_test.py` | Coverage for every changed term |
| `src/agents/training/reward_invariants_e2e_test.py` | Fixed for the escalating/ramped model |

*Section B (move outcome) files are listed in `impl_step5_move_outcome.md`.*
