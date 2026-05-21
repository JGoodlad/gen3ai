# Implementation: Step 9 — Signal Routing and Reward Overhaul

This step contains three major, independent improvements made in the same sprint:

1. **TurnDelta Conditioner** — the effectiveness and move-order signals added in Step 7
   were reaching the projection as raw scalars but bypassing all five team-attention
   paths. A new shared MLP now routes them additively into the active-role tokens
   *before* attention runs, so every Pressure/Safety/Synergy/Threat/OppSynergy head
   can reason about last-turn matchup and speed context.

2. **N=5 Turn History with MHA** — the single-TurnDelta slot (39 dims) was replaced
   with a 5-turn history block (195 dims). All five deltas are embedded through the
   shared move/type tables, positional encodings are added, and a 3-head MHA
   self-attention block produces one 99-dim history-informed vector for the projection.
   The model can now detect bounce loops, momentum, and multi-turn speed trends.

3. **Reward Function Overhaul** — nine new reward signals and two corrections make the
   reward function considerably more precise: escalating repetition tax, HP-scaled faint
   cost, futile-setup penalty, setup-while-dying penalty, status-wasted penalty,
   boost-utilization reward, explosion-block bonus, se_switch restriction to voluntary
   switches, and volatile-status move removal from the status-wasted check. A new E2E
   fuzz test validates 11 reward invariants across 30 real battles.

Supporting changes: per-opponent reward tracking in eval, concurrent-safe stall detection,
team-pool bias expansion, clip_range constant, and greedy-argmax replay.

---

## 1. TurnDelta Conditioner

### Motivation

Step 7 added `our_effectiveness_onehot(4)`, `opp_effectiveness_onehot(4)`, and
`move_order(2)` to the TurnDelta block. In Step 8's network these 10 scalars entered
the model only in the final projection (`Linear(562→512)`), **after** all five
attention passes over the 12 Pokémon role tokens had already finished.

This meant:

- The Pressure head (our active ← their team) had no knowledge of whether our attack
  was SE or resisted last turn.
- The Safety head (our team ← their active) could not condition on whether the opponent
  outsped us.
- The Threat and Synergy heads similarly saw stale role tokens.

In practice the model had to infer type-matchup signals from the raw effectiveness
scalars appearing late, without the team-structure context those signals most need.

### Design

A new shared MLP — the **TurnDelta Conditioner** — maps the 10-dim strategic TurnDelta
slice `[our_eff(4), opp_eff(4), order(2)]` into role-token space (128 dims) and adds
the result to the active-slot tokens before any attention path runs. This mirrors the
existing `active_ctx_to_role` path (which injects boosts/volatiles into active tokens
pre-attention) and costs nothing in projection-dimension bookkeeping.

The opp active token receives a **perspective-flipped** input so the shared MLP sees
consistent `[attacker_eff, defender_eff, we_first, opp_first]` framing from both sides:

```
Our active input:  [our_eff(4), opp_eff(4), we_first(1), opp_first(1)]
Opp active input:  [opp_eff(4), our_eff(4), opp_first(1), we_first(1)]
```

### Architecture

```python
TD_STRATEGIC_DIM    = EFF_DIM * 2 + ORDER_DIM  # 10
TD_STRATEGIC_OFFSET = TURN_DELTA_DIM - 10       # 29: tail of the 39-dim TurnDelta block

self.td_conditioner = nn.Sequential(
    nn.Linear(TD_STRATEGIC_DIM, 64),
    nn.ReLU(),
    nn.Linear(64, ROLE_TOKEN_SIZE),  # → 128
)
```

Applied in `forward_internal`, after the active-ctx biases and before the five attention
heads:

```python
td_strategic   = turn_delta_feat[:, TD_STRATEGIC_OFFSET:]   # [B, 10]
our_td_bias    = self.td_conditioner(td_strategic)           # [B, 128]
opp_td_input   = cat([opp_eff, our_eff, opp_first, we_first], dim=1)
opp_td_bias    = self.td_conditioner(opp_td_input)           # [B, 128]
role_tokens[our_active_idx] += our_td_bias
role_tokens[opp_active_idx] += opp_td_bias
```

`ARCH_SIGNATURE` bumped to `gen3_td_cond_v1`; old checkpoints fail cleanly at startup.

### Test Coverage

7 new unit tests in `features_extractor_test.py`:

| Test | Validates |
|---|---|
| `test_td_strategic_constants_consistent` | `TD_STRATEGIC_DIM=10`, `TD_STRATEGIC_OFFSET=29` match TurnDelta layout |
| `test_td_conditioner_shape` | Maps `[B, 10] → [B, ROLE_TOKEN_SIZE]` |
| `test_our_effectiveness_changes_output` | Non-zero our_eff changes network output |
| `test_opp_effectiveness_changes_output` | Non-zero opp_eff changes network output |
| `test_move_order_changes_output` | Non-zero order bits change output |
| `test_td_conditioner_wired_pre_attention` | Zeroing conditioner weights changes output, proving it is applied pre-attention |
| `test_our_opp_asymmetry` | Swapping our_eff/opp_eff produces different outputs (perspective flip is active) |

---

## 2. N=5 Turn History with MHA Attention

### Motivation

Replay analysis at ~99M steps found ~30% of battles containing A→B→A→B bouncing
switches within 3–4 turns. With only the immediately previous TurnDelta visible, the
model had no memory of what it had just done. Adding N=5 turns of history allows:

- Detecting and penalising bounce loops (A→B→A→B within 3 turns visible in history)
- Recognising attack momentum (KO'd two mons in a row — press the attack)
- Tracking move-order trends (outsped 3 turns running — speed context present)
- Stall spiral detection (flat HP-delta pattern for 3+ turns)

### Observation Vector Change

The single 39-dim TurnDelta slot is replaced with a 5 × 39 = 195-dim history block:

| Block | Before | After | Offset |
|---|---|---|---|
| Base encoder | 1103 | 1103 | 0 |
| Prev-turn action mask | 11 | 11 | 1103 |
| **TurnDelta history (N=5)** | **39** | **195** | 1114 |
| **Total** | **1153** | **1309** | — |

Within the 195-dim block (offset 1114): oldest turn at `[0:39]`, most recent at
`[156:195]`. Zero-padded at episode start. Index N−1 carries the same data as the
former single-TurnDelta slot, so no information is lost.

### Network Changes

```python
N_HISTORY_TURNS = 5

# Embedding dim per slot: 2*move_emb + 2*type_emb + 35 scalars = 99
self.turn_history_pos_emb = nn.Embedding(N_HISTORY_TURNS, 99)    # positional
self.turn_history_attn    = nn.MultiheadAttention(99, 3, batch_first=True)
self.turn_history_norm    = nn.LayerNorm(99)
```

In `forward_internal`:

```python
history_slots   = turn_history_raw.view(B, N, TURN_DELTA_DIM)  # [B, 5, 39]
embedded_slots  = stack([embed_slot(history_slots[:, t]) for t in range(N)], dim=1)
embedded_slots += turn_history_pos_emb(arange(N))              # [B, 5, 99]
attn_out, _     = turn_history_attn(embedded_slots, embedded_slots, embedded_slots)
attended        = turn_history_norm(embedded_slots + attn_out)
turn_delta_embedded = attended[:, -1, :]                        # [B, 99] — most-recent
```

The most-recent position (index N−1) is used in the projection because it aggregates
context from the full window via the attention mechanism. This replaces the former
89-dim single-delta embedding; the projection input is auto-discovered via a dummy
forward, so no projection constant required updating.

### EpisodeTracker Changes

Added `_actions: list[int]` parallel to `_history`. `record()` commits the pending
action before appending the new context, so `_actions[i]` is always the action taken
from `_history[i]`.

New method:

```python
def prev_N_delta_vecs(self, n: int, encoder: TurnDeltaEncoder) -> np.ndarray:
    """Return (n, 39) array of encoded TurnDeltas, oldest-first, zero-padded."""
    result = np.zeros((n, encoder.dimension), dtype=np.float32)
    available = min(n, len(self._history) - 1, len(self._actions))
    for i in range(available):
        action  = self._actions[-1 - i]
        ctx_prev = self._history[-2 - i]
        ctx_curr = self._history[-1 - i]
        delta    = TurnDelta.build(ctx_prev, ctx_curr, action)
        result[n - 1 - i] = encoder.encode(delta)
    return result
```

`gen3_env.embed_battle()` replaces the single-delta call with
`episode_tracker.prev_N_delta_vecs(N_HISTORY_TURNS, delta_encoder).flatten()`.

### Inference Player

`Gen3Player` now maintains a `dict[str, EpisodeTracker]` keyed by `battle_tag` and
calls `record()` / `prev_N_delta_vecs()` identically to the training env.
`_battle_finished_callback` cleans up both the episode tracker and, in this step, the
new stall-logger dict (see §4). Eval win rates now reflect the model's full-history
capability rather than single-frame inference.

### Model Versioning

`ModelVersion` gains `n_history_turns: int` (added to `_WEIGHT_FIELDS`). Version bumped
to 2. Migration: old v1 models default to `n_history_turns=1`.

### Test Coverage

| File | New tests |
|---|---|
| `episode_tracker_test.py` (new) | 14 tests: `_actions` sync, `prev_N_delta_vecs()` shape/padding/ordering/action index/last-slot invariant |
| `features_extractor_test.py` | 9 tests: pos_emb/attn/norm module existence, shape, functional (history changes output), pos_emb wired-in proof |
| `player_test.py` | 5 tests: per-battle tracker isolation and cleanup |
| `snapshot_test.py` | `n_history_turns` weight-field mismatch test + migration test; stale v1→v2 assertion fixed |

Also see `designs/ai_v3/impl_step3_turn_history.md` for the detailed design rationale.

---

## 3. Reward Function Overhaul

### Motivation

Analysis of training replays at ~100M steps revealed several categories of behaviour
the reward function was either failing to discourage or accidentally rewarding:

- The agent repeated the same zero-effect move without escalating cost (flat −0.02 tax).
- Setup moves at stat cap (e.g. Calm Mind at +6 SpA) were free — the move fired but
  had zero mechanical effect.
- Setup below 40% HP was not penalised, so mons would attempt Dragon Dance while
  near-KO.
- Status moves (Toxic, Thunder Wave) aimed at already-statused or immune targets
  produced no `mon.status` change; no penalty fired.
- Attacking with +2 Attack was rewarded identically to attacking unbooosted.
- Sacrificing a high-HP mon vs a low-HP mon incurred the same faint cost.
- Ghost or Protect blocking Explosion only gave the standard faint reward with no
  extra bonus for the strategic play.
- The `se_switch` bonus fired even on forced post-faint replacements and when the
  opponent was already fainted.
- `attract`, `confuseray`, `supersonic` were in `STATUS_INFLICTING_MOVES` despite
  creating volatile status (`mon.status` stays None) — `status_wasted` always misfired.

### New Constants

| Constant | Value | Purpose |
|---|---|---|
| `FAINT_BASE` | `0.5` | Minimum faint reward/penalty (replaces flat `FAINTED_VALUE = 2.0`) |
| `FAINT_HP_SCALE` | `2.0` | Linear HP multiplier: faint value = `FAINT_BASE + FAINT_HP_SCALE × hp_before` |
| `FUTILE_SETUP_PENALTY` | `−0.3` | Setup move with no boost delta (at ±6 cap) |
| `SETUP_LOW_HP_THRESHOLD` | `0.40` | HP fraction below which setup is penalised |
| `SETUP_LOW_HP_MAX_PENALTY` | `−0.10` | Max setup-low-HP penalty (at 0% HP); linear to 0 at threshold |
| `STATUS_WASTED_PENALTY` | `−0.3` | Status move with no resulting `mon.status` change |
| `BOOST_UTILIZED_SCALE` | `0.03` | Reward per boost stage × damage dealt |
| `EXPLOSION_BLOCK_BONUS` | `1.0` | Ghost immune or Protect blocked opponent Explosion |
| `REPETITION_TAX_SCALE` | `(−0.02, −0.05, −0.10, −0.20)` | Flat-effect repeats: cost escalates with consecutive count |
| `REPETITION_TAX_ZERO_EFFECT_SCALE` | `(−0.05, −0.10, −0.20, −0.30)` | Zero-effect repeats: steeper escalation |

### Reward Signal Summary Table

| Signal | Before | After | Notes |
|---|---|---|---|
| `faint_ours` | −2.0 flat | −(0.5 + 2.0 × hp_before) | Sacrificing a 10% HP mon costs −0.7; a 100% HP mon costs −2.5 |
| `faint_opp` | +2.0 flat | +(0.5 + 2.0 × hp_opp_before) | Symmetric |
| `repetition_tax` | −0.02 flat | −0.02/−0.05/−0.10/−0.20 (escalating) | Doubled tier when previous attack had zero effect |
| `futile_setup` | *(none)* | −0.3 | Setup move at ±6 stat cap |
| `setup_low_hp` | *(none)* | −0.10 × (1 − hp/0.40) | Linear in HP below 40% threshold |
| `status_wasted` | *(none)* | −0.3 | Status move → no `mon.status` change |
| `boost_utilized` | *(none)* | atk/spa_stage × 0.03 × damage | Attack while holding active stat boosts |
| `explosion_block` | *(none)* | +1.0 | Ghost/Protect blocked opp Explosion (we took 0 damage) |
| `explosion` | −3.0 (mutual KO) | 0.0 (mutual KO) | `faint_ours` already covers the loss — no double-count |
| `se_switch` | voluntary + forced | voluntary only, opp alive | Removed false positives on forced post-faint replacements |

### Implementation Details

**HP-scaled faint — snapshot timing:**
`record_action()` captures `_our_active_hp_before` and `_opp_active_hp_before` from the
pre-action `BattleContext`. These are the HP fractions *at decision time*, not after
the attack, so the cost is anchored to what was actually risked.

**Escalating repetition tax:**
`_consecutive_attack_repeats` increments each time the same move index is chosen
consecutively; resets to 0 on any switch or different move. The index into the scale
is `min(consecutive_repeats − 1, 3)`. `_last_attack_had_effect` (set at end of each
turn as `opp_hp_delta.sum() < 0`) selects the normal or zero-effect scale for the
**next** repeat.

**Futile setup detection:**
Uses `delta.our_boost_delta.sum() == 0` as the criterion — the boost array carries the
per-stat change for this turn. If no stat changed, the move had zero mechanical effect
(already at cap, or Taunt blocked it). Guards: not if `our_failed_to_move` (already
penalised) and not if `we_fainted`.

**Status-wasted detection:**
`_compute_status_reward` was refactored to return `(reward, d_opp)` where `d_opp` is
the delta in the opponent's statused-mon count. `status_wasted` fires when the move
was a `STATUS_INFLICTING_MOVES` member, the opp didn't switch out, and `d_opp == 0`.
Guards: not if `our_failed_to_move`.

**Volatile-status correction:**
`attract`, `confuseray`, `supersonic` removed from `STATUS_INFLICTING_MOVES`. These
create volatile status (`attract_count`, `confusion`) that poke-env does not reflect in
`mon.status`, so the wasted check always misfired for them regardless of outcome.

**Boost-utilization reward:**
Fires only when `effective_boost = max(atk_stage, spa_stage) > 0` and the move dealt
damage. Uses boosts captured in `record_action()` (`_our_boosts_before = ctx.our_boosts.copy()`).

**Explosion mutual-KO correction:**
Previously charged `−3.0` when the opponent used Explosion and both sides fainted.
`faint_ours` already deducts the full HP-scaled faint cost, so the additional `−3.0`
was double-counting. The `explosion` field now only fires for `+2.0` (opp survived,
strategic loss for them). The new `explosion_block` fires separately when `our_hp_delta == 0`
(we took zero damage — Ghost immunity or Protect).

**se_switch restriction:**
Two new early-return guards in `_compute_se_switch_bonus()`:
- `if last_reward_metadata.get("type") != "VOLUNTARY": return 0.0` — excludes roar,
  post-faint forced replacements
- `if opp_mon.fainted: return 0.0` — excludes switching in after the opponent already fainted

### New E2E Fuzz Test

`src/agents/training/reward_invariants_e2e_test.py` — 30 real battles, 11 invariant checks:

| Invariant | What it verifies |
|---|---|
| Faint reward ≥ 0.5 + FAINT_HP_SCALE × hp | HP-scaled formula lower-bounded |
| Repetition tax escalates | Consecutive same-move tax increases each turn |
| Zero-effect repeats taxed more | Zero-effect scale ≥ normal scale at each tier |
| Futile setup fires at cap | `futile_setup < 0` when boost_delta == 0 and BOOST_MOVES |
| Setup-low-hp proportional | Penalty increases as HP decreases below threshold |
| Status-wasted fires | `status_wasted < 0` when status move had no effect |
| Boost-utilized fires | `boost_utilized > 0` when attacking with active boosts |
| Explosion block fires | `explosion_block > 0` when ghost/protect blocks Explosion |
| se_switch only voluntary | `se_switch > 0` never fires on forced switches |
| No double-count on mutual KO | `explosion == 0` when mutual KO occurs |
| Status-volatile exclusion | `status_wasted` never fires for attract/confuse/supersonic |

---

## 4. Evaluation Infrastructure

### Per-Opponent Reward Tracking

Before this step, `EvalCallback` only logged win rates; there was no per-episode reward
tracked during eval, making it impossible to distinguish "winning faster" from "winning
with better play quality."

**`RewardTracker`** (`src/agents/training/reward_tracker.py`) — standalone per-battle
reward accumulator that mirrors the Gen3Env deferred-reward pattern without requiring
Gen3Env:

```
begin_turn(ctx, action_idx)     → latch pre-action state
complete_pending(curr_ctx, battle) → settle previous turn when next choose_move fires
finalize(battle)                → settle last pending turn at battle end
```

**`RewardTrackingMixin`** — mixin for `RLPlayer` subclasses. Hooks `choose_move()` and
`_battle_finished_callback()`. Maintains a `dict[str, RewardTracker]` keyed by
`battle_tag` so concurrent battles don't share state.

**`EvalRLPlayer`** (`RewardTrackingMixin` + `RLPlayer`) logs `eval/mean_reward_vs_{name}`
to TensorBoard for each opponent type after every eval episode batch.

### Concurrent-Safe Stall Detection

The shared `_stall_logger` and `_last_battle_tag` attributes in `Gen3Player` caused
data races when multiple eval battles ran concurrently. Both are replaced with a
`dict[str, StallLogger]` keyed by `battle_tag`. Stall loggers are created on first use
in `_handle_stall()` and cleaned up in `_battle_finished_callback()`.

The `try-catch` in `RLPlayer.choose_move()` and `StatTrackingRLPlayer.choose_move()`
was also removed — exceptions now surface rather than silently returning a random move.

---

## 5. Supporting Changes

### Team Pool Expansion

The trainee team pool was expanded from a small curated set to all valid Gen 3 OU
teams in `data/teams/`, with a 50% bias toward a curated high-quality subset:

```python
Gen3Teambuilder(teams=all_teams, bias_teams=curated_teams, bias_prob=0.50)
```

`Gen3Teambuilder` gains `bias_teams` and `bias_prob` constructor parameters.
`yield_team()` samples from `bias_packed_teams` with probability `bias_prob` and from
`packed_teams` otherwise. Both pools are validated at construction time.

### Clip Range Constant

`clip_range=0.2` was specified as a literal in two places (direct call and resume path).
Extracted to `CLIP_RANGE = 0.20` at module level. Both `MaskablePPO()` instantiation
and the `model.clip_range = lambda _: CLIP_RANGE` path now reference the constant.

### Greedy Replay

The replay recorder was using stochastic action sampling (`model.predict(obs)`) when
saving battle replays. Changed to greedy argmax (`np.argmax(action_probs)`) so saved
replays show the model's best-known strategy rather than a sampled variant. Masked
logits (−∞ for illegal actions) ensure the argmax always picks a legal move.

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/model/features_extractor.py` | `TD_STRATEGIC_DIM/OFFSET` constants; `td_conditioner` MLP; `N_HISTORY_TURNS=5`; `turn_history_pos_emb/attn/norm` modules; N-turn embedding+attention forward block; `TURN_DELTA_EMBED_DIM` corrected 89→99 |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE → gen3_td_cond_v1`; `n_history_turns` field added to `ModelVersion`; `MODEL_CONFIG_VERSION` 1→2; migration v1→v2 defaults `n_history_turns=1` |
| `src/agents/training/episode_tracker.py` | `_actions: list[int]`; `prev_N_delta_vecs(n, encoder)` method |
| `src/agents/training/gen3_env.py` | `embed_battle()` replaces single delta with `prev_N_delta_vecs().flatten()` |
| `src/agents/observation/state_encoder.py` | `get_layout()` exposes `turn_history_offset`, `turn_history_dim`, `n_history_turns`; `dimension` 1153→1309 |
| `src/agents/inference/player.py` | Per-battle `EpisodeTracker` dict; `prev_N_delta_vecs()` in `embed_battle()`; per-battle `StallLogger` dict (concurrent-safe); `_battle_finished_callback` cleanup |
| `src/agents/training/reward_manager.py` | `FAINT_BASE/FAINT_HP_SCALE`; `FUTILE_SETUP_PENALTY`; `SETUP_LOW_HP_*`; `STATUS_WASTED_PENALTY`; `BOOST_UTILIZED_SCALE`; `EXPLOSION_BLOCK_BONUS`; `REPETITION_TAX_*_SCALE` tuples; `BOOST_MOVES`; `STATUS_INFLICTING_MOVES` (volatile moves removed); 4 new `_compute_*` methods; `faint_ours/opp` HP-scaled; `explosion` mutual-KO fix; `explosion_block` new field; `se_switch` restriction; `status_wasted` plumbed through; `_our/opp_active_hp_before`, `_our_boosts_before` snapshot fields |
| `src/agents/training/reward_tracker.py` | New file: `RewardTracker`, `RewardTrackingMixin` |
| `src/agents/training/battle_recorder.py` | Refactored to use `RewardTracker` |
| `src/agents/training/eval_callback.py` | `EvalRLPlayer` (mixin + player); per-opponent reward logging |
| `src/agents/training/replay_recorder.py` | Greedy argmax instead of stochastic predict |
| `src/utils/teambuilder.py` | `bias_teams`, `bias_prob` constructor params; `yield_team()` biased sampling |
| `src/main/train_rl_agent.py` | `CLIP_RANGE = 0.20` constant; trainee pool expanded to all teams with 50% bias |
| `src/agents/model/features_extractor_test.py` | 7 TurnDelta conditioner tests + 9 N-turn history tests |
| `src/agents/training/episode_tracker_test.py` | New file: 14 tests for `_actions` + `prev_N_delta_vecs()` |
| `src/agents/inference/player_test.py` | 5 per-battle tracker isolation + concurrency tests |
| `src/agents/model/snapshot_test.py` | `n_history_turns` weight-field mismatch test; migration assertions |
| `src/agents/training/reward_tracker_test.py` | New file: `RewardTracker` deferred-pattern tests + concurrency isolation |
| `src/agents/training/reward_manager_test.py` | Tests for all 9 new reward signals |
| `src/agents/training/reward_invariants_e2e_test.py` | New file: 11 reward invariants, 30 real battles |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` 1153→1309 |
| `designs/ai_v3/impl_step3_turn_history.md` | New file: detailed N-turn history design rationale |
