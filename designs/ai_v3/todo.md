# AI v3 — Future Work

---

## 1. Learning Rate Annealing

**Where:** `src/main/train_rl_agent.py`

Wang (2024) found constant LR plateaued at ~55% winrate vs SimpleHeuristicsPlayer;
their annealing schedule reached ~80%. The formula:

```
ℓ(x) = peak_lr / (8x + 1)^1.5    where x = training progress 0→1
```

SB3 accepts a callable for `learning_rate` that receives `remaining_progress_fraction`
(1→0), so `x = 1 - remaining`. Replace `learning_rate=args.lr` with:

```python
def make_lr_schedule(peak):
    def schedule(remaining):
        x = 1.0 - remaining
        return peak / (8.0 * x + 1.0) ** 1.5
    return schedule

learning_rate=make_lr_schedule(args.lr)
```

TensorBoard logs `train/learning_rate` automatically — should show a smooth decay curve.

---

## 2. Reduce clip_range and Learning Rate ✓ DONE

**Where:** `src/main/train_rl_agent.py`

Current values use SB3 defaults (`clip_range=0.2`, `lr=3e-4`). Wang (2024) used
`clip_range=0.0829` and a peak LR of ~6e-5 (with annealing). Both suggest our updates
are currently too large.

Suggested first step (conservative, easy to reason about):
- `clip_range`: 0.2 → 0.15 — reduces max policy jump per update without being as
  aggressive as the thesis; well-motivated by our large rollout size (96×2048=196K steps)
- `lr`: 3e-4 → 1.5e-4 — halves the LR, gets closer to the thesis's range without
  committing to a full annealing schedule; pair with LR annealing (§1) once validated

These can be added as hardcoded values first (like `gae_lambda`) or as CLI args if
you want to keep them easy to tune.

---

## 3. poke-env: Delegating Move `last_move` Gap

**Where:** `src/poke_env/battle/pokemon.py` → `Pokemon.moved()`
**Research:** `src/agents/training/poke_env_gaps/README.md`

### Problem

`TurnDelta.opp_move_id` is sourced from `battle.opponent_active_pokemon.last_move`, which
poke-env tracks via `Move._is_last_used`. When a delegating move (Metronome, Nature Power,
Assist, Mirror Move) fires its delegated action, poke-env calls `moved(delegated_id,
reveal=False)`, which sets `move=None` and clears all `_is_last_used` flags. Result:
`last_move = None` after the turn, even though the opponent clearly acted.

Affected moves in Gen 3:
- **Metronome** — picks a random move from the entire move pool
- **Nature Power** — becomes Swift in standard Showdown terrain
- **Assist** — picks a random teammate move (Delcatty, Persian)
- **Mirror Move** — copies the opponent's last move (Pidgeot, Swellow)

Sleep Talk is **not** affected (poke-env uses `pass` for it, not `reveal=False`), except
when delegation fails (0 PP) — in that case `last_move = "sleeptalk"`, which is correct.

### Current Impact

Low. None of these moves are standard in serious gen3ou teams. The model handles the gap
gracefully via `opp_move_known = False`. Confirmed by fuzz testing: `our_move_slot_unknown`
is 0 across 30K transitions, meaning our side is never affected.

### Proposed Fix (5-line poke-env change)

In `Pokemon.moved()`, always track the move ID regardless of `reveal`:

```python
if use:
    self._last_used_move_id = to_id_str(move_id)   # new field, always set
    for m in self.moves.values():
        m._is_last_used = m is move
```

Then expose `last_move_id: str | None` as a property that returns `_last_used_move_id`
directly, bypassing the `moves` dict scan. `BattleContext.opp_last_move_id` in
`gen3_env.py` would read this instead of `last_move.id`.

### Also: Explosion / Self-Destruct faint gap

When the opponent uses Explosion and faints, `opponent_active_pokemon` is already the
new switch-in by the time `_get_observation()` runs, so `last_move = None` for the
switch-in. Captured by `TurnDelta.opp_fainted = True` and `opp_prev_active`. Acceptable.

Fix requires intercepting `|faint|` in `AbstractBattle._parse_message()` to snapshot
`last_move` before the active slot changes. Slightly more invasive than the `moved()` fix.

---

## 4. Status and Stat-Stage Deltas in `TurnDelta`

**Where:** `src/agents/training/battle_context.py` → `TurnDelta`

### Problem

`TurnDelta` currently tracks HP deltas and faint events but not:
- **Status conditions** — burn, paralysis, sleep, freeze, poison applied or cured this turn
- **Stat stages** — Calm Mind boosts, Intimidate drops, etc.

These matter for reward signal (e.g., reward for inflicting burn) and future observation
encoding (knowing the opponent is now paralyzed changes move selection).

### Complexity

- **Aromatherapy / Heal Bell** clears the entire team's status at once — needs per-slot
  before/after snapshots, not a single delta.
- **Stat stages reset on switch** — need to track stage values per active slot across turns.
- **Intimidate** applies on switch-in, before the first move — needs careful turn ordering.

Add `our_status_delta: dict[str, str | None]` and `opp_status_delta` (slot → new status),
plus `our_stage_delta: np.ndarray` (6-stat vector per slot) when the architecture is ready
to consume them.

---

## 5. `TurnDeltaEncoder` — One-Turn Memory in the Observation Vector ✓ DONE

**Where:** `src/agents/observation/turn_delta_encoder.py`
**Implemented in Step 3 (same session as TurnDelta).** 29-dim block appended to obs by `gen3_env.embed_battle()`. See CLAUDE.md for layout.

### Problem

The current observation vector encodes the raw battle state (HP, status, moves, matchups)
but nothing about what happened *last turn*. The model must infer momentum signals
(the opponent just used Rock Slide, we should expect flinch pressure) from patterns in
consecutive obs frames, which is slow to learn for a feedforward network.

### Proposed Design

Append a fixed-dim block to the observation vector encoding the previous turn's
`TurnDelta`. Keeps the feedforward architecture (no LSTM required for basic one-turn
memory):

```
TurnDeltaEncoder output (~32 dims):
  our_move_id_embed       (16,)  — move embedding, zeros if we switched
  our_switched            (1,)   — bool
  our_failed_to_move      (1,)   — bool
  our_cant_reason_onehot  (5,)   — [par, slp, frz, flinch, confusion]
  opp_move_id_embed       (16,)  — zeros if opp switched or move unknown
  opp_switched            (1,)   — bool
  opp_failed_to_move      (1,)   — bool
  opp_cant_reason_onehot  (5,)   — same categories
  our_hp_delta_sum        (1,)   — scalar damage we took
  opp_hp_delta_sum        (1,)   — scalar damage we dealt
  we_fainted              (1,)
  opp_fainted             (1,)
  opp_move_known          (1,)   — False signals Explosion gap or new active mon
```

The `TurnDelta.empty()` sentinel (first turn of episode) maps to an all-zeros block.

### Trade-offs

- **Pro:** Gives the model clear signal for Rock Slide flinch value, paralysis disruption,
  Sleep Talk sequencing — things the current obs can only imply.
- **Pro:** No architecture change required; just widens the projection layer input.
- **Con:** Adds ~32 dims permanently. Verify `Gen3FeaturesExtractor` projection input
  updates correctly (it auto-discovers dim via dummy forward pass, so no hardcoding needed).

---

## 6. Observation / Encoding: Volatile Count Encoding ✓ DONE (partial)

**Where:** `src/agents/observation/active_context.py`, `src/agents/observation/pokemon.py`

**Done:** Sleep counter (`min(ctr,4)/4`) and toxic counter (`min(ctr,8)/8`) added to
per-Pokémon vector so they flow through role tokens and all 5 attention paths.
Perish Song upgraded from binary to scalar (`turns_left/3`) in active context.

**Gen 3 sleep nuance to be aware of (not yet encoded):**
Sleep lasts 1–4 turns in Gen 3. Using Sleep Talk or Snore while asleep increments
`status_counter` like a normal turn, but those increments are **discarded on switch-out**
due to an engine oversight — the counter reverts to its pre-Sleep-Talk value. The current
encoding uses `mon.status_counter` as-is; it will be slightly inflated for mons that
used Sleep Talk then switched. Not critical to fix — poke-env likely tracks this
correctly for our own mons (via Showdown messages), and opponent sleep counter is
already approximate.

**Architectural gap (separate item below):** Boosts and volatiles in active context
are invisible to all 5 attention paths — see item 9.

**Remaining:** Move PP / Encore countdown — low priority.

---

## 7. Turn-History Memory

**Where:** `src/agents/model/features_extractor.py`, `src/agents/training/episode_tracker.py`

Out of scope for the current breaking-change wave, but the natural next step
after the one-turn `TurnDelta` block already in the observation. Two-phase
plan, in order:

1. **Sliding window first.** Append the last K `TurnDelta` blocks (or the
   last K `our_active_refined` tokens after attention) to the projection
   input. K configurable, default 3. This is the cheapest possible
   recurrence approximation: no recurrent state, no architecture change,
   only a wider concat. Easy to ablate by setting K=1.

2. **GRU pass.** Once the sliding window is in place and the model has
   learned to exploit it, replace the stacked-window block with a small
   `nn.GRUCell(128, 128)` over the per-turn role-token deltas. Hidden state
   threaded through `EpisodeTracker` (extension point already wired in
   Step 1). Adds true recurrence without committing to a full LSTM/
   Transformer history.

`EpisodeTracker._history` already exists and both designs build on it. The
sliding-window prototype should land first to establish a baseline that the
GRU has to beat.

---

## 8. Hidden Power — Opponent Type Inference

**Where:** `src/agents/observation/moves.py`, `src/agents/observation/turn_delta_encoder.py`

### Problem

Hidden Power is a critical competitive mechanic in Gen 3 (Celebi, Zapdos, Jolteon, etc.
commonly run HP Water, HP Ice, HP Fire). Gen 3 Showdown uses the base "hiddenpower" move ID
internally and sets the actual type at runtime via `onModifyMove`. The `|move|` battle log
emits "Hidden Power" (no type suffix) — see `dist/sim/battle-actions.js:380`. poke-env
cannot recover the HP type from this message.

### Current Mitigation (already applied)

- Opponent HP in both `moves.py` and `turn_delta_encoder.py` encodes `type_id=0` (unknown
  sentinel, distinct from Normal=1) and `basePower=70` (competitive assumption).
- `TypeEncoder.IDX_TO_TYPE[0] = "UNKNOWN"` so display shows `hiddenpower [Unknown, 70bp]`.
- Our own HP moves encode correctly (typed variant comes from the Showdown request).

### Proper Fix (future)

Two avenues, both non-trivial:

**a) Damage-effectiveness inference**: After `|move|` and `|-supereffective|` / `|-resisted|`
messages, infer the HP type from the effectiveness and the target's known types. Needs
poke-env message-stream access and type-chart lookup. Often ambiguous (e.g. a 2× hit on a
Water/Ground mon could be HP Grass or HP Ice).

**b) Explicit HP type tracking via poke-env extension**: Add a `hp_type: str | None`
attribute to poke-env's `Pokemon` class, populated from team-preview or from the first
move-use log if Showdown ever exposes it. Gen 3 Random Battles have no team preview, so
this only helps in non-random formats.

Until one of these is implemented, the model must infer the opponent's HP type from patterns
(which mons use HP, HP delta magnitude, matchup context). The `type_id=0` encoding gives it
a clean "I don't know" signal to reason from.

---

## 9. Architecture Gap: Boosts and Volatiles Blind to Attention

**Where:** `src/agents/model/features_extractor.py`

### Problem

Boosts (stat stages) and volatile effects (Taunt, Confusion, Substitute, etc.) are encoded
in the **active context** stream (23 dims → 32-dim MLP → concatenated at the projection head).
This means all 5 attention paths (Pressure, Safety, Synergy, Threat, Opp Synergy) are
**completely blind** to them.

Concrete examples:
- Pressure ("what threatens our active from their bench?") cannot see that their benched
  Blissey is at +6 SpA after three Calm Minds.
- Safety ("which of our bench can switch in safely?") doesn't know our active Skarmory
  is Taunted and can't use Spikes or Roost.

This is not a bug — it's a design gap from having status counters and volatile effects
arrive too late in the pipeline (after attention has already run).

### Proposed Fix

For the **active mon only**, inject the active context encoding into that mon's role token
**before** the attention paths run:

```python
# After role encoding, find the active slot and add the active-ctx projection
our_active_idx = our_active_flags.argmax(dim=1)  # [B]
# Project active context (23 dims) → role token size (128 dims)
our_ctx_boost = self.active_ctx_to_role(our_ctx_raw)  # [B, 128]
# Add into the active slot's role token
role_tokens[:, our_active_idx, :] += our_ctx_boost
# Same for opponent
opp_active_idx = opp_active_flags.argmax(dim=1) + TEAM_SIZE
opp_ctx_boost = self.active_ctx_to_role(opp_ctx_raw)
role_tokens[:, opp_active_idx, :] += opp_ctx_boost
```

Where `self.active_ctx_to_role = nn.Linear(ACTIVE_CONTEXT_DIM, ROLE_TOKEN_SIZE)`.

Benched mons have no boosts/volatiles, so only injecting into the active slot is correct.
`ROLE_TOKEN_SIZE` stays at 128 — no downstream dim changes needed.

### Priority

Medium. The projection MLP can still use boosts/volatiles to influence final logits,
so the model can partially compensate. But strategic attention-level reasoning (e.g.,
"their bench +4 Blissey is a bigger threat than my typing suggests") requires this fix.
