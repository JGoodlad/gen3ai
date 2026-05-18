# Hidden Power Type Inference — Design & Implementation Plan

## Problem

Gen 3 Showdown sends `|move|...|Hidden Power|...` with no type suffix. poke-env stores the
opponent's HP move as `_id = "hiddenpower"`, so we can never recover the type from the battle
log. Our current mitigation (`type_id=0 / Unknown, 70bp`) is honest but leaves the model
blind to one of the most important Gen 3 metagame dimensions.

HP type is frequently decisive: Celebi almost always runs HP Fire or HP Ice; Zapdos runs HP
Ice or HP Water; Jolteon runs HP Grass. Misidentifying an opponent's HP type is one of the
most common causes of bad human matchup decisions in Gen 3 OU.

---

## Why Not the Simpler Approaches

**Parsing `-supereffective` / `-resisted` messages:** These are in poke-env's
`MESSAGES_TO_IGNORE` and carry only the target identifier — no move type, no magnitude.
Even if parsed, a single super-effective hit on Starmie (Water/Psychic) is consistent with
5+ HP types. Not viable.

**Pure poke-env extension (`hp_type` attribute):** The battle log gives us nothing to
populate it with for Gen 3 Random Battles. It's the right storage layer but has no data
source on its own.

**Fixing local Showdown fork to send the type:** Works for self-play training, but on the
actual ladder (human opponents) we use standard Showdown which sends no type. The model
trained this way would have a feature that's always "Unknown" at ladder time. Invalid.

---

## Chosen Approach: HP Evidence Block per Opponent Slot

### Core idea

After each turn where the opponent used Hidden Power, we:
1. **Rule-based step (in the env):** Infer the effectiveness tier from `opp_hp_delta` +
   our active mon's types + approx opponent Sp.Atk (from species base stats). This gives one
   of: `immune / resisted / neutral / super_effective`. The ±15% random roll is absorbed by
   rounding to tier — super-effective hits are ~2× neutral so the bands don't overlap.
2. **Store the event** in `EpisodeTracker._opp_hp_events[slot]`: a list of
   `(our_type1_id, our_type2_id, effectiveness_onehot)` tuples, capped at K=3 (rare to see
   more than 2–3 HP uses per battle).
3. **Inject per-slot into the role encoder input** for each opponent slot as a fixed-size
   HP evidence block: `K × (type1_emb[16] + type2_emb[16] + effectiveness_onehot[4])` =
   K×36 dims. Reuse the shared type embedding table already in the network.
   Slots where HP has never been used → all zeros.

### Where it plugs into the network

The **role encoder input** for each opponent slot is the right injection point. The role
encoder already processes the full per-mon vector (species, stats, moves, types) into a
128-dim role token. Adding the HP evidence block here means the role token carries "what I
know about this mon's HP type" naturally alongside everything else.

The role token then flows through all five attention paths:
- **Pressure** (our active ← their team) — most impacted: helps evaluate threat level of
  an unrevealed HP type
- **Safety** (our bench ← their active) — helps identify which of our mons can safely
  switch into an HP user
- **Opp Synergy** (their team ← their team) — helps model infer team composition context

No changes needed to the attention paths or the projection layer — the projection input
dimension is auto-discovered via dummy forward pass in `FeaturesExtractor.__init__`.

### What the model learns

- **Rule-derived signals it doesn't have to discover:** Effectiveness tier directly
  constrains possible HP types — the model sees structured evidence, not raw damage floats.
- **Metagame priors:** Species embedding carries implicit prior (Celebi pattern → model
  learns to concentrate on Fire/Ice even without evidence). Evidence then sharpens it.
- **Uncertainty-aware play:** When hp_evidence is all-zeros (HP never used), the model
  has learned to hedge vs. opponent mons that commonly carry HP. As evidence accumulates
  across turns, strategy adjusts.

### Credit assignment

The payoff for correctly inferring HP type is delayed (comes when you make the right
matchup decision several turns later). The HP evidence block gradient flows through the
role token → attention → projection → policy — standard credit assignment. Will be slow
to learn in early training; metagame priors from species embeddings bootstrap it.

---

## Implementation Steps

### Step 1 — Rule-based effectiveness inference utility

**File:** `src/agents/observation/hp_type_inference.py` (new)

```python
def infer_hp_effectiveness(
    hp_delta: float,           # abs(our HP fraction lost this turn)
    our_type1: str,            # e.g. "ELECTRIC"
    our_type2: str | None,     # e.g. "FLYING", or None
    opp_spa: int,              # opponent's base Sp.Atk from species data
    our_spd: int,              # our base Sp.Def from species data
    our_level: int = 50,
) -> str:  # "immune" | "resisted" | "neutral" | "super_effective"
```

Uses the Gen 3 damage formula to compute expected neutral damage fraction, then buckets
the observed delta into effectiveness tier. Use `GenData.from_gen(3).type_chart` (already
loaded in the codebase).

Returns `None` if the signal is too noisy (e.g., multi-hit turn, HP delta too small to
classify). Call only when `TurnDelta.opp_move_id == "hiddenpower"`.

### Step 2 — EpisodeTracker: store HP events

**File:** `src/agents/training/episode_tracker.py`

```python
# New field:
_opp_hp_events: dict[int, list[HPEvent]]   # slot index → list of events
# HPEvent = (type1_id: int, type2_id: int, effectiveness: str)
```

- Initialize to `{i: [] for i in range(6)}` on reset.
- After each turn where `TurnDelta.opp_move_id == "hiddenpower"`:
  - Determine which opponent slot was the attacker
  - Call `infer_hp_effectiveness(...)` using the current `TurnDelta`
  - Append `HPEvent(our_active_type1_id, our_active_type2_id, effectiveness)` to that slot
  - Cap list at K=3 (discard oldest if needed)

### Step 3 — BattleContext: carry hp_events snapshot

**File:** `src/agents/training/battle_context.py`

Add `opp_hp_events: dict[int, list[HPEvent]]` to `BattleContext`. Populated by
`BattleContext.from_battle()` from `EpisodeTracker._opp_hp_events` (passed in as arg).

### Step 4 — FeaturesExtractor: HP evidence block in role encoder

**File:** `src/agents/model/features_extractor.py`

For each opponent slot before the role encoder:
1. Look up `opp_hp_events[slot]` from the current BattleContext (passed via obs or a side
   channel — see note below).
2. Build `hp_evidence_block`: shape `[K, 36]` where each row is
   `concat(type1_emb[16], type2_emb[16], effectiveness_onehot[4])`. Pad with zeros.
3. Flatten to `[K*36]` and concatenate onto the per-slot role encoder input.

**Note on passing hp_events to FeaturesExtractor:** The cleanest option is to encode the
HP events as a fixed-size numeric block and append it to the observation vector (like the
prev\_mask and TurnDelta blocks). The env's `embed_battle()` builds this block from
`EpisodeTracker._opp_hp_events`, and FeaturesExtractor slices it out of the obs.
Avoids any side-channel architecture.

Observation block layout: 6 slots × K events × (type1\_id, type2\_id, eff\_class) =
6 × 3 × 3 = **54 raw ints** appended to obs. FeaturesExtractor embeds them using the
existing type embedding table + a 4-dim effectiveness embedding.

### Step 5 — Model versioning

Bump `MODEL_CONFIG_VERSION` and add migration in `model_version.py`. New checkpoints
required (expected — rapid iteration project).

---

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/observation/hp_type_inference.py` | **New** — effectiveness tier inference utility |
| `src/agents/training/episode_tracker.py` | Add `_opp_hp_events`, update on HP use |
| `src/agents/training/battle_context.py` | Add `opp_hp_events` field |
| `src/agents/training/gen3_env.py` | Build HP events obs block in `embed_battle()` |
| `src/agents/model/features_extractor.py` | Slice HP events block; embed and concat to opp role encoder inputs |
| `src/agents/model/model_version.py` | Bump version, add migration |
| `designs/ai_v3/todo.md` | Mark item #6 as "in progress / designed" |

---

## Verification

1. **Unit test `hp_type_inference.py`:** Given a known HP type (e.g., HP Water), simulate a
   hit on a Water-type mon and confirm effectiveness = "neutral"; on a Fire-type mon →
   "super_effective".
2. **Unit test obs block:** Confirm the obs dimension increases by exactly 54 and the HP
   events block round-trips correctly through `embed_battle()` / FeaturesExtractor slice.
3. **Smoke test:** `train_rl_agent.py --debug --steps 10000` — check obs integrity passes,
   no crashes, `[ModelVersion] FATAL` does not appear.
4. **All existing unit tests pass.**

---

## Open Questions

- **K=3 events:** Is 3 the right cap? Gen 3 battles average ~30 turns; HP is used perhaps
  1–5 times. K=3 captures the most common cases. Could go to K=4 if needed.
- **Level assumption:** The formula assumes level 50. Gen 3 Random Battles use level 100.
  Adjust the formula accordingly (the constant factor cancels so effectiveness tier is
  unaffected).
- **Obs block encoding:** Storing raw int IDs (type1, type2, eff_class) as float32 in the
  obs is slightly inelegant but consistent with how TurnDelta move IDs and type IDs are
  stored. Alternative: one-hot encode everything (~13 dims/event → 6×3×13=234 dims).
  Raw IDs fed through embeddings are probably better.
