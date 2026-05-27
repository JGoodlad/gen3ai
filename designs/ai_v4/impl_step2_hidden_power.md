# Implementation: Step 2 — Hidden Power Type Inference

This step lets the network reason about opponent Hidden Power types. A new
`HiddenPowerTracker` maintains a per-species candidate distribution that is narrowed
every time the opponent uses HP, exposed to the model as a 17-dim per-slot block.
Resolving "which of our mons actually got hit" is driven by `TurnDelta` (not raw
action ints), so voluntary switches, Baton Pass, Roar, forced replacements, and even
the cycle-back-to-same-species case all return the right target. Two Gen 3 ability
quirks (Thick Fat and the Flash Fire-vs-frozen interaction) were folded into
`effective_multiplier` so calculated effectiveness matches Showdown.

Primary themes: candidate elimination from observed effectiveness, single-source-of-truth
action decoding via `TurnDelta`, status-aware ability quirks, an independent fuzz
validator that mirrors the resolver without sharing code, and a clean architecture
break via `ARCH_SIGNATURE`.

---

## Motivation

### The gap

Gen 3 Showdown sends Hidden Power as `|move|...|Hidden Power|...` — the type suffix is
stripped at the protocol level for both sides. Before this step:

- The **matchup matrix** computed opp HP effectiveness using the `gen3_moves.json`
  Normal-type dummy. Since HP can never be Normal in Gen 3, every cell involving opp HP
  was structurally wrong.
- The **move slot** encoded `type_id=0` (unknown sentinel) for every observed opp HP,
  so the role encoder could not reason about coverage (Ice vs. Dragon, Fire vs. Steel).
- The model saw effectiveness in the rolling N-turn history, but that signal fell off
  the window and required the model to relearn the type chart from RL signal.

### What we can infer

Every time opp uses HP, two facts are determined: our Pokémon's full type + ability,
and the effectiveness tier (immune / resisted / neutral / super-effective) from
`battle.opp_last_effectiveness`. The type chart is deterministic — for each of the 16
possible HP types, we check whether it produces the observed effectiveness against the
hit mon. Types that don't match are zeroed out of the candidate set.

Common convergence patterns:

| HP hits | Remaining candidates |
|---------|---------------------|
| Blissey (Normal) at 2× | Fighting only — certain on turn 1 |
| Lanturn (Water/Electric, Volt Absorb) at 0× | Electric only |
| Weezing (Poison, Levitate) at 0× | Ground only |

Hitting Normal- or ability-immune mons converges in one observation. Multi-typed mons
with overlapping weaknesses may take two or three.

### Why prior probabilities, not binary flags

Candidate entries carry the **prior probability** from competitive usage stats rather
than 1/0 possible-eliminated flags. Jolteon's `ICE: 0.70, GRASS: 0.25` signals the
likely guess immediately; after Ice is eliminated, the remaining `GRASS: 0.25` still
preserves the signal that this is the minority variant. No renormalization happens
during narrowing — magnitudes stay stable so the model learns a single interpretation.

---

## What Changed

### Per-Pokémon slot: 79 → 96 dims; POKEMON_FULL_DIM: 80 → 97

The 17-dim HP block appends after the existing 79 dims:

| Field | Dims | Own slots | Opp slots |
|-------|------|-----------|-----------|
| `hp_revealed` flag | 1 | 1.0 (always — own state is known) | 1.0 once HP has been seen *or* ruled out; 0.0 otherwise |
| HP candidate probs (in `HIDDEN_POWER_TYPE_ORDER`) | 16 | one-hot at the known type; all-zero if no HP move | tracker output (sparse after narrowing; all-zero after ruled out) |

The `(hp_revealed, probs.any())` pair disambiguates four states linearly:

| State | `hp_revealed` | `probs.any()` |
|-------|---------------|---------------|
| Opp HP not yet seen | 0 | 0 |
| Opp HP narrowed via observation | 1 | 1 |
| Opp HP ruled out (4 moves seen, none is HP) | 1 | 0 |
| Own mon with HP | 1 | 1 |
| Own mon without HP | 1 | 0 |

Own-team HP type is recovered through a poke-env patch: `Pokemon._update_from_teambuilder`
now passes `raw_id` so typed HP moves (e.g. `"hiddenpowergrass"`) retain their type and
70 base power instead of collapsing to bare `"hiddenpower"` → Normal / 0bp. The encoder
reads the resolved type from `mon.moves` directly. Opp HP is ruled out by
`HiddenPowerTracker.mark_no_hp(species)`, called once 4 moves have been observed for that
species with no HP among them — this flips `hp_revealed` without populating any probs.

### New and changed constants

| Constant | Before | After | Notes |
|----------|--------|-------|-------|
| `POKEMON_HP_BLOCK_OFFSET` | (new) | `79` | Appends after the 18-dim spread block |
| `POKEMON_HP_REVEALED_OFFSET` | (new) | `79` | `hp_revealed` flag at the first byte |
| `POKEMON_HP_PROBS_OFFSET` | (new) | `80` | 16-dim probability vector |
| `POKEMON_HP_BLOCK_DIM` | (new) | `17` | `1 + 16` |
| `POKEMON_VECTOR_DIM` | `79` | `96` | `79 + 17` |
| `POKEMON_FULL_DIM` | `80` | `97` | `96 + 1` (active flag) |

All top-level offsets derive from `POKEMON_FULL_DIM` and update automatically:

| Constant | Before | After |
|----------|--------|-------|
| `OFFSET_OPP_TEAM` (= 6 × FULL_DIM) | 480 | 582 |
| `OFFSET_CONTEXT` (= 2 × OPP_TEAM) | 960 | 1164 |
| `OFFSET_GLOBAL` (= CONTEXT + 2 × 23) | 1006 | 1210 |
| `OFFSET_REACTIVE` (= GLOBAL + 13) | 1019 | 1223 |

### Observation vector dimensions

12 Pokémon × 17 new dims = 204 additional dims in the base encoder output:

| Block | Before | After | Notes |
|-------|--------|-------|-------|
| Our team (6 × FULL_DIM) | 480 | 582 | +17 dims per slot × 6 |
| Opp team (6 × FULL_DIM) | 480 | 582 | +17 dims per slot × 6 |
| Active context ×2 | 46 | 46 | Unchanged |
| Global env | 13 | 13 | Unchanged |
| Reactive + matchups | 300 | 300 | Unchanged |
| Prev-turn action mask | 11 | 11 | Unchanged |
| Turn history (5 × 39) | 195 | 195 | Unchanged |
| **Base dim** | **1319** | **1523** | |
| **Total** | **1525** | **1729** | |

### Architecture version

`ARCH_SIGNATURE` changed from `"gen3_spread_v1"` to `"gen3_hp_v1"`. The total_dim
mismatch alone would have caught old checkpoints via `check_compatible()`, but the
signature bump makes the failure message explicit ("architecture family mismatch")
rather than a generic dim-list diff.

---

## Implementation Details

### Constants (`src/agents/observation/constants.py`)

```python
POKEMON_HP_BLOCK_OFFSET = 79   # 61 + 18 (spread end)
POKEMON_HP_REVEALED_OFFSET = 79
POKEMON_HP_PROBS_OFFSET = 80
POKEMON_HP_BLOCK_DIM = 17      # 1 hp_revealed flag + 16 candidate-type probs
POKEMON_VECTOR_DIM = 96        # 61 + 18 (spread) + 17 (HP block)
POKEMON_FULL_DIM = 97          # 96 + 1 (active flag appended by state_encoder)
```

### Tracker (`src/agents/training/hidden_power_tracker.py`)

`HiddenPowerTracker` owns `_priors: dict[str, dict[str, float]]` loaded from
`data/pokemon/gen3_hidden_power_priors.json` (computed from Smogon Gen3 OU usage via
`src/scripts/compute_hidden_power_priors.py`). State is a `dict[str, np.ndarray]` keyed
by species: each entry is the live (16,) float32 candidate vector.

```python
HIDDEN_POWER_TYPE_ORDER: list[PokemonType] = [
    PokemonType.BUG, PokemonType.DARK, PokemonType.DRAGON, PokemonType.ELECTRIC,
    PokemonType.FIGHTING, PokemonType.FIRE, PokemonType.FLYING, PokemonType.GHOST,
    PokemonType.GRASS, PokemonType.GROUND, PokemonType.ICE, PokemonType.POISON,
    PokemonType.PSYCHIC, PokemonType.ROCK, PokemonType.STEEL, PokemonType.WATER,
]
```

`observe(species, effectiveness, target_mon)`:

1. On first observation for `species`: build the (16,) prior vector from
   `_priors[species]` (flat 1/16 if absent from the file).
2. For each surviving type `t`, if `effective_multiplier(t, target_mon) != effectiveness`,
   zero `state[i]`.
3. If the result is all-zero, raise `ValueError`. The message distinguishes "species
   had a prior entry → tracker bug" from "species has no prior entry → data gap, add
   it to `gen3_hidden_power_priors.json`". This is a hard failure rather than a silent
   reset because either case means the upstream logic is wrong and should be fixed.

`get_probs(species)` returns the live vector (or a fresh zero array if HP has not
been observed for that species this episode). `reset()` clears all state.

### Resolving the actual HP target (`src/agents/training/episode_tracker.py`)

The model only sees correct candidate elimination if `observe()` is called with the
mon that was on our field *at the moment HP resolved* — not necessarily
`prev.our_active`. Gen 3 simultaneous resolution makes this nontrivial:

| Scenario | Target |
|----------|--------|
| No side change | `prev.our_active` stayed on the field |
| Voluntary switch (last_action 0..5) | Switch resolves at priority +6 → switch-in is the target |
| Baton Pass + we moved first | BP fires first → switch-in is the target |
| Baton Pass + opp moved first | HP hits `prev.our_active` before BP fires |
| `prev.our_active` newly fainted | HP killed them on the field; they were the target |
| Voluntary switch + switch-in fainted | Switch-in is the target; identified from
                                          `newly_fainted - {prev.our_active}` |
| Voluntary switch + switch-in fainted + forced replace cycled the same mon back | Same as above |

`_resolve_hp_target(battle, prev, curr, delta: TurnDelta)` consumes the structured
`TurnDelta` rather than re-decoding `last_action`. The action fields (`our_switch_to`,
`our_move_id`) already encode "did we switch" / "was it Baton Pass" — duplicating that
logic would put two sources of truth in the codebase.

```python
voluntary_switch = delta.our_switch_to is not None
baton_pass = delta.our_move_id == BATON_PASS and prev.our_active != curr.our_active

if voluntary_switch:
    switch_first = True
elif baton_pass:
    switch_first = delta.we_moved_first is True
else:
    switch_first = False
```

The resolver returns an `_HpTargetMon` dataclass (not the live `Pokemon`) with status
overridden — see "Historical status" below.

### Where the tracker plugs into `record()`

`EpisodeTracker.record()` builds the new context, then immediately offers an HP
observation via `_maybe_observe_hidden_power()`:

```python
def _maybe_observe_hidden_power(self, battle, ctx: BattleContext) -> None:
    if not self._history or ctx.phase != "move_selection":
        return
    if ctx.opp_last_move_id != "hiddenpower" or ctx.opp_last_effectiveness is None:
        return
    prev = self._history[-1]
    if prev.our_active == "NONE" or prev.opp_active == "NONE":
        return

    delta = TurnDelta.build(prev, ctx, self._last_action)
    target = _resolve_hp_target(battle, prev, ctx, delta)
    if target is not None:
        self._hidden_power_tracker.observe(
            prev.opp_active, ctx.opp_last_effectiveness, target
        )
```

The `phase == "move_selection"` gate is critical: a forced-switch context can carry a
stale `opp_last_effectiveness` from the prior turn (when end-of-turn damage triggers
the switch after `battle.turn` has already ticked), and re-observing there would
double-count with now-misaligned `prev_*` state. Each HP event is observed exactly
once, at the next move_selection context. The poke-env property
`opp_last_effectiveness` gates on `turn_set == self._turn - 1`, which together with
the phase gate makes it impossible to lose an event or double-count one.

### Order swap: record-before-encode

The model wants to see HP narrowing *from* the just-fired HP, not *after* the next
HP. That means the tracker must be updated **before** the env encodes the obs.
`gen3_env.embed_battle` and `Gen3Player.embed_battle` were both reordered:

```python
def embed_battle(self, battle):
    if battle is self.battle1 and not battle.finished:
        mask = Gen3ActionMasker.get_mask(battle).astype(np.int8)
        if mask.sum() > 0:
            self._tracker.record(battle, mask)

    if battle is self.battle1:
        obs = self.observation_encoder.encode(
            battle, hp_tracker=self._tracker.hidden_power_tracker
        )
```

The `obs` field on `BattleContext` was vestigial (set in `from_battle()`, read by
nobody) and blocked the reorder by forcing `record()` to take the encoded obs as an
input. It was removed; `record(battle, mask)` is now the full signature.

### Ability quirks: unified table + Flash Fire-frozen (`src/agents/gen3_mechanics.py`)

All Gen 3 abilities that modify type-effectiveness now live in one dict:

```python
ABILITY_TYPE_MULTIPLIER: dict[str, dict[PokemonType, float]] = {
    "levitate":    {PokemonType.GROUND:   0.0},
    "voltabsorb":  {PokemonType.ELECTRIC: 0.0},
    "waterabsorb": {PokemonType.WATER:    0.0},
    "flashfire":   {PokemonType.FIRE:     0.0},
    "thickfat":    {PokemonType.ICE:      0.5, PokemonType.FIRE: 0.5},
}
```

`effective_multiplier(move_type, mon)` returns
`type_chart * ABILITY_TYPE_MULTIPLIER.get(ability, {}).get(move_type, 1.0)`, with one
exception inline-conditioned:

```python
if ability == "flashfire" and getattr(mon, "status", None) == Status.FRZ:
    return base   # Flash Fire is suppressed while the holder is frozen in Gen 3
```

The frozen branch comes from `pokemon-showdown/data/mods/gen3/abilities.ts` — a frozen
Flash Fire mon does **not** get the immunity and the Fire move thaws it. Without this
quirk, the tracker's calculated multiplier (0×) doesn't match Showdown's reported
multiplier (0.5×) and Fire is wrongly eliminated.

The previous two-dict layout (`ABILITY_TYPE_IMMUNITY` and `ABILITY_DAMAGE_HALVE`) was
collapsed into the single table above — they shared one purpose and a unified dict
makes the contract explicit.

### Bucketed effectiveness comparison

poke-env stores `opp_last_effectiveness` in Showdown's reported `{0.0, 0.5, 1.0, 2.0}`
buckets — the protocol emits `|-immune|`, `|-resisted|`, neutral, `|-supereffective|`
without exposing raw multipliers. A 4× HP Grass on Swampert (Water/Ground) gets reported
as 2.0; a 0.25× HP Grass on Skarmory (Steel/Flying) gets reported as 0.5. The tracker
was comparing raw `effective_multiplier()` output (4.0, 0.25) against the bucketed
report and wrongly zeroed the correct candidate when no type matched the unbucketed
multiplier exactly.

`bucket_effectiveness(mult)` in `gen3_mechanics` collapses any raw multiplier into the
same four buckets:

```python
def bucket_effectiveness(mult: float) -> float:
    if mult == 0.0:   return 0.0
    if mult < 1.0:    return 0.5
    if mult == 1.0:   return 1.0
    return 2.0
```

`HiddenPowerTracker.observe()` pipes the calculated multiplier through this helper
before comparing. The fuzz validator mirrors the same bucketing in its parallel
implementation.

### Ruling out HP via `mark_no_hp()`

`HiddenPowerTracker.mark_no_hp(species)` is called by `state_encoder` once the opp
has revealed 4 moves for that species and none of them is Hidden Power. The tracker
sets a per-species `_ruled_out` flag and `get_probs()` returns an all-zero vector
thereafter; `is_known(species)` returns True even though the probs are empty. This
is what powers the "HP ruled out" state in the per-mon block (hp_revealed=1,
probs all-zero) — a definitive signal that the model can use to commit team-building
threat assessments without leaving a stale prior on the table.

### Historical status snapshot

Fire moves *thaw* their target. A frozen Arcanine hit by HP Fire takes resisted damage
(0.5×), then immediately loses the freeze. Reading `mon.status` at the next
move_selection sees `None` (no longer frozen) and wrongly computes a 0× immunity —
the Fire candidate then gets eliminated against the 0.5× observation.

`BattleContext.our_team_status: dict[str, Status | None]` snapshots every mon's status
at the start of each turn. `_resolve_hp_target` builds an `_HpTargetMon` with status
overridden to that historical value:

```python
@dataclass(frozen=True)
class _HpTargetMon:
    species: str
    type_1: "PokemonType"
    type_2: "PokemonType | None"
    ability: "str | None"
    status: "Status | None"

# at resolution time:
return _HpTargetMon(
    species=live_mon.species,
    type_1=live_mon.type_1,
    type_2=live_mon.type_2,
    ability=live_mon.ability,
    status=prev.our_team_status.get(species, live_mon.status),
)
```

The dataclass is the explicit contract — every attribute `effective_multiplier()` reads
is present, and the frozen wrapper prevents accidental mutation. The tracker treats it
as a duck-typed `Pokemon`.

### Encoder wiring (`src/agents/observation/state_encoder.py`)

`encode()` now accepts an optional `hp_tracker`. For each opponent slot it calls
`hp_tracker.get_probs(mon.species)` and threads the (16,) array into
`PokemonEncoder.encode(..., hp_probs=...)`:

```python
def encode(self, battle, hp_tracker=None):
    ...
    for i in range(TEAM_SIZE):
        mon = opponents[i] if i < len(opponents) else None
        hp_probs = (
            hp_tracker.get_probs(mon.species)
            if (hp_tracker is not None and mon is not None)
            else None
        )
        mon_vec = self.pokemon_encoder.encode(mon, battle, is_own=False, hp_probs=hp_probs)
```

The encoder is layout-only — it does not own the tracker, has no knowledge of priors,
and treats `None` as "no data, leave block at zero".

### Per-mon encoding (`src/agents/observation/pokemon.py`)

```python
def encode(self, mon, battle, is_own=False, hp_probs=None, hp_known=False) -> np.ndarray:
    ...
    if is_own:
        vec[POKEMON_HP_REVEALED_OFFSET] = 1.0
        idx = _own_hp_type_index(mon)        # scans mon.moves for "hiddenpower<type>"
        if idx is not None:
            vec[POKEMON_HP_PROBS_OFFSET + idx] = 1.0
    elif hp_known:
        vec[POKEMON_HP_REVEALED_OFFSET] = 1.0
        if hp_probs is not None:
            vec[POKEMON_HP_PROBS_OFFSET : POKEMON_HP_PROBS_OFFSET + 16] = hp_probs
```

Own slots derive the type index from the move object — the poke-env raw_id patch
preserves the typed variant so `move.type` returns the correct `PokemonType`. Opp slots
flow through the explicit `hp_known` gate so we can distinguish "narrowed to sparse
probs" from "ruled out" (both have `hp_revealed=1` but only narrowed has non-zero probs).
`state_encoder` provides `hp_known` from `tracker.is_known(species)`.

### Smogon priors data

`tools/smogon_stats_downloader/sync.py` fetches the Smogon Gen3 OU chaos JSON to
`data/pokemon/gen3_smogon_stats.json`. `src/scripts/compute_hidden_power_priors.py`
extracts `hiddenpower*` move keys per species, normalises by total HP usage, and writes
`data/pokemon/gen3_hidden_power_priors.json` keyed by lowercase species name. 173
species, ~608k observations.

### Feeding HP into the move processor

The 17-dim per-mon HP block puts the candidate distribution in the **role encoder**'s
input, but every per-move feature (type embedding, type matchups) still treated bare
`"hiddenpower"` as a typeless move. Two paths were rewired to consume the same
distribution:

**1. Weighted type embedding (`features_extractor.py`).** For move slots whose move-id
equals `HIDDEN_POWER_MOVE_NUM = 237` (all 16 typed variants and the bare form already
share this num in `gen3_moves.json`), the move's type embedding is replaced with the
probability-weighted blend of the 16 candidate-type embeddings:

```python
hp_probs = pokemon_part[:, :, POKEMON_HP_PROBS_OFFSET : POKEMON_HP_PROBS_OFFSET + 16]
hp_type_rows = self.type_embedding(self.hp_type_idx_map)     # [16, type_emb_dim]
soft_type_emb_per_pk = hp_probs @ hp_type_rows                # [B, 12, type_emb_dim]
is_hp_slot = (all_move_ids == HIDDEN_POWER_MOVE_NUM)
embedded_move_types = torch.where(
    is_hp_slot.unsqueeze(-1),
    soft_type_emb_per_pk.unsqueeze(2).expand(-1, -1, num_moves, -1),
    embedded_move_types,
)
```

`hp_type_idx_map` is a (16,) buffer built at init that maps each
`HIDDEN_POWER_TYPE_ORDER` index to its row in `type_embedding`. For a one-hot p (own
team), the weighted sum collapses to `type_embedding(known_type)` — algebraically
identical to the pre-change path, so own behaviour is unchanged.

**2. Expected matchup (`reactive.py`).** The 144-dim `our_matchups` / `their_matchups`
blocks compute `effective_multiplier(move.type, opp)` per cell. For HP move slots the
default `move.type` is Normal (poke-env's fallback for bare `"hiddenpower"`), so every
opp-HP matchup cell was structurally wrong. The new `_hp_expected_multiplier()` helper
takes the tracker's per-species distribution and returns the probability-weighted
effectiveness:

```python
def _hp_expected_multiplier(move, attacker, opp, hp_tracker) -> float:
    if move.id == "hiddenpower" and hp_tracker is not None and attacker is not None:
        probs = hp_tracker.get_probs(attacker.species)
        if probs is not None and probs.sum() > 0:
            return float(sum(
                probs[i] * effective_multiplier(HIDDEN_POWER_TYPE_ORDER[i], opp)
                for i in range(16) if probs[i] > 0
            ))
    return effective_multiplier(move.type, opp)
```

Typed HP variants (own team) fall through to the plain `effective_multiplier(move.type)`
path — equivalent to the one-hot weighted sum, so own matchup cells are unchanged.
`ReactiveEncoder.encode()` now takes an optional `hp_tracker` parameter, threaded from
`state_encoder.encode()` alongside the existing per-mon tracker plumbing.

**3. HP probs as a raw feature in `move_network`.** The expected matchup is a
projection of the distribution onto effectiveness against currently-known opps. It
loses two signals:

- **Uncertainty/variance.** A one-hot Ice HP and a `[0.5 Ice, 0.5 Fire]` HP can have
  the same expected matchup against your team but very different risk profiles.
- **Threat against future-revealed opps.** The 144-dim matchup matrix only covers
  currently-revealed mons.

The 16-dim distribution is broadcast per Pokémon to its 4 move slots and zeroed for
non-HP slots (`torch.where` on `is_hp_slot`), then appended to the `move_network` input
alongside the existing 6-dim matchup scalars.

### Matchup validity: unknown vs immune

The 144-dim matchup matrices initialise to zero and only overwrite cells where both
`move` and `their_mon` exist. This conflated four scenarios into the same cell value:

- Move slot empty (no move at that index)
- Opp slot unrevealed
- Genuine 0× immunity (Ghost-vs-Normal, Levitate-vs-Ground, etc.)
- Bucketed 0× from `bucket_effectiveness`

Per-cell validity bits — `move_known AND species_known(opp)` — are computed inside the
feature extractor from the existing observation fields (`POKEMON_SPECIES_KNOWN_OFFSET`
and the move slot's `known` flag) and appended to `move_network` input as 6 dims per
move slot (one per opponent). With this, `matchup=0, valid=1` means "real immune wall"
and `matchup=0, valid=0` means "no information." No observation-dim change needed; the
extractor derives the validity tensor from already-present scalars.

### Move processor capacity bump

The two new feature blocks (16-dim hp_probs + 6-dim matchup validity) grow the
`move_network` input from 58 to 80 dims. To keep the expansion ratio sensible the
hidden layer was bumped:

| Constant | Before | After |
|----------|--------|-------|
| `move_network` input dim | 58 | 80 |
| `MOVE_NET_HIDDEN[0]` (hidden layer) | 64 | 96 |
| `MOVE_NET_HIDDEN[1]` (output dim) | 32 | 32 |

Old checkpoints fail `check_compatible()` on the `move_network` Linear shapes — expected
under the project's rapid-iteration stance, and surfaces as a clean error at startup
via the model-version round-trip smoke test.

### Independent fuzz validator (`hidden_power_tracker_fuzz_e2e_test.py`)

The fuzz test reimplements HP-target resolution from raw battle state with a parallel
`_HpTargetMon` and parallel resolution logic. **This duplication is deliberate** — if
the test shared `_resolve_hp_target` with the production code, a bug in either piece
would slip through. The fuzz drives 500–1000 real Showdown battles against an
opponent that biases toward HP, and asserts on two layers:

1. **Invariant** — after every `observe()` call, every surviving candidate type
   satisfies `effective_multiplier(type, target) == observed_effectiveness`.
2. **Ground truth** — at battle end, for every opp species that used HP, the true HP
   type (known from the fixed team) must still be a non-zero candidate.

The test teams exercise the interesting edge cases:

| Test team mechanic | Why |
|---|---|
| Lanturn (Volt Absorb), Vaporeon (Water Absorb), Weezing (Levitate), Arcanine (Flash Fire) | All four ability immunities |
| Snorlax (Thick Fat) | The damage-halving ability path |
| Arcanine + opp Ice Beam | Triggers the Flash Fire-vs-frozen quirk in real play |
| Vaporeon Baton Pass | Move-action side change |
| Zapdos Roar | Opp phazing (which doesn't co-occur with HP; verifies we don't mis-attribute) |
| Raikou Rest + Sleep Talk + HP Grass | Sleep Talk → HP delegation |
| Skarmory hit by HP Grass (0.25×) | Exercises the `bucket_effectiveness` 0.25 → 0.5 collapse |
| Salamence / Swampert / Forretress hit by 4× HP | Exercises the 4× → bucketed-2.0 path |

The fuzz now cross-rotates 3 our-team variants × 2 opp-team variants, sampled per
battle by `Gen3Teambuilder`. At 3000 battles: 27531 observations across 8 opp species,
100% invariant pass, zero ground-truth failures.

The fuzz validator also caught a self-bug: its `_resolve_target` Case A returned the
live `Pokemon` directly instead of wrapping it through the historical status snapshot,
so the Flash Fire-vs-frozen evaluation read post-faint `FNT` status and wrongly
eliminated Fire. Both return paths were refactored through a shared
`_snapshot_target()` helper that mirrors the production code.

---

## Edge Cases

### Switch-in survives → identified by `curr.our_active`

Voluntary switch with no faint: `curr.our_active` is the switch-in.
`newly_fainted - {prev.our_active}` is empty, so the resolver returns `curr.our_active`.

### Switch-in fainted → identified by `newly_fainted - {prev.our_active}`

If the switch-in died from HP, `curr.our_active` is a forced replacement — not the
target. `prev.our_active` is alive in the back (switched out safely), so subtracting it
from `newly_fainted` leaves exactly the dead switch-in as the singleton.

### Cycled-back-to-same-species

Vaporeon → Lanturn → (Lanturn KO'd by HP) → Vaporeon. `prev.our_active == curr.our_active
== "vaporeon"`, but `prev_was_switch == True` and `newly_fainted == {"lanturn"}`. The
resolver returns Lanturn, not Vaporeon — driven by what *we* did, not visible side state.

### `prev.our_active` newly fainted

If our mon fainted from the HP itself (no switch), `prev.our_active` is in
`newly_fainted`. The resolver returns them immediately — they were on the field when
HP hit, and they died from it.

### Baton Pass with opp moving first

HP fires first, hits the BP user (`prev.our_active`). BP fires next if the user
survives. `delta.we_moved_first is False` flips `switch_first` off, so the resolver
returns `prev.our_active`.

### Forced switch from end-of-turn damage

A mid-turn faint from a non-HP source (toxic damage at end-of-turn) triggers
`phase="forced_switch"`. The resolver is skipped at that context — the previous
move_selection already observed any HP from that turn, and observing again with the
forced-switch ctx would double-count.

### Fire move thaws the Flash Fire target

The historical status snapshot in `prev.our_team_status` returns the frozen status,
so `effective_multiplier(FIRE, frozen_arcanine)` returns `base` (0.5×). The Fire
candidate survives the observation. Without the snapshot, the live mon's `status` is
already `None` and the multiplier wrongly returns 0×.

### Multiple newly-fainted mons (impossible in Gen 3 singles, defensive only)

If `newly_fainted - {prev.our_active}` has more than one element, the resolver raises
`RuntimeError`. This cannot happen in Gen 3 singles because only one mon takes a
single-target move per turn, but the explicit error is safer than picking arbitrarily.

---

## Test Suite

### `HiddenPowerTracker` unit tests (`hidden_power_tracker_test.py`, 14 tests)

| Test | What it validates |
|------|-------------------|
| `test_blissey_2x_leaves_fighting_only` | Pure Normal at 2× → only Fighting survives |
| `test_volt_absorb_lanturn_0x_leaves_electric_only` | Ability immunity isolates Electric |
| `test_levitate_weezing_0x_leaves_ground_only` | Levitate isolates Ground |
| `test_thick_fat_snorlax_05x_leaves_ice_and_fire` | Thick Fat damage-halve path |
| `test_flash_fire_frozen_arcanine_05x_leaves_fire` | Frozen Flash Fire falls through |
| `test_flash_fire_healthy_arcanine_0x_leaves_fire_only` | Healthy Flash Fire 0× immunity |
| `test_prior_weights_preserved_for_surviving_types` | Prior probabilities, not 1/0 flags |
| `test_species_not_in_priors_uses_flat_prior` | Unknown species → flat 1/16 start |
| `test_all_zero_with_prior_entry_raises` | Tracker-bug ValueError path |
| `test_all_zero_without_prior_entry_raises_data_gap` | Data-gap ValueError path |
| `test_idempotent` | Same observation twice == single observation |
| `test_get_probs_before_observation_returns_zeros` | Pre-observation state |
| `test_reset_clears_state` | `reset()` semantics |
| `test_multiple_observations_narrow_further` | Chained narrowing across mons |

### Encoder dimension tests

| Test file | What changed |
|-----------|--------------|
| `pokemon_test.py::test_pokemon_encoder_dimension` | `POKEMON_VECTOR_DIM` 79 → 96 |
| `pokemon_test.py::test_pokemon_encoder_empty` | `POKEMON_VECTOR_DIM` 79 → 96 |
| `state_encoder_test.py` | `EXPECTED_BASE_DIM` 1319 → 1523; `EXPECTED_OBS_DIM` 1525 → 1729 |

### `BattleContext` / `EpisodeTracker` tests

The `obs` field removal cascaded into ~80 tests that constructed `BattleContext`
directly or called `record(battle, mask, obs)`. All were updated to the new signature.

### Move-processor wiring tests

| Test file | What it validates |
|-----------|-------------------|
| `reactive_test.py::test_one_hot_distribution_collapses_to_exact_effectiveness` | One-hot p reproduces exact effectiveness (own-team regression guard) |
| `reactive_test.py::test_two_type_distribution_gives_expected_value` | `[0.5 Ice, 0.5 Grass]` HP vs Salamence = `0.5*4 + 0.5*0.25 = 2.125` |
| `reactive_test.py::test_uniform_distribution_equals_mean_over_all_16_types` | Uniform 1/16 = mean of all 16 type effectivenesses (analytic ground truth) |
| `reactive_test.py::test_typed_hp_variant_uses_move_type_directly` | Own-team typed `"hiddenpowerice"` falls through to `effective_multiplier(move.type)` |
| `reactive_test.py::test_no_tracker_falls_back_to_move_type` | No-tracker path preserves legacy behaviour for inference outside training |
| `reactive_test.py::test_non_hp_move_unchanged` | Non-HP moves ignore the tracker entirely |
| `reactive_test.py::test_levitate_blocks_ground_hp` | One-hot Ground HP vs Levitate = literal 0× (real immunity, not unknown) |
| `features_extractor_hp_test.py::test_hp_type_idx_map_matches_type_encoder` | The (16,) buffer maps HP slot i → row in `type_embedding` for that type |
| `features_extractor_hp_test.py::test_one_hot_hp_distribution_collapses_to_direct_lookup` | `p @ type_embedding[hp_type_idx_map] == type_embedding(known_idx)` for one-hot p |
| `features_extractor_hp_test.py::test_hp_slot_detection_only_fires_on_canonical_move_num` | `is_hp_slot` mask triggers only on `move_num == HIDDEN_POWER_MOVE_NUM` |
| `features_extractor_hp_test.py::test_matchup_validity_distinguishes_unknown_from_immune` | Unrevealed opp (validity 0) vs revealed-with-zero-matchup (validity 1) produce different forward outputs |
| `features_extractor_hp_test.py::test_forward_runs_with_hp_probs_in_obs` | Smoke: populated hp_probs in obs produces finite forward output |

### Fuzz E2E (`hidden_power_tracker_fuzz_e2e_test.py`)

Runs N battles (default 500) against `HiddenPowerSpammer` (always picks HP when
available). Asserts the per-observation invariant and the end-of-battle ground truth
for each species in `OPP_HP_GROUND_TRUTH`. Diagnostic dumps the per-species observation
log, the last 15 turn entries, and the raw protocol around the buggy turn(s) when any
assertion fails. See "Independent fuzz validator" above for the assertion details.

```bash
export PYTHONPATH=$PYTHONPATH:src
python src/agents/training/hidden_power_tracker_fuzz_e2e_test.py 500
```

Requires a live Showdown server (`npm run showdown`).

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/gen3_mechanics.py` | Unified `ABILITY_TYPE_MULTIPLIER`; Flash Fire-vs-frozen branch in `effective_multiplier`; `Status` import from poke_env |
| `src/agents/gen3_mechanics_test.py` | Import `ABILITY_TYPE_MULTIPLIER` (was `ABILITY_TYPE_IMMUNITY`) |
| `src/agents/training/hidden_power_tracker.py` | **New** — tracker, `HIDDEN_POWER_TYPE_ORDER` |
| `src/agents/training/hidden_power_tracker_test.py` | **New** — 14 unit tests |
| `src/agents/training/hidden_power_tracker_fuzz_e2e_test.py` | **New** — independent fuzz validator (500-battle target) |
| `src/agents/training/battle_context.py` | Drop vestigial `obs` field; add `our_team_status`, `our_fainted_species`, `opp_fainted_species` |
| `src/agents/training/episode_tracker.py` | `_HpTargetMon` dataclass, `_resolve_hp_target(... delta)`, `_maybe_observe_hidden_power`, `BATON_PASS` constant; `record(battle, mask)` (drop obs) |
| `src/agents/training/gen3_env.py` | `record()` before `encode()`; pass `hp_tracker=` to encoder |
| `src/agents/training/battle_recorder.py` | Drop `obs` arg in `_build_ctx` |
| `src/agents/training/reward_tracker.py` | Drop `obs` arg in two `BattleContext.from_battle` calls |
| `src/agents/inference/player.py` | Record before encode; pass `hp_tracker=` to encoder; return obs dict directly |
| `src/agents/observation/constants.py` | `POKEMON_HP_BLOCK_*` offsets; `POKEMON_VECTOR_DIM` 79→96; `POKEMON_FULL_DIM` 80→97 |
| `src/agents/observation/pokemon.py` | `hp_probs` param to `encode()`; HP block write; `get_layout()` entry; docstring updated |
| `src/agents/observation/state_encoder.py` | `hp_tracker` param to `encode()`; per-opp-slot tracker lookup |
| `data/pokemon/gen3_hidden_power_priors.json` | **New (generated)** — 173 species' HP usage distributions |
| `data/pokemon/gen3_smogon_stats.json` | **New (generated)** — raw Smogon Gen3 OU chaos JSON |
| `src/scripts/compute_hidden_power_priors.py` | **New** — normalises the chaos JSON into the priors file |
| `tools/smogon_stats_downloader/sync.py` | **New** — fetches the Smogon chaos JSON |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE` `"gen3_spread_v1"` → `"gen3_hp_v1"` |
| `src/agents/observation/pokemon_test.py` | `POKEMON_VECTOR_DIM` 79→96 in dim assertions |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_BASE_DIM` 1319→1523; `EXPECTED_OBS_DIM` 1525→1729 |
| `src/agents/training/battle_context_test.py` | Drop `obs=` from `_ctx` helper and all `from_battle()` calls |
| `src/agents/training/episode_tracker_test.py` | Drop `obs=` / extra `np.zeros(10)` from all `record()` calls |
| `src/agents/training/reward_manager_test.py` | Drop `obs=` from `_ctx` helper |
| `src/agents/training/reward_invariants_e2e_test.py` | Drop `obs=` from `from_battle()` |
| `src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py` | Drop `obs=` from `from_battle()` |
| `src/agents/inference/player_test.py` | Stub `encode()` + `Gen3ActionMasker.get_mask` via `_stubbed_encode` context manager |
| `src/poke_env/battle/pokemon.py` | `_update_from_teambuilder` passes `raw_id` so typed HP moves keep their type/basePower |
| `src/poke_env/battle/hidden_power_raw_id_test.py` | **New** — validates the raw_id patch round-trips HP type through the move object |
| `src/agents/training/hidden_power_tracker.py` (later commits) | `mark_no_hp(species)` ruled-out path; per-species observation log dumped on the impossible-observation `ValueError` |
| `src/agents/gen3_mechanics.py` (later commits) | `bucket_effectiveness(mult)` helper collapsing raw multipliers into the {0.0, 0.5, 1.0, 2.0} reported buckets |
| `src/agents/observation/moves.py` | `HIDDEN_POWER_MOVE_NUM = 237` module constant (canonical HP detector) |
| `src/agents/observation/reactive.py` | `_hp_expected_multiplier()` helper; `ReactiveEncoder.encode()` takes `hp_tracker=`; applied to both 4-dim active-move and 144+144-dim matchup paths |
| `src/agents/observation/reactive_test.py` | **New** — 7 unit tests for the expected-effectiveness helper |
| `src/agents/model/features_extractor.py` | `MOVE_NET_HIDDEN` 64→96; `hp_type_idx_map` buffer; weighted type embedding for HP slots; appended `hp_probs_per_slot` (16) + `matchup_validity` (6) to `move_network` input (58→80) |
| `src/agents/model/features_extractor_hp_test.py` | **New** — 5 unit tests for the weighted embedding, validity mask, and forward smoke |
