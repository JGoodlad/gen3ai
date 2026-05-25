# Implementation: Step 4 — Team Completion Enrichment

Extends the Step 3 team completion model with richer training data and additional
predicted fields, before it is wired into MCTS in Step 5.

**Step 3 status at entry:** pipeline running on 18,869 ladder replay records.
Species top-1 target: ≥ 25%. Moves are multi-hot BCE; items supervised where revealed.
Abilities and Hidden Power type are not yet predicted.

---

## Motivation

Three gaps remain after Step 3:

**1. Partial move coverage.** Replays reveal only the moves actually used in a game — on
average 1.82 moves per Pokémon. The model is trained against noisy labels (unrevealed
moves are treated as 0, but many are actually in the moveset). The 39 curated teams in
`data/teams/` have all 4 moves fully specified; mixing them in eliminates label noise for
those records, improving the move head.

**2. No ability prediction.** Gen 3 abilities are never revealed by the Showdown protocol
(no `|-ability|` lines in Gen 3 unlike later generations). They must come from curated
teams, which specify abilities explicitly. Knowing the opponent's ability matters for MCTS
rollouts: Levitate blocks Ground moves, Intimidate drops Attack on switch-in, Arena Trap
prevents escape.

**3. Hidden Power type unknown.** `Hidden Power` appears as a single move key in replays
regardless of type. In curated teams the type is explicit (`Hidden Power [Ice]`). The type
is tactically significant — HP Ice is standard on Zapdos variants; HP Fire targets steel
types. The model should distinguish them.

---

## Data Sources after this Step

| Source | Records | Completeness |
|--------|---------|--------------|
| Ladder replays (`data/replay_teams.jsonl`) | 37,940 | Species always known; moves/items partial; ability/HP type unknown |
| Curated teams (`data/teams/sample/`, `data/teams/others/`) | ~78 (39 teams × 2 players equiv) | All fields complete: 4 moves, item, ability, HP type |

Curated team records are weighted up during loss computation to compensate for their small
share of the total data. Because they have complete ground truth, every loss term fires on
every slot — no `item_known` or ability masks needed.

---

## Schema Changes

### `PokemonRecord` additions

```python
@dataclass
class PokemonRecord:
    species: str
    moves: list[str]          # normalized move keys, e.g. "thunderbolt", "hiddenpower"
    item: Optional[str]
    ability: Optional[str]    # NEW: normalized ability key, e.g. "naturalcure"; None if unknown
    hp_type: Optional[str]    # NEW: Hidden Power type key, e.g. "ice"; None if HP not in moveset
                              #      or type not known (replay-only records)
```

`hp_type` is non-None only when:
1. `"hiddenpower"` is in `moves`, AND
2. The type is known (parsed from a curated team's `Hidden Power [Type]` annotation)

Replay records always have `hp_type = None` even when Hidden Power was used, because the
Showdown protocol does not emit the type in Gen 3.

### Normalization

`Hidden Power [Ice]` in Showdown text normalizes to `move="hiddenpower"` + `hp_type="ice"`.
The move key stays `hiddenpower` in all cases — the dataset does not create separate
`hiddenpowerice`, `hiddenpowerfire` etc. move IDs, since the move embedding should be
shared. Type is a separate supervised field.

---

## Curated Team Parser

New file: `src/agents/training/team_completion/team_parser.py`

Parses Showdown text format (e.g. `data/teams/sample/023a2d47648b85e6.txt`) into
`TeamRecord` objects compatible with the replay JSONL schema.

Key parsing rules:
- Species line: `Name (M/F) @ Item` or `Name @ Item` or bare `Name`
- Ability line: `Ability: AbilityName`
- Move lines: `- MoveName` (up to 4); `- Hidden Power [Type]` → `move="hiddenpower"` + `hp_type="type"`
- EVs / IVs / Nature: **ignored** — not predicted by the model
- Winner: `False` for all (curated teams are not battle records; winner field is unused in the loss)

The existing `TeamLoader` in `src/utils/team_loader/loader.py` already discovers and reads
the raw text files. The new parser wraps it to produce `list[TeamRecord]`.

Integration with `build_datasets()`: add a `curated_teams_dir` argument. When provided,
curated team records are appended to the training split (never val — they're too few to
split meaningfully). Each curated team record gets a `is_curated=True` tag so the dataset
can up-weight it.

---

## Model Changes

### New head: ability

```python
self.ability_head = nn.Linear(ROLE_TOKEN_DIM, num_abilities)  # cross-entropy
```

Add the ability embedding to the frozen backbone set if present in the checkpoint:
```python
self.ability_emb = nn.Embedding(num_abilities, 16)   # frozen if backbone has it
```

Loss term:
```python
+ CE(ability_logits, target_ability) × ability_known_mask   # only where ability is known
```

`ability_known_mask` is `True` only for curated team records. Replay records never
supervise this head.

### New head: Hidden Power type

```python
HP_TYPES = 16   # all possible HP types in Gen 3
self.hp_type_head = nn.Linear(ROLE_TOKEN_DIM, HP_TYPES)   # cross-entropy
```

Loss term:
```python
+ CE(hp_type_logits, target_hp_type) × hp_known_mask   # only where HP is in moveset AND type known
```

`hp_known_mask` is `True` for curated records where `hp_type is not None` only.

At inference, apply `hp_type_head` only at masked slots where the model predicts
`hiddenpower` in the top-4 moves.

### Updated loss

```
L = CE(species_logits, target_species)
  + BCE(move_logits, target_move_multihot)
  + CE(item_logits, target_item)    × item_known_mask
  + CE(ability_logits, target_ability) × ability_known_mask
  + CE(hp_type_logits, target_hp_type) × hp_known_mask
```

### `model_config.json` additions

```json
{
  "arch_signature": "team_completion_v2",
  "num_abilities": 76,
  "hp_types": 16
}
```

Bump `ARCH_SIGNATURE` to `"team_completion_v2"` so Step 3 checkpoints cannot be
accidentally loaded into a Step 4 model.

---

## Evaluation Metrics (additions)

| Metric | Description |
|--------|-------------|
| `ability_top1` | Top-1 accuracy on masked slots where ability is known (curated records only) |
| `hp_type_top1` | Top-1 accuracy on masked HP slots where type is known |

Both are sparse (only fire when curated records appear in the val batch), so log them only
when `ability_total > 0` / `hp_total > 0` to avoid misleading 0% readings.

---

## Training

No new CLI flags required. `build_datasets()` detects the curated dir automatically when
called with `curated_teams_dir=os.path.join(repo_root, "data", "teams")`.

Because curated records have full labels (especially all 4 moves), even a small weight
boost of 5–10× per record provides a meaningful correction to the move BCE label noise.

Recommended run:
```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.train_team_completion \
  --backbone models/<run>/checkpoint_<N>_steps.zip \
  --data data/replay_teams.jsonl \
  --epochs 300 --batch-size 512 --lr 1e-3
```

(Curated teams are loaded automatically alongside the replay JSONL.)

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `src/agents/training/team_completion/team_parser.py` | **New**: Showdown text → `TeamRecord`; wraps `TeamLoader` |
| `src/agents/training/team_completion/replay_parser.py` | Add `ability: Optional[str]` and `hp_type: Optional[str]` to `PokemonRecord` |
| `src/agents/training/team_completion/team_dataset.py` | Add `curated_teams_dir` support to `build_datasets()`; ability/hp_type tensors; curated weight |
| `src/agents/model/team_completion_model.py` | Add `ability_head`, `hp_type_head`; bump `ARCH_SIGNATURE` to `v2`; update loss |
| `src/main/train_team_completion.py` | Pass `curated_teams_dir`; log new eval metrics |

---

## Verification

1. **Parser sanity** — run `team_parser.py` on the 39 curated files; confirm all 6 Pokémon
   per team are parsed with 4 moves, item, ability, and `hp_type` where applicable.
   Spot-check: Zapdos should have `hp_type="ice"` or `hp_type="grass"` depending on the team.

2. **Dataset mixing** — print dataset summary: `n_replay`, `n_curated`, `frac_with_ability`,
   `frac_with_hp_type`. Expect ability coverage ~100% of curated records, ~0% of replay records.

3. **Ability sanity** — after convergence, given a masked Blissey slot with no context,
   top-1 ability prediction should be `naturalcure`. Given Gengar, `levitate`.

4. **HP type sanity** — given a masked Zapdos where the model predicts Hidden Power is in
   the moveset, top-2 HP types should include `ice`. Given Salamence, `fire`.

---

## Final State

Step 4 is complete when:
- Curated teams are parsed and mixed into training without errors
- `ability_top1` ≥ 70% on curated val records (most Gen 3 OU abilities are deterministic per species)
- `move_recall_at_4` improves vs. Step 3 baseline (curated records supply the missing moves)
- `hp_type_top1` ≥ 50% (15 types, random = 6.7%)

**Ready for Step 5: MCTS**

The enriched completion model predicts species, full movesets, items, abilities, and Hidden
Power types — everything the MCTS simulator needs to construct a plausible opponent team.
