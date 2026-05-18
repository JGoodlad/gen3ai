# AI v4 — Team Completion Model

The agent currently has no model of what the opponent is carrying. During MCTS rollouts
(and even without MCTS, for planning purposes), we need to sample plausible complete
opponent teams from partial observations — revealed Pokémon, seen moves, inferred items.

Raw usage stats give marginal distributions but miss the joint structure that actually
defines Gen 3 OU team-building: Focus Punch Tar ≠ DD Tar, physical Salamence implies
special or Pursuit Tar, weather-clear teams don't run Tyranitar. A learned model capturing
these correlations is a force multiplier for any search-based planning layer.

The key insight: the existing PPO model already contains a strong team-encoding backbone
(embedding tables + role encoder + synergy attention), trained on millions of battles.
The team completion model **freezes this backbone and adds only a small new head** —
dramatically reducing the data needed to learn team-level joint structure.

---

## 1. Self-Play Team Logger

**Where:** `src/agents/training/gen3_env.py` → `Gen3Env.reset()`

### Problem

The PPO training loop plays millions of battles where both teams are fully known to the
server. This is the richest possible source of Gen 3 OU team composition data — real teams
that were validated, queued into the pool, and actually played against each other. None of
it is currently logged.

### Insertion point

In `Gen3Env.reset()` (line 115), immediately **before** `super().reset()` is called, both
`self.battle1` and `self.battle2` are still alive and fully populated. `battle1.team`
gives our team; `battle1.opponent_team` gives the opponent's. After `super().reset()` is
called the battles are deallocated.

### Design

Add an optional `team_log_path: Path | None` argument to `Gen3Env.__init__()`. When set,
log one JSON line per completed episode before reset:

```python
{
  "our_team":   [<showdown_text_per_mon>, ...],   # 6 entries
  "opp_team":   [<showdown_text_per_mon>, ...],   # 6 entries
  "winner":     "ours" | "opp" | "tie",
  "n_turns":    int
}
```

**Missing piece:** There is currently no `Pokemon → Showdown text` serializer. The
`Pokemon` object in poke-env stores species, item, ability, moves, HP, EVs, nature —
everything needed. Write `serialize_pokemon(mon: Pokemon) -> str` in a new utility
`src/agents/team_model/team_serializer.py`. Use `mon.species`, `mon.item`, `mon.ability`,
`list(mon.moves.keys())` (move IDs), and the team-text format already parsed by
`Gen3Teambuilder`.

### Also

Wire the `team_log_path` argument through `train_rl_agent.py` CLI as `--team-log`. Enabled
by default under a sensible path (`models/<run_id>/teams.jsonl`). Low overhead — one JSON
write per episode, which averages 25 turns.

---

## 2. Team Completion Dataset

**Where:** `src/agents/team_model/dataset.py` (new)

### Design

`TeamCompletionDataset` loads complete teams from two sources:

**Source A — existing 770 teams** (`data/teams/`):
Use `TeamLoader.get_all_teams()` (already in `src/utils/team_loader/loader.py`) to load
all teams. Parse each Showdown-format team file with the existing `Gen3Teambuilder`
infrastructure. This gives ~770 complete 6-mon teams immediately, no training required.

**Source B — self-play JSONL** (from Step 1):
Load from `teams.jsonl` written during training runs. Each entry is already in parsed form.

Each complete team of 6 yields **63 training examples** (all non-trivial bitmask patterns
for 1–5 masked slots). With 770 teams that's ~48K examples before any self-play data
lands; with millions of self-play games it scales arbitrarily.

Each training example is a tuple:
- `observed_slots`: list of encoded Pokémon vectors for revealed slots (using frozen backbone)
- `mask_pattern`: bitmask indicating which slots are hidden
- `labels`: species ID, item ID, ability ID, 4 × move ID per hidden slot

Mask pattern is sampled randomly during `__getitem__` so each epoch sees different
masking for the same team.

---

## 3. Team Completion Model Architecture

**Where:** `src/agents/team_model/completion_model.py` (new)

### Design

The model has two clearly separated parts:

**Frozen backbone (loaded from PPO checkpoint, `requires_grad=False`):**

| Component | Dims | Source |
|-----------|------|--------|
| Species embedding | 32D | `features_extractor.py:42` |
| Move embedding | 16D | `features_extractor.py:46` |
| Item embedding | 16D | `features_extractor.py:50` |
| Ability embedding | 16D | `features_extractor.py:54` |
| Type embedding | 16D | `features_extractor.py:58` |
| Move processor | 58→32D per slot | `features_extractor.py:83` |
| Role encoder | 259→128D per mon | `features_extractor.py:98` |

Loading: `torch.load(checkpoint_path)` extracts these module weights by name. They are
copied into the completion model and frozen. No PPO training logic is loaded.

**New trainable head:**

1. **Mask token** — a learned 128D embedding, substituted in place of unknown slot role
   tokens. Analogous to `[MASK]` in BERT. One parameter, initialized to small random values.

2. **Completion transformer** — 2-layer `nn.TransformerEncoder` (4 heads, 128D model dim,
   256D feedforward) over the 6 role token slots. Operates on the mixed sequence of
   observed role tokens and mask tokens. Captures cross-slot conditioning: "given Skarmory
   is in slot 0, what belongs in slot 3?"

3. **Output heads** (applied only to masked slots):
   - Species head: `Linear(128 → num_species)` → softmax
   - Item head: `Linear(128 → num_items)` → softmax
   - Ability head: `Linear(128 → num_abilities)` → softmax
   - Move heads ×4: `Linear(128 → num_moves)` → softmax

   Move heads are independent (not autoregressive over moves within a slot). Conditioning
   on species happens implicitly through the role token, which encodes species via the
   frozen role encoder.

### Training objective

BERT-style masked slot prediction. Loss is cross-entropy, summed only over masked slots.
Unmasked slots contribute zero loss — the model is not penalized for ignoring them.

```
L = Σ_{i ∈ masked} [CE(species_logits_i, species_label_i)
                   + CE(item_logits_i, item_label_i)
                   + CE(ability_logits_i, ability_label_i)
                   + Σ_j CE(move_logits_i_j, move_label_i_j)]
```

### Training stages

**Stage 1 — bootstrap on 770 existing teams:**
- Frozen backbone, only new head trains
- LR: 1e-3, batch: 32, epochs: 200
- Expected outcome: learns basic co-occurrence patterns from the curated team pool

**Stage 2 — scale on self-play JSONL:**
- Optionally unfreeze role encoder with LR: 1e-5 (backbone); new head at LR: 3e-4
- Batch: 256, continuous training as new self-play data arrives
- Expected outcome: learns real joint distribution, format-specific archetypes

**Stage 3 — domain adapt on ladder data (see Step 4):**
- LR: 1e-5 throughout (avoid forgetting self-play priors)
- Expected outcome: learns human-specific biases (standard sets, popular cores)

---

## 4. Ladder Daemon

**Where:** `src/agents/team_model/daemon.py` (new)

### Design

Subclass `Player` (from `src/poke_env/player/player.py`) and override
`_battle_finished_callback()` (line 168), which fires at the end of every battle with
the complete `AbstractBattle` object. At that point `battle.team` (our side) is fully
populated and `battle.opponent_team` contains everything the opponent revealed.

```python
class LadderDaemonPlayer(RLPlayer):
    def _battle_finished_callback(self, battle: AbstractBattle):
        self._fetch_and_log_replay(battle.battle_tag)
```

After the battle ends, fetch the replay JSON from Showdown's replay API:

```
GET https://replay.pokemonshowdown.com/{battle_tag}.json
```

Showdown replay JSON contains `|poke|` messages in the `log` field that declare both
teams in full at the start (species, level, gender, shiny flag — not moves/items, but
combined with the observed battle log, significantly narrows the posterior). Parse these
plus the battle log to build the most complete team record possible.

Write to a separate `ladder_teams.jsonl` with a flag distinguishing "fully observed" mons
(sent into battle) vs. "species only" (benched and never revealed). The dataset loader
in Step 2 masks unrevealed move/item/ability slots accordingly rather than treating them
as training signal.

`requests` is already a dependency (used in `ps_client.py:11`).

### Running the daemon

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  src/agents/team_model/daemon.py \
  --model models/latest.zip \
  --team-log models/ladder_teams.jsonl \
  --n-battles 100
```

Runs N ladder games, logs both sides' team data. Can run alongside training — daemon uses
the latest checkpoint for moves but the primary purpose is data collection.

---

## 5. MCTS Integration

**Where:** `src/agents/inference/` (new, when MCTS is implemented)

### Design

The team completion model plugs into MCTS as the **world-sampling step** in PIMC
(Perfect Information Monte Carlo). At the start of each MCTS search:

1. Encode revealed opponent slots through the frozen backbone → role tokens
2. Fill unrevealed slots with the learned mask token
3. Run completion transformer → per-slot distributions
4. Sample N complete team hypotheses (top-k sampling, k=5 per slot)
5. For each hypothesis: this becomes the "sampled world" for one MCTS trajectory

This replaces the flat usage-stat sampling that would otherwise be needed. The completion
model's joint distribution ensures sampled teams are realistic — "given Skarmory, sample
from the conditional distribution" produces Blissey, Pursuit Tar, Magneton, etc., rather
than independently sampling from marginals.

N is determined by the computational budget (Wang 2024 achieved 1000–2000 MCTS rollouts
in 10 seconds with 20 parallel workers). Each rollout uses one sampled team hypothesis,
so N = number of rollouts.

### Prerequisite

MCTS itself is not implemented yet. This step depends on a future design doc. The team
completion model can be trained and validated independently of MCTS — it's useful on its
own for opponent modeling during pure PPO inference.

---

## 6. Damage-Calc Constraint Inference (Optional Enrichment)

**Where:** `src/agents/team_model/constraint_tracker.py` (new)

### Problem

The team completion model conditions on revealed species, seen moves, and inferred items.
Items are often invisible until a move interaction reveals them (Leftovers HP recovery,
Choice Band damage magnitude, Lum Berry curing status). The model currently treats these
as unknown.

### Design

A `BattleConstraintTracker` processes the battle log turn-by-turn and narrows the
posterior over each opponent Pokémon's held item using the Gen 3 damage formula:

```
Damage = floor(floor(floor(2*100/5 + 2) * BasePower * Atk/Def / 50) + 2) * modifier
```

Observables per turn: HP fraction delta, move base power, type matchup (known from
species data), STAB, weather modifier. Unknown: Atk stat (constrained by Nature/EV),
item modifier.

Key detectable signals:
- **Choice Band**: all physical attacks deal consistently ~1.5× expected neutral damage
- **Leftovers**: HP recovers by exactly 1/16 max HP each turn the mon doesn't take damage
- **Berry consumption**: single-turn HP jump visible when berry activates

The tracker outputs a `{slot: {item_id: probability}}` posterior, updated each turn.
This is fed as additional conditioning signal to the completion model at inference time
alongside the observed role tokens.

This is the most complex component and is not needed for the base model to be useful.
Implement after Stage 2 training (Step 3) is producing sensible team completions.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/team_model/team_serializer.py` | `Pokemon → Showdown text` serializer |
| `src/agents/team_model/dataset.py` | `TeamCompletionDataset` |
| `src/agents/team_model/completion_model.py` | Architecture + frozen backbone loader |
| `src/agents/team_model/train.py` | Training script (staged) |
| `src/agents/team_model/daemon.py` | Ladder observer daemon |
| `src/agents/team_model/constraint_tracker.py` | Damage-calc item inference (Step 6) |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/training/gen3_env.py` | Add `team_log_path` arg, log teams in `reset()` |
| `src/main/train_rl_agent.py` | Wire `--team-log` CLI arg through to `Gen3Env` |

---

## Verification

1. **Step 1 smoke test:** Run `train_rl_agent.py --debug --steps 10000 --team-log /tmp/teams.jsonl`;
   confirm JSONL is written, each line parses cleanly, teams round-trip through
   `serialize_pokemon()` and back.
2. **Step 2 sanity:** Load `TeamCompletionDataset` on the 770 existing teams; confirm 63×770
   examples generated, no empty move slots on populated mons.
3. **Step 3 qualitative checks after Stage 1 training:**
   - Given Skarmory alone → top-3 species predictions include Blissey
   - Given physical Salamence → "Pursuit Tar" scores higher than "Calm Mind Tar" in item/move
     distribution
   - Given Tyranitar (Sand Stream) → weather-setter species score low in remaining slots
4. **Step 3 quantitative:** Perplexity on 10% held-out teams from the 770-team pool;
   should be lower than a flat-usage-stats baseline.
5. **Step 4 daemon:** Run 10 ladder games; confirm `ladder_teams.jsonl` written with correct
   team structure, replay fetch succeeds, `|poke|` parsing captures both teams' species.
