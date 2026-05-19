# AI v5 — Replay Corpus, Team Prediction, and Ladder-Weighted Training

Two systems with a shared data dependency: a replay scraper that builds a
corpus of real ladder teams, and a hidden-team predictor that uses that corpus
to reason about unseen opponent slots during play.

---

## Motivation

The ai_v3 agent trains against a flat pool of 770 sample teams. This has two
problems at the ladder stage:

1. **Wrong distribution.** Real Gen 3 OU ladder has a meta: some teams appear
   constantly (Skarmory/Blissey/Suicune cores), others are rare. A flat pool
   over-represents obscure teams and under-represents the ones that actually
   matter for win rate.

2. **Hidden information is ignored.** When 3 of the opponent's 6 slots are
   unrevealed, the agent has no signal about what might be there. A predictor
   trained on real team compositions can say "Skarmory + Blissey seen → 70%
   chance there's a Suicune" and feed that into the observation.

Both problems have the same root fix: a corpus of real ladder replays.

---

## 1. Replay Collection ✓ DONE (live spectator)

**Goal:** build a corpus of complete Gen 3 OU battle logs.

### What was built

`src/main/collect_replays.py` — a long-running daemon that connects to Showdown as a
guest, discovers active Gen 3 OU battles via `/query roomlist gen3ou`, spectates up to
20 concurrent rooms, and saves each `.log` file the moment the battle finishes.

Joining mid-battle is fine: Showdown sends the full scrollback from turn 1 on connect,
so no partial replays are produced. All data flows over a SOCKS5 proxy (routed through a
GCP VM) to keep the home IP off Showdown's throttle lists.

Currently collecting into `replays/showdown/1/` (daemon running).

### Comparison to API scraping

The live approach produces exactly the same log format as the Showdown replay archive
(`replay.pokemonshowdown.com`), but with zero scraping latency and no pagination. The
replay API is still useful for **historical data** (battles played before we started
collecting). If a historical backfill is needed:

```
GET https://replay.pokemonshowdown.com/api/replays?format=gen3ou&page=N
# then per replay:
GET https://replay.pokemonshowdown.com/gen3ou-XXXXXXXX.json
```

The JSON `.log` field is the same Showdown protocol format our parser already handles.
A backfill script would be a thin wrapper around the existing parsing pipeline from Step 2.

### Log format and parsing

Each `.log` file is raw Showdown protocol (one message per line). Consumer rules:
- `|win|username` → `battle.won_by(username)` (not via `parse_message`)
- `|tie` → `battle.tied()`
- All other lines → `battle.parse_message(line.split("|"))`

Full format documented in `designs/ai_v5/impl_step1_replay_collection.md`.

---

## 2. Ladder-Weighted Opponent Sampling

**Goal:** replace the flat 770-team pool with a frequency-weighted sample from
real ladder teams.

### How it works

Replace `Gen3Teambuilder(all_teams)` for the frozen opponent with a sampler
that draws teams proportional to their ladder frequency:

```python
# Conceptual — exact implementation TBD
weights = [team["count"] for team in ladder_teams]
chosen = random.choices(ladder_teams, weights=weights, k=1)[0]
```

A smoothing floor (e.g. `max(count, min_weight)`) prevents ultra-rare teams
from being completely excluded, while still reflecting the real meta distribution.

### Trainee sampling

The trainee (our agent) should continue drawing from the 32 curated sample teams
during Phase 1 training — the goal is a generalist, not meta-specialization.
Meta-specialization ("fine-tuning" on the 5 best ladder teams) is a Phase 2 step
after the generalist is mature.

---

## 3. Hidden-Team Predictor

**Goal:** given the opponent's revealed slots mid-game, predict the distribution
over unseen slots and feed that signal into the observation.

### Framing

A masked autoencoder / conditional distribution model:

- **Input:** revealed species IDs (known slots), masked tokens for unknown slots,
  current turn context
- **Output:** per-unknown-slot distribution over species IDs (or a top-K
  shortlist + confidence)

Trained on the replay corpus: for each replay, generate `(partial_reveal,
full_team)` pairs at each turn where a new mon was revealed.

### Integration into observation

Two options (not mutually exclusive):

**a) Soft embedding** — compute the expected species embedding as a weighted sum
over the predicted distribution for each unknown slot, and substitute it into
the opponent team encoding where `species_known = 0`. The network already handles
unknown slots with `species_known = 0.0`; this replaces the zero vector with an
informed prior.

**b) Explicit confidence scalar** — append a per-slot confidence score to the
opponent Pokémon vector. The agent learns to weight revealed vs. predicted
information differently.

Option (a) is lower risk and requires no architecture change — start there.

### Training data format

```
# One record per (turn, reveal event) in a replay
{
  "revealed_so_far": ["skarmory", "blissey", null, null, null, null],
  "full_team": ["skarmory", "blissey", "suicune", "tyranitar", "zapdos", "gengar"],
  "turn": 8
}
```

---

## 4. Ladder Team Finder (Phase 2)

After the generalist model is trained, run it on ladder with each of the 32
sample teams for N games each and rank by win rate. The top ~5 become the
deployment pool — analogous to how human players identify their best teams
through practice and specialize.

This is a deployment decision, not a training constraint. No code change needed
until Phase 2.

---

## Build Order

1. ✓ **Replay collection** — live spectator daemon collecting continuously
2. **Ladder-weighted opponent sampling** — drop-in replace for training; fast win
   (needs enough replays to have reliable frequency data — collect ~10K first)
3. **Hidden-team predictor** — standalone model, trained offline on corpus
4. **Observation integration** — wire predictor output into `gen3_env.embed_battle()`
5. **Ladder team finder** — run after Phase 1 generalist is mature

---

## Open Questions

- How many replays are available for Gen 3 OU on Showdown? If the archive is
  thin, supplement with simulated replays from ai_v3 self-play.
- Should the predictor output be a hard top-1 prediction or a soft distribution?
  Soft is safer early (avoids confidently wrong predictions poisoning the obs).
- Rate-limiting on the Showdown replay API — need to check before scraping at scale.
