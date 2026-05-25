# AI v6 — Todo

The generalist trained in v3–v5 is a strong all-rounder. v6 has two goals: (1) identify
the best teams and specialise a model per team, then take them to the ladder; (2) integrate
cheap MCTS into the training loop so the final model trains against better-quality action
selections on both sides — bridging toward v7's full training-time MCTS via the Rust sim.

---

## Step 1 — Team Evaluation

Run the v5 MCTS agent through all 32 sample teams against the v4 league pool, measuring
win rate per team. Select the top 3. See `designs/ai_v6/impl_step1_team_eval.md`.

**Design questions to resolve:**
- **Games per team**: 200 gives ~±7% confidence interval at 50% win rate. Increase to
  500 if rankings are tight (within a few percent of each other).
- **Opponent pool**: use the full v4 league (PFSP-weighted) or a fixed benchmark set
  (e.g., best league snapshot + SimpleHeuristicsPlayer + MCTSPlayer)? Fixed benchmark is
  more reproducible; PFSP-weighted is more realistic.

---

## Step 2 — Per-Team Specialisation

Fine-tune one model per top-3 team, starting from the v5 generalist checkpoint, with the
team fixed for the duration. See `designs/ai_v6/impl_step2_specialisation.md`.

**Design questions to resolve:**
- **Opponent distribution during fine-tuning**: league pool (as in v4 self-play) or the
  other two specialised teams as additional opponents? The latter adds within-stable
  coverage but risks over-fitting to the stable matchups.
- **Run length**: 10–15M steps per team. Start at 10M; extend if win rate is still rising.
- **Parallelism**: the three fine-tuning runs are independent — run them concurrently on
  separate GPUs if available.

---

## Step 3 — Ladder Run

Deploy each of the three specialised MCTS players on the real Showdown ladder. Track ELO
per team and use the results to decide which team is the primary ladder entry.
See `designs/ai_v6/impl_step3_ladder.md`.

**Design questions to resolve:**
- **Account strategy**: one account per team (clean ELO per team), or rotate teams on a
  single account? Separate accounts give cleaner data.
- **Games per team**: 100 ladder games per team to get a stable ELO estimate.
- **Stopping criterion**: stop when ELO has been stable (±30 points) for 30+ consecutive
  games. A rising ELO after 100 games means keep playing, not stop.

---

## Step 4 — Cheap MCTS in Training

Integrate shallow action sampling into the PPO self-play data collection loop. The Node.js
bridge (from ai_v5 Step 5) is too slow for deep MCTS during training, but a very shallow
sweep — `K=3` rollouts per legal action, `max_depth=1` — adds only ~30ms per decision and
requires no changes to the bridge protocol.

**What changes:**
- The training env's `choose_move()` runs flat action sampling instead of a raw policy
  forward pass.
- Both sides see improved action quality — not just our agent.
- The PPO policy now trains on trajectories where actions were chosen with at least a
  one-ply lookahead and 3-rollout confirmation per option.

**Expected benefit:** the model learns to value positions that are favourable for MCTS-like
play, rather than positions that happen to look good to the raw policy. This should transfer
positively when the model is later used with full MCTS at inference time.

**Tuning knobs:**
- `K` (rollouts per action): start at 3. Increase if throughput permits.
- `max_depth`: keep at 1 for training (pure value-head leaf evaluation, minimal RNG
  accumulation). Depth 1 means: fork, take action, ask `V_θ` for the resulting state.
- **Throughput**: measure steps/second before and after. The target is ≤ 2× slowdown vs.
  raw policy training. If slower, reduce `K` or disable for bench/utility Pokémon (only
  run sampling when the active Pokémon has a non-trivial choice).

**Design questions to resolve:**
- **Both sides vs. our side only**: running sampling on both sides doubles the bridge
  calls. Start with both sides for data quality; if throughput is unacceptable, sample
  only for our agent and use the raw policy for the opponent.
- **League opponent**: should league agents also use action sampling, or only the
  learning agent? League agents with sampling are harder opponents; without sampling they
  are consistent with how they were evaluated in v5.

**Stopping criterion:** v6 Step 4 is complete when:
- Training with `K=3, max_depth=1` sampling runs without throughput regression > 2×
- Win rate (v5 checkpoint evaluated with full MCTS) does not regress — the model trained
  with cheap MCTS should be at least as strong as the raw-policy baseline
- Ladder ELO (with full MCTS at inference) is ≥ Step 3 baseline

---

## Future Work — Meta Alignment

The current setup measures win rate against the v4 league, which is a proxy for ladder
performance. The league may not reflect the real Gen 3 OU meta distribution — certain
teams and strategies appear far more frequently on the ladder than in self-play.

Two directions to explore in a future version:

**Ladder-weighted team selection**: after the initial ladder run, feed collected replays
back into the team completion model and re-run the team evaluator, this time weighting
opponents by their ladder frequency rather than league composition. The "best team"
ranking may shift significantly once the opponent distribution reflects real ladder play.

**Meta self-alignment**: continuously update the opponent pool during specialisation
fine-tuning using replays collected from the ladder. The agent trains against what it
actually encounters, closing the gap between league proxy and ladder reality. This is a
training loop, not a one-shot evaluation — it requires the ladder daemon (v5 Step 1) to
run concurrently with fine-tuning.

---

## Handoff to v7

v6 ends with a strong specialised agent on the ladder and the infrastructure for cheap
MCTS in training. v7 picks up here by replacing the Node.js bridge with a Rust sim
(PyO3, in-process), enabling:
- 50 000+ rollouts/turn at inference (vs. ~1 000 with the JS bridge)
- Full MCTS during training data generation (previously blocked by JS bridge throughput)
- Training against MCTS-quality opponents — the final frontier for data quality
