# AI v6 — Todo

The generalist trained in v3–v5 is a strong all-rounder. v6 is about identifying the
three teams it performs best with, specialising a model for each one, and taking them to
the ladder.

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
