# Design + Experiment Record — Public-Information Value Function (human-replay-supervised)

**Status:** SCRATCH EXPLORED (2026-07-05), NOT built into main. Verdict: the *aggregate* public value is
real, cheap, and worth an integration test; identity/structure did **not** beat aggregates at scratch scale.
Companion to `design_curriculum_uncertainty.md` and `designs/learning/pbs_value_functions_and_search.md`
(the theory). Scratch scripts live in `tmp/` (uncommitted): `pubval_experiment.py`, `pubval_richobs.py`,
`pubval_transformer.py`.

## Motivation (why this, from the value-function-is-the-hub thesis)

The measured limiter is the **value function**: it's blind to defensive/positional value (defensive AUC ≈ 0.50
vs offense ≈ 0.63–0.71), so the advantage is ~0 on positional decisions and the policy never learns a game
plan ("trivially okay at matchups, not great at games"). Fixing V needs a value signal from *outside* the RL
bootstrap. A **public-information value function** — `V_pub(public board) → P(win)`, trained purely offline on
human replays — is that exogenous signal: it's value-independent (supervised on real outcomes), universal
(the public board is a common language across teams), and the intended use is a **shared-trunk aux target** so
its positional features enrich the RL critic's representation (leak-safe: the RL head still learns `V^π`, only
the trunk features are shared — never wire `V_pub` into GAE, it's `V^human` not `V^π`).

## The corpus

`/home/goodlad/dev/gen3ai/replays/showdown/gen3ou/<date>/*.log` — **164,230** gen3ou rated-ladder replay logs
(Showdown protocol; player ratings are in the `|player|` lines, so rating-filter/stratify is trivial; `|win|`
gives the outcome). Parsing: reconstruct the **public** board per `|turn|` from the log (revealed mons + public
HP/status/boosts, hazards, weather, revealed moves), generate positions from **both perspectives** (mirror),
label every position by the game's terminal outcome. `data/replays` is empty; the corpus is under `replays/`.

## Experiments (CPU-only, split-by-GAME held-out)

| # | Representation | Model | Test AUC | Notes |
|---|---|---|---|---|
| 0 | material (mon-count + HP diffs) | logistic | ~0.70 | baseline |
| **1** | **crude aggregates** (+ hazards/status/boosts, 17 feats) | **logistic** | **0.734** | **best**; MLP also 0.734 |
| 2 | rich identity (species base-stats/types/type-matchups, 122 feats) | logistic | 0.733 | **no gain** over crude |
| 2 | rich identity | MLP 2×128 | 0.689 test / **0.914 train** | overfit |
| 3 | **per-mon + moves tokens** (learned species+move embeddings) | **transformer** | **0.699 peak → 0.638** | overfits (declines w/ epochs) |

**Verification checks (all passed on exp 1):**
- **Leakage guard:** turn-1 AUC ≈ **0.51** (chance — can't predict a symmetric opening), rising monotonically
  to ~**0.80–0.84** by the late game (more revealed → more predictable). Correct shape, no outcome leakage.
- **Calibration:** predicted buckets match realized win-rate closely — [0–0.2)→0.11, [0.4–0.6)→0.50,
  [0.8–1.0)→0.89 — well-calibrated out of the box.
- **Beats material:** +0.04 AUC over material-only (public *positional* aggregates add real value beyond raw
  material).

## Verdict (honest)

1. **The public value function is real, cheap, and validated.** ~0.73 AUC overall, ~0.80 late-game, clean
   leakage, calibrated — from a *17-feature logistic trained in seconds on CPU*. The cheapest model is the best.
2. **Identity/structure did NOT beat simple aggregates at scratch scale.** Adding species/type/move identity via
   richer features, an MLP, or a transformer with learned embeddings all **matched or underperformed** the
   aggregate baseline and **overfit** the within-game correlation (the transformer's test AUC *declines* with
   training; the MLP hit train 0.91 / test 0.69). The public signal is dominated by **material + hazards +
   status**, not *which specific mons*.
3. **KEY CAVEAT — data-limited.** Scratch used 7–20k games; the overfitting is severe partly because ~758
   species+move embeddings can't stabilize on so few *independent games*. The **full corpus is 164k (≈20×)**,
   which would drastically shrink the overfit gap — a properly-regularized transformer on the full corpus
   *might* extract identity value. Scratch **cannot rule this out**. Separately, Pokémon's irreducible variance
   (dice + hidden teams) likely caps *public* win-prediction around ~0.74–0.80 regardless of model.

## Recommendation (what to actually build)

- **Do NOT chase raw AUC beyond 0.74** with a fancier model — that's not the point and scratch says it overfits.
- **Build the CHEAP version and run the transfer test.** Wire the **aggregate** public value (the ~17-feature
  logistic, or the equivalent over the obs pipeline's *public/revealed* subset) in as a **shared-trunk aux
  target**, and measure the one thing that matters: **does the critic's defensive-AUC-by-style climb?** (Reuse
  the calibration-by-style analysis from the value-conditioning probe.) That is the real question — *positional-
  discrimination transfer*, not win-AUC — it's nearly free, and it doesn't hinge on beating 0.74.
- **The full-corpus transformer is a bigger bet, a follow-on only.** Attempt it (164k games, embedding dropout +
  early stopping + weight decay, per-mon+move tokens) *only if* the cheap aggregate version shows the transfer.
  Rating-filter but note more-data > higher-rating at these scales (≥1400 filter cost data and slightly lowered
  AUC in exp 1/2).
- **As a standalone tool** the aggregate `V_pub` is immediately useful as a universal position evaluator for the
  prober / curriculum scoring, independent of any training run.

## Practical build notes

- Reuse the obs pipeline's **public/revealed** subset instead of hand-features (the encoder already separates
  revealed vs believed). Full 2992-dim obs is NOT buildable from a replay (needs each player's own hidden team +
  legal-action requests) — public is the faithfully-learnable slice.
- Supervised, terminal-outcome (Monte-Carlo) target — keeps it value-independent (no bootstrap circularity).
- Always split held-out **by game** (positions within a game are correlated) and re-check the **turn-1 ≈ 0.5**
  leakage guard on any new feature set.
