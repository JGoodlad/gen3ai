# Implementation: Step 3 — ELO / Skill Rating

A single, absolute, drift-free number that tracks the model's strength over training and
**across runs** — the progress signal self-play win-rate cannot give.

> **Status: BUILT & shipped.** This is an as-built record. The engine, the live training
> integration, the offline analyzer, the launcher badge, and the one-time bot anchor are all
> in code and verified. It partially delivers the measurement layer `impl_step2_league_play.md`
> references ("resolving the ELO ambiguity… not a win rate against a moving pool").

---

## Motivation

Once training is mostly self-play **pool play**, win-rate stops being legible:

- The promotion gate only promotes a snapshot when `win_rate_vs_pool > promote_threshold`,
  and the pool is a **sliding window of recent selves**. So `win_rate_vs_pool` is a treadmill
  pinned near 50–65% **by construction** — it cannot trend up no matter how much the model
  improves.
- `win_rate_vs_bots` saturates near 100% once the fixed bots are solved.

Neither tracks real progress during extensive pool play. The fix: the 9 eval bots (`random` +
8 archetypes) are a **fixed yardstick** that never changes within or across runs. Anchor a
rating model to them and every frozen snapshot lands on one absolute scale.

Crucially, **no new battles are needed for the model's rating**: every eval cycle already plays
the trainee (greedy) vs all 9 bots and vs up to 5 pool sentinels, `EVAL_GAMES` each. Those
win-records *are* a tournament-matrix row; we accumulate them and fit ratings.

---

## Two layers (don't conflate them)

| Layer | What it rates | How it's produced | Where |
|---|---|---|---|
| **Snapshot ELO** | the model at each step | **incrementally**, inline in the eval cycle that already runs | `eval/elo` live + the offline curve |
| **Bot anchor** | the 9 bots relative to each other | a **one-time** bot-vs-bot round-robin (model-independent) | `data/gen3_bot_elo_anchors.json` |

The snapshot ELO is fully incremental from live evals — it needs no special run and works on
the `random=0` fallback before any anchor exists. The bot anchor exists for one structural
reason: **in live evals the bots only ever play the model, never each other**, so from live
data their relative strengths are only knowable indirectly (and blur out once the model
saturates them). A direct round-robin (a) pins their relative ratings precisely and (b) is
model-independent, so the same bots get **one canonical rating across every run** → snapshot
ELOs are directly comparable run-to-run. It's computed once (bots never change) and reused
forever.

---

## The rating model

**Anchored Bradley-Terry** (a.k.a. logistic Elo / BayesElo), fit in **batch** by penalized MLE:

```
P(i beats j) = sigmoid( (R_i − R_j) · ln(10)/400 )          # 400 Elo pts per decade
```

- Each bot is a player `bot:<name>`, each snapshot `snap:<step>`. A snapshot is the **same
  player** whether it appears as a cycle's trainee or later as a sentinel (unified by step) —
  this links the whole ladder.
- The 9 bots are **pinned** to the anchor ratings; only snapshots are free (fallback: pin
  `random` at `base`=1000, others float).
- A weak Gaussian prior (σ≈400 Elo, centered at `base`) keeps perfect (100-0) records finite.
- Solved by **damped Newton-Raphson** — a backtracking line search guarantees descent on the
  concave objective; a full Newton step can otherwise diverge when the data and a pinned anchor
  are far apart (regression-tested in `elo_test.py::test_inconsistent_anchors_stay_finite`).
- Per-player uncertainty (the valuable part of Glicko) = `sqrt(diag(H⁻¹))` from the inverse
  Hessian → "ELO 1532 ± 40".

**Why batch-BT and not online Elo / Glicko-2:** online K-factor Elo is order-dependent and
accumulates path error — bad for a ladder where each snapshot plays one cycle. Glicko-2's
volatility models *skill drift over time*, but our snapshots are **frozen** (fixed strength
forever) — the drift is the *sequence* of snapshots, which the ELO-vs-step curve shows. Batch
MLE is ms-cheap (a few Newton steps over ~tens of players), so we compute the globally-consistent
answer directly. Glicko remains an easy additive option; not adopted.

**Known caveat (acceptable for v1):** the trainee acts greedy, sentinels act stochastic@temp,
so a snapshot's rating slightly blends its greedy/stochastic strength — a uniform shift that
preserves the trend.

---

## Data flow: input vs output

```
                  ┌─ INPUT (raw results) ─────────────────────────────────────┐
 every eval cycle │ <run>/eval_results.jsonl   (append-only, 1 line/cycle):    │
 (already runs) ──┤   {step, n_games, bots:{name→wr}, sentinels:[{step,wr}]}   │
                  └───────────────────────────┬───────────────────────────────┘
                                              │  fit_elo(rows, pin=anchors)
                  ┌─ OUTPUT (computed ELO) ────▼───────────────────────────────┐
                  │ eval/elo + eval/elo_ci  → TensorBoard + TUI 🏅 badge        │
                  │ metadata.json:latest_eval.elo   (rides the existing block)  │
                  │  → per-checkpoint sidecar .json + snapshot_history (stamped)│
                  └────────────────────────────────────────────────────────────┘
```

Only **one new file** (`eval_results.jsonl`) — the complete, append-only, restart-safe record
of every cycle (the fit needs all cycles; metadata holds only the latest, the snapshot stamps
are sparse). The ELO *output* rides the **existing** `latest_eval` block, so it propagates into
the sidecar + `snapshot_history` exactly the way win-rates already did — no new storage
mechanism. A 100M-step run's jsonl is ~50 lines (~15 KB).

**File placement (deliberate):**
- `data/gen3_bot_elo_anchors.json` — the **only** runtime input the model reads; immutable until
  bots change → lives in `data/`.
- `designs/ai_v5/elo_calibration/{gen3_bot_elo_games.json, *_heatmap.png}` — calibration
  provenance (resume state) + viz; *artifacts*, not runtime data → live with the design work.
- The model's ELO — `metadata.json` / sidecar / `snapshot_history` / `eval_results.jsonl`, all
  in the run dir. **No model-specific data in `data/`.**

---

## Module map

| File | Role |
|---|---|
| `src/agents/training/elo.py` | **pure engine** (numpy only): `EvalRow`, `load_rows` (log/tb/meta), `load_bot_anchors`, `fit_elo` (snapshot ladder) + `fit_pairwise` (round-robin) — both lower into one `_aggregate` + `_fit_aggregated` + damped-Newton `_newton` (grad/Hessian assembled once in `_grad_and_hessian`, shared by the step and the SE); `win_prob`/`ci95` are the single source for the BT formula + the 95% multiplier; `EloFit` (`snapshot_curve`/`rating_for_step`/`bot_ratings`) |
| `src/agents/model/snapshot.py` | `append_eval_result_row` → `eval_results.jsonl` |
| `src/agents/training/eval_callback.py` | `record_elo` (shared helper: append + refit + record `eval/elo`), called by **both** callbacks; `replay_last_eval_to_tui` re-publishes ELO on resume |
| `src/agents/training/selfplay_callback.py` | calls `record_elo` in `_collect_pending` (bots + sentinels) |
| `src/agents/training/bot_elo_calibration.py` | incremental/resumable bot round-robin driver → anchor; `--merge` for fleet stores |
| `src/agents/training/bot_elo_store.py` | pure resumable game-count store (load/save/accumulate/merge/results) — split out so the calibration script stays focused + the store is unit-testable (`bot_elo_store_test.py`) |
| `src/main/elo.py` | offline analyzer CLI (ladder + Elo-vs-step curve; `--source tb` backfill) |
| `src/main/launcher/format.py` + `app.py` | `🏅 ELO` badge + `ELO`/`ELO 95% CI` eval-summary row |

The `record_elo` helper is the **single integration point** for both eval paths, so the
bot-only and self-play paths can't drift. The engine is dependency-light (no torch/SB3) so it
imports into the callbacks *and* the CLI — one fit implementation, two consumers.

---

## Live training integration

- `record_elo` runs **inline on the training thread** when the (already non-blocking) eval
  cycle's results are collected — once per `EVAL_FREQ_STEPS` (2M). It appends the row, reads the
  tiny jsonl, fits (~ms over tens of players), and records `eval/elo` + `eval/elo_ci`. **No new
  thread or process; negligible throughput cost.** (The bot calibration fleet is a separate
  *offline* job, never part of training.)
- The live number is the best estimate from data **so far** (batch-BT is global → early points
  retro-adjust as more cycles land); `python -m main.elo` re-fits canonically.
- **Resume:** the run dir is stable across the launcher's periodic restarts
  (`_insert_or_replace_run_dir_arg`), so `eval_results.jsonl` **accumulates continuously** — the
  ladder doesn't reset every 3h. And `replay_last_eval_to_tui` re-publishes `eval/elo` from the
  resumed `metadata.json` on startup, so the 🏅 badge shows immediately instead of blanking until
  the next cycle.

---

## The bot anchor (one-time, offline, no server)

`python -m agents.training.bot_elo_calibration` plays every bot pair **in-process via the bridge**
(no server, no :8001 risk), fits BT (`random` pinned at 1000), and writes the anchor.

**Incremental & resumable:** game counts accumulate to the store and the anchor is re-fit after
every pair-chunk, so you never need a big pause — run it in the background (even alongside
training at low `--concurrency`), Ctrl-C anytime (partial anchor is valid), resume later. A
`git_hash` check warns if bot logic changed (`--reset` to discard). `--max-minutes` self-stops.

**Fleet + merge** (used to mint the production anchor): per-process throughput is capped by the
serialized battle-start lock (~2.3 games/s), so the lever is *more processes, not more
concurrency*. N independent processes each accumulate to their own store; `--merge` sums them
(independence verified — different RNG per process). The minted anchor:

- **86,000 bot-vs-bot battles**, 2,000–2,700 per pair, ~80 min on a 16-core box.
- Bot ratings at **±7 Elo**; `mean_abs_err = 0.99%` (the ensemble is nearly transitive — a single
  Elo per bot is faithful), worst pair off 3.8%.
- Ladder: `heuristic2` 1639 ≈ `aggressive_v2` 1630 → `aggressive` 1512, `random` pinned 1000.

---

## Offline analyzer

`python -m main.elo <run_dir> [--source auto|log|tb|meta] [--out DIR]` — prints a ranked ladder
(bots + snapshots, Elo ± 95% CI) and writes `elo_ratings.json` + `elo_curve.png` (Elo-vs-step
with CI band + bot anchor lines). `--source tb` **backfills an already-running run straight from
TensorBoard** with zero training change; `--out` elsewhere analyzes a live run without touching
it.

---

## Verification

- `src/agents/training/elo_test.py` — synthetic ground-truth ladder recovery, anchoring,
  perfect-score finiteness, **damped-Newton divergence guard**, translation-equivariance,
  dedup-by-step, loaders, `fit_pairwise`.
- `eval_callback_test.py` — resume re-publishes `eval/elo` (`test_replay_last_eval_republishes_pool_block`).
- End-to-end: a `--debug --self-play --use-showdown-bridge` run wrote `eval_results.jsonl` + a live
  `eval/elo`; the offline CLI + the real anchor produce the ladder/curve. Calibration validated by
  the 86k-battle production run.

---

## Future (optional)

- Push the anchor to 5,000 games/pair (resume; ±~5 Elo) — currently 2,000 (±7).
- A global greedy-vs-stochastic offset parameter to remove the v1 caveat.
- Feed the anchored ratings into the league payoff matrix (PFSP) per `impl_step2`.
