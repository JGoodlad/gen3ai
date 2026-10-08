# P3 of the poke-env retirement — the anchors' OUR side on the Rust stack: the matched comparison

**Date:** 2026-10-07 · **Code:** the P3 build on top of `f7567a9f` (worktree, before landing) · **Box:** CPU only,
load 30–40 on 16 threads (a live training run held the GPU) · **Servers:** local only — the anchors harness's own
front end on 9500 / 9501 (auto-picked); no public server, no 8000 / 8001, no human contact.

## The question

P3 moves OUR side of an external-anchor read (`python -m main.anchors`) off the poke-env websocket client
(`main.play` → `RLPlayer`: vendored poke-env parses the protocol, the Python battle layer folds it, the PYTHON
encoder builds the row) and onto the Rust stack: the websocket front end runs IN the anchors process and our side
is an in-process slot of it, deciding on `sim_bridge`'s core observation frame (the reader training runs). The
opponent stays an external upstream-poke-env process on the socket. Does the change move the anchor number?

Plan gate (survey §A4.5, P3): "byte-identical ACTIONS vs today's `RLPlayer` client on the same seeded battles, both
still present". Brief gate: a matched comparison, the win-count shift with a CI, zero protocol / parse failures.

## Protocol

- **Checkpoint:** `models/rb_x5ab_blob_s1008/final_model.zip` (step 15,048,279; sha256 `e251b30d1e51bb86…`), a
  current-architecture Rustboro-era X5 `blob` arm, loaded `bare` by both paths.
- **Opponent:** `metamon:SmallRL` (ckpt 40, `0a00a759`), greedy, its own process and interpreter.
- **Matched:** each cell played TWICE — `new` = `--our-transport core` run under `utils.poke_env_blocker`
  (every `import poke_env` raises and is recorded; none was), `old` = `--our-transport poke-env` (the legacy
  client, subprocess front end). Same `--seed-base 914007` (each battle's sim seed), same `--team-seed` (both
  sides' draws), same role-balanced halves, `--capture-dir` on both (every CHOOSE / FORCELOSE written to each
  battle's child, and every per-side chunk relayed).
- **Cells:** `away_greedy` (Metamon's 20-team set, 40 games), `home_greedy` (our 719-team pool, 40 games),
  `away_t1seeded` (away, our side SAMPLING at T = 1.0 with `GEN3AI_POLICY_SEED=11` on both paths, Metamon greedy,
  20 games) — the last checks the sampled branch draws from the same generator in the same order.
- Scripts: `scripts/batch.sh` (the run; it points at `/tmp/p3work` paths of the session that ran it),
  `scripts/read.sh` (the wrapper: `PYTHONPATH`, the worktree's own release `sim_bridge`, the blocker),
  `scripts/analyze.py` (the read). Banked: each cell's `games.jsonl` + `summary.json` per path, and
  `analysis.json`. The 66 MB of captures were not banked; `analyze.py` re-derives the identity counts from them.

## Result

| cell | new (core slot) W/L/T | old (poke-env client) W/L/T | shift new − old, Newcombe 95% | battles byte-identical | our decisions |
|---|---|---|---|---:|---:|
| away · greedy | 19/21/0 | 19/21/0 | **0.000 [−0.209, +0.209]** | 40 / 40 | 2,252 |
| home · greedy | 24/16/0 | 24/16/0 | **0.000 [−0.206, +0.206]** | 40 / 40 | 2,418 |
| away · ours T = 1.0 seeded | 4/16/0 | 4/16/0 | **0.000 [−0.247, +0.247]** | 20 / 20 | 908 |
| **pooled** | **47/53/0** | **47/53/0** | **0.000 [−0.136, +0.136]** | **100 / 100** | **5,578** |

- **"Byte-identical"** = in every one of the 100 battle pairs, each side's sequence of commands to the battle
  child (11,108 choices over both sides) AND every per-side protocol chunk the front end relayed are equal. The two
  paths played the SAME 100 games, decision for decision, so the paired shift is exactly 0 on every game; the
  Newcombe interval above treats the cells as independent samples and is the conservative bound. Paired: 0
  discordant games of 100 ⇒ the discordance rate is below 3 / 100 at 95 % (rule of three), so |shift| < 0.03.
- **Protocol / parse failures: zero.** `status: OK` on all six reads; 0 ERROR lines in all six front-end logs;
  0 `core_slot_error`; 0 defaults and 0 re-decides on the legacy path; `regime_verified_decisions` and
  `peer_clean` true on all six; Metamon's `argmax_match_rate` 1.0000 throughout. No game reached the 250-turn
  forfeit.
- **The sampled cell** read `our_argmax_match_rate` 0.574 on BOTH paths (sampling engaged, same draws).
- **Wall:** equal within noise (away 2,381 s vs 2,320 s; home 1,567 vs 1,538; t1 296 vs 291, under load ~35;
  a capped one-game read 37.3 s vs 37.0 s back to back). The core slot is not a speed change here; the
  opponent's per-decision cost and the box dominate.

**Verdict: EQUIVALENT BY CONSTRUCTION on this protocol — byte-identical actions on 100 / 100 seeded battle pairs.**
The `our_transport` stamp is provenance, not a regime boundary: readers do not refuse a mix (contrast P2, whose
protocol changed seats, mirroring and the draw fold).

## Findings

1. **The legacy client's per-row `n_decisions` is CUMULATIVE over the half**, not per game (`RLPlayer._n_decisions`
   is never reset; the observer copies it per finished battle — e.g. 27, 75, 120, 154 … on the away half). The core
   slot writes it per game (27, 48, 45, 34 …, equal to `our_argmax_decisions` on both paths). Any reader that summed
   or averaged the old field over rows read a triangular number. Pre-existing; a `rust_core_slot` row's
   `n_decisions` now means "this game".
2. **The legacy path stays for three shapes the slot cannot serve:** `--server node` / `--server-uri` (no in-process
   front end), a `bot:` our-side (a Python roster bot is a poke-env `Player`; the Rust bot ports run only inside the
   Rust env core, not on a `sim_bridge` slot). `--our-transport auto` (the default) picks it for those and PRINTS
   why; `src/main/anchors/session.py` therefore keeps its lazy poke-env imports, and only `runner.py` left the
   import allowlist (136 → 135 (after P5)). Retiring the rest needs P4 (`--server node`) and a Rust-bot slot (or dropping
   `bot:` our-side cells).
3. **One checkpoint, one opponent.** The identity was shown for a current-architecture checkpoint against
   `SmallRL`; the mechanism (row equality ⇒ action equality) does not depend on the opponent, but a checkpoint
   whose Python and core rows differ would show it here first. The plan's P4 "two roads, one row" gate is the
   standing instrument for row equality.
