# X26 (70M) vs the toy at its 100M cap: INTERIM, DESCRIPTIVE (results)

Registered before any game in [`REGISTRATION.md`](REGISTRATION.md) (`60c36f37`). Played 2026-10-16 19:09–19:43 PDT (34 min,
≈ 1.5 games/s) on CPU under rule 5b, at main `d5880c87`'s `src/`. Read by [`read_cells.py`](read_cells.py) →
[`result/read.json`](result/read.json). Family `xvt_interim` in `models/_ledger`. The read script had a one-line orientation bug
(1 × 3 matrix indexed as 3 × 1) fixed after the games and before any number was read; no game was affected.

**Owner's question:** is production stronger than the toy? **Answer at this point: yes, modestly.** X26's 70M checkpoint scores
**+4.4 pp** (90 % [+1.5, +7.3]) against the toy's 100M-cap finals.

**This is DESCRIPTIVE and INTERIM. It is NOT R-T2** (`design_x26_readiness.md` §3.5), which is pre-registered for X26's PLATEAU
checkpoint; X26 is a live run and still climbing, and the toy is still climbing at its cap too. No §3.6 class is read from this.

## The numbers (production is the player; mirrored pairs, 500 per cell, 0 aborted games)

| toy opponent (100M cap) | production W / L / D (1,000 games) | production score | − 50 |
|---|---|---|---|
| `rb_toy_ent_s4001b` | 549 / 443 / 8 | 55.30 % | +5.30 pp |
| `rb_toy_ent_s4002` | 522 / 474 / 4 | 52.40 % | +2.40 pp |
| `rb_toy_ent_s4003` | 552 / 442 / 6 | 55.50 % | +5.50 pp |
| **pooled (3,000 games)** | 1,623 / 1,359 / 18 | **54.40 %** | **+4.40 pp** |

Seeds-as-units interval (the three toy seeds are the units; one-sample t, df 2, SE 1.00 pp):
- **90 %: [+1.48, +7.32]**, 95 %: [+0.09, +8.71].
- Production's own seed variance is NOT in the interval: there is ONE X26 seed and ONE checkpoint (the n = 1 column limit of
  [`../toy_crosses_2026-10-16/rect_stat.py`](../toy_crosses_2026-10-16/rect_stat.py)). With df 2 the interval is wide.

## Read in context (descriptive, all against different references)

- The toy at its 100M cap scored +10.2 pp against production at 15M (R-T6, [`../toy_crosses_2026-10-16/RESULTS.md`](../toy_crosses_2026-10-16/RESULTS.md)).
  Production at 70M beats the same toy checkpoints by +4.4 pp. So production's lead over the toy has grown from "behind at 15M"
  to "ahead at 70M", but the two reads use different opponents and do not form one curve.
- X26 at 70M is far from its plateau, so R-T2 (plateau against the toy's cap) can only be larger in production's favour if X26 keeps
  climbing; this read says nothing about that.

## FINDINGS (standing rule 7)

1. **Not R-T2, no class.** One production seed, df 2, a live run. Do not quote +4.4 pp as G_p(cap).
2. **Seed spread:** s4002 is the weakest toy for production (+2.4 pp vs +5.3 / +5.5); the cell-level sampling noise alone is ≈ 1.6 pp
   (SD of a 1,000-game score), so the three cells are not clearly different from one another.
3. **The X26 file is a live run's checkpoint** (`checkpoint_70000008_steps.zip`), read-only; all four references load at main
   `d5880c87` (strict loader, historical kwargs).
