# R0-prod TRANSFER read: is the best responder's edge P\*-specific, or general piloting of its 5 offense teams?

## 1. Registered 2026-10-22, BEFORE any game of this read

**The question.** R0-prod's best responder (BR, `rb_x35_r0_s3531`) beats its target P\* (X26 280M) by H = +11.1 pp,
with a LEARNED part of +13.6 pp (H − gap(0); `../curve_rb_x35_r0_s3531.json`). Is that P\*-specific exploitation, or
did the BR learn to pilot its 5 offense teams better against anyone?

**The test.** Play the BR's START (k = 0, P\*'s own weights, `rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`)
and its k = 20M file (the curve's final read point, `rb_x35_r0_s3531/checkpoints/checkpoint_300000097_steps.zip`)
against an opponent that is NOT P\*. The procedure is the curve's own (`main.x35.r0.play_curve`): the BR side pilots
its recorded offense pin (5 teams), the opponent draws from the pool, greedy vs greedy, 80 games per team = 400 a file,
seed 20261020 (the curve's `READ_SEED`), Rust eval core (`untaught_meter.play_cells`). The deals are the curve's (the
meter's seed depends on (seed, team index) only).

- **Opponent 1 (primary):** production C0 s3501's final, `rb_x35_c0_s3501/final_model.zip` (300,061,940 steps).
- **Opponent 2 (secondary, if CPU time allows):** pre-plateau X26 230M,
  `rb_x26_s2601/checkpoints/checkpoint_230000074_steps.zip`.

**The statistic.** Transfer(O) = P(BR k=20M beats O) − P(BR start beats O), Newcombe 95 % interval
(`agents.training.best_response_gap.newcombe_diff_ci`), ties are not wins. The reference is the learned part vs P\*,
L = +13.6 pp; the half bar is 0.5 × L = **+6.8 pp**.

**The bands (deterministic; rule 8 at 0.0005).**
- **TARGET-SPECIFIC** iff upper(Transfer) < 0.068 − 0.0005.
- **GENERAL** iff lower(Transfer) > 0.068 + 0.0005.
- **MIXED** otherwise.
- Any file with > 25 % timeouts: **INCONCLUSIVE**.

Opponent 1 is classed; opponent 2 is classed by the same bands but reported as secondary. Also reported,
never classed: the matched single-point learned part vs P\* (k = 20M − k = 0 = (242 − 190) / 400 = +13.0 pp), and
Transfer / L.

**Named before playing.** C0 s3501 is a fork of P\* itself (+20M of production self-play), and X26 230M is P\*'s
ancestor: neither is outside P\*'s lineage, so a GENERAL verdict shows transfer to P\*'s family, not to every
policy. No strong production opponent outside this lineage exists at the pin.
