# K9(b) FLIP-JUDGE on the static-screen u1480 stop (2026-10-08)

**Question.** `rb_st_static_s1001` stopped at update 1480 on K9(b)'s excluded-share ceiling alone, because one
Salamence hypothesis's cut pair sat in a near tie that `--token-encoding static` repeats in every row of that species
([`k9_static_tie_2026-10-08/`](../k9_static_tie_2026-10-08/README.md), F-ST-10). The flip-judge
(`gen3_behaviour_tie_flip_judge_v1`) judges such a row under BOTH resolutions of its one tie instead of excluding it.
What does the dump's excluded share fall to, what stays excluded, and what does it cost?

## The rule as built

- A current row EXCLUDED only because its near ties sit at ONE element of ONE recorded selection call is
  JUDGED instead. One element means one near pair of X5's sort, or one threshold element.
  - A sort pair qualifies only when it is ISOLATED: both neighbouring sorted pairs sit at least epsilon away.
    Three near-tied keys have more than two resolutions.
  - The recorder (`tie_margins.TieMargins`, `near_eps`) notes it as `flippable`.
- **Both resolutions.** The first is the one the probe's own forward took. The second comes from ONE extra probe
  forward of the same rows under `tie_margins.TieFlip`, with that element resolved the other way: the pair's two
  candidates swapped, or the bool inverted.
  - The second forward runs only when some such row fails the bar as-is.
  - The row PASSES if either resolution's |log π − log μ| is under the bar.
  - It FAILS (a mismatch, FATAL like any judged row) if neither is.
- **Out of scope:** rows with near ties at two or more calls or elements, an `argmax` / `topk` / `threshold_self`
  tie, a NaN margin, and a non-isolated pair. These stay EXCLUDED as before.
- **Unchanged:** epsilon (2e-4), the bar (1e-4) and the ceiling (0.15).
- **Reporting:** the excluded share is reported AFTER the flip-judge. `behaviour/excluded_frac_before_flip`,
  `rows_judged_by_flip`, `rows_flip_forward`, `rows_flip_resolved` and `flip_forward_ms` sit beside it.
- **The flip forward must reproduce the recording one.** It needs the same MARGIN call sequence up to the last flip
  and the same two tied candidates, and every row at no tie must stay within the bar of its first value (rows are
  independent). Otherwise it raises `TieMarginError`: FATAL under `fatal`, and under `warn` the rows stay excluded.

## Method (CPU only, no GPU lease)

`replay.py`: the `k9_static_tie` harness (the `k9_early_probe` build plus the `static_fm` arm). It plays a seeded
2,048-row complete-game Rust-collector rollout (16 envs × 128 steps, T2 eager on CPU, a seeded random p2) at the
given weights, then runs the REAL probe (`consistency.behaviour_probe`, `warn`) on every row as one micro-batch.

A FORCED read follows. It resolves EVERY flippable row the other way in one `TieFlip` forward and compares that
resolution with the stored behaviour log-prob. It also times the bare forward, the recorder before this change and
the recorder with near tracking, each as a median of 3.

- **At the PIN (the real dump):** a scratch detached `6c6d2e09` worktree with this commit's `tie_margins.py` and
  `consistency.py` dropped in. The pin's own copies are byte-identical to this commit's parent, so the flip-judge
  code is exactly HEAD's. The weights are `models/rb_st_static_s1001/behaviour_violation_u1480_policy.pt`, read
  only. Output: `replay_pin_u1480.json`.
- **At HEAD (a constructed reproduction):** HEAD (v145) cannot load the dump (v143) as it is. The version break
  ties the op's `out_gain` from 138 per-position gains to 29 (max within-tie spread 0.137) and deletes the dead
  value tower. `convert_u1480.py` copies the other 748 tensors bit for bit, maps each tied gain to the mean of the
  positions it covers, and writes the result outside `models/`. Output: `replay_head_u1480.json`.

Each run: `scripts/ops/mem_cap.sh 24`, `CUDA_VISIBLE_DEVICES=`, about 3 min, peak 2.8 GB. torch 2.8.0+cu126.

## Result

| | before (excluded) | flip-judged | after (excluded) | judged max \|Δ\| |
|---|---|---|---|---|
| **pin, the u1480 dump** | 209 (**10.2 %**) | 156 | 53 (**2.6 %**) | 5.5e-6 |
| HEAD, constructed reproduction | 212 (10.4 %) | 144 | 68 (3.3 %) | 4.3e-6 |

At the pin, 209 matches the root-cause replay's 209 row for row. The 156 flip-judged rows are X5's sort pairs: the
154 Salamence cut-pair rows plus 2 other cut pairs. What stays excluded (53):

| site | rows | why it stays |
|---|---|---|
| `hypothesis_set.py` argsort | 38 | the bit-equal typed-HP presence ties at ≈ 1e-9: three or more keys tie, so the pair is not isolated |
| `damage_op.py` argmax (dominant move) | 10 | an `argmax` tie is not expressed |
| `damage_kinds.py` `fixed >= hp` | 4 | near at two or more elements |
| `hypothesis_set.py` argsort | 1 | near at two calls |

HEAD reads the same mix (45 / 7 / 15 / 1).

**Forced read.** At the pin, the other resolution moved log π(a|s) on 155 of the 156 rows (max 4.3e-3, median
7.3e-5). That is the taken action's log-prob; the root-cause README's 1.25e-2 is over the full masked distribution.
Every flip-judged row passed AS IS, so the probe ran no flip forward: in this harness T2 is the same eager CPU
forward as the learner. Rows at no tie moved by exactly 0 in the flip forward. On the live GPU path T2 is compiled
and can take the other resolution, which is the case the flip forward exists for. `consistency_test` plants it.

**Cost** (CPU, 2,048 rows, median of 3):

| forward | time |
|---|---|
| the bare probe forward | 2.28 s |
| under the old recorder | 2.88 s |
| under the recorder with near tracking | 2.27 s |

The near tracking is within this box's noise. The flip forward, when it runs, is ONE more probe forward of the same
rows (2.07 s here), at most once per update and only on an update where a flip-judged row fails as is. On the GPU
the probe forward measured 62.6 ms plus 27.7 ms for the recorder (`k9_behaviour_exclusion`), so the bound is
≈ 0.1 s, under 0.25 % of a ≈ 40 s update. **UNVERIFIED on the GPU**: no GPU lease was held.

## Not claimed

- **The live composition.** The live probe read 17.5 % at u1480 where this harness reads 10.2 %. What the
  flip-judge would have left live is **UNVERIFIED**; the harness's 10.2 % → 2.6 % is the measured statement.
- **What fires on the GPU.** No GPU run was made, so how often the flip forward fires there, and how often the
  flipped resolution is the one that passes, is **UNVERIFIED**.
- **The other stays.** The 38 typed-HP floor ties and the argmax ties stay excluded. Expressing them (a three-way
  sort cluster, an argmax runner-up) is not built.

## Files

`replay.py`, `convert_u1480.py` (drivers), `replay_pin_u1480.json`, `replay_head_u1480.json`.
