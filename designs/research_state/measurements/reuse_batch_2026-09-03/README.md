# Reuse batch (2026-09-03): C1 vs B2 per-team rows, the shared baseline, the rev-4 reference

Per-team untaught rows behind the ledger entries "HEADLINE — C1 (loss OFF, 40% team bias ON) DOES
NOT ROB" and "C1 vs B2 COMPLETE, three depths". Written by the training session's scorer; copied
here with sha256 verified against its manifest (first 16 hex chars listed below). Instrument stamp:
stochastic policy · rev-1 24M opponent snapshot · team set M (8 untaught teams) · n=200/team ·
CRN prefix — the same stamp as `../dose_cell_2026-09-02/`.

| file | cells | total | sha256[:16] |
|---|---:|---:|---|
| `untaught_R2ACTION.json` | 8 | 932/1600 | bf6107a1ba5b50c0 |
| `untaught_R4ACTION.json` | 8 | 828/1600 | b7c7a94217bb2c16 |
| `untaught_C1_p1M.json` | 8 | 952/1600 | 30ee08eb07e89f24 |
| `untaught_C1_mid.json` | 8 | 932/1600 | 83f9a778e8f0988f |
| `untaught_C1_end.json` | 8 | 972/1600 | 89e622175bdf42ae |
| `untaught_B2_p1M.json` | 8 | 790/1600 | e19f34a1b7bfafe3 |
| `untaught_B2_mid.json` | 8 | 838/1600 | 8a39dea90a0f251c |
| `untaught_B2_end.json` | 8 | 888/1600 | 74a0ecd1523a06b2 |

Three things a later reader needs, or the files mislead:

1. **The baseline is part of the measurement, not context.** Every Δ in the ledger tables is
   against `untaught_R2ACTION.json` (the fork parent, 932/1600 = 0.5825), measured ONCE and reused
   across the whole batch by design — the dose-cell rows use the same file. Without it the rows
   are uninterpretable.
2. **`untaught_C1_mid.json` totals 932/1600 — the SAME total as the baseline.** It is not the parent
   scored by mistake: the shas differ, the per-team rows differ by −14…+17 wins, and the scorer's
   log names `ai_v9_141_C1_0901/checkpoints/checkpoint_30065184_steps.zip`. A coincidence, verified
   twice before it was reported. Anyone diffing totals will suspect a mix-up; this line exists so
   they check the rows instead.
3. **The depth labels are not the dose arms' depths.** C1 and B2 live on the 150k checkpoint grid,
   so `p1M` = +1.05M and `mid` = +1.95M steps past the fork, against the dose arms' exact +1.00M /
   +2.00M on the 500k grid. Within the C1–B2 pair the depths are matched exactly (same grid, same
   step numbers); across cells they differ by 0.05M. `end` is +4.45M on both.

`untaught_R4ACTION.json` is rev-4's ENDPOINT only; no p1M/mid rows of rev-4 exist on this instrument.
N1/N2 (the narrow fold-replicate pair, six files) are added here with their own manifest lines when
scored.

## N1/N2 — the narrow fold-replicate pair (added 2026-09-04)

`ai_v9_142_N1_0901` / `ai_v9_143_N2_0901`: byte-identical argvs (one token apart, the `--run-name`
value), seed 42 both, pin `19d25a11` both, coef 0.1761, target `action`, gas=2, **controller LIVE**
(no `--fork-lr`, no freeze) — so their spread is the fold-replicate floor for controller-live folds
at gen-era dose, not for frozen-dose folds. Their 6 taught teams are disjoint from the untaught 8.

| file | sha256[:16] | file | sha256[:16] |
|---|---|---|---|
| `untaught_N1_p1M.json` | 486a5c32e028ae13 | `untaught_N2_p1M.json` | 7fb39346e5adddb8 |
| `untaught_N1_mid.json` | c889a099f51bccd7 | `untaught_N2_mid.json` | 569a2674371210bd |
| `untaught_N1_end.json` | 38c9e4760785d194 | `untaught_N2_end.json` | a92f52721c29c10a |

N1 − N2: p1M +4.62 [−0.12, +9.87] · mid +2.25 [−0.81, +5.13] · end +5.94 [+1.94, +10.25]; the three
are not distinguishable from each other; pooled over the three depths 4.27 [+1.23, +6.92] — the bar
ruled for this instrument (ledger 2026-09-04 00:50). Re-read script: the training session's
`floor_reread.py` (takes depths as argv; loss-off arms take `max(4.19, floor)`).
