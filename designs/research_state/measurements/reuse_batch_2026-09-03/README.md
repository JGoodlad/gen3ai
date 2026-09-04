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
