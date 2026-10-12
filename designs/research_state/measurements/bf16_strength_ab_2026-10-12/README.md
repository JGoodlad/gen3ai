# bf16 STRENGTH A/B — measurements

Registration: [`designs/endstate/design_bf16_strength_ab.md`](../../../endstate/design_bf16_strength_ab.md)
(REGISTERED 2026-10-12, `981013f4`, before any seed). EB = `--arch endstate --trunk-precision bf16`, E = `--arch endstate`
(fp32), both at P_bf16 `f45d0c27`.

- [`futility_rule.py`](futility_rule.py) — stage 1's decision rule (§4), CODED BEFORE ANY CELL IS PLAYED: the 4 × 4
  cross of the 5M finals (rows EB 2001–2004, columns E 2001–2004), `main.h2h.cross.cross_stat` at n = 4 (df 6),
  the two-sided 90 % interval Δ̂ ± t_{0.95,6} √V̂ (t = 1.9432); STOP (futility) iff its upper end < −2.0 pp, rule 8 at
  1e-9 (a tie does not stop); otherwise CONTINUE to stage 2. `python futility_rule.py self-check` (run from a
  checkout's `src`) reads three constructed matrices: CONTINUE, STOP, and the rule-8 tie (CONTINUE, `near_boundary`).
- Stage 2's rule is the closing test's coded rule, unchanged:
  [`endstate_closing_test_2026-10-09/closing_rule.py`](../endstate_closing_test_2026-10-09/closing_rule.py) `decide`.

The cross artifacts (plan, shas, cells, h, verdict) are added here when each stage's cross is read.
