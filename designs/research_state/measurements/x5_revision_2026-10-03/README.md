# X5 design revision: the measurements behind it (2026-10-03)

The numbers the revised [`design_x5_belief_tokens.md`](../../../endstate/design_x5_belief_tokens.md) quotes for
its review items M1, M3, M4 and M5–M8. CPU only, read-only, no checkpoint loaded, nothing under `models/` touched.
Every script is seeded (including the QMC inside scipy's multivariate-normal CDF that sets the O'Brien–Fleming constant); a re-run reproduces every output byte for byte (checked by running each twice).

Run from `scripts/` with `PYTHONPATH=<checkout>/src:.` and the `gen3ai_torch28` interpreter.

| script | item | what it measures | output |
|---|---|---|---|
| `m1m3.py` | M1, M3 | On the 719 pool teams × random reveal orders (11,504 decisions; r = 0 rows are reported but never occur in a battle, because the lead is always revealed): (1) the LOGISTIC FIXED-SIZE marginals π = σ(a + τ), τ by a 64-step bisection, in fp64 and fp32, with and without a Newton step — the residual \|Σπ − k\| and the fp32-vs-fp64 gap; (2) the capped-πps rate and its infinite-BCE rate; (3) OTHER's mass and the recall of the true hidden mons by hypothesis budget, for three beliefs: the Smogon T0 prior (cold start) and two IN-SAMPLE pool-memorising proxies fitted per r to the banked head's on-pool NLL where reachable. | `out/m1m3.log`, `out/m1m3_out.json` |
| `gs_sim.py`, `gs_sim2.py`, `gs_sim3.py`, `gs_extra.py` | M5–M8 | The group-sequential non-inferiority design on the mirrored head-to-head CROSS (looks at 3, 5, 8 seeds per arm): O'Brien–Fleming z-boundaries mapped to t at the same nominal p (Pocock 1977), the boundary constant then scaled ×1.02 so the WORST simulated type-I error over σ ∈ {2.5, 3.43, 4.8} × σ_int ∈ {0, 1.5} is ≤ 0.05; type-I, power, futility stops and expected GPU-hours per σ and margin; the paired-by-seed design and a fixed n = 8 test for comparison; σ 5.21 and 8.0 (the A − A2 contrast). | `out/gs_*.log`, `out/gs_results.json` |
| `m7.py` | M4, M7 | σ_run from the A − A2 same-seed cross-pin pair and the pooled values; the SD of the four E10 levels; the power an outside-panel NON-INFERIORITY gate would have. | `out/m7.log` |

**Limits, stated where the note uses them.**
- The memorising proxies use SPECIES only. Even exact in-sample memorisation cannot reach the head's per-slot matched
  NLL at r = 1–2 (3.02 / 2.64 against the head's 2.24 / 2.51), so the live head may be sharper than any proxy here.
  Its OTHER mass under X5 is UNVERIFIED.
- The simulation's σ is the per-run SD on the HEAD-TO-HEAD scale. No H2H run floor is measured; the grid borrows the
  untaught meter's values.
