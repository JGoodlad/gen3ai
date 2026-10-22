# G1, AM-2's variance gate: PASS (2026-10-22)

**σ̂ = 0.408 pp on 4 df, against the 1.2 pp bar: PASS** (`../x35_gate_g1.json`, written once by
`python -m main.x35.kit gate g1_input.json --stage .. --write`, exit 0). It reads only variance, never an arm's Δ̂.

| final | step | S (pp) | se (pp) |
|---|---|---|---|
| C0 s3501 | 300,061,940 | 57.52 | 0.55 |
| C0 s3502 | 300,061,805 | 56.60 | 0.56 |
| C0 s3503 | 300,061,491 | 56.81 | 0.55 |
| A1 s3501 | 300,060,532 | 57.51 | 0.55 |
| A1 s3502 | 300,061,732 | 57.11 | 0.55 |
| A1 s3503 | 300,061,979 | 56.89 | 0.55 |

- **The spread is measurement noise.** The mean se of one S is 0.55 pp, larger than σ̂ (0.41), so no seed-to-seed
  spread beyond the meter is visible. The X26 drift pre-read (`../x26_drift_preread/`) put σ near 0.55–0.8 pp.
- **Without A1 s3503 (descriptive only):** σ̂ = 0.427 pp on 3 df. Its u200 CompileCanary was UNCONFIRMED, and the
  orchestrator ruled the final FIT (the fixture class) before the gate was written.

## How it was read

- S = `main.x35.readout.population_score` against the registered R (`../reference_population_production.json`, content
  `cef0960e…`, 7 members). Each cell is 500 mirrored pairs on `main.h2h play-many`: greedy vs greedy, the eval team
  builder (no panel), CPU, `torch_threads` 1. The regime id is `adfdbee824c9eb0a`, the R freeze's and the pre-read's own.
- se(S) = sqrt(Σ se_m²) / 7, the pre-read's declared per-final interval.
- The driver is [`g1_read.py`](g1_read.py), the pre-read's machinery: each final (`models/<run>/final_model.zip`, step
  ≥ P\* + 20M) is copied with its sha256 checked, and is read only after its `X35END: DONE` line. R is re-checked
  before every unit. Units are 40-minute, single-heavy-slot chunks at `nice 19` on the agent cores.
- The ledger is `~/gen3ai_archive/x35_g1/ledger` (family `x35_g1_2026-10-22`). It holds 210 rows and 21,000 pairs, all
  at commit `3f3a2ed41` and none voided. [`results.json`](results.json) holds every per-member score and se.

## Hazards (FINDINGS)

1. **A1 s3502 was still training at the start** (10:31). The driver waited for its DONE line, so nothing changed.
2. **The playing code is the read's worktree commit `3f3a2ed41`** (main `9e49c0f1` + the AM-X52-i′ note), not an X35
   pin. Its regime id equals the R freeze's.
3. **`results.json`'s `regime_ids` field is empty**: the ledger's `Cell` carries no regime attribute. The regime was
   checked on the raw rows instead (all 210 rows read `adfdbee824c9eb0a`).
