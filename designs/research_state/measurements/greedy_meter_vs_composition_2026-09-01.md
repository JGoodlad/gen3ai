# The rev-2 contradiction: METER and COMPOSITION each carry about half

**2026-09-01.** Pre-registered before collection (session relay, 2026-08-31 ~23:20), scored
without adjustment. Artifacts: `greedy_meter_inputs/greedy_{REV1FIN,R2ACTION}.json` (3,200 new
battles), collector `greedy_meter_arm.py`.

## 0. The contradiction this cell was built to settle

Rev-2's untaught hop had two published values that appeared to disagree:

| source | value | meter |
|---|---|---|
| probe Q (`rev3_untaught_pulldown_2026-08-30.md`) | **−7.06pp** [−10.56, −3.50] z=−3.86 | **greedy**, rev-1 `final_model.zip` (25M) opponent, **team set Q** |
| M9 (`plain_training_robbery_2026-08-31.md`) | **+0.88pp** [−1.62, +3.56] n.s. | **stochastic**, rev-1 `snapshot_000024000000.zip` (24M) opponent, **team set M** |

Same contrast (`R2ACTION − REV1FIN`), same n=200/team, same rust bridge — but three axes differ
at once, so neither number refutes the other and the campaign had been quoting the first while
scoring rev-4's REPRO-1 failure on the second. The sets overlap on **three of eight** teams.

## 1. Frozen predictions

> - greedy/set-M ≈ **−7pp** → the difference is **policy regime**
> - greedy/set-M ≈ **0** → the difference is **composition** (set Q selected its own answer)
> - anything in between → **both contribute, and neither confound can be dropped from any
>   untaught-8 claim**

## 2. Method — one flag, and only one

The collector is a **verbatim copy** of the standing meter
(`axis_split_untaught_arm.py`) with `stochastic=True → False`. The full diff is the two code
lines plus the docstring sentence that describes them; nothing else was touched, because that
file's own docstring warns that re-implementing the meter is what makes a result
uninterpretable. Everything else is inherited unchanged: fixed rev-1 @24M target drawing from
the validated 719-team pool, team set M in its seed-bearing order, seed family `1000 + 9 + i`,
n = 200, rust bridge, CPU, `nice -n 15`, per-team resumable.

Both arms (`G_REV1FIN`, `G_R2ACTION`) ran sequentially, deliberately chained behind M9's
binding cell so they took no cores from it.

## 3. Result

| team | greedy rev1 | greedy r2a | Δ | stoch rev1 | stoch r2a | Δ |
|---|---|---|---|---|---|---|
| `U_61590463` | 0.590 | 0.520 | −7.0 | 0.585 | 0.555 | −3.0 |
| `U_90b94599` | 0.565 | 0.525 | −4.0 | 0.575 | 0.585 | +1.0 |
| `U_92832108` | 0.515 | 0.495 | −2.0 | 0.550 | 0.510 | −4.0 |
| `U_9909f2e9` | 0.585 | 0.565 | −2.0 | 0.575 | 0.640 | +6.5 |
| `U_9d5f8458` | 0.625 | 0.535 | −9.0 | 0.565 | 0.545 | −2.0 |
| `U_ce35b736` | 0.555 | 0.480 | −7.5 | 0.565 | 0.565 | +0.0 |
| `U_dbf81d8e` | 0.605 | 0.660 | **+5.5** | 0.595 | 0.660 | **+6.5** |
| `U_f7ba5702` | 0.560 | 0.545 | −1.5 | 0.580 | 0.600 | +2.0 |

| cell | rev-2 hop | 95% CI (bootstrap over teams) | z |
|---|---|---|---|
| **greedy / set M** *(this probe)* | **−3.44pp** | [−6.19, −0.31] | **−2.29** |
| stochastic / set M (M9) | +0.88pp | [−1.62, +3.50] | +0.67 |
| greedy / set Q (probe Q) | −7.06pp | [−10.56, −3.50] | −3.86 |

## 4. Verdict — the middle branch, split almost evenly

The full gap is **7.94pp** (−7.06 → +0.88). This cell splits it:

| component | held fixed | contrast | size |
|---|---|---|---|
| **policy regime** | team set M | greedy −3.44 vs stochastic +0.88 | **4.32pp** |
| **composition** | greedy | set Q −7.06 vs set M −3.44 | **3.62pp** |

Neither confound dominates; each carries roughly half. **Both must therefore be stamped on
every untaught-8 number in the ledger**, and no greedy result may be quoted beside a stochastic
one, nor any set-Q result beside a set-M one.

## 5. What this does NOT rescue

**−3.44pp sits inside the replicate floor.** M9 measured two byte-identical no-fold runs
separating by **4.19pp** on this same untaught meter (and 3.70pp on the taught-9 meter). The
greedy/set-M robbery is smaller than that, so even on the regime that maximises it, rev-2's hop
does not clearly clear what two identical runs produce by themselves. It is directionally
negative on 7 of 8 teams, which is worth something the pooled magnitude is not — but a
single-run arm cannot separate the two.

Per-team sign agreement between the regimes is **5 of 8**, near chance, and the one strongly
positive team (`U_dbf81d8e`, +5.5 / +6.5) agrees across both. So the regimes are not measuring
noise *independently* — they share the extremes and disagree in the middle, which is the
signature of a real but small effect read through two different samplers.

## 6. Consequences

1. **Probe Q's −7.06 is not "the" rev-2 number** and was never comparable to the stochastic
   family it has been quoted against. It is a valid greedy/set-Q measurement.
2. **"Every gen-era fold robs" fails on its anchor case** under any meter-consistent reading:
   rev-2 is +0.88 (stochastic) or −3.44 (greedy, inside the noise floor). Only rev-4's −6.50
   with its floor-concentration signature (z=−4.32) clearly survives.
3. **The remaining honest confound is the OPPONENT checkpoint** — probe Q used rev-1
   `final_model.zip` (25M), this cell uses the `snapshot_000024000000.zip` (24M). It is folded
   into the "composition" column above and is NOT separately identified. A third cell (greedy /
   set Q / 24M opponent) would isolate it; it was not run, and the 3.62pp attributed to
   composition should be read as *composition + opponent checkpoint*.

## 7. Cross-references

`plain_training_robbery_2026-08-31.md` (the replicate floor and rev-4's surviving robbery) ·
`axis_split_taught_untaught_2026-08-31.md` (the stochastic family's fold table) ·
`rev3_untaught_pulldown_2026-08-30.md` (probe Q, the greedy family).
