# Offline collateral-KL — off-slice displacement for arms the live monitor cannot instrument

**Reproducibility, first, because it is the thing most easily misread.** This measurement is
byte-reproducible **only at `concurrency=1` with the four seeds pinned** — and it takes *both*
halves. Seeds alone are not enough: seeded at `concurrency=3`, two runs still produced 1193 vs 1141
states with arm levels up to `+0.043` apart, because interleaved battles consume the shared random
streams in a scheduling-dependent order. At `concurrency=1` two runs are byte-identical. The script
now **refuses** `concurrency != 1` (override with `OKL_ALLOW_CONCURRENCY=1`, accepting unquotable
levels). The three pre-pin artifacts are kept deliberately — the spread between them *is* the
evidence for the caveat.

Verify the canonical run by re-running it; the artifact should hash to
`e59abe295465a9a6798bd0d3e7b3219f0adad749b0cb2a63a1df718841adcdbf`.

```bash
export PYTHONPATH=$PYTHONPATH:src
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py \
  /tmp/okl.json 24 1
```

## 📌 The STATE RECIPE below now has an in-tree successor — `python -m main.untaught_meter`

**2026-09-06.** The two halves this file established — *all five global-RNG seams pinned* **and**
*`concurrency=1`*, with the refusal rather than a warning — plus the equal-weight cluster bootstrap
over the 8 teams on ONE shared resampling index set, are now the standing implementation in
`src/agents/training/untaught_meter.py` (CLI `src/main/untaught_meter.py`). `untaught_teams.json`
beside this README is that CLI's **default team manifest**, read in this order, and at `--seed 0`
its pool draw is `random.Random(61000 + ti)` — this script's, unchanged.

Two things the successor adds that this script could not:

* a **continuation-control column**. Ledger 2026-09-06 (cell 2) measured a plain continuation moving
  the untaught win rate +3.45pp [+0.46, +6.48] with no fold machinery at all, so a delta against a
  frozen parent overstates a fold by whatever the parent would have gained anyway. `--control`
  pools continuation arms and carries their own max-pairwise replicate floor.
* **ref resolution through the ONE choke point**, with the resolved file and its rung recorded — the
  defect probe H8 found (`best_model.zip` is not always a run's last checkpoint) is unrepresentable
  there.

**This script and its artifacts are NOT superseded and must not be regenerated.** It measures a
KL, not a win rate; its five artifacts are the evidence for the reproducibility caveats below, and
the three pre-pin runs exist precisely to show what an unseeded draw looks like. Read this file for
*why* the recipe is what it is; run the meter when you want a new number.

## Why this exists

The live `--distill-anchor-monitor` reports `collateral_kl_vs_parent`, but it needs the **off-slice
split**, which exists only when a distill term is live. A **coef-0 control arm** — C1, the loss-off
arm of the reuse batch — therefore resolves the instrument OFF *by construction*, no matter which
flags are passed. That is a structural limit, not a configuration mistake, so this offline
recomputation is the **standing method** for any loss-off arm, not a one-night workaround.

## Recipe

| | |
|---|---|
| states | the fold parent (`ai_v9_59_R2ACTION_0827/final_model.zip`) pilots each of the **8 untaught teams** against rev-1's 24M snapshot, 3 battles/team, capturing every decision. Off-slice **by construction** — the untaught 8 are disjoint from every teacher slice |
| statistic | `masked_kl_rows` **imported** from `agents.training.instrumented_ppo.distill_anchor`, never reimplemented: forward `KL(parent‖arm)` over legal actions, illegal → `-inf` on both sides |
| aggregation | per-team mean, then an **equal-weight cluster bootstrap over the 8 teams**, 20k draws, with **one fixed resampling index set shared by every arm** so an arm-vs-arm difference is paired on the same team draws |
| seeds | sim `[team+1,2,3,4]`; pilot policy `71000+team`; opponent policy `72000+team`; pool sequence `61000+team` |
| gate | the three dose arms must reproduce their **logged ordering** or the artifact stamps C1/B2 `UNINTERPRETABLE` |

## Two traps this file exists to stop

**Read `cluster_mean` / `cluster_ci95`, not `offline_collateral_kl`.** The latter is the
**state-weighted** mean and can disagree in **sign**: on the unseeded 1213-state draw, `C1 − B2` was
`+0.0309` pooled but `−0.0188` clustered — 6 of 8 teams had C1 below B2, and two teams that
contributed more states carried the pooled sign. Team-weighted is this program's unit everywhere
else; it is the unit here too.

**Never merge this column with the callback's logged column.** Same formula, different state
distribution — the callback accumulates over the fold's own rollout states, this scores one fixed
off-slice batch. The levels are not comparable and the offline ones sit far lower.

## Result (canonical seeded run, 1100 states, 8 teams)

| arm | offline (clustered) | what it is |
|---|---|---|
| R4DOSE12 | 0.3062 | dose 0.53× v8 |
| R4DOSE6 | 0.3502 | dose 1.06× v8 |
| R4DOSE3 | 0.4416 | dose 2.12× v8 |
| B2 | 0.3938 | the fold, coef 0.1761 |
| C1 | 0.3702 | **the loss-off control**, coef 0 |

Paired, on the same team draws:

| contrast | delta | 95% CI | verdict |
|---|---|---|---|
| `R4DOSE3 − R4DOSE12` | +0.1203 | [+0.0663, +0.1795] | **SEPARATES** |
| `C1 − B2` | −0.0245 | [−0.0841, +0.0267] | **spans zero** |
| `B2 − R4DOSE3` | −0.0392 | [−0.1066, +0.0174] | spans zero |

**Calibration: PASS** — the offline dose ordering (0.3062 / 0.3502 / 0.4416) reproduces the logged
ordering (0.5446 / 0.5832 / 0.6047), so C1 and B2 are interpretable.

So the meter **resolves the dose axis and does not resolve C1-vs-B2 at 8 clusters.** That is a limit
of *this* instrument. The untaught win-rate instrument *does* separate C1 from B2 (3/3 against the
pooled 4.27pp floor). **The two are not in contradiction** and must not be presented as such — one
meter is silent, it does not dissent.

⚠️ A dissociation claim ("C1 displaced *more* than B2, so displacement magnitude does not predict
untaught damage") was drafted off the first run's bare point values and **withdrawn before it was
sent**, once the cluster CI showed the difference spanning zero. The point estimates that produced
it are in `run_unseeded_948states.json`.

## Files

| file | states | seeds | conc | note |
|---|---|---|---|---|
| `run_seeded_conc1_1100states.json` | 1100 | pinned | 1 | **canonical** — byte-reproducible |
| `run_seeded_conc3_1193states.json` | 1193 | pinned | 3 | evidence that seeds alone are insufficient |
| `run_seeded_conc3_1141states.json` | 1141 | pinned | 3 | its replicate — 52 states apart |
| `run_unseeded_1213states.json` | 1213 | none | 3 | pre-pin draw; carries the pooled/clustered sign trap |
| `run_unseeded_948states.json` | 948 | none | 3 | pre-pin draw; the withdrawn dissociation came from here |

`../floor_reread.py` re-reads every untaught delta against the ruled **pooled floor 4.27pp**
(`FLOOR_OWN_DEPTH=1` reproduces the set-aside own-depth policy for checking).

⚠️ Any level quoted from an **unseeded** file is a draw, not a measurement: the same arm moved
`+0.04…+0.075` between the two unseeded runs. Only orderings and paired differences survive there.
