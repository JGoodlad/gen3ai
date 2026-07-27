# Retention ablation — does distilled per-team skill survive without the teachers?

**Status: ANSWERED (arm A′, 2026-07-26). Distilled skill does NOT wash out — it decays to a stable
equilibrium at ~76% of the distilled gain, with NO retention mechanism of any kind.**

## The question

The exploiter→distill loop raises per-team piloting. But if the skill decays once the teachers are
removed, the programme is O(N teachers) **forever** — you can never retire a teacher, and the loop
does not scale. The owner posed this directly: *"we forever have to keep all of the existing distillers
around because as soon as we remove them from the training pool over time we're gonna likely regress."*

## The framing that made it answerable

Separate **acquisition** from **retention**, and separate two causes of decay:

- **Forgetting** — the skill is still valuable but the maintenance gradient loses to noise/interference.
  Bad; worth fixing.
- **Obsolescence** — the skill was an exploit of a *specific* opponent, and as self-play moves away it
  stops paying. **Correctly discarded**; fighting it would be harmful.

Obsolescence is driven by the opponent distribution moving, which in self-play means **pool promotions**.
So arm A′ **freezes the pool** — any decay observed there is forgetting, not obsolescence.

## Design (arm A′)

Forked from the distilled `ai_v8_14` checkpoint (`checkpoint_292100648`), run
`ai_v8_15_retention_A_frozen_0726`, with every retention aid removed:

| | setting |
|---|---|
| teachers (distillation) | **none** (`--distill-*` removed) |
| teachers as opponents | **none** (`--stable-opponents` removed) |
| team-PFSP | **off** (`--team-pfsp measure` — tracks, never biases) |
| pool | **FROZEN** — `--promote-threshold 1.1` (unclearable, so no promotions) + the parent's `snapshots/` copied in |
| everything else | identical to the distillation run (arch flags, `team-block-episodes`, `film-grad-accum`) |

**Measurement** (`tmp/retention_probe.sh`): per-team win rate on the 10 taught teams, piloted **greedy**
vs a **FIXED `ai_v8_04` reference**. The fixed reference is the essential design choice — measuring
against the evolving pool would confound the curve with a drifting yardstick. 40 games/team (±~0.045).

**Interpretive bounds:** un-distilled floor **0.438** (ai_v8_04 on these teams) · distilled baseline
**0.710** · def-10 specialist ceiling **~0.72**.

## Result

| steps since distill | WR on the taught teams | Δ |
|---|---|---|
| 0 | **0.710** | — |
| +3M | 0.6875 | −0.023 |
| +6M | 0.6450 | −0.043 |
| +10M | 0.6425 | −0.003 |
| +15M | **0.6475** | +0.005 |

**Decay happened EARLY, DECELERATED, and STOPPED at ~0.645** — three consecutive points flat within
noise across 9M steps.

**Retention = (0.645 − 0.438) / (0.710 − 0.438) ≈ 76%** of the distilled gain, unaided.

## Interpretation

1. **Teachers can be RETIRED.** The model does not regress to the old plateau. The O(N)-scaffolding
   fear is dissolved; the loop is `generalist → exploiters → distill → retire`.
2. **Distillation is BOOTSTRAPPING, not life support.** The acquisition/retention split held up
   empirically — which also means the plateau was **optimization difficulty**, not objective
   indifference (a skill the objective did not value would have decayed to the floor).
3. **Retention is an EQUILIBRIUM, not a binary.** This is the leaky-bucket prediction confirmed:
   decay stops where the restoring force (`∝ P(team) × value of the skill`) balances erosion
   (interference + gradient-noise diffusion). Not perfect retention, not collapse — a *level*.

## What it re-prices

| arm | prior role | new role |
|---|---|---|
| **B** (`--team-pfsp onesided`) | prevent collapse — necessary | **optional**: recover the last ~24% (0.645 → 0.71) by raising the equilibrium |
| **C** (teacher-as-opponent) | the cheap retention rung | **lower priority** |
| **D** (always-on distillation) | the feared default | **not needed** |

Theory note for arm B: PFSP raises `P(team)` for weak teams by only **~2–3×** (cap-bounded), so it should
raise the *equilibrium*, not eliminate decay. It is also a **lagged controller** (EMA β=0.5, every 3
rollouts), so it can only reject disturbances slower than its loop time.

## Caveats (honest)

- 400 games/point (±~0.045); the flat conclusion rests on three points, not a dense curve.
- The frozen pool held only **2 snapshots** — stationary but narrow, so the opponent distribution is
  thinner than a normal run's.
- 15M steps is a modest horizon; decay slower than the noise floor cannot be excluded.
- Arms A (normal pool), B, C were **not run** — this is the control arm only. The normal-pool arm would
  additionally include obsolescence, so its decay could legitimately be larger *and that would be fine*.

## Reproduce

```
tmp/launch_ai_v8_15_retention.sh          # arm A' (frozen pool, no aids)
tmp/retention_probe.sh <run_dir> <games>  # per-team WR vs the FIXED ai_v8_04 reference
tmp/retention_curve.sh                    # auto-samples at +3M / +6M / +10M / +15M
```

**Operational gotcha:** offline probes contend with training for cores (load hit 28 on a 16-core box),
so a 400-game probe took 10–50 min depending on contention, and one probe **died silently** mid-run.
Any orchestration around this must verify a *new* line was written rather than re-reading the last one.

## See also

- Memory: `project_multiteam_distill_payoff` (the distillation that produced the model),
  `project_exploiter_fork_vs_scratch` (the count sweep + fork≫scratch).
- `designs/ai_v8/exploiter_batch_strategy.md` — the retention protocol + the acquisition/retention framing.
- `designs/learning/conditioning_architectures.md` §5b — why the conditioning path is the bottleneck.
