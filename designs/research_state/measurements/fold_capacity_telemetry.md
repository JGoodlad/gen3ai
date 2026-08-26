# Fold-era capacity telemetry — the first real read of `capacity/*`

**Date:** 2026-08-25 · **Ordered by:** the ledger's `978b1aa` sharpness-probe entry ("the telemetry
read is ORDERED — the instrument was built for exactly this and then forgotten in its first real
incident").

**Question.** The surviving finding from the sharpness probe is that 3M-step tocks moved to a
GLOBALLY different policy (KL(teacher‖base) ≈ 0.43–0.48 nats on-pin, 0.36–0.41 off-pin). The
proposed mechanism is a **TUG-OF-WAR**: the distill KL drags the shared trunk toward that
different function while PPO pulls elsewhere. Every fold run carried `--capacity-telemetry`, so
the mechanism is testable on disk without a new battery.

---

## 0. Headline

**Signature fired: INTERFERENCE, WEAKLY — on one of the two meters that the interference cell
needs, and with the other meter's evidence pointing the wrong way. Half the registered battery
(the COLLAPSE meter) never produced a single sample in any run, so the collapse cell is
UNTESTABLE, not refuted.**

Three things, in order of how load-bearing they are:

1. **`capacity/feature_velocity{,_cos,_rel}` is ABSENT from all six runs — zero samples, ever.**
   Not "flat", not "small": the tag does not exist in any tfevents file. Cause is a
   cadence/restart-window interaction (§2). The COLLAPSE signature ("velocity falling at constant
   grad_norm") therefore **cannot be evaluated at all**.
2. **`halfbatch_cosine` is significantly LOWER in every distill arm than in the coefficient-0.0
   control**, over an exactly matched step grid from a common fork: pooled **−0.0304, 95% CI
   [−0.0580, −0.0122], p = 0.001**. Direction 4/4 across arms. This is the interference meter, and
   it fired.
3. **The canary half of the interference prediction did NOT fire.** `canary_recovery` supports no
   claim at the honest granularity (n = 2 reset episodes per arm; §6), and `canary_loss` moves the
   **opposite** way from the prediction — the distill arms fit the canary *better*, at every
   single logged step.

And the sobering context: **tick-1's own 10M collapse window is FLAT on every capacity scalar**
(§8). The instrument distinguishes the arms from their control; it does not track the collapse
trajectory.

---

## 1. What was read, and the reader trap

TB events span multiple files per run (one per launcher child) and counters are per-child. Every
number below is a **union over all `events.out.tfevents*` under `<run>/tb/`**, deduped on
`(tag, step, child)`.

| run | label | tfevents files | children with data | capacity points | step span |
|---|---|---|---|---|---|
| `ai_v9_29_rev1_0823` | rev-1 | 6 | **5** | 248 | 0.197M – 25.068M |
| `ai_v9_34_tick1_0824` | tick-1 | 3 | 3 | 97 | 25.264M – 35.095M |
| `ai_v9_38_fdA_coef03_0825` | fdA | 1 | 1 | 31 | 25.264M – 28.115M |
| `ai_v9_39_fdB_lossonly_0825` | fdB | 1 | 1 | 31 | 25.264M – 28.115M |
| `ai_v9_40_fdC_ecology_0825` | fdC | 1 | 1 | 31 | 25.264M – 28.115M |
| `ai_v9_42_fdE_single_0825` | fdE | 1 | 1 | 28 | 25.264M – 27.919M |

⚠️ **One of rev-1's six event files carries ZERO scalars** (`…3766534.0`, max_step 0 — a child that
wrote a file and nothing else). A reader that picks "the largest file", "the newest file" or "the
one with data" would silently return 41 of rev-1's 248 points. Union, always.

Per-child point counts (this is also the train()-call count — capacity scalars are logged once per
`train()`):

```
rev-1    5 children  [c0 n=61 0.20-6.00M] [c1 n=53 6.29-11.40M] [c3 n=45 11.70-16.00M]
                     [c4 n=48 16.22-20.84M] [c5 n=41 21.14-25.07M]     (c2 = the empty file)
tick-1   3 children  [c0 n=44 25.26-29.49M] [c1 n=40 29.79-33.62M] [c2 n=13 33.92-35.09M]
fdA/fdB/fdC  1 child  n=31 ·  fdE  1 child  n=28
```

**Tags present in all six runs:** `canary_age`, `canary_loss`, `canary_loss_reset`,
`canary_recovery`, `canary_resets`, `canary_steps`, `halfbatch_cosine`,
`halfbatch_grad_norm_ratio`.
**Tags absent from all six runs:** `feature_velocity`, `feature_velocity_cos`,
`feature_velocity_rel`.

`capacity/canary_steps` = **480.0 in every arm at every point** — the canary itself was healthy
(the documented silent-failure tell is `canary_steps` reading 0, and it never does).

### Run state on disk

- **fdE is LIVE as of this read** (pid 736341, `-m main.launcher`, tb file mtime 19:19 vs read at
  19:20). Its 28 points are a partial series of an in-flight run; it has no `final_model.zip` and
  no `capacity_battery.json`. Every fdE number here is provisional.
- **Arm D (`ai_v9_41_fdD_gated_0825`) does not exist on disk** — no run directory, no metadata
  referencing it. It was never launched under that name. Nothing in this document covers it.
- fdA / fdB / fdC are complete (each `final_model.zip` + `capacity_battery.json` present).

---

## 2. 🚨 The COLLAPSE meter never ran — mechanism, and why it is the same bug class as the rev-1 duty cycle

`CapacityTelemetry.finish_train` gates the velocity probe on:

```python
if (self.probe_obs is not None and self.velocity_every > 0
        and self.train_calls % self.velocity_every == 0):
    current = probe_features(model, self.probe_obs)
    if current is not None:
        out.update(feature_velocity_metrics(current, self.prev_features))
        self.prev_features = current
```

and `feature_velocity_metrics(current, None)` returns `{}` by construction (a velocity needs two
points). Every run used the default `--capacity-velocity-every 50`.

`train_calls` is an attribute of the PPO object, which is **rebuilt per launcher child**. Measured
train-calls per child above: **28 to 61**. So:

- fdA / fdB / fdC / fdE (28–31 calls) and tick-1's three children (44 / 40 / 13) **never reached
  call 50**. Zero velocity probes.
- rev-1's c0 (61) and c1 (53) each reached call 50 **exactly once**, took a probe, found
  `prev_features is None`, emitted `{}`, and stored the point. Neither reached call 100.

**Total across all six runs: 2 probes taken, 0 samples emitted.** The tag's absence is complete and
fully explained.

This is the **same defect shape as the rev-1 hour-2 duty-cycle incident** ledgered on 2026-08-23
(`--cf-label-lag-steps 150k` against a hardcoded 2.4M checkpoint interval): two individually
sensible defaults — a 50-`train()` velocity cadence and a 3-hour launcher restart — that are
**jointly impossible**, with no gate multiplying them. The canary and cosine cadences are counted
in *minibatches* and survive a restart window fine; the velocity cadence is counted in `train()`
calls, of which a restart window contains ~30–60, and it needs **two** of them.

**Fix shape (not applied here, recorded for the owner):** either count velocity in minibatches like
the other two, or default `velocity_every` to something a restart window clears twice (≤10), or
publish a `capacity/velocity_probes` counter so a zero is visible rather than an absent tag. The
last is the cheapest and matches the `canary_steps`-reads-0 precedent already in the module.

---

## 3. The arms

All five forked from **`models/ai_v9_29_rev1_0823/final_model.zip` @ 25,264,368 steps**, all with
`--warmstart-bc-steps 4000`, `--canary-reset-steps 1000000`, `--capacity-cosine-every 50`,
`--capacity-velocity-every 50`. **Their capacity step grids are byte-identical** (verified: every
arm's steps are a prefix of fdC's), which is what makes the paired analysis in §5 possible.

| arm | `--distill-coef` | `--distill-value-feat-coef` | `--stable-opponent-pfsp` (teacher in the ecology) | teachers | commit |
|---|---|---|---|---|---|
| **fdC** (control) | **0.0** | 0.0 | **yes** | tock1 k4 + tock1b rain (present, unused by the loss) | `3d639cc4` |
| fdA | 0.3 | 0.15 | yes | tock1 k4 + tock1b rain | `3d639cc4` |
| fdB | 1.0 | 0.5 | no | tock1 k4 + tock1b rain | `3d639cc4` |
| fdE | 1.0 | 0.5 | no | tock1c q6 (single) | `b4f77661` |
| tick-1 | 1.0 | 0.5 | yes | tock1 k4 + tock1b rain | `5e63ecb5` |

**fdC is not a no-teacher control — it is an ECOLOGY-ONLY control** (the teachers are in the
opponent pool; only the loss is off). That is the right control for a tug-of-war claim, because it
holds the opponent distribution fixed and varies only the gradient term.

⚠️ **fdB and fdE drop `--stable-opponent-pfsp`**, so their rollout distributions differ from fdC's
on a second axis. **fdA vs fdC is the only single-variable contrast** (same ecology, same teachers,
same commit, coef 0.3 vs 0.0), and it is the weakest of the three distill-vs-control cosine gaps.
Read fdA as the load-bearing comparison and fdB/fdE as corroboration that may be inflated.

---

## 4. Matched-window means (25.264M → 28.115M)

Arm means over the window every arm shares. `n` points in the last row.

| scalar | fdC (ctrl) | fdA | fdB | fdE | tick-1 |
|---|---|---|---|---|---|
| **`capacity/halfbatch_cosine`** | **0.1411** | 0.1136 | 0.1119 | 0.0995 | 0.1022 |
| `capacity/halfbatch_grad_norm_ratio` | 0.7925 | 0.7957 | 0.7881 | 0.8037 | 0.7899 |
| **`capacity/canary_loss`** | **0.5639** | 0.5472 | 0.5377 | 0.5026 | 0.5311 |
| `capacity/canary_recovery` | 0.9339 | 1.0783 | 0.9627 | 0.9882 | 1.0364 |
| `capacity/canary_steps` | 480 | 480 | 480 | 480 | 480 |
| `train/grad_norm` | 1.4771 | 1.3979 | 1.7232 | 1.8530 | 1.6577 |
| `distill/kl` | — | 0.0848 | 0.0404 | 0.0245 | 0.0396 |
| `train/entropy_loss` | −0.7654 | −0.7717 | −0.7713 | −0.7662 | −0.7810 |
| `train/approx_kl` | 0.0299 | 0.0232 | 0.0251 | 0.0223 | 0.0241 |
| `train/explained_variance` | 0.7711 | 0.7355 | 0.7337 | 0.7499 | 0.7018 |
| `grad/policy_share` | 0.5243 | 0.5012 | 0.4623 | 0.5275 | 0.4408 |
| `grad/value_share` | 0.1821 | 0.1881 | 0.1910 | 0.1481 | 0.2033 |
| *n points* | 31 | 31 | 31 | 28 | 30 |

Within-window SD (for judging whether a mean gap is real):

| scalar | fdC | fdA | fdB | fdE | tick-1 |
|---|---|---|---|---|---|
| `halfbatch_cosine` | 0.0697 | 0.0764 | 0.0796 | 0.1031 | 0.1064 |
| `canary_loss` | 0.0319 | 0.0292 | 0.0300 | 0.0317 | 0.0333 |
| `canary_recovery` | 0.4683 | 0.5277 | 0.4257 | 0.4125 | 0.4809 |
| `train/grad_norm` | 0.0415 | 0.0730 | 0.1008 | 0.1131 | 0.1332 |

Note there is **no `grad/distill_share`** in the `grad/*` family — the distill term was never wired
into the gradient-balance readout, so the tug-of-war's most direct measurement (what fraction of
the shared-trunk gradient the KL owns) does not exist. `grad/policy_share` falling in fdB (0.462)
and tick-1 (0.441) against fdC (0.524) is the nearest available proxy.

---

## 5. Paired-by-step contrasts (arm − fdC at the SAME step)

Because the step grids are identical, the common trajectory can be differenced out. Moving-block
bootstrap (L = 5, B = 20 000) over the per-step difference series.

| scalar | arm | n | mean Δ | 95% CI | p |
|---|---|---|---|---|---|
| **`halfbatch_cosine`** | fdA | 31 | **−0.0275** | [−0.0541, −0.0039] | **0.019** |
| | fdB | 31 | **−0.0293** | [−0.0560, −0.0081] | **0.006** |
| | fdE | 28 | **−0.0384** | [−0.0691, −0.0183] | **0.0001** |
| | tick-1 | 30 | −0.0420 | [−0.0979, +0.0251] | 0.275 |
| **pooled mean(fdA,fdB,fdE)** | | 28 | **−0.0304** | **[−0.0580, −0.0122]** | **0.001** |
| `canary_loss` | fdA | 31 | −0.0167 | [−0.0205, −0.0139] | 0.000 |
| | fdB | 31 | −0.0262 | [−0.0314, −0.0230] | 0.000 |
| | fdE | 28 | −0.0656 | [−0.0695, −0.0617] | 0.000 |
| | tick-1 | 30 | −0.0342 | [−0.0446, −0.0243] | 0.000 |
| | *pooled* | 28 | −0.0365 | [−0.0401, −0.0342] | 0.000 (sign **0/28**) |
| `canary_recovery` | fdA | 20 | +0.1444 | [+0.0932, +0.1986] | 0.000 |
| | fdB | 20 | +0.0288 | [−0.0030, +0.0677] | 0.077 |
| | fdE | 17 | −0.0304 | [−0.0708, +0.0077] | 0.111 |
| | tick-1 | 19 | +0.0778 | [+0.0589, +0.0982] | 0.000 |
| | *pooled* | 17 | +0.0501 | [+0.0289, +0.0655] | 0.000 |
| `halfbatch_grad_norm_ratio` | fdA / fdB / fdE / tick-1 | | +0.003 / −0.004 / +0.013 / −0.001 | all include 0 | 0.65 / 0.66 / 0.24 / 0.47 |
| `train/grad_norm` | fdA | 31 | −0.0792 | [−0.1256, −0.0400] | 0.000 |
| | fdB | 31 | +0.2461 | [+0.1841, +0.2993] | 0.000 |
| | fdE | 28 | +0.3749 | [+0.3062, +0.4362] | 0.000 |
| | tick-1 | 30 | +0.1799 | [+0.0970, +0.2448] | 0.000 |
| `train/explained_variance` | fdA | 31 | −0.0355 | [−0.0419, −0.0270] | 0.000 |
| | fdB | 31 | −0.0373 | [−0.0469, −0.0254] | 0.000 |
| | fdE | 28 | −0.0201 | [−0.0328, −0.0095] | 0.001 |
| | tick-1 | 30 | −0.0691 | [−0.1069, −0.0364] | 0.000 |
| `grad/policy_share` | fdA | 31 | −0.0231 | [−0.0540, +0.0001] | 0.050 |
| | fdB | 31 | −0.0620 | [−0.1157, −0.0298] | 0.000 |
| | fdE | 28 | +0.0092 | [−0.0301, +0.0260] | 0.954 |
| | tick-1 | 30 | −0.0833 | [−0.1383, −0.0472] | 0.000 |

**Why the cosine result is not an artifact of the distill term itself.** `halfbatch_trunk_cosine`
runs the **plain PPO surrogate only** (clipped policy loss + `vf_coef`·MSE — deliberately not the
run's full fold, per the module docstring). The distill KL cannot enter the probe. So a lower
cosine in a distill arm says: *the trunk that the KL has shaped yields a more internally
inconsistent **PPO** gradient across two i.i.d. halves of one on-policy batch.* That is exactly the
functional form the tug-of-war predicts, and it is a statement about the representation rather
than about the extra loss term's presence.

**⚠️ The residual confound is the state distribution.** Each arm's cosine is measured on that arm's
own rollout data. fdB and fdE also changed their opponent ecology, so their gaps are not purely
trunk effects. fdA holds the ecology fixed and still shows −0.0275 [−0.0541, −0.0039] — that is the
clean version of the claim, and it is the smallest of the three.

---

## 6. 🚨 Why `canary_recovery` supports NO claim (and why the paired test above lied about it)

`canary_recovery` = post-reset EMA loss of the re-seeded target ÷ its pre-reset loss. >1 means the
new random target is still worse-fit than the retired one; it decays toward ~1 as the head re-fits.
The documented read rule is **compare at a matched `canary_age`**, because the value is a point on
a decay curve.

The per-point bootstrap in §5 reports a "significant" pooled +0.0501. **It is reading the reset
schedule, not a capacity difference.** Consecutive recovery points inside one reset episode are an
EMA ratio of the same two numbers, so they are near-deterministic replicas. The honest unit is one
`(child, reset episode)`:

| run | reset episodes in the matched window | per-episode mean recovery | arm mean |
|---|---|---|---|
| fdC (ctrl) | 2 | 1.343, 0.434 | 0.888 |
| fdA | 2 | 1.543, 0.511 | 1.027 |
| fdB | 2 | 1.334, 0.509 | 0.921 |
| fdE | 2 | 1.279, 0.455 | 0.867 |
| tick-1 | 2 | 1.434, 0.490 | 0.962 |

Every arm shows the **same two-episode shape** (episode 1 ≈ 1.28–1.54, episode 2 ≈ 0.43–0.51), and
the between-episode spread (≈0.9) is **an order of magnitude larger** than any between-arm
difference (≤0.16). At matched age the arms interleave: fdE 0.867 < fdC 0.888 < fdB 0.921 < tick-1
0.962 < fdA 1.027 — the control sits *second* of five, and the two coefficient-1.0 arms sit on
either side of it.

The canary state is not checkpointed, so it re-inits per child and the reset counter never exceeds
2 in a 2.85M-step arm (rev-1's 25M run gets 20 episodes across 5 children; tick-1 gets 7).
**`canary_recovery` has n = 2 per fold arm. It cannot answer this question.** Do not quote the
§5 recovery row.

**`canary_loss` moves against the prediction.** The interference cell predicts the canary
degrading; instead the distill arms fit the synthetic targets **better** than the control at
**every single step** (sign 0/28, pooled −0.0365 [−0.0401, −0.0342]). Read literally — a lower
`canary_loss` means the trunk supplies *more* recoverable obs structure — this is the opposite of
representation narrowing. Two honest readings, and the data does not separate them:

- *Consistent with tug-of-war:* the KL holds the trunk away from the "collapsed onto the policy's
  current answers" attractor the canary was built to detect, so richness stays higher while task
  agreement falls. Richness up, coherence down.
- *Confound:* the gap is already present at the **first** logged point (fdC 0.597 vs fdA 0.597,
  fdB 0.603, fdE 0.560, tick-1 0.560 — fdC is not cleanly highest at t=0, but is highest from the
  second point onward and stays highest throughout), and the canary head re-initialises per child
  and trains on arm-specific data. A level offset that opens within ~100k steps is not
  distinguishable here from a distributional difference in the probe's own inputs.

---

## 7. Divergence from a common origin — the accelerated-decay test does NOT clear

All five arms carry identical weights at 25.264M, so their cosine at the fork must agree, and it
does (early-third 0.150–0.160 for all four 3M arms). The registered interference story predicts the
distill arms then decay faster. Thirds of the matched window:

| arm | early | mid | late | decay (late − early) | 95% CI |
|---|---|---|---|---|---|
| fdC (ctrl) | 0.1600 | 0.1455 | 0.1199 | −0.0401 | [−0.0858, +0.0080] |
| fdA | 0.1547 | 0.1011 | 0.0876 | −0.0672 | [−0.1252, −0.0098] |
| fdB | 0.1498 | 0.1104 | 0.0786 | −0.0711 | [−0.1363, −0.0051] |
| fdE | 0.1507 | 0.1066 | 0.0470 | −0.1037 | [−0.1857, −0.0200] |
| tick-1 | 0.0223 | 0.1373 | 0.1469 | +0.1246 | [+0.0429, +0.2046] |

Difference-in-differences vs fdC (bootstrap over independent draws):

| arm | DiD | 95% CI | p |
|---|---|---|---|
| fdA | −0.0271 | [−0.0977, +0.0441] | 0.455 |
| fdB | −0.0311 | [−0.1081, +0.0490] | 0.447 |
| fdE | −0.0637 | [−0.1589, +0.0326] | 0.192 |
| tick-1 | +0.1647 | [+0.0694, +0.2570] | 0.001 |

**Verdict on §7: NOT significant.** The three 3M distill arms all point the right way and order
roughly by pull, but every CI includes zero — with n≈10 per third and a per-point SD of 0.07–0.10,
this test has no power. **The level difference (§5) is real; the "the gap OPENS" claim is not
supported.**

⚠️ **tick-1's early third (0.0223) is a noise artifact, not a finding.** Its first eight per-point
cosines are `+0.077, −0.050, +0.210, −0.004, −0.108, −0.011, +0.159, −0.002` — the probe's
per-point spread swamps its own mean over ten points. tick-1's DiD sign flip is entirely that. Do
not read tick-1 into the divergence table.

**Dose ordering is imperfect.** Late-third cosine against the measured distill pull:

| arm | late cosine | mean `distill/kl` | `--distill-coef` |
|---|---|---|---|
| fdE | 0.0470 | 0.0245 | 1.0 |
| fdB | 0.0786 | 0.0404 | 1.0 |
| fdA | 0.0876 | **0.0848** | 0.3 |
| fdC | 0.1199 | — | 0.0 |
| tick-1 | 0.1469 | 0.0396 | 1.0 |

fdA carries the *largest* residual KL (0.0848 — its weaker coefficient leaves the student further
from the teacher) yet the *mildest* cosine depression, and tick-1 at coefficient 1.0 has the
highest late cosine of all. **There is a clean loss-present / loss-absent split; there is no
monotone dose-response.** The honest statement is binary, not graded.

---

## 8. What these scalars look like NORMALLY (rev-1, 0.2M → 25.07M, 5 children)

The long baseline is what stops a low cosine at 25M being read as pathology.

| window (M steps) | `halfbatch_cosine` | `canary_loss` | `canary_recovery` | `train/grad_norm` | `halfbatch_grad_norm_ratio` |
|---|---|---|---|---|---|
| 0.0 – 5.0 | 0.6139 | 0.5131 | 1.5025 | 1.3410 | 0.7947 |
| 5.0 – 10.0 | 0.3391 | 0.5143 | 0.9211 | 1.2602 | 0.8119 |
| 10.0 – 15.0 | 0.2370 | 0.5115 | 0.7446 | 1.1891 | 0.8184 |
| 15.0 – 20.0 | 0.1576 | 0.5319 | 0.9123 | 1.1914 | 0.8193 |
| 20.0 – 25.1 | 0.1127 | 0.5196 | 0.8446 | 1.1416 | 0.8091 |

Whole-run slopes: `halfbatch_cosine` **−0.0235 / 1M, r = −0.77** (a strong, monotone decline —
exactly the module docstring's "falling slowly as the gradient shrinks toward a stationary point");
`canary_recovery` −0.0294 / 1M, r = −0.26 (no trend); `canary_loss` flat at ~0.51–0.53 across 25M.

**So the fold arms fork into a regime where the cosine is already at 0.11 and normally declining.**
The 0.15–0.16 the arms show in their early third is a small bump above rev-1's terminal 0.113 —
plausibly the warm-start / fresh-optimizer transient. The interference reading in §5 is a
**relative** claim (arm vs its own control at matched steps), and must stay one: there is no
calibrated alarm level for this scalar, as the landing entry says.

---

## 9. tick-1's full 10M series — the instrument is FLAT across the actual collapse

tick-1 is the richest series and the run the collapse was named from.

| window (M) | `halfbatch_cosine` | `canary_loss` | `canary_recovery` | `train/grad_norm` | `distill/kl` | `train/entropy_loss` |
|---|---|---|---|---|---|---|
| 25.2 – 27.0 | 0.0785 | 0.5399 | 1.4408 | 1.6510 | 0.0418 | −0.7940 |
| 27.0 – 29.0 | 0.1236 | 0.5041 | 0.7745 | 1.6645 | 0.0353 | −0.7488 |
| 29.0 – 31.0 | 0.1008 | 0.5324 | 1.0882 | 1.6883 | 0.0328 | −0.7346 |
| 31.0 – 33.0 | 0.1148 | 0.5409 | 1.0103 | 1.6913 | 0.0317 | −0.7333 |
| 33.0 – 35.2 | 0.1151 | 0.5268 | 1.0100 | 1.6489 | 0.0313 | −0.7346 |

Whole-run slopes: `halfbatch_cosine` **+0.0037 / 1M (r = +0.12)**, `canary_recovery` −0.0089 / 1M
(r = −0.05), `canary_loss` +0.0010 / 1M (r = +0.08), `train/grad_norm` +0.0015 / 1M (r = +0.05).

**Every one of them is flat.** Whatever tick-1 did over 10M steps, no capacity scalar degraded
along the way. `distill/kl` declines smoothly 0.042 → 0.031 (the student converging toward the
teachers) and `train/entropy_loss` rises −0.794 → −0.735, both monotone and unremarkable.

This is the single most important caveat on the §5 result: **the interference signature is an
offset between arms, not a trajectory within one.** A meter that separates a treatment from its
control but stays flat while the treated run does whatever it did is telling you about the
*condition*, not about the *damage*.

---

## 10. Triage verdict

The registered triage table, and what actually happened:

| cell | condition | fired? |
|---|---|---|
| **INTERFERENCE** (widen / pace) | canary degrades **AND** cosine falls | **PARTIAL — cosine fell (p = 0.001 pooled, 4/4 direction); the canary did NOT degrade** (`canary_recovery` null at honest n; `canary_loss` moved the *opposite* way, 0/28) |
| **COLLAPSE** (fix targets, not width) | canary degrades + cosine flat + velocity low | **UNTESTABLE — `feature_velocity` produced zero samples in all six runs** (§2). The cosine was not flat, so this cell was not entered on the evidence available; it is not refuted. |
| **IDLE** (do nothing) | all flat | **NOT fired between arms** (cosine, canary_loss, explained_variance, grad_norm all separate the distill arms from the control) — but **fired WITHIN tick-1's own 10M collapse window**, where every scalar is flat (§9) |

**Overall: the tug-of-war hypothesis gets weak, direction-consistent support on its one working
meter, and no support from the other.** The honest headline is closer to *"the instrument's first
real case is a half-null, and half of the battery never ran"* than to *"tug-of-war confirmed"*.

**What the evidence does support, stated at the strength it earns:**

1. Adding a distill KL to this trunk **measurably lowers the PPO objective's self-agreement across
   two halves of one on-policy batch** — pooled −0.030 [−0.058, −0.012], present at coefficient 0.3
   with the ecology held fixed. That is a real interference reading, in the mechanism's predicted
   direction, on a probe the distill term cannot contaminate.
2. It is **binary in the loss, not graded**: fdA (coef 0.3, largest residual KL) and fdB (coef 1.0)
   are indistinguishable, and tick-1 (coef 1.0) has the *highest* late cosine. Any "turn the
   coefficient down" prescription is **not supported** by this data.
3. It travels with a **consistent critic cost**: `train/explained_variance` is lower in all four
   distill arms (−0.020 to −0.069, all p ≤ 0.001), worst in tick-1. That is the most robust
   arm-vs-control difference in the whole read — larger effect, tighter CI, and 4/4.
4. The trunk is **not** losing representational richness: `canary_loss` is lower (better) in every
   distill arm at every step. Whatever the KL is costing, it is not the supply of recoverable obs
   structure the canary measures.

**What it does NOT support, and must not be quoted as:**

- ❌ "canary recovery degrades in the distill arms" — n = 2 reset episodes per arm; the arms
  interleave at matched age with the control **second of five**. The §5 recovery row is an artifact
  of a per-point bootstrap over a deterministic reset schedule.
- ❌ "the cosine gap widens as the collapse develops" — the DiD is null (§7), and tick-1's 10M
  series is flat (§9).
- ❌ any grading of the effect by coefficient (§7 dose table).
- ❌ any collapse claim in either direction — the meter does not exist in these runs.

---

## 11. Limitations

- **3M is a short series.** fdA/fdB/fdC have 31 capacity points and fdE 28. The per-point cosine SD
  is 0.07–0.10 against effects of 0.03. Only the paired-by-step design (identical step grids from a
  common fork) makes the level contrast readable at all, and it still cannot see a trend.
- **n = 1 run per cell.** Every CI here is over within-run sampling noise, not over run-to-run
  variation. Two arms differing at p = 0.001 on a within-run bootstrap is not the same claim as a
  reproducible arm effect.
- **fdE is live.** Its series is truncated at 27.919M and its numbers will change.
- **fdB and fdE also changed the opponent ecology** (`--stable-opponent-pfsp` absent), so only
  fdA vs fdC is single-variable — and it is the weakest of the three.
- **fdE ran on a different commit** (`b4f77661` vs `3d639cc4`) and a different teacher
  (tock1c q6), tick-1 on `5e63ecb5`. Only fdA/fdB/fdC are commit-matched.
- **The canary is not checkpointed** and re-inits per launcher child; `canary_age` and
  `canary_resets` restart at 0 each time. Recovery must be read within a restart window and at
  matched age — done in §6, and it is what kills the recovery claim.
- **The cosine's state distribution is arm-specific** (each arm measures on its own rollouts). No
  design here separates "the trunk became inconsistent" from "the data became harder".
- **No calibrated alarm level exists for any of these scalars** — the landing entry says trend
  only, and rev-1 (§8) shows the cosine falling 5× over a healthy run. Every number here is
  arm-relative by necessity.
- **`grad/distill_share` does not exist**, so the tug-of-war's most direct quantity — the KL's
  share of the shared-trunk gradient — was never measured. `grad/policy_share` is the proxy used.
- **Outcome anchoring is deliberately omitted.** Each run's `snapshot_ladder/ladder.json` is fit
  over that run's own frozen pairs (10–66 pairs, differing sentinel sets, and the fold arms inherit
  pre-fork snapshots from the rev-1 lineage), so cross-run Elo comparison would violate the
  matched-snapshot-count rule. The strength verdict on these arms lives in the fold investigation's
  own arena, not here.

## 12. Recommendations

1. **Fix the velocity cadence before the next fold arm** — count it in minibatches like the canary
   and cosine, or default it low enough that a 3-hour restart window clears it twice, and publish a
   `capacity/velocity_probes` counter so a zero reads as a zero rather than an absent tag. Until
   then the COLLAPSE cell of the triage table is decorative.
2. **`canary_recovery` needs longer arms or more frequent resets to be usable.** At
   `--canary-reset-steps 1000000` a 3M arm gets two episodes. Either shorten the reset interval for
   short arms or stop reading recovery on them.
3. **Add `distill` to the `grad/*` balance family.** The one number that would settle the
   tug-of-war — the distill term's share of, and cosine against, the shared-trunk gradient — is the
   one number nobody logged, and every other `*_coef` term already has it.
4. **`train/explained_variance` was the strongest and cleanest arm-vs-control signal in this read**
   (−0.020 to −0.069, 4/4, all p ≤ 0.001), and it is not part of the capacity battery. Whatever
   the fold costs, the critic is where it shows most reliably.
