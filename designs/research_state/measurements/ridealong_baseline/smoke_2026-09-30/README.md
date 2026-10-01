# Ride-along heads reader: plumbing smoke on UNTRAINED heads (2026-09-30)

**What this is.** This is the first run of `python -m main.ridealong_read` (`src/main/ridealong_read/`).
It reads three EXISTING ai_v14 checkpoints: K2 final (91.1M), N0 final (75.0M) and C_fix final (83.1M),
the three the X4 pre-read used. None of the three was trained with the ride-along heads. So every JSON
says `heads: "fresh-untrained"`: the reader attached FRESH heads with the baseline spec (V ensemble 5,
RND, A 5, B 5) from `build_ridealong` on a private RNG.

**What the numbers can and cannot tell you:**
- The ensemble disagreement and A are their **randomized priors only**.
- RND's predictor is **untrained**. Its observation statistics and error z-score were fitted on the TRAIN
  split's rows, then every row was scored.
- So meters (i)–(iv) are a **plumbing check and a FLOOR** that a trained head must beat. They are not a
  measurement of the heads.
- Meter (v), the RND input-choice read, **is** a real measurement. It trains its own predictors offline.

The real read comes from the baseline GPU run, whose checkpoints carry trained heads (`heads: "trained"`).

Every interval is a 95 % percentile bootstrap that resamples BATTLES, with B = 1000 and seed 20260930.
This is the X4 pre-read's method (`bootstrap` block in every JSON). Rank statistics use exact weighted
mid-ranks (`src/main/ridealong_read/boot.py`). Bank: `m5_laneS/bank_v1` (20,712 decisions, 580 battles),
re-encoded byte-equal to the recording (19,964 / 19,964 checkable rows). Truth: `m5_laneS/truth_v2`
(1,600 turns, 64 seeds, three greedy continuations; each checkpoint is quoted against its OWN).

## The command

```bash
export PYTHONPATH=$PYTHONPATH:src
M=/home/goodlad/dev/gen3ai/models
python -m main.ridealong_read \
  --checkpoint $M/ai_v14_07_g0p_k2/final_model.zip=K2final \
  --checkpoint $M/ai_v14_01_base/final_model.zip=N0final75M \
  --checkpoint $M/ai_v14_06_lbat_ctrl_fix/final_model.zip=C_fixfinal \
  --drift-pair $M/ai_v14_01_base/checkpoints/checkpoint_45752064_steps.zip \
               $M/ai_v14_01_base/final_model.zip $M/ai_v14_07_g0p_k2/final_model.zip \
  --out designs/research_state/measurements/ridealong_baseline/smoke_2026-09-30 \
  --archive ~/gen3ai_archive/ridealong_read/smoke_2026-09-30 --threads 4
```

The whole read takes ~4 min on CPU: a 49 s re-encode, ~35 s per checkpoint, and ~35 s for the RND
training. The per-row arrays (V, logits, π, `value_pooled`, every head output) are in
`~/gen3ai_archive/ridealong_read/smoke_2026-09-30/<label>.rows.npz` (40 MB, not committed). Files here:
`<label>.json` (one per checkpoint) and `rnd_input_choice.json`.

## What each meter means, and what the smoke read

**(i) Does disagreement or novelty predict V's actual error?**
- Setup: error = |V − z| on the bank, with win z = 1 and loss z = 0. DRAWS ARE EXCLUDED (1,865 rows), and
  `draws_as_loss` is the sensitivity row. The score is judged on AUROC for |V − z| > 0.5, Spearman, and
  the top-decile ÷ bottom-decile mean error.
- Columns:
  - `ens_std`: std of the members' probabilities.
  - `ens_logit_std`: the same in logit space.
  - `rnd_z`: RND novelty.
  - `ref_v_entropy`: V's own binary entropy, the REFERENCE.
- `auroc_within_ref_quintiles` re-reads each AUROC inside quintiles of V's entropy. That is the de-confounded
  read, and it is the one to quote.
- Results in the table below: RND is at chance. The untrained ensemble reaches 0.63–0.66, but only about
  0.54–0.56 once V's own uncertainty is held fixed.
- **Hazard.** A std of member probabilities is mechanically largest where the logits sit near 0, which
  is where V itself is near 0.5. So the raw `ens_std` AUROC partly measures V's own uncertainty.

| | K2 final | N0 final | C_fix final |
|---|---|---|---|
| `ens_std` AUROC | 0.656 [0.615, 0.691] | 0.630 [0.587, 0.669] | 0.653 [0.610, 0.690] |
| `ens_std` AUROC within V-entropy quintiles | 0.557 | 0.541 | 0.540 |
| `ens_std` Spearman | 0.439 [0.356, 0.509] | 0.387 [0.303, 0.462] | 0.436 [0.351, 0.505] |
| `ens_std` top ÷ bottom decile error | 3.86 [1.89, 9.64] | 2.83 [1.60, 5.78] | 3.89 [1.78, 11.6] |
| `rnd_z` AUROC | 0.518 [0.482, 0.551] | 0.546 [0.510, 0.580] | 0.521 [0.485, 0.555] |
| reference: V-entropy AUROC | 0.740 [0.704, 0.772] | 0.714 [0.676, 0.750] | 0.745 [0.708, 0.777] |

The per opponent class and per phase splits are in each JSON. The exploiter class has 748 rows, and
every interval there spans 0.5.

**(ii) Does A see the truth?**
- Setup: on the 1,600 truth turns, compare A's member mean with the truth's per-action values over the
  legal actions. The policy's own logits are the reference row.
- Reads:
  - the within-turn Spearman ρ;
  - argmax regret on DECISIVE turns (±1 scale);
  - on decisive turns, how often a near-best action the policy STARVES (π < 1 %, truth gap ≤ 0.1) falls
    in the score's top 2, against chance.

Untrained A (a random prior) is at chance, as expected:

| | K2 final | N0 final | C_fix final |
|---|---|---|---|
| A: within-turn ρ | −0.007 [−0.032, 0.020] | −0.007 [−0.031, 0.017] | 0.003 [−0.025, 0.030] |
| logits: within-turn ρ | 0.180 [0.152, 0.206] | 0.146 [0.120, 0.168] | 0.169 [0.141, 0.194] |
| paired ρ(A) − ρ(logits) | −0.187 [−0.220, −0.151] | −0.152 [−0.186, −0.119] | −0.166 [−0.201, −0.130] |
| A: starved near-best in top 2 (chance) | 0.240 [0.207, 0.273] (0.265) | 0.266 [0.229, 0.307] (0.268) | 0.248 [0.212, 0.286] (0.264) |
| logits: same | 0.038 | 0.030 | 0.022 |
| argmax regret on decisive turns, A / logits | 0.400 / 0.303 | 0.414 / 0.342 | 0.390 / 0.293 |

Cross-check against the X4 pre-read: the policy's argmax regret on decisive turns is 0.303 / 0.342 /
0.293 here, against 0.312 / 0.345 / 0.299 there. That read used held-out seeds 8–63. This one uses all
64 seeds.

**(iii) Are the starved near-best moves flagged as uncertain?**
- Setup: A's member spread (`adv_std`) on starved near-best actions, against every other legal action.
- The control that matters: starved near-best against starved NOT-near-best, i.e. "good and unplayed"
  against merely "unplayed".
- **Hazard found by the first read, and FIXED in the heads.** The first read measured the spread of A
  CENTRED UNDER π. Each member's value on an action π plays is subtracted from every member, so the
  spread there shrinks toward 0 while an unplayed action keeps the full spread. Even the untrained
  prior then read "starved actions are more uncertain": 0.66–0.67 against other legal actions on the
  truth turns, and 0.73–0.75 starved vs fed bank-wide. `adv_std` is now each member centred on its
  OWN legal-action mean (uniform weights, `ridealong_heads.RideAlongHeads.readout`). That removes the
  members' free offsets without favouring any action. This rerun is on that definition.

| | K2 final | N0 final | C_fix final |
|---|---|---|---|
| `adv_std` (uniform-centred): starved-near vs other legal | 0.509 [0.485, 0.532] | 0.545 [0.516, 0.575] | 0.523 [0.499, 0.545] |
| `adv_std`: starved-near vs starved-other | 0.496 [0.471, 0.521] | 0.534 [0.499, 0.566] | 0.517 [0.493, 0.539] |
| uncentred spread (`adv_raw_std`): starved-near vs other legal | 0.465 [0.435, 0.493] | 0.484 [0.451, 0.518] | 0.503 [0.472, 0.532] |
| bank-wide `adv_std`, starved vs fed | 0.512 | 0.516 | 0.511 |

These are the untrained FLOORS X26's R4 is read against, on the same checkpoint.

**(iv) Disagreement or novelty against |V − V_truth| on the truth turns.**
- The truth rows carry per-action values, not a state value. V_truth is therefore DERIVED: the truth value
  of the continuation policy's greedy root action, mapped to [0, 1]. This is valid only against the
  checkpoint's own continuation.
- The AUROC is for |V − V_truth| > 0.25, because V_truth is continuous.
- V is optimistic, by +0.107 / +0.090 / +0.096 on [0, 1]. That is +0.18 to +0.21 on ±1, consistent with
  the X4 pre-read's +0.14 to +0.19.
- `ens_std` AUROC is 0.615 [0.577, 0.650] / 0.548 [0.510, 0.584] / 0.605 [0.562, 0.644], and 0.535–0.599
  within V-entropy quintiles.
- `rnd_z` AUROC is 0.53–0.54, with every interval spanning or touching 0.5.
- The V-entropy reference is 0.75–0.78.

## (v) RND input choice: observation or trunk features? A REAL measurement

**Split** (battle-level, by source; no battle straddles it; the reader asserts this):
- TRAIN = the N0@36M, 46M, 56M and 66M eval cycles: 10,147 rows.
- HELD-OUT = the rest: 10,565 rows.
- UNSEEN-TEAM = held-out rows whose team never appears in TRAIN: 2,312 rows.
- EXPLOITER = the 748 held-out rows against the round-0 exploiter.
- Most exploiter rows are also unseen teams, so each label is also read inside the other's complement.

**Training.** Both predictors are the heads' own `RndNovelty` (the same target and predictor shapes,
normalisation fitted on TRAIN, ±5 clip). They train with Adam at the ride-along rate (3e-4), minibatch
256, fixed seed, and are read after 3, 10 and 30 epochs.
- obs-RND reads the 2761-dim observation.
- feature-RND reads the 128-dim detached `value_pooled`.

**Drift.** feature-RND is trained on the features of N0 at 45.75M (A). The SAME states are then scored
through two later checkpoints of the lineage (B): N0 final (75M), and K2 final (91.1M, via C_fix). Two
versions are read:
- A's frozen normalisation (`drift`);
- the normalisation re-fitted on B's features (`drift_renormalised`, which removes a pure shift or
  rescale, as a live running normaliser partly would).

| novelty AUROC (held-out rows), 10 epochs | obs-RND | feature-RND, same checkpoint (K2 / N0 / C_fix) | feature-RND on A |
|---|---|---|---|
| unseen team vs seen | **0.741 [0.684, 0.797]** | 0.719 / 0.712 / 0.710 | 0.693 [0.639, 0.744] |
| unseen team vs seen, exploiter rows removed | 0.728 | 0.735 / 0.728 / 0.724 | 0.693 |
| exploiter vs other | **0.733 [0.659, 0.797]** | 0.617 / 0.603 / 0.612 | 0.663 [0.587, 0.741] |
| exploiter vs other, seen teams only | 0.682 [0.578, 0.757] | 0.627 / 0.590 / 0.607 | 0.676 |

These hold at every epoch count. At 3 and 30 epochs, obs-RND reads unseen 0.681 → 0.755 and exploiter
0.712 → 0.731.

| drift: feature-RND trained on A, the same held-out states through B (10 epochs) | B = N0 final | B = K2 final |
|---|---|---|
| mean novelty inflation B ÷ A (no drift = 1.0) | **8.2×** | **6.7×** |
| share of states above A's own 90th percentile (no drift = 0.10) | 1.00 | 1.00 |
| drift AUROC: the same state through A vs through B (none = 0.5) | 0.999 [0.999, 1.000] | 0.998 [0.996, 1.000] |
| … with the normalisation re-fitted on B: inflation / above-p90 / drift AUROC | 2.6× / 0.58 / 0.881 [0.866, 0.897] | 2.5× / 0.56 / 0.870 [0.854, 0.887] |
| unseen-team AUROC through B (0.69 through A) | 0.535 (renormalised 0.587) | 0.546 (renormalised 0.595) |
| exploiter AUROC through B (0.66 through A) | 0.534 (0.511) | 0.485 (0.515) |

The same pattern holds at 3 and 30 epochs: frozen inflation is 6.7–9.7× against N0 final, and 2.1–3.0×
after re-normalising.

**Verdict: obs-RND. Tag: MEASUREMENT on the fixed bank.**
- With no drift at all, obs-RND is at least as good as feature-RND:
  - unseen team: 0.74 vs 0.71–0.72;
  - exploiter: 0.73 vs 0.60–0.62 on the checkpoint's own features.
- Feature-RND's novelty is DOMINATED by representation drift over 30M steps of one lineage. The states it
  was trained on read as 2.5–8× more novel once the trunk moves. That holds even after re-normalisation,
  and the drift AUROC stays at 0.87–0.88 after it.
- Through the later checkpoint, its unseen-team signal collapses to 0.53–0.60 and its exploiter signal
  to about 0.5.
- Obs-RND's input does not depend on the checkpoint, so its drift is zero by construction.

**Caveats** (each one is also a finding in the build report):
- **Obs-RND memorises rows.** Held-out seen-team rows against the TRAIN rows themselves score 0.74 / 0.86 /
  0.91 at 3 / 10 / 30 epochs. That is a mix of row memorisation and the policy's era shift across sources.
- **Drift is measured on 30M-step gaps, not on a live run.** Between PPO updates the drift per rollout is
  far smaller. A live feature-RND whose predictor keeps training would partly track it. The number that
  applies to a live run is the per-interval drift at the run's snapshot cadence. It was not measured here.
- **The split is confounded with time.** Held-out sources come later in the lineage, so "novel" also
  means "a later policy's states".

## Findings from building the reader

1. **The `ens_std` meter is confounded with V's level** (sigmoid compression). ACTED ON: the training-time
   `ridealong/ens_*_err` meters now score the LOGIT-space spread (`ens_logit_std`) and log V's own
   entropy beside it (`ridealong/ref_v_entropy_*`). X26's R1 is read WITHIN V-entropy quintiles
   (`auroc_within_ref_quintiles`), against this smoke's untrained floor. Note the logit spread of
   untrained priors ALSO reads 0.63–0.65 raw and 0.53–0.55 within quintiles here, so the within-quintile read
   and the same-checkpoint floor are both needed.
2. **The `adv_std_starved` vs `adv_std_fed` pair was a centring artefact at init.** FIXED in the heads:
   `adv_std` is uniform-legal-centred now (see (iii)). The bank-wide starved-vs-fed AUROC at init
   fell from 0.73–0.75 to 0.51–0.52.
3. **(iv)'s V_truth is derived.** The truth rows carry no state value. The reader uses the continuation's
   greedy root action, so (iv) exists only for a checkpoint that IS a truth continuation.
