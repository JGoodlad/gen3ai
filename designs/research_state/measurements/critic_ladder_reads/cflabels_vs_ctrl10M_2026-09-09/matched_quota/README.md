# Matched-quota re-read — does `cond.own_team_r2.t1` survive a frame the size of the control's?

**Verdict: ARTEFACT.** On a frame matched to the control's the arm's turn-1 own-team R² falls from
**+0.060 to a median of −0.025** over 30 subsample seeds (2.5/97.5 across-seed spread
[−0.140, +0.249]) — statistically indistinguishable from the control's own **−0.024** — and the
delta collapses from the registered **+0.0841 [+0.0323, +0.1825] DETECTED** to
**−0.0013 [−0.2436, +0.3461] NOT DETECTED**, with the sign a coin flip across seeds (15/30
positive; 2 seeds' own CIs clear zero upward, 2 clear it downward). The arm's own R² is a smooth
increasing function of its frame size, which is the signature the prediction below named for the
artefact. `identity.bias.ALL` is a different kind of object and is NOT implicated.

Read with: `bash run.sh [TMP]`. Scripts: `subsample.py` (the subsampled view + the ten rows),
`analyze.py` (the tables), `identity_matched.py` (§4). Points and frames:
`matched_quota_points.json`, `identity_matched.json`; full tables: `tables.md`,
`identity_tables.md`. **Nothing was written under `models/`** — every view is a tree of symlinks
into the archive plus a rewritten manifest, materialized under the job tmp dir and deleted after
each read.

---

## 1. The question, the prediction, and the decision rule — stated before the reading

The registered read (`../critic_read.md`) compared an arm traced at outcome quota **40/40/10** with
a control traced at **5/10/5**. Every gauge row is Horvitz-Thompson reweighted by the cycle's own
capture rates, so the LOSS-ENRICHMENT selection is handled (rule 17). What reweighting does not
touch is **decoder power**: `cond.own_team_r2.*` is an out-of-fold score of a ridge FIT on the
frame, with battle-grouped folds and a per-team leave-one-battle-out label that needs
`MIN_TEAM_BATTLES = 4`. The arm's decoder was fit on **429 battles**, the control's on **104**.

> **PREDICTION (written before the numbers).** If the +0.060 is a LEVER effect, the arm's own-team
> R² is a property of its critic and must survive a shrunken frame: it stays near +0.060 at the
> control's frame size, and the frame-size curve is FLAT. If it is a DECODER-POWER artefact, the R²
> is a property of how much the decoder had to learn from, and it must FALL toward the control's
> −0.024 as the frame shrinks, with the curve climbing monotonically in frame size.
>
> **DECISION RULE.** ARTEFACT if (a) the matched-frame median R² is inside the control's own
> interval AND (b) the matched delta's CI covers zero AND (c) the curve is monotone increasing in
> frame size. EFFECT if the matched median holds near +0.060 and the matched delta's CI clears
> zero. INCONCLUSIVE otherwise — in particular if the matched delta's CI covers zero only because
> it is enormous, while the point estimate holds.

All three artefact conditions are met, and the point estimate does not hold, so (c)'s escape hatch
into INCONCLUSIVE is not reached.

## 2. Method

**The caps are the control's REALIZED profile, not its nominal quota.** Under battle-level
work-stealing each shard unit carries `max(1, ceil(quota / n_shards))`, so the control's nominal
5/10/5 landed as **8 traced wins and up to 12 traced losses** per opponent. Subsampling the arm to
the literal 5/10 would over-shrink it below the control's own frame, so the matched rung is
**8/12/5** — and it lands at **197 battles against the control's 198**.

For each seed and each rung: per opponent, keep at most `cap` traced wins / losses / draws drawn
without replacement from the arm's traced battles (`numpy.random.default_rng([seed, opp_index,
0xC0FFEE])`); **recompute the capture rates for the subsample** (`kept_wins / battles_won`, and
likewise for losses and draws, against the manifest's own denominators) so the HT weights describe
the view that is actually read; materialize a symlink tree with that manifest; run
`main.ops.conditioning_meters.conditioning_block` at the registered read's own `seed=0` and
`boot=2000`. Draws are excluded from every conditioning row by `extract_cycle` regardless, so the
draw cap is cosmetic. The strength axis (for `cond.elo_slope`) is computed once against the REAL
run directory and injected — it is a function of the snapshot ladder and the manifest's true win
rates, neither of which a subsample touches.

**Self-check:** run on the arm and the control AS TRACED, this path reproduces
`../critic_read.json`'s conditioning points to the printed digit (control −0.0237 / +0.7895 /
+0.0098 …; arm +0.0604 / +0.0837 / +0.5552 …), and `identity_matched.py` reproduces all six
identity strata under all three weightings. The tool is the same tool.

Deltas use `critic_readouts.independent_delta` — the difference of the two runs' independent
battle-clustered bootstraps — and the registered three-way label. **No replicate floor exists**, so
every DETECTED here would be against ZERO. **30 subsample seeds** on the two matched rungs, **20**
on each frame-size rung, 2,000 bootstrap draws per block.

Two intervals are reported per row because they answer different questions: **per-seed** is the
median seed's own battle-clustered CI (what the registered read would have printed had the arm been
traced at the control's quota on that draw); **pooled** concatenates every seed's draws before
differencing, so it prices the subsample choice on top of the battle resampling. Pooled is the
honest one for "would a matched read have detected this".

**One rung was added after the first reading, and it is disclosed as such.** Battle-matching leaves
the arm's own-team decoder with only **62** battles against the control's 104 — the arm carries more
distinct trainee teams per battle (106 teams / 197 battles vs 76 / 198), so fewer of its teams reach
the 4-battle threshold. That makes battle-matching *unfair to the arm*. The `decoder_matched` rung
(caps 11/16/5, 247 battles, **102** decoder battles) was added to give the arm the control's decoder
frame. It moves the answer in the arm's favour and still does not detect.

## 3. The reading

### Frames

| view | battles | teams | states | own-team t1 decoder battles |
|---|---|---|---|---|
| control, as traced (5/10/5 nominal) | 198 | 76 | 6,080 | **104** |
| arm BATTLE-MATCHED (8/12/5, median of 30 seeds) | 197 | 106 | 6,257 | 62 |
| arm DECODER-MATCHED (11/16/5, median of 30) | 247 | 118 | 7,692 | **102** |
| arm ×2 (16/24/10, median of 20) | 326 | 132 | 10,114 | 176 |
| arm ×4 (32/48/10, median of 20) | 531 | 167 | 16,120 | 353 |
| arm FULL, as traced (40/40/10) | 627 | 183 | 18,739 | 429 |

### The registered row, and the two matched frames

| frame | arm `cond.own_team_r2.t1` | control | Δ (pooled) | label | seeds whose own CI clears 0 |
|---|---|---|---|---|---|
| **as read (arm FULL vs control)** | +0.0604 | −0.0237 | **+0.0841 [+0.0323, +0.1825]** | **DETECTED** (vs zero) | — |
| **battle-matched** (30 seeds) | −0.0250 [−0.1404, +0.2491] | −0.0237 | −0.0013 [−0.2436, +0.3461] | NOT DETECTED | 2 up / 2 down |
| **decoder-matched** (30 seeds) | +0.0038 [−0.0770, +0.2352] | −0.0237 | +0.0275 [−0.1162, +0.3785] | NOT DETECTED | 3 up / 0 down |

Same story on `cond.own_team_r2.all`: as read +0.0837 vs −0.0098, Δ +0.0935 [+0.0252, +0.1836]
DETECTED; battle-matched arm −0.0467, Δ −0.0369 [−0.3011, +0.2041] NOT DETECTED; decoder-matched arm
−0.0034, Δ +0.0063 [−0.1150, +0.1846] NOT DETECTED.

### The frame-size curve — the artefact's own signature

| rung | decoder battles | `cond.own_team_r2.t1` (median [2.5, 97.5] over seeds) |
|---|---|---|
| battle-matched | 62 | −0.0250 [−0.1404, +0.2491] |
| decoder-matched | 102 | +0.0038 [−0.0770, +0.2352] |
| ×2 | 176 | +0.0369 [−0.0416, +0.1321] |
| ×4 | 353 | +0.0684 [+0.0437, +0.0999] |
| full (as traced) | 429 | +0.0604 |
| *control, for reference* | *104* | *−0.0237 [−0.1122, −0.0067]* |

Monotone in frame size up to ×4, where it saturates at the value the full frame reports. This is the
climbing curve the prediction named. It also says the control's own −0.024 is **not** evidence that
the control's critic knows nothing about its team — at 104 decoder battles neither arm can tell.

### The other nine rows

Nothing else changes label in the direction of a new claim, but **two of the registered
detections do not survive either**, and for two different reasons:

* `cond.spread_ratio_raw.t1_3` — as read Δ −0.2256 [−0.5187, −0.1046] DETECTED. Matched:
  −0.1079 [−0.3718, +0.1291] NOT DETECTED. Here the arm's OWN point moves with the frame
  (+0.1077 full → +0.2254 matched; the arm-against-itself delta is +0.1177 [+0.0131, +0.3546],
  the only DETECTED row in Table C) — an uncorrected, unclamped between-cell spread ratio is
  itself frame-size dependent, because the noise it does not subtract grows as cells shrink.
* `cond.spread_ratio_raw.all` — as read Δ −0.2945 [−0.6714, −0.0316] DETECTED. Matched:
  −0.2597 [−0.6432, +0.1065] NOT DETECTED. Here the POINT barely moves (arm +0.5108 → +0.5456);
  only the interval widens. That is not an artefact, it is a power loss: at the control's frame
  size the read simply cannot resolve a delta of that size. Both of these were unregistered
  companion rows; the registered `cond.spread_ratio.t1_3` was NOT DETECTED before and after.

`cond.spread_delta.*`, `cond.elo_slope` and `cond.opp_class_auc.t1` are stable in point and label
across every frame (`cond.elo_slope` moves by 0.0001 between full and matched).

## 4. The identity row (`identity.bias.*`) — NOT implicated, and here is why

**Structurally.** `identity.bias` is a WEIGHTED MEAN of the per-state quantity `V − MC`. Nothing is
fit; nothing is held out. A weighted mean's expectation does not depend on how many states entered
it, given correct weights — which rule 17's capture-rate IPW supplies. The own-team R² is the
opposite kind of object: an out-of-fold score of a model FIT on the frame, whose expectation rises
with the training set. The decoder-power channel convicted in §3 therefore cannot reach the identity
row.

**Empirically.** Restricting the arm's committed 800-state payload to the battles the matched
subsample keeps (~350 states, 143 battles), with the frame mass and the capture rates recomputed for
the subsample, over 20 seeds:

| stratum | `ipw` on FULL | `ipw` on MATCHED (median [2.5, 97.5] over seeds) | `pop` matched | `raw` matched |
|---|---|---|---|---|
| ALL | −0.0016 | **+0.0108 [−0.0193, +0.0398]** | +0.0955 (full +0.0225) | +0.1452 (full +0.0734) |
| early (≤10) | −0.0429 | −0.0425 [−0.0761, +0.0029] | +0.0509 (full −0.0178) | +0.0869 (full +0.0244) |
| mid (11–24) | +0.0229 | +0.0395 [+0.0022, +0.0804] | +0.1355 (full +0.0485) | +0.1814 (full +0.0976) |
| late (≥25) | +0.0182 | +0.0315 [−0.0063, +0.1131] | +0.1144 (full +0.0375) | +0.1761 (full +0.1055) |

The registered `ipw` weighting is stable — every stratum's matched across-seed interval contains its
full-frame value — while the un-reweighted `pop` and `raw` companions move by 0.05–0.10, which is a
direct demonstration that HT reweighting is doing its job on the selection axis as the frame
shrinks. The identity delta's point would, if anything, move AWAY from zero
(+0.0403 → ≈ +0.053 = +0.0108 − (−0.0418)).

🚨 **This is a stability check, not a matched re-read, and it is not offered as one.** A true
matched-quota `cf_audit` would draw a fresh 800 states from the smaller tree; restricting the
existing payload leaves ~350. **The control's own labelled payload is not in this measurement
directory, so no matched identity DELTA and no matched identity CI is computed here.** A
normal-approximation projection — scaling the arm's bootstrap SE by `sqrt(381/143)` and leaving the
control's alone — puts the matched delta near +0.053 [+0.014, +0.092], i.e. still clearing zero;
that is arithmetic on two published intervals, **not a measurement**, and must not be quoted as
one. Producing the real number needs fresh Monte-Carlo rollouts, which this job did not run.

## 5. Ledger-ready paragraph

> ### 2026-09-09 · RETRACTION (partial) · the cflabels own-team R² detection is a DECODER-POWER ARTEFACT of the quota asymmetry, not a lever effect — matched-frame Δ −0.001 [−0.244, +0.346] NOT DETECTED
>
> The 2026-09-09 READ of `ai_v12_12_ladder_cflabels` reported `cond.own_team_r2.t1` Δ +0.0841
> [+0.0323, +0.1825] DETECTED and flagged the quota asymmetry (arm 40/40/10, control 5/10/5) as an
> uncovered power/decoder confound with a matched re-read dispatched. It is now read
> (`measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/matched_quota/`, same
> `conditioning_meters` code, seed 0, boot 2000, **30 subsample seeds**, capture rates RECOMPUTED
> for each subsample so rule 17 still holds, nothing written under `models/`). Subsampled to the
> control's REALIZED per-opponent capture profile (8 wins / 12 losses — the nominal 5/10 inflates
> under shard work-stealing), the arm's frame is 197 battles against the control's 198 and its
> turn-1 own-team R² falls from **+0.060 to a median −0.025** [−0.140, +0.249], i.e. onto the
> control's own **−0.024**; the delta is **−0.0013 [−0.2436, +0.3461] NOT DETECTED**, positive on
> exactly 15 of 30 seeds. Battle-matching is if anything unfair to the arm — it carries more
> distinct teams per battle, so only 62 of its battles clear `MIN_TEAM_BATTLES` against the
> control's 104 — so a DECODER-MATCHED rung (11/16/5, 102 decoder battles) was read too: arm
> +0.0038 [−0.0770, +0.2352], Δ **+0.0275 [−0.1162, +0.3785] NOT DETECTED**, 3/30 seeds clearing
> zero upward. The frame-size curve is monotone in the decoder's own battle count (62 → −0.025,
> 102 → +0.004, 176 → +0.037, 353 → +0.068, 429 → +0.060), which is the artefact's signature and
> not an effect's. `cond.own_team_r2.all` behaves identically (Δ +0.0935 → −0.0369 battle-matched,
> +0.0063 decoder-matched, both NOT DETECTED). **Two unregistered companion detections also fail
> to survive**, for different reasons: `cond.spread_ratio_raw.t1_3` (Δ −0.2256 → −0.1079) because
> the UNCORRECTED, unclamped ratio is itself frame-size dependent — the arm's own value moves
> +0.1177 [+0.0131, +0.3546] between its full and matched frames — and `cond.spread_ratio_raw.all`
> (Δ −0.2945 → −0.2597) purely through power, its point barely moving. The registered
> `cond.spread_ratio.t1_3`, `cond.spread_delta.*`, `cond.elo_slope` and `cond.opp_class_auc.t1` are
> stable in point and label. **The identity row is NOT implicated**: `identity.bias` is a weighted
> MEAN, not a fit, so frame size moves its variance and not its expectation; restricting the arm's
> payload to the matched battles with the frame mass and capture rates recomputed leaves the
> registered `ipw` point inside its across-seed interval in every stratum (ALL −0.0016 → +0.0108
> [−0.0193, +0.0398]) while the un-reweighted `pop`/`raw` companions move 0.05–0.10 — but the
> control's labelled payload is not held here, so **no matched identity delta or CI is computed**
> and none is claimed. **Standing consequence: a DECODER-BASED conditioning row (`own_team_r2.*`,
> and any future fitted row) may not be compared across arms traced at different quotas.** The
> ladder must equalise the frame before the decoder rows are read — subsample the richer arm to the
> poorer one's realized profile, over ≥20 seeds, and report the across-seed spread beside the
> battle-clustered CI. Tag: **MEASURED · NOT DETECTED (own-team, matched frame) · the 2026-09-09
> DETECTED is WITHDRAWN.** No claim is made about the cf-label lever in either direction: the
> matched read is under-powered by construction, and a lever effect of the published size would not
> be visible at 104 decoder battles.

## 6. Recommendation — one line

**Make quota equalisation part of the ladder read for the DECODER-BASED rows**: `critic_read`
should subsample the richer arm to the poorer arm's realized per-opponent capture profile over ≥20
seeds and report the across-seed spread beside the battle-clustered CI (cheap — a whole rung is
~10 s of CPU on a symlink view), rather than re-tracing the control at the higher quota, which
costs a full eval cycle per arm and still leaves every already-read arm incomparable.

## 7. Hazards

1. 🚨 **`cf_audit.build_frame` REFUSES a battle with no `_reconstruction.json` sibling** and
   returns an EMPTY frame with the count in its `skipped` counter — no exception. A first version of
   `identity_matched.py` symlinked only `_states.npz` + `_summary.json`, so `pop_weights` divided by
   a zero mass, every weight became 0, and the bias read `nan` for all 20 seeds. `materialize` now
   links **every** sibling of each kept battle. A NaN was the only tell; had the stat degraded
   rather than failed, this would have shipped.
2. ⚠️ **A nominal quota is not a realized one.** The control's `5/10/5` is `8` wins and up to `12`
   losses per opponent on disk, because each shard unit carries `max(1, ceil(quota / n_shards))`.
   Matching the nominal numbers would have shrunk the arm ~35 % below the control and made the
   artefact look larger than it is. Always match on what the manifest RECORDS.
3. ⚠️ **Battle-matching is not decoder-matching.** `MIN_TEAM_BATTLES = 4` means the decoder frame is
   a nonlinear function of team diversity, and the arm has more teams per battle. Equal battle
   counts gave the arm 62 decoder battles against the control's 104. Any future quota equalisation
   should match the frame the row is actually fit on, not the tree.
4. ⚠️ **`cond.spread_ratio.t1_3` reads a clamped `0.000` on the arm at every frame.** Per the
   meter's own docstring the clamp makes it biased and non-monotone and the INTERVAL is the read —
   the point is not "no spread". It is reported here unchanged, not interpreted.
5. ⚠️ **The strength axis is injected, not recomputed per view.** It is a function of the run's
   snapshot ladder and the manifest's true win rates, which subsampling does not touch; computing it
   against a symlink view would fail outright (no `eval_results.jsonl`). `cond.elo_slope` moving by
   0.0001 across every rung is the consistency check on that.
6. ⚠️ **Only the ARM's identity payload is committed in this measurement directory.** §4 is
   therefore an arm-side stability check with a stated normal-approximation projection, and no
   matched identity delta is claimed.
