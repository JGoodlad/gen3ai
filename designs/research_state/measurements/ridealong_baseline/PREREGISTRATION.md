# X26 — the ride-along baseline: pre-registration (2026-09-30)

**Owner, 2026-09-30:** *"Once we are ready to train on the GPU, establish a baseline just with the
ensemble of value heads to test uncertainty, the trick where you predict a random output to see how
often you've seen the state, and the Q, A and B values. Start getting baselines before we add more
complicated arms."*

**What this run is.** The FIRST GPU training run on the M5 Rust core after the switch is the
production recipe plus the four DETACHED ride-along heads (`gen3_ridealong_heads_v1`, config v126;
`designs/ARCHITECTURE.md` §3.4, `designs/model/readouts_and_value_routes.md`). The heads observe and
never steer. `ridealong_update_test` pins one real update with the heads ON vs OFF as BIT-IDENTICAL
in the policy, trunk and V parameters, the PPO optimizer state, the PPO scalars and the RNG. So this
run IS the production-recipe run, with meters attached. **Every later arm compares against THIS run:
X23 (policy sharpness) first, then X5, X13, and any learner or recipe arm.** They compare on strength
(the standard reads) AND on the ride-along meters below. An arm carries the same four flags so its
meters exist; they cost it nothing in learning.

## The argv

```
python -m main.launcher --restart-interval-hours 3 --device cuda --arch production \
  <the K10(a) RECIPE block, applied by --arch production once it lands> \
  --critic winprob --terminal-indicator --victory-value 1.0 --draw-penalty 0 \
  --env-core rust \
  --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 \
  --steps <the production run's registered length> --run-name <assigned at launch>
```

- **The recipe.** K10(a)'s recipe block in `designs/production_config.json` is not on main as of
  this registration. If it has not landed at launch, the recipe is N0's (`ai_v14_01_base`)
  RECORDED training flags (`metadata.json:original_command`), with every value that differs from
  the parser default typed out, per the K10(a) note on the five silent defaults. `python -m
  main.checkargs --argv "…"` must print the ARCH SURFACE as matching and no refused combination.
  The four ride-along flags are `family=CRITIC`, so they are off the ARCH surface by declaration.
- **`--opp-intent-coef` > 0 is required** (production's 0.05, written by `--arch production`). It
  is what aligns the one-ahead opponent labels that B trains on.
- **Launch gates, in addition to the SOP's:** `ridealong_update_test` and `ridealong_heads_test`
  green at the launch commit. `ridealong/*` present in TensorBoard from the first update.
  `ridealong/disabled` = 0 at every read. If it reads 1, the heads hit a non-finite step and
  stopped: that read point is VOID for the heads, and the run itself is unaffected.

## Overhead (GPU learner benchmark, `learner_benchmark run --device cuda --ridealong`)

MEASURED, 2026-09-30, arm C's real 98,304-row buffer, `--compile-trainer` on, one RTX 3080 Ti, box
load 30–50 on 16 cores (contended: warned, not stretched). The ride-along config is the baseline spec
(ensemble 5, RND, A 5, B 5), attached to the loaded policy exactly as a v126 build makes them.
- **Heads trained on all 10 PPO epochs:** update 75.97 s vs 67.05 s baseline (median of 3), **+8.9 s =
  +13 %**. The bracketed `ridealong` phase is 6.44 s per update. That is far past the ~2 % instrument
  budget, so the heads now train on PPO's FIRST epoch only (`RIDEALONG_EPOCHS` = 1) with fused Adam.
- **Heads on epoch 0 only:** ESTIMATED at about a tenth of the above (~0.9 s, ~1.3 %). The re-measure
  is PENDING and will be recorded here.

## The read

**Instrument:** the offline reader `python -m main.ridealong_read` (CPU forward passes, no games;
it lands with its plumbing smoke in `smoke_2026-09-30/`). It reads the run's
own checkpoints, which carry the TRAINED heads (`policy.ridealong`), on two sets:
- the fixed M5 Lane S bank (`m5_laneS/bank_v1`: 20,712 decisions, 580 battles, outcome-labelled);
- the X4 pre-read's 1,600 ground-truth turns (`m5_laneS/truth_v2`, variant M per F-X4-1).

**Floors.** Every verdict is read against the SAME reader on the SAME checkpoint with FRESH heads
(randomized priors only). The heads must beat their own untrained floor, not only chance.
`smoke_2026-09-30/` holds those floors for the three pre-read checkpoints.

**Read points:** 10M and 25M (trajectory only; no verdict), and the run's FINAL checkpoint (the
verdict). **Intervals:** 95 % battle-clustered percentile bootstrap, 1,000 draws, fixed seed (the
pre-read's method). The TB `ridealong/*` series are MONITORING: they are on-policy rows the heads
just trained on, and are never a verdict.

| # | question | meter | decision rule (FINAL checkpoint) |
|---|---|---|---|
| R1 | Does ENSEMBLE disagreement predict V's actual error BEYOND V's own uncertainty? | the members' spread in LOGIT space (`ens_logit_std`). PRIMARY: its AUROC for \|V − z\| > 0.5 on the bank read WITHIN quintiles of V's own binary entropy (`auroc_within_ref_quintiles`). A spread of member PROBABILITIES is mechanically largest where V is near 0.5, so the raw AUROC partly re-reads V's own uncertainty: the smoke's UNTRAINED priors score raw 0.63–0.66 but within-quintile 0.54–0.56. Secondary: the raw AUROC, and the top/bottom-decile error ratio | **USABLE** for allocation iff the within-quintile AUROC's lower bound > 0.56, i.e. above the untrained-prior floor of 0.54–0.56, **and** the top/bottom-decile ratio's lower bound > 1.25. **NOT DETECTED** iff the within-quintile interval contains the untrained floor, re-read by the same reader on the SAME checkpoint with fresh heads. Anything else is **WEAK** |
| R2 | Does RND novelty predict V's error, and flag unfamiliar states? | the same AUROC for `rnd_z`. Coverage: the AUROC of `rnd_z` for held-out rows whose TEAM is absent from the reader's train split vs present | error: the R1 rule. **COVERAGE METER** iff the unseen-team AUROC's lower bound > 0.55 |
| R1×R2 | Which is the better allocator? | the paired AUROC difference (ensemble − RND), same resamples | the winner is the one whose paired interval excludes 0. Otherwise **TIED** (use both) |
| R3 | Does A rank actions consistently with the ground truth? | within-turn Spearman of A (the member mean) vs truth, over legal actions, on the 1,600 turns. Reference rows: the policy's own logits on the same turns, and the pre-read's one-ply Q̂ (ρ 0.226–0.243) | **CONSISTENT** iff ρ_A's lower bound > 0. **ADDS BEYOND THE POLICY** iff the paired ρ_A − ρ_logit has lower bound > 0. **A IS A POLICY ECHO** iff that paired interval contains 0 while `corr(policy logit, A)` > 0.5. Argmax-A regret on decisive turns is reported beside the policy argmax's |
| R4 | Are the starved near-best moves flagged as uncertain? | AUROC of the per-action A member spread (`adv_std`: each member centred on its OWN legal-action mean, uniform weights, NOT under π; π-centring pulls the members together on the actions π plays, so a π-centred spread flags starved moves even untrained, AUROC 0.73–0.75 at init). It compares STARVED near-best actions (π < 1 %, truth within 0.1 of the best) with (a) every other legal action and (b) the starved actions that are NOT near-best, on the truth turns | **FLAGGED** iff (b)'s lower bound > 0.55, i.e. the spread singles out the GOOD starved moves, not starvation as such, **and** (a) also clears the fresh-heads floor on the same checkpoint. This decides whether A's spread can steer where counterfactual labels or search effort go (X25 USE), when those come back in scope |
| R5 | Does B carry outcome information? | the Brier score of Q_played = V + A(a) + B(b) vs Brier(V) on bank rows where the opponent's action is in α's support. Paired difference, same resamples | **B INFORMATIVE** iff Brier(V) − Brier(Q_played) has lower bound > 0. Report the α support's label rate (1 − the miss rate) beside it |

**Not claimed by this run.** Strength. The heads cannot move it, by construction and by test, so
the run's strength reads (ladder, anchors, untaught meter) are the PRODUCTION RECIPE's own numbers.
They are recorded as the strength baseline for later arms, not as a result about the heads.

**Declared limits.**
- A and B learn from GAE advantages (λ = the run's `policy_gae_lambda`, V-bootstrapped). Their
  ceiling is therefore bounded by V's within-turn blindness, which the pre-read measured. R3 against
  Q̂ is a like-for-like floor, not a target.
- B is the simple pre-X5 parameterisation: α's support (seats by move id + SWITCH), conditional on
  the opponent choosing a listed option. It is to be re-based onto X5's flat pointer.
- The heads train on PPO's FIRST epoch only (each rollout row once; `RIDEALONG_EPOCHS` = 1, set by
  the GPU overhead measurement). V trains on all epochs, so the ensemble members see each row once
  where V sees it ten times. R1 asks whether their DISAGREEMENT predicts V's error, not whether
  they match V.
- The heads' Adam state resets at every launcher restart (it is not checkpointed).
- On a pre-v126 checkpoint the reader can only attach FRESH heads. Its numbers there are a plumbing
  smoke, never a read (`smoke_2026-09-30/`).
