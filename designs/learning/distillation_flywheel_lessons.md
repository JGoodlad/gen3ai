# The distillation flywheel — what the v8 reproduction campaign actually taught us

*Status note as of 2026-09-01. Why "fold the exploiters back into the generalist" gave +69 ELO
once (v8, ai_v8_14) and has not gifted since; what a fold measurably does to a student; and the
instrument lessons that changed how every fold number must be read. Intuition first, then the
measured mechanism, then where each piece lives in the code and the ledger.*

**TL;DR.** The flywheel is: train narrow **exploiters** (best responses to the current generalist
on a few pinned teams), then **distil** them back into the generalist (the "fold"). It worked once
(v8: +69 anchored ELO, and +5.4pp on teams it was never taught, z=4.8). Three gen-era reproductions
(rev-2, rev-3, rev-4) did not reproduce it; rev-4 *robbed* untaught teams (−6.5pp). After nine
probes the surviving facts are: (1) **a fold's benefit is LOCAL** — taught teams gain ~+6pp, and the
gain does not radiate; (2) **the run-to-run noise floor is ~4pp**, which is the same size as most of
the "significant" fold effects we had been scoring, so single-run-vs-single-run fold comparisons
are underpowered by construction; (3) "every gen-era fold robs" is dead — rev-2's robbery was half
*meter* (greedy vs stochastic play) and half *composition* (which teams, which opponent
checkpoint), and inside the noise floor once the meter is held fixed; (4) what survives of the
robbery is rev-4's **shape** (damage concentrated on teams where the parent was already competent),
not its magnitude; (5) v8's gift is **not diluted taught content** — it is a broad behavioural
re-weighting of *when to switch* in winning positions, orthogonal to what the taught teams learned;
(6) the **critic is not the vehicle** of the gift — folds move the policy more than the critic, and
only the policy's movement correlates with per-team gain; (7) **the optimizer matters and
maturity does not**: v8's fold ran 3.2–6.6× gentler per step, `--lr` is INERT on a fork, and
distill-term lr 1e-4 Pareto-dominates 3e-4 on the real ingredients 6/6. The open suspect for the
sign flip is the one ingredient no arm varied: **student maturity** (v8's parent had 277M steps;
ours ~30M) — the one-week run is that experiment. Every fold number now carries a meter stamp
(policy regime · opponent checkpoint · team set) and a replicate floor beside it.

---

## 1. Intuitive level

**The flywheel as a story.** Self-play stalls: the generalist plays every team adequately and none
expertly (the amortization gap, see `generalist_specialist_amortization_gap.md`). So you hire
tutors. Each tutor (an *exploiter*) is a copy of the generalist fine-tuned on two or three teams
until it beats the generalist on those teams. Then you sit the generalist down with all its tutors
and have it imitate them (the *fold*, a distillation loss added to PPO). If the tutors' lessons
generalise, the generalist gets better at everything — the wheel turns, and you repeat with fresh
tutors against the improved generalist.

**What v8 saw.** The generalist got better at the taught teams (obviously) *and* at teams no tutor
had ever seen — the gift. That untaught gain is the entire value of the flywheel, because taught
teams are a rounding error of the 719-team pool.

**What we saw when we tried again.** The taught teams still improved. The untaught teams got
*worse*. Three explanations competed:
- *Narrowness robs*: teaching two teams hard makes you forget the rest (catastrophic interference).
- *Coverage*: the gift is just spillover from teams that resemble taught ones.
- *Something about the student*: an older, more settled student absorbs lessons differently.

**What the probes found, in plain words.**
- The tutors' lessons on their own teams are one thing; the untaught change is a *different* thing
  entirely (the two behavioural fingerprints are orthogonal, cosine 0.14 against a 0.60 ceiling).
  So the gift is not "a weaker copy of the lesson" — it is a side effect on the student's general
  habits, specifically *when it chooses to switch while ahead*.
- Two students trained identically from the same parent with no tutors at all differ by ~4pp on
  the same teams. That is the size of most of our "fold effects". We had been reading coin flips
  as verdicts.
- Rev-2's famous −7pp robbery shrank to +0.9pp when measured the way rev-4 was measured. Half the
  gap was that the student was scored playing greedily instead of sampling; half was which teams
  and which opponent snapshot. Neither number was wrong; they measured different things.
- The critic (the value head) barely changes off the taught teams; the policy does. The gift
  travels through *behaviour*, not through *evaluation*.

**The concrete example that makes the local-vs-global point.** Teach the student Skarm-Bliss stall
for 12M steps. On stall teams it stops attacking into resists and pivots more (+6pp). On an offense
team it never saw, v8's student *also* started switching less when losing a matchup and slightly
more when winning one, and won +5pp more. The gen-era student cut the losing-matchup switches by the
same amount — but got *greedier* when ahead (+5pp more super-effective attacks taken, −5pp switches
when ahead on mons) and gained nothing. Same lesson, opposite side effect. Amount of change did not
predict gain (ρ = −0.13); *direction* did (ρ = +0.57 on switch-while-winning).

## 2. Technical level

**The objective.** A fold minimises PPO's loss plus a distillation term, in action-form (a KL
between the teacher's action distribution and the student's on the student's own states), weighted
`--distill-coef`, restricted toward teacher teams by `--distill-team-bias`. Action-form beat
target-form by +7.6pp at matched dose (KL-alone corrupted; rank decoupled from performance). The
value side is a FitNets-style cosine hint on `value_pooled`, because scalar value distillation
crystallised the value classifier's rank.

**Extraction vs transfer are different quantities.** *Extraction* = teacher − reference on the
taught teams, measured on a matched zero-head-start harness with seniority as a separate term
(the matched-extraction row rule: +0.035 became +0.000 by fixing the baseline alone). The teacher
ceiling is ~0.69 on meter-class teams and 0.574 on coverage teams; funding (1.25M vs 2.5M/team)
moves extraction (−0.12 `ordered`) but rung 1 found teacher quality does not drive fold quality
(z=−0.44). *Transfer* = what the student keeps: ~+6pp on taught teams, dose-saturated, and the
untaught delta — the quantity the whole programme turns on.

**The noise floor, and what our intervals actually contain.** Our CIs are binomial or
battle-clustered: they contain *battle sampling* variance and nothing else. Two byte-identical
no-fold runs (`R2PLAIN` vs `R2CTRL`, same parent, seed, steps, source) differ by **−3.70pp**
taught and **−4.19pp [−6.94,−1.37]** untaught. Any effect of that size between two separately
trained runs is therefore undecidable without replicates. The teacher-count contrast the 40-team
fleet's shape rests on (−2.89pp, z=−2.14) sits below this floor; the shape is still unbiased but
the evidence for it is weaker than its z implied. Team-cluster dispersion must be *measured*, not
imported: it was 2.52× in one cell and 0.62–1.31× (mean ≈0.98) across 15 others, because at
n=200/team binomial noise swamps team heterogeneity.

**The meter is three axes at once.** Rev-2's untaught hop read −7.06pp (probe Q) and +0.88pp (M9)
for the same contrast because three things moved together: greedy vs stochastic policy, rev-1
`final_model.zip` (25M) vs its 24M snapshot as the opponent, and team set Q vs M (overlap 3/8).
The missing cell (greedy, set M) reads −3.44pp: **regime ≈ 4.3pp, composition + opponent
checkpoint ≈ 3.6pp**. Rule: a greedy result may not be quoted beside a stochastic one, and every
untaught number carries its stamp. The opponent-checkpoint axis is not yet separately identified
(a cell is running as this note is written).

**What a fold delivers to the critic (M5).** Off-slice, a fold moves the critic's *level*
(reliability/bias) not its *resolution* (Murphy resolution fell 13% on parent-generated states in
v8's own gifting fold; every critic meter's CI straddles zero in all four cells). The policy is the
larger mover and the only meter ordered against the gift: `KL(fold‖parent)` vs per-team gain
ρ = +0.51 [+0.05,+0.82]. So the critic is neither the vehicle nor the casualty of the gift.

**The optimizer, not the age (M7).** Maturity does not reduce distillation harm in the
distillability index; the *step size* does. `--lr`, `--batch-size` and `--n-steps` are INERT on a
fork (the resume path restores the checkpoint's optimizer; `model_build.py` prints "ignored on
resume"), so every gen-era fold ran at whatever the parent had annealed to, and v8's fold was
3.2–6.6× gentler per step than any gen-era fold. The lr licensing probe (18 cells, the actual rev-4
ingredients) found distill-term lr 1e-4 Pareto-dominates 3e-4 on 6/6 arms: collateral KL −39%,
absorption ceiling higher everywhere, net teacher content unchanged — the whole saving is the
content-free overshoot half. The live levers on a fork are `--grad-accum-steps`, `--n-epochs`,
`--min-lr`.

**Shape.** Count dominates conditioning (20→10 teams/teacher +0.077 SIG); a free per-team code
did not close the N=20 gap (+0.024 n.s.), so the conditioning-signal theory is unsupported.
Narrow beats wide at fixed compute (−0.0275, z=−2.88, confounded with per-team budget). Fold
quality tracks distinct teacher COUNT and is indifferent to team count — hence the 20×2 fleet.
The sweet-spot hypothesis (~10 teams/teacher) fuses two axes and remains unmeasured as a curve.

**What the robbery is, once the meter is fixed.** Rev-4's −6.50pp is ~1.6× the floor, measured
once, with no matched plain control (`R4PLAIN` is the named price: one `--distill-coef 0` fork
≈ 2 GPU-h plus one untaught arm). Its *shape* survives: −8.67pp (z=−4.32) on teams where the
parent was >0.55 and +0.00 on floor-level teams, while both no-fold replicates invert that
ordering. Plain training measured twice (−0.37 and −4.56) is not the robber's pattern. Repair
follows coverage; it does not radiate.

**Exploitability.** The "flat exploitability curve" was a composition artifact (two points, not
three; the meter mixed taught and untaught teams). Exploiters do NOT farm boundary states: their
wins are early, dice-fair, launched from already-ahead positions — knowledge and line-prep
against a static victim on pinned teams. That inversion holds under ground-truth Monte-Carlo
anchors (boundary-share ratio 0.10 under truth vs 0.59 under the head).

## 3. Where this lives in our architecture and record

- Flags: `--exploiter`, `--distill-teacher`, `--distill-coef`, `--distill-team-bias`,
  `--distill-value-coef` / `--distill-value-feat-coef` (the FitNets hint), `--stable-opponent-*`,
  `--trainee-teams`; the fork-inert set is documented in `src/main/train/model_build.py` and
  `src/agents/training/CLAUDE.md`.
- Meters: `python -m main.exploitability` (bookkeeping over admission artifacts, the meter/coverage
  split, refuses schema drift); the untaught-8 instrument
  (`designs/research_state/measurements/axis_split_untaught_arm.py`, `plain_training_robbery.py`,
  `greedy_meter_arm.py` — verbatim copies of one meter with one flag flipped, by design); the
  behavioural fingerprint (`v8_fold_behavioral_fingerprint_2026-08-31`, 25 model-free axes read
  from `LegalActions` + `gen3_data`, ports across architectures); `distillability_index_probe`;
  `lr_licensing_probe.py`.
- Team promotion: `python -m main.promote_teams` (seed-recorded uniform draw, exclusions in
  `designs/ai_v12/promotion_exclusions.json` — rebuilt from run metadata after going stale with
  its union size unchanged).
- Ledger: the scorecard `1a77edf` (REPRO-1..5), the M1–M9 entries of 2026-08-31, the 2026-09-01
  meter/composition resolution, the fleet launch entry and the funding-split chain entry.
- Memory: `project_v8_reproduction_scorecard`, `project_fold_transfer_is_local`,
  `project_fold_failure_eliminations`, `project_teacher_ceiling`, `project_fork_inert_flags`,
  `feedback_matched_extraction_row`, `project_distillability_index`.

## 4. What is running and what is owed

- The 40-team fleet (20 teachers × 2 teams × 1.5M, target R2ACTION) completes ~Wed 02:30–04:00;
  the 8-arm funding split (2.5M/team) is chained to it. Its load-bearing read is the UNTAUGHT cut,
  meter-stamped, with the replicate floor printed beside every delta.
- Owed measurements: `R4PLAIN` (makes rev-4's robbery causal); R3SELF's untaught cell; the
  opponent-checkpoint isolation cell (in flight); the selectivity-axis test across the fleet's
  folds (`switch|ahead_on_mons`, `switch|winning_matchup`, `take_SE_attack` vs win rate — M4's
  surviving next test).
- The maturity experiment is the one-week run; it needs pre-registration as such.
- Standing instrument gap: a change of switch TARGET (`SWITCH→SWITCH`, 21.6% of flips) is
  invisible to every rate-based behavioural metric we have.

## Synthesis

The flywheel's promise was that narrow lessons would radiate. The measured truth is that lessons
stay local, that the radiating part of v8's win was a *habit* change (selectivity about switching
when ahead) rather than transferred content, that our instruments could not tell a ~4pp fold
effect from two identical runs, and that the fold's optimizer step — not the student's age — is
the ingredient we can already prove matters. The programme is not dead: it is re-based on
replicates, stamped meters and a gentler step, with maturity as the one untested ingredient and
a one-week run as its test.

## See also

- `generalist_specialist_amortization_gap.md` — why the generalist needs tutors at all.
- `on_policy_self_distillation.md` — the search-as-teacher variant of the same loss.
- `credit_assignment_and_value_errors.md` §4 — what the same campaign taught about the critic.
- `designs/research_state/critic_calibration_plan.md`, `ladder_readiness.md`; root `CLAUDE.md`
  → *Exploitability*; `src/agents/training/CLAUDE.md` → distillation and fork flags.
