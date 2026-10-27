# Adversarial review of the X53-N1 design and the all-null strategy (DRAFTS at `cb85e54f`), 2026-10-26

**Scope.** [`design_x53_n1_fresh_breadth.md`](../../../endstate/design_x53_n1_fresh_breadth.md) ("X53") and
[`strategy_if_levers_null_2026-10-26.md`](../../../endstate/strategy_if_levers_null_2026-10-26.md) ("the strategy"),
both DRAFT. Neither doc is edited here.

**Context read.** `design_beat_metamon_path.md`, `UNDERSTANDING.md`, `sota_survey_2026-10-26.md` §5, the N1-P PREREG
and its `power/`, the D3 look-1 record, the belief-prior-mix results, the X51 table, the X61 review, the X59-S
PREREG §3, X26's `original_command` and `production_config.json`'s recipe. **The piloting-gap analysis had not
landed** (no commit by `f93386c76`), so nothing here leans on it.

**Method.** Two deterministic CPU Monte Carlos, run under rule 5b (nice 19, cores 5-7 and 13-15, `mem_cap.sh`
short lane, no GPU):
- [`power_x53.py`](power_x53.py) → [`power_x53.log`](power_x53.log): X53 §8's chain rebuilt from its prose
  (300,000 draws a cell), with two readings of the Stage-B interval, three effect time courses and two seed models.
- [`power_d1_small.py`](power_d1_small.py) → [`power_d1_small.log`](power_d1_small.log): N1-P's registered rule
  (its own `power.py`, reused) at +1.0 and +1.5 pp, the effects the strategy guesses for its T2 forks.

Abbreviations are the drafts': pp = percentage points; LB / UB = the bounds of a 95 % interval; P\* = X26 @280M;
C0 = X35's plain-continuation control; CF = X53's fresh control; F = `forum72`; H = `heldout72_hpfix`; P72 = the
procedural panel; U = H, F and P72 at equal weight (goal (b)'s meter); S = in-house strength; σ = the true
between-run SD of a run's F score; GPU-h = hours of our local card; lane-h = one CPU lane for an hour.

---

## Verdicts

| doc | verdict | in one line |
|---|---|---|
| **X53-N1** | **NOT READY TO REGISTER. The direction is sound.** | A fresh same-pin pair, with F as the primary, is the right shape. But the GAIN rule's error rate hangs on a seed SD floor of 1.5 pp. That floor is half what the X61 review recommended for fresh runs, and nothing measures σ before launch (M1). F is the panel where P\* has the least deficit (M2). 100M sits on the steep part of F's curve (M3). The reported effect is inflated by the screen (M4). A SCREEN-STOP licenses nothing (M5). |
| **The strategy** | **REVISE BEFORE USE.** | It names *resolution* as the likeliest reason for the nulls. Then it routes resolution to a stack fork that it guesses at +1.0 pp, read by a rule that calls a true +1.0 pp GAIN 0.19–0.20 of the time and NULL 0.32–0.47 (B1). It leaves out the zero-GPU resolution read that the fork data already allow. Its evidence for resolution is selected (M6). Its ranking ignores that six of its eight rows are one breadth bet, and that a lever's decision value is P(GAIN), not its expected pp (M7). Its escalation point is where the budget runs out, not a decision (M8). The fork-extension branch that X53 §4 sends to it does not exist in it (M9). |

---

## Findings at a glance

| # | sev | doc | finding | fix |
|---|---|---|---|---|
| B1 | **BLOCKER** | strategy | Resolution is the diagnosis, but no step of the tree can resolve +1–1.5 pp: D1's rule calls a true +1.0 pp GAIN 0.19–0.20 of the time and NULL 0.32–0.47. The zero-GPU resolution read is missing | add a registered resolution read to T1; size every T2 fork for +1.5 pp or drop it |
| M1 | MAJOR | X53 | The false-GAIN rate is controlled only while σ ≤ 1.5 pp, and the draft's σ = 2 figure is low. σ for a fresh run is unmeasured, and the X61 review set a 3 pp fresh-run floor | measure σ before launch, floor at 3 pp, pin the estimator |
| M2 | MAJOR | X53 | F is P\*'s best unfamiliar panel (0.530; premium 6.8 pp). The power is quoted at +4 and +6 pp, i.e. 60–90 % of F's whole premium | quote power at +1.5 to +3 pp; add an unhazarded panel with a larger deficit |
| M3 | MAJOR | X53 | X26's F rose +6.0 pp from 161M to 280M. A 100M Δ mixes pace with level. Under a late or dipping time course, the early stops kill a true +4 pp 21–24 % of the time, not 5 % | read the trajectory; make L50 non-binding; plan CF to the plateau |
| M4 | MAJOR | X53 | The screen includes pair 1 in the pooled estimate. Given GAIN, Δ̂ is inflated by +0.9 to +3 pp at plausible effects, and TRADE hands that number to the owner | estimate from the unselected pair, or condition the MLE |
| M5 | MAJOR | X53 | NULL is unreachable, which is confirmed. But SCREEN-STOP and UNDECIDED license nothing, so ≈ 40 GPU-h can end with the lever still open, and the strategy's rank 6 can buy it again | register what SCREEN-STOP retires, with its achieved power |
| M6 | MAJOR | strategy | The evidence for resolution keeps only the four positive estimates. With λ = 1 and X26c added, the pooled estimate halves to +0.9 pp. The belief channel is called both "closed" and evidence | pool by a registered rule over one family; drop "closed" |
| M7 | MAJOR | strategy | The ranking treats six breadth rows as independent bets. N5 starts from P\*, so it escapes none of the four explanations. The ranking sorts by expected pp, not P(GAIN). The costs use look 1 and leave out CPU | rank by P(GAIN) per GPU-h plus lane-h; one breadth bet; N5 from scratch |
| M8 | MAJOR | strategy | The 90 GPU-h exit equals T2 + T3 + T4's planned spend, so it is not a decision. By the doc's own numbers, only the scale bet reaches the gap | ask the owner at the end of T1, with an information trigger |
| M9 | MAJOR | both | The missing options: fork extension (dose versus plasticity: X53 §4 sends it to the strategy, which lacks it), the toy Option T, batch and rollout scale, and specialist distillation | add the rows, each with a cost or a reason it is not ranked |
| m1–m8 | MINOR | X53 | lockstep mechanics, the cost and wall-time wording, the "most null runs end early" claim, the comparative hypothesis, Stage C's cross-pin gate, the plateau tick's meter, team clustering, the licence change | below |
| m9–m12 | MINOR | strategy | P(all miss) bookkeeping, the plasticity probe's confound, N2's cost basis, CPU absent from the exit | below |

---

## X53-N1

### What holds

- **The arms are matched by construction.** Both are fresh `--arch production` runs at one tagged pin. They take X26's
  non-recipe tokens (X26's `original_command`: the four ride-along families, `--checkpoint-every-steps 1000000`),
  and the team mix is the only expected difference, checked by `checkargs`.
- **Stopping at 100M is safe for the recipe.** It has no schedule tied to `--steps`: `anneal_lr_start_steps` is
  null, and the LR follows the adaptive KL controller.
- **The case against X26 as the control (§2) is right.** It is cross-pin and cross-seed, and the obs, op-input and
  label changes of 10-26 lie between the two.
- **F is the right primary *against generator adjacency*.** 19 % of F's full sets are in the corpus, against 62 % of
  H's and 56.5 % of P72's.
- **Reproduced:** for σ ≤ 1 pp, the draft's power table and its GPU-h column agree with this review's Monte Carlo
  to within ≈ 0.01. The examples: Δ = 0, σ = 0: GO 0.067, GAIN 0.0002. Δ = +4, σ = 0: 0.890 / 0.64. Expected
  GPU-h at σ = 1: 40.8 / 55.4 / 69.9 / 76.5 against the draft's 41 / 56 / 70 / 77.
- **"NULL is unreachable" is confirmed.** The smallest Stage-B half-width over every simulated draw is 3.5 pp at
  2,400 games and 3.1 pp at 9,600. The 1.5 pp floor alone contributes 2.94 pp, so no game count reaches D1's ±3 pp
  band.
- **The cost arithmetic is right:** 100M ÷ 1,410 steps/s = 19.7 h per arm. X26 averaged ≈ 1,414 steps/s over 280M.

### M1 (MAJOR). The GAIN rule's error rate depends on an unmeasured σ, and the floor comes from the wrong regime

- **The draft's floor is half the earlier review's.** Stage B widens its pooled interval by max(measured spread,
  1.5 pp). The measured spread has 2 degrees of freedom (two runs per arm). The X61 review's M2 has already set out
  why that cannot size σ: "Two FRESH runs from step 0 diverge from the first team draw, and their spread is
  unmeasured". Its fix was a **fresh-run floor of 3 pp**. X53 uses 1.5 pp and cites only fork data (FINDING 1).
- **The false-GAIN rate, recomputed** for the whole chain (`power_x53.log` §A). "max" uses the measured spread,
  floored; "floor" uses 1.5 pp fixed. The draft does not say which it means.

  | true σ | false GAIN, "max" | false GAIN, "floor" | the draft says |
  |---|---:|---:|---:|
  | 1 pp | 0.005 | 0.006 | 0.01 |
  | 2 pp | **0.033** | **0.046** | 0.02 |
  | 3 pp | **0.061** | **0.098** | — |
  | 4 pp | **0.076** | **0.135** | — |

- **What follows.** The draft's σ = 2 figure is low, and its table stops where the fresh-run risk begins.
  - At σ = 3 pp, the "floor" reading gives a false GAIN one time in ten.
  - The power at +4 pp also moves with σ: 0.64 at σ = 0, 0.41–0.50 at σ = 2, and 0.32–0.45 at σ = 3.
  - **More Kakuna games do not fix this once σ ≥ 2** (`power_x53.log` §C). At +4 pp and σ = 2, GAIN is 0.41,
    0.45 and 0.47 at 2,400, 4,800 and 9,600 F games a run.
- **Fix:**
  - **Measure σ before registration, at zero GPU.** `rb_es_prod_s2001`–`s2008` are eight FRESH `--arch production`
    seeds to 15M (pin `95d014fa`; `original_command` read here). F against Kakuna on them at 600 games each is
    ≈ 13 lane-h. That gives a lower-step proxy with 7 degrees of freedom: a lower bound on fresh-run divergence, not
    a 100M value. Pair it with the CF s5301 − X26 @100M line the draft already plans.
  - **Floor at 3 pp, as the X61 review set.** Pin "max" (not a fixed floor) in the registration. Have the power
    table run to σ = 4, and print the false-GAIN rate at each σ.
  - If the proxy reads σ ≥ 2.5 pp, Stage B needs ≥ 3 runs per arm, or a t-interval on the runs (df = 2 per arm is
    nearly useless). Say so in the registration rather than finding out at the read.

### M2 (MAJOR). F is where P\* has the least deficit, and the power is quoted at implausibly large F effects

- **X51's P\* cells:**

  | panel | P\* | premium (home72 − panel) |
  |---|---:|---:|
  | home72 | 0.598 | — |
  | F | **0.530** | **6.8 pp** |
  | H | 0.497 | 10.1 pp |
  | P72 | 0.433 | 16.5 pp |

- **Goal (b)'s deficit is mostly not on F.**
  - F is already at the level goal (b) asks of U (≈ 0.53). The 4.3 pp the goal lacks sits mostly in P72 and H.
  - Those are exactly the two panels M4 hazards for any N1-trained model.
  - So a GAIN on F says little about goal (b). An N1-trained model's road to goal (b) runs through `forum216b`
    alone (X54), which the draft says.
- **The quoted power is at effects F cannot plausibly show.**
  - The "one screen" quotes +4 pp (0.39–0.64) and +6 pp (0.71–0.98). +4 pp on F is 60 % of F's whole familiarity
    premium, and +6 pp is 90 %: breadth would have to make unfamiliar human teams almost as easy as P\*'s practised
    ones.
  - The draft's own GUESS is +3 [−1, +7].
  - At a realistic +2 pp, the whole chain reads GAIN only **0.06–0.16** (σ = 0–2, "max"). At +3, GAIN lies between
    those and the +4 figures.
- **Fix:**
  - Quote the power at +1.5, +2 and +3 pp in the "one screen".
  - Add an **unhazarded human panel with a larger deficit** as a co-primary or the primary. The forum v2 manifest
    that `forum216b` was carved from has more pastes; the panel-validity read describes it. The panel must have a
    corpus set coverage ≤ F's 19 %, and it is registered before any X53 game.
  - Doubling the primary's team count also cuts the team-sampling term (m7).

### M3 (MAJOR). 100M is mid-climb on F, so the Δ mixes pace with level, and the early stops assume the wrong shape

- **The fact.** X26's F went 0.470 → 0.530 between 161M and 280M (X51, +6.0 pp). H went 0.448 → 0.497. This is
  where most of P\*'s unfamiliar-team piloting was won. X37b's "most in-house gain by 100M" does not carry over to F.
- **What it means.**
  - N1F pilots pool teams half as often as CF (≈ 2,000 against ≈ 4,000 per team by 100M). Half its games are on
    incoherent procedural sets.
  - A 10–20 % slower climb, read at 100M on a slope like X26's, costs ≈ 1–3 pp of F (MY JUDGEMENT, from the slope),
    whatever the plateau difference will be. The reverse (a breadth arm that climbs faster and plateaus at the same
    place) reads as a GAIN.
  - The Procgen analogy (§1) is a fixed budget at 200M, where the curves have separated. It is not a mid-climb read.
- **The early stops** (`power_x53.log` §B; σ = 1, rising base 0.30 / 0.40 / 0.45). Under a true +4 pp at 100M:

  | time course of the effect | FUTILITY-STOP | HARM + FUTILITY | final GAIN |
  |---|---:|---:|---:|
  | the draft's (¼ at 25M, ½ at 50M) | 0.05 | — | 0.55 |
  | "late" (0 until 100M) | **0.21** | — | **0.48** |
  | "dip" (−½ at 25M, 0 at 50M) | — | **0.24** | **0.47** |

  At σ = 3 the dip course stops 36 % of true +4 pp runs early. The draft's FINDING 2 names this risk but does not
  price it.
- **Fix:**
  - Make L50's futility stop **non-binding** (report only), or require it on two consecutive looks.
  - Keep L25's harm stop only for a catastrophic bar.
  - Read F's trajectory at L50, L100 and (on GO) L150 for both arms. A curve-crossing read (both arms' F against
    steps) says "pace" or "level"; one point cannot.
  - Plan CF to the plateau as part of Stage C, not as an option (m5). Without it, the plateau claim is cross-pin.

### M4 (MAJOR). The screen inflates the reported effect

- **The mechanism.** Pair 1 enters Stage B only if its Δ̂ ≥ +2 pp, and the pooled Δ̂ then includes that same pair.
  The chain's error rates include this, because the Monte Carlo simulates it. The **estimate** does not.
- **The bias, conditional on GAIN** (`power_x53.log` §A, "max"):

  | true Δ | σ = 1 | σ = 2 | σ = 3 |
  |---|---:|---:|---:|
  | +2 pp | **+2.3 pp** | +3.1 | +4.2 |
  | +4 pp | +0.9 | +1.9 | +3.0 |
  | +6 pp | +0.2 | +0.9 | +2.0 |

- **Who reads the inflated number.** TRADE hands the owner Δ̂_F / −Δ̂_P. Stage C's CF option is gated on N1F's
  plateau F. At a true +2 pp, the TRADE ratio's numerator is roughly doubled.
- **Fix:** register Stage B's effect estimate as pair 2's Δ̂ alone (unselected), or as the conditional MLE given the
  screen. Report the pooled Δ̂ beside it, labelled "screen-selected". The GAIN test can stay pooled, since its error
  rate is simulated.

### M5 (MAJOR). "NULL is unreachable" is honest, but leaving the miss branches unlicensed is not acceptable at ≈ 40–114 GPU-h

- **Under a true null**, the chain spends ≈ 39–45 GPU-h (`power_x53.log`: 38.6 at σ = 0, 45.0 at σ = 4). It ends
  in SCREEN-STOP ("a low-power screen, never a NULL") or a Stage-B UNDECIDED ("the orchestrator rules").
- **Nothing says what that retires.** The strategy's rank 6 ("fresh breadth without N1-P's GO") can buy the same
  ≈ 39 GPU-h again on a later trigger.
- **The data from a miss do say something.** For example, the chain ends without a GO under a true +4 pp with
  probability 0.11 at σ = 0 and 0.35 at σ = 3. A registered statement can carry that.
- **Fix.** Register a **RETIRE** outcome. On SCREEN-STOP, or a Stage-B UNDECIDED with Δ̂ < +2, "breadth from step
  0 with `n1_proc_v1` at 0.5" is retired for goal (b) in this era. The record prints the chain's power at +3 and
  +4 pp at the σ that M1's proxy measured. It is re-opened only by a new corpus or share (a new lever) or by new
  evidence named in advance.
- **Better still:** a Stage-B KILL class. Retire if UB < +3 pp: the effect no longer matters for F, given M2's
  headroom.

### MINOR (X53)

- **m1. The lockstep has no mechanism.**
  - `--restart-interval-hours` restarts by wall clock, not at L25 or L50. Stopping at an exact step means
    `--steps 25M`, then a `--model` resume. That is an argv change, so check it with `checkargs`, and confirm that the
    KL-controller, optimizer, snapshot-pool and PFSP state carry over exactly as in a restart.
  - The stop rules do not need the arms paused. Both runs can keep training while the 25M and 50M checkpoints (saved
    every 1M) are read, at the cost of the GPU-h on a stopped run.
  - Pausing idles the card for the read (L50 ≈ 16 lane-h ≈ 5 h on rule 5b's three cores). §6's "Stage A ≈ 1.7 days"
    leaves that out.
- **m2. The throughput is unverified at this pin.** The Rust full-parity batches (moves 364 of 369) and the 10-26
  obs and label changes lie between the X35 pin's measured 1,388–1,433 steps/s and HEAD. The 1,000 steps/s HOLD
  bar accepts a Stage A of up to ≈ 55 GPU-h without comment. Measure 1M steps at the pin (one fresh run, ≈ 12 min)
  before quoting 39.
- **m3. "The early stops end most null runs by 100M" is wrong as worded.** The early stops end 17–38 % of null
  runs (harm + futility, σ = 0–4). SCREEN-STOP happens AT 100M, after the full 39 GPU-h. The expected-cost figure
  (39–45) is right; the sentence is not.
- **m4. The comparative hypothesis is not what the design tests** (§1: "from step 0 the effect is larger"). X53's
  Δ (fresh, 100M, 120 pilots per corpus team, half the pool dilution over the whole history) and N1-P's Δ (P\* + 20M,
  24 pilots, 4 % dilution, another pin) differ in dose, stage, pin and base level. Drop the comparison, or define it
  as descriptive.
- **m5. Stage C's gate uses the cross-pin comparison that §2 rejects.** CF goes to the plateau "only if N1F's plateau
  F beats P\*'s 0.530". §2 argues that 160+ commits could move F by a few pp, which is exactly a gate's margin. For
  any adoption question, run CF to the plateau by default (+36 GPU-h, and price it). Goal (b) itself is absolute and
  needs no control.
- **m6. Stage C's plateau tick reads the in-house meter**, which X37b showed is blind to piloting. X26's F still
  rose +6 pp after the in-house curve flattened, so the tick can stop N1F early on the lever's own meter. Add an F
  read at the tick, or a floor of 250M.
- **m7. Team clustering is ignored.** Every interval is a binomial over games on 72 fixed teams. It generalises to
  F's teams, not to unfamiliar human teams. With team-level effect heterogeneity τ, the team-sampling SE of the mean
  effect is τ / √72. That is ≈ 0.6–1.2 pp for τ = 5–10 pp (GUESS), the same size as the seed floor. Report a
  team-cluster bootstrap beside the rule. The rule is D1's, so this is inherited.
- **m8. The licence is widened without saying so.** N1-P §6 licenses "a FRESH run … against X26 @100M (≈ 21 GPU-h)".
  The draft spends 39–114 GPU-h. §2 justifies the change well, but the orchestrator must ratify the wider spend, and
  the draft should say that it differs from the licence. Separately: N1H shares CF, so a second comparison against
  one control needs its multiplicity stated, and its timing fixed (X59-S's read decides whether N1H exists, and so
  where the pair runs, §9). That means X53 cannot pick cloud or local until X59-S reads.

---

## The strategy

### What holds

- **The arithmetic of P(all miss) is right:** 0.8 × 0.85 × 0.85 = 0.578. FINDING 1's direction is right:
  positively correlated misses raise it.
- **The four explanations** (plasticity, capacity, resolution, the meter) are a useful frame.
- **A zero-GPU week before any fork** is the right instinct.
- **"No row except the scale bet reaches 4.3 pp"** is honest, and it is the doc's most important sentence (M8).
- **The search and ensemble entries are present**, with their reasons (the brief's "test-time search with the
  real-opponent referee" is E1-0's machinery, in T5).

### B1 (BLOCKER). The tree has no step that can resolve the effect size it believes in

- **The chain of claims:**
  - §2 calls resolution "the explanation most likely to be true": each lever is +1–2 pp, and a ±4 pp read cannot see
    it.
  - Its remedy is the stack fork (rank 3). The stack's own GUESS is **+1.0 pp** ("sub-additive"), smaller than D1's
    point estimate alone.
  - The stack is read by D1's rule. At that size the rule calls GAIN one time in five, and **NULL** (which retires
    the lever) one time in three to two.
- **The power** (N1-P's registered rule on F, `power_d1_small.log`):

  | true Δ | final GAIN, σ = 0 | final GAIN, σ = 1 |
  |---|---:|---:|
  | +1.0 pp | GAIN 0.19 · NULL 0.47 · UNDECIDED 0.34 | GAIN 0.20 · NULL 0.32 · UNDECIDED 0.47 |
  | +1.5 pp | GAIN 0.36 · NULL 0.32 · UNDECIDED 0.32 | GAIN 0.34 · NULL 0.23 · UNDECIDED 0.43 |

  N2, H-TEAMS and N3 are guessed at +0.5 to +1.5 and read by the same rule. The NULL column is the trap: D1's ±3 pp
  band retires a real +1 pp lever more often than it adopts one. The earlier plateau NULL and UNDECIDED reads are
  therefore as consistent with +1–1.5 pp as with 0.
- **So T2 spends ≈ 37–60 GPU-h** on reads that the doc's own diagnosis says will come back UNDECIDED or NULL. The T6 exit
  then fires on schedule (M8).
- **The missing read costs no GPU.**
  - Fork seeds overdispersed by ≈ 0 (D1's five, N1-P review M2; D3's φ̂ 0.60). So for the X35 frame the binding
    noise is Kakuna games, which are CPU.
  - The finals of D1 (5 seeds), D3 (3, 5 after look 2), N1-P (3–5), X52 (2) and the shared C0 (5) are all banked.
- **Fix: a registered RESOLUTION READ in T1.**
  - New unit seeds, a rule written before any game, and F (and U where the corpus allows: M4's hazard applies to
    N1-P).
  - Sized for +1.5 pp at 80 % power: ≈ **17,400 F games per arm** (≈ 48 lane-h an arm at 360 games a lane-h). C0 is
    shared, so the C0 + D1 + D3 slice is ≈ 145 lane-h.
  - A pre-registered **breadth-family** pooled contrast (D1, D3 and N1-P against C0, one estimand: "breadth at the
    plateau, any corpus") is cheaper than per-arm reads. It is also the test the resolution hypothesis actually
    makes.
- **What it settles.** If its UB is below +1.5 pp, resolution is dead and the tree goes straight to the plasticity
  and capacity branch, or to the owner (M8). If its LB is above 0, the stack fork is licensed with a power target it
  can meet: ≥ 3,500 F games per seed final, and 5 seeds.
- **The CPU is real.** It is ≈ 2–3 days of the three agent cores. Still, it costs less of the scarce resource than
  one fork that cannot answer.

### M6 (MAJOR). The evidence for resolution is selected, and the belief channel is used both ways

- **The selection.** §2 cites "four point estimates between +1 and +2 pp": D1 +2.10, D3 +1.27, X52 unpractised
  +1.3, and λ = 0.25 belief +1.5. In the same scope the doc leaves out:
  - λ = 1 belief: −1.50 [−5.47, +2.47];
  - X26c: +0.08 [−3.90, +4.06];
  - D3's own F: +0.0 [−6.7, +6.7];
  - the archive and SPRT pools (NULL).
- **The pooled numbers** (inverse-variance; SEs from the published intervals):
  - the four positives: **+1.65 pp, SE 0.96**;
  - with λ = 1 and X26c added: **+0.92 pp, SE 0.80**, interval [−0.64, +2.49].
  - These levers are not one effect, so neither pool is an estimate of anything. The point is that the "+1–2 pp"
    pattern exists only after selection.
- **The belief channel is used both ways.** §1 lists it as "already closed for goal (b)" ("the prior mix moved no
  Kakuna games"). §2 uses its +1.5 as evidence. The results README says "an effect below ≈ 5 pp is not excluded":
  it is NOT DETECTED, not closed.
- **Fix:** drop "closed" (say "not detected at ±4–5 pp; not ranked"). Make resolution's evidence the
  breadth-family read of B1, defined before any number is pooled.

### M7 (MAJOR). The ranking is not honest about shared mechanism, detectability or cost

1. **Six of the eight rows are one bet.** N5, the stack, H-TEAMS, N3, fresh breadth and the scale bet all act
   through team breadth.
   - Conditional on N1-P (10⁴ teams) and D3 missing, they fail together much more often than independent guesses
     imply.
   - The "conditional" GUESSES barely move. The survey's N5 (+1 to +5) becomes +1.5 (−1, +5), with its upper tail
     unchanged.
   - §3's "0.10–0.25 each" invites a reader to compound seven chances. Say "one breadth bet at ≈ 0.15–0.25 in
     total, plus N2 and N4".
2. **N5 at rank 1 escapes none of the doc's four explanations.**
   - The student starts from P\* (survey §5 N5: "a student from P\*"), which is plasticity. It is the same width,
     which is capacity. It is one read, which is resolution.
   - Its corpus is P\*'s own play on 10⁴ unfamiliar teams, which is the deficient piloting being fixed. The
     advantage filter relies on a critic that §4's last routing line suspects on exactly those teams.
   - "On X59-S's machinery" is also inexact: X59-S's students are FRESH (`main.fresh_checkpoint.build_fresh_model`,
     X59-S PREREG §3).
   - A **from-scratch** N5 student would at least escape plasticity. State which one is meant, and re-guess it.
3. **Expected pp is the wrong numerator.**
   - A +1.5 pp lever read by D1's rule is adopted with probability ≈ 0.34–0.36 (B1), and a +1.0 pp lever with
    ≈ 0.19–0.20, not 1. The decision value of a
     row is ≈ P(GAIN on its registered rule) × the gain, per cost.
   - That flattens ranks 1–5 to near-ties. The doc's own "the ranking separates costs better than effects" is the
     admission. Say it in the table, as a P(GAIN) column.
4. **Cost basis.**
   - "GPU-h to the first decisive read" uses look 1 (12.3) for the forks. But look 1 decides rarely: N1-P's
     registered CONTINUE is 0.45, and UNDECIDED-final is the modal class. On the final's 20.5 GPU-h, N2's
     1.2 per 10 GPU-h becomes 0.73.
   - CPU, which the cloud plan names as the bottleneck, is outside the ratio. N5's 75–115 lane-h is ≈ 25–38 h of
     the three agent cores, and E4's 300–470 lane-h is ≈ 4–6.5 days.
   - Rank by GPU-h and lane-h together, or add a lane-h column to the ratio.

### M8 (MAJOR). The escalation is where the budget runs out, not a decision

- **The arithmetic.**
  - T2's three forks (≈ 37–60 GPU-h at look 1 or final), plus T3 (39, or 13–15 for the toy), plus N5's 8 come to
    ≈ 60–107.
  - So "≈ 90 GPU-h without a GAIN" fires on roughly any path that runs the plan to its end.
  - By the doc's own §3, even a GAIN on every rank 1–5 lever does not close the 4.3 pp gap. Only the scale bet's
    point estimate does.
- **What the owner question depends on.** It is "fund ≈ 150–250 GPU-h and a second card, or re-scope (b)". Its
  answer depends on T1's information, not on spending 90 GPU-h. A second card has a lead time.
- **Fix.** Escalate at the **end of T1**, with B1's resolution read and the plasticity probe in hand, and an
  information trigger:
  - if the breadth-family UB < +1.5 pp, and the plasticity probe does not read LOSS, escalate at once (no cheap
    lever remains);
  - otherwise run T2 with a cap, and escalate at ≈ 40 GPU-h without a GAIN.
  - Count lane-h in the exit too (m12).

### M9 (MAJOR, both docs). Missing options, and a branch that has no home

- **Fork extension: dose versus plasticity.**
  - X53 §4 rejects extending N1-P's seeds and C0's (+80M each, ≈ 32 GPU-h) as X53's follow-up. It calls this "the
    cheap complement if N1-P is UNDECIDED with Δ̂ > 0 (the strategy doc)". The strategy doc has no such row: its
    UNDECIDED-Δ̂ > 0 branch goes to fresh breadth (rank 6).
  - X53 also keeps the toy "Option T" as "the cheap screen if N1-P is UNDECIDED". The strategy has no row for it
    either.
  - **Dose is a fifth explanation the strategy misses.** A 20M fork gives each corpus team ≈ 24 pilots and removes
    ≈ 4 % of a pool team's exposure. Plasticity predicts that an extension stays flat. Dose predicts that it keeps
    climbing.
  - This is the cheapest GPU read that separates them, and it uses the banked frame and an existing control
    (extend C0 too).
- **Batch and rollout scale.**
  - Breadth spreads each team's gradient signal ≈ 17× thinner (12,000 teams against 719). The standard partner of
    diversity is a larger batch (the gradient-noise-scale argument; OpenAI Five and AlphaStar both scaled batch with
    diversity). This is MY JUDGEMENT, not re-fetched.
  - Neither doc lists it, even as "not ranked, because …".
  - The effective batch is 65,536 (2,048 × 32 in recent production argv; `UNDERSTANDING.md`).
- **Distillation of practised specialists.**
  - X52 practised +3.2 [−0.2, +6.8], so practice works on the practised teams. Test-time adaptation is kept only as
    a deployment regime.
  - Practise K human-built teams (not the panels), then distil the K specialists into the generalist (policy
    distillation). That turns the regime into a training lever.
  - It is costly (≈ 0.6 GPU-h a team), but at K ≈ 100 it is ≈ 60 GPU-h: comparable to rank 7, and aimed at the
    premium directly.
- **Reward shaping** and **opponent-pool diversity at the fresh start.** These are fine to leave unranked, but
  give the reason. Shaping has no mechanism tied to piloting. Pool diversity is inside X53's CF and N1F by
  construction.
- **More seeds of what worked** is B1's resolution read.
- **Fix:** add the extension and Option T rows (with their triggers, matching X53 §4), the batch row, and the
  distillation row, each costed or with its reason not to rank.

### MINOR (strategy)

- **m9. The P(all miss) bookkeeping.**
  - D3's 0.15 is its *registration* prior. After look 1 (+1.27, CONTINUE), the input should be D3's look-2 power
    at its look-1 posterior.
  - X59-S's "HUMAN-DATA-BETTER" is not a Kakuna GAIN on goal (b)'s meter. Mixing it into "all three miss a GAIN"
    mixes outcome types.
  - With D3 and N1-P perfectly correlated, the figure is 0.68. State the range 0.58–0.68.
- **m10. The plasticity probe is confounded by the training stage.**
  - Dormant-unit fraction, effective rank and fine-tune speed all differ between X26 @25M and P\* for reasons
    besides lost plasticity. A less-trained network starts at a higher loss, so it drops faster.
  - The literature probes plasticity by fitting a *fixed new target* to a fixed loss (MY JUDGEMENT, as the doc's
    own; not re-fetched).
  - It needs registered bars before T3 routes ≈ 39 GPU-h on it.
  - A behavioural complement: N1-P's argv forked from X26 @100M for 20M at the X35 pin, against the same fork from
    P\*. The difference in Δ is plasticity in the meter's own units (≈ 8–16 GPU-h).
- **m11. N2's p and its look-1 cost.** The flag's mask probability p is unregistered (a second lever if tuned). The
  12.3 GPU-h is look 1 only (M7.4).
- **m12. The exit counts GPU-h only.** T1 (40–60 lane-h), N5 (75–115), every fork look (≈ 44) and E4 (300–470)
  share three agent cores. Without a lane-h cap, the CPU sets the wall time, as the cloud plan's FINDING says.

---

## FINDINGS (standing rule 7)

1. **The piloting-gap analysis had not landed** (no commit by `f93386c76`). The strategy's routing on it was not
   checked against its numbers.
2. **This review's Monte Carlo rebuilds X53's chain from its prose.** Where the prose is ambiguous, the model is
   stated: the Stage-B estimator (two variants), the seed effect shared across looks or redrawn per look, and no
   early stops in Stage B. The registration's own `power.py` on the registered code supersedes it.
3. **The +6.0 pp F climb from 161M to 280M is one run, one seed (X51)**, at 400 games a cell (F's Newcombe +0.060
   [−0.009, +0.128], n.d.). M3's argument uses its direction and size, not a detected effect.
4. **`rb_es_prod_s2001`–`s2008` were read only for `original_command` and checkpoint steps** (15,000,075). Whether
   they are playable at a current `main.reads` commit, and their exact era, are UNVERIFIED.
5. **No game was played and no GPU was used.** Both Monte Carlos ran in the short lane (peak 0.11 GB).
