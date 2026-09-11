# THE N-CURVE — would the win head condition if the policy moved SLOWER?

The owner's question, 2026-09-10. The [head refit](../winprob_head_refit_2026-09-09/README.md)
showed that the **terminal 0/1 label reproduces the online conditioning failure offline, from a
FROZEN policy**, at ~951 battles, and argued the result is scale-free because the between-cell
share of a proper scoring rule is a property of the loss and not of *n*. That is an argument. This
measures it: the **N-curve** of the same refit on a **stationary** dataset **23× larger** — one
frozen checkpoint, three independent eval seeds, ~24,000 battles and ~720,000 states per substrate,
nested subsets at N = 1k / 2k / 4k / 8k / 16k / ~21.5k battles, every fit scored on ONE fixed
held-out set of 2,400 battles.

Run 2026-09-10, offline, CPU only, nothing written under `models/`. Reproduce with `./run.sh`.
The prediction was registered in [`PREDICTION.md`](PREDICTION.md) **before any curve existed** and
is scored in §8.

---

## VERDICT

**Four findings. The second is larger than the one the measurement was commissioned for.**

**1 — On the owner's question: (ii) FLAT on `ctrl10M`, (i) RISE-THEN-SATURATE on `ctrl10M_b`, and
the two agree once you look at where each curve LANDS.** The terminal-label head converges to the
**same** place on both substrates — a turns-11–24 between-opponent spread ratio of **0.693** and
**0.665**, an all-states ratio of **0.609** and **0.557** — and it gets there by **N ≈ 4,000
battles ≈ 1.25 PPO rollouts**. On `ctrl10M`, whose online head is ALREADY at that level (0.759 /
0.652), the curve is flat from 1k to 21.5k: Δ at turns 11–24 **−0.000 [−0.020, +0.021]**, at all
states **−0.001 [−0.017, +0.014]**, turns-4–10 opponent-class AUC **−0.004 [−0.011, +0.002]**. On
`ctrl10M_b`, whose online head sits below it (0.464 / 0.388), the curve climbs **+0.182 [+0.158,
+0.209]** at turns 11–24 and then stops. **More stationary data buys a head at most the
conditioning the better of two identically-configured runs already has, and buys it inside one
rollout's worth of episodes.** A slower policy, a value replay or a stationary window is therefore
worth **at most** that gap and nothing beyond it — and the head has ~6.8 rollouts of stationary
data at N_max with no further movement. **The owner's hypothesis is REFUTED as a route past the
current ceiling**, and the ceiling itself is now measured rather than argued.

**2 — 🚨 THE PROBE READ'S TURN-1 OPPONENT DECODE IS AN OWN-TEAM CONFOUND, and that changes what
three measurements' headline meters mean.** [`frame_check.py`](frame_check.py): in the probe read's
frames the **trainee's OWN TEAM alone** predicts the opponent's class at **AUC 0.856** (CTRL) and
**0.877** (A), because only **37 of 180** and **47 of 216** distinct trainee teams ever face a
sentinel there. The own team is in the observation verbatim at turn 1. On these matched-team
frames all **602/602** teams face every opponent, the own-team channel carries nothing about the
class (**0.508**) — and neither does `value_pooled` (**0.502 / 0.508** against the probe read's
published 0.861 / 0.846, which re-runs here at 0.845 / 0.853 on its own extraction). **The opponent is genuinely unobservable at turn 1** (Gen 3 has no team preview) and
becomes observable only as it plays: `value_pooled` → class **0.50 (t1) → 0.67 (t1–3) → 0.82
(t4–10)**. So on a matched-team frame a turn-1 spread ratio of 0 is **BAYES-OPTIMAL, not a
defect** — the probe read's own PRE-REGISTERED expectation (ii), which it believed it had
refuted, was right — and the window in which "does the head condition on the opponent" has an
answer is turns ≥ 4. **In that window the online head already conditions**: its prediction decodes
the opponent's class at AUC **0.723 / 0.700** against `value_pooled`'s 0.812 / 0.848 and a null of
0.52, and its turns-11–24 spread ratio is **0.759 / 0.464**.

**3 — a third result the measurement did not go looking for: the CONDITIONAL target, the head
refit's recommended re-pricing, is HARMFUL on a matched-team frame on every meter but one.** At
N_max it takes the turns-11–24 ratio from the online head's 0.759 **down** to 0.465 on `ctrl10M`
and from 0.464 to 0.439 on `ctrl10M_b`, and Brier over all states from 0.1012 to **0.1275** (the
terminal refit reads 0.0972). The one meter it wins is the **ordering** of opponents: its turns-4–10
class AUC is **0.830 / 0.854**, matching `value_pooled` itself and beating both the online head and
the terminal refit with the delta's CI clear of zero. A target built from information the features
do not carry teaches the ORDER and destroys the AMPLITUDE.

**4 — "not enough passes" is refused a second time, harder.** 5× the gradient steps with NO early
stopping at N_max makes every meter worse: turns-4–10 class AUC **−0.078 [−0.099, −0.057]** and
**−0.048 [−0.067, −0.030]**, Brier over all states **+0.024 [+0.018, +0.030]** and **+0.019
[+0.014, +0.023]**, with the calibration slope diverging into saturation.

---

## 1. The three readings, registered before the run

Reproduced from [`PREDICTION.md`](PREDICTION.md); `summarize.py` applies them **in code**.

| | the curve | what it would mean | the treatment |
|---|---|---|---|
| **(i)** | the TERMINAL-label meters RISE with N toward the conditional target's level | non-stationarity / discard is the binding constraint | a slower policy, a value replay, a stationary window |
| **(ii)** | they stay FLAT from 1k to ~21k | the target's NOISE SHARE is the constraint; more stationary data does not help | target-side re-pricing; the owner's hypothesis refuted for this regime |
| **(iii)** | they rise only past ~10k battles | a data threshold exists — quantify it against what a rollout supplies | depends on where the threshold falls |
| **guard 0** | `cond_oracle` (the conditional target emitted verbatim) must reach a turns-1–3 ratio whose CI lower bound clears 0.80, or the (b) column is INCONCLUSIVE | the head refit's rule, kept verbatim | — |

**The rule needed one amendment, made in code and reported beside the original.** The registered
(ii) clause required `mlp_cond` to be **detected above** `online` at every N as a liveness check.
On this frame the conditional arm is detected **below** the online head on most buckets (finding 3),
which the registered wording could not express — it tested `detected`, not `detected AND positive`.
The amended rule (a) puts the sign in, and (b) uses guard 0 as the liveness check instead, since
`cond_oracle` reaching a ratio of **0.998 [0.941, 1.045]** proves the meter is movable on this
frame. `summarize.py` emits BOTH verdicts; the tables carry both.

---

## 2. The frame — and why it is stationary by construction

`eval_trace_gen` was run three times against the **same checkpoint** with three seeds. The three
`snapshot.zip` files are **md5-identical** per substrate and `extract.py` REFUSES otherwise: "one
frozen policy" is the premise, not an assumption.

| | `ai_v12_11_ladder_ctrl10M` @10M | `ai_v12_15_ladder_ctrl10M_b` @10M |
|---|---|---|
| trees (seed × games × 12 opponents) | 20260909×400 · 20260910×800 · 20260911×800 | same |
| states · battles · teams | **732,626 · 23,891 · 602** | **716,727 · 23,877 · 602** |
| per-tree overall true win rate | 0.8631 / 0.8596 / 0.8625 | 0.8400 / 0.8585 / 0.8517 |
| draws excluded (no binary outcome) | 109 | 123 |
| capture | **FULL** — every `capture_rate_*` is 1.0 or None, so **every HT weight is exactly 1.0** (asserted in `extract.py`, not assumed) | same |
| frozen-forward QC, max \|V_fwd − V_rec\| | **2.21e-06** | **1.94e-06** |
| held-out set (fixed across every N) | 2,400 battles / 74,403 states, 200 per opponent | 2,400 / 72,651 |
| training pool | 21,491 battles | 21,477 |

**Why this frame and not the head refit's.** The head refit pooled four or five training CYCLES —
four or five different policies — which is what made "non-stationarity" impossible to separate from
"too little data" there. Here the cycle index is an **eval seed**, so the three cells of one
opponent are **replicates**: a free noise check, and the head refit's hazard 2 (a conditional
target with no cycle term is unreadable on a rapidly improving run) cannot bite. Both cell keyings
are run — 36 (seed, opponent) cells at 400/800 games each, and 12 opponent cells with the seeds
pooled at 2,000 games each — and they agree on the quantity the reading turns on: the turns-11–24
trend of the terminal refit reads **−0.000 [−0.020, +0.021]** vs **+0.003 [−0.014, +0.021]** on
`ctrl10M` and **+0.182 [+0.158, +0.209]** vs **+0.184 [+0.160, +0.209]** on `ctrl10M_b`.

**The outcome side of the identity is the MANIFEST's win rate at that cell's own game count (400 or
800), not the held-out subsample's.** The outcome's true between-opponent spread is a property of
the policy and the opponent set; estimating it from 400–800 games rather than from the ~67 held-out
battles in a cell is strictly better AND is the same quantity at every N, which is what an N-curve
needs. The prediction side is the held-out battles only.

### Optimisation is budgeted in STEPS, not epochs

An epoch at 21k battles is 21× an epoch at 1k, so "the same number of epochs" hands the large-N
fits 21× the optimisation and makes a rise uninterpretable. Every fit here gets the same budget:
Adam, lr 1e-3, batch 1024, max 60,000 steps, validation every 25 steps on a grouped-by-battle
split of the training battles, patience 400 checks (**10,000 steps** without improvement). In
epochs that is 3.2 at N=1,000 and 0.8 at N=21,491 — the fits converge fast and then overfit, which
is what §6 is for.

| N (battles) | training states | as PPO rollouts (48 envs × 2,048 steps = 98,304) |
|---|---|---|
| 1,000 | 24,236 | 0.31 |
| 2,000 | 49,047 | 0.62 |
| 4,000 | 99,086 | **1.25** |
| 8,000 | 195,772 | 2.49 |
| 16,000 | 390,683 | 4.96 |
| 21,491 | 527,443 | **6.70** |

---

## 3. 🚨 THE FRAME FINDING — what a turn-1 "opponent decode" was actually decoding

[`frame_check.py`](frame_check.py), [`frame_check.json`](frame_check.json). Run over four
extraction dirs, including the probe read's own, which survives on disk.

| frame | battles | teams | teams ever facing a sentinel | **own-team LOO → opp_class AUC** | `value_pooled` → opp_class @t1 |
|---|---|---|---|---|---|
| probe read `A` @74M | 951 | 216 | **47 / 216** | **0.877** | 0.853 (published 0.846) |
| probe read `CTRL` @10M | 873 | 180 | **37 / 180** | **0.856** | 0.845 (published 0.861) |
| this frame, `ctrl10M` | 23,891 | 602 | **602 / 602** | **0.508** | **0.502** |
| this frame, `ctrl10M_b` | 23,877 | 602 | **602 / 602** | **0.507** | **0.508** |

The trainee's own team is in the observation verbatim from turn 1. In the probe read's frames a
decoder asked *"is this a sentinel?"* at turn 1 could answer it by reading **our** team — an
artefact of how that eval assigned teams, not information about the opponent. The probe read's
own machinery anticipated exactly this failure mode on the team axis (its `_perm_labels(kind=
"group")` identity-mediation reference, and the hazard it records about a "null" of 0.98) but the
reference was never run on the opponent axis.

**Observability, per turn bucket, on the stationary frame** (`value_pooled` → opponent class, AUC;
permutation null ≈ 0.52):

| | t1 | t1–3 | t4–10 | t11–24 | t25+ |
|---|---|---|---|---|---|
| `ctrl10M` | **0.502** | 0.671 | **0.819** | 0.801 | 0.789 |
| `ctrl10M_b` | **0.508** | 0.699 | **0.851** | 0.801 | 0.777 |

**What this does and does not overturn.** It does NOT touch the head refit's central result, which
compares two TARGETS scored on the same held-out outcome and needs no claim about turn-1
observability. It DOES retire the probe read's headline framing — *"the head is handed the answer
and does not use it"* — at turn 1, and with it the strength of the argument from the value path to
a target-side treatment. And it makes one row of this measurement uninterpretable by construction,
which is why the turn-1 opponent rows below carry a ⚠️ and the reading is taken at turns ≥ 4.

**The own-team axis survives and is the one to read at turn 1**: the own team IS observable there,
`value_pooled` decodes its leave-one-battle-out win rate at R² **0.266 / 0.224** against a null of
0.000, and the online head's prediction reads **0.114 / 0.062**.

---

## 4. The curves

Full tables, both substrates and both cell keyings:
[`tables_ai_v12_11_ladder_ctrl10M_cell.md`](tables_ai_v12_11_ladder_ctrl10M_cell.md) ·
[`tables_ai_v12_15_ladder_ctrl10M_b_cell.md`](tables_ai_v12_15_ladder_ctrl10M_b_cell.md) ·
(`_opponent.md` beside each). **Bold = the delta's own battle-clustered CI is clear of zero.**

### (a) the TERMINAL 0/1 target — the one the online head actually trains on

`ctrl10M` — the online head reads t1 0.000 · t1–3 **0.176** · t11–24 **0.759** · all **0.652**

| meter | 1k | 2k | 4k | 8k | 16k | 21.5k | Δ(N_max−N_min) | per doubling |
|---|---|---|---|---|---|---|---|---|
| spread ratio @t1–3 | 0.123 | 0.098 | 0.090 | 0.115 | 0.139 | 0.128 | +0.005 [−0.012, +0.016] | +0.001 |
| **spread ratio @t11–24** | 0.694 | 0.648 | 0.647 | 0.722 | 0.719 | 0.693 | **−0.000 [−0.020, +0.021]** | −0.000 |
| spread ratio @all | 0.610 | 0.553 | 0.560 | 0.629 | 0.633 | 0.609 | −0.001 [−0.017, +0.014] | −0.000 |
| **opp-class AUC @t4–10** | 0.735 | 0.715 | 0.723 | 0.735 | 0.738 | 0.730 | **−0.004 [−0.011, +0.002]** | −0.001 |
| own-team R² @t1 | 0.112 | 0.099 | 0.107 | 0.090 | 0.125 | 0.131 | **+0.018 [+0.006, +0.030]** | +0.004 |

`ctrl10M_b` — the online head reads t1 0.021 · t1–3 **0.097** · t11–24 **0.464** · all **0.388**

| meter | 1k | 2k | 4k | 8k | 16k | 21.5k | Δ(N_max−N_min) | per doubling |
|---|---|---|---|---|---|---|---|---|
| spread ratio @t1–3 | 0.071 | 0.092 | 0.120 | 0.132 | 0.169 | 0.151 | **+0.081 [+0.064, +0.096]** | +0.018 |
| **spread ratio @t11–24** | 0.483 | 0.572 | 0.669 | 0.659 | 0.754 | 0.665 | **+0.182 [+0.158, +0.209]** | +0.041 |
| spread ratio @all | 0.406 | 0.473 | 0.541 | 0.545 | 0.613 | 0.557 | **+0.151 [+0.133, +0.175]** | +0.034 |
| **opp-class AUC @t4–10** | 0.703 | 0.702 | 0.712 | 0.723 | 0.745 | 0.738 | **+0.035 [+0.026, +0.044]** | +0.008 |
| own-team R² @t1 | 0.059 | 0.053 | 0.062 | 0.089 | 0.077 | 0.104 | **+0.044 [+0.030, +0.060]** | +0.010 |

**Read the two tables against each other, because separately they say opposite things and together
they say one thing.** The end points are nearly equal — 0.693 vs 0.665 at turns 11–24, 0.609 vs
0.557 at all states, 0.730 vs 0.738 on the class AUC — while the start points are not. The
terminal-label head converges to a substrate-independent level, and **whether the curve appears to
rise is a fact about where that run's own online head already sits**, not about how much stationary
data the refit was given. Against `online` at N_max the same rows read:

| Δ(refit at N_max − online) | @t1–3 | @t11–24 | @all |
|---|---|---|---|
| `ctrl10M` | **−0.048 [−0.093, −0.035]** | **−0.065 [−0.085, −0.042]** | **−0.043 [−0.064, −0.026]** |
| `ctrl10M_b` | **+0.055 [+0.033, +0.071]** | **+0.201 [+0.178, +0.232]** | **+0.169 [+0.150, +0.193]** |

i.e. with 6.7 rollouts of stationary data and no non-stationarity at all, the refit does not reach
`ctrl10M`'s online head and overshoots `ctrl10M_b`'s to exactly the same place.

### (b) the CONDITIONAL target — the head refit's re-pricing, on a matched-team frame

At N_max, `ctrl10M` (`ctrl10M_b` in brackets):

| | online | MLP · terminal | MLP · conditional | `cond_oracle` (ceiling) | `value_pooled` |
|---|---|---|---|---|---|
| spread ratio @t11–24 | **0.759** (0.464) | 0.693 (0.665) | **0.465** (0.439) | 0.998 (0.978) | — |
| spread ratio @all | **0.652** (0.388) | 0.609 (0.557) | **0.409** (0.382) | 0.998 (0.978) | — |
| opp-class AUC @t4–10 | 0.723 (0.700) | 0.730 (0.738) | **0.830** (0.854) | 0.964 (0.979) | 0.812 (0.848) |
| own-team R² @t1 | 0.114 (0.062) | 0.131 (0.104) | 0.136 (0.058) | 0.056 (0.040) | 0.266 (0.224) |
| Brier @all | 0.1012 (0.1045) | **0.0972** (0.0988) | 0.1275 (0.1258) | 0.1235 (0.1238) | — |
| calibration slope @all | 0.96 (1.12) | 1.02 (0.97) | 1.30 (1.38) | 0.93 (0.89) | — |

**The conditional target wins the ORDERING and loses the AMPLITUDE.** Its prediction ranks
opponents better than anything except the target itself — 0.830 / 0.854 against `value_pooled`'s
own 0.812 / 0.848 — while its between-opponent spread is the SMALLEST of any condition, and its
Brier is 0.03 worse than the terminal refit's. The mechanism is not subtle: the conditional label
is a per-battle constant built partly from information the features do not carry, so a head fitted
to it regresses hard toward the base rate (calibration slope 1.30–1.38, i.e. **under-dispersed**)
and keeps only the component it can see. The head refit could not see this trade because its frame's
features leaked the opponent through the team.

### (c) where in N the movement happens — the segment split

Turns 11–24, `mlp_term`, each segment with its own battle-clustered CI:

| substrate | 1k → 8k | 8k → 21.5k |
|---|---|---|
| `ctrl10M` | **+0.028 [+0.004, +0.054]** | **−0.029 [−0.045, −0.012]** |
| `ctrl10M_b` | **+0.176 [+0.155, +0.202]** | +0.006 [−0.008, +0.017] |

`ctrl10M_b`'s rise — 0.483 → 0.572 → **0.669** → 0.659 → 0.754 → 0.665 — is complete by N ≈ 4,000
battles (**1.25 rollouts of episodes**) and the last 13,500 battles add nothing. `ctrl10M`'s tiny
early gain is **given back** over the same span: its last segment is detected and NEGATIVE. Neither
substrate is still climbing at N_max, and neither has a threshold anywhere near 10,000 battles — so
reading **(iii)** is refused on both.

---

## 5. The reading, per substrate

| substrate | rule's verdict (amended) | as the registered wording would have emitted it | in words |
|---|---|---|---|
| `ctrl10M` | **(ii) FLAT** | MIXED | no detected trend on any spread bucket or on the class AUC; the online head is already above the refit's converged level |
| `ctrl10M_b` | **(i) RISE, SATURATED by N=8,000** | (i) RISE | a real, large rise that is complete inside 1.25 rollouts and lands where `ctrl10M` starts |

**The campaign-level statement the two support jointly, and neither alone:** *stationary data is
not the binding constraint on the win head's opponent conditioning past ~4,000 battles, and the
level it buys is one the better of two identically-configured control runs already reaches online.*

### The (iii) arithmetic, since one substrate does have a threshold

The threshold is at **N ≈ 4,000 battles ≈ 99,000 states ≈ 1.25 PPO rollouts** at this run's
geometry (48 envs × 2,048 steps = 98,304 states ≈ 3,170 episodes per rollout, `--n-epochs 10`).
An online head therefore already receives, **inside a single policy iteration**, data of the order
the offline curve needs to saturate — and it carries its weights across iterations on top of that.
**There is no data deficit to fix.** A value replay over a stationary window of 4,000 battles would
hand the head what it already gets from one rollout; a slower policy would buy the same thing more
slowly. This is the arithmetic the head refit's scale-free argument asserted and could not show.

---

## 6. The MORE-OPTIMISATION control — "not enough passes", refused at 23× the scale

5× the gradient steps the early-stopped fit used, no early stopping, final weights taken.

| contrast at N_max | Δ ratio @t1–3 | Δ opp-class AUC @t4–10 | Δ own-team R² @t1 | Δ Brier @all |
|---|---|---|---|---|
| `ctrl10M` · terminal_long − terminal | **+0.066 [+0.035, +0.126]** | **[−0.099, −0.057]** | [−0.053, +0.004] | **[+0.0182, +0.0301]** |
| `ctrl10M_b` · terminal_long − terminal | +0.017 [−0.003, +0.080] | **[−0.067, −0.030]** | **[−0.059, −0.006]** | **[+0.0138, +0.0233]** |

More passes **raise** the turns-1–3 spread ratio on one substrate while **lowering** the opponent
decode and **raising** Brier on both — the signature of a head driving predictions toward 0/1 on the
training battles, not of a head learning the opponent. The head refit eliminated the
optimisation-HISTORY hazard from the other side (a fine-tune of the online weights lands where a
scratch fit lands). Together the two close treatment class 1 completely.

🚨 **A saturated prediction makes the calibration slope diverge, and it did**: the first pass of
this arm printed a slope of **−6.0e9**. `readout.py` now REFUSES a slope outside ±50 rather than
printing it; the underlying fits are unchanged.

---

## 7. Counter-hypotheses

**(a) Eval-population vs training-population states.** These frames are **eval battles against a
fixed 12-opponent panel (9 scripted bots, 3 frozen self-play sentinels)**, not training rollouts.
Training states come from a self-play pool that moves, at a different bot/pool mix, with
exploration noise in the trainee's own actions. Three things follow, and only the third is a real
threat. (1) The opponent panel is narrower than the training pool's — which if anything makes the
conditioning problem EASIER here, so a null is conservative. (2) The greedy-sentinel eval regime is
the one `ctrl10M`'s own eval runs, so the meter is comparable to every other read on this substrate.
(3) **The team-assignment design differs, and §3 shows that difference is decisive for what a
turn-1 decode means.** Whether TRAINING rollouts assign the trainee's team independently of the
opponent — which would make the matched frame the right model of the online problem — is **NOT
established here** and is the single largest open question this measurement leaves. **SURVIVES as
the principal limit.**

**(b) "The curve is flat because the meter is dead on this frame."** Refuted by the same table three
ways: `cond_oracle` reaches a turns-1–3 ratio of **0.998 [0.941, 1.045]**, the online head itself
reads 0.759 at turns 11–24 on `ctrl10M`, and `ctrl10M_b`'s curve moves **+0.182** on the same
meter under the same code. **ELIMINATED.**

**(c) Early stopping is what is flat, not the head.** The budget is in steps and is identical at
every N (§2), the patience window is 10,000 steps against best-steps of 75–825, and §6 runs the
largest N with no early stopping at all and 5× the steps. **ELIMINATED.**

**(d) The held-out set is too small to see a trend.** 2,400 battles / ~74,000 states, fixed across
N, and the same bootstrap resolves a **+0.182** move on `ctrl10M_b` and a **+0.099** move in the
conditional arm on `ctrl10M`. The flat rows' CIs are ±0.02 on a scale where the effect of interest
is ≥0.2. **ELIMINATED.**

**(e) The conditional target's ceiling moves with N, so the (b) column is not one comparison.** True
and reported: `cond_oracle`'s turns-1–3 ratio runs 0.695 → 0.998 as its factors get more battles
and need less shrinkage. That is why the (b) column is read as a fraction of ITS OWN ceiling at each
N, and why guard 0 is checked at N_max. **SURVIVES as a caveat on the (b) column only.**

**(f) The head refit's own caveats are inherited where they apply.** The noise-corrected, clamped
spread ratio is a biased non-monotone operator and a point estimate of 0.000 can sit below its own
bootstrap interval (its hazard 1) — every turn-1 row here has that shape and the unclamped
`ratio_raw` companion is computed in the same pass. HT reweighting is inert here (all weights 1.0),
so its hazard 5 cannot bite. Its hazard 2 (a conditional target with no cycle term) cannot bite
either: the "cycles" are eval seeds of one checkpoint.

**(g) Extraction hazards from the probe read.** Its hazard 5 — off-cycle rows are a probe of the
FINAL model on earlier states, not a replay — is **NOT inherited**: every state here was generated
by the very checkpoint that forwards it, and the QC is a hard equality (max \|V_fwd − V_rec\| =
2.2e-06 / 1.9e-06) rather than an off-cycle mean.

---

## 8. Scoring the pre-registration

| # | predicted | outcome |
|---|---|---|
| 1 | terminal spread ratio FLAT in N | **RIGHT on `ctrl10M`, WRONG on `ctrl10M_b`.** The resolution — a common converged level reached by 1.25 rollouts — was not predicted |
| 2 | turn-1 opponent-class AUC flat and near its null | **RIGHT**, for a reason the prediction did not know: the opponent is unobservable at turn 1 on this frame at all |
| 3 | DISSENT: own-team R² will RISE, detected, staying below the conditional refit and `value_pooled` | **RIGHT** (+0.018 [+0.006, +0.030] and +0.044 [+0.030, +0.060]; 0.131 / 0.104 against `value_pooled`'s 0.266 / 0.224) — but it rises by 0.004–0.010 per doubling, which is not a lever |
| 4 | `mlp_cond` DETECTED above `online` on the spread ratio at every N | **WRONG, and informatively so** — it is detected BELOW at almost every N (finding 3) |
| 5 | more optimisation moves nothing on the opponent axis | **WRONG in direction, right in conclusion**: it moves it DOWN (−0.078 / −0.048 on class AUC) |
| 6 | Brier improves with N for `mlp_term` while the opponent meters do not | **PARTLY** — Brier @all 0.0976 → 0.0972 (`ctrl10M`) is flat too; the dissociation is between the CONDITIONAL arm's ordering gain and its Brier loss instead |

The bar the prediction set for "a slower policy is a real lever" — **+0.03 per doubling on the
turns-1–3 spread ratio, sustained** — is met on `ctrl10M_b` at turns 11–24 (+0.041) and **not
sustained**: the 8k → 21.5k segment adds +0.006. On `ctrl10M` it is not met at all.

---

## 9. What this means for the treatment

- 🚫 **The owner's lever — slow the policy / replay a stationary window — is NOT worth building.**
  The curve saturates at 1.25 rollouts and the level it saturates at is one an online control
  already reaches. Bounded, measured, and small.
- 🚫 **Head-side optimisation stays RULED OUT**, now from both sides: no basin (head refit §7) and
  no pass deficit (§6 here, at 23× the data).
- ⚠️ **The cf-label arm's premise needs re-reading before it is judged.** `ai_v12_12_ladder_cflabels`
  was registered on the turn-1–3 spread ratio and the turn-1 own-team R². §3 says the turn-1
  opponent component of the first meter is unobservable on a matched-team frame, and §4(b) says a
  low-variance conditional target **lowers** the spread ratio while raising the opponent ORDERING.
  **Read that arm on the turns-4–10 opponent-class AUC of its prediction and on Brier/resolution,
  not on the turn-1–3 spread ratio alone** — otherwise a successful arm will read as a failure and
  a failed one as a success.
- ✅ **The highest-value unbuilt lever is unchanged and now better motivated: opponent-stratified
  weighting of the win-prob loss** (`value_terms._win_prob_loss`). It raises the between-cell share
  of the objective without introducing a target the features cannot support — which is exactly the
  failure mode §4(b) exposes in the conditional target.
- 🔬 **The measurement to run next is cheap and it is the one that decides (a):** does a TRAINING
  rollout assign the trainee's team independently of the opponent? If it does, the matched frame is
  the right model and the turn-1 mixture defect is not a defect. If it does not, the online head has
  a turn-1 tell the eval frame denies it, and every turn-1 meter on this campaign needs the own-team
  mediation reference run beside it.
- 📏 **A standing rule this earns:** *before reading a decode of X from a representation, decode X
  from the nuisance variables the observation carries anyway.* Here the nuisance was the trainee's
  own team, it explained 0.86 of a 0.86 AUC, and three measurements had already been built on top.

---

## 10. Hazards hit (a hazard is a finding)

1. 🚨 **A TURN-1 "OPPONENT DECODE" CAN BE AN OWN-TEAM DECODE, AND WAS.** §3. AUC 0.856 / 0.877 from
   the trainee's team alone on the probe read's frames, 0.508 on a matched-team frame. The probe
   read had the right machinery (an identity-mediation reference null) and did not point it at this
   axis. **Any decode whose label is constant within a nuisance group must be read against that
   group's own predictive power first.**
2. 🚨 **ON A MATCHED-TEAM FRAME A TURN-1 SPREAD RATIO OF 0 IS BAYES-OPTIMAL.** If the opponent is
   unobservable, the optimal critic's per-opponent means are all equal while the outcome's are not,
   so the identity is violated **by the Bayes critic**. The mixture identity is only a defect
   meter in windows where the conditioning variable is measurable w.r.t. the critic's information
   set — here, turns ≥ 4.
3. 🚨 **A SATURATED PREDICTION MAKES THE CALIBRATION SLOPE DIVERGE.** The no-early-stop arm printed
   −6.0e9 before the guard was added. Any calibration slope read off a head trained past its
   validation optimum needs the same refusal.
4. ⚠️ **AN EPOCH BUDGET SILENTLY SCALES THE OPTIMISATION WITH N.** An N-curve on an epoch budget
   measures data and passes together. Budget in steps.
5. ⚠️ **A RISE IN AN N-CURVE CAN BE A SMALL-N ARTEFACT.** `ctrl10M_b`'s +0.182 is real and its CI is
   clear of zero, and it is still not evidence that stationary data buys conditioning — the same
   fits on `ctrl10M` start where `ctrl10M_b` ends. **The end point, not the slope, is the quantity.**
6. ⚠️ **THE CONDITIONAL TARGET'S CEILING RISES WITH N** (0.695 → 0.998) because shrinkage falls, so
   the (b) column is six different comparisons and must be read as a fraction of its own ceiling.
7. ⚠️ **THE TWO SUBSTRATES' ONLINE HEADS DIFFER BY MORE THAN THE EFFECT BEING CHASED** (turns-11–24
   ratio 0.759 vs 0.464, two runs of the same configuration). Rule 19's family: this is a
   run-level component, and a single-substrate read of any amplitude row here would have been
   wrong in one direction or the other.

---

## 11. Recommendation — one paragraph

**Do not build the slower-policy / value-replay / stationary-window lever, and re-read the queued
cf-label arm on different meters before judging it.** The N-curve bounds what removing
non-stationarity is worth: the terminal-label head converges to a turns-11–24 between-opponent
spread ratio of 0.665–0.693 and an all-states ratio of 0.557–0.609 on both substrates, reaches that
level by ~4,000 battles (1.25 PPO rollouts at this run's geometry), and then stops — one substrate's
last segment is flat and the other's is detected NEGATIVE. Since an online head already receives
that much data inside a single policy iteration and carries its weights across iterations on top,
there is no data deficit for a slower policy to fill; the whole available gain is the distance
between the two control runs' own online heads (0.464 vs 0.759), which is a run-level variance term
(rule 19's family), not a lever. Spend the GPU instead on **opponent-stratified weighting of the
win-prob loss** — it raises the between-cell share of the objective without asking the head to
predict information its features do not carry, which is precisely how the conditional target fails
here (best opponent ORDERING of any condition at 0.830/0.854 AUC, worst AMPLITUDE at 0.439/0.465
spread ratio, and +0.03 of Brier). And spend one cheap CPU hour first on the question §3 opens:
whether a TRAINING rollout assigns the trainee's team independently of the opponent — because if it
does not, every turn-1 opponent meter on this campaign has the own-team mediation term sitting
inside it, exactly as the probe read's did.

---

## 12. Ledger paragraph (for the orchestrator to append — this file does NOT edit the ledger)

> **2026-09-10 · MEASUREMENT · THE N-CURVE — more stationary data does NOT buy the win head
> opponent conditioning past ~4,000 battles (1.25 rollouts), and the probe read's turn-1 opponent
> decode was an OWN-TEAM confound.** The owner asked whether the win head would improve if the
> policy moved slower, i.e. if non-stationarity were removed so the head could accumulate data on a
> fixed target; the head refit had argued its own null was scale-free, which was an argument, not a
> measurement. Measured: `ai_v12_11_ladder_ctrl10M` and `ai_v12_15_ladder_ctrl10M_b` @
> `step_10000032`, three offline `eval_trace_gen` trees each (seeds 20260909/20260910/20260911 ×
> 400+800+800 games × 12 opponents) whose `snapshot.zip` files are **md5-identical** (refused
> otherwise) — **732,626 states / 23,891 battles / 602 teams** and **716,727 / 23,877 / 602**, FULL
> capture so every HT weight is exactly 1.0, frozen-forward QC max |V_fwd−V_rec| 2.2e-06 / 1.9e-06.
> The win head alone was refit from scratch on frozen `value_pooled` against (a) the terminal 0/1
> label and (b) the per-cell × own-team LOO conditional label, on nested-by-battle subsets at N =
> 1k/2k/4k/8k/16k/~21.5k, **an identical GRADIENT-STEP budget at every N** (not epochs — an epoch
> budget scales the optimisation with N), scored OUT OF SAMPLE on ONE fixed held-out set of 2,400
> battles / ~74k states, every interval a battle-clustered bootstrap and every claim on the delta's
> own CI. **(1) THE TERMINAL-LABEL HEAD CONVERGES TO A SUBSTRATE-INDEPENDENT LEVEL AND GETS THERE
> BY N ≈ 4,000 BATTLES ≈ 1.25 PPO ROLLOUTS** (48 envs × 2,048 steps = 98,304 states ≈ 3,170
> episodes): turns-11–24 spread ratio **0.693** and **0.665** at N_max, all-states **0.609** and
> **0.557**. On `ctrl10M`, whose ONLINE head already reads 0.759 / 0.652, the curve is FLAT — Δ(21.5k
> − 1k) at turns 11–24 **−0.000 [−0.020, +0.021]**, all states **−0.001 [−0.017, +0.014]**,
> turns-4–10 opponent-class AUC **−0.004 [−0.011, +0.002]** — and its 8k → 21.5k segment is
> detected NEGATIVE (−0.029 [−0.045, −0.012]). On `ctrl10M_b`, whose online head reads 0.464 /
> 0.388, it RISES **+0.182 [+0.158, +0.209]** at turns 11–24 and SATURATES: 1k→8k +0.176 [+0.155,
> +0.202], 8k→21.5k +0.006 [−0.008, +0.017]. **Reading: (ii) FLAT on `ctrl10M`, (i) RISE-SATURATED
> on `ctrl10M_b`; (iii) refused on both. The owner's hypothesis is REFUTED as a route past the
> current ceiling** — the whole available gain is the gap between two identically-configured control
> runs' own heads (rule 19's run-level family), and an online head gets 1.25 rollouts of data inside
> one policy iteration. **(2) 🚨 THE PROBE READ'S TURN-1 OPPONENT DECODE IS AN OWN-TEAM CONFOUND.**
> `frame_check.py`: in the probe read's frames the TRAINEE'S OWN TEAM alone predicts the opponent's
> class at **AUC 0.856** (CTRL) / **0.877** (A), because only **37 of 180** and **47 of 216** teams
> ever face a sentinel there; the own team is in the observation verbatim at turn 1. On these
> matched-team frames all 602/602 teams face every opponent, own-team → class reads **0.508**, and
> `value_pooled` → class reads **0.502 / 0.508** against the probe read's 0.846/0.861. **The
> opponent is genuinely UNOBSERVABLE at turn 1** (no team preview) and becomes observable as it
> plays — pooled → class 0.50 (t1) → 0.67 (t1–3) → **0.82 (t4–10)** — so **a turn-1 spread ratio of
> 0 is BAYES-OPTIMAL on a matched-team frame, not a defect**, and the probe read's own
> PRE-REGISTERED expectation (ii), which it believed refuted, was right. In the window where the
> question has an answer the ONLINE head already conditions: class AUC **0.723 / 0.700** (null 0.52,
> `value_pooled` 0.812 / 0.848), turns-11–24 ratio 0.759 / 0.464. This does not touch the head
> refit's target-vs-target result; it retires the probe read's "the head is handed the answer"
> framing at turn 1 and makes the turn-1–3 spread ratio the WRONG registered meter for the cf-label
> arm. **(3) THE CONDITIONAL TARGET IS HARMFUL ON A MATCHED-TEAM FRAME ON EVERY METER BUT THE
> ORDERING**: at N_max it wins the turns-4–10 class AUC (**0.830 / 0.854**, above `value_pooled`'s
> own 0.812 / 0.848 and the online head's 0.723 / 0.700, delta CIs clear of zero) while taking the
> turns-11–24 ratio DOWN to 0.465 / 0.439 and Brier over all states from 0.1012 / 0.1045 to **0.1275
> / 0.1258** (the terminal refit reads 0.0972 / 0.0988), calibration slope 1.30 / 1.38
> (under-dispersed). A target built partly from information the features do not carry teaches the
> ORDER and destroys the AMPLITUDE. **(4) "NOT ENOUGH PASSES" IS REFUSED A SECOND TIME AT 23× THE
> SCALE**: 5× the gradient steps with NO early stopping lowers the turns-4–10 class AUC by **[−0.099,
> −0.057]** and **[−0.067, −0.030]** and raises Brier by **[+0.018, +0.030]** and **[+0.014,
> +0.023]** — with the head refit's basin elimination, treatment class 1 is closed from both sides.
> Counter-hypotheses: the eval-vs-training POPULATION gap SURVIVES as the principal limit (these are
> eval battles against a fixed 12-opponent panel, and whether a TRAINING rollout assigns the
> trainee's team independently of the opponent is NOT established — the single largest open
> question); "the meter is dead on this frame" ELIMINATED (`cond_oracle` reaches 0.998 [0.941,
> 1.045] and `ctrl10M_b` moves +0.182 under the same code); early stopping ELIMINATED (step budget
> identical at every N, patience 10,000 steps against best-steps of 75–825, plus §6). Hazards: a
> saturated prediction makes the calibration slope DIVERGE (−6.0e9 printed before a ±50 refusal was
> added); an EPOCH budget silently scales optimisation with N; **a rise in an N-curve can be a
> small-N artefact — the END POINT, not the slope, is the quantity**, since `ctrl10M`'s fits start
> where `ctrl10M_b`'s end. **Consequence:** do NOT build the slower-policy / value-replay /
> stationary-window lever; re-register `ai_v12_12_ladder_cflabels` on the turns-4–10 opponent-class
> AUC and Brier/resolution rather than the turn-1–3 spread ratio; opponent-stratified weighting of
> the win-prob loss remains the highest-value unbuilt lever and is now better motivated; and run the
> cheap CPU check of the TRAINING-rollout team assignment before any further turn-1 meter is read.
> Measurement: `designs/research_state/measurements/winprob_refit_ncurve_2026-09-10/`.

---

## 13. Proposed amendment to `UNDERSTANDING.md` §4.2b (text only — NOT applied here)

> Append to §4.2b, after the head-refit paragraph:
>
> **More stationary data does not buy the win head opponent conditioning, and the probe read's
> turn-1 opponent decode was an own-team confound.** [MEASURED,
> `winprob_refit_ncurve_2026-09-10`] The win head was refit on frozen `value_pooled` at N =
> 1k → ~21.5k battles drawn from ONE checkpoint (`step_10000032`, three md5-identical eval trees,
> ~24,000 battles / ~720,000 states per substrate, full capture so every HT weight is 1.0), on an
> identical gradient-STEP budget at every N, scored on one fixed held-out set of 2,400 battles.
> Under the TERMINAL 0/1 label the head converges to a substrate-independent level — turns-11–24
> between-opponent spread ratio **0.693 / 0.665**, all-states **0.609 / 0.557** — and reaches it by
> **N ≈ 4,000 battles ≈ 1.25 PPO rollouts** (48 × 2,048 = 98,304 states ≈ 3,170 episodes). On
> `ai_v12_11_ladder_ctrl10M`, whose ONLINE head already reads 0.759 / 0.652, the curve is FLAT
> (Δ at turns 11–24 −0.000 [−0.020, +0.021]) and its last segment is detected NEGATIVE; on
> `ai_v12_15_ladder_ctrl10M_b` (online 0.464 / 0.388) it rises +0.182 [+0.158, +0.209] and
> saturates by 8k. **Removing non-stationarity is therefore worth at most the gap between two
> identically-configured control runs' own heads, and an online head already receives that much
> data inside one policy iteration: a slower policy, a value replay or a stationary window is NOT a
> lever.** 🚨 **On a matched-team frame the opponent is UNOBSERVABLE at turn 1** — `value_pooled` →
> opponent class reads AUC **0.502 / 0.508** against the probe read's 0.846 / 0.861, because in the
> probe read's frames the TRAINEE'S OWN TEAM alone predicts the class at **0.856 / 0.877** (only
> 37 of 180 and 47 of 216 teams ever faced a sentinel there, against 602 of 602 here). The opponent
> becomes observable as it plays (0.50 → 0.67 → 0.82 by turns 4–10), **so a turn-1 spread ratio of 0
> is Bayes-optimal, not a defect**, and the probe read's headline "the head is handed the answer and
> does not use it" does not hold at turn 1. In the window where the question has an answer the
> online head DOES condition (class AUC 0.723 / 0.700, null 0.52). The CONDITIONAL target wins the
> opponent ORDERING (0.830 / 0.854, above `value_pooled`'s own 0.812 / 0.848) and loses the
> AMPLITUDE (ratio 0.465 / 0.439, Brier 0.1275 / 0.1258 against the terminal refit's 0.0972 /
> 0.0988) — so the cf-label arm must be read on turns-4–10 class AUC and Brier/resolution, not on
> the turn-1–3 spread ratio. **Open:** whether a TRAINING rollout assigns the trainee's team
> independently of the opponent; if not, every turn-1 meter on this campaign carries the same
> own-team mediation term.

---

## 14. Files

[`PREDICTION.md`](PREDICTION.md) (registered before any curve existed) · `extract.py` (three trees,
one md5-verified frozen forward, `value_pooled` + meta only) · `ncurve.py` (the held-out draw, the
nested subsets, the conditional target per subset, the step-budgeted fits, the 5× no-early-stop
control) · `readout.py` (the mixture identity with a per-cell `n_games`, the turn-bucketed decodes,
Brier, the calibration slope, every delta's CI) · `frame_check.py` (the own-team → opponent-class
leak and the observability ladder) · `summarize.py` (the reading rule in code, amended and
registered verdicts side by side) · `run.sh` (end to end, ~40 min per substrate) ·
`tables_<run>_<key>.md` · `reading_<run>_<key>.json` · `ncurve_stats_<run>.json` ·
`frame_check.json` · `extract_meta_<run>.json`. The 128-dim feature tensors and the prediction
columns are NOT committed; `run.sh` regenerates them.
