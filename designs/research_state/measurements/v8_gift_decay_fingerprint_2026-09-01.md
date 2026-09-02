# M9b — WHAT DID v8's LAST ~2.5M FOLD STEPS UNDO? The peak→final behavioural fingerprint

**Status: PRE-REGISTERED 2026-09-01 22:16:01 PDT (2026-09-02T05:16:01Z), before the first
measurement cell.** Results are appended below only after the battles run.

---

## 0. PRE-REGISTRATION (frozen — nothing in this block may be edited after the first cell)

### The question

`v8_gift_timing_2026-09-01.md` measured the untaught gain of v8's fold
(`ai_v8_14_distill3_0725`, forked from `ai_v8_04_distill_4teacher_0722` at 277,583,267) against
its fork parent at nine points along the fold. The curve is a **transient hump**: it peaks at
**+9.67pp [+6.79, +12.50]** at `checkpoints/checkpoint_290115536_steps.zip` (+12.53M fold steps)
and then decays **monotonically** — +8.25 at 291,106,373 · +7.03 at 292,100,648 · **+4.98pp
[+2.15, +7.62]** at `final_model_interrupted.zip` (+15.04M). The fold ran `--distill-coef 1.0`
against the same three teachers throughout.

**So: what did the last ~2.5M steps UNDO?** That is a question about *which decisions changed*,
not about how many win-rate points moved — and decisions are the one trace that ports across the
ai_v9 architecture rewrite (ledger `d392e80`).

### The registered hypotheses

Let `U(a→b)` be the 25-axis behavioural delta vector between arms `a` and `b` measured on
UNTAUGHT boards, and `T(a→b)` the same on TAUGHT boards. All vectors are measured on the
**identical** `(obs, action_mask)` rows.

> **H1 — OVERSHOOT / content-free regime.** The peak→final change on untaught boards is a
> *reversal* of the parent→peak change: the axes that moved on the way up move back on the way
> down. Operationally: `cos(U(parent→peak), U(peak→final)) < 0`.
>
> **H2 — CONTINUED TEACHER PULL.** The peak→final change on untaught boards looks like MORE of
> the TAUGHT-side fingerprint — the teachers' content leaking further into boards it was never
> taught on. Operationally: `cos(T(parent→peak), U(peak→final)) > 0`.
>
> **H3 — SOMETHING ELSE.** Neither cosine clears its noise ceiling, and a *distinct* axis set
> moves — e.g. the greedier play the gen-era non-gifting fold showed (`take_SE_attack` up,
> `switch|ahead_on_mons` down; M4 §6b).

> **TS — the TAUGHT-side question.** On taught boards, does the peak→final change continue in
> the teachers' direction (absorption still rising) or reverse? Operationally, the sign of
> `cos(T(parent→peak), T(peak→final))`.

### The registered bars — scored against INTERVALS, never points

A bar on a point estimate is a vacuous guard (ledger 2026-09-01). Every cosine below is scored
against a **cluster-bootstrap-over-TEAMS 95% percentile interval** — teams resampled, BOTH
vectors recomputed from the resampled teams, cosine recomputed — and **UNDECIDED is a real
outcome** that will be reported as one.

> **B1 (H1).** **PASS** iff the 95% interval on `cos(U(parent→peak), U(peak→final))` lies
> entirely **below zero**. **FAIL** iff it lies entirely **above** zero. **UNDECIDED** iff it
> straddles zero.

> **B2 (H2).** **PASS** iff the 95% interval on `cos(T(parent→peak), U(peak→final))` lies
> entirely **above zero**. **FAIL** iff entirely below. **UNDECIDED** iff it straddles.

> **B3 (H3).** **PASS** iff B1 and B2 are both non-PASS **AND** the three largest-|z| axes of
> `U(peak→final)` are **disjoint** from the three largest-|z| axes of `U(parent→peak)`.
> **FAIL** iff either B1 or B2 PASSes, or the top-3 sets intersect. B3 is deliberately the
> residual bar: it can only fire when the two named accounts do not.

> **B4 (TS).** **CONTINUES** iff the 95% interval on `cos(T(parent→peak), T(peak→final))` lies
> entirely above zero; **REVERSES** iff entirely below; **UNDECIDED** otherwise.

**Every cosine is reported beside its NOISE CEILING** — `sqrt(r_a · r_b)`, where each `r` is the
split-half-over-teams reliability of that vector (M4's control; two noisy vectors agree less than
one, and this programme has already turned a −0.020 into a +0.206 by installing it). A cosine is
never quoted without it, and a **permutation null** (shuffling one vector's axis labels, 4,000
draws, both tails) is reported beside every one.

### The instrument — REUSED, not rebuilt

| | |
|---|---|
| script | `v8_gift_decay_fingerprint_probe.py` (this directory) |
| ancestor | `v8_fold_behavioral_fingerprint_probe.py` (M4) — the cell constants and the 25-axis `classify_decision` are **IMPORTED from it**, not re-typed, so the basis cannot drift |
| axes | M4's 25 model-free behavioural axes read from `LegalActions` + `gen3_data` (never from the obs vector), in five groups: class shares · conditional switching · move quality · switch-target quality · the forced replacement |
| arms | `parent` = `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` (277,583,267) · `peak` = `ai_v8_14_distill3_0725/checkpoints/checkpoint_290115536_steps.zip` (290,115,536, **+12.53M**) · `final` = `ai_v8_14_distill3_0725/final_model_interrupted.zip` (292,623,779, **+15.04M**) |
| scoring | **all three arms act**, and at every decision **all three are scored on the identical `(obs, action_mask)`** — two extra forwards per decision, no extra battles |
| cells | M4's own: the 16 untaught probe teams + the 6 taught controls, × probe P's 8 fixed opponent teams, **4 games each, × 3 arms** |
| fixed reference opponent | `ai_v8_03_zarch_control_0718/final_model_interrupted.zip` — an ancestor of every arm and equal to none |
| CRN | `random.Random(f"{team_sha}:{opp_sha}")` → 4-int sim seeds, probe P's construction verbatim; game index *i* is the same battle for every arm, and these are the same CRN prefix subsample M4 used |
| regime | greedy both sides (`stochastic=False`), pinned single team per side, **node** bridge, no server, CPU, `nice -n 15`, ≤3 processes in flight, 1 battle in flight per process, `GEN3AI_TIMEOUT_SCALE=12` |
| era pin | `b13b30b289c5eaba136a930a4ab63451e209fbe5`, a PRIVATE copy of the era checkout, `PYTHONPATH=<era>/src` |
| totals | untaught **1,536** battles (16×8×4×3) · taught **576** (6×8×4×3) |
| statistics | cluster bootstrap over **TEAMS**, 4,000 resamples, equal-weight-over-teams point estimate — M4's and probe P's convention, so every number is on their scale |

**The five reproducibility seeds do not exist at this commit** (`$GEN3AI_{PLAYER,TEAM,POLICY,POOL,
STALLER}_SEED` landed 2026-08-30, after `b13b30b`; a grep over the era tree finds zero
references). Determinism comes from the same three things it came from for probe P, M4 and the
timing probe: no policy draw (`stochastic=False`), no team draw (one pinned team per side), no
dice draw (an explicit 4-int sim seed per battle). Recorded, not asserted.

### Order of work

**Untaught first** (H1/H2/H3 all live there), then taught (B2's reference vector and B4). If the
taught pass does not finish, the untaught tables and B1/B3 stand on their own and B2/B4 are
reported as NOT RUN.

### Reproduction checks fixed before data

1. `parent` and `final` are M4's own two arms, so `U(parent→final)` and `T(parent→final)` from
   this run must reproduce M4's published vectors. Reported as a cosine; a low value means the
   instrument drifted and invalidates everything downstream.
2. The ACID gate at load: all four networks (three arms + reference) distinct in parameter
   space, same parameter count. A mis-resolved path loading one zip twice reads as a perfect
   null.
3. Per-cell `finished == requested`; any shortfall is printed, never absorbed.

### What this probe CANNOT say

1. **One fold, one lineage, one decay.** Three points on `ai_v8_14`'s trajectory.
2. **Behaviour is not mechanism.** "The axes moved back" does not say *why* they moved back —
   the pool, the PFSP weights, the LR schedule and the accumulated distillation dose all move
   with step.
3. **Co-occurrence, not causation.** The behavioural deltas and the win-rate deltas are the same
   battles.
4. **Greedy, not stochastic**, and against ONE fixed reference opponent.
5. **`peak` is a SELECTED arm** — it was chosen as the maximum of the timing curve, so any
   contrast involving it is upward-biased in magnitude and is descriptive, not a registered test
   of the peak's height. The bars above are about the *shape* of a change, not its size.

---

## 1. THE HEADLINE

**The last ~2.5M fold steps UNDID essentially nothing. They ADDED — and what they added is the
TAUGHT-side fingerprint, applied to boards the fold never taught on, where it costs win rate.**

Three readings, all on identical boards:

1. **The ascent survives.** The behavioural change from the fork parent to the fold's ENDPOINT is
   **cosine 0.864 [+0.671, +0.953]** with the change from the parent to the PEAK, at **92%** of its
   magnitude (‖parent→final‖ / ‖parent→peak‖ = 0.92). The noise ceiling those two vectors' own
   split-half reliabilities permit is **0.751**, so the observed 0.864 is *at* the ceiling. The
   endpoint is not a policy that walked back; it is the peak's policy plus something.
2. **The decay vector is mostly a NEW direction, not a reversal.** Projecting `peak→final` onto
   `parent→peak` gives k = **−0.205**: only **16.4%** of the decay's energy lies along the ascent,
   **83.6%** is orthogonal to it.
3. **That new direction is the teachers'.** `cos(TAUGHT parent→peak, UNTAUGHT peak→final)` =
   **+0.559 [+0.104, +0.684]**, permutation p = **0.0015**, sign agreement 19/25 — against a noise
   ceiling of 0.486, i.e. **at or above** what these two vectors' reliabilities permit. Against
   M4's *published* taught vector (`taught parent→final`) it reads **+0.644 [+0.156, +0.781]**,
   p = 0.0003, ceiling 0.556.

| bar | verdict | number |
|---|---|---|
| **B1** — H1, peak→final is a REVERSAL of parent→peak (untaught) | **PASS as registered · but the registered rule is near-vacuous; the mechanism is REFUTED by the sharper null** | cos = **−0.405 [−0.619, −0.090]**, ceiling 0.777. The shared-endpoint arithmetic null expects **−0.672 [−0.774, −0.556]** and the observed value sits **above** it (p_right = 0.000) — *less* anti-aligned than chance |
| **B2** — H2, untaught peak→final looks like the TAUGHT fingerprint | **PASS** | cos = **+0.559 [+0.104, +0.684]**, perm p = 0.0015, ceiling **0.486** |
| **B3** — H3, neither named account fires and a distinct axis set moves | **FAIL** | B2 passed. (The top-3 axis sets *are* disjoint — the second clause holds — but the bar required both) |
| **B4** — TS, does the TAUGHT side keep going in the teachers' direction? | **UNDECIDED** on the registered cosine · **CONTINUES** on the arithmetic decomposition | cos(T(p→peak), T(peak→final)) = **+0.279 [−0.270, +0.569]**, ceiling 0.357. But `cos(T(p→peak), T(p→final))` = **+0.965** with ‖p→final‖/‖p→peak‖ = **1.125** — the taught-side change kept *growing*, 12.5% further along the same direction |

**And the cost is asymmetric, which is the finding that matters.** The same last 2.5M steps are
worth **−1.56pp [−8.33, +5.21], n.s.** on the six teams the fold taught, and **−5.66pp
[−12.11, −0.20]** on the sixteen it did not. The decay is *the same behavioural change on both
slices* (cos(T(peak→final), U(peak→final)) = **+0.539**, ceiling 0.572, perm p = 0.005) — it is
just **1.72× larger** on the untaught boards, and only there does it cost anything.

> **In one sentence: the fold did not forget its gift; it kept absorbing its teachers, and past
> ~+12.5M steps the marginal absorption was stall-shaped content that its taught teams could
> afford and the rest of the pool could not.**

---

## 2. THE WIN RATES — the timing curve reproduced on M4's cells

**Meter stamp — every number in this document carries it:** `regime` greedy both sides
(`stochastic=False`), node bridge, no server, CPU, `nice -n 15`, 3 processes in flight,
1 battle in flight per process, `GEN3AI_TIMEOUT_SCALE=12`, era pin
`b13b30b289c5eaba136a930a4ab63451e209fbe5` (a private copy) · `opponent`
`ai_v8_03_zarch_control_0718/final_model_interrupted.zip`, fixed for every arm · `cells` M4's own
16 untaught + 6 taught teams × probe P's 8 fixed opponents × **4 CRN games** × **3 arms** ·
**2,112 battles · 80,882 dual-scored decisions · 528 cells · ZERO errors and ZERO short cells.**

| slice | arm | fold Δ | pooled WR | gain vs the arm before it | 95% cluster CI | z | teams + |
|---|---|---:|---:|---:|---|---:|---:|
| untaught (16 teams, 1,536 battles, 54,791 decisions) | `parent` | 0.000M | 0.3770 | — | — | — | — |
| | `peak` | +12.532M | 0.4902 | **+11.33pp** | [+6.45, +16.60] | +4.43 | 14/16 |
| | `final` | +15.041M | 0.4336 | **−5.66pp** | [−12.11, −0.20] | −1.92 | 4/16 |
| | *(endpoint vs parent)* | | | **+5.66pp** | [+0.98, +10.16] | +2.39 | 11/16 |
| taught (6 teams, 576 battles, 26,091 decisions) | `parent` | 0.000M | 0.4479 | — | — | — | — |
| | `peak` | +12.532M | 0.6927 | **+24.48pp** | [+16.67, +31.77] | +6.24 | 6/6 |
| | `final` | +15.041M | 0.6771 | **−1.56pp** | [−8.33, +5.21] | −0.44 | 2/6 |
| | *(endpoint vs parent)* | | | **+22.92pp** | [+15.63, +32.29] | +5.40 | 6/6 |

The endpoint rows are M4's own numbers recovered exactly (M4: untaught **+5.66pp**, taught
**+22.92pp** — the same battles, so this is an identity check, not an independent measurement).
The `parent→peak` rows reproduce the timing probe's hump at a quarter of its games per cell
(timing probe, 16 games/cell: **+9.67pp [+6.79, +12.50]**; here, 4 games/cell: **+11.33pp
[+6.45, +16.60]** — each inside the other's interval).

### 2.1 The instrument checked itself, four ways

| check | expected | here | verdict |
|---|---|---|---|
| ACID at load: all four networks distinct, same parameter count | — | ‖parent−peak‖ 222.93 · ‖parent−final‖ **238.92** · ‖peak−final‖ 17.41 · ‖parent−ref‖ **53.33**; 3,512,397 params each | passes; the two starred figures are M4's own, to 2 dp |
| `parent→final` untaught axis vector vs M4's published one | ≈1 | **cosine 0.9851**, max per-axis |Δ| 0.011 | reproduces |
| `parent→final` taught axis vector vs M4's published one | ≈1 | **cosine 0.9803**, max per-axis |Δ| 0.015 | reproduces |
| cells completing their requested games | 528/528 | **528/528, 0 short** | clean |

A mis-resolved path that loaded one zip twice would read as a perfect null, so distinctness is a
GATE rather than a nicety. `peak` and `final` are 17.41 apart in parameter space against the
fold's 238.92 of total travel — 7.2% of the journey, and the arms are unambiguously different
networks.

---

## 3. WHAT THE LAST 2.5M STEPS ACTUALLY DID — the untaught `peak→final` axes

Δ is *final − peak on the same boards*, equal-weight over the 16 untaught teams, cluster
bootstrap over teams. The full 25-axis tables for all three arm pairs and both slices are in
`…_tables.md`, regenerated from the JSON.

| axis | n | rate at `peak` | Δ (final−peak) | 95% CI | z |
|---|---:|---:|---:|---|---:|
| `take_SE_attack\|SE_available` | 11994 | 0.620 | **−0.0235** | [−0.0328, −0.0144] | **−5.03** |
| `switch\|low_hp(<1/3)` | 6017 | 0.165 | **+0.0300** | [+0.0182, +0.0458] | **+4.18** |
| `take_best_damage\|>=2_attacks` | 25136 | 0.405 | **−0.0116** | [−0.0175, −0.0058] | −3.81 |
| `attack_rate` | 44552 | 0.494 | **−0.0161** | [−0.0251, −0.0076] | −3.64 |
| `switch\|ahead_on_mons` | 11890 | 0.212 | **+0.0108** | [+0.0050, +0.0171] | +3.48 |
| `attack_at_all\|>=2_attacks` | 25136 | 0.612 | **−0.0175** | [−0.0277, −0.0072] | −3.34 |
| `other_status_rate` | 44552 | 0.122 | **+0.0043** | [+0.0008, +0.0075] | +2.51 |
| `switch\|winning_matchup` | 14746 | 0.163 | **+0.0109** | [+0.0025, +0.0194] | +2.51 |
| `switch_rate` | 44552 | 0.212 | **+0.0075** | [+0.0012, +0.0131] | +2.45 |
| `switch\|losing_matchup` | 10453 | 0.319 | **+0.0103** | [+0.0021, +0.0189] | +2.40 |

**It stops clicking attacks — including the super-effective one and the maximum-damage one — and
pivots instead, most of all out of a nearly-dead mon and when it is ahead.** That is the taught
fingerprint verbatim. M4's published taught vector reads `attack_at_all` **−0.0644**,
`take_SE_attack` **−0.0634**, `attack_rate` **−0.0336**, `take_best_damage` **−0.0295**,
`switch_to_resist` **+0.0243**, `switch|winning_matchup` **+0.0305** — the same six axes, the same
six signs, at roughly a third of the amplitude.

**And it is NOT what the ascent did.** The `parent→peak` untaught vector is a *switch reduction*:
`switch|behind_on_mons` **−0.0308** (z −5.25), `switch|early` **−0.0491** (z −4.31),
`switch|high_hp` **−0.0300** (z −4.29), `switch|losing_matchup` **−0.0479** (z −3.98),
`switch_rate` **−0.0245** (z −3.93), with `attack_rate` **+0.0184** (z +3.65) — the fold got *more*
committed and attacked *more* on the way up, and then partly gave the switch reduction back while
also clicking worse. The two top-3 |z| sets are disjoint, which is B3's second clause.

**Divergence, for scale.** `parent→peak` argmax disagreement **0.306**, mean KL 0.239, mean |ΔV|
3.44. `peak→final` **0.154**, KL 0.056, |ΔV| 1.59. The decay is half the behavioural size of the
ascent and a quarter of its KL — consistent with the norm ratio 0.51 — yet it costs half the win
rate back. **Amount is not the story; direction is** (the same lesson M4 drew from the gen-era arm).

`SWITCH→SWITCH` remains the largest single flip bucket on every pair (21.2%, 21.2%, 21.7%), and
the switch-target agreement rises from 74.9% (`parent→peak`) to **88.3%** (`peak→final`) — the
decay changes *whether* to pivot far more than *to whom*.

---

## 4. THE TAUGHT SIDE — absorption is still rising, and it is free there

| axis | n | rate at `peak` | Δ (final−peak) | 95% CI | z |
|---|---:|---:|---:|---|---:|
| `attack_rate` | 22583 | 0.293 | **−0.0061** | [−0.0086, −0.0033] | −4.47 |
| `forced_repl_resists\|resist_avail` | 940 | 0.281 | **−0.0225** | [−0.0428, −0.0086] | −2.46 |
| `switch\|we_are_boosted` | 4084 | 0.063 | **+0.0076** | [+0.0024, +0.0158] | +2.01 |
| `attack_at_all\|>=2_attacks` | 6614 | 0.583 | −0.0101 | [−0.0219, +0.0012] | −1.67 |
| `take_SE_attack\|SE_available` | 4045 | 0.626 | −0.0049 | [−0.0119, +0.0022] | −1.34 |

Same direction, a third the amplitude of the untaught decay (‖T(peak→final)‖ = 0.0305 vs
‖U(peak→final)‖ = 0.0527), and **no measurable win-rate cost** (−1.56pp, CI straddling zero).

**B4 is UNDECIDED as registered and CONTINUES on the decomposition, and the two are not in
conflict — they are limited by different things.** The registered cosine
`cos(T(parent→peak), T(peak→final))` = +0.279 with a CI of [−0.270, +0.569] against a **noise
ceiling of only 0.357**: the taught vectors are built from six teams and their split-half
reliabilities are 0.304 / 0.420, so this contrast simply has too little resolution to clear zero.
The decomposition asks the same question with far more of the signal intact and answers cleanly:
`cos(T(parent→peak), T(parent→final))` = **+0.965**, and the endpoint vector is **12.5% LONGER**
than the peak vector. On its taught teams the fold at +15.04M is the fold at +12.53M, only more so.

**The registered verdict stands as UNDECIDED.** The decomposition is reported beside it, never in
place of it.

---

## 5. SCORING THE BARS — and the one that turned out near-vacuous

### B1 — "peak→final is a reversal of parent→peak": **PASS as registered. The mechanism is REFUTED.**

The registered rule fires: cos = −0.405, CI [−0.619, −0.090], wholly below zero, permutation
p_left = 0.030 against a label-shuffle null centred at −0.077.

**But the registered rule is close to vacuous, and this is recorded rather than quietly fixed.**
`peak→final` is not an independent vector: by construction `D = (parent→final) − (parent→peak) =
B − A`, so `cos(A, D)` carries an ARITHMETIC component. If `B` were unrelated to `A` at a similar
norm, the expected cosine is `≈ −‖A‖/‖B−A‖ ≈ −0.7`. A negative cosine is therefore the DEFAULT,
and B1 would have fired just as readily under H3.

The honest null keeps the shared endpoint and destroys only the axis-level correspondence
(permute `B`'s labels, recompute `cos(A, perm(B) − A)`): it reads **−0.672, 95% [−0.774, −0.556]**.
The observed **−0.405 sits ABOVE that band** (p_right = 0.000). **The decay is LESS anti-aligned
with the ascent than chance requires**, which is the opposite of a reversal.

The non-vacuous restatement of H1 — *pure unlearning means the endpoint lies back along the ascent
direction, shorter* — is answered by `cos(A,B)` = **+0.864** with `‖B‖/‖A‖` = **0.92**: the
endpoint lies along the ascent and is barely shorter. **Nothing was walked back.**

*This was written into the analyzer before the taught pass ran and after the untaught pass, so it
is a post-hoc control on a pre-registered bar. It is reported as one: the registered verdict is
PASS, and the null it does not contain is printed beside it.*

### B2 — "the untaught decay looks like the taught fingerprint": **PASS.**

cos = **+0.559**, cluster CI **[+0.104, +0.684]** wholly above zero, permutation p = **0.0015**,
sign agreement 19/25. Ceiling 0.486 (limited by the taught ascent's 0.304 reliability at six
teams), so `cosine / ceiling` = **1.149** — the observed agreement is as high as this instrument
can register. The same contrast against M4's published taught vector reads **+0.644 [+0.156,
+0.781]**, p = 0.0003, ratio-to-ceiling 1.159.

⚠️ **A ratio above 1.0 is not "better than perfect".** The disattenuation divides by a noisy
reliability estimate; all it licenses is *"indistinguishable from a fully shared direction, given
how noisy these vectors are"*. The cross-slice bootstrap resamples the two team sets
independently, which is the honest choice and the reason the interval is wide.

### B3 — "something else": **FAIL**, because B2 passed.

The bar had two clauses and only the second holds: the three largest-|z| axes of `U(parent→peak)`
(`switch|behind_on_mons`, `switch|early`, `switch|high_hp`) are **disjoint** from those of
`U(peak→final)` (`switch|low_hp`, `take_SE_attack`, `take_best_damage`). The decay genuinely moves
a different axis set from the ascent — it is just not an *unnamed* one.

**H3's specific prediction is half right, in the informative way, and worth recording.** It guessed "greedier play, as the
gen-era non-gifting fold showed — `take_SE_attack` up, `switch|ahead_on_mons` down". The decay
moves exactly those two axes and moves both **in the opposite direction**: `take_SE_attack`
**−2.35pp**, `switch|ahead_on_mons` **+1.08pp**. v8's last 2.5M steps did not get greedy; they got
*less* greedy, further past the point where less-greedy paid.

### B4 — the taught side: **UNDECIDED** (see §4).

---

## 6. WHAT THIS CHANGES

**Does:**

1. **"What did the last 2.5M steps undo?" has an answer, and it is *nothing*.** The gift was not
   unlearned. The endpoint holds 92% of the ascent's behavioural magnitude at cosine 0.864. Any
   account of the decay built on forgetting, plasticity loss, or drift-away-from-the-gift is
   attacking a phenomenon that did not occur.
2. **The decay is CONTINUED DISTILLATION, measured.** The fold ran `--distill-coef 1.0` against the
   same three teachers throughout, and the marginal content it absorbed in its last sixth is the
   teachers' own fingerprint arriving on boards the teachers never covered. It is a dose effect on
   an untaught population, not an optimisation pathology.
3. **It sharpens M4's `(i)/(ii)` reading rather than repeating it.** M4 found the *endpoint's*
   untaught vector near-orthogonal to the taught one (cosine 0.14, reproduced here at
   **0.186 [−0.236, +0.472]**). This probe shows why that is compatible with the taught content
   leaking: the untaught endpoint is a **SUM** of two roughly orthogonal things — a large
   commit-more/pivot-less ascent that is the gift, and a smaller stall-shaped teacher residue that
   is the tax. Averaged, they read as "not the taught vector". Decomposed in time, the second one
   is exactly the taught vector.
4. **The operational lever the timing probe named is now mechanistically motivated, not just
   empirical.** "Stop the fold at the untaught peak" is, on this evidence, "stop before the
   marginal distillation dose exceeds what the untaught population can absorb". That predicts the
   stopping point should track the *distillation schedule*, not the step count — a testable
   difference, and one no probe here ran.

**Does not:**

1. **n = 1 fold, n = 1 decay.** Three points on `ai_v8_14`'s trajectory. Whether any other fold
   decays this way, or at all, is untouched.
2. **Behaviour is not mechanism.** "The marginal change is teacher-shaped" is consistent with
   continued distillation and with several other things that also make a policy more stall-like
   (the pool's composition drifting, the LR schedule, PFSP re-weighting). Nothing here isolates the
   distillation term; an arm with `--distill-coef` annealed to 0 over the last 2.5M would.
3. **Co-occurrence, not causation.** The behavioural deltas and the win-rate deltas are the same
   battles.
4. **`peak` is a SELECTED arm** — chosen as the argmax of the timing curve — so every magnitude
   involving it is upward-biased. The bars are about the *shape* of a change, and the shape
   statistics (cosines, the projection) are not the quantity that selection inflates; the +11.33pp
   ascent is.
5. **The taught slice is six teams**, with vector reliabilities of 0.30 and 0.42. It is why B4 is
   UNDECIDED, and it caps B2's ceiling at 0.486. A wider taught set is the cheapest strengthening
   available.
6. **Greedy play, one fixed reference opponent** (`ai_v8_03`), and battles that are a CRN prefix
   subsample of probe P's — the same subsample M4 used, so this is not an independent redraw.
7. **The axes are a fixed 25-coordinate basis** inherited from M4. A change living outside that
   basis is invisible here; the `SWITCH→SWITCH` target choice, which M4 flagged as a standing
   instrument gap, is still one (though its agreement *rose* to 88.3% across the decay, so the
   decay is not hiding there).

---

## 7. ARTIFACTS

| file | holds |
|---|---|
| `v8_gift_decay_fingerprint_2026-09-01.json` | every number above — three arm pairs × two slices × 25 axes, the four bars, the arithmetic nulls, the informational cosines, the per-team win rates |
| `v8_gift_decay_fingerprint_2026-09-01_tables.md` | the full axis tables, **regenerated from the JSON** so the markdown cannot drift from the numbers |
| `v8_gift_decay_fingerprint_2026-09-01_rows_untaught.jsonl.gz` · `…_rows_taught.jsonl.gz` | **all 80,882 dual-scored decision rows** — per-action class vector, board strata, and all three arms' argmax and value on the identical board. New axis definitions are testable offline; only the three 11-float probability vectors were dropped (they are ~⅔ of the bytes and no axis reads them) |
| `v8_gift_decay_fingerprint_2026-09-01_cells.jsonl.gz` | all 528 cells — per-cell win counts, per-game outcome vectors, battle tags, and the deferral counters |
| `v8_gift_decay_fingerprint_probe.py` | the three-arm dual-scoring battle pass (imports M4's selection and classifier; resume-safe) |
| `v8_gift_decay_fingerprint_analyze.py` | the analysis — pure arithmetic over recorded rows, no model and no battle |
| `v8_gift_decay_fingerprint_bank.py` | the row-banking step |
| `v8_gift_decay_fingerprint_inputs/selection.json` | the team selection, with byte-equality against M4's copy and list-equality against the timing probe's **asserted by the script that wrote it** |
| `v8_gift_decay_fingerprint_inputs/arms.json` | the three arms + the reference: path, step, fold delta, size and **sha256** |
| `v8_gift_decay_fingerprint_inputs/seeds.json` | the environment, the regime, the CRN construction, and the note on the five inert seed vars |
| `v8_gift_decay_fingerprint_inputs/environment.json` · `probe_meta_*.json` | interpreter/numpy/worktree HEAD; the probe's own ACID block and wall time |
