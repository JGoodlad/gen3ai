# Differentiation vs breadth — does teams-per-exploiter drive behavioural specialization? (2026-08-28)

**Question.** The plasticity forensics found that v8-era exploiter teachers *differentiate* — their
argmax agreement with the parent drops on their own pinned teams relative to off-slice — while the
current-era R2 fleet (2 pinned teams each) does not. The **breadth hypothesis** says the difference
is teams-per-teacher: many teams cannot be satisfied by one global policy shift, two can.

**Answer: breadth is NOT the cause. The registered FLAT reading is selected.**

Across a **length-controlled** ladder spanning **2 → 3 → 4 → 9** pinned teams, differentiation does
not move. The weighted slope is **+0.0003 ± 0.0013 per extra pinned team (z = +0.23)** on the metric
the forensics used, and **+0.0003 ± 0.0084 (z = +0.03)** on a sharper distribution-controlled metric
built for this probe. Integrated over the ladder's whole 2 → 9 span, breadth buys **−0.018** — the
wrong sign, and inside noise.

**The axis that does move it is fork LENGTH.** At fixed breadth 9, tripling the fork (3 M → 9 M)
shifts the differentiation delta by **+0.039 ± 0.016 (z = +2.43)** — more than breadth's entire
2 → 9 range, in the opposite direction to breadth's point estimate. That matters for reading v8,
because **the cited v8 points are themselves long forks** (see the correction below).

**Two corrections to the brief's premises, both material.**

1. **v8's teachers pinned 3, 10 and 10 teams — not 23 each.** 23 is the *total across the three*,
   which is how the ledger phrases it ("23-teams-across-3"). Verified from each run's
   `metadata.json` `--trainee-teams`. Two of the three v8 points therefore sit *inside* our ladder's
   breadth range rather than an order beyond it, which removes the only reason to think the ladder
   was too short to see the effect.
2. **The cited v8 on/off numbers are measured on the FINAL checkpoints — 7.4–18.7 M steps past the
   fork — not at the forensics' matched 3.13–3.15 M length.** (Confirmed in the forensics JSON: its
   `onslice_kl` off-slice values are exactly its "as-distilled finals" row, not its `_m`
   matched-length row.) So the v8-vs-gen contrast is **breadth-comparable but not length-matched**,
   and our ladder tops out at a fork 2.5–6× shorter than v8's. This is now the leading candidate
   explanation, and this probe cannot close it.

**One nuance the forensics' metric missed, reported honestly.** Once the trajectory-distribution
confound is removed (below), *every* teacher at *every* breadth shows a small, consistently-signed
differentiation of **+0.028 pooled** — the R2 forks are weakly team-selective after all, not
literally undifferentiated. It is breadth-flat, and it is roughly a third of v8's (confounded,
longer-fork) effect. The headline is unchanged; the forensics' "never became behaviourally distinct
at all" phrasing is slightly too strong.

---

## The ladder

| teacher | run | **K** teams | fork | primary Δ (z) | controlled Δ (z) | per-team sd_excess (Q *p*) |
|---|---|---:|---:|---:|---:|---:|
| F5a | `ai_v9_53_R2F5a_0826` | 2 | 3.0 M | −0.034 (−3.11) | **+0.013** (0.67) | 0.0071 (8.8e−02) |
| F5b | `ai_v9_54_R2F5b_0826` | 2 | 3.0 M | +0.034 (2.96) | **+0.035** (1.59) | 0.0075 (8.6e−02) |
| F5c | `ai_v9_55_R2F5c_0826` | 2 | 3.0 M | −0.020 (−1.83) | **+0.019** (1.18) | 0.0000 (3.9e−01) |
| F5d | `ai_v9_56_R2F5d_0826` | 2 | 3.0 M | −0.004 (−0.33) | **+0.037** (2.00) | 0.0231 (6.0e−04) |
| F5e | `ai_v9_57_R2F5e_0826` | 2 | 3.0 M | −0.024 (−2.23) | **+0.034** (1.77) | 0.0333 (3.2e−07) |
| **tock-1c** | `ai_v9_36_tock1c_q6_0824` | **2** | 3.0 M | −0.008 (−0.69) | **+0.039** (2.36) | 0.0000 (2.1e−01) |
| **tock-1b** | `ai_v9_32_tock1b_rain_0824` | **3** | 3.0 M | −0.025 (−2.22) | **+0.013** (0.91) | 0.0087 (9.5e−02) |
| **tock-1a** | `ai_v9_31_tock1_k4_0824` | **4** | 3.0 M | −0.022 (−2.00) | **+0.036** (2.25) | 0.0346 (1.0e−05) |
| **tock-2.0 @3M** | `ai_v9_44_…/checkpoint_28067760` | **9** | 3.0 M | −0.027 (−2.42) | *NOT DEFINED* | 0.0080 (9.5e−02) |
| tock-2.0 @9M | `ai_v9_44_tock2_v8shape_0825` | 9 | 9.0 M | +0.012 (1.05) | *NOT DEFINED* | 0.0113 (7.4e−02) |
| — | — | — | — | — | — | — |
| *v8 semistall3* † | `ai_v8_06_…` | *3* | ***~7.4–18.7 M*** | *+0.058 (3.68)* | *not measured* | *not measured* |
| *v8 pool10* † | `ai_v8_09_…` | *10* | ***~7.4–18.7 M*** | *+0.047 (2.96)* | *not measured* | *not measured* |
| *v8 defensive10* † | `ai_v8_13_…` | *10* | ***~7.4–18.7 M*** | *+0.134 (8.59)* | *not measured* | *not measured* |

† **CITED** from `plasticity_forensics_v8_vs_gen_2026-08-28.json`, not re-measured (v8 needs era
pinning, out of scope). z-scores computed here from that file's proportions and n's. **These are
final-checkpoint reads** — the forensics' matched-3.1 M-length v8 checkpoints were used for its
other phases, not for this on/off table.

Δ is **off-slice agreement − on-slice agreement**: positive = the teacher changed its chosen action
*more* on its own pinned teams = differentiated.

### Cell means and the slope

| breadth K | primary Δ (n) | controlled Δ (n) |
|---:|---|---|
| 2 | −0.009 (6) | **+0.029** (6) |
| 3 | −0.025 (1) | **+0.013** (1) |
| 4 | −0.022 (1) | **+0.036** (1) |
| 9 | −0.007 (2) | *undefined* |

| slope of Δ on K | n | slope ± SE | z | 95 % CI |
|---|---:|---|---:|---|
| primary, all | 10 | **+0.00030 ± 0.00131** | +0.23 | [−0.0023, +0.0029] |
| primary, length-controlled (3 M only) | 9 | −0.00255 ± 0.00169 | −1.51 | [−0.0059, +0.0008] |
| controlled, all (= length-controlled) | 8 | **+0.00025 ± 0.00836** | +0.03 | [−0.0161, +0.0166] |

Inverse-variance weighted least squares. **The primary slope is the tight one**: its CI rules out
more than ±0.003 of Δ per extra team, so even 2 → 23 teams could buy at most ~0.06, with the sign as
likely negative. The length-controlled primary slope is *negative* at z = −1.51 — if anything,
broader teachers here differentiate slightly **less**.

---

## Why there are two metrics

**Primary — own-trace ON vs parent-trace OFF.** The forensics' construction, reproduced so the v8
row stays comparable: on-slice states come from the teacher's own `eval_traces`, off-slice from the
parent's. **Its confound is inherited and load-bearing** — the two halves differ in *state
distribution* as well as in team, because a fork's traces are generated by the fork's own policy.
This probe found the confound is not benign: it flips the sign. The reproduced R2 fleet numbers
agree with the forensics to ~0.01 (e.g. F5a on/off 0.768/0.734 here vs 0.768/0.741 there), so the
replication is sound; the metric is what is limited.

**Controlled — one shared bank, ON vs CROSS.** Built for this probe. tock-2.0 pins **all nine**
teams in the universe, so its traces supply states on every team from **one common policy's
trajectory distribution**. For each other teacher, ON = bank states on its pinned teams, CROSS =
bank states on the other teams. Teacher checkpoint, state source and parent are identical on both
sides; **team identity is the only thing that varies.** Bank: 3404 decisions from
`ai_v9_44/eval_traces/step_28000032`, per-team n = 107–830.

That is the read to trust, and it is the one that is flat.

### The state sets

| set | source | n |
|---|---|---:|
| off-slice (shared) | `ai_v9_29_rev1_0823/eval_traces/step_24000000`, teams **outside** the 9-team universe | 3000 (pool 7595) |
| bank (shared) | `ai_v9_44/eval_traces/step_28000032`, all 9 teams | 3404 |
| on-slice, per teacher | that teacher's own matched trace step, full on-pin pool | 3063–4170 |

**Every exploiter's traces are 100 % on-pin** — the trainee only ever pilots its pinned teams in
eval — so on-slice pools need no filtering, and there are no fork-trajectory states anywhere outside
the nine teams.

---

## Team-conditional structure — the sharper read, and it also says no

If breadth forced team-conditional behaviour, a broad teacher's agreement should **vary by team**
more than a narrow one's. Cochran's Q on per-team agreement over its own pinned teams, with the
binomial noise floor subtracted (`sd_excess`):

- **Heterogeneity is real for some teachers and absent for others, with no relation to breadth.**
  Significant: `tock1a` (K = 4, 0.0346, *p* = 1e−05), `F5e` (K = **2**, 0.0333, *p* = 3e−07),
  `F5d` (K = **2**, 0.0231, *p* = 6e−04). Zero excess: `F5c` (K = 2), `tock1c` (K = 2).
  A 2-team fork produces as much team-to-team spread as a 4-team one.
- **The broadest teacher is the most UNIFORM.** tock-2.0 at K = 9 has the *lowest* excess
  heterogeneity on the ladder (0.0080, *p* = 0.09 at 3 M; 0.0113, *p* = 0.07 at 9 M) — and **six of
  its nine** per-team agreements sit within ±0.02 of 0.745. Given the most teams to differentiate
  among, it differentiated among them least. That is the direct opposite of the mechanism's
  prediction.

The bank profile makes the same point: each teacher's pinned columns sit inside the spread of its
unpinned ones, and the largest per-team swings track *which team it is* rather than who pinned it —
across the nine 3 M teachers, bank team 0 reads 0.729–0.869 and team 7 reads 0.642–0.776, whoever
has it pinned. **Per-team variance is mostly a property of the team, not of the specialization.**

---

## The length-controlled sub-ladder — the cleanest cell

`tock-1c` / `tock-1b` / `tock-1a` are K = **2 / 3 / 4**, all **+3.0 M** fork steps, one fork each,
same parent, same frozen target. **`tock-2.0` joins them at K = 9**: it has a checkpoint at
**exactly 28,067,760 steps** — the same total as the 3 M tocks' finals — *and* an
`eval_traces/step_28000032` matching theirs. The length-controlled ladder is therefore a full
**2/3/4/9**, not the trio the brief anticipated.

| K | teacher | primary Δ | controlled Δ | per-team sd_excess |
|---:|---|---:|---:|---:|
| 2 | tock-1c | −0.008 | +0.039 | 0.0000 |
| 3 | tock-1b | −0.025 | +0.013 | 0.0087 |
| 4 | tock-1a | −0.022 | +0.036 | 0.0346 |
| 9 | tock-2.0 @3M | −0.027 | *undefined* | 0.0080 |

**Verdict: non-monotone and inside noise.** The controlled Δ across 2/3/4 goes 0.039 → 0.013 →
0.036, a spread entirely contained within the *five* K = 2 R2 forks' own range (0.013–0.039). The
primary Δ is flat-to-negative and, if anything, most negative at the widest breadth. There is no
breadth signal in the cleanest cell in the study.

---

## The length axis — the one thing that does move

The brief flagged that breadth and fork length co-vary on this ladder. **They need not**: reading
tock-2.0 at its 28,067,760-step checkpoint discharges the confound for the breadth question
entirely, and every slope marked "length-controlled" uses 3.0 M forks only.

That leaves length free to be measured on its own, at fixed K = 9:

| tock-2.0 | agreement (on) | KL(parent‖teacher), on | primary Δ | per-team sd_excess |
|---|---:|---:|---:|---:|
| **@3 M** | 0.746 | 0.304 | **−0.027** | 0.0080 (*p* = 0.09) |
| **@9 M** | 0.614 | 0.621 | **+0.012** | 0.0113 (*p* = 0.07) |
| change | −0.132 | ×2.04 | **+0.039 ± 0.016, z = +2.43** | n.s. |

Tripling the fork roughly doubles the distance from the parent **and flips the differentiation delta
positive** — a shift more than twice the size of breadth's entire 2 → 9 effect (−0.018) and in the
opposite direction. tock-2.0 @9 M is also the only gen-era teacher whose primary Δ is positive, and
the only one whose fork length approaches v8's 7.4–18.7 M range.

**How much weight this carries: directional, not established.** n = 1 pair; it is measured on the
*primary* (distribution-confounded) metric, and the two readings use different on-slice trace steps
(`step_28000032` vs `step_34000032`), so part of the shift may be distributional. Per-team
heterogeneity does **not** rise significantly with length either (0.0080 → 0.0113, both n.s.), so
length moves the aggregate delta without demonstrably creating team-conditional structure. It is a
lead, sized and named, not a result.

---

## Which axis is left

The elimination is informative because the ladder is unusually well controlled. Every run on it:

- forks the **same parent** — `ai_v9_29_rev1_0823/final_model.zip`, 25,067,760 steps;
- against the **same frozen target** — `…/snapshots/snapshot_000024000000.zip`;
- draws from the **same 9-team universe** — the R2 fleet's five pairs and tock-1a/1b/1c's 4 + 3 + 2
  both partition exactly the same nine team files, and tock-2.0 pins all nine;
- shares **121 identical launch flags**. The only differences in the entire argv are
  `--trainee-teams` (the manipulation), tock-2.0's `--steps`, and `--compile-opponents-strict` (an
  unversioned perf knob, absent in tock-1a/1b).

With breadth flat at fixed length, the surviving candidates are **fork length** and **parent
maturity / era-recipe**. Length now has a within-archive measurement pointing the right way
(z = 2.43) *and* co-varies with the v8 comparison (v8's teachers ran 2.5–6× longer forks than any
gen-era teacher), which makes it the cheapest next test: **a 9 M fork at K = 2**, which would cross
the two axes and is the one cell the archive does not contain.

Parent maturity is the other candidate and is in worse shape than it looks: v8's parent had 277 M
steps to rev-1's 25 M, but the forensics already **refuted** the natural story there — the 277 M v8
parent showed *no* Lyle capacity loss (1.154) while the 25 M rev-1 showed mild loss (0.948). So
"converged parents differentiate better" is not supported by any measurement in this archive either.

**Implication for the budget law: it does not need a breadth rider.** Spending exploiter budget on
more teams per teacher has no measured effect on how team-selective the resulting teacher is —
consistent with this project's standing finding that **count dominates conditioning**. If budget is
to be spent on differentiation at all, the evidence points at fork *length*, not team *count*.

---

## Method

Read-only over `models/`, CPU-only, **current architecture, no era pinning** — every ladder run
records `arch_signature = gen3_critic_route_wave_v1` and obs 2501, identical to HEAD, so all eleven
checkpoints load under current code.

Per teacher, versus the shared parent, on each state set: **top-1 agreement** of the argmax over
legal actions, **KL(parent ‖ teacher)** in nats over the legality-masked softmax, and the value
correlation. Δ's SE is the two-proportion SE; per-team heterogeneity is Cochran's Q against the
binomial floor, with `sd_excess = √(sd²_obs − sd²_expected)`.

### Acid tests — the input reconstruction is exact at both ends

The traces record only `obs`; the Dict observation has ~15 further auxiliary channels, filled with
"unknown" defaults. Validated by forwarding **the snapshot the trace itself shipped** and comparing
to the recorded logits (the forensics standard, repeated once for this pipeline and then extended to
the teacher end, which additionally proves checkpoint↔trace alignment):

| | max abs Δ logits | corr | top-1 agreement |
|---|---|---|---|
| parent, `ai_v9_29_rev1/step_24000000` | 1.91e−05 | 0.9999999999996 | **1.000** |
| teacher, `tock1a/step_28000032` | 2.57e−05 | — | **1.000** |
| teacher, `F5d/step_28000032` | 2.77e−05 | — | **1.000** |
| teacher, `tock2/step_34000032` | 3.72e−05 | — | **1.000** |

Float32 noise. The parent row reproduces the forensics' gen-era row exactly.

**Step-gap sensitivity (a caveat that does *not* bias the result).** A teacher's `final_model` sits
~67 k steps past the trace it is scored on, and agrees with that trace's own snapshot at only
**0.90–0.93** top-1 — larger than the differentiation deltas themselves. It **cancels in the
controlled contrast** by construction: ON and CROSS use the *same* teacher checkpoint on the *same*
bank, so a checkpoint offset shifts the level, not the ON-vs-CROSS difference. For scale, those same
snapshots agree with the *parent* at only 0.48–0.76, so fork distance dominates the step gap.

---

## MISSING cells and caveats

| Cell | Status |
|---|---|
| **controlled read at K = 9** | **Structurally undefined.** tock-2.0 pins the entire 9-team universe, so it has no within-bank cross-slice, and no fork-trajectory states exist outside the nine (every exploiter trace is 100 % on-pin). The top rung has the primary metric only. |
| **mitigation sets** (parent-trace states on each teacher's pins) | **Computed where possible, UNINFORMATIVE.** Only **0.90 %** of the parent's 7664 pooled decisions land on the nine-team universe; per-teacher coverage is 0 / 0 / 43 / 0 / 26 / 26 / 43 / 0 / 69 / 69, and **4 of 10 teachers have ZERO parent battles on their pins**. Every computed cell has n ≤ 69 and is flagged `UNDERPOWERED`; the six that exist scatter in both directions (tock-1c 0.654 vs 0.720 off-slice; tock-1b 0.884 vs 0.736). Reported for coverage honesty, used in no verdict. |
| **v8 controlled read** | Not computed — v8 requires era pinning (out of scope), and the bank construction needs a teacher pinning the whole universe, which the v8 fleet does not provide. v8 is cited on the **primary** metric only, so the v8-vs-gen contrast is like-for-like on that metric alone. |
| **a long fork at narrow breadth** | **Absent from the archive** — the cell that would cross breadth against length. Named above as the cheapest next test. |
| **v8 length-matched on/off** | The forensics computed matched-3.1 M-length v8 checkpoints for its other phases but its on/off table uses the finals. A matched-length v8 on/off read would settle how much of v8's differentiation is length; it is not in the JSON and re-deriving it needs era pinning. |
| v8 breadth | Corrected: **3 / 10 / 10**, not 23 each. |
| K = 3 and K = 4 have **n = 1** | Single forks; the K = 2 cell has six. The slope weights accordingly, but the ladder's power sits at K = 2. |
| on/off state-distribution confound (primary metric) | Inherited from the forensics and **not** fixed in the primary metric — which is exactly why the controlled metric was built. Drawn identically at every rung, so the *breadth* contrast is still read cleanly; the *sign* of a primary Δ is not trustworthy on its own. |
| tock-2.0 @9 M trace step | Its on-slice states come from `step_34000032` while every 3 M teacher uses `step_28000032`. Labelled, and never mixed into a length-controlled statistic. |

## Reproduction

Scripts under `tmp/` in this worktree (untracked, `tmp/` is gitignored): `pa_states.py` (state sets
+ the shared bank), `pa_forward.py` (CPU forwards + acid test; `pa_forward.py acid` for the acid
test alone), `pa_analyze.py` (agreement / KL / Cochran Q), `pa_finalize.py` (JSON assembly). Every
raw number is in the sibling `.json`.
