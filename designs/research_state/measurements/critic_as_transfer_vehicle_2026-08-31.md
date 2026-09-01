# M5 — IS THE CRITIC THE OFF-SLICE VEHICLE?

**2026-08-31 · owner-ordered.** One of the six probes aimed at the (i)/(ii) fork the rev-4
scorecard opened: *either v8's gift came from something no gen-era arm varies, or the fold metrics
do not measure what produced it.*

Artifacts beside this file: `critic_as_transfer_vehicle_2026-08-31.json` (every number) ·
`critic_as_transfer_vehicle_2026-08-31_tables.md` (the machine-rendered tables — every table below
is copied from it verbatim, nothing typed by hand) ·
`critic_as_transfer_vehicle_2026-08-31_cooccurrence_ci.json` (bootstrap CIs + partial correlations
for §5) · `critic_as_transfer_vehicle_probe.py` (phases `collect | analyze | report`, resumable).

---

## HEADLINE — three findings, and the registered predictions fail on both clauses

1. **The critic is NOT the off-slice vehicle.** In all four untaught cells the fold moved the
   critic's within-battle ordering *less* than an equal stretch of ordinary training did
   (**0.86–0.95×** the no-fold control), while v8's fold moved the **policy** 1.28–1.30× more than
   the control. The big mover is the policy, and it is the only meter that shows any ordering
   against the per-team gift (Spearman **+0.512 [+0.047, +0.815]**). Every critic meter's CI
   straddles zero in every cell.
2. **What a fold reliably delivers to the critic off-slice is LEVEL, not RESOLUTION** — and this is
   the same in the era that gifted and the era that did not. Reliability (calibration error) falls
   in **4 of 4** untaught cells (v8 0.0028→0.0006 and 0.0117→0.0032; gen 0.0265→0.0133 and
   0.0145→0.0073) and the over-optimism bias shrinks toward zero every time, while Murphy
   resolution moves between −6% and +2% of itself in three cells and **falls 13% (35% on early
   states)** in the fourth.
3. **Critic RESOLUTION improvement is LOCAL, exactly like behaviour.** On the **taught** slice
   resolution rises in both eras (v8 0.0588→0.0633, early **0.0082→0.0126, +54%**; gen
   0.0501→0.0565, early 0.0175→0.0202) — the one place it moves. The critic learns to discriminate
   where the teacher took it, and nowhere else.

**A fourth result, unregistered and arguably the most useful one:** the gen-era fold is
*behaviourally smaller than its own no-fold control* — rev-3's fold changed R2-ACTION **less** than
the preceding 4.07M steps of plain training did, on every meter (KL 0.91×, TV 0.95×, |Δz(V)| 0.92×,
within-battle de-ranking 0.88× on untaught; 0.85/0.72/0.71/0.83× on taught). v8's fold is above its
control on the policy axis; rev-3's is below its control on all of them. See §6 for the caveat that
keeps this from being a clean era law.

---

## The question, and why it is not "did the fold work"

Distilled behaviour can only generalise off-slice if the value function prices it correctly in
contexts the teacher never visited. Three prior measurements motivate suspecting the critic:

* the critic pathway runs ~7× LOWER effective rank than the policy pathway at every checkpoint
  (steady state, not decay — the plasticity audit);
* the critic is the **main casualty** of action-level distillation — off-slice `|ΔV|` 4–9 on a ±12
  scale within a handful of steps (`distillability_index_gen_2026-08-28.md` §5.6);
* the win-prob head is aggregate-calibrated but **per-state blind** — per-state |error| 0.278
  against an aggregate bias of +0.036, AUC 0.679 against ground truth's 0.970
  (`exploiter_fingerprint_truthcheck_2026-08-31.md`), with the outgoing-KO channel never routed to
  the critic at all (`ko_boundary_decodability_2026-08-31.md`).

So: **after a fold, on states from teams it never taught, what changed — the policy's action
distribution or the critic's valuation — and which of the two ORDERS with the measured per-team
gift?**

## Registered predictions (fixed in the mission brief, before any data)

1. v8's fold improves off-slice critic **RESOLUTION** (not just level) on untaught teams; the
   gen-era fold does not.
2. Policy-side change is LARGER than critic-side change in both eras, **but only the CRITIC-side
   change ORDERS with the untaught gift** (i.e. the big mover is not the vehicle).

Both are scored in §7. **P1 FAILS on its first clause** (v8 does not improve off-slice resolution
— on parent-generated states it degrades it) while accidentally passing on the second; **P2 passes
on its first clause and FAILS on its second**, with the sign of the evidence pointing at the policy.

---

## 1. Method

**Two eras, each a (parent → fold) pair plus a fixed reference opponent that is an ancestor of both
and equal to neither.**

| era | parent | fold | fixed reference opponent | published untaught outcome of the fold |
|---|---|---|---|---|
| **v8** | `ai_v8_04_distill_4teacher_0722` `final_model_interrupted.zip` | `ai_v8_14_distill3_0725` | `ai_v8_03_zarch_control_0718` | **GIFTED +5.42pp** [+3.44, +7.42] (probe P) |
| **gen** | `ai_v9_59_R2ACTION_0827` `final_model.zip` | `ai_v9_70_R3ACTION_0828` | `ai_v9_29_rev1_0823` | **null −0.75pp** [−4.56, +3.00] (probe Q) |

**States are GENERATED, not read from eval traces, and the reason is a mask.** The recorded
`states.npz` carries no `action_mask` and its `logits` are already-normalised log-probs, so the
`logits > -1e8` recovery returns ALL-LEGAL — the documented vacuous-guard trap that put 38.4%
phantom legality into a year of flip/KL audits. This probe plays its own battles and takes the mask
straight out of `embed_battle`, where it is the server-authoritative one.

**One pass, three networks, identical inputs.** For each probe team the actor pilots that ONE pinned
team against the reference opponent (greedy both sides, in-process bridge, no server, CPU); at every
decision with ≥2 legal actions the **parent**, the **fold** and a **control** checkpoint are scored
on the *identical* observation and the *identical* mask inside the same process. Nothing is
re-derived offline, so a state cannot drift between arms.

**The meters.**

| axis | meter |
|---|---|
| POLICY change | masked `KL(fold‖parent)`, total variation, argmax agreement |
| CRITIC change | Spearman ρ(V) pooled **and within-battle**, mean \|Δz(V)\| after per-arm z-scoring, ρ of the within-battle TD sequence, win-prob level shift |
| CRITIC quality | AUC(V, outcome), AUC(win-prob, outcome), and the **Murphy decomposition** of the win-prob Brier into RELIABILITY (level error) and RESOLUTION (discriminating content) |

The Murphy split answers the crux directly: `Brier = reliability − resolution + uncertainty`, so
*reliability* is exactly "is the level right" and *resolution* is exactly "does it separate states
that end differently". AUC is a second, purely rank-based resolution meter sharing no arithmetic
with it, and the two agree everywhere below.

**The critic's ranking over ACTIONS is SUBSTITUTED, and the substitution is named.** The critic is a
`V`, not a `Q`; a genuine per-action ranking needs one-ply materialisation of every legal successor
(the prober's `lookahead`) — seconds per action per state, and no such driver exists on the v8-era
tree. Reported instead: the ranking the critic induces **over states**, computed *within battle*
(the pooled version is dominated by the global who-is-winning axis that both critics get right),
plus the rank correlation of the within-battle **TD sequence** — the credit the critic assigns to
the transitions actually taken. Decision-relevant, but not the action ranking; no claim here should
be read as if it were.

**PopArt is why |ΔV| is z-scored.** The arms carry different PopArt shift/scale, so a raw `|ΔV|`
conflates an affine re-scaling with a change of shape. Each arm's V is z-scored within the cell
before differencing.

### 1.1 The matched-noise control — without it "the policy moved more" is not a statement

KL (nats) and rank de-correlation (dimensionless) are different units. Every fold change is
therefore also computed for a **parent ← an-earlier-checkpoint-of-the-parent's-own-run** pair —
ordinary training, no fold — and reported as a RATIO. That is what makes the comparison scale-free.

| era | control checkpoint | control span | fold span | matched? |
|---|---|---|---|---|
| v8 | `ai_v8_04/checkpoints/checkpoint_269716291_steps.zip` | 7.46M (269.72M → 277.18M) | ~14.8M | **half** — the control understates ordinary movement, so every v8 ratio is an **over-estimate** |
| gen | `ai_v9_59/snapshots/snapshot_000024000000.zip` | 4.07M (24.00M → 28.07M) | ~4.55M | yes |

### 1.2 Labels: many states with 1-draw labels, not few states with tight-MC ones

The mission offered the truthcheck's tight-MC method and asked which was chosen. **Chosen: realized
battle outcomes, at scale.** A battle's outcome is one draw from the true value distribution at
every state that battle contains, so AUC, the Murphy split and the reliability curve are all
consistently estimable from single draws under battle-clustered inference; what R=40 MC buys is the
resolution of an *individual anchor*, which is what the truthcheck needed (it classified single
states into a boundary band) and this probe does not. The budget bought **30,108 states** instead of
~600. The cost is stated in Limits §8.2: the label is the *actor's* continuation, so the arm that
generated the states is judged on-policy and the other off-policy — which is exactly why both state
sets were collected, and why the two v8 cells disagree in the way they do.

### 1.3 Design and provenance

`n = 12` paired battles per (team, actor); the opponent's team draw and the sim seeds come from one
`random.Random(20260831)` sequence, identical across teams and actors. **60 team-cells, 720
battles, 30,108 scored states, ZERO dropped battles.**

| cell | teams | classes | battles | states |
|---|---|---|---|---|
| `v8/parent` | 22 | 16 untaught + 6 taught | 264 | 11,235 |
| `v8/fold` | 16 | untaught only | 192 | 6,967 |
| `gen/parent` | 14 | 8 untaught + 6 taught | 168 | 7,089 |
| `gen/fold` | 8 | untaught only | 96 | 4,817 |

Team sets are **inherited, never re-drawn**: v8's 22 are probe P's own probe set, resolved from its
published per-team shas against the era pool (all 22 present); gen's 8 untaught are probe Q's
pre-registered selection; gen's 6 taught are one per rev-3 teacher cluster F6a–F6f, read at run time
from each fleet arm's recorded `--trainee-teams`.

**The v8 arm runs in an era-pinned worktree** (`/tmp/probeP_v8era` @ `b13b30b2`) on the **node**
bridge — the era's rust bridge predates the seedless-seed fix `bc00d4d` and would replay one dice
stream. The gen arm runs on the current tree on rust. Both arms of any comparison always live in one
era and one tree; nothing is compared across trees.

**ACID.** Every collection process refuses to start unless parent and fold state-dicts differ:
measured L2 **238.923** (v8) and **51.835** (gen). A mis-resolved path loading one zip twice would
otherwise read as a perfect null.

---

## 2. POLICY change vs CRITIC change — the headline tables

Battle-clustered bootstrap 95% CIs (4,000 resamples), cluster = battle.

### 2.1 v8 — the era whose fold GIFTED (+5.42pp untaught)

**states generated by the PARENT arm**

| class | teams | battles | states | WR | KL(fold‖parent) | TV | argmax-agree | \|Δz(V)\| | ρ(V) pooled / within-battle | Δwin-prob level |
|---|---|---|---|---|---|---|---|---|---|---|
| untaught | 16 | 192 | 7293 | 0.431 | 0.2421 [0.2317, 0.2530] | 0.2313 [0.2262, 0.2366] | 0.685 [0.671, 0.699] | 0.238 [0.214, 0.269] | 0.9378 / 0.9114 | -0.0553 [-0.0678, -0.0440] |
| taught | 6 | 72 | 3942 | 0.380 | 0.2963 [0.2639, 0.3328] | 0.2578 [0.2424, 0.2743] | 0.632 [0.596, 0.663] | 0.295 [0.263, 0.329] | 0.9092 / 0.8708 | -0.0029 [-0.0198, 0.0125] |

**states generated by the FOLD arm** (untaught only)

| class | teams | battles | states | WR | KL(fold‖parent) | TV | argmax-agree | \|Δz(V)\| | ρ(V) pooled / within-battle | Δwin-prob level |
|---|---|---|---|---|---|---|---|---|---|---|
| untaught | 16 | 192 | 6967 | 0.374 | 0.2422 [0.2315, 0.2525] | 0.2299 [0.2247, 0.2349] | 0.691 [0.676, 0.706] | 0.258 [0.236, 0.282] | 0.9296 / 0.9137 | -0.0563 [-0.0670, -0.0451] |

### 2.2 gen — the era whose fold did NOT gift (−0.75pp, n.s.)

**states generated by the PARENT arm**

| class | teams | battles | states | WR | KL(fold‖parent) | TV | argmax-agree | \|Δz(V)\| | ρ(V) pooled / within-battle | Δwin-prob level |
|---|---|---|---|---|---|---|---|---|---|---|
| untaught | 8 | 96 | 5059 | 0.522 | 0.3645 [0.3361, 0.3936] | 0.2394 [0.2252, 0.2535] | 0.722 [0.697, 0.746] | 0.240 [0.219, 0.260] | 0.9313 / 0.8889 | -0.0501 [-0.0617, -0.0384] |
| taught | 6 | 72 | 2030 | 0.371 | 0.3864 [0.3365, 0.4376] | 0.1987 [0.1830, 0.2139] | 0.790 [0.765, 0.814] | 0.242 [0.221, 0.263] | 0.9327 / 0.8882 | -0.0464 [-0.0573, -0.0356] |

**states generated by the FOLD arm** (untaught only)

| class | teams | battles | states | WR | KL(fold‖parent) | TV | argmax-agree | \|Δz(V)\| | ρ(V) pooled / within-battle | Δwin-prob level |
|---|---|---|---|---|---|---|---|---|---|---|
| untaught | 8 | 96 | 4817 | 0.569 | 0.3892 [0.3456, 0.4416] | 0.2505 [0.2312, 0.2729] | 0.710 [0.680, 0.740] | 0.260 [0.229, 0.293] | 0.9279 / 0.9045 | -0.0370 [-0.0562, -0.0207] |

**Absolute KL is not comparable across eras** (different architectures, different action-entropy
regimes — gen's KLs run 1.5-2.1x v8's on every pair, control included). Only the within-era ratios
in §3 are.

---

## 3. The scale-free comparison — fold change ÷ matched no-fold control

Ratio > 1 means the fold moved that meter *more* than a comparable stretch of ordinary training did.

| cell | class | KL | TV | argmax-DISagreement | \|Δz(V)\| | V within-battle DE-ranking |
|---|---|---|---|---|---|---|
| **v8/parent** | untaught | **1.28×** | 1.10× | 1.05× | **0.94×** | **0.95×** |
| v8/parent | taught | **1.48×** | 1.16× | 1.12× | 1.13× | 1.29× |
| **v8/fold** | untaught | **1.30×** | 1.11× | 1.12× | **1.05×** | **0.92×** |
| **gen/parent** | untaught | **0.91×** | 0.95× | 1.00× | 0.92× | 0.88× |
| gen/parent | taught | 0.85× | 0.72× | 0.72× | 0.71× | 0.83× |
| **gen/fold** | untaught | 0.94× | 0.94× | 0.93× | 1.12× | 0.86× |

**Read the untaught rows.** v8's fold is the only one that clears its control on the policy meter,
and it clears it on the policy meter *only*: KL 1.28×/1.30× while both critic-ordering meters sit at
or below 1.00×. rev-3's fold clears nothing — it is at or below its control on every axis. The two
eras differ in *whether the fold did anything unusual to the policy at all*; neither era's fold did
anything unusual to the critic's ordering off-slice.

The underlying pairs, so the ratios can be audited:

| cell | class | pair | KL | TV | argmax-agree | \|Δz(V)\| | ρ(V) within-battle | ρ(TD) within-battle |
|---|---|---|---|---|---|---|---|---|
| v8/parent | untaught | fold_vs_parent | 0.2421 | 0.2313 | 0.685 | 0.238 | 0.9114 | 0.8357 |
| v8/parent | untaught | parent_vs_control | 0.1892 | 0.2106 | 0.701 | 0.253 | 0.9069 | 0.8282 |
| v8/parent | taught | fold_vs_parent | 0.2963 | 0.2578 | 0.632 | 0.295 | 0.8708 | 0.7845 |
| v8/parent | taught | parent_vs_control | 0.1997 | 0.2223 | 0.670 | 0.261 | 0.8994 | 0.7981 |
| v8/fold | untaught | fold_vs_parent | 0.2422 | 0.2299 | 0.691 | 0.258 | 0.9137 | 0.8383 |
| v8/fold | untaught | parent_vs_control | 0.1858 | 0.2072 | 0.724 | 0.246 | 0.9060 | 0.8235 |
| gen/parent | untaught | fold_vs_parent | 0.3645 | 0.2394 | 0.722 | 0.240 | 0.8889 | 0.8312 |
| gen/parent | untaught | parent_vs_control | 0.4021 | 0.2528 | 0.723 | 0.260 | 0.8744 | 0.7995 |
| gen/parent | taught | fold_vs_parent | 0.3864 | 0.1987 | 0.790 | 0.242 | 0.8882 | 0.8624 |
| gen/parent | taught | parent_vs_control | 0.4560 | 0.2761 | 0.707 | 0.342 | 0.8646 | 0.8154 |
| gen/fold | untaught | fold_vs_parent | 0.3892 | 0.2505 | 0.710 | 0.260 | 0.9045 | 0.8505 |
| gen/fold | untaught | parent_vs_control | 0.4158 | 0.2655 | 0.688 | 0.233 | 0.8892 | 0.8056 |

⚠️ **The v8 ratios are over-estimates** (its control spans half the fold's interval, §1.1). The
*direction* of that bias is known and it works against the era contrast, not for it: correcting for
it would pull v8's 1.28× toward 1.0 and would not move gen's 0.91×. **The clean half of the era
contrast is therefore the gen side — a fold that is measurably SMALLER than its own matched control
— and the v8 side should be read as "at least not below its control", which is still the opposite
of gen.**

---

## 4. CRITIC QUALITY — resolution vs level, the crux

`Brier = reliability − resolution + uncertainty`; **reliability is the LEVEL error, resolution is
the discriminating content.** AUC is the independent rank-only resolution meter.

| cell | class | arm | AUC(V) | AUC(win-prob) | RESOLUTION | RELIABILITY | bias | Brier |
|---|---|---|---|---|---|---|---|---|
| v8/parent | untaught | control | 0.823 [0.771, 0.869] | 0.821 [0.777, 0.862] | 0.0739 | 0.0226 | +0.135 | 0.1929 |
| v8/parent | untaught | **parent** | 0.848 [0.807, 0.885] | 0.843 [0.803, 0.880] | **0.0844** | 0.0028 | +0.041 | 0.1625 |
| v8/parent | untaught | **fold** | 0.822 [0.782, 0.862] | 0.817 [0.775, 0.857] | **0.0734** | **0.0006** | −0.015 | 0.1720 |
| v8/parent | taught | control | 0.751 [0.672, 0.824] | 0.756 [0.680, 0.829] | 0.0429 | 0.0592 | +0.222 | 0.2509 |
| v8/parent | taught | **parent** | 0.773 [0.705, 0.838] | 0.784 [0.717, 0.848] | **0.0588** | 0.0131 | +0.090 | 0.1902 |
| v8/parent | taught | **fold** | 0.766 [0.692, 0.838] | 0.792 [0.718, 0.861] | **0.0633** | 0.0134 | +0.087 | 0.1852 |
| v8/fold | untaught | control | 0.831 [0.793, 0.867] | 0.824 [0.785, 0.860] | 0.0712 | 0.0462 | +0.196 | 0.2085 |
| v8/fold | untaught | **parent** | 0.817 [0.778, 0.853] | 0.820 [0.780, 0.856] | **0.0700** | 0.0117 | +0.097 | 0.1751 |
| v8/fold | untaught | **fold** | 0.827 [0.789, 0.860] | 0.823 [0.786, 0.859] | **0.0716** | **0.0032** | +0.041 | 0.1649 |
| gen/parent | untaught | control | 0.829 [0.765, 0.883] | 0.817 [0.752, 0.871] | 0.0771 | 0.0245 | +0.144 | 0.1959 |
| gen/parent | untaught | **parent** | 0.818 [0.753, 0.874] | 0.810 [0.748, 0.863] | **0.0736** | 0.0265 | +0.149 | 0.2009 |
| gen/parent | untaught | **fold** | 0.807 [0.737, 0.867] | 0.800 [0.733, 0.855] | **0.0726** | **0.0133** | +0.099 | 0.1888 |
| gen/parent | taught | control | 0.737 [0.670, 0.800] | 0.739 [0.665, 0.809] | 0.0389 | 0.0729 | +0.258 | 0.2667 |
| gen/parent | taught | **parent** | 0.749 [0.674, 0.818] | 0.773 [0.699, 0.839] | **0.0501** | 0.1098 | +0.313 | 0.2914 |
| gen/parent | taught | **fold** | 0.795 [0.729, 0.855] | 0.791 [0.720, 0.856] | **0.0565** | **0.0797** | +0.266 | 0.2557 |
| gen/fold | untaught | control | 0.798 [0.730, 0.858] | 0.789 [0.703, 0.867] | 0.0636 | 0.0136 | +0.111 | 0.1942 |
| gen/fold | untaught | **parent** | 0.812 [0.745, 0.873] | 0.802 [0.720, 0.876] | **0.0696** | 0.0145 | +0.109 | 0.1896 |
| gen/fold | untaught | **fold** | 0.801 [0.734, 0.859] | 0.788 [0.696, 0.863] | **0.0657** | **0.0073** | +0.072 | 0.1860 |

**EARLY stratum** — first half of each battle. A state in the last third of a decided game is
near-terminal and both critics call it, which pushes every AUC toward a shared ceiling; the early
stratum is where a resolution difference can exist at all.

| cell | class | states | arm | AUC(win-prob) | RESOLUTION | RELIABILITY | bias |
|---|---|---|---|---|---|---|---|
| v8/parent | untaught | 3498 | control | 0.742 [0.686, 0.796] | 0.0411 | 0.0471 | +0.214 |
| v8/parent | untaught | 3498 | **parent** | 0.748 [0.689, 0.803] | **0.0440** | 0.0072 | +0.078 |
| v8/parent | untaught | 3498 | **fold** | 0.695 [0.630, 0.758] | **0.0288** | **0.0010** | +0.006 |
| v8/parent | taught | 1919 | control | 0.561 [0.458, 0.668] | 0.0053 | 0.1237 | +0.312 |
| v8/parent | taught | 1919 | **parent** | 0.587 [0.480, 0.697] | **0.0082** | 0.0366 | +0.136 |
| v8/parent | taught | 1919 | **fold** | 0.589 [0.476, 0.711] | **0.0126** | 0.0313 | +0.118 |
| v8/fold | untaught | 3344 | control | 0.711 [0.653, 0.764] | 0.0300 | 0.0720 | +0.266 |
| v8/fold | untaught | 3344 | **parent** | 0.702 [0.642, 0.759] | **0.0281** | 0.0175 | +0.130 |
| v8/fold | untaught | 3344 | **fold** | 0.690 [0.631, 0.745] | **0.0262** | **0.0051** | +0.056 |
| gen/parent | untaught | 2459 | control | 0.728 [0.646, 0.803] | 0.0432 | 0.0510 | +0.216 |
| gen/parent | untaught | 2459 | **parent** | 0.711 [0.628, 0.788] | **0.0342** | 0.0443 | +0.208 |
| gen/parent | untaught | 2459 | **fold** | 0.702 [0.617, 0.778] | **0.0350** | **0.0311** | +0.167 |
| gen/parent | taught | 958 | control | 0.584 [0.481, 0.689] | 0.0067 | 0.1317 | +0.356 |
| gen/parent | taught | 958 | **parent** | 0.653 [0.550, 0.750] | **0.0175** | 0.1649 | +0.400 |
| gen/parent | taught | 958 | **fold** | 0.674 [0.564, 0.774] | **0.0202** | **0.1235** | +0.350 |
| gen/fold | untaught | 2338 | control | 0.671 [0.560, 0.779] | 0.0245 | 0.0299 | +0.166 |
| gen/fold | untaught | 2338 | **parent** | 0.694 [0.588, 0.799] | **0.0314** | 0.0298 | +0.156 |
| gen/fold | untaught | 2338 | **fold** | 0.682 [0.568, 0.779] | **0.0310** | **0.0216** | +0.127 |

### 4.1 The verdict, stated as the mission asked: **which moved, resolution or level?**

**LEVEL — everywhere, in both eras, on both team classes.** Reliability falls from parent to fold in
**5 of 6** cell×class rows, the sixth flat (0.0028→0.0006 · 0.0131→0.0134, the flat one · 0.0117→0.0032 ·
0.0265→0.0133 · 0.1098→0.0797 · 0.0145→0.0073) and the bias always moves toward zero, in v8's
untaught case straight through it (+0.041 → −0.015). Folds systematically de-bias the critic's
optimism; that is the one thing they reliably do to it.

**RESOLUTION — only on the TAUGHT slice.** Untaught: −13% (v8 parent-states), +2% (v8 fold-states),
−1% (gen parent-states), −6% (gen fold-states) — three of four negative, none large, and the AUC
meter agrees row for row. Taught: **+8% (v8, and +54% on early states)** and **+13% (gen, +15%
early)**. The critic's ability to tell a won position from a lost one improves where the teacher
took it and not one team further.

That is the same locality the behaviour shows (`project_fold_transfer_is_local`: +6pp only on taught
teams). It is the *critic's* version of the same fact, measured independently.

---

## 5. Co-occurrence with the per-team gift

Per-team meters against probe P's / probe Q's own published per-team fold−parent win-rate delta.
Spearman with a 20,000-resample bootstrap over TEAMS, plus the partial correlation given the
parent's win rate on that team (the obvious confound: a team with more room to gain might also be a
team the fold changed more).

| cell | n | KL | TV | \|Δz(V)\| | ΔAUC(win-prob) | Δresolution |
|---|---|---|---|---|---|---|
| **v8/parent** | 16 | **+0.512 [+0.047, +0.815]** *(partial +0.508)* | +0.288 [−0.253, +0.738] | −0.276 [−0.662, +0.265] | −0.175 [−0.829, +0.471] | −0.188 [−0.724, +0.365] |
| **v8/fold** | 16 | +0.435 [−0.109, +0.753] *(partial +0.448)* | +0.203 [−0.344, +0.603] | +0.082 [−0.429, +0.550] | +0.321 [−0.236, +0.692] | +0.374 [−0.221, +0.741] |
| gen/parent | 8 | +0.214 [−0.643, +0.857] | +0.333 [−0.500, +0.929] | +0.095 [−0.738, +0.929] | +0.333 [−0.452, +0.905] | +0.333 [−0.500, +1.000] |
| gen/fold | 8 | +0.071 [−0.643, +0.786] | +0.143 [−0.619, +0.857] | +0.214 [−0.548, +0.905] | +0.214 [−0.667, +0.833] | +0.286 [−0.524, +0.857] |

*(The two v8 ΔAUC cells are computed over **15** teams, not 16: `048182d1e9` won 0 of 12 battles in
both v8 cells, so its AUC is undefined and the pair is DROPPED rather than rank-imputed. The
`report` phase's own inline line ranks the NaN and therefore prints −0.194 / +0.265 for those two
cells; the NaN-dropped values in this table are the correct ones and are what
`..._cooccurrence_ci.json` carries.)*

**One cell of one meter excludes zero, and it is the POLICY meter.** `KL(fold‖parent)` on
v8's parent-generated states: **+0.512 [+0.047, +0.815]**, unchanged by conditioning on the parent's
per-team win rate (+0.508; gift-vs-parent-WR is itself ρ = −0.071, so there is no confound to
remove). It reproduces in sign and near-magnitude on the independently-generated fold state set
(+0.435), where it just fails to exclude zero.

**No critic meter orders with the gift anywhere.** Every critic CI straddles zero in all four cells,
and the two v8 cells *disagree in sign* on ΔAUC and Δresolution (−0.18/−0.18 vs +0.32/+0.37) — a
meter whose sign depends on which arm generated the states is not measuring a property of the gift.

The gen rows carry n = 8 and, more importantly, probe Q's per-team deltas span only −8.5pp…+7.5pp
with per-team CIs of roughly ±9pp: **that x-axis is mostly noise, so the gen co-occurrence rows are
uninformative by construction and are reported for completeness, not as evidence.**

The per-team rows for the v8 cells (the ones the correlations are computed from) are in
`..._tables.md` and in the JSON under `cells/<cell>/gift_cooccurrence/rows`.

---

## 6. What this says about the (i)/(ii) fork

The probe was dispatched on the hypothesis that **the critic is the vehicle** and that the gen era's
under-resolved critic is why its folds do not radiate. **The hypothesis is not supported, on its own
terms and by its own registered predictions.**

* Off-slice, *neither* era's fold moves the critic's ordering more than ordinary training does.
* Off-slice, *neither* era's fold improves the critic's resolution; both improve its level.
* The one thing that differs between the eras — v8's fold clearing its control on the policy meter
  while rev-3's fails to clear its control on anything — is a **policy-side** difference, and the
  policy meter is also the only one with a detectable ordering against the per-team gift.

**What replaces it as the live reading.** v8's fold made a *bigger behavioural change* off-slice
than rev-3's did, relative to what each network was doing anyway, and the size of that change tracks
the size of the gift across teams. That is consistent with the coverage/breadth account already in
the ledger (v8 taught 22 teams across 3 teachers; rev-3 taught 12 across 6) and with the maturity
account the fallback tree activated — a 277M-step parent has a slower ordinary-training clock, so
the same fold registers as a larger relative displacement. **This probe cannot separate those two**;
it only removes the critic from the list.

**The gen-era observation worth carrying forward on its own:** rev-3's fold changed R2-ACTION
*less* than 4.07M steps of plain training did, on every meter and both team classes. A fold that is
below its own no-fold noise floor has nothing to radiate — which is a mechanism for "no gift" that
needs no critic story at all, and which is directly checkable on any future fleet before its
untaught cut is measured.

---

## 7. Predictions scored

**P1 — "v8's fold improves off-slice critic RESOLUTION (not just level) on untaught teams; the
gen-era fold does not." → FAILS on the first clause; the second is right for the wrong reason.**
v8's fold *degrades* off-slice resolution on parent-generated states (0.0844 → 0.0734, −13%; early
0.0440 → 0.0288, −35%; AUC 0.843 → 0.817) and leaves it flat on its own states (0.0700 → 0.0716).
The gen fold indeed does not improve it (0.0736 → 0.0726) — but that is not a contrast, because
neither does v8's. The prediction's underlying claim, that off-slice resolution is where the eras
differ, is false: **they differ on the policy axis and agree on the critic axis.** What v8's fold
*does* improve is the level (bias +0.041 → −0.015, reliability 0.0028 → 0.0006), which the
prediction explicitly excluded.

**P2 — "policy-side change is LARGER than critic-side change in both eras, but only the CRITIC-side
change ORDERS with the untaught gift." → first clause PASSES, second clause FAILS with the sign
reversed.** Against the matched control, the policy meter is the larger mover in v8 (1.28–1.48× vs
0.94–1.13×) and is the only meter that separates the eras; in gen both are below 1.0 with the policy
meter still the higher of the two on untaught. And the ordering against the gift is carried by the
**policy** meter (+0.512 [+0.047, +0.815], the only interval excluding zero anywhere in §5), not the
critic — **the big mover IS, on this evidence, the vehicle.**

---

## 8. Limits — read before quoting

1. **The ACTION-ranking substitution (§1).** The mission asked for the change in the ranking the
   critic induces over legal actions. That needs one-ply successor materialisation, which does not
   exist on the era tree; what is reported is the ranking over *states* within a battle plus the TD
   sequence. If the fold re-ordered the critic's *action* preferences without re-ordering its state
   preferences, this probe would not see it. That is the single largest gap and the natural next
   instrument (`lookahead` on the gen era, where the driver exists).
2. **Single-draw labels are the ACTOR's continuation.** With states generated by the parent, the
   parent's V is judged on-policy and the fold's off-policy — which biases the level comparison
   toward the parent, and is the most likely explanation for why v8's resolution reads −13% on
   parent states and +2% on fold states. Both sets were collected for exactly this reason and the
   **conclusion is only drawn where the two agree** (level improves in both; the critic-change
   ratios are ≤1 in both; no critic meter orders with the gift in either). Where they disagree
   (the sign of Δresolution) the finding is reported as *not established*, never averaged.
3. **The v8 control spans half the fold's interval**, so every v8 ratio in §3 is an over-estimate
   (§1.1). The era contrast's clean half is the gen side.
4. **Two eras is two points, and they differ in everything** — architecture (2992-dim vs 2501-dim
   obs), parent maturity (277M vs 28M steps), teacher count and breadth, reference opponent,
   bridge implementation. Nothing here isolates *which* difference matters; the within-v8
   co-occurrence over 16 teams is the better-powered internal test and it is the one carrying the
   §5 claim.
5. **n = 12 battles per (team, actor)** makes every *per-team* AUC/resolution noisy; the by-class
   rows (72–192 battles) are the powered reads and the per-team numbers exist only to feed the
   rank correlations, which are themselves reported with bootstrap CIs precisely because of it.
   One team (`048182d1e9`) won 0 of 12 in a v8 cell, so its AUC is undefined and is recorded as
   MISSING, never imputed.
6. **The gen co-occurrence x-axis is noise** (probe Q's per-team CIs are ±9pp against an 16pp
   spread). Those rows are not evidence.
7. **Greedy, fixed reference opponent, one pinned team per battle** — the standing per-team
   piloting meter's regime, chosen for comparability with probes P and Q, not because it is the
   distribution the models train on.
8. **Not measured here:** whether the fold's *level* correction is worth anything behaviourally.
   De-biasing an optimistic critic is a real improvement in Brier (0.1625→0.1720 on v8 untaught is
   actually a *worsening*, because the resolution loss outweighs the reliability gain; 0.1751→0.1649
   on the fold's own states is an improvement) but PPO's advantage is invariant to a constant value
   offset, so a pure level shift is close to behaviourally inert by construction. That is an
   argument, not a measurement.

---

## 9. Reproducing

```bash
# v8 arm — ERA-PINNED tree, node bridge (the era's rust bridge predates bc00d4d)
export PYTHONPATH=/tmp/probeP_v8era/src            # git worktree @ b13b30b2
P=designs/research_state/measurements/critic_as_transfer_vehicle_probe.py
nice -n 15 python $P collect --era v8  --actor parent --n 12 --shard 0/2   # + 1/2
nice -n 15 python $P collect --era v8  --actor fold   --n 12 --kinds untaught --shard 0/2
# gen arm — current tree, rust bridge
export PYTHONPATH=$PYTHONPATH:src
nice -n 15 python $P collect --era gen --actor parent --n 12 --shard 0/2
nice -n 15 python $P collect --era gen --actor fold   --n 12 --kinds untaught --shard 0/2
nice -n 15 python $P analyze && nice -n 15 python $P report
```

CPU only, 1 torch thread, `nice -n 15`, two shards = two cores; `models/` is read-only throughout.
Wall cost: **74 min of collection** on 2 niced cores beside a live 20-arm GPU fleet, plus ~8 min of
bootstrap. Per-state rows are written to `$M5_SCRATCH` (default `/tmp/m5_critic_vehicle`) and are
**deliberately not committed** — 24k rows × 37 floats regenerate in 74 min and the JSON carries
every aggregate. Every markdown table in §2–§4 is rendered by `--phase report` straight from the
analysis JSON.
