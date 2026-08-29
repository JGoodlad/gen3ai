# PROBE M — does the WIN-PROB head price what the hand-coded BIAS tax taxes?

*Measured 2026-08-29 · **7 current-arch runs** (`ai_v9_29_rev1_0823` + the R2/R3 era:
`ai_v9_58_R2CTRL`, `_59_R2ACTION`, `_60_R2TOPK`, `_61_R2KL`, `_62_R2PLAIN`, `_70_R3ACTION`) ·
**5,035 battles / 147,204 decisions**, every eval-trace opponent class · CPU-only, no model
loaded, `nice 15` · data `bias_tax_head_alignment_2026-08-29.json`.*

## The question, and what happened to it

The mission asked whether the learned outcome signal already prices what the hand-coded BIAS taxes
tax, and where they disagree. The first step — enumerate the ACTUAL BIAS members from code rather
than assume them — **collapsed the scope before a single number was measured**, and that collapse
is the first finding.

---

## 0. THE BIAS-MEMBER ENUMERATION — 29 declared, **exactly 1 live**

`RewardBreakdown._REGISTRY` (`reward_manager.py:265-292`) is the declared, exhaustive reward
registry. `registry_fields(RewardClass.BIAS)` returns **29** names. But membership is not
activity: `_bias_term_active(config, name)` gates each one, and **`--all-shaping-pbrs` — DEFAULT-ON
since 2026-08-18 — returns `False` for every BIAS term except `no_progress_tax`**
(`reward_manager.py:410-411`, a single unconditional `if asp: return False`).

Every one of the seven runs records the identical composition in its own `metadata.json`:

```
1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)
```

`--no-progress-penalty 0.15`, `--bias-additivity 1.0` (λ=1 ⇒ the accumulate-refund is identically
zero, so the charge is **fully additive**), `--stall-pbrs` off. Seven runs, three different git
hashes, one composition.

| BIAS member | one-line semantics | live? |
|---|---|---|
| **`no_progress_tax`** | **flat `−0.15` on a "charged NO_OP" window — a deliberate, obs-knowable wheel-spin, when a switch was legal** | **YES — the only one** |
| `stall_tax` | progressive per-turn tax past `STALL_TAX_START_TURN` (turn-count, progress-blind) | zeroed by `all_shaping_pbrs` |
| `explosion_block` | Ghost-immunity / Protect blocked an opponent Explosion | zeroed |
| `finishing_blow` | a damaging move secured the KO | zeroed |
| `self_ko_penalty` | HP-scaled penalty for self-KOing a healthy mon (Explosion/Self-Destruct) | zeroed (also `--self-ko-hp-penalty 0.0`) |
| `roar` | successful phaze | zeroed |
| `futile_attack` | attacked into an immunity / no effect | zeroed |
| `futile_setup` | setup move used at the ±6 stat cap | zeroed |
| `setup_low_hp` | setup move chosen below 40% HP | zeroed |
| `boost_utilized` | attacked while holding active stat boosts | zeroed |
| `status_wasted` | status-inflicting move had no effect | zeroed |
| `spikes` | a Spikes layer was added | zeroed |
| `futile_spikes` | Spikes used at the 3-layer cap | zeroed |
| `matchup_penalty` | staying in a bad type matchup | zeroed |
| `dead_matchup_tax` | escalating penalty for staying in a 0×-only matchup | zeroed |
| `stay_risk_tax` | belief-risk-scaled penalty for staying into a high P(KO) with a safe pivot | zeroed (also `--switch-bias-weight 0.0`) |
| `escape_risk_bonus` | belief-risk-scaled reward for escaping a high-P(KO) spot | zeroed (idem) |
| `switch_base` | flat per-voluntary-switch **subsidy** | zeroed |
| `switch_bouncing_tax` | penalty for immediately switching back | zeroed |
| `repetition_tax` | same attack repeated consecutively | zeroed |
| `struggle_tax` | Struggle-loop penalty | zeroed |
| `pivot_protect` / `pivot_status` / `pivot_damage` | what the opponent did on our switch turn | zeroed |
| `se_switch` | our switch-in has a super-effective move vs the opponent active | zeroed |
| `escape_threat_switch` | switched out under a revealed SE threat | zeroed |
| `sleep_out` / `sleep_in` | sleeping-mon rotation credit / penalty | zeroed |
| `status` | count-diff standing value of status on the opponent | zeroed |

> **The mission's framing — "the anti-stall FAMILY: heal-war grace, stall penalties, draw-related
> terms" — describes a reward function that has not run in production since 2026-08-18.** The
> heal-war grace is not a reward term at all (it is `HEAL_FREEZE_GRACE`, a *branch inside*
> `ProgressClock`); `stall_tax` is dead; the draw term (`--draw-penalty`) is TERMINAL, not BIAS.
> What is left is one flat charge. **The probe re-scoped to measure that one charge exhaustively
> rather than measure a family that does not exist.**

---

## 1. THE RECONSTRUCTION — exact, model-free, and gated

`no_progress_tax` is not re-simulated here; it is **read off the recorded observations**. That is
possible because of the Markovian design's own contract: `ProgressClock.update()` sets `self.n`
(which the obs encoder writes as the board scalar `turns_since_progress`) and `self.last_penalty`
(which `Gen3RewardManager._apply_progress_clock` reads) **in the same call**, so obs and reward key
on one number.

**The decode.** The obs scalar is `log(1+min(n,10))/log(11)` — an exact 11-point lattice, so the
integer `n` is recovered losslessly and any deviation is detectable. Column **1602** was located
*empirically* (the only column in the board block on the lattice AND obeying counter dynamics) and
it coincides with the documented offset (`reactive.py:218-224`, board `vec[2]`).

**The charge condition is exact by SOURCE, not by sampling.** Every path in `update()` that sets a
non-zero `last_penalty` (the capped-Spikes short-circuit, the wasted-self-cure short-circuit, and
the final NO_OP branch) executes the identical three lines — increment `n`, then charge iff
`legal.switches` is non-empty — and every path that sets `last_penalty = 0.0` (forced-switch
early-return, progress, exogenous denial, in-grace heal) leaves `n` untouched. So

> **`n` incremented ⟺ a charge fired, modulo `legal.switches`**

with one exception, which is excluded: at the `PROGRESS_CLOCK_CAP = 10` clamp an increment is
invisible (2,924 rows, 2.0%). And `legal.switches` is exactly mask bits 0-5 —
`Gen3ActionMasker.mask_from_legal` sets `mask[sw.slot]` from that very list, and
`gen3_env.embed_battle` hands the SAME `LegalActions` object to the masker and to the clock.

**The fold→window alignment was MEASURED, not assumed — and the intuitive answer was wrong.**
`TurnDelta.phase_is_forced_switch` reads `curr_ctx.phase`, and `curr_ctx` is built at embed time,
i.e. from the **upcoming** request. Testing both candidate alignments against the clock's
documented sit-out (`update()` returns early on a forced switch):

| the fold sits out when… | n | windows where the clock nonetheless moved |
|---|---|---|
| **decision t** is a forced switch | 10,424 | **8,710 (83.6%)** |
| **decision t+1** is a forced switch | 10,442 | **0 (0.0%)** |

*(aggregate over the seven runs' validation pass — 2,677 battles, 78,664 steps.)*

Alignment is therefore `t+1`, decisively. Three further gates pass on all seven runs: **0 illegal
clock transitions** in those 78,664 steps; **all 2,677** validated battles open on the same `n`
(=1, one degenerate `reset()`-time fold — not a charged window, and the tax reads a difference);
and the mask-derived `forced switch` label agrees with the summary's independently-recorded `phase`
field on **147,204 / 147,204** rows.

**Classes** (attributed to decision *t*, whose reward `r_t` carries the charge):

| class | rule | n | share |
|---|---|---|---|
| **TAXED** | n rose, a switch was legal | **34,235** | **23.3%** |
| NEUTRAL | n held at 0 (progress-or-freeze, untaxed) | 53,325 | 36.2% |
| PROGRESS | n reset to 0 from >0 | 29,678 | 20.2% |
| SITOUT | the fold's request is a forced switch — clock sits out | 19,503 | 13.2% |
| TRAPPED | n rose, no switch legal — the helplessness exemption | 4,293 | 2.9% |
| FROZEN | n held >0 — exogenous denial, or an in-grace heal | 3,246 | 2.2% |
| CAP | n == 10, increment invisible — **excluded from every arm** | 2,924 | 2.0% |

---

## 2. THE TAX CENSUS — what actually gets charged

**23.3% of every decision the model makes carries the charge.** 6.80 charges per battle × `−0.15` =
**`−1.02` reward units per battle, 3.4% of a win** (`VICTORY_VALUE = 30`).

But the composition is not what the term's name describes:

| action kind | n | P(taxed \| kind) | share of ALL charges |
|---|---|---|---|
| **voluntary switch** | 19,957 | **0.733** | **42.7%** |
| **forced switch** (post-faint replacement) | 19,466 | **0.639** | **36.3%** |
| move | 106,798 | 0.067 | 20.9% |
| struggle | 983 | 0.009 | 0.03% |

> **`no_progress_tax` is 79% a SWITCH tax and 36% a post-faint-replacement tax.** Attacking is
> charged on 6.7% of decisions; switching on 73%.

This is structural, not incidental. The clock's `_is_progress` credits our-attributed damage, a
status landing, a hazard layer, a forced opponent commit, a winning residual, a boost rise, a new
Substitute, a Wish cast — **none of which any switch action can produce**. A switch's only escape
routes are opponent-determined (they also switched) or a pre-existing residual. The charge is
therefore near-constant across the entire switch branch of the action space: it does not
discriminate *between* switches, it prices *switching*.

**The implied preference, in the reward's own units:**

| | expected charge per decision |
|---|---|
| choose a voluntary switch | `−0.111` |
| choose a move | `−0.010` |
| **differential against switching** | **`−0.101`** ( = `−0.00168` of win probability) |

**And the tax fires on isolated turns, not on stalls.** Of the 34,235 charges, **24,277 (70.9%) are
at `n_t = 0`** — the first no-progress turn in a streak — and only 1,521 (4.4%) at `n ≥ 4`. A term
whose counter is named `turns_since_progress` and whose obs scalar saturates at 10 spends seven
charges in ten on windows with no streak at all.

Game-phase: taxed decisions sit at turn p50 **12** (p10 3 / p90 36) against the population's p50 13
(p10 3 / p90 39) — very slightly earlier, not phase-concentrated.

---

## 3. ALIGNMENT — the definitions, stated before computing

φ(sₜ) is the win-prob head's `P(win)` **recorded at decision time** in `states.npz` (no forward
pass; the head's own read of the state it acted in).

- **`d_out(t) = φ(s_{t+1}) − φ(s_t)`** — the head's verdict on the window the tax charges. The
  charge for `a_t` lands in `r_t` and is decided by the fold producing `obs_{t+1}`, so `d_out` is
  the head's pricing of *exactly* that window.
- **`d_in(t) = φ(s_t) − φ(s_{t−1})`** — was the head already declining coming in.
- **ε = 0.0025**, which is not arbitrary: it is **the tax's own size in win-prob units**
  (`0.15 / 60`, the terminal reward spanning `−30…+30`). So the binary asks the mission's question
  literally — *did the head price this window at least as costly as the rule charges for it?*
- **ALIGNED-consequence** := `d_out < −ε`.  **ALIGNED-anticipation** := `d_in < −ε` — the
  registered prediction's literal reading ("already depressed/declining **before** the tax fires").
- **OVER-TAXED** := taxed AND `d_out ≥ 0`.

**Why the control is not optional.** φ is approximately a martingale, so *any* decision shows a
decline about half the time. It is also **optimistic**: the mean φ at a battle's first recorded
decision is **0.745** against a realized 49.6% win rate, and the mean `d_out` over all 147,204
decisions is **−0.0070** — a calibration drift that shifts *every* arm negative. A raw "X% of taxed
decisions decline" measures that drift. Every claim below is carried by a **contrast**, CIs
cluster-bootstrapped over **battles** (4,000 resamples).

### 3.1 The headline, against the ≥70% bar

| | ALIGNED-consequence | ALIGNED-anticipation |
|---|---|---|
| **TAXED** (n = 34,235) | **0.457** [0.451, 0.464] | **0.558** [0.551, 0.564] |
| **UNTAXED control base rate** (n = 110,045) | **0.447** [0.442, 0.451] | **0.418** [0.414, 0.422] |

Sensitivity — the ordering does not depend on ε: consequence reads 0.515 / 0.457 / 0.364 at
ε = 0 / 0.0025 / 0.01, against a base rate of 0.486 / 0.447 / 0.384.

> **The registered ≥70% bar is REFUTED on both readings.** The consequence reading is
> **45.7% against a 44.7% base rate** — a +1.0pp excess, i.e. the tax fires essentially at chance
> with respect to whether the head prices the window as costly. The anticipation reading (the
> registered one) is **55.8% against 41.8%** — a real +14pp excess, but still 14 points short of
> the bar, and §3.3 shows most of it is a selection effect.

### 3.2 Stratified by action kind — MANDATORY, and it is where the sign flips

The taxed and untaxed sets have wildly different action composition (73% of switches vs 6.7% of
moves), and switches and moves have different φ dynamics for reasons that have nothing to do with
the tax. An unstratified contrast measures the composition.

| kind | arm | n | mean `d_out` | in reward units | ALIGNED-conseq |
|---|---|---|---|---|---|
| **move** | TAXED | 7,175 | **−0.0108** [−0.0125, −0.0092] | −0.650 | 0.457 |
| | UNTAXED | 97,786 | −0.0080 [−0.0086, −0.0074] | −0.481 | 0.442 |
| **switch** | TAXED | 14,619 | **−0.0014** [−0.0025, −0.0003] | −0.082 | 0.418 |
| | UNTAXED | 5,103 | **−0.0116** [−0.0142, −0.0091] | −0.698 | 0.536 |
| **forced sw.** | TAXED | 12,432 | −0.0054 [−0.0063, −0.0045] | −0.324 | 0.504 |
| | UNTAXED | 6,966 | −0.0024 [−0.0047, −0.0001] | −0.147 | 0.443 |

Contrasts (taxed − untaxed), one CI each:

| kind | Δ mean `d_out` | Δ ALIGNED-conseq | Δ ALIGNED-antic |
|---|---|---|---|
| move | **−0.0028** [−0.0046, −0.0011] **SIG** | +0.0147 [−0.0039, +0.0334] n.s. | +0.0196 [+0.0032, +0.0354] SIG |
| **switch** | **+0.0103** [+0.0076, +0.0131] **SIG — WRONG SIGN** | **−0.118** [−0.137, −0.099] **SIG — WRONG SIGN** | +0.0347 [+0.0176, +0.0518] SIG |
| forced sw. | −0.0030 [−0.0054, −0.0004] SIG | +0.0611 [+0.0460, +0.0758] SIG | +0.0780 [+0.0633, +0.0923] SIG |

> **On MOVES the tax points the right way** (taxed moves cost 0.28pp more win probability than
> untaxed moves — though the *rate* is not elevated, so the difference lives in the tail).
> **On SWITCHES — 43% of all charges — the tax points the WRONG way.** Among voluntary switches
> the ones it charges are worth **+1.03pp MORE** win probability than the ones it exempts, and the
> alignment rate is **11.8 points BELOW** the untaxed switches. Within the switch branch the rule's
> discrimination is inverted.

### 3.3 The matched control — the primary read, and it is NULL

For every taxed decision, the control is drawn from **the same battle, within ±3 game turns, of the
same action kind**. 14,149 pairs.

| paired difference (taxed − its own matched controls) | |
|---|---|
| mean `d_out` | **−0.00106 [−0.00316, +0.00100]** — **NULL** |
| ALIGNED-consequence | **+0.0131 [−0.0018, +0.0286]** — **NULL** |
| ALIGNED-anticipation | +0.0350 [+0.0220, +0.0476] — SIG, but 3.5pp |

> **Once you hold the battle, the game phase and the action kind fixed, the windows the tax charges
> are priced by the head EXACTLY like the windows it does not.** The +14pp anticipation excess of
> §3.1 shrinks to **+3.5pp**; nearly all of it was the tax firing disproportionately at moments
> where we were already switching because we were already behind. The tax carries **no
> win-probability information** beyond what "it is turn T of this battle and this is a switch"
> already says.

### 3.4 What the tax IMPLIES about switching vs what the head says

| | in win-prob units |
|---|---|
| what the tax expresses (differential charge against switching, §2) | **−0.00168** |
| what the head measures (mean `d_out` on switches − on moves) | **+0.00419** [+0.00297, +0.00539] **SIG** |

> Opposite signs, and the disagreement is ~2.5× the tax's own size. *(Descriptive, not causal:
> switches and attacks are chosen in different states, so this contrasts contexts as well as
> actions. But the tax is applied to the same selection, so it is the right comparison for "does
> the hand-coded rule agree with the learned signal about which action kind is costly".)*

---

## 4. THE OVER-TAX SET — 48.5%, and it is enormous

**16,593 of 34,235 taxed decisions (48.5%) have `d_out ≥ 0`** — the head sees no outcome cost at
all. 14,841 (43.4%) are *strictly rising* past ε. Across the whole over-tax set the mean `d_out` is
**+0.0327**: the average over-taxed decision **gained 3.3 percentage points of win probability**
while being charged `−0.15`.

Since exactly one BIAS term is live, "which terms produce them" resolves to *which behaviours*:

| kind | n in over-tax set | share |
|---|---|---|
| voluntary switch | 8,028 | 48.4% |
| forced switch | 5,485 | 33.1% |
| move | 3,074 | 18.5% |

**Named example class — Protect**, the archetype of what the rule was written for and its single
most-charged move (1,782 charges):

| Protect | n | mean `d_out` | ALIGNED-conseq | over-taxed |
|---|---|---|---|---|
| TAXED | 1,782 | **−0.00047** [−0.00287, +0.00191] | 0.354 | **0.573** |
| UNTAXED | 1,384 | −0.00123 [−0.00524, +0.00293] | 0.423 | 0.540 |
| contrast | | +0.00076 [−0.00397, +0.00570] **n.s.** | | |

> A taxed Protect costs the head **nothing measurable** — its CI straddles zero — and 57% of taxed
> Protects *gain* win probability. Other frequent over-taxed moves: `recover` (297), `softboiled`
> (244), `substitute` (133), `earthquake` (126), `leechseed` (123).

**The structural over-tax: 12,432 charges on ZERO-AGENCY decisions.** 36% of all charges land on
forced switches — post-faint replacements, where the agent's only legal actions are switches and
therefore *no available action can produce progress by the clock's own definition*. The charge is
near-unconditional on the choice made. `ProgressClock`'s own comment
(`progress_clock.py:160-163`) states the intent — *"Forced-switch / post-faint replacement: only
switches were legal → the clock sits out"* — but the guard reads `curr_ctx.phase`, i.e. the
**upcoming** request (§1), so it spends its 19,503 exemptions on the *preceding* window (the turn
in which our mon was KO'd — a full-agency decision) and charges the replacement it was written to
exempt. **The exemption is misdirected by exactly one window.**

---

## 5. THE UNDER-TAX HUNT — and it mostly EXONERATES the exemptions

The reverse question: stall-shaped windows the rule exempts, where the head prices real cost.

| exempt class | n | mean `d_out` | in reward units | vs TAXED |
|---|---|---|---|---|
| **SITOUT** (the misdirected forced-switch exemption) | 19,503 | **−0.0509** [−0.0524, −0.0493] | **−3.054** | **−0.0461** [−0.0478, −0.0443] SIG |
| **FROZEN — exogenous denial** (miss / cant / blocked) | 2,060 | **−0.0486** [−0.0535, −0.0441] | −2.917 | — |
| TRAPPED (no switch legal) | 4,293 | −0.0064 [−0.0087, −0.0040] | −0.384 | −0.0016 [−0.0040, +0.0008] **n.s.** |
| **FROZEN — heal inside `HEAL_FREEZE_GRACE`** | 1,186 | **+0.0162** [+0.0117, +0.0208] | **+0.971** | **+0.0210** [+0.0164, +0.0257] SIG |

Read honestly, **three of the four exemptions are correct and the mission's prior for this arm does
not hold:**

- **The heal-war grace is CORRECT, and it is the cell the mission named.** A productive heal inside
  the 2-window grace **gains 1.6pp of win probability** — 70% of them have `d_out ≥ 0` and the
  alignment rate is 24.5%, the lowest of any arm. Forgiving them is right; charging them would be
  the over-tax. *(The hunt was for untaxed heal-loops the head prices as costly; the head prices
  them as **profitable**.)*
- **The exogenous-denial freeze is correct in the sense that matters.** Those windows are
  genuinely ruinous (−4.9pp) — but their cost is a miss, a full-paralysis or a Protect, i.e. dice,
  not policy. A reward that charged them would be taxing luck. This is a case where the head's
  signal and the rule *should* disagree.
- **The trapped exemption is a clean null** — trapped no-ops cost statistically the same as taxed
  ones, and by construction the agent had no alternative.
- **Only SITOUT is a genuine defect**, and it is the §4 off-by-one seen from the other side: the
  costliest class in the entire corpus (−5.1pp, 10× the taxed set, 13.2% of all decisions) is
  exempted, while the zero-agency class it was meant to exempt is charged. The remedy is not "tax
  the KO window" — `pbrs_material` already prices a faint, policy-invariantly — it is that the
  exemption is pointed at the wrong window.

---

## 6. SPLITS

**By outcome (taxed decisions).** LOSS mean `d_out` −0.0076, ALIGNED 0.491, over-taxed 0.472;
WIN mean `d_out` −0.0009, ALIGNED 0.410, over-taxed 0.503. The alignment is better in losses and
essentially absent in wins — the tax is a better signal when the game is already going badly.

**At the BATTLE level the tax DOES track losing — and that is the trap.**

| | wins (2,496) | losses (2,539) | difference |
|---|---|---|---|
| taxed **rate** | 0.2025 [0.1990, 0.2061] | 0.2457 [0.2425, 0.2488] | **+0.0432** [+0.0383, +0.0480] SIG |
| taxed **count** | 5.71 [5.51, 5.91] | 7.87 [7.64, 8.12] | **+2.17** [+1.86, +2.48] SIG |

Losses are also longer (31.8 vs 26.7 decisions), which explains part of the count gap but not the
rate gap. So: **the taxed behaviour predicts losing at the battle level while carrying no
per-decision cost under the matched control (§3.3).** That is the classic marker-vs-cause split —
the tax fires when you are behind (you switch defensively, you cannot make progress), so it
functions as *a penalty on being behind*, which `pbrs_material` already delivers and delivers
policy-invariantly.

**By opponent class** (taxed, ALIGNED-conseq): `sentinel` 0.481 · `staller` 0.461 · `heuristic`
0.442 · `heuristic2` 0.439 · `aggressive` 0.432 · `setup_sweep` 0.424 · `random` **0.326**. Alignment
is worst against a random opponent — the one whose losses are least attributable to our stalling.

**By run**: 0.411–0.491, no outlier; `ai_v9_62_R2PLAIN` taxed windows cost nothing at all
(+0.0002 [−0.0019, +0.0022]). The finding is a property of the reward function, not of one run.

---

## 7. THE REGISTERED PREDICTIONS, SCORED

| prediction | verdict |
|---|---|
| **ALIGNMENT ≥70% on the stall-tax class** | **REFUTED.** Consequence **45.7%** vs a **44.7%** control base rate (+1.0pp); anticipation **55.8%** vs **41.8%** (+14pp) — neither clears 70, and the matched control cuts the anticipation excess to **+3.5pp** and the consequence excess to a **null**. Probe L's 96% whiff prior did not transfer: there the head was ranking *actions* on a class the model provably mis-plays; here the tax is a *state-window* charge that fires on a quarter of all decisions. |
| **A NONZERO over-tax set exists** | **HELD, and far larger than "nonzero": 48.5% of all charges** (16,593), mean `d_out` **+0.0327**. Produced by the single live term `no_progress_tax`; by behaviour it is 48% voluntary switches, 33% zero-agency forced switches, 19% moves — Protect the most-charged single move at 1,782, with a cost CI straddling zero. |
| **The under-tax set — no prediction** | **The finding is that three of four exemptions are RIGHT.** The heal-war grace protects turns worth **+1.6pp**; the denial freeze protects turns whose cost is dice; the trapped exemption is a null. The one real defect is **SITOUT** — the costliest class in the corpus (−5.1pp, 13.2% of decisions) exempted by a one-window misdirection that simultaneously charges 12,432 zero-agency replacements. |

## The selected reading

**The hand-coded tax and the learned outcome signal are not aligned, not anti-aligned, but
LARGELY ORTHOGONAL — and where they do interact on the tax's largest cell, the sign is inverted.**
Under the matched control the tax carries no win-probability information at all. It is not
principally an anti-stall term in practice: it is a **flat 10:1 tax on switching over attacking**
(`−0.101` reward units per decision, versus a head that rates switching `+0.0042` of win
probability *better*), it spends 36% of its charges on decisions with no alternative, 71% of its
charges on windows with no streak, and 48.5% of them on windows the head scores as *gaining*.

Two of its findings are actionable independently of the alignment verdict, because they are defects
against the rule's **own stated intent** rather than against the head:

1. **The forced-switch sit-out is off by one window** (`curr_ctx` is the upcoming request). Its
   comment says post-faint replacements sit out; they are charged 63.9% of the time, and the
   exemption lands on the KO window instead.
2. **A switch can never satisfy `_is_progress`**, so the term prices an action *kind* rather than
   discriminating within it — a consequence of the predicate's construction, not of any tuning.

Neither requires a new measurement to act on. The alignment result says something narrower and
worth stating plainly: **the win-prob head is not a ready-made replacement teacher for this tax**,
because at the state level it does not know anything about these windows that the battle-and-phase
context does not already say. That is the opposite of Probe L's action-ranking result, and the
difference is the level: Probe L composed the head with a *simulator* to get a per-action ranking;
here the head is read as a state scalar, and a state scalar cannot express "this action was the
wheel-spin".

## Caveats and cuts

1. **The one-ply tier was CUT**, as the mission's budget rule directs. Every read here is the
   state-level φ trajectory. A decision-level ranking (re-roll each legal action, score the
   successor) could still find that the head dispreferres the taxed *action* even where it prices
   the *window* at zero — that would sharpen §4 and cannot be ruled out by this probe.
2. **`d_out` is the head's CLAIM, not realized outcome.** Nothing was confirmed by rollouts.
3. **The head is systematically optimistic** (mean φ at first decision 0.745 vs a 49.6% realized
   win rate; mean `d_out` −0.0070 over the corpus). This shifts every arm negative, which is why no
   claim rests on an absolute rate. It is also an independent calibration finding.
4. **The last decision of every battle is excluded** (no `φ(s_{t+1})`), so the largest φ moves — the
   terminal ones — are outside every arm. This is symmetric between wins and losses.
5. **CAP rows (2,924, 2.0%) are excluded** — at `n = 10` an increment is invisible, so they cannot
   be told apart from a freeze. Sensitivity is favourable: their mean `d_out` is −0.0010 and their
   alignment 0.285, so folding them in as taxed would *lower* alignment further.
6. **§3.4's switch-vs-move contrast is descriptive, not causal** — different states, not a treatment
   effect. Flagged in place.
7. **Eval traces only** — snapshot-vs-opponent games, not the training distribution. Opponent
   coverage is broad (7 classes, `sentinel_*` 36% of battles) but the ladder distribution is absent.
8. **The tax is reconstructed, not logged.** The equivalence is proved from source (§1) rather than
   sampled against a live reward manager; a bridge-replay cross-check of `last_penalty` was not run.
   The four independent gates (lattice, transitions, opening value, sit-out alignment 0/1513, plus
   phase agreement 147,204/147,204) are what stands in for it.
9. **`ai_v9_29_rev1_0823` supplies 52% of the decisions.** The by-run split shows no outlier.

## Reproduce

The two producer scripts ship beside this file. CPU-only, no model, ~4 minutes total.

```bash
export PYTHONPATH=$PYTHONPATH:src
cd designs/research_state/measurements
for r in ai_v9_29_rev1_0823 ai_v9_58_R2CTRL_0827 ai_v9_59_R2ACTION_0827 ai_v9_60_R2TOPK_0827 \
         ai_v9_61_R2KL_0827 ai_v9_62_R2PLAIN_0827 ai_v9_70_R3ACTION_0828; do
  nice -n 15 python bias_tax_head_alignment_census.py --run models/$r --out /tmp/probeM/census_$r.jsonl
done
nice -n 15 python bias_tax_head_alignment_analyze.py --census '/tmp/probeM/census_*.jsonl' \
    --out bias_tax_head_alignment_2026-08-29.json --boot 4000
```

The census **fails loudly** rather than producing numbers if the clock column drifts, if a
`has_state` gap breaks the counter dynamics, or if the fold→window alignment stops holding.
