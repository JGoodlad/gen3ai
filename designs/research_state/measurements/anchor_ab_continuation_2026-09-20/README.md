# THE A/B CONTINUATION, READ EXTERNALLY — `SmallRL` and `SyntheticRLV2`, greedy, BOTH team sets, 2026-09-20

**The owner's question, verbatim:** *"Can you get the A/B continuation against metamon teams to see
if we can see the improvement externally."*

**The answer, in one sentence:**

> 🚨 **No — not at the size the internal meters say.** Over **8,400 anchor battles** (1,200 per arm
> per `SmallRL` team set across 12 team seeds, plus a 1,200-game `SyntheticRLV2` era gate), the
> continuation `ai_v13_09_wcont` is **+0.039 [−0.000, +0.079] on Metamon's own teams**, **+0.020
> [−0.019, +0.058] on ours** and **+0.048 [−0.019, +0.113]** against `SyntheticRLV2`, against its
> frozen parent — **NOT DETECTED in any cell, at the registered n = 400 and again at the extended
> n = 1200**. The instrument is not blind while it says so: on the very same games it resolves
> **C − F at +0.120 and +0.098** and **F − W at −0.081 and −0.078**, every one of those four CIs
> clear of zero. The **ORDER** the internal meters give (continuation > frozen parent > fold path)
> reproduces externally on both `SmallRL` team sets; the **MAGNITUDE** does not — a post-hoc
> combination of all three cells puts the external edge at **+0.032 [+0.007, +0.058]**, roughly
> **one fifth** of the untaught meter's +15.50 pp. The **`+0.140`** from the n = 100 read was seed
> luck: the same contrast at 12× the games is **+0.039**.

**And the row that DID reproduce:** the era-1 fold path `ai_v13_08_fold1_cont` is **0.08 BELOW its
own frozen parent** on both team sets, CIs clear of zero — the internal read's *"the fold did not
merely fail to pay, it COST"* confirmed by an agent nobody here trained.

Pre-registered in [`PREDICTION.md`](PREDICTION.md), committed **`0f110483`** before the first
battle, with **AMENDMENT 1** (`0920f197`) fixing the unconditional extension to n = 1200 before any
pooled contrast was computed. **Branch reached: (c)** — see §4.

---

## 0. THE ARMS, resolved

| arm | what it is | `.zip` | step | rung | loader |
|---|---|---|---:|---|---|
| **W** | the FROZEN PARENT — the 75M win-prob flywheel arm | `models/ai_v13_02_flywheel_winprob/final_model.zip` | **75,005,952** | `explicit_zip` | `bare` |
| **C** | **the CONTINUATION** `ai_v13_09_wcont` — W + 12M, **no teachers**, frozen dose | `models/ai_v13_09_wcont/final_model.zip` | **87,097,344** | `explicit_zip` | `bare` |
| **F** | the era-1 **FOLD PATH** `ai_v13_08_fold1_cont` — same parent, same dose, same endpoint | `models/ai_v13_08_fold1_cont/final_model.zip` | **87,097,344** | `explicit_zip` | `bare` |

Steps read from each zip's own SB3 `data` blob and **confirmed again by the tool on every row it
wrote**. C and F end on the same step to the step, which is what makes C − F a matched contrast.

## 1. WHAT WAS RUN

`python -m main.anchors`, the tool of record. Greedy-vs-greedy (SOP rule 2), verified **per decision
on both sides**; Metamon `@0a00a759`, `SmallRL@ckpt40` / `SyntheticRLV2@ckpt48`, Showdown pin
`e0551883f`; CPU only, `OMP_NUM_THREADS=1`, `nice 15`, each cell on its own 9500–9599 port started
and stopped by its own PID. **:8000 and :8001 were never touched.** Run from the worktree
`gen3ai-wt/anchor_ab` (SOP: a campaign is run from its own worktree) with absolute model paths into
the main checkout's `models/`; **nothing was written under `models/`**.

Each cell is **N × 100-game sub-cells at distinct `--team-seed`s**, pooled. 🚨 **The seeds are
different from the 2026-09-20 read**, which ran at the tool's default `20260914`:

* registered: **20260919, 20260929, 20260939, 20260949**
* AMENDMENT 1 extension: **20260959, 20260969, 20260979, 20260989, 20260999, 20261009, 20261019, 20261029**
* the `SyntheticRLV2` era gate takes the **first four** (20260919 – 20260949) — 400 games per arm

**The same seeds run for all three arms**, and the post-hoc alignment check confirms the arms faced
the *same draw in the same slot*: **every shared (seed, half, index) slot carried the same
`our_team`** — 1200 of 1200 in each of the six `SmallRL` cross-arm comparisons and 400 of 400 in
each of the three `SyntheticRLV2` ones. That licenses the paired descriptor in §3.3; the registered
primary stays the unpaired Newcombe interval.

**Cost, for the SOP's table.** `SmallRL` ran at **~2 min per 100-game cell** with two lanes on a
quiet box (the SOP's ~15 min was measured at load 25–31 with a live training arm); `SyntheticRLV2`
at **~13 min**, the 200M-parameter build dominating. The whole campaign — 84 sub-cells, 8,400
battles — took **under 3 hours of wall clock**.

## 2. THE RESULTS TABLE — arm × cell × win rate

Win rate is OURS; **ties count in the denominator and not the numerator**. Role-balanced inside
every sub-cell (50 games we challenge, 50 they challenge).

### 2.1 `metamon:SmallRL`, greedy, **AWAY** — Metamon's own 20 `competitive` teams (the owner's "metamon teams")

| arm | n | W / L / T | **win rate** | Wilson 95 % | ours-challenge / peer-challenge | mean turns |
|---|---:|---|---:|---|---|---:|
| **C** — the continuation | 1200 | 711 / 489 / 0 | **0.5925** | **[0.5644, 0.6200]** | 0.595 / 0.590 | 52.95 |
| **W** — the frozen parent | 1200 | 664 / 535 / 1 | **0.5533** | [0.5251, 0.5812] | 0.540 / 0.567 | 51.40 |
| **F** — the fold path | 1200 | 567 / 631 / 2 | **0.4725** | [0.4444, 0.5008] | 0.473 / 0.472 | 57.84 |

### 2.2 `metamon:SmallRL`, greedy, **HOME** — our 719-team pool

| arm | n | W / L / T | **win rate** | Wilson 95 % | ours-challenge / peer-challenge | mean turns |
|---|---:|---|---:|---|---|---:|
| **C** — the continuation | 1200 | 773 / 427 / 0 | **0.6442** | **[0.6167, 0.6708]** | 0.655 / 0.633 | 43.48 |
| **W** — the frozen parent | 1200 | 749 / 448 / 3 | **0.6242** | [0.5964, 0.6511] | 0.615 / 0.633 | 43.24 |
| **F** — the fold path | 1200 | 655 / 539 / 6 | **0.5458** | [0.5176, 0.5738] | 0.522 / 0.570 | 43.91 |

### 2.3 The twelve sub-cell rates per arm — the run-to-run spread, printed rather than hidden

| cell | arm | the twelve 100-game rates |
|---|---|---|
| away | C | 0.62 0.45 0.56 0.61 0.61 0.69 0.56 0.59 0.60 0.64 0.56 0.62 |
| away | W | 0.51 0.53 0.55 0.59 0.50 0.51 0.56 0.53 0.60 0.58 0.50 0.68 |
| away | F | 0.43 0.59 0.47 0.49 0.34 0.49 0.41 0.36 0.55 0.57 0.45 0.52 |
| home | C | 0.60 0.67 0.68 0.66 0.66 0.69 0.65 0.66 0.63 0.56 0.66 0.61 |
| home | W | 0.61 0.67 0.54 0.65 0.62 0.64 0.70 0.64 0.56 0.62 0.65 0.59 |
| home | F | 0.50 0.61 0.53 0.55 0.56 0.49 0.62 0.53 0.58 0.42 0.58 0.58 |

🚨 **A single 100-game sub-cell of this cell ranges over 0.34 wide within one arm** (`F` away:
0.34 → 0.59). That is the whole content of the SOP's warning that 100 games resolve ~±0.10, and it
is why the n = 100 banked numbers below moved so much.

## 3. THE THREE CONTRASTS, per cell

**The registered bar: the 95 % CI on the DIFFERENCE excludes zero.** A CI that straddles is **NOT
DETECTED** — never "equivalent", never "no difference". The 0.090 run-level floor is the stricter
secondary, reported beside it.

### 3.1 🚨 THE REGISTERED READ — n = 400 per arm per cell, the four registered seeds ALONE

| cell | contrast | Δ | Newcombe 95 % | **REGISTERED VERDICT** | vs the 0.090 floor |
|---|---|---:|---|---|---|
| away | **C − W** | **+0.0150** | [−0.0537, +0.0835] | **NOT DETECTED** | WITHIN FLOOR |
| away | C − F | +0.0650 | [−0.0042, +0.1333] | NOT DETECTED | WITHIN FLOOR |
| away | F − W | −0.0500 | [−0.1185, +0.0192] | NOT DETECTED | WITHIN FLOOR |
| home | **C − W** | **+0.0350** | [−0.0316, +0.1012] | **NOT DETECTED** | WITHIN FLOOR |
| home | C − F | +0.1050 | [+0.0371, +0.1715] | **DETECTED** | WITHIN FLOOR |
| home | F − W | −0.0700 | [−0.1374, −0.0017] | **DETECTED** | WITHIN FLOOR |

### 3.2 THE EXTENDED READ — n = 1200 per arm per cell, all twelve seeds (AMENDMENT 1)

| cell | contrast | Δ | Newcombe 95 % | verdict | vs the 0.090 floor |
|---|---|---:|---|---|---|
| **away** | **C − W** | **+0.0392** | **[−0.0004, +0.0786]** | **NOT DETECTED** *(by 0.0004)* | WITHIN FLOOR |
| **away** | **C − F** | **+0.1200** | **[+0.0802, +0.1593]** | **DETECTED** | WITHIN FLOOR (CI covers 0.090) |
| **away** | **F − W** | **−0.0808** | **[−0.1205, −0.0409]** | **DETECTED** | WITHIN FLOOR (\|Δ\| < 0.090) |
| **home** | **C − W** | **+0.0200** | [−0.0185, +0.0584] | **NOT DETECTED** | WITHIN FLOOR |
| **home** | **C − F** | **+0.0983** | [+0.0591, +0.1371] | **DETECTED** | WITHIN FLOOR (CI covers 0.090) |
| **home** | **F − W** | **−0.0783** | [−0.1174, −0.0389] | **DETECTED** | WITHIN FLOOR (\|Δ\| < 0.090) |

**The registered and the extended read AGREE on every row's direction, and on five of six verdicts.**
The one that moves is `C − F` on away — NOT DETECTED at n = 400, DETECTED at n = 1200 — which is
the extension doing exactly what it was registered to do.

⚠️ **NOTHING clears the 0.090 secondary floor, in either direction.** For `C − F` the point estimate
is above it but the CI still covers it (clause (b) fails); for `F − W` the point estimate is below
it outright (clause (a) fails). That floor is a **BETWEEN-SEED** quantity and three of these six
rows are within-lineage contrasts, so it is the wrong null in kind for them — but it is the only
floor measured on this cell and it is reported unedited.

### 3.3 ⚠️ THE PAIRED DESCRIPTOR DISAGREES ON ONE ROW, AND THE REGISTERED VERDICT STANDS

The arms faced the **same team in the same slot, 1200 of 1200 times**, so a paired bootstrap over
matched game slots is legitimate and strictly tighter. It was registered as a **DESCRIPTOR ONLY**,
in advance, precisely so that a headline could not come to depend on it:

| cell | contrast | unpaired Newcombe (REGISTERED) | paired bootstrap (descriptor) |
|---|---|---|---|
| **away** | **C − W** | **[−0.0004, +0.0786] NOT DETECTED** | **[+0.0025, +0.0758] — excludes zero** |
| away | C − F | [+0.0802, +0.1593] | [+0.0817, +0.1575] |
| away | F − W | [−0.1205, −0.0409] | [−0.1175, −0.0433] |
| home | C − W | [−0.0185, +0.0584] NOT DETECTED | [−0.0150, +0.0542] |
| home | C − F | [+0.0591, +0.1371] | [+0.0608, +0.1350] |
| home | F − W | [−0.1174, −0.0389] | [−0.1150, −0.0425] |

🚨 **The away `C − W` row sits exactly on the line: the registered interval misses zero by 0.0004 on
the wrong side, the paired one clears it by 0.0025.** The honest statement is the one that was
registered — **NOT DETECTED** — and the second honest statement is that this is the weakest possible
version of a null: the continuation is **very probably slightly ahead of its parent on Metamon's own
teams, by something near +0.04**, and no reading of this data supports +0.14.

## 4. 🚨 THE BRANCH REACHED — **(c)**, AT BOTH SAMPLE SIZES

`PREDICTION.md` §4 filed three branches. **(c) — "NOT DETECTED on either team set at n = 400" — is
what happened**, and it was the branch I gave the *lowest* probability (0.15). It also survives the
3× larger extension, which the registration did not require it to, **and it repeats on a third cell
the branches did not cover**: `SyntheticRLV2` away reads +0.0475 [−0.0190, +0.1134], NOT DETECTED
(§6).

| branch | filed P | met? |
|---|---:|---|
| (a) C − W detected on BOTH team sets | 0.55 | ❌ |
| (b) detected on exactly ONE | 0.30 | ❌ |
| **(c) NOT DETECTED on either at n = 400** | **0.15** | ✅ **— and again at n = 1200** |

The three side bets: **P(C − F detected on away) = 0.75** → ✅ at n = 1200, ❌ at the registered
n = 400. **P(F − W detected on away) = 0.20** → ✅ at n = 1200 (in the NEGATIVE direction).
**P(the away C − W point estimate lands in [+0.05, +0.25]) = 0.70** → ❌, it is **+0.039**.

### 4.1 What branch (c) says, written before the data and quoted unchanged

> *"🚨 The internal +15.50 pp did NOT transfer to an external opponent at a resolvable size. That is
> a finding about the METERS, not only about the arm … It would NOT license 'the continuation did
> not improve' — `SmallRL` is one opponent at one skill level and a ceiling/floor effect against it
> is a live alternative — but it would demote every internal number of this era from 'competence' to
> 'competence **against our own lineage**' until an external read says otherwise."*

**That stands, with one qualification the data adds and the registration could not anticipate: the
instrument demonstrably had the power.** On the same 7,200 games it put four CIs clear of zero at
effect sizes of 0.078–0.120. A blindness explanation is therefore not available for the C − W row:
the external anchor can see an eight-point difference between our own arms, and the continuation's
edge over its parent is simply not eight points. **The internal untaught meter says +15.50 pp;
the external anchor says +3.9 pp on their teams and +2.0 pp on ours, neither resolvable.** The two
instruments are measuring different things, and the burden now sits on the claim that the internal
one is measuring competence.

### 4.2 The one thing that DID reproduce externally: the ORDER, and the fold path's cost

**C > W > F on both team sets, with the two F rows resolved.** The era-1 fold path is **below its
own frozen parent** by −0.081 [−0.121, −0.041] on Metamon's teams and −0.078 [−0.117, −0.039] on
ours. The internal read's second headline — *"against the right baseline the era-1 fold did not
merely fail to pay, it COST"* — is **confirmed by an agent nobody here trained, on both team sets,
at 1,200 games each.** That is the strongest externally-validated statement this read produces, and
it is not the one the read was commissioned to test.

## 5. HAZARDS — each one a finding

### 5.1 🚨 H-A — `regime_verified` READ FALSE ON 24 OF 72 SUB-CELLS, AND THE REGIME WAS MATCHED ON ALL 72

Hazard W-J from the 2026-09-20 read is **not a one-off; it fires on a third of sub-cells.**

| | |
|---|---:|
| sub-cells run (SmallRL) | **72** |
| composite `regime_verified == true` | **48** |
| composite `regime_verified == false` | **🚨 24** |
| distinct per-decision `argmax_match_rate` values across ALL 144 halves | **{1.0000}** |
| sub-cells with `status != "OK"` | **0** |

The composite flag is an **AND over both halves' `regime_check_ok`**, and a peer's flag goes false
if the peer process raises **anywhere** — including after its last decision, in results aggregation,
which is where Metamon's `RecursionError` on a forfeit desync (H5) lands. **Every half of every
sub-cell reported `argmax_match_rate` exactly 1.0000**, so the regime was matched on all 7,200
games. **Reading the composite flag as "the regime failed" would have voided a third of this
campaign.** Both numbers are recorded per sub-cell in `out/*/summary.json` and in
`results_n1200.json`; the tool would benefit from separating "the peer crashed after its work" from
"the regime was not matched", and that is a backlog row, not a defect in this read.

### 5.2 ⚠️ H-B — I EDITED THE LANE SCRIPT WHILE BASH WAS STILL READING IT

`bash` reads a script incrementally by **byte offset**, so editing a running script moves the ground
under the interpreter. Patching `run_lane.sh` to add the extension jobsets while the registered
lanes were mid-flight ended both lanes in `syntax error near unexpected token ')'` **after** the
`for` loop — the loop itself had already been parsed whole, so **all 24 registered sub-cells ran and
completed, and none is affected** (verified: 24/24 `summary.json`, all `status: OK`). It was luck,
not design. **The fix, applied for the extension: run from a FROZEN COPY** outside the edited tree
(`/home/goodlad/.claude/jobs/anchor_ab/run_lane_frozen.sh`). The rule generalises — never edit a
shell script that a live process is executing.

### 5.3 The forfeit / timeout row — the registered INCONCLUSIVE rule does not fire

| | `SmallRL` total |
|---|---:|
| games | **7,200** |
| games hitting the 250-turn forfeit cap | **41 (0.57 %)** |
| ties | **12 (0.17 %)** |
| sub-cells FAILED or INCONCLUSIVE | **0** |
| `team_source_asymmetry` | **false on every sub-cell** |
| `n_defaults` / `n_redecides` | **0 / 0** everywhere |

🚨 **`main.anchors` has no "timeout" outcome by design** — a stalled series is a NAMED failure
(`no_progress` / `short_series`) that voids the sub-cell, so the registered *">25 % timeouts ⇒
INCONCLUSIVE"* rule reads on excluded sub-cells, of which there are **none**. The forfeit-at-cap
rate of **0.57 %** matches the SOP's banked 0.5 % exactly.

### 5.4 ⚠️ The n = 100 comparison, and what it costs to believe one

| cell (SmallRL greedy away) | banked n = 100 (seed 20260914) | **here, n = 1200** | moved by |
|---|---:|---:|---:|
| continuation `ai_v13_09_wcont` | 0.640 | **0.5925** | −0.048 |
| arm W | 0.500 | **0.5533** | +0.053 |
| fold path `ai_v13_08_fold1_cont` | 0.430 | **0.4725** | +0.043 |
| **C − W** | **+0.140** | **+0.039** | **−0.101** |
| **C − F** | **+0.210** | **+0.120** | **−0.090** |

Every arm moved toward the middle and **the headline contrast shrank by a factor of 3.6**. The
n = 100 read said so itself — it registered this row as DESCRIPTIVE with P(clears) = 0.15 and it did
not clear — so nothing here contradicts it. **It is, however, the cleanest demonstration available
that a 100-game anchor point estimate is not a quantity to carry forward**, and the SOP's tier-B
recommendation of 100 games should be read as *a detector of ~0.15-sized movement*, not as a
measurement of a ~0.05-sized one.

### 5.5 Team coverage

`distinct_our_teams` per 100-game sub-cell: **16–18 of 20** on away, **47–50 of 719** on home. The
away cell therefore samples its team space nearly exhaustively every sub-cell and the home cell does
not — one more reason the twelve home sub-cells are pooled rather than compared.

## 6. THE ERA GATE — `metamon:SyntheticRLV2`, greedy, away, n = 400 per arm

The tier-C cell, raised from 200 to 400 games per arm by AMENDMENT 1. `SyntheticRLV2` (ckpt 48,
200.9M params) is the reference that is **ahead of us**.

| arm | n | W / L / T | **win rate** | Wilson 95 % | four sub-cells | mean turns |
|---|---:|---|---:|---|---|---:|
| **C** — the continuation | 400 | 154 / 245 / 1 | **0.3850** | [0.3386, 0.4336] | 0.42 0.35 0.34 0.43 | 54.27 |
| **F** — the fold path | 400 | 139 / 261 / 0 | **0.3475** | [0.3025, 0.3954] | 0.28 0.39 0.28 0.44 | 56.61 |
| **W** — the frozen parent | 400 | 135 / 263 / 2 | **0.3375** | [0.2929, 0.3852] | 0.30 0.32 0.35 0.38 | 55.30 |

| contrast | Δ | Newcombe 95 % | verdict | paired descriptor |
|---|---:|---|---|---|
| **C − W** | **+0.0475** | [−0.0190, +0.1134] | **NOT DETECTED** | [−0.0125, +0.1075] |
| C − F | +0.0375 | [−0.0292, +0.1037] | NOT DETECTED | [−0.0225, +0.1000] |
| F − W | +0.0100 | [−0.0556, +0.0755] | NOT DETECTED | [−0.0525, +0.0700] |

🚨 **THE ERA BAR FAILS FOR ALL THREE ARMS, AND NOT NARROWLY.** The bar is *Wilson lower bound >
0.50*; here the **UPPER** bound is below 0.50 for every arm (0.4336 / 0.3954 / 0.3852). By the SOP's
re-registered fire rule clause (2) — *"the ABSOLUTE bar (upper bound < 0.50 fires) applies at ≥ 75M"*
— **this cell FIRES on all three arms.** `SyntheticRLV2` leads every arm of this lineage by 12–16
points at 87M steps.

⚠️ **The C − W direction agrees with `SmallRL` and the cell still cannot resolve it.** +0.0475 with a
half-width of 0.066 is the same story one opponent over: a small positive that n = 400 cannot
separate from zero. 🚨 **Do NOT read the C − F and F − W rows here as contradicting §3.2.** At
n = 400 this cell's half-width is 0.066, so its +0.038 and +0.010 are *consistent with* the
`SmallRL` cells' +0.10 and −0.08 as much as with zero; the honest summary is that the era gate has
no power at these sizes, not that the fold path recovers against a stronger opponent.

⚠️ **Descriptive, not a contrast:** the banked `ai_v12_02_winprob_critic` @75M read **0.420
[0.328, 0.518]** on this cell at n = 100 (SOP §4), against arm W's **0.3375 [0.293, 0.385]** here.
Different runs, different eras, different n — the intervals overlap and **nothing may be taken from
the comparison**; it is printed only so the number is not later discovered and read as a drop.

### 6.1 Integrity, this cell

400 games per arm, 1,200 total, **0 sub-cells failed**, **9 games at the 250-turn cap (0.75 %)**,
3 ties, `team_source_asymmetry` false throughout, `distinct_our_teams` 17–18 of 20 per sub-cell,
`n_defaults` / `n_redecides` 0 / 0. `argmax_match_rate` = **1.0000 on all 24 halves**; the composite
`regime_verified` read **true on 5 of 12** sub-cells — hazard H-A again, at a higher rate than on
`SmallRL`, and again with every per-decision instrument matched.

## 7. ⚠️ ONE POST-HOC DESCRIPTOR — the three cells combined

**NOT REGISTERED, and reported as a descriptor only.** An inverse-variance fixed-effects combination
of the three cells' `C − W` estimates (each cell's own binomial variance; the cells are independent
games but pool across two team sets and two opponents, which the SOP says to report separately):

| contrast | combined estimate | 95 % | cells |
|---|---:|---|---|
| **C − W** | **+0.0320** | **[+0.0065, +0.0575]** | +0.0392 away · +0.0200 home · +0.0475 synthv2 |
| C − F | +0.0984 | [+0.0727, +0.1241] | +0.1200 · +0.0983 · +0.0375 |
| F − W | −0.0658 | [−0.0916, −0.0401] | −0.0808 · −0.0783 · +0.0100 |

🚨 **Read this for its SIZE, never as a verdict.** Every per-cell registered verdict on `C − W` is
NOT DETECTED and that is the result. What the combination adds is a magnitude: **the continuation's
external edge over its frozen parent is on the order of +3 percentage points**, with an upper end
around +6. **The internal untaught meter's +15.50 pp is outside that interval by a factor of
roughly five.** That is the finding, and it is a finding about the relationship between the two
instruments rather than about either one alone.

## 8. THE READING

### 8.1 One paragraph, every hedge in place

> *Against `metamon:SmallRL` at a greedy-vs-greedy regime verified on every decision of all 7,200
> battles, the plain continuation `ai_v13_09_wcont` @87,097,344 is **+0.039 [−0.000, +0.079]** over
> its frozen parent `ai_v13_02_flywheel_winprob` @75,005,952 on Metamon's own `competitive` teams
> and **+0.020 [−0.019, +0.058]** on our 719-team pool — **NOT DETECTED on either, at the registered
> n = 400 and again at the extended n = 1200** — and **+0.048 [−0.019, +0.113]**, also NOT DETECTED,
> on a 1,200-game `SyntheticRLV2` era gate that **FIRES on all three arms** (every upper bound below
> 0.50). On the same games the anchor resolves the continuation **+0.120 [+0.080, +0.159]** and
> **+0.098 [+0.059, +0.137]** over the era-1 fold path, and the fold path **−0.081 [−0.121, −0.041]**
> and **−0.078 [−0.117, −0.039]** BELOW the frozen parent — so the instrument is not blind at this
> effect size. A post-hoc inverse-variance combination of the three cells puts the continuation's
> external edge at **+0.032 [+0.007, +0.058]**; the internal untaught meter's **+15.50 pp** for the
> same contrast is outside that interval by about 5×. The **ordering** the internal meters give
> (continuation > frozen parent > fold path) reproduces externally on both `SmallRL` team sets; the
> **magnitude** does not. Team seeds 20260919–20261029, different from the n = 100 read's default
> 20260914; 50 of 8,400 games hit the 250-turn forfeit cap (0.60 %); zero sub-cells failed;
> `regime_verified` read false on 31 of 84 sub-cells while all 168 halves reported
> `argmax_match_rate` = 1.0000 (hazard H-A).*

### 8.2 What this does NOT say

SOP §6 applies unchanged. Specific to this read:

* **It does not say the continuation did not improve.** It says the improvement against these
  opponents is on the order of +0.03, not +0.14, and that no single cell here resolves it. A
  ceiling/compression effect against one opponent at one skill level remains live — and the era gate
  firing on all three arms is exactly the shape a compression story would take.
* **It does not attribute the C − F gap to any single lever.** The fold path differs from the
  control in the distill LOSS, in `--distill-team-bias 0.4`, **and** in its opponent ecology. Three
  levers, one contrast; the three-lever split arm queued on 2026-09-20 is the instrument for that.
* **It does not put any arm on an absolute scale.** No bot cell and no anchor-vs-anchor cell was
  run; these are anchor-relative win rates, not ratings.
* **It says nothing about the open ladder.** Both clients speak to a pinned local server.
* **It cannot say the untaught meter is wrong.** It says the two instruments disagree by ~5× on the
  same contrast, and that the external one — which cannot drift with our training — is the smaller.

## 9. WHAT IS IN THIS DIRECTORY

| path | what |
|---|---|
| `PREDICTION.md` | the pre-registration (`0f110483`) + AMENDMENT 1 (`0920f197`), both committed before the cells they govern |
| `results_registered_n400.json` | the REGISTERED read — the four registered seeds alone |
| `results_n1200.json` | the EXTENDED read — all twelve seeds, plus the `SyntheticRLV2` cell |
| `out/<arm>_<opponent>_<teamset>_s<seed>/` | **84 sub-cells**: `summary.json`, `games.jsonl.gz` (gzipped — 11 MB raw), and both halves' `peer_*_report.json` (the per-decision regime instruments). The peer stdout logs, the Showdown logs and Metamon's own battle CSVs are NOT committed |
| `scripts/run_lane.sh` | the campaign harness — one lane of the job list |
| `scripts/analyze.py` | pools the sub-cells, puts Newcombe on each contrast, keeps the composite regime flag and the per-decision rates apart, and verifies the cross-arm draw alignment before the paired descriptor may speak |

🚨 **`scripts/run_lane.sh` is the post-hazard-H-B version** (it carries the extension jobsets). The
registered 24 sub-cells were run by the same file before those jobsets were appended; the jobs it
dispatched are byte-identical in either version, and the edit is what caused H-B.

## 10. §Ready-to-append ledger paragraph

*(for `designs/research_state/ledger.md`; **this job did not edit the ledger, `UNDERSTANDING.md` or
any design note** — it wrote only this directory)*

---

### 2026-09-20 · MEASUREMENT (MAJOR) · 🚨 **THE CONTINUATION'S +15.50 pp DOES NOT APPEAR EXTERNALLY: 8,400 anchor battles put `ai_v13_09_wcont` at +0.039 [−0.000, +0.079] over arm W on Metamon's own teams and +0.020 [−0.019, +0.058] on ours — NOT DETECTED on either, at the registered n = 400 and again at n = 1200 — while the SAME games resolve the era-1 fold path 0.08 BELOW its own frozen parent on both team sets. The ORDER reproduces; the MAGNITUDE is ~5× smaller. The n = 100 read's +0.140 was seed luck.**

Owner's question ("can you get the A/B continuation against metamon teams to see if we can see the
improvement externally"), answered through the tool of record `python -m main.anchors`, greedy-vs-
greedy verified per decision on **every one of 8,400 battles**; Metamon `@0a00a759`, Showdown pin
`e0551883f`, CPU, `OMP_NUM_THREADS=1`, ports 9500–9592, nothing written under `models/`. Three arms
at explicit zips — arm W `ai_v13_02_flywheel_winprob` @75,005,952, the continuation
`ai_v13_09_wcont` @87,097,344, the era-1 fold path `ai_v13_08_fold1_cont` @87,097,344 (the two
continuations end on the same step to the step). Twelve team seeds 20260919–20261029, **different
from the 2026-09-20 read's default 20260914**, the same seeds for all three arms, and the arms faced
**the same draw in the same slot 1200/1200** in every cross-arm comparison. Pre-registered at
`0f110483`; AMENDMENT 1 (`0920f197`) raised n from 400 to 1200 per arm per cell **unconditionally,
committed before any pooled contrast was computed**, because the cell ran ~7× cheaper than the SOP's
cost table budgets (~2 min per 100 games on a quiet box, not ~15). Artifact:
`designs/research_state/measurements/anchor_ab_continuation_2026-09-20/`.

**`SmallRL` greedy, n = 1200 per arm per team set.** AWAY (Metamon's own 20 `competitive` teams):
continuation **0.5925 [0.5644, 0.6200]**, arm W 0.5533 [0.5251, 0.5812], fold path 0.4725 [0.4444,
0.5008]. HOME (our 719 pool): continuation **0.6442 [0.6167, 0.6708]**, arm W 0.6242 [0.5964,
0.6511], fold path 0.5458 [0.5176, 0.5738]. The three registered contrasts, Newcombe on the
difference — 🚨 **C − W: +0.0392 [−0.0004, +0.0786] away and +0.0200 [−0.0185, +0.0584] home, NOT
DETECTED on both, and NOT DETECTED on both at the registered n = 400 too (+0.0150 and +0.0350)**;
**C − F: +0.1200 [+0.0802, +0.1593] away and +0.0983 [+0.0591, +0.1371] home, DETECTED on both**;
**F − W: −0.0808 [−0.1205, −0.0409] away and −0.0783 [−0.1174, −0.0389] home, DETECTED on both, in
the NEGATIVE direction.** 🚨 **The instrument therefore had the power and used it** — four CIs clear
of zero at effect sizes 0.078–0.120 on the very same games where C − W straddles. **Branch (c) of
the registration is what happened, and it was the branch I gave the LOWEST probability (0.15).** The
away C − W row sits on the line: the registered unpaired interval misses zero by 0.0004 on the wrong
side while the matched-draw paired bootstrap — a DESCRIPTOR, registered as such in advance — reads
[+0.0025, +0.0758]. The registered verdict stands and the weakness of the null is stated with it.
An unregistered inverse-variance combination of all three cells puts C − W at **+0.0320 [+0.0065,
+0.0575]** — printed for its SIZE, not as a verdict: **the external edge is ~3 pp where the untaught
meter reads 15.5, a factor of ~5.**

**The era gate fires on all three arms.** `SyntheticRLV2` greedy away, n = 400 per arm: continuation
**0.3850 [0.3386, 0.4336]**, fold path 0.3475 [0.3025, 0.3954], arm W 0.3375 [0.2929, 0.3852]. The
bar is *Wilson lower > 0.50*; here every **upper** bound is below 0.50, so by the SOP's clause (2)
the absolute bar fires on the whole lineage at 87M. All three contrasts NOT DETECTED at this n
(C − W +0.0475 [−0.0190, +0.1134]) — the cell agrees in direction and has no power at 400 games, and
its C − F / F − W rows must **not** be read as contradicting the `SmallRL` cells.

**🚨 What replicates and what does not.** The n = 100 `SmallRL` away read banked
continuation 0.640 / arm W 0.500 / fold path 0.430 and a **+0.140** headline contrast; at 12× the
games the same three arms read 0.5925 / 0.5533 / 0.4725 and **+0.039**. Every arm moved toward the
middle and the contrast shrank 3.6×. That read registered the row as DESCRIPTIVE with P(clears) =
0.15 and did not claim it, so nothing is contradicted — but **a 100-game anchor point estimate is a
detector of ~0.15-sized movement and is not a measurement of a ~0.05-sized one**, and a single
100-game sub-cell of this very cell ranged 0.34–0.59 *within one arm* across twelve seeds.
**What DOES reproduce externally is the era-1 fold path's COST**: below its own frozen parent by
~0.08 on both team sets, confirmed by an agent nobody here trained. That was the internal read's
second headline and it is now the only one of the two with external support.

**Two hazards, both findings.** (1) 🚨 **`regime_verified` read FALSE on 31 of 84 sub-cells while
all 168 halves reported `argmax_match_rate` = 1.0000** — hazard W-J is not a one-off, it fires on
~37 % of sub-cells, because the composite flag is an AND over both halves' `regime_check_ok` and a
peer's flag goes false if the peer process raises anywhere, including in post-game aggregation where
Metamon's `RecursionError` on a forfeit desync lands. **Reading the composite as "the regime failed"
would have voided a third of this campaign.** Backlog row: separate "the peer crashed after its
work" from "the regime was not matched". (2) ⚠️ **editing a running bash script corrupted both
lanes' tails** — `bash` reads a script by byte offset, so appending the extension jobsets mid-flight
ended the lanes in a syntax error *after* the loop; all 24 registered sub-cells had already been
dispatched and completed (24/24 `status: OK`), which was luck. Fixed by running from a frozen copy
outside the edited tree. Integrity otherwise clean: **0 of 84 sub-cells failed**, 50 of 8,400 games
at the 250-turn cap (0.60 %, matching the SOP's banked 0.5 %), 15 ties, `team_source_asymmetry`
false everywhere, `n_defaults` / `n_redecides` 0 / 0, `model_rung` `explicit_zip` and `model_loader`
`bare` on every row.

**Reading.** The untaught meter and the external anchor disagree by ~5× on the same contrast, and
the one that cannot drift with our training is the smaller. That does not make the continuation's
gain unreal — the sign agrees in all three cells and the combined interval clears zero — and it does
not make the untaught meter wrong. It does mean **every "+15 pp" of this era should be quoted as
competence AGAINST OUR OWN LINEAGE until an external cell says otherwise**, and it means the era
gate's verdict (all three arms 12–16 points behind `SyntheticRLV2` at 87M) is the number that should
set expectations, not the internal one. Tag: **MEASUREMENT (MAJOR) · 8,400 ANCHOR BATTLES · C − W
NOT DETECTED ON EITHER TEAM SET AT n = 400 AND n = 1200 · THE FOLD PATH IS 0.08 BELOW ITS OWN PARENT
EXTERNALLY · ORDER REPRODUCES, MAGNITUDE ~5× SMALLER · ERA GATE FIRES ON ALL THREE ARMS · HAZARD
W-J FIRES ON 37 % OF SUB-CELLS WITH EVERY PER-DECISION RATE AT 1.0000**.

---

## Addendum 2026-09-21 — HAZARD W-J IS CLOSED, AND ALL 31 SUB-CELLS FLIP TO VERIFIED

The backlog row this campaign filed ("separate *the peer crashed after its work* from *the regime
was not matched*") is shipped as **`gen3_anchor_regime_split_v1`**, and the split has been applied
retroactively to this directory's own saved evidence.

| | |
|---|---:|
| sub-cells | **84** |
| old composite `regime_verified` FALSE | **31** (arms C 6 / F 14 / W 11; 3,100 of 8,400 games) |
| → `regime_verified_decisions` TRUE under the split | **31** |
| → still unverified | **0** |
| peer `rc` on all 31 | **1** |
| peer error class on all 31 | **`RecursionError`**, 40 occurrences |
| per-decision `argmax_match_rate` on all 31 | **1.0** — the only value that appears |
| rows-vs-half-report disagreements | **0 / 84** |

**Every one of the 31 was a clean regime with a dirty exit.** Metamon raises a `RecursionError` in
its post-game teardown when our side forfeits at turn 250 — after its last decision, after every
game is played and recorded — so the peer exits 1 and a flag that ANDed the two questions read
FALSE. Under the split, `regime_verified_decisions` is true on all 84 sub-cells and `peer_clean` is
false on 31.

🚨 **Nothing else is re-derived and nothing else changes.** No win rate, no Wilson interval, no
contrast and no verdict in `results_registered_n400.json` or `results_n1200.json` is recomputed
here: this addendum explains a FLAG, and the games were always played and always recorded. The
campaign's reading — C − W NOT DETECTED on either team set at n = 400 and n = 1200, the fold path
0.08 below its own parent externally, the era gate firing on all three arms — stands exactly as
written above.

Reproduce: `python designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/regime_split_rederive.py`
→ [`regime_split_rederived.json`](regime_split_rederived.json) (per sub-cell: both half reports,
the rates, the rc, the error, the old composite and the two new fields). The definitions it applies
are the shipped ones, and `src/main/anchors/regime_split_test.py` pins them.
