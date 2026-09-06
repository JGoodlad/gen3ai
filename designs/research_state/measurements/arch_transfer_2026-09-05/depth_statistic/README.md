# THE DEPTH STATISTIC — does the trained policy leave value on the table that only DEPTH can recover?

*Probe for the ARCH→TRANSFER batch, 2026-09-05. Read-only over `models/`; CPU-only, `nice 10`,
BLAS pinned to 1 thread. Worktree `agent-a5d93fae566ec1b73` at `f91f93c3`.*

**Why this exists.** `designs/learning/entity_tokens_biases_pointers.md` → Part 4 → *"Depth, and the
looped transformer"* names one pre-probe as the gate on any looped-trunk or deeper-trunk proposal:

> "The right pre-probe already exists: the better-line search records the depth at which it finds an
> improvement over the played action. If depth-2 improvements are rare, extra hops inside the
> network would not have found them either; if they are common, that is the case for depth, looped
> or stacked."

This file turns that sentence into a number with an interval.

---

## 1. PRE-REGISTRATION (written before any statistic was computed)

### (a) The statistic

Over decisions where the **search** engaged, split by the **realized search depth**:

1. **PRIMARY — the depth-engagement fraction.** Of searched decisions in the depth-enabled arm, the
   fraction whose realized depth is ≥2. This is an **upper bound** on "the fraction of improvements
   that need depth ≥2": an improvement can only be a depth-≥2 improvement on a decision that
   actually reached depth 2. Wilson 95% CI, clustered read reported beside the raw binomial.
2. **PRIMARY — the depth-attributable action change.** Paired across two arms run on the SAME seeded
   games with the SAME width caps, differing only in `--max-depth` (1 vs 3): the action-change rate
   (`n_changed / n_searched`) in the depth-3 arm minus the depth-1 arm. The depth-1 arm's changes
   are the ones a one-hop search finds; the excess is what only a second ply found.
3. **SECONDARY — the depth dividend in outcome units.** Paired mirror win rate (depth 3) − (depth 1)
   on matched game indices, against the 0.50 mirror null. This is the ΔP(win) analogue the
   pre-registration asked for; the per-decision ΔP(win) the phrasing implies is **not recoverable
   from this instrument** (§2), and the reason is recorded rather than papered over.
4. **CONTEXT — gate and futility shares**, from the committed defensive cells: the fraction of
   decisions the defensive gate FORCES (plays the policy, no search) and the futility split. Those
   are decisions where depth cannot matter by construction, so they bound the population any depth
   lever could ever address.

### (b) The prediction and the decision rule

**Prediction (registered):** depth-≥2 improvements will be RARE. Grounds: the ledger's 2026-08-23
reading that at default caps width absorbs the whole budget (realized depth 1.00, deepen rate 0%),
the standing "trunk is healthy / ceiling is structural holes not capacity" verdict, and the fact that
the v8-era `damage_refine_rounds` in-forward loop measured near-inert three times.

**Decision rule, fixed before the data:**

| reading | verdict |
|---|---|
| **< 20%** of confirmed improvements need depth ≥2 | **depth is NOT the binding lever** — the note's standing verdict holds, no build proposal |
| **20–40%** | **UNDECIDED** — needs a bigger cell before anyone builds |
| **> 40%** | **depth is a LIVE lever** — deserves a build proposal |

Applied to the PRIMARY pair: statistic 1 is the upper bound and statistic 2 is the point estimate.
If the **upper bound** (statistic 1) already falls below 20%, the verdict is NOT-BINDING and
statistic 2 cannot rescue it — an improvement that never reached depth 2 was not found at depth 2.

### (c) How many decisions do I expect?

From the committed defensive cells: **42.4 decisions/game**, of which ~88% are clean move-selection
(the rest are forced switches the search declines by rule). At 30 games × 2 orientations = 60
orientation-games per arm, I expect **≈2,200–2,600 searched decisions per arm**, ~2,500 as the
planning figure. A Wilson interval on a 10% rate at n=2,500 is ≈ ±1.2 pp, so the 20% and 40% rails
are separated by many interval widths — the instrument is not the limiting factor for the PRIMARY
statistic. The SECONDARY (win-rate) statistic at 30 pairs carries a paired CI of roughly ±0.10 and
is therefore **weak by design**; it is reported as context, and no verdict rests on it.

### (d) Vocabulary

**SIGNIFICANT** = the interval excludes the null. **WITHIN FLOOR** = the effect is smaller than the
instrument's own noise floor. **NOT DETECTED** = the interval straddles the null and the design had
the power to see the registered effect size.

*(Deviation, recorded: the cell was run at **50** games × 2 orientations per arm, not the 30 planned
above — the depth-1 arm turned out cheap enough to afford it inside the CPU budget. The pre-registered
rails and decision rule are unchanged; the larger n only tightens the intervals.)*

---

## 2. FINDING ZERO — the committed artifacts CANNOT answer this question, and the CLI says the opposite of the truth

Two things had to be established before any number was computed. Both are findings in their own right.

### 2a. Every committed search cell is depth-1 BY CONSTRUCTION

The three defensive-search cells are the only committed per-decision search provenance in the tree.
All three were launched with `--max-depth 3`, and all three realized depth 1 on every game:

| cell | orientation-games | decisions | `--max-depth` | `max_depth_realized` observed | `n_deepened` TOTAL |
|---|---|---|---|---|---|
| `defensive_search_first_cell` | 400 | 16,942 | 3 | {0, 1} | **0** |
| `defensive_search_iter2` | 1,600 | 68,585 | 3 | {0, 1} | **0** |
| `defensive_search_iter3` | 1,275 | 55,958 | 3 | {0, 1} | **0** |

**141,485 decisions, zero deepened.** The cause is structural, not a budget accident: `racing` and
`defensive` are root ALLOCATORS and *"a racing round is depth 1 (racing and iterative deepening are
not composed)"*. `--max-depth 3` on those strategies is inert. The `racing_root_selection` memo says
so in its own limitations section and names this probe's exact design as the open question:
*"Whether iterative deepening on a narrowed beam beats width is a separate and probably more
interesting question."*

Two further reasons those rows could not have answered it even had they deepened: they are
**per-GAME, not per-decision** (counters only — no per-decision action, score or depth), and
`_score_world` **overwrites** the depth-1 `values` with the deepened ones, so no artifact anywhere
retains the depth-1 argmax beside the deepened one. The literal instrument the learning note names —
the prober's `better-line` — has **no committed results at all**.

The only cells that ever did deepen are the 2026-08-23 grid readings, and the ledger marks them
**VOID**: taken under the depth-≥2 chunk-gap defect, so *"every deepened arm was scored on a holed
replay or dropped."*

### 2b. 🚨 HAZARD — the CLI's own `--help` still forbade the measurement its code now supports

`--max-depth`'s help string read, until this probe:

> "WARNING depth>=2 is built and gated but its successor replay has an OPEN fidelity defect (see
> deepen.py); it fails safe as a counted search_error, but **do not publish a depth-2 number yet**."

That defect is **FIXED** — `gen3_search_depth2_chunk_gap_v1`, commit `16b4bf0` (2026-08-29), ledger
*"TASK #38 CLOSED"*. `deepen.py`'s module docstring was updated to say so ("**THE DEPTH-≥2 REPLAY
DEFECT IS DIAGNOSED AND FIXED**"); `__main__.py`'s help string was not. The two files contradicted
each other, and the one a reader meets first — `--help` — carried a stale veto on exactly this probe.
This is the CLAUDE.md-named class: *an allowlist entry can outlive its own fix and then mislead every
reader after, including a subagent briefed from it.*

**Corrected in this worktree** (`src/main/search_dividend/__main__.py`, help string only — no
behaviour change; no test pinned the string; the three static gates are green). It now names the fix
commit and keeps the "2026-08-23 readings stay VOID" half, which is still true.

---

## 3. The fresh cell

Because 2a holds, a fresh cell was required. It is the minimum that isolates depth: **two arms, the
same seeded games, the same width caps, differing ONLY in `--max-depth`.**

```
nice -n 10 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src python -m main.search_dividend \
  /home/goodlad/dev/gen3ai/models/ai_v9_59_R2ACTION_0827/final_model.zip \
  --arm honest --budget 1 --games 50 --games-seed 7 --opponents self \
  --max-opp 2 --max-worlds 1 --max-dice 1 --max-depth {1|3} \
  --device cpu --battle-timeout-s 1800 --battle-idle-s 120 --out {d1|d3}.jsonl
```

* **`--max-opp 2 --max-worlds 1 --max-dice 1` is the depth lever, and it is mandatory.** Width is
  spent FIRST in the registered order, so at the default caps (6/8/8) it absorbs the whole clock and
  every decision reports depth 1 — the two arms would have been identical and the null would have
  been a configuration, not a finding. This is the ledger's own registered lever ("the depth lever is
  `--max-opp`/`--max-dice` narrowing, NOT budget").
* **`--root-strategy grid`** (the default) — the only strategy iterative deepening composes with.
* **`--opponents self`** is the mirror: the searched side against the SAME network with search off,
  so the no-effect point is **0.50 by construction**, side-swapped so the team draw differences out.
* **`--games-seed 7`** is the historical mirror cells' seed, so game *g* is the same pinned dice and
  team draw the defensive cells played.
* Checkpoint `models/ai_v9_59_R2ACTION_0827/final_model.zip`, `arch_signature
  gen3_critic_route_wave_v1` == the tree's current signature (no arch drift). `models/` read-only.

**Hygiene.** 100 orientation-games per arm, **100/100 finished, 0 errors, 0 timeouts, 0 ties** in
both. Battle wall 1,100 s (depth-1) + 2,595 s (depth-3) = **1.03 CPU-hours**, two processes at
`nice 10`, BLAS pinned to 1 thread, CPU only. **The box was BUSY throughout** — load average 20–31 on
16 cores (a live training run shares the machine), a contention factor ≈1.3–1.9. The budget is a
wall-clock deadline, so under load both arms buy fewer arms per second; they ran **concurrently**, so
the confound is shared and the paired contrast is the readable quantity. The absolute win rates are
not comparable to the historical idle-box mirror cells.

Artifacts here: `depth1_rows.jsonl.gz`, `depth3_rows.jsonl.gz`, `cell_summary.json`,
`depth_statistic_probe.py` (re-runs both halves).

---

## 4. The table

### 4a. Did depth actually engage? (yes, overwhelmingly)

| | depth-1 control | depth-3 arm |
|---|---|---|
| orientation-games | 100 | 100 |
| decisions | 3,938 | 4,000 |
| searched (fallbacks excluded) | 3,433 | 3,373 |
| mean realized depth | 1.000 | **1.831** |
| max realized depth | 1 | **3** |
| mean beam (our actions carried into the deeper ply) | 0.00 | **2.73** |
| **DEEPENED (realized depth ≥ 2)** | 0 / 3,433 = 0.0000 | **2,574 / 3,373 = 0.7631 [0.7485, 0.7772]** |

The lever worked. Three quarters of searched decisions got a second ply, with ~2.7 root actions
re-scored one hop deeper, and some went to a third.

### 4b. THE STATISTIC — did the second ply change anything?

| | depth-1 control | depth-3 arm |
|---|---|---|
| action CHANGED (search action ≠ policy action) | 2,245 / 3,433 = **0.6539** [0.6379, 0.6697] | 2,210 / 3,373 = **0.6552** [0.6390, 0.6711] |

| quantity | value | verdict |
|---|---|---|
| **depth-attributable change rate** (d3 − d1, Newcombe) | **+0.0013 [−0.0213, +0.0238]** | **NOT DETECTED** |
| …as a share of all depth-3 changes | **+0.19%** | — |
| **depth-engagement fraction** (the pre-registered UPPER BOUND) | 0.7631 [0.7485, 0.7772] | opportunity, not improvement |

### 4c. Depth-conditional outcome (the ΔP(win) analogue)

| arm | paired mirror win rate | 95% CI | pairs |
|---|---|---|---|
| depth-1 control | 0.2100 | [0.1354, 0.2846] | 50 |
| depth-3 | 0.2300 | [0.1450, 0.3150] | 50 |
| **DEPTH DIVIDEND (d3 − d1), paired on the 50 shared game indices** | **+0.0200** | **[−0.0919, +0.1319]** | 50 |

**NOT DETECTED**, at a width that would have caught ±9 pp — weak by design, as pre-registered, and no
verdict rests on it. The row worth noting separately: **both arms sit ~27 pp BELOW the 0.50 mirror
null.** Narrow-width search on this critic is strongly harmful, replicating the standing ledger
verdict ("depth-1 search on today's critic is NEGATIVE… the critic cannot tell branches apart at the
margins the search acts on") and sharpening it — narrowing the marginalization to `m_opp=2, K=1, R=1`
removes the averaging that was holding the leaf noise down.

### 4d. The population where depth cannot matter by construction (committed cells)

| cell | GATE FORCED (policy played, no search) | futility of raced | deadline-truncated share of futility | separated | overruled |
|---|---|---|---|---|---|
| first cell | **0.7385** [0.7314, 0.7455] | 0.8432 [0.8314, 0.8542] | 3,301/3,301 = **100%** | 0.1568 | 0.0180 |
| iter2 | **0.7515** [0.7480, 0.7549] | 0.5458 [0.5379, 0.5538] | 8,223/8,229 = **99.93%** | 0.4542 | 0.0582 |
| iter3 | **0.7504** [0.7465, 0.7542] | 0.6446 [0.6361, 0.6530] | 7,941/7,941 = **100%** | 0.3554 | 0.0011 |

**~74–75% of decisions are FORCED** — the critic's `|P(win) − 0.5| ≥ 0.15` gate plays the policy
instantly because being overruled would not matter. Depth cannot help there by construction. Of the
~26% raced, a further 55–84% end in futility (never separated), essentially all clock-ended. So even
before the depth statistic is read, the addressable population is roughly **a tenth of decisions** —
and §4b says the second ply re-ranks ~0% of them.

---

## 5. VERDICT

**Depth is NOT the binding lever. The learning note's standing verdict HOLDS.**

Against the pre-registered rule: the point estimate for improvements requiring depth ≥2 is **+0.19%
of changes, interval [−2.1 pp, +2.4 pp] on the rate** — far below the 20% rail, by an order of
magnitude.

The pre-registration's upper-bound shortcut did **not** fire (depth-engagement is 76%, not <20%), so
the verdict rests where the rule said it then must: on the point estimate. And that is the strongest
form the answer could have taken. **This is not "depth never got a chance."** Depth got three
quarters of the decisions, a ~2.7-action beam and up to three plies — and re-ranked essentially none
of them. The second hop of who-threatens-whom, computed by a real simulator on real successor states,
reproduced the first hop's ordering.

That is a direct answer to the note's question. If a *perfect* second hop — an actual game-tree ply,
not an approximation of one inside a network — does not change which action looks best, extra
message-passing hops inside the trunk would not have found those improvements either, because there
are almost none there to find. Convergent with the note's own strongest prior: the v8-era
`damage_refine_rounds` in-forward loop measured near-inert three times.

**Recommendation: no looped-trunk or deeper-trunk build proposal is justified by a depth argument.**
The binding constraint this cell keeps pointing at is the **leaf** — a critic that cannot separate
branches at the margins the search acts on (both arms 27 pp below a by-construction 0.50 null). Same
conclusion iter2 reached by another route ("the missing dividend was never budget-limited after all:
it is the *leaf*") and the same one the R-ladder closed on.

---

## 6. CAVEATS — read before quoting any of this

1. **SEARCH depth is not NETWORK depth.** A search ply is an exact simulator transition plus a fresh
   critic evaluation; a transformer layer is one round of message passing over 29 seats. The
   inference "no search-depth dividend ⇒ no network-depth dividend" is an ANALOGY the learning note
   itself proposes, not an identity. It is a strong analogy — a real ply is strictly more informative
   than an approximated one — but it is the weakest joint in the argument and the honest place to
   attack this verdict.
2. **The critic scoring the leaves is the SAME network.** A depth-2 search improvement would be
   evidence the POLICY lacks something the CRITIC can see. Its ABSENCE is therefore ambiguous between
   "there is nothing at depth 2" and "this critic cannot see what is at depth 2." Given the measured
   leaf noise (per-leaf dice sd 0.0115 against a top1−top2 margin of 0.0213), the second reading is
   live — and it points at critic resolution rather than trunk depth, which is where the verdict
   lands anyway.
3. **`n_changed` is "search ≠ policy", not "ply 2 ≠ ply 1".** No artifact retains the depth-1 argmax
   beside the deepened one (`_score_world` overwrites `values`), so the depth-attributable change is
   a difference of RATES across two arms, not a per-decision join. The arms play identical games only
   until their first differing action. At n≈3,400 searched decisions per arm the rate contrast is
   solid, but it is an aggregate; a *decision-level* statistic would need a new recorder (sized: one
   field in `_score_world`'s diagnostics carrying the depth-1 argmax — cheap, and it would also serve
   ladder requirement 3).
4. **The MAX-backup optimism bias runs WITH the effect, not against it.** Depth ≥2 backs up with MAX
   over our actions, and `E[max] ≥ max E` under noise, so a deeper arm should change MORE actions for
   purely statistical reasons. The measured excess is +0.19%. The true "genuine improvement" share is
   therefore ≤ the measured one — a conservative reading, and the ledger's *"never report a
   max-selected gap without its null sim"* rule honoured by design rather than by simulation.
5. **BUDGET CONFOUND, and it is the real one.** The depth arm buys its plies out of the SAME 1 s
   clock. It is not "depth-1 plus a ply"; it is the same money spent differently. A dividend could
   exist at a budget where depth is additive rather than substitutive. What this cell prices is the
   registered trade — *at a fixed deadline, is a second ply worth more than the width it displaces?*
   — and the answer is no. Note the width was ALREADY narrowed to `m_opp=2, K=1, R=1` in BOTH arms,
   so the depth arm is not displacing width; it is spending the leftover. That makes the null
   stronger, not weaker.
6. **One checkpoint, one arm, one opponent tier.** `honest` (belief-determinized), `ai_v9_59` final,
   the mirror. The `oracle` arm has one world by construction and would spend its leftover on dice;
   depth against a bot roster is a different population (the gate forces 82.5% there, not 74%).
7. **The depth-3 arm carries 115 non-`not_move_selection` fallbacks** (`prefix_gate_failed` 66,
   `root_failed` 46, `search_error` 3) against the control's 1, concentrated in 3 of 100 games. These
   are determinized-world prefix-gate and driver failures; every one falls back to the POLICY action,
   so they bias the win rate **toward** the 0.50 null, and they are excluded from `n_searched` so the
   change-rate denominators stay clean.
8. **Contention.** Load average 20–31 on 16 cores throughout. Absolute win rates are not comparable
   to the idle-box historical mirror cells; the paired within-batch contrast is.

