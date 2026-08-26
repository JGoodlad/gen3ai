# PLAYOFF FORMAL READ — the top-2 terminal-rollout mirror cell, final (2026-08-25)

**The bankable read of the search-dividend PLAYOFF cell.** The oracle sweep demoted to a SCREEN;
the top-2 actions settled by PAIRED rollouts to a terminal under common random numbers; the playoff
may override the critic only when `|mean d| ≥ 2·SE` over ≥4 pairs, else it plays the POLICY's
action. Data: `tmp/search_dividend/playoff_10s.jsonl` (80 rows = 40 games × both orientations,
mirror vs the same network, seed 11) + `playoff_10s.log`. Code:
worktree `agent-a6c1260df535a67fc` (the build WITH the playoff observability fixes).
Checkpoint: `models/ai_v9_29_rev1_0823/final_model.zip`. Headline numbers:
[`playoff_formal_read.json`](playoff_formal_read.json).

**Budget provenance:** the registration said 10 s/decision; the cell RAN at **20 s** and the file
keeps the `playoff_10s` name. The deviation was ratified before launch and is itself a
measurement: at 10 s the realized budget bought R = 3 paired rollouts, below the instrument's own
`MIN_PAIRS = 4` floor, so a 10 s cell would have read 100% inconclusive **by construction** —
raising the budget is honest, lowering the floor would have manufactured a verdict.

**Exclusion (pre-registered rule 1):** **g0/o1 is EXCLUDED** — a tainted v1 build-era game
(pre-observability-fix): its 17 `playoff_error` decisions / 204 failed pairs are a build artifact
and its `fallback_details` is `[]`, so its failures cannot be diagnosed. It is directionally
neutral (every error falls back to the policy action), so the exclusion is hygiene, not
result-shopping. Footnote: g0/o0 also predates the fix but carried **zero** errors and zero failed
pairs and is retained; excluding only o1 leaves game 0 unpaired, so the paired estimate runs over
**39** swap-pairs. Both with- and without-exclusion numbers are quoted below; they do not differ in
any conclusion.

---

## 1. The headline — the three rates WITH the paired win rate (never the win rate alone)

Over the **2,710** decisions at which the screen produced a top-2 (excluding g0/o1; 3,314 decisions
total in the cell):

| outcome | n | rate | meaning |
|---|---|---|---|
| **screen_decisive** | 425 | **15.7%** | screen's top pick IS the policy's action by > the leaf-noise margin — nothing to arbitrate, no rollouts spent |
| **resolved (played)** | 52 | **1.9%** | the paired rollouts cleared 2·SE and the playoff acted |
| **inconclusive** | 1,911 | **70.5%** | ground truth at R ≈ 10 paired terminal rollouts could NOT separate the critic's top-2 → policy action |
| **error (0 completed pairs)** | 322 | **11.9%** | every pair of the decision failed (diagnosed in §2) → policy action |

| win rate | value |
|---|---|
| **paired (39 swap-pairs, excl g0/o1)** | **0.436 [0.362, 0.509]** — null 0.50 **not excluded** |
| paired (all 40 pairs) | 0.450 [0.37, 0.53] |
| raw (79 orientation-games, excl) | 0.443 (35/79, 0 ties) |
| raw (all 80) | 0.450 [0.35, 0.56] |

Support numbers: realized **R = 10.1** pairs per playoff that ran (n = 1,963), **15.2 s** of
playoff wall per ran decision; 114 rollouts (0.29%) hit the 250-turn cap (scored 0.5 by the shared
`gen3_cf_draw_at_cap_v1` rule). The playoff changed the action on **36 of 3,314 decisions = 1.1%**
of the game it played — this arm is very nearly the policy null, by the 2·SE gate's own choice.
Decisions reaching a SETTLED search verdict (screen_decisive + played) = 477/2,710 = **17.6%**.

**Mirror caveat (rule 6):** `--opponents self` carries no ELO anchor; these are direct win rates,
never read through the anchored-ELO fit. The mirror's no-effect point is 0.50 by construction.

## 2. failed_pairs = 4068, diagnosed (the number the preliminary read owed)

The diagnosis is one identity: **on every row, `n_playoff_failed == 12 × playoff_error`**
(339 × 12 = 4,068 across all 80 rows; 322 × 12 = 3,864 excluding g0/o1; zero deviations). So the
4,068 is not a scatter of dropped pairs across healthy decisions — **failures are wholesale**: an
affected decision loses ALL 12 of its pairs, and no decision that lost some pairs concluded on the
rest. The 4,068 headline count is therefore just the 339 error-stage decisions counted twelve
times; the clean-n concern in the preliminary read resolves to "11.9% of screened decisions
errored out and fell back to the policy".

Where and why (from the captured `fallback_details`; the field caps at 3 unique texts per game, so
cause counts are a **sample, not a census** — but every post-fix error row carries at least one):

- **Bridge no-progress reject loop** (13 of 17 captured texts): the nested rollout battle wedges
  on `local_sim_bridge error: no-progress reject loop on p2: 9 consecutive refusals with no
  committed decision` — an invalid-choice refusal loop inside the counterfactual replay, across
  many move/switch kinds and turns (turn 4 to turn 88).
- **Prefix replay decision-count mismatch** (4 of 17): `prefix replay produced N decisions for a
  branch at index N — prefix_chunks and prefix_actions disagree, so the arms would branch from the
  wrong state` — the record's prefix and its action list disagreeing at the branch point.

The 322 error decisions cluster in **12 games** (13 rows counting the excluded g0/o1) and lean
orientation-1 (223 vs 116). Every one falls back to the policy action, so the failure mode biases
the cell **toward the 0.50 null** — the 0.436 paired read is, if anything, conservative.

Separately, the SCREEN itself died wholesale in four games (`g6/o1`, `g8/o0`, `g31/o1`, `g36/o1`:
a `search_error` followed by `root_failed` storms — 170 root_failed / 60 prefix_gate_failed / 5
search_error across the cell). Blast radius was **contained within each game**: the following game
is clean in all four cases.

## 3. Timeout hygiene (rule 5)

**Zero timeouts occurred.** The run raised the backstops precisely because a timed-out game
poisons the shared SearchSession for subsequent games (`--battle-timeout-s 5400`,
`--battle-idle-s 180`); the longest game was 1,610.9 s wall against the 5,400 s backstop, and
neither log contains a timeout or idle-kill line. The known poison defect had no opportunity to
fire; no blast-radius bound is needed for this cell.

## 4. `n_playoff_reversed` — endorsed post-cell (rule 4)

The schema addition is endorsed, and this cell shows exactly why it is needed: the reversal count
— of the 52 resolved playoffs, how many picked the screen's **a2 over its a1** — is **not
derivable** from the shipped counters. The nearest derivable proxy is changed-vs-policy: **36 of
the 52 resolved playoffs (69%) played a non-policy action** (`n_changed` can only come from the
played stage — screen_decisive keeps the policy action by definition and every fallback returns
it). But changed-vs-policy conflates "playoff endorsed the screen's override of the policy" with
"playoff overturned the screen", which is the crispest leaf-bias number this instrument could
produce. Next playoff cell records it per decision.

## 5. Verdict

**No — at 20 s/decision, a terminal-ground-truth playoff on top of the critic screen buys nothing
measurable.** Ground truth refuses **70.5%** of the critic's contested top-2 comparisons, settles
only **1.9%**, changes **1.1%** of all decisions, and the paired mirror reads **0.436
[0.36, 0.51]** — the null is not excluded and the point sits below it. This is the pre-registered
**honest-NULL branch** (neither the ≥0.50 "rollouts cure the bias" reading nor the 0.19–0.27
"screen selection is the residual disease" reading): the 2·SE gate did its job, and the rollout
budget could not resolve the pairs.

What it means, composed with the prior banked verdicts: the mirror battery showed plain depth-1
search COSTS ~17–22pp; the R-ladder showed the harm is leaf **bias**, flat across a 32× dice
sweep; this cell closes the loop **by refusal** — when an unbiased arbiter (terminal rollouts,
CRN-paired, in the mirror where the self-rollout is the exact estimand) adjudicates the same
contested comparisons, it declines ~70% of them and the harm disappears (0.44–0.45 vs 0.19–0.33)
without any dividend appearing. The plain arms' overrides were noise-artifacts of critic blur
almost in their entirety. The `top1−top2` margin on contested decisions is genuinely below the
noise floor of even a 10-pair terminal-rollout estimate at 20 s/decision.

**For the parked search program: this read CONFIRMS the standing disposition and does not change
it.** Search stays parked behind critic calibration / R1; the mirror table remains the
critic-resolution meter to re-run after each critic milestone. The playoff pattern — screen +
honest arbiter + refuse-on-noise — remains the deployment template IF search ever earns its way
back, with the caveat that 20 s/turn is a science instrument, not a ladder-deployable config
(it out-accrues the 150 s ladder timer). Total cell cost: 9.14 h wall.
