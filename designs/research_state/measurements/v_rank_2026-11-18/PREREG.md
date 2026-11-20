# V-RANK: does the search's value rank ONE position's moves as well on unfamiliar teams as at home? — PREREG (2026-11-18)

Written and committed BEFORE any game of this read is played (FULL tier, blind). Design:
[`../../../endstate/design_v_rank_on_u.md`](../../../endstate/design_v_rank_on_u.md) (§3–§4, with the deviations of
§9 below). Queue item Q9, model lever MF-4 ([`../../EXPERIMENT_BACKLOG.md`](../../EXPERIMENT_BACKLOG.md)). Spec:
[`read_spec.json`](read_spec.json) (`main.reads`, rule `v_rank`). Driver: `main.value_search.v_rank`; estimator:
`main.value_search.v_rank_stat`; the branches' play path: `main.anchors --our-force`. Pin: `c933afd5a` (the commit
that lands the driver, the estimator and the rule). NOT LAUNCHED: the orchestrator schedules the lanes (§8).

**Abbreviations.**
- P\* = the named baseline `pstar_plateau` (`python -m main.baselines path pstar_plateau`): `rb_x26_s2601`
  @280,000,170, sha256 `814823c5…91ed`, greedy. Kakuna = Metamon's Kakuna, checkpoint 34, greedy.
- home72 = 72 pool teams. U = the unfamiliar panels H (`heldout72_hpfix`), F (`forum72`) and P72 (`p72`), at equal
  panel weight.
- π = P\*'s greedy policy; π's top = the action it plays.
- S = `r0s4_honest@10`: the honest one-ply search (8 worlds consistent with our information state, fresh dice, the
  4-turn leaf with P\* on both sides and the critic V, piKL λ 0.015), from turn 10; run offline on the kept honest
  input log through the live searcher's own `_search` and root guards.
- **Q̂(a)** = S's value of legal action a at a decision. **y(a)** = the result for us (win 1, tie ½, loss 0) of the
  REAL game against Kakuna in which we play a at that decision and π everywhere else, on the deal's own seed.
- **β** = the within-position slope of y on Q̂ (§4): the share of Q̂'s action differences that are real against
  Kakuna. β = 1: realised one for one; β = 0: noise.
- A deal = one game of a 20-game role-balanced unit (panel, unit seed, game). A block = one deal of each U panel
  and three home72 deals. LB / UB = a 95 % interval's bounds. GUESS = an unmeasured effect; ESTIMATE = a cost from
  measured rates; MY JUDGEMENT = an inference no read states.

---

## 1. The question, and what it decides

A one-ply argmax needs the value to rank the SIBLING moves of one position. H-MID found P\*'s critic miscalibrated in
LEVEL on unfamiliar teams, but a level bias common to every sibling cancels in an argmax, so it is not evidence that
the leaf mis-ranks siblings on U. **This read asks that directly, with a truth that holds no opponent model:** at
a position of a raw P\* vs Kakuna game, is the slope of the real outcome on Q̂ lower when P\* pilots an unfamiliar
team than when it pilots a home72 team?

**It decides MF-4's funding** (MY JUDGEMENT; the design's §4.3 gated S-U, which S′-LIVE and S′-HONEST have since
answered in games):

| class | MF-4 |
|---|---|
| **RANK-NULL-ON-U** (either look) | **FUNDED**: the leaf-value step (a leaf critic fine-tuned on banked Kakuna outcomes, cross-fit by panel, ≈ 1–2 GPU-h; then the Q head on that evidence). β_home is reported beside it: a null on both sets says the deficit is not U-specific, and the step is still the one a non-self leaf value answers |
| **RANK-WORSE-ON-U** | **FUNDED**, as a U-specific within-node deficit |
| **RANK-HOLDS-ON-U** | **NOT FUNDED on this evidence.** Never a statement that search pays (the slope is 0.54 on P\*'s own states while search gains nothing there; review M6) |
| UNDECIDED | unchanged; the read is reported |

It is a meter of the MODEL and claims no search gain. It moves no recipe and adopts nothing by itself.

## 2. The ground truth

Every arm is a real game against Kakuna. π's arm is the raw game, already played; every other legal action's arm is
a FORCED BRANCH: the deal replayed from turn 1 with π, the named action played at the position, π after it. On one
engine our play and Kakuna's replay exactly, so the branch reaches the position byte for byte. **G-BRANCH** checks
that live in every branch before the forced action is sent (the battle's index, the honest input log's sha256, the
choice tokens, the row's sha256 and π's greedy action all equal the raw decision's), and afterwards the branch's
capture must equal the raw game's protocol through the branch turn.

**Kakuna's replay is not perfectly deterministic** (the dlive read: 9 of 2,100 branches, 0.43 %, diverged before
their state; AM-DL-PEER). So the peer-divergence classifier is registered from the start: an attempt whose capture
shows the PEER leaving the raw game's line before the position is a PEER DROP, not a G-BRANCH failure; every other
G-BRANCH failure VOIDs. One null control a raw unit (π's own action forced) must replay its raw game whole.

## 3. The read

**Raw games** (`read_spec.json`, `main.reads` anchor groups; whole 20-game units, P\* greedy with
`load.input_log: "10-30"`, opponent `metamon:Kakuna`). Unit seeds are fresh: no committed spec uses 2000–2159.

| look | blocks | U units (each of H, F, P72) | home72 units | deals a set | positions a set (≈ 1.95 a deal) |
|---|---|---|---|---|---|
| stage 0 | 5 | games 0–4 of unit seed 2000 | games 0–4 of 2100, 2101, 2102 | 15 | ≈ 30 |
| **look 1** (this spec) | 100 | seeds 2000–2004 (group `u1`) | seeds 2100–2114 (group `home1`) | 300 | ≈ 590 |
| **look 2, final** (a second spec, §7) | 200 by default; the sizing rule may raise it to at most 400 | + 2005–2009 (to 2019 at the cap) | + 2115–2129 (to 2159 at the cap) | 600 (≤ 1,200) | ≈ 1,170 (≤ 2,340) |

A block b is one deal of each U panel (unit seed 2000 + b // 20, game b % 20) and three home72 deals (unit seeds
2100 + 3·(b // 20) + j, j = 0, 1, 2, game b % 20), so both sets hold the same number of deals and a size change never
moves an earlier deal.

**Positions** (outcome-blind; `v_rank select`, written once). Eligible: our kept decisions with ≥ 2 legal actions.
Stratum A: turns 10–19; stratum B: turns 20–30. In each stratum of each deal, the eligible decision with the
smallest `sha256("vrank_v1|<deal>|<turn>|<decision index>")`. A stratum with no eligible decision adds nothing. Each
state is stamped with the team we piloted.

**Q̂ and the shadow arms** (`v_rank score`, offline, resumable). At each selected state: S's Q̂ for every legal
action with the per-world leaf values (refused unless each action's mean is the search's own Q̂), log π, and an
exclusion if no honest world fits; then the self-referee `oracle_leaf8` (the TRUE world, every legal action played
8 times to the end with P\* on both sides; REPORTED only). Every row carries the scoring zip's sha256. The scorer
prints no value. A state S excludes gets no branch; the exclusion rate is reported by panel.

**Branches** (`v_rank plan`, then `main.x35.kakuna run` per shard). One `main.anchors --our-force` one-game unit per
legal action other than π's at every searched state, and one null control per raw unit. The plan is a function of
the selection and the scores; it is extended as more states are scored and never changes a planned unit.

**Counting.** A game counts by `main.anchors.results.unit_counts` with the forfeit-win guard. y(π's top) is the raw
game's result; every other y is its branch's. **A position enters iff S valued it and every one of its branches
counted.** A position with a branch that never counted is left out WHOLE and counts against `max_drop_frac`.

**One engine, one pin, one set of bytes.** Every game and every scored state on the registered engine
`port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130`, at the pin, on P\*'s sha256. Nothing is imported from an
earlier read.

## 4. The estimand and the decision rule

For a position p with legal actions a: q_c = Q̂(a) − mean_p Q̂, y_c = y(a) − mean_p y. Per panel,

β_panel = Σ_p Σ_a q_c · y_c / Σ_p Σ_a q_c².

β_U = the mean of β_H, β_F, β_P72; β_home = home72's; **Δβ = β_U − β_home**.

**Intervals: a percentile bootstrap over the TEAMS WE PILOT, within panel** (every position of every deal a team
played moves together), B = 2,000, one `numpy.random.default_rng(20261027)`, the U panels drawn first
(`heldout72_hpfix`, `forum72`, `p72`), then home72; the 2.5 / 97.5 percentiles. This departs from the design's deal
bootstrap: the registration tiers require a team-clustered interval on every Kakuna number, and a unit deals each of
its 10 teams twice. The deal bootstrap is REPORTED beside it (`beta_deal_clustered`).

| class | rule (`null_bar` 0.25, `margin` 0.25) |
|---|---|
| **RANK-NULL-ON-U** | UB(β_U) < 0.25. It dominates: Δβ's class is recorded beside it |
| **RANK-HOLDS-ON-U** | LB(Δβ) > −0.25 and LB(β_U) > 0 |
| **RANK-WORSE-ON-U** | UB(Δβ) < 0, and not RANK-HOLDS-ON-U |
| UNDECIDED | otherwise |

- **Rule 8.** Bars are compared exactly; a bound within 0.0005 of a bar that decides the class reads UNDECIDED
  (CONTINUE at look 1), flagged `boundary`.
- **Two looks.** Look 1 (100 blocks) reads RANK-NULL-ON-U only: NULL STOPS the read; anything else is CONTINUE and
  no other class is computed. Look 2 reads every class on all the blocks. A false NULL where β_U sits at the bar is
  ≤ 2 × 2.5 % over the two looks (the Bonferroni bound).
- **VOID** (named, before any estimate): counted raw units on two engines, an unknown one or another than the
  registered one, on other bytes than P\*'s, carrying a search stamp or a regime other than greedy; a state scored on
  other bytes or another preset, or without its per-world values or its `oracle_leaf8` record; a selected state
  whose hash, stratum or team is not its draw's; a counted raw unit with no selected state; planned branches that
  are not exactly the other legal actions of the searched states; a G-BRANCH failure that is not a classified peer
  drop; a counted branch whose capture is not the raw game's through the branch turn; a null control that changed
  its game (peer drops excepted); a branch that forced another action; more than 10 % (`max_drop_frac`) of a panel's
  searched positions incomplete. A VOID is repaired at its root cause and re-read, or the read ends VOID.
- **REPORTED, never decided:** β per panel and by stratum; the deal-clustered interval; the contender slope, the
  partial slope given π's logits, α_top, H and the piKL pick's IMP (the search-facing contrasts: reviews M6 / M3
  put their power near z ≈ 1.4, so they cannot confirm or refute anything here); the self-referee decomposition
  (the slope of Q_self on Q̂, of y on Q_self, corrected for the 8-seed noise); Q̂'s Monte Carlo reliability and β
  corrected for it; Σ q_c² a position; the legal-action count; the exclusion rate, the species hit, the share of
  deals with a stratum-B position; the peer drops; every part's CPU.

## 5. Stage 0 and the sizing rule (outcome-free)

Stage 0 scores and branches the first 5 blocks (≈ 60 positions) and `v_rank stage0` reads, with NO outcome: s̄ (the
mean Σ q_c² a searched position, both sets pooled), the exclusion rate, the legal-action count, G-BRANCH and the
teeth, and every part's CPU.

- **The sizing rule:** final blocks = 200 × min(2, max(1, 0.030 / s̄)), rounded up to a multiple of 20. Its output is
  written to `record/stage0.json` in this folder BEFORE look 1 is decided.
- **If s̄ < 0.015 the cap binds** (400 blocks, ≈ 240 CPU-h at the smoke's rate): the orchestrator decides whether to
  spend it before look 1 is decided. Stopping at the default 200 blocks is then a registered option; a size between
  the two is not.
- **Look 1 is 100 blocks whatever the rule says** (the design had it at half the final blocks; a fixed look keeps
  one committed spec). Stage 0's positions are look 1's first 5 blocks and enter the read like any other.
- **A stage-0 tooth, an exclusion rate over 15 % on any panel, or a branch drop rate over 3 %** HOLDS the read for
  the orchestrator before look 1's lanes are filled (MY JUDGEMENT; S′-HONEST's budget read 3 % exclusions on U,
  5 % on F).

## 6. Power, predictions and cost

**Power (the design's §5.2: a normal approximation from the self-referee probe on banked pool states, DEALS
INDEPENDENT).**

| positions a set | SE β_U | SE Δβ | P(NULL) at β_U = 0 / 0.10 | P(WORSE) at Δβ = −0.27 | P(HOLDS) at Δβ = 0 |
|---|---|---|---|---|---|
| 600 (look 1) | 0.099 | 0.141 | 0.71 / 0.33 | (not read) | (not read) |
| **1,200 (look 2, 200 blocks)** | **0.070** | **0.099** | **0.95 / 0.57** | **0.78** | **0.71** |
| 1,200 at the banked Kakuna spread (Σ q_c² 0.012) | 0.111 | 0.157 | 0.61 / 0.27 | 0.41 | 0.36 |
| 2,400 at that spread (the cap) | 0.079 | 0.111 | 0.89 / 0.48 | 0.68 | 0.61 |

False WORSE at Δβ = 0 is 2.5 %; false HOLDS at Δβ = −0.27 ≈ 1.5 %; a false NULL at the probe's β 0.54 ≈ 0.
**These are upper bounds on the registered rule's power.** The registered interval resamples teams: ≈ 38 distinct
teams a U panel at look 1 and ≈ 56 at 200 blocks (72 a panel, 10 a unit), ≈ 64 and ≈ 71 on home72. A team effect on
the slope widens the interval by an amount no banked data measures (UNVERIFIED); the read reports both intervals, so
its own design effect is read off the record. The entering positions are also ≈ 5–10 % fewer than the table's (1.95
positions a deal, exclusions, dropped positions).

**Predictions (GUESSES, before any game):** β_home ≈ 0.45, β_U ≈ 0.40, Δβ ≈ −0.05. RANK-HOLDS-ON-U 0.40,
UNDECIDED 0.38, RANK-WORSE-ON-U 0.12, RANK-NULL-ON-U 0.10 (the design's 0.45 / 0.33 / 0.12 / 0.10, HOLDS lowered for
the wider interval). s̄ ≈ 0.015–0.025.

**Cost (ESTIMATE from the build smoke: unregistered unit seed 2900, one unit each of H and home72, a live training
run beside it; 8 scored states, 44 one-game units).**

| part | measured | a selected position |
|---|---|---|
| a raw game in a 20-game unit | 9.0–10.8 CPU-s | ≈ 5 CPU-s |
| S at a searched state | 2.8–15.7 CPU-s, mean 6.3 | ≈ 6 |
| `oracle_leaf8` | 1.3–13.5 CPU-s, mean 8.0 | ≈ 8 |
| an honest-world exclusion (1 of 8 states; scored twice) | 205 and 219 CPU-s | ≈ 6–27 at a 3–12 % rate |
| a one-game forced unit (44 of them) | 21.7–30.1 CPU-s, mean 24.8; 25.8 s of wall | ≈ 150 (6 branches at the smoke's mean of 7 legal actions) |
| **all** | | **≈ 3.0 CPU-min (2.5–3.5)**, against the design's 2.4 |

| size | positions | CPU-h |
|---|---|---|
| stage 0 (5 blocks) | ≈ 60 | ≈ 3 |
| look 1 (100 blocks) | ≈ 1,170 | ≈ 60 (50–70) |
| **final (200 blocks)** | **≈ 2,340** | **≈ 120 (100–140)** |
| the cap (400 blocks) | ≈ 4,680 | ≈ 240 (200–280) |

Wall: the branch units are short-lane jobs, at most four heavy slots at once, so ≈ 6,700 units at look 1 are ≈ 12 h
on four shards when the slots are free, and the final ≈ 25–30 h (UNVERIFIED: the smoke ran two shards).

## 7. Look 2

If look 1 reads CONTINUE, a SECOND spec (`read_spec_look2.json`, committed here before any look-2 game) is this
spec with: `read` `v_rank_look2_2026-11-18`; groups `u1` and `home1` each carrying
`"import": {"state": "<A>/look1", "spec_sha256": "<the sha256 of this folder's read_spec.json>", "group": "u1" |
"home1", "models": ["pstar"]}`; fresh groups `u2` (unit seeds 2005 to 2000 + B/20 − 1) and `home2` (2115 to
2100 + 3·B/20 − 1), B = stage 0's final blocks; `u_groups: ["u1", "u2"]`, `home_groups: ["home1", "home2"]`,
`look: 2`; and each artefact list extended by look 2's own file (`selected2.jsonl`, `scored2`, `branches2`, made by
the same verbs on the fresh groups' raw states). Nothing else changes: the pin, the engine, the bars, the bootstrap,
`max_drop_frac`. The import path was executed on the smoke (a look-2 spec importing both groups reproduced the
look-1 numbers).

## 8. The launch (the orchestrator schedules it; nothing below has been run on a registered seed)

Rule 5b throughout: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, nice 19 on the agent cores, no GPU, no lease.

```bash
P=c933afd5a
W=/home/goodlad/dev/gen3ai-worktrees/vrank_play
git -C /home/goodlad/dev/gen3ai worktree add --detach $W $P
cd $W && scripts/bootstrap.sh --skip-env --no-check
export PYTHONPATH=$W/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CPUS=$($PY -m utils.core_split agent-cpus); LOW="nice -n 19 taskset -c $CPUS"
scripts/ops/mem_cap.sh --max-s 570 12 $LOW $PY -c \
  "from utils.rust_env.build import ensure_built; print(ensure_built('release'))"
M=/home/goodlad/dev/gen3ai/designs/research_state/measurements/v_rank_2026-11-18
A=/home/goodlad/gen3ai_archive/v_rank_2026-11-18/live; mkdir -p $A
ENG='port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130'
MODEL=$($PY -m main.baselines path pstar_plateau)
RAW="--raw-state $A/look1/anchor/u1/pstar --raw-state $A/look1/anchor/home1/pstar"

# 1. the raw games: 30 units x 20 games, ≈ 1.7 CPU-h
$PY -m main.reads check-spec --spec $M/read_spec.json
$PY -m main.reads plan --spec $M/read_spec.json --state $A/look1
(setsid nohup $LOW $PY -m main.reads run --state $A/look1 --lane anchor --hours 3 --concurrency 4 \
   > $A/raw1.log 2>&1 &)
PID=$($PY -m main.reads pid --state $A/look1 --lane anchor --wait 180)   # watch /proc/$PID
$PY -m main.reads assemble --state $A/look1                               # every unit counted, unplayed 0

# 2. the draw, once
$LOW $PY -m main.value_search.v_rank select $RAW --out $A/selected1.jsonl

# 3. stage 0: the first 5 blocks, no outcome read
scripts/ops/mem_cap.sh 8 $LOW $PY -m main.value_search.v_rank score --selected $A/selected1.jsonl \
  --model "$MODEL" --search r0s4_honest@10 --engine "$ENG" --out-dir $A/scored1 \
  --games 0-4 --unit-seeds 2000,2100,2101,2102
$LOW $PY -m main.value_search.v_rank plan --selected $A/selected1.jsonl --scored-dir $A/scored1 $RAW \
  --out $A/branches1 --shards 4
for k in 0 1 2 3; do (setsid nohup $LOW $PY -m main.x35.kakuna run --state $A/branches1/s$k \
   > $A/branch1_s$k.log 2>&1 &); done                                      # one supervisor a shard, each to its end
$LOW $PY -m main.value_search.v_rank classify --branch-state $A/branches1
$LOW $PY -m main.value_search.v_rank stage0 --selected $A/selected1.jsonl --scored-dir $A/scored1 \
  --branch-state $A/branches1 --params $M/stage0_params.json --out $A/stage0.json
# -> commit $A/stage0.json as $M/record/stage0.json (§5), BEFORE step 5

# 4. the rest of look 1: two scorers side by side (each resumable; rerun to resume), then the branches
for i in 0 1; do (setsid nohup scripts/ops/mem_cap.sh 8 $LOW $PY -m main.value_search.v_rank score \
   --selected $A/selected1.jsonl --model "$MODEL" --search r0s4_honest@10 --engine "$ENG" \
   --out-dir $A/scored1 --part $i/2 > $A/score1_p$i.log 2>&1 &); done
$LOW $PY -m main.value_search.v_rank plan --selected $A/selected1.jsonl --scored-dir $A/scored1 $RAW \
  --out $A/branches1 --shards 4                                            # extends; rerun as scoring advances
#   ... the step-3 supervisor loop again per shard, until every unit has ended ...
$LOW $PY -m main.value_search.v_rank classify --branch-state $A/branches1

# 5. look 1's decision
$LOW $PY -m main.reads decide --state $A/look1
$LOW $PY -m main.reads record --state $A/look1 --out $M/record
```

A lane's wait is one bounded watcher on its PID, never on `$!` (that is `setsid`'s): a branch supervisor's PID is
the last `supervisor_start` event of its shard's `units.jsonl`; a scorer's is found by its argv in `/proc`. No verb
above prints an outcome before step 5.

## 9. Deviations from the design, hazards and limits (FINDINGS)

1. **The pipeline is the dlive read's, not the design's one-process deal unit.** Raw games are whole 20-game units
   through `main.reads`; the shadow arms run offline on the kept log; every branch is a one-game unit. π's arm is a
   game inside a 20-game series while every other arm is a lone one-game replay: G-BRANCH, the capture prefix and
   the null controls are the check that the two are the same game (both smoke nulls were identical).
2. **Q̂ is `r0s4_honest@10`'s**, not the design's leaky `r0s4`: the honest search is a function of our information
   state and is the search that reached goal (b). The read is about that search's value, in turns 10–30.
3. **The interval is team-clustered** (§4), so §6's power is optimistic by an unmeasured amount. The opponent's team
   is not in the unit's records, so the cluster is one-way.
4. **Look 1 is fixed at 100 blocks** (§5).
5. **Not built:** the level-at-10 subgroup; the cross-engine comparison of raw games with X51 (fresh seeds).
6. **`oracle_leaf8` is a referee configuration of the scorer**, not a playable preset.
7. **A position with a dropped branch is excluded whole.** At the dlive read's 0.43 % a branch and ≈ 6 branches a
   position that is ≈ 3 % of positions (≈ 6 % on F), under the 10 % cap; whether a peer drop depends on the forced
   action is unmeasured, so the excluded positions are reported by panel and cause.
8. **The search-facing contrasts cannot be decided here** (reviews M6, M3): the slope is dominated by siblings π
   would never play, and RANK-HOLDS-ON-U licenses nothing about search.
9. **P72 is weak and procedural**; β per panel is reported, and an equal-weight β_U pulled down by P72 alone still
   reads as U.
10. **The cost is ≈ 25 % over the design's**, most of it one-game start-up; a multi-force deal unit would roughly
    halve the branch cost and is not built.
11. **The smoke's outcomes were seen** by its builder (7 positions on unit seed 2900, a pipeline check; the rule read
    UNDECIDED with two teams a panel). Seed 2900 is not in this read and none of its numbers is quoted as a result.
12. **A two-look read with a sizing rule**: the false-NULL bound is §4's; the other classes are read once.

## Errata (dated; operational command corrections only: no rule, number, bar, seed or class changes)

- **ERRATA 2026-11-19 (an operational command correction, NOT a rule change; found at look 1,
  `record/operator_notes.md`).** §8 step 5's `main.reads record --state $A/look1 --out $M/record` names a directory,
  but the verb's `--out` writes ONE file. Look 1's record was written with `--out $M/record/v_rank_look1.json`; look
  2's is `--out $M/record/v_rank_look2.json`. Nothing the read decides or reports changes.
