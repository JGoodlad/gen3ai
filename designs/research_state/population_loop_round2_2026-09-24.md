# Population loop, round 2 — PRE-REGISTRATION (2026-09-24)

**Status: REGISTERED before any round-2 number exists.** B2 launched at 17:43 PT 09-24. §§1–6 were
committed as `3265ec83` at 17:54 PT, when `models/ai_v13_27_popr2_loop/eval_results.jsonl` did not
yet exist (checked at 17:54:39). The follow-up commit adds the reader and side-check kit, the §4.4
rule script, and the findings from validating them. It changes no primary, bar, rule or branch.
**One declared amendment, made before any extension exists:** §4.4's row order S, C, B, I → **S, B,
C, I**, so that a rise in both readers is never read as RB alone closing (found while mechanising the
table). At 18:12 B2 still had no eval row. The round-2 generalists (B2, C2) were specified by round 1's
§5 branch N+ and are the Training Run session's launches. This session trained nothing, touched no
GPU, and wrote nothing under `models/`. The reader launch kit and the convergence side-check kit are in
[`measurements/population_loop_r2_2026-09-24/`](measurements/population_loop_r2_2026-09-24/README.md)
(PREPARED BUT NEVER EXECUTED).

**Template:** [`population_loop_round1_2026-09-23.md`](population_loop_round1_2026-09-23.md) (the
registration) and [its read](measurements/population_loop_r1_2026-09-23/read/README.md). Every rule of
evidence there carries over unless this file says otherwise, with the reason.

**The question, one round deeper.** B, the loop generalist, absorbed the two offense best responders it
trained against. A fresh best responder then found a gap on it 10 pp smaller than on the control, but that
difference was NOT DETECTED at the registered bar. **If B keeps training with that fresh best responder
(RB) added to its opponents, does the NEXT fresh best responder find a smaller gap on it (B2) than on the
control continued as long (C2)?** And across the rounds, does the loop's gap FALL round over round?

---

## 1. The arms, one line each (every code carries its description, every time)

| code | run name | what it is | depends on | GPU-h |
|---|---|---|---|---:|
| **G0** (the plateau parent) | `ai_v13_12_plateau` | exists; the round-0 target | — | 0 |
| **A** (round-0 offense exploiter of G0, 1.78× dose) | `ai_v13_18_teach5_offense_hidose` | exists; **gap +14.00 pp [+9.18, +18.55]**; the recipe every reader copies token-exactly | — | 0 |
| **A2** (A's seed-1002 replicate, the reader floor) | `ai_v13_26_popr0_exploit5_offense_s1002` | exists; **gap +12.00 pp [+7.15, +16.62]** | — | 0 |
| **B** (round-1 loop generalist) | `ai_v13_22_popr1_loop` | exists; G0 +8M vs {A, `ai_v13_13`} at share 0.40 | — | 0 |
| **C** (round-1 no-exploiter control) | `ai_v13_23_popr1_ctrl` | exists; B's argv at share 0.0 | — | 0 |
| **RB** (round-1 reader of the loop) | `ai_v13_24_popr1_read_loop` | exists; **gap +7.00 pp [+2.10, +11.76]**, curve still RISING (H-1) | — | 0 |
| **RC** (round-1 reader of the control) | `ai_v13_25_popr1_read_ctrl` | exists; **gap +17.00 pp [+12.25, +21.43]**, curve flat | — | 0 |
| **B2** (round-2 loop generalist) | `ai_v13_27_popr2_loop` | **LAUNCHED 17:43 PT 09-24** (launcher pid 1365781, pin `6eb9c776`). B continued +8M at B's own frozen 0.20× dose, stable set **{A, `ai_v13_13`, RB}** at share 0.40, PFSP on, retirement off (`--stable-opponent-mastered-wr 1.01`), no distillation, seed 1001 | B ✓, RB ✓ | ~4.5 |
| **C2** (round-2 no-exploiter control) | `ai_v13_28_popr2_ctrl` | C continued +8M, B2's argv with the share at **0.0** (the same three loaded and evaluated, never trained against). Queued behind B2 on the chain | C ✓ | ~4.5 |
| **RB2** (round-2 reader of the loop) | `ai_v13_29_popr2_read_loop` | a FRESH offense exploiter, A's recipe token-exact, re-pointed at B2's final | B2 finished | ~4.0 |
| **RC2** (round-2 reader of the control) | `ai_v13_30_popr2_read_ctrl` | the same, re-pointed at C2's final | C2 finished | ~4.0 |
| **RB+** (RB extended, convergence side-check) | `ai_v13_31_popr1_read_loop_ext` | a FORK of RB's final, same target (B), same frozen 1.78× dose, +4M (+50 %) | — | ~2.0 |
| **RC+** (RC extended, convergence side-check) | `ai_v13_32_popr1_read_ctrl_ext` | a FORK of RC's final, same target (C), same dose, +4M (+50 %) | — | ~2.0 |

**The B2 / C2 argvs are the Training Run session's** (`/home/goodlad/.claude/jobs/popr2_2026-09-24/argv_{B2,C2}.txt`,
diffed this session against round 1's `argv_B.txt`). **Only four values move from round 1:**
`--run-name`, `--model` (B's / C's final at 103,219,200), `--steps 103158272 → 111219200` (a TOTAL:
103,219,200 + 8,000,000, landing 111,280,128), and `--stable-opponents` gains
`models/ai_v13_24_popr1_read_loop/final_model.zip`. B2 and C2 differ only in the share (0.4 / 0.0),
the run name and the model. RB's five teams bind automatically: `resolve_stable_opponents` reads each
opponent's recorded `trainee_teams`, and A, `ai_v13_13` and RB record the same five offense teams
(`9eb3abdc52876a63, ac17a9dde5, a185b2d193, 9ba039ba8a, b904dbe059`). B2's launch banner confirms
it: `🐴 [STABLE] 3 cross-run opponent(s) … ext_ai_v13_24_popr1_read_loop … @111,280,128 [rung=explicit_zip]`,
`[ForkLR] … pinning LR to 2.80e-05 and FREEZING`, `[MATCHUP ef5242cffd]`, and the pool seeded with 20
snapshots from `ai_v13_22_popr1_loop`.

### 1.1 The mix B2 sees (from the code, as round 1 §2.3 derived it)

`sf = 0.90` (B's `win_rate_vs_bots` 92.75 % ≥ 0.80 ⇒ the 0.10 floor; B2's banner prints
`self_play_fraction=90%`). Stable slice = 0.9 × 0.40 = **0.36**, self-play pool 0.54, bots 0.10 — the
same as B. **Per specialist, the slice now splits THREE ways: ≈ 0.12 each**, against B's 0.18. PFSP
weights `max(0.05, 1 − rate)` at B's last-cycle rates (vs A 0.41, vs `ai_v13_13` 0.41, vs RB ≈ 0.40
from RB's endpoint 0.60) are ≈ 0.59 : 0.59 : 0.60, so ≈ uniform. **A finding (P-1): the exposure per
specialist fell by a third at a fixed share.** Round 1 chose 0.40 so each specialist got 0.18; round 2
at 0.40 gives each 0.12, which is between the split arm's 0.09 and round 1's 0.18. This is what §5's
round-1 rule specified, so it stands. The manipulation check (§4.1) is what tells whether it was enough.

### 1.2 Step arithmetic

A rollout is 48 × 2,048 = 98,304 steps; `--steps` on a fork is a TOTAL and lands on the next rollout
boundary (ceil).

| arm | fork step | `--steps` | lands | post-fork budget |
|---|---:|---:|---:|---:|
| B2, C2 | 103,219,200 | 111,219,200 | **111,280,128** | 8,060,928 |
| RB2, RC2 | 111,280,128 | 119,280,128 | 119,341,056 | **8,060,928 (A's)** |
| RB+, RC+ | 111,280,128 | 115,280,128 | 115,310,592 | **4,030,464 (+50 %, 41 rollouts)** |

---

## 2. What is carried, and what is re-decided

- **Reader recipe: A's, token-exact** (`ai_v13_18_teach5_offense_hidose`'s recorded `original_command`,
  with `--run-name`, `--model`, `--exploiter` and `--steps` changed). Frozen `--fork-lr 2.5e-4
  --fork-lr-freeze` (dose 3.815e-8 = 1.78× v8), seed 1001, the five offense teams,
  `--exploiter-keep-bots --exploiter-bot-fraction 0.5`, `--eval-battles 100`, `--eval-sentinel-greedy`,
  pin `6eb9c776`. **The budget, the dose and the regime match round 1's readers and A, or
  `main.best_response_gap` refuses.** That is also why the readers are NOT lengthened to answer H-1:
  a longer RB2/RC2 would refuse against A, RB and RC, and the round-over-round chain would break. H-1
  is answered by a separate side-check instead (§4.4).
- **The bar: F = 2.00 pp is REUSED, bar = max(F, 5.0) = 5.0 pp.** Why not re-measure F: (1) F is a
  property of the READER instrument (the recipe, dose and budget), and none of those change. (2) F
  enters only through `max(F, 5.0)`, so a re-measurement moves the bar only if a new seed pair differed
  by more than 5.0 pp. The one pair in hand differed by 2.00, and a 5.0 pp run-to-run spread would be
  2.5× what was seen. (3) It costs ≈ 4 GPU-h (a seed-1002 RB2′) on the critical path. **The honest limit:**
  F is ONE pair at ONE target (G0), so it is a two-replicate range, not a variance estimate. The floor
  at B2 or C2 is not measured. If round 2 reaches branch T, rule 22's replicate pair re-measures it for free.
- **Guards:** the same two, the same floors, the same KILL rule, but on B2 − C2 (§4.3).
- **New in round 2:** an EQUIVALENT row (§5; round 1 lacked one), the convergence side-check (§4.4),
  and a registered round-over-round read (§4.2.3).

---

## 3. The GPU queue (single GPU; the Training Run session owns every launch)

Measured on this box in round 1: generalists **≈ 4 h 05 m – 4 h 39 m** per +8.06M, readers **≈ 3 h 29 m
– 4 h 28 m**. Under round 1's CPU load (25–38 on 16 cores), A2 trained at 378 fps (H-8). So these ETAs
are central estimates with a ±1 h band per arm.

| order | arm | start ≈ (PT) | end ≈ | GPU-h | note |
|---:|---|---|---|---:|---|
| 1 | **B2** (round-2 loop generalist) | 17:43 09-24 (running) | 22:15 | 4.5 | chain `popr2_2026-09-24/chain.sh` |
| 2 | **C2** (round-2 control) | 22:15 | 02:45 09-25 | 4.5 | same chain, ADJACENT to B2 (matched contention) |
| 3 | **RB2** (reader of B2) | 02:45 | 06:45 | 4.0 | `launch_RB2.sh`; refuses until B2 is finished at 111,280,128 |
| 4 | **RC2** (reader of C2) | 06:45 | 10:45 | 4.0 | `launch_RC2.sh`; ADJACENT to RB2 |
| 5 | **RB+ then RC+** (convergence side-check) | 10:45 | 14:45 | 4.0 | the two ADJACENT (matched contention). Depends on nothing in round 2; may run anywhere the GPU is free, so long as the pair stays adjacent |
| | **total new GPU** | | | **≈ 21 (range 19–24)** | |

**Why the side-check goes LAST by default.** It does not change how round 2 is launched: the reader recipe
is fixed by the matching rule (§2), whatever the side-check shows. It changes only how round 2 is READ
(§4.4 table). Putting it last keeps round 2's primary on its earliest path. If the GPU would otherwise
sit idle (a block on B2/C2), the Training Run session may pull the pair forward; the pair is never split.

**CPU reads** (§4.3), incremental, start after C2 finishes and overlap RB2/RC2: G-U ≈ 3,200 new battles
plus the 1,600-battle reproduction check (~2–3 h at 4 workers); G-A 24 new 100-game units (~8.5 min each
under load, ≈ 3.5 h serial). The meter takes seconds. **Round-2 primary ETA ≈ 11:00 PT 09-25; the
side-check descriptor ≈ 15:00 PT 09-25.**

---

## 4. The read

### 4.1 Manipulation check — did B2 absorb what it was shown? (read FIRST)

B2 and C2 evaluate greedy-vs-greedy against all three specialists every cycle (100 games each,
`eval_results.jsonl` `externals`). **M2 = [B2's pooled rate over its LAST TWO cycles against all
three] − [C2's same], n = 600 per arm, Newcombe 95 %. ABSORBED iff M2's CI lower bound > 0.**
Descriptors beside it, never folded: per specialist (vs A, vs `ai_v13_13`, vs RB; n = 200 each), and
B2's per-cycle curve. **The one that matters most is vs RB**, the specialist B2 has not seen before.
If B2 absorbs A and `ai_v13_13` further but not RB, that reads as "absorbed the old ones", not "the
loop engaged at depth 2". Registered use: reported beside the branch; it does not change the branch
(M2 is the rule). Prior **P(ABSORBED) ≈ 0.55** — round 1 absorbed at 0.18 per specialist; round 2 has
0.12 (P-1).

### 4.2 PRIMARY — Δ2 = gap(RB2) − gap(RC2)

```bash
export PYTHONPATH=$PYTHONPATH:src
# 4.2.1 the matched gate: 0 mismatches (budget / dose / regime), or the read is VOID
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose \
    ai_v13_29_popr2_read_loop ai_v13_30_popr2_read_ctrl --check
# 4.2.2 THE read. --rounds is REQUIRED: B2 and C2 land on the SAME step (F2)
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose \
    ai_v13_30_popr2_read_ctrl ai_v13_29_popr2_read_loop \
    --rounds ai_v13_30_popr2_read_ctrl=2 ai_v13_29_popr2_read_loop=3 \
    --json <out>/brgap_r2.json --md <out>/brgap_r2.md
```

- **Statistic:** the `round 3 − round 2` offense row = **gap(RB2) − gap(RC2)**, stat `pooled` (4
  post-fork cycles × 100 = 400 games per reader), Newcombe 95 %. The tool's VERDICT line will read
  "UNREADABLE — fewer than two archetypes" (F1) and is NOT the verdict. The rule below is.
- **Bar = 5.0 pp** (§2).
- **Rule (unchanged):** OUTSIDE iff |Δ2| > bar AND the Newcombe CI excludes the bar point on that
  side. EQUIVALENT iff the whole CI sits inside [−bar, +bar]. Otherwise NOT DETECTED. **NOT DETECTED
  is never "equivalent", and never "no effect".**
- 🚨 **Power, stated before the number.** n = 400 per reader at rates near 0.6 ⇒ Newcombe half-width
  **≈ 6.7 pp** (round 1's realised CI [−16.60, −3.27] is half-width 6.67). **A detection needs Δ2 ≲
  −11.7 pp. Round 1's own point estimate (−10.00) would MISS again.** EQUIVALENCE needs a half-width
  below 5.0, which is **unreachable at n = 400** unless both rates sit near 0.9 or above. The row
  exists (§5) because the rule defines it, not because it can fire. Round 2, like round 1, tests for a
  LARGE effect.
  - **Orchestrator option, NOT the default (unchanged from round 1):** raise RB2's and RC2's
    `--eval-battles` to 200. The two readers stay matched to each other (half-width ≈ 4.8 pp;
    detection at Δ2 ≲ −9.8), but `cycle_games` then differs from A's, RB's and RC's, so the
    round-over-round chain (§4.2.3) needs `--allow-unmatched` and carries the confound in its header.
    Choosing it is a registration change and must be committed before RB2 launches.
- **Secondary (confirmatory, never folded):** `python -m main.best_response_gap <reader> --play 400
  --greedy` on RB2 and RC2, seed 0, concurrency 1. Its gap(RB2) − gap(RC2) must agree with the primary
  in sign for a detection to stand. Run only if the primary is OUTSIDE (round 1's D5: it cannot change
  any other branch).
- **Descriptors beside it, never folded:** gap(RC2) − gap(RC) (what a second plain block did; prior:
  inside the bar) · gap(RB2) − gap(RB) · each reader's per-cycle curve, with **its convergence read
  as in H-1**: the pooled rate over cycles 3–4 minus that over cycles 1–2. A rise of ≥ 5 pp on one
  reader and not the other is a named caveat on Δ2, as H-1 was on Δ1.
- **Prior:** P(OUTSIDE, below) ≈ **0.25**; P(NOT DETECTED) ≈ **0.68**; P(OUTSIDE, above) ≈ **0.07**;
  P(EQUIVALENT) < 0.01 (power). Up from round 1's 0.20 below, because Δ1's point estimate is −10.00.
  Capped by the power arithmetic above, by P-1, and by H-1 (part of Δ1 may be reader speed).

#### 4.2.3 The round-over-round question — does the loop's gap FALL across rounds?

This is the question `main.best_response_gap` exists for ("the loop is working iff the gap FALLS round
over round"). It is asked on the two CHAINS, each in its own invocation, with one reader per round. (F3's silent
collapse is FIXED at main, `26015897`. Two readers of one target now POOL as replicates, and two
readers of DIFFERENT targets in one round are a typed refusal. So mixing the chains in one
invocation fails loudly, but it still does not answer the question.)

```bash
# the LOOP chain: G0 → B → B2, read by A → RB → RB2
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose ai_v13_24_popr1_read_loop ai_v13_29_popr2_read_loop \
    --rounds ai_v13_18_teach5_offense_hidose=1 ai_v13_24_popr1_read_loop=2 ai_v13_29_popr2_read_loop=3 \
    --json <out>/brgap_loop_chain.json --md <out>/brgap_loop_chain.md
# the CONTROL chain: G0 → C → C2, read by A → RC → RC2
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose ai_v13_25_popr1_read_ctrl ai_v13_30_popr2_read_ctrl \
    --rounds ai_v13_18_teach5_offense_hidose=1 ai_v13_25_popr1_read_ctrl=2 ai_v13_30_popr2_read_ctrl=3 \
    --json <out>/brgap_ctrl_chain.json --md <out>/brgap_ctrl_chain.md
```

(The targets' steps already order the chains, 95.2M → 103.2M → 111.3M. `--rounds` is passed anyway,
so a tie-break can never reorder them. Both chains' round-1 halves were run at main this session:
`measurements/population_loop_r2_2026-09-24/validation/brgap_{loop,ctrl}_chain_r1half.txt`, −7.00
[−13.68, −0.23] and +3.00 [−3.58, +9.54], as round 1's read printed.) Each chain prints two per-round Newcombe deltas. The loop chain's
round-1 delta is already known: gap(RB) − gap(A) = **−7.00 [−13.68, −0.23]**. **Registered reading:
DESCRIPTOR, never the verdict.** The loop "falls round over round" iff the loop chain's `round 3 −
round 2` row is below zero, AND the control chain's same row is not. This is read as a pattern, not
tested, because (a) the round-1 half is already observed, so any test that folds it in was chosen
after seeing it (a forking-paths hazard), and (b) gap(RB) carries H-1: if RB was not converged, then
gap(RB) is biased low and gap(RB2) − gap(RB) is biased HIGH. **For the same reason, a pooled Δ over
rounds 1 and 2 (half-width ≈ 4.7 pp) is NOT registered as a test.** It may be printed as a descriptor
and labelled "includes data seen before registration".

### 4.3 KILL guards — B2 − C2 (both 16M past G0, so the contrast is CUMULATIVE over two loop rounds)

Both guards are read INCREMENTALLY (owner rule 2026-09-24, ORCHESTRATOR_SOP §2): minutes-long units,
durable per-unit rows, resumable, detached. The round-1 drivers are reused with only the refs changed.

- **G-U, the untaught 8.** Tree **`6eb9c776`** (the arms' pin), in its own worktree with the submodule
  initialised and the `dist` / `node_modules` links made (**H-9**: without them the teambuilder's Node
  validation fails with `Cannot find module …/deps/pokemon-showdown`), and its own release `sim_bridge`.
  The driver is round 1's `read/scripts/gu_driver.py` (the meter's `play_cells` loop body, pure per
  (ref, team, battle)). The refs are `popr2_loop=models/ai_v13_27_popr2_loop/final_model.zip`,
  `popr2_ctrl=models/ai_v13_28_popr2_ctrl/final_model.zip`, and
  **`plateau_b1=models/ai_v13_12_plateau/final_model.zip` re-run IN FULL as the reproduction check**.
  Settings: `untaught_meter_opponent` (`ai_v9_29_rev1_0823@24,000,000`), `--seed 0`, 200 games per
  team, concurrency 1, units of (ref × team × 25 battles).
  **Reproduction check: `plateau_b1` must read 963/1600 = 60.19 pp EXACTLY, or the cell is VOID.**
  Paired bootstrap over the 8 TEAMS, VERBATIM from round 1 (20,000 draws, seed 20260915). **Floor 3.69
  pp. KILL iff B2 − C2 is OUTSIDE and BELOW** (|Δ| > 3.69 and the CI excludes −3.69). Descriptors, which
  may reuse round 1's banked rows (same tree, same seeds, same pure function; the reproduction check is
  the license): B2 − B, C2 − C, B2 − G0, C2 − G0. 🚨 **Stated before the number:** round 1's B − C was
  −2.44 [−6.31, +0.88], within the floor but negative. A second −2.4 on top would put B2 − C2 near
  −4.9, OUTSIDE the floor by point estimate. The guard is cumulative by construction, and a KILL there
  is a real reading, not bad luck.
- **G-A, the external anchor.** Tree **`7c511161`**, NOT the pin. `6eb9c776`'s `main.anchors` predates
  `6e25c380` (`regime_verified_decisions`), `33338688` (`--server rust`) and `558586d0` (serial
  challenges), and without `9a189de8` a Metamon `competitive` team file HANGS (round 1 §0). Settings:
  `python -m main.anchors --model <final_model.zip> --opponent metamon:SmallRL --regime greedy
  --teamset {away,home} --games 100 --team-seed S --seed-base S --device cpu --nice 15`,
  `OMP_NUM_THREADS=1`, `--server rust`, one cell at a time, **S ∈ {0,10,20,30,40,50}**, the same S for
  B2 and C2 (round 1's D3). That is **24 units = 1,200 games per model**. The driver is round 1's
  `read/scripts/ga_driver.sh` with the models changed. Every row must have `regime_verified_decisions`
  true and `argmax_match_rate` 1.0. **KILL iff B2 − C2 (pooled 1,200, Newcombe) is OUTSIDE and BELOW
  the 0.110 floor.** Descriptors: B2 − B, C2 − C, B2 − G0, from round 1's banked units (same tree,
  same seeds). **H-3 carried:** a Metamon post-game `RecursionError` fires in BOTH halves (4 of 6 in
  round 1 were in the ACCEPTOR half, contrary to the SOP's H17). A complete half (50/50,
  `argmax_match_rate` 1.0, regime verified) with `peer_clean` false is COUNTED, with the unit named in
  the read. A half with missing games is re-run from game 1.

### 4.4 The CONVERGENCE SIDE-CHECK (owner default; the orchestrator's recommendation)

**It is NOT a re-verdict of round 1.** Round 1 stays **NOT DETECTED → N+**, whatever this returns.
Its numbers never enter round 1's registered row, and `best_response_gap` would refuse to put them
there (the budget is 4,030,464, not 8,060,928). It is a DESCRIPTOR of one question: **does round 1's Δ
survive once BOTH readers have trained 50 % longer?** Equivalently, was part of Δ1 = −10.00 reader
learning SPEED on a harder target (H-1: RB 0.52 / 0.59 / 0.57 / 0.60, still rising; RC 0.67 / 0.68 /
0.64 / 0.69, flat)?

- **What runs.** RB+ = `ai_v13_31_popr1_read_loop_ext` and RC+ = `ai_v13_32_popr1_read_ctrl_ext`.
  Each is a **FORK of the reader's own `final_model.zip`** (RB @111,280,128, RC @111,280,128) into a
  NEW run directory. It is **never a resume of the reader's own directory**: that would append cycles
  to RB's and RC's `eval_results.jsonl`, move their recorded budget, and make every round-1 and
  round-over-round read refuse. **Argv:** RB's / RC's recorded `original_command`, token-exact except
  `--run-name`, `--model` (the reader's final), and `--steps 115280128` (+4,000,000 ⇒ lands
  115,310,592 = **+4,030,464, exactly +50 %**). The `--exploiter` target is UNCHANGED (B's final for RB+,
  C's final for RC+). The dose is **named and frozen, as in the parents:** `--fork-lr 2.5e-4
  --fork-lr-freeze`, 1.78×. The seed is 1001. The teams, bot fraction, eval battles and regime are the
  same, and so is pin `6eb9c776`.
- **Matched-budget handling.** Both are extended by the SAME +4,030,464, so RB+ and RC+ are matched
  to EACH OTHER in budget, dose and regime. `best_response_gap --check` on the pair alone must print
  0 mismatches. They are NOT matched to A, RB or RC, so they are never read in the same invocation as
  those.
- **The read (descriptor):**
  ```bash
  python -m main.best_response_gap ai_v13_31_popr1_read_loop_ext ai_v13_32_popr1_read_ctrl_ext --check
  python -m main.best_response_gap ai_v13_32_popr1_read_ctrl_ext ai_v13_31_popr1_read_loop_ext \
      --rounds ai_v13_32_popr1_read_ctrl_ext=2 ai_v13_31_popr1_read_loop_ext=3 \
      --json <out>/brgap_ext.json --md <out>/brgap_ext.md
  # the table below, mechanised (reads RB, RC, RB+, RC+ through the meter's own reader)
  python designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/convergence_rule.py \
      --json <out>/convergence_rule.json
  ```
  **Δ+ = gap(RB+) − gap(RC+)**, stat `pooled` over the extension's eval cycles. At an eval every 2M on
  absolute steps, that is the cycles at 112M and 114M: **2 × 100 = 200 games per reader, half-width
  ≈ 9.5 pp.** Beside it, the six-point curve of each reader (the parent's 4 + the extension's 2), and
  **the late window**: the pooled rate over cycles 3–6 (400 games: the parent's last two cycles plus the
  extension's two) for each reader, and its difference, Δ_late (half-width ≈ 6.7 pp).
  - 🚨 **Honest limit, stated first:** convergence cannot be PROVEN at this n. A flat 200-game
    extension is consistent with a ±9 pp slope. This check can show a reader still CLIMBING; it cannot
    certify one has stopped.

| outcome (registered before any number) | condition | what it means for round 2's reading |
|---|---|---|
| **S — the Δ SURVIVES** | Δ_late ≤ −5.0 AND RB+'s pooled rate is within 5 pp of RB's own cycles 3–4 (0.585) | Round 1's Δ was not mostly reader speed. Round 2's primary reads as registered, and a T in round 2 needs no extra qualifier |
| **B — BOTH still climbing** | RB+ AND RC+ each ≥ their own cycles 3–4 + 5 pp | Neither reader converged at 8M. Δ compares matched budgets but not converged best responses, for either arm. Read round 2 as in C. Additionally the whole reader instrument's 8M budget is in question (A was flat, so this would be new) |
| **C — the Δ CLOSES** | RB+'s pooled rate ≥ RB's cycles 3–4 + 5 pp, AND Δ+ > −5.0 | Part of Δ1 was RB still learning. **Round 2's primary still governs by its rule**, but every reading of it carries "at an 8M reader budget": a smaller gap on B2 may mean *slower to exploit*, not *less exploitable*. **A T in round 2 is then a CANDIDATE that needs RB2+ / RC2+ (the same +50 % extension of RB2 and RC2, ≈ 4 GPU-h) BEFORE rule 22's replicate pair is spent.** And the orchestrator is asked whether round 3 (if any) should re-register a longer reader budget for ALL readers, A's included (a new round-0 A′ at 12M), since the chain cannot be matched otherwise |
| **I — INCONCLUSIVE** | none of the above (the expected case at n = 200) | nothing changes; the H-1 caveat stays attached to Δ1 as worded in round 1's read, and round 2's own reader curves (§4.2 descriptor) are the next evidence |

The rows are read in the order **S, B, C, I**; the first that applies governs. B comes before C so
that a rise in BOTH readers is never read as RB alone closing. RB's cycles 3–4 pooled = (57 + 60) /
200 = 0.585, and RC's = (64 + 69) / 200 = 0.665, from round 1's read. **The conditions are on POINT
estimates** (it is a descriptor), and every interval is printed beside them. At n = 200 against 200
a "rise ≥ 5 pp" is well inside noise (half-width ≈ 9.5), which is why no row moves a verdict.
**Mechanised:** `measurements/population_loop_r2_2026-09-24/scripts/convergence_rule.py` reads the
four runs through the meter's own `read_exploiter` and prints Δ+, Δ_late, both rises and the row.
It was exercised on weight-free stand-ins (`scripts/brgap_ext_standin_sim.py`, scenario counts, not a
prediction; `validation/convergence_rule_standin.txt`). There, `best_response_gap --check` on the
stand-in pair read **0 mismatches at budget 4,030,464**, the pair read gave half-width ≈ 9.3 pp, and
**RB together with an extension was REFUSED on budget** (8,060,928 vs 4,030,464), as registered.

---

## 5. Pre-registered branches — what each outcome means and what round 3 does

Read in this order; the first that applies governs. **The stopping rule carries: round 1 counted 1 of
3 (N+).**

| | condition | reading | round 3 |
|---|---|---|---|
| **V** | `--check` refuses (§4.2.1), or `plateau_b1` fails to reproduce 963/1600, or a STOP-list fails (B2's TB `train/stable_fraction` ≠ 0.36 or C2's ≠ 0.00 after the first eval; either's `matchup_hash` ≠ `ef5242cffd`; a reader not landing at 119,341,056) | the read is VOID | fix and re-run the failed piece; nothing is interpreted |
| **K** | a KILL guard fires (G-U or G-A: B2 − C2 OUTSIDE BELOW) | **the loop bought exploitability with generality, cumulatively over two rounds.** Δ2 is reported, never promoted | B2 is NOT a round-3 parent. Back to the orchestrator; no launch |
| **R** | Δ2 OUTSIDE, ABOVE | **the second round made the loop MORE exploitable** than the control | stop the loop at this configuration; counts toward the stopping rule (**2 of 3**) |
| **T** | Δ2 OUTSIDE, BELOW, the secondary agrees in sign, the guards are clean | **THE LOOP TURNS AT DEPTH 2 — a CANDIDATE** (rule 22: one pair, one seed). If §4.4 read **C** or **B**, it is a candidate at an 8M reader budget only | FIRST, if §4.4 read C/B: RB2+ / RC2+ (≈ 4 GPU-h). THEN the seed-1002 replicate at depth 2: B2′ (B +8M, seed 1002, the same set and share), C2′ (C +8M, seed 1002, share 0.0), and their readers at A's recipe (≈ 17 GPU-h); rule 3 re-measures the floor F from the two reader pairs. If it agrees: round 3 continues B2 with RB2 added, and the cross-archetype question opens (round 1 §2.3's runner-up set with balance and stall) |
| **N+** | Δ2 NOT DETECTED, manipulation ABSORBED (M2 lower bound > 0) | B2 beat what it saw, and a fresh best responder is not measurably weaker at this power, **for the second round running** | counts **2 of 3**. Round 3 = one round deeper again (RB2 joins B2's set, share 0.40, the same dose), and it is the LAST round at this configuration. Before round 3 is registered, the orchestrator decides whether to buy the power the question needs (the `--eval-battles 200` option, or a pooled 3-round read registered BEFORE round 3's numbers exist), because a third NOT DETECTED at n = 400 ends the loop on power alone |
| **M** | Δ2 NOT DETECTED, manipulation NOT ABSORBED | **the loop did not engage at depth 2.** Δ2 tested nothing about generalization. Most likely cause stated in advance: P-1, the per-specialist exposure fell 0.18 → 0.12 | counts **2 of 3**. Round 3 changes ONE lever on B2 and C2 together: **share 0.40 → 0.60** (per specialist 0.18 again, over three). If that also fails to absorb, D_g 0.20× → 0.39× (round 1 §2.2's runner-up) |
| **E** | Δ2 EQUIVALENT (the whole CI inside [−5.0, +5.0]); unreachable at n = 400 unless both rates are near 0.9 | **a positive null:** the second loop round moved the gap by less than the bar, WITH the evidence to say so. Distinct from N+ | counts **2 of 3**. Round 3 is NOT "one deeper" at the same configuration, since an equivalence says depth at this configuration buys less than the bar. It changes one lever instead: share 0.60 (as M). If §4.4 read C/B, an EQUIVALENT is qualified "at an 8M reader budget" like T |

**STOPPING RULE (carried, count updated):** three rounds in which Δ is not OUTSIDE BELOW (branches R,
N+, M, E) ⇒ **the loop does not turn at this parent.** Report it; the next question is a different
parent. **Round 1 counted 1. A round-2 outcome in {R, N+, M, E} makes it 2, and round 3 is the last.**
A branch-V round counts nothing and is re-run. A branch-K round ends the loop at this parent regardless
of the count (the loop's premise is that generality does not pay for it).

The §4.4 side-check changes NO branch and NO count. It changes only the qualifier on T and E and the
pre-condition on T's replicate spend.

---

## 6. Hazards carried forward, and new ones (each is a finding, not a footnote)

- **F1 — no VERDICT on a one-archetype round** (still true at main): the tool prints "UNREADABLE". The
  per-archetype row is read by the rule in §4.2.
- **F2 — siblings need `--rounds`.** B2 and C2 both land on 111,280,128; RB2 and RC2 on 119,341,056.
  Round 1 reproduced it LIVE (H-2: the unassisted listing REVERSED the mapping). Every invocation in
  this file passes `--rounds`.
- **F3 — FIXED at main (`26015897`), still honoured.** Same-target, same-archetype readers now pool
  as replicates, with the between-replicate spread printed; a different target in the same round is a
  typed refusal. Checked this session: A + A2 in round 1 pool to **+13.00 [+9.60, +16.28]**, spread
  −2.00 [−8.65, +4.68], and the round-1 primary reproduces EXACTLY at main, **−10.00 [−16.60, −3.27]**
  (`validation/brgap_r1_with_A2_pooled.txt`). The registered invocations still hold one reader per round.
  🚨 **RB and RB+ share a TARGET KEY (B's final @103,219,200)**, so in one invocation the tool would
  treat them as replicates if the budget check did not refuse first. They are never co-invoked.
- **H-1 — RB's curve was still rising.** Addressed by §4.4 (descriptor) and by the registered
  convergence descriptor on RB2/RC2 (§4.2). It never moves a verdict by itself.
- **H-3 — Metamon's post-game `RecursionError` fires in the ACCEPTOR half too** (4 of 6 in round 1).
  G-A counts a complete half and names the unit (§4.3). The SOP's H17 wording and `main.anchors`'
  `peer_recursion_upstream` stamp (only on a complete `peer_challenge` half) are unfixed at the time of
  writing: a doc and tool finding for the anchors owner, not fixed here.
- **H-9 — a pin worktree needs its submodule AND the `dist` / `node_modules` links** for G-U (Node
  validates the untaught teams). Round 1's registration worktree failed `git submodule update --init`
  (F11). The read's pin worktrees succeeded. Check both BEFORE starting G-U, or the first unit dies on
  `Cannot find module`.
- **The anchors tree is `7c511161`, not the pin** (§4.3). B2, C2 and round 1's banked B / C / G0 units
  all play on it, so any tree effect cancels in B2 − C2 and in every descriptor.
- **P-1 — the per-specialist exposure fell 0.18 → 0.12** at the carried share 0.40 (§1.1). The
  registered consequence is M's round-3 lever.
- **P-2 — the STOP-list item "[ModelVersion] Round-trip smoke test PASSED" is STALE** (Training Run,
  2026-09-24): a real launch never prints it; only the `--debug` smoke does. It is dropped from every
  round-2 STOP-list. Its absence proves nothing, but a `[ModelVersion] FATAL` still stops the launch.
- **P-3 — B2's stable set contains a READER.** RB is a best responder to B, not to G0, and it pilots
  the same five teams as A and `ai_v13_13`. So B2 trains against three offense policies on one five-team
  slice. RB2 is then the fourth best responder in a row on that same slice (F7 carried: a falling gap
  means "no longer loses this matchup", inside the offense subgame; generality is the guards' job).
- **P-4 — the matchup hash cannot confirm the readers are matched** (F12 carried): RB2, RC2, RB+, RC+
  each print a NEW hash by construction (the exploiter-target path is hashed). Only B2 and C2 must both
  print `ef5242cffd`, and B2 did. Reader matching is `--check`'s job.
- **P-5 — B2 and C2's argvs live in the Training Run session's job directory**
  (`/home/goodlad/.claude/jobs/popr2_2026-09-24/`), not in the repo. The measurement directory commits
  verbatim copies, so the registration does not depend on a scratch path.
- **P-6 — a reader extension is a FORK, so the ForkLR guard re-pins the rate. VERIFIED by executing.**
  RB's and RC's checkpoints already hold the frozen 2.5e-4, and the extension names the same value,
  so the fork is dose-neutral by construction. checkargs on the REAL extension argvs (parents and
  targets exist) gives the pinned parser at 129 / 2 / 0, `[ForkLR] ✓ … (--fork-lr 0.00025
  --fork-lr-freeze)`, and the ARCH surface clean. The dry-run reads `--steps 115,280,128 vs checkpoint
  at 111,280,128 (sidecar) → +4,000,000`, role `FORK of …/ai_v13_24_popr1_read_loop` (and of `…_25_…`
  for RC+). A fork does NOT carry the reader's eval history: `read_series` marks any row at or below
  `fork_step` as the parent's, so an inherited file could not be credited to RB+ anyway. No warm-start
  runs (`model_build.py` builds one only under `--warmstart-consensus`, absent from every reader argv).
  No self-play pool is involved (RB has no `snapshots/`: exploiter mode). ⚠️ RB and RC each also hold a
  `final_model_interrupted.zip` (07:10 and 10:41 PT 09-24, periodic-restart saves, OLDER than
  `final_model.zip`). The argvs name `final_model.zip` explicitly, and the launch script checks that
  the parent run is finished at 111,280,128.
- **K-1 — `main.best_response_gap` WRITES `best_response_gap.json` INTO CWD BY DEFAULT**
  (`DEFAULT_JSON`, `src/main/best_response_gap.py`). Run from the main checkout (it must run from a repo
  root, since recorded team paths are repo-relative), it dirties MAIN. **This session did exactly that
  once** (18:02 PT, three invocations overwriting one file). The file was moved out
  (`/tmp/popr2_stray_best_response_gap.json`) within a minute, and main is clean. **Every registered
  invocation that reads passes `--json <out>`.** `--check` writes nothing. A tool finding for the meter
  owner: default to no file, or refuse a cwd inside a git checkout's tracked tree.
- **K-2 — the RB2 / RC2 stand-in target is an EXPLOITER, not a generalist.** Their argvs were validated
  against `ai_v13_24_popr1_read_loop` @111,280,128 (the step B2 and C2 land on), because B2's and C2's
  finals do not exist. What the stand-in cannot validate: that B2's / C2's `model_config.json` inherits
  nothing a reader would trip on (round 1's F6, which came back clean for B and C). The real launch
  scripts re-run checkargs + `--dry-run` against the real target before launching.
- **K-3 — round 1's STOP-lists could not be re-verified from banners** (round 1 H-4: the banners are
  not in the child logs). The round-2 STOP-lists therefore add a post-first-eval item each (the
  `externals.ext_<target>` cell with counts [w, 100]) that CAN be checked from files.
</content>
</invoke>
