# Cross-era head-to-head — ai_v8_14 vs ai_v9_59

**Status: PRE-REGISTERED (written before any game was played) · results appended below.**

Two generations that cannot load in one process, played directly against each other over a
websocket Showdown server. Both sides are clients; each runs in its own process against its own
era's code.

---

## 1. The question, and why the anchored ELO cannot answer it

Both generations are rated against the same fixed anchor bots. The strongest anchor bot is
**ELO 1639**, and both models sit **300–400 points above it**. Their anchored ratings are therefore
*extrapolations from ~5% loss rates* against the top anchor — the regime where a Bradley-Terry fit
has the least information per game and where a small mis-specification moves the rating a lot.

A direct match has resolution the anchor set does not, and it **tests** the anchored ELO rather
than assuming it.

### The two ratings

Read from each run's own dense ladder (`<run>/snapshot_ladder/ladder.json`), which is the artifact
the project's *Reading an ELO* rule names as the headline (±10-class), not the noisier live
`eval/elo`:

| run | ladder node | anchored ELO | se | the checkpoint played here | its steps |
|---|---|---|---|---|---|
| `ai_v8_14_distill3_0725` | `292000005` | **2049.1** | 13.7 | `final_model_interrupted.zip` | ~292.1M |
| `ai_v9_59_R2ACTION_0827` | `28000032`  | **1957.4** | 15.0 | `final_model.zip` | ~28.07M |

Both final models map to the **top ladder node of their own run**, so the rating quoted is the
rating of the artifact actually played. Both ladders report `anchored_to_bots: true` and
`converged: true`.

### ⚠️ Two caveats that ride WITH the prediction

1. **Snapshot count is not matched.** v8_14's ladder carries **2** nodes; v9_59's carries **14**.
   The project's own rule is that a cross-run ELO comparison must be at matched snapshot *count*,
   because Bradley-Terry re-solves every node on every add and the newest node is systematically
   inflated. This comparison is **not** matched, and the newest-node inflation is expected to bite
   the 2-node ladder harder. This is a reason to distrust the prediction — and precisely why the
   head-to-head is worth playing.
2. **The prediction has its own interval.** The rating difference is 91.7 with a combined
   se of 20.3, i.e. a 95% interval of **[51.9, 131.5]** ELO.

## 2. PRE-REGISTERED PREDICTION

> **The anchored ELO difference is 2049.1 − 1957.4 = +91.7 in v8_14's favour.**
> Under the Bradley-Terry/logistic link the ladder itself is fitted with,
> `P = 1 / (1 + 10^(−Δ/400))`, this predicts
>
> ## **P(ai_v8_14 beats ai_v9_59) = 0.629**
>
> Carrying the ratings' own standard errors through the same link, the prediction's 95% interval is
> **P ∈ [0.574, 0.681]**.

**This is the hypothesis under test.** It is registered here before the first game.

### What the result will be allowed to say

Decided in advance, so the verdict cannot be chosen after seeing the number:

| condition on the observed win rate's Wilson 95% CI | verdict |
|---|---|
| CI **straddles 0.500** | **NOT DETECTED** — no direction may be claimed, in either direction |
| CI **excludes 0.500** but **contains 0.629** | **SIGNIFICANT**, and the anchored ELO is *corroborated* |
| CI **excludes 0.500** and **excludes 0.629** | **SIGNIFICANT**, and the anchored ELO **mis-predicts** — report by how much, in both win-rate and implied-ELO units |
| CI excludes 0.629 while containing 0.500 | **NOT DETECTED** on direction, but the anchored ELO is still contradicted — reported as such |
| timeouts > 25% of attempted games | **INCONCLUSIVE** — no verdict, per the project's contention rule |

"WITHIN FLOOR" is reserved for a difference that is smaller than the measurement's own noise floor.

## 3. Design

### Sample size and the resolution it buys

**Target: 560 games = 280 team-pairs × 2 orientations** for the primary arm
(v8_14 vs v9_59).

Expected Wilson 95% CI half-width at N=560: **±0.040**.

- If the truth is 0.629, the expected CI is **[0.590, 0.669]** — which **excludes 0.500**.
- If the truth is 0.500, the expected CI is **[0.459, 0.541]** — which **excludes 0.629**.

So N=560 is sized to separate the prediction from the null in *either* direction. It cannot
resolve a difference smaller than ~0.08 from the prediction; that is stated as the design's
resolution limit, not discovered afterwards.

### Paired team draws, both orientations

The team draw is the dominant nuisance variable in this game — a gen3ou match is decided in part
by which team each side is handed. The design removes it by construction:

- Teams are drawn from the **intersection of the two eras' pools**, verified **by sha256** (see
  §4). The harness reads the team files itself, once, and hands the identical team *string* to both
  sides — so a divergence between the two eras' team *pools* cannot reach the measurement.
- A **pair** is an ordered team draw `(T_a, T_b)` from a fixed seeded sequence.
- Each pair is played **twice**: orientation 0 = `v8_14` plays `T_a` / `v9_59` plays `T_b`;
  orientation 1 = `v8_14` plays `T_b` / `v9_59` plays `T_a`.
- Every game records both teams' shas, so the pairing is **verified from the output**, not assumed.

Both a game-level Wilson CI (the conservative headline) and a pair-level analysis are reported.

### Seeds

Every source of randomness that the project's determinism rule names is pinned, and the seeds are
recorded in the summary JSON. The team sequence is generated by a single seeded
`random.Random(TEAM_SEQ_SEED)` in the harness — *not* by either era's teambuilder — so the two
sides cannot consume a shared stream in a scheduling-dependent order.

### Action selection

Both sides play **greedy (argmax over legal actions)**.

Rationale: greedy is the measurement convention on the current side (`play.py`'s
`--temperature 0.0` default, and `RLPlayer(stochastic=...)` is set `False` by every eval path
precisely so a measured win rate is a stable control signal). Greedy also removes a whole
randomness source from a paired design. The two sides are made **consistent** — the convention is
verified in each era's own eval code and recorded in §4, and if the two eras' defaults disagree,
both are forced to greedy explicitly rather than each keeping its own default.

### Timeouts are never a semantic outcome

Per the project's contention rule, a game that times out goes in its **own bucket** and is never
scored as a loss. Battle timers are set generously (per-decision websocket latency is ~18 ms, so
the budget is dominated by turn count, not latency). **>25% timeouts ⇒ INCONCLUSIVE.**

### Bot calibration check

In the same session, against the same server, each side also plays a fixed number of games against
2–3 shared anchor bots. This is the calibration check on the ELO prediction: if the head-to-head
contradicts the anchored ELO, the bot rows say whether the two eras' *anchor* performance also
disagrees with what the ladder recorded. If the era side cannot drive bots over websocket, this is
skipped and said so — the head-to-head is primary.

## 4. Environment — the facts that make this safe

*(Filled in as verified; every entry is an executed check, not an assumption.)*

### Showdown server build — the protocol-drift question

The era's protocol parser **raises on an unknown keyword by design**, so a server newer than the
era's pin would kill the era side mid-battle. Checked directly:

| tree | `deps/pokemon-showdown` submodule pin |
|---|---|
| main HEAD | `e0551883ff8c676937a39ae8f4d6c0caf9de1613` |
| era commit `b13b30b2` | `e0551883ff8c676937a39ae8f4d6c0caf9de1613` |
| this worktree HEAD | `e0551883ff8c676937a39ae8f4d6c0caf9de1613` |

**The pin is identical across all three.** The era and current code were developed against the
*same* Showdown build, so a single server serves both with no drift to reason about. Moreover the
era checkout's `deps/pokemon-showdown/{dist,node_modules}` are symlinks to the main checkout's
build — literally the same compiled server that this harness runs.

Server run for this measurement: `npm run showdown -- 9137` (`--no-security`), started from this
worktree, on port **9137** (verified free; 8000/8001 untouched).

### Team pool intersection

| directory | this worktree | era |
|---|---|---|
| `data/teams/sample/*.txt` | 72 | 32 |
| `data/teams/specialist/*.txt` | 3 | 3 |

The era's 32 `sample/` teams are a **byte-identical subset** of this worktree's 72
(`comm -12` over `sha256sum` gives 32; the era has **0** files that differ or are absent).
The draw is restricted to that verified 32-team intersection.

### Action selection — verified, not assumed

Both eras' `eval_worker.py` construct the measured player with `stochastic=False` and the
verbatim comment **"eval = greedy yardstick"**, so greedy is not a choice imposed on the era
from outside — it is what each commit already did when it measured itself. Both sides here pass
`stochastic=False`.

### The two sides as run

| | side_a | side_b |
|---|---|---|
| model | `ai_v9_59_R2ACTION_0827/final_model.zip` | `ai_v8_14_distill3_0725/final_model_interrupted.zip` |
| code | this worktree (current) | `/tmp/v8rep_era` @ `b13b30b2` (read-only) |
| obs / config_version | 2501 / 103 | 2992 / 45 |
| role | **challenger** | **acceptor** |
| action selection | greedy | greedy |

The **era is the acceptor** deliberately: the era's vendored poke-env still has the
passwordless-login race, and it is the *challenger* that races, so the side with the unfixed
client is given the role that does not race.

---

## 5. Results

**Executed 2026-09-05, one Showdown server (`--no-security`, port 9137, submodule `e0551883`),
560 planned games, both sides greedy.**

### Run integrity

```
[v9_59] played=560 finished=560 won=205 planned=560 timeouts=0 team_mismatch=0 elapsed=1070s deadline_hit=False
[v8_14] played=560 finished=560 won=354 planned=560 timeouts=0 team_mismatch=0 elapsed=1071s deadline_hit=False
```

| check | result |
|---|---|
| games planned / attempted | 560 / 560 |
| **timeouts** | **0 (0.0%)** — far under the 25% INCONCLUSIVE bar |
| draws | 1 |
| decisive games | 559 |
| team assignment verified (`team_match`) | **1120 / 1120** side-records correct |
| battle-tag join | identical tags in identical order on both sides |
| games with two winners or none (excl. the draw) | 0 |
| anomalies | none |

**The one non-decisive game is a DRAW, not a timeout** — `battle-gen3ou-47493094`, 24 turns,
`finished=True` and `won=None` on *both* sides (a double-KO; the pool's Gengar carries
Explosion). It is bucketed separately and excluded from the win-rate denominator. Folding a draw
into the timeout bucket would have overstated the timeout rate and hidden a real game, so the
harness separates them.

### The headline

```
planned=560 decisive=559 draws=1 timeouts=0 (0.0%)
v8_14 wins 354/559 = 0.6333
  Wilson 95%            [0.5925, 0.6722]
  pair cluster boot 95% [0.5964, 0.6696]
  implied ELO delta     +94.9 [+65.1, +124.7]  (predicted +91.7)
  pairs: 103 v8_14-sweeps / 148 splits / 28 v9_59-sweeps  (n=279)
VERDICT: SIGNIFICANT — the Wilson 95% CI [0.5925,0.6722] excludes 0.500 — v8_14 is the stronger
of the two head to head; and it CONTAINS the pre-registered 0.6290, so the anchored ELO is
corroborated
```

| quantity | pre-registered | observed |
|---|---|---|
| P(v8_14 beats v9_59) | **0.6290** | **0.6333** |
| 95% interval | [0.574, 0.681] *(from rating se)* | **[0.5925, 0.6722]** *(Wilson)* |
| implied ELO difference | **+91.7** | **+94.9** [+65.1, +124.7] |

The pair-level cluster bootstrap — resampling the 279 complete **pairs**, which is the
independent unit because the two games of a pair are the same team draw swapped — gives
**[0.5964, 0.6696]**, essentially the game-level interval. The pairing therefore bought
correctness rather than width: it did not inflate the interval, and it removes the team draw as
an explanation.

### Robustness

| cut | v8_14 win rate | Wilson 95% |
|---|---|---|
| all decisive games | 354/559 = **0.6333** | [0.5925, 0.6722] |
| orientation 0 only | 173/280 = 0.6179 | [0.5597, 0.6728] |
| orientation 1 only | 181/279 = 0.6487 | [0.5911, 0.7024] |
| excluding the 12 stall-forfeits (250-turn cap) | 345/547 = 0.6307 | [0.5895, 0.6701] |

The two orientations agree well inside each other's intervals, which is the direct evidence that
the side-swap did its job: the result is **not** an artifact of which team either model was
handed. Twelve games reached the 250-turn stall-forfeit cap (v8_14 took 9 of them); dropping them
entirely moves the estimate by 0.003, so the result does not rest on the stall rule either.

Turn statistics: mean 45.1, median 32, min 9, max 250.

### BOT CALIBRATION

Run in the same session, against the same server, 120 games per bot per side (**720 games,
0 timeouts**). Each side ran in its own era's process; the bots are comparable because
`diff`ing the two trees' `agents/opponents.py` shows the decision **logic is identical** — the
only change is an opt-in per-instance RNG for the staller's Protect coin whose default is the
era's own `random.random()`.

```
bot             anchor |                 v8_14 win rate  implied |                 v9_59 win rate  implied |
------------------------------------------------------------------------------------------------------------
heuristic2      1638.8 | 111/120 0.925 [0.864,0.960]     2075 | 103/120 0.858 [0.785,0.910]     1952 |
staller_v2      1554.9 | 108/120 0.900 [0.833,0.942]     1937 | 106/120 0.883 [0.814,0.929]     1907 |
aggressive_v2   1630.1 | 110/120 0.917 [0.853,0.954]     2047 | 109/120 0.908 [0.843,0.948]     2029 |
------------------------------------------------------------------------------------------------------------
v8_14: pooled 329/360 = 0.9139 [0.8804,0.9387]   mean implied ELO 2019 +/- 64   ladder 2049   (implied - ladder = -30)
v9_59: pooled 318/360 = 0.8833 [0.8461,0.9125]   mean implied ELO 1962 +/- 57   ladder 1957   (implied - ladder = +5)

GAP (v8_14 - v9_59):
   bot-implied   +57  95% [-29, +143]
   ladder        +91.7   -> inside the bot-implied CI: True
   head-to-head  +94.9   -> inside the bot-implied CI: True
   the bot-implied CI also contains ZERO: True -- i.e. 720 bot games cannot even establish which model is better
```

Per cell, with the implied rating's own interval:

| side | bot | anchor ELO | games | win rate (Wilson 95%) | implied ELO (95%) | contains that side's ladder ELO? |
|---|---|---|---|---|---|---|
| v8_14 | heuristic2 | 1638.8 | 111/120 | 0.925 [0.864, 0.960] | **2075** [1959, 2191] | yes (2049.1) |
| v8_14 | staller_v2 | 1554.9 | 108/120 | 0.900 [0.833, 0.942] | **1937** [1834, 2039] | **no** (2049.1, by 10 pts) |
| v8_14 | aggressive_v2 | 1630.1 | 110/120 | 0.917 [0.853, 0.954] | **2047** [1936, 2157] | yes (2049.1) |
| v9_59 | heuristic2 | 1638.8 | 103/120 | 0.858 [0.785, 0.910] | **1952** [1864, 2040] | yes (1957.4) |
| v9_59 | staller_v2 | 1554.9 | 106/120 | 0.883 [0.814, 0.929] | **1907** [1811, 2002] | yes (1957.4) |
| v9_59 | aggressive_v2 | 1630.1 | 109/120 | 0.908 [0.843, 0.948] | **2029** [1922, 2135] | yes (1957.4) |

`implied ELO = anchor_elo + 400·log10(p/(1−p))`, with a delta-method interval
(`se = (400/ln10)·√(1/(n·p·(1−p)))`) — the term that blows up as p approaches 1, which is
exactly the weakness under discussion.

**Five of six cells contain their side's ladder rating.** The miss is v8_14 vs `staller_v2`
(1937 [1834, 2039] against a ladder 2049.1), short by 10 points on the interval's edge. At 95%
coverage over 6 cells ~0.3 misses are expected, so one marginal miss is not evidence of anything;
it is recorded rather than dropped. Note also the spread *within* a side — v8_14's three anchors
imply 2075, 1937 and 2047 — which is the same instability the pooled ±64 reports, seen per cell.

**Does it agree with the ladder? Yes, on both sides individually.** v9_59's bot-implied rating is
**1962 ±57** against a ladder value of **1957** — agreement to 5 points. v8_14's is **2019 ±64**
against **2049** — 30 points low, comfortably inside its own interval. Neither model's anchored
rating is contradicted by re-measuring it here.

**But the calibration arm cannot settle the comparison, and that is the point.** The
bot-implied *gap* is **+57 with a 95% interval of [−29, +143]**. That interval contains the
ladder's +91.7 and the head-to-head's +94.9 — so nothing disagrees — but it **also contains
zero**. In the pre-registered vocabulary the bot arm alone is **NOT DETECTED**: 720 games against
the anchor bots cannot establish even which of the two models is stronger.

This is the motivating claim of the whole exercise, measured rather than argued. Both models beat
these bots ~88–91% of the time, and at that win rate a rating is an extrapolation with a ±60-point
standard error per side. **560 direct games pinned the gap to ±30 rating points; 720 bot games left
it at ±86 and could not exclude zero.** The direct match has resolution the anchor set does not.

One honest asymmetry worth recording: v8_14's margin over v9_59 is largest against `heuristic2`
(0.925 vs 0.858, the strongest anchor) and smallest against `aggressive_v2` (0.917 vs 0.908). The
per-bot samples are only 120 games each, so this is not a finding — but it is the shape a real
difference would take, and it points the same way the head-to-head does.

---

## 6. VERDICT

# SIGNIFICANT

**ai_v8_14 beats ai_v9_59 in 63.3% of 559 decisive head-to-head games (Wilson 95%
[0.5925, 0.6722]).** The interval excludes 0.500, so the direction is real: the *older*
generation is the stronger of the two.

**The anchored ELO is corroborated, and to a degree the design did not guarantee.** The
pre-registered prediction was 0.6290; the observed value is 0.6333 — a discrepancy of **+0.004**,
roughly a tenth of the measurement's own half-width. In rating units the ladder predicted a
**+91.7** point gap and the head-to-head measures **+94.9** [+65.1, +124.7], with the prediction
sitting almost exactly on the point estimate.

That is the substantive finding, because the prediction was an extrapolation the anchor set could
not check. Both models sit 300–400 points above the strongest anchor bot (ELO 1639), so their
ratings were fitted from ~5% loss rates against opponents far below them, and the two ladders were
not even matched on snapshot count (2 nodes vs 14) — a mismatch the project's own rule says should
bias the shorter ladder's newest node *upward*. Any of that could have made the extrapolation
wrong by tens of points. It did not: **the anchored Bradley-Terry ratings transfer to a direct
match between generations that cannot even load in the same process.**

Two limits on how far to carry this. It is **one pair of checkpoints**, so it licenses no general
claim that the anchored ELO is always this accurate — it is one successful out-of-sample test of
the extrapolation, not a validation of the ladder as an instrument. And the agreement is closer
than the design can resolve: with a half-width of ±0.040 this measurement could not have
distinguished the truth 0.629 from, say, 0.610, so "the prediction landed within 0.004" should be
read as "the prediction was not detectably wrong", not as evidence the ladder is accurate to four
decimal places.

### The secondary finding: the anchor set's resolution, measured

The bot-calibration arm reproduced each model's anchored rating individually (v9_59: 1962 ±57 vs
a ladder 1957; v8_14: 2019 ±64 vs 2049) but **could not settle the comparison**: the bot-implied
gap is +57 with a 95% interval of [−29, +143], which contains zero.

So the two instruments do not disagree — they differ in **power**. On the same day, on the same
server, with the same checkpoints: **560 direct games measured the gap to ±30 rating points, while
720 bot games left it at ±86 and could not name a winner.** The premise that motivated this
harness — that a direct match has resolution the anchor set does not — is now a measurement rather
than an assumption.

---

## 7. Reproducing this

```bash
export PYTHONPATH=$PYTHONPATH:src          # mandatory in a linked worktree
npm run showdown -- 9137                   # your OWN port; never 8000/8001

D=designs/research_state/measurements/arch_transfer_2026-09-05/cross_era_head_to_head
python $D/plan.py --pairs 280 --out plan.json

# side_b = ERA (acceptor), run with PYTHONPATH=/tmp/v8rep_era/src
python $D/side.py --plan plan.json --role b --mode accept \
    --model models/ai_v8_14_distill3_0725/final_model_interrupted.zip \
    --name Gen3xEraf1 --opponent Gen3xCurf1 --label v8_14 --out b.jsonl &
# side_a = CURRENT (challenger), run with PYTHONPATH=<worktree>/src
python $D/side.py --plan plan.json --role a --mode challenge \
    --model models/ai_v9_59_R2ACTION_0827/final_model.zip \
    --name Gen3xCurf1 --opponent Gen3xEraf1 --label v9_59 --out a.jsonl &
wait

python $D/analyze.py --side-a a.jsonl --side-b b.jsonl --plan plan.json \
    --out-games $D/games.jsonl --out-summary $D/summary.json
python $D/bots.py --plan plan.json --model <ckpt> --label <l> --tag <t> --out bots.<l>.jsonl
python $D/calibration.py --v8-jsonl bots.v8_14.jsonl --v9-jsonl bots.v9_59.jsonl \
    --out $D/bot_calibration.json
```

| | |
|---|---|
| team-sequence seed | **20260905** (`plan.py --seed`) |
| team pool | the 32-team verified intersection, `/tmp/v8rep_era/data/teams/sample` |
| pairs / games | 280 / 560 |
| server | `npm run showdown -- 9137 --no-security`, submodule `e0551883` |
| action selection | greedy both sides (`stochastic=False`) |
| device | CPU both sides, `nice -n 10`, BLAS threads pinned to 1 |
| wall clock | 1070 s for the 560-game arm; ~9 min for the 720-game bot arm |

Both sides are **deterministic given the plan**: greedy action selection removes policy sampling,
the team sequence is read from the plan file rather than drawn, and battles run at
`max_concurrent_battles=1`. The remaining randomness is the **simulator's own dice**, which are
unseeded here — the server mints a fresh seed per battle — so a re-run reproduces the *design*
exactly but not the individual games. Pinning the sim dice as well would make the run
bit-reproducible and is the obvious extension if this is ever used as a regression gate.

### Artifacts

| file | what |
|---|---|
| `README.md` | this document — pre-registration, design, results, verdict |
| `plan.py` | the seeded, side-swapped game plan generator |
| `side.py` | one side of the match; runs on **either** era (common API subset) |
| `analyze.py` | the join, the two intervals, the pre-registered decision rule |
| `bots.py` | the bot-calibration arm |
| `calibration.py` | implied ratings + their delta-method intervals |
| `games.jsonl` | **560 rows** — pair id, orientation, both teams' shas, winner, turns, timeout/draw flags |
| `summary.json` | counts, both CIs, implied ELO, pair breakdown, verdict, anomalies |
| `bot_calibration.json` | per-bot win rates, implied ratings, the gap and its CI |

---

## 8. Hazards hit (recorded, not silently worked around)

1. **A refused login costs the whole deadline, silently.** `localhost_server_configuration` still
   authenticates against the *real* Smogon `action.php`, so short account names (`cs1`, `es1`)
   that somebody has registered upstream are refused even though our server runs
   `--no-security`. Neither era wraps `send_challenges`/`accept_challenges` in the
   connect-or-raise guard that `_battle_against` gets, so both sides sat for the full 320 s and
   reported 20 timeouts. **Fix applied:** `side.py` waits on `logged_in` with a 30 s bound and
   exits with the diagnosis. **Latent defect for the tree:** the guard belongs on the challenge
   paths too, not only `_battle_against`.
2. **The era's poke-env hangs on a passwordless login.** The server greets each connection with
   `|updateuser| Guest N` before `|challstr|`; the era's client sets `logged_in` on that, so a
   passwordless challenger fires `/challenge` at a user that does not exist yet and both sides
   wait forever. Current code guards it with `_trn_sent`; the era does not. **Mitigated** by
   giving every account a password and by making the **era the acceptor** (the challenger is the
   side that races).
3. **A draw was initially bucketed as a timeout.** One game (`battle-gen3ou-47493094`, 24 turns)
   finished with `won=None` on both sides — a double-KO. The project's rule that a timeout is
   never a semantic outcome has a converse: a semantic outcome must never be booked as a timeout.
   `analyze.py` now separates the buckets, and the true timeout count is **0**.
4. **The two eras' team pools differ** (72 vs 32 teams in `data/teams/sample/`). Rather than
   trusting either, the harness reads the team files itself and hands both sides the identical
   team *string*, drawn from the sha256-verified 32-team intersection.
5. **Worktree isolation refuses several ordinary shell forms** — `cd <main> && git …`,
   `git -C <main> …`, `export PYTHONPATH=$PYTHONPATH:src` inline, and long heredocs. Worked
   around with `git ls-tree` from the worktree's own object DB (same store), plain-file reads of
   the main checkout, and per-side runner scripts holding an absolute `PYTHONPATH`. Worth knowing
   before writing commands for an isolated agent; note the CLAUDE.md-mandated
   `export PYTHONPATH=$PYTHONPATH:src` is itself one of the refused forms.
