# X65 ExIt-0: teach P\* what the search S′ plays, and test whether it learns it

**Status: REGISTERED 2026-11-05** ([`../research_state/measurements/x65_exit0_2026-11-05/PREREG.md`](../research_state/measurements/x65_exit0_2026-11-05/PREREG.md)).
**Builds B1–B5 BUILT and REVIEWED 2026-11-05** (§7; package `src/main/x65/`, the read rule
`main.reads.rules_distill_play`; [`REVIEW.md`](../research_state/measurements/x65_exit0_2026-11-05/REVIEW.md): TRAIN,
every finding disposed in PREREG Amendments 1 and 2). **The bank is built and registered**: 268,905 decision rows,
97,395 label rows, every one verified against its `frame_sha256` (manifest sha256 `d13fd55f…2b7a`). **The play pin
is `320a535d4`**, whose play path is byte-identical to S′-LIVE's `0c76b521e` and whose `main.reads` plans the specs.
Nothing is trained or played: training goes to the GPU session on the orchestrator's word. The specs' remaining
`PENDING` values (the student zips and sha256s, the pin, the later looks' import hashes) are filled once the students
are exported.

**Terms.**
- **P\*** is X26 @280M (`pstar_plateau`).
- **S′** is the one-ply search `r0s4_rolls@10`, the one S′-LIVE played. On each decision it runs 8 worlds with
  `dice="rolls"`, scores each action by 4-turn playouts (8 per action, critic at the leaf), and acts by piKL at
  λ = 0.015.
- **U** is the three unfamiliar panels: H (`heldout72_hpfix`), F (`forum72`) and p72.
- **ExIt** is expert iteration: search improves the policy's choices, and the network is trained toward them.
- **π′** is S′'s piKL target, π′ ∝ π · exp(Q/λ). Its mode is the action S′ plays.

## 0. The answer in one paragraph

**Recommended:** distil S′ into P\* from the 97,601 searched decisions S′-LIVE has already banked. Each banked decision
holds Q̂ for every legal action and P\*'s log-probabilities, so π′ can be computed exactly. The game's capture rebuilds
the observation bit for bit: 976 of 977 frames matched in a probe (§4.1). So the labels cost **no new search CPU**.

The design fits one P\* student per panel fold. Each student is trained on two panels' labels with a KL anchor to P\*.
It is then read against the real Kakuna on the third panel, whose teams it never saw. The control at look 1 is
S′-LIVE's banked P\* games on the same deals.

**Cost.**
- GPU: ≈ 0.5 GPU-h.
- Kakuna lane: ≈ 23 CPU-h at look 1, ≈ 96 CPU-h at most with look 2.
- Builds: ≈ 2.5–3 agent-days.

**Why it is worth running whatever S′-HONEST finds.** The student plays with no search and sees only its own
observation, so its result is leak-free by construction, whatever its labels knew. What its labels knew (S′'s leaks)
can only make them worse targets.

**Power.** 0.78–0.82 at a true +2.3 pp, half of S′'s gain (§5).

**The GPU stays mostly idle under this plan** (§6). No candidate keeps it busy for days without one of three things:
the owner's yes, a multi-day build, or slowing S′-HONEST's live read when the two overlap. The days-long GPU work is
**ExIt-1** (§5.1), ≈ 1.5–2.5 GPU-days, funded only by ExIt-0's GO together with the placebo's SEARCH-INFO
(Amendment 1). It is sequenced to finish inside S′-HONEST's
6–8-agent-day build.

## 1. The situation this answers (2026-11-05)

- **The GPU has been idle since 11-03.** Growing the network at the plateau reads NO-GO three times: X47 depth, X48
  per-kind experts and N4 FFN width. The breadth family is NULL or UNDECIDED: D1, D3, N1-P and X63 Stage A. Human
  labels are worse than self-play (X59-S).
- **Search is the one measured lever with a game-level gain.** S′-LIVE read GO-LEAKY: G_U +4.60 pp [+2.66, +6.48],
  and the searched arm scores 0.545 on U. But S′ is leak-inclusive. [`design_sprime_honest.md`](design_sprime_honest.md)
  lists six leak channels, and on S′-LIVE's decisions a Speed is pinned in 23.5 % and a hidden counter is live in
  6.9 %. S′ also costs ≈ 4.6 search CPU-min a game.
- **CPU is the binding limit, and the agent cores are the Kakuna lane.** S′-HONEST's build is ≈ 6–8 agent-days. Its
  live read is then ≈ 450–960 CPU-h, about 3–14 days at concurrency 6. A training run cuts the lane's pace sharply:
  - the lane falls back to 2 units at a time (AM-K6's revert clause, root rule 5b);
  - S′LIVE-OP-5 measured 3–5 deals an hour at that setting during a live (toy) run, against 25–50 at concurrency 6
    with no run;
  - a production run's effect is UNVERIFIED (FINDING X65-F1).

So a GPU run that overlaps S′-HONEST's play costs that read roughly 5–10× in wall time. The window between now and
S′-HONEST's play is the GPU's.

## 2. The candidates, weighed with measured costs

| candidate | GPU | CPU (agent cores) | ready in | P(useful) and why | verdict |
|---|---|---|---|---|---|
| **(i-a) ExIt-0: distil S′ from the BANKED labels, supervised, panel cross-fit** | ≈ 0.5 GPU-h (X59-S measured ≈ 0.4 GPU-h for a 1.2M-row, 3-epoch student; this one is ≈ 1M row-passes) | label bank ≈ 1 CPU-h (measured 0.012 CPU-s a frame, §4.1); read ≈ 23 / ≤ 96 CPU-h (S′-LIVE's raw units: 29 CPU-s each) | ≈ 1.5 agent-days to the first GPU minute, ≈ 3 to the first game | GUESS 0.35–0.5 for a GO. One supervised step on ≈ 65k labels of S′, whose gain is +4.6 pp; the leak-share is unknown | **RECOMMENDED** |
| (i-b) ExIt from FRESH offline labels (self-play states, S′ re-run) | as (i-a) | **9.5 CPU-s a label**, measured: S′-LIVE's 258 search CPU-h ÷ 97,601 labels (worlds 138, exclusions 49, playouts 66). 100k labels ≈ 260 CPU-h, plus the games that host them. The honest builder is ×1.2–1.5 on worlds (`design_sprime_honest.md` §4) | days | same as (i-a), and later | **REJECTED now:** it costs as much CPU as S′-HONEST's read, on the same cores. The bank gives 97.6k labels free |
| (i-c) the in-loop search factory (`design_value_discrimination.md` §13) | +3–12 % GPU (its ESTIMATE) | at f = 0.25 % of a production run's ≈ 1,333 decisions/s, ≈ 3.3 labels/s × 9.5 CPU-s ≈ **32 cores**. §13.5's ESTIMATE of 0.08 CPU-s a world was ≈ 60× low; measured 5.07 CPU-s a search | 6–9 agent-days | — | **REJECTED:** about 10× over the box's 3 agent cores |
| (i-d) ExIt-1: PPO + an auxiliary KL to the bank’s π′ (§5.1) | ≈ 35 GPU-h (8 × P\*+20M at 1,333 steps/s = 4.4 GPU-h each) | the forks' own collectors; read ≈ ExIt-0's | ≈ 2 agent-days of build (a trainer loss over an offline label buffer) | conditional on (i-a) | **the GPU-days follow-on, funded by ExIt-0's GO and the placebo's SEARCH-INFO** |
| (ii) `D_MODEL` growth at P\* | ≈ 26 GPU-h (3 + 3 forks) | ≈ 6,000 Kakuna F games (≈ 50 CPU-h) | `D_MODEL` is not a growable field: widening the residual stream through LayerNorm is not function-preserving by zero padding. A build of 2–3 agent-days (GUESS) | low: the plateau read NO-GO for depth (X47), experts (X48) and FFN width (N4, "used, buys nothing") | **REJECTED**: three capacity NO-GOs, a build, and a multi-day CPU-heavy run |
| (iii) from-scratch scale (a wider network from step 0) | ≈ 8–12 GPU-days. X26's 380M took ≈ 79 GPU-h at 1,333 steps/s; a 2× `D_MODEL` run is GUESS 1.5–2.5× slower a step, plus a same-pin 1× control | the run's collectors for the whole span, overlapping S′-HONEST's play | needs the owner | N4's NO-GO removed its cheap support (not refuted) | **NOT the winner**: the owner's question stays open (STATUS §6) |
| (iv-a) X60-T, human-built teams in the pool (`design_human_replay_lever.md` §6.4) | ≈ 20 GPU-h (5 × 4.1) | ≈ 6,000–12,000 Kakuna games | its training clause is PROPOSED, not built; REVIEW F's fixes are owed | GUESS 0.2: a breadth lever, and its sibling D1 read +2.10 [−0.92, +5.11] | **RUNNER-UP** (§3) |
| (iv-b) a league or exploiter arm | days | days | an R0-style build | A1 and SPRT NULL; exploiters are counterplay, not skill | rejected |
| (iv-c) "fill the GPU": a fresh same-pin C0 now | 13 GPU-h (3 seeds) | collectors | 0 (argv validated, §8) | ≈ 0 alone. It is useful only as ExIt-1's control, at ExIt-1's own pin, which does not exist yet | rejected now |

## 3. Why ExIt-0 wins, and why X60-T is the runner-up

1. **It is the only lever aimed at a measured +4.6 pp.** Every other candidate starts from an UNDECIDED or NULL
   family. A network that keeps half of S′'s gain on unseen teams (+2.3 pp) closes about half of goal (b)'s ≈ 4 pp
   gap with no search at play time.
2. **It is the first rung of the owner's goals 4 and 5.** The AlphaZero loop is search improves, the network
   distils, then the stronger network is the next search's prior. Whether P\* can distil S′ is the question every
   later rung depends on. It has never been asked here: the in-loop factory was never built, and the distillation code
   was deleted in L3.
3. **It is nearly free of the binding resource.** The labels are already paid for (292 CPU-h, by S′-LIVE). The
   training is GPU-minutes and about one core. Look 1's control is banked. Its ≈ 23 CPU-h look 1 fits inside
   S′-HONEST's build window several times over.
4. **It is worth running whatever S′-HONEST finds** (§4.4). An honest read of a network trained on leaky labels is
   honest. If S′-HONEST reads GO, ExIt-0 says whether the gain can be banked into the network. If it reads NULL,
   ExIt-0 is a second, independent test of whether S′'s choices carried any honest, learnable signal. A GO there would
   mean training labels may come from a leaky (cheaper) builder.

**X60-T is the runner-up** because it is the registered human-data route and the only multi-day GPU design already
written. It loses on four counts:
- its training clause is not built and REVIEW F's fixes are owed;
- the breadth family's priors are weak;
- its read is 6,000–12,000 Kakuna games on the same lane;
- it does not compound with search.

It becomes the GPU's work if ExIt-0 reads NOT-DELIVERED or NULL.

## 4. The design (the PREREG owns the numbers; this is the why)

### 4.1 The label bank: zero search CPU, guarded bit for bit

- **Each label.** S′-LIVE's counted searched units record every searched decision in `search_decisions.jsonl`
  (schema `gen3_live_search_decision_v1`, `src/main/anchors/search_side.py`). A record holds `q` and `log_pi` over
  the legal actions, `played`, `frame_sha256` (the sha256 of the live core row) and the leak block.
  π′ = softmax(log π + Q/λ) at λ = 0.015, exactly the distribution whose mode S′ played (`src/main/value_search/rules.py`
  `pikl_scores`).
- **Each observation.** Every unit keeps a capture (the battle seed and every command). Replaying the honest input log
  through `main.value_search.worlds.prepare_root` rebuilds our row.
- **The probe** (outcome-blind, at `c365f637e`; [`frame_probe.json`](../research_state/measurements/x65_exit0_2026-11-05/frame_probe.json)):
  **976 of 977 rebuilt rows are bit-identical to `frame_sha256`**, at 0.012 CPU-s a row. The one miss is our second
  decision inside one request update. There the capture's arrival order is not `honest_log`'s (update, side) order,
  so a naive prefix is a different log. The bank build (B1) rebuilds that order. Every row must match its
  `frame_sha256`; a mismatch is dropped and counted, never kept.
- **Outcome-blind.** The builder reads `search_decisions.jsonl` and each capture's commands. It never reads
  `games.jsonl` or a result line.
- **Size** ([`bank_count.json`](../research_state/measurements/x65_exit0_2026-11-05/bank_count.json)): H 32,020,
  F 33,398, p72 32,183 searched decisions; 35,738 departures from P\*'s top choice. Exclusions (no consistent world, so
  P\* played) carry no label.
- **Anchor rows.** Every decision of our side in the training folds' games, both arms, rebuilt the same way, with P\*
  itself as the teacher (computed online). Raw games keep no `frame_sha256`, so these rows rest on the path the
  labels verify. A wrong anchor row only weakens the anchor; it cannot corrupt a label.

### 4.2 The folds: no student ever sees a read team

Both sides of a U deal draw from the same panel, so a fold holds out a whole panel:

| student | trains on labels and anchor rows from | reads on |
|---|---|---|
| S_H | F + p72 (≈ 65.6k labels) | H |
| S_F | H + p72 (≈ 64.2k) | F |
| S_P | H + F (≈ 65.4k) | p72 |

Panel adjacency is reported, not removed: p72's sets are Smogon usage draws, and H's hidden sets are usage-filled.

### 4.3 The recipe (one fixed recipe; the inner validation picks among two learning rates and the checkpoints)

- **Start and loss.** Start from P\*'s weights. The loss is KL(π′ ‖ π_θ) on label rows plus β · KL(π_P\* ‖ π_θ) on
  anchor rows, with β = 1, over legal actions only.
- **Batches and optimiser.** 256 label + 256 anchor rows a batch; AdamW (weight decay 1e-5, eps 1e-5), 200 warm-up
  steps, then a constant learning rate; gradient-norm clip 0.5; fp32; eval mode.
- **What trains.** No gradient flows into the value head's own parameters. The shared trunk moves, so V's drift is
  reported.
- **Training seeds.** 6511 / 6512 / 6513.
- **The inner validation.** Learning rate ∈ {1e-5, 3e-5} × checkpoints at every quarter epoch over 4 epochs. The
  pick is the lowest inner-validation loss, on 10 % of the TRAINING folds' deals (sha256 of the deal key, first byte
  mod 10 = 0). It never touches the read panel.
- **The scale of the move.** PPO moves P\* ≈ 15 optimiser steps an update at ≈ 1e-4. This recipe's ≈ 1,000 steps at
  1e-5 to 3e-5 is ≈ 7–20 updates' worth of displacement.

### 4.4 Is it safe to train on leaky targets? Can it switch?

- **The read is honest by construction.** The student plays greedy on its own observation through the bare loader. It
  has no path to the battle seed, the true team or the opponent's commands. A leak can only make a label a worse
  target: π′ at a leak-exposed decision is a per-world best response the observation cannot support (strategy
  fusion). The student learns its average given the observation. **A leak can bias ExIt-0 toward NULL, never toward a
  false GO.**
- **It is not a deployment of S′.** No search runs at play time.
- **It is not a use of S′-LIVE's licence.** GO-LEAKY licenses only a leak-honest replication, and "no training
  change". ExIt-0 is its own registration, a research arm that changes no production training.
- **The switch.** S′-HONEST's live read will bank the same decision schema on ≈ 3,840 fresh deals: ≈ 138k honest
  labels, with zero extra search CPU, if it keeps `q`, `log_pi`, `frame_sha256` and the capture on every searched
  decision (FINDING X65-F2: a request to its builder).
  - **ExIt-0H** re-runs this registration's recipe and folds on those labels. One lever: the label source.
  - It reads on fresh deals disjoint from every other read, and is registered after S′-HONEST lands.
  - The pair (ExIt-0 leaky against ExIt-0H honest) is the direct answer to "may training labels come from a leaky
    builder?". It matters because honest worlds cost ×1.2–5.9 the leaky builder's.
- **Reported:** the delivery meter (§5) split by each label's leak exposure (the `leak` block's fields).

## 5. The read, its rule, its power

- **The delivery gate, played before any Kakuna game (zero lane CPU).**
  - On each held-out panel's banked searched decisions, take Δagree = (the student's argmax agreeing with S′'s pick)
    − (P\*'s agreeing). P\* agrees on the non-departures only.
  - Pool over folds. Take the 95 % LB from a game-clustered bootstrap.
  - **DELIVERED** iff LB > 0. NOT-DELIVERED stops the read: no game is played, and the class is NOT-DELIVERED.
- **Look 1: 2,700 deals, the banked control.**
  - Each student plays its held-out panel's 900 S′-LIVE deals (U unit seeds 300–344), each as a lone one-game unit
    against Kakuna (checkpoint 34, greedy).
  - It plays at `320a535d4` (PREREG Amendment 2), whose play path is byte-identical to S′-LIVE's play pin
    `0c76b521e`, on engine E4.
  - It is paired by deal with the banked P\* raw game.
  - **G-BANK:** first re-play 135 banked raw units at that pin (one per panel and unit seed). At least 133 must be
    byte-identical; the measured rate of Kakuna divergence is 9 in 2,100. Otherwise look 1 plays a fresh raw arm too
    (+22 CPU-h).
- **Look 2: 4,500 fresh deals.** It runs only if look 1 is neither GO nor NULL. U unit seeds 600–674, 1,500 deals a
  panel; both the student and P\* raw are played fresh.
- **The statistic.** G_X is the mean over the three panels of the mean D = y(student) − y(raw). Its SE comes from a
  within-panel unit-seed bootstrap. Drops are imputed at their worst case, as S′-LIVE's were.
- **The classes:**
  - **GO** iff LB > 0 at z = 2.576 (look 1) or z = 2.0 (look 2).
  - **NULL** iff UB(95 %) < +2.0 pp.
  - else **UNDECIDED**.
  - A bound within 0.0005 of its bar never crosses it (root rule 8).
- **Power** (two-look Monte Carlo, normal; [`power.py`](../research_state/measurements/x65_exit0_2026-11-05/power.py) →
  `power.txt`; the share of deals where the two arms' results differ is taken as 0.45–0.50, since a student departs
  from turn 1 while S′ shared 10 turns, d = 0.341):

  | true Δ | 0 | +1.0 | +1.5 | +2.0 | **+2.3 (half of S′)** | +3.0 | +4.6 (all of S′) |
  |---|---|---|---|---|---|---|---|
  | P(GO) | **0.026** | 0.22–0.24 | 0.43–0.47 | 0.66–0.71 | **0.78–0.82** | 0.95–0.96 | 1.00 |
  | P(NULL) | 0.70–0.74 | 0.26–0.28 | 0.12 | 0.04 | 0.02 | 0.004 | 0 |

- **Reported on GO, ≈ 37 CPU-h:**
  - **Attribution:** a PLACEBO student whose Q̂ is permuted among the legal actions within each decision (same rows,
    recipe and sharpness; no information), on look 1's deals.
  - **Goal (a)'s floor:** the student and P\* against Kakuna on home72, 900 paired deals.
  - **The distilled fraction** G_X / G_S′ on look 1's deals. It is biased low: S′'s +4.60 was read on these deals.

### 5.1 What each class licenses (the PREREG's table is binding)

- **GO:** "a network distilled from S′'s leak-inclusive choices beats P\* on teams it never saw, against Kakuna,
  deployably". It licenses:
  - registering ExIt-0H;
  - a candidate for X54's sealed `forum216b` confirmation, on X54's own terms;
  - **only if the dose-matched PLACEBO (§7: Q̂ permuted, same recipe, rows and seeds) also reads SEARCH-INFO**
    (Amendment 1, answering REVIEW R-M2), registering **ExIt-1**: P\*+20M PPO forks with an auxiliary KL to the
    bank's π′, cross-fit as here, 3 folds × 2 seeds plus 2 same-pin C0 seeds ≈ 35 GPU-h, the GPU-days rung. GO with
    NOT-SEPARATED licenses only "a fine-tuned P\* beats P\*", and ExIt-1 is not funded on it.
- **NULL:** one supervised step on ≈ 65k labels keeps < 2 pp. It does not close ExIt (more labels, honest labels, an
  in-loop form); ExIt-1 is not funded, and X60-T takes the GPU.
- **NOT-DELIVERED:** the student did not learn S′'s choices on unseen teams. It closes nothing about larger budgets.
- **UNDECIDED:** nothing licensed; no extension without a new registration.

## 6. The GPU, said plainly

This plan uses ≈ 0.5 GPU-h, plus ≈ 35 GPU-h on a GO. Three things could fill days now, and none is right:
- a from-scratch scale run needs the owner;
- `D_MODEL` growth and X60-T need builds of days and have weak priors;
- a C0 alone buys nothing.

What changes this:
1. **The owner's yes on the rented CPU box** (STATUS §6.2). S′-HONEST's read then plays off-box, so local training no
   longer competes with it.
2. **FINDING X65-F1**, a 1-h measurement nobody has made: production-fork fps with the trainer pinned to cores 0–4 and
   8–12, with the Kakuna lane's pace measured beside it. If the lane keeps most of its 25–50 deals an hour, training and
   S′-HONEST can overlap.

## 7. The builds (≈ 2.5–3 agent-days; each fails on revert)

| # | build | estimate | built as |
|---|---|---|---|
| B1 | the bank: rows rebuilt in `honest_log`'s order, π′, a `frame_sha256` check on every label row (drop and count), anchor rows, fold and inner-validation keys, a manifest with sha256; outcome-blind | 0.75 d | `python -m main.x65.bank build / check` |
| B2 | the fine-tune: P\*-started, the soft KL, the anchor teacher, the inner-validation pick and resumable state | 0.5 d | `python -m main.x65.student train / pick` |
| B3 | the export: P\*'s zip with the student's policy, written at the play pin; logits bit-identical to the trainer's on the 64-row fixture; one bare-loader smoke game | 0.25 d | `python -m main.x65.export write / smoke` |
| B4 | the delivery meter and the rule `distill_play` in `main.reads` (two looks, worst-case imputation, the boundary rule), constructed-case tests and a by-hand script | 0.75 d | `python -m main.x65.delivery score / gate`; `main.reads.rules_distill_play`; `by_hand.py` |
| B5 | the read spec: G-BANK, the per-fold model labels, look 2's seeds, `check-spec` exit 0 | 0.25 d | the measurement folder's read spec(s) |

**How B1 rebuilds a row.** The capture keeps the commands in ARRIVAL order without the request update each one
answered, so B1 re-derives the update the way the front end counted it: it replays the arrival-order prefix once
(stopping before our last command, so no replay reaches the result), counts each side's `|request|` lines (a
re-request after `[Unavailable choice]` asks the same update) and cuts each of our decisions' log exactly as
`honest_log` did, then rebuilds the row through `prepare_root`. The probe's one miss (our second decision inside
update 24, after a trapped switch was refused) is its regression fixture (`src/main/x65/testdata/`). Every label row
must hash to its `frame_sha256` or it is dropped and counted. A raw game has no digest, so an anchor row with none
must pass two other checks or be dropped and counted: the honest cut's row equals the arrival-order replay's, and
P\*'s greedy action on the row is the one the game recorded (a row whose two best logits lie within 1e-4 is excluded
from that check, root rule 8).

**Readings the builds took where the PREREG is silent** (for the adversarial review to accept or amend):
- the inner-validation `deal` key is `search_side.deal_key`'s digest (the searched records' `deal`, shared by a
  deal's raw and searched games), not the record's `panel:seed:game` string;
- an anchor batch is drawn from a fresh permutation of the anchor rows each label epoch, so an epoch uses
  ≈ 230 × 256 of them, not all;
- every decision of our side is an anchor row, including one-legal-action decisions (KL 0) and both arms' identical
  pre-turn-10 prefixes (so those rows weigh twice);
- π_P\* is computed once, from a frozen copy of P\*'s bytes, before the first step;
- the inner-validation L is a float64 mean over all held-out label rows plus β times one over all held-out anchor
  rows; an epoch's last label batch is the remainder;
- the delivery gate's a(P\*) uses the recorded `pi_argmax`; a(student) the exported zip; root rule 8 applies to its LB.

## 8. The launch, checked

- **The GPU step is not a `main.launcher` run.** It is B2's offline fine-tune under the GPU lease, so there is no
  launcher argv to validate for it.
- **ExIt-1's control argv was validated at `c365f637e`:**
  ```
  python -m main.launcher --model models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip \
    --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --seed 6501 \
    --run-name rb_x65_c0_s6501
  ```
  - `main.checkargs` passes: the fork architecture is compatible, and every ARCH key matches the production mirror.
  - `--dry-run` refused only because `CUDA_VISIBLE_DEVICES=''` was set: this unit holds no lease. The disk,
    Showdown and architecture checks passed.
  - Its treatment flag does not exist yet. ExIt-1's registration names it.

## 9. Risks and the self-check

- **Kakuna-specific learning.** The labels come from Kakuna games, and the read is against Kakuna. On GO, goal (a)'s
  home72 floor is read, and ExIt-1 adds the in-house meter.
- **Under-power at small capture.** At +1 pp, P(GO) is ≈ 0.23. An UNDECIDED there is honest: it is a power
  statement, not "no effect".
- **Recipe sensitivity.** One recipe, two learning rates picked blind on the training folds. A NULL is "at this
  recipe".
- **Banked-control drift.** G-BANK.
- **The bias of S′'s own gain.** The distilled fraction is descriptive, and biased low.

## Decision record

| date | decision | chosen | rejected, and why | evidence |
|---|---|---|---|---|
| 2026-11-05 | **This doc CREATED and X65 ExIt-0 REGISTERED** (research designer, orchestrator brief: one GPU experiment, worth running whatever S′-HONEST finds) | ExIt-0: supervised distillation of S′ into P\* from S′-LIVE's banked labels, panel cross-fit, read against Kakuna on the held-out panel with the banked raw arm at look 1 | fresh-label ExIt (9.5 CPU-s a label, measured); the in-loop factory (≈ 32 cores at f = 0.25 %); `D_MODEL` growth (not growable; three capacity NO-GOs); from-scratch scale (the owner's question; ≈ 8–12 GPU-days); X60-T (unbuilt; the runner-up); a bare C0 | §2; `frame_probe.json` 976 / 977; `bank_count.json`; `power.txt`; S′-LIVE's record; `design_sprime_honest.md` §4 |
| 2026-11-05 | **B1–B5 built** (X65 builder, orchestrator brief: build exactly as registered) | B1–B4's meter in `src/main/x65/` (a new loop, reusing X59-S's helpers, HOM's export form and `policy_logits_t`); the rule and the spec in `main.reads` | X59-S's `train` as-is: its loss (set-NLL), its fresh init, its cosine schedule and its rowset format are all different from §3's, so it would be a fork with every line changed | an 18-shard sample (360 games, 17,134 rows): 6,007 / 6,007 label rows and all 6,621 searched-arm digests verified, every recorded `log_sha256` and re-score log reproduced, 0 anchor rows failing either check; the readings in §7 are for the review |
| 2026-11-05 | **ExIt-1 waits for the placebo** (PREREG Amendment 1; REVIEW R-M2, the orchestrator's ruling) | GO funds ExIt-1 only with the placebo's SEARCH-INFO; the placebo runs on GO at either look, on look 1's deals | GO alone: the control is not dose-matched, so a GO can be fine-tuning rather than S′'s choices | `REVIEW.md` R-M2 |
| 2026-11-05 | **The pin amendment** (PREREG Amendment 2) | the bank `d13fd55f…2b7a` (97,395 / 97,395 label rows verified); the play pin `320a535d4`; every REVIEW finding disposed | pinning `0c76b521e` itself: its `main.reads` cannot plan the specs; a successor built on `0c76b521e`: not needed, since `320a535d4`'s play path is byte-identical | `playpath.compare`; the export check at `320a535d4`; `bank_manifest.json` |
