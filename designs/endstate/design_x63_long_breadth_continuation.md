# X63 — a long breadth continuation of the plateau: +100M on a union corpus, pool share 0.2

**Status: REGISTERED 2026-10-29, launch-ready for Stage A, NOT LAUNCHED.** The PREREG is
[`x63_2026-10-29/PREREG.md`](../research_state/measurements/x63_2026-10-29/PREREG.md); it wins any disagreement with
this page. This page was the DRAFT of 2026-10-27 (`49fbf75a`), revised in place after its adversarial review
([`reviews/x63_review_2026-10-27.md`](reviews/x63_review_2026-10-27.md), `ca51bf99`: one BLOCKER, five MAJORs) and a
second, blockers-only review (§10). Everything the review found is answered in §10, fixed or declined with a reason.
The launch is the Training session's, through the arm chain (kit arm `x63u`), on the orchestrator's GO.

It turns recommendation 2 of the context-generalisation survey
([`context_generalisation_2026-10-27.md`](../research_state/literature/context_generalisation_2026-10-27.md) §7,
`5d554c23`: continue the plateau ≥ 100M on a broad team distribution with the pool at ≤ 0.2) into an experiment. Goal
(b) and its meter are [`design_beat_metamon_path.md`](design_beat_metamon_path.md)'s; the ranking rules are
[`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md)'s. The accepted rulings it applies
are [`precommit_reads_2026-10-27.md`](precommit_reads_2026-10-27.md) §4 (`090730a9`). The power Monte Carlo is
[`power_x63_v2.py`](../research_state/measurements/x63_2026-10-29/power/power_x63_v2.py) → `.log` (CPU, 7 s).

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280M (the named baseline `pstar_plateau`); **X26@380M** = X26's `checkpoint_380000195_steps.zip` (sha256
`7e0ebe94…`); **X26c** = `rb_x26c_s2601` (X26@380M + 100M at `f4b25238`, no lever; FLAT); **X63-C** =
`rb_x63c_s6302` (X26c's argv with seed 6302, trained 10-28/29, arm-end K9(b) PASS, `8a7f293f`); **X63-U** = the
treatment; F = `forum72`; H = `heldout72_hpfix`; P72 = the procedural panel; home72 = our 72 sample teams; S =
in-house strength against the frozen reference population R on home72; σ = the true between-seed SD of a +100M
continuation's F score; σ̂_CC = σ measured from X26c v X63-C; lane-h = one wall-hour of the Kakuna lane at
`--concurrency 1` on the agent cores (160–230 games an hour); GUESS = an unmeasured effect; MY JUDGEMENT = an
inference no read states.

---

## 0. One screen

- **The question.** Does continuing the plateau network for 100M steps with 80 % of each side's teams drawn from a
  17,876-team union corpus make it pilot unseen human-built teams (F) better against Kakuna than the same network
  continued on the pool alone? It is NOT a dose read (review M2): against the 20M breadth forks it changes the steps,
  the parent, the share and the corpus at once, so a GAIN is "a long union continuation at share 0.8", never "dose".
- **The arm.** X63-U s2601: X26@380M forked at its own corpus-only pin (`f4b25238` + one commit adding
  `x63_union_v1`), X26c's argv token for token plus `--team-mix x63_union_v1@ALL:0.8 --team-mix-sides both`, +100M
  to 480M. Kit arm `x63u`.
- **The controls already exist.** X26c (seed 2601) and X63-C (seed 6302) are both X26@380M + 100M at `f4b25238` with
  no lever. Stage A reads **1 arm against 2 controls** and measures σ̂_CC at the same time (the review's B1 fix 3).
- **The mix** (§3): x62 recombined sixes 12,000 + N1-P's procedural teams at its registered prefix tier K3 (3,000) +
  D3's set-mutated teams 2,876, drawn uniformly over teams: **0.537 / 0.134 / 0.129** of all draws, the pool 0.2. The
  procedural third the draft had is cut to a seventh, because F, the decider, reads context novelty and barely sees
  n1 (review M3). Forum teams stay out (F would carry its own population).
- **Cost** (§5): Stage A **19.9 GPU-h** (X26c's measured 100M); Stage B +19.9 (one arm seed; its control already
  exists); Stage C +39.8 (a pair) only on a near miss. **E[GPU-h] 36–45 over the GUESS rows; at most 79.6.** Lane:
  Stage A's deciding games 18,000 (78–113 lane-h), of which the controls' 12,000 can play BEFORE the arm finishes.
- **The read** (§6): Kakuna on F at the X35-frame shared play pin `86cfdf7e`'s regime (E1, the post-H22 Kakuna pack;
  the play pin every X35-frame read shares), played from its X63 successor, which adds only X63's rule, 4,000 F games a model at +100M; home72 against Kakuna at 2,000 games a model
  for goal (a)'s floor and the price.
- **The rule** (§7): a Stage-A screen (CONTINUE iff Δ̂_A ≥ +1.0 pp, else **NOT PURSUED**, no era claim), then 2 v 2,
  then 3 v 3 on a near miss, with a seed term σ_eff = max(1.0 pp, σ̂) and a Student-t quantile on σ̂'s degrees of
  freedom whenever σ̂ exceeds the floor. **False GAIN ≤ 0.05 for σ ≤ 3 pp** (0.06 if the arm's spread is twice the
  control's); GAIN at a true +3 pp 0.91 / 0.64 / 0.45 / 0.32 at σ 0 / 1 / 1.5 / 2 pp.
- **The price** (§6.3): on a GAIN, **net = Δ_home72 + r · Δ_F** (both against Kakuna), with **r the owner's exchange
  rate, written before X63-U launches** (STATUS §6). The S ceiling of the draft is gone (review M4).
- **P(GAIN) ≈ 0.28** over the registered GUESS (+1.5 pp (−1.5, +4.5)) and the review's σ prior; ≈ 0.04 per
  resource-day.

---

## 1. Why a long continuation, and why now

**What the survey found** (§0 and §7 rec 2 there). Every breadth result in the literature trained on the broad
distribution for 10⁸ steps (Procgen 200M, CoinRun 256M, VGC-Bench, OpenAI Five). Honor of Kings, the one recipe that
starts from specialists, followed its fixed-lineup teachers with a long continuation on random lineups. P\* is the
specialist; its next step in that recipe is a long broad continuation (MEDIUM-LOW confidence: other domains' step
counts).

**What our reads say.**
- **Plain continuation adds nothing.** X26c, 380M → 480M on the pool: +0.08 pp [−3.90, +4.06] against Kakuna, S
  −0.01 pp. Movement against X26c and X63-C is the union's, not the steps'.
- **The premium is team-level** (H-CTX SUPPORTED, `6f126c4af`): on new sixes of familiar pool sets (rc72b) P\* scores
  0.514, below F's 0.540 and 10 pp under home72 on one engine. The A-K check (`fb84c260f`) finds rc72b about 28 Elo
  weaker as a panel under a neutral pilot, uniformly; the premium stays mostly team-level. So recombined sixes (x62)
  are the component aimed at what F lacks.
- **20M forks of breadth read +1 to +2 pp, none resolved.** D1 H +2.10 [−0.92, +5.11]; D3 final +2.33 [−0.69, +5.35],
  in-house non-inferior; N1-P look 1 F +0.75 [−1.56, +3.06] with a resolved in-house cost (S −3.27 [−4.20, −2.33]).
- **A 20M fork pilots a corpus team ≈ 24 times a side; X63 pilots each ≈ 128 times** (2.85M games per 100M × 0.8 /
  17,876, D1's rule). X52 needed ≈ 4,000 pilots to show a practised +3.2 pp, so X63 is still "fresh" per team.
- **The learning-rate controller is not the plateau's cause** (LR-F UNDECIDED, final, −1.13 pp [−2.45, +0.18] on S),
  and the plasticity probe read NO-LOSS through 480M: the trunk still bends.

---

## 2. The arms and the controls

| arm | run | start | lever | role |
|---|---|---|---|---|
| **X63-U** | `rb_x63u_s2601` (Stage B `rb_x63u_s6302`, Stage C `rb_x63u_s6303`) | X26@380M | `--team-mix x63_union_v1@ALL:0.8 --team-mix-sides both` | treatment |
| **X26c** | `rb_x26c_s2601` (trained) | X26@380M | none | control, seed 2601 (X63-U s2601's argv twin) |
| **X63-C** | `rb_x63c_s6302` (trained), Stage C `rb_x63c_s6303` | X26@380M | none: X26c's argv, seed changed | control replicate |
| P\* | `rb_x26_s2601` @280M | — | — | REPORTED anchor only |

**The argv** (the kit builds it from X26c's immutable `original_command`; the PREREG holds the `checkargs` and
`--dry-run` transcripts): `--model …/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip --behaviour-check warn
--steps 480000195 --checkpoint-every-steps 1000000 --device cuda --team-mix x63_union_v1@ALL:0.8 --team-mix-sides
both --seed 2601 --run-name rb_x63u_s2601`, with the launcher's `--restart-interval-hours 24 --pin-commit <X63U pin>`.
`checkargs` against X26c differs by the four mix tokens and the run name only. A fork inherits no mix, and X26c had
none.

**The pin.** `f4b25238` + ONE commit confined to `src/utils/team_corpora/` holding `x63_union_v1`'s three files and
its `CORPORA` line (the AM-N1P pattern; `main.x35.batch` checks it as `AM-X63U`). The arm runs the controls' code byte
for byte. The 10-26 / 10-27 GIGO fixes are absent from every arm and control alike.

### 2.1 Why the controls are valid, and their limits

- **Matched** (read from the runs, the review's "What holds"): the parent file (sha `7e0ebe94…`), the pin, the role
  (a fork seeded from X26's snapshots; X63-C's pool-seed check equals X26c's), the argv but the seed, the restart
  interval, the 1M checkpoint cadence, the local card, arm-end K9(b) PASS on both.
- **Read files, pinned by sha256 in the read spec** (review m7): X26c `final_model.zip` (480,081,209, `cba7a1f8…`;
  X26c's later 480 → 489.6M segment branched at 480,000,085 and never wrote this file); X63-C `final_model.zip`
  (480,080,819, `e778c4a4…`); the 400M / 430M nodes of both (REPORTED only).
- **Two seeds give σ̂_CC one degree of freedom.** Its 95 % interval runs ≈ 0.45 σ̂ to ≈ 32 σ̂, and with true σ = 1.5 pp
  a single pair reads exactly 0 a third of the time (`power_x63_v2.log` B). So σ̂_CC never stands alone: it enters the
  pooled σ̂ of §7, behind a floor and a t quantile.
- **No banked Kakuna row imports.** X26c's rows were played on E1 under Kakuna's legacy pack; the shared pin is its
  own regime (new ids on every panel). Every control game is played fresh.
- **The learning-rate path is an outcome, not inherited** (review M1). The KL controller is live in every arm. X26c
  held 8.37e-5 to 471M and stepped to 6.98e-5; X63-C's median over its run is 6.98e-5 (`python -m main.dose`). An
  arm's rate falls when its updates move the policy more, which the lever may cause. The lever is therefore
  "breadth under the production KL controller", a legitimate bundle; §6.4 reports the path and pre-labels a
  divergence.
- **Not valid for a P\*-start arm** at any dose (X26's own 280 → 380M is cross-pin; the draft's §2.1 argument stands).
  X26@380M is on P\*'s plateau (X26c's 480M equals P\* on U, S flat, plasticity NO-LOSS).

---

## 3. The mix

### 3.1 The components and the gates that set them (all three read before this registration)

| component | teams in X63 | what it varies | gate read | in |
|---|---:|---|---|---|
| `x62_recomb_v1@ALL` (built by this unit, X62's G0; X62 §2 exactly) | 12,000 | context: sixes of familiar STRICT-class pool sets, ≤ 2 shared with any pool team | H-CTX **SUPPORTED** (premium team-level) | yes |
| `n1_proc_v1@K3` (N1-P's registered prefix tier) | 3,000 | context and sets: procedural, ≈ 72 % of slots new sets | N1-P look 1 **CONTINUE** ⇒ n1 IN (ruling 1) | yes, a seventh of draws |
| `d3_setmut_v1@ALL` | 2,876 | sets inside familiar sixes (4–5 shared with one pool team) | D3 look 2 **UNDECIDED, final**: H +2.33 [−0.69, +5.35]; P −0.45 [−1.16, +0.25] | yes (ruling 4: out only on H LOSS or Δ_P's UB < −2.0; UB is +0.25) |
| `w_forum_v1` | — | human-built sixes | — | **no**: F is drawn from its population |

**Why n1 at K3 and not all 12,000** (review M3: "choose the mix by the question"). F decides. F is 70 % pool sets and
26 % recombined-like (X62 §1.2), and n1 covers 19 % of F's full sets against 62 % / 56.5 % of H's / P72's. At its full
size n1 would take 0.357 of draws for a component F can barely read, while H-CTX now says the premium F lacks is
team-level, which is x62's target. n1 stays IN (ruling 1), at the registered prefix tier K3, so set novelty is still
present without a builder change: x62 0.537, n1 0.134, d3 0.129 of all draws. D3's sixes keep 4–5 pool sets, so the
pool-LIKE share is ≈ 0.33, above the survey's ≤ 0.2 in spirit; the price (§6.3) is what guards the pool side.

**x62's coverage of F** (the M3 re-measure, PREREG §3): the STRICT rule withdraws 86 of the pool's 658 sets, yet x62
still carries the set of 0.699 of F's mon slots against the pool's 0.701, so STRICT costs F almost nothing. n1 at K3
still covers 0.48 / 0.41 of H's / P72's full sets (0.62 / 0.56 at 12,000): H and P72 stay hazarded.

### 3.2 How it is built (`utils.team_corpora.build_x63_union`)

One registered corpus, because `--team-mix` takes one: the three components loaded through the verifying reader at
their registered tiers, concatenated in a fixed order, uniform over teams (Procgen's sampling; a component-weighted mix
would be a parser and Rust-table build that buys nothing here). The builder:
- refuses a duplicate six across components;
- **refuses any team sharing ≥ 3 sets with ONE rc72b team** (review m3: rc72b is in `diagnostic_panels.json`, which
  postdates the pin, so the pin's startup guard never sees it, and its radius is 4); a test fails if the check goes;
- records every panel's max-shared histogram (heldout72, H, F, P72, `forum216b`, `n1p72`, rc72b), the
  cross-component near-duplicates (≥ 5 shared), each component's tier sha, and **the material pack-regime share**:
  teams whose Frustration plays at base power 1 under the pin's pre-`d8d674b9` packer (STATUS §7 risk 4: above 2 % of
  teams the arm would need a packer-backport pin).

The startup guard (`team_mix_guard`, radius 4 against every held-out panel including the sealed `forum216b`, which
`heldout_panels.json` lists at the pin: review m3) re-checks the union at launch. The crash and label-path soak, the
startup team phase and the per-process RSS at the pin are measured (review m5) and recorded in the PREREG §3.

### 3.3 The variants, each with its GUESS (review m1)

| variant | when | draws x62 / n1 / d3 | Δ_F GUESS at +100M (80 %) | P(GAIN), σ prior |
|---|---|---|---|---|
| **U3-K3 (registered)** | now | 0.537 / 0.134 / 0.129 | **+1.5 (−1.5, +4.5)** | **0.28** |
| U3-K3 after a PR-1′ KILL | PR-1′ reads before X63's decide | same | +0.5 (−2.0, +3.0) | 0.16 |
| U3-K3 after a PR-1′ BREADTH-GAIN | same | same | +2.5 (−0.5, +5.5) | 0.39 |
| no-n1 (x62 + d3, 14,876) | only on an N1-P look-2 LOSS before launch (§8) | 0.645 / — / 0.155 | +1.5 (−1.5, +4.5) | 0.28 |

σ prior (the review's, MY JUDGEMENT): 0.5 / 1 / 1.5 / 2 / 3 pp at 0.15 / 0.25 / 0.25 / 0.2 / 0.15. Basis of +1.5
(MY JUDGEMENT): the 20M breadth reads on unfamiliar panels cluster at +0.75 to +2.3 with D1's relevant number its H
+2.1 (not its F +4.0, its own population: review m1); F's premium below home72 is ≈ 7.5 pp on one engine and H-CTX
puts it in the sixes, which x62 trains on directly; a 100M dose at ≈ 128 pilots a team. The draft's +2.0 is lowered
because its basis cited D1's own-population F.

---

## 4. The dose and the read points

- **+100M (to 480M) is the registered dose**: the field's order of magnitude for a breadth result and the controls'
  length.
- **Only +100M decides.** The +20M / +50M nodes (400M / 430M) are played at 1,200 F games per model as X63-C's own
  PREREG registered them, REPORTED, after every deciding game. At that size a 1 v 1 Δ has a 2.0 pp SE, so they are a
  picture, never a trend claim (review M2, E1).
- **No dose claim** (review M2 fix (b), chosen over (a) because (a) costs 24–35 lane-h for a contrast whose seed term
  is UNVERIFIED to cancel). X63 does not replace the strategy's fork extension (rank 6), which holds the corpus,
  share and parent fixed. A dose extension past 480M is licensed only by a GAIN, as its own registration.

---

## 5. Seeds, stages and cost

The rate: X26c 19.92 GPU-h for 100M at 1,402 steps/s; X63-C 20.7 h at 1,348 steps/s (`8a7f293f`). The table uses 19.9.

| stage | new runs | GPU-h | deciding Kakuna games | lane-h | when |
|---|---|---:|---|---:|---|
| **A** | `rb_x63u_s2601` | 19.9 | F 4,000 + home72 2,000 for each of X63-U, X26c, X63-C = 18,000 | 78–113 | now (the controls' 12,000 can play first) |
| **B** | `rb_x63u_s6302` | 19.9 | 6,000 | 26–38 | trains speculatively before Stage A's verdict (AM-X63-BSPEC); played and read only on Stage A CONTINUE |
| **C** | `rb_x63u_s6303`, `rb_x63c_s6303` | 39.8 | 12,000 | 52–75 | Stage B UNDECIDED with Δ̂_B ≥ +1.5 pp |
| **at most** | 4 runs | **79.6** | 36,000 | 156–226 | |

- **Expected GPU-h** (`power_x63_v2.log` E): 41 over the registered GUESS, 36 after a PR-1′ KILL, 45 after a PR-1′
  BREADTH-GAIN; 24–33 under a true null.
- **REPORTED games**, played only when the lane has room after the deciding games: H and P72 at 400 a model (with
  the n1 set-coverage line), the 400M / 430M nodes at 1,200 F a model (X63-U, X26c, X63-C), P\* on F at 1,200,
  rc72b at 400 a model (the union is measured at ≤ 2 shared against it). ≈ 12,000 games, 52–75 lane-h.
- **S** (in-house, REPORTED only): ≈ 1.5 heavy-slot-h a model (ESTIMATE), when the heavy slots have room.
- **The card**: all local (the card rule forbids comparing across card types; both controls trained here).
- **Wall time.** Stage A trains ≈ 21 h. The lane is the binding cost; REVIEW_2 counts 84,000–122,000 Kakuna games
  already queued, so the order on the lane is the orchestrator's (review m2). A rented CPU box (STATUS §6 item 2)
  would play Stage A in a day.
- **Per resource-day**: 0.28 / ((41 GPU-h + ≈ 125 expected lane-h) / 24) ≈ **0.04** (review m2's 0.045 band).

---

## 6. The read

### 6.1 Primary: goal (b) on F against Kakuna

- **Engine and regime: the X35-frame shared play pin, `x35-shared-play-86cfdf7e`** (`86cfdf7ef`, FROZEN;
  [`designs/tools/reads.md`](../tools/reads.md) §4): main's harness with every game path restored to `f4b25238`'s
  bytes, so OUR side plays exactly as it trained (E1, the legacy pack); Kakuna's pack post-H22. Checked there: 220 /
  220 captured games equal choice by choice against banked E1 units. This is the review's preferred option (M5) and
  the play pin of every X35-frame read (PR-1′, X62, X48's C0), so no engine × treatment interaction can enter.
  **The games are played from its named successor** (PREREG §5: `86cfdf7e` + one commit adding only §7's
  `main.reads` rule, `x63_kvk`, play path byte-identical), because the frozen pin's spec parser refuses a rule it does
  not hold (the second review's blocker 1).
- **Cell:** Kakuna checkpoint 34, greedy against greedy, CPU, `forum72`'s committed config, role-balanced 20-game
  units, unit seeds disjoint from every other read at this pin (REVIEW_2 m2; the PREREG names the block), anchor
  group `primary`; 200 units (4,000 games) per model at +100M. Each unit records its engine stamp and `peer_pack`;
  two stamps in a group is INCONCLUSIVE.
- **Statistic** (k arms against k controls; Stage A k_U = 1, k_C = 2): Δ_F = mean over arm seeds of F − mean over
  control seeds of F, each a plain proportion of that model's games. The interval and classes are §7's. A
  team-cluster bootstrap over F's 72 teams is REPORTED beside it.

### 6.2 Goal (a)'s floor: home72 against Kakuna

2,000 games per model at +100M, same cell and pin. **FLOOR PASS iff the arm's pooled home72 Wilson LB > 0.50.** It is
goal (a)'s definition, not a price guard (review M4: it passes a drop of up to 7.6–8.5 pp from P\*'s 0.598); the
price is §6.3.

### 6.3 The price: home72 against Kakuna, at the owner's exchange rate

The literature makes an in-pool drop the expected sign of breadth working (CoinRun's train score, VGC-Bench's seen
teams; D1 −1.6, N1-P −3.27 in S), so a harm bar at 0 would kill the lever for its predicted side effect. The draft's S
ceiling fired on the width of a Welch interval at ≈ 1 degree of freedom (P(TRADE) 0.32 at a true cost of 0; review
M4) and is withdrawn. The price is X62's form, with F (X63's clean panel) in place of U:

- **net = Δ_home72 + r · Δ_F**, both arm − control against Kakuna, k v k.
- **PRICE WITHIN iff net's point estimate ≥ 0** (rule 8: within 1e-9 of 0 is not WITHIN); **PRICE EXCEEDED iff net's
  95 % UB < 0**, SE_net² = SE_home72² + r² · SE_F², each SE as §7's (its own σ̂, the same floor and quantile).
- **r is the owner's**, in words "how many pp of goal (a)'s margin one pp of goal (b) is worth". It is written into
  the PREREG by a dated amendment **before X63-U s2601 launches** (the second review: once the arm trains, its
  pool win rate and arm-end records preview the cost r prices). Recommended default **r = 1** (STATUS
  §6): both goals are the same opponent and the same win rate, goal (a) holds ≈ 5 pp of margin over its bar
  (LB 0.549) while goal (b) is ≈ 4 pp short, and r = 1 is X62's default, so the two breadth designs price the same
  trade alike. If no r is written by then, the read computes the break-even **r\* = −Δ̂_home72 / Δ̂_F** and the verdict
  on a GAIN that passes the floor is **UNPRICED**: the owner rules on it afterwards, recorded as a post-hoc decision, never as the
  registered verdict.
- **The protected-team caveat** (review m9). The trainee draw is `default_biased` (`bias_prob` 0.1) toward the sample
  teams, which ARE home72, so a home72 team keeps ≈ 2.1× the pilots of another pool team at share 0.2. home72 is
  the best-protected 72 of the 719 pool teams: the price is measured where it is smallest, and is stated so. No
  non-sample pool panel against Kakuna exists; building one is its own unit (declined here, §10).

### 6.4 REPORTED, never decided

- **The learning-rate path** (review M1): `python -m main.dose` of every arm and control (sidecar source), median rate
  over 380–480M. **Pre-registered label:** if the arm's median sits ≥ 2 controller rungs (≥ 1.44×) below the
  controls' median, any verdict carries "confounded with the KL controller's response to the lever", and the
  follow-up is a breadth arm at a frozen `--fork-lr` (its own registration).
- σ̂_CC, the pooled σ̂ and the unselected estimate (Stage B's new arm seed alone, which the screen did not select).
- H and P72 against Kakuna with the n1 set-coverage line (N1-P PREREG §8's M4 hazard): a gain there may be the
  generator. U is printed.
- rc72b, P\* on F, the 400M / 430M nodes, S on home72 (D1's NON-INFERIOR bar printed beside it, deciding nothing).
- H-MID's Δbias on the arm's and the controls' F and home72 captures.

**Predictions (GUESS, registered before any game):**

| quantity | point | 80 % interval |
|---|---|---|
| **Δ_F at +100M (primary)** | **+1.5 pp** | (−1.5, +4.5) |
| Δ_home72 against Kakuna | −2.0 | (−5.0, +1.0) |
| Δ_S on home72 | −2.0 | (−4.5, +0.0) |
| Δ_H / Δ_P72 (n1-hazarded) | +1.5 / +1.5 | (−1.5, +4.5) |
| σ (a +100M continuation's F seed SD) | 1.25 pp | (0.5, 3.0) |
| arm median rate / controls' | 0.85× | (0.6×, 1.0×) |

---

## 7. The decision rule, its power, and what each verdict licenses

**The seed term.** σ̂² = ½ (s²_arm + s²_control) − ½ (mean binomial variance of the arm models + of the control
models), each s² the sample variance (ddof 1) over that side's seeds; a side with one seed has no s² and drops out of
both halves (so at Stage A σ̂ = σ̂_CC). σ̂ = √max(0, ·); its df is Σ (seeds − 1) over the sides: 1 at Stage A, 2 at
Stage B, 4 at Stage C. **σ_eff = max(1.0 pp, σ̂).** SE² = Σ p̂(1 − p̂)/G over the arm models
/ k_U² + the same over the control models / k_C² + σ_eff² (1/k_U + 1/k_C). The quantile q is 1.96 when σ̂ ≤ 1.0 pp,
else Student-t at 0.975 on σ̂'s df (4.303 at df 2, 2.776 at df 4): the review's B1 fix 2, second form.

**The rule** (rule 8: a statistic within 1e-9 of its threshold takes the conservative side).
1. **Stage A (screen; X63-U s2601 against X26c and X63-C):** CONTINUE iff Δ̂_A ≥ +1.0 pp. Else **NOT PURSUED**: the
   record states the achieved power and claims nothing for the era (review B1 fix 1). σ̂_CC is printed.
2. **Stage B (2 v 2; + X63-U s6302):** **GAIN iff LB > 0**; **RETIRE iff UB < +1.5 pp** (PR-1′'s KILL bar);
   **INCONCLUSIVE iff σ̂ > 3.0 pp**; else UNDECIDED. GAIN and RETIRE also need σ̂ ≤ 3.0 pp.
3. **Stage C (3 v 3; + X63-U s6303 and X63-C s6303), only if B is UNDECIDED with Δ̂_B ≥ +1.5 pp:** the same classes,
   final. The looks are not alpha-corrected; the false-GAIN column below includes both and the screen's selection.
4. The floor (§6.2) and the price (§6.3) are read only on a GAIN.

**Power** (`power_x63_v2.log` A, 40,000 draws a cell, F base 0.53, 4,000 F games a model):

| true Δ_F | σ = 0 | σ = 1 pp | σ = 1.5 pp | σ = 2 pp | σ = 3 pp | arm 2 / control 1 pp | arm 3 / control 1.5 pp |
|---|---|---|---|---|---|---|---|
| 0 (false GAIN) | 0.001 | 0.023 | 0.042 | 0.050 | 0.049 | 0.048 | 0.059 |
| +2.0 | 0.47 | 0.35 | 0.27 | 0.20 | 0.13 | 0.27 | 0.18 |
| +3.0 | 0.91 | 0.64 | 0.45 | 0.32 | 0.19 | 0.44 | 0.28 |

P(NOT PURSUED) at a true +2 pp is 0.15 / 0.26 / 0.31 / 0.35 / 0.39 across the same σ; INCONCLUSIVE at a true null
reaches 0.06 / 0.19 at σ 2 / 3 pp. RETIRE essentially never fires at k ≤ 3 (it needs Δ̂ ≈ −1.6 pp after a +1 pp
screen): a stop is NOT PURSUED or UNDECIDED, never an era claim.

**The alternatives** (the same log): the draft's rule (floor 1.0, z) has false GAIN 0.07–0.08 at σ ≥ 2 pp; a 1.5 pp
floor with z has 0.08 at σ = 3 pp and +3 pp power 0.68 at σ = 0; a 1.5 pp floor with the t quantile is as safe as
the registered rule (≤ 0.052) but loses ≈ 0.2–0.3 of power at σ ≤ 1 pp. The registered form is the only one with
false GAIN ≤ 0.05 across σ ≤ 3 pp that keeps the draft's power where σ is small.

**The verdicts.**

| verdict | rule | what it licenses |
|---|---|---|
| **ADOPT-CANDIDATE** | GAIN, PRICE WITHIN, FLOOR PASS, every VOID guard clear | (1) a dose extension past 480M (its own registration); (2) the arm's final as a candidate plateau baseline: its X54 read on `forum216b` and a `main.baselines` proposal are their own units. **Its lineage hazards H and P72** (n1 covers 62 % / 56.5 % of their full sets: review m8), so a baseline proposal needs replacement held-out panels drawn outside n1's coverage; (3) X53-N1's prior rises |
| **TRADE** | GAIN and FLOOR, neither WITHIN nor EXCEEDED | to the owner with net's interval; not adopted at the registered r |
| **OVER-RATE** | GAIN and (FLOOR fails, whatever r; or EXCEEDED) | not adopted; goal (a) is at risk, or the cost is clearly over the rate |
| **UNPRICED** | GAIN, FLOOR PASS, no r written before X63-U launched | the owner, with r\*; recorded as a post-hoc decision |
| **NOT PURSUED** | Stage A Δ̂_A < +1.0 pp | no claim; the record prints P(NOT PURSUED) at a true +2 / +3 pp and the σ̂_CC it saw |
| **RETIRE** | k ≥ 2 and UB < +1.5 pp | "a 100M union continuation of the plateau at share 0.8, with these corpora" is retired for goal (b) in this era |
| **UNDECIDED** | Stage C (or B when C does not fire) reads neither | the orchestrator, with σ̂ and the unselected estimate |
| **INCONCLUSIVE** | σ̂ > 3.0 pp | more seeds or stop (the orchestrator); σ̂ is itself a finding for X53-N1's σ gate |
| **VOID** (one seed) | PREREG §8's guards | never read; re-run at the same pin and seed after the cause is fixed |

---

## 8. Gates: what read first, and the one still pending

All three registered gates read before this registration, and the union is frozen at the pin:
- **H-CTX: SUPPORTED** ⇒ x62 in (U3), X62's corpus built (its G0).
- **N1-P look 1: CONTINUE** ⇒ n1 IN without waiting for look 2 (ruling 1), at K3 (§3.1).
- **D3 look 2: UNDECIDED, final**, H not LOSS, Δ_P's UB +0.25 > −2.0 ⇒ d3 IN (ruling 4).

**N1-P look 2 (s3504–s3505 training 10-29; its read lands after X63-U launches). The amendment is pre-written here and
in the PREREG §9, before its verdict:**

| N1-P look-2 verdict | lands | X63 does |
|---|---|---|
| GAIN / NULL / UNDECIDED-final | any time | nothing changes: n1 is IN at 0.134 of draws |
| **LOSS** | before X63-U s2601 launches | HOLD the launch; rebuild the union without n1 (x62 + d3, 14,876 teams), a new pin and a dated amendment; GUESS row "no-n1" (§3.3) |
| **LOSS** | after X63-U s2601 launches | X63-U trains and is read as registered: the composition is frozen at launch, every later arm seed keeps the same union (pooling needs one treatment), and a 20M LOSS at share 0.5 says little about 0.134 at 100M. Every X63 verdict then carries "n1, 0.134 of draws, read LOSS at 20M"; on NOT PURSUED / RETIRE the no-n1 union is the named follow-up, as its own registration |

**Not gates.** PR-1′ (it reads resolution at 20M; X63 runs on its own; its class only moves the GUESS row, §3.3).
X59-S / X60 (no human-data corpus in X63). X48 / N4 (capacity, independent). LR-F (read, final; J16 does not apply).

**Before launch** (each a HOLD on failure; done and recorded in the PREREG unless marked): the union built, soaked and
startup-measured at the pin; `checkargs` against X26c = the mix tokens and run name only; the launcher `--dry-run`;
the kit's pin, corpus and pool-seed checks; the material pack share ≤ 2 %; the GPU lease and the arm chain (the
Training session's).

---

## 9. How it interacts with the rest

- **X53-N1** (from scratch on 10⁴ teams). X63 asks whether a long breadth continuation of the plateau moves F; X53
  whether breadth from step 0 shapes the representation. X63's σ̂ is a continuation's seed spread, the nearest
  measured figure to X53's σ gate, not X53's σ itself.
- **X62.** Its corpus is built here (its G0) and is X63's largest component. X62's 20M fork remains its own
  registration (its §6 tree); X63 Stage A goes first (ruling 5).
- **PR-1′** shares the play pin and the lane. Its KILL retires "≤ 20M breadth forks at these corpora" and does not
  reach X63 (REVIEW_2); it moves X63's GUESS row (§3.3).
- **LR-F** read UNDECIDED, final: no breadth + floor arm is licensed by it. X63's controller path is REPORTED (§6.4).
- **The strategy's fork extension (rank 6)** stays the dose read; X63 does not replace it (M2).

---

## 10. RESPONSE to the reviews

**Review 1** (`ca51bf99`, 2026-10-27):

| item | handled |
|---|---|
| **B1** σ unmeasured; the one-seed screen retires for the era | FIXED. (1) A Stage-A stop is NOT PURSUED; RETIRE needs k ≥ 2. (2) σ_eff = max(1.0 pp, σ̂) with the t quantile on σ̂'s df whenever σ̂ > 1.0 pp; false GAIN ≤ 0.05 for σ ≤ 3 pp, stated at σ 2 and 3 pp beside the headline (§7). (3) X63-C was run first (`rb_x63c_s6302`, done 10-29) and plays in Stage A, so σ̂_CC exists at the screen. (4) The wobble trigger is dropped |
| **M1** the learning rate is the controller's response | FIXED: §2.1 corrected ("breadth under the production KL controller"); `main.dose` of every model REPORTED; the ≥ 1.44× divergence label pre-registered (§6.4). The PREREG is written after LR-F's final read (the controller is not the plateau's cause) |
| **M2** not a dose read | FIXED by (b): every dose claim removed (§0, §1, §4, §9); X63 does not replace the fork extension; the extension trigger is a GAIN only |
| **M3** mix by convenience | FIXED: n1 cut to its K3 prefix (0.134 of draws), a GUESS and P(GAIN) per variant (§3.3), U2 not run (H-CTX SUPPORTED makes it moot; it was the KILLED branch), x62's F coverage measured (PREREG §3) |
| **M4** the price has no working guard | FIXED: the S ceiling is withdrawn; net = Δ_home72 + r·Δ_F at the owner's r (§6.3); the floor is labelled goal (a)'s definition |
| **M5** engine fairness | FIXED: the frozen shared play pin (E1 game paths, the review's preferred option, built and checked 220 / 220) for every model |
| m1 GUESS per gate / PR-1′ class | FIXED (§3.3) |
| m2 cost and queue | FIXED: S and the REPORTED games counted (§5); the queue position is the orchestrator's, and the controls' 12,000 games may play before the arm ends |
| m3 rc72b at radius 3 | FIXED: the union builder refuses it, tested; the measured histogram is in the PREREG |
| m4 throughput gate admits a restart | FIXED: the gate is ≥ 1,200 steps/s (a PAGE), and ≤ 1 restart (the resume restores the optimizer's rate) does not VOID a seed (PREREG §8) |
| m5 the table's startup and memory | FIXED: measured at the pin (PREREG §3) |
| m6 the kit's parent field | FIXED by X63-C's kit work: a registered start node (`x26_380m`), reused by `x63u` |
| m7 pin X26c's final | FIXED: every model is pinned by sha256 in the read spec |
| m8 an ADOPT uses up H and P72 | FIXED: stated in the ADOPT row |
| m9 the price is on the protected teams | STATED (§6.3); a non-sample pool panel against Kakuna is DECLINED here: none is registered, building one is a panel build plus games, and the bias's direction is known (the price is understated) |
| m10 FINDING 1 stale | FIXED: §11 item 1 |

**Review 2** (a fresh blockers-only re-review, 2026-10-29; PREREG §13) found three, all fixed before registration:
(1) the frozen play pin's spec parser refuses a rule it does not hold, so the read plays from a successor pin carrying
only the new `x63_kvk` rule (§6.1); (2) r's deadline moved from the first X63-U game to X63-U's launch, closing a
window in which in-pool training signals preview the cost r prices (§6.3); (3) a GAIN with no r and a failed floor is
OVER-RATE, by a stated precedence (§7). It also checked that sharing seed 2601 with X26c makes the test more cautious,
not less.

---

## 11. FINDINGS (standing rule 7)

1. **The strategy doc** was corrected in place by REVIEW_2 (`69c7cd34`): X63 is ranked second, the 20M stack fork is
   superseded by it, and PR-1′'s KILL does not reach X63. X63 does not claim the fork extension's (rank 6) dose
   question (M2).
2. **σ is still unmeasured at registration.** σ̂_CC comes with Stage A's games; until then every error rate in §7 is
   conditional on σ. The rule is built to stay ≤ 0.05 false GAIN up to σ = 3 pp, not beyond.
3. **The read plays from a successor of the frozen shared pin** (`65b998da`: `86cfdf7e` + the `x63_kvk` rule only,
   play path byte-identical), because the frozen pin's spec parser refuses a rule it does not hold. The successor is
   a new named pin, as `reads.md` §4 requires for any change to a frozen one.
4. **The price is measured on the protected teams** (§6.3).
5. **X62's corpus has no soak record of its own**: all 12,000 of its teams are inside the union's soak (178,760
   battles, 0 core errors; and the pin's own soak, 0 core errors).
6. **D3's sixes are near-pool**, so the pool-like share is ≈ 0.33, and 59 of them share 3 sets with a forum72 team
   (D3 was built at radius 4): the pool's own overlap, inherited at half the pool's rate (PREREG FINDING 3).
8. **Per-worker memory of a full launch is unmeasured**; the team phase is 31 s at 0.71 GB and a whole-table env core
   peaks at 0.62 GB (PREREG §3). The first-hour throughput gate guards it.
7. **The kit holds Stage A's seed (2601) and Stage B's (6302).** Stage B's `rb_x63u_s6302` joined by AM-X63-BSPEC
   (PREREG §17, 2026-10-30) to train on idle GPU before Stage A's verdict. Its games play only on a Stage A CONTINUE;
   otherwise it is labelled "a speculative replicate, not read for any decision". Stage C's pair is added by a dated
   amendment after its gate reads.

---

## 12. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-27 | **CREATED (DRAFT)** (X63 designer) | X63-U: X26@380M + 100M at X26c's pin with a union mix at 0.8; X26c as the Stage-A control; a 1 v 1 screen, 2 v 2 with a 1 pp allowance, 3 v 3 on a near miss; F against Kakuna; home72 floor; an S ceiling | a P\*-start arm; X26's 280 → 380M as control; forum teams in the union; component weights; a 1 v 1 verdict; a home72 harm bar at 0 | the survey (`5d554c23`); X26c records; the strategy doc (`7574743d`); `power_x63.log` |
| 2026-10-27 | X62's revision propagated | the recombined third inherits X62's revised builder | — | `design_x62_team_recombination.md` §2.2 |
| 2026-10-29 | **REVISED IN PLACE and REGISTERED** (X63 registration worker, orchestrator brief; review 1's B1, M1–M5, m1–m10 answered) | Stage A = 1 arm v 2 controls (X63-C trained); NOT PURSUED, never RETIRE, on a screen stop; σ_eff = max(1.0, σ̂) with the t quantile above the floor, INCONCLUSIVE at σ̂ > 3 pp; n1 at its K3 tier (draws 0.537 / 0.134 / 0.129 / pool 0.2); the shared play pin `86cfdf7e`; home72 at 2,000 games; price net = Δ_home72 + r·Δ_F with r the owner's; no dose claim; the throughput gate at 1,200 steps/s; the N1-P look-2 amendment pre-written; x62_recomb_v1 and x63_union_v1 built and soaked | the draft's z rule (false GAIN 0.07–0.08 at σ ≥ 2 pp); a 1.5 pp floor (loses 0.2–0.3 power at σ ≤ 1 pp); a dose read by +20M points at 4,000 games (24–35 lane-h, seed term unverified to cancel); the S ceiling; n1 at 12,000; choosing r for the owner | `power_x63_v2.log`; H-CTX (`6f126c4af`), A-K (`fb84c260f`), D3 look 2 (`25a73617`), N1-P look 1, LR-F final; X63-C arm-end (`8a7f293f`); `reads.md` §4 |
| 2026-10-30 | **Stage B's seed trains speculatively** (AM-X63-BSPEC, PREREG §17; Stage-B-spec worker, orchestrator brief) | `rb_x63u_s6302` added to the kit at X63U_PIN, argv = s2601's with only the seed and run name changed; trains on the idle GPU before Stage A's verdict; played and read only on Stage A CONTINUE, else labelled "a speculative replicate, not read for any decision"; no rule, game or Stage A timing changes | training Stage B only after CONTINUE (≈ 1 day later if it fires, with the GPU idle meanwhile) | the GPU frees 10-30 ≈ 19:00 with nothing queued; Stage A's verdict lands 10-31 / 11-01 on CPU |
