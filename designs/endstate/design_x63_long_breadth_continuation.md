# X63 — a long breadth continuation of the plateau: 100M+ steps on a union corpus, pool share 0.2

**Status: DRAFT (2026-10-27), explicit-only. Nothing is built, registered or launched.** It turns recommendation 2 of
the context-generalisation survey
([`context_generalisation_2026-10-27.md`](../research_state/literature/context_generalisation_2026-10-27.md) §7,
`5d554c23`: "dose before more 20M plateau forks"; continue P\* ≥ 100M on the union corpus with the pool at ≤ 0.2) into
an experiment. Goal (b), its meter and every read cited are
[`design_beat_metamon_path.md`](design_beat_metamon_path.md)'s. The ranking rules (P(GAIN) per full cost, CPU
included, one lever per A/B) are [`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md)'s
(`7574743d`). The corpus it reuses is [`design_x62_team_recombination.md`](design_x62_team_recombination.md)'s; its
from-scratch cousin is [`design_x53_n1_fresh_breadth.md`](design_x53_n1_fresh_breadth.md). The power Monte Carlo is
[`power_x63.py`](../research_state/measurements/x63_design_2026-10-27/power_x63.py) → `.log` (CPU, numpy, 2 s).

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170 (X26's plateau, the named baseline `pstar_plateau`); **X26@380M** = X26's
`checkpoint_380000195_steps.zip` (sha256 `7e0ebe94…`); **X26c** = `rb_x26c_s2601`, X26@380M forked for 100M with no
lever (FLAT, [`x26c_continuation_2026-10-25/README.md`](../research_state/measurements/x26c_continuation_2026-10-25/README.md));
C0 = X35's 20M plain-continuation control of P\*; F = `forum72`; H = `heldout72_hpfix`; P72 = the procedural panel;
U = H, F and P72 at equal weight (goal (b)'s meter); home72 = 72 pool teams; S = in-house strength against the frozen
reference population R on home72; σ = the true between-seed SD of a +100M continuation's F score; **lane-h** = one
wall-hour of the Kakuna anchor lane at `--concurrency 1` on rule 5b's agent cores, at the MEASURED 160–230 games an
hour (the strategy doc's definition); GUESS = an unmeasured effect; ESTIMATE = a cost from measured rates; MY
JUDGEMENT = an inference no read states.

---

## 0. One screen

- **The question.** Does a long continuation of the plateau network on a broad team distribution (100M steps, 80 % of
  each side's teams from a ≈ 27,000-team union corpus) pilot unseen human-built teams better against Kakuna than the
  same network continued on the pool alone? Every breadth read so far was a 20M fork giving each corpus team ≈ 24
  pilots a side; this gives each ≈ 85, on the production network, with no toy in between.
- **The arm.** **X63-U**: X26@380M forked at X26c's own pin `f4b25238` (+ one corpus-only commit), X26c's argv token
  for token plus `--team-mix x63_union_v1@ALL:0.8 --team-mix-sides both`, +100M to 480M, checkpoints every 1M.
- **The control is X26c, and it is valid only from its own start** (§2). It is X26@380M + 100M at `f4b25238` with no
  lever: same parent checkpoint, same pin, same argv, same seed. **For an arm forked from P\* @280M it is a control at
  NO dose**, and X26's own 280→380M segment is cross-pin (`9c033489`) and not a fork, so it is DESCRIPTIVE only. So
  the arm starts at 380M and the brief's 300 / 330 / 380M read points become **400 / 430 / 480M** (+20 / +50 /
  +100M). One seed; its Kakuna rows must be RE-PLAYED (old engine E1, and forum72's pre-`b0441623` pack regime).
- **The mix** (§3): the union of X62's recombined sixes (12,000), N1-P's procedural teams (12,000) and D3's
  set-mutated pool teams (2,876), drawn **uniformly over teams** (0.357 / 0.357 / 0.086 of all draws), the pool 0.2.
  **Forum teams (`w_forum_v1`) are left out** so that F, the primary panel, stays free of its own population (F is
  D1's forum population; strategy FINDING 6). One `--team-mix` takes one corpus, so the union is built as ONE new
  corpus, not composed on the command line.
- **Seeds and GPU-h** (§5): Stage A one arm seed, **19.9 GPU-h** (X26c's measured 100M); Stage B one more arm seed
  plus one pool-only control replicate (+39.8); Stage C one more pair only on a near miss (+39.8). **Expected 28–38
  GPU-h under a null, 50–69 at a true +2 pp; at most 99.6.** CPU: Stage A 66–95 lane-h, each later stage 45–65.
- **The read** (§6): Kakuna on **F**, every model on ONE engine and the post-`b0441623` pack regime, 4,000 F games a
  model at +100M; the trajectory at +20 / +50M (REPORTED); home72 against Kakuna as goal (a)'s FLOOR (LB > 0.50) and S
  on home72 as the in-pool PRICE, registered as an expected cost, not a harm bar at 0.
- **The rule** (§7): Stage A is a SCREEN (CONTINUE iff Δ̂ ≥ +1.0 pp, else RETIRE); Stage B reads 2 v 2 with a 1 pp
  seed allowance (GAIN iff LB > 0); Stage C (3 v 3) only if B is UNDECIDED with Δ̂ ≥ +1.5 pp. **A true +3 pp reads GAIN
  0.91 / 0.86 / 0.71, +2 pp 0.46 / 0.45 / 0.40 (σ = 0 / 0.5 / 1 pp); false GAIN 0.001 / 0.005 / 0.026**, rising to
  0.057–0.069 at σ 1.5–2 pp (FINDING 3). P(GAIN) over the GUESS ≈ 0.43, about **0.06 per resource-day**.
- **The gates** (§8): H-CTX (rc72b) decides whether the recombined third exists; N1-P look 1 and D3 look 2 can each drop
  their component on a LOSS. PR-1 is not a gate: it tests resolution at 20M, X63 tests dose (§9).

---

## 1. Why a long continuation, and why now

**What the survey found** (§0 and §7 rec 2 there). Every breadth result in the literature trained on the broad
distribution for 10⁸ steps (Procgen 200M, CoinRun 256M, VGC-Bench's whole run, OpenAI Five). The one recipe that
starts from specialists, Honor of Kings, followed its fixed-lineup teachers with a LONG continuation on random
lineups from the whole pool. P\* is already the specialist; its next step in that recipe is a long broad continuation,
not another 20M fork. The survey ranks this second, after X62, and calls it the cheap cousin of X53-N1 (MEDIUM-LOW
confidence: the step counts are other domains').

**What our reads say.**
- **Plain continuation adds nothing.** X26c, 380M → 480M on the pool: U +0.08 pp [−3.90, +4.06], S −0.01 pp. So any
  movement X63 shows against X26c is the breadth, not the extra steps.
- **20M forks of breadth read +1 to +2 pp, none resolved.** D1 (1,981 forum teams, 0.5): U +2.10 [−0.92, +5.11], F
  +4.0 on its own population, in-pool S −1.61 [−2.90, −0.32]. D3 look 1: +1.27 [−2.63, +5.16]. N1-P is being read.
- **A 20M fork is a small dose.** At share 0.5 a corpus team is piloted ≈ 24 times a side; a pool team ≈ 395 (X62
  design §2, from D1's rule of ≈ 570k games per 20M). X52 needed ≈ 4,000 pilots a team to show a practised +3.2 pp.
- **The plateau can still learn.** The plasticity probe read NO-LOSS through 480M (strategy §2), so a long fork is not
  doomed by a frozen trunk.
- **The critic only changes through its targets** (H-MID, REVIEW M1). Breadth is the one lever that changes V's
  self-play targets as well as the policy.

**The dose explanation, made testable.** The strategy names five explanations of a null breadth family; "dose"
predicts that Δ keeps rising with more breadth steps. Its own dose read (rank 6, the fork extension: N1-P s3501 and C0
s3501 each +40M) costs ≈ 16 GPU-h and ≈ 180 lane-h for one corpus at +40M. X63 asks the question at +100M, on the
union, against a control that already exists.

---

## 2. The arms, and whether X26c is a valid control

| arm | run name (proposed) | start | the lever | role |
|---|---|---|---|---|
| **X63-U** | `rb_x63u_s2601` (Stage B `rb_x63u_s6302`, Stage C `rb_x63u_s6303`) | X26@380M | `--team-mix x63_union_v1@ALL:0.8 --team-mix-sides both` | treatment |
| **X26c** | `rb_x26c_s2601` (banked) | X26@380M | none | the Stage-A control, 0 GPU-h |
| **X63-C** | `rb_x63c_s6302` (Stage C `rb_x63c_s6303`) | X26@380M | none: X26c's argv with the seed and run name changed | the control replicates |
| P\* | `rb_x26_s2601` @280M | — | — | REPORTED anchor (is the arm above the named baseline?) |

**The argv** (X26c's `original_command`, read through `main.lineage`, plus the lever; a sketch, NOT a launch command):
`--restart-interval-hours 24 --pin-commit <X63 pin> --model …/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip
--behaviour-check warn --steps 480000195 --checkpoint-every-steps 1000000 --device cuda --team-mix
x63_union_v1@ALL:0.8 --team-mix-sides both --seed 2601 --run-name rb_x63u_s2601`. The registration prints
`python -m main.checkargs` of each arm against X26c's record; the only expected difference is the two mix tokens (and
the seed and run name for the replicates). A fork does not inherit a mix (team curriculum doc, "Restart vs fork"),
and X26c had none.

**The pin.** `f4b25238` (X26c's, E1, the X35 production pin) + ONE commit confined to `src/utils/team_corpora/` that
holds the union's files and its `CORPORA` line: the AM-N1P pattern (`main.x35.batch.N1P_PIN`), checked by
`main.x35.batch.arm_pin_check` against the union's tree. So the arm runs X26c's code byte for byte. The 10-26 and 10-27 GIGO fixes
(the observation, op-input and label changes since) are absent from both arms, as in every X35 arm.

### 2.1 Is X26c a valid control, and at what dose?

**Valid, for this arm, at every dose from +1M to +100M, as ONE seed.**
- **Matched:** the parent checkpoint (X26@380M, sha `7e0ebe94…`), the pin (`f4b25238`, `pin_history` one entry),
  the role (a fork; `lineage.role` = fork, the opponent pool seeded from X26's snapshots), the argv, the seed (2601),
  the restart interval (24 h), the checkpoint cadence (1M, so 400 / 430 / 480M exist), the card (the local 3080 Ti),
  and arm-end K9(b) PASS.
- **Read at matched steps:** X63-U @400 / 430 / 480M against X26c @400 / 430M and its `final_model.zip`
  (480,081,209, sha `cba7a1f8…`, the file X26c's own read played).
- **Its limits:**
  1. **One seed.** A 1 v 1 contrast cannot separate the lever from a seed's luck, so Stage A is a screen and the
     decision is read with control replicates (§5, §7).
  2. **Its Kakuna rows cannot be imported.** They were played on E1 under forum72's legacy pack (Kakuna's `sft431`
     Gyarados with Frustration at base power 1), which `b0441623` made its own regime; an import across it refuses
     (`designs/tools/reads.md` §4). X26c read only 400 F games at 480M anyway. Every X26c checkpoint is re-played.
  3. **Its home72 against Kakuna was never read** (its FINDING 8): re-played too.
  4. **Beyond +109.6M it does not exist.** X26c was later extended toward `--steps 580000195` and stopped at
     489,640,763 (dose segment `rb_x26c_s2601_seg2_480M_489M`, 2.07 GPU-h, 0 restarts; not in its README, FINDING 6).
     A dose extension past 480M resumes X26c to its recorded target (≈ 18 GPU-h).
- **Not valid for a P\*-start arm at any dose.** From P\* the pool-only paths are C0 (+20M, five seeds, `f4b25238`: a
  valid control at +20M only) and X26's own 280→380M (one seed, X26's pin `9c033489`, a same-run continuation, not a
  fork). The X53 design's cross-pin argument (163 commits touch `src/` between the two pins, including OU-visible Rust
  parity batches) makes the second DESCRIPTIVE. A P\*-start X63 would need its own pool-only fork to 380M per control
  seed (+19.9 GPU-h each) and would lose X26c.

**Why X26@380M is an acceptable start.** It is on the same plateau as P\*: X26c's 480M scores U level with P\*
(+0.08 pp), S was flat over 380 → 480M, and the plasticity probe reads no loss through 480M. X26@380M itself was never
read against Kakuna; Stage A reads it on F as the trajectory's origin when the lane has room (REPORTED).

---

## 3. The mix

### 3.1 The components

| component | teams | what it varies (X62 §1.2, measured) | panel hazard | in X63 |
|---|---:|---|---|---|
| `x62_recomb_v1` (X62, NOT BUILT) | 12,000 | **context**: every set a pool set (572 of the pool's 658 under X62's STRICT Hidden Power rule), ≤ 2 shared with any pool team | none (radius-3 refusal of every panel, rc72b and `forum216b`) | yes, if H-CTX is not KILLED |
| `n1_proc_v1` (N1-P) | 12,000 | context **and** sets: 72 % of slots new sets, procedural, incoherent | H and P72 (62 % / 56.5 % of their full sets in the corpus); F 19 % | yes, unless N1-P reads LOSS |
| `d3_setmut_v1` (D3) | 2,876 | sets inside familiar sixes: 4–5 sets shared with one pool team | none | yes, unless D3 reads LOSS |
| `w_forum_v1@ABC` (D1) | 1,981 | contexts, mostly familiar sets, human-built | **F is its own population** (D1: F +4.0 vs H +2.1) | **no** |

**Why the forum teams are out.** The survey's union names them, and they are the only human-built component. But the
read is F (the brief's, and the breadth family's shared panel), and F is drawn from the forum population D1 trains
on. With them in, no current panel is unhazarded for the union (the strategy's FINDING 6): F carries the forum source,
H and P72 carry N1-P's generator. Without them F is clean for every component (X62: none; N1-P: F is its one
unhazarded panel; D3: none). Their dose would have been small anyway (1,981 of ≈ 28,900 teams, ≈ 5.5 % of draws), and
X62's sixes carry what they add (familiar sets in new contexts; 31 % of D1's teams are recombined-like). The four-way
union (U4) is named in §10 item 7 as the variant whose read would need the sealed `forum216b`, which is X54's.

### 3.2 How the mix is composed

`--team-mix` takes ONE `<corpus>[@<tier>]:<share>` (`agents.training.matchup_spec.parse_team_mix`; the team curriculum
doc). So the union is one new registered corpus, **`x63_union_v1`** (proposed), built by a builder beside
`build_n1_proc` (proposed name `build_x63_union`):
- it loads each component at its registered tier through the verifying reader (`load_dir`, sha256s, the label-path
  guard), concatenates the rows with `curator` set to the component's name, and refuses a duplicate six across
  components; it reports the cross-component near-duplicates (≥ 5 shared sets) in the manifest, since X62's draw
  does not refuse D3 or N1-P teams (the X62 review's m8);
- its manifest records each component's name, tier, `tier_sha256` and count, so the union's identity names its parts;
- one tier `ALL` (a dose split by tier is not this design's question);
- the soak at the X63 pin (`utils.team_corpora.soak`, 10 battles a team as N1-P's, ≈ 269,000 battles, ≈ 70–80 min at
  nice 19: ESTIMATE scaled from X62 §2's 30–35 min per 120,000).

**Uniform over teams, not weighted by component.** A draw from the union is uniform over its teams (the `--team-mix`
semantics), so each component's share is its count's. This is Procgen's sampling (uniform over levels), it needs no
code change, and every corpus team gets the same ≈ 85 pilots a side. A component-weighted mix would need either a
multi-corpus `--team-mix` (a build: the parser, `TeamMix`, the builder's draw, the spec hash and the Rust team table)
or duplicated rows; neither buys information this design needs.

**The draw shares** (the trainee's non-corpus 0.2 is the pool with its usual 10 % sample bias; the opponent's is the
uniform pool):

| union (by the gates, §8) | teams | x62 | n1 | d3 | pool | pilots a side per corpus team over 100M |
|---|---:|---:|---:|---:|---:|---:|
| **U3** (the expected case) | 26,876 | 0.357 | 0.357 | 0.086 | 0.2 | ≈ 85 |
| U2: H-CTX KILLED (no x62) | 14,876 | — | 0.645 × 0.8 = 0.516 | 0.124 | 0.2 | ≈ 153 |
| N1-P LOSS (no n1) | 14,876 | 0.516 | — | 0.124 | 0.2 | ≈ 153 |
| D3 LOSS (no d3) | 24,000 | 0.4 | 0.4 | — | 0.2 | ≈ 95 |

(Pilots by D1's rule, ≈ 2.85M games per 100M: 2.85M × 0.8 / teams. A pool team gets ≈ 790 a side against X26c's
≈ 3,960, a fifth.) D3's teams keep 4–5 of a pool six, so the pool-LIKE share of U3's draws is ≈ 0.29, not 0.2
(FINDING 4).

**The startup guard** (`main.train.matchup_setup.team_mix_guard`) re-checks the union at the pin: no exact copy or
near neighbour (≥ 4 shared sets) of a held-out panel, no validator drop, the TEAM-MIX banner and 0 pool copies. Each
component passed it alone, so the union passes by construction; the guard runs anyway.

**The startup cost.** N1-P's 12,000-team table took 18.5 s of team phase and trained at 1,388–1,433 steps/s, level
with C0's. A 26,876-team table is UNVERIFIED (FINDING 5); the arm chain's first-hour throughput gate (≥ 1,000 steps/s)
is the guard.

---

## 4. The dose, and the checkpoints read

- **+100M (to 480M) is the registered dose.** It is the shortest that matches the field's order of magnitude for a
  breadth result (10⁸ steps) and the longest the control covers (X26c to 480M; 489.6M exists).
- **Read points: +20M (400M), +50M (430M), +100M (480M).** Only +100M decides. +20M is comparable in dose with the
  20M forks (D1, D3, N1-P), but from another start, so the comparison is DESCRIPTIVE. The trajectory is REPORTED as
  the dose curve.
- **The dose extension, +200M (580M), is a separate registration**, licensed only by a GAIN or by an UNDECIDED whose
  trajectory rises (Δ̂ at +100M above Δ̂ at +50M by ≥ 1 pp, REPORTED, never a verdict). It costs ≈ 19.9 GPU-h for the
  arm and ≈ 18 for resuming X26c to its recorded 580M target, plus the reads.

---

## 5. Seeds, stages and cost

The rate: X26c's own dose record, **380M → 480M in 19.92 GPU-h at 1,402 steps/s** (startup and restarts included).
The brief's measured fork rate (≈ 1,450 steps/s) gives 19.2. The table uses 19.9.

| stage | new runs | GPU-h | F games (Kakuna) | lane-h | when |
|---|---|---:|---:|---:|---|
| **A** | `rb_x63u_s2601` | 19.9 | 8,000 at +100M (arm and X26c, 4,000 each) + 4,800 trajectory (1,200 per model at +20 and +50M) + 2,400 home72 (1,200 each at +100M) = 15,200 | 66–95 | after the gates (§8) |
| **B** | `rb_x63u_s6302`, `rb_x63c_s6302` | 39.8 | 8,000 F + 2,400 home72 = 10,400 | 45–65 | Stage A CONTINUE |
| **C** | `rb_x63u_s6303`, `rb_x63c_s6303` | 39.8 | 10,400 | 45–65 | Stage B UNDECIDED with Δ̂_B ≥ +1.5 pp |
| **at most** | 5 runs | **99.6** | 36,000 | 156–225 | |
| expected, true Δ = 0 (σ 0–1 pp) | | **28–35** | | ≈ 75–120 | |
| expected, true Δ = +2 pp (σ 0–1 pp) | | **59–69** | | ≈ 110–175 | |

- **Seeds.** Stage A uses 2601, X26c's own, so the arm's argv differs from its control's only by the lever. The
  replicates pair seeds (6302, 6303) the same way. The RNG streams diverge at the first team draw (the mix's extra
  `random()` call), so the pairing buys tidiness, not correlation.
- **In-house S** (the price, §6.3): S of each final against R at D1's regime. X26c's S2 at 480M (58.79, regime
  `d4de0fd7…`, E1) imports only if the S read plays on E1; otherwise it is re-played. ≈ 1.5 heavy-slot-h a model
  (ESTIMATE from D3 look 2's 7.5 h for its finals; UNVERIFIED).
- **Agent-days ≈ 2** before Stage A: the union builder and soak (≈ 0.5), the kit arm, its pin and a kit test of the
  `n1p is C0 plus the mix` kind (≈ 0.5), a `main.reads` rule for the k v k contrast with the seed allowance (≈ 0.5),
  the PREREG (≈ 0.5). X62's corpus build is X62's own (its G0, ≈ 1 agent-day).
- **The card.** All local: the card rule (`cloud_gpu_plan.md` §1.2) forbids a comparison across card types, and X26c
  trained on the local card. A rented card could carry a Stage-B or C pair only together with its own control, and
  pooling across cards is not licensed by that rule (UNVERIFIED that it would be).
- **Wall time.** Stage A trains in ≈ 20 h; its read is the binding part (66–95 lane-h ≈ 3–4 days at `--concurrency 1`).
  **X26c's replays need no GPU and no arm**: they can play as soon as the read's engine is registered, before the arm
  launches.

---

## 6. The read

### 6.1 Primary: goal (b) on F against Kakuna

- **Cell:** Kakuna checkpoint 34, greedy against greedy, CPU, `forum72`'s committed config, role-balanced 20-game
  units, new unit seeds, anchor group `primary`; 200 units (4,000 games) per model at +100M.
- **ONE engine, ONE regime.** Every game of every model (the arms, X26c's checkpoints, the replicates, the REPORTED
  anchors) is played from one playing commit under `main.reads` (one state), after `b0441623` (the
  `gen3_peer_showdown_pack_v1` regime, SOP H22). Each unit records its engine stamp and `peer_pack`; two stamps in a
  group is INCONCLUSIVE, and no banked forum72 row from before H22 is imported (the import refuses).
  - **Which engine, in order of preference:** (a) PR-1's registered engine, if PR-1 is registered by then (the
    strategy's "tree's eval engine"), so X63's numbers sit beside PR-1's; (b) otherwise a hybrid play pin like D3 look
    2's AM-D3-L2 (`c4a5c37b6`), but on the other side of H22: our game paths at `f4b25238`'s bytes (E1, both arms'
    training engine) under a post-`b0441623` harness and peer pack, checked by re-playing banked E1 units on panels
    H22 left unchanged (home72, H, P72) game for game, choice by choice. Whether such a hybrid can be assembled is
    UNVERIFIED (H22 added `utils/bridge` files that AM-D3-L2 restored to `f4b25238`'s bytes; FINDING 7); (c) failing
    both, one post-`b0441623` commit of `main` for every model. Every option is fair to the contrast; (b) keeps the
    play nearest the training engine.
- **Statistic:** Δ_F = mean over arm seeds of F − mean over control seeds of F, each a plain proportion of its model's
  games. Interval: normal, SE² = Σ p̂(1 − p̂) / G over the 2k models / k² + σ_eff² · 2 / k, with **σ_eff = max(1.0 pp,
  σ̂)** and σ̂ the within-arm seed spread net of its binomial part (the AM-5 idea, with a registered floor because df
  is 2–4). A team-cluster bootstrap over F's 72 teams is REPORTED beside it.
- **Classes** (rule 8: a bound within 1e-9 of a threshold ⇒ UNDECIDED): §7.

### 6.2 Goal (a)'s floor: home72 against Kakuna

1,200 games per model at +100M, same cell and engine. **FLOOR PASS iff the arm's pooled home72 LB > 0.50**, so goal (a)
stays MET (P\* 0.598 [0.549, 0.644]). It is a floor on the arm, not a contrast with X26c: the literature makes an
in-pool drop the expected sign of breadth working (CoinRun's train score, Hill, VGC-Bench's seen teams; D1's −1.6 pp),
so a harm bar at 0 would kill the lever for its predicted side effect (the X62 review's B1 measured how: X62's
NON-INFERIOR co-primary holds only 0.07 of the time at D1's cost). Δ_home72 = arm − X26c is REPORTED against its
registered GUESS (§6.4).

### 6.3 The in-pool price: S on home72

S of each final against R (D1's co-primary regime), Δ_S = arm − control, Welch 97.5 %. **Registered as a PRICE, with a
ceiling, not as non-inferiority:** the verdict is TRADE (§7) iff Δ_S's LB ≤ −6.0 pp (≈ 4× D1's point estimate at a
fifth of the dose, MY JUDGEMENT). D1's NON-INFERIOR bar (LB > −2.0) is REPORTED beside it and decides nothing.

### 6.4 REPORTED, never decided

- **The dose curve:** Δ_F at +20 / +50 / +100M (1,200 games per model at the first two).
- **The plateau wobble:** X26c's F at 380 (if played) / 400 / 430 / 480M. Its spread net of the binomial part is a
  zero-GPU hint of σ (two seeds far apart on a treadmill differ like two far-apart checkpoints of one lineage, MY
  JUDGEMENT), printed beside σ̂.
- **The unselected estimate:** Stage B's new pair alone (and C's), which the screen did not select; Stage A's pair is
  selected for Δ̂_A ≥ +1.0, so the pooled Δ̂ carries a winner's curse.
- **H and P72** against Kakuna, 400 games per model at +100M, each with the n1 set-coverage line (N1-P PREREG §8's M4
  hazard): a gain there may be the generator. **U** is printed, and goal (b)'s own number may be read from F alone.
- **rc72b** (held-out recombinations of pool sets), only if the union is measured to keep its radius-3 distance from
  rc72b (FINDING 8).
- **P\*** on F (1,200 games): is the arm above the named baseline, not only above X26c?
- **H-MID's Δbias** on the arm's and X26c's F and home72 captures (X62's M3): does a breadth-trained critic see more of
  the premium?

**Predictions (GUESS, registered before any game; given the gates in §8 favourable):**

| quantity | point | 80 % interval |
|---|---|---|
| **Δ_F at +100M (primary)** | **+2.0 pp** | (−1.0, +5.0) |
| Δ_F at +20M / +50M | +0.5 / +1.2 | (−2.0, +3.0) / (−1.5, +4.0) |
| Δ_home72 against Kakuna | −2.0 | (−5.0, +1.0) |
| **Δ_S on home72 (the price)** | **−2.5** | (−4.5, −0.5) |
| Δ_H / Δ_P72 (hazarded by n1) | +2.5 / +3.0 | (−1.0, +6.0) / (−1.0, +7.0) |
| Δ_F if N1-P and D3 both read NULL | +1.0 | (−1.5, +4.0) |

Basis (MY JUDGEMENT): X26c is flat, so Δ is breadth alone; F's whole home72 − F premium is 6.8 pp and a 100M dose
recovering ≈ 30 % of it gives ≈ +2; D1's 20M read F +4.0 on its own population and H +2.1; the survey's dose argument.

---

## 7. The decision rule, its power, and what each verdict licenses

**The rule.**
1. **Stage A (screen, 1 v 1):** X63-U s2601 against X26c at +100M. **CONTINUE iff Δ̂_A ≥ +1.0 pp** (a point threshold,
   not an interval: one seed per side cannot carry a seed allowance worth reading). Else **RETIRE**.
2. **Stage B (2 v 2):** the classes on the four models. **GAIN iff LB > 0** (and the INCONCLUSIVE guard clear);
   **RETIRE iff UB < +1.5 pp** (PR-1's KILL bar); **INCONCLUSIVE iff σ̂ > 2.0 pp**; else UNDECIDED.
3. **Stage C (3 v 3), only if B is UNDECIDED with Δ̂_B ≥ +1.5 pp:** the same classes on six models. The two looks are
   not alpha-corrected; the power table's false-GAIN column includes that.
4. The goal-(a) floor (§6.2) and the price ceiling (§6.3) are read only on a GAIN.

**Power** (`power_x63.log`, 40,000 draws a cell, base F 0.53, 4,000 F games a model):

| true Δ_F | σ = 0: GAIN / RETIRE | σ = 0.5 pp | σ = 1 pp | σ = 1.5 pp | E[GPU-h], σ 0–1 |
|---|---|---|---|---|---:|
| 0 | **0.001** / 0.81 | 0.005 / 0.77 | 0.026 / 0.71 | 0.057 / 0.66 | 28–35 |
| +1.0 | 0.056 / 0.49 | 0.089 / 0.49 | 0.136 / 0.50 | 0.155 / 0.50 | 47–48 |
| +1.5 | 0.20 / 0.32 | 0.23 / 0.35 | 0.26 / 0.39 | 0.24 / 0.42 | 54–61 |
| **+2.0** | **0.46** / 0.18 | 0.45 / 0.22 | 0.40 / 0.29 | 0.33 / 0.34 | 59–69 |
| **+3.0** | **0.91** / 0.04 | 0.86 / 0.06 | 0.71 / 0.13 | 0.53 / 0.20 | 63–67 |
| +4.0 | 0.995 / 0.003 | 0.98 / 0.01 | 0.88 / 0.05 | 0.67 / 0.10 | 61–62 |

The rest of each row is UNDECIDED or INCONCLUSIVE (the latter 0.03–0.05 at σ 1, 0.09–0.17 at σ 1.5).

**The alternatives, from the same log:**
- **No Stage C** (2 v 2 only): +2 pp 0.26 / 0.28 / 0.28 (σ 0 / 0.5 / 1), +3 pp 0.73 / 0.67 / 0.55, for ≈ 10–17 fewer
  expected GPU-h at +2 pp (≈ 1–3 under a null). Stage C buys ≈ +0.15–0.2 at +2 pp where it is needed.
- **3 v 3 at once:** the same or slightly lower power than the registered shape at +2 / +3 pp, for ≈ 16–18 more
  expected GPU-h at +2 pp and ≈ 7–9 more under a null.
- **A +0.5 pp screen:** +2 pp 0.48 / 0.47 / 0.42, false GAIN unchanged, ≈ 4–6 more expected GPU-h under a null.
- **Stage A read as a verdict** (1 v 1 with the 1 pp allowance): +3 pp 0.32 / 0.34 / 0.39. Read **without** the
  allowance (binomial SE only), a true null reads GAIN 0.026 / 0.050 / 0.11 / 0.18 at σ 0 / 0.5 / 1 / 1.5 pp: **a single
  pair must never be read as a verdict.**

**P(GAIN) over the GUESS** (normal; SD = 80 % width / 2.563, `pgain_rows.py`'s approximation): **0.47 / 0.43 / 0.35**
at σ 0.5 / 1 / 1.5 pp for +2.0 (−1.0, +5.0); 0.31 / 0.29 / 0.24 for the both-NULL GUESS +1.0 (−1.5, +4.0).
**Per resource-day:** ≈ 0.43 / ((≈ 55 GPU-h + ≈ 130 lane-h) / 24 ≈ 7.7) ≈ **0.06**, in the band of the strategy's
ranks 5–9 (0.06–0.07) and below PR-1's stack (0.12); the lane is three quarters of the cost, so a rented CPU box
roughly doubles the ratio (MY JUDGEMENT arithmetic).

**The verdicts.**

| verdict | rule | what it licenses |
|---|---|---|
| **ADOPT-CANDIDATE** | F GAIN, goal-(a) FLOOR PASS, Δ_S's LB > −6.0 pp, every VOID guard clear | (1) the dose extension (§4) to find the continuation's own plateau; (2) the arm's final as a candidate new plateau baseline: its X54 read on `forum216b` (the goal-(b) claim) and a `main.baselines` proposal are their own units, and nothing joins the recipe on this read; (3) X53-N1's prior rises; if N1-P did not read GO, the owner is asked whether the from-scratch version is worth its 150–250 GPU-h (survey rec 2: this read is its decider) |
| **TRADE** | F GAIN, and the floor or the ceiling fails | the owner, with the exchange rate Δ̂_F / −Δ̂_S and the home72 number |
| **RETIRE** | Stage A Δ̂_A < +1.0 pp, or a later UB < +1.5 pp | "a 100M union continuation of the plateau at pool share 0.2, with these corpora" is retired for goal (b) in this era, with the achieved power printed. The dose explanation loses its plateau form; capacity (N4) and the scale bet remain (strategy T3). It cannot read NULL: a screen stop at a true +2 pp happens 0.18–0.29 of the time |
| **UNDECIDED** | Stage C (or B when C does not fire) reads neither | the orchestrator, with the trajectory, σ̂ and the unselected estimate |
| **INCONCLUSIVE** | σ̂ > 2.0 pp | more seeds or stop (the orchestrator); σ̂ itself is a finding for X53's σ gate |
| **VOID** (a seed) | the TEAM-MIX banner (`x63_union_v1`, 0 panel hits, 0 pool copies), the recorded pin, the first-hour throughput ≥ 1,000 steps/s, arm-end K9(b), one engine and one regime in the read | never read; re-run at the same pin and seed after the cause is fixed |

---

## 8. Gating: what must read first

The PREREG is written after all three reads below land, before any X63 game; the union's composition is frozen there.

| read | outcome | effect on X63 |
|---|---|---|
| **H-CTX** (rc72b, `premium_recovery`) | SUPPORTED or UNDECIDED | the recombined third is in: X62's G0 builds `x62_recomb_v1` (on UNDECIDED it builds and holds), and the union is U3 |
| | KILLED | no x62: the premium is set-level and the union is U2 (n1 + d3, new sets); GUESS unchanged, the union now carries only set novelty |
| | INCONCLUSIVE / VOID | HOLD, as X62's own tree does |
| **N1-P look 1** (+ look 2 on CONTINUE) | GAIN | n1 in. X53-N1 is licensed by its own PREREG; X63 still runs (a different question, at about half X53's Stage-A GPU); the order is the orchestrator's |
| | NULL / UNDECIDED (final) | n1 in: a 20M NULL is weak evidence against breadth at a larger dose (survey §6.2); the GUESS falls to the both-NULL row if D3 also misses |
| | LOSS | n1 out (incoherent procedural sets cost) |
| **D3 look 2** | GAIN / NULL / UNDECIDED | d3 in |
| | F LOSS, or P INFERIOR | d3 out |
| H-CTX KILLED **and** N1-P LOSS | — | X63 is not run: only the near-pool d3 remains |

**Rulings of 2026-10-27** (precommit memo §4, accepted before the reads landed): (1) N1-P CONTINUE ⇒ n1 is IN and this PREREG does not wait for look 2 (only a LOSS removes n1). (4) d3 leaves X63 iff D3's registered H class is LOSS, or Δ_P's 97.5 % UB < −2.0 pp; "non-inferiority not shown" alone keeps d3 in. The "F LOSS, or P INFERIOR" row above reads as this.

**Not gates.**
- **PR-1** (§9): it reads resolution at 20M; X63 reads dose. Its engine, if registered first, is X63's engine (§6.1).
- **X59-S / X60:** X63 carries no human-data corpus (`n1_hrep_v1` is never merged into a union; the human-data
  architecture doc's rule).
- **LR-F, the N4 toy:** independent questions (§9).
- **X64** (the toy context-count curve): it informs the GUESS, and it routes the survey's rec 2 through its own
  SHRINKS-U class; this doc does not wait on it (§9, FINDING 12).

**Before launch** (each a HOLD on failure): the union built and soaked at the X63 pin; `python -m main.checkargs` diff
against X26c = the mix tokens only; the launcher `--dry-run`; the kit's pin and corpus checks; `forum216b` confirmed in
the startup guard's panel set (FINDING 9); the GPU lease (root rule 5) and the arm chain (`python -m main.ops.arm_chain`)
with the arm's own pin.

---

## 9. How it interacts with X53-N1, X62, PR-1 and the rest

- **X53-N1** (from scratch, n1 at 0.5, behind N1-P's GO and its σ gate). The survey calls X63 its cheap cousin and the
  read that decides whether X53 is worth its GPU. They answer different questions: X53 whether breadth shapes the
  representation from step 0; X63 whether dose at the plateau moves F. On N1-P GO both are licensed; X63 Stage A costs
  ≈ 20 GPU-h against X53 Stage A's ≈ 39, and neither blocks the other. On N1-P not GO, X53 is strategy rank 11 behind a
  plasticity LOSS that did not happen; an X63 ADOPT-CANDIDATE is then the one result that re-opens it (to the owner).
  X63's σ̂ (a continuation's seed spread) is not X53's σ (a fresh run's), but it is the nearest measured figure.
- **X62** (P\*-fork, x62 at 0.5, 20M). X63 needs X62's corpus, not its fork read. X62's fork is the attribution read
  (is the recombined component itself worth anything at 20M?); X63 is the dose read on the union. X62's revised tree
  (`d997a146`'s B2) retires its own 20M fork on a PR-1′ KILL but keeps the corpus for X63, and sends the corpus to
  X63's union on N1-P GAIN or PR-1′ BREADTH-GAIN; X63 is a home for it in every branch where H-CTX is not KILLED. If both run, X62's read goes first on the lane only if its registration says so: the survey ranks X62 first.
- **PR-1 and the strategy's tree.**
  - Lane contention is the real interaction: PR-1 165–238 lane-h, X62's read 126–181, X63's Stage A 66–95, all on one
    lane. Unless a rented CPU box is approved (the owner question PR-1 already carries), they run in series: weeks.
    X63's X26c replays can interleave from day one.
  - **PR-1 KILL** retires "breadth at the plateau, ≤ 20M, at these corpora" and the strategy adds "no further plateau
    breadth fork". X63 is the ≥ 100M form, and KILL-at-20M with GAIN-at-100M is exactly what the strategy's "dose"
    explanation predicts. Whether X63 runs after a KILL is the orchestrator's ruling; this doc recommends it does
    (FINDING 1).
  - **PR-1 BREADTH-GAIN** licenses the strategy's breadth STACK (rank 3: a 20M P\*-fork on the union of forum, d3 and
    n1). X63 is that stack's long form without the forum teams; the survey's rec 2 ("dose before more 20M forks")
    argues X63 replaces the 20M stack fork. Recommended, not applied (FINDING 1).
  - **The fork extension** (rank 6: N1-P s3501 and C0 s3501 +40M, ≈ 16 GPU-h, ≈ 180 lane-h) is the strategy's dose
    read. X63 asks it at 2.5× the dose with a free control for ≈ 4 more GPU-h and ≈ half the lane-h, at the price of
    a union instead of one corpus. Recommended as its replacement (FINDING 1).
- **LR-F** (P\* continued with a floor on the learning rate, GPU queue next). X63 inherits X26c's learning-rate path,
  so its Δ is "breadth at that path". An LR-F GAIN would make a breadth + floor arm the natural follow-up, as its own
  registration (one lever per A/B).
- **X64** ([`design_x64_toy_context_curve.md`](design_x64_toy_context_curve.md), the survey's rec 5: three fresh toy
  arms at 719 / 3,719 / 12,719 teams, ≈ 50 GPU-h + ≈ 103–147 lane-h, gated on rc72b not KILLED and X62's corpus).
  Its SHRINKS-U class "licenses" the long union continuation (rec 2, i.e. X63) to ask for its GPU; its FLAT / GROWS
  classes argue against it. This doc treats X64 as information, not a gate: X63's Stage A (≈ 20 GPU-h, 66–95 lane-h)
  is cheaper than X64 and asks the production question directly, while X64's transfer to production is LOW
  confidence by its own survey (the toy's premium is about half P\*'s). If X64 has read first, its class moves X63's
  GUESS (SHRINKS-U: the favourable row; FLAT / GROWS: the both-NULL row or lower). The two compete for the same lane
  and both need X62's corpus; which runs first is the orchestrator's (FINDING 12).
- **N4** (capacity). Independent. On an N4 toy GO the scale bet (N4 × N1 from scratch) is the owner's; X63's union is
  a candidate corpus for it.
- **D1 / D3 / N1-P banked finals.** Not comparable in Δ (another start, another dose); their rows are context only.

---

## 10. Adversarial self-check

1. **"X26c is one seed; any 1 v 1 contrast is seed noise."** Agreed, which is why Stage A only screens and the decision
   needs 2 v 2 or 3 v 3 with a registered seed allowance. Read 1 v 1 without the allowance, a true null reads GAIN
   0.11 at σ 1 pp (§7).
2. **"The 1 pp allowance is a guess; at σ ≥ 1.5 pp false GAIN is 0.057–0.069."** True (FINDING 3). The guard
   (σ̂ > 2 pp ⇒ INCONCLUSIVE) catches 0.09–0.17 of draws there. The registration raises the allowance to 1.5 pp if,
   before any X63 game, PR-1's φ̂ implies σ_s ≥ 0.75 pp at 20M (its own trigger for "seeds are the only remedy"), or if
   X26c's plateau wobble (§6.4, playable first) exceeds 1.5 pp. At 1.5 pp power at +3 falls to ≈ 0.5.
3. **"Starting at 380M is not 'continuing P\*'."** It is X26 100M later on the same plateau (X26c's 480M = P\* on U;
   S flat; plasticity NO-LOSS). Starting at P\* costs a new pool-only control per seed and loses the free one. The
   arm's P\* line (§6.4) answers whether it beats the named baseline, not only X26c.
4. **"The union confounds which component works."** By design: X63's lever is the dose of a broad distribution, as in
   the literature it tests. Component attribution belongs to the 20M reads (X62, N1-P, D3). A GAIN makes a better
   baseline candidate whichever part carried it; a RETIRE retires the union at this dose, not each part.
5. **"Uniform over teams lets procedural, incoherent teams take a third of the draws."** Yes (0.357). N1-P's read is
   the gate: a LOSS drops them. An n1-driven gain would show on H and P72 (hazarded) more than F; F is the decider
   because it is clean of n1's generator.
6. **"A pool share of 0.2 breaks goal (a)."** The floor (home72 LB > 0.50) guards it, and the price is registered
   before any game. D3's near-pool sixes make the pool-like share ≈ 0.29.
7. **"The survey's union includes the forum teams; leaving them out departs from the recommendation."** It does, to
   keep F clean (§3.1). The four-way union U4 has no clean panel: F carries the forum source, H and P72 the
   generator, and `forum216b` is X54's sealed panel drawn from the same forum source. U4 is a separate registration
   if the orchestrator prefers it, read on U with every panel's hazard line.
8. **"F has little headroom (P\* 0.530, premium 6.8 pp); +3 pp is 44 % of the premium."** True; the realistic range
   is +1.5 to +3, where this design's power is 0.2–0.9. The PR rule's 17,400 games an arm would not help much: the
   seed allowance, not the games, dominates the interval (4,000 → 8,000 games a model moves SE from 1.27 to 1.15 pp at
   2 v 2).
9. **"The read engine is not the training engine."** Option (b) of §6.1 keeps our game paths at E1; either way both
   arms and the control play on one engine, and a cross-engine divergence (≈ 1 % of games, ours) cannot bias a
   same-engine contrast.
10. **"The screen and the final reuse Stage A's pair: selection bias."** The power table includes it (false GAIN is
    computed through the screen); the unselected Stage-B estimate is REPORTED (§6.4).
11. **"X63 trains at a 10-day-old pin without the 10-26 / 10-27 fixes."** The price of a free control. A HEAD-pin X63
    needs its own control seeds (+19.9 GPU-h each) and a new P\*-lineage start at that pin's code; an ADOPT-CANDIDATE
    at `f4b25238` is a candidate, and moving it to HEAD is the baseline proposal's problem, as for every X35 arm.
12. **"X26c FLAT already says more steps do nothing; why spend 20–100 GPU-h?"** Because X26c is the control that
    makes X63 cheap: every pp X63 shows over it is the breadth's. X26c's flatness is exactly what isolates the lever.
13. **"In-house meters are blind to the deficit; S cannot see the gain."** S is read only for the price (pool-side
    piloting, which it sees). The gain is read only on Kakuna.
14. **"The literature's step counts are from-scratch runs, not continuations."** Honor of Kings is the continuation
    precedent; the rest set the order of magnitude. Confidence stays MEDIUM-LOW (the survey's).

---

## 11. FINDINGS (standing rule 7)

1. **Not applied to the strategy doc (explicit-only, out of this unit's scope):** this doc recommends that X63 replace
   the strategy's rank 6 (fork extension) as the dose read and its rank 3 (the 20M breadth stack) as the union's form,
   and that a PR-1 KILL not retire X63 (KILL's own scope is ≤ 20M). The orchestrator rules; the strategy's §3.2 and §4
   are unchanged, and `EXPERIMENT_BACKLOG.md` has no X63 row.
2. **X62's corpus is not built.** The X62 review's prototype draw
   ([`x62_review_2026-10-27.md`](reviews/x62_review_2026-10-27.md)) fits 12,000 sixes under the ≤ 4 cap, so U3's
   12,000 holds. X62's revision applied the review's M1, and the union inherits it: rc72b refused at radius 3, the
   STRICT Hidden Power rule (86 of the pool's 658 sets and 6 species leave the recombined third; its set TVD from the
   pool is 0.17), and near-duplicates against D1, D3 and N1-P reported (none ≥ 5 in the prototype).
3. **The false GAIN exceeds 0.05 at σ ≥ 1.5 pp** (0.057 / 0.069 at 1.5 / 2.0). No read has measured a +100M
   continuation's seed spread; C0's 20M φ̂ (0.6–0.77) was read at 420–1,200 games a seed.
4. **D3's sixes are near-pool** (4–5 shared sets), so U3's pool-like share is ≈ 0.29, above the survey's ≤ 0.2 in
   spirit; a stricter reading drops d3 (U = x62 + n1, 24,000 teams).
5. **A 26,876-team table's startup and throughput are UNVERIFIED**; N1-P's 12,000 were level with C0.
6. **X26c's second segment** (480 → 489.6M toward `--steps 580000195`, 2.07 GPU-h, 0 restarts, interrupted) is
   recorded only in its dose record, not in its README or the ledger. It does not touch the 480M final the read uses.
7. **The hybrid play pin (§6.1 b) is UNVERIFIED.** AM-D3-L2 restored `src/utils/bridge/` to `f4b25238`'s bytes; H22's
   peer pack added files there. Whether E1's game paths and the post-H22 peer can share a play path is a build check.
8. **rc72b's distance to the union is not yet guaranteed.** rc72b's own build refused overlap with D1, D3 and N1-P
   (the X62 review's m8), and X62's revised draw refuses rc72b at radius 3 (its review's M1; without it 491 of 12,000
   prototype teams sat at ≥ 3). rc72b is REPORTED only once the built union is measured at ≤ 2 against it.
9. **Whether the startup guard's panel set includes the SEALED `forum216b` is UNVERIFIED** here; `w_forum_v1` and
   X62 exclude it by construction, `n1_proc_v1` and `d3_setmut_v1` by their builds' panel lists (not re-checked).
10. **The S cost per model (≈ 1.5 heavy-slot-h) is an ESTIMATE**, and X26c's banked S2 imports only on E1.
11. **The kit (`main.x35.kit`) is built around P\*-forks;** whether it can describe an X26@380M start without a change
    is UNVERIFIED (part of the ≈ 0.5 agent-day kit item).
12. **X64 (landed concurrently, 2026-10-27) routes rec 2 behind its own SHRINKS-U class; this doc does not.** The
    two designs disagree on whether the toy curve gates the production continuation; the orchestrator rules. X64's
    lane (≈ 103–147 lane-h) adds to the contention of §9.
13. This unit ran no game, used no GPU and wrote no code outside the measurement script; the Monte Carlo ran on the
    agent cores at nice 19 (2 s).

---

## 12. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-27 | **This doc CREATED (DRAFT)** (X63 designer, orchestrator brief; docs and a CPU Monte Carlo) | X63-U: X26@380M + 100M at X26c's pin with `--team-mix x63_union_v1@ALL:0.8 --team-mix-sides both`, the union of x62 + n1 + d3 drawn uniformly over teams; X26c as the matched Stage-A control (valid from its own start, one seed, re-played); a 1 v 1 screen, a 2 v 2 read with a 1 pp seed allowance, a 3 v 3 Stage C on a near miss; F against Kakuna at one engine and the post-H22 regime; home72 as goal (a)'s floor and S as a registered price; gated on H-CTX, N1-P look 1 and D3 look 2, not on PR-1 | a P\*-start arm (X26c is no control for it at any dose; a new control per seed); X26's 280→380M segment as the control (cross-pin, not a fork); the forum teams in the union (F would lose its clean status); a multi-corpus `--team-mix` or component weights (a build that buys nothing here); a 1 v 1 verdict; a harm bar at 0 on home72 | the survey (`5d554c23`) §0, §6.2, §7; X26c README and dose records; the strategy doc (`7574743d`) §2, §3; X62 §1.2, §2; X53 §2; the team curriculum doc; `reads.md` §4; `power_x63.log` |
| 2026-10-27 | **X62's revision propagated** (X62 revision worker, orchestrator brief; docs only) | the recombined third inherits X62's revised builder: rc72b refused at radius 3, the STRICT Hidden Power rule (86 of 658 pool sets and 6 species withdrawn), near-duplicates reported; §9's X62 bullet follows X62's new tree (a PR-1′ KILL retires X62's 20M fork, not the corpus); FINDINGS 2 and 8 updated. Nothing else in this doc changed | — | [`design_x62_team_recombination.md`](design_x62_team_recombination.md) §2.2, §6, §10 |
