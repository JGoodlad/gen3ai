# X53-N1 — breadth at 10⁴ teams from step 0: a fresh production pair

**Status: REVISED DRAFT (2026-10-26), explicit-only. Nothing is built, registered or launched.** Revised after the
adversarial review [`strategy_review_2026-10-26/REVIEW.md`](../research_state/measurements/strategy_review_2026-10-26/REVIEW.md)
(`6fca2c06`: NOT READY to register, the direction sound; M1–M5, M9 and m1–m8 on this doc; §14 disposes of each). It
is the follow-up that N1-P's GO licenses
([`x61p_n1_production_2026-10-26/PREREG.md`](../research_state/measurements/x61p_n1_production_2026-10-26/PREREG.md)
§6: ADOPT-CANDIDATE or TRADE ⇒ "register X53-N1"). N1-P's look-1 seeds have trained (s3501–s3503,
`x35_breakers_stage_2026-10-17/dose/`); its read is in flight. **If N1-P does not read GO, this doc is not run as
written**: [`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md) (rank 11) says when a
from-scratch breadth pair still earns its GPU. The parent plan is
[`design_beat_metamon_path.md`](design_beat_metamon_path.md) §6.5. Its registration (§10's skeleton, made whole) is a
separate unit, written after N1-P's read and **after Stage 0 (§8.1)**, before any launch. The revision's Monte Carlo
is [`power_x53_revised.py`](../research_state/measurements/strategy_revision_2026-10-26/power_x53_revised.py) → `.log`.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170 (X26's plateau); C0 = X35's plain-continuation control; F = `forum72`; H = `heldout72_hpfix`; P72 = the
procedural panel; U = H, F and P72 at equal weight (goal (b)'s meter); S = in-house strength against the frozen
reference population R on `home72`; σ = the true between-run SD of a FRESH run's F score; GPU-h = hours of our local
RTX 3080 Ti; **lane-h = one wall-hour of the Kakuna anchor lane at `--concurrency 1` on rule 5b's agent cores, at the
MEASURED 160–230 games an hour** (D3 look 1; N1-P look 1's projection, `designs/tools/reads.md` §3); **GUESS** = an
unmeasured effect size; **ESTIMATE** = a cost from measured rates.

---

## 0. One screen

- **The question.** Does a production network trained FROM STEP 0 with 12,000 procedural Smogon-usage teams in its
  draw pilot unseen human-built teams better against Kakuna than the same network trained on the pool alone?
  N1-P asks this at the plateau, where plasticity confounds a NULL; this pair removes that confound.
- **The arms.** Two fresh `--arch production` runs at ONE pin: **N1F** (`--team-mix n1_proc_v1@ALL:0.5
  --team-mix-sides both`, N1-P's share) and **CF** (no lever). X26's checkpoints at the same steps are a DESCRIPTIVE
  third line, never the control.
- **Stage 0 decides whether X53 is affordable at all (zero GPU).** The design's power hangs on σ, which nobody has
  measured. Stage 0 reads F against Kakuna on the eight fresh production seeds `rb_es_prod_s2001`–`s2008` at 15M
  (1,200 games each, ≈ 42–60 lane-h): **GO iff the proxy's σ̂ ≤ 1.5 pp and its 80 % upper bound ≤ 2.5 pp.** Above
  that, two or four fresh seeds per arm cannot resolve a realistic effect, and the question goes to the owner as the
  scale bet's.
- **The power at REALISTIC effects** (the revised chain, σ at the proxy's value, one Stage-B pair): a true **+2 pp**
  reads GAIN **0.21 / 0.15** (σ 1 / 1.5 pp), **+3 pp 0.43 / 0.30**, +4 pp 0.67 / 0.49. With three Stage-B pairs, +3 pp
  reads 0.53 / 0.39 for ≈ 116–120 expected GPU-h. False GAIN 0.015–0.019. F is P\*'s best unfamiliar panel (0.530,
  premium 6.8 pp), so +4 pp would be 60 % of F's whole premium: the realistic range is +1.5 to +3.
- **The cost.** Stage A (one seed per arm to 100M): **≈ 39 GPU-h**, at the MEASURED fork rate, re-measured at the pin
  before it is quoted (§6). Expected A + B under a true null ≈ 46–52 GPU-h (the L50 futility stop is gone, §8.2).
  Stage C (N1F to its plateau) ≈ +36, and CF to its plateau ≈ +36 by default for any adoption question. CPU at the
  measured rate: Stage 0 ≈ 42–60 lane-h, Stage A ≈ 69–99.
- **What it can say.** GAIN, or RETIRE (§8.3: a SCREEN-STOP or an unpromising Stage B retires "breadth from step 0
  with `n1_proc_v1` at 0.5" for goal (b) in this era, with the chain's achieved power printed). It cannot read NULL.
- **The human-team corpus** (`n1_hrep_v1`, PROPOSED) is a separately toggled third arm **N1H** sharing CF, local
  only, so its presence decides where the pair runs (§9). Where X53 runs therefore waits on X59-S's read.

---

## 1. What it tests, and why from step 0

**The mechanism.** The deficit behind goal (b) is PILOTING an unfamiliar team (X37a PILOT_c +0.123; facing costs
≈ 0). X26 plays a team well in proportion to how often it has piloted it (the familiarity premium, ≈ 11 pp raw;
`design_beat_metamon_path.md` §4). The field's answer is breadth: every system that generalises across teams trained
on 10⁴–10⁵ of them (`sota_survey_2026-10-26.md` §5 N1; Procgen's held-out gap closes at ≈ 10⁴ levels).

**Why a plateau fork is not the whole test.** Every breadth test so far added teams to a network whose
representation formed on 719 of them: D1 (1,981 forum teams, H +2.10 [−0.92, +5.11]), D3 (2,876 mutated sets, look 1
+1.27 [−2.63, +5.16]), X52 (36 practised teams), and N1-P (12,000). Procgen trains from scratch on N levels. A
from-scratch run asks whether breadth shapes the representation differently, which no plateau fork can.

**What it does NOT test.** X53's Δ (fresh, 100M, ≈ 120 pilots per corpus team, half the pool dilution over the whole
history) and N1-P's Δ (P\* + 20M, ≈ 24 pilots, ≈ 4 % dilution, another pin) differ in dose, stage, pin and base
level. Comparing them is DESCRIPTIVE only; "larger from step 0" is not a hypothesis this design tests.

**The prior.** Expected Δ_F at 100M given N1-P GO: **+2 pp, 80 % [−1, +5]** (GUESS; lowered from +3 because F's
whole premium is 6.8 pp, review M2).

---

## 2. The arms

| arm | run name (proposed) | the lever | role |
|---|---|---|---|
| **N1F** | `rb_x53_n1f_s5301` (s5302… at Stage B) | `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` | treatment |
| **CF** | `rb_x53_cf_s5301` (s5302… at Stage B) | none | the matched control; also the shared control of any later fresh arm at this pin (N1H, a wide arm) |
| X26 trajectory | `rb_x26_s2601` @25M / 50M / 100M (every 1M checkpoint exists) | — | DESCRIPTIVE only: cross-pin, one seed |
| **N1H** (conditional) | `rb_x53_n1h_s5301` | `--team-mix n1_hrep_v1@…:0.5 --team-mix-sides both` | the human-team arm, separately toggled (§3.2); shares CF |

Both arms take `--arch production` (architecture and recipe) and exactly the non-recipe tokens X26's
`original_command` carried (read through `main.lineage`), plus their seed and run name. The registration prints the
resolved-config diff of each arm against X26's record (`python -m main.checkargs`); the only expected difference
between N1F and CF is the team mix.

**Why CF, and not X26's trajectory as the control.** It would save ≈ 20 GPU-h, and it would make the read
uninterpretable:
- X26 trained at pin `9c033489`. From there to the X35 production pin `f4b25238` alone, 163 commits touch `src/`, and
  HEAD adds the 10-26 GIGO fixes (an observation, an op-input and a label change; `MODEL_CONFIG_VERSION` 162) and the
  Rust core's full-parity batches, which are OU-visible.
- Each could move F by a few pp; the lever is expected to move it by a few pp. So N1F − X26 mixes the lever with ten
  days of code and with seed. CF − X26 @100M stays as a free, crude cross-pin look at a fresh replicate's drift.

**Why one seed per arm first.** A two-seed pair to 100M is ≈ 79 GPU-h before anything is known. Stage A spends half,
and Stage B adds pairs only when the first one screens positive.

**Multiplicity with N1H.** N1F − CF and N1H − CF are two comparisons against one control, each read at its own
registered α (two-sided 0.05); the registration states that no family-wise correction is applied and why (two
distinct corpora, two distinct decisions).

---

## 3. The corpus and the panel

### 3.1 `n1_proc_v1`, at N1-P's share

- **What it is** (X61's PREREG §1 has the full card): 12,000 whole teams from the Smogon facade alone, the pinned
  validator, level 100, no team within 2 mon sets of any held-out panel team; the label-path soak PASS (120,000
  battles). It is on main since `de269480`.
- **The share is 0.5, on both sides**, because that is the share N1-P's GO measured. Another share is a second lever.
- **The dose** (ESTIMATE, D1's rule of ≈ 570k games per 20M): at 100M each corpus team is piloted ≈ 120 times per
  side, ≈ 330 by 280M. Each pool team gets ≈ 2,000 pilots per side by 100M (CF: ≈ 4,000).
- **The M4 hazard applies unchanged** (N1-P PREREG §8): 62 % of H's full sets and 56.5 % of P72's are in the corpus
  (F: 19 %). Hence F is the primary, H and P72 are REPORTED with their set-coverage line, and a goal (b) claim by an
  N1-trained model leans on F and on the sealed `forum216b` (X54).
- **The known weakness:** procedural sets are incoherent by construction (P72 rates −146 Elo, TSR). Δ_H − Δ_F
  (reported) separates "learned the generator" from "learned to pilot".

### 3.2 The primary panel: F, widened if the source allows

- **F has the least headroom** (P\* 0.530 on F, premium 6.8 pp; H 10.1, P72 16.5), so a GAIN on F says little about
  goal (b) except through `forum216b`. **A larger-deficit unhazarded panel does not exist in our sources** (MY
  JUDGEMENT): humans field usage sets, so every human panel's sets are covered by `n1_proc_v1`'s usage draws (H 62 %).
  The fix is to quote power at realistic F effects (§8), not to change the panel.
- **F+ (team count, review m7).** If the forum v2 manifest's remainder outside `forum216b` yields ≥ 72 teams at
  ≤ 19 % corpus set coverage, the primary is **F+ = F ∪ that carve** at equal weight, registered before any X53 game;
  otherwise F alone, and the registration says so (UNVERIFIED: the remainder's size). Doubling the teams halves the
  team-sampling variance τ² / 72 (≈ 0.6–1.2 pp SE for τ = 5–10 pp, GUESS). A team-cluster bootstrap is REPORTED
  beside every rule.

### 3.3 The human-team corpus, as its own toggled arm

`n1_hrep_v1` is PROPOSED in `human_data_architecture.md` §9 item 6 and specified in X61's PREREG §10: full-six
ladder sides at actor rating ≥ 1500 from ONE registered snapshot, the panel exclusion BEFORE the Smogon fill, Hidden
Power repaired, the pinned validator and the soak, `"human_data": true` in its manifest. It is HUMAN DATA: its own
corpus, its own `--team-mix` toggle, never merged into `n1_proc_v1`. REVIEW F's F-M7 (one spec, a throwing guard on
`--team-mix` for a `human_data` corpus) applies before it is built.

- **When it runs.** Only if X59-S does not close human data (X59-S PREREG §5).
- **Its read** is N1H − CF on F, the same rule.
- **Its size is UNVERIFIED.** Below ≈ 10⁴ distinct full-six teams after the exclusion it is not an N1-scale arm.
- **Its cost:** the corpus build (≈ 1 agent-day) and one more fresh arm (≈ 20 GPU-h to 100M), sharing CF.
- **It runs LOCALLY only** (the cloud plan §4), so a planned N1H keeps CF local, and with it N1F (§9).

---

## 4. Fork or fresh

| option | GPU-h to a decisive look | what it answers | verdict |
|---|---|---|---|
| **Fresh pair, N1F vs CF** (this doc) | ≈ 39 (one seed each to 100M) | does breadth from step 0 shape a network that pilots unseen teams better? No plasticity confound | **CHOSEN**, behind Stage 0 |
| Extend N1-P's seeds and C0's (+40M each) | ≈ 16 | dose: does more breadth exposure keep raising a plateaued network's F? | not X53's follow-up; it is the strategy's rank 6 (the dose-vs-plasticity read) |
| Fork both arms from an early X26 checkpoint (25M) at today's pin | ≈ 30 | breadth from 25M | rejected: a third code version inside both arms; saves only ≈ 5 GPU-h per arm |
| Fresh toy pair (Option T, `main.x61.arms`) | ≈ 13 toy GPU-h | the same question on the 1.42M toy | rejected for a GO; listed in the strategy as the cheap screen if N1-P is UNDECIDED |
| N5-S: a fresh offline student on a 10⁴-team corpus vs one on 719 teams | ≈ 8 | breadth from step 0 at student scale | the strategy's rank 7; the cheap fresh-breadth screen when N1-P has no GO |
| N1F vs X26's trajectory only | ≈ 20 | — | rejected (§2) |

---

## 5. The pin

- **Chosen: main at registration, tagged** (an annotated tag on origin), for BOTH arms and every stage. It carries
  `n1_proc_v1`, the AM-D1-FIX label guard, the pinned-data root, and the 10-26 GIGO fixes.
- **Rejected: N1-P's pin `5f8f6cfa`.** X53's control is CF, so that pin's reason does not carry, and a ≈ 55 GPU-h
  lineage would bake in known observation and label defects.
- **The cost of choosing main:** the Rust full-parity programme is mid-flight (364 of 369 moves; the B6–B10 review
  found 5 OU-reachable divergences). Both arms share the pin's engine, so the contrast is clean; the registration
  records the programme's state.
- **The reads' engine.** Every X53 Kakuna game, both arms and every look, plays at ONE eval harness commit held in a
  pinned worktree, with `server_version` and the sim binary's sha256 recorded per unit (REVIEW F's F-M2 pattern; the
  engine-mix audit had not landed). X26's descriptive cells may live in their own `main.reads` state.

---

## 6. Steps, GPU-hours, seeds, mechanics

**The rate.** At the X35 production pin, 20M forks ran at 1,384–1,433 steps/s with or without the corpus (MEASURED,
`dose/`). The Rust parity batches and the 10-26 changes lie between that pin and HEAD, and a fresh run's early rate is
unmeasured: **the registration measures 1M steps of a fresh run at the pin (≈ 12 min of GPU) and quotes the stages at
that rate**; the table assumes 1,410.

| milestone | steps | GPU-h per arm (ESTIMATE) | cumulative, both arms |
|---|---:|---:|---:|
| L25 | 25M | 4.9 | 9.9 |
| L50 | 50M | 9.9 | 19.7 |
| **L100 (Stage A screen)** | 100M | 19.7 | **39.4** |
| Stage B: each further pair to 100M | 100M | 19.7 | +39.4 a pair |
| Stage C: N1F s5301 100M → its plateau (250M floor, ≤ 300M) | ≈ 280M | +35.5 | +35.5 |
| Stage C: CF s5301 to the plateau (default for any adoption question) | ≈ 280M | +35.5 | +35.5 |

- **Seeds:** 5301 for Stage A, 5302 (and 5303–5304 if m_B = 3) for Stage B; both arms share each seed.
- **No lockstep pause.** Both arms train straight to 100M (sequentially or in alternating restart segments on one
  card, the training session's call; `--restart-interval-hours` restarts by wall clock, so no argv changes mid-run).
  Each look reads the 1M checkpoints as an arm passes them. The only stop (L25's catastrophic harm bar) acts when both
  arms have passed 25M and the L25 read is in; GPU-h trained past 25M meanwhile (≤ ≈ 5 GPU-h) is its price.
- **Wall time on our card:** Stage A ≈ 1.7 days of GPU; the L100 read adds ≈ 2–4 days of the anchor lane.

---

## 7. The meters along the way

Every Kakuna cell is greedy against greedy, ckpt 34, CPU, role-balanced 20-game units, through `main.reads`, at the
one eval engine (§5).

| look | primary (decides) | reported, never decides | games / lane-h (measured rate) |
|---|---|---|---:|
| **Stage 0** | σ̂ from F on `rb_es_prod_s2001`–`s2008` @15M (1,200 each) | per-seed F; the pin and era of each run | 9,600 / 42–60 |
| launch | — | VOID guards: the TEAM-MIX banner (N1F), 0 held-out hits, 0 pool copies, the recorded pin; first-hour steps/s ≥ 1,000 (else a HOLD); compile canary clean | 0 |
| **L25** | the catastrophic harm stop on Δ_F (600 per arm) | X26 @25M on F (600); the in-run eval | 1,800 / 8–11 |
| **L50** | — (no stop) | Δ_F (1,200 per arm); X26 @50M (1,200); S per arm (3,500 pairs, the S lane) | 3,600 / 16–23 |
| **L100** | **the screen: Δ_F (or Δ_F+) at 2,400 per arm** | H, P72, home72 (420 each per arm) with the coverage line; Δ_H − Δ_F; the REVERSE cell (400 per arm); S per arm; X26 @100M (2,400); CF − X26; **the trajectory: each arm's F at 25M / 50M / 100M** | 10,520 / 46–66 |
| Stage B L100, each pair | the pooled Δ_F over every pair, and P | as above; **the Stage-B-only Δ̂** (§8.2) | ≈ 8,100 a pair / 35–51 |
| Stage C | the plateau stop: `python -m main.plateau tick` on N1F every 10M from 250M, **with an F read at each tick** (1,200 games) | F at 150M and 200M (1,200 each): the trajectory | ≈ 5–8 a tick |
| **N1F's plateau** | the goal reads: X54 on `forum216b` (1,600–2,400 games) + home72 (400, goal (a)'s guard) | U with its coverage line; N1F vs P\* on F and home72 (descriptive, cross-pin); CF's plateau F (same-pin attribution) | ≈ 13–18 |

- **The trajectory, not one point, says "pace" or "level".** X26's F rose +6.0 pp from 161M to 280M (X51; one seed,
  n.d.), so 100M is mid-climb on F. A breadth arm that climbs slower but plateaus higher (or faster to the same place)
  shows as crossing curves; the record reports both arms' F against steps and calls the 100M Δ a mix of pace and
  level.
- **The in-house meter steers nothing about goal (b)** (X37b); the Stage C tick reads F beside it, and its 250M floor
  keeps the tick from stopping N1F on the lever's blind meter (X26's F rose after its in-house curve flattened).

---

## 8. The decision points

Rule 8 applies throughout: a bound or point estimate within 1e-9 of a threshold takes the undecided / continue
branch.

### 8.1 Stage 0 — the σ gate (zero GPU)

- **The proxy.** `rb_es_prod_s2001`–`s2008` are eight FRESH `--arch production` seeds to 15M (pin `95d014fa`). F
  against Kakuna at 1,200 games each. σ̂² = the sample variance of the eight F scores minus their mean binomial
  variance (df 7); its 80 % upper bound from χ²₇. **Its direction to 100M is unknown** (runs may diverge further or
  converge); it is the only fresh-run spread we can measure without GPU. The CF s5301 − X26 @100M line (cross-pin) is
  a second, cruder look once Stage A has run.
- **GO iff σ̂ ≤ 1.5 pp AND its 80 % upper bound ≤ 2.5 pp.** Then the registration fixes **the seed floor at max(1.5
  pp, σ̂)** and **m_B = 1 if σ̂ ≤ 1.0 pp, else 3**, and prints the chain's false-GAIN rate at σ = the upper bound.
- **Otherwise X53 is not registered as designed.** At σ ≥ 2 pp even three Stage-B pairs read a true +3 pp as GAIN only
  0.28 (≈ 113 GPU-h expected); the fresh-breadth question goes to the owner with the scale bet (the strategy's §5).
- **Playability is UNVERIFIED** (the review's FINDING 4): the eight runs' configs must play at the eval engine's
  commit; a run that does not is dropped (≥ 6 needed), never patched.

### 8.2 Stages A–C

| stage, look | rule | consequence |
|---|---|---|
| any | a VOID guard fails (§7 launch row; K9(b) at each restart; AM-6's ceilings; two engine stamps in one read) | that run or read is VOID: re-run at the same pin and seed after the cause is fixed; nothing is read |
| A, L25 | **HARM-STOP** iff the Newcombe UB of Δ_F (600 per arm) < −5.0 pp | stop both arms: a catastrophic early cost. FINAL for the stage |
| A, L50 | no stop; reported | — |
| A, L100 | **SCREEN-GO** iff Δ̂_F ≥ +2.0 pp (2,400 per arm) | Stage B. Otherwise **SCREEN-STOP ⇒ RETIRE** (§8.3) |
| B, L100 | over all 1 + m_B pairs: Δ̂ pooled, normal interval with per-run seed SD max(the measured within-arm spread, the floor); the sign rule (every pair's Δ̂ ≥ 0). **GAIN** iff LB > 0 and the sign rule; **LOSS** iff UB < 0; **KILL** iff not GAIN and UB < +3 pp; otherwise UNDECIDED. Co-primary P: Welch on S, **NON-INFERIOR** iff LB > −2.0 pp | GAIN + P NON-INFERIOR = ADOPT-CANDIDATE; GAIN without it = TRADE. Either ⇒ Stage C. KILL, LOSS, or UNDECIDED with the Stage-B-only Δ̂ < +2 pp ⇒ RETIRE (§8.3). UNDECIDED otherwise ⇒ the orchestrator rules, with the RETIRE record's numbers in hand |
| C, plateau | X54 on `forum216b`, home72 (goal (a) must stay met: LB > 0.50), the coverage line, CF's plateau F | goal (b) is claimed only on X54's registered terms; adoption of `n1_proc_v1` into production needs CF's same-pin plateau |

**The estimate handed on (review M4).** The GAIN test pools every pair (its error rate is simulated with the screen
inside it). The **effect estimate** reported, and the numerator of TRADE's exchange rate, is the **Stage-B pairs
alone** (not screen-selected: its bias given GO is 0.00 ± 0.04 pp in the Monte Carlo), with its own interval; the
pooled Δ̂ is printed beside it labelled "screen-selected". Conditional on GAIN any estimate still carries the
significance filter's inflation (Stage-B-only +1.0 to +1.9 pp at a true +3 pp, σ 1–1.5, m_B 1); the record says so.

### 8.3 RETIRE (review M5)

On SCREEN-STOP, KILL, LOSS, or a Stage-B UNDECIDED with the Stage-B-only Δ̂ < +2 pp: **"breadth from step 0 with
`n1_proc_v1` at 0.5, F primary" is RETIRED for goal (b) in this era.** The record prints the chain's power at +2, +3
and +4 pp at Stage 0's σ̂, so the reader sees what was and was not excluded. It is re-opened only by a new lever (a
new corpus, such as `n1_hrep_v1`, or another share) or by evidence named in advance (the strategy's PR-1 reading
BREADTH-GAIN after the RETIRE). The strategy's rank 11 cannot buy it again.

### 8.4 Power

[`power_x53_revised.py`](../research_state/measurements/strategy_revision_2026-10-26/power_x53_revised.py), 100,000
draws a cell; F's base rising 0.30 / 0.40 / 0.45 at 25M / 50M / 100M; a run-level seed effect of SD σ shared by a
run's looks; the floor = σ (Stage 0's value); GAIN is the whole chain's final GAIN.

| true Δ_F at 100M | σ = 1 pp, m_B = 1 | σ = 1.5, m_B = 1 | σ = 1.5, m_B = 3 | σ = 2, m_B = 3 | σ = 3, m_B = 3 |
|---|---|---|---|---|---|
| 0 (false GAIN) | 0.015 | 0.017 | 0.011 | 0.011 | 0.012 |
| +2 | 0.21 | 0.15 | 0.18 | 0.13 | 0.08 |
| **+3** | **0.43** | **0.30** | **0.39** | 0.28 | 0.15 |
| +4 | 0.67 | 0.49 | 0.62 | 0.47 | 0.27 |
| E[GPU-h], null / +3 | 46 / 66 | 48 / 65 | 65 / 116 | 70 / 113 | 77 / 108 |

- **The draft's chain, recomputed (review M1),** flat base 0.49, false GAIN by the true σ: with its 1.5 pp floor,
  0.005 / 0.034 / 0.060 at σ = 1 / 2 / 3 pp (the draft said ≤ 0.02); with a fixed 3 pp floor, 0.000 / 0.003 / 0.018,
  but then a true +3 pp reads GAIN only 0.01–0.12. A fixed 3 pp floor makes X53 blind; a measured σ is the fix.
- **The early stops (review M3).** Under the draft's stops, a true +4 pp that arrives late is stopped early 0.22 of
  the time (dip course 0.24) and reads GAIN 0.47–0.48. Under the revised stops it is stopped ≤ 0.003 and reads GAIN
  0.56 whatever the time course. The price is ≈ +7 GPU-h expected under a null (46–52 against 39–45).
- **NULL stays unreachable:** the seed term alone keeps the pooled half-width above D1's ±3 pp band.
- **KILL is rarely reachable** (≤ 0.08 under a null at m_B = 3, ≈ 0 at m_B = 1): RETIRE rests mostly on SCREEN-STOP
  and the Stage-B-only Δ̂.
- The registration replaces this with a Monte Carlo of the registered code (`power.py`, as N1-P's), at Stage 0's σ̂.

---

## 9. The second GPU: would the cloud plan run it?

**Not as approved; it should be the first workload a widened approval covers**, and where it runs waits on X59-S's
read (N1H). Against [`../ops/cloud_gpu_plan.md`](../ops/cloud_gpu_plan.md):

- **The card rule (§1.2) costs X53-N1 nothing extra.** Its control CF runs fresh beside N1F on the same card type;
  only X26's descriptive line crosses cards.
- **Nothing from `models/` goes up.** A fresh run has no parent checkpoint (UNVERIFIED: that it reads no other file
  under `models/` at startup; the box dry-run answers it).
- **The plan's shape does not fit.** It runs one box per 20M fork (≈ 2.5 box-h) with a 240-minute deadman, and refuses
  a second instance. A fresh arm is ≈ 12 box-h to 100M on a 5090 (at 1.65×). It needs a longer deadman or segment
  boxes (pull the checkpoint, resume on a fresh box: a restart, which the arms take anyway).
- **Two identical cards at once** halve the wall time; that needs the one-instance rule lifted (an owner question).
- **The cost** at $0.50–0.70/h: Stage A ≈ 24 box-h ≈ **$13–18**; each Stage B pair the same; Stage C ≈ $12–16 per
  arm. **≈ $50–85 if every stage runs with CF to its plateau**, above the $20 the current approval asks for.
- **Where it runs:** with N1H planned, LOCAL (human-derived teams never reach a box); without N1H and with a widened
  approval, the CLOUD; otherwise local. A bought 5090 replaces the rental.

---

## 10. The pre-registration skeleton

To be written whole after N1-P's read AND Stage 0, before any launch, as `measurements/x53_n1_fresh_<date>/PREREG.md`:

1. **Question and mechanism** (§1), N1-P's verdict that licensed it, and **Stage 0's σ̂, its bound and the gate's
   outcome**, cited by their records.
2. **The arms' argv** from a fresh-pair kit (on `main.x61.arms`'s pattern), `checkargs` exit 0 at the pin,
   `--dry-run` transcripts, each arm's resolved-config diff against X26's record, the pin's tag and `git describe`,
   the full-parity programme's state, and the measured 1M-step rate.
3. **The corpus card** by reference (X61 §1), its sha256s, the banner line, the set-coverage line of every panel by
   `pool ∪ n1_proc_v1`, and F+'s carve if it exists (§3.2).
4. **The pre-launch checks:** the `--debug` smoke with the lever at the pin, the parity gate, the tag on origin.
5. **The looks and rules** of §7–§8 verbatim: the 1e-9 rule, the floor and m_B from Stage 0, the sign rule, the
   floor on counted games per look, the one-engine rule, the `main.reads` specs (`check-spec` exit 0).
6. **The REPORTED lines** and why each decides nothing.
7. **The VOID guards**, which are module-checked and which are the operator's.
8. **Power:** the Monte Carlo of the registered code at Stage 0's σ̂ and at its bound.
9. **Predictions** (GUESS), per look and per class.
10. **Cost** per stage in GPU-h, lane-h and, on the cloud, dollars; where it runs (§9); **the licence widening**:
    N1-P §6 licenses "a FRESH run … against X26 @100M (≈ 21 GPU-h)", this spends ≈ 46–120 expected, so the
    orchestrator ratifies it in writing.
11. **What it cannot say:** one share, one corpus, one pin; NULL unreachable; goal (b) only through X54; what RETIRE
    does and does not exclude.
12. **FINDINGS at registration.**

---

## 11. Cost

| item | GPU-h | lane-h (measured rate) | agent-days |
|---|---:|---:|---:|
| Stage 0 (σ proxy) | 0 | ≈ 42–60 | ≈ 0.3 |
| the fresh-pair kit, the specs, the registration, the 1M-step rate | ≈ 0.2 | ≈ 1 (checks) | ≈ 1 |
| Stage A (L25, L50, L100) | ≈ 39 | ≈ 69–99 | ≈ 0.3 (operation) |
| Stage B, per pair (on SCREEN-GO; m_B = 1 or 3) | ≈ 39 | ≈ 35–51 | ≈ 0.2 |
| Stage C, N1F to its plateau (on GAIN or TRADE) | ≈ 36 (cap) | ≈ 45–70 (ticks with F, X54) | ≈ 0.3 |
| Stage C, CF to its plateau (default for adoption) | ≈ 36 | ≈ 15–25 | ≈ 0.1 |
| N1H (conditional): corpus build + one arm to 100M | ≈ 20 | ≈ 25–40 | ≈ 1.5 |
| **every stage, m_B = 1, CF to its plateau, no N1H** | **≈ 150** | **≈ 205–305** | **≈ 2.2** |

Expected GPU-h under a true null ≈ 46–52 (m_B = 1) or 65–77 (m_B = 3). The CPU reads are the binding lane: Stage 0
plus Stage A is ≈ 110–160 lane-h, ≈ 5–7 days at `--concurrency 1` while a run is live; the strategy's rented CPU box
(§3.1 there) would take both in ≈ 1 day.

---

## 12. FINDINGS (standing rule 7)

1. **A fresh production run's seed-to-seed spread on F is unmeasured**; Stage 0 measures a 15M proxy whose direction
   to 100M is unknown. Above σ ≈ 2 pp, no affordable seed count resolves a realistic effect (§8.4).
2. **The effect's time course is a GUESS**; the revised stops make the chain insensitive to it (§8.4), and the
   trajectory is reported.
3. **`--team-mix` takes ONE corpus** (`agents.training.matchup_spec.parse_team_mix`), so N1F and N1H are separate arms.
4. **The X35 kit is fork-shaped**; a fresh pair needs its own kit arms (≈ 0.5 agent-day inside §11's day).
5. **The fresh-run throughput, and whether a fresh run reads anything under `models/`, are UNVERIFIED** (§6, §9).
6. **Stage 0's runs (`rb_es_prod_s2001`–`s2008`, pin `95d014fa`) are UNVERIFIED as playable** at a current
   `main.reads` commit, and their era is unread.
7. **F+'s source is UNVERIFIED**: whether the forum v2 manifest's remainder holds ≥ 72 teams at ≤ 19 % coverage.
8. **The engine-mix audit had not landed**; §5 follows REVIEW F's F-M2 fix pattern.
9. **This unit ran no game and used no GPU.** The Monte Carlo ran on the agent cores at nice 19 in ≈ 8 s.

---

## 13. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-26 | **This doc CREATED (DRAFT)** (research-strategy worker, orchestrator brief) | X53-N1 as a fresh production PAIR at one pin, N1F (`n1_proc_v1` at 0.5, both sides) vs CF, in lockstep; X26's trajectory descriptive; F primary (M4); stages A (L25 harm stop, L50 futility stop, L100 screen at Δ̂_F ≥ +2.0), B (a second pair, D1's classes pooled, seed floor 1.5 pp) and C (N1F to its plateau, X54's goal read); N1H as a separately toggled, local-only third arm; main at registration as the pin; the cloud as the preferred home once an approval covers it | N1F vs X26's trajectory; extending N1-P's forks; an early-X26 fork; the toy; N1-P's pin; two seeds per arm from the start; a share other than 0.5 | N1-P PREREG §6, §8; `dose/`; `sota_survey_2026-10-26.md` §5; `cloud_gpu_plan.md` §1.2–§4; the §8 Monte Carlo |
| 2026-10-26 | **REVISED DRAFT after the review** (research-strategy worker, orchestrator brief; docs + a CPU Monte Carlo) | Stage 0, a zero-GPU σ gate on `rb_es_prod_s2001`–`s2008` (GO iff σ̂ ≤ 1.5 and its 80 % bound ≤ 2.5; floor max(1.5, σ̂); m_B 1 or 3 by σ̂); power quoted at +2 / +3 pp; L25 harm only at a catastrophic −5 pp bar, L50 report-only, the trajectory read; the Stage-B-only estimate handed on; a RETIRE outcome and a KILL class; CF to its plateau by default for adoption; the Stage C tick with an F read and a 250M floor; F+ if the source allows; one eval engine; no lockstep pause | a fixed 3 pp floor (it makes X53 blind: +3 pp reads GAIN ≤ 0.12); keeping the L50 futility stop (it kills a late +4 pp 0.22 of the time); a larger-deficit unhazarded panel (none exists in our sources); the comparative hypothesis (descriptive only) | [`REVIEW.md`](../research_state/measurements/strategy_review_2026-10-26/REVIEW.md) (`6fca2c06`); [`power_x53_revised.py`](../research_state/measurements/strategy_revision_2026-10-26/power_x53_revised.py) → `.log`; D3 look-1 README FINDING 5 and `reads.md` §3 (the lane rate) |

---

## 14. RESPONSE to the review (`6fca2c06`)

| finding | disposition | what changed, or why not |
|---|---|---|
| **M1**: the false-GAIN rate rests on an unmeasured σ; the floor is half the X61 review's | **FIXED, the floor in modified form** | false GAIN re-derived at σ 2–3 (§8.4: 0.034 / 0.060 under the draft's floor); σ measured at zero GPU first (Stage 0, §8.1); the floor is max(1.5 pp, σ̂) and m_B is set by σ̂, with the false-GAIN rate printed at σ̂'s bound. A FIXED 3 pp floor is rejected because it makes the design blind (+3 pp reads GAIN ≤ 0.12); where σ̂ cannot be measured, X53 is not registered rather than run blind |
| **M2**: F has the least headroom; power quoted at implausible effects | **FIXED in part** | power quoted at +2 / +3 pp (§0, §8.4); the prior lowered to +2 [−1, +5]; F+ widens the team count if the source allows. The larger-deficit unhazarded panel is **rejected**: none exists in our sources (§3.2) |
| **M3**: 100M is mid-climb; the early stops assume the wrong shape | **FIXED** | L50's futility stop removed (report-only); L25's harm stop only at −5 pp; the trajectory at 25M / 50M / 100M (and 150M / 200M in Stage C) reported as pace-vs-level; CF to its plateau by default for adoption |
| **M4**: the screen inflates the estimate | **FIXED** | the effect handed on (and TRADE's numerator) is the Stage-B pairs alone; the pooled Δ̂ is labelled screen-selected; the residual significance-filter inflation is stated |
| **M5**: SCREEN-STOP licenses nothing | **FIXED** | the RETIRE outcome (§8.3) with its power printed, and the KILL class (UB < +3 pp); the strategy's rank 11 cannot re-buy a RETIRED bet |
| **M9**: the fork-extension branch has no home | **FIXED** | the strategy's rank 6 (dose vs plasticity); Option T listed there; §4 points to both |
| **m1**: the lockstep has no mechanism | **FIXED** | no pause: both arms train straight through and the looks read the 1M checkpoints (§6) |
| **m2**: the throughput is unverified at this pin | **FIXED** | 1M steps measured at the pin at registration; costs quoted at that rate |
| **m3**: "the early stops end most null runs" is wrong as worded | **FIXED** | the sentence is gone; expected null cost is stated (46–52 GPU-h) |
| **m4**: the comparative hypothesis is not what the design tests | **FIXED** | descriptive only (§1) |
| **m5**: Stage C's gate uses the cross-pin comparison §2 rejects | **FIXED** | CF to its plateau by default for any adoption question, priced (+36 GPU-h) |
| **m6**: the plateau tick reads the blind in-house meter | **FIXED** | an F read at each tick and a 250M floor |
| **m7**: team clustering ignored | **FIXED in part** | a team-cluster bootstrap reported beside every rule; F+ if the source allows; the rule itself stays binomial (inherited from D1) |
| **m8**: the licence is widened without saying so; N1H's multiplicity and timing | **FIXED** | the widening is stated for the orchestrator to ratify (§10 item 10); N1H's multiplicity stated (§2); where X53 runs waits on X59-S (§0, §9) |
