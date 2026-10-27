# X53-N1 — breadth at 10⁴ teams from step 0: a fresh production pair

**Status: DRAFT (2026-10-26), explicit-only. Nothing is built, registered or launched.** It is the follow-up that
N1-P's GO licenses ([`x61p_n1_production_2026-10-26/PREREG.md`](../research_state/measurements/x61p_n1_production_2026-10-26/PREREG.md)
§6: ADOPT-CANDIDATE or TRADE ⇒ "register X53-N1"). N1-P's look-1 seeds have trained (s3501–s3503, each ≈ 4.0–4.4
GPU-h, `x35_breakers_stage_2026-10-17/dose/`); its read has not run. **If N1-P does not read GO, this doc is not
run as written**: the companion [`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md)
says when a from-scratch breadth run still earns its GPU. The parent plan is
[`design_beat_metamon_path.md`](design_beat_metamon_path.md) §6.5. Its registration (§10's skeleton, made whole)
is a separate unit, written after N1-P's read and before any launch.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170 (X26's plateau); C0 = X35's plain-continuation control; F = `forum72`; H = `heldout72_hpfix`; P72 = the
procedural panel; U = H, F and P72 at equal weight (goal (b)'s meter); S = in-house strength against the frozen
reference population R on `home72`; GPU-h = hours of our local RTX 3080 Ti; lane-h = one CPU lane for an hour
(≈ 360 Kakuna games); **GUESS** = an unmeasured effect size; **ESTIMATE** = a cost from measured rates.

---

## 0. One screen

- **The question.** Does a production network trained FROM STEP 0 with 12,000 procedural Smogon-usage teams in its
  draw pilot unseen human-built teams better against Kakuna than the same network trained on the pool alone?
  N1-P asks this at the plateau, where plasticity confounds a NULL; this run removes that confound.
- **The arms.** Two fresh `--arch production` runs at ONE pin, in lockstep: **N1F** (the lever
  `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both`, N1-P's share) and **CF** (no lever: the fresh-run frame's
  C0). X26's own checkpoints at the same step counts are a **descriptive** third line, never the control: 160+ `src/`
  commits separate X26's pin from today's.
- **The cost.** Stage A (one seed per arm to 100M): **≈ 39 GPU-h** (MEASURED rate: the corpus costs no throughput at
  the production pin, 1,388–1,433 steps/s against C0's 1,384–1,424). Stage B (a second seed pair, only on a screen
  GO): +39. Stage C (N1F to its plateau, only on GAIN): ≈ +36. **≈ 114 GPU-h if every stage runs**; ≈ 39–44 expected
  under a true null, because the early stops end most null runs by 100M (§8).
- **The decision.** F is the primary (H and P72 are fed by the corpus's own generator, N1-P's M4 hazard). A
  one-seed screen at 100M, then a two-seed read by D1's classes. The design detects +4 pp with probability 0.39–0.64
  and +6 pp with 0.71–0.98, depending on the seed-to-seed spread of a FRESH run on F, which nobody has measured
  (FINDING 1). It cannot read NULL (§8).
- **The human-team corpus** (`n1_hrep_v1`, PROPOSED, not built) is a separate, toggled third arm **N1H** sharing CF,
  never mixed into `n1_proc_v1`; it runs locally only, so its presence decides where the whole pair runs (§9).
- **The second GPU.** X53-N1 is the cleanest cloud workload in the queue: it brings its own control, so the card rule
  costs it nothing, and a fresh run uploads no checkpoint. But the cloud plan's approval ($20, C0v and D3) does not
  cover it, and its one-box-per-seed, 240-minute deadman shape is sized for 20M forks (§9).

---

## 1. What it tests, and why from step 0

**The mechanism.** The deficit behind goal (b) is PILOTING an unfamiliar team (X37a PILOT_c +0.123; facing costs
≈ 0). X26 plays a team well in proportion to how often it has piloted it (the familiarity premium, ≈ 11 pp raw;
`design_beat_metamon_path.md` §4). The field's answer is breadth: every system that generalises across teams trained
on 10⁴–10⁵ of them (`sota_survey_2026-10-26.md` §5 N1; Procgen's held-out gap closes at ≈ 10⁴ levels).

**Why a plateau fork is not the whole test.** Every breadth test so far added teams to a network whose
representation formed on 719 of them: D1 (1,981 forum teams, H +2.10 [−0.92, +5.11]), D3 (2,876 mutated sets, look 1
+1.27 [−2.63, +5.16]), X52 (36 practised teams), and N1-P (12,000). Procgen trains from scratch on N levels. X37b
found that most of X26's in-house gain came by 100M: the representation is shaped early. A from-scratch run asks
whether breadth shapes it differently, which no plateau fork can.

**What a GO from N1-P means for the prior.** N1-P GO says breadth moves even a plateaued network's F piloting. The
hypothesis here is that from step 0 the effect is larger. That is a GUESS: VGC-Bench's team-count gains were from
scratch, but its budget is 1–2 % of ours. Expected Δ_F at 100M given N1-P GO: **+3 pp, 80 % [−1, +7]** (GUESS).

---

## 2. The arms

| arm | run name (proposed) | the lever | role |
|---|---|---|---|
| **N1F** | `rb_x53_n1f_s5301` (s5302 at Stage B) | `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` | treatment |
| **CF** | `rb_x53_cf_s5301` (s5302 at Stage B) | none | the matched control; also the shared control of any later fresh arm at this pin (N1H, a wide arm) |
| X26 trajectory | `rb_x26_s2601` @25M / 50M / 100M (every 1M checkpoint exists) | — | DESCRIPTIVE only: cross-pin, one seed |
| **N1H** (conditional) | `rb_x53_n1h_s5301` | `--team-mix n1_hrep_v1@…:0.5 --team-mix-sides both` | the human-team arm, separately toggled (§3.2); shares CF |

Both arms take `--arch production` (architecture and recipe) and exactly the non-recipe tokens X26's
`original_command` carried (read through `main.lineage`), plus their seed and run name. The registration prints the
resolved-config diff of each arm against X26's record (`python -m main.checkargs`); the only expected difference
between N1F and CF is the team mix.

**Why CF, and not X26's trajectory as the control.** It would save ≈ 20 GPU-h, and it would make the read
uninterpretable:
- X26 trained at pin `9c033489`. From there to the X35 production pin `f4b25238` alone, 163 commits touch `src/`
  (FP32_ALT's τ solve among them), and HEAD adds the 10-26 GIGO fixes (an observation change, an op-input change and a
  label change; `MODEL_CONFIG_VERSION` 162) and the Rust core's full-parity batches, which are OU-visible.
- Each of those could move F by a few pp. The lever is expected to move it by a few pp.
- So N1F − X26 mixes the lever with ten days of code, and with seed. It stays as a descriptive line, and
  CF − X26 becomes a free, crude look at how far a fresh replicate of production drifts (§7).

**Why one seed per arm first.** A two-seed pair to 100M is ≈ 79 GPU-h before anything is known. Stage A spends half
of that, and its early stops end most null runs sooner (§8). Stage B adds the second pair only when the first one
screens positive.

---

## 3. The corpus

### 3.1 `n1_proc_v1`, at N1-P's share

- **What it is** (X61's PREREG §1 has the full card): 12,000 whole teams from the Smogon facade alone, the pinned
  validator, level 100, no team within 2 mon sets of any held-out panel team; the label-path soak PASS (120,000
  battles). It is on main since `de269480`, so a pin taken from main carries it with no side commit.
- **The share is 0.5, on both sides**, because that is the share N1-P's GO measured. A different share (X61's toy
  used 0.7; the survey suggested the pool at a 0.3 minority) would be a second lever, and nothing has measured it.
- **The dose** (ESTIMATE, D1's rule of ≈ 570k games per 20M): at 100M each corpus team is piloted ≈ 120 times per
  side, ≈ 330 by 280M. Each pool team gets ≈ 2,000 pilots per side by 100M (CF: ≈ 4,000). By 280M that is ≈ 5,500
  against CF's ≈ 11,000. (§4 of the beat-Metamon doc's rule gives ≈ 9,300 for P\*; the two estimates order the levers
  and are not quoted as measurements.)
- **The M4 hazard applies unchanged** (N1-P PREREG §8): 62 % of H's full sets and 56.5 % of P72's are in the corpus
  (F: 19 %). Hence F is the primary, H and P72 are REPORTED with their set-coverage line, and a goal (b) claim by an
  N1-trained model leans on F and on the sealed `forum216b` (X54), whose owner is told before its unsealing is spent
  on this model.
- **The known weakness:** procedural sets are incoherent by construction (P72 rates −146 Elo, TSR). From step 0 the
  network spends half its games on them. Δ_H − Δ_F (reported) is the line that separates "learned the generator" from
  "learned to pilot".

### 3.2 The human-team corpus, as its own toggled arm

`n1_hrep_v1` is PROPOSED in `human_data_architecture.md` §9 item 6 and specified in X61's PREREG §10: full-six
ladder sides at actor rating ≥ 1500 from ONE registered snapshot, the panel exclusion at ≥ 3 move-revealed fits
BEFORE the Smogon fill, Hidden Power repaired, the pinned validator and the soak, `"human_data": true` in its
manifest. Under the owner's rule it is HUMAN DATA: its own corpus, its own `--team-mix` toggle, never merged into
`n1_proc_v1`.

- **When it runs.** Only if X59-S does not close human data (SELF-PLAY-DATA-BETTER, with Δ_labels not
  HUMAN-LABELS-BETTER and Δ_states' LB ≤ 0, closes it at this volume; X59-S PREREG §5). Δ_states' LB > 0 is the read
  that supports it most directly: human STATE coverage helps a student on unfamiliar teams.
- **Its read** is N1H − CF on F, the same rule. H is drawn from Metamon's `hl_05_26` ladder teams, so it is
  ladder-adjacent to this corpus too; F stays the primary.
- **Its size is UNVERIFIED.** The data card counts 1.80M informative rows and says 15.9 % of ≥ 1500 full-six sides
  lie in the panels' region; how many DISTINCT full-six teams survive the exclusion was not counted. Below ≈ 10⁴ it is
  not an N1-scale arm, and the registration says so.
- **Its cost:** the corpus build (≈ 1 agent-day, ESTIMATE) and one more fresh arm (≈ 20 GPU-h to 100M), sharing CF.
- **It runs LOCALLY only** (the cloud plan §4: no human-derived data reaches a rented box), so a planned N1H keeps CF
  local, and with it N1F (the card rule, §9).

---

## 4. Fork or fresh

| option | GPU-h to a decisive look | what it answers | verdict |
|---|---|---|---|
| **Fresh pair, N1F vs CF** (this doc) | ≈ 39 (one seed each to 100M) | does breadth from step 0 shape a network that pilots unseen teams better? No plasticity confound; it is Procgen's protocol | **CHOSEN** |
| Extend N1-P's seeds and C0's (e.g. s3501, +80M each) | ≈ 32 | does MORE breadth exposure keep raising a plateaued network's F? | rejected as the GO's follow-up: it keeps the plasticity confound N1-P's NULL branch names; it is the cheap complement if N1-P is UNDECIDED with Δ̂ > 0 (the strategy doc) |
| Fork both arms from an early X26 checkpoint (25M) at today's pin | ≈ 30 | breadth from 25M | rejected: its first 25M are X26's pin and pool-only, a third code version inside both arms; saves only ≈ 5 GPU-h per arm |
| Fresh toy pair (the review's Option T, `main.x61.arms`) | ≈ 13 toy GPU-h | the same question on the 1.42M toy | rejected for a GO: the toy's familiarity premium is about half P\*'s (X51) and sits below its own GO bar (review B2), and its transfer to production is uncalibrated. Kept as the cheap screen if N1-P is UNDECIDED |
| N1F vs X26's trajectory only | ≈ 20 | — | rejected (§2): cross-pin and one seed each |

**The case for fresh, in one line:** the plateau forks have now asked the plasticity-confounded question four times
(D1, D3, X52, N1-P), and a GO from the fourth is the moment to ask the clean one.

---

## 5. The pin

- **Chosen: main at registration, tagged** (an annotated tag on origin, as D3's and N1-P's pins), for BOTH arms and
  every stage. It carries `n1_proc_v1`, the AM-D1-FIX label guard (`label_path_defects`, which N1-P's pin omits),
  the pinned-data root, and the 10-26 GIGO fixes.
- **Rejected: N1-P's pin `5f8f6cfa`.** It keeps the contrast with C0 clean for N1-P, but X53-N1's control is CF, so
  that reason does not carry. Training a ≈ 55 GPU-h lineage, which may become the next production lineage, on code
  with known observation and label defects would bake them into the result.
- **The cost of choosing main:** the Rust core's full-parity programme is mid-flight (364 of 369 moves). The pin
  freezes whatever state it is in. Both arms share it, so the contrast is clean, and the registration records how far
  the programme had got.
- **The reads' playing commit.** X26's checkpoints (an older config) and N1F / CF (config ≥ 162) must be playable by
  the read's commit. P\* replayed X51 game for game at `4ee655648` (the belief-prior-mix guard), so older checkpoints
  play identically at newer commits at least that far. The X26 cells may live in their own `main.reads` state; they
  are descriptive.

---

## 6. Steps, GPU-hours, seeds, lockstep

**The rate.** At the X35 production pin, 20M-step forks ran at 1,388 / 1,413 / 1,433 steps/s with the corpus (N1-P)
and 1,384 / 1,418 / 1,424 without (C0), 4.0–4.4 GPU-h per 20M (MEASURED, `dose/`). X26 took ≈ 55 GPU-h to 280M at
≈ 1,400 steps/s. A FRESH run's early rate (shorter or longer early games, a near-empty snapshot pool) is
**UNVERIFIED**; the table assumes 1,410.

| milestone | steps | GPU-h per arm (ESTIMATE) | cumulative, both arms |
|---|---:|---:|---:|
| L25 | 25M | 4.9 | 9.9 |
| L50 | 50M | 9.9 | 19.7 |
| **L100 (Stage A screen)** | 100M | 19.7 | **39.4** |
| Stage B: s5302 pair to 100M | 100M | 19.7 | +39.4 |
| Stage C: N1F s5301 100M → its plateau (≤ 300M) | ≈ 280M | +35.5 | +35.5 |
| Stage C option: CF s5301 to the plateau (same-pin attribution) | ≈ 280M | +35.5 | +35.5 |

- **Seeds:** 5301 for Stage A, 5302 for Stage B; both arms share each seed (seeds are draws: the same pin and seed
  diverge at rollout 1, measured 2026-09-09).
- **Lockstep.** The arms advance in segments that end at L25, L50 and L100, so neither arm runs ahead of a stop rule
  (a segment end is a normal restart under `--restart-interval-hours`). On ONE card the segments alternate CF,
  N1F; on two identical cards they run side by side (§9).
- **Restarts** are routine and identical in kind for both arms; each restart's step is recorded.
- **Wall time on our card:** Stage A ≈ 1.7 days; all three stages ≈ 4.8 days.

---

## 7. The meters along the way

Every Kakuna cell is greedy against greedy, ckpt 34, CPU, role-balanced 20-game units, through `main.reads`.

| look | primary (decides) | reported, never decides | lane-h (ESTIMATE) |
|---|---|---|---:|
| launch | — | VOID guards: the TEAM-MIX banner (N1F), 0 held-out hits, 0 pool copies, the recorded pin; first-hour steps/s ≥ 1,000 (else a HOLD); compile canary clean | 0 |
| **L25** | the harm stop on Δ_F (600 F games per arm) | X26 @25M on F (600); the in-run eval | ≈ 5 |
| **L50** | the futility stop on Δ_F (1,200 per arm) | X26 @50M on F (1,200); S per arm (3,500 pairs) | ≈ 16 |
| **L100** | **the screen: Δ_F at 2,400 F games per arm** | H, P72 and home72 (420 each per arm) with the set-coverage line; Δ_H − Δ_F; the REVERSE cell (we pilot F, Kakuna pilots home72; 400 per arm); S per arm; X26 @100M on F (2,400); CF − X26 | ≈ 33 |
| Stage B L100 | the pooled two-seed Δ_F (4,800 per arm) and P | as above | ≈ 33 |
| Stage C | the plateau stop: `python -m main.plateau tick` on N1F every 10M from 150M (Tier 1's GSPRT, newest node vs 50M back) | F at 200M (1,200) | ≈ 3 per tick + reads |
| **N1F's plateau** | the goal reads: X54 on `forum216b` (1,600–2,400 games) + home72 (400, goal (a)'s guard) | U with its coverage line; N1F vs P\* on F and home72 (descriptive, cross-pin) | ≈ 8–11 |

- **The plateau-tick schedule** is the one X26's plateau was read by (Tier 1 offline, U9a). Stage C stops at the first
  PLATEAU decision or at 300M, whichever comes first, so the 280M figure is a cap estimate, not a target.
- **The in-house meter steers nothing about goal (b)** (X37b: it is blind to the piloting deficit); S reads only
  the in-pool cost, as D1's co-primary P does.
- **Kakuna on the unfamiliar panels at every look** is F only. H and P72 are hazarded for any N1-trained model.

---

## 8. The decision points and the early stops

Rule 8 applies throughout: a bound or point estimate within 1e-9 of a threshold takes the undecided / continue
branch.

| stage, look | rule | consequence |
|---|---|---|
| any | a VOID guard fails (§7 launch row; K9(b) at each segment end, AM-6's ceilings) | that run is VOID: re-run at the same pin and seed after the cause is fixed; nothing is read |
| A, L25 | **HARM-STOP** iff the Newcombe UB of Δ_F (600 per arm) < −2.0 pp | stop both arms: breadth slows early learning beyond what 100M is likely to recover. FINAL for the stage |
| A, L50 | **FUTILITY-STOP** iff Δ̂_F ≤ −2.0 pp (1,200 per arm) | stop both arms. FINAL |
| A, L100 | **SCREEN-GO** iff Δ̂_F ≥ +2.0 pp (2,400 per arm) | Stage B. Otherwise **SCREEN-STOP**: both arms stop at 100M; "N1 from step 0 not shown, one seed" (a low-power screen, never a NULL) |
| B, L100 | D1's classes on the pooled Δ_F (two seed pairs) with the registered between-seed term; the sign rule: both pairs' Δ̂ ≥ 0. **GAIN** iff LB > 0 and the sign rule; **LOSS** iff UB < 0; otherwise UNDECIDED. Co-primary P: Welch on S, **NON-INFERIOR** iff LB > −2.0 pp | GAIN + P NON-INFERIOR = ADOPT-CANDIDATE; GAIN without it = TRADE (the exchange rate Δ̂_F / −Δ̂_P to the owner). Either ⇒ Stage C. UNDECIDED or LOSS ⇒ stop; the orchestrator rules |
| C, plateau | report N1F's X54 read, home72 (goal (a) must stay met: LB > 0.50) and the coverage line | goal (b) is claimed only on X54's registered terms. CF goes to its plateau too only if N1F's plateau F beats P\*'s 0.530 (X51) and the recipe question (adopt `n1_proc_v1` into production) needs a same-pin attribution |

**The between-seed term.** With two seeds per arm, AM-5's φ̂ has too few degrees of freedom to stand alone. The
registration fixes a floor: the pooled interval uses max(the measured spread, 1.5 pp per run) as the seed SD.

**Power** (Monte Carlo of this chain, 300,000 draws a cell; game noise at F's base 0.49; a run-level seed effect of
SD σ shared by all looks of one run; the effect at 25M and 50M assumed to be ¼ and ½ of the 100M effect, a GUESS):

| true Δ_F at 100M | σ = 0: SCREEN-GO / final GAIN | σ = 1 pp | σ = 2 pp | expected GPU-h for A + B (σ = 1) |
|---|---|---|---|---:|
| 0 | 0.07 / 0.00 | 0.15 / 0.01 | 0.26 / 0.02 | 41 |
| +2 | 0.46 / 0.06 | 0.47 / 0.12 | 0.48 / 0.13 | 56 |
| +4 | 0.89 / 0.64 | 0.81 / 0.58 | 0.71 / 0.39 | 70 |
| +6 | 0.99 / 0.98 | 0.96 / 0.92 | 0.86 / 0.71 | 77 |

- Under a true null the early stops end ≈ 16–26 % of runs at L50, and SCREEN-STOP ends most of the rest at 100M; the
  false-GAIN rate is ≤ 0.02.
- **NULL is unreachable.** With a 1.5 pp seed floor the pooled interval is wider than D1's ±3 pp band, so the pair
  can say GAIN or not-shown, never "no effect". That is the honest reach of two fresh seeds per arm.
- The registration replaces this with a Monte Carlo of the registered code (`power.py`, as N1-P's), and states the
  25M / 50M effect fractions it assumes.

---

## 9. The second GPU: would the cloud plan run it?

**Not as approved, and it should be the first workload a widened approval covers.** Against
[`../ops/cloud_gpu_plan.md`](../ops/cloud_gpu_plan.md):

- **The card rule (§1.2) costs X53-N1 nothing extra.** No comparison may cross card types, and every seed of an arm
  runs on one type. D3 on the cloud needs a cloud C0 (C0v, ≈ 12 GPU-h) because its control is banked locally.
  X53-N1's control is CF, run fresh beside N1F, so both go on the same card type and no banked row is needed. The only
  cross-card line is X26's trajectory, which is descriptive anyway.
- **Nothing from `models/` goes up.** A fresh run has no parent checkpoint or snapshot pool to upload, so approval
  question (b) (uploading P\* and its snapshots) does not apply. Results come back by the plan's pull-back (§2.4).
  **UNVERIFIED:** that a fresh production run reads no other file under `models/` at startup; the box dry-run answers
  it.
- **The plan's shape does not fit.** It runs one box per 20M fork (≈ 2.5 box-h) with a 240-minute deadman, and its
  kit refuses a second instance. A fresh arm is ≈ 12 box-h to 100M on a 5090 (at the measured 1.65×) and ≈ 34 box-h to
  280M. It needs either a longer deadman or segment boxes (pull the checkpoint, resume on a fresh box: a restart,
  which the arms already take). The lockstep segments (§6) are a natural box boundary.
- **Two identical cards at once halve the wall time** (CF and N1F side by side), but that needs the one-instance rule
  lifted for a pair of the same card type. That is an owner question.
- **The cost** at $0.50–0.70/h (cloud plan §3, plus ≈ $0.05/h and ≈ $0.15 a box): Stage A ≈ 24 box-h ≈ **$13–18**;
  Stage B the same; Stage C ≈ 22 box-h ≈ $12–16 (+ the same for the CF option). **≈ $38–52 if every stage runs**,
  above the $20 the current approval asks for.
- **The card gate.** The plan's C0v gate screens whether a 5090 trains P\*'s fork like our card. X53-N1 needs only its
  mechanical bars on each run: the parity gate, the compile canary, `compile/recompiles_after_lock` 0, zero unplanned
  restarts, and K9(b) PASS at each segment end. CF itself is the cloud baseline.
- **Where it runs, as a rule:**
  - with N1H planned: the whole experiment LOCAL (human-derived teams never reach a box, and N1H shares CF);
  - without N1H, and with the owner's widened approval: the whole experiment on the CLOUD, while our card keeps the
    fork queue (D3 look 2, X59-S, the strategy doc's forks);
  - otherwise local: Stage A takes our card for ≈ 1.7 days.
  - A bought 5090 replaces the rental: then all of X53-N1 runs on it, and X26's line becomes cross-card (descriptive
    anyway).

---

## 10. The pre-registration skeleton

To be written whole after N1-P's read, before any launch, as `measurements/x53_n1_fresh_<date>/PREREG.md`:

1. **Question and mechanism** (§1), and N1-P's verdict that licensed it, cited by its record.
2. **The arms' argv** from a kit (a fresh-pair kit on `main.x61.arms`'s pattern: a fresh pair at one pin), with
   `checkargs` exit 0 at the pin and `--dry-run` transcripts; the resolved-config diff of each arm against X26's
   record; the pin's tag and its `git describe`; the full-parity programme's state at the pin.
3. **The corpus card** by reference (X61 §1), its sha256s, the startup banner line each N1F run must print, and the
   set-coverage line of every panel by `pool ∪ n1_proc_v1`.
4. **The pre-launch checks:** the `--debug` smoke with the lever at the pin (N1-P never ran it, its FINDING 6),
   the parity gate, and the tag on origin.
5. **The looks and rules** of §7–§8 verbatim, with the 1e-9 rule, the seed floor, the sign rule, the floor on counted
   games per look (≥ the planned games minus one unit per arm, else INCONCLUSIVE), and the `main.reads` specs
   (`check-spec` exit 0).
6. **The REPORTED lines** and why each decides nothing (M4).
7. **The VOID guards**, which are module-checked and which are the operator's (N1-P's FINDING 11 pattern).
8. **Power:** the Monte Carlo of the registered code, with the false-GAIN rate of the screen-then-pool chain.
9. **Predictions** (GUESS), per look and per class.
10. **Cost** per stage in GPU-h, lane-h and, on the cloud, dollars; where it runs (§9).
11. **What it cannot say:** one share, one corpus, one pin; NULL unreachable; goal (b) only through X54.
12. **FINDINGS at registration.**

---

## 11. Cost

| item | GPU-h | CPU lane-h | agent-days |
|---|---:|---:|---:|
| the fresh-pair kit, the specs, the registration | 0 | ≈ 1 (checks) | ≈ 1 |
| Stage A (L25, L50, L100) | ≈ 39 | ≈ 54 | ≈ 0.3 (operation) |
| Stage B (on SCREEN-GO) | ≈ 39 | ≈ 33 | ≈ 0.2 |
| Stage C, N1F to its plateau (on GAIN or TRADE) | ≈ 36 (cap) | ≈ 25 (ticks, F at 200M, X54) | ≈ 0.3 |
| Stage C option, CF to its plateau | ≈ 36 | ≈ 10 | ≈ 0.1 |
| N1H (conditional): corpus build + one arm to 100M | ≈ 20 | ≈ 33 | ≈ 1.5 |
| **every stage, no options** | **≈ 114** | **≈ 113** | **≈ 1.8** |

The CPU reads are the next bottleneck (cloud plan §3 FINDING): Stage A's ≈ 54 lane-h is ≈ 18 h on rule 5b's three
agent cores while a run is live.

---

## 12. FINDINGS (standing rule 7)

1. **A fresh production run's seed-to-seed spread on F against Kakuna is unmeasured.** Fork seeds overdispersed by
   ≈ 0 (D1's five, N1-P review M2); fresh runs diverge from rollout 1, and nothing says they converge to the same F
   at 100M. The design's power swings from 0.64 to 0.39 at +4 pp across σ = 0–2 pp. CF s5301 − X26 @100M is the
   first crude look (cross-pin as well as cross-seed).
2. **The effect's time course is a GUESS.** The early stops assume a quarter of the 100M effect at 25M and half at
   50M. If breadth costs early learning and pays late, the L50 futility stop fires on a true effect more often than
   §8 says.
3. **`--team-mix` takes ONE corpus** (`agents.training.matchup_spec.parse_team_mix`). N1F and N1H are therefore
   separate arms by construction, which the human-data rule wants anyway; a union arm would need a union corpus.
4. **The X35 kit is fork-shaped** (C0's argv, P\* as the start). A fresh pair needs its own kit arms, on
   `main.x61.arms`'s pattern (≈ 0.5 agent-day inside §11's day).
5. **The fresh-run throughput, and whether a fresh run reads anything under `models/`, are UNVERIFIED** (§6, §9).
6. **This unit ran no game and used no GPU.** The power table comes from a 2-second CPU Monte Carlo of the chain in
   §8, outside the repo; the registration redoes it on the registered code.

---

## 13. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-26 | **This doc CREATED (DRAFT)** (research-strategy worker, orchestrator brief) | X53-N1 as a fresh production PAIR at one pin, N1F (`n1_proc_v1` at 0.5, both sides) vs CF, in lockstep; X26's trajectory descriptive; F primary (M4); stages A (one seed each to 100M: L25 harm stop, L50 futility stop, L100 screen at Δ̂_F ≥ +2.0), B (a second pair, D1's classes pooled, seed floor 1.5 pp) and C (N1F to its plateau by the Tier-1 tick, X54's goal read); the human-team corpus as a separately toggled, local-only third arm N1H; main at registration as the pin; the cloud as the preferred home once an approval covers it | N1F vs X26's trajectory (cross-pin, one seed); extending N1-P's forks (plasticity confound); an early-X26 fork; the toy (premium below its bar); N1-P's pin (known defects in a ≈ 55 GPU-h lineage); two seeds per arm from the start (≈ 79 GPU-h before any signal); a share other than 0.5 (a second lever) | N1-P PREREG §6, §8; `x35_breakers_stage_2026-10-17/dose/` (n1p, c0 rates); `sota_survey_2026-10-26.md` §5; `cloud_gpu_plan.md` §1.2, §2, §3, §4; this doc's §8 Monte Carlo |
