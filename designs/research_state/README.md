# Research State


> **Measured numbers live in [`measurements/`](measurements/)** — the raw audit outputs, each
> carrying its own checkpoint, step, state count and date. Cite a file there rather than
> re-typing a percentage: that is what makes a stale number detectable by comparing dates
> instead of by re-deriving the result.

The **single source of truth for what we're trying, what we know, what we don't, and what's left
to find.** Version-agnostic (it tracks the ongoing hunt, not one `ai_vN`). Maintained deliberately
by agents — see the protocol below and the `feedback-research-state` memory.

> The whole reason this exists: this project keeps forming plausible hypotheses, and most die under
> scrutiny (8 killed in one session). That knowledge is the asset. A finding that isn't written here
> evaporates and gets re-discovered (or re-believed) next session. **Write the kills, the
> not-knowns, and the pros/cons — not just the wins.**

## Layout

**The APPEND-ONLY record and the MUTABLE views are different documents, and the difference is the
point.** `ledger.md` records what was believed *at the time* and is never edited; everything else
here states what is believed *now* and is rewritten whenever the ledger says so. When the two
disagree, **the ledger wins and the view is wrong** — fix the view.

### The record (append-only)

- **[ledger.md](ledger.md)** — the running hypothesis record: every dispatch with its predictions
  registered BEFORE the data, every verdict scored against them, every kill written as a kill. It
  opens with an at-a-glance status table and then runs chronologically. **Entries are cited by
  their landing sha** (e.g. `1d5a866` = probe M's verdict) — that is this folder's stable
  entry id, since the headings are prose.
- **[measurements/](measurements/)** — the raw audit outputs behind every number quoted anywhere,
  each carrying its own checkpoint, step, state count and date. A `.md` companion beside a `.json`
  is the probe's own write-up. Cite a file here rather than re-typing a percentage.

### The mutable views (rewritten to match the record)

- **This README** — the orientation: what the layout is, [where the programme is
  now](#where-the-programme-is-now-2026-08-30), the [defect genres](#the-defect-genres--five-named-failure-classes),
  the maintenance protocol, the amortizability gate, the build bar, and
  [the frontier](#the-frontier--what-else-might-be-there).
- **[levers/](levers/)** — one file per OPEN or ACTIVE lever, each with the full
  **Known / Not-known / Pros / Cons / Status / Next-test** structure (`levers/_template.md`). Killed
  levers don't normally get a file — their one-line cause-of-death lives in the ledger and in the
  frontier table below; the two killed files that DO survive
  (`opp_pp_observability`, `compile_opponents_net_value`) are kept because the *way* they died is
  reusable.

### Standing programme documents

- **[bait_loop_hunt.md](bait_loop_hunt.md)** — the PRE-REGISTERED gen-16 hunt for the bait/loop
  pathology (we fire an immune move into a voluntary pivot, repeatedly): the baselines, the four
  bars, the two registered confounds, the launch-window cell-liveness check, and the pre-committed
  fork for each of the three end-of-run outcomes. Instrument: `main.prober.query loops`.
- **[hodge_predictions.md](hodge_predictions.md)** — the measured Hodge **non-transitivity
  baselines** (gen-13…17: width excess, noise floor, cyclic fraction, all at 21 players / 174 edges
  / 814 triangles) plus the PRE-REGISTERED spinning-top predictions P1/P2/P3, each with a numeric
  threshold stated against that floor. Carries the game-count confound that voids any naive
  cross-generation width comparison. Instrument: `python -m main.elo <run> --hodge-bootstrap 300`.
- **[capacity_battery.md](capacity_battery.md)** — the regularly-evaluatable SATURATION battery on
  the shared trunk (effective rank · Lyle trainability vs a fresh net · probe decodability · param
  census), its per-metric VALIDITY notes, and the gen-17 baseline row. Ordered because the flywheel
  piles distilled skills into one fixed-capacity network, so saturation has to be visible BEFORE a
  long fruitless hunt. Instrument: `python -m main.capacity <run>` (~7 s, offline). **Tripwire, not
  verdict** — the `PR(K_ū)=17` retraction is why every metric there ships what would have to
  CONFIRM it.
- **[ladder_readiness.md](ladder_readiness.md)** — the PUBLIC-LADDER gap list (WORKS / FIXED /
  SIZED), against the owner's permanent "we must be able to play online" constraint and the
  external Metamon milestone (~Elo 1511 / GXE 64). Carries the two measurements that killed the
  two loudest fears — **zero protocol drift** over a real 59-replay ladder corpus and **18 ms** per
  decision against a 150 s timer — Showdown's actual (discretionary) bot policy with sources, the
  end-to-end smoke results, and a go-live checklist. Instruments: `python src/main/play.py`,
  `python src/main/ladder_drift_scan.py`.
- **[wang_search_reconciliation.md](wang_search_reconciliation.md)** — why Wang 2024's MCTS ablation
  (+31 pts, same-policy mirror) and our depth-1 search-dividend probe (−21 pts, same-policy mirror)
  are the **same phenomenon at different noise levels**, sourced section-by-section from the thesis.
  The crux: **Wang's leaves are the same trained critic ours are** — "rollout-to-end" is NOT what he
  did, so do not cite him for it. What he has and we lack is R=1000–2000 samples per decision on a
  tree that persists across turns, a PUCT policy prior inside selection, and **visit-count argmax
  chosen explicitly over Q-argmax for the variance reason we then walked into**. Carries the diff
  table, the noise/margin ladder, and the one experiment that separates variance from a critic-bias
  floor. Instrument: `python -m main.search_dividend` (the R-ladder is flags only, zero code).
  ⚠️ **Every search-dividend cell measured before 2026-08-29 carries the allocator-noise caveat**
  (`d2a0212`): at the battery's own 1 s budget the grid agrees with its own large-budget argmax on
  only **86.1%** of decisions, so ~1 in 7 "searched" decisions was allocator noise. Re-read any
  load-bearing battery conclusion against that before quoting it.
- **[exploiter_coverage_board.md](exploiter_coverage_board.md)** (+ `.json`) — GENERATED, always-current:
  which of the 32 vetted `data/teams/sample/` teams have a banked exploiter teacher, and the proposed
  arms for the rest. The coverage denominator the flywheel's steering equation multiplies against.
- **[substrate_exploiter_gates.md](substrate_exploiter_gates.md)** — the pre-registered kill-capable
  gates for the v93–v95 conditional-mechanics substrate (owner-supplied scenarios, teams identified by
  roster). The substrate shipped and gen-16 ran it; these gates are the un-spent half.
- **[cf_r1_runbook.md](cf_r1_runbook.md)** — the counterfactual **label factory** (R1) runbook: the
  cost model, the sampler design, the label-quality prerequisites. Now load-bearing for two live
  programmes at once — the win-prob head's grounded labels and the E5 Q-head's supply side.
- **[critic_calibration_plan.md](critic_calibration_plan.md)** · **[gigo_signal_probe.md](gigo_signal_probe.md)**
  — two smaller standing plans (critic reliability; the GIGO/order-mismatch tripwire).
- **[metamon_replay_feasibility.md](metamon_replay_feasibility.md)** — the external-corpus feasibility
  study behind the (owner-PUNTED) human-replay chapter.
- **[`../ops/TRAINING_RUN_SOP.md`](../ops/TRAINING_RUN_SOP.md)** — the era-independent operating
  procedure for a training run (launch checks, watchers + the 55-minute cron, kill/relaunch, reading).
- **`gen{11..16}_endofrun_runbook.md`** — one per generation: the pre-registered end-of-run battery for
  that generation, its §-numbered gates, and what each gate returned. Historical once its generation
  closes, except where a § is explicitly still open (`gen14_endofrun_runbook` §(c) was closed by probe O
  on 2026-08-29, `32c39df`).
- **[learning_notes/](learning_notes/)** — the durable concept notes a probe spun off (plasticity,
  distillability, PSRO/exploitability, imperfect information, …). ⚠️ Several carry **STATUS BANNERS**
  marking which of their claims a later verdict refuted — read the banner first; the note bodies were
  deliberately NOT rewritten, so the refutation stays visible in the artifact.
- **[The frontier](#the-frontier--what-else-might-be-there)** (below) — the standing list of candidate
  levers NOT yet (fully) investigated. This is the working surface for "there has to be more."

## Where the programme is NOW (2026-08-30)

> Every live programme, written as the *current* belief with the ledger sha that established it —
> plus the kill table (§2) and the build state (§7). Nearly all of it changed in the week of
> 2026-08-26 → 08-30. **If a claim below carries no sha, it is not yet measured.**

### 1. The flywheel capstone (rev-1 → rev-2 → rev-3): transfer is REAL and LOCAL; the bar MISSED

- **rev-2 (`e4c13a9`) — HOLDS BY LETTER, QUALIFIED.** All three pre-registered bars passed
  (R2-ACTION − R2-CTRL **+0.0741 z=5.47**), and the honest sentence beside it: **vs its own parent
  +0.0161 z=1.33 n.s.** The control fell −0.0580, so the fold *prevented a 5.8pp loss*; it did not
  add gain. The monotone content ordering survives the control entirely: **action +1.6 > top-3 −2.4
  > full-KL −3.2 > nothing −5.8**.
- **Content is REAL and teacher-SPECIFIC (`d4d551d`).** The ZapDug natural experiment — one team
  pinned by two teachers, taught by only one because `_distill_mask()` breaks on first match —
  gives a difference-in-differences of **≥+4.0pp**, a lower bound (shared content cancels in a DiD).
  *The flywheel has demonstrably transferred content at least once.* Third result, unasked:
  **ALIGNMENT ≠ BENEFIT** — R2-KL absorbed the most teacher shift and finished 4.8pp behind
  R2-ACTION. Copying the teacher's DIRECTION hurts; copying its DECISION pays.
- **rev-3 (`ade78c1`) — the absolute improvement bar MISSED**, called honestly with no optional
  stopping (R3-ACTION +0.0174 z=1.29 · HI +0.0211 z=1.57 vs rev-1). **What is real: +6pp on the 3
  COVERAGE teams, z=2.57, REPLICATED at a second dose.** Transfer works where HEADROOM exists —
  coverage-team headroom 25.7pp vs meter-team 7.7pp. The bar was pointed at mined-out teams.
- **THE TEACHER CEILING (`61608ac`) — and the death of the budget law.** Teachers converge to
  **~0.6881 [0.672, 0.704]** set-mean, INVARIANT to budget (1.5 vs 2.5M/team: +0.0019 z=0.16) and to
  target strength. Target rose +0.0587, extraction fell −0.0569 — *the same number*. **Extraction
  was never a teacher property; it is headroom to a fixed ceiling.** The budget law's celebrated
  prospective confirmation (rev-2's sd 0.0098 cluster) was constant headroom in disguise.
- **THE TREADMILL (`ade78c1`) — the fold REDISTRIBUTES.** R2-ACTION lost **−5.9pp z=2.5 on untaught
  coverage teams** while gaining on taught ones; rev-3 then recovered it. Each narrow fold damages
  the uncovered and the next revolution repairs the last. This is the strongest argument FOR high
  coverage: **breadth is the anti-treadmill lever, not merely an adder.**
- **The steering equation (`36fc4af`):** general gain ≈ per-team gain × coverage fraction ×
  retention. At 9–12 taught teams of 719, complete transfer of +8pp/team moves full-pool h2h by
  **~0.1–0.2pp** — so "does not generalise" and "cannot be seen by this instrument" are both true
  sentences here, and the h2h is a REGRESSION guard, never a corroboration requirement.
- **Standing narrative, frozen before the evidence (`91d5125`):** breadth-per-teacher →
  generalizable content → the fold's externality flips from ROBBERY to GIFT → compounding = v8's
  +69. Four pending rows, each with its damage condition stated: R3-SELF must NOT reproduce the
  +6pp; probe P's off-lineage arm must keep v8-untaught ≥0; **probe Q** must show rev-3's own
  untaught pull-down ≈ −4..−6; **rev-4's 3×8-vs-6×2 shape discriminator** must show broad teachers
  at the same ceiling with reduced-or-reversed robbery. Then the **40-team revolution**, teams
  selected BY HEADROOM (`main.exploitability` prints the per-class read for exactly this).

### 2. Seven accounts of v8's +69 ELIMINATED — each with the measurement that killed it

| Account | Killed by | The number |
|---|---|---|
| **Parent RIGIDITY / plasticity** | plasticity forensics `181c1d5` | P1 & P3 **REFUTED-OPPOSITE** — the 277M v8 parent shows NO capacity loss (Lyle 1.154) while plastic rev-1 shows mild loss (0.948); v8's teacher deltas were TRUNK-heavy (0.47 vs 0.28). "Converged in loss" ≠ "rigid in capacity" |
| **Dark-knowledge TAILS** | probes D `92ad277` + E `2122878` | The tail is certifiably **NOISE** — five exploiters agree about their tails no better than with a run that never had a teacher (inter-fork 0.327 vs no-fork control 0.306). And v8's tails are not special: **0.349 vs gen's 0.344 like-for-like**, CI [−0.021,+0.033]. The full-KL re-entry path lost its only empirical pillar |
| **BREADTH → differentiation** | probe A `673a694` | Slope **+0.0003 ± 0.0013 (z=0.23)** across a 2/3/4/9-team ladder, recipe-controlled. What DOES move is fork LENGTH (+0.039 ± 0.016 at fixed K=9, 3M→9M) |
| **Opponent CURRICULUM (C1)** | F6-CURR, scored in `ade78c1` | **NULL, z=−1.40** at rev-3 scale (power rules out >4.5pp only). The then-top-ranked candidate, killed at measured power |
| **BUDGET (per-team steps)** | rev-3 admission `61608ac` | +0.0019 **z=0.16** between 1.5M and 2.5M per team. The law is dissolved, not merely unconfirmed |
| **DOSE** | rev-3 `ade78c1` | R3-ACTION-HI at 0.35 bought nothing for 4.5M steps — carry the CHEAP arm forward |
| **ECOLOGY / adaptive team-PFSP** | probe P interim P2 | The adaptive-repair signature the treadmill account predicted is not in v8's artifacts. *(P-final and its off-lineage arm are still running — this row is the interim read and is the one most likely to move.)* |

**The last untested supply axis is teams-per-teacher BREADTH for TRANSFER** (not for
differentiation, which probe A flattened — a different metric, and the tension is named, not
hidden). It is also the axis v8 differs on most: 23 distinct teams via 3 teachers at 10/3/10,
against rev-3's 12 via 6 at 2. Rev-4 adjudicates at fixed total compute.

### 3. The search programme: as a PLAYER it dead-ends; as a TEACHER it is the road

- **Probe G (`5f98d26`) — pair first.** Critic error decomposes into a per-DECISION shared offset
  **0.728 of true MSE** (cancels between siblings, so shallow paired search is favored — and does
  NOT cancel across depth) and a **0.272 differential** residual. Contrastive critic training is
  therefore **SIZED at ≤5.7pp of per-decision regret, not convicted** — a later lever, not the
  constraint. **The bankable surprise: ranking by the one-ply WIN-PROB head beats the action the
  policy actually played (+0.0219 [+0.0089,+0.0364]); the scalar V head does not clear zero.**
  Any search must read the win-prob head, not V.
- **Probe H (`79e8b11`) — the "forced decisions" premise is REFUTED.** Search flips **69.4%** of
  decisions and no cheap policy-confidence feature separates the flips (gap/entropy/top-1 at or
  below the random null). What IS separable is flip **COST**: 83% of the dividend sits in 22.7% of
  decisions, findable only by **|P(win) − 0.5|**. *The policy does not know when search will
  overrule it; the critic knows when being overruled would not matter.*
- **Probe I (`d2a0212`) — racing, and a retroactive caveat.** Racing buys 1.47× on the deadline axis
  / 1.87–2.40× on spend. The separation distribution is **U-SHAPED with an empty middle** (52.2%
  never separate; the rest separate at the floor) — so non-separation is itself a mid-search triage
  signal. And the allocator-noise finding above, which every historical battery cell now carries.
- **Defensive paired search, iteration 1 (`4cf81fd`) — search STOPPED LOSING.** Mirror **0.4937**
  [0.4448, 0.5427] vs honest_1s's 0.2929 (Δ +0.2008), beating playoff_10s at **1/20 the budget**,
  on literally-identical seeds. Stretch (CI > 0.50) honestly not met.
- **Iteration 2 (`35dbc3c`) — the mechanism moved EXACTLY to spec and the dividend is ZERO.**
  Overrules 1.8% → 5.82% (13×), separated-of-raced 0.157 → 0.454 (95% of I's ceiling), and the win
  rate landed **0.5003 [0.4803, 0.5203] — the point estimate IS the null.** Mechanism analysis of
  record: the **WINNER'S CURSE of a biased instrument** — CRN pairing removes dice noise and the
  shared offset, so what racing certifies is the leaf's residual DIFFERENTIAL bias (RMS 0.122,
  larger than most true gaps) as much as signal. *Statistical separation of a biased reader is not
  correctness.*
- **Probe K (`2af60c2`) — the leaf is PARTIALLY EXONERATED.** Re-judged under opponent-marginalized
  ground truth, iteration 2's overrules bought **+0.0474 [+0.0216, +0.0730] per decision — REAL**,
  and G's edge was not a frozen-opponent artifact. The game-level zero stands as fact; its
  *attribution to the leaf* does not.
- **The transfer cell (`deb0bc9`) — COMPOUNDING convicted.** 8,100 games / 4,050 paired units:
  A−B = +0.0020 against a naive expectation of +1.16pp ⇒ **transfer coefficient τ = 0.17
  [−0.34, +0.68], which EXCLUDES 1.0.** Checkpoint and the bot half of population are removed with
  the dividend still absent. Good behavior worth keeping: the triage gate **auto-scales dose to
  headroom** (92.6% forced vs saturated bots, 9× fewer overrules than in the mirror).
- **Standing conclusion:** per-decision gains are real and **do not compose in play**. Search's
  value is as **TEACHER and data** — exactly where the ai_v12 programme went. This is consistent
  with, not a challenge to, the owner's hard constraint that search never rides on the model.

### 4. The win-prob programme: the head KNOWS, and only an explicit teacher can move the policy

- **The BAROMETER / COACH distinction (`b070d6e`).** Two "shapings" had been conflated. The live
  `win_prob_mode="shaping"` @0.05 is **REPRESENTATION** shaping — it pushes outcome-predictive
  features into the trunk and exerts **zero force on behavior** (no gradient path from predict-wins
  to choose-winning-actions). And the head's labels are self-referential: habitual whiffs that still
  win 55% teach it "55%", never "the whiff was the mistake". *Reward-level PBRS with φ = P(win) is a
  DISTINCT, still-UNSANCTIONED proposal — do not quote it as adjudicated.*
- **Probe L (`bda8382`) — the head knows 96.4% of the whiffs AT DECISION TIME.** 0.964
  [0.948, 0.978] against a ≥60% bar, median margin 0.049 win-prob units against a within-decision
  sd of 0.00062 — two orders of magnitude. It is **whiff-SPECIFIC** (+0.213 vs hit-pivot, +0.342 vs
  no-pivot), not probe G's generic edge. α flags THE PIVOT, not the whiff — the correct division of
  labor. **Repeat offenders refuted by a ceiling: the head is at 1.000 on the FIRST click of a
  loop** — it knew immediately, forever, and was ignored every time, because **the policy samples
  the head's preferred action at median p = 0.002.**
- **…and the "shaping" lever is structurally REFUTED, with the argument the registration lacked.**
  Trunk share 1.02% at cosine **−0.133 AGAINST** the policy gradient; the reward registry has no
  win-prob member; even hypothetical PBRS @0.05 is homeopathic. "Raise the dose" names no real
  mechanism. **The head's ranking is not a quantity the network computes — it is the head COMPOSED
  WITH A SIMULATOR (one re-roll per action), a composition PPO never performs. No coefficient can
  deliver it; only an explicit teacher that materializes the ranking and writes it back.**
- **The three-route taxonomy (`1984dc7`).** Route 1 = PBRS φ=P(win) at the REWARD level (suppress
  without knowing the alternative; protected by the telescoping-invariance theorem; UNSANCTIONED).
  Route 2 = ranking DISTILLATION at the TARGET level (prescribe the alternative without carrying
  why; **no shield** — it imports the head's differential bias, i.e. iteration 2's winner's curse).
  Route 3 = confirmed defensive-search OVERRULES at INFERENCE. **Route 3 feeds route 2**: confirmed
  overrules are route 2's highest-quality training targets — the AlphaZero loop in miniature, search
  manufacturing the curriculum.
- **All three BUILT and dormant (`f5c8a77`, `ade5a11`).** And the build **caught its own
  two-order-of-magnitude arm-sizing error**: the PBRS coefficient ladder {0, 0.1, 0.3} was sized
  against an assumed terminal reward of order 1, but `VICTORY_VALUE = 30` — the arm would have
  measured nothing and the null would have read as a verdict. Restated as {0, 3, 9} fractions of
  VICTORY_VALUE. *No PBRS term has ever actually run.*
- **E5, the closed loop (`5edbd05`):** predict (a per-action win-prob readout on the pointer head's
  own action tokens) → ground (factory re-rolls as labels) → prioritize (the CfEvidentialHead's
  evidence as the sampler) → teach (the same labels double as route-2 targets) → **measure: the
  AMORTIZATION RESIDUAL (Q-head vs true re-roll) IS the value of one-ply search as a number.**
  Shipped dormant at v107; **the producer does not yet emit `q_labels`** — the named open piece.
- **Value foundations (`596608e`, `8b83cff`):** the shaped-return critic is *definitionally* correct
  for its job (GAE cannot mix currencies), the win-prob head is the correct GAME value, and the
  two-head structure is the automatic consequence of choosing shaped rewards — not a design accident
  to repair. **The critic holds no knowledge; its entire job is policy-gradient variance
  reduction** — epistemically second-tier BY DESIGN, operationally first-tier BY NECESSITY. The
  only error was the battery using instrument A for job B, found by probe G and fixed.

### 5. The clean world: three arms, and every hand-tuned term retired

- **Structure (`e22bd08`, `627ab58`).** Terminal ∈ {+1, −1} with **draw = −1**, and three arms whose
  every pairwise difference is a named quantity: **SPARSE** (no potential) · **SELF-φ** (the run's
  own live head) · **FROZEN-φ** (a mature prior-generation head, which also restores the PBRS
  invariance theorem EXACTLY). SELF−SPARSE = the value of self-shaping; FROZEN−SELF = the value of
  maturity; FROZEN−SPARSE = the total worth of outcome-grounded shaping. The incumbent comparison
  comes free via h2h + anchored ELO.
- **draw = −1 is load-bearing, not a preference (`cfbc9bf`).** {+1, −1, 0} would make the 250-turn
  stall the best non-winning outcome in an arm with zero anti-stall terms. **Stall rate is a PRIMARY
  endpoint.** And **NO anti-stall bias at launch** — with draw = loss, stalling is weakly dominated,
  so a bias is only needed if the model lands in a can't-win-won't-lose optimum, which is an
  EMPIRICAL condition. The escalation rule is pre-registered: the bias enters only if the stall-rate
  endpoint fires, and enters as a registered change, never a mid-run patch.
- **🚨 The famine claim is NARRATIVE, not measurement (`4d22ae4`).** "The early run would face
  near-sparse rewards — the label-density famine that made shaping necessary" was stated as fact and
  is **unmeasured lore: no sparse-reward arm has ever run in this project.** That is why the
  PURE-SPARSE control arm exists and why a paired ~5M sparse-vs-shaped pre-test sizes the full arms
  before three generation-scale runs are committed. Same genre as the probe-E lesson: *the claims
  most worth auditing are the ones load-bearing enough that nobody thought to check.*
- **Two rulings that look like details and are not (`cfbc9bf`, `db9bb5c`):** the **coefficient
  spelling** is coef 2 on φ = p, never 2p − 1 (the affine constant at γ<1 pays a per-step bonus for
  LONGER episodes — wrong sign — and the terminal φ:=0 convention is correct for [0,1] and wrong for
  [−1,+1]); and **PBRS pays transitions only**, so constants are free (the critic baseline absorbs
  them) while offsets are charged. Corollary worth keeping: under a good potential **V_shaped goes
  CONSTANT** — so "V is directly readable as expected outcome" holds for the SPARSE arm and the head
  only; in shaped arms the outcome-readable quantity is V_shaped + coef·φ, which is what the
  scaffolding gauge computes.
- **PopArt RETIREMENT (`2d38a4a`)** is registered for the sparse and clean arms: its job (scale-30
  drifting shaped returns swamping the trunk) is deleted by the reward design. ±1's real gift is
  **stationarity**, not range.

### 6. The reward audit: the one live BIAS term is a switch tax nobody re-derived

- **Scope collapse first (probe M, `1d5a866`).** Of 29 registry BIAS members, `--all-shaping-pbrs`
  (default-ON since 2026-08-18) leaves exactly **ONE** live: production reward = **1 TERMINAL + 7
  PBRS + 1 BIAS (`no_progress_tax`)**. The remembered anti-stall family has not run in months.
- **Alignment REFUTED, over-tax HELD far larger than predicted.** The tax carries no win-prob
  information beyond phase and action kind (45.7% vs a 44.7% matched control). **48.5% of charges
  have Δφ ≥ 0** — and the decomposition is behavioral: **73% of voluntary switches are charged vs
  6.7% of moves; 79% of all charges land on switches; 36% on ZERO-AGENCY post-faint replacements.**
  The implied differential is **−0.101 reward/decision against switching** while the head rates
  switching **+0.0042 win-prob BETTER** (significant) — opposite signs.
- **HYPOTHESIS REGISTERED (not yet causal): the `no_progress_tax` is a candidate CAUSE of the
  long-standing under-switching pathology** (policy ~16% vs strong humans ~30%). A reward term
  paying −0.101/decision against switching for a whole generation is exactly the shape that produces
  it. **The causal arm needs NO code: `--no-progress-penalty 0.0`.**
- **Probe N (`cfbc9bf`) — the intent verdict is COMPOSITION DRIFT.** Charging a voluntary switch was
  **designed in writing** — but inside a reward where a switch also collected **+0.35 net** of
  bonuses (`SWITCH_BASE_BONUS` 0.5, se_switch, escape_threat, pivot_*). Commit `928a00b`
  (2026-06-12) zeroed every counterweight and kept the toll; the net sign on switching flipped
  **+0.35 → −0.15** and nobody re-derived the term. The manager still carries a comment describing
  the dead world.
- Three implementation defects confirmed line-by-line: the SITOUT guard is off by one window (it
  exempts the KO turn and charges the replacement); **no switch can ever satisfy `_is_progress`** —
  the term prices an action KIND, not progress; and the trapped gate reads the upcoming legal too.
  Fixes **landed flag-gated OFF at config v106** (`132d198`) — retrain-class, enable-able after the
  verdict, deliberately not touched mid-campaign.

### 7. Build state — the ai_v12 wave is COMPLETE (5/5), and everything is dormant

| Wave | Landed | What now exists |
|---|---|---|
| **A** (`132d198`) | config **v105** | The clean world is FLAG-REACHABLE end-to-end (`[Reward] composition: 1 TERMINAL + 0 PBRS + 0 BIAS`) + the frozen-φ `--win-prob-pbrs-source` loader |
| **B** (`195ce9f`) | — | The harvest/factory backbone: `main.harvest` → `winprob_finetune` → `harvest_meter` |
| **C** (`ade5a11`) | config **v107** | The E5 Q-win-prob head on the pointer's action tokens, `read_only`, every input detached inside the forward (trunk exposure is *unrepresentable*, not defaulted off) |
| **D** (`132d198`) | config **v106** | The three no-progress-tax / staller-RNG fixes, all OFF |
| **E** (`4867537`) | — | The **scaffolding gauge** (TB `train/scaffolding_gauge` + offline) and **`python -m main.exploitability`** |

Also shipped this week and dormant: **v104 route-1 PBRS**, the **winprob teacher + confirm** modes,
**defensive search + racing** (`--root-strategy racing`, `--defensive-confirm`), the **`signal/`
TB group** (advantage density × outcome entropy — *the PAIR is the instrument; singly each
misleads*), and the **exploiter ladder curriculum** (`--exploiter-ladder`).

**🔴 The harvest pilot's FAILURE is the finding (`195ce9f`).** Naive head fine-tuning **REGRESSED**,
and only the untouched LONG-WIN control caught it: the labels were excellent **and for the wrong
states** (fit set turns 60–152; 29.3% of eval turns beyond its max). ***A label factory that never
samples the region its meter scores is extrapolating.*** Damage scaled with the fit-set mean offset
across two independent runs, so selection bias is convicted and a hyperparameter is not; hence
`--anchor-coef` defaults ON, because 0.0 was **measured** destructive.

**The exploitability curve's first reading is FLAT** (`4867537`): rev-2 → rev-3 delta mean net
exploitability **+0.0185 [−0.0100, +0.0470] — no detectable change**, consistent with the ceiling
reframe. Per-class headroom is the usable part: **meter teams 0.0856 vs coverage teams 0.2575.**
This is now the standing meter the flywheel must eventually bend.

**In flight:** R3-SELF (the +6pp falsification) · probe P final · probe Q (rev-3's own untaught
pull-down) · the ai_v12 launch triad (adversarial review of the five landings / the launch runbook /
the `q_labels` producer) · the 40-team slate · the global-random sweep.

## The defect genres — five named failure classes

Five distinct ways this codebase has produced confidently-wrong measurements. They are indexed here
because **each one was found the expensive way, and each generalises**: a new probe design should be
read against this list before it runs.

| Genre | What it is | Specimens |
|---|---|---|
| **recorded ≠ effective** | A flag is recorded in `cli_args`, the argv looks right, and the code path never fires. **Five specimens** — this is the house's most common defect | td_aux provenance · the pinned-key near-miss · `--distill-team-bias` gated on `_distill_pairs`, empty at coef 0, so the capstone CONTROL differed in two variables (`d4d551d`, fixed `6ff4c04`) · `exploiter_bot_fraction` INERT without `--exploiter-keep-bots` — **this one invalidated a kill** (`92ad277`) · the admission harness's team dict disagreeing with recorded `--trainee-teams` at 16/36 cells, which would have ordered a full relaunch on fabricated grounds (`61608ac`) |
| **composition drift** | Every piece individually correct; the ENSEMBLE's meaning inverted by sibling deletion. **Invisible to every per-piece test by construction** | The `no_progress_tax` switch toll, designed against +0.35 of counterweights that `928a00b` deleted, flipping the net sign to −0.15 (`cfbc9bf`) |
| **silent-inert** | A mechanism that "ran" leaves no artifact, so nobody can tell whether it engaged | The standing defense is **verifiability by construction** — a state artifact written by the mechanism itself. `exploiter_temp_state.json` is what let probe C *prove* v8's ratchet ran; `exploiter_ladder_state.json` and `ladder_state.json` were built to that pattern deliberately (`d883601`) |
| **global-random coupling** | Two arms share the process-wide `random` module, so one arm's draws move the other's | `Gen3StallerPlayer`/`V2` flip Protect on module-level `random` — cross-arm coupling in *every* paired-eval design, found by the transfer cell's own falsifier (`deb0bc9`). Unbiased there (3A/1B); fix landed OFF at v106; the genre-wide audit is dispatched |
| **untested INTERSECTIONS** | Each flag tested alone; the COMBINATION never | The `value_from_dist` lesson. Named as the primary hunting ground for the ai_v12 adversarial review: clean-world × value_from_dist × compile × distill (`c06e386`) |

Two method lessons of the same weight, from the same week:

- **Allocator noise (`d2a0212`)** — ~1 in 7 historical "searched" battery decisions was the
  allocator, not the search. A standing caveat on every pre-2026-08-29 search-dividend cell.
- **Extrapolating factories (`195ce9f`)** — see the harvest pilot above. The generalisation: *a
  probe's fit region and its meter's region are two different sets, and nobody checks the second.*

## Maintenance protocol (what the memory enforces)

After **any** investigation that changes our belief, update this folder in the same pass:
1. **Update the lever file** (or create it from `_template.md` when a frontier item becomes active):
   move facts into **Known**, open questions into **Not-known**, the upside into **Pros**, the caveats
   into **Cons**. Be honest — a confirmed Con (e.g. "pervasive in wins too") is as valuable as a Pro.
2. **Update the ledger row** — status, the load-bearing number, the re-verify command.
3. **Tend the frontier** — add any new candidate lever surfaced; mark one investigated/ruled-out.
4. **Apply the honesty gates** before promoting a finding to Known (see ledger.md → method): is it
   outcome-conditioning? falsifier-myopia? legitimate-in-context? exploration vs learned? Always
   adversarially verify a *confirming* measurement (we were overturned 3-for-3 by careful rechecks).
5. **A measurement cited by a committed doc must itself be committed in the same pass.** Learned
   2026-08-17: the gen-13.5 evidence base — including the dV readings that LICENSED a shipped,
   irreversible deletion — sat uncommitted in a worktree while `designs/CLAUDE.md` already cited
   it; one `git clean` from gone. A citation to an uncommitted file is a dangling pointer wearing
   provenance clothes.
6. **Read the new arm against [the defect genres](#the-defect-genres--five-named-failure-classes)
   before it runs, and verify EFFECT rather than argv.** The recorded≠effective genre alone has five
   specimens — arms that looked correctly configured and were not. The cheap prophylactics: confirm the flag moved a
   telemetry quantity, make the mechanism write its own state artifact, and test the INTERSECTION
   you are about to run rather than each flag alone.

## The amortizability gate — route every oracle finding through L1–L4

The falsifier is an **ORACLE** (hindsight + re-rolls), so *"a better action existed"* is its DEFAULT
output and *"needs search/MCTS"* is the degenerate conclusion it is **built to reach** — concluding it
proves only that the oracle is an oracle. A finding is a **lever** only if the improvement is
**AMORTIZABLE** into the single-forward-pass network. Route every oracle-found improvement through:

- **L1 obs-info** — the deciding info was ABSENT from the obs but is known/addable → **obs feature** (cheap).
- **L2 representation** — info WAS in the obs; the net mis-used / mis-valued it → **arch / training /
  value-target**. *Test:* linear-probe the trunk (`probe`); recoverable ⇒ L2 (head fault), not ⇒ L1.
- **L3 anticipation** — needed the opponent's TYPICAL reply → **opponent-action / forward-model aux
  head** (amortized anticipation, NOT runtime search). *Test:* was the opp move predictable from
  history/priors, or idiosyncratic/hidden?
- **L4 deliberation** — genuinely needs per-state multi-ply AND the opp move is not anticipatable → the
  ONLY "search" bucket, and it ships as an **OFFLINE TEACHER distilled into the net** (AlphaZero /
  Expert Iteration), never as runtime compute.

**HARD CONSTRAINT (owner):** no lever may put search/MCTS **on the model** (inference or the training
loop). Search is a *teacher / offline diagnostic* only. *"Cheap vein mined out → MCTS"* is not a default
— it is a falsifiable claim about the SIZE of L4 that must beat the measured L1+L2+L3 mass.

## Decision posture — the build bar (say YES)

Falsification is a **means** (don't waste a retrain on a dud), not the goal (a better model). The
**honesty gates govern what we CLAIM TO KNOW; they are NOT the bar for what we BUILD.** The census says
there is no single remaining big lever (confirmed blunders explain a few points; the rest is diffuse
grind) → the EV-optimal strategy SHIFTS from *explore* (hunt an elephant) to **EXPLOIT** (build the
portfolio of moderate, amortizable gains and measure the aggregate). Progress here is a long tail of
small, compounding, stacked wins — not one breakthrough.

A lever is **GO-TO-BUILD** when ALL hold (this bar is deliberately LOWER than "Known"):
1. **Amortizable** — passes the L1–L4 gate (a feedforward change, or an offline teacher, can capture it).
2. **Positive EV** — a plausible mechanism + non-trivial headroom (≥~1% wr or a clear behavioural fix);
   it need NOT be proven dominant.
3. **Falsifiable-after-build** — we name the metric that must move in the retrain, so a no-op is cheap
   to detect and abandon.
4. **Bounded cost** — obs feature / reward term / aux head, not a moonshot.

**Portfolio rule:** because no single lever fixes the gap, **STACK** the GO-TO-BUILD levers into the
next FRESH run (the resume-immutable boundary several already require) and measure the aggregate, rather
than A/B-ing each forever. *Honest tradeoff:* stacking **confounds attribution**. Mitigate by stacking
only mechanistically-INDEPENDENT, cheap levers, each with its own offline pre-retrain proxy; isolate a
lever only when it is expensive or risky. The live GO-TO-BUILD queue is in [ledger.md](ledger.md).

## The frontier — what else might be there

Ranked by where the *unexplained loss mass* most plausibly lives, with the honest status. The census
finding to keep front-of-mind: **the confirmed blunders (self-KO, attack-mismatch) explain only a few
points of the ~18% bot-loss gap. The dominant remaining mass is NOT more blunders.**

**The five candidates that became live programmes this week are listed first**; the standing census
rows follow unchanged except where a measurement moved them.

| Candidate lever | Why it might be there | Status |
|---|---|---|
| **Coverage / teams-per-teacher breadth (the anti-treadmill lever)** | The flywheel's transfer is REAL (DiD ≥+4.0pp) and LOCAL (+6pp where headroom exists, z=2.57, replicated), but each narrow fold ROBS the untaught (−5.9pp) and the next revolution repairs it — the treadmill. At 9–12/719 coverage the loop cannot move general strength by arithmetic. Seven accounts are dead (budget, dose, curriculum, breadth-for-differentiation, rigidity, tails, ecology); breadth-for-TRANSFER is the last one standing and the axis v8 differs on most. | 🟡 **ACTIVE — the era's main line.** rev-4's 3×8-vs-6×2 shape discriminator, then the 40-team revolution with teams picked BY HEADROOM → [levers/coverage_and_breadth.md](levers/coverage_and_breadth.md) · `ade78c1` `61608ac` `91d5125` |
| **Search as a TEACHER (routes 2/3, E5)** | Per-decision search gains are REAL (+4.7pp, opponent-marginalized) and **do not compose in play** (τ = 0.17, excludes 1.0). So the value is not at inference — it is in the TARGETS search manufactures: confirmed overrules distilled back, then amortized into a per-action Q-win-prob readout. | 🟡 **ACTIVE, BUILT, dormant.** ai_v12 routes 2+3 landed; the Q head is v107 `read_only`; the `q_labels` producer is the named gap → [levers/search_as_teacher.md](levers/search_as_teacher.md) · `deb0bc9` `f5c8a77` `5edbd05` |
| **Win-prob head empowerment (labels · uncertainty · harvest)** | The head KNOWS 96.4% of whiffs at decision time and is ignored (policy samples its preferred action at p=0.002). It is the leaf every search reads, the potential every clean-world arm would use, and the auditor that convicted the switch tax. It is also the binding constraint in three programmes at once. | 🟡 **ACTIVE.** Harvest/factory landed; the pilot regressed and named its guard rails → [levers/win_prob_head_empowerment.md](levers/win_prob_head_empowerment.md) · `bda8382` `195ce9f` `9cb825c` |
| **The CLEAN WORLD (terminal ±1 + PBRS-φ only)** | Every hand-tuned reward term is a hypothesis nobody re-derives; the switch tax proves the class. Three arms (sparse / self-φ / frozen-φ) at draw = −1 price the whole scaffolding era, and every pairwise difference is a named quantity. | 🟡 **REGISTERED, flag-reachable, not launched.** Gated on the 5M sparse pre-test + rev-3/rev-4 obligations → [levers/clean_world_reward.md](levers/clean_world_reward.md) · `e22bd08` `627ab58` `4d22ae4` |
| **The `no_progress_tax` switch tax → the under-switching pathology** | 73% of voluntary switches are charged vs 6.7% of moves; the implied −0.101 reward/decision runs AGAINST switching while the head rates switching +0.0042 BETTER. Composition drift: the toll was designed against +0.35 of counterweights that were later deleted. **This is a candidate CAUSE of the 16%-vs-30% under-switching gap.** | 🔬 **OPEN — causal arm costs NOTHING** (`--no-progress-penalty 0.0`); fixes landed OFF at v106 → [levers/no_progress_switch_tax.md](levers/no_progress_switch_tax.md) · `1d5a866` `cfbc9bf` |
| **Stall-tail head over-confidence (the 0.999 tails)** | The clock fix HELD (81.2% → 22.2% over seven generations) but **34.8% of cap tails still end φ ≥ 0.5 on games that lose by construction** — 4.3× the ordinary-loss rate. The clean world leans on the head hardest at its historically weakest point. | 🔬 **OPEN, sequenced: harvest → reducibility probe → architecture only if the diet fails** → [levers/stall_tail_overconfidence.md](levers/stall_tail_overconfidence.md) · `32c39df` `b63a96f` |
| **General multiplicative structure (GLU · MI junctions · extremal pooling)** | The 0.999 tails and the "lost my last Roar mon" family are the same object: a rare VETO that a sum-through-a-squash builds only as a slope. The deficiency is sample ECONOMICS, not expressiveness. | 🔲 **REGISTERED, deliberately UNBUILT** — owner ruling: no hand-picked aggregates; every rung must be earned by a measured gap, and the data leg (harvest) runs first → [levers/multiplicative_structure.md](levers/multiplicative_structure.md) · `07e9a54` `00c5a11` |
| **Strong-opponent positional grind** | The census's biggest finding: ext/pool losses are LOW-variance NEUTRAL grinds (dice std ~0.01), not blunders or surprise-OHKOs. We have **no mechanism** for this mass yet — it's the least-understood and likely largest lever. | 🔲 **UNEXPLORED** (the priority) |
| **The EARLIER decision (multi-ply)** | Surprise-deaths + recovery/setup "blunders" were repeatedly found to be *committed by the lethal turn* — the real mistake is 2–3 turns upstream. We've only ever anchored on the death/crater turn. | 🔲 still unexplored **as an obs/representation question** — but its *search* form is now measured and negative: fixing individual upstream decisions at inference buys +4.7pp per decision and **τ = 0.17 of it survives to the game** (`deb0bc9`). The compounding failure is the finding; anchoring falsify earlier remains untried |
| **Forward-model / opponent-action head** | Predict opp action (attack/switch/which-move). | ❌ **FALSIFIED 2 ways (2026-06-12)** — *and note the successor that DID ship works differently: the production α/β opponent-intent heads are consumed by the damage op's Σα·f reductions and by search α-pruning, not read as a prediction. Probe L found α flags **the pivot** correctly (+0.209 vs no-pivot) and is null on **the whiff** — a correct division of labor, with the whiff knowledge living in the win-prob head (`bda8382`).* — (a) VoI oracle: perfectly knowing the opp action is worth ~0.03 mon, loss≈win ([[project_l3_oracle_grind_l4]]); (b) representation probe: the trunk+POLICY-HEAD already model the opponent — opp_switches 0.89/0.90, move-TYPE 0.93/0.96, switch-TARGET 0.88, big-hit 0.75 — so an aux head re-encodes present info & can't fix a decision problem. DON'T build. `project_opp_action_head_falsified` |
| **Surprise-OHKO belief coverage (H3)** | Belief under-fires on 52% of lethal healthy deaths. Recoverability now RESOLVED: **42% avoidable** (64% switch-to-a-wall), 33% LUCK (belief was right), 25% committed. | 🔬 **GO (gated)** — see [levers/surprise_ohko_coverage.md](levers/surprise_ohko_coverage.md); pair with under-switching |
| **Opponent-PP observability (the stall-class blind spot, C3)** | The critic is over-confident precisely on stall losses (win_prob 0.7–0.98 vs resampled-dice 0.0–0.4) while LEVEL-calibrated elsewhere (C2) — and the opponent's PP ledger, the quantity that decides a Gen-3 stall war, is encoded as ALWAYS FULL in the obs (`moves.py:129-130`; no usage tracker exists). The win-prob head is MC-supervised, so the miss is missing INPUT or distribution, not objective. | ❌ **KILLED 2026-08-17** — pre-registered probe (`gen13_endofrun_runbook.md` §8) ran same day, zero deviations: primary ΔAUC **−0.0026** CI [−0.018, +0.010], p=0.12; confident-slice PP-only AUC 0.595 < the 0.65 bar. The MC-supervision argument survives, but the successor it handed to — the **training DISTRIBUTION of stall games** — is **ALSO KILLED 2026-08-17** (`measurements/gen13_stall_coverage.json`): stall trajectories are 3.0% of training decisions vs 0.21% of matched sentinel eval losses, a ~14x OVER-exposure, opposite in sign to the hypothesis. **Both mechanisms are now dead while the phenomenon stands** — revive only with a NEW, pre-registered mechanism → [levers/opp_pp_observability.md](levers/opp_pp_observability.md). **UPDATE 2026-08-29 (`32c39df`): the phenomenon has been re-measured and it SHRANK and MOVED.** The deadline clock cut positive-V-at-final-decision on cap losses **81.2% → 22.2%** across seven generations, and what remains is a *conditional* residual — 34.8% of cap tails ending φ ≥ 0.5, concentrated where the position prior is strongest. That residual has its own lever and its own (data-first) sequencing → [levers/stall_tail_overconfidence.md](levers/stall_tail_overconfidence.md). Note the 14× stall OVER-exposure cited in this row covers heal-war states, **which the head reads RIGHT** — it is not the failing joint |
| **Hidden-team belief (the un-falsified gap)** | The opp-ACTION head was FALSIFIED, but it named a SEPARATE real gap: the opponent's HIDDEN team. Gen3 has no team preview → the ~3 unrevealed slots are ABSENT from the obs (a probe CAN'T recover them). Pre-build learnability: revealed→hidden beats the usage prior **+7pp recall / +8–10pp top-1**. Subsumes the bench/switch-in half of H3. | 🛠 **SHIPPED — and the "NOT RUN" status this row carried until 2026-08-30 was STALE.** `opp_belief_aux_coef 0.05` / `opp_belief_cls_k 6` / `opp_belief_slots` are all **ACTIVE** in the production flag table (`designs/ARCHITECTURE.md`), and the `HiddenOppBeliefPool` (768 dims, POLICY side only — the vf half read dV 0.0000 and was deleted) is in the live phase chain. **What is still UNMEASURED is the original question: does it HELP the policy?** No A/B has isolated it → [levers/hidden_team_belief.md](levers/hidden_team_belief.md). `project_hidden_team_belief_built` |
| **Under-switching = policy COMMITMENT lever** | Anchored to strong humans (faithful human-agreement probe): humans switch ~30% of voluntary decisions, our argmax ~16%. **But REFRAMED 2026-06-12: NOT representation/valuation — the policy's SOFT switch-prob (0.28) already ≈ human (0.30); the gap is ARGMAX/commitment** (it assigns ~human switch mass but under-commits). So the knob is argmax sharpness — `--switch-bias-weight` / `ent-coef` / temperature — no arch build. ("info-blind switching" was REFUTED: the corr was a progress-confounded null; double-switch 57% proves info-conditioning.) | 🚨 **A CANDIDATE CAUSE FOUND 2026-08-29 — this row's "no arch build, tune the sharpness knob" framing is superseded.** Probe M measured the one live BIAS term paying **−0.101 reward/decision against switching** (73% of voluntary switches charged vs 6.7% of moves) while the win-prob head rates switching **+0.0042 BETTER** — opposite signs, for a whole generation (`1d5a866`). Probe N traced it to composition drift (`cfbc9bf`). **The causal arm is free (`--no-progress-penalty 0.0`) and switch rate is its registered endpoint** — run that before any commitment/sharpness tuning → [levers/no_progress_switch_tax.md](levers/no_progress_switch_tax.md). `project_opp_action_head_falsified` |
| **Attack type-mismatch obs feature (H2)** | Confident resisted/immune picks; small (~fraction of a %). Cheap obs effectiveness feature. | ✅ confirmed small — [levers/attack_type_mismatch.md](levers/attack_type_mismatch.md) |
| **Team / matchup draw quality** | Some losses may be bad team draws, not policy errors. The team-pool weighting (yak_attack) was a data-dist bug. | 🟡 partially addressed |
| **Decision-time search / MCTS** | Highest ceiling (Wang2024, rank-8 Elo). The amortized levers (forward-model) are the search-free cousins. | ⛔ user RULED OUT (kept as the ceiling-setter) — **and now measured, which makes the constraint cheap to honor.** The best search we can build (defensive paired, `4cf81fd`) got search to **stop losing** (0.4937 vs honest_1s 0.2929) and no further: 13× more evidence-certified overrules moved the win rate onto the null **exactly** (`35dbc3c`), per-decision gains are real but τ = 0.17 survives to the game (`deb0bc9`). *Search as a PLAYER dead-ends at these checkpoints; search as a TEACHER is the surviving road.* |
| **Human ladder data (ai_v11 chapter / treadmill break)** | Self-play only explores its own convex hull; **263,159** ladder replays (2.8 GB, 2026-05-18→2026-08-02 — the old "~102k" is stale; collection appears STOPPED at 2026-08-02) are an EXTERNAL distribution — a candidate mechanism for the positional-grind lever + the "self-play treadmill" ceiling. **FAITHFUL human-agreement probe BUILT 2026-06-12** (`src/agents/bc/log_reader.py` turn-1 full-team injection makes the acting side's obs/mask faithful; `src/main/human_agreement.py`): overall ≈35% match vs strong humans (≈1-in-3; low ≠ bad — two strong humans also disagree; agreement = behavioural DISTANCE). **Headline finding: the policy UNDER-SWITCHES ~2× vs strong humans (16% vs 30%)** — this OVERTURNED v0's "modest 16v19" (an artifact of 43% of human switches being unrepresentable pre-fidelity-fix). The faithful `log_reader` is the reusable FOUNDATION for the non-search roadmap (BC / team-completion / offline eval). **The human half of that headline now reproduces MODEL-FREE: the corpus's own switch share is 28.96% over 30,146 reconstructed ≥1500 decisions** (`measurements/human_replay_faithfulness_census_1500.json`, 2026-08-18). **FAITHFULNESS CENSUS RUN 2026-08-18** — the reconstruction is far more partial than "a spectator log under-represents move alternatives" conveyed: fully-faithful decisions (6/6 own bench AND 4/4 moves on the acting mon) are **16.70%**; own **item is known on 3.93%** of own mons; only **8.64%** of own mons reach 4/4 moves; **3.15% of sides fail to parse** (`UnknownVolatileError`, and `human_agreement.py` swallows them silently); own **spread is FABRICATED and flagged `spread_known=1`** (all-31 IVs / 0 EVs / neutral nature — a wrong value asserted as known, feeding the `d1`/`d2` outgoing physics); and the faithful stratum is **loss-enriched 1.29×**, so any outcome-labelled objective needs outcome-balanced weights. | 🔶 **PUNTED (owner, 2026-08-18) — drafted, censused, NOT implementation-ready.** The draft is [`designs/ai_v11/design_human_replay_objectives.md`](../ai_v11/design_human_replay_objectives.md) (four-rung ladder ordered by OOD-robustness, gates pre-registered, faithfulness census committed) — the partial-information nuance is exactly why it needs refinement before any rung is built. NOT a gen-17 commitment; re-open by owner decision only. The replay COLLECTOR was restarted 2026-08-18 (proxied, tmux window) so the corpus grows meanwhile. **The COMMITMENT lever is NOT unchanged as of 2026-08-29** — this row's own headline number (16% vs 30%) now has a candidate CAUSE that lives in the REWARD, not the network: the `no_progress_tax` pays ≈−0.101/decision against switching (`1d5a866`, `cfbc9bf`). See the under-switching row. The opp-action head remains FALSIFIED and is still not a path here. Memory `project_human_agreement_probe` |

| **De-amortization: per-team conditioning** | The generalist plays every team with one averaged policy; a specialist is far stronger (0.438 → 0.72 piloting). The exploiter→distill loop CLOSES it (**D1**) and the skill STICKS without teachers (**D2**, ~76%). | ✅ **RESOLVED 2026-07-28 (D4)**: the scaling limit is **COUNT (+0.077 SIG)**, not conditioning (+0.028, marginal, and mechanistically a SHARED modulation — LoRA/MoE ruled out). ⚠️ **Two 2026-08-29/30 amendments.** (1) The *teacher* side has a **CEILING at ~0.6881, invariant to budget and to target strength** — extraction is headroom, not a teacher property (`61608ac`). (2) N=2 sat BELOW the tested range, so "N ≤ 10" does not by itself pick the shape; **5×8 = 40 satisfies both the N≤10 bound and v8's structure**, and rev-4 adjudicates. The **team-dial bar stands re-affirmed** (`c0619a3`): a gate/dial ships only after the gated quantity is shown to PREDICT performance — the exact bar the LUT arm failed. `designs/ai_v8/design_conditioning_ceiling_arms.md` |

| **Measurement/infra correctness** (are our experiments running the model we think they are?) | M1 found that SB3 silently destroyed every zero-init in the extractor, so no "identity-at-init" toggle ever actually started at identity — a confound that sat under K10 and the whole D4 conditioning line, undetected, because every test builds the module directly instead of through a policy. | 🔴 **NO LONGER "one found" — this is the highest-yield row in the table.** M1 was the first of what are now **five named defect genres**, and **five fresh specimens landed in the week of 2026-08-26 alone**; one of them *invalidated a kill* (`92ad277`) and one would have ordered a full relaunch on fabricated grounds (`61608ac`). Full index + the specimens: [the defect genres](#the-defect-genres--five-named-failure-classes). **The generalised lesson stands and has widened:** an invariant asserted only where the production path is bypassed is not an invariant — and a *recorded* flag is not an *effective* one. The standing defenses are (a) verify effect from telemetry, never from the argv, and (b) make the mechanism write its own artifact. **A SIXTH genre landed 2026-08-31 — *the metric's model of the data is wrong* (`measurements/obs_conditioning_2026-08-31.md`):** the observation's participation ratio was read as an input-richness collapse (37.76 gen-12 → 16.20 gen-14), and a covariance PR treats every column as a MAGNITUDE — but **433 of 2023 live obs columns are raw dex numbers cast with `.long()` into `nn.Embedding`, carrying 99.993% of the raw variance**, so the statistic was measuring Showdown's numbering. On the 1,590 columns the network reads as numbers the PR is flat (45.14 → 44.62 → 45.70) and the whole dynamic range is 5.6×. Unlike the other five this one is not about a flag or a control — the code did exactly what it said; the *estimator's* assumption about its input was false. Defense: **before quoting a magnitude statistic, check that every column is consumed as a magnitude** (the fourth rule in [capacity_battery.md](capacity_battery.md)). |

**The honest steer (rewritten 2026-08-30).** The de-amortization vein is still the active one, but
the week reshaped *which knob it turns*: seven accounts of v8's +69 died, the teacher CEILING
dissolved the budget law, and what survived is **coverage / breadth-per-teacher**, promoted from "an adder" to
**the anti-treadmill lever** by the discovery that each narrow fold robs the untaught. Beside it, two
genuinely new veins opened, both from instruments rather than hypotheses: **the win-prob head is a
better judge than anything that consumes it** (96.4% whiff knowledge, ignored at p=0.002; it also
audited and convicted a live reward term), and **the reward stream contains hypotheses nobody
re-derives** — which is what the clean world exists to price. The blunder-hunting vein remains mined
out, and the **strong-opponent positional grind** is still the largest row with no mechanism, though
the treadmill/coverage account now claims part of what used to look like an unexplained flat curve.

*One caution earned this week:* three of the four biggest belief changes came from **auditing a
claim that was load-bearing enough that nobody had checked it** — the budget law, v8's tails, the
"shaping was necessary" famine. That is now a standing move, not a coincidence.

**Programme sequencing (2026-08-30, `fdd2934` + `ade78c1`).** The 2026-08-17 substrate-before-
flywheel decision is DISCHARGED — the v93–v95 substrate shipped and gen-16 ran it; its un-spent
half is the exploiter A/B gates in [substrate_exploiter_gates.md](substrate_exploiter_gates.md).
The live sequencing is three workstreams:

1. **Capstone completion (GPU, critical path).** R3-SELF (the +6pp falsification) → probe P final +
   probe Q → **rev-4's 3×8-vs-6×2 shape discriminator** → the 40-team revolution in the winning
   shape, teams selected by headroom. The wheel-turns-twice replication is a standing requirement,
   not a nicety.
2. **Builds (CPU, parallel).** DONE 5/5 this week (waves A–E). Remaining: the `q_labels` producer,
   the launch runbook + adversarial review of the five landings, and the post-verdict enablement of
   the v106 reward fixes. **The gated-readout / SwiGLU rungs are deliberately NOT built** — they are
   behind the harvest-then-reducibility measurement by owner ruling.
3. **Era opening (ai_v12).** The 5M sparse-vs-shaped pre-test sizes it; then the clean-world three
   arms + the re-sized route-1 ladder + the zero-code tax-off arm, against endpoints registered in
   advance: stall rate, switch rate, the scaffolding gauge, the whiff census.

**The standing gates on all of it:** no lever ships without passing the amortizability gate above;
no claim is promoted to Known without its honesty gates; and **no arm is registered without its
readings written down before the data exists** — the discipline that let four separate accounts be
killed this week by their own pre-registered predictions rather than by argument.
