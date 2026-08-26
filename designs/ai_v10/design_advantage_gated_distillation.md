# design — ADVANTAGE-GATED DISTILLATION: making the distill gradient and PPO's gradient agree by construction

> **[STATE 2026-08-25 night] FORWARD DESIGN — nothing built, nothing scheduled.** Written
> overnight for morning adjudication, ordered by the **ARM E** ledger entry (`d87393d`, §"Two roads
> forward", road 2). It is the **deep-branch** response to the fold's failure: five +3M arms plus
> tick-1 have eliminated every cheaper explanation, and every one of them manipulated *where* or
> *how hard* the distillation KL applies. **None manipulated what it asks for.** This document
> designs that change.
>
> **Arm F is running while this is written** (pure-distill phase vs simultaneity). §5 states, in
> advance, how each F outcome edits this design — so the design is readable the moment F relays,
> without a rewrite.
>
> Companions: [`design_flywheel_tick_tock.md`](design_flywheel_tick_tock.md) (the operational loop
> whose **D-F** this document formally contests), [`design_counterfactual_value_grounding.md`](design_counterfactual_value_grounding.md)
> (the label-factory pattern rung (b) instantiates), [`../research_state/critic_calibration_plan.md`](../research_state/critic_calibration_plan.md)
> (the era's binding constraint; §3.1's judge reads the very estimate that plan is fixing — §7.4
> states how the two compose).

---

## 0. The object, in one paragraph

The current distillation loss asks the student to match a frozen specialist's **full action
distribution** on every state where the student pilots that specialist's pinned team. Five
code-matched arms have established that optimizing this objective *successfully* collapses the
shared trunk's representation rank and damages play **globally** — including on states the KL
never touches — even for a single teacher independently verified as +11.6pp stronger. The working
mechanism is **representational incompatibility expressed as gradient interference**: the on-pin
KL drags the shared trunk toward a globally-different function while PPO's own gradient pulls
elsewhere, and the trunk's compromise is to lose dimensions.

This design changes **what the KL asks for**, so that the distillation gradient points only where
PPO's own gradient already points. Two axes, and keeping them apart is the whole structure:

| axis | question | rungs |
|---|---|---|
| **TARGET FORM** | *what* does the term ask the student to become? | **(c)** action-level CE instead of full-distribution KL |
| **JUDGE** | *who certifies the teacher is right here?* | **(a)** the student's own advantage (free) · **(b)** paired rollouts (costly, correct) |
| *(neither)* | *how hard / how far?* | **(d)** trust region — a magnitude control, and magnitude is **measured dead** |

---

## 1. The evidence base — the five-arm negative space

Everything in this section is measured, code-matched, and in `../research_state/ledger.md`. It is
reproduced here because a design that re-derives a killed hypothesis wastes a generation.

### 1.1 What is ESTABLISHED

All arms are **+3M continuations forked from `rev-1 final`**, identical but for the named change.
Primary meter: per-team piloting on 9 pinned teams, paired.

| arm | change | piloting Δ | retention | `rank/policy_pr` (pi_features) |
|---|---|---|---|---|
| **tick-1** | the original fold (coef 1.0, ecology ON, 3 teachers) | pooled **−4.0pp**; ladder **−97.8** CI [−139.9, −55.7] | **−47%** | 19.6 → **12.4** |
| **fdA** | coef **0.3**, ecology ON | **−5.5pp** z=−4.85 | — | **12.50** |
| **fdB** | coef 1.0, ecology **OFF** (loss channel isolated) | **−7.9pp** z=−7.05 | **−87%** | **12.50** |
| **fdC** | coef **0.0**, ecology ON (pure ecology control) | **−1.2pp** n.s., spans zero on all 9 cells | — | **21.87** — *above* rev-1 on every capacity tap |
| **fdE** | coef 1.0, **ONE** teacher (tock-1c, +11.6pp verified), its own 2 teams, hard-gated | **−7.2pp** z=−6.41 | **−80%** | **13.57** |

Read off that table:

1. **The loss channel is convicted; the ecology is exonerated.** fdC's null is flat across all nine
   cells — teachers-as-opponents is safe, so double-sided defence stays available.
2. **The coefficient is not a lever.** 12.50 at 0.3 and 12.50 at 1.0 — identical to two decimals
   from *separate arms*. A tunable dose would grade; this **switches**.
3. **The state-gate is not a lever.** fdE is maximally gated (one teacher, its own two teams, hard
   gate) and lands statistically on fdB.
4. **Teacher count is not a lever.** Multi-teacher averaging exonerated by fdE.
5. **`pi_features` is BINARY: 21.87 with no KL, 12.5–13.6 with ANY KL.**
6. **The localization claim is dead.** IN-gate and OUT-gate states are **both** damaged in every
   arm (every figure negative, 5/6 significant). The "worse OUT" directional reading from a
   preliminary n=100 cell did not survive full n and is recorded as unsupported in either
   direction.
7. **The optimization worked.** tick-1: KL 0.098→0.033, agreement →0.92. fdE matched its teacher
   best of any arm (agree **0.938**, KL fell 3.6×) — on a verified +11.6pp teacher — and lost 8.8pp
   on those very teams. **Successful optimization of the objective caused the damage.**
8. **The teachers were good.** All three tocks pass the admission gate net of seniority
   (a +8.3 z=3.35 · b +8.8 z=3.63 · c +11.6 z=4.84; seniority ≈0; 9/9 teams positive). This is
   **negative transfer**, not failed transfer.

### 1.2 The two mechanism probes

**Sharpness (`978b1aa`) — the narrowness theory REFUTED, 8/8 cells reversed.** Teacher entropy is
slightly *higher* than the base on identical paired states (on-pin Δ **+0.04..+0.08 nats**, top-1
*lower*; neither model near degenerate, 0.34–0.47 of uniform). **Do not carry "the teachers are
over-sharp scripts" into any part of this design.** What survives: KL(teacher‖base) ≈ **0.43–0.48
nats on-pin and 0.36–0.41 off-pin** — 3M steps moved the teachers to a **globally different**
policy, barely slice-specific.

**Telemetry triage (`88091ca`) — the interference signature FIRES.** Pooled Δ half-batch trunk-gradient
cosine **−0.030, p=0.001** in the distill arms vs control: the KL-shaped trunk yields a measurably
more *internally inconsistent* plain-PPO gradient. And the effect is **binary in coefficient
(0.3 ≈ 1.0)** — matching the rank switch exactly, from an independent instrument.

Composed: **an on-pin KL toward a globally-different (not narrower) teacher policy fights PPO's own
gradient in the shared trunk; the conflict — not the magnitude — does the damage.**

### 1.3 The frame this design is built on — OBJECTIVE AGREEMENT

The v8 tension (`ai_v8_14`: same recipe, same on-pin gating, same bias 0.4, same coef 1.0 →
**+69 anchored ELO**, piloting 0.438→0.710) rules out raw divergence as the poison: the v8 teachers
were *more* diverged (9–20M post-fork vs our 3M) and folded fine.

The surviving frame: **a fold pays when the teacher's difference lies in directions PPO's own
experience corroborates**, and collapses when the difference is idiosyncratic style PPO keeps
contradicting. v8's teachers were better vs varied play — both objectives pulled the same way.
Ours encode the quirks of beating one frozen opponent — PPO contradicts them everywhere else.

**This document's single design requirement follows directly: the distill gradient must point only
where PPO's gradient already points, by construction rather than by hope.**

### 1.4 What is NOT established

- **WHY**, mechanically, at the level of weights. §1.2 is a signature, not a proof.
- Whether the damage is **recoverable** (arm F's second half speaks to this).
- Whether **fold tolerance is trained.** One speculative lead, flagged as such: v8's student was
  itself a product of prior folds (`v8_04 = distill_4teacher`), and `rev-1` is a never-folded fresh
  trunk. §6.4 makes this the kill-branch's successor question.
- The **teacher prescription** (v8's literal shape: ~9M+ budgets on ~10-team slices) remains a live
  and *orthogonal* lever. This document assumes today's teachers and changes the loss; the two are
  composable and should not be confounded in one arm. **Lineage refinement (2026-08-25 evening,
  read from `metadata.json`, not memory):** the tock-1 teachers are `rev-1 + ~3M` forks — *closer*
  cousins by ancestry than v8's ~9–20M forks off a 276M parent — so ancestry is NOT the variable;
  **distributional distance is**, and two mechanisms decouple the two on a young trunk:
  **plasticity** (3M steps on a 25M-step net renovates; 20M on a converged 276M net barely moves)
  and **narrowness** (tock-1 trained on K=4-team slices; v8's teachers covered 23 teams across 3 —
  narrow objectives on plastic nets drift globally, which is what the flatness probe and fdE's
  IN+OUT damage both measured). Tock-2.0 deliberately BUNDLES both (9M × all nine teams); if it
  folds cleanly the separating follow-up is one arm (9M-narrow or 3M-broad), pre-named here so the
  bundle is not read as settling the ingredient. **2026-08-26 update — the sharpness re-run
  answered the separating question without that arm, against the narrowness half:** tock-2.0
  (9M × 9 teams) sits FARTHER from base (KL(T||B) 0.66–0.76 vs the 3M teachers' 0.32–0.50) while
  *broader*-entropy (11/12 cells dH > 0) — if narrowness drove distance, breadth should have
  offset the step effect and did not (steps and breadth moved together, so this is directional,
  not a clean 2×2 — but the direction is decisive for the operative question). **Narrowness is
  dead as the distance driver; PLASTICITY survives alone**: v8's teacher proximity was a property
  of its converged 276M parent, not of teacher breadth, and cannot be reproduced by any teacher
  recipe on a ~25M trunk. The 9M-narrow/3M-broad follow-up is SUPERSEDED. Extraction agrees the
  lever is dead: tock-2.0's net +0.0875 equals tock-1b's 3M row exactly, below tock-1c's +0.1162.
- **Teacher CONTENT quality is not separated by any current gate** (owner question, 2026-08-25:
  is the exploiter's edge transferable skill, or memorized exploitation of one opponent's distance
  from Nash?). The admission gate measures extraction **against the target**, which is
  exploit-shaped by construction — a teacher could pass on pure opponent-memorization. Proposed
  amendment (**OWNER ADJUDICATION ITEM #2**): add a **transfer term** to admission — the teacher's
  edge measured against opponents it never trained on (held-out pool members or the fixed bots);
  an edge that vanishes off-target is Nash-distance memorization and does not fold, however clean
  the channel. This is a CONTENT gate, orthogonal to everything else in this document (which
  repairs the CHANNEL); the v8 +69 being *anchored* (vs fixed bots) is the existing evidence that
  at least some exploit content transfers.

---

## 2. The design requirement, stated precisely

Let `s` be a state, `π_S` the student, `π_T` the teacher, `a` the action the actor sampled, and
`Â(s,a)` the GAE advantage PPO computes for it.

PPO's policy gradient at `s` moves `log π_S(a|s)` in the direction `sign(Â(s,a))`. Today's distill
term moves the whole distribution toward `π_T(·|s)`. On a state where `Â(s,a) > 0` and
`argmax π_T(s) ≠ a`, **the two terms have opposite signs on the same logit.** That is the
interference, written out.

> **The requirement.** Every state on which the distill term fires must be a state where reducing
> `π_S(a|s)` is *also* what PPO wants — and the term's target must be a single action rather than a
> whole distributional shape, so that agreeing on the *direction* does not smuggle in a *function*.

Two consequences that shape the whole ladder:

- **A gate alone is a sixth "where" manipulation.** The five arms said *where* is not a lever. A
  design that only adds a smarter mask has a strong prior of reproducing the null. The gate is
  necessary but is *not* the novel axis.
- **The target form is the axis nothing has ever touched.** Full-distribution KL is the one
  constant across tick-1, fdA, fdB, fdE, *and* the +69 arc. It is the only unexamined variable in
  the whole record.

---

## 3. THE GATING LADDER — cheapest first, each with its judge

Every rung is stated as **(judge, target, cost, what it inherits, what it is blind to)**.

### 3.1 Rung (a) — DISAGREEMENT + TEACHER-ADVANTAGE gating (free, no rollouts)

**Fires on** `s` where **both**:
1. `argmax π_T(·|s) ≠ a` (the sampled action) — *there is something to teach*; and
2. `Â(s,a) < −τ` — *the student's own experience says the action it took was a mistake here*.

**Judge:** the student's own critic, through the GAE advantage already sitting in
`rollout_data.advantages` (`ppo.py:363`), normalized exactly as the PPO term normalizes it
(`ppo.py:365`), so `τ` is in the same units the clip objective uses.

**Why this specific judge and not a Q-comparison.** The brief's phrasing —
*"the teacher's action has higher estimated advantage under the student's own critic"* — needs
`Q_S(s, a_T)`, which we do not have: the critic is a **state**-value head, and the only per-action
value estimate available is a one-ply reroll (rung (b)'s machinery, ~seconds). The free,
*exactly-equivalent-in-sign* substitute is the dual: **gate on the student being demonstrably wrong
rather than on the teacher being demonstrably right.** On a negative-advantage state PPO pushes
probability **away from `a`**, and a CE toward `a_T ≠ a` also pushes probability away from `a`.
The two gradients agree on the decisive logit **by construction**, and they agree *using the same
number*, not two estimates of it.

**The circularity, stated honestly.** This judge reads the critic the era has convicted:
`sd_true_excess` within-decile spread 0.11–0.36, of which ~39% is an irreducible hidden-information
floor (`critic_calibration_plan.md` §0). Three things make it acceptable *as a v1*, and none of
them make it good:

1. **It reads only the SIGN, for the sampled action** — the coarsest and most blur-robust read
   available, and the exact quantity PPO already acts on.
2. **A wrong sign costs nothing NEW.** A mis-signed `Â` mis-teaches PPO and the distill term
   *identically*, in the same direction. The term therefore adds no error axis the run did not
   already have — which is precisely the "objective agreement" property, and it survives a bad
   critic.
3. **The "teacher is better" half does not come from the critic at all.** It comes from the
   **teacher-admission gate** (extraction at 800/arm, now a universal standing rule): a
   population-level, rollout-verified fact. The honest split is *state-level "student wrong" from
   the critic, teacher-level "better on this slice" from measured extraction.*

**What it is blind to:** whether the teacher's alternative is *the* better action, or merely a
different one. It can pull toward a second mistake on any state where both models are wrong.
Rung (b) is the fix, and it is deferred on cost, not on merit.

**Registered risk — selection bias (the PER lesson).** The gated subset is *not* a random subset of
on-pin states: `Â < 0` selects the states where the critic over-valued the action, i.e. the
conviction region / optimizer's-curse states. The R1 read measured that surprise-prioritized
sampling **HURT** (B−A **+0.065**) — priority without importance correction, the classic
prioritized-replay defect. The defect does not transfer *mechanically* (that was a value-regression
target; this is a policy CE with a bounded weight), but the family is the same. **The mitigation is
the experiment itself: G1 vs G2 in §6 IS the with/without-selection contrast.**

### 3.2 Rung (b) — ROLLOUT-JUDGED gating (the correct judge, deferred on arithmetic)

**Fires on** a state whose dispute has been adjudicated **offline** by paired rollouts: play the
teacher's action and the student's action from the same state under **common random numbers**, R
pairs each, to termination; keep the state only if the paired difference clears a significance
gate.

**This is the label-factory pattern applied to distillation, and almost all of it exists.** The
three pieces:

| piece | exists as | note |
|---|---|---|
| the substitution primitive | `utils/bridge/counterfactual.py:337` `replay_counterfactual(record, …, substitute_choice=…, post_t_seed=…)` | `post_t_seed` reseeds at the divergence turn — the CRN knob |
| the paired arbiter + refusal rule | `main/search_dividend/playoff.py:284` `PlayoffRunner.adjudicate` | CRN on **both** axes (sim seed *and* torch seed, `:316-325`); a failed pair contributes **nothing**, never a half-pair (`:326-331`); conclusive iff `n ≥ MIN_PAIRS(4)` **and** `|mean| ≥ 2.0·SE` with an SE floor (`:172-184`) |
| the 3-tier "verified strictly better" gate | `agents/training/teacher/produce.py:83-152` | tier 1 = *search finds A\**; **rung (b) replaces tier 1 with "the teacher proposes A\*"** and keeps tiers 2–3 verbatim |

**And the state source is already there.** `<run>/cf_records/` is the **training-side** tap
(`--cf-records`): the bridge emits a `__RECON__` frame at the end of *every* training episode and
the ring writes it out; `obs_materializer.scan_record` replays a bare record with no `states.npz`
and yields `RecordDecision(index, turn, action, choice, obs, mask)`. **So disputed states from
on-policy TRAINING rollouts can be judged today** — not only offline eval traces. (The prober tier
— `falsify`/`lookahead`/`better_line` — *does* require a `*_reconstruction.json` sibling next to a
`*_summary.json`; the producer tier does not.)

**And the delivery channel is already there, twice over.** The natural one is the
`CorrectionBuffer` (`teacher/buffer.py:47`): a bounded recency ring of
`Correction(obs, action_mask, better_action, advantage, confirmed_value, step_produced, opponent)`
— whose `advantage` field is *already defined* as a **rollout-confirmed win-rate improvement,
never a critic advantage**. A teacher-proposed verdict populates that record exactly. Filled by a
callback from worker shards, sampled per minibatch with its own forward at `ppo.py:850-867`.

**⇒ Rung (b) ⊂ Rung (c) by construction.** The buffer's consumer is
`DistillTerms._searchteacher_loss` — an **advantage-weighted CE toward a single action**. Rung (b)
is therefore not a different loss from rung (c); it is rung (c)'s target form with a *better judge*.
That is the cleanest fact in this document, and it is why the ladder's two axes must be kept apart.

**Two caveats that are load-bearing:**

1. **No opponent identity on a training record.** `cf_producer.py:36-51` (the ECOLOGY DECISION):
   every rollout is played by the current snapshot on both sides at temperature 1.0, stamped
   `opponent: "self_current"`. For a *paired action comparison* this is less damaging than for an
   absolute value label (both arms face the same ecology, so the difference is still a valid
   comparison *under that ecology*) — but it is a comparison in a self-play population, and it must
   be stamped as one. **Preferred source: `teacher/generate.py`**, which plays fresh frozen-trainee
   battles and records full traces with a **known** opponent.
2. **Declared coverage bounds** ride along unchanged: move rounds only; `turn ≥ MIN_LABELABLE_TURN
   = 2` (a *sampler* choice, not a capability limit, since `gen3_search_turn1_open_v1`); records
   ending in a forfeit are not full-replay-anchorable.

**The cost table is §A.** Its conclusion, stated here because it is the design decision: **rung (b)
produces on the order of 10²–10³ verdicts per arm against a 3×10⁶-state on-policy stream (coverage
~0.0x%). It therefore cannot gate an on-policy KL at all** — it can only feed an off-policy buffer
term where each verdict is resampled many times, and it needs a **flywheel-cadence** production
window (hours), not an arm-cadence one. **Rung (b) is v2.**

### 3.3 Rung (c) — ACTION-LEVEL targets (the axis nothing has touched)

Replace the per-state forward KL

```
  Σ_a π_T(a|s) · [ log π_T(a|s) − log π_S(a|s) ]
```

with an **advantage-weighted cross-entropy toward the teacher's chosen action**:

```
  w(s) · CE( π_S(·|s), argmax_a π_T(a|s) )        w(s) = clamp( exp(|Â(s,a)| / β), max=20 )
```

with a **top-K generalization**: distill toward the teacher's top-K probabilities renormalized over
the legal set (`K=1` = pure argmax CE; `K = n_legal` recovers the KL). `K` is the dial between the
two extremes and makes the axis *measurable* rather than binary.

**Why this is the load-bearing rung.** Full-distribution KL is the one constant across every arm in
§1.1 **and** across the +69 arc. It is the only variable in the record that has never moved. And
the mechanism argues for it directly: §1.2 says the teachers differ **globally** (off-pin KL 0.36–0.41
vs on-pin 0.43–0.48) and are **flatter**, not sharper — so what the KL is copying is the teacher's
whole *low-confidence tail shape*, a function of 11 numbers per state that the trunk must jointly
represent with the general policy's. An argmax CE copies **one bit of ordering information** and
leaves the shape to PPO.

**This contests an owner decision, explicitly.** `design_flywheel_tick_tock.md` **D-F**:
*"Distillation is ALWAYS full-distribution: the aux loss targets the teacher's whole policy
distribution (KL), never hard actions — 'dark knowledge is very rich.'"* That decision was made
2026-08-18, **before** the five-arm negative space existed. Dark knowledge *is* rich; the record now
says that in a **shared trunk under simultaneous PPO** it is also the thing being fought over.
**This is adjudication item #1 for the morning.** The top-K dial exists precisely so the owner can
keep some of D-F if he wants it (`K=3` is a defensible middle).

**Cost:** free. One static method, ~25 lines, no new process, no new obs key.

### 3.4 Rung (d) — TRUST-REGION variants (low prior, ranked last)

Clip the pull toward the teacher against a budget on how far the student may move from itself:

```
  L_d = distill_coef · min( KL(π_T‖π_S) , κ )       or      · KL(π_T‖π_S) · 1[ KL(π_S‖π_S,old) < κ ]
```

**Verdict: LOW PRIOR, and the reason is measured.** A trust region is a **magnitude** control, and
magnitude is dead — `pi_features` reads 12.50 at coefficient 0.3 *and* at 1.0, and the interference
cosine is binary in coefficient from an independent instrument. Two independent meters say the
harm does not scale with pull strength, so bounding pull strength should not remove it.

**The one variant that is not a magnitude control**, and is therefore worth keeping on the ladder:
a **per-state** trust region that *disables* the term where matching the teacher would require a
large move — i.e. exactly where the two policies are representationally far apart. That is a
content-conditioned gate rather than a scalar clamp. It is still a "where" manipulation (§2), so it
inherits the five-arm prior. **Keep as the fallback that could rescue D-F if the owner declines
rung (c).**

### 3.5 The ladder, summarized

| rung | axis | judge | cost | verdict |
|---|---|---|---|---|
| **(c)** | TARGET FORM | *n/a* | free (one loss fn) | **SHIP IN v1 — the only untouched axis** |
| **(a)** | JUDGE | student's own `Â` sign | free (already in the minibatch) | **SHIP IN v1 — but see the dose confound, §6.2** |
| **(b)** | JUDGE | paired CRN rollouts, refusal rule | ~10–25 s per usable verdict, 4 workers (§A) | **v2 — flywheel-cadence, off-policy buffer only** |
| **(d)** | magnitude | *n/a* | free | **PARK — magnitude is measured dead; keep only the per-state variant as a D-F rescue** |
| **(e)** | GRADIENT (surgery) | the measured PPO↔distill gradient cosine | one projection per fold step (PCGrad-style) | **PRE-REGISTERED RUNG 3 — fires only on §6.4's KILL (G1 AND G2 both collapse)** |

Rung (e), stated before any result can bias it: if BOTH arms collapse, the conflict is
irreducible at the **loss** level — every axis will have been manipulated, including what the loss
asks for — and the remaining lever is the **optimizer** level: project the distill gradient onto
the plane orthogonal to PPO's whenever their cosine is negative (PCGrad, Yu et al. 2020), i.e.
referee the collision instead of preventing it. It is listed LAST deliberately: it alters PPO's
descent direction (its own failure surface on a touchy optimizer), and it manages a conflict the
cheaper rungs try to eliminate at source. Its judge already runs — the same gradient-cosine
telemetry that convicted the interference (Δcos −0.030, p = 0.001) adjudicates whether projection
cures it, with the rank tripwire as the second meter. Ordering vs the §6.4 fold-tolerance arm:
**(e) runs first** — it is one code change in the fold step and reuses the standard +3M arm shape,
where the fold-tolerance arm needs a pre-conditioning phase plus a paired run.

---

## 4. RANK PROTECTION AS A FIRST-CLASS DESIGN ELEMENT

**No fold runs blind again.** The collapse signature is now known to two decimal places, from five
arms, and the instrument that would have caught it on day one was already running and was **read
five days late** — that is the lesson this section exists to make structural.

### 4.1 The tripwire spec

**Signal.** `rank/policy_pr` — the participation ratio of `pi_features`, from
`rank_metrics.rank_probe` (`ppo.py:1056`). It is sampled **once per `train()`** on the first
minibatch via one `no_grad` forward, is already logged, and is already surfaced in the launcher TUI
(`main/launcher/format.py:115`). It costs **nothing new**; the tripwire is bookkeeping over an
existing scalar. It does **not** require `--capacity-telemetry` (that flag owns the canary and the
half-batch cosine; `rank_probe` runs whenever the trunk is shared).

**Baseline.** `pr_base` = median of `rank/policy_pr` over train() calls `[W_skip, W_skip + W_base)`,
with `W_skip = 5` (skip the resume/compile transient) and `W_base = 20`. Recorded once and logged as
`rank/policy_pr_baseline` on every subsequent call, so any post-hoc reader gets it free.

**Statistic.** `pr_ema` = EMA of `rank/policy_pr`, half-life 10 train() calls. `rank_probe` samples
one minibatch per call and is noisy; the EMA plus the persistence rule below is what makes it a
verdict rather than a flicker.

**Thresholds and response.**

| level | condition | response |
|---|---|---|
| WARN | `pr_ema < 0.90 · pr_base` for 3 consecutive checks | one line to the launcher event stream (`main.launcher.ipc.emit`), `rank/policy_pr_ratio` logged |
| **TRIP** | `pr_ema < 0.80 · pr_base` for 3 consecutive checks | loud event + `rank/tripwire_fired = 1` latched; under `--rank-tripwire abort`, the callback returns `False` (SB3 stops `learn()` cleanly, checkpoint saved) |

**Calibration against the record — this is the sentence that makes the threshold defensible.**
Every KL arm fell **21.87 → 12.5–13.6**, a **38–43%** drop; `fdC` (coef 0.0) rose *above* rev-1 on
every capacity tap. **A 20% threshold fires on all five known-bad arms and on none of the known-good
controls.** The band between 20% and 38% is the honest margin.

**Failure semantics.** `rank_probe` returns `{}` for a non-Gen3 extractor and on any capture
failure. **A missing reading is "no reading", never a trip and never an all-clear** — it is logged
as `rank/tripwire_no_reading` and the persistence counter does not advance. A diagnostic must never
crash a run, and must never *silently* stop speaking either.

**Default.** `--rank-tripwire warn`. `abort` is opt-in per arm. A tripwire that ends a run by
default would be a new way to lose a training window.

### 4.2 The second meter, and the "did the instrument fire" rule

Every arm in §6 runs with **`--capacity-telemetry` ON** so the **half-batch trunk-gradient cosine**
is available — the instrument that carried the interference verdict (Δ −0.030, p=0.001).

**But the telemetry triage found half its own battery silent**: the canary/collapse half either did
not fire or never sampled in the distill arms — an instrument-coverage gap discovered on its first
real case. So the arm-read protocol carries an explicit check:

> **An instrument that did not fire is reported as "did not fire", never as "no problem."** The arm
> read asserts `capacity/canary_samples > 0` and a non-empty cosine series before quoting either as
> evidence. A silent meter is an unmeasured arm.

### 4.3 The term's own liveness

`grad/distill_share` **does not exist today** — the parser says so in as many words (`parser.py:808`:
*"only the search-teacher's `grad/searchteacher_share` and `grad/opd_share` are"*). That gap is
tolerable for a term whose dose is a coefficient and whose row-count is fixed; it is **not**
tolerable for a *gated* term, whose dose is the coefficient **times a fraction that moves during
training**. §6.2 turns this into a hard requirement.

New metrics, all cheap: `distill/gated_frac` · `distill/n_gated` · `distill/gate_agree_rate`
(student argmax == teacher argmax *on gated rows*) · `distill/mean_gate_adv` · and
**`grad/distill_share`** folded into the existing grad-balance probe.

---

## 5. HOW ARM F EDITS THIS DESIGN

Arm F is a **pure-distill phase** (KL only, PPO off) followed by a PPO resume: it separates *"KL
alone corrupts the trunk"* from *"KL × PPO simultaneous conflict corrupts the trunk."* Its outcome
does not change whether this design proceeds — it changes which rungs matter.

### 5.1 F-CLEAN — the trunk survives the KL-only phase and PPO resumes healthily

**Reading:** simultaneity is the culprit. The objective-agreement frame is *confirmed* and gains a
second implementation: **sequencing** (phases) achieves agreement by never letting the two gradients
coexist, rather than by aligning them.

**Edits:**
- Sequencing becomes a **real fix candidate and an implementation option** for v1 (a phase schedule
  in the callback, not a loss change).
- **Rung (a) is downgraded from required to optional.** In a phase there is no PPO gradient to agree
  with, so the advantage gate has nothing to align to.
- **Rung (c) SURVIVES and stays in v1.** F-clean proves PPO can *repair* a globally-different
  function written into a shared trunk; it does not prove the function should have been written
  there. An argmax CE writes less to repair. The cheapest strong arm becomes **phase + action-level
  target**.
- Rung (d) rises slightly (a phase is naturally trust-regioned by its length).
- The **+69 re-explanation** gains a lead: v8's fold may have had more PPO repair budget.

### 5.2 F-COLLAPSES — the KL-only phase alone collapses `pi_features`

**Reading:** simultaneity is **exonerated**; the KL's *content* is everything. This is the outcome
that makes this document's §3.3 the only road.

**Edits:**
- **Sequencing dies as a fix.** Do not spend an arm on phases.
- **Rung (c) becomes the entire v1**, and its priority over rung (a) hardens: a gate cannot help a
  term whose damage does not need PPO to be present.
- **Rung (d) dies with it** — a magnitude/where control cannot rescue a content defect that fires
  with no opposing gradient at all.
- **Rung (a) is demoted to a dose reducer**, honestly labelled as such, and G1 (ungated) becomes the
  scientifically primary arm by an even wider margin.
- The interference framing of §1.2/§1.3 needs amendment: the cosine result stands as *measured*, but
  interference would then be a **consequence** of the incompatible representation rather than its
  cause. **Amend §1.3 in place if this fires; do not narrate.**

### 5.3 F-AMBIGUOUS — collapse during the phase, recovery after the PPO resume

**Reading:** the most informative outcome and the one nobody has pre-registered. It says the damage
is **not a permanent basin** — the trunk's rank is recoverable — and moves the whole question from
"what collapses it" to **"what is the repair rate, and does the +69 arc simply have more repair
budget than a +3M arm?"**

**Edits:**
- Add a **repair meter** to §6: `rank/policy_pr` recovery half-life in train() calls after the term
  is switched off, measured on a tail segment of each arm.
- Both rungs survive; the design gains a third axis (**exposure schedule**) that is currently
  unexplored: fold early, then train clean.
- **This outcome partially rescues D-F** — if the trunk repairs, full-distribution KL may be
  affordable given enough post-fold PPO. That is an owner call, and it is expensive (repair budget
  is measured in millions of steps).

---

## 6. PRE-REGISTERED EVALUATION

### 6.1 The arm template

Standard era template, unchanged: **+3M continuations forked from `rev-1 final`**, code-matched,
identical config but for the named term. `--capacity-telemetry` ON. All value-side distill
coefficients (`--distill-value-coef`, `--distill-value-feat-coef`) set to **exactly** what fdB/fdE
used, and recorded — the arm is the **policy-term form change alone**, and a second moving part
makes it unreadable. Teachers: **the existing tocks, unchanged** (the teacher prescription of §1.4
is a separate, orthogonal lever and must not ride along).

| arm | term | dose | purpose |
|---|---|---|---|
| **G1** | action-level CE, **ungated**, on-pin (rung (c) alone) | coef tuned to match fdB's `grad/distill_share` | **THE DISCRIMINATOR** — isolates TARGET FORM at a matched dose |
| **G2** | action-level CE **+ advantage gate** (rungs (c)+(a)) | coef raised so `grad/distill_share` ≈ fdB's | **THE PRODUCT** — the design's actual proposal |
| *fdC* | coef 0.0, ecology ON | — | **control, ALREADY RUN** — no re-run needed |
| *fdB* | full KL, coef 1.0, ecology OFF | — | **KL comparator, ALREADY RUN** |

Two new arms. Two already on disk.

### 6.2 🚨 THE DOSE CONFOUND — pre-registered, and it is the reason G1 leads

The advantage gate fires on roughly `on_pin × P(disagree) × P(Â<0)` of minibatch rows.
Estimating from the record: disagreement ~8–20% (tick-1's agreement converged to 0.92, fdE's to
0.938) and `P(Â<0) ≈ 0.5` under normalized advantages ⇒ **the gated row count is ~10–20× smaller
than the ungated one.**

**Therefore a healthy G2 is uninterpretable on its own** — "the gate worked" and "the dose was too
small to hurt" predict the identical result, and the tested coefficient range (0.3–1.0, a 3.3×
span) does not reach 10–20×. Two mitigations, both binding:

1. **Dose-match on GRADIENT SHARE, not on coefficient.** `grad/distill_share` (§4.3) must exist
   before either arm launches, and G2's coefficient is set so its share lands **within 2×** of
   fdB's. This is the single build item without which the experiment cannot be read.
2. **G1 carries the scientific weight.** G1 fires on exactly the rows fdB fired on, so it has **no
   dose confound at all**; G1 vs fdB is a clean one-variable contrast on the never-manipulated axis.
   G2 vs G1 is then the gate's own contrast — and, not incidentally, the importance-correction check
   that §3.1's PER risk demands.

### 6.3 The meters (all four, every arm)

1. **PRIMARY — per-team piloting on the 9 pinned teams, n=100, paired vs `rev-1 final`.** The
   `tmp/piloting_mirror_eval.py` equal-pilot mirror pattern. A population meter, deliberately: the
   gated subset is a biased subset (§3.1), so no on-gate readout may serve as primary.
2. **RETENTION = piloting − extraction**, on the same reference / teams / target. The fold's standing
   meter. Reference values on the failed arms: tick-1 **−47%**, fdB **−87%**, fdE **−80%**.
3. **CAPACITY ROW** — `rank/policy_pr` (terminal and trajectory), `rank/trunk_pr`,
   `rank/value_cls_pr`, plus `capacity/*` half-batch cosine, **with the §4.2 did-it-fire check**.
4. **TERM LIVENESS** — `grad/distill_share`, `distill/gated_frac`, `distill/gate_agree_rate`,
   `distill/n_gated`, and the tripwire's own record (fired / not, at what step).

### 6.4 Success, kill, and the branches between

**SUCCESS** (per arm): **retention > 0** — the fold lands at or above the pre-fold base on the
taught teams — **AND** terminal `rank/policy_pr ≥ 0.85 · pr_base` (the healthy band; the five failed
arms all fell 38–43% and would fail this, fdC rose and would pass).

**PARTIAL — G1 healthy, G2 collapses (or the reverse).** A real result, not a muddle: it separates
target form from gate. G1-healthy/G2-collapsed says the **gate** is the harmful half (and given
§6.2, most likely via some interaction the design did not anticipate — re-spec before another arm).
G1-collapsed/G2-healthy is the dose confound firing; read `grad/distill_share` before believing it.

**NULL-DOSE — `distill/gated_frac` < 2% of minibatch rows, or `grad/distill_share` more than 2×
below fdB's.** The arm **DID NOT FIRE**. Report it that way and re-launch with a corrected
coefficient. *Never* report a null-dose arm as "no effect" — three preliminary-cell
non-replications in a single day are already on the record.

**KILL — both G1 and G2 land in the 12–14 `pi_features` band with negative retention.** Then:

> **The fold concept is rejected for never-folded trunks.** Every axis will have been manipulated —
> where, how hard, how many, *and what it asks for* — with the same all-or-nothing collapse. At that
> point the +69 arc is the anomaly requiring explanation, not the failures, and the live question
> becomes the **fold-tolerance-is-trained** speculation of §1.4: `v8_04` was itself
> `distill_4teacher`, and `rev-1` is a fresh trunk that has never been folded.
>
> **Two successor experiments are pre-registered here, in order, so the kill is not a dead end.**
> **First, rung (e) — gradient surgery** (§3.5): the loss level is exhausted, so referee the
> collision at the optimizer level (project the distill gradient off PPO's when their cosine goes
> negative); one code change, the standard +3M arm shape, adjudicated by the same gradient-cosine
> telemetry that convicted the interference plus the rank tripwire. **Second, the fold-tolerance
> arm** — pre-condition `rev-1` with a *self*-distillation phase (a fold with nothing to learn),
> then apply the real fold, and ask whether the trunk that has been through one fold survives the
> second. If it does, the flywheel's D-G cadence needs a warm-up revolution and the whole era's
> fold sequencing changes.

---

## 7. SCOPE — the smallest credible v1

### 7.1 What v1 IS

**One loss function, one gate, three flags, and the tripwire.**

- `agents/training/instrumented_ppo/distill_terms.py`: one new pure staticmethod
  `_gated_action_distill_loss(student_logits, teacher_logits, action_mask, distill_mask,
  advantages, sampled_actions, *, top_k, tau, beta, gate)` → `(loss, metrics) | None`. ~40 lines.
  Masked-mean over gated rows exactly as `_distill_loss` masks over on-pin rows, so **per-row
  gradient magnitude is preserved and only the row FRACTION changes** (which is what makes §6.2's
  share-matching meaningful).
- `ppo.py` fold (~lines 742–841): the per-teacher loop already computes `_sel`; the change is to
  multiply `_sel` by the gate and dispatch on `--distill-target`. ~15 lines. The teacher forward,
  the student-logit reuse, the per-teacher balancing and every value-side term are **untouched**.
- The tripwire: a small callback reading `model.logger.name_to_value["rank/policy_pr"]` in
  `_on_rollout_end`, emitting via `main.launcher.ipc.emit`. ~60 lines + tests.
- `grad/distill_share`: fold the distill term into the existing `grad_balance_metrics` call
  (`ppo.py:1040`) as one more `aux_terms` entry. ~3 lines.

**Nothing else.** No new process, no new obs key (the existing training-only `distill_mask` integer
team-id is reused verbatim), no model-version change, no `check_compatible` entry.

### 7.2 What v1 is NOT (deferred, with the reason)

| deferred | reason |
|---|---|
| **Rung (b)'s judge + producer** | §A's arithmetic: flywheel-cadence, off-policy-buffer-only. Deferred on **cost, not merit** — it is the correct judge |
| **Rung (d)** | magnitude is measured dead (two independent instruments) |
| Value / FitNets distill changes | a second moving part makes the arm unreadable (§6.1) |
| Teacher-recipe changes (9M+ budgets, ~10-team slices) | orthogonal lever, must not confound the loss arm |
| Phase scheduling | gated on arm F (§5.1) |

### 7.3 C6-style BUILD-vs-ENABLE separation

Standard and non-negotiable:

1. **BUILD** behind `--distill-target kl` (the default) and `--distill-gate none` (the default), so
   an unflagged run is **byte-identical** to today. Land with unit tests on the pure loss function:
   `K = n_legal` + `gate=none` must reproduce `_distill_loss`'s KL to floating-point tolerance
   (the *identity* test that makes the new path a superset rather than a replacement); `K=1`
   reproduces `_searchteacher_loss`'s CE; empty gate returns `None`, never a NaN.
2. **Prove the byte-identity** on a `--debug --steps 10000` smoke, both defaults on.
3. **THEN enable** in G1, then G2. Never in the same landing.

### 7.4 How this composes with the critic-calibration plan

The two lines touch at exactly one point and it is a **composition, not a competition**:

- Rung (a)'s judge reads the GAE advantage — the estimate `critic_calibration_plan.md` Layers 1–2
  exist to improve. **The gate's quality therefore rises as the era's primary line lands**, with no
  work on this side.
- Rung (b)'s judge **is** that plan's label factory with a different estimand (an *action
  comparison* rather than a *state value*), sharing the producer, the reconstruction stack and the
  CRN discipline. Building (b) later reuses infrastructure the plan is already paying for.
- **The hazard, named:** `distill/gate_agree_rate` and friends must **never** be read as a
  critic-calibration meter. Two lines reading each other's instruments is how a confound gets
  ratified. The calibration meters are `sd_true_excess`, the mirror table, `value_pooled` PR and the
  §2 paired-head read — and nothing here touches them.

### 7.5 Flag surface sketch

All **train-loop knobs**: not versioned, not in `check_compatible`, inherited on a flagless resume
exactly like `distill_coef` (`config.py:300`).

```
--distill-target {kl,action}       default kl      # THE TARGET FORM (rung c). kl = today, byte-identical.
--distill-topk K                   default 1       # with `action`: teacher top-K renormalized (1 = argmax;
                                                   #   n_legal ≈ kl). The D-F dial.
--distill-gate {none,advantage}    default none    # THE JUDGE (rung a). none = today's on-pin-only.
--distill-gate-tau TAU             default 0.0     # gate on Â < -TAU, in NORMALIZED advantage units
                                                   #   (the same normalization the clip objective uses)
--distill-beta BETA                default 1.0     # AWR temperature for the |Â| weight (mirrors
                                                   #   --search-teacher-beta; w clamped at 20, as there)

--rank-tripwire {off,warn,abort}   default warn    # §4.1
--rank-tripwire-drop FRAC          default 0.20
```

**Validation** (in `config.py::validate`, the existing pattern): `--distill-topk` and
`--distill-gate` require `--distill-target action`; `--distill-target action` requires
`--distill-coef > 0`; `--distill-gate-tau` requires `--distill-gate advantage`. Each a
`parser.error`, and each reachable by `python -m main.checkargs`.

---

## 8. WHAT WOULD MAKE THIS DESIGN WRONG

Registered in advance, so a failure is diagnosed rather than rationalized.

1. **G1 collapses at a matched dose.** Then the target form is not the axis either, every axis has
   been eliminated, and §6.4's kill fires. This is the single most likely way to be wrong.
2. **The gate never fires** (`distill/gated_frac` ≈ 0). Then the disagreement estimate of §6.2 was
   wrong — most likely because on-pin agreement is already ~0.95 at fork time (fdE started from a
   student only 3M diverged from its teacher, and FitNets was 0.995-aligned at hour one). The
   remedy is a **fresher/broader teacher**, which returns to §1.4's orthogonal lever.
3. **The tripwire fires on the control.** Then 20% is inside the noise band of a one-minibatch
   probe and the EMA/persistence rule is under-damped. Re-derive from fdC's own `rank/policy_pr`
   series before trusting any arm's trip.
4. **Arm F is ambiguous** (§5.3) and the trunk repairs. Then the whole framing — *"the objective is
   the wrong object"* — weakens toward *"the objective is affordable at a longer horizon"*, and the
   era's question becomes budget rather than form.
5. **`Â`'s sign is uninformative at the states that matter.** The critic's blur is concentrated in
   ~10–20% of states and ~39% of the conviction-region excess is irreducible. If the disputed states
   are *exactly* the blurred ones — plausible, since disagreement and uncertainty co-occur — then
   the free judge is a coin flip there and only rung (b) can speak. `distill/gate_agree_rate`'s
   trajectory is the early warning.

---

## Appendix A — COST TABLE FOR RUNG (b)

All figures carry their source. **Scope a cost model to its measured path before citing it** — the
banked *162 ms/label* is the **materializer** (one-ply) path and is **not** applicable here;
rung (b)'s labels are rollouts-to-end, whose cost is dominated by `choose_move` forwards.

### A.1 The unit costs

| quantity | value | source |
|---|---|---|
| producer wall that is rollouts | **93%**, of which 93% is `choose_move` | `cf_producer.py:130-148` |
| B=1 CPU decision | **26.3 ms eager → 4.1 ms compiled (6.4×)** | `cf_producer.py:147-148` |
| warm producer, load-fair interleaved beside a live trainer | **8.09 → 1.81 s per label** (one arm, R=8) ⇒ **≈0.23 s / rollout-to-end** | ledger `53870dd`; `cf_producer.py:167-172` |
| cost-model rollout-to-end | **221 ms / line**; **792 ms** for a win-prob at R=8 | `ledger.md:1384` |
| playoff sizing (paired, quiet) | **~1.6 s / decision at R=8 pairs** ⇒ **≈0.10 s / rollout** | `wang_search_reconciliation.md:332` |
| contention scaling | **878 ms/label @ load 7 → 2,787 @ load 25** (×3.2 for ×3.6 load) | `ledger.md:1509` |
| intra-process rollout concurrency | **a WASH** (conc=1 **3.86 s** vs conc=8 **4.17 s**) — parallelism must be by PROCESS | `cf_producer.py:1560-1567` |
| producer batching | **a NULL** (rpc 4/8/16 → 86/43/99 per hour) — compute-bound | `ledger.md:3895-3906` |
| throttle | `--max-labels-per-hour` default **2000** | `cf_producer.py:1532` |

Two independent sources bracket the per-rollout cost at **0.10 s (quiet) – 0.23 s (beside a live
trainer)**. Use the band, never a point.

### A.2 Cost per ATTEMPT (one disputed state adjudicated)

An attempt = 2 arms × R paired rollouts under shared CRN. The candidate screen (replay the record,
one teacher forward + one student forward per decision at 4.1 ms compiled) is **<2%** of the bill
and is ignored below.

| R (pairs) | rollouts | wall @0.23 s (busy) | wall @0.10 s (quiet) |
|---|---|---|---|
| 4 — `MIN_PAIRS`, the arbiter's floor | 8 | 1.8 s | 0.8 s |
| 8 | 16 | 3.7 s | 1.6 s |
| 12 — `DEFAULT_ROLLOUTS` | 24 | 5.5 s | 2.4 s |
| 24 | 48 | 11.0 s | 4.8 s |

### A.3 Cost per USABLE VERDICT — where the arithmetic bites

The refusal rule earns its keep and costs dearly. Measured on the completed 80/80 playoff cell:

- **70% of contested top-2 comparisons were INDISTINGUISHABLE** at R≈10 paired terminal rollouts.
- screen-decisive **15.8%**; the playoff overrode the policy on only **7.4%** of decisions.
- ⚠️ **Raising R does not rescue this.** The R-ladder was **flat across a 32× dice sweep**, and the
  playoff's own verdict is that the indistinguishability is largely **real**, not a power problem. A
  paired-binomial resolution improves as `1/√R`, so halving the detectable effect costs **4×**.

Of the ≤30% that separate, assume half favour the teacher (planning number; the teachers are
verified +8–11pp at the population level, so this is conservative-to-fair).
**⇒ usable-verdict yield ≈ 0.15 per attempt at R≈8–12.**

| | busy (0.23 s/rollout) | quiet (0.10 s/rollout) |
|---|---|---|
| per usable verdict, R=8, 1 process | **~25 s** | **~11 s** |
| **with 4 nice-10 background processes** | **~6.2 s** ⇒ **~580 / hour** | **~2.7 s** ⇒ **~1,330 / hour** |
| under heavy trainer contention (×3.2) | | **~180–420 / hour** |

Consistent with the producer's own 2,000/h throttle and with the post-warm-path expectation of
**~1,200–1,500/h** (`ledger.md:4348-4352`).

### A.4 The conclusion that defers rung (b)

A **+3M arm is roughly 0.5–1.5 h of wall clock**. At §A.3's rates that is **~300–2,000 usable
verdicts** — against a **3×10⁶-state** on-policy stream, i.e. **coverage ~0.01–0.07%**, and
**1.5–10%** of the `CorrectionBuffer`'s 20,000-record ring.

Three consequences, all design-determining:

1. **Rung (b) can never gate an on-policy KL.** Only an off-policy buffer term, where each verdict
   is resampled across many minibatches — which is exactly what `_searchteacher_loss` already does.
2. **Rung (b) is a FLYWHEEL-cadence item.** A corpus worth training on needs a production window
   measured in tock-hours, not arm-hours. It belongs beside a tock, filling the ring while the
   exploiters train, and is folded at the next tick.
3. **It is nearly free to build once someone wants it**, and that is worth writing down: the
   substitution primitive, the CRN arbiter with its refusal rule, the 3-tier gate, the training-side
   state tap, the record schema, the shard transport, the ring buffer and the AWR loss **all exist**.
   The build is *one producer* whose proposer is the teacher instead of the beam
   (`teacher/produce.py` tier 1), plus a source switch to `teacher/generate.py` so the opponent is
   **known** rather than `self_current`. Estimated ~1 agent-day.
