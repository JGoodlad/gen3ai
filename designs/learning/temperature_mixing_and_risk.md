# Temperature, Nash mixing, and risk awareness — what a policy's distribution *means*

**TL;DR.** A policy is a distribution over actions, and three different questions hide inside
"how sharp should it be?": (1) **Distillation temperature** — how much of a teacher's
distribution to copy. Our measured answer is T→0 (copy only the argmax): an RL teacher's tail
is contaminated by the entropy bonus and extrapolation noise, so full-KL matching transplants
garbage (`--distill-target action`, `--distill-topk 1` — the +7.6pp result). (2) **Play
temperature vs Nash** — temperature cannot produce equilibrium mixing, because Nash
probabilities are set by the *opponent's* indifference conditions, not by our own action
values; the right transformation of a contaminated policy is support restriction (top-p), not
temperature. Mixing must be *learned*, and the league (flywheel = fictitious play with the
fold as the averaging operator) is the machine that learns it. (3) **Risk appetite** — under
a win/loss objective, "gamble when behind, consolidate when ahead" is not psychology; it is
the curvature of the P(win) sigmoid, and an agent maximizing P(win) inherits it for free.
Whether *our* policy actually modulates risk is UNMEASURED — the ai_v12 capstone probe
(`designs/ai_v12/probe_risk_modulation_capstone.md`) exists to measure it.

---

## 1. Distillation temperature: when a teacher's tail is treasure and when it is poison

**Intuitive.** Temperature is a focus knob on a distribution: T→0 shows only first place,
moderate T shows the runner-up structure, high T flattens toward uniform. Classical
distillation ("dark knowledge", Hinton 2015) says the runner-up structure is the payload — a
classifier saying *dog 0.9, wolf 0.09, car 10⁻⁶* transmits similarity geometry no hard label
contains. The math: as T rises, the KL-matching gradient approaches logit-difference
matching; temperature dials between "copy the teacher's decisions" (low T) and "copy its
ranking geometry" (high T). The literature's corroboration: label-smoothed teachers — whose
tails are deliberately de-structured — distill *worse* (Müller et al. 2019).

**Our measured heresy.** Full-distribution KL at matched dose *corrupted* the student;
pure action-form top-1 won by **+7.6pp (z=6.0)** — i.e. our optimal temperature measured as
freezing. Reconciliation (theory, flagged as such): a supervised classifier's tail is
sculpted by data and is honest; an RL teacher's tail is sculpted by the **entropy bonus**
(`--ent-coef 0.02` pays the policy to keep mass on actions it has no reason to prefer) plus
unconverged noise on rarely-visited states. Only part of the tail is genuine "second-best
line" structure; KL matching forces the student to reproduce all of it with equal fidelity —
and the gradient's probability-ratio weighting spends real capacity on the teacher's 3%
flotsam.

**The robbery connection (hypothesis).** The distill loss lands on the *student's* states
(team bias 0.4 — an invisible constant, category #10 of the mechanism map). Off the
teacher's competence slice, its distribution is dominated by entropy residue and
extrapolation — full-KL would transplant *confidently-shaped noise* wherever state
distributions brush. A concrete candidate mechanism for how a narrow fold robs; predicts
robbery looks like entropy-artifact behavior (indifference among bad options), checkable by
the behavioral-fingerprint instrument.

**Honest status.** The entropy-residue account fits two data points (KL corrupts; top-1
wins). A rival fits them too: an optimization pathology in the KL-through-PPO interaction,
independent of tail content. Distinguishing cell, never run: the **top-k ladder** (k ∈
{1,2,3} at matched dose). Tails-as-poison ⇒ damage grows smoothly with k; optimization
pathology ⇒ k=2 behaves like k=1 and full-KL fails discontinuously.

**Three temperature dials, set differently on purpose.** Generation (self-play/pool
opponents): warm (`--self-play-temp 1.0`) — diversity of what the student *faces*. Targets:
frozen (top-1) — decisiveness in what the student *copies*. Evaluation: greedy
(`stochastic=False`) — determinism for CRN pairing and variance; the honest cost is that we
measure the policy's mode, not its distribution.

## 2. Temperature vs Nash: support, ratios, and why mixing must be learned

**The trap.** In rock-paper-scissors, a correct 33/33/33 at T=1 is not "temperature doing
its job" — the *learned distribution* is the equilibrium and T=1 merely plays it undistorted.
Temperature is a post-hoc transformation sliding along a one-parameter family (sharpened ↔
flattened versions of what was learned), and that family almost never passes through Nash.
If the policy is Nash, every T≠1 *creates* exploitability; if it isn't, no T repairs it.

**The indifference principle** (the deep fact). At a mixed equilibrium, every action in the
support has *exactly equal* expected value against the opponent's mixture — that's why
mixing is free — and everything outside the support is strictly worse. Nash never plays a
bad move 5% of the time; a 5% action is precisely as good as the 60% one. The counterintuitive
half: the *ratios* are not preference strengths — they are chosen to make the **opponent**
indifferent, computed from *their* payoffs. Corollary: softmax-over-own-values at any
temperature has the wrong functional form for equilibrium mixing. Mixing ratios must be
learned from adversarial pressure; that is what self-play is *for*.

**What Nash looks like in gen3.** Mostly pure (dominated decisions — one action best
regardless of the simultaneous choice), with sharp pockets of mixing at **guess points**:
double-switch predictions, sack-vs-protect, Choice-locked picks, and the lead choice (a
genuinely simultaneous blind game — a deterministic lead is "always rock" on a ladder with
rematches). Neither T=0 (kills the pockets) nor high T (bleeds EV at dominated states) is
right; the correct object is state-dependent.

**What training converges to.** Entropy-regularized self-play converges (two-player
zero-sum) to a **quantal response equilibrium (QRE)** — a softmax-response fixed point whose
temperature is set by the entropy coefficient; ent-coef → 0 recovers Nash. So at T=1 we play
a slightly over-mixed Nash: nearly free at guess points (payoffs are flat there), a real EV
leak at dominated states. This also refines §1: some of the tail *is* strategic mixing Nash
requires — components (a) equilibrium mixing, (b) entropy subsidy, (c) noise are entangled
in the same mass.

**Top-p beats temperature for the repair.** Given that contamination, truncation
(nucleus/top-p) deletes the low-mass tail — where (b) and (c) concentrate — while leaving
the *ratios among survivors* untouched, which the indifference principle says must be
preserved. Temperature rescales all ratios, damaging (a) to fight (b)+(c). Failure mode:
generous p is needed or genuine 2–5% equilibrium actions (poker bluff frequencies) get
amputated.

**Leagues and cycling** (summary — detail in `population_game_theory.md`). Naive best-response
dynamics cycle at guess points (each side over-corrects, orbiting the mixed point). Two fixes,
both of which we run: *smoothing* (the entropy term damps the cycle into a convergent spiral —
ent-coef is the convergence mechanism for mixed equilibria, not just exploration) and
*averaging* (fictitious play converges by averaging historical best responses — **the flywheel
is fictitious play at era scale**: exploiters are the best-response oracle, the distillation
fold is the averaging operator; this is the PSRO construction). A league's pool is monotone
memory, so gross cycles ("always rock") die; residual cycling survives in exactly three
places: (i) the sampling-window leak (PFSP down-weights old counters toward zero pressure —
functional forgetting), (ii) **the weight-interference leak — the measured treadmill (rev-2
robs, repair follows coverage) is a league cycle running through catastrophic interference
one level below the pool**, pinned by coverage breadth, and (iii) the game's *irreducible*
mixed content, which equilibrium prices rather than removes — visible as a stable nonzero
`eval/hodge_cyclic_fraction` (growing cyclic fraction = a forgetting channel winning;
stable floor = correctly priced RPS content).

## 3. Risk awareness: the sigmoid is the risk policy

**The dissolution.** "Humans gamble when behind — intrinsic bias or rational response?" is
hard to answer in humans because prospect theory (genuine loss-domain risk-seeking bias) and
the correct policy point the same way. For a P(win) maximizer the question dissolves:
under a **threshold objective** (win/lose; six mons at 1% beats one at 100%), variance
helps below the threshold (mass must cross it) and hurts above (your mass crosses back).
Pulled goalies, fourth-down aggression, must-win sharp openings — all the optimal policy of
the objective, no psychology required.

**The compact statement.** Plot P(win) against position: an S-curve. **Risk appetite is its
curvature.** Convex below the midpoint ⇒ Jensen says spreads beat the sure middle ⇒ gamble;
concave above ⇒ consolidate. The same coin flip is correct at 30% and wrong at 70%.
**Risk-neutral in probability space = risk-sensitive in material space** — an agent
maximizing a calibrated P(win) inherits the entire risk policy with no explicit risk system.

**The catch, and the ai_v12 connection.** The free lunch requires the optimized quantity to
*be* P(win). Material-flavored shaping partially maximizes expected material — risk-neutral
exactly where correctness demands curvature. PBRS is policy-invariant in theory (the
clean-world design keeps the lunch); the old hand-shaping biases had no such guarantee.
**The ai_v12 sparse arm (±1 terminal) is risk-correct by construction in the limit** — the
clean-world experiment is a risk-correctness experiment, not only a credit-assignment one.

**Our model: knows vs uses.** KNOWS the score, substantially, measured: win-prob head knows
96.4% of whiffs (probe L); calibration CLI splits critic-overvalued from lost-position;
OHKO belief AUC 0.79; the historical gaps were *specific* (clock cliff: positive V on the
final decision in 13/14 timeout losses — fixed by `gen3_deadline_clock_v1`, 81%→22%; the
self-KO floor leak). USES it to modulate risk: **UNMEASURED** — and knowing/using have split
before (bait verdict: head knew, policy fired, credit convicted). Circumstantial hints cut
both ways (stall-while-ahead ⇒ over-conservatism ahead; all-or-nothing 6-0 losses ⇒
risk-blindness behind).

**The probes** (all offline, existing tooling): the **accuracy-tradeoff curve** — P(chose
Hydro Pump over Surf | both legal) as a function of recorded win-prob; falling curve =
risk-correct, flat = risk-blind; **Explosion timing** — boom frequency vs V (correct play
booms from behind); **the general curve** — per-action outcome spread via CRN rerolls
(the falsifier's aleatoric instrument repointed), correlated against win-prob. Registered as
the ai_v12 capstone: `designs/ai_v12/probe_risk_modulation_capstone.md`.

## Where this lives in our architecture

- Distill target form: `--distill-target action --distill-topk 1` (the +7.6pp cell); the
  entropy source of tail contamination: `--ent-coef 0.02`; team-bias constant:
  `--distill-team-bias 0.4` (mechanism-map category #10, never varied).
- Temperatures: `--self-play-temp` / `--stable-opponent-temp` (generation, warm);
  eval greedy (`stochastic=False` in the eval players); no play-time top-p exists today.
- Cycle meter: `eval/hodge_cyclic_fraction` / `eval/hodge_width_elo`
  (`agents/training/hodge.py`, `python -m main.elo`).
- Win-prob head: `--win-prob-mode` (v107 adds `--q-winprob-mode read_only`); calibration:
  `python -m main.prober.query calibration`; the clean world: `designs/ai_v12/launch_runbook.md`.
- Reroll/spread machinery: `utils/bridge/reconstruction` `reroll_many`, the falsify
  aleatoric split (`python -m main.prober.query falsify`).

## Synthesis

One object — the policy's conditional distribution — answers three different masters. As a
*distillation target* its tail must be honest, and ours measured contaminated, so we copy
decisions, not geometry. As a *strategy* its mixing must satisfy opponent-indifference at
guess points, which no temperature can synthesize and only adversarial pressure (the league,
i.e. fictitious play, i.e. the flywheel) can teach. As a *risk policy* it needs no separate
machinery at all — a calibrated P(win) objective's curvature is the whole of correct risk
behavior, which is the quietest and maybe strongest argument for the clean-world program:
±1 terminals don't just fix credit assignment, they make gambling-when-behind the literal
optimum. The recurring theme: sharpness, mixing, and risk are all statements about *which
parts of a distribution are signal* — and every dial we set (ent-coef, topk, temperature,
support) is a claim about that, whether we make the claim explicitly or not.

## See also

- `designs/learning/imperfect_information_and_equilibria.md` — information sets, why belief
  conditioning is Nash-required while behavior-prediction conditioning is priced exploitation.
- `designs/learning/population_game_theory.md` — PSRO/league construction in full.
- `designs/learning/win_prob_decomposition.md` — the head this note's §3 leans on.
- `designs/learning/credit_assignment_and_value_errors.md` — why knowing ≠ using (the bait
  conviction).
- `designs/ai_v12/design_winprob_behavior_coupling.md` + `launch_runbook.md` — the clean
  world; `probe_risk_modulation_capstone.md` — §3's probes, pre-registered.
- Ledger 2026-08-30 entries: temperature/tails, sweet-spot hypothesis, category map,
  behavioral fingerprint.
