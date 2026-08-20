# Credit assignment and the geometry of value errors

*Why "critic surprise" is the project's named enemy, what the λ dial actually trades, and how a
bootstrapped critic's errors propagate. Intuition first, then the machinery, grounded in our
measured record.*

## 1. Intuitive: one bit, forty decisions

A battle ends win or lose: ~1 bit of ground truth spread across ~40 decisions. Reinforcement
learning's central problem is deciding WHICH decisions that bit belongs to. The critic is the
amplifier that converts one terminal outcome into a per-decision signal (advantages) — which
means **a wrong critic mis-assigns credit at every decision simultaneously**. That is why the
sentinel sweep's headline — the win-prob head reading a median **0.827 on the top-50 decisions
that lost their games** — matters more than any single behavioral pathology: a policy trained on
a confidently-wrong critic is being taught the wrong lessons everywhere at once, invisibly.

## 2. The machinery

**Baselines and variance.** The policy gradient is `∇log π(a|s) · (signal)`. Using raw returns as
the signal is unbiased but drowns in variance; subtracting a baseline V(s) (the critic) removes
variance without adding bias — IF V is right. The whole actor-critic bargain is: trade the
baseline's bias risk for the return's variance.

**GAE's λ is the bias-variance dial, stated precisely.** The advantage estimator blends TD
residuals over horizons: λ→1 recovers the Monte-Carlo return (unbiased, high variance — the
1-bit problem in full); λ→0 uses the one-step TD error (low variance, but every drop of critic
error becomes signal bias). Everything between splits the difference geometrically. The practical
reading: **the more you trust the critic, the lower λ can go; the worse the critic, the more λ is
protecting you** — and a project whose named enemy is critic error should remember its λ choice
is that protection.

**Bootstrap error propagation — the self-referential part.** The critic trains toward targets
built FROM ITSELF (`r + γV(s′)`). An error at state s′ becomes a wrong target for s, which
becomes a wrong target for the state before it: value errors travel BACKWARD through time, and a
systematic bias (e.g. over-valuing boards with a live sweeper behind) self-reinforces until some
ground truth (a terminal) interrupts it. This is why the TD-aux line (ledger C5) is a
*consistency* pressure — it makes the critic agree with itself along real transitions — and why
its measured ceiling is exactly what theory predicts: **consistency suppresses noise but cannot
create signal**. A critic can be perfectly self-consistent and consistently wrong.

**Scale arbitration on a shared trunk.** With γ≈0.9999 the returns run to ±hundreds, and the
value MSE gradient can swamp the shared trunk (the measured `grad/value_policy_logratio`
pathology). PopArt is the fix in production: the head trains in normalized space and the POP
rescale keeps the de-normalized function unchanged. `vf_coef` is the residual arbitration knob —
resume-immutable because silently changing the actor/critic gradient ratio mid-run changes what
the trunk is FOR.

**What distributional heads buy — and what they measurably did not buy here.** A return
DISTRIBUTION makes tail risk visible (the awareness stack: `knew_by_turn`, `blind_loss`, the
stall signature of bottom-atom mass under a positive mean — things a scalar V cannot express) and
enables calibration audits (PIT: gen-10 read pit_mean 0.396, coverage80 0.44 vs nominal 0.80 —
optimistic AND overconfident). What it did NOT buy: a strength lever via tail-reweighting — the
measured verdict is that our value residuals are sub-Gaussian, no heavy tail to re-weight. Ledger
rule: the distributional head is an INSTRUMENT here, not a lever.

**The instrument hierarchy** (the ledger's standing method): post-hoc ablation measures
DEPENDENCE; lesion-eval measures HARM-AS-TRAINED; a training-time A/B measures CAUSATION. Value
errors specifically demand the calibration layer too — recorded V(s) vs realized return G(s),
selection-aware (the prober's reliability curve), because eval quotas over-capture losses and a
loss-conditioned V−G is biased positive by construction.

## 3. The map of critic-failure causes (each with a different cure)

1. **Input coverage** — the critic literally cannot see the fact. Cure: obs work. Measured
   instance: the deadline clock (13/14 timeout losses carried positive final V → 2/9 after).
2. **Distribution** — the failing states are rare in training. Measured refutation available:
   stall states were 14× OVER-exposed, killing that account for the stall pathology.
3. **Representation/target** — the trunk carries the fact but the value target can't use it
   (ledger H1/C4: tokens carry no trade signal; a constant beat every head on self-KO ΔV).
4. **Horizon/bootstrap bias** — errors propagating from later states (the TD-aux territory).

The sweep's critic-surprise finding is currently UNATTRIBUTED among 3 and 4 — which is why C6
requires a new pre-registered mechanism before anyone touches the critic again.

**The question you can answer after this note:** *the win-prob head says 0.83 and the game is
lost this turn — which of the four causes is it, and which instrument separates them?* (1 is
ruled out per-case by checking the obs holds the killing fact; 2 by exposure counts in training
traces; 3 vs 4 by whether a probe can decode the danger from the trunk while the value target
misses it — decodable-but-unvalued ⇒ 3, undecodable ⇒ the fact never survived to the trunk.)

Related notes: [`marginalization_and_uncertainty.md`](marginalization_and_uncertainty.md) (the
distributional head's mean/support mechanics); PopArt and the reward composition live in
`src/agents/model/CLAUDE.md` and `src/agents/training/CLAUDE.md`.
