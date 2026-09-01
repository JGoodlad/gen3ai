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

## 4. What the 2026-08-31 campaign settled about the critic (status as of 2026-09-01)

Section 3 left the critic-surprise finding unattributed between causes 3 (representation/target)
and 4 (horizon/bootstrap). The KO-boundary work moved it, decisively, to **3 — and narrowed 3 to
the READOUT**.

**Intuition.** Ask the model "will Surf knock out Tyranitar?" The physics block inside the network
already knows: the `DamageOperator` ships a closed-form 16-roll KO fraction as its own channel,
and on a constructed Starmie/Tyranitar sweep it tracks the true KO probability at slope +1.17,
r=0.995. The win-prob head, on the *same forward pass*, moves at slope +0.02. The fact is in the
building; the head does not read it. Proof that it is readout and not capacity: a **linear** map of
the head's own input (`value_pooled`) decodes true win-probability at AUC 0.845 where the head
scores 0.588 — chance on open KO races. Expressiveness (GLU, more layers) is acquitted; the target
and readout are convicted.

**The population-scale signature: aggregate-calibrated, resolution-blind.** Under ground-truth
Monte-Carlo anchors (333 anchors, 13,320 rollouts) the head's mean bias straddles zero while
per-state |error| is 0.28 and `sd_true_excess` is 0.25–0.43 in every predicted bin. It prices the
HP bar smoothly and is blind to roll structure: revealing a threat halves the mask's
truth-tracking amplitude. The bias is *larger* at common faint counts (P(Surf) at true
KO-indifference 0.79 at 5-5 → 0.92–0.93 at 2-2..4-4), so the constructed probe was conservative,
with one unexplained exception at 1-1 where the policy is unbiased and orders 22/22 sweep points.

**Whose losses live at the boundary.** The credit-surface hypothesis (exploiters win by farming
50/50 states the critic misprices) was refuted with the sign inverted in all three generations,
and the inversion got 5.8× stronger under truth: exploiter-win swings are essentially never at
true boundary (share 0.03); *target*-win swings are (0.27). The boundary is where the victim claws
back, not where the exploiter wins. The credit defect itself stands; the claim about who exploits
it fell.

**The bot gap is credit, not conditioning (SI-2).** One-ply search under the model's OWN marginal
recovers 70% of the available crater improvement; an oracle over the bot's reply adds ~0.6pp of a
7.4pp gap. Knowing the opponent's move is cheap; valuing the resulting state is what is missing.

**Design rule banked (decision-focused value learning).** Value error matters only where it
changes a decision, so the right objective is induced-policy regret, concentrated at the
sigmoid's steep middle. But at the boundary, decision relevance and label NOISE arrive together
(a 50/50 outcome is a coin), so "care more" must be implemented as MORE SAMPLES (re-seed,
multi-rollout: the harvest's Beta-evidence priority, `--q-top-n`), never as larger per-sample
weights, which amplify exactly the noise you cannot afford. `--value-tail-weight` is the
hand-rolled cousin of Emphatic TD's interest function; the meta-learned version stays shelved
until a hand rule measurably underperforms.

**And the fold does not fix it (M5).** Distillation moves the critic's off-slice *level*, not its
*resolution*, in both eras; the policy is the mover. Whatever repairs the readout will come from
the value target and its label supply (the counterfactual label factory, the twin/evidential/Q
heads gated behind `gen3_cf_*` flags), not from teachers.

**The question you can answer after this section:** *the head says 0.66 on a state whose true
value is 0.91 — is that a capacity problem?* No: check whether a linear probe on `value_pooled`
recovers it (it does, at 0.845). It is a target/readout problem, and the cure is label supply at
the boundary with more samples, not a wider head.

See also: `distillation_flywheel_lessons.md` (the policy side of the same campaign),
`win_prob_decomposition.md` (the error taxonomy this refines), root `CLAUDE.md` → *Exploitability*,
`designs/research_state/measurements/ko_boundary_decodability_2026-08-31.md`.
