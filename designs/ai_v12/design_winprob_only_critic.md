# design — THE WIN-PROB-ONLY CRITIC: one value function, in outcome units

> **[STATE 2026-09-06]** **DESIGN + GAP AUDIT. Nothing here is built and nothing is sanctioned to
> run.** The owner direction (2026-09-06) is that the next generation uses the win-probability head
> as the critic's ONLY signal — a cleaner value function to serve as the starting point for long
> (3-day, 75M-step) runs and for search, whose leaf is this critic. This document states what is
> true now, what the end state is with a recommendation on every open choice, the calibration gate
> that must hold before a long run is trusted, and the migration. §6 is the ranked gap list a build
> works from.
>
> Era: **ai_v12** — the win-prob → behavior-coupling chapter, of which this is the terminal move.
> [`design_winprob_behavior_coupling.md`](design_winprob_behavior_coupling.md) opened it with three
> routes that turn a barometer into a coach; **this is the fourth and most direct: stop treating it
> as a side channel and make it the value function.** Route 1 (`--win-prob-pbrs-coef`) is
> superseded by it (§3.7) and route 2 is orthogonal (it targets the POLICY, not the critic).
>
> **The measured baseline this design must beat is committed**:
> [`designs/research_state/measurements/winprob_critic_baseline_2026-09-06/`](../research_state/measurements/winprob_critic_baseline_2026-09-06/).
>
> **It deviates from a standing plan, and that is stated up front, not buried** — see §0.

---

## 0. The one thing to read before agreeing with this document

[`designs/research_state/critic_calibration_plan.md`](../research_state/critic_calibration_plan.md)
§3 specifies a **staged promotion** for exactly this change:

> heads-only twins (running) → **win-prob TRUNK-OPEN** only after a POSITIVE §2 read → **shadow
> critic** promotion → the real critic. […] **Never skip a stage — each one's meters license the
> next.**

**This design skips to the last stage.** It is an owner-ordered generation change, not a research
arm, and pretending otherwise would be the worse error. The honest accounting:

| what the skipped stages would have bought | what substitutes for it here |
|---|---|
| a within-run paired read of grounded-label heads (twins B/C) | nothing. **Not substituted.** |
| a passive shadow critic's divergence-vs-live-V, published before promotion | the OFFLINE baseline (§4) — the same question asked of the head that already exists, on traces already on disk, with a bar registered before the run |
| the `sd_true_excess` / mirror-table meters read on an arm | **§4's gate, pre-registered** — but two of the four plan meters are not yet runnable offline (§6, gaps M1/M2) |

The plan's own **KILL** clause is unaffected and still applies to the label-factory line. What this
generation tests is a different proposition from the plan's: not *"can grounded labels sharpen the
critic"* but *"is the critic's TARGET — a shaped, discounted, PopArt-normalized return — the wrong
quantity to be predicting at all"*. The two are complements. If this generation fails, the label
factory is still the lever, applied to a cleaner target.

---

## 1. WHAT IS TRUE NOW — the exact chain, end to end

Everything in this section is read from HEAD (`407b27c0`, `MODEL_CONFIG_VERSION` **107**,
`ARCH_SIGNATURE` **`gen3_critic_route_wave_v1`**, `MIGRATION_FLOOR` **96**) and from
`designs/production_config.json`. File:line references are to `src/`.

### 1.1 The reward stream

`RewardClass` (`agents/training/reward_manager.py:49-61`) is a three-way census every launch prints
and every save records into `metadata.json` as `reward_composition`:

| class | contract | production composition |
|---|---|---|
| **TERMINAL** | the ±`victory_value` win/loss. "Emitted as-is; never shaped/blended/flag-affected." | **1 term** |
| **PBRS** | pure potentials, ALWAYS telescoping (Φ(terminal) = 0), objective-neutral | **7 terms** |
| **BIAS** | soft shaping whose additive↔telescoping mix is `--bias-additivity` (λ = 1.0 ⇒ fully additive) | **1 term** (`no_progress_tax`) |

Production reward config (§6.3 of `ARCHITECTURE.md`, all resume-immutable via
`check_reward_config`): `all_shaping_pbrs` **true** · `victory_value` **30.0** · `draw_penalty`
**−35.0** · `no_progress_penalty` 0.15 · `mat_alive_weight` 1.25 · `bias_additivity` 1.0 ·
`hand_shaping` **true** · `pbrs_material` **true** · `pbrs_belief` **true** · `stall_pbrs` false.

The terminal, precisely (`clean_world.py:44-54`): **a win scores +30, a decisive loss and a rare
pre-cap tie score −30, a 250-turn TIMEOUT scores −35.** The ordering is load-bearing and the flag
help says so: `--draw-penalty` must stay ≤ −`victory_value`, or a 250-turn stall becomes the best
non-winning outcome and a losing agent's optimal play is to run out the clock.

**The advantage-invariance of the PBRS class is STRUCTURAL, not asserted by a test.** It rests on
two facts: Φ(terminal) = 0 (so every episode's shaping telescopes to −Φ(s₀), a per-start-state
constant), and `PBRS_GAMMA = 0.9999` **must equal** the PPO `gamma`
(`agents/training/reward_weights.py:101`; `gamma=0.9999` at `main/train/model_build.py:691`, pinned
by an assert at build time per `reward_registry_test.py:52`). ⚠️ **UNVERIFIED that any test asserts
advantage-invariance directly** — `reward_registry_test.py` pins the class registry's coverage and
the γ equality; the invariance itself is the Ng–Harada–Russell (1999) theorem applied to those two
facts, which `winprob_pbrs.py`'s module docstring states in full. Treat "advantage-invariant by
construction" as a *derivation from pinned premises*, not as a green test.

### 1.2 Returns and GAE

Stock SB3 (`buffers.py:403-438`), unmodified:

```
delta        = r_t + γ·V(s_{t+1})·nonterminal − V(s_t)
last_gae_lam = delta + γ·λ·nonterminal·last_gae_lam
returns      = advantages + values
```

`values` are written by `policy.forward` / `policy.predict_values`, both of which route through
`_critic_value` and are **denormalized — REAL reward units**. Rewards are raw env rewards.
Therefore **advantages and returns are in raw shaped-return units**, running to ±hundreds at
γ = 0.9999. Advantages are then per-minibatch normalized `(A − mean)/(std + 1e-8)`
(`instrumented_ppo/ppo.py:477-487`) and never touch PopArt.

The only pre-GAE modification in the tree is `--win-prob-pbrs-coef` (§1.6), which mutates
`rollout_buffer.rewards` in place and re-runs `compute_returns_and_advantage`. At coefficient 0 the
module is not even imported.

### 1.3 PopArt — and the two currencies

`PopArtNormalizer` (`agents/model/popart.py:43-117`) keeps EMA `mu`/`sigma` (β = 0.1, σ floored at
1e-2) of the rollout's real-unit returns, updated once per `train()`
(`instrumented_ppo/ppo.py:401-405`). The vocabulary is
[`designs/learning/popart_value_scale_and_currencies.md`](../learning/popart_value_scale_and_currencies.md):

> the network earns and thinks in **z-dollars**; the environment pays and settles in **raw-yen**;
> there is exactly one exchange window (`_denorm` in `predict_values`).

The currencies in play today, named:

| quantity | currency | where converted |
|---|---|---|
| env reward, GAE δ, advantages, `returns` | **raw** (shaped, γ-discounted, ±hundreds) | — |
| `value_dist_head` atoms + logits, its CE target | **z** (normalized return; ±12 = ±12σ) | `popart.normalize(returns)` at `ppo.py:872-875` |
| `_critic_value` output | **raw** | `_denorm` at `policy.py:202-203` |
| `win_head` logit → σ(·) | **probability** — a THIRD currency, convertible to neither | never |
| `td_aux` residual | raw ÷ `popart.sigma` | `aux_terms.py:92` |
| `cf_shadow` target | z | `cf_terms.py:448` |

**The third currency is the whole problem.** P(win) is commensurable with nothing else in the
table, which is why `main.scaffolding_gauge` ships *two* gauges and a units disclaimer instead of a
conversion, and why `--defensive-leaf` had to be a measured choice rather than an obvious one.

### 1.4 `value_dist_head` — the head that IS the critic

Class `ValueDistHead` (`agents/model/aux_value_heads.py:50-100`):
`LayerNorm(128) → Linear(128,128) → ReLU → Linear(128, bins)`, reading `value_pooled`.
Production **51 atoms over [−12, +12]** ⇒ Δ = 0.48, HL-Gauss σ_g = 0.75·Δ = 0.36 — **all in z
units**. The support is a NON-persistent buffer (`aux_value_heads.py:91`), so it is not in the
`state_dict` and is enforced on resume by `check_value_dist` (`model_version/resume_checks.py:172-190`).

HL-Gauss target (`instrumented_ppo/value_terms.py:104-121`): erf-difference of a Gaussian at the
normalized return over each atom's bin, edge bins absorbing the tails, renormalized, cross-entropy
against `log_softmax(logits)`.

`value_from_dist` **true** makes it load-bearing (`policy.py:173-203`):

```python
if self._value_from_dist:
    ...
    return self._denorm(head.mean(logits))      # E[Z], denormalized
return self._denorm(self.value_net(latent_vf))
```

There is **no fallback** — a missing head or a stale stash raises. All four value sites
(`forward`, `evaluate_actions`, `predict_values`) go through it. The CE is weighted at `vf_coef`
(**0.5**) rather than `value_dist_coef` when `value_from_dist` is on (`ppo.py:880-881`), and the
scalar path is dropped outright: `_vf_term = 0.0 if value_from_dist else self.vf_coef * value_loss`
(`ppo.py:557-560`).

**Four facts about this that no document stated before, from the 2026-09-06 audit:**

1. 🚨 **PopArt's POP surgery never touches the head that is the production critic.**
   `popart.update(returns, self.policy.value_net)` (`ppo.py:403-405`) preserves `value_net`'s
   denormalized output across a statistics move. `value_dist_head` gets no such correction, and its
   atoms are in z units — so **every (μ, σ) update silently shifts the real-unit E[Z] of every
   state.** Small per step at β = 0.1; uncorrected in total. The popart note flags exactly this
   class ("exact output-preservation exists only for LINEAR heads … re-fit continuously"), but the
   re-fit is the CE's, not a POP correction, and `policy.py:174-177`'s "same PopArt peg as the
   scalar" is true of the (de)normalization maps and **not** of the POP half.
2. **`value_tail_weight` (0.3, resume-immutable, ACTIVE in the production table) is INERT on the
   trained critic.** It lives in `_value_loss_from_se` (`value_terms.py:139-155`), which feeds only
   the scalar MSE that `ppo.py:560` zeroes under Phase B. The HL-Gauss CE has no tail weighting.
3. **`value_net` is frozen by OMISSION, not by `requires_grad`.** It appears in no loss graph, but
   it is still an optimizer parameter and is still POP-rescaled every `train()`.
4. **Under Phase B the real critic loss is tagged `"aux"`**, not `"value"`, in the per-term
   noise-scale and grad-balance bookkeeping (`ppo.py:569-570`, `ppo.py:882`). The grad-balance probe
   compensates (`ppo.py:1310-1313`); **the noise-scale groups do not** — so
   `train/noise_scale_value` has been reporting a term whose weight is 0.0.

### 1.5 `win_head` — where it sits, what trains it

Class `WinProbHead` (`agents/model/aux_value_heads.py:16-47`): `LayerNorm(128) → Linear(128,128) →
ReLU → Linear(128,1)`, reading `value_pooled` — **structurally identical to `ValueDistHead` but for
the output width.** Built at `extractor_build.py:783-786`; forward at `extractor_forward.py:726-728`:

```python
wp_in = value_pooled if self.win_prob_mode == "shaping" else value_pooled.detach()
self.stash.win_prob_logits = self.win_head(wp_in)
```

Stashed at `last_win_prob_logits`; **never concatenated into `pi` or `vf`** — leak-safety, because
its label is the privileged future outcome.

**The label.** `agents/training/wrappers.py:485-493` at episode end:
`won = 1.0 if (b is not None and b.won is True) else 0.0`.
`WinProbLabelCallback` (`agents/training/win_prob_callback.py:75-102`) back-fills that scalar to
**every step of the finished episode, undiscounted (γ_win = 1), Monte-Carlo, never bootstrapped**;
`win_mask` is 1 only for steps whose episode terminated inside this rollout buffer, so the trailing
in-progress episode contributes nothing.

🚨 **DRAWS ARE LABELLED AS LOSSES.** `won is None` (a draw / the 250-turn timeout) falls through to
`0.0`. There is no third class and no mask-out. Every decision of a drawn episode trains the head
toward "loss" — while the REWARD scores that same episode at −35, i.e. **worse** than the −30 loss.
The head's label and the objective's terminal already disagree about draws today.

**The loss** (`instrumented_ppo/value_terms.py:20-86`):
`per = BCE_with_logits(logits, target, reduction="none"); loss = (per*mask).sum() / n_known`,
returning `None` when `n_known == 0`. Folded at `ppo.py:811-824` as
`win_prob_term = self.win_prob_coef * wp_loss`, tagged `"aux"`. Gate: mode ≠ `none` AND
`win_prob_coef ≠ 0`.

**`--win-prob-mode` values** (`clean_world.py:320-331`, STRUCTURAL + resume-immutable):

| value | module built | grad into trunk | mechanism |
|---|---|---|---|
| `none` | no | — | `win_head = None` |
| `read_only` | yes | **NO** | `value_pooled.detach()` at `extractor_forward.py:727` |
| `shaping` | yes | **YES** | the live tensor at the same line |

Production **`shaping` at coefficient 0.05**. As
[`design_winprob_behavior_coupling.md`](design_winprob_behavior_coupling.md) §1 establishes, that
"shaping" is *representation* shaping only — a feature subsidy in the trunk. **There is no gradient
path anywhere from predicting wins to choosing winning actions.**

### 1.6 `winprob_pbrs` — what route 1 does with the head

`--win-prob-pbrs-coef` (`agents/training/winprob_pbrs.py`) is trainer-side buffer augmentation
applied in `collect_rollouts` after collection and before `train()`:

```
r'(s,a,s') = r(s,a,s') + coef · ( γ·φ(s') − φ(s) ),   φ = σ(win-prob logit)
```

φ is computed in one batched `no_grad` forward and converted to numpy before touching
`rollout_buffer.rewards`, so **it carries no gradient, structurally**. Returns and advantages are
then recomputed, and PopArt (which reads `rollout_buffer.returns` at the top of `train()`) sees the
shaped stream. `--win-prob-pbrs-source` selects which model supplies φ; it resolves through the one
choke point `fixed_opponent_pool.resolve_model_ref`, so a bare run directory means that run's LAST
SNAPSHOT (`gen3_last_snapshot_resolution_v1`). Both flags default OFF and the module is not imported
at coefficient 0.

Its own docstring names the caveat this design turns into a deletion argument: the Ng shield assumes
a **fixed** φ, and ours is a head inside the network being trained.

### 1.7 Everything else hanging off `value_pooled`

| head | flag | production | trained on | detach |
|---|---|---|---|---|
| `QWinProbHead` | `q_winprob_mode` (`none`\|`read_only` — **no `shaping`, deliberately**) | **OFF / LATENT** | `--q-winprob-coef` (per-action counterfactual binomial NLL) + `--q-winprob-onpolicy-coef` (biased taken-action fallback) | **every** input, unconditionally |
| `CfEvidentialHead` | `cf_evidential` (+ `--cf-evidential-coef`/`-reg`) | OFF | Beta-Binomial marginal likelihood of rollout COUNTS | unconditional |
| twins B/C (`WinProbHead` ×2) | `cf_twin_heads` (+ `--cf-twin-coef`) | OFF | B: single-outcome labels; C: tight-MC labels; both + head A's on-policy BCE | unconditional |
| `ShadowValueHead` | `cf_shadow_critic` (+ `--cf-shadow-coef`) | OFF | `popart.normalize(mc_return)` — a value twin that never enters GAE | unconditional |
| — | `--cf-winprob-coef` | OFF | tight-MC counterfactual labels, applied to **`win_head` itself** | `cf_head_only` (default true) |
| — | `--td-aux-coef` | **0.0 / OFF** | `δ = (V(s_t) − r_t − γV(s_{t+1})) / popart.sigma`, both ends live | — |

`QWinProbHead` emits [B, 11] in action-space order from the **same per-action tokens the pointer
head scores**, with a zero-init `q_score` — eleven `P(win|s,a)` for one forward. It is the amortized
one-ply search leaf and it is already in the currency this design adopts.

### 1.8 The search leaf

`src/main/search_dividend/search.py:250-274` (`batch_scores`) reads
`policy.predict_values` and, separately, `features_extractor.last_win_prob_logits`:

```python
if mode == "value" or (wp is None and mode in ("auto", "win_prob")):
    return values, "value"
```

`--score auto` is the battery default and **silently falls back to the value head** when a run
trained none. `defensive.py:190-197` names why that is unacceptable for a defensive search:

> A default that can quietly become the losing arm is not a default.

Probe G measured it: `--defensive-leaf winprob` beat the played action by **+0.0219 [+0.0089,
+0.0364]**, where the scalar value head's **+0.0135 [−0.0007, +0.0280]** does not clear zero. So
`LEAVES = ("winprob", "value")` with `winprob` the default and `check_leaf` raising unless the
scorer delivered what the strategy asked for.

**The state of play in one sentence:** the quantity the search's best-measured leaf uses, the
quantity every calibration instrument can score, and the quantity the whole error taxonomy is
written in — is a 0.05-weighted side readout that no gradient reaches from the policy; while the
critic that actually assigns credit predicts a shaped, discounted, PopArt-normalized return that is
commensurable with nothing and whose head silently drifts under the normalizer.

---

## 2. WHY — what the record already says

Not a fresh argument; a collection of things this tree has already measured.

1. **The defect is the READOUT and the TARGET, not capacity** — `credit_assignment_and_value_errors.md`
   §4: the `DamageOperator` tracks true KO probability at slope +1.17, r = 0.995 on a constructed
   sweep while the win-prob head moves at slope +0.02 on the *same forward pass*; a **linear** probe
   of `value_pooled` decodes true win probability at **AUC 0.845** where the head scores 0.588.
   "Expressiveness (GLU, more layers) is acquitted; the target and readout are convicted."
2. **The distributional lever is dead as a strength lever.** Ledger rule: "the distributional head
   is an INSTRUMENT here, not a lever" — residuals are sub-Gaussian, no tail to re-weight.
3. **The search's best leaf is already this head** (probe G, §1.8), and the search-dividend line is
   the reason the owner wants a clean critic at all.
4. **The measured baseline says the head is well calibrated in the MEAN and starved of RESOLUTION**
   (§4.1) — which is the *good* precondition for promotion: promoting a head whose reliability is
   0.002 risks little level damage, and resolution is what a real training signal can add.
5. **`--defensive-leaf`'s existence is itself the argument.** A codebase that has to *measure* which
   of its two value readouts to trust, and then build `check_leaf` so the answer cannot silently
   change, has one value readout too many.

---

## 3. THE END STATE — every choice, with the argument and a recommendation

**One sentence:** the critic's output is `P(win) ∈ [0,1]`, trained on the terminal outcome; the
reward stream is that terminal and nothing else; PopArt, the distributional head and the shaped-return
currency are deleted; GAE runs in probability units at γ = 1.

### 3.1 The critic's parameterization

**RECOMMEND: one logit `z`, critic value `V(s) = σ(z)`.** The network's output stays a logit (the
BCE is `binary_cross_entropy_with_logits`, numerically the right thing) and the *value* consumed by
GAE, by search and by every readout is the probability.

Not the raw logit as V: GAE is a linear operator and a linear combination of logits is not the logit
of anything; advantages must be in the units the objective pays.

**Where the head physically lives.** `WinProbHead` and `ValueDistHead` are the same module up to
output width, both reading `value_pooled`. **RECOMMEND: keep `WinProbHead` and route
`_critic_value` to it**, rather than re-purposing `value_net`. Reason: `value_net` is fed
`latent_vf` through the assembler, and v96 (`gen3_critic_route_wave_v1`) already established that
`vf_combined` IS `value_pooled` — so the two paths carry the same content, and using the head that
already exists keeps the v89 orphaned-branch class unrepresentable.

### 3.2 Draws — say how

Three options:

| option | argument for | argument against |
|---|---|---|
| **(a) draw = 0 (a not-win)** | matches today's label; makes "P(win)" literally P(win); a proper scoring rule for the outcome the ladder pays | cannot express "worse than a loss", which today's `--draw-penalty −35` does |
| (b) draw = 0.5 | the natural probability reading of a tie | the objective does not pay 0.5 for a draw, so the critic would be systematically wrong exactly where stalling tempts — this project's recorded stall pathology |
| (c) third class, masked out | no wrong label | a drawn episode's ~250 decisions then produce NO learning signal at all, and they are the episodes where the signal is most needed |

**RECOMMEND (a): a draw is a not-win, `y = 0`, and the fall-through becomes EXPLICIT.** Today
`won is None ⇒ 0.0` is an accident of a boolean test; it should be a named branch with a comment
saying a draw is scored as a not-win *by decision*, and a `train/draw_rate` scalar so the frequency
is visible rather than inferred.

🚨 **Name the cost, do not bury it.** A critic in [0,1] **cannot represent "a timeout is worse than
a loss"**. Today's terminal does (−35 vs −30), and the clean-world flag help calls that ordering
load-bearing: without it "a losing agent's optimal play is to run out the clock". Under this design
the anti-stall pressure must come from somewhere else. It has two candidates, both already built:
the **deadline clock** in the obs (v67 `gen3_deadline_clock_v1`, three scalars — the fix that took
positive-V-on-timeout-loss from 13/14 to 2/9) and the **`no_progress_tax` BIAS term**. See §3.3 and
gap **B4**.

### 3.3 The reward stream — PBRS survive, or delete?

**The case for KEEPING PBRS.** Every PBRS term is policy-invariant by construction (§1.1), so it
*cannot* change the optimal policy; it only speeds credit assignment, which on a problem of ~1 bit
per ~40 decisions is exactly what is scarce. Deleting a free accelerant to satisfy a definition is a
bad trade.

**The case for DELETING it.** The identity this whole design rests on is `V(s) = P(win | s)`. Under
PBRS with Φ(terminal) = 0 and γ ≈ 1, the return from `s` telescopes to `R_T − Φ(s)`, so the critic
that minimizes its loss learns `V_game(s) − Φ(s)` — **not P(win)**. The identity is not approximately
broken, it is exactly broken by a known function. You would then have to add Φ back at every
consumer (the search leaf, the calibration gate, every offline instrument), which reintroduces the
two-currency boundary this design exists to remove, with a *learned, drifting* Φ instead of a
normalizer. And the invariance argument cuts both ways: **if PBRS cannot change the optimum, it
cannot be why a run succeeds** — it changes dynamics only, so deleting it costs speed, not
correctness.

**RECOMMEND: DELETE the hand shaping — `--no-hand-shaping`.** The composition becomes
**1 TERMINAL + 0 PBRS + 0 BIAS**. Three supporting facts:

* The flag **already exists and is already resume-immutable and value-checked**
  (`clean_world.py:19-29`), built in the ai_v12 wave-A landing. This is not new machinery.
* It exists precisely because `--no-all-shaping-pbrs` cannot get you there — that flag is *also* the
  BIAS class's master gate, so disabling it silences 5 potentials while **reviving 25 BIAS terms**.
* `--victory-value 1.0` is the matching half, and its help text already anticipates this exact
  composition: *"Pair `--victory-value 1.0` with `--draw-penalty -1.0` (draw = loss) and make stall
  rate + mean game length a PRIMARY endpoint."*

**The exception to carry, and it is the one risk of the recommendation.** `--no-hand-shaping` zeroes
`no_progress_tax` along with the rest of the BIAS class, and §3.2 just removed the terminal's
anti-stall ordering. That is **two** stall defences dropped in one step, leaving only the obs
deadline clock. **RECOMMEND: `--no-hand-shaping` for the arm, with `no_progress_tax`
independently re-armable** (gap **B4** — no flag re-arms it today) held as the contingency, and
**stall rate + mean episode length as a PRIMARY, pre-registered, kill-condition-bearing endpoint**,
not a monitored one.

### 3.4 The value target and the loss

**RECOMMEND: BCE-with-logits against the terminal outcome, MC, undiscounted, on every decision of a
finished episode — i.e. exactly today's `win_prob` loss — promoted to be the value loss.**

The critical consequence, stated plainly: **the critic's target stops being bootstrapped.** Today's
critic regresses `returns = advantages + values`, which is built from itself; the win-prob label is
a pure Monte-Carlo outcome. That removes the self-referential error propagation
`credit_assignment_and_value_errors.md` §2 describes, and it removes the mechanism by which "a
systematic bias self-reinforces until some ground truth interrupts it".

It also costs the variance reduction bootstrapping buys. **This is a real trade and it is the single
biggest open risk in the design.** Two mitigations, both already in the tree:

* GAE still bootstraps for the ADVANTAGE (λ < 1) even when the critic's own target is MC. The
  critic's target and the advantage estimator are separate choices, and this design changes only
  the first.
* `--td-aux-coef` (§3.9) is precisely the consistency pressure that puts the bootstrap back as an
  *auxiliary* rather than as the target, which is what its ledger-C5 measured ceiling
  ("consistency suppresses noise but cannot create signal") makes it good for.

**Masking.** `win_mask` today excludes the trailing in-progress episode. Under promotion that means
**a fraction of every rollout produces no critic gradient at all** — the tail of each env's
trajectory. At `n_steps 2048` against the ~16–29 mean turns the eval cycles record this is a small
fraction, but it is not zero and it is not currently measured (`coverage` is logged by the win-prob loss; gap **M3**).
The alternative is to bootstrap the trailing segment from the current critic — which reintroduces
exactly the self-reference. **RECOMMEND: keep the mask, publish the coverage as a first-class
scalar, and pre-register a threshold below which the arm is inconclusive.**

### 3.5 PopArt — precisely what happens

**RECOMMEND: OFF (`--no-use-popart`).** The reasoning is not "V ∈ [−1,1] so it's small enough"; it
is that **PopArt's job no longer exists**:

* PopArt exists because *"RL returns drift in scale as the policy improves"* and raw-scale value
  gradients swamp the shared trunk (measured: `value_share` 1.0 → 0.39, ELO 1400 → 1850 when
  shipped). With a terminal-only ±1 objective the return is **bounded by construction and
  stationary for the life of the run** — μ and σ cannot drift because the payoff set is fixed at
  {win, not-win}. There is no scale to track.
* The BCE gradient with respect to the logit is `σ(z) − y ∈ [−1, 1]` regardless of anything — the
  same O(1) anti-swamping protection PopArt was engineering, obtained for free from the loss.
* Deleting it **collapses the two currencies into one**. The popart note's "one exchange window"
  becomes zero exchange windows; `_denorm` disappears; every cross-head comparison, distillation
  target and GAE hookup stops needing to name a currency.
* It removes §1.4's hazard 1 outright: there is no POP surgery to be missing from the critic head.

**Two consequences to handle rather than assume away:**

* `use_popart` false **un-forces `--clip-range-vf none`**. `combination_checks.py:295-303` refuses a
  clip only when PopArt is on. So the flag becomes live again — see §3.6.
* Several terms divide by `popart.sigma` (`td_aux` at `aux_terms.py:92`) or normalize through it
  (`cf_shadow` at `cf_terms.py:448`). **Every one must be audited for the `popart is None` path**
  before an arm runs — gap **B2**. UNVERIFIED whether those paths are exercised by any current
  test.

### 3.6 The PPO knobs

| knob | production | recommendation |
|---|---|---|
| `vf_coef` | 0.5 | **KEEP the flag, KEEP it resume-immutable, RE-TUNE the value.** It now multiplies a BCE, not an MSE/CE over 51 atoms; 0.5 does not transfer between loss families, and the ratio it arbitrates (`grad/value_policy_logratio`) is exactly what silently changing it would break. **RECOMMEND: a startup line naming the loss family beside the coefficient**, so a resumed run cannot inherit a number tuned for a different objective without saying so. |
| `clip_range_vf` | forced `none` by PopArt | **KEEP OFF for the first arm** (one variable at a time), but it is now *legal and meaningful* — clipping a probability is well defined where clipping an un-normalized return was not. Register it as a later ladder rung, not a launch flag. |
| `--draw-penalty` | −35.0 | **REFUSE it under the new objective.** With the terminal at {0,1} there is no −35 to give. Silently ignoring it is the failure class `checkargs` exists to end. **RECOMMEND a `combination_checks` entry**: `--draw-penalty` other than the value implied by §3.2 is an error when the win-prob critic is on. Gap **B5**. |
| `--victory-value` | 30.0 | **1.0**, with `--draw-penalty -1.0` per its own help text — or, if §3.2's "draw = not-win" is taken literally in the label, the terminal pays `+1 / 0 / 0`. **RECOMMEND: keep the reward at ±1 and let the LABEL be the `{1,0}` map**, so `y = (r_T + 1)/2`; that keeps `victory_value`'s existing symmetric semantics and one guard rail (`draw_penalty ≤ −victory_value`) intact instead of inventing a second convention. |
| `--value-tail-weight` | 0.3, ACTIVE-but-inert (§1.4) | **DELETE.** It is the hand-rolled cousin of Emphatic TD's interest function, and the design rule already banked in `credit_assignment_and_value_errors.md` §4 forbids its shape here: at the decision boundary, relevance and label NOISE arrive together, so "care more" must be *more samples*, "never as larger per-sample weights, which amplify exactly the noise you cannot afford". A per-sample weight on a Bernoulli likelihood is that antipattern exactly. |
| `--adaptive-batch` / per-term noise scale | on/off per run | **FIX the grouping first** (§1.4 fact 4): with the critic promoted, its loss must be tagged `"value"` again, or `train/noise_scale_value` keeps describing a zero-weighted term. Gap **B3**. |

### 3.7 The `--win-prob-pbrs-*` pair

**RECOMMEND: DELETE both `--win-prob-pbrs-coef` and `--win-prob-pbrs-source`, and delete
`agents/training/winprob_pbrs.py` with them.**

The argument is that promotion *subsumes* them, and subsumes them in the stronger form. Route 1 adds
`coef·(γφ(s′) − φ(s))` to the reward, where φ is the win-prob head. But `γV(s′) − V(s)` is precisely
the TD residual the critic already computes and GAE already turns into the advantage. Once
`V ≡ φ`, route 1 is **adding the advantage to the reward and then computing the advantage of that** —
double counting, with the added liability that its Ng shield is at its structurally weakest (the
theorem assumes a fixed φ; ours is the head being trained, and route 1's own docstring says the
lever "wants a MATURE φ" and that "enabling it on a fresh run tests the shield's worst case").

Route 1 was the right mechanism for coupling a barometer to behaviour *while it remained a
barometer*. This design removes the barometer.

`--win-prob-pbrs-source` also resolves a model reference through `resolve_model_ref`; deleting the
flag removes one consumer of that choke point and nothing else.

> **🚨 OWNER AMENDMENT, 2026-09-06 — the recommendation above is SUPERSEDED. Refused, not deleted:
> "self path to be deleted at the default flip; frozen path kept refused for one generation."**
> Implemented in `gen3_winprob_critic_mode_v1`; `agents/training/winprob_pbrs.py` and everything
> behind `--win-prob-pbrs-source` are left INTACT.
>
> **(a) The SELF-φ path is refused for the REASON above.** Source unset (or pointing at this run's
> own head) under `--critic winprob` exits with §3.7's double-counting argument stated verbatim in
> the message: with `V ≡ φ`, `γφ(s′) − φ(s)` IS the TD residual GAE already turns into the
> advantage, so route 1 adds the advantage to the reward and then takes the advantage of that.
>
> **(b) A FROZEN external source is ALSO refused in this build — but as DEFERRED, and the message
> says so in those words.** Exact Ng invariance *does* hold for a fixed φ: the critic then learns
> `P(win) − φ_frozen`, which is recoverable at inference by adding φ back. So it is held for a
> later FROZEN-φ ablation, not judged wrong, and the ledger's registered SPARSE / SELF-φ / FROZEN-φ
> ladder (`designs/ai_v12/launch_runbook.md`) stays **one edit** away. A refusal that read "this is
> wrong" would retire a live research rung by accident.
>
> **(c) It is a BOOLEAN, not a scalar** (owner, same day: *"it seems like it would be a boolean, not
> a scalar"*). The frozen rung is declared NOW as **`--win-prob-pbrs-frozen <run|zip>`** — on/off by
> presence, **no coefficient** — with the coefficient fixed internally and printed at startup.
>
> **THE DERIVATION, from the code's OWN currency.** Under `--critic winprob` the terminal is the
> WIN INDICATOR (`+victory_value` on a win, `0.0` otherwise) at `--victory-value 1.0`, so the
> undiscounted return is `1{win}` and `V(s) = P(win|s) ∈ [0,1]`. φ = σ(win-prob logit) is therefore
> **already in the value currency**, and the currency-matched shaping coefficient is exactly **1.0**
> — the potential is one unit of V per unit of V. (The alternative currency the owner names gives
> the same answer scaled: with a ±1 terminal and `V = 2p − 1`, φ = p needs a coefficient of **2**.
> This tree's terminal is the indicator one, hence 1.0. Writing it as a ±1 terminal instead would
> also break `successor_potential`'s `φ(terminal) := 0` convention, which is the correct zero for a
> [0,1] potential and the MIDDLE of a [−1,+1] one.)
>
> `--win-prob-pbrs-coef` is refused under `winprob` with the message *"no coefficient: the potential
> is currency-matched; the dose ladder belonged to the shaped critic."* **Nothing under `--critic
> shaped` changes** — the old pair keeps its meaning, its dose ladder and its `--win-prob-pbrs-source`
> spelling until the default flip.

### 3.8 `value_dist_head` — delete, or re-parameterize over P(win)?

**RECOMMEND: DELETE the head and its whole flag family. Do NOT re-parameterize it over P(win).**

The decisive argument is a counting one, and it is stronger than the usual "we measured it null":

> Under a terminal-only objective, **the return from any state takes at most two values** (win / not-win
> — three, if draws are ever separated). A categorical distribution over a two-point support is a
> **Bernoulli**, and a Bernoulli is fully specified by its mean. The 51-atom head would therefore not
> be a richer parameterization of the new target; it would be the *same* parameterization with 50
> redundant degrees of freedom and a discretization error.

Everything else agrees: the ledger already rules the distributional head an INSTRUMENT and not a
lever here (sub-Gaussian residuals, no tail to re-weight), and the fixed-support-versus-drifting-scale
tension the popart note describes dissolves along with PopArt.

**The cost, named.** The awareness stack reads the distribution — `knew_by_turn`, `blind_loss`, "the
stall signature of bottom-atom mass under a positive mean", and the PIT calibration audit
(`pit_mean`, `coverage80`) are things a scalar V cannot express. Under a Bernoulli critic these
become functions of the single P(win): PIT collapses to the reliability curve §4 already computes
(which is strictly *more* informative for a binary outcome), and the bottom-atom-mass stall signature
becomes "P(win) low" — coarser. **UNVERIFIED how many live consumers read `last_value_dist_logits`
or `value_dist` from traces**; enumerating them is gap **A2**, and any that survive need a stated
replacement before the deletion lands.

**What if the objective later stops being terminal-only?** Then the return regains a continuum and a
distributional head regains a job. That is a re-entry condition, and it should be written into the
deletion the way `value_intent`'s re-entry condition survived its own deletion at v96.

### 3.9 The other value-adjacent flags

| flag | verdict | reason |
|---|---|---|
| `--td-aux-coef` | **KEEP** — and it gets *more* useful | It is the bootstrap put back as a consistency auxiliary rather than as the target (§3.4). Its residual `(V(s) − r − γV(s′))` is now in probability units and the `/ popart.sigma` divisor disappears (gap **B2**). Its measured ceiling — "consistency suppresses noise but cannot create signal" — is exactly the right job description for an auxiliary beside an MC target. **Still OFF for the first arm.** |
| `--cf-winprob-coef` (+ `--cf-head-only`, `--cf-label-likelihood`, `--cf-label-lag-steps`, `--cf-records*`) | **KEEP, and note the promotion** | This term applies tight-MC counterfactual labels **to `win_head` itself**. Once `win_head` is the critic, it stops being a side-head experiment and becomes *the label factory feeding the critic directly* — which is the standing plan's destination reached from the other side. `--cf-head-only` (default true) becomes the meaningful trunk-exposure dial. |
| `cf_twin_heads` (+ `--cf-twin-coef`) | **KEEP** | Heads B and C become twins **of the critic**, so the plan's within-run paired read (B−A isolates coverage, C−B isolates variance reduction) becomes a read on the live value function instead of on a proxy. Strictly more valuable after promotion. |
| `cf_shadow_critic` (+ `--cf-shadow-coef`) | **REPURPOSE, do not delete** | Its stated job is "the staged PROMOTION PATH for critic surgery, not the surgery" — accumulating shadow-vs-live divergence as a published number. After promotion that job is done in its current direction, but the *mirror* is newly wanted: a shaped-return shadow beside the P(win) critic, publishing what the old objective would have said. One line changes (the label source); the head, the flag and the meter survive. |
| `cf_evidential` (+ coef, reg) | **KEEP, OFF** | Axis 5 of the error taxonomy — the model's confessed uncertainty over P(win). Its pre-registered meter (`width_vs_blur_spearman`) is unaffected by promotion and its head reads a detached `value_pooled` either way. |
| `q_winprob_mode` (+ two coefs) | **KEEP, OFF for arm 1 — the natural arm 2** | Eleven `P(win|s,a)` from one forward, now in the **same currency as the critic**, which is what it always wanted to be. It is the amortized search leaf and the direct answer to SI-2's finding that "knowing the opponent's move is cheap; valuing the resulting state is what is missing". Deliberately not composed with the critic change (one variable). |
| `--value-from-dist` / `--allow-value-from-dist-change` | **DELETE** | The flag names a choice between two critics; after §3.8 there is one. |
| `--value-dist-mode` / `-bins` / `-vmin` / `-vmax` / `--value-dist-coef` | **DELETE** | With their head. `check_value_dist` goes with them. |
| `--value-tail-weight` | **DELETE** | §3.6. |
| `--win-prob-pbrs-coef` / `--win-prob-pbrs-source` | **DELETE** | §3.7. |
| `--win-prob-mode` | **KEEP the flag, CHANGE its meaning — or delete it** | Once the head is the critic, `none` is unrepresentable (a run with no critic) and `read_only` means "the critic's gradient does not reach the trunk", which is a genuinely interesting and genuinely different arm — it is the value-vs-representation split the whole `belief_grad_mode` family already parameterizes. **RECOMMEND: keep the flag, drop `none`, and rename it in help to say it now governs the CRITIC's trunk exposure.** A structural flag whose legal set shrinks is a `check_compatible` change either way. |
| `--win-prob-coef` | **DELETE, replaced by `vf_coef`** | Two coefficients on one loss is the ambiguity that made `_ce_w = self.vf_coef if value_from_dist else self.value_dist_coef` necessary. One critic, one coefficient. |

### 3.10 The search leaf

**RECOMMEND: `--score` collapses to a single legal value; `auto` is DELETED; `--defensive-leaf`
keeps its two-value shape only as long as a second readout exists, i.e. not past this change.**

* `search.py:270`'s `if mode == "value" or (wp is None …): return values, "value"` must become a
  **refusal**. Under the new critic `predict_values` and the win-prob head are the same number, so a
  fall-back is either a no-op or a bug, and the class of "a default that can quietly become the
  losing arm" is what `check_leaf` was written against.
* `check_leaf` becomes vacuous but **KEEP it** — it is the guard that makes adding a second readout
  later safe, and deleting a guard because it currently has nothing to catch is how the
  choice-reject allowlist entry outlived its own fix.
* Every committed search-dividend cell was measured with `--defensive-leaf winprob`, so the
  **existing cells transfer** — they were already measuring this critic. That is a real and
  underappreciated benefit: the defensive-search operating point (`--defensive-wp-margin 0.15`, the
  futility rule, the contested deadline) is calibrated on P(win) and does not need re-deriving.

> **DECIDED AND BUILT — `gen3_winprob_critic_mode_v1`, 2026-09-06 (gap B10 closed).** Three rulings,
> and the first is a deliberate departure from "`auto` is DELETED" above.
>
> 1. **`auto` dies as a BEHAVIOUR, and survives as a spelling that RESOLVES.** `--score auto` is
>    the CLI default, so deleting it outright would make the flagless invocation —
>    `python -m main.search_dividend <ckpt>` — a usage error on every winprob checkpoint, and the
>    battery is required to run end to end on this critic. What made `auto` dangerous was never the
>    spelling but the **fall-back**: it prefers the win-prob head *and silently drops to the value
>    head* when the run trained none, which is exactly the substitution `check_leaf` exists to
>    catch. On a winprob policy that fall-back is unreachable by construction — one readout, so
>    `auto` can only ever resolve to it. So it resolves to `win_prob` and the resolution is
>    **ANNOUNCED** (`[search_dividend] critic=winprob: --score auto -> win_prob …`), never left for
>    a reader to infer from silence. An EXPLICIT `--score value` RAISES.
> 2. **`--defensive-leaf` collapses to `winprob`; `value` is REFUSED.** Probe G's control arm does
>    not exist on this critic — `predict_values` and the win-prob head are the same number, so a
>    `value` arm would be the winprob arm wearing a different label and would report a control that
>    was never run. It is refused rather than aliased. **No ledger retirement is owed**: `shaped`
>    keeps both leaves and can still re-run G's control, so the arm is retired *on this critic*,
>    not deleted.
> 3. **The refusal happens at STARTUP, not in `batch_scores`** — and this is the part the design
>    did not anticipate. `SearchEngine.choose` catches every exception from a search as a counted
>    `search_error` that plays the policy action (correct: a search failure must never cost the
>    battle). A deep `ValueError` on the flagless default would therefore turn *every decision* into
>    a policy fallback and report the arm's dividend as ≈0, with nothing in the log saying the
>    battery never searched — a config error arriving through the error path as a silent null
>    result. `defensive.resolve_for_critic` runs once in `__main__.main` off the LIVE policy's
>    `_critic_mode` before any `SearchConfig` is built; `batch_scores`'s own check stays as the
>    library-level backstop for callers that are not the CLI.
>
> `check_leaf` and `LEAVES` are **KEPT** exactly as this section asked.
>
> **Proven end to end** on a CPU cell against a `--critic winprob` smoke checkpoint (`--arm base
> --arm honest --budget 1 --games 2 --opponents self --max-depth 1`): 8 games, 111 decisions
> searched on the `honest` arm, `changed=0.55`, zero `search_error` fallbacks, mirror + ELO blocks
> rendered. The output is in the B10 handoff note, and the numbers are a plumbing demonstration on
> a 10k-step toy — **not a dividend measurement**.

### 3.11 GAE γ and λ in the new currency

**γ: RECOMMEND 1.0** (from 0.9999). The argument: γ < 1 exists to make an infinite-horizon return
finite and to express time preference. Neither applies. The episode is *hard-terminated* at 250
turns (`MAX_TURNS`, which is also the forfeit deadline), and the objective — "win" — carries no time
preference at all; a win on turn 200 is worth exactly a win on turn 20. **At γ = 1 with a
terminal-only reward, `V(s)` is EXACTLY `P(win | s)`** — the identity the entire design rests on
holds without an approximation term. At γ = 0.9999 over 250 turns the discount factor is 0.975, so
the identity would be off by ~2.5% at the start of a long game, which is the same order as the ECE
this design is trying to improve.

Two things to verify before taking it (gap **B6**): SB3's truncation handling
(`next_non_terminal`) at γ = 1 on a 250-turn cap, and the `PBRS_GAMMA == model.gamma` assert, which
must not fire when PBRS is deleted.

> **BOTH VERIFIED, and the first one was BROKEN — `gen3_winprob_critic_mode_v1`, 2026-09-06.**
> Measured on a real bridge battle with `StallConfig.threshold` lowered: the cap FORFEITS
> (`gen3_env.action_to_order`), Showdown answers `|win|<opponent>` so `battle.won` is **`False`**
> (a plain loss — NOT the `None` a tie gives), and `PokeEnv.calc_term_trunc`
> (`poke_env/environment/env.py:951`) then returned **`terminated=False, truncated=True`**, because
> its test is "was exactly one side wiped" and a forfeit leaves six mons alive a side. SB3
> (`sb3_contrib/ppo_mask/ppo_mask.py:251-260`, and `async_vec_env.collect_rollouts_async:216-219`)
> therefore bootstrapped `r += γ·V(s_last)` — at γ = 1 with a terminal-indicator reward of 0 that
> is `0 + V(s_last)` against a target of `V(s_last)`, a TD error of **identically zero**, removing
> the timeout from the loss entirely. G7's stall rate would have been a kill condition on a signal
> the critic never received.
>
> **THE GENERAL FACT, which is what makes the fix unambiguous: this env never truncates in the SB3
> sense at all.** `calc_term_trunc` sets EITHER flag only when `battle.finished`, so a `truncated`
> out of it never means "an episode was cut off mid-flight" — the one thing `TimeLimit.truncated`
> is supposed to mean — it means "finished, and not by a wipe": the 250-turn cap, or a genuine
> tie. Both are OUTCOMES. Under `winprob` they are re-labelled TERMINAL by
> `wrappers.resolve_episode_end`, which is the design's intent and the only reading consistent with
> `V(s) = P(win | s)`: a cap loss really is a state whose win probability is 0.
>
> `shaped` is UNCHANGED, and what it does is worth stating rather than leaving implicit: a cap
> forfeit and a tie both bootstrap `0.9999·V(s_last)` on top of a terminal reward that already paid
> `--draw-penalty` — two wrongs that partly cancel, and a `shaped`-era question this design does
> not reopen.

**λ: RECOMMEND leaving it exactly where it is for the first arm.** `credit_assignment_and_value_errors.md`
§2 states the dial precisely — "the more you trust the critic, the lower λ can go; the worse the
critic, the more λ is protecting you". This design's whole thesis is that the critic becomes more
trustworthy, so **lowering λ is the PAYOFF, not part of the experiment**. Moving it in the same arm
would confound the two. Register a λ ladder as the first follow-up, gated on §4's calibration gate
passing.

---

## 4. THE CALIBRATION GATE — what must hold before a long run is trusted

A 3-day 75M-step run is ~72 GPU-hours. The gate exists so that commitment is made against a number,
not a hope. **All of it is offline, model-free, and runs on eval traces the run writes anyway.**

### 4.1 The bar to beat — MEASURED, committed

[`designs/research_state/measurements/winprob_critic_baseline_2026-09-06/`](../research_state/measurements/winprob_critic_baseline_2026-09-06/),
`ai_v9_59_R2ACTION_0827`, 474 battles / 12,694 decisions, **selection-reweighted** to the eval
cycle's own recorded win rates (see the warning in §4.2):

| step | stratum | Brier [95% CI] | **skill** [95% CI] | ECE | **reliability** | **resolution** | unc |
|---|---|---|---|---|---|---|---|
| 26,000,016 | all | 0.1207 [0.1022, 0.1427] | **+0.336** [0.266, 0.391] | 0.0249 | 0.0013 | **0.0618** | 0.1817 |
| 26,000,016 | bot | 0.0767 [0.0612, 0.0973] | +0.291 [0.185, 0.374] | 0.0395 | 0.0027 | 0.0337 | 0.1082 |
| 26,000,016 | pool | 0.1591 [0.1277, 0.1950] | +0.290 [0.189, 0.368] | 0.0667 | 0.0064 | 0.0711 | 0.2242 |
| 28,000,032 | all | 0.1216 [0.0979, 0.1473] | **+0.265** [0.170, 0.343] | 0.0349 | 0.0020 | **0.0445** | 0.1653 |
| 28,000,032 | bot | 0.0723 [0.0565, 0.0920] | +0.222 [0.101, 0.306] | 0.0228 | 0.0012 | 0.0215 | 0.0929 |
| 28,000,032 | pool | 0.1705 [0.1269, 0.2203] | +0.208 [0.067, 0.319] | 0.0875 | 0.0103 | 0.0530 | 0.2152 |

**What the baseline says, and it shapes the gate:** reliability is ~0.002 — the head is *already
calibrated in the mean*. Resolution is 0.062 / 0.045 against an available uncertainty of 0.182 /
0.165, i.e. it separates **27–34%** of what there is to separate. **So the gate must be a
RESOLUTION gate.** A promotion that improves ECE and leaves resolution flat has moved the meter that
was never the disease — the "wrong-meter trap" the calibration plan already recorded once (label
heads re-centred without sharpening).

### 4.2 🚨 The selection confound, and why the raw numbers invert the verdict

The eval recorder's quota is loss-enriched: the captured slice's outcome rate is **0.46** while the
same cycles' `eval_results.jsonl` records **0.901 vs bots / 0.702 vs pool**. Unweighted, the same
traces read ECE **0.237 / 0.281** and skill **+0.071 / −0.080** — i.e. "grossly optimistic, and at
28M worse than the base rate". Both readings are artifacts. **Every number in this gate is
selection-reweighted, and any future quote of a raw one must say so.** The prober's `calibration`
verb already carries the sibling self-diagnosis (`captured_win_fraction`, `bias_on_wins`,
`bias_on_losses`) for the same reason on the scalar critic.

### 4.3 The gate, pre-registered

Read on the new run's own eval traces, per checkpoint, **`bot` and `pool` separately — never
pooled**, cluster-bootstrapped over battles.

| # | criterion | bar | why this bar |
|---|---|---|---|
| **G1** | **resolution** | strictly greater than the matched-stratum baseline, CI clearing it | the disease (§4.1); the primary endpoint |
| **G2** | **reliability** | per stratum, **no worse than the same-stratum baseline** (≤ 0.005 is ASPIRATIONAL — see the ruling below) | promotion must not *un*-calibrate the head; the risk is that a head steering the policy that generates its own labels drifts |
| **G3** | **ECE** | per stratum, **no worse than the same-stratum baseline** (≤ 0.05 is ASPIRATIONAL — see the ruling below) | a reader-facing bound, and a genuine no-regression clause once it is stated against the predecessor rather than against a number |
| **G4** | **skill** | > 0 with the CI clearing 0, both classes | the floor: a critic no better than the base rate cannot assign credit |
| **G5** | **`sd_true_excess`**, floor-subtracted, per population | not worse than the era's recorded value | the standing plan's headline meter. ⚠️ **gap M1** — not yet runnable from traces alone |
| **G6** | **the MIRROR TABLE** | no cell crossing 0.50 | the plan's behavioural-resolution meter; a crossing cell is the wake-search signal. ⚠️ **gap M2** |
| **G7** | **stall rate + mean episode length** | no increase over the era, pre-registered threshold | §3.2/§3.3 removed two anti-stall defences; this is a KILL condition, not a monitor |
| **G8** | **`win_mask` coverage** | ≥ a pre-registered floor | §3.4 — a rollout fraction with no critic gradient. gap **M3** |
| **G9** | **capacity `value_pooled` PR** | should RISE off the 2.5 steady state | the plan's third meter; already runnable (`python -m main.capacity`) |

#### 🚨 OWNER RULING 2026-09-06 — G2 and G3 are PER-STRATUM RELATIVE bars

**The finding** (reported by `gen3_critic_gate_v1`'s own self-comparison, and now pinned by
`critic_gate_test.py` against the committed artifact): §4.3's absolute G2 (reliability ≤ 0.005)
and G3 (ECE ≤ 0.05) bars **are already breached by the baseline of §4.1 on the `pool`
stratum** — reliability **0.0064 / 0.0103** and ECE **0.0667 / 0.0875** at 26M / 28M — while the
G3 row above called itself a no-regression clause "the reweighted baseline already passes". That is
true **pooled** and true on `bot`; it is **false on `pool`**, and both criteria are registered over
"in both classes". As written, the arm had to clear a bar its own predecessor never cleared, which
is not a no-regression clause but a new and unmet requirement smuggled in as one.

**The ruling.** G2 and G3 are **per-stratum RELATIVE** bars. On each gated stratum (`bot`, `pool`,
never pooled) the arm's reliability and ECE must be **no worse than the baseline's same-stratum
value at the matched checkpoint** — the baseline's own row at that step where the artifact carries
it, and otherwise (the ordinary case, since a new arm's step grid will not coincide with the
baseline's at all) the artifact's steps reduced by `--baseline-reduce`, defaulting to the **LAST**
for these two. G1 keeps `max`, the strictest resolution to beat, and does not step-match: it is a
single pre-registered bar by design. Every row records which of the two it got. **The matched
half is load-bearing, not a nicety** — reduced alone, the arm's 26M row is judged against the
baseline's 28M value, and a generation compared with ITSELF is then reported as inferior to itself
(measured: `bot` reliability 0.0027 [0.0013, 0.0080] against 28M's 0.0012 reads FAIL). Read against
the arm's cluster-bootstrap CI:

* **PASS** if the arm's point estimate is **≤** the baseline's value, **or** if the arm's CI
  **contains** that value — *non-inferiority*, and deliberately never a claim that the arm is
  better;
* **FAIL** only when the arm's **whole CI sits above** the baseline's value.

A non-finite CI cannot support a non-inferiority claim, so there the point estimate alone decides
and the row says so — a missing interval never converts a worse point estimate into a pass.

**The absolute numbers stay, and stay printed — as ASPIRATIONAL targets that gate nothing.** Every
row carries whether the arm meets them *and whether the baseline does*, so the distance from the
number §4.3 wrote down remains visible without it being a bar an arm can fail on. **G1 (resolution,
the primary endpoint) and G4 (skill) are unchanged**; the promotion is still judged on whether it
SEPARATES better, which is the disease §4.1 identified.

*Implementation:* `main.critic_gate_design.RELATIVE_BARS` / `RELATIVE_RULE` /
`OWNER_RULING_2026_09_06` (the rule as DATA, read by the tool and asserted by the test),
`main.critic_gate.relative_verdict` (the three clauses), `baseline_bar(..., at_step=)` (the matched
checkpoint, with `matched` recorded and printed), and a report column naming **which clause decided
each row** beside the baseline value it was decided against.

*Verified* on `ai_v9_59_R2ACTION_0827` compared with itself: **G2 and G3 PASS on every stratum**
(every delta exactly 0.0000, every row decided by `point <= base`) while **G1 stays false
everywhere** — a generation is non-inferior to itself and does not out-resolve itself, so the
composed verdict is §5.5's falsification clause, which is the correct reading of that comparison.

**G1–G4, G7 and G9 are runnable today** (G1–G4 by the instrument built alongside this design; G9 by
the capacity battery). **G5, G6 and G8 are gaps.** A gate with three unrunnable criteria is a gate
that will be quietly reduced to the runnable ones under time pressure, so §6 ranks them.

### 4.4 The instruments — what they read, and what is missing

| instrument | reads | computes | can it score this critic? |
|---|---|---|---|
| `python -m main.prober.query calibration` | recorded **`values`** + rewards ⇒ V(s) vs realized discounted return G(s), binned by V | reliability curve in *reward* units, `bias_on_wins` / `bias_on_losses` / `captured_win_fraction`, and the `critic_overvalued` vs `lost_position` split of `falsify_scan`'s NEUTRAL bucket | **NO today** — it never touches `win_probs` and has no notion of outcome probability. **After promotion, YES and for free**: `values` *is* P(win), and its reliability curve becomes an outcome calibration in the right units. Its `overvalue_tau` (5.0, reward units) must be re-scaled — gap **B7**. |
| `python -m main.scaffolding_gauge` | `values` **and** `win_probs` + per-battle outcome, per step, with an `--opponent` filter | rank gauge (Spearman), calibrated-affine gauge (`rms`, `bias`, `brier_head`, `readout_penalty`), the db9bb5c constancy row; cluster-bootstrap CIs over battles | **partially, and now fully.** It read the right columns but only as a *divergence* between the two readouts. `--reliability` / `--reliability-reweight` (built in this pass) add the Brier / skill / ECE / Murphy split against the truth, stratified by opponent class. **After promotion the gauge's own subject dissolves** — with one readout there is nothing to diverge from, so §6 **A3** retires the two divergence gauges and keeps the reliability half. |

**Built in this pass** (deliverable 4), and what produced §4.1:

* `agents/training/scaffolding.reliability_table(win_probs, outcomes, bins=, weights=)` — Brier,
  Brier skill score, ECE, MCE, the Murphy REL−RES+UNC split with its binning residual reported, and
  Kish `ess`.
* `main.scaffolding_gauge --reliability [--reliability-bins N]` — per step, stratified
  `all` / `bot` / `pool` / per-opponent, cluster-bootstrap CIs over battles. Opt-in: the default
  JSON and render are unchanged, pinned by a test.
* `main.scaffolding_gauge --reliability-reweight` + `true_win_rates()` / `selection_weights()` —
  the §4.2 correction, read from the run's own `eval_results.jsonl`, **refusing** rather than
  falling back when the true rates cannot be resolved, and reporting weight-0 strata by name.

---

## 5. MIGRATION

### 5.1 Version

Per `src/agents/model/CLAUDE.md`'s playbook this is a **structural change**, not an optional feature:

| what | from | to | why |
|---|---|---|---|
| `MODEL_CONFIG_VERSION` | 107 | **108** | fields removed (`value_dist_*`, `value_from_dist`, `value_tail_weight`, `win_prob_coef`) and `win_prob_mode`'s legal set shrinks |
| `ARCH_SIGNATURE` | `gen3_critic_route_wave_v1` | **new** (e.g. `gen3_winprob_critic_v1`) | `value_dist_head`'s parameters leave the `state_dict`; the critic route changes |
| `MIGRATION_FLOOR` | 96 | **108** | the surviving code cannot rebuild a `value_dist_head`, so a pre-108 config must be REFUSED with a diagnosis rather than walked — the v75 rule, and the v88/v96 precedent |

**⇒ FRESH WEIGHTS. The signature bump forbids a warm start**, exactly as gen-14 and gen-16 were
forced fresh. This is not a cost of the design so much as a property of it: a critic trained to
predict a shaped return cannot be warm-started into predicting a probability.

### 5.2 What breaks resume

Resume-immutable fields that change value, each of which FATALs a flagless resume with a message
naming the flag (`check_reward_config` / `check_value_dist` / `check_compatible`):

`hand_shaping` (true→false) · `pbrs_material` (true→false) · `pbrs_belief` (true→false) ·
`victory_value` (30.0→1.0) · `draw_penalty` (−35.0→−1.0) · `all_shaping_pbrs` ·
`use_popart` (true→false) · `vf_coef` (re-tuned) · `value_from_dist` (deleted) ·
`value_dist_vmin`/`vmax` (deleted) · `win_prob_mode` (legal set shrinks).

Frozen eval / pool / distill opponents are unaffected by the reward fields (`check_compatible`
excludes them — their forward never reads the reward) but **are** affected by the signature, so a
gen-108 run cannot take a pre-108 opponent, teacher or sentinel. That is the usual fresh-generation
consequence and it means the new run seeds its pool from scratch.

### 5.3 The deletion list — in the `gen3_dead_flag_purge_v1` style

Every row: the flag, what goes with it, and the reason. A deletion whose reason is only "unused" is
not on this list.

| flag(s) | deletes | reason |
|---|---|---|
| `--value-dist-mode`, `--value-dist-bins`, `--value-dist-vmin`, `--value-dist-vmax`, `--value-dist-coef` | `ValueDistHead`, `_value_dist_loss`, the HL-Gauss target, `check_value_dist`, 4 registry rows, the `value_dist` trace column | §3.8 — a categorical over a two-point support is a Bernoulli; the head is not a richer parameterization of the new target, it is the same one with 50 redundant degrees of freedom |
| `--value-from-dist`, `--allow-value-from-dist-change` | the `_critic_value` branch, `set_value_from_dist`, `check_value_from_dist`, the `combination_checks` entry | names a choice between two critics; there is one |
| `--win-prob-pbrs-coef`, `--win-prob-pbrs-source` | `agents/training/winprob_pbrs.py`, the `collect_rollouts` hook, its `combination_checks` entry | §3.7 — with `V ≡ φ` it adds the advantage to the reward and then takes the advantage of that; and its Ng shield is weakest exactly where it would now run |
| `--win-prob-coef` | the separate coefficient | one critic, one coefficient (`vf_coef`); two on one loss is what forced `_ce_w`'s conditional |
| `--value-tail-weight` | `_value_loss_from_se`'s weighting | §3.6 — already inert under Phase B, and its shape is the banned one (weights, not samples, at the noisy boundary) |
| `--score auto` (search) | the `batch_scores` fall-back branch | §3.10 — a fall-back that cannot fire is dead; one that can is the losing arm |
| `--draw-penalty` at any value ≠ §3.2's | *nothing* — a REFUSAL, not a deletion | the flag stays meaningful in raw-terminal mode; it must refuse under the new objective rather than be ignored |
| `use_popart` | nothing deleted — **set false** | §3.5. **KEEP the flag and the module**: the two-currency machinery is correct and will be wanted again the moment the objective stops being terminal-only |

⚠️ **`--value-tail-weight` and `use_popart` are resume-immutable**, so each deletion/flip must ship
its migration branch and its refusal message in the same pass — the v75 rule.

### 5.4 The ONE fresh-run launch command

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.launcher \
  --run-name ai_v12_01_winprob_critic \
  --restart-interval-hours 3 \
  --steps 75000000 \
  --n-envs 64 --batch-size 16384 --grad-accum-steps 4 --n-epochs 10 \
  --n-steps 2048 --lr 0.0003 --ent-coef 0.02 \
  --device cuda --log-level periodic \
  \
  --no-hand-shaping \
  --victory-value 1.0 --draw-penalty -1.0 \
  --no-use-popart \
  --win-prob-mode shaping \
  --vf-coef <RE-TUNED — see §3.6; do not inherit 0.5> \
  --gamma 1.0 \
  --self-play
```

> **⚠️ SUPERSEDED by `gen3_winprob_critic_mode_v1` (2026-09-06). The command above LAUNCHES but is
> NOT this design's arm** — that was §5.4's own warning ("it would launch TODAY'S critic"), and the
> fix is one flag. **Use this instead:**
>
> ```bash
> export PYTHONPATH=$PYTHONPATH:src
> python -m main.launcher \
>   --run-name ai_v12_01_winprob_critic \
>   --restart-interval-hours 3 \
>   --steps 75000000 \
>   --n-envs 64 --batch-size 16384 --grad-accum-steps 4 --n-epochs 10 \
>   --n-steps 2048 --lr 0.0003 --ent-coef 0.02 \
>   --device cuda --log-level periodic \
>   \
>   --critic winprob \
>   --no-hand-shaping --terminal-indicator \
>   --victory-value 1.0 --draw-penalty 0 \
>   --vf-coef <RE-TUNED — see §3.6; do NOT inherit 0.5, it multiplies a BCE now> \
>   --self-play
> ```
>
> Three differences from the block above, each one load-bearing:
>
> 1. **`--critic winprob`** is what makes it this arm at all (gap B1).
> 2. **`--terminal-indicator` and `--draw-penalty 0`** replace `--draw-penalty -1.0`. §3.2 chose
>    "a draw is a not-win, `y = 0`", and §3.6 chose to keep the reward at ±1 — but those two are
>    INCONSISTENT with §3.1's `V(s) = σ(z) ∈ [0,1]`: a `+1/−1` return against a `[0,1]` critic
>    gives every terminal TD error a systematic, state-dependent offset (a loss reads `−1 − V`
>    against the truth `0 − V`). The indicator terminal (`+victory_value` on a win, `0.0` on a
>    loss, a tie AND a timeout) is what makes `V(s) = E[return] = P(win|s)` hold exactly. This is a
>    CORRECTION to §3.6's "keep the reward at ±1", not an addition to it.
> 3. **`--no-use-popart`, `--win-prob-mode shaping` and `--gamma 1.0` are dropped** — all three are
>    IMPLIED by `--critic winprob` (their argparse defaults are the `None` sentinel, so "unset" is
>    representable). Passing them explicitly is harmless and still validates.
>
> Validate with `python -m main.checkargs --argv "…"` and then `python -m main.launcher --dry-run`,
> in that order, exactly as this section already says.
>
> ### The `--vf-coef` RECOMMENDATION — **start at 0.5, and READ the first-rollout line**
>
> `<RE-TUNED>` above is not an answer, and the reason it could not be one is that **`--vf-coef`
> multiplies a different quantity under each critic while keeping its name**:
>
> | critic | what `--vf-coef` multiplies | its magnitude |
> |---|---|---|
> | `shaped` | an MSE between a shaped return and V, in PopArt-NORMALISED units | O(1) after normalisation — of a raw return whose scale is ±30, i.e. **O(100)** unnormalised |
> | **`winprob`** | the win-prob head's **BCE against a Bernoulli outcome** | **ln 2 ≈ 0.693** per sample at initialisation (a zero logit is P = 0.5), falling from there |
>
> So the historical 0.5 was calibrated against the first quantity and carries **no information**
> about the second. Two orders of magnitude separate the unnormalised scales, and the PopArt
> normalisation that closed that gap is REFUSED under this critic (§3.4) — there is no scale to
> track on a bounded stationary Bernoulli payoff, so nothing keeps the two comparable any more.
>
> **RECOMMENDATION: launch the first arm at `--vf-coef 0.5` and read the banner it prints on the
> first rollout, rather than re-tuning blind.** The reasoning:
>
> * Under PopArt the `shaped` value loss was already O(1) by construction, so 0.5 was never a
>   number about ±30 — it was a number about a normalised residual. A BCE that starts at 0.693 and
>   falls is in that same O(1) band, so 0.5 is a *defensible* starting point rather than an
>   inherited accident. What it is not is a MEASURED one.
> * The quantity that decides is not the coefficient but the **RATIO of the value term to the
>   policy term**, which is a property of the run and cannot be predicted from either flag. It is
>   also already the thing `grad/value_share` and `grad/value_policy_logratio` report per rollout —
>   the arm's own instruments, and the ones to steer by past the first reading.
> * A wrong coefficient here is not a silent failure. `train/noise_scale_ratio_value` and
>   `grad/value_share` both move with it, and §4's gate reads `win_prob/critic_resolution`, which a
>   swamped or starved critic degrades visibly.
>
> **The line the operator reads** (printed once, on the first rollout that carries a scorable
> win-prob label, from `instrumented_ppo/calibration.vf_coef_scale_line`; it does NOT latch on a
> rollout it could not read, and it is excluded from the checkpoint so every launcher restart
> re-prints it beside the `[CRITIC]` banner):
>
> ```
> 🎯 [CRITIC] winprob — first rollout scale: value term = --vf-coef 0.5 x BCE 0.1567 = 0.0784,
>    against |policy loss| 0.0002 -> 476x.  ⚠️ --vf-coef now multiplies a BCE, not the
>    shaped-return MSE 0.5 was tuned for: a BCE at a 0.5 base rate is ln 2 ~ 0.693 per sample at
>    init and falls, where that MSE on a +-30 return was O(100). …
> ```
>
> ⚠️ **That sample is from a CPU `--debug` smoke (1 env, 10k steps) and its 476x is NOT a
> production reading** — a toy's clipped surrogate sits at ~2e-4, which is the denominator, not a
> statement about the coefficient. It is reproduced here to show the SHAPE of the line, and it
> makes the reading rule concrete: **a ratio in the tens or worse means the critic is swamping the
> policy on the shared trunk** — cut `--vf-coef` by that factor — while a ratio far below 1 means
> the critic is starved. Confirm against `grad/value_policy_logratio` (0 = balanced), which is the
> aux-independent version of the same question and the one that keeps reading after rollout 1.
**Status of this command, measured 2026-09-06.** Everything above **except `--gamma 1.0`
validates today** — `python -m main.checkargs --argv "…"` accepts all 18 remaining flags and prints
`✓ this command still launches`. Two things stand between that and the design:

* **`--gamma` is not a flag** (γ is hardcoded at `model_build.py:691`) — gap **B6**.
* **It would launch TODAY'S critic, not this one.** With no `--value-dist-mode`, `value_from_dist`
  is false and the scalar `value_net` is the critic; `win_prob_mode shaping` still buys only the
  0.05-weighted side readout. **Gap B1 is what makes this argv mean what §3 says**, and until it
  lands the command is a clean-world reward arm, not a win-prob-critic arm. Anyone launching it
  early would get a legitimately interesting run and would be entitled to believe it was this one.
* Note the composition `--no-hand-shaping` + `--victory-value 1.0` + `--no-use-popart` is exactly
  the hazard `ai_v12_intersection_test.py` §1 pins (±1 returns landing inside ~4 of 51 atoms, "a
  critic quantized to ~0.5 on a ±1 scale, feeding GAE, silently"). §3.8's deletion resolves it — but
  only after it lands, and the interval between B1 and A4 is where that trap is live.

**Validate with `python -m main.checkargs --argv "…"` and then `python -m main.launcher --dry-run`,
in that order, before the real launch** — the argv-only tool answers "do these cohere", the dry run
resolves the actual launch on the box (role, run dir, pin, inherited-vs-argv config) without
creating a run dir, a worktree or a child.

### 5.5 The pre-registered read

**All of it runs as one command: `python -m main.critic_gate <run> --parent <ref> --control
<continuation refs…>`** — the ladder at matched snapshot count, §4.3's gate with G1 primary, the G7
kill condition and the untaught meter with its control, composed (never re-derived) into one
markdown report + JSON, with `--check` to resolve every input first.

Registered **before** the run, three endpoints and a control:

1. **ANCHORED LADDER at matched SNAPSHOT COUNT** — `<run>/snapshot_ladder/ladder.json` (dense, ±10),
   not `eval/elo` (±29), and only once the run is finished, because BT re-solves every node on every
   add and the newest is systematically inflated (gen-10's 12M fell 2089 → 2021 over 12 refits).
   Compared against the current generation **at matched snapshot count, never at matched step**.
2. **THE CALIBRATION GATE** (§4.3), per checkpoint, `bot` and `pool` separately, selection-reweighted.
   G1 (resolution) is primary; G7 (stall) is a kill condition.
3. **THE UNTAUGHT METER — with a CONTINUATION CONTROL, and this is the part that is easy to get
   wrong.** The 2026-09-06 CELL 2 ledger entry established that a plain +1.08M continuation of a
   mature parent reproduces the entire "gift" a fold was credited with: re-based on a continuation
   control, v8's celebrated +4.64pp becomes ≈ +1.2pp and is not significant. **A frozen parent is
   the wrong baseline** — it credits an arm with progress the baseline would have made anyway.
   Here the arm is a FRESH run, so the corresponding error is comparing it against
   `ai_v9_59_R2ACTION_0827` *frozen at 28.1M*. **The comparator is the current generation CONTINUED
   to matched snapshot count** — which, as of 2026-09-06, is what G5
   (`ai_v9_195/196/197_G5PLAIN{A,B,C}_0906`) is being run to establish on our parent. If G5's
   continuation gift fires, the frozen-parent reading of this arm is the one that gets retired.
   ⚠️ **UNVERIFIED at the time of writing whether G5 has been scored.**

**What would falsify the design, stated before the data:** G1 flat (resolution unmoved) with G2–G4
passing means the promotion bought calibration this head already had and nothing else — the
wrong-meter trap, and the target/readout diagnosis of §2 would survive intact while *this* remedy
for it would not. That must be reported as loudly as a pass.

---

## 6. THE GAP LIST — ranked by build cost

**Legend:** 🟢 done in this pass · 🔵 hours · 🟡 a day · 🔴 a design decision of its own.

### Build

| id | gap | file | cost |
|---|---|---|---|
| 🟢 **M0** | reliability curve + Brier + skill + ECE + Murphy split, stratified by opponent class, with the selection reweighting | `agents/training/scaffolding.py`, `main/scaffolding_gauge.py` | **DONE** |
| 🟢 **B1** | **DONE 2026-09-06** as a MODE (`--critic winprob`), default `shaped`, so the OFF path is byte-identical. `_ce_w` survives, because the distributional critic survives under `shaped` | `agents/model/{critic_mode,policy}.py`, `instrumented_ppo/ppo.py` | **DONE** |
| 🟢 **B2** | **DONE 2026-09-06 — the audit found all three ALREADY branch**, and `_value_loss_from_se` never had a branch to lose (it takes SE in the caller's chosen space). The missing half was the TESTS, which now pin each one: under `--critic winprob` these stop being the rare path | `agents/model/critic_mode_test.py` | **DONE** |
| 🟢 **B3** | **DONE 2026-09-06** — under `winprob` the promoted BCE folds as `_ntg.add("value", …)` and the grad-balance value term follows the critic. `shaped` keeps its `aux` tag | `instrumented_ppo/ppo.py` | **DONE** |
| 🟢 **B4** | **DONE 2026-09-06** — `--arm-no-progress-tax`, a resume-immutable `RewardConfig` bool defaulting OFF. Re-arms the tilt ALONE; the other 24 BIAS terms stay zeroed | `agents/training/reward_manager.py`, `parser/clean_world.py` | **DONE** |
| 🟢 **B5** | **DONE 2026-09-06 — THIRTEEN entries, not two**: the two named plus PopArt, `--value-dist-mode`, `--value-from-dist`, `--win-prob-coef`, `--value-tail-weight`, both `--win-prob-pbrs-*`, `--win-prob-pbrs-frozen`, and the three REQUIRED reward flags | `main/train/combination_checks.py` | **DONE** |
| 🟢 **B6** | **DONE 2026-09-06.** `--gamma` is a flag (shaped default read from `PBRS_GAMMA` itself), INERT on a resume and stated as such, and the PBRS assert is GATED on a potential actually being folded on BOTH build paths. **The truncation half was a 🔴 and is now closed** — MEASURED on a real bridge battle: the 250-turn cap forfeits, `battle.won=False`, six mons alive a side, and `PokeEnv.calc_term_trunc` (`poke_env/environment/env.py:951`) returned **`terminated=False, truncated=True`**, so `MaskablePPO.collect_rollouts` (`ppo_mask.py:251-260`) bootstrapped `r += γ·V(s_last)` — at γ=1 the tautology, verified by revert (the row read exactly `V`). Fixed at `wrappers.resolve_episode_end` (winprob-gated, ONE seam upstream of both rollout loops); `shaped` byte-identical and its bootstrap documented. **This env never truncates in the SB3 sense at all**: either flag requires `battle.finished`, so a `truncated` here means "finished, not by a wipe" — the cap and a tie — which is a terminal outcome. (c) is clean: SIGTERM `os._exit`s and the partial buffer dies with it; the eval reset-mid-battle forfeit fills no buffer | `agents/training/wrappers.py`, `main/train/env_factory.py` | **DONE** |
| 🔵 **B7** | re-scale the prober `calibration` verb's `overvalue_tau` (5.0 = reward units) for probability units | `main/prober/session/aggregate.py` | hours |
| 🟡 **B8** | the version bump itself: 108, new `ARCH_SIGNATURE`, `MIGRATION_FLOOR` 108, the migration branches and refusal messages for every deleted resume-immutable field | `agents/model/model_version/` | a day |
| 🟢 **B9** | **DONE 2026-09-06** — the three-way branch is named in `wrappers.py`, `info["win_draw"]` publishes it, and the rate lands as **`signal/draw_rate`** (not `train/`: `SignalMetricsCallback` has a pinned prefix contract, and the rate's siblings are `signal/outcome_*`) | `agents/training/{wrappers,signal_callback}.py` | **DONE** |
| 🟢 **B10** | **DONE 2026-09-06.** The collapse is DECIDED and built: `defensive.{leaves_for_critic,scores_for_critic,resolve_for_critic}` narrow both legal sets off the LIVE policy's `_critic_mode` — `--defensive-leaf` to `winprob` alone, `--score` to `win_prob` alone, with an explicit `value` REFUSED on either. **`--score auto` dies as a RESOLUTION, not a refusal**, and that is the decision: `auto` is the CLI default, so refusing it would make the flagless invocation a usage error on every winprob checkpoint, while what made it dangerous — the silent fall-back to the value head — is unreachable when there is one readout. It resolves to `win_prob`, ANNOUNCED at startup and recorded. **`check_leaf` is KEPT** (vacuous here; the guard outlives the thing it currently has nothing to catch). Resolution is at STARTUP, not in `batch_scores`: `SearchEngine.choose` swallows every exception as a counted `search_error`, so a deep refusal on the flagless default would make every decision a policy fallback and report the arm at ~0 in silence — the deep check stays as the library backstop. Probe G's `value` control arm is **retired on this critic only** (the two readouts are one number there); `shaped` keeps it untouched, so no ledger retirement is owed. Proven end to end on a CPU cell — see below | `main/search_dividend/{defensive,search,__main__}.py` | **DONE** |

### A2 — THE CONSUMER CENSUS (DONE, 2026-09-06). Read this before deleting anything.

`gen3_winprob_critic_mode_v1` implemented the mode without touching the head, and this is the
census that licensed that choice. Scope: every consumer of `value_dist_head` / `ValueDistHead` /
`value_from_dist` / `last_value_dist_logits` / the `value_dist` trace column / the five
`--value-dist-*` flags / `check_value_dist` / `_value_dist_loss` / the 51-atom output and everything
built on it (PIT, `coverage80`, `knew_by_turn`, `blind_loss`, the bottom-atom stall signature).
Method: tree-wide grep, each hit READ for what it does.

**THE STRUCTURAL FACT the whole census turns on:** the head is built iff
`value_dist_mode != "none"` (`extractor_build.py:807-814`), and **almost every consumer gates on
the MODE STRING, not on `value_dist_head is None`**. So "skip the build, leave the mode set" is a
state ~15 sites cannot represent — which is why `--critic winprob` REFUSES a non-`none`
`value_dist_mode` on the RESOLVED value rather than merely not building the head.

| # | area | representative sites (file:line) |
|---|---|---|
| 1 | model / policy | `aux_value_heads.py:50-103` (the class), `:88` (the non-persistent `atoms` buffer), `:96-100` (`mean(logits)` = the E[Z] the Phase-B critic returns) · `extractor_build.py:31,76-77,796-814` · `extractor_forward.py:734-736` (the ONE writer, already `None`-guarded) · `extractor_stashes.py:54` · `extractor_api.py:210` · `features_extractor.py:120` · **`policy.py:184-201`** — `_critic_value`'s load-bearing read, RAISING on a missing head/stash · `policy.py:132,141,155-166,245,272,288` · `critic_route_audit.py:28-30,61-62` |
| 2 | PPO loss | `value_terms.py:89-141` (`_value_dist_loss`, the HL-Gauss target + PIT) · `hparams.py:139-143` · `ppo.py:237,263,264-266` (**the gate reads the MODE**), `:557-560` (Phase B drops the scalar term), `:867-884` (the CE fold, tagged **`aux`**, weighted `vf_coef` under Phase B), `:1601-1607` |
| 3 | noise-scale / grad-balance | `ppo.py:882` (group `aux`) · `noise_scale_terms.py:21,109,118,162` (`add` is a passthrough) · `ppo.py:1204,1214,1310-1312` (**grad-balance's value term SWAPS to the CE under Phase B**) · `grad_balance.py:96-102` |
| 4 | prober | `engine/analyze.py:20-23,27-66,249-255,309` · `engine/views.py:534` · `model.py:583-603` (`value_dist_at`), `:767-779` (`value_dist_support`), `:807,845-846` · `session/core.py:152-172` (`_dist_support`, model-FREE off `model_config.json`) · `session/reading.py:145-147` · `session/scans.py:38-42,55-77,85-105,119,127,245-252,325-329,402,473-474,498` · `awareness.py:1-35,47-70,230-241,245-266` · `engine/intent.py:152-153` · `lookahead.py:195,209` · `better_line.py:81,402-403` · `web/app.py:420,810-813,1100-1129` · `web/templates/partials/analyze_result.html:282,325-334` · `web/fixture_run.py:120-126,160,201-203` · `main/endofrun.py:43-47,184-215` |
| 5 | TensorBoard | `ppo.py:1601-1607` (`value_dist/{ce,entropy,std,pit_mean,mean_abs_err}`) · `launcher/format.py:101-103,161-165,316-318` · `parser/distillation.py:456-457` |
| 6 | eval worker / traces | PRODUCER `inference/player.py:465-467,625-635` (already returns `None` with no head) · WRITER `battle_recorder.py:179-181,202-205,223-224` (**omits the npz key entirely** when absent) · readers `analyze.py:34`, `awareness.py:247,261`, `session/reading.py:145`, `session/scans.py:60` — all KeyError/None-guarded. `summary.json` carries no `value_dist` column |
| 7 | counterfactual family | **NO direct consumer.** `cf_terms.py:420,427,472-483` and `prober/model.py:760-764` reach it only INDIRECTLY, through `policy._critic_value`. `cf_audit`, `cf_q_labels`, `cf_label_buffer`, `q_winprob_terms`, `ShadowValueHead`, `CfEvidentialHead`, the twins: zero references |
| 8 | stashes / readouts | `extractor_stashes.py:54` · `extractor_api.py:210` · `extractor_forward.py:734-736` · `tier_contract.py:122` (declared **tier 3 DELIVER**) · `arch_tables.py:48,75-78,95` · `delivery_graph.py:93,338-341,774-781,810-813` + `delivery_graph_snapshot.json` |
| 9 | version machinery | `fields.py:256-266,370-373,377,520-526` · `compat.py:317-319,509-512,567-570` (**FATAL on `value_dist_mode` / `value_dist_bins` mismatch**) · `resume_checks.py:127-152,170-190` · `migrations.py:105-106,132-133` · `construct.py:30,165-169,255-259,276,313` · `flag_registry.py:181-191` · `snapshot.py:796-800,842-849,1171-1175,1266-1269,1308,1359-1360` · `config.py:95-118,456-460,591,631-632` · `combination_checks.py:119,288-291,408-421` · `model_build.py:144,390,400,443-447,455-457,684,725` · `lifecycle.py:91` · `run_io.py:120` · `designs/production_config.json:88-95` |
| 10 | tests | 30 files / 185 references. Heaviest: `dist_critic_test.py` (24), `prober/awareness_test.py` (21), `value_dist_head_test.py` (21), `prober/web/app_test.py` (15), `prober/engine_test.py` (13), `instrumented_ppo_test.py` (11) |
| 11 | docs / generated | `ARCHITECTURE.md:226,244,694-697,972-988` · `flag_registry.md:89-92,137-138,157` · `architecture_graph.dot:19,30,131,147,164` + the viewer · `delivery_graph_snapshot.json` · `CHANGELOG.md` (37 refs) · `designs/model.md:62` · four `CLAUDE.md` leaves |

**If the head is simply NOT BUILT** (`value_dist_head is None`, other flags at their defaults):

* **(a) guards cleanly, LOSES a feature** — the writer, both prober support functions, `lookahead`,
  `better_line`, the web chart, the delivery graph, `arch_tables` (reports ABSENT / the coefficient
  INERT), and `session/scans.py:38-42`, which returns a structured `{"error": …}`. What goes dark:
  `knew_by_turn` / `lead_time` / `blind_loss` / `mean_tail_divergence` (the bottom-atom stall
  signature) / `coverage80` / `pit_mean` / the whole `value_dist/*` TB family / `endofrun`'s
  awareness verdicts (which read `"UNAVAILABLE"`, correctly).
* **(b) RAISES, deliberately** — `policy._critic_value:189-196` under `value_from_dist`, and
  therefore every rollout, every PPO epoch, `td_aux`, `cf_terms` and the prober's `live_v` at once.
  Plus `check_compatible` / `check_value_from_dist` / `check_value_dist` on a resume, and
  `extractor_build.py:798-806` if the mode is set with `bins == 0`.
* **(c) SILENTLY WRONG — the set that shaped this mode's refusals.** `ppo.py:264-266` + `:871`
  gate on the MODE and then skip the CE with an inner `is not None`, so a run trains with **no
  distributional loss** while every flag says it is on. `ppo.py:1310-1312` then reports the FROZEN
  scalar head's pull as `grad/value_share` (the 2026-07-22 catch). `prober/model.py:593,774` and
  `session/core.py:165` build a support for a head that does not exist. `awareness.py:26-33`'s
  `fit_denorm` is exact only under `value_from_dist` and would return NUMBERS, not errors, in a
  changed currency. And `session/scans.py:60-62` would keep computing PIT/`coverage80` against
  gen-10 baselines across two different currencies.

**Consequence for a future deletion (A4):** the head cannot be removed by not building it. The
delete must take the MODE with it, and §(c) is the checklist.

### Delete

| id | gap | file | cost |
|---|---|---|---|
| 🔵 **A1** | `--win-prob-pbrs-*` + `agents/training/winprob_pbrs.py` + its tests + its `combination_checks` entry | `agents/training/winprob_pbrs.py` | hours |
| 🟢 **A2** | **DONE 2026-09-06** — the consumer census, its "what breaks if the head is absent" split, and the finding that ~15 sites gate on the MODE STRING rather than on the head. It is the section immediately above this table (*A2 — THE CONSUMER CENSUS*), not a separate artifact | tree-wide; started at `agents/model/extractor_api.py:210` | **DONE** |
| 🟡 **A3** | retire the scaffolding gauge's two DIVERGENCE gauges once there is one readout; keep the reliability half | `main/scaffolding_gauge.py`, `agents/training/scaffolding.py` | a day |
| 🟡 **A4** | the `value_dist_*` / `value_from_dist` / `--value-tail-weight` / `--win-prob-coef` purge, with registry rows, generated tables, `production_config.json` and `ARCHITECTURE.md` in the same pass | `agents/model/flag_registry.py` + the five hand-synced surfaces | a day |

### Measure

| id | gap | file | cost |
|---|---|---|---|
| 🔵 **M3** | publish `win_mask` coverage as a first-class scalar (G8) | `instrumented_ppo/value_terms.py:63-85` | hours |
| 🟡 **M4** | a **weighted** `affine_gauge`, so the §4.1 finding-5 comparison (an affine map of V out-predicting the head, raw only) can be made on the reweighted population instead of the quota | `agents/training/scaffolding.py` | a day |
| 🔴 **M1** | **`sd_true_excess` runnable offline from eval traces** (G5). It currently needs tight-MC labels and the determinization instrument, i.e. the counterfactual factory — not a trace read. Either build the trace-only approximation or **strike G5 from the gate and say so** | `agents/training/cf_audit.py`, `main/prober/` | decision |
| 🔴 **M2** | **the MIRROR TABLE as an offline instrument** (G6) | — | decision |
| 🔴 **M5** | a **pre-registered stall threshold** for G7, derived from the era's recorded stall rate rather than chosen after the fact | `designs/research_state/` | decision |

**The ordering that matters:** **A2 before A4** (know what you break before you break it), **B2 and
B3 before any arm** (they are correctness, not features), and **M1/M2 resolved before the gate is
quoted** — a gate with unrunnable criteria degrades to its runnable ones the first time a run is
waiting.

---

## 7. Open questions this document does not settle

1. **Does an MC-only critic target lose more to variance than it gains in correctness?** §3.4's
   trade is argued, not measured. The cheapest read is an offline one: compare the MC label against
   the bootstrapped return on existing traces and measure the variance of each. Not done.
2. **UNVERIFIED: how much the `--no-hand-shaping` composition costs in learning SPEED.** The
   invariance theorem says it costs no correctness; nothing in this tree has measured the dynamics
   cost, and a 75M-step run is an expensive place to find out. A short paired A/B at ~2M steps
   before committing the long run would price it.
3. **UNVERIFIED whether `win_head`'s architecture is right for a critic.** It was sized as a side
   readout (`LayerNorm → 128 → ReLU → 1`). §2's evidence says expressiveness is acquitted *for the
   defect measured*, which is not the same as saying the shape is right once gradients flow the
   other way.
4. **The label is still self-referential** — outcomes under the current policy — and
   `design_winprob_behavior_coupling.md` §1 names that as the deeper reason the head cannot coach.
   Promotion does not fix it; it makes the loop tighter. The counterfactual label factory
   (`--cf-winprob-coef`) is the standing answer and it now points directly at the critic (§3.9).
5. **Whether `--win-prob-mode read_only` is the interesting arm.** A critic whose gradient does not
   reach the trunk is a real and testable position (it is what the belief heads' `label_only` mode
   already is), and it is the cleanest way to ask whether the critic's *representation* pressure is
   helping or crowding.

---

## 8. Provenance

Design written 2026-09-06 against HEAD `407b27c0`. Reads: `designs/ARCHITECTURE.md` §3.4/§6,
`designs/research_state/critic_calibration_plan.md`,
`designs/learning/{win_prob_decomposition, credit_assignment_and_value_errors, popart_value_scale_and_currencies}.md`,
`designs/ai_v12/{design_winprob_behavior_coupling.md, todo.md}`, the ledger's 2026-09-06 CELL 2 and
G5 entries, and a line-level audit of `agents/model/{value_readouts,aux_value_heads,policy,popart,flag_registry}.py`,
`agents/training/{reward_manager,winprob_pbrs,wrappers,win_prob_callback,scaffolding}.py`,
`agents/training/instrumented_ppo/`, `main/train/parser/clean_world.py`,
`main/search_dividend/{search,defensive}.py` and `main/prober/session/aggregate.py`.

Measured baseline:
[`designs/research_state/measurements/winprob_critic_baseline_2026-09-06/`](../research_state/measurements/winprob_critic_baseline_2026-09-06/).

Every claim not carried by a file:line reference or by that measurement directory is marked
**UNVERIFIED** in place.
