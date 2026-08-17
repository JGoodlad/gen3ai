# Activation functions — is ReLU the right choice for our model?

**TL;DR.** ReLU is a *defensible default* and almost certainly not what is costing us strength —
but the question "should different parts use different activations?" has an answer we already
live by without having written it down: **our model uses three distinct nonlinearity tiers**, and
only one of them is a free choice.

| Tier | What it is | What we use | Is it a choice? |
|---|---|---|---|
| **Generic** | our MLP hidden layers, transformer FFNs, head bottlenecks | **ReLU** | **Yes** — swappable |
| **SB3 policy/value tower** | `net_arch = [512,512]`, both branches | **tanh** — an SB3 *default* we never set | Yes, but currently implicit |
| **Bounded scorer** | the pointer head's per-action projection | **tanh** | Yes, but deliberate |
| **Semantic** | damage op, belief posteriors, attention | sigmoid / softmax / clamp | **No** — these are the math |

⚠️ The tower row is easy to miss: we pass `net_arch` in `policy_kwargs` but **never**
`activation_fn`, so `MaskableActorCriticPolicy`'s default `nn.Tanh` silently governs 4 of the
policy path's layers. See §2.4.

The generic tier is the only place a GELU/SiLU swap even means anything. The single site where
the choice has a real structural consequence is the extractor's **output** projection
(`features_extractor.py` (`Gen3FeaturesExtractor.forward`)), which makes every feature the policy and critic see
non-negative. Everything else is a 2-layer-deep, LayerNorm-sandwiched MLP where the literature's
ReLU-vs-GELU gap is small.

**Almost nothing about activations has been measured in this repo** — a grep for `GELU`/`SiLU`/`Mish`
over `src/` returns zero hits. Every claim below about *our* model is mechanism, not measurement,
and is marked where it matters. The **one exception** is the extractor-output site: its dying-unit
risk was probed on 2026-08-16 and came back **zero dead units of 512 in both heads** over 2479 real
states (§2.3, ledger K11), which kills the strongest argument for swapping it.

---

## 1. The intuitive level

### What an activation actually does

A stack of linear layers is just one linear layer — `W₂(W₁x)` collapses to `(W₂W₁)x`. The
activation is the only thing between the layers that stops the collapse. So its job description
is short: **be nonlinear, and don't wreck the gradient.**

Everything else is a tradeoff between those two.

### Why ReLU won

ReLU is `max(0, x)`. Below zero it outputs nothing; above zero it is the identity.

Its virtue is the **gradient**. Sigmoid and tanh saturate: push the input far from zero and the
slope goes to ~0, so the gradient signal dies on its way back through the layers. This is why deep
networks were hard to train before ~2011. ReLU has a slope of exactly 1 on its entire positive
half — the gradient passes through *undiminished*, no matter how deep. It is also one branch
instruction, which is as cheap as a nonlinearity gets.

### Its one real defect: the unit can die

The mirror of "slope exactly 1 above zero" is "slope exactly **0** below zero". A unit whose
input is negative for every example in the batch receives *zero* gradient — not a small gradient,
none. It cannot learn its way back. It is dead for the rest of training.

The smooth alternatives (GELU, SiLU/Swish) are ReLU with the corner rounded off:

```
ReLU:  ____/          hard corner at 0, exactly flat to the left
GELU:  ___╱           soft corner, slightly negative dip, never exactly flat
```

The dip below zero is the point. A GELU unit that is currently "off" still passes a small
gradient, so it can be revived. It also lets a unit output a *small negative* value, which ReLU
structurally forbids.

The cost: a few more flops per element, and a slightly less crisp sparsity pattern (ReLU's exact
zeros are genuinely useful — they make the representation sparse and the gradients structured).

### The concrete Pokémon version of "why it might matter here"

Take the pointer head scoring `switch to Skarmory`. Suppose one hidden channel has learned to mean
"the opponent's active threatens my Skarmory". That is a signal that should be able to go
**negative** — "actually Skarmory *walls* this" is not the same statement as "no threat".

Under ReLU the channel can only say *threat* or *silence*. To express "safe" the network must
allocate a **second** channel meaning "safety", and the two must be learned to cancel downstream.
That is not fatal — it is exactly what networks do, and 128 dims is plenty of room to spend on it
— but it is a real cost paid in width, and it is the honest form of the "should we use something
signed?" question.

---

## 2. The technical level

### The three properties that separate the candidates

1. **Gradient at the origin and below.** ReLU: 1 above, 0 below, undefined at 0. GELU/SiLU: smooth
   everywhere, small-but-nonzero below. This is the dying-unit axis.
2. **Signedness of the output.** ReLU and GELU are (essentially) one-sided; tanh is symmetric and
   bounded. Determines whether the layer's output can express a signed quantity directly.
3. **Cost.** ReLU is a compare-and-select. Exact GELU needs an `erf`; the tanh approximation needs
   a `tanh`. On GPU at large batch this is invisible (the layer is memory-bandwidth-bound and the
   activation fuses into the epilogue). On **CPU at batch 1** — which is exactly our rollout
   opponents' regime — it is a larger relative share, though still small next to dispatch overhead.

### Why the dying-ReLU risk is *low* in our trunk specifically

Three structural facts work in our favour:

- **We are shallow.** `TRANSFORMER_N_LAYERS = 2` (`arch_constants.py`), and every MLP is 2 layers
  (`ROLE_ENCODER_HIDDEN = [256,128]`, `MOVE_NET_HIDDEN = [96,32]`). Dying-ReLU compounds with
  depth; at 2 layers there is very little to compound.
- **LayerNorm is everywhere.** Post-LN transformer blocks (`norm_first=False`,
  `team_transformer.py`) and a `LayerNorm` in front of every head MLP re-center activations
  each block, which is precisely the condition under which a unit's pre-activation stops drifting
  permanently negative.
- **Nothing is depth-starved.** The classic dying-ReLU disasters are deep plain MLPs with a large
  learning rate and no normalization. That is not our model.

So the textbook argument for switching is weaker here than it is in general.

### Where the choice *does* have a structural consequence

`features_extractor.py` (`Gen3FeaturesExtractor.forward`):

```
pi_features = ReLU(pi_pre)
vf_features = ReLU(vf_pre)
```

This is the extractor's **return value** — the entire interface to `Gen3DualHeadMaskablePolicy`.
Two consequences follow, and neither is about training dynamics:

1. **The representation is confined to the positive orthant.** Every feature the policy and critic
   consume is `≥ 0`. Any signed quantity — advantage-like, "better/worse than neutral", a
   directional matchup read — must be encoded as a *pair* of one-sided channels rather than one
   signed channel.
2. **A dead unit here is dead for both heads at once.** Elsewhere a dead unit costs one channel in
   one block; at the final projection it costs a channel of the shared interface.

This is the one site where "would a smooth or signed activation change the model's expressive
shape?" has an unambiguous *yes*. **UNVERIFIED:** whether it changes measured strength.

#### MEASURED 2026-08-16 — consequence 2 (dead units) is FALSE here

The dying-unit half of the argument above is the one that would justify a generation, and it does
not survive contact with the model. `tmp/relu_deadunit_probe.py` hooks `projection` /
`value_projection` (their outputs *are* `pi_pre` / `vf_pre`) and reads gen-12 @14.2M over **2479
real greedy on-distribution decision states**:

| | `pi_pre` | `vf_pre` |
|---|---|---|
| **dead (never fires)** | **0 / 512** | **0 / 512** |
| always-on | 0 | 0 |
| near-dead (<1% of states) | 6 | 4 |
| active on <10% of states | 90 | 82 |
| mean active fraction | 0.465 (median 0.428) | 0.483 (median 0.457) |

By the rule of three, a unit reading dead at n=2479 has a true activation rate below 0.12%. With
**zero** dead *and* **zero** always-on, every unit genuinely modulates: this is a textbook-healthy,
roughly zero-centred ReLU gate, not a dying layer. The ~17% of units firing on <10% of states are
sparse specialists, not corpses.

**Two traps this measurement sets, both worth internalising:**

- **A dead COUNT is meaningless without its sample bound.** The pilot at n=97 reported "3 pi / 5 vf
  dead" — entirely artifact: at n=97 a unit that truly fires 1% of the time reads dead 38% of the
  time. The probe prints the rule-of-three bound for exactly this reason.
- **"≈55% of pre-activation mass is clipped" is NOT lost information.** The network *trained* under
  ReLU and arranged its representation so the discarded half carries what it does not need. That
  number describes ReLU's operating point; reading it as a 55% loss is the same error one level up.

#### …and consequence 1 (the orthant tax) is dead too — because capacity is not binding

The channel-pair argument is a **capacity** argument, and capacity has a precondition that is easy
to skip: *it costs nothing unless the interface is full.* `tmp/relu_capacity_probe.py` (n=4944
states, n/d = 9.7×) asks that first:

| | `pi` | `vf` |
|---|---|---|
| effective dim (participation ratio) of the post-ReLU interface | **26.9 / 512** | **30.8 / 512** |
| variance actually TRANSMITTED to the tower (`λ_k‖Wv_k‖²`) | 23.2 | 30.8 |
| anti-correlated pre-activation pairs (corr < −0.8) | **0** | **0** |

The interface carries an effectively **~25–31 dimensional** signal through 512 channels — ~5–6% of
the budget — and this is not a structural cap: both projections are *compressions* (1177→512,
1369→512), so full rank is reachable and the deficit is a property of the learned representation.
Spending two channels to carry a sign is free when ~480 contribute almost nothing. And the
mechanism is not even visible: **zero** anti-correlated pairs in either head (most negative −0.711 /
−0.721), so the model does not appear to be paying the tax in the first place.

⚠️ **One wrong analysis is preserved here because the error is instructive.** The first version of
the follow-up compared the tower's raw read-weight energy `‖Wv_k‖²` against the variance curve,
found it spread across ~460 of 512 dims, and concluded the tower amplifies low-variance directions
(so PR understates capacity). That is the **null**: `‖Wv_k‖²` measures only how `W` is *oriented*,
and is flat in `k` for isotropic `W` — the measured curve matched a matched-scale random-`W`
baseline to within ~1pp. The quantity a consumer actually receives is variance *transmitted*,
`λ_k‖Wv_k‖²`, and by that measure the PR reading stands. **A curve that matches its own null is not
a finding.**

**Two limits stated honestly:** every measure here is variance-based, and a low-variance direction
*can* be decisive — an ablation→ΔKL sweep would close that gap; and pairwise correlation cannot see
a sign encoded across a distributed subspace rather than a clean pair.

**Ledger K11: both legs measured dead. Not worth a generation slot** — and since the swap is
`ARCH_SIGNATURE`-bumping and fresh-only, it cannot be A/B'd mid-run nor folded into another
generation without confounding it, so "cheap to try later" is not on the table. Re-run both probes
if the activation ever changes; they are the pre-registered metrics.

There is also a historical note worth keeping: `model_version.py:326` records that the retired FiLM
conditioning generators were zero-init and applied **post-projection, pre-ReLU**. A modulation
signal starting at exactly zero has to grow through the ReLU's gate, and the gate is closed on
whichever channels are currently negative. That is a mechanistically plausible contributor to the
FiLM line's null results — but the conditioning work has a *separate*, measured explanation (see
`conditioning_architectures.md` and the count-dominates-conditioning result), so this is a
footnote, not a re-opened case.

### 2.4 The policy path end to end — four nonlinearity stages, three of them squashing

The full route from trunk to action logits is not "ReLU trunk → pointer head". It is:

```
trunk blocks .............. ReLU          (team_transformer.py: BiasedEncoderLayer / TeamTransformer)
extractor OUTPUT .......... ReLU          (:3223, applied :4009-4010)   → pi_features ≥ 0
SB3 mlp_extractor ......... Linear(512) tanh  Linear(512) tanh          → latent_pi ∈ [-1,1]
pointer head .............. tanh          (:2374)  → zero-init scorer   → 11 logits
```

The middle stage is the one nobody chose. `train_rl_agent.py:3766` builds `policy_kwargs` with
`net_arch` and the extractor kwargs, and **no `activation_fn`** — so
`MaskableActorCriticPolicy`'s signature default (`activation_fn: type[nn.Module] = nn.Tanh`,
`sb3_contrib/common/maskable/policies.py:51`) applies. That is the classic PPO/MuJoCo default and
is a well-tested choice for on-policy control, but it arrived here by omission rather than by a
decision about *our* discrete, 11-action, heavily-structured policy.

The structural observation that follows: between the trunk and the logits the signal passes
through **one one-sided stage and three saturating ones**. Nothing about that is known to be
harmful — the pointer head's zero-init scorers are where logit sharpness actually comes from, and
a bounded `latent_pi` is arguably *good* for the pointer head's stability — but it is the shape of
the policy path, and it was never a deliberate design.

**UNVERIFIED:** whether the tanh tower costs anything here. It is an SB3 default, not a measured
result on this model.

### Why the pointer head is tanh, and why that is correct

`pointer_head.py` (`PointerNativeActionHead`) scores every action through `tanh`, then a **zero-initialised**
linear scorer:

```
m = tanh(move_proj(move_token) + ctx)
move_logits = move_score(m) * move_valid
```

This is a different job from a trunk MLP. These outputs become **logits over an 11-way softmax**,
so:

- **Boundedness is a feature.** `tanh ∈ [-1,1]` means no single action's score can run away early
  in training and collapse the policy entropy. A ReLU here is unbounded above and would need the
  zero-init scorer to do all the taming.
- **Signedness is required.** "This move is bad" and "this move is unremarkable" are genuinely
  different logits, and the comparison is *relative* across actions.
- **Saturation is harmless.** The usual anti-tanh argument is vanishing gradient through depth.
  There is no depth here — it is one projection before the score.

So the model already answers "do different parts need different functions?" with **yes**, and the
answer is not arbitrary: the trunk needs *gradient flow* (ReLU), the scorer needs *bounded signed
comparison* (tanh).

### The semantic tier — not a choice at all

In the `DamageOperator` and the belief heads, the nonlinearity **is the quantity**, and swapping it
would be a physics bug, not a tuning change:

- `torch.sigmoid(move_belief_logits)` (`damage_op.py:589`, `:1756`) — a per-move **probability** the
  opponent carries that move. Sigmoid because independent multi-label, not softmax.
- `sigmoid((our_spe - opp_spe) / (opp_spe_std / 1.702))` (`damage_op.py:763`) — a deliberate
  **normal-CDF approximation**; the `1.702` is the constant that makes a logistic match a Gaussian.
  This is not "an activation", it is `P(we outspeed)` under speed uncertainty.
- The damage clamps (`_DMG_CHIP_CAP`, `:656-659`) — **physical bounds** on a damage roll as a
  fraction of max HP.
- `log_softmax` for species posteriors (`t0_species.py:85`) — and note the `gen3_species_posterior_spelling_v1`
  rule there: it must be spelled `log_softmax(...).exp()` and never `torch.softmax`, because
  Inductor's CPU backend cannot codegen the latter. Nonlinearity choice interacts with the compile
  path.
- `F.relu(target - std)` in `belief_bank.py:302` — a VICReg **hinge**, i.e. `max(0, ·)` used as a
  one-sided penalty. Not an activation at all; the name is a coincidence.

Anyone doing a global find-and-replace of `ReLU` would break the first tier and leave the third
untouched only by luck. The tiers must be edited by hand.

---

## 3. Where this lives in our architecture

| Site | File | Activation |
|---|---|---|
| Transformer FFN (trunk) | `team_transformer.py` — `TeamTransformer` (post-LN) | ReLU |
| Edge-bias encoder-layer clone | `team_transformer.py` — `BiasedEncoderLayer` | ReLU |
| `MoveLatentEncoder` MLP | `encoders.py` — `MoveLatentEncoder` | ReLU |
| Shared move network | `encoders.py` — `PokemonEncoder.move_network` | ReLU |
| Per-mon role encoder | `encoders.py` — `PokemonEncoder.role_encoder` | ReLU |
| Value head / `WinProbHead` bottleneck | `aux_value_heads.py` — `WinProbHead` / `ValueDistHead` | ReLU |
| **Extractor output projection** | `features_extractor.py` — `self.activation`, applied in `forward` | ReLU |
| **SB3 policy/value tower `[512,512]`** | `train_rl_agent.py:3766` (no `activation_fn` passed) | **tanh** (SB3 default) |
| **Pointer (action) head** | `pointer_head.py` (`PointerNativeActionHead`) | **tanh** |
| `OppIntent` / `PairReduce` / team-completion MLPs | `opp_intent.py:72`, `pair_reduce.py:77`, `team_completion_model.py:134` | ReLU |
| Damage op, belief posteriors, attention | `damage_op.py`, `t0_species.py`, everywhere | sigmoid / softmax / clamp |

Relevant constants (`arch_constants.py`): `D_MODEL = 128`, `TRANSFORMER_N_LAYERS = 2`,
`TRANSFORMER_FFN_DIM = 256`, `TRANSFORMER_N_HEADS = 4`, `POINTER_HIDDEN = 64`.

### The two hazards if you ever do swap one

1. **It is retrain-class but NOT shape-changing.** An activation swap leaves every weight shape
   identical, so `model_config.json` / `check_compatible` (`agents/model/snapshot.py`) will
   **not** catch it — an old checkpoint would load cleanly into a network computing a different
   function, with no `[ModelVersion] FATAL`. Any such change must bump `ARCH_SIGNATURE`
   (`model_version.py:1011`) deliberately, exactly as the model-versioning playbook in
   `src/agents/model/CLAUDE.md` requires for a structural change.

   Sharpened: `ModelVersion.from_layout_and_policy_kwargs` records
   `net_arch=list(policy_kwargs.get("net_arch", NET_ARCH))` (`model_version.py:1461`) and **no
   activation field at all** — so the tower's shape is versioned while its nonlinearity is not.
   A corollary worth knowing: because we never pass `activation_fn`, an SB3/sb3-contrib upgrade
   that changed that default would silently change 4 layers of the live policy, and no gate in
   this repo would notice. Passing `activation_fn=nn.Tanh` explicitly would pin today's behaviour
   at zero behavioural cost — a cheap defensive change independent of any decision to swap.
2. **It must re-pass the compile gates.** `--compile-opponents` (the measured 6.53× B=1 CPU
   forward) and `--compile-trainer` (1.75× on the PPO step) both depend on Inductor lowering the
   whole graph. GELU lowers fine in general, but this graph already has two known Inductor
   refusals (the damage op's `atomic_add` scatter on the C++ backend; `torch.softmax` on the last
   dim). `extractor_compiles_test.py` is the default-on gate.

### How a swap would have to be judged

Per the fresh-generation policy (`project_fresh_generation_equivariance` / `designs/`), an
activation change is judged by **anchored ELO, generation vs generation**, fit offline with
`python -m main.elo` at run end with matched snapshot counts — not by a mid-run A/B and never by a
narrated mid-run ELO delta.

---

## 4. Synthesis

ReLU is a reasonable default for our trunk and is very unlikely to be the binding constraint on
strength — our network is 2 layers deep, LayerNorm-sandwiched, and the plateau work has repeatedly
localised the ceiling to *structural* holes (belief, conditioning, critic credit assignment)
rather than to capacity or optimisation smoothness. Swapping ReLU→GELU across the trunk is the
kind of change that buys a fraction of a percent in the literature and would be indistinguishable
from noise at our ELO error bars.

The genuinely interesting version of the question is narrower and structural: the extractor
**returns** `ReLU(·)`, so the entire policy/critic interface is non-negative, and every signed
quantity must be spent as a channel pair. That is the one site where the activation changes what
the representation can *express* rather than how smoothly it trains.

**But it is no longer worth a generation slot, and that is a measurement rather than a judgement**
(§2.3 above, ledger K11). The argument had two legs; the load-bearing one is dead. Dying units at
this site would have been a concrete, unrecoverable defect — and there are **zero** of them across
2479 real states, in both heads, with zero always-on either. What remains is the orthant-efficiency
leg alone: real, but a capacity argument in a model whose plateau work has repeatedly localised the
ceiling to *structural* holes (belief, conditioning, critic credit assignment) rather than to
capacity. Since the swap is `ARCH_SIGNATURE`-bumping and fresh-only, it cannot be A/B'd mid-run and
cannot ride along with another generation without confounding it — so "cheap to fold in later" is
not actually available. The honest ranking puts it below every open structural lever.

The transferable lesson is about *where* activations matter: at narrow, un-renormalised
**interfaces** that many consumers read — not in wide, LayerNorm-sandwiched hidden layers. This
projection is the former and the trunk is the latter, so if you ever spend attention on one
activation here, this is still the right one to look at. "Right one to look at" and "will move ELO"
are different claims, and only the free diagnostic separates them.

And the answer to "do different parts need different functions?" is that they already have them,
along the axis that actually matters: **our own generic positions take the cheapest
gradient-preserving nonlinearity (ReLU); the action scorer takes a bounded signed one (tanh); and
the damage op and belief heads take whichever function *is* the quantity being computed (sigmoid,
softmax, clamp)** — where it is not a hyperparameter at all. The one part of that split we did not
author is the SB3 tower's tanh, which is a default rather than a decision; the cheap first move
there is to pin it explicitly, not to change it.

## See also

- `src/agents/model/CLAUDE.md` — architecture-constant rules, the model-versioning playbook,
  the dual-head policy and pointer-head contract
- `designs/ARCHITECTURE.md` — the current phase chain, per-head inputs, the `DamageOperator` block
- `src/agents/training/CLAUDE.md` → Compiled CPU opponents / Compiled GPU trainer — why the B=1 CPU
  forward is the throughput-relevant regime
- `designs/learning/entity_tokens_biases_pointers.md` — why the action head is a pointer, and what
  it scores
- `designs/learning/marginalization_and_uncertainty.md` — why the damage op's sigmoids are
  probabilities being marginalised, not activations
- `designs/learning/conditioning_architectures.md` — the FiLM line and its measured null
