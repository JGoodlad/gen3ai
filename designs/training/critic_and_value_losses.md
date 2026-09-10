# Training — the critic mode and the value losses

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## THE VALUE LOSS has a MODE — `--critic {shaped,winprob}` (`gen3_winprob_critic_mode_v1`)

**Default `shaped`; a flagless run is byte-identical.** Design of record:
`designs/ai_v12/design_winprob_only_critic.md`. The model-side half (the route, the version gate)
is `src/agents/model/CLAUDE.md` → *The CRITIC MODE*; this is what `train()` does about it.

| | `shaped` | `winprob` |
|---|---|---|
| the value TERM | `vf_coef · _value_loss_from_se(...)`, or the HL-Gauss CE at `vf_coef` under `value_from_dist` | `vf_coef · _win_prob_loss(...)` — the head's **BCE against the terminal outcome** |
| noise-scale group | `value` | **`value`** |
| the scalar `value_loss` | the loss | a DIAGNOSTIC only (its term is dropped, `_vf_term = 0.0`), and computed UNCLIPPED |
| `--win-prob-coef` | weights the auxiliary BCE, tagged `aux` | refused — the BCE is the value loss now |
| gate | `win_prob_coef != 0` | `win_prob_coef != 0` **or** the critic — the head's own loss cannot be switched off by a coefficient |

**The `"value"` tag is the point of `gen3_value_diagnostics_v1`'s sibling finding, applied one
critic over.** §1.4 of the design records that under `--value-from-dist` the REAL critic loss was
folded as `_ntg.add("aux", …)` while `_vf_term` was 0.0 — so `train/noise_scale_value` spent that
entire era describing a term with weight zero, and the grad-balance probe had to compensate
separately. The promoted BCE joins `value`, and `grad/value_share`'s term follows the critic
(`win_prob_term if critic_winprob …`) for the same reason: a `grad/value_share` measuring the
FROZEN scalar head's pull is the 2026-07-22 catch (`grad/value_dist_share` stuck at ~0.05).

**`win_prob/critic_*` — the P(win)-currency reliability read, once per rollout.** Under `winprob`
only, from `agents.training.scaffolding.reliability_table` — **imported, never re-implemented**, so
the live number and `python -m main.scaffolding_gauge --reliability` are the same statistic and a
run's series is comparable with the committed 2026-09-06 baseline. It reads the rollout BUFFER's
`values` (which under this critic ARE the P(win) GAE bootstrapped from) against
`win_target`/`win_mask`, i.e. the DEPLOYED quantity — where the `win_prob/ece` family above reads
the head's logits per minibatch. Keys: `critic_{brier,skill,ece,mce,reliability,resolution,uncertainty,decomp_residual,base_rate,n}`.

🚨 **`critic_resolution` IS the meter; `critic_reliability` is not.** A base-rate forecaster scores
a perfect 0 reliability and a useless 0 resolution. The committed baseline measured this head at
reliability ~0.002 against a resolution of 0.062 out of an available 0.182 — already calibrated in
the MEAN, starved of SEPARATION — so a promotion that improves ECE and leaves `critic_resolution`
flat has moved the meter that was never the disease. ⚠️ It is computed on the TRAINING population
rather than the loss-enriched eval quota, so it needs no selection reweighting and its LEVEL is
**not** comparable with the offline gate's — only its trend. An unmeasurable rollout publishes
**`{}`, never zeros**: a calibration of nothing and a perfect calibration must not render the same.

**What `winprob` does to the REWARD, and the one thing it gives up.** The stream becomes the
TERMINAL **win indicator** alone (`--no-hand-shaping` + `--terminal-indicator` +
`--victory-value 1.0` + `--draw-penalty 0`, all four REQUIRED and each named by its own
`combination_checks` refusal), so the undiscounted return is exactly `1{win}` and, at `--gamma 1.0`,
`V(s) = P(win | s)` with no approximation term. The cost is stated rather than buried: **a critic
bounded in [0,1] cannot represent "a timeout is worse than a loss."** `--draw-penalty`'s
`−35 < −30` ordering is unrepresentable there, so the anti-stall pressure is the obs deadline clock
plus **`--arm-no-progress-tax`** (design gap B4) — which re-arms `no_progress_tax` alone under
`--no-hand-shaping`, without reviving the other 24 BIAS terms. **Stall rate and mean episode length
are PRIMARY, kill-condition-bearing endpoints on a `winprob` arm.**

**THE DRAW BRANCH IS EXPLICIT, and `signal/draw_rate` states its frequency** (design §3.2 / gap
B9). `battle.won` is a TRI-STATE — True / False / **None**, the last being a draw or the 250-turn
timeout — and `MaskableAgentWrapper.step` used to reach `0.0` for the third case through a boolean
test, i.e. by accident. It is now a named branch: **a draw is scored as a NOT-WIN by decision**
(`y = 0`), because that makes "P(win)" literally P(win); 0.5 would make the critic systematically
wrong exactly where stalling tempts; and masking the episode out would leave its ~250 decisions
with no learning signal at all. It is SCORED, never dropped — `info["win_draw"]` rides beside
`win_outcome`, `SignalMetricsCallback` counts it on the terminal scan it already runs (both rollout
paths), and `signal/draw_rate` + `signal/n_terminals` publish the rate per rollout.

⚠️ **It is `signal/draw_rate`, not the `train/draw_rate` the design proposes**, for two reasons:
that callback carries a PINNED prefix contract (every row it emits must start with `signal/`), and
on the merits the draw rate is an OUTCOME statistic whose literal siblings — `signal/outcome_win_rate`,
`signal/outcome_entropy` — are computed from the same terminal `info` in the same loop.

⚠️ **The label and the objective DISAGREE about draws under the shaped terminal, and that is a real
property of that composition rather than a defect here.** The label scores a timeout as a not-win
(`y = 0`, the same as a loss) while the reward pays `--draw-penalty` (−35, i.e. WORSE than the −30
loss). Under `--terminal-indicator` they agree: a timeout pays 0.0, exactly like a loss.

**`--gamma` is now a flag** (design gap B6; it was hardcoded at `model_build.py`'s
`InstrumentedMaskablePPO(...)` call). Its `shaped` default is `reward_weights.PBRS_GAMMA` itself
rather than a retyped `0.9999`, and the `PBRS_GAMMA == reward_config.gamma == model.gamma` assert
is now **GATED on a hand potential actually being folded** on both build paths — under
`--no-hand-shaping` every `_fold_*_pbrs` early-returns, so there is nothing for the invariance
claim to be about and an ungated assert would refuse a coherent run. It is **INERT ON A RESUME**
like `--lr`: SB3 restores the checkpoint's own γ, the resume path SAYS so, and it re-points
`reward_config.gamma` at the value actually in force so the two cannot silently disagree.

### 🚨 THE 250-TURN CAP IS A TERMINAL, NOT A TRUNCATION — and it was the other way round (B6)

**THIS ENV NEVER TRUNCATES IN THE SB3 SENSE, AND THAT IS THE WHOLE FINDING.**
`PokeEnv.calc_term_trunc` (`poke_env/environment/env.py:951`) sets EITHER flag only when
`battle.finished`; it then splits a finished battle by *how* it finished — exactly one side wiped ⇒
`terminated`, anything else ⇒ `truncated`. "Anything else" is the **250-turn cap forfeit** and a
genuine **tie**. Both are OUTCOMES. Nothing here is a time limit that interrupted an episode
mid-flight, which is the one thing `TimeLimit.truncated` is supposed to mean.

MEASURED 2026-09-06 on a real bridge battle with `StallConfig.threshold` lowered to 6:
`gen3_env.action_to_order` returns a `ForfeitBattleOrder` at `turn >= threshold` (`== MAX_TURNS` in
production), Showdown answers `|win|<opponent>`, and the boundary reported `finished=True`,
`won=False`, **six mons alive a side**, `terminated=False, truncated=True`.

That flag then becomes `info["TimeLimit.truncated"]` in `DummyVecEnv`/`SubprocVecEnv`, and
`MaskablePPO.collect_rollouts` (`sb3_contrib/ppo_mask/ppo_mask.py:251-260`, mirrored by our
`async_vec_env.collect_rollouts_async:216-219`) does `rewards[idx] += gamma * V(s_last)`. Under
`--critic winprob` the terminal reward is the win indicator — **0** for a cap loss — and γ is
**1**, so the last step's target becomes `0 + 1.0·V(s_last) = V(s_last)`: **a TD error of
identically zero.** The timeout leaves the loss entirely, and a policy that stalls to the cap is
taught nothing about it — while G7's stall rate is a KILL CONDITION on that arm, i.e. the gate
would have been reading a signal the critic never received. Verified by revert: the row reads
exactly `V`.

**`agents/training/wrappers.resolve_episode_end` is the fix** — under `winprob` only, a
`trunc and not term` end is re-labelled TERMINAL. It sits at the END of
`MaskableAgentWrapper.step`, so every outcome consumer above it still sees the flags the sim
produced, and it is the ONE seam upstream of BOTH rollout loops (which is why it is here and not in
`Gen3Env.calc_term_trunc`). The mode arrives as `MaskableAgentWrapper(critic=…)` from
`env_factory`. Pinned by `winprob_truncation_test.py`, including the SB3 composition through a real
`collect_rollouts`.

**`shaped` is UNCHANGED and byte-identical, and what it does is worth stating rather than leaving
implicit:** a cap forfeit and a tie both bootstrap `0.9999·V(s_last)` on top of a terminal reward
that already paid `--draw-penalty`. Two wrongs that partly cancel — the shaped terminal
double-counts the ending while the bootstrap removes it — and re-deriving that composition is a
`shaped`-era question this change does not reopen.

**Two adjacent paths are CLEAN, checked rather than assumed.** A launcher SIGTERM
(`lifecycle.abort_training`) saves a checkpoint and `os._exit`s, so the partially-collected rollout
buffer is discarded and never reaches an update — SB3 does not persist it. The eval
reset-mid-battle forfeit lives in `PokeEnv.reset`, which returns only an observation and fills no
rollout buffer. Neither can leak a truncation into the loss.

⚠️ **CORRECTION to a comment that shipped with the mode: the cap forfeit is a LOSS, not a draw.**
`won_by` (`poke_env/battle/abstract_battle.py:1619-1626`) sets `_won = False` on `|win|<opponent>`;
`won is None` is `|tie|` ALONE. The label is 0.0 either way so nothing downstream moved, but
`signal/draw_rate` counts **ties and not timeouts** — the series that watches the cap is
`signal/stall_rate` / mean episode length, which is why those and not `draw_rate` are G7's kill
condition.

### `--vf-coef` multiplies a BCE now, and the first NON-DEGENERATE update SAYS what that is worth

`--vf-coef` means a different quantity under each critic while keeping its name: under `shaped` an
MSE on a PopArt-normalised shaped return (O(100) unnormalised, on a ±30 scale); under `winprob` the
win-prob head's **BCE against a Bernoulli outcome**, which is `ln 2 ≈ 0.693` per sample at
initialisation and falls. The 0.5 default was tuned against the first and carries no information
about the second, and the normalisation that made the two comparable is refused here.

So `calibration.announce_vf_coef_scale` prints, ONCE, two numbers that answer different questions:
the **raw BCE** (a statement about the HEAD — `ln 2` is chance, well below it means the head
already calls lopsided games) and the **ratio of the value term's shared-trunk gradient norm to
the policy term's** (the statement about the COEFFICIENT — the value norm scales linearly in
`vf_coef`, so the ratio IS the factor to divide it by, `≫1` cut / `≪1` raise).

🚨 **IT USED TO DIVIDE THE VALUE TERM BY `|policy loss|`, AND THAT DENOMINATOR IS DEGENERATE BY
CONSTRUCTION.** On epoch 1 the clipped surrogate has `ratio ≡ 1` and sits at its stationary point,
so `|policy loss| ≈ 0`. The live arm `ai_v12_01_winprob_critic` printed `value term = 0.5 × BCE
0.1330 = 0.0665 against |policy loss| 0.0004 → **165×**` on rollout 1 — and its own gradient series
reads UNREADABLE at rollout 1 (`grad/policy_norm_shared` exactly 0.0 against a value norm of 7.53),
**91×** at rollout 2 and **4.6×** by rollout 17. 165× was not a noisy estimate of 4.6×; it was a
number about the epoch, and a ratio of LOSS MAGNITUDES was the wrong quantity anyway — what
competes on a shared trunk is the gradient.

**The norms are READ from `grad_balance_metrics`, never recomputed.** That read-only
`autograd.grad(retain_graph=True)` probe already runs per-term on every `train()`, so the printed
ratio is exactly `10 ** grad/value_policy_logratio` and no second backward pass is run for a
banner. **The threshold is on the NORM, not the epoch** (`MIN_POLICY_GRAD_NORM`, 1e-6): the probe
samples ONE minibatch per `train()` and it is in epoch 1 by construction, so there is no later
epoch to wait for inside a reading — and epoch 1 is not degenerate in general, since the policy
GRADIENT at `ratio ≡ 1` is `A·∇log π ≠ 0`. It does NOT latch on an update it could not read (no
scorable label, no grad probe, or a policy norm under the floor — the next rollout tries again),
and `_vf_scale_announced` is in `_excluded_save_params`, so a launcher restart re-prints it beside
the startup `[CRITIC]` banner. It is a PRINT, not a scalar: past the first reading the right
instrument is `grad/value_policy_logratio` (0 = balanced). Reading rule: design §5.4.

## `--win-prob-strata-weight` — OPPONENT-STRATIFIED weighting of the win-prob BCE

`gen3_winprob_strata_weight_v1`, config **v115**, landed 2026-09-09 as **arm 7 of the critic
ladder**. Default **0.0 = OFF and the loss is BIT-identical**; **`--critic winprob` is REQUIRED**
(`combination_checks.winprob_strata_needs_the_winprob_critic`). Code:
`instrumented_ppo/value_terms.py::_win_prob_strata_weights` + `_win_prob_loss`, called once per
rollout from `ppo.train()`. Gate: `src/agents/training/winprob_strata_weight_test.py`.

### The defect it targets, and why the treatment is on the LOSS rather than the head

[`winprob_head_refit_2026-09-09`](../research_state/measurements/winprob_head_refit_2026-09-09/README.md)
refit the win head alone on a FROZEN `value_pooled`, out of fold and HT-reweighted, on two
substrates. Against the **terminal 0/1 target the online head actually trains on** it reproduced the
online failure exactly (turn-1–3 between-opponent spread ratio 0.149 → 0.000 on arm A, 0.066 → 0.068
on the ladder control, both deltas straddling zero); against the same label with its per-episode
variance removed the **same head, same features, same optimiser** recovered a DETECTED part
(0.149 → 0.323, 0.066 → 0.259). A head initialised FROM the online weights lands where a scratch
head lands under both targets, so the head is **not in a basin** and the head-side-optimisation
treatment class (value replay, periodic refit, head-specific lr) is RULED OUT.

**The mechanism is arithmetic** (§6 of that measurement, `variance_shares.json`):

| | total variance of the label | share BETWEEN (cycle, opponent) cells |
|---|---|---|
| arm A · terminal 0/1 | 0.1649 | **10.2 %** |
| CTRL · terminal 0/1 | 0.1595 | **14.4 %** |
| arm A · conditional | 0.0367 | 24.0 % |
| CTRL · conditional | 0.0341 | 58.8 % |

Nine-tenths of the terminal label is per-episode noise plus within-cell board state, so a head
minimising a proper scoring rule buys its resolution wherever it is cheapest — the board and its
own team — and the opponent component is ~10 % of the objective **at any sample size**. That
statement is scale-free, which is what carries it across the 2,500× gap between the offline 951
battles and the online head's ~2.4 M episodes. §11 names this lever as *"the highest ratio of
expected effect to cost on this list"*: it raises the between-cell share **directly**, needing no
new labels, no new machinery and no extra rollout cost, where counterfactual labels buy the same
re-pricing at the cost of producing them.

### The arithmetic

Over the rollout buffer's **KNOWN** rows (`win_mask == 1`), with per-class counts `n_c`, frequencies
`f_c = n_c / N` and `s` = the flag:

```
raw_c = min(f_c ** (-s), CAP)          CAP = 8.0   (_STRATA_WEIGHT_CAP)
w_c   = raw_c / Z,   Z = (Σ_c n_c · raw_c) / N     ⇒  mean(w) over the known rows == 1
loss  = Σ_i BCE_i · mask_i · w_{c(i)} / N
```

Three properties, each of which a test pins:

* **`s = 0` returns no vector at all**, so the caller takes the original masked mean UNCHANGED and
  the loss is bit-identical — not "approximately equal". Every archived arm is that baseline.
* **The mean weight is exactly 1**, so `s` re-prices the MIX without rescaling the value gradient.
  Without this, an arm confounds "balanced the classes" with "raised the critic's step size".
* **At `s = 1` and no clipping, `n_c · w_c` is constant in `c`** — each class contributes `N / C` to
  the objective. `s` interpolates the exponent, monotonically.

The **DENOMINATOR stays `N`**, not `Σ mask·w`. Dividing by the weighted count would renormalise per
minibatch and undo the buffer-level balance the weights were computed to produce.

**Computed ONCE per rollout, over the whole buffer, and held constant for every epoch and
minibatch.** Per-minibatch frequencies would make the class balance itself a sampling-noise term,
and the mean-weight-1 normalisation is only meaningful over the population the frequencies came
from. It is read after `_align_opp_intent_labels`, the only writer of `opp_class` in the call (a
semantic no-op — the class is constant within an episode).

### The VOCABULARY, and the option that was rejected

**Shipped: the four `opp_class` codes** — `bot` / `pool` / `stable` / `exploiter`
(`agents.model.opp_intent.OPP_CLASS_NAMES`). That is the label key `gen3_env` declares under the
win-prob gate, and it is **all the env knows per step**.

🚨 **REJECTED — one class per BOT NAME plus a single `selfplay` class**, which is the finer
stratification the measurement's 14-opponent cell structure would suggest. It is not available:
the bot archetype and the pool snapshot's step are drawn per EPISODE inside
`MaskableAgentWrapper._select_episode_opponent` and **never reach the observation**; shipping it
would mean a new obs key per identity, which `value_sidecar.py` had already declined for the same
reason. **The consequence is stated rather than buried: this lever balances the between-CLASS
share, and the residual heterogeneity WITHIN the bot class (random vs. the heuristics) stays in
episode proportion.** It bounds what the arm can move, and it is the first thing to revisit if arm 7
moves the meters part-way.

The two-way `bot` / `selfplay` collapse was also considered and NOT taken: `stable` and `exploiter`
are genuinely different opponent populations when they are present at all, and collapsing them
would silently re-price a `--stable-opponents` or `--exploiter` run's mix in a way its operator
never asked for. Four codes is what the env records, so four codes is what the objective is
stratified by.

### 🚨 THE CAP BINDS AT THE PRODUCTION MIX, DELIBERATELY

Uncapped, a class holding 1/1000 of the buffer would ask for 1000× and one rollout's handful of rows
would carry the whole value gradient. `CAP = 8.0` binds at `f < 1/8` when `s = 1`, and at the
measured post-promotion mix it **does** bind:

| | frequency | raw (uncapped) | raw (capped) | `w_c` | share of the objective |
|---|---|---|---|---|---|
| `bot` | 0.10 | 10.0 | **8.0** | 4.444 | **0.444** |
| `pool` | 0.90 | 1.111 | 1.111 | 0.617 | **0.556** |

So `s = 1` delivers **44/56, not 50/50** — a **4.4×** re-pricing of the between-class signal instead
of 5×, in exchange for a hard bound on the per-row gradient weight. A reader comparing
`strata_share_*` against parity has to know this, so `winprob_strata_weight_test.py` pins the exact
numbers and this table is what it pins them against.

### The TB read — `win_prob/strata_*`, once per rollout

| key | what it says |
|---|---|
| `strata_weight` · `strata_rows` · `strata_n_classes` | the `s` in force, the KNOWN rows it was computed over, how many classes were present |
| `strata_w_<class>` | the per-class weight vector — the lever itself, one series per class |
| `strata_frac_<class>` | the class's raw episode proportion — the BEFORE picture |
| `strata_share_<class>` | its share of the objective AFTER weighting — the AFTER picture |
| `strata_w_entropy` | normalised entropy of that share distribution; **1.0 = perfectly balanced**, and the one number that says the lever landed |
| `strata_w_min` / `strata_w_max` / `strata_capped` | the spread, and how many classes hit the cap |
| `loss` vs `loss_unweighted` (per minibatch) | what the objective actually minimised, vs what it would have been unweighted — this separates "the weights moved the loss" from "the head got better" |
| `strata_row_w_mean` (per minibatch) | the realised mean weight; ≈1 or the normalisation is broken |
| `strata_active` | **1 = the weights were applied; 0 = they were not, and the row beside it says why** |

🚨 **AN ABSENT `strata_*` FAMILY MEANS THE FLAG IS OFF, AND NOTHING ELSE.** Whenever the flag is on
the family is published EVEN WHEN NO WEIGHTING APPLIES, carrying `strata_active = 0` with
`strata_n_classes` and `strata_rows` saying why — one class present (a bot-only curriculum: every
`--debug` run, and any run before the self-play pool seeds), or a rollout whose labels were never
back-filled. **This build's first smoke landed in exactly that state and could not be told apart
from a plumbing break**, which is why the inactive case reports rather than vanishing.

### What it is NOT applied to

The counterfactual and twin-head callers of `_win_prob_loss` (`cf_terms.py`,
`_cf_twin_onpolicy_terms`) stay **unweighted**, and a test pins that: they score FOREIGN recorded
states whose opponent mix belongs to the label factory, not to this rollout, so reweighting them by
this rollout's frequencies would be a category error.

### Recording and resume

`win_prob_strata_weight` is a **v115 `ModelVersion` field of the `td_aux_coef` class**: recorded in
`model_config.json` for provenance and for flagless-resume read-back (`_resolve`), **never** compared
by `check_compatible` or any `check_*` — it reweights a loss and touches no forward pass or weight
shape, and gating a frozen eval/pool/distill opponent on it would be a false rejection. A pre-v115
config migrates to `0.0`, which is a RECORD and not a guess: the field did not exist. It is **not** a
`flag_registry.py` row — that registry declares EXTRACTOR toggles, and this builds no module (the
`td_aux_coef` / `cf_*_coef` / `intent_label_bot_weight` precedent). It IS declared in
`arch_tables._COEF_MODULE` (→ `win_head`) so a production config that ever adopts it cannot be
silently dropped from the generated table the way `intent_label_bot_weight` was from v97.

## `--win-prob-lambda` — λ-RETURN targets for the win-prob BCE

`gen3_winprob_lambda_v1` (config **v116**, the critic ladder's **arm 8**). Default
**`1.0` = OFF and the loss is BIT-identical**; **`--critic winprob` is REQUIRED** (refused in
`combination_checks`). The strata weight above re-prices *which states* the BCE is bought from;
this one changes *what the BCE regresses toward*.

### The defect it targets

The same one, from the other side. Under `--critic winprob` the value loss is a BCE against **one
terminal bit copied to every state of the episode** (`WinProbLabelCallback`). The head refit
([`winprob_head_refit_2026-09-09`](../research_state/measurements/winprob_head_refit_2026-09-09/README.md)
§6, §11) proved the win-prob critic's conditional miscalibration is a **TARGET** defect, not a head
defect: only **10.2 % / 14.4 %** of that label's variance lies BETWEEN (cycle, opponent) cells, so an
early-stopped on-policy learner shrinks the weak axes toward the marginal. The measured shape of
that is the turn-1 read — the critic barely separates opponents or its own team at turn 1
(between-opponent spread ratio **~0.1**) although the features carry both, while **mid- and
late-game values already separate opponents far better (~0.5–0.8 over all states)**.

**So the information exists inside the episode; it just never reaches turn 1.** A λ-return hands an
early state a blend of the network's OWN later estimates, and that channel carries far less noise
than the single terminal draw.

### The recursion

γ = 1 and the clean-world stream is **terminal-only** (`--no-hand-shaping --terminal-indicator`), so
an n-step return has no intermediate reward term at all and **IS** `V(s[t+n])`. The λ-weighted
average over n collapses to one backward pass per episode:

```
row t ENDS its episode   ⇒  G[t] = y                       (the outcome, exactly)
otherwise                ⇒  G[t] = (1−λ)·V(s[t+1]) + λ·G[t+1]
```

Expanded, a state **d** steps from its terminal keeps weight **λ^d on the outcome** and the rest on
later values — published as `win_prob/lambda_bootstrap_frac` (the share of scored rows with
λ^d < 0.5) rather than left to be assumed. At λ = 0.9 the outcome still holds half the target 6–7
steps out and ~4 % of it at 30.

🚨 **`V` is the RECORDED, pre-update value** — `rollout_buffer.values`, which under this critic *is*
`sigmoid(win logit) ∈ [0,1]` (`policy._critic_value`). Recorded and not re-forwarded inside
`train()` on purpose: a target recomputed from the CURRENT weights would move under its own gradient
across the 10 epochs, which is the classic self-referential-target divergence. The collection-time
values are a fixed point of this rollout by construction.

The loss itself is **unchanged** — the same masked-mean `binary_cross_entropy_with_logits`, now
against a SOFT target in [0,1]. BCE is a proper scoring rule for the target's *expectation*, so a
soft target is exactly the right generalisation and nothing about the head, the coefficient or the
`vf_coef` routing moves.

### The BUFFER BOUNDARY — `--win-prob-lambda-truncated {bootstrap,mask}`

A rollout ends mid-episode in every env column. Those states have **no outcome**, so today they
carry `win_mask = 0` and are excluded. Under λ < 1 their successor is `s_T`, whose value is the same
`model._last_obs` bootstrap SB3's own GAE uses (and `winprob_pbrs` takes), so the recursion needs no
special case at all: `G = (1−λ)·V(s_T) + λ·V(s_T) = V(s_T)`.

* **`bootstrap`** (the default) gives them that target and **UNMASKS** them — states that carry no
  target today now contribute. `win_prob/lambda_unmasked` counts them per rollout.
* **`mask`** leaves them excluded exactly as today.

It is a FLAG rather than a constant so a read can attribute an effect to *the target change* rather
than to *the extra rows*. It is **INERT at λ = 1.0**: the recursion is skipped whole, which is what
makes the default bit-identical including this convention.

🚨 **An episode that ended with NO recorded outcome is never unmasked.** The trailing in-progress
segment is computed from `episode_starts` and `model._last_episode_starts`, not inferred from
`win_mask == 0` — an episode that finished without a `win_outcome` in its info is ALSO unlabelled,
its rows would anchor at `y = 0`, and bootstrapping those would train the head against a fabricated
loss. A test pins it.

### The TB read — `win_prob/lambda_*`, once per rollout

Computed in the callback (it needs the buffer's `[n_steps, n_envs]` shape, before `get()` flattens
it) and folded into the ordinary `win_prob/` prefix in `train()`. **An ABSENT `lambda_*` family
means λ = 1.0, and nothing else.**

| tag | what it says |
|---|---|
| `lambda` | the λ in force |
| `lambda_rows` | states scored by the BCE this rollout |
| `lambda_unmasked` | states the truncation branch ADDED — read before attributing anything to λ |
| `lambda_bootstrap_frac` | share of scored rows whose target is MOSTLY later values (λ^d < 0.5) |
| `lambda_weight_mean` | mean λ^d — how much of the objective is still the outcome |
| `lambda_target_shift` | mean \|G − y\| on rows that HAVE an outcome — how far the targets moved |
| `lambda_loss` / `lambda_loss_terminal` | BCE of the RECORDED V against G, and against y |
| `lambda_truncated_bootstrap` | 1.0 = truncated episodes were bootstrapped in; 0.0 = masked |
| `lambda_bootstrap_fallback` | 1.0 = the `_last_obs` forward was unavailable and V(s[last]) stood in |

🚨 **`lambda_loss` and `lambda_loss_terminal` are scored on the SAME states with the SAME
predictions** — the collector's recorded V — so their difference isolates the target change and
cannot be a step of learning. They are deliberately not a post-update per-minibatch pair: the
recursion overwrites `win_target` in place, so `y` does not survive the buffer's shuffle, and a
post-update pair would confound the two effects.

### How it composes

* **`--win-prob-strata-weight`** — orthogonal and composable. Strata multiplies each row's BCE by
  its opponent class's weight; λ changes what that row's BCE is *against*. Neither reads the other.
* **The counterfactual labels (`--cf-winprob-coef`)** — **disjoint, no precedence needed.** The cf
  term (`cf_terms.cf_winprob_term`) re-applies the head to FOREIGN recorded states sampled from
  `<run>/cf_labels/` with their own tight-MC labels; it never touches `win_target` and never sees a
  rollout row. The two losses are added, as they are today.
* 🚨 **The value SIDECAR's `target` column follows the flag.** `ValueSidecarCallback` runs
  immediately after `WinProbLabelCallback` and reads `win_target` — so under λ < 1 that column is
  the **λ-return**, not the raw outcome, and `target_known` covers the rows the truncation branch
  unmasked. The header's `win_prob_lambda` field says which.

### Recording and resume

`win_prob_lambda` and `win_prob_lambda_truncated` are **v116 `ModelVersion` fields of the
`td_aux_coef` class**: recorded in `model_config.json` for provenance and for flagless-resume
read-back (`_resolve`), **never** compared by `check_compatible` — they re-aim a loss target
computed in a post-collection callback and touch no forward pass or weight shape. A pre-v116 config
migrates to `1.0` / `"bootstrap"`, which is a RECORD and not a guess: λ = 1.0 **is** the
terminal-bit target every prior run trained against. Not a `flag_registry.py` row (no extractor
module); `win_prob_lambda` IS declared in `arch_tables._COEF_MODULE` (→ `win_head`).

## `--win-prob-dense-aux` — DENSE AUXILIARY targets beside the win-prob BCE

`gen3_dense_aux_v1` (config **v117**, the critic ladder's **arm 9**). Default
**`0.0` = OFF and BIT-identical** — bit-identical by not BUILDING the head at all, so there is no
module, no obs key, no callback and no term. **`--critic winprob` is REQUIRED** (refused in
`combination_checks`), and `--win-prob-mode != none` is a `flag_registry` `requires` the extractor
constructor enforces.

The three flags above all act on ONE loss — one scales it, strata re-prices its MIX, λ re-aims it.
This one does not touch that loss at all. It **adds 25 targets beside it**, on the same
`value_pooled`.

### The defect it targets, and the literature's answer

The same measurement, taken to its conclusion. Under `--critic winprob` the value loss is a BCE
against one terminal bit copied to every state, and only **~10 %** of that label's variance lies
BETWEEN opponents
([`winprob_head_refit_2026-09-09`](../research_state/measurements/winprob_head_refit_2026-09-09/README.md)
§6), so the head shrinks the weak axes — opponent, own team — toward the marginal although its
features carry them. **Four 10M levers moved nothing at ±0.01 on bot resolution** (ledger *THE ARMS
AT 400 GAMES*). Every one of them re-weighted, re-aimed or re-priced the SAME one bit.

KataGo (Wu 2019, §3) answers a one-bit terminal signal differently: keep the win target and add
**auxiliary targets that share the win's CAUSE** — ownership of every point of the board, and the
final score — reported there as a large gain in learning efficiency. Our analogue of "ownership of
every point" is **per-Pokémon end-of-battle outcomes**. That is 25 numbers per state instead of one
bit, and 24 of them are facts about a NAMED ENTITY the state's own observation also carries, so the
gradient they deliver runs along exactly the per-entity axes a pooled bit cannot separate.

### The head and its 25 outputs

`DenseAuxHead` (`agents/model/dense_aux_head.py`): `Linear(D_MODEL, 64) → ReLU → Linear(64, 25)`,
output layer ZERO-INIT, reading the same `value_pooled` the win head reads.

| outputs | target | scored by |
|---|---|---|
| `0..11` | SURVIVAL of slot k (our 6, then theirs 6) — 1 if un-fainted at termination | BCE, targets in {0,1} |
| `12..23` | slot k's FINAL HP FRACTION at termination, in [0,1] | BCE against the soft target |
| `24` | TURNS LEFT: `log1p(terminal_turn − this_turn) / log1p(250)`, clipped | BCE against the soft target |

`aux_loss = coef × mean(the three masked-mean terms present)` — a mean of TERMS, not a pooled mean
over 25 columns, so twelve survival outputs cannot outvote the one turns output twelve to one, and a
minibatch in which a block is entirely masked DROPS that block rather than contributing a zero.

🚨 **Every output is a sigmoid logit scored by BCE, including the two that are not Bernoulli means**,
and that uniformity is a decision with three reasons. (1) A sigmoid keeps a bounded-in-[0,1] target
in range BY CONSTRUCTION — no clamping, and no MSE-through-a-saturating-sigmoid vanishing gradient
at 0 and 1, which is exactly where the HP mass sits (a fainted mon is exactly 0). (2) BCE with a
soft target is a proper scoring rule for a [0,1] mean, the same family as the win head's own loss.
(3) One family puts all three terms in the same NATS scale, so the single coefficient means one
thing across them. ⚠️ **The price: a BCE against a soft target has a non-zero floor** (the target's
own entropy), so `aux_hp_loss` and `aux_turns_loss` never approach 0 even for a perfect predictor —
`aux_hp_mae` and `aux_turns_mae` are the interpretable reads and are published beside them.

🚨 **THE PER-SIDE KO COUNTS ARE DERIVED, NEVER PREDICTED.** `6 − Σ survived` per side is a linear
function of the survival block, so a separate output would be a linearly-dependent target adding no
information — and it is the ONE target that could not honour the mask, because a count is a sum over
slots some of which are unscored. It is published as a meter (`win_prob/aux_ko_mae_*`) computed from
the survival head over the UNMASKED slots.

### The targets, and the TWO masks that are ANDed

`agents/training/dense_aux.py` builds them from the trainee's own finished `battle1` at the done
step — the same seam `info["win_outcome"]` is published from, so the opponent's HP here is the
PUBLICLY known percentage, which is what actually happened. `DenseAuxLabelCallback` back-fills them
to every state of the episode exactly as `WinProbLabelCallback` back-fills the win bit, and masks
the trailing in-progress episode for exactly the same reason.

Slot order is the OBSERVATION's own: our slots are `battle.team` order, theirs are
`battle.opponent_team` REVEAL order — read through the encoder's own
`ObservationEncoder.get_team_list`, never re-derived, so the target and the features cannot disagree.

A slot is scored only where all three hold:

1. **the episode finished inside this rollout buffer** (the `win_mask` condition, verbatim);
2. **TERMINAL AVAILABILITY** — the slot has an end-of-battle fact at all. 🚨 A battle can end with
   three of the opponent's party never revealed; those slots have no observed fainted flag and no
   observed HP and are MASKED, never fabricated as "alive at full HP". That label would be wrong in
   a DIRECTION — the un-revealed mons of a team we beat are disproportionately the ones it never got
   to send out;
3. **PER-STATE VISIBILITY** — the slot names an entity THIS state's observation carries. Reveal
   order makes the final opponent order a prefix-stable extension of every earlier state's, so slot
   i at turn t is the same mon as slot i at termination for every i below the reveal count and is an
   all-zero absent block above it. 🚨 Scoring an unrevealed slot would anchor a label to a feature
   block encoding nothing — the same "target with no feature correspondence" defect this arm exists
   to remove, one level down. The env emits the visibility as a REAL present-state value in the
   `aux_mask` placeholder and the callback ANDs it in.

The turns output names no slot, so it is masked by (1) alone.

### Where the gradient goes — and why it is NOT detached

The head is **not called by the forward at all** (the `CfEvidentialHead` contract), so pi and vf are
BIT-IDENTICAL at an ARBITRARY weight in it and a rollout, an eval and the prober pay nothing. The
term (`ppo.train()`, folded as an `aux` term at `--win-prob-dense-aux`, never at `vf_coef` — there
is one critic and these are not it) applies it to the `value_pooled` the minibatch's
`evaluate_actions` stashed, **live and undetached**. That is the arm and not a leak: the point of a
dense auxiliary target is gradient into the shared trunk along axes the pooled bit cannot carry,
exactly as the win-prob loss does under `shaping` (which `--critic winprob` implies).
`grad/dense_aux_share` is the verification, and unlike `grad/cf_evidential_share` it must NOT read 0.

### λ, and the precedence rule

🚨 **`--win-prob-lambda` DOES NOT REACH THESE TARGETS, and must not.** They are TERMINAL FACTS, not
returns: there is no recorded per-state estimate of "slot 4's final HP" for a recursion to blend, and
blending would make the target a mixture of an outcome and a prediction of a different quantity. The
precedence is STRUCTURAL rather than conventional — the λ recursion overwrites `win_target` /
`win_mask` in place and names no `aux_*` key — and `dense_aux_test` pins it with a source scan.
Under λ < 1 the two simply coexist: the win term regresses toward a soft λ-return, the aux terms
toward the battle's own ending.

Orthogonal to `--win-prob-strata-weight` (that one weights rows of a different loss) and disjoint
from the counterfactual labels (foreign states, own labels, no shared key).

### TensorBoard

All under `win_prob/aux_*`, so an ABSENT family means the flag is off and nothing else. From the
label callback: `aux_active`, `aux_coverage` (rows whose episode finished inside the buffer),
**`aux_masked_frac`** (the fraction of the 25 outputs NOT scored — the honest size of the opponent's
unrevealed party, and the number to read before any aux loss). From the loss: `aux_loss`,
`aux_survival_loss`, `aux_hp_loss`, `aux_turns_loss`, `aux_scored_frac`, **`aux_auc_own` vs
`aux_auc_opp`** (survival AUC per SIDE — a pooled AUC would hide exactly the asymmetry the arm is
built to move; ties take their average rank, so a zero-init head reads exactly 0.5, and a side with
one class absent is OMITTED rather than logged), `aux_hp_mae`, `aux_turns_mae`,
`aux_ko_mae_own` / `aux_ko_mae_opp`.

### Recording and resume

ONE flag, TWO recorded v117 fields, because they are gated differently.

* **`dense_aux` (bool) is STRUCTURAL** — the `value_true_team` mould. ON builds a head whose params
  ARE the state_dict delta; its only output is a training-side loss, so no width changes anywhere
  and **a bool compare in `check_compatible` is the only thing that could reject a flipped flag**. A
  resume that dropped it would silently delete a trained head; one that added it would begin
  supervising a fresh zero-init head inside a trained trunk and call the result the same arm. It is
  a `flag_registry.py` row, `derived=True` off `win_prob_dense_aux` (the `opp_belief_slots` /
  `opp_intent` pattern), `family=CRITIC`, `requires=("win_prob_mode",)`.
* **`win_prob_dense_aux` (float) is the `td_aux_coef` class** — recorded for provenance and
  flagless-resume read-back, never compared. So **a resume may RE-DOSE the arm freely and may not
  add or remove its parameters**, which is the honest split. Declared in
  `arch_tables._COEF_MODULE` → `dense_aux_head`.

**NO ARCH_SIGNATURE bump**, for `value_true_team`'s reasons: the 2501-dim observation VECTOR is
unchanged (the three label keys ride SEPARATE Dict keys, the `win_target` precedent — and none of
them is an `extra_obs_keys` row, because the extractor never reads them), no existing module moves,
the head is built LAST and the forward never calls it. So an OFF run on this code is bit-identical
to the same run on v116 and every existing checkpoint still resumes. `family=CRITIC` keeps it off
the ARCH surface, so `--arch production --win-prob-dense-aux 1.0` is the documented launch and
`checkargs` reports ARCH SURFACE = production mirror.

## PopArt value-target normalization (`--use-popart`)

The fix for the swamping the diagnostics above reveal. `train()` reads `self.popart =
getattr(self.policy, "popart", None)` (built by the policy when `--use-popart`; see
`src/agents/model/CLAUDE.md` → PopArt for the math + version-checking). When present: once per
`train()` (before the epochs) `popart.update(self.rollout_buffer.returns, self.policy.value_net)`
advances the running `(mu, sigma)` **and** POP-rescales `value_net`; the value loss then becomes
`MSE(popart.normalize(returns), popart.normalize(values))` — the **normalized**-space loss, so the
value gradient into the shared trunk drops by ≈`sigma²` and stops swamping the policy. The policy's
value sites de-normalize, so `rollout_buffer.values` / GAE / advantages stay real-unit — the policy
path is untouched. **`--use-popart` requires an explicit `--clip-range-vf none`** (errors otherwise —
self-documenting config; clipping is unnecessary with value normalization, and would clip in
un-normalized units). New diagnostics ride the same generic metrics path:
`popart/mu`, `popart/sigma` (watch them track `train/return_mean`/`return_std`),
`popart/value_weight_norm` (POP keeps it bounded). Under PopArt `train/value_loss` is the normalized
loss (≈O(1)) and `grad/value_policy_logratio` should fall from a large positive value toward ~0 (the
aux-independent value/policy balance — `grad/value_share` also drops but moves with the aux count, so the
log-ratio is the cleaner confirmation it worked).

## Tail-weighted value loss (`--value-tail-weight`)

A probe-driven critic-tail lever (off by default). A representation probe found the critic's TD-residual
tail is fat and barely anticipated (the V-tail crater the `eval/td_resid_tail` CVaR@5% already tracks),
so `InstrumentedMaskablePPO._value_loss_from_se` replaces the plain `F.mse_loss` at all **three** value
sites (PopArt-normalized / unclipped / clipped) with a **CVaR blend**:
`value_loss = (1−β)·MSE + β·mean(worst _VALUE_TAIL_FRAC=10% squared errors)`, computed in whichever
space the branch uses (NORMALIZED under PopArt, so the tail selection matches the loss scale). At **β=0
it is `se.mean()`, byte-identical to `F.mse_loss`** (the default no-op). β>0 makes the critic prioritise
the big over-claim misses it under-prices; it is **symmetric in error sign**, so V stays an unbiased
mean estimate and the GAE advantages the policy reads are unaffected — a weighting change, not a new
target. The hparam is set on the model after construction (like `_async_rollout`), **resume-immutable**
(recorded in `model_config.json`, FATAL to change on resume via `ModelVersion.check_value_tail_weight`,
`MODEL_CONFIG_VERSION` v11; excluded from `check_compatible` since a frozen opponent never runs the value
loss), and **not weight-shape** (no `ARCH_SIGNATURE` bump). The v10
`value_active_readout` value-head fix that used to pair with it is **deleted** (v88
`gen3_dead_flag_purge_v1` — it was never enabled in a gen-8+ run and the multi-seed readout /
`--value-threat-inject` superseded it; a checkpoint recording it ON is refused by the migration). Validate by watching
`eval/td_resid_tail` fall.
Tests: `instrumented_ppo_test.py` (β=0 == MSE, β>0 == the exact blend).

## TD-consistency auxiliary (`--td-aux-coef`, `td_aux.py`)

**What it fixes.** The critic's only signal is a PER-STATE regression, `MSE(V(s_t), G_t)`. That
constrains each state's LEVEL and says nothing about the DIFFERENCE between two adjacent states — so
independent per-state noise ε in V arrives in `ΔV` at `2·Var(ε)`, exactly where the truth is nearly
constant. Since ΔV is what GAE reads, that is injected advantage noise on **every** transition, not
just the dramatic ones. `--td-aux-coef λ` adds the Bellman identity the critic already owes, as an
explicit loss:

```
loss += λ · mean_pairs[ ( V(s_t) − r_t − γ·V(s_{t+1}) )² ]
```

Both residual ends carry gradient (the residual-gradient / Baird form — see the *Cons* in the
pre-registration). `λ = 0.0` is the default and the whole block is skipped, so an OFF run is
byte-identical. **Pre-registered band: 1.0–3.0, 3.0 the favourite; `λ ≤ 0.1` measured significantly
WORSE than control offline, so the small-coef regime is to be avoided, not treated as "a bit of the
effect".** Full pre-registration (rung-1 evidence, the honest ceiling, the rung-2 gates):
`designs/research_state/levers/td_consistency_aux.md` (ledger C5). Do not edit that file — it is the
pre-registration.

**Where the pairs come from — this is the whole engineering problem.** `RolloutBuffer.get()` yields
a RANDOM PERMUTATION, so a PPO minibatch contains **no adjacent pairs at all**; the pairs have to be
drawn from the buffer's surviving `[n_steps, n_envs]` structure. `td_aux.sample_contiguous_pairs`
draws `TD_AUX_STATES` (512) rows as contiguous per-env runs of `TD_AUX_SEG_LEN` (16) and pairs their
adjacent rows. Four facts make it correct:

- **Row convention.** After the first `get()`, `observations` are `swap_and_flatten`ed to ENV-MAJOR
  (`row = env·n_steps + t`), so temporal adjacency survives; the sampler returns rows in exactly
  that convention and `_td_aux_term` **raises** if `generator_ready` is False rather than indexing
  an un-flattened array (which would silently mis-pair states with rewards at any `n_envs > 1`).
- **`rewards` / `episode_starts` are NOT in `get()`'s flatten list**, so they stay `[n_steps,
  n_envs]` and are read in their native shape (rewards are swapped to env-major at use).
- **Episode boundaries DROP the pair, never zero it.** `episode_starts[t+1] == 1` means the
  successor begins a new episode, so (t, t+1) is not a transition; zeroing would train
  `V(s_t) → r_t` at every battle end. This also disposes of SB3's time-limit bootstrap (which folds
  `γ·V(s_term)` into the stored reward at the done step): that row's successor always starts an
  episode, so the pair never forms.
- **Segments, not random pairs.** L contiguous states serve L−1 pairs off L forwards — the
  "K+1 forwards serve K pairs" economy the pre-registration calls for, ~2× cheaper per pair than
  independent pair sampling. Rung 1 also found whole-battle batching beat a random-permutation
  control by 12%, so the within-segment correlation is a feature.

**It runs per MINIBATCH, with its own sample and its own critic forward** — modelled on the
search-teacher / OPD folds, not on the once-per-`train()` diagnostic probes. Those are read-only;
this one carries gradient, and a once-per-`train()` fold would give it ONE contribution against the
value loss's `n_epochs × n_minibatches` (~240 in production), so λ would have to be ~240× rung-1's
band to mean the same thing. Cost is bounded by `TD_AUX_STATES`, not by `batch_size`: one extra
512-state critic forward per minibatch, ≈10% of the train step at production shapes.

**The value path is `policy.predict_values`, never a hand-rolled one.** That method is what routes
to the DISTRIBUTIONAL head's mean under `--value-from-dist` (where the scalar `value_net` is FROZEN)
and applies PopArt's de-normalization — reading `value_net` directly would train a critic the run
does not use.

**Units.** `predict_values` returns REAL-unit values and the buffer's rewards are real-unit, so the
raw residual is real-unit. But under PopArt the value loss trains in NORMALIZED space, so the
residual is divided by σ — which *is* the normalized-space residual, since
`normalize(V) − normalize(r + γV′) = (V − r − γV′)/σ` (the μ cancels). λ therefore keeps the meaning
rung 1 calibrated in both regimes; σ = 1.0 with PopArt off.

**Metrics (`td_aux/` prefix).** `resid_rms` is the headline — the quantity being minimised, the live
counterpart of the offline ΔV-dispersion instrument, and it should FALL. `resid_mean` (SIGNED) is
the no-harm watch: rung 1's decomposition says this is dispersion suppression, so a bias drifting
away from ~0 means the residual-gradient term is shifting the LEVEL rather than tightening it — read
it beside `train/explained_variance`. Also `loss`, `n_pairs`, `scale` (the σ the residual is
expressed in) and `pair_drop_frac` (share of candidate pairs lost to episode boundaries). The
shared-trunk pull rides `grad/td_aux_share` + `grad/td_aux_policy_cosine`; the term reaches the trunk
through the CRITIC path only, so `td_aux_share` against `value_share` is the read for "is the
consistency term crowding out the level regression it is meant to complement".

**Class: `training_coef`.** Scales a loss, touches no forward pass ⇒ NO `ARCH_SIGNATURE` bump, NOT in
`check_compatible` and no `check_*` of its own; recorded on `ModelVersion` (`MODEL_CONFIG_VERSION`
v90) purely for provenance and so a **flagless resume inherits it** via `_resolve`, exactly like
`--opp-belief-aux-coef`. It is deliberately NOT in `agents/model/flag_registry.py` — that registry's
scope is extractor architecture toggles, and this reaches the extractor not at all.

Tests: `td_aux_test.py` — the sampler (env-major row convention, (t, t+1) adjacency, boundary pairs
DROPPED not zeroed, the all-boundary degenerate → `None` not 0.0, the segment economy, fail-loud on a
flattened `episode_starts`), the residual math on a hand-built case, the PopArt scale identity, both
ends carrying gradient, and on a REAL `train()`: coef-0 byte-identity (asserted twice — identical
parameters AND the sampler monkeypatched to raise, so a future sampler change cannot perturb an off
run), coef>0 moving the update and logging every metric, gradient landing on `value_net`, and the
un-flattened-buffer refusal.

## Distributional value head (`--value-dist-mode` / `--value-dist-coef`)

The training half of the v29 interpretability side head (model side: `src/agents/model/CLAUDE.md` →
distributional value head). A categorical readout off `value_pooled` whose softmax is the critic's
predicted **return DISTRIBUTION** — the shape the scalar V collapses (sharp = confident, wide =
uncertain, bimodal = coinflip). **Phase A** (interpretability-only): it does NOT replace the scalar
critic, so the GAE/advantage/value-loss path is untouched — this loss is an ADD-ON, like the win-prob
aux. Design + the K1 honesty frame: `designs/ai_v6/design_distributional_value_critic.md`.

- **Loss (`instrumented_ppo._value_dist_loss`).** **HL-Gauss** (Farebrother et al. 2024): build a
  Gaussian-smoothed soft target by integrating `N(target, σ_g²)` (σ_g = 0.75·Δ) over each atom's bin,
  with the two EDGE bins absorbing the outer tails (graceful out-of-support handling), then cross-entropy
  against the head's `log_softmax`. `train()` reads the stashed `last_value_dist_logits` + the rollout
  return as the target, **PopArt-normalized when the scalar critic is** (so the target lands in the head's
  support space — set `--value-dist-vmin/vmax` to a normalized range like ±5 under `--use-popart`). Folded
  at `value_dist_coef`. Pure + static → unit-tested in `value_dist_loss_test.py`.
- **Metrics (`value_dist/*`).** Aggregate interpretability health under its own TB prefix (the
  `grad/`/`popart/`/`win_prob/` group convention): `ce`, `entropy` + `std` (fall as the critic sharpens),
  `pit_mean` (≈ 0.5 ⟺ **calibrated** — the PIT anchor), `mean_abs_err` (`|E[Z] − return|` in support
  units). Ride the generic logger → TensorBoard + launcher TUI (`value_dist/*` labels in `format.py`).
- **Versioning.** `value_dist_mode` (str) + `value_dist_bins` (int) are version-checked structural toggles
  (fresh-only); the support `vmin`/`vmax` is resume-immutable (`check_value_dist`); `value_dist_coef` is
  **training-only**, read back on a flagless resume (like `win_prob_coef`). Threaded into
  `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles`.
- **Forensic trace + prober.** `RLPlayer._value_dist` reads the stashed logits at capture (softmax ⇒ the
  per-atom distribution) → `BattleRecorder.states_arrays` writes a `value_dist [T, bins]` npz array (key
  OMITTED when the head is off → the prober's KeyError "unavailable" path; NaN rows = uncaptured). The
  prober renders the per-decision **histogram** + mean/std/P10–P90/entropy/bimodality
  (`engine.build_value_dist` → `ValueDistView`, model-free; in the Summary panel + the `analyze` CLI). See
  `src/main/prober/CLAUDE.md`.
- **Honesty gate.** Ledger **K1 already killed the distributional critic as a WIN-RATE lever** (sub-Gaussian
  residuals — no tail). This is justified on INTERPRETABILITY only; its strongest use is upgrading the
  prober calibration/`falsify-scan` luck-vs-mistake split (predicted spread vs realized return = a
  within-model PIT). "Learns ≠ helps" — validate calibration (PIT ≈ uniform), not win-rate.
- **Tests.** Unit: `value_dist_loss_test.py` (HL-Gauss math + diagnostics), `agents/model/
  value_dist_head_test.py` (module build, off byte-identical, grad gating, the v29 version gate),
  `main/prober/engine_test.py` (`build_value_dist`). End-to-end `--debug --debug-eval --use-bridge=node
  --value-dist-mode read_only` smoke captures a trace whose npz carries `value_dist`.

