# Training — the critic mode and the value losses

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## THE VALUE LOSS has a MODE — `--critic {shaped,winprob}` (`gen3_winprob_critic_mode_v1`)

**Default `winprob` (the bare-argv flip, deletion pass D2); `shaped` is the historical critic, selectable on
`--env-core python` only.** PopArt, the distributional value head, `value_from_dist`, the value-tail weight and
the aux-BCE coefficient were DELETED with the shaped critic's levers (deletion pass L1, config v131 —
`designs/deleted_flags.md`); the sections below that describe them are gone. Design of record:
`designs/ai_v12/design_winprob_only_critic.md`. The model-side half (the route, the version gate)
is `src/agents/model/CLAUDE.md` → *The CRITIC MODE*; this is what `train()` does about it.

| | `shaped` | `winprob` |
|---|---|---|
| the value TERM | `vf_coef · mean((returns − values)²)` (clipped when `--clip-range-vf` is set) | `vf_coef · _win_prob_loss(...)` — the head's **BCE against the terminal outcome** |
| noise-scale group | `value` | **`value`** |
| the scalar `value_loss` | the loss | a DIAGNOSTIC only (its term is dropped, `_vf_term = 0.0`), and computed UNCLIPPED |
| the win-prob BCE | an auxiliary at a fixed weight 1.0, tagged `aux` | the value loss itself (weight `vf_coef`) |
| gate | the win head exists | the win head exists — no coefficient can switch the head's own loss off |

**The `"value"` tag is the point of `gen3_value_diagnostics_v1`'s sibling finding, applied one
critic over.** §1.4 of the design records that under the (since deleted) distributional-critic route the
REAL critic loss was folded as `_ntg.add("aux", …)` while `_vf_term` was 0.0 — so
`train/noise_scale_value` spent that entire era describing a term with weight zero, and the grad-balance
probe had to compensate separately. The promoted BCE joins `value`, and `grad/value_share`'s term follows
the critic (`win_prob_term if critic_winprob …`) for the same reason.

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
TERMINAL **win indicator** (`--terminal-indicator` + `--victory-value 1.0` + `--draw-penalty 0`,
all three REQUIRED and each named by its own
`combination_checks` refusal), so the undiscounted return is exactly `1{win}` and, at `--gamma 1.0`,
`V(s) = P(win | s)` with no approximation term. The cost is stated rather than buried: **a critic
bounded in [0,1] cannot represent "a timeout is worse than a loss."** `--draw-penalty`'s
`−35 < −30` ordering is unrepresentable there, so the anti-stall pressure is the obs deadline clock
(the reward has no anti-stall term since the shaped reward path was deleted, 2026-09-26). **Stall rate and mean episode length
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
rather than a retyped `0.9999` (the `PBRS_GAMMA == model.gamma` assert went with the hand
potentials when the shaped reward path was deleted, 2026-09-26). It is **INERT ON A RESUME**
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
MSE on a shaped return (O(100) on a ±30 scale); under `winprob` the
win-prob head's **BCE against a Bernoulli outcome**, which is `ln 2 ≈ 0.693` per sample at
initialisation and falls. The 0.5 default was tuned against the first and carries no information
about the second (and the PopArt normalisation that once made the two comparable is deleted).

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

**Compiled (`--compile-trainer`, torch 2.8): DECLARED** (`gen3_r1_declared_levers_v1`). The lever is
part of the compile region R1's signature from the FLAG; a rollout whose labelled rows hold one
opponent class (any run before the pool seeds) feeds R1 neutral weights (ones) — bit-identical to
the unweighted BCE. Before 2026-10-01 such an arm died at its first post-seed update.
`designs/training/compile_flags.md` "R1's DECLARED LEVERS".

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

## `--win-prob-lambda` — DELETED (deletion pass L2)

`gen3_winprob_lambda_v1` (v116, the critic ladder's arm 8; flags `--win-prob-lambda`, `--win-prob-lambda-truncated`) re-aimed the BCE at a λ-return over the collector's recorded values, `G[t] = (1−λ)·V(s[t+1]) + λ·G[t+1]` anchored at the outcome, to move within-episode information backward along a less noisy channel than one terminal bit (only 10.2 % / 14.4 % of that bit's variance lies between opponent cells, `winprob_head_refit_2026-09-09`). It read NOT DETECTED and was deleted with the `win_prob/lambda_*` tags and the sidecar's pre-λ outcome stash; the BCE target is now ALWAYS the terminal outcome (`WinProbLabelCallback` only back-fills it). Old value-sidecar files can still carry a λ-return `target` (`value_sidecar.md`). Recoverable at pin <= 475bd817.

## `--win-prob-dense-aux` — DELETED (deletion pass L2)

`gen3_dense_aux_v1` (v117, the critic ladder's arm 9) added a 25-output head on `value_pooled` predicting end-of-battle facts (survival and final HP of the twelve slots, turns-left) as KataGo-style auxiliary targets, with an un-detached input so its gradient reached the shared trunk. It moved nothing at the registered bar (ledger *THE ARMS AT 400 GAMES*) and was deleted with its head, callback, obs keys (`aux_target` / `aux_mask` / `aux_turn`), `win_prob/aux_*` / `grad/dense_aux_share` tags and the structural ModelVersion field `dense_aux`; a checkpoint recorded with it ON is refused on every load (`model_version/retired_levers.py`). Recoverable at pin <= 475bd817.

## `--win-prob-rollout-target` — DELETED (deletion pass L2)

`gen3_winprob_rollout_target_v1` (v118, the critic ladder's arm 10, flags `--win-prob-rollout-{target,r,mode,weight}`) replaced a sampled state's terminal bit with `wins / R` from R continuations replayed out of the `cf_records` ring: NEW bits per state rather than one outcome bit copied to ~30 states. It cost ~26x a production rollout's simulation budget at 1/32 and R = 8, was labelled synchronously by short-lived child processes, and read nothing that justified keeping it (ledger *THE ARMS AT 400 GAMES* and the arm-10 verdicts). The modules, the `win_prob/rollout_*` TB tags, the supply lever `win_prob_rollout` and the value sidecar's measured-fraction `target` rows are gone; `record_key` / `index_records` moved to `agents/training/cf_records.py` for the fork arm. Recoverable at pin <= 475bd817.

## THE FORK ARM (`--fork-fraction`) lives in its own doc

`--fork-fraction` (`gen3_fork_v1`, config v120) is the next knob on this loss and the only one that
adds **STATES** rather than re-pricing, re-weighting or re-aiming the ones collection happened to
visit: contested decisions are FORKED, the branches are played to a terminal by the current policy,
and their transitions enter the same PPO buffer with their own GAE/λ-returns and their own outcomes
as `win_target`. **Plain BCE, no ranking term** — the pairwise ranking loss is CLOSED as a lever at
this depth (−0.0107 [−0.0249, +0.0028], NOT DETECTED). It also touches the POLICY term (the fork
step is masked out of it) and the BUFFER (injected rows), which is why it does not live here:
[`designs/training/forks.md`](forks.md).

🚨 It REFUSES `--win-prob-strata-weight` — that flag prices rows by `win_margin`, which the env's
reward manager computes and an injected row cannot supply.

## `--win-prob-rollout-weight` — DELETED (deletion pass L2)

The R-rollout ANCHOR loss weight (`gen3_winprob_rollout_weight_v1`, v119) multiplied the per-row BCE of the rows the R-rollout target had anchored (the `win_row_w` obs key) so a treatment on ~0.12 % of the loss mass could move the head. It went with the target it weighted; the arithmetic (anchored share `f·k / (1 + f·(k−1))`) is in the ledger. Recoverable at pin <= 475bd817; `win_row_w` no longer exists as an obs key, an env-core label or a compile-declared lever.

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
to the run's critic (`critic_mode`) — reading `value_net` directly would train a critic the run does not use.

**Units.** `predict_values` returns REAL-unit values and the buffer's rewards are real-unit, so the
residual is real-unit — the same space the value loss trains in (the PopArt σ division this term once
carried is deleted with PopArt).

**Metrics (`td_aux/` prefix).** `resid_rms` is the headline — the quantity being minimised, the live
counterpart of the offline ΔV-dispersion instrument, and it should FALL. `resid_mean` (SIGNED) is
the no-harm watch: rung 1's decomposition says this is dispersion suppression, so a bias drifting
away from ~0 means the residual-gradient term is shifting the LEVEL rather than tightening it — read
it beside `train/explained_variance`. Also `loss`, `n_pairs` and `pair_drop_frac` (share of candidate pairs lost to episode boundaries). The
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
flattened `episode_starts`), the residual math on a hand-built case, both
ends carrying gradient, and on a REAL `train()`: coef-0 byte-identity (asserted twice — identical
parameters AND the sampler monkeypatched to raise, so a future sampler change cannot perturb an off
run), coef>0 moving the update and logging every metric, gradient landing on `value_net`, and the
un-flattened-buffer refusal.
