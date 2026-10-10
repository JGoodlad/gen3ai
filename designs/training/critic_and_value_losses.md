# Training — the critic mode and the value losses

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## THE VALUE LOSS has a MODE — the win-prob critic (`gen3_winprob_critic_mode_v1`; the `--critic` flag is deleted)

**`winprob` is the ONLY trainable critic and the default (the bare-argv flip, deletion pass D2); `shaped` is the historical critic — a typed `--critic shaped` is refused at parse time (U3), an old shaped checkpoint still LOADS (opponent, meters, prober) and a resume or fork of one is refused `FATAL_CONFIG` (D4: run it pinned).** PopArt, the distributional value head, `value_from_dist`, the value-tail weight and
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
TERMINAL **win indicator** (indicator, victory 1.0, draw 0.0 — constants of the namespace; the flags `--terminal-indicator`, `--victory-value`, `--draw-penalty` are DELETED, P11b batch (c)), so the undiscounted return is exactly `1{win}` and, at gamma 1.0 (a constant too),
`V(s) = P(win | s)` with no approximation term. The cost is stated rather than buried: **a critic
bounded in [0,1] cannot represent "a timeout is worse than a loss."** The shaped
`−35 < −30` draw-penalty ordering is unrepresentable there, so the anti-stall pressure is the obs deadline clock
(the reward has no anti-stall term since the shaped reward path was deleted, 2026-09-26). **Stall rate and mean episode length
are PRIMARY, kill-condition-bearing endpoints on a `winprob` arm.**

**THE DRAW BRANCH IS EXPLICIT, and `signal/draw_rate` states its frequency** (design §3.2 / gap
B9). `battle.won` is a TRI-STATE — True / False / **None**, the last being a draw or the 250-turn
timeout — and the deleted Python `MaskableAgentWrapper.step` used to reach `0.0` for the third case through a boolean
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
(`y = 0`, the same as a loss) while the reward pays the draw penalty (−35, i.e. WORSE than the −30
loss). Under the indicator terminal they agree: a timeout pays 0.0, exactly like a loss.

**`gamma` is NOT a flag** (`--gamma` existed from design gap B6, when it was hardcoded at `model_build.py`'s
`InstrumentedMaskablePPO(...)` call; it was DELETED in P11b batch (c)). The discount is `WINPROB_GAMMA` = 1.0, a constant of every trainer namespace. SB3 restores a checkpoint's own γ on a resume, like `--lr`, the resume path SAYS so, and it re-points
`reward_config.gamma` at the value actually in force so the two cannot silently disagree.

### 🚨 THE 250-TURN CAP IS A TERMINAL, NOT A TRUNCATION — and it was the other way round (B6)

**THIS ENV NEVER TRUNCATES IN THE SB3 SENSE, AND THAT IS THE WHOLE FINDING.**
`PokeEnv.calc_term_trunc` (`poke_env/environment/env.py:951`) sets EITHER flag only when
`battle.finished`; it then splits a finished battle by *how* it finished — exactly one side wiped ⇒
`terminated`, anything else ⇒ `truncated`. "Anything else" is the **250-turn cap forfeit** and a
genuine **tie**. Both are OUTCOMES. Nothing here is a time limit that interrupted an episode
mid-flight, which is the one thing `TimeLimit.truncated` is supposed to mean.

MEASURED 2026-09-06 on a real bridge battle with `StallConfig.threshold` lowered to 6:
the Python env's `action_to_order` (deleted in U3; the Rust env core enforces the same cap) returned a `ForfeitBattleOrder` at `turn >= threshold` (`== MAX_TURNS` in
production), Showdown answers `|win|<opponent>`, and the boundary reported `finished=True`,
`won=False`, **six mons alive a side**, `terminated=False, truncated=True`.

That flag then becomes `info["TimeLimit.truncated"]` in `DummyVecEnv`/`SubprocVecEnv`, and
`MaskablePPO.collect_rollouts` (`sb3_contrib/ppo_mask/ppo_mask.py:251-260`, mirrored by our
deleted async collector) does `rewards[idx] += gamma * V(s_last)`. Under
the win-prob critic (the only critic) the terminal reward is the win indicator — **0** for a cap loss — and γ is
**1**, so the last step's target becomes `0 + 1.0·V(s_last) = V(s_last)`: **a TD error of
identically zero.** The timeout leaves the loss entirely, and a policy that stalls to the cap is
taught nothing about it — while G7's stall rate is a KILL CONDITION on that arm, i.e. the gate
would have been reading a signal the critic never received. Verified by revert: the row reads
exactly `V`.

**The fix, as it stands now:** the Rust collector's `store.end(...)` records the stall forfeit as a plain LOSS (`forfeit=True`, outcome 0.0, the episode terminal), so the cap loss is trained, not bootstrapped away. (The Python fix — `wrappers.resolve_episode_end`, re-labelling a `trunc and not term` end TERMINAL under `winprob` at the end of `MaskableAgentWrapper.step`, pinned by `winprob_truncation_test.py` through a real SB3 `collect_rollouts` — was deleted with the Python env core in U3.)

**`shaped` is UNCHANGED and byte-identical, and what it does is worth stating rather than leaving
implicit:** a cap forfeit and a tie both bootstrap `0.9999·V(s_last)` on top of a terminal reward
that already paid the draw penalty. Two wrongs that partly cancel — the shaped terminal
double-counts the ending while the bootstrap removes it — and re-deriving that composition is a
`shaped`-era question this change does not reopen.

**Two adjacent paths are CLEAN, checked rather than assumed.** A launcher SIGTERM
(the deferred abort, `main/train/deferred_abort.py`, run at a safe point) saves a checkpoint and
`os._exit`s, so the partially-collected rollout buffer is discarded and never reaches an update — SB3 does not persist it. The eval
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

## `--win-prob-strata-weight` — DELETED (deletion pass P11c)

`gen3_winprob_strata_weight_v1` (v115, the critic ladder's arm 7) re-weighted the win-prob BCE by
inverse opponent-class frequency (`s` the exponent, weights capped at 8.0) to stop the bot-heavy mix
dominating the head's objective. It read NOT DETECTED / NOT CONFIRMED (its replicate reversed sign;
`designs/research_state/winprob_critic_ladder_2026-09-08.md`) and no recipe, backlog or end-state row
named it, so it was deleted with `_win_prob_strata_weights`, the `win_prob/strata_*` tags, its R1
per-update `var` tensors (`TrainSetup._r1_levers`) and the `ModelVersion` field. A config recording a
non-zero value is refused on every load (`model_version/retired_levers.py`). The BCE is plain and
unweighted across opponent classes. Detail and history: `designs/deleted_flags.md`.

## `--win-prob-lambda` — DELETED (deletion pass L2)

`gen3_winprob_lambda_v1` (v116, the critic ladder's arm 8; flags `--win-prob-lambda`, `--win-prob-lambda-truncated`) re-aimed the BCE at a λ-return over the collector's recorded values, `G[t] = (1−λ)·V(s[t+1]) + λ·G[t+1]` anchored at the outcome, to move within-episode information backward along a less noisy channel than one terminal bit (only 10.2 % / 14.4 % of that bit's variance lies between opponent cells, `winprob_head_refit_2026-09-09`). It read NOT DETECTED and was deleted with the `win_prob/lambda_*` tags and the sidecar's pre-λ outcome stash; the BCE target is now ALWAYS the terminal outcome (`WinProbLabelCallback` only back-fills it). Old value-sidecar files can still carry a λ-return `target` (`value_sidecar.md`). Recoverable at pin <= 475bd817.

## `--win-prob-dense-aux` — DELETED (deletion pass L2)

`gen3_dense_aux_v1` (v117, the critic ladder's arm 9) added a 25-output head on `value_pooled` predicting end-of-battle facts (survival and final HP of the twelve slots, turns-left) as KataGo-style auxiliary targets, with an un-detached input so its gradient reached the shared trunk. It moved nothing at the registered bar (ledger *THE ARMS AT 400 GAMES*) and was deleted with its head, callback, obs keys (`aux_target` / `aux_mask` / `aux_turn`), `win_prob/aux_*` / `grad/dense_aux_share` tags and the structural ModelVersion field `dense_aux`; a checkpoint recorded with it ON is refused on every load (`model_version/retired_levers.py`). Recoverable at pin <= 475bd817.

## `--win-prob-rollout-target` — DELETED (deletion pass L2)

`gen3_winprob_rollout_target_v1` (v118, the critic ladder's arm 10, flags `--win-prob-rollout-{target,r,mode,weight}`) replaced a sampled state's terminal bit with `wins / R` from R continuations replayed out of the `cf_records` ring: NEW bits per state rather than one outcome bit copied to ~30 states. It cost ~26x a production rollout's simulation budget at 1/32 and R = 8, was labelled synchronously by short-lived child processes, and read nothing that justified keeping it (ledger *THE ARMS AT 400 GAMES* and the arm-10 verdicts). The modules, the `win_prob/rollout_*` TB tags, the supply lever `win_prob_rollout` and the value sidecar's measured-fraction `target` rows are gone; `record_key` / `index_records` moved to `agents/training/cf_records.py` for the fork arm (deleted with the Python fork arm, L5). Recoverable at pin <= 475bd817.

## THE FORK ARM (`--fork-fraction`) lives in its own doc

`--fork-fraction` (`gen3_fork_v1`, config v120) is the next knob on this loss and the only one that
adds **STATES** rather than re-pricing, re-weighting or re-aiming the ones collection happened to
visit: contested decisions are FORKED, the branches are played to a terminal by the current policy,
and their transitions enter the same PPO buffer with their own GAE/λ-returns and their own outcomes
as `win_target`. **Plain BCE, no ranking term** — the pairwise ranking loss is CLOSED as a lever at
this depth (−0.0107 [−0.0249, +0.0028], NOT DETECTED). It also touches the POLICY term (the fork
step is masked out of it) and the BUFFER (injected rows), which is why it does not live here:
[`designs/training/forks.md`](forks.md).

## `--win-prob-rollout-weight` — DELETED (deletion pass L2)

The R-rollout ANCHOR loss weight (`gen3_winprob_rollout_weight_v1`, v119) multiplied the per-row BCE of the rows the R-rollout target had anchored (the `win_row_w` obs key) so a treatment on ~0.12 % of the loss mass could move the head. It went with the target it weighted; the arithmetic (anchored share `f·k / (1 + f·(k−1))`) is in the ledger. Recoverable at pin <= 475bd817; `win_row_w` no longer exists as an obs key, an env-core label or a compile-declared lever.

## TD-consistency auxiliary (`--td-aux-coef`) — DELETED (deletion pass P11c)

A Bellman-residual auxiliary over contiguous rollout pairs (`gen3_td_consistency_aux_v1`, v92;
`td_aux.py`, `_td_aux_term`, the `td_aux/*` tags and `grad/td_aux_share`) that penalised the
critic's `ΔV` between adjacent states. The ladder's `tdaux` arm read NOT DETECTED
(`designs/research_state/levers/td_consistency_aux.md`), and nothing live named it, so it was
deleted. A resume or fork of a run that recorded a non-zero value is refused by
`model_version/retired_levers.py`. Detail: `designs/deleted_flags.md`.

## From the training leaf (moved 2026-10-10)

> These sections headed `src/agents/training/CLAUDE.md` until its 2026-10-10 cleanup; moved here as they
> stood (minus statements verified FALSE). Where an earlier section of this doc says the same in more
> detail, both are current; fix both in the same pass.

### THE VALUE LOSS has a MODE — the win-prob critic (`gen3_winprob_critic_mode_v1`)

**`winprob` is the ONLY trainable critic** (a CONSTANT of every trainer namespace — `src/main/train/parser/objective.py`, `parser.set_defaults`; the `--critic` flag is DELETED, P11b batch (b), and a typed one is refused with its reason from `designs/deleted_flags.md`; `CRITIC_TRAINABLE_MODES` is deleted). `shaped` is the historical critic (`CRITIC_SHAPED`, still in `CRITIC_MODES` and the meaning of an ABSENT record, so an old shaped checkpoint still LOADS as an opponent, in meters and in the prober); a resume or fork of one is refused `FATAL_CONFIG` (D4) — run it pinned to its own commit. The `shaped` column below is the historical contrast. Design of record:
`designs/ai_v12/design_winprob_only_critic.md`; the model-side half is `src/agents/model/CLAUDE.md`
→ *The CRITIC MODE*.

| | `shaped` | `winprob` |
|---|---|---|
| the value TERM | `vf_coef · mean((returns − values)²)` (clipped under `--clip-range-vf`) | `vf_coef · _win_prob_loss(...)` — the head's **BCE against the terminal outcome** |
| the scalar `value_loss` | the loss | a DIAGNOSTIC only (its term is dropped), computed UNCLIPPED |
| PopArt · `--value-dist-*` · `--value-from-dist` · `--value-tail-weight` · `--win-prob-coef` · every `--win-prob-pbrs-*` | **DELETED** (L1, config v131: `designs/deleted_flags.md`; a checkpoint that recorded one ON is refused, `model_version/retired_levers.py`) | **DELETED** |

🚨 **`winprob` has the win-indicator terminal (indicator, victory 1.0, draw 0.0) and gamma 1.0 BY CONSTRUCTION** (constants of the namespace, `parser/objective.py` `set_defaults`; a typed flag is refused with its reason; a recorded non-production reward is refused on a resume by `check_reward_config`), so the undiscounted return
is exactly `1{win}` and `V(s) = P(win|s)` with no approximation term.
🚨 **THE COST IS STATED, NOT BURIED: a critic bounded in [0,1] cannot represent "a timeout is worse
than a loss."** The anti-stall pressure is the obs deadline clock (the reward has no anti-stall term), and
**stall rate and mean episode length are PRIMARY, kill-condition-bearing endpoints on a `winprob`
arm.** 🚨 **A `winprob` run's `train/*` value tags are not comparable with a `shaped` run's.**

🚨 **`critic_resolution` IS the meter; `critic_reliability` is not.** A base-rate forecaster scores
a perfect 0 reliability and a useless 0 resolution — the committed baseline measured this head at
reliability ~0.002 against a resolution of 0.062 out of an available 0.182, so a promotion that
improves ECE and leaves `critic_resolution` flat has moved the meter that was never the disease.
The `win_prob/critic_*` family comes from `scaffolding.reliability_table`, **imported, never
re-implemented**, and an unmeasurable rollout publishes **`{}`, never zeros**.

🚨 **A DRAW IS SCORED AS A NOT-WIN BY DECISION** (`y = 0`), never dropped and never 0.5 — that is
what makes "P(win)" literally P(win). ⚠️ `signal/draw_rate` counts **ties and not timeouts**, so the
series that watches the 250-turn cap is `signal/stall_rate` / mean episode length.
🚨 **THE 250-TURN CAP IS A TERMINAL, NOT A TRUNCATION** — this env never truncates in the SB3 sense;
a cap forfeit used to arrive as `truncated`, so SB3 bootstrapped `V(s_last)` onto a 0 reward at
γ=1 and the timeout left the loss entirely with **a TD error of identically zero**. Fixed in
the Python env wrapper's `resolve_episode_end` (deleted with that core in U3; the Rust collector's complete-game rule serves it now), under `winprob` only. **`gamma` is a constant of the namespace (1.0; no flag), and SB3 restores a checkpoint's own gamma on a
resume like `--lr`.**

#### `--fork-fraction` — THE FORK ARM, contested-state EXPLORING STARTS (`gen3_fork_v1`, v120)

**Default `0.0` = OFF and BIT-identical** — no fork object built, no obs key declared, no row injected.
**It runs on the win-prob critic (the only critic)**; the arm's replay
ring, `--cf-records`, was deleted in deletion pass L4 and the Python arm's code in L5. Detail:
[`designs/training/forks.md`](forks.md).

🚨 **ONE implementation: the Rust port** (the Python arm — callback, replay-ring child process,
buffer subclass, decision-time handle capture — is DELETED, deletion pass L5; `fork_arm.py` keeps
its selector and meters and `fork_buffer.py` its FILL table, which the port imports). The arm is the COLLECTOR's fork phase (`rust_rollout/fork.py`,
`gen3_fork_rust_v1`, forks.md §14) — DECLARED and OFF, deferred by the owner's one-ply scope
(2026-10-01). It replays the core's finished input log on Lane I playout handles, keys every branch
draw on the PARENT's keyed-draw key (a parent-action branch IS the parent — gate
`rust_rollout/fork_crn_integration_test.py`) and puts each branch game into the complete-game FIFO
after its parent, so **branch rows COMPETE for the update's D** rather than doubling the buffer, and
a branch plays the parent's REAL policy opponent where its slot still serves it
(`fork/opp_substituted` is the rest). Both are DEPARTURES from the arm that read NOT DETECTED on
2026-09-16 — that read does not transfer unchanged. Needs no extra flag.

🚨 **WHY — the head ranks siblings at CHANCE.** `paired_refit_discrimination_2026-09-14` measured
the promoted win-prob critic's pairwise accuracy on successors ONE MOVE APART at **0.5169
[0.4800, 0.5524]**, while a FROZEN trunk with only the head's four tensors refit on counterfactual
successors reaches **0.6032** (+0.0863, DETECTED) and a pairwise RANKING term buys **nothing**
(−0.0107, NOT DETECTED). Pairwise accuracy is a RANK statistic, so the ordering was in
`value_pooled` all along and the on-policy stream never asked for it: **the DATA is the lever, not
the loss form.** A rollout visits exactly ONE successor per decision; this manufactures the
siblings. At a contested decision the battle is forked, three branches (the policy's top-2 + ONE
uniformly random legal action) are played to a terminal by the CURRENT policy, and their
transitions enter the SAME PPO buffer. **Plain BCE, NO ranking term — closed as a lever.**

🚨 **THE MASK RULE IS UNIFORM: the FORK STEP is out of the policy term for EVERY branch**, the
top-2 included, and the term is RENORMALISED over the kept rows (not just zeroed — a masked
`.mean()` would silently lower the effective policy LR by the fork rate). Masking only the random
branch would re-weight the policy gradient by the branch MIX. The fork step stays fully in the
VALUE terms. Carrier: the `fork_pg_m` obs key.

🚨 **THE PREFIX IS COUNTED ONCE** — a branch's rows begin AT the fork step. The fork STATE appears
once per branch with a DIFFERENT action; that is the exploring start, not a duplicate.

🚨 **`--fork-crn dice_and_draws` (default) pairs the DICE *and* the policy draws.** `cf_q_labels`
paired only the dice — a concrete, testable account of its null — and
`rust_rollout/fork_crn_integration_test.py` proves a parent-action branch reproduces the parent byte for byte.

⚠️ **The ecology approximation** (the Python arm's largest caveat; the Rust port shrinks it to the
bot share): a branch is played against the parent's REAL policy opponent where its slot still serves
it, else a SELF-LIKE one, and those rows are labelled `opp_class = POOL`; `fork/opp_substituted`,
`fork/branch_share` and `fork/bot_share` price it.

🚨 **A fork dropped at the row budget has ALREADY BEEN PLAYED**, so the ask is bounded by the
previous rollout's MEASURED `fork/rows_per_fork`. Read **`fork/rate`**, **`fork/branch_share`**,
**`fork/tie_rate`**, **`fork/random_wins`**, **`fork/pairwise_acc`** (IN-SAMPLE; the endpoint is a
held-out read) and **`fork/sim_steps_share`** (the cost).

**Full detail — the currency argument, the cap-terminal measurement, the `--vf-coef` BCE
announcement and every value-side flag below — is in
[`designs/training/critic_and_value_losses.md`](critic_and_value_losses.md).**

#### The other value-side flags, in one place

| flag | default | what it does, and the one thing to know |
|---|---|---|
| `--vf-coef` | `0.5` | multiplies a BCE under `winprob`, an MSE on a shaped return under `shaped` — **the 0.5 default carries no information about the first**. The startup announcement prints the raw BCE and the value/policy shared-trunk gradient RATIO (`10 ** grad/value_policy_logratio`); it never divides by `|policy loss|`, which is ≈0 by construction on epoch 1. Fixed for a run's lifetime |
