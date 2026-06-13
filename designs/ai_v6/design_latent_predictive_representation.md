# AI v6 — Latent Predictive Representation (Meaning B)

An **amortized 1-ply anticipation** objective that shapes the shared representation, culminating
in the **per-action outcome-token injection** idea — *without* ever running search or the
simulator at decision time.

> **Status:** design (forward-looking). **Class:** L3 amortized-anticipation lever
> (`designs/research_state/`). **Build posture:** cheapest-and-safest-first ladder; the
> outcome-token injection is the *payoff* stage (Stage 4), deliberately **not** Stage 1.
> **Carried verdict (judge panel + adversarial red-team):** CONDITIONAL GO on the cheap
> de-risking ladder; hard kill-gates and anti-leak guardrails before any injection;
> STOP-flag on the value-path leak.

---

## 1. Motivation & framing

### Meaning B, not Meaning A

We want the single-forward-pass policy/value to become **anticipatory** — to act as if it had
"looked one ply ahead" — *without a search or a simulator call at decision time*. This is
**Meaning B**: a **latent predictive objective that shapes the shared representation** so "what
happens next" is baked into the weights.

This is explicitly **not** Meaning A (MuZero/Dreamer: learn a dynamics model to *plan inside
of*). The project ruled that out —
`designs/ai_v5/research_selective_search_compute_constrained_rl.md`: *"Do NOT learn the chance
model — Stochastic MuZero solves a problem you don't have (no simulator). You HAVE Showdown."*
And the owner's hard constraint (`designs/research_state/README.md`): **no lever may put
search/MCTS on the model at inference OR in the training loop; search is a teacher / offline
diagnostic only.**

This design therefore **supersedes the ai_v6 MCTS plan** (`designs/ai_v6/impl_step5_mcts.md`)
as the route to "anticipation" (see §9). It is feedforward end to end.

### Why this is L3 and passes the amortizability gate

The research_state L3 test: *"was the next-step outcome (the opponent's typical reply / the
consequence of my action) predictable from history/priors (anticipatable), or idiosyncratic /
hidden (irreducible)?"*

Our objective **is** that test, made trainable: an auxiliary loss that only succeeds to the
extent the next decision state is predictable from the current trunk + the candidate action. At
inference there is no tree, no rollout, no sim call — one forward pass produces the anticipatory
policy/value. The amortization is the whole mechanism: the 1-ply lookahead that would otherwise
require forking the sim per action is *compiled into the weights*.

### The elegant insight: the sim is a SUPERVISION oracle, never a planner

We have the exact, fast simulator. We **never** call it in the forward pass. But during normal
on-policy rollout the env *already* steps Showdown for the action taken and **already folds the
realized `TurnDelta`** of that transition (`gen3_env.py:87`, `self._pending_delta`, computed
once and reused by the reward at `:137`). That is free, on-policy ground truth for "what the
taken action did." We use it as a supervised target for a per-action predictor. Inference cost:
zero sim calls. Training cost: read an already-computed struct.

---

## 2. The core mechanism

### Real arch constants (single source of truth: `features_extractor.py`)

```
ROLE_TOKEN_SIZE = D_MODEL = 128                       (:34, :44)
TRANSFORMER_N_LAYERS = 2, N_HEADS = 4, FFN_DIM = 256  (:45-47)
PROJECTION_DIM = 512                                  (:35)
N_HISTORY_TURNS = 10                                  (:40)
TEAM_SIZE = 6 → TeamTransformer sequence = 6 our + 6 their + 10 history + 1 global = 23 tokens
TOKEN_TYPE_{OUR_TEAM,THEIR_TEAM,HISTORY,GLOBAL} = 0..3, NUM_TOKEN_TYPES = 4   (:50-54)
ACTION_SPACE_SIZE = 11   (0–5 switch, 6–9 move[request-slot], 10 struggle)   (agents/action/constants.py:3)
```

The dual head is a **readout split over one shared body**: `forward_internal` runs
`ObsUnpack → PokemonEncoder → TeamTransformer → CLSPool`, producing
`our_team_pooled / their_team_pooled / our_active_refined` (policy) and `value_pooled` (critic,
`:960`). `ProjectionAssembler.forward` concatenates `pi_parts` / `vf_parts` and returns
`(pi_combined, vf_combined)`; projection input dims are **auto-discovered** by a dummy zero-obs
forward in `__init__` (`:910–914`), so appended readouts need no manual dim edit.

### The predictor module: `OutcomePredictor`

A new phase `nn.Module` (own file `src/agents/model/outcome_predictor.py`, mirroring the
`HiddenOppBeliefPool` precedent). It holds:

- a **board-summary** input `z = value_pooled` `[B, 128]` (the whole-board "who's winning"
  critic readout — the single most semantically loaded 128-vector in the trunk);
- a learned **action-embedding bank** `nn.Embedding(ACTION_SPACE_SIZE=11, A_EMB=32)`;
- a **per-action generator** `g`: an MLP `[128 + 32 → 256 → 128]` producing one outcome token
  per action, `o = g([z ; action_emb_a]) → [B, 11, 128]`;
- a small **aux-decode head** `[B, 11, 7]` mapping each token to a grounded outcome prediction
  (below).

The action is a **model input** to `g`, so `g` is a function over all 11 actions but is only
ever *supervised* on the one taken — the Q-function-from-single-transitions resolution (§3.1).

### What the tokens predict (the 7-field grounded target, off the realized `TurnDelta`)

Read directly from the folded `TurnDelta` (`turn_delta.py`: `we_fainted`, `opp_fainted`,
`we_moved_first`, `our_effectiveness`, `our_hp_delta`, `opp_hp_delta`) — no second fold:

| # | field | type | loss |
|---|---|---|---|
| 0 | `faint_ours` (`we_fainted`) | binary | BCE |
| 1 | `faint_opp` (`opp_fainted`) | binary | BCE |
| 2 | `moved_first` (`we_moved_first`, `None`→mask) | binary | BCE (masked) |
| 3 | `our_hp_delta_frac` (Σ our `our_hp_delta`, our-active) | reg [−1,1] | Huber |
| 4 | `opp_hp_delta_frac` (Σ opp `opp_hp_delta`, opp-active) | reg [−1,1] | Huber |
| 5 | `eff_bucket` (`our_effectiveness` → {immune,nve,neutral,se}) | 4-way | CE (masked: switch/None) |
| 6 | `dmg_bucket` (quantized realized damage of our action) | 5-way | CE (masked: no hit) |

All discrete or bounded → **collapse-proof by construction** (no constant-output trivial optimum
exists for an external-fact target). This is the de-risked supervised-first rung of
`designs/ai_v5/design_offense_and_opponent_belief.md` §B3.

### The loss

Added at the single PPO loss-sum site (`instrumented_ppo.py:235`,
`loss = policy_loss + ent_coef·entropy_loss + vf_coef·value_loss`):

```
loss = policy_loss + ent_coef·entropy_loss + vf_coef·value_loss + aux_coef · L_outcome
```

`L_outcome` is evaluated **only at the taken action's token** (gather `o_{a_taken}` via
`rollout_data.actions`), decoded, and compared to the 7-field target. `aux_coef` is a
resume-immutable value-meaning hparam (§4 Stage 1), default `0.0`, start value `~0.1`.

### Where the tokens inject (Stage 4 — the payoff)

The 11-token bank is appended to the **`ProjectionAssembler` concat** first (lowest-risk read
path — the verified `value_active_readout` / `hidden_opp_belief` concat precedent at `:842–857`),
with a learned outcome-CLS query pooling the bank to `[B, 128]` appended to `pi_parts` (policy
side; the value side benefits only via the shared trunk — §3.3). Only if that read-path arm
beats the loss-only arm do we escalate to true cross-attention injection (CLSPool memory
extension at `:735–749`, or TeamTransformer sequence injection at `:674` with
`TOKEN_TYPE_OUTCOME = 4`, `NUM_TOKEN_TYPES 4→5`). Illegal actions are key-padding-masked via the
action mask; ≥1 action is always legal, so no all-masked row → no attention NaN (the same
invariant `HiddenOppBeliefPool` relies on, `:802`).

---

## 3. Counterfactual resolution, collapse, and leakage guardrails

This is where designs die. Every guardrail below is **mandatory**.

### 3.1 Counterfactual: action-as-input generalization (with the sharp caveat)

On-policy rollout observes only the taken action's outcome. We supervise `g` **only on
`a_taken`**; because `a` is an *input*, `g(z, a)` is defined for all 11 actions and generalizes
at inference — exactly like a Q-function trained from single-action transitions evaluates all
actions.

**The caveat the red-team is right about, and mitigations that are NOT optional:**

1. **Consumer mismatch.** A Q-net's consumer is `argmax` (robust to miscalibrated absolutes).
   Our Stage-4 consumer is *attention*, which can weight a confidently-wrong token. → Stage 4
   only ships if it beats the loss-only arm; Stages 1–3 have no attention consumer at all.
2. **Adversarial selection bias.** The behavior policy **under-switches** (documented,
   repeated). So switch tokens (slots 0–5) are the *least* supervised and most likely wrong —
   and attending over a wrong imagined switch-consequence could *entrench* under-switching.
   Both required at Stage 4:
   - **Per-action-class generators** `g_move` / `g_switch` / `g_struggle` (a switch's
     consequence — bring in a bench mon, take entry damage, no hit lands — is structurally
     different from a move's; one shared `g` under-fits the rarely-taken switch hardest).
   - **Inverse-propensity aux weighting**: up-weight `L_outcome` on rarely-taken actions so the
     under-explored switch tokens get more signal per sample.
3. **Coverage is policy-temperature-limited.** `--self-play-temp` spreads coverage only over
   actions the policy *considers*, not ones it irrationally avoids. → If Stage 3's behavioural
   gate shows switch behaviour not moving, the **offline counterfactual reroll (Stage 5b) is
   promoted from optional polish to the principled fix.**

### 3.2 Collapse: grounded discrete targets dodge it; SPR carries it

Stages 1–4 use grounded discrete/bounded targets → **collapse is structurally absent** (an
external label has high loss at the constant-output optimum). A self-predictive latent target
(BYOL/SPR, Stage 5a, opt-in) reintroduces the predict-your-own-embedding collapse and is
therefore quarantined behind: EMA target-encoder + stop-grad + asymmetric predictor MLP + a
**VICReg variance floor**, plus a mandatory `aux/latent_std` monitor whose collapse
(`std → 0` while `loss → 0`) is an explicit NO-GO. The discrete co-head stays as the **banked
fallback**. If a self-predictive variant is ever chosen, an **MSE** target against `value_pooled`
is forbidden under PopArt (value-scale drift makes it non-stationary) — **cosine/normalized
only**.

### 3.3 Leakage — the STOP-flag the red-team is correct about

**The value-path leak (Stage 4, severe).** `value_pooled` (`:960`) IS the critic's projection
input. If we inject outcome tokens into the value readout **and** train them to encode the
realized next-turn delta, the critic gains a near-label of `r_t` / the immediate transition for
the taken action — exactly the action it will most often pick next. That dishonestly inflates
`explained_variance`, shrinks `value_loss`, and **biases GAE advantages** (a "peeking" critic
understates advantage variance). **Hard requirements:**

- **Stage 4 injects into the POLICY pools only** (`our_cls` / `their_cls` / the `pi_parts`
  concat). The value head benefits *solely through the shared-trunk gradient*, never by reading
  the token directly. **OR**, if value-side injection is ever attempted, the token is
  **stop-gradient on the value-memory path** so the critic attends to it as a fixed feature but
  the aux-label gradient never flows through the critic's read.
- **Ablation gate:** token-in-value-memory must NOT improve `explained_variance` / `value_loss`
  beyond token-in-policy-only by more than noise. If it does, that's leakage → revert.
- **`eval/td_resid_tail` is DISQUALIFIED as a gate the moment a token touches the value path** —
  a peeking critic is unsurprised by construction. Behavioural gates must be *policy-action*
  metrics (§4) thereafter.

**The obs-target leak (all stages).** The 7-field target must be **unreachable by `ObsUnpack` /
the extractor forward** — the future must never reach the policy *input*. A unit test asserts
`ObsUnpack` does not read the target key. Routing decision (§5): the target rides a **new Dict
obs key**, NOT inside the flat `observation` vector (which `ObsUnpack` peels), precisely so a
slice can't accidentally expose it.

**Gradient-fight on the shared trunk.** `aux_coef·L_outcome` is a **third** gradient on
`SHARED_TRUNK_PHASES = ("embeddings", "pokemon_encoder", "team_transformer", "assembler")`
(`grad_balance.py:47`) — the same trunk the value loss already swamps (the reason PopArt exists).
**Required:** add `grad/aux_share` and `grad/policy_aux_cosine` to `grad_balance.py` and gate the
run: if `aux_share > ~0.4` or `policy_aux_cosine < 0` sustained, lower `aux_coef` or stop. The
HP-delta Huber terms are bounded fractions (different scale from PopArt-normalized returns), so
they do NOT go through PopArt; `aux_coef` does double duty (scale + weight) — watch it.

---

## 4. Incremental stages

Cheapest/safest first. Each is independently shippable with a named falsifiable metric.
**Win-rate signal: anchored ELO + prober `triage` / `falsify-scan` only —
`win_rate_vs_pool` is gate-pinned ~50% and FORBIDDEN as a signal.**

### Stage 0 — FREE offline kill-gates (zero training, zero arch)

The highest-leverage move in the plan: kill a dud before any retrain.

**What ships:** three offline probes on the **existing** best checkpoint + existing eval traces
(no new run):
1. **Anticipatability probe** (extends the prober representation-probe harness): linear-probe
   the frozen trunk's `value_pooled` / `our_active_refined` to predict the realized next-turn
   `TurnDelta` (faint-ours / faint-opp / we-moved-first / effectiveness), **split loss-causal
   vs ordinary** states.
2. **Headroom probe:** the trunk already predicts incoming self-KO at **AUC ≈ 0.79** (the v10
   incoming-belief finding). If next-turn-faint AUC is *already* ~0.79, the aux loss has
   near-zero headroom → **STOP**.
3. **Mis-targeting probe** (`falsify-scan` on the loss corpus): what fraction of loss-craters
   are *anticipatable-but-not-acted-on* (a reward/credit problem) vs *genuinely surprising*?
   Cross-check the recorded surprise-OHKO provenance (~36% just-switched, ~42% new/unrevealed
   move — a large irreducible slice that **bounds the addressable fraction**).

**Go/no-go:** GO only if next-turn faint/effectiveness AUC `> ~0.65` on a meaningful fraction of
loss-causal states **AND** materially below the ~0.79 incoming-belief ceiling (headroom exists)
**AND** a meaningful slice of craters is anticipatable-but-not-acted-on (not purely a credit
problem). If outcomes are largely idiosyncratic/hidden (AUC ~0.5) or the trunk already
anticipates as well as the incoming belief, **STOP the L3 lane and spend the cycle on the
reward/credit levers** (`--switch-bias-weight`, `--self-ko-hp-penalty`) that target the
demonstrated bottleneck.

**Falsifiable metric:** prober probe AUC for next-turn-faint-ours/opp on loss-causal decisions;
the anticipatable-but-unacted crater fraction from `falsify-scan`.

**Files:** extend `src/main/prober/engine.py` + `query.py` (a new probe target); no training code.

### Stage 1 — target plumbing + head, NO learning (`aux_coef = 0.0`)

Prove the plumbing is inert before turning on learning.

**What ships:** the new Dict obs key `outcome_target` (§5); the `OutcomePredictor` module built
**gated on `aux_coef > 0`** (so `aux_coef=0` reproduces the baseline state_dict **byte-for-byte**
— the `opp_belief_cls_k` playbook → **no `ARCH_SIGNATURE` bump**); `evaluate_actions` widened to
surface the taken-action decode; `aux_coef` as an `InstrumentedMaskablePPO` class attr (default
`0.0`, the `value_tail_weight:0.0` template at `:97`).

**Arch/training change:** `MODEL_CONFIG_VERSION 12→13`, `_migrate_config setdefault`, `aux_coef`
field + `check_outcome_aux` (resume-only, excluded from `check_compatible` — a frozen opponent
never runs the aux loss). **No `ARCH_SIGNATURE` bump** (gated construction).

**Go/no-go (`no_op_equivalence`):** `train/value_loss` + `train/policy_gradient_loss`
**byte-identical** to baseline at `aux_coef=0` (else the plumbing leaked gradient);
`obs_build_benchmark.py` + `trainer_turn_benchmark.py` show no regression (the mandatory obs
gate); the bridge fuzz test confirms `outcome_target == protocol truth` on 200 battles; the
obs-leak unit test passes.

**Falsifiable metric:** byte-identity of the two PPO losses at `aux_coef=0`.

### Stage 2 — aux loss ON, NO injection (the cheapest real test)

Turn on `aux_coef ~0.1` with the 7-field collapse-proof discrete target. Pure trunk-shaping; the
tokens are NOT yet attended by any head. Answers *"does a predictive objective help the
representation?"* for the minimum spend.

**Arch/training change:** none beyond Stage 1's module; `aux_coef > 0`. Add `grad/aux_share` +
`grad/policy_aux_cosine` to `grad_balance.py`.

**Go/no-go:** (a) `L_outcome` falls AND taken-action faint-prediction accuracy rises above the
Stage-0 probe baseline (the head can fit); (b) PPO health stable — `approx_kl`, `value_loss`,
`win_rate_vs_bots` within noise, `aux_share < 0.4`, `policy_aux_cosine ≥ 0`; (c) discrete
next-KO AUC `> 0.70`. If the head can't fit the taken-action outcome, the target schema is wrong
— fix before measuring behaviour.

**Falsifiable metric:** `aux/next_ko_auc > 0.70` within ~10M steps; `aux_share < 0.4`.

### Stage 3 — BEHAVIOURAL A/B of the shaped trunk (no new code)

The gate the project's history demands. The incoming-damage belief was wired, critic-read,
AUC-0.79-calibrated — and the policy **still under-switched** (the decorate-the-trunk trap). A
better probe AUC is **not** a policy win.

**What ships:** a full anchored-ELO paired run `aux_coef=0` vs `0.1`, with prober `triage` /
`falsify-scan`.

**Go/no-go:** a NAMED BEHAVIOURAL metric must move at flat-or-better ELO — **surprise-OHKO
read-rate down from the ~55% baseline**, OR `ignored_threat_death` triage share down, OR
`eval/td_resid_tail` CVaR@5% less-negative (valid here only because no token touches the value
path yet). If the trunk is better-shaped but behaviour is flat → that is the **explicit trigger
to escalate to injection (Stage 4)**, not to abandon. If ELO regresses or `td_resid_tail`
worsens → STOP.

**Falsifiable metric:** `eval/td_resid_tail` (CVaR@5%) AND one of {surprise-OHKO read-rate,
`ignored_threat_death` share}, anchored-ELO-controlled.

### Stage 4 — INJECT the per-action outcome tokens (the centerpiece)

Entered only when Stage 3 proves the trunk is better-shaped but the policy doesn't act on it —
precisely the condition under which *attending over imagined consequences* is the mechanism most
likely to convert anticipation into action.

**What ships, in escalating risk (each its own revert gate):**
- **4a — concat read (lowest risk):** an outcome-CLS query pools the 11-token bank → `[B,128]`,
  appended to `pi_parts` only (policy pools; value benefits via shared trunk, §3.3). The verified
  `hidden_opp_belief` concat precedent (`:842–857`).
- **4b — CLSPool cross-attention memory:** append the bank to `our_cls` / `their_cls` memory
  (`:735–738`); `value_cls` **only with stop-grad** or not at all (§3.3).
- **4c — TeamTransformer sequence injection:** `cat` at `:674` with `TOKEN_TYPE_OUTCOME = 4`,
  `NUM_TOKEN_TYPES 4→5`, sequence `23→34` tokens, key_padding_mask widened (bidirectional —
  team/history attend back onto the imagined outcomes).

Plus the §3.1 mitigations: **per-action-class generators** + **inverse-propensity aux weighting**.

**Arch/training change:** `ARCH_SIGNATURE` bump (`gen3_incoming_crit_split_v1 →
gen3_outcome_token_v1`); old checkpoints fail loudly (rapid-iteration project allows it).
Auto-discovery sizes the widened projections.

**Go/no-go:** the injection arm must **beat the Stage-3 loss-only arm** on the SAME behavioural
metric beyond anchored-ELO CI, **AND** non-trivial outcome-token attention mass (the CLS queries
actually attend to legal tokens, checked via attention weights), **AND** the value-leak ablation
(§3.3) is clean. **Revert if no lift** — ship the cheaper loss-only arch and delete the injection
structure (no dead arch weight). Must clear `obs_build_benchmark.py` + `trainer_turn_benchmark.py`
(the GPU is ~86% idle so 23→34 tokens is cheap in wall-clock, but the new Dict key's per-step
memcpy must not regress the CPU obs-build bottleneck).

**Falsifiable metric:** Δ(behavioural metric) of inject-arm minus loss-only-arm,
anchored-ELO-controlled; mean outcome-token attention mass from the policy CLS queries above a
fixed floor.

### Stage 5 — OPT-IN richer targets (each its own gate; never required for v1)

- **5a — SPR/BYOL latent target:** swap the grounded 7-field decode for a next-trunk-latent
  target with the **full collapse stack** (EMA target + stop-grad + asymmetric predictor +
  VICReg floor; cosine target only). Ships only if it beats the grounded arm on the Stage-4
  behavioural metric **with `aux/latent_std` above floor** (no collapse). The discrete head stays
  as the banked fallback.
- **5b — offline counterfactual reroll:** generate ground-truth next states for the **non-taken**
  actions offline via the bridge reconstruction record (`utils/bridge/reconstruction.py`,
  bridge-eval traces with a `*_reconstruction.json` sibling), distilled into the predictor in a
  held-out pass. **Promoted from optional to the principled fix** if Stage 3/4 shows switch-action
  behaviour not moving (the §3.1 selection-bias fix). Strictly offline, behind the
  one-sided/omniscient wall — never the live loop, never search.

**Go/no-go:** 5a — beats the grounded arm without collapse. 5b — non-taken-action decode
calibration on held-out reroll labels materially exceeds the action-as-input baseline AND the
behavioural metric improves; else the free generalization sufficed (the expected outcome).

---

## 5. Ground-truth target extraction (sim → TurnDelta → rollout buffer)

**Source.** `gen3_env.embed_battle` already folds the realized t→t+1 `TurnDelta` once (`:87`,
`self._pending_delta`, reused by `calc_reward` at `:137` — no double fold) via the event-sourced
battle layer (`turn_delta.build_from_events`). The 7 fields are read straight off it — a single
source of truth via a new `outcome_target_encoder.py` (mirrors `turn_delta_encoder.py`).

**The one-step-delay pairing.** The delta describes the transition *caused by the action taken at
the PREVIOUS decision*. So `embed_battle` for decision `t+1` writes
`outcome_target = encode(self._pending_delta from the t→t+1 fold)`, pairing it with `(s_t, a_t)`
in the buffer. SB3 stores per-step obs (not next-obs), so the target *must* ride the obs of the
step it labels — the one-step-delay solves it (the same pattern as the td-residual backfill).

**Routing — a NEW Dict obs key, not the flat vector.** The obs Dict is assembled in
`poke_env/environment/env.py` as `{"observation": <flat>, "action_mask": <mask>}` (space at
`:424–429`; dicts at `:591`/`:663`; `Gen3Env.observation_space` at `gen3_env.py:42–45`). We add a
third key `outcome_target: Box(shape=(7,), float32)` at those four sites, populated from
`embed_battle`'s cached delta. **Why a Dict key, not a slice of `observation`:** `ObsUnpack` peels
the flat `observation` vector — putting the target there risks the policy *reading the future*. A
separate key is provably unreachable by the extractor forward (the obs-leak unit test pins this).
It flows untouched through `MaskableDictRolloutBuffer` and the async collector's generic
`obs_keys` loop (`async_vec_env.py`), and SB3 stores it per-step like `action_mask`.

**Async correctness.** The async collector fills each env's buffer column contiguously; the new
key rides the same per-column path as `action_mask`. The fuzz test (§7) pins the `s_t ↔ target`
pairing off-by-one-correct under **both** sync and `--async-rollout`.

---

## 6. Files to create / modify

**Create:**
- `src/agents/model/outcome_predictor.py` — `OutcomePredictor` (action-embed bank + per-action
  `g`; Stage 4: per-class `g_move`/`g_switch`/`g_struggle`; + 7-field decode head; Stage 5a:
  EMA/predictor/VICReg).
- `src/agents/observation/outcome_target_encoder.py` — pure `TurnDelta → 7-field target` (single
  source of truth for the schema).
- `src/agents/model/outcome_predictor_test.py` — phase-level: shape/NaN-safety on all-zeros dummy
  forward, taken-action gather correctness, all-illegal-but-one mask no-NaN, `aux_coef=0`
  inertness.
- `src/agents/training/outcome_target_fuzz_test.py` — bridge fuzz: `outcome_target` at t+1 ==
  protocol-observed realized outcome of the action taken at t; pairing off-by-one-correct under
  sync **and** `--async-rollout`.

**Modify:**
- `src/agents/model/features_extractor.py` — register `OutcomePredictor` in `__init__` (gated on
  `aux_coef>0`, before the dummy forward `:910`); wire in `forward_internal` after `cls_pool`
  (`:960`); Stage 4 injection in `ProjectionAssembler`/`CLSPool`/`TeamTransformer`;
  `TOKEN_TYPE_OUTCOME`/`NUM_TOKEN_TYPES` (Stage 4c only).
- `src/agents/model/policy.py` — `evaluate_actions` widened to surface the taken-action decode;
  `forward`/`get_distribution`/`predict_values` **unchanged** (inference drops the aux head).
- `src/agents/training/instrumented_ppo.py` — `aux_coef` class attr (`:97` template);
  `+ aux_coef·L_outcome` at the loss sum (`:235`); gather the taken-action token via
  `rollout_data.actions`.
- `src/agents/training/grad_balance.py` — add `grad/aux_share` + `grad/policy_aux_cosine`.
- `src/agents/training/gen3_env.py` — `outcome_target` Dict key in `observation_space` (`:42–45`);
  populate in `embed_battle` from the cached `_pending_delta` (one-step-delayed).
- `src/poke_env/environment/env.py` — add the `outcome_target` key to the space (`:424–429`) and
  the obs dicts (`:591`, `:663`).
- `src/agents/model/model_version.py` — `aux_coef` field, `MODEL_CONFIG_VERSION 12→13`,
  `_migrate_config setdefault`, `check_outcome_aux` (resume-only); `ARCH_SIGNATURE` bump at
  Stage 4.
- `src/main/train_rl_agent.py` — `--aux-coef` flag → `features_extractor_kwargs` + post-build
  attr set (the `--value-tail-weight` template); assert obs-build benchmark unaffected.
- `src/main/prober/engine.py` + `query.py` — Stage-0 anticipatability/headroom/mis-targeting
  probes.
- Docs (always-current rule): `src/agents/model/CLAUDE.md`, `src/agents/training/CLAUDE.md`, root
  `CLAUDE.md` obs/versioning tables.

---

## 7. Verification plan (project idiom)

- **Unit / phase** (`outcome_predictor_test.py`, alongside `phase_modules_test.py`): hand-built
  `ExtractorContext` / `value_pooled`; shape + NaN-safety on all-zeros (the dummy forward runs on
  zeros — masks derived from zero obs must not produce an all-masked attention row); taken-action
  gather correctness; `aux_coef=0` byte-identical inertness.
- **Obs-leak unit test** (mandatory, §3.3): assert `ObsUnpack` / the extractor forward never reads
  `obs["outcome_target"]` — the future cannot reach the policy input.
- **Bridge fuzz** (`outcome_target_fuzz_test.py`, the canonical pattern — real battles in-process,
  intercept protocol, validate in `choose_move`): `outcome_target` written at t+1 matches the
  protocol-observed realized faints/effectiveness/we-moved-first of the action taken at t; pairing
  off-by-one-correct under sync **and** `--async-rollout`.
- **No-op equivalence** (Stage 1): `instrumented_ppo_test.py`-style — at `aux_coef=0`,
  `train/value_loss` + `train/policy_gradient_loss` byte-identical to baseline.
- **Benchmarks** (mandatory obs gate, `src/agents/observation/CLAUDE.md`): `obs_build_benchmark.py`
  + `trainer_turn_benchmark.py` before/after — the new Dict key's per-step memcpy must not regress
  the CPU bottleneck.
- **Smoke** (`--debug --steps 10000`): `[ModelVersion] Round-trip smoke test PASSED`; `aux/*`
  scalars present; for Stage 5a, `aux/latent_std` above floor.
- **Leak ablation** (Stage 4): token-in-value-memory vs token-in-policy-only —
  `explained_variance` / `value_loss` must not improve beyond noise (else leakage → revert).
- **Grad-balance gate** (Stage 2+): `grad/aux_share < 0.4`, `grad/policy_aux_cosine ≥ 0`
  sustained.

---

## 8. Risks, the honest null hypothesis, and the cheap falsifier

**The honest null (the modal outcome, with direct in-repo precedent):** the incoming-damage
belief was wired in, critic-read, AUC-0.79-calibrated, high-saliency — and the policy **still
under-switched and plateaued**. The behavioural gap was partly **credit-assignment / reward**
(stay-into-KO advantage stayed positive; the self-KO finding where the critic over-valued the
post-trade board and neutralized a correct reward), NOT representation. PPO improvement is driven
by the **advantage** signal, not by how richly the trunk encodes the future. So the honest prior
is: `aux/loss` falls, probe AUC rises, `td_resid_tail` moves, **ELO flat or slightly down**.
Worse, a value-path leak makes the advantage *more* biased → ELO down.

**The cheap proxy that falsifies a dud for ZERO retrain (Stage 0, mandatory):** (1) probe the
**existing** checkpoint's next-turn-faint AUC — if already ~0.79, headroom is near-zero, STOP;
(2) `falsify-scan` the loss corpus — if most craters are anticipatable-but-not-acted-on, this is
mis-targeted, prefer the reward/credit levers; (3) the surprise-OHKO provenance (~36%
just-switched + ~42% new/unrevealed) already bounds the addressable fraction to a minority. These
GATE the build.

**Other live risks:** representation-without-behaviour (defended by behavioural-only gates at
every stage + Stage 4's injection as the act-on mechanism); switch-token miscalibration
entrenching under-switching (defended by per-class generators + inverse-propensity weighting +
Stage 5b promotion); aux-vs-PPO gradient fight (defended by `grad/aux_share`); collapse on Stage
5a (defended by the full BYOL stack + `latent_std` NO-GO); the "free labels" overstatement (real
plumbing — a Dict key threaded through the buffer + async collector + one-step-delay + a fuzz
test).

---

## 9. Composition with ai_v6 MCTS + team-completion; relation to ai_v7/v8

**Supersedes the ai_v6 MCTS plan as the anticipation route.** `impl_step5_mcts.md` proposed
decision-time search; the owner ruled out search/MCTS on the model (inference OR training loop).
This design delivers the *same goal* — an anticipatory policy/value — as a **feedforward L3
lever**. MCTS, if ever revisited, is confined to the L4 bucket (an **offline teacher** whose
targets are distilled, never a runtime tree) and is out of scope here.

**Re-homes the orphaned team-completion predictor.** `src/agents/model/team_completion_model.py`
(a masked-slot transformer with frozen backbone embeddings + species/item/move heads, imported by
nothing in the live pipeline) is the canonical re-home target: its per-slot
supervised-prediction-head pattern is exactly the `OutcomePredictor` decode head's structure.
Restructuring it onto the shared trunk (trainable body, shared embeddings) rather than its
standalone offline trainer is a natural follow-on — a second aux head (predict the opponent's
hidden party) sharing the same plumbing.

**Composes with the existing reward/credit levers.** If Stage 0 or Stage 3 shows the bottleneck
is credit-assignment not representation, the cycle goes to `--switch-bias-weight` /
`--self-ko-hp-penalty` — and a future run can **stack** this aux head with those reward levers
(the GO-TO-BUILD exploit-not-explore posture: stack independent cheap amortizable levers into one
fresh run and measure the aggregate).

**Relation to ai_v7/v8.** This is the **L3 spine** for the predictive-representation line. ai_v7
is the natural home for: (a) the opponent-action / hidden-party aux head (re-homed
team-completion); (b) the Stage-5a SPR latent if the grounded target saturates; (c) the Stage-5b
offline counterfactual augmentation if action-as-input generalization proves insufficient on
switches. ai_v8 is where an offline-teacher (L4, search-distilled targets) would compose with this
shaped trunk — the shaped representation is the substrate a distilled teacher would write into,
never a runtime planner.

---

**Bottom line.** Build the cheap, collapse-proof, leak-free spine (Stages 0–3) first; it is
testable for the price of one paired run with kill-gates that fire offline. The outcome-token
injection (Stage 4) is the payoff, entered only when Stage 3 proves the trunk anticipates but the
policy doesn't act — the exact condition under which attending over imagined consequences is the
mechanism that converts anticipation into wins. If the Stage-0 offline probes show the trunk
already anticipates as well as the incoming belief, **kill the lever** and spend the cycle on the
demonstrated reward/credit bottleneck.
