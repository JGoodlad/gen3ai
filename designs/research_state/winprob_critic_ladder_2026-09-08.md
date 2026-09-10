# THE WIN-PROB CRITIC LADDER — short fresh arms, read at 10M by the critic meters

**Status: DESIGN NOTE (explicit-only, `designs/` rules). Written 2026-09-08. Nothing launched,
nothing landed, nothing written under `models/`.**

Owner's direction (2026-09-08, verbatim): *"We would like to work towards a win probability based
value function. This is strategically the direction we want to go. Consider the handcrafted
alternatives as a suboptimal implementation. What is the best path forward?"*

So hand-crafted shaping is out as a direction, and the question is narrowed to: **how do we make the
win-prob critic itself good, fastest and with the most certainty?** This note audits the four levers
that could do that, and lays out a ladder of one-lever arms read by the CRITIC meters — never by
strength.

The control is **arm A at 10M**, `ai_v12_02_winprob_critic`, whose 10M read already exists
(`designs/research_state/measurements/winprob_critic_10M_read_2026-09-07/`). Every arm below is arm
A's own recorded `original_command` with one substitution.

---

## 0. What we are trying to fix, in one paragraph

Arm A (`--critic winprob`: V = sigmoid(win logit), BCE against the terminal win indicator, γ = 1, no
shaping, shared trunk) trained 75M steps. The head is INFORMATIVE — skill score +0.166 [+0.097,
+0.231], resolution +0.0476 [+0.0348, +0.0635] — but it **fails its primary bar**: G1 resolution at
matched strata is 0.0452 [0.0331, 0.0615] against the shaped-era readout's 0.0618, failing 60 of 60
rows, and it is **optimistic against its own Monte-Carlo continuation by +0.0965 [+0.0671, +0.1268]**,
entirely late-game (turn ≥ 25: +0.1687 [+0.0963, +0.2474]; turn ≤ 10: +0.0015 [−0.0335, +0.0385]).
Meanwhile the value gradient's share of the shared trunk fell from 0.44 to ~0.10 over training.

The audit below says the defect is overwhelmingly a **TARGET** problem (lever L2), not a capacity
problem and not a privileged-information problem — and that the fix already exists in the tree behind
a flag that arm A left at zero.

---

## 1. AUDIT

### L1 — ASYMMETRIC (PRIVILEGED) CRITIC · **BUILT 2026-09-08 — `--value-true-team`**

> **STATUS UPDATE (2026-09-08, `gen3_value_true_team_v1`, commit `d94de2e3`).** The audit below
> was right that no value path saw the opponent's true team; it is now a flag, default OFF. What it
> got wrong is the COST of turning it on: the note predicted "FRESH-RUN-ONLY … `check_compatible`
> will reject every existing checkpoint", because it assumed the channel had to widen the 2501-dim
> observation. It does not — the privileged block rides a SEPARATE Dict obs key (`opp_true_team`,
> the `win_target` / `belief_species` precedent), the readout is built LAST, and the injection into
> `value_pooled` is additive, so **there is no `ARCH_SIGNATURE` bump and every existing checkpoint
> still resumes**; only the arm that turns the flag on is fresh-run-only, and only because a
> STRUCTURAL bool cannot flip mid-run. The rest of the audit stands, including the ladder-safety
> argument, which is now asserted rather than argued: `ProjectionAssembler` returns `vf_combined`
> **as** `value_pooled` and `pi_combined` as a concat that never contains it, so `pi` is
> BIT-identical under an arbitrary perturbation of the privileged key at a large random weight
> (`true_team_value_test.py`), and a policy-only backward leaves the route's projection with zero
> gradient.
>
> Two things the audit did not anticipate, both load-bearing for reading the arm:
> * **The key must exist at EVAL, not only in training.** Both critic meters take V and P(win) from
>   the eval traces' npz, so a channel present only during rollout collection would have the arm
>   measured on a V the run never trained. `LocalBattleRunner` — the transport for bridge training,
>   bridge eval AND `replay_counterfactual` — sets an `_opp_player` back-reference on both sides,
>   and `RLPlayer` reads it. **`cf_audit` and `main.critic_gate` therefore read this arm
>   unaffected.** At ladder play there is no runner and the all-zero "unknown" block goes instead.
> * **The prober's offline forwards REFUSE on such a checkpoint** (`ProbeModel._pin`): a V rebuilt
>   from the recorded observation vector alone is V stripped of exactly the privilege the arm
>   exists to measure. Model-free prober commands are unaffected.
>
> The argv is `/home/goodlad/.claude/jobs/9ab51de6/tmp/argv_E_truevalue.txt`
> (`ai_v12_14_ladder_truevalue`, 10M): `checkargs` exit 0, 0 unrecognized, and a clean ARCH SURFACE
> — the flag is `family=CRITIC`, so the surface key set did not grow and no existing argv's verdict
> moved. The audit below is preserved as written.

### L1 — the audit as written (2026-09-08, before the build)

**Verdict: NO value path sees the opponent's true team/sets/items.** The two flags whose names
suggest otherwise are both ON in arm A and are *not* privileged:

- `--value-entity-pool` and `--value-entity-pool-full` are both `True` in arm A's recorded
  `cli_args` (`models/ai_v12_02_winprob_critic/metadata.json`), as is `--value-threat-inject`. They
  appear in arm A's `original_command` and they are part of the **production ARCH surface** —
  `python -m main.checkargs` on arm A's argv prints `✓ every ARCH-surface key matches the production
  mirror` (`production_config@360f8378dd90`).
- Structurally they cannot be privileged: the model consumes a **single flat 2501-dim observation
  vector** (`Gen3ObservationEncoder.get_layout()`), and `Gen3FeaturesExtractor` splits that one
  encoding into the `(pi_features, vf_features)` tuple that
  `Gen3DualHeadMaskablePolicy` consumes. Both heads read the same upstream tensor. `value_entity_pool*`
  therefore selects *which pooled view of the already-observed entities is routed to the value head* —
  a routing choice, not an information channel. ⚠️ **UNVERIFIED at the tensor-construction line**: a
  dedicated audit of the pool builder was dispatched and had not returned when this note was written.
  Nothing in the design docs or the flag table describes these as privileged, and no
  `privileged` / `oracle` / `opp_true` / `hidden_team` value-path flag appears in the run's config.

**What it would take to build.** The env has both teams (the bridge constructs both packed teams and
the recorded `__RECON__` frame carries "both packed teams", `src/agents/training/cf_producer.py:44-47`),
and the ground-truth opponent team **already enters the training loop as a LABEL** — that is what the
belief supervision is: `opp_belief_aux_coef`, `move_belief_coef`, `spread_belief_coef`,
`hp_type_belief_coef`, `item_belief_coef` are all live at 0.05 on arm A, fed through
`src/agents/training/belief_bank.py` from training-only obs keys. A privileged critic channel would
**reuse exactly that plumbing** — the same obs-dict-label trick `WinProbLabelCallback` uses for the
win target (`src/agents/training/win_prob_callback.py:9-22`): declare a training-only obs key holding
the true opponent team encoding, route it into `vf_features` only, and zero it at inference (where
`predict_values` is never called on the ladder path — `src/main/play.py` runs the policy head only).

**Ladder safety.** The property that makes this safe is that `pi_features` and `vf_features` are
separate tensors returned by the extractor and consumed by separate heads; a channel added to the
vf tensor provably cannot reach the policy. ⚠️ That divergence line should be cited exactly before any
build is authorised — it is the whole safety argument.

**Cost: ~12–20 engineering hours, and FRESH-RUN-ONLY** (it changes the observation space and the
value-path weight shapes, so `check_compatible` will reject every existing checkpoint —
`src/agents/model/model_version/`). **Not on the first rung.** It is also the lever with the worst
strategic fit: a critic that needs privileged inputs is not the critic that ships, and every
downstream use we want (search leaf, distillation target, calibration instrument) is on the
un-privileged net. It belongs on the ladder only as a **CEILING PROBE** — "how much of the residual
error is irreducible uncertainty about the opponent's team?" — not as a shippable arm.

### L2 — TARGET VARIANCE · **this is the defect, and the fix is a flag that arm A left at zero**

**What the target actually is today.** The win-prob head is supervised by the **pure Monte-Carlo
terminal indicator, undiscounted, back-filled to every state of the episode**:

> "the outcome is a FUTURE quantity, only known when the battle ends… the per-episode outcome is
> propagated backward to every step of its episode… The win label is **undiscounted** (γ_win = 1):
> every step of an episode carries that episode's actual outcome"
> — `src/agents/training/win_prob_callback.py:9-27`

Trailing in-progress episodes get `win_mask = 0` and are excluded, never trained toward a fabricated
label (same docstring). Under `--critic winprob` that BCE **is** the value loss, at `vf_coef`, tagged
`"value"`, and the scalar `value_loss` term is dropped —
`src/agents/training/instrumented_ppo/train_setup.py:129-138`.

**There is ZERO bootstrapping in the value target.** Note that `gae_lambda` is **hardcoded 0.80**
(`src/main/train/model_build.py:533` on the resume path, `:789` on the fresh path — it is not a CLI
flag), but under `winprob` that λ governs only the ADVANTAGE the policy sees, because the scalar
value term is dropped. The critic's own target is the episode-constant 0/1 label.

**Why that is the defect.** Every state of a lost 40-turn battle receives the identical label 0. The
target therefore carries **no information whatever about where in the episode a state sits**, and
nothing in the loss constrains `V(s_t) − V(s_{t+1})`. That is verbatim the defect
`src/agents/training/td_aux.py:3-9` opens with:

> "The critic's only training signal is a PER-STATE regression… That constrains each state's LEVEL
> and says nothing whatever about the DIFFERENCE between two adjacent states"

**And it shows up in the artifact.** Computed here from
`measurements/winprob_critic_75M_read_2026-09-08/identity_labels.json` (800 rows / 214 battles,
battle-clustered bootstrap, 4 000 draws, seed 0):

| quantity | value |
|---|---|
| corr(turn, V) | **+0.0887** |
| corr(turn, MC) | **−0.2202** |
| **Δ = corr(turn,V) − corr(turn,MC)** | **+0.3089, CI [+0.0828, +0.5101]** — excludes 0 |
| mean within-battle sd(V) | 0.1041 |
| mean within-battle sd(MC), R = 8 | 0.2141 (binomial floor 0.1732 ⇒ true ≈ 0.126) |

As a battle goes on the true win probability **falls**; the critic's output slightly **rises**. Its
within-battle dynamic *range* is roughly right (0.104 against a true ≈ 0.126) — its *sign against the
clock* is not. That is a differential failure, exactly the class a per-state MC regression cannot
see, and it is coherent with test 3's independent finding (commit `73c929e1`) that the head is
compressed toward the base rate: pessimistic early, optimistic late, resolution at 27 % of the
base-rate cap and flat over four cycles.

**The fix already exists, built and pre-registered.** `src/agents/training/td_aux.py` —
`--td-aux-coef λ` — adds the Bellman residual as an explicit loss:

    L_td = λ · mean_t[ ( V(s_t) − r_t − γ·V(s_{t+1}) )² ]

residual-gradient form, both ends live (no detach), per the pre-registration in
`designs/research_state/levers/td_consistency_aux.md`. **Rung 1 (offline, frozen tokens) already MET
its pre-registered gate at λ = 1.0 and λ = 3.0** (`td_aux.py:11-14`); this module is the live-training
half and has never been run on a win-prob critic. Its properties under `--critic winprob` are unusually
clean:

- γ = 1 and r = 0 on every non-terminal, so the residual is exactly `V(s_t) − V(s_{t+1})` — a
  **TD-consistency term in probability space**, which is precisely the missing differential constraint.
- PopArt is refused under `winprob` (`combination_checks.py:318-324`), and `scale` is 1.0 with PopArt
  off, so λ keeps the meaning rung 1 calibrated (`td_aux.py` docstring, point 4).
- Episode boundaries are DROPPED, never zeroed (point 2) — no fabricated `V(s_t) = r_t` at terminals.
- Cost: `TD_AUX_STATES = 512`, `TD_AUX_SEG_LEN = 16` (`td_aux.py:56-66`) — one extra critic forward of
  512 states per minibatch, sized at **~10 % of the train step**.

**Arm A had `td_aux_coef = 0.0`** (`metadata.json`). This is the single highest-value untried lever
in the tree.

**The MC-from-current-policy route also exists**, and is the second half of L2: `cf_producer` emits
per-state MC win-fraction labels from the current snapshot, played **stochastically at temperature
1.0 on both sides** — "CURRENT snapshot on BOTH sides, sampling stochastically at temperature 1.0 —
the regime the training actor itself plays in" (`src/agents/training/cf_producer.py:51-53`), stamped
`opponent: "self_current"`. Consumed during training as a supervised auxiliary at `--cf-winprob-coef`
(`src/agents/training/cf_terms.py:185-186`, likelihood `binomial` or `bce`). Arm A ran it OFF
(`cf_records = False`, `cf_winprob_coef = 0.0`, `cf_head_only = True`). This is a **variance-reduced
target** (R rollouts per state instead of one realised trajectory) but it is *not* a differential
constraint, and it costs real rollout compute — a producer cycle is the expensive lever, TD is the
cheap one. ⚠️ **UNVERIFIED**: the exact per-cycle cost numbers in `designs/training/cf_grounding.md`
were not re-read for this note; quote them from that doc before committing GPU.

### L3 — CAPACITY / GRADIENT SHARE · **one knob, and it is fresh-run-only**

- **`vf_coef = 0.5` on arm A**, and under `winprob` it is the ONE coefficient on the value loss by
  construction — `combination_checks.py:342-348` REFUSES an explicit `--win-prob-coef` alongside
  `--critic winprob` precisely so there cannot be two weights on one loss.
- **It is resume-immutable.** `src/main/train/model_build.py:527` sets `model.vf_coef = args.vf_coef`
  with the comment `# == the saved value (enforced above); set explicitly for parity` — the resume
  path *enforces* equality with the checkpoint's value before setting it. **Changing `vf_coef`
  therefore requires a FRESH RUN**, which is why it is a ladder arm and not a mid-run adjustment.
- **The instrumentation exists**: `grad/value_share` is produced by
  `src/agents/training/grad_balance.py` (`:99` — "`value_term = vf_coef*value_loss`"), which is what
  measured the 0.44 → 0.10 collapse.
- ⚠️ **UNVERIFIED / pending**: whether `Gen3DualHeadMaskablePolicy` honours SB3's
  `share_features_extractor=False` (a separate value trunk), whether any value-head width flag exists,
  and whether any of the August step-size controllers targets value share. A dedicated audit was
  dispatched and had not returned when this note was written. **Treat "a separate value trunk exists"
  as NOT ESTABLISHED** — the extractor returning a `(pi_features, vf_features)` tuple from one module
  is prima facie *incompatible* with SB3's two-extractor path, so assume it needs a build until shown
  otherwise.

**Read of the lever.** A falling `grad/value_share` is a symptom that admits two readings: the critic
is being starved, or the critic has stopped having anything to learn *from this target* and its
gradient has legitimately decayed. **L2 predicts the second reading** — a target with no differential
content saturates. That is why `vf_coef` is a *control* arm on this ladder rather than a favourite:
if raising it does nothing, the starvation reading is dead and L2 owns the whole defect.

### L4 — THE LATE OPTIMISM · **(a) and (b) both ELIMINATED; the target is the cause**

**(a) The 250-turn cap — CLOSED, twice over.**
`src/agents/training/wrappers.py:13-52` (`resolve_episode_end`): under `--critic winprob` a *truncated*
end — the 250-turn cap forfeit and a genuine tie — is **re-labelled TERMINAL**, so SB3 does not apply
`rewards[idx] += gamma * V(s_last)`. Without that, at γ = 1 the last step's target would be
`0 + 1.0·V(s_last) = V(s_last)`, a tautology with identically-zero TD error, and "a policy that stalls
to the cap would be taught nothing about it". The re-labelling means a cap loss carries win-indicator 0
and **is counted in the BCE**. The specification and a composition test through the real SB3
`collect_rollouts` path are `src/agents/training/winprob_truncation_test.py:1-30`.
Second closure, from the artifact: `identity_labels.json` spans turns **2–164**; one battle exceeds
turn 100 and **none reaches 200**, so no timed-out battle is in the audited population at all.

**(b) Greedy-vs-sampled mode gap — ELIMINATED, and it runs the WRONG WAY.**
The MC continuation plays the **trainee GREEDY** — `build_trainee` returns
`RLPlayer(..., stochastic=False)`, docstring "A GREEDY `RLPlayer` (the measured policy)",
`src/main/prober/replay.py:55-68`. The **opponent** plays in the recorded regime, stochastic at temp
1.0 for a checkpoint sentinel (`_RECORDED_CKPT_STOCHASTIC = True`, `:69-72`). Training's actor samples
at temp 1.0. A greedy copy of a net is strictly stronger than a temp-1.0 sample of it — measured at
`+0.037 [+0.007, +0.066]` on 477 states when this was fixed for the opponent side (`replay.py:91-98`).
So the greedy-trainee MC label is biased **HIGH**, and the measured `V − p̂ = +0.0965` is a
**conservative, understated** estimate of the optimism relative to the training regime. The mode gap
cannot explain the optimism away; correcting for it makes the optimism larger.

**(c) The target — CONFIRMED, and it is the account that survives.** The turn-tracking measurement in
§L2 above, plus test 3 (`73c929e1`): the head is compressed toward the base rate, pessimistic early
and optimistic late, with resolution at 27 % of the base-rate cap and **flat over four cycles** —
i.e. not a transient that more steps would fix. An episode-constant γ = 1 label is exactly the target
that produces a base-rate-compressed, clock-blind head.

---

## 2. THE LADDER

Every arm is **arm A's recorded `original_command`** with `--pin-commit`, `--run-name` and `--steps`
stripped and one lever substituted. All four argvs were validated:

```
$ python -m main.checkargs --argv "$(cat argv_<ARM>.txt)"
  unrecognized                   : 0
  ARCH SURFACE vs designs/production_config.json  [production_config@360f8378dd90]
    ✓ every ARCH-surface key matches the production mirror
  ✓ this command still launches
```

— **exit 0 on all four**, and the ARCH SURFACE line is clean on all four, so the 2026-09-06
stripped-architecture failure mode cannot recur here. Argv files:
`/home/goodlad/.claude/jobs/9ab51de6/tmp/argv_{A10M,B_td,C_vf,D_cf}.txt`.

| # | arm | the one substitution | hypothesis (one sentence) | pre-registered bar at 10M vs A@10M | cost | needs built? |
|---|---|---|---|---|---|---|
| **1** | **B — TD-consistency** | `--td-aux-coef 1.0` | A Bellman residual in probability space supplies the *differential* constraint the episode-constant label cannot, so V starts tracking the game clock and resolution rises. | **G1 resolution strictly above A@10M with the DELTA's CI excluding 0** (primary). Secondary, identity test: `corr(turn,V) − corr(turn,MC)` moves toward 0 with its CI excluding A's +0.3089; late-bucket bias (turn ≥ 25) below A@10M's. | ~4.6 h GPU (10M @ ~600 fps, 48 envs) **+ ~10 % train-step overhead** ⇒ ~5.1 h | **NOTHING** — flag exists, rung 1 passed offline |
| **2** | **C — capacity control** | `--vf-coef 1.5` | If the critic is *starved* rather than saturated, tripling its share of the shared trunk raises resolution; if it is saturated, nothing moves and L2 owns the defect. | Same G1 bar. **This arm is informative either way** — a null is a positive result that kills the starvation reading. Also read `grad/value_share` at 10M vs A's trajectory. | ~4.6 h GPU | **NOTHING** — flag exists; fresh-run-only because `vf_coef` is resume-enforced (`model_build.py:527`) |
| **3** | **D — MC-variance reduction** | `--cf-records --cf-winprob-coef 0.5` | Replacing a one-sample terminal label with an R-rollout MC win fraction from the current policy cuts target variance and lets the head resolve within-decile structure. | Same G1 bar. | ~4.6 h GPU **+ the cf producer's rollout compute** (⚠️ read `designs/training/cf_grounding.md`'s cost model before committing — this is the only arm whose cost is not bounded by the flag) | **NOTHING** — flag exists |
| **4** | **E — privileged critic** | *(no argv — does not exist)* | An upper bound: how much of the residual error is irreducible uncertainty about the opponent's team? | Ceiling probe only; never a shippable arm. | ~4.6 h GPU **+ 12–20 eng-h** | **BUILD, fresh-run-only** (obs space + vf weight shapes ⇒ `check_compatible` rejects every checkpoint) |

**Ranking by expected information per GPU-hour, and by certainty:**

1. **B (TD-consistency)** — highest on both axes. It is the only arm whose mechanism is *directly*
   implicated by a measurement made here (the clock-tracking sign inversion, Δ +0.3089 CI [+0.083,
   +0.510]), it is the only one with a **pre-registered offline gate already passed**, it costs one
   flag and ~10 % throughput, and it needs zero engineering.
2. **C (vf_coef)** — second on *certainty of learning something*, because a null result is as
   valuable as a positive one: it closes the gradient-share reading of the 0.44 → 0.10 collapse.
   Cheapest possible arm (one number, no overhead).
3. **D (cf MC labels)** — good mechanism, but it attacks label *variance* while the measured defect
   is label *content*: an R-rollout MC label is still episode-terminal in character and still says
   nothing about `V(s_t) − V(s_{t+1})`. It also has the only unbounded cost on the ladder.
4. **E (privileged)** — highest build cost, and it answers a question about the *ceiling* rather than
   about the shippable critic. Do it only after B/C/D have located the residual.

---

## 2b. READS — one row per arm as it lands (`main.ops.critic_read`, ARM − CONTROL, control = `ai_v12_11_ladder_ctrl10M`@10M)

| arm | lever | G1 resolution · bot Δ | identity bias · late Δ | turn-contrast Δ | verdict | where |
|---|---|---|---|---|---|---|
| `ai_v12_10_ladder_vf15` | `--vf-coef 1.5` | +0.0140 [−0.0080, +0.0389] | +0.0486 [−0.0305, +0.1311] | +0.1634 [−0.0807, +0.3733] | **NOT DETECTED** on all four; identity bias on ALL states +0.048 [+0.014, +0.082] MORE optimistic (vs ZERO, no floor) | `measurements/critic_ladder_reads/vf15_vs_ctrl10M_2026-09-08/` · ledger 2026-09-08 *READ · critic ladder arm 1* |
| `ai_v12_13_ladder_tdaux` | `--td-aux-coef 1.0` | +0.0149 [−0.0045, +0.0343] | −0.0151 [−0.0977, +0.0654] | +0.0092 [−0.2117, +0.2030] | **NOT DETECTED** on all four; G2/G3 worse on points (n.d.); **the clock inversion is absent at 10M** (control −0.094), so the lever's mechanism is untestable at this length | `measurements/critic_ladder_reads/tdaux_vs_ctrl10M_2026-09-09/` · ledger 2026-09-09 *READ · critic ladder arm tdaux* |
| `ai_v12_15_ladder_ctrl10M_b` (THE FLOOR) | `--seed 1001` (replicate) | +0.0102 [−0.0083, +0.0288] | +0.0558 [−0.0188, +0.1335] | +0.0153 [−0.2329, +0.2390] | the run-to-run scale: identity bias ALL differs by **0.070** between replicates (clear of zero), spread ratio t1–3 by 0.028, own-team R² by 0.012, ladder by **45 Elo**; every DETECTED-vs-zero on vf15/cflabels is WITHIN FLOOR ⇒ **zero detected registered rows on any arm** | `measurements/critic_ladder_reads/ctrl10M_b_vs_ctrl10M_2026-09-09_FLOOR/` + `replicate_floor_10M.json` · ledger 2026-09-09 *THE FLOOR* |
| `ai_v12_17_ladder_strata` | `--win-prob-strata-weight 1.0` (realized bot share 0.473 vs 0.5) | +0.0052 [−0.0121, +0.0161] | −0.0153 [−0.0929, +0.0585] | −0.0047 [−0.2340, +0.1806] | **NOT DETECTED / WITHIN FLOOR** on every registered row; spread ratio t1–3 Δ −0.049 n.d.; own-team R² Δ −0.054 n.d.; 400-game read pending | `measurements/critic_ladder_reads/strata_vs_ctrl10M_2026-09-09/` · ledger 2026-09-09 *READ · critic ladder arm 7* |
| **arm 8 — `ai_v12_19_ladder_lambda09`, `--win-prob-lambda 0.9`** | **BUILT `28ece02a`**, pin it; after `ctrl10M_c` (launch gated on two increment tests) | λ-return targets in probability space: `G[t] = (1−λ)V(s[t+1]) + λG[t+1]`, outcome only at the episode end, recorded pre-update V, truncated rows bootstrapped; read on bot resolution at 400/800 games + own-team R², identity test as the guard | — | — | — | registered; ledger 2026-09-09 *BUILT + REGISTRATION · arm 8* |
| `ai_v12_16_ladder_ctrl10M_c` (THE FLOOR, draw 2) | `--seed 1002` (replicate) | −0.0010 [−0.0158, +0.0106] | +0.0088 [−0.0702, +0.0852] | +0.0673 [−0.1615, +0.2725] | second replicate difference: identity bias ALL +0.015 (draw 1 +0.070) — the floor is a RANGE, bar = the wider draw; three-draw spreads: 45 Elo, G7 ratio 0.93–1.07, first jump 0.21 | `measurements/critic_ladder_reads/ctrl10M_c_vs_ctrl10M_2026-09-09_FLOOR2/` · ledger 2026-09-09 *THE FLOOR, SECOND DRAW* |
| `ai_v12_12_ladder_cflabels` | `--cf-records --cf-winprob-coef 0.5` (+ `--checkpoint-every-steps 500000`) | +0.0054 [−0.0121, +0.0191] | +0.0380 [−0.0382, +0.1122] | +0.1928 [−0.0549, +0.4084] | **NOT DETECTED** on every registered row; the own-team R² +0.084 detection was a DECODER-POWER ARTEFACT of the 4× quota (WITHDRAWN, `matched_quota/`; re-read by the quota-matched tool in `critic_read_v3.md` as **+0.0077 [−0.1353, +0.0816] NOT DETECTED**); head unbiased vs its continuation (Δ +0.040 [+0.007, +0.071], stands vs zero) | `measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/` · ledger 2026-09-09 *READ* + *RETRACTION* |

No replicate floor exists until `ai_v12_15_ladder_ctrl10M_b` lands; every DETECTED before then is against zero.

All three rows above were regenerated on 2026-09-09 by the quota-matched tool (`critic_read_v3.md/json` beside each `critic_read.md`). **`vf15` and `tdaux` are SYMMETRIC pairs** — they and the control all ran the default quota, realized cap 8/12 on every side — so their v3 delta rows are **bit-for-bit identical to v2**; only `cflabels` was matched, and only its five frame-sensitive rows moved.

---

## 2c. STATUS after the probe read and the head refit (2026-09-09)

The three offline reads (`winprob_mixture_diagnostic_2026-09-09`, `winprob_probe_read_2026-09-09`, `winprob_head_refit_2026-09-09`) re-rank the ladder without a GPU-hour:

| arm | status | why |
|---|---|---|
| `cflabels` (`--cf-records --cf-winprob-coef 0.5`) | **PROMOTED — the arm that matters**; running | the only queued arm in the supported class (target variance / conditional label); read on the turn-1–3 spread ratio and turn-1 own-team R² beside the registered rows |
| **arm 7 — `ai_v12_17_ladder_strata`, `--win-prob-strata-weight 1.0`** | **BUILT `f871e79f`**, pin it; after `ctrl10M_b` (launch gated on the increment's functional test) | balances the win-prob BCE across the four opponent CLASSES (per-name unavailable — the id never reaches the buffer); 8× cap ⇒ 44/56 at the production mix; read on `cond.spread_ratio.t1_3` + `cond.own_team_r2.t1` |
| `ctrl10M_b` | unchanged — the replicate floor, next | two lever arms lean +0.014 / +0.015 on bot resolution; the floor says whether that is the control's shortfall |
| `tdaux` | read NOT DETECTED; **DEMOTED** | a bootstrap regresses on the head's own unconditioned V at turn 1; and the clock inversion it targets is absent at 10M |
| `truevalue` | **DEMOTED to last** | a representation probe on a head that discards its representation |
| `vf15` | read NOT DETECTED | capacity is second-order and appears only under the conditional target; the separate value trunk stays HELD |
| §12 FiLM conditioning (mixture sketch) | **STRUCK** | `value_pooled` already carries the opponent and the own team |
| head-side optimisation (value replay, periodic refit, head lr) | **RULED OUT** | the idealised version reproduced the failure offline from scratch and from the online weights |

---

## 2d. THE HIGH-POWER OFFLINE RE-READ (2026-09-09) — when the constraint is POWER, not effect size

Five arms read at 100 games, **zero detected registered rows**, and the replicate floor
(`ctrl10M_b` vs `ctrl10M` — two draws from the SAME configuration, never a seed sweep: the seed does
not reach the battle stream) showing every DETECTED-vs-zero on `vf15` / `cflabels` to be WITHIN
FLOOR. The floor read also named the reason, and it is not "the arms are identical": the
battle-clustered CI on the turn-1-3 spread ratio is ~±0.4 around a control value of ~0.12 with only
12 opponent cells, resolution rows resolve only deltas above ~0.03, and the run-to-run floor on the
low-variance rows (bot resolution 0.010, own-team R² 0.012, spread ratio t1-3 0.028) sits **below**
the battle-level CI width. A read whose floor is smaller than its own error bar is not measuring the
arms; it is measuring how few battles it saw.

Every arm keeps its 10M checkpoint (`eval_traces/step_10000032/snapshot.zip`), so the read can be
retaken at any size for CPU and no GPU. `python -m main.ops.eval_trace_gen <run>@10000032 --games
400 --sentinels 6 --out DIR` replays the cycle offline through the same `main.eval_worker` path,
capturing EVERY battle instead of the live outcome quota's ~200 — ~4,800 traced battles a side.
`main.ops.critic_read … --arm-traces DIR --control-traces DIR` then reads the pair from those
cycles, taking the registered G1-G4 rows from the real run's ladder (which an offline cycle has
none of, and cannot have).

🚨 **THE HIGH-POWER ROWS ARE A SEPARATE TABLE WITH A SEPARATE FLOOR.** They are a different number
of games and a different capture rule from the 100-game reads in §2b, so they are not rows of that
table and must never be pasted into it. `critic_read` enforces the half of this it can — it REFUSES
to difference an offline frame against a live one at all, so no single report can mix them — and
the reporting rule covers the rest. The floor for the hp table is its own pair: `ctrl10M_b` vs
`ctrl10M`, generated at the identical spec. Without it, a high-power DETECTED is a detection against
zero, and the 2026-09-09 RETRACTION is exactly what a bigger frame does to a fitted row on its own.

⚠️ **The sentinel count did NOT rise on these runs.** `--sentinels 6` asks for six; each ladder arm
saved four snapshots, one of them at the read step (excluded as a self-mirror), so the cycles carry
**three** pool sentinels — the same 12 opponent cells as the live read. The power here comes from
4× the games and ~24× the traced battles, not from more cells, and the between-opponent spread
ratio's cell count is therefore unchanged. Closing that would need a run that snapshots more often;
it is not something the generator can supply after the fact.

## 2d. THE ARMS AT 400 GAMES (2026-09-09) — the slot decision

Offline 4,800-battle cycles on both sides (`main.ops.eval_trace_gen`), quota-matched; the replicate pair agrees to ±0.007 on bot resolution. `vf15` +0.003 [−0.005, +0.010], `tdaux` −0.002 [−0.010, +0.005], `cflabels` +0.001 [−0.006, +0.009]; spread ratio t1–3 all inside ±0.06 (replicate difference 0.066). **No lever moves the decision rows at 10M. No replicate is spent (`cflabels_b` dropped). Arm 8 = `--win-prob-lambda 0.9` (λ-return targets), then dense auxiliary targets.** Ledger 2026-09-09 · *THE ARMS AT 400 GAMES*.

---

## 3. THE FIRST PAIR

**Launch B, then C — sequentially, both at `--n-envs 48`.**

**Why not in parallel.** The GPU takes one arm at a time at 48 envs. Two 24-env arms would fit, but
the training loop is **rollout-bound** (memory: `project_throughput_profile`), so halving envs roughly
halves each arm's fps: each would need ~9.3 h for 10M, against ~9.2 h to run both sequentially at 48.
**There is no wall-clock saving, and it costs the matched control** — A@10M was measured at 48 envs,
and `n_envs` changes the effective batch composition and the self-play mixture rate, so a 24-env arm
is no longer a one-lever change against A@10M. Sequential at 48 envs preserves the only clean
comparison we have. Run B first; C is the control that interprets B's result either way.

**Validated argvs** (each is arm A's `original_command` minus `--pin-commit f971caf2`, `--run-name`,
`--steps`, plus the lever; both verified `✓ this command still launches` with a clean ARCH SURFACE):

```bash
export PYTHONPATH=$PYTHONPATH:src

# ARM B — TD-CONSISTENCY.  Lever: --td-aux-coef 1.0 (arm A had 0.0).
python -m main.launcher $(cat /home/goodlad/.claude/jobs/9ab51de6/tmp/argv_B_td.txt)
#   ... --critic winprob --no-hand-shaping --terminal-indicator --victory-value 1.0
#       --draw-penalty 0 --vf-coef 0.5 --td-aux-coef 1.0
#       --steps 10000000 --run-name ai_v12_13_ladder_tdaux

# ARM C — CAPACITY CONTROL.  Lever: --vf-coef 1.5 (arm A had 0.5).
python -m main.launcher $(cat /home/goodlad/.claude/jobs/9ab51de6/tmp/argv_C_vf.txt)
#   ... --critic winprob --no-hand-shaping --terminal-indicator --victory-value 1.0
#       --draw-penalty 0 --vf-coef 1.5
#       --steps 10000000 --run-name ai_v12_10_ladder_vf15
```

🚨 **The argv files are the source of truth, not the abbreviated comments above.** Re-run
`python -m main.checkargs --argv "$(cat …)"` immediately before launching — HEAD may have moved since
this note was written, and a fresh arm is judged by the CURRENT parser.

**How each is read at 10M** — 🚨 **THE READ IS ONE COMMAND** (built 2026-09-08, ledger
*INSTRUMENT · main.ops.critic_read*):

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read <arm> --control ai_v12_11_ladder_ctrl10M \
    --step 10000032 --out <outside models/> --ledger-line   # --floor-json when a floor exists
```

It resolves each run's LAST COMPLETE eval cycle (a manifest whose `selection` block is written —
i.e. the cycle has COLLECTED), runs all three meters, and emits `critic_read.md` +
`critic_read.json` with every registered quantity as **ARM − CONTROL** and the delta's own
battle-clustered CI. It REFUSES rather than emit a partial table: no manifest, an uncollected
cycle, npz without `win_probs`, an anchor rate under 0.90, or a draw/timeout share over 25% exits
2 naming the cause. The control's readout is cached per run, so the second arm read against the
same control costs only its own half — with both halves cached, a full pair re-read including the
whole conditioning section is **~40 s**.

🚨 **THE READ IS QUOTA-MATCHED BY DEFAULT (2026-09-09), AND THAT IS NOT COSMETIC.** The arms run
at trace quota **40/40/10** against the control's **5/10/5**, so the arm's read frame is 2–4× the
control's — and a conditioning row whose estimator is a **FIT on the frame** (`cond.own_team_r2.*`,
`cond.opp_class_auc.t1`) or an **UNCORRECTED second moment** (`cond.spread_ratio_raw.*`) has an
expectation that moves with frame SIZE. Horvitz-Thompson reweighting corrects the loss-ENRICHMENT;
it does not correct that. `main.ops.quota_match` therefore subsamples the RICHER side to the
poorer side's **realized** per-opponent capture profile (the cap on disk — a nominal 5/10/5 lands
as **8/12** under shard rounding, and matching the nominal numbers over-shrinks by ~35%) over 21
seeded in-memory draws with the capture rates recomputed per draw, and the row's label is decided
on the **matched** delta. The as-traced number is printed beside it marked **UNMATCHED** and
carries **no label**; a second **decoder-matched** rung is searched for and printed, because
`MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity
and equal battle counts can leave the richer arm's decoder with FEWER battles than the control's.
`--no-quota-match` opts out and then those rows print `UNMATCHED — not a reading` INSTEAD of a
label — on an unequal frame neither DETECTED nor NOT DETECTED is a claim the report may make. Every
other row (the noise-corrected spread ratio, the spread delta, the Elo slope, the gate rows, the
identity rows) is a weighted mean or a regression on cell means and is read AS TRACED. The
2026-09-09 RETRACTION is what this prevents: `cond.own_team_r2.t1` read **+0.0841 [+0.0323,
+0.1825] DETECTED** as traced and **+0.008 [−0.135, +0.082] NOT DETECTED** matched. Matching a pair
costs **~8 s of CPU** and reproduces `cflabels_vs_ctrl10M_2026-09-09/matched_quota/` — including
finding the same 11/16 decoder rung that measurement hand-picked. The two realized profiles are
printed in the report header whether or not anything was matched, so a symmetric pair says so in
print. 🚨 **This applies to a CONTROL-vs-CONTROL read too** — the replicate floor
(`ctrl10M_b` vs `ctrl10M`) carries the same asymmetry with the same sign, and a floor magnitude
read off an unmatched decoder row would bake the artefact into every later arm's bar.

🚨 **PASS `--step` WHENEVER THE ARM'S LAUNCHER MAY STILL BE ALIVE.** `--on-live skip-newest` is the
default and it DROPS the newest cycle when any process still names the run, so a finished 10M arm
whose launcher had not yet exited is read at **8M**. That happened to the 2026-09-09 tdaux read.
The tool now prints WHICH cycle it chose and WHY — for both runs, on stdout before anything
expensive runs and again in the report header — and `--step N` pins both runs (`--control-step N`
pins the control alone). A run with no `step_N` REFUSES, naming the steps it has.

What it composes:

1. **Identity test** — `python -m agents.training.cf_audit <run> --step <10M step dir> --impl rust
   --rollouts 8 --states 800 --anchors 150 --seed 0 --out <outside models/>`, **no `--checkpoint`**
   (ledger `1a1ad063`). Read: `V − p̂` with its battle-clustered CI, by turn
   bucket (early ≤ 10 / mid 11–24 / late ≥ 25), and the `corr(turn,V) − corr(turn,MC)` contrast
   against A's **+0.3089 [+0.0828, +0.5101]**. Reported under three selection weightings, with
   the registered one starred: capture-rate reweighted for the biases (rule 17), and the POOLED
   UNWEIGHTED pair for the turn-contrast, because that is the estimand A's +0.3089 is.
2. **G1 resolution at matched strata** — `python -m main.critic_gate`, `all` / `bot` / `pool`,
   against A@10M's own rows from `measurements/winprob_critic_10M_read_2026-09-07/`. A FRESH arm
   has no parent, so the tool passes `--parent v9_fold_parent --famine-comparator off
   --skip-meter`: G1–G4 read their bars from the committed calibration baseline, and the ladder /
   famine / G7 sections that would use a parent are not reported — **strength is not read.**
3. **CONDITIONING** (added 2026-09-09) — `main.ops.conditioning_meters`, promoted out of
   `measurements/winprob_mixture_diagnostic_2026-09-09/` (the spread identity, the Elo slope) and
   `measurements/winprob_probe_read_2026-09-09/` (the own-team leave-one-battle-out target, the
   battle-grouped folds, the HT reweighting), so the ladder and those measurements share ONE
   implementation. **These are the rows the current ladder is read on**, because three offline
   reads established that the critic's defect is a conditioning failure in the win head — it
   emits one near-marginal win probability regardless of opponent AND of its own team. Ten rows:

   | row | what it is | a calibrated head reads |
   |---|---|---|
   | `cond.spread_ratio.t1_3` ⭐ | between-opponent spread of `V` ÷ that of the outcome, turn 1–3, each side corrected for its own sampling noise | **1.0** (it is an identity: `E[V\|opp] = E[y\|opp]`) |
   | `cond.spread_ratio_raw.t1_3` | the same, UNCORRECTED and UNCLAMPED — the monotone companion | — |
   | `cond.spread_delta.t1_3` | `sd(V) − sd(outcome)`, corrected | 0.0 |
   | `cond.spread_ratio{,_raw}.all`, `cond.spread_delta.all` | the same three over ALL states | 1.0 / — / 0.0 |
   | `cond.elo_slope` | slope of bias (`V` − true win rate) on opponent Elo, per 100 Elo | 0.0 |
   | `cond.own_team_r2.t1` ⭐ | own-team **leave-one-battle-out** win-rate R² decoded from `V` alone, turn 1 | high; the head's is ~0.01 against ~0.67 available in `value_pooled` |
   | `cond.own_team_r2.all` | the same over all turns | — |
   | `cond.opp_class_auc.t1` | opponent-CLASS (pool vs bot) AUC of `V`, turn 1 | high; chance is 0.5 |

   Every row is computed on the **RECORDED** `win_probs` of the read cycle — that column IS the
   cycle's own model, so **no model forward is done** — HT-reweighted by that cycle's capture
   rates, with the interval resampling battles WITHIN their opponent cell (the roster is a fixed
   pinned set) and redrawing the outcome side from its own Binomial(`battles_played`, true win
   rate). The two ⭐ rows are HEADLINES and appear in the ledger quote.

   🚨 **The frozen-forward-vs-recorded distinction matters only ACROSS cycles, and this reads
   one.** A recorded `V` pooled over several cycles carries the trainee's own improvement between
   cells and reads as separation the head does not have — CTRL's pooled recorded ratio is 1.079
   against 0.066 for a frozen forward (`winprob_head_refit_2026-09-09` §12 hazard 3). This module
   never pools cycles.

   🚨 **A SPREAD RATIO CAN SIT BELOW ITS OWN INTERVAL, for two independent reasons.** (1) The
   noise-corrected ratio is **CLAMPED** — each side's sampling variance is subtracted and a
   negative result floored at zero, a biased non-monotone operator — so a 0.000 routinely carries
   a CI like [0.21, 0.53] (`winprob_head_refit_2026-09-09` §12 hazard 1); the tdaux arm reads
   exactly that, 0.0000 [0.0000, 0.5489], against an unclamped 0.1984. (2) A **resampled
   between-group variance is UPWARD BIASED**, so even the UNCLAMPED `ratio_raw` can sit below its
   own draws (tdaux: 0.1984 against [0.2117, 0.6061]); the mixture diagnostic reports the same
   shape on its η² rows. **The interval is the read**, and both effects hit arm and control
   alike, so the DELTA is the quantity least disturbed by either.

   🚨 **The Elo-slope row is OMITTED WITH A REASON, never reported on two scales.**
   `fit_ladder` silently returns an UNANCHORED ladder when called from the wrong cwd — ratings
   near 1000 instead of near 2000, `anchored_to_bots: false`, no error (mixture-diagnostic hazard
   1). The strength axis chdirs to the repo root, REFUSES an unanchored refit, and verifies the
   positional `sentinel_k` → snapshot map against the manifest's own win counts. `--cond-ladder
   off` skips the axis; every other conditioning row is unaffected, because the spread identity
   needs no strength axis at all.

🚨 **Arm A's own 10M eval traces are GONE** (`--keep-eval-trace-steps 20` trimmed them; A's tree
now starts at 36M). A matched-10M delta against A cannot be recomputed from traces — the committed
numbers in `measurements/winprob_critic_10M_read_2026-09-07/` are the record, and a fresh read
against A resolves to A@74M, which is a CROSS-STEP, cross-regime comparison and must be labelled
as one.

🚨 **The bar is stated as a DELTA with a CI, not two overlapping bars.** Per the standing rule, an
"improved" or "equivalent" sentence requires the *difference's* interval; A@10M and the new arm are
both measured, so the delta CI is available and must be the thing quoted.

**The composition step.** If B clears its bar and C is null, the story is closed at the target and the
next arm is **B + D** (TD-consistency plus MC-variance-reduced labels), which attack content and
variance respectively and should compose. If B clears and C *also* moves, the two are confounded on a
shared trunk and the composition step is **B at the vf_coef that C found**, read against B alone. If
B is null, the differential-constraint account is dead, and the ladder's weight shifts to D and then
to E as a ceiling probe.

---

## 4. Readiness summary — what is a flag today, and what is a build

| lever | flag | ready today? |
|---|---|---|
| **L2 · TD-consistency in probability space** | `--td-aux-coef` | **YES** — built, offline rung 1 passed, arm A left it at 0.0 |
| **L2 · MC win-fraction auxiliary labels** | `--cf-records --cf-winprob-coef` | **YES** — built; cost model must be read first |
| **L3 · critic gradient share** | `--vf-coef` | **YES**, but **fresh-run-only** (resume-enforced, `model_build.py:527`) |
| **L3 · separate value trunk / wider value head** | — | ⚠️ **NOT ESTABLISHED** — assume a build until the extractor audit lands |
| **L1 · privileged (asymmetric) critic** | — | **NO** — ~12–20 eng-h, fresh-run-only, and a ceiling probe rather than a shippable arm |

**Open items this note did not close** (both dispatched, neither returned before writing): the
tensor-level confirmation that `value_entity_pool_full` is a routing choice and not an information
channel, and whether `Gen3DualHeadMaskablePolicy` can honour `share_features_extractor=False`. Neither
changes the first pair.
