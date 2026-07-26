# Design — Search-as-Teacher (selective Expert Iteration)

**Status: BUILT — Phase 0 (selection) + Phase 1 (AWR policy distillation) + the SUPPLY+POOL upgrade
(`--teacher-persistent`).** Off by default (`--search-teacher` absent / `--search-teacher-coef 0` ⇒
byte-identical to stock PPO). Code in `src/agents/training/teacher/` (`selection`, `buffer`, `produce`,
`opponent_resolver`, `generate`, `callback`) + `src/main/search_teacher_worker.py` (per-cycle) +
`src/main/search_teacher_persistent_worker.py` (persistent) + the AWR aux loss in `instrumented_ppo.py`.

**The supply+pool upgrade (`--teacher-persistent`, the §10/§11 "produce continuously" lever, BUILT):**
the per-cycle mode reads eval traces (a trickle every ~2M steps); the persistent mode runs a
LONG-LIVED worker pool that GENERATES its own fresh losses — the frozen trainee vs sampled current
opponents (recent pool snapshots + bots), recorded via the eval path (`teacher/generate.py` →
`begin_forensic_cycle` + `run_local_battles`) — and searches them continuously, dripping corrections
into the buffer instead of a 2M-step burst. It never touches the training hot path (a frozen-snapshot
side activity, like eval); the parent RE-FREEZES the snapshot every `--teacher-refresh-steps` (default
500k) so long-lived workers track the moving policy, and ingests correction shards incrementally each
`_on_step`. Because the worker CHOSE the opponent, the exact-opponent is known directly (no
sentinel-resolution fragility — `produce_correction` gets the opp ckpt straight from the generator).
Validated: one worker published 8 verified-better corrections from self-generated battles in ~150 s
(vs the per-cycle trickle). The §3 "where to search" funnel runs `falsify_gate=False` here (supply is
plentiful → the CONFIRM is the gate; falsify is for the scarce per-cycle path). **Lifecycle-hardened
after an adversarial review** (a multi-process, long-lived pool must self-heal): the parent respawns a
crashed worker on a step-backoff (`teacher/workers_alive`); snapshots keep the latest 3 (numeric
version key) and the worker existence-checks + try/excepts every snapshot/opponent load (a pruned/corrupt
file skips, never crashes); `_ingest` deletes-before-buffering (a delete failure drops, never
duplicates); the per-iter `ProbeSession` + warm `SearchSession` are bounded (context-manager close +
periodic recycle); and the buffer is `_excluded_save_params` (it holds a `threading.Lock` → otherwise
`model.save()` crashes the pre-train roundtrip smoke for every `--search-teacher` run; transient like the
rollout buffer). The "expert" (the
depth-limited beam search, `main/prober/better_line.py`) and the verification tier (rollout-confirm,
`utils/bridge/counterfactual.py`) are the proven, faithful foundation. This doc specifies turning that
search into a **training signal**, applied **selectively** (a compute budget, not every turn) so the
policy distills what the search found it should have done. **Phase 2 (off-policy value term — wired
but `--search-teacher-value-coef 0` by default) + Phase 3 (priority sampling, win/negative balance,
plateau gating) are follow-ons.**

This is **Expert Iteration (ExIt) / search-based policy improvement** — the AlphaZero/DAGGER family —
but (a) applied to a few high-leverage turns instead of every move, and (b) injected as an
**auxiliary supervised loss** rather than by replacing the PPO rollout (the searched action is
off-policy, so it must not enter the on-policy buffer). It is the offline-teacher plateau-breaker the
frontier roadmap names (`project_model_frontier_roadmap`, `project_positional_grind_decomposition`).

---

## 1. The two signals → the two heads

The search at a state `s` produces two distinct targets that behave very differently:

| signal | target | head | bias under losses-only selection |
|---|---|---|---|
| **policy** | the verified-better action `A*` (the divergence-ply move) | actor | ~unbiased ("A* beats B here" is locally true regardless of episode outcome) |
| **value** | the rollout-**confirmed** return of `s` (NOT the critic's optimistic backed-up value) | critic | **biased** (outcome-correlated → losses-only shifts V down) |

The **value signal is the higher-confidence place to start**: the documented failure is critic
tail-blindness (`eval/td_resid_tail`, `triage`'s `critic_blindspot`), and search directly corrects
V(s) at exactly the craters the critic mis-prices. It also can't collapse the policy. The **policy
signal is higher-leverage per correction but riskier** (entropy collapse, opponent-overfit) — add it
second, once the value path is proven.

**Loss to fold into `InstrumentedMaskablePPO.train()`** (one more aux term beside the belief/win-prob/
value-dist losses — `instrumented_ppo._*_loss` is the exact existing pattern):

```
L_teacher = coef_v · MSE(V(s), confirmed_value(s))            # v1
          + coef_pi · CE(π(·|s), A*)         (advantage-weighted)   # v2
```

No importance sampling, no off-policy policy-gradient correction — both terms are **supervised**
(regression + classification of a target), so they sidestep PPO's on-policy assumption entirely.

---

## 2. What changes in SB3 PPO (and what does NOT)

**Unchanged:** GAE, advantage, the clip objective, the rollout buffer, the on-policy cadence. The
searched action is off-policy; it must NEVER enter `rollout_buffer` or the policy-gradient path.

**New (all mirror existing infra):**
1. **A side buffer** — a growing DAGGER dataset of `(obs, action_mask, A*, confirmed_value, weight)`
   tuples, separate from the rollout buffer. Bounded ring (recency) so stale corrections age out.
2. **An aux-loss term** in `instrumented_ppo.train()` — sample a minibatch from the side buffer each
   `train()` step, fold `coef · L_teacher`. Rides the existing aux-coef threading + the grad-balance
   probe (`grad/searchteacher_share` / `_policy_cosine` — the live "is the teacher fighting the actor"
   signal, exactly like every other aux head).
3. **A background search-worker pool** — rides the `--eval-workers` subprocess pattern
   (`agents/training/eval_worker.py`, `spawn_eval_workers`): workers drain a queue of `(trace, turn)`
   candidates, run the search + confirm on the just-written eval traces, and write verified
   corrections into the side buffer. Runs on `--eval-device` / spare cores, **never blocks training**.

So the surface area is: **one aux term + one side buffer + one worker pool** — no core-algorithm
change. Versioning: the side-buffer/loss is training-only (a coef like `ent_coef`, NOT version-locked,
read-back on resume); it adds no forward/weight change → no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION`
bump (`coef = 0` ⇒ byte-identical default).

---

## 3. Selection — where to spend the budget

**Do not conflate "where the model is wrong" with "what keeps the training set representative."**
Outcome (loss) is only a cheap *prior* for where mistakes cluster.

- **Where to search (mistake-likelihood):** the per-decision **TD-residual / value-crater**
  (`r + γV(s′) − V(s)` most negative), pre-filtered by `falsify` to keep only **reducible MISTAKES**,
  not aleatoric **LUCK** (don't teach against dice). The prober already computes all of this:
  `scan`/`triage` (worst-ΔV craters) → `falsify` (luck vs mistake). This two-stage funnel — *cheap
  model-free triage, then expensive search only on the promising turns* — is what protects the budget.
- **The crater is the consequence, not the cause.** Search a small **window** (the crater turn + 1–2
  turns *before*) and let the search *attribute* which turn had the recoverable line. Evidence from
  the better-line demo: searching the crater (turn 27) returned "already lost"; searching turn 26
  found Spore (+17.8 ΔV). "1 turn before" is also where the nets are competent enough for the search
  to be **trustworthy** (the search leans on the policy's top-k priors + the critic's leaf values).
- **Keep the corrected SET representative** to avoid biasing the critic down: include **~15–25%
  non-loss turns**, AND keep **"no better line" results as negative examples** (search confirms the
  played move was fine) so the critic doesn't learn "every state I examine is worse than I thought."

---

## 4. The strictly-better gate (verified, not critic-claimed)

A correction is distilled **only if it passes the 3-tier gate** (the better-line tool already implements
all three):
1. search backed-up value of `A*` > played action's value, by a margin (cheap, critic-trusted);
2. **rollout-confirm**: play `A*` to the end vs the EXACT opponent, `N` games, and the **Wilson lower
   bound** of the win-rate exceeds the played line's realized rate (CI-gated → *verified* better);
3. (optional) `falsify` already proved it was a mistake, not luck.

Tier 2 is load-bearing: the demo showed the critic over-valuing Spore (95% predicted vs 62% confirmed).
**Distill the CONFIRMED value, never the critic's optimistic backed-up value.** Only CI-gated,
confirmed-better corrections become targets — this is what makes the signal *strictly better* rather
than *hallucinated better*.

---

## 5. The EXACT opponent (per owner request — high-value)

In training the opponent that produced the trajectory is **known and reloadable**: a self-play
sentinel (`models/<run>/snapshots/`), a stable opponent (`ext_<run>`), or a fixed bot. The trace is
already namespaced by opponent (`eval_traces/step_N/<opponent>/`) and the
`*_reconstruction.json` carries the resolved seed + teams.

- **Interior-ply opponent** = the **exact** reloaded opponent model, via the already-built
  `interior_opponent="ckpt"` / `opponent_ckpt` path (NOT the `self_model_approx` proxy the offline
  prober defaults to). This makes the searched line faithful to the real opponent the policy faces.
- **Confirm-rollout opponent** = the **same** exact opponent (`replay_counterfactual`'s
  `opponent_ckpt` already supports it). This **kills the opponent-overfit edge case**: we are not
  proving "A* beats a proxy," we are proving "A* beats *this* opponent" — exactly the distribution the
  policy is being trained against.
- Resolution: the worker maps the trace's opponent label → its checkpoint (sentinels via the
  `SnapshotPool` path, stable via `fixed_opponent_pool`, bots are reproducible from
  `_EVAL_OPPONENT_SPECS`). Model opponents (sentinels/stable) get the exact-model interior + confirm;
  **heuristic bots** have no obs→action model, so they use the recorded move at the divergence ply +
  the bot itself in the confirm rollout (the bot IS reproducible — `replay_counterfactual` rebuilds
  it), and the interior plies fall back to the trainee proxy (flagged, depth-1 stays faithful).

---

## 6. Compute budget

Unit = **searches per eval cycle** (eval fires every `EVAL_FREQ_STEPS` ≈ 2M steps):
`N = budget_seconds / (search_time + confirm_rollouts · rollout_time)`. With the optimized search
(~3.4 s warm) + e.g. 8 confirm games (~1.2 s each ≈ 10 s), a correction costs ~13 s; a handful of
background workers on spare cores yield a few hundred verified corrections per cycle — a small but
very high-quality supervised drip. **Prioritize** the budget by `|ΔV| · P(reducible mistake)` (the
falsify pre-filter ranks this cheaply) = prioritized expert iteration. The budget is a hard knob
(`--teacher-search-budget`), capped per cycle; the worker stops when spent.

---

## 7. When to apply

Turn it on at **plateau**, not early (`td_resid_tail` stuck, ELO flat — the self-play-treadmill
convergence). Early, plain PPO improves fast and search-teaching is wasted compute. Late, the search
injects an external improvement signal self-play can't generate. Gate: `--search-teacher` off by
default; recommended enable when `eval/td_resid_tail` has been flat for K cycles.

---

## 8. Metrics

- **Helped the policy:** agreement rate with the search recommendation on a **held-out** set of
  verified-better turns (never trained on); ELO / `win_rate_vs_bots` / `win_rate_vs_pool`.
- **Helped the critic (watch FIRST):** `eval/td_resid_tail` → 0; the prober `calibration` reliability
  gap shrinks; `train/explained_variance` ↑.
- **Didn't hurt:** entropy not collapsing; `train/return_mean`/`return_std` not lurching (value-shift
  canary); `grad/searchteacher_policy_cosine` not strongly negative.
- **Teacher yield:** fraction of searched turns that pass the confirm gate (`teacher/yield`) — if low,
  the budget is being spent where there's nothing to teach; re-tune the triage filter.
- New TB group `teacher/*` (its own prefix, like `belief/`/`win_prob/`): `corrections_per_cycle`,
  `yield`, `mean_confirmed_dwin`, `value_coef_share`, `policy_coef_share`, `buffer_size`.

---

## 9. Edge cases

- **Search over-optimism** → distill the rollout-CONFIRMED value, not the backed-up value (§4).
- **Opponent-specific lines** → use the EXACT opponent in interior + confirm (§5); optionally confirm
  vs ≥2 pool opponents and down-weight lines that only beat one.
- **Staleness** — the policy at train-time N+k may already do the right thing at a state searched at
  N → re-verify the policy still disagrees before distilling, and age the side buffer by recency.
- **Faithfulness decays with depth** — the interior opponent past the divergence is approximate even
  with the exact model (it's greedy, not the real stochastic sample) → distill primarily the
  **first (divergence) action** + the value; treat deeper PV actions as soft / value-only.
- **Selection feedback loop** — searching where the critic is surprised then training it not to be
  surprised there can teach memorization → the held-out set + the calibration metric guard it.
- **Value-head drift** — small `coef_v`, balanced selection (wins + negatives), watch the value-scale
  diagnostics (`popart/*`, `train/return_*`).
- **Aux fighting the actor** — the grad-balance probe (`grad/searchteacher_policy_cosine`); lower the
  coef on a strong negative.

---

## 10. Performance (this is now in the training loop — optimized + planned)

Profiled: the bottleneck is **obs materialization** (~58%); the `serializeBattle` clones are ~2%.
**Done** (the search is now ~3.4 s warm at depth 2, down from ~5.65 s — ~1.66×):
- **L1** ONE shared `replay_battle` feeds both the anchor choice-map and the opponent's
  `infer_action_indices` history (was two full replays).
- **L2** per-node policy forwards **batched** (`action_probs_batch`, was one forward per node).
- **L3** `materialize_decisions(encode_only_at={target})` — encode the obs ONLY at the decision the
  search reads; every other prefix decision is **track-only** (`Gen3Player.track_decision`, the
  tracking half of `embed_battle` extracted byte-identical → live obs path unchanged; bit-for-bit
  pinned by the clone-parity fuzz).
- **L4 (this doc)** **warm `SearchSession` reused across battles** — `open_root(record=…)` + the driver
  clears its node cache, so a background worker pays the ~0.68 s Node spawn ONCE, not per search
  (measured 1.2× on a 6-search worker run; the spawn was 17% of each search). `better_line_decision`
  takes an injected `session=` (nullcontext → not closed).

**Planned (deferred levers, ranked):**
- **Per-worker model + mapping cache** (trainee + exact-opponent loaded once) — trivial, the
  `eval_worker` cache pattern.
- **Batch searches by battle** — a worker that searches several turns of ONE loss replays it once
  (the `replay_battle` 0.70 s shared across turns, not per turn).
- **Resume the materializer from the anchor** (the deferred big lever) — checkpoint poke-env tracker
  state at the anchor and feed only the suffix per node, instead of re-tracking the prefix. Highest
  remaining headroom on the 0.90 s per-node materialize, but needs careful poke-env state handling +
  the bit-for-bit parity test as the gate. NOT needed for v1.

---

## 11. Build plan (phased; v1 is the low-risk slice)

- **Phase 0 — selection harness (model-free, cheap).** A `teacher/` candidate-ranker: over recent eval
  traces, `triage`/`scan` the worst-ΔV turns, `falsify`-gate to mistakes, expand to the
  crater±window. Output: a ranked `(trace, turn)` queue. (Reuses the prober engine wholesale.)
- **Phase 1 — VALUE-ONLY teacher (the recommended v1).** Background workers run the search + confirm
  (exact opponent) on the queue, CI-gate, write `(obs, confirmed_value)` to the side buffer; add the
  single MSE aux term + the `grad/searchteacher_*` probe + the `teacher/*` metrics. No policy
  distillation. Validate on `td_resid_tail` + `calibration`. Cannot collapse the policy; directly
  attacks the documented critic weakness; reuses eval-worker + aux-loss infra almost wholesale.
- **Phase 2 — POLICY distillation.** Add the advantage-weighted CE term on the confirmed `A*`, with
  the grad-balance probe watching `_policy_cosine`. Validate on the held-out agreement rate + ELO.
- **Phase 3 — prioritization + budget tuning.** `|ΔV|·P(mistake)` priority, per-cycle budget cap,
  recency aging, the deferred perf levers (§10) if the budget is tight.

**Honesty gate (mirrors every belief-head):** the search *learns* a better line ≠ it *helps* the
policy. v1's success criterion is `td_resid_tail`/calibration improving on a fresh run A/B
(`coef = 0` control), not "the search found corrections." **RETRACTED (2026-07-24): the former
"~⅔ of grind losses are uncoachable team-draw" ceiling is DEAD.** That attribution was CIRCULAR — our
own policy/critic judged recoverability, so "uncoachable" only meant "uncoachable *at this strength*"
— and a strong human beats our bot on the same teams. Treat those losses as HEADROOM, not a ceiling:
there is no known per-turn-uncoachable floor to plan around.
```
