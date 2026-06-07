# Design — Reward fix for policy under-switching

**Status:** design (not implemented). Forward-looking; explicit-only doc.
**Motivates:** the confirmed under-switch pathology on `run_20260606_204351` (@44M).
**Relates to:** `design_reward_annealing.md`, `design_incoming_damage_obs.md`, `project_popart`.

---

## 1. Problem

The policy does not switch enough. Forensic (prober, `run_20260606_204351` @44M): the policy's
**switch-probability mass inverts vs the incoming-KO belief** — 37% switch-mass at P(KO)<0.2, but
only 18% (median 7%) at P(KO)≥0.8. It switches *less* as death becomes *more* certain. ~200 genuine
under-switch mistakes (high P(KO), an alive low-pko pivot existed, it stayed in and our mon died);
structural (plateaus ~80% over 8M→44M, not improving). The `gen3_incoming_damage_v1` belief is in the
obs and the **critic reads it heavily** (value-saliency ~5× the avg dim) — so the model *can see* the
threat; the gap is that the **reward/value signal doesn't make the policy act on it**.

## 2. Current reward — review

The reward (`reward_manager.py`) sums ~30 unweighted `RewardBreakdown` fields into `bd.total`,
returned **raw** through `gen3_env.calc_reward` into the PPO transition — **no clipping / scaling /
VecNormalize**. A ±30 terminal (`VICTORY_VALUE`) dominates a dense shaping band of ≈±0.1..±3.25; the
dense band is where every stay-vs-switch decision is actually decided. `gamma=0.9999`,
`gae_lambda=0.80` → near-undiscounted long return but only a ~5-step effective GAE credit horizon.

**Base spine:** `hp_ours/hp_opp = ±2.0·Δhp_frac`; `faint_ours = −(0.5 + 2.0·hp_before + 0.75)` =
−1.25..−3.25 (asymmetric — carries the 0.75 material penalty); `faint_opp = +0.5..+2.5` (no material
term); `win_loss = ±30`. **Switch stack:** `switch_base +0.5` (flat, spam-gated to 0 back-to-back);
`escape_threat_switch +0.25` and `matchup_penalty −0.15` (the two **state-conditioned** terms);
`se_switch +0.2`; pivot/sleep bonuses. **Taxes:** repetition/bouncing/dead-matchup (escalating with
floors), futile, stall.

**The immediate economics already favor switching** — so "suicidal trades pay" is **falsified**:
`finishing_blow` is suppressed when `we_fainted` ([reward_manager.py:870](../../src/agents/training/reward_manager.py:870)), and a clean death nets ≈−4.5 vs a switch's ≈0.0..+0.5. Dying is correctly punished.

### Why it under-switches (three compounding mechanisms)

1. **Revealed-SE gating (the smoking gun).** The only two state-conditioned switch-tilting terms —
   `escape_threat_switch` ([:416](../../src/agents/training/reward_manager.py:416)) and
   `matchup_penalty` ([:592](../../src/agents/training/reward_manager.py:592)) — both gate on a single
   boolean `_prev_opp_se_threat`, set by `_update_opp_se_threat`
   ([:689](../../src/agents/training/reward_manager.py:689)) which iterates **only revealed** opp moves
   and requires type-effectiveness ≥2×. **This is the exact narrow signal the *old* obs had.** A
   model-free scan of 130 loss traces @44M found **71% of high-P(KO) under-switch mistakes (220/311 at
   P(KO)≥0.5; 156 at P(KO)=1.0) had NO revealed SE opp move** — so for the 71% majority *both* switch
   terms are completely **silent** even though the belief flags the kill. The gate is also blind to
   damage-**magnitude** threats (a neutral/resisted move that still OHKOs via base-power / Choice Band /
   +stages never trips `mult≥2.0`).
2. **A constant subsidy can't un-invert.** `switch_base` is a flat +0.5 at P(KO)=0.05 or 0.95 — it
   shifts the *mean* appeal of switching but structurally cannot change the *relative ordering across
   P(KO)*, which is exactly the inverted curve to fix.
3. **Credit-assignment back-loading + value scale.** `faint_ours` is correctly attributed to the stay
   decision's step, but it is a one-shot ~−2.65 that is a small fraction of the ±30 return the critic
   must also track (the `project_popart` value-swamping problem — no reward normalization), while the
   *benefit* of dodging it by switching is a discounted future non-event. With `gae_lambda=0.80` the
   deferred credit horizon is short.

### Other smells (found en route)
- `explosion` is a hardcoded `+2.0` literal ([:862](../../src/agents/training/reward_manager.py:862)),
  not a named constant — a reward-hack surface (bait/survive self-KOs); should be named.
- Overlapping switch shaping (`switch_base`, `escape_threat_switch`, `se_switch`, pivot/sleep) stacks
  additively to +1.0..+1.3 with no priority — muddies attribution.
- `matchup_penalty` is **flat** −0.15/turn while `dead_matchup_tax` escalates — even the firing path is
  a weak trickle, not mounting pressure.
- `faint_opp` has no material bonus while `faint_ours` carries −0.75 — preservation is only ever a
  *penalty-avoidance*, never a positive: the policy learns "a healthy mon is a faint waiting to happen"
  rather than "an asset to protect."
- No reward normalization; the ±30 terminal dwarfs shaping ~60–100×; only per-batch advantage
  normalization keeps the dense band learnable — fragile if the return scale shifts.

## 3. Recommended design — hybrid (both behind one fresh-run-only flag, default OFF)

> **Hard scope constraint (user, 2026-06-07): the terminal `VICTORY_VALUE = 30` is OUT OF SCOPE —
> do not touch it.** This design does not: PBRS uses the absorbing convention `Φ(terminal) = 0`, so
> the shaping contributes only a **policy-invariant constant** `−Φ(s_0)` to the episode return and
> **never alters the ±30 win/loss reward**. The "no reward normalization / ±30 dwarfs shaping" smell
> in §2 is a **known, deliberately-deferred** future item (annealing / PopArt / normalization) — NOT
> part of this change. Reward annealing in particular is explicitly excluded (it would scale the
> switch shaping toward zero — the wrong direction here).
>
> **Decisions locked (user, 2026-06-07):** implement **both** (A) and (B) together (sub-flags retained
> so a 3-arm A/B is still possible); the re-gate **ORs** the belief with the existing revealed-SE gate
> (never loses the current firing path); the shaping is **coupled to the belief-toggle** (Φ and the
> re-gate read 0 when the incoming belief is ablated, keeping reward and obs consistent). A/B baseline
> (fresh vs resume) is a run-time choice, decided when the run is started.

**(A) Belief re-gate of the existing switch terms (the cheap dark-signal fix).** Snapshot, alongside
`_update_opp_se_threat`, the **active mon's** imminent KO risk from the belief:
`_prev_active_ko_risk = max(phys_pko, spec_pko)·(1 − p_outspeed)` for the active slot. Gate
`escape_threat_switch` and `matchup_penalty` on `_prev_active_ko_risk ≥ SWITCH_RISK_THRESHOLD`
(default 0.5) **instead of / OR-ed with** the revealed-SE boolean. Same bounded ±0.15/+0.25 magnitudes —
this just lights them up for the 71% dark majority. Correctly **active-centric** (it reads the active
slot, the mon actually facing the hit). *Not* policy-invariant (a direct nudge), but small, bounded,
and grounded in a raw fact.

**(B) Potential-based reward shaping over the belief (the credit-assignment bridge).** Add
`pbrs_material = PBRS_GAMMA·Φ(s′) − Φ(s)` (Ng 1999 — policy-invariant; telescopes to
`γ^T·Φ(s_T) − Φ(s_0)`, so it cannot change the optimal policy or be farmed). `PBRS_GAMMA` **must equal
the PPO gamma** (assert at construction + in the round-trip test).

### 3.1 The potential Φ — corrected from the workflow's draft

> **Correction (important).** The workflow's draft Φ = `−W·Σ_alive(hp_i · pko_i·(1−outspeed_i))`
> summed over **all** alive mons is **wrong twice**: (i) the belief computes each mon's P(KO) vs the
> opp active **regardless of whether it's active or benched** (`compute_team_block` loops every
> defender), so a *benched* doomed mon still reads pko≈1 — meaning switching the doomed mon to the
> bench barely moves that sum, so the shaping would **not reward switching** (the one thing it exists
> to do); and (ii) with that sign, a faint *removes* a risk term and *raises* Φ → it would **reward
> fainting**. Both are fixed by making the imminent risk **active-gated** and the potential
> **expected-surviving-material**:

```
Φ(s) = W · [ Σ_{alive i} hp_frac_i ]              # total surviving HP material
       − W · ( hp_frac_active · pko_active · (1 − outspeed_active) )   # minus the ACTIVE mon's expected imminent loss
```
where `W = PBRS_RISK_WEIGHT` (default 2.0), `pko_active = max(phys_pko, spec_pko)` for the active slot,
read from the belief block; benched mons contribute their full HP (they are **not hit this turn**).
`Φ` clamped to `[0, W·TEAM_SIZE]`; `Φ(terminal) = 0`; `self._prev_phi = None` in `reset()` so the first
decision of each episode adds no shaping and episodes never cross-pair.

**Why this works (the three properties):**
- **Rewards switching the doomed mon out.** A full-HP active mon at certain imminent KO is discounted
  to ~0 in Φ(s) (`hp·1 − hp·1·1 = 0`). Switch it to the bench → it's no longer the active mon, its full
  HP now counts → Φ(s′) rises by ≈`W·hp` → `F ≈ +W·hp` *at the switch decision* (the credit-assignment
  bridge). Staying-and-dying: it was already valued ~0, so `F≈0` (no shaping reward) — the base
  `faint_ours` still carries the loss.
- **Never rewards fainting** (kills the critique's failure-ii): a faint removes a *positive* material
  contribution → Φ drops or is flat, never rises.
- **Policy-invariant** (kills failure-i): Φ is a pure function of state `s` (alive set, HPs, who's
  active, the deterministic belief) — no action term — so PBRS telescopes.

### 3.2 Change set (`src/agents/training/reward_manager.py`)
- New constants `SWITCH_SHAPING_ENABLED` (False), `PBRS_RISK_WEIGHT` (2.0), `PBRS_GAMMA` (=PPO gamma),
  `SWITCH_RISK_THRESHOLD` (0.5); new `RewardBreakdown.pbrs_material` field (auto-summed into `total`);
  `self._prev_phi` instance member.
- `_compute_pbrs_phi(self, battle, live) -> float`: read the belief slot block, compute Φ per §3.1.
  **Reuse the already-built obs block** rather than a second `encode_block` if the FPS gate
  (below) shows a regression — thread it from the env (`gen3_env`) into the reward manager.
- In `process_turn_reward` after `live = battle.live_view()`: `phi_next = _compute_pbrs_phi(...)`;
  if enabled and `_prev_phi is not None`: `bd.pbrs_material = PBRS_GAMMA·phi_next − _prev_phi`; then
  `_prev_phi = 0.0 if terminal else phi_next`. In `reset()`: `_prev_phi = None`.
- Re-gate `escape_threat_switch` / `matchup_penalty` on `_prev_active_ko_risk` per §3-(A).

### 3.3 Flag, anti-hacking, retrain
- **Flag:** `--use-switch-shaping` (default OFF, **fresh-run-only + resume-immutable**, recorded in
  `model_config.json` and value-checked by `ModelVersion` like `--vf-coef` / `enabled_beliefs` — a flip
  on resume is FATAL). Sub-flags for the A/B: enable re-gate only vs PBRS only vs both;
  `--pbrs-risk-weight`, `--switch-risk-threshold`.
- **Anti-hacking:** PBRS is policy-invariant by construction; `PBRS_GAMMA==model.gamma` asserted;
  Φ-as-surviving-material (faint never raises Φ); Φ(terminal)=0 + reset-cleared; Φ clamped; the re-gate
  only toggles existing bounded terms (no new unbounded magnitude). **Guard tests:** a
  potential-invariance test (Σ F over a fixed trajectory = `γ^T·Φ_T − Φ_0`); a **faint-monotonicity**
  test (Φ must not increase on an isolated our-faint); a **switch-reward** test (Φ rises when a doomed
  active mon is switched to a safe bench mon — this is the test that catches the all-mon-Φ bug); and a
  re-gate-fires test on a constructed high-P(KO)-no-revealed-SE state.
- **Retrain-class** but **no ARCH bump** (reward values change, not the obs/arch — checkpoints still
  load). A shaped checkpoint's value head isn't comparable to baseline; the flag is fresh-run-only.
  PBRS leaves the optimal policy unchanged — the effect is faster/cleaner switch-curve correction, not
  a new equilibrium.

## 4. A/B + verification
- **FPS gate FIRST:** `_compute_pbrs_phi` is a belief read on the reward path. Run
  `trainer_turn_benchmark.py` before/after; if >~3–5% turn-CPU regression, thread the env's
  already-built block in instead of recomputing.
- **Unit:** the four guard tests above. **Smoke:** `--debug --steps 10000 --use-switch-shaping` →
  round-trip PASSED (new config key), episodes finish, no NaN.
- **A/B (3 arms, matched seeds/steps):** (1) OFF baseline, (2) re-gate only, (3) re-gate + PBRS.
  **Primary metric = the prober switch-prob-vs-P(KO) curve** (the exact diagnostic that found the
  inversion): success = the curve *un-inverts* (switch-mass at P(KO)≥0.8 moves from ~18% toward
  ≥37%) and the ~200 under-switch mistakes drop materially. **Secondary:** `win_rate_vs_bots`/ELO and
  value `explained_variance` non-regression; no rise in bouncing/stall tax (guard against
  over-switching the other way); watch for the reward-hack signature (avg episode reward drifts up
  without ELO moving → revert).

## 5. Runner-up
**Belief re-gate alone** (no PBRS): the cheapest defensible single change — lights up the existing
terms for the 71% dark majority, no new term, no encode-on-reward FPS risk, no invariance proof needed.
Weaker because it's a small bounded nudge that doesn't bridge the deferred-credit gap and isn't
policy-invariant. The right minimal fix if FPS or the PBRS Φ subtlety rules out (B).

## 6. Resolved decisions (user, 2026-06-07)
1. **Belief-toggle coupling:** YES — shaping respects the `enabled_beliefs` mask (Φ/re-gate read 0
   when the incoming belief is ablated); add the `ModelVersion` value-check coupling the two.
2. **FPS vs module boundary:** start with a self-contained recompute behind the FPS gate; thread the
   env's already-built block in only if the benchmark regresses (>~3–5% turn-CPU).
3. **A/B baseline:** deferred to run-time (not an implementation blocker).
4. **Re-gate scope:** OR the belief with the existing revealed-SE gate (safer; never loses the current
   firing path).
5. **Scope of this run:** implement (A)+(B) together; sub-flags retained for the 3-arm A/B.
6. **`VICTORY_VALUE = 30`:** OUT OF SCOPE — unchanged (see the §3 scope constraint).
