# Design — Markovian / PBRS reward redesign + feature-encoder enrichment

**Status:** design (not implemented). Forward-looking; explicit-only doc.
**Goal:** make **every** reward term either **PBRS** (objective-neutral) or **Markovian-w.r.t.-the-
observation** (a clean, obs-keyed bias), and enrich the feature encoder so every Markovian term keys
on something the model can actually observe.
**Coordinates with (does not redo):** `design_reward_switching.md` (shipped belief PBRS + re-gate,
commit `7483dd1`), `design_incoming_damage_obs.md` (the belief block, `gen3_incoming_damage_v2`),
`project_popart` (return normalization).
**Replaces the need for:** `design_reward_annealing.md` for the shaping half (see §6.2).

> **Hard scope constraint (user):** `VICTORY_VALUE = 30` (the ±30 terminal) is **out of scope and
> untouched.** Both PBRS potentials use the absorbing convention `Φ(terminal) = 0`, so shaping
> contributes only a policy-invariant per-episode constant `−Φ(s_0)` and never alters the ±30
> win/loss reward.

This doc was pressure-tested by an adversarial review pass before being written; §9 is the findings
ledger (what broke in the draft and how it was fixed). The most consequential corrections are folded
inline and flagged **[RT-n]**.

---

## 1. Conceptual frame — the two axes and the rules

Every `RewardBreakdown` field (`reward_manager.py:41–95`) is classified on **two orthogonal axes**:

- **Axis 1 — Objective-neutral (PBRS) vs Objective-changing (a bias).** A potential-based shaping
  `F(s→s′) = γ·Φ(s′) − Φ(s)` telescopes to `γ^T·Φ(s_T) − Φ(s_0)` — a path-independent constant — so
  it **cannot change the optimal policy** (Ng et al. 1999). Everything that adds/subtracts at a single
  step and does not telescope is a *bias*: it deliberately changes the objective.
- **Axis 2 — Markovian-w.r.t.-the-OBSERVATION vs history-dependent.** "Markovian" here is relative to
  the **agent's obs**, not the true game state. A term is Markovian-w.r.t.-obs iff its trigger is a
  deterministic function of `(obs_t, action_t)` (the current board + the in-obs history + the action),
  so `V(s)` can predict its expectation with **no irreducible variance** and the policy can learn the
  contingency state-conditioned. A term that keys on a **manager-internal cross-turn counter not
  exposed in the obs** (`self._consecutive_*`, `self._prev_*`, `self._last_*`) is history-dependent
  **even if** that counter is mechanically computable — because the network cannot see it.

**The four rules this design applies:**

1. **PBRS for terms that should NOT bias the objective** (material / win-margin: `hp_ours`, `hp_opp`,
   `faint_ours`-HP-part, `faint_opp`). Reformulate as a state potential `Φ_mat`; `Φ_mat(terminal)=0`
   so the episode return is purely the ±30 terminal ("a win is a win"). `PBRS_GAMMA` **must** equal
   the PPO `gamma` (§7.3).
2. **A term MEANT to bias the policy** (anti-spam, status/hazard/switch nudges) stays
   additive/objective-changing — do **not** PBRS it (that deletes its effect). Instead make it
   **Markovian**: a deterministic function of the obs, never of hidden internal history.
3. **Encode the counter the reward keys on, at the reward's granularity.** Don't hand the model a raw
   10-turn window and hope it re-derives a counter — give it the scalar. Reducible-from-the-window ≠
   clean-from-step-1.
4. **Provide facts, let it learn.** New obs features are raw known facts/counters, not baked strategy.

**Granularity note [RT-1].** The reward fires **once per decision window** (`calc_reward` →
`process_turn_reward`, `env.py:600`), **not** once per game-turn. A faint/phaze/forced-switch splits
one game-turn into ≥2 decision windows, each its own `process_turn_reward` call. All telescoping math,
density tables, and the offline falsifier below are indexed over **decision windows** (= the rows in
`states.npz`/`summary.json`), not `|turn|N` boundaries. Telescoping is granularity-agnostic, so the
conclusions hold — but an implementer who indexes by game-turn will get spurious "telescoping broken"
failures on every faint turn.

### 1.1 The reward registry — three classes, one fold loop

The four rules are operationalized through a **single registry that is the source of truth** for every
reward component. The manager folds the per-turn reward by **iterating the registry and applying the
per-class treatment** — there is no per-term special-casing in the fold loop; a term's **class** + its
declared computation drive everything. The three classes:

| class | what it is | shaping | flag-affected |
|---|---|---|---|
| **TERMINAL** | the ±30 win/loss | never shaped/blended | **no** — the only true objective-changing reward; untouchable by any flag |
| **PBRS** | pure potential hints, **always telescoping**, **always objective-neutral** (`Φ(terminal)=0`) regardless of any flag | `γ·Φ(s′) − Φ(s)` | **no** |
| **BIAS** | soft, intended-but-optional shaping whose additive-vs-telescoping mix is set by `--bias-additivity` (§1.2) | accumulate-and-refund (§1.2) | **yes** |

**Registry entry** = `{ name, class, compute }`:
- **TERMINAL / PBRS** declare a **state potential `Φ(s)` computable from the OBS with `Φ(terminal)=0`**
  (TERMINAL is the degenerate `Φ≡0` plus the ±30 emission).
- **BIAS** declares its **current additive per-turn value**, plus — *only if it will be swept below
  additivity 1.0* — a **state potential (its running accumulator)** so the telescoping refund stays
  low-variance (§1.2).

`RewardBreakdown` and the prober telemetry are **derived from the registry** (each entry contributes its
named component), so the per-component breakdown still resolves and stays in lockstep with the fold.
**Adding a reward = one registry entry** (declare name + class + compute); nothing else changes.

**PBRS members:** `Φ_mat` (material/HP/faint/margin — §2), and the already-shipped switch
risk-potential (`pbrs_belief`, `design_reward_switching.md`). `Φ_hazard`/`Φ_status` are the
*telescoping forms* of the BIAS spikes/status terms, reached by the flag (§1.2, §2.6–2.7) — they are
registered in **BIAS**, not PBRS, so the default run keeps them additive (= today).
**BIAS members:** the anti-no-progress clock (§4) and every other soft shaping (the `futile_*` family,
the switch nudges, `roar`, `spikes`/`status` credit, `stall_tax`, …).

### 1.2 The bias-additivity flag

`--bias-additivity` (config key `bias_additivity`), float `[0,1]`, **default `1.0`**. It affects **only
the BIAS class** (PBRS and TERMINAL ignore it). It is the per-run knob for *how much shaping is allowed
to change the objective* — from a full additive bias to a pure telescoping hint:

- **`1.0` = fully additive.** Implemented via **accumulate-and-refund**: each BIAS term emits exactly its
  **current per-turn value** and refunds **nothing** → **byte-identical to today's bias rewards.**
- **`0.0` = fully telescoping** (pure PBRS hint): refund `−acc` at the terminal → episode contribution `0`.
- **`λ` in between:** refund `−(1−λ)·acc` → episode contribution `= λ·acc`, where `acc = Σ_t b_t` is the
  term's accumulated per-turn value.

```
episode contribution of a BIAS term = Σ_t b_t  +  refund   where refund = −(1−λ)·acc
                                     = acc − (1−λ)·acc      = λ·acc
```

**Why accumulate-refund, not per-turn scaling (`λ·b_t`).** Per-turn scaling changes every per-turn value,
so `λ=1` is only a *numerical* no-op; accumulate-refund leaves **every per-turn value untouched** and only
adds a terminal-side refund, so `λ=1` is a **structural, byte-exact no-op** — provable by the no-op
equivalence test (§7) — and the bias's **credit-assignment timing is preserved** (it still lands at the
turn it's earned); only the *net* objective bias is dialed to `λ·acc`. This is the exact property
event-style terms (a one-time `+0.5` spikes bonus) need.

**Low-variance refund (the `Φ` the registry asks BIAS terms to declare).** A `−(1−λ)·acc` *lump at the
terminal* is high-variance (the value head would have to predict the whole episode's accumulated bias).
Instead deliver it as a telescoping shaping over the term's **running accumulator** `Φ_acc(s_t) = acc_t`
(readable from the obs / a registry-owned counter), **without** the absorbing `Φ(terminal)=0` zeroing —
so the difference `−(1−λ)·(γ·Φ_acc(s_{t+1}) − Φ_acc(s_t)) ≈ −(1−λ)·b_{t+1}` spreads the refund per-turn
and the critic can predict it. At `λ=1` no accumulator/refund is active at all (zero extra terms) — the
no-op is total.

**Resume-immutable + fresh-run-only**, recorded in `model_config.json`, value-checked by `ModelVersion`
(same machinery as `--vf-coef` / the switch-shaping flag). It is a **per-run constant — NOT annealed
within a run** (§6.2 contrasts this with reward annealing).

### 1.3 The single-variable default run (call this out)

> **At the defaults (`bias_additivity = 1.0`), the BIAS class behaves exactly as today (additive,
> byte-identical) and MATERIAL telescopes (`Φ_mat`).** So the **only reward-behavior change vs the
> current run is the material-PBRS clutch-fix** — material no longer banks the lead, so every win
> returns `+30` and every loss `−30`. This is a **deliberate, always-on change** (NOT gated behind a
> flag — we have decided to just fix it), and it is a **clean single-variable change**, ideal for
> attribution against the run we are keeping alive.

**Material's default behavior is therefore NOT today's, by design.** Everything else stays put at the
defaults. **The bias redesign is staged:** the no-progress clock (§4), the obs-keyed reframes (§3), and
the spikes/status de-bias (§2.6–2.7) are all **registered BIAS entries enabled in subsequent A/B arms**,
not the first attribution run — so the first run isolates the material clutch-fix. The two later levers
are orthogonal: **(a) lowering `bias_additivity`** sweeps the *existing* biases from additive toward
telescoping; **(b) swapping in the redesigned BIAS entries** (clock replacing the anti-spam family, the
reframes) is a separate enable. Both ride the registry; neither touches PBRS or the ±30 terminal.

> **Staging caveat to confirm:** if instead you want the *first* run to ship the full redesigned BIAS set
> (clock + reframes, additive), that run is no longer strictly single-variable — material *and* the
> bias-structure change together. The recommendation here is the strict single-variable run (material
> only) first, then the bias arms; flag if you'd rather bundle.

---

## 2. The material PBRS potential `Φ_mat`

### 2.1 The bias being removed

The current `base` spine (`process_turn_reward:907–918`) applies **unconditionally** every window:

```
bd.hp_ours    =  HP_VALUE · Σ our_hp_delta          # ±2.0 · Δhp_frac
bd.hp_opp     = −HP_VALUE · Σ opp_hp_delta          # ±2.0 · Δhp_frac
bd.faint_ours = −(FAINT_BASE + FAINT_HP_SCALE·hp_before + FAINT_MATERIAL_PENALTY)   if we_fainted
              = −(0.5 + 2.0·hp_before + 0.75)  ∈ [−1.25, −3.25]
bd.faint_opp  =  (FAINT_BASE + FAINT_HP_SCALE·hp_before)                            if opp_fainted
              =  (0.5 + 2.0·hp_before)         ∈ [+0.5, +2.5]
```

These are a **raw running tally** of material change. Summed over an episode they do **not** telescope —
they accumulate path-dependently, so two trajectories that both end in a win return *different* totals.
Measured: a clutch win returns ≈ **+26** while a dominant win returns ≈ **+47**, and a faint costs
≈ **−5.82 even in won games**. The argmax is therefore tilted toward clean/dominant wins over clutch
wins — wrong, because **a win is a win** (the only objective truth is `win_loss = ±30`).

### 2.2 The definition — material potential over the *declared* teams **[RT-5][RT-6]**

```
Φ_mat(s) = MAT_HP_WEIGHT   · ( Σ_{i ∈ ours}  hp_frac_i  −  Σ_{j ∈ opp}  hp_frac_j )
         + MAT_ALIVE_WEIGHT · ( n_alive_ours  −  n_alive_opp )

  MAT_HP_WEIGHT    = HP_VALUE = 2.0
  MAT_ALIVE_WEIGHT = FAINT_BASE + FAINT_MATERIAL_PENALTY = 1.25   (see §2.4 for the derivation)

Φ_mat(terminal) = 0                                       # absorbing convention
```

**Both sides are summed over the DECLARED team size, not the revealed set.** This is the single most
important structural decision in §2 and it resolves two real defects the draft missed:

- **Reveal discontinuity [RT-5].** `live.opp.mons` is *revealed mons only* (`live_view.py:204`), and
  opp HP is `%`-based. A naive sum over revealed mons would **jump by ≈+1.0** every time the opponent
  reveals a new mon (it enters the sum at full HP) — injecting a spurious `F_mat ≈ −2.0` the agent
  cannot cause. Computing the opp side over `live.opp.team_size` (`live_view.py:202`) and treating
  **unrevealed opp mons as full-HP-alive** removes the jump entirely: the opp sum starts at a constant
  baseline (`6.0`) and only ever *decreases*.
- **Start-state variance [RT-6].** With the declared-team formulation, at `s_0` both sides are 6 mons
  at full HP and 6 alive: `Φ_mat(s_0) = 2·(6−6) + 1.25·(6−6) ≈ 0`. The per-episode constant `−Φ_mat(s_0)`
  is therefore ≈ 0 with **near-zero cross-episode variance**, instead of swinging in `~[+9,+18]` with
  the team and the reveal schedule. This matters because the constant lands in the **value-loss target**
  (§2.3), and this run is already fighting value-swamps-the-trunk (`project_popart`, `grad_balance.py`).

Concretely: `Σ_{opp} hp_frac_j = Σ_{revealed} hp_frac + (team_size − revealed_count) · 1.0`, and
`n_alive_opp = team_size − (revealed fainted count)`. Our side is fully known (6 mons). `hp_frac_i` is
the LiveView `LivePokemon.hp_fraction` already read by `_belief_potential_and_risk` (`reward_manager.py:760`);
the alive counts are the `report_episode` form (`:1063–1064`). **All reads come from the single
`live = battle.live_view()` already built at `:904`** — no second `encode_block`, cheaper than the
shipped belief Φ.

Bound: `|Φ_mat| ≤ 2·6 + 1.25·6 = 19.5`. Mirror the shipped Φ's defensive clamp `Φ_mat = clip(Φ_mat,
−19.5, +19.5)`, but the synthetic-trajectory test (§7.1) must keep the clamp inert (a clamped step is
not a pure potential difference and breaks telescoping).

### 2.3 Telescoping (over decision windows) and the terminal walkthrough **[RT-1][RT-2]**

The shaping is `F_mat,t = PBRS_GAMMA · Φ_mat(s_{t+1}) − Φ_mat(s_t)`, `t` over decision windows. Summed
discounted:

```
Σ_t γ^t · F_mat,t  =  γ^T · Φ_mat(s_T)  −  Φ_mat(s_0)   =   −Φ_mat(s_0)     (since Φ_mat(s_T)=0)
```

a single per-episode constant. It is the same for **every** trajectory from a fixed `s_0` (a fixed
team), so it **cannot change the argmax** — it only reshapes *where* the dense credit lands. With the
declared-team formulation `Φ_mat(s_0) ≈ 0`, so the constant is ≈ 0 and varies negligibly across teams.

**Density is preserved** (the whole point — PPO still gets per-window credit over 30+ window games).
With `γ = 0.9999 ≈ 1`, `F_mat ≈ ΔΦ_mat`:

| current term | current value | `ΔΦ_mat` | match |
|---|---|---|---|
| our chip `δ` HP-frac | `−2.0·δ` | `−2.0·δ` | identical |
| opp chip `δ` dealt | `+2.0·δ` | `+2.0·δ` | identical |
| our mon faints from `hp_before` | killing-blow `hp_ours` (`−2·hp_before`) + `faint_ours` non-HP part | `−2·hp_before` (HP term) `− MAT_ALIVE_WEIGHT` (alive flips) | see §2.4 |
| opp mon KO'd from `hp_before` | `+0.5 + 2·hp_before` | `+2·hp_before + MAT_ALIVE_WEIGHT` | see §2.4 |

**Terminal window — the highest-risk silent bug [RT-2].** When our move KOs the opp's last mon, that
single decision window's `process_turn_reward`: (a) sets `win_loss = +30`; (b) the post-kill board can
have `Φ_mat(s′) ≈ +19.5` (a 6-0 with our full team alive). The PBRS term **must** zero `Φ_mat(s′)` on
the terminal window — exactly as the shipped belief PBRS already does at `:1006–1010` via
`(0.0 if is_terminal else phi_next)` — so that

```
F_mat(terminal) = PBRS_GAMMA · 0 − _prev_phi_mat  =  −_prev_phi_mat     (a small negative)
```

If an implementer drops the `is_terminal→0` gate (an easy slip — "the mon is alive, a non-zero
`phi_next` feels natural"), the winning window adds `≈ +19.5 − _prev_phi_mat` instead of `−_prev_phi_mat`,
injecting a path-dependent **+19.5 dominant-win bonus** — re-creating the exact bias this design removes,
**larger than the original**. This must be a walked-through proof and **the first guard test written**
(§7.1). (On a forced-switch/post-faint window the incoming block is all-zeros so `Φ_belief` reads
0-risk, but `Φ_mat` reads only HP/alive and is unaffected — the one place `Φ_mat` is *more* robust than
`Φ_belief`.)

### 2.4 Removing `FAINT_MATERIAL_PENALTY`, and `MAT_ALIVE_WEIGHT` **[RT-3][RT-9]**

The current `−0.75 FAINT_MATERIAL_PENALTY` is a deliberate "preserve mons / counter the 6-0 dynamic"
bias. **It is removed.** Preservation is only *instrumentally* good: a mon is worth keeping because it
helps you win, and the ±30 terminal + the dense `Φ_mat` material signal already encode that. A standalone
preservation *bias* is objective-changing and not needed — and measurably harmful (it priced a *necessary
clutch sacrifice* the same as a careless one, which is part of the clutch-vs-dominant inversion).

**The per-mon-alive term restores the discrete-faint density without the bias.** Pure HP-fraction
shaping prices a faint as continuous with chip (a mon fainting from 1% HP moves `Σ hp` by only −0.01).
But losing the *6th* of a Pokémon is a discrete strategic event (the win condition is mons-to-zero).
`MAT_ALIVE_WEIGHT·Δn_alive` reproduces the discrete step `FAINT_BASE` gave — as part of a **potential**,
so it telescopes and cannot bias the optimum.

**Why removal is safe [RT-3].** Do **not** argue "the +30 propagates back to the sacrifice" — under
`gae_lambda = 0.80` the effective credit horizon is ~5 windows, so the terminal reaches a 30-window-prior
sacrifice only through the *bootstrapped* `V(s)` that is itself under training. The correct argument:
**faint pain is preserved because the alive-term drop is charged IMMEDIATELY at the faint window via
`F_mat`** (zero horizon needed), exactly as the old `−0.75` was. The honest accounting:

```
OLD immediate full-HP faint pain  =  killing-blow hp_ours (−2·hp_before) + faint_ours non-HP (−1.25)
                                  ≈  −2·hp_before − 1.25   →  −3.25 at hp_before=1.0 ... (with the −0.5 base, ≈ −5.25 total per the killing window)
NEW immediate full-HP faint pain  =  −2·hp_before − MAT_ALIVE_WEIGHT
                                  ≈  −2·hp_before − 1.25   →  −3.25 at hp_before=1.0
```

Setting `MAT_ALIVE_WEIGHT = FAINT_BASE + FAINT_MATERIAL_PENALTY = 1.25` matches the old *non-HP*
immediate faint magnitude (the part not already carried by the HP term), from a **stated invariant**,
not a "midpoint feel." This is a deliberate, modest reduction in the very-flat-penalty regime; the
false-negative guard (§7.3: faint-rate / mons-lost-per-game must not rise) is pre-registered to catch
any real loss of faint-aversion.

**The symmetric opp-alive term is a credit-density change, NOT an aggression bias.** Making the alive
term symmetric raises the *immediate* opp-KO credit by `+MAT_ALIVE_WEIGHT − FAINT_BASE = +0.75` vs the
old `faint_opp`. Because it is part of a **potential**, it **cannot** bias the optimum toward aggression
(it telescopes); it only makes KO/faint credit land more immediately. Still pre-register a
reckless-trade guard (§7.4) to catch impl bugs / value-head effects.

### 2.5 `explosion` / `explosion_block` / `finishing_blow` re-derivation **[RT-4]**

Once HP/faint become PBRS, re-check the Explosion-button exploit. A healthy 1-for-1 mutual KO under
`Φ_mat`: `ΔΦ_mat = 2.0·((−1)−(−1)) + 1.25·((−1)−(−1)) = 0` — materially neutral, **correct**. So the
**`finishing_blow` `we_fainted` guard (`:879`) MUST stay** — without it the `+0.5` tips a materially-
neutral mutual KO positive, re-teaching Explosion-as-a-free-KO-button.

- `finishing_blow` (`+0.5`): **keep** as a per-`(s,a)` Markovian bias keyed on `(opp_fainted ∧ ¬we_fainted
  ∧ move.base_power>0)` — all this-window outcomes. Guard verbatim.
- `explosion` (`+2.0` literal): the survive-the-Explosion credit is **subsumed by `Φ_mat`** (opp lost a
  mon, we lost nothing → `F_mat ∈ [+1, +3]`). **Delete ONLY the `bd.explosion = 2.0` assignment**
  (`reward_manager.py:931` + its comment). **Keep the enclosing `if not delta.we_fainted:` block and the
  nested `explosion_block`** (`:928–937`) — deleting the whole branch silently loses `explosion_block`.
- `explosion_block` (`+1.0`): **keep** as a small named per-`(s,a)` bias (no-selling the Explosion via
  type-immunity / Protect / 0 damage — observable from the matchup block). Rename the literal
  (`EXPLOSION_SURVIVE_BONUS` is now unused; keep `EXPLOSION_BLOCK_BONUS`).

### 2.6 Hazard control — a BIAS term whose telescoping form is `Φ_hazard` (the `spikes` split)

The `spikes` term mixes two things that belong in different classes, and the **layer-added credit is
material-like, not a strategic prior** — it should be PBRS, not a bias.

- **`SPIKES_LAYER_BONUS = +0.5/layer` is, by its own comment, a "credit assignment bridge"** — the
  value of a layer is realized diffusely (every opp switch-in takes entry chip many turns later). That
  is the textbook PBRS use case. Crucially, as a *bias* it **double-counts**: it pays `+0.5` for setting
  the layer **and** the realized chip is *already* rewarded by `hp_opp` → `Φ_mat`. Worked example —
  3 layers set, 4 switch-ins at 25%: the current reward pays `0.5·3 (set) + 2.0·0.25·4 (chip) = +3.5`
  for spikes; the realized value is only the `+2.0` chip. The extra `+1.5` is the objective bias.

  **The telescoping form (reached via `bias_additivity → 0`, §1.2):** the layer-credit's running
  accumulator is the potential `Φ_hazard(s) = HAZARD_WEIGHT · (opp_spike_layers − our_spike_layers)`,
  `HAZARD_WEIGHT = SPIKES_LAYER_BONUS = 0.5`. At `λ = 1` (**default**) the term is the **additive
  `+0.5/layer` of today** (byte-identical); as `λ → 0` it telescopes to `−Φ_hazard(s_0) = 0` →
  contributes **zero to the episode return**, the spikes value flowing **solely** through `Φ_mat` (the
  chip) with **no double-count**. Rapid Spin is handled correctly at `λ<1`: spinning a layer drops
  `Φ_hazard` → a negative payback that cancels the `+w`, so a layer that never chips anyone nets zero.
  **Registry placement: `spikes` is a BIAS entry** (class = BIAS), declaring its additive per-turn value
  **and** `Φ_hazard` as its accumulator-potential for the low-variance refund (§1.2). It is **not** in
  the always-on PBRS `Φ_total` (§6.1); the `bias_additivity` knob is exactly the dial for the
  double-count-vs-exploration judgment this section raised — default keeps today's additive bias for the
  single-variable run; lower it to test the telescoping no-double-count form.

- **The `SPIKES_WASTE_PENALTY = −0.2` (using Spikes at 3 layers) stays a Markovian bias** — a genuine
  objective-changing nudge keyed on `(move is_hazard ∧ opp_spike_layers == 3)`, both observable at
  decision time (`is_hazard` in the move-effect block, layer count in global-env). It becomes a clean
  per-`(s,a)` `futile_spikes` penalty in the futile family (§3.1). `_prev_opp_spikes` disappears from
  the bias path entirely (the PBRS uses `_prev_phi_hazard`; the progress clock's "layer added" check and
  the futile-waste both read the current layer count from `live`).

**Why this departs from the task's literal "don't PBRS hazards" rule.** Rule 2 ("a hazard nudge must
stay additive — PBRS deletes the effect") is right for terms whose *whole* value is the bias, but it
fails for the **layer-added credit**: PBRS *preserves* the bridge (`+w` still lands at the setup turn)
and deletes only the over-valuation — the effect the rule feared losing is intact. The rule *does* hold
for the **futile-waste**, which is exactly the piece kept as a bias. Caveat (state it): PBRS gives no
*net* incentive to set hazards — it reshapes credit for the realized-chip incentive (`Φ_mat`) to land
at the setup decision. For a competent model experiencing spikes chip constantly this is correct; the
progress clock also treats a new layer as a progress event (a small Markovian nudge to set hazards),
which covers the cold-start exploration gap. **Generalization → §2.7:** the same logic is applied to
`status` (#29) — its `±0.3` is likewise a diffuse-value bridge that double-counts the Toxic/burn chip
`Φ_mat` already rewards.

### 2.7 Status — a BIAS term whose telescoping form is `Φ_status` (the `status` split)

The same logic as §2.6 applies to `status` — its value is eventually realized materially, so the
inflict/receive credit is a bridge whose double-count is dialed by `bias_additivity`, and only the
*wasted application* is a genuine always-additive bias.

The credit's accumulator-potential is `Φ_status(s) = STATUS_WEIGHT · (opp_statused − our_statused)`
(count of non-fainted statused mons per side). At `λ = 1` (**default**) the term is today's **additive
`±0.3`** (byte-identical); as `λ → 0` it telescopes to `−Φ_status(s_0) = 0` → **zero net return
contribution**, no double-count with the Toxic/poison/burn chip `Φ_mat` already rewards. The current
term's symmetry falls out naturally: us getting statused drops `Φ_status` (negative); **us curing our
own status** (Rest/Lum/Aromatherapy) raises it (positive); the opponent curing theirs drops it — the
exact analogue of Rapid Spin for hazards. **Registry placement: a BIAS entry** declaring the additive
value + `Φ_status` as its accumulator; **not** in the always-on PBRS `Φ_total` (§6.1). `status_wasted`
(#14) is the Markovian futile piece (parallel to `futile_spikes`). **Weight is uniform over status
types** — do NOT weight by type (that bakes strategy); `Φ_mat` already differentiates the realized value
(Toxic-on-a-wall produces far more chip than Thunder-Wave-on-a-dying-mon).

**The sharper caveat (status is heterogeneous — be explicit).** Status value is only *partly* material:
Toxic/poison/burn are damaging (realized in `Φ_mat` → clean bridge, like spikes), but paralysis / sleep
/ freeze are **non-damaging tempo** ("the opponent loses turns"), whose value is only *diffusely and
stochastically* material (a missed turn → damage we wouldn't have dealt → `Φ_mat`, eventually). PBRS is
still **valid** (policy-invariant for any potential — it cannot move the optimum), but the realized
signal it bridges for tempo-status is noisier than for damaging status. The old uniform `+0.3`
*guaranteed* a standing incentive to status; `Φ_status` removes the net incentive and relies on the
agent learning the diffuse payoff. Worst case is not a wrong optimum — it is *slower learning* of
tempo-status value (the agent under-using Thunder Wave / sleep until it experiences the downstream).
**Hedge (pre-registered guard, §7.4):** *status-application rate must not collapse* vs the baseline; if
it does, the tempo signal was too weak — restore a small standing bias for the **non-damaging** statuses
specifically (the same shape as the `FAINT_MATERIAL_PENALTY`-removal faint-rate guard). This is the only
part of the status conversion that is judgment, not mechanics.

**The dividing line, restated in registry terms:** value eventually realized materially (HP / win) wants
a **telescoping (potential) treatment** — *always-on* for `Φ_mat` / `Φ_belief` (the **PBRS class**), and
*flag-dialed* for `spikes` / `status` (**BIAS class**, telescoping form `Φ_hazard` / `Φ_status` reached
as `bias_additivity → 0`); value that is *only* the immediate nudge stays an **additive bias** (the
no-progress clock, the `futile_*` family, the small switch tilts). The reason `spikes`/`status` land in
BIAS rather than PBRS is the **exploration caveat** (a telescoping potential gives no *net* incentive to
set hazards / inflict status — only a credit reshuffle): the `bias_additivity` knob lets a run choose,
per the §7.4 guard, between today's additive incentive (`λ=1`, default) and the no-double-count
telescoping form (`λ→0`). `Φ_mat`/`Φ_belief` are *unconditionally* telescoping because their material is
realized so reliably that no exploration incentive is needed.

---

## 3. The registry — class-tagged term table (deliverable A)

`RewardBreakdown` has **31 fields**. This is the **registry's class-tagged listing** (§1.1): each field's
**registry class** (TERMINAL / PBRS / BIAS) drives its fold treatment, alongside its Axis-2
(Markovian-vs-history) verdict, its key, whether that key is in the obs today, and its reformulation. The
**"class" column is the registry class**; "Markovian-bias" = a BIAS term whose trigger is (or is made) a
deterministic function of `(obs, action)`. Class coverage is exhaustive and non-overlapping (every field
→ exactly one class — the registry-coverage test, §7) — **PBRS:** #1–4 (→ `Φ_mat`), #31 (`pbrs_belief`);
**TERMINAL:** #5; **BIAS:** all the rest. The BIAS rows respond to `--bias-additivity` (§1.2); at the
default `1.0` they are byte-identical to today. The spikes/status "→ PBRS" reformulations (#15, #29) are
the **`bias_additivity → 0` telescoping form** of those BIAS terms (§2.6–2.7), *not* a move into the
PBRS class.

| # | field | trigger (fn:line) | magnitude | class (registry) | Axis-2 (as written) | key | key in obs today? | reformulation |
|---|---|---|---|---|---|---|---|---|
| 1 | `hp_ours` | `:907` | `±2·Δhp` | **PBRS** (material) | Markovian | our HP frac | **yes** (own team, `POKEMON_HP_OFFSET=67`) | → **Φ_mat** |
| 2 | `hp_opp` | `:908` | `±2·Δhp` | **PBRS** | Markovian | opp HP frac | **yes** (opp team) | → **Φ_mat** |
| 3 | `faint_ours` | `:909` | −1.25..−3.25 | **Mixed**: HP part PBRS, **0.75 flat = bias (REMOVED)** | Markovian | our mon faints + `hp_before` | **yes** | HP part → **Φ_mat** + alive term; the −0.75 bias **deleted** (§2.4) |
| 4 | `faint_opp` | `:911` | +0.5..+2.5 | **PBRS** | Markovian | opp faints + `hp_before` | **yes** | → **Φ_mat** (symmetric alive term carries the discrete KO) |
| 5 | `win_loss` | `:912–916` | ±30 | **Terminal truth** | Markovian (terminal flag) | won/lost/finished | n/a (absorbing return) | **UNCHANGED — out of scope.** `Φ(terminal)=0` for both potentials |
| 6 | `explosion` | `:928–931` | `+2.0` | **Bias** | Markovian | opp self-KO & we survived | partial (belief Explosion flag + history) | **subsumed by Φ_mat** — delete the literal only (§2.5) |
| 7 | `explosion_block` | `:932–935` | `1.0` | **Bias** | Markovian | we no-sold the Explosion | partial (matchup immunity) | **keep**, named, per-`(s,a)` |
| 8 | `finishing_blow` | `:867–890` | `0.5` | **Bias** | Markovian | KO w/ damaging move & lived | **yes** | **keep** — `we_fainted` guard verbatim (§2.5) |
| 9 | `roar` | `:457–466` | `±0.2` | **Bias** | **History** (`_prev_opp_boosts`) | Roar forced switch w/ spikes/+boosts | partial (current boosts/spikes in obs; *prev* snapshot not) | **reframe** to current-obs opp-boosts + opp-spikes at the Roar decision; drop `_prev_opp_boosts` |
| 10 | `futile_attack` | `:784–813` | −0.05 / −0.5 | **Bias** | Markovian | attack did no net dmg / immune | partial→yes (matchup 0× + transition) | **keep** as the per-`(s,a)` futile no-op discriminator (§4) |
| 11 | `futile_setup` | `:815–824` | −0.3 | **Bias** | Markovian | setup at ±6 cap | **yes** (boosts block + `is_boost`) | **keep** per-`(s,a)` |
| 12 | `setup_low_hp` | `:826–835` | −0.10 | **Bias** | Markovian | setup below 40% HP | **yes** | **keep** per-`(s,a)` |
| 13 | `boost_utilized` | `:849–865` | `0.03·b·dmg` | **Bias** | Markovian | attacked w/ +stages | **yes** | **keep** per-`(s,a)` (small; partly overlaps Φ_mat damage credit) |
| 14 | `status_wasted` | `:837–847` | −0.3 | **Bias** | Markovian | status move whiffed | **yes** (`status_will_land` in move-effect block) | **keep** per-`(s,a)` |
| 15 | `spikes` | `:773–782` | +0.5/layer, −0.2 | **BIAS** (layer-credit telescopes at `bias_additivity→0`) | layer-credit was History (`_prev_opp_spikes`); waste is Markovian | layer added / wasted at 3 | partial (current count in global env) | **BIAS, default additive = today (§2.6).** Layer-credit's accumulator is `Φ_hazard` (the `λ→0` telescoping form → no double-count with Φ_mat's chip); the −0.2 waste → Markovian `futile_spikes` keyed on `(is_hazard ∧ opp_layers==3)`. Drop `_prev_opp_spikes` |
| 16 | `matchup_penalty` | `:626–634` | −0.15 flat | **Bias** | **Markovian (post shipped re-gate)** | stayed in vs threat | **yes** (belief block) | **keep** — shipped re-gate; escalation, if wanted, rides `turns_since_progress` |
| 17 | `dead_matchup_tax` | `:636–678` | −0.10·n | **Bias** | **History** (escalation `_consecutive_dead_matchup_stays`) | 0×-only stay, N turns | partial (0× from matchups; count not) | **firing condition kept** as flat futile bias; **escalation → progress clock** (§4) |
| 18 | `switch_base` | `:447–448` | `0.5·spam_mult` | **Bias** | **History** (spam-gate `last_switch_turn`) | voluntary switch, not spam | partial | **drop the spam-gate** → clean per-action bias **[RT-R3]**; spam handled by the clock |
| 19 | `switch_bouncing_tax` | `:439–445` | −0.15·n | **Bias** | **History** (`_last_switched_from`, `_consecutive_bounces`) | A↔B oscillation | **no** | **subsumed by the progress clock** (§4); add a bounce feature only if A/B shows a residual exploit |
| 20 | `repetition_tax` | `:350–356` | −0.03/−0.15·n | **Bias** | **History** (`_last_action_idx`, `_consecutive_attack_repeats`, `_last_attack_had_effect`) | same move repeated | partial/no | **fully subsumed**: 86% no-op part → progress clock; **14% productive-repeat part eliminated** (§4) |
| 21 | `struggle_tax` | `:360–365` | −0.5 | **Bias** | **History** (`_consecutive_struggle`) | struggle loop ≥3 | partial (`forced_struggle` bit in obs; count not) | **subsumed by the progress clock**; single-turn struggle is forced & unpenalized |
| 22 | `pivot_protect` | `:531–559` | `0.10` | **Bias** | Markovian | switched while opp Protected | partial (transition) | **keep** per-`(s,a)` |
| 23 | `pivot_status` | `:561–568` | `0.10` | **Bias** | Markovian | switched into immune status | partial (switch-in types in obs) | **keep** per-`(s,a)` |
| 24 | `pivot_damage` | `:570–602` | +0.10/+0.15 | **Bias** | Markovian | switch improved the matchup | **yes** (matchup block) | **keep** — cleanest Markovian switch-quality term |
| 25 | `se_switch` | `:468–509` | `0.2` | **Bias** | **History** (`_last_opp_seen_by` gate) | switch-in has SE move (once/matchup) | partial (SE-fact in matchups; gate not) | **drop the once-per-matchup gate** → fold the offensive-threat tilt into `pivot_damage`, or keep flat per-`(s,a)` |
| 26 | `escape_threat_switch` | `:450–451` | `0.25` | **Bias** | **Markovian (post shipped re-gate)** | switched out while threatened | **yes** (belief block) | **keep** — the canonical obs-keyed switch nudge |
| 27 | `sleep_out` | `:604–613` | `+0.25` | **Bias** | Markovian | rotated a sleeping mon to bench | **yes** (status one-hot) | **keep** per-`(s,a)` |
| 28 | `sleep_in` | `:615–624` | `−0.25` | **Bias** | Markovian | sent in a sleeping mon | **yes** | **keep** per-`(s,a)` |
| 29 | `status` | `:511–519` | `±0.3` | **BIAS** (inflict-credit telescopes at `bias_additivity→0`) | inflict-credit was History (`_prev_*_statused`); waste is Markovian | status gained/lost | partial (current counts + transition events in obs) | **BIAS, default additive = today (§2.7).** The inflict/receive credit's accumulator is `Φ_status` (the `λ→0` telescoping form → no double-count with Φ_mat's Toxic/burn chip); `status_wasted` (#14) is the Markovian futile piece. Drop the `_prev_*_statused` diff |
| 30 | `stall_tax` | `:993–995` | ramp, −0.5 cap | **Bias** | **Markovian** (keys on `battle.turn`) | game past turn 60 | **yes** (global-env log-turn) | **keep, GENTLER** (§4.3): the progress clock can't see defensive stalls, so retain a soft absolute-turn term (re-tuned to ~−10, not −21.3) alongside the clock + the turn-250 forfeit |
| 31 | `pbrs_material` | `:1006–1010` | `W=2`, `γ=0.9999` | **PBRS** | Markovian | the belief potential `Φ_belief` | **yes** (belief block) | **keep, RENAME → `pbrs_belief`** (it is the belief PBRS, not material); new `pbrs_material` = `Φ_mat` |

### 3.1 The anti-spam collapse (rule 2 + rule 3, with evidence)

`repetition_tax`, `switch_bouncing_tax`, `dead_matchup_tax`(-escalation), and `struggle_tax` all encode
the **same bad** — spending a turn without progress — each via a **private history counter the network
cannot see** (`_consecutive_attack_repeats`, `_consecutive_bounces`, `_consecutive_dead_matchup_stays`,
`_consecutive_struggle`). Five escalation schedules proxying one quantity. **Collapse them into ONE
Markovian "no-progress" signal** keyed on `turns_since_progress` (§4), exposed as an obs feature (§5).

| term | disposition | why |
|---|---|---|
| `repetition_tax` (86% no-op) | **subsumed** by the clock | a no-op repeat *is* a no-progress turn (`_last_attack_had_effect=False`); evidence: 86% of today's repetition_tax fires on no-op turns |
| `repetition_tax` (14% productive) | **eliminated** | a still-damaging repeat is correct play; the clock never fires on a productive turn → the over-reach disappears |
| `switch_bouncing_tax` | **subsumed** | an A↔B bounce burns tempo with no board change = no progress |
| `dead_matchup_tax` escalation | **subsumed** | a 0×-only stay is no-progress by definition |
| `dead_matchup_tax` firing cond. | **kept** as a flat per-`(s,a)` futile bias | "all my damaging moves are 0×" is obs-derivable from the matchup block |
| `struggle_tax` | **subsumed** | a struggle loop is the purest no-progress pattern |
| `futile_attack/immune/setup`, `status_wasted`, `futile_spikes` (§2.6) | **kept** as one-shot per-`(s,a)` futile penalties | the single-turn no-op cost; already obs-keyed; the clock adds the *cross-turn escalation* on top (no double-charge — §4) |
| `stall_tax` | **kept, gentler** (§4.3) | the clock is offense-centric and can't see *defensive* stalls (heal/Protect wars); a soft absolute-turn term covers them (re-tuned to ~−10 — its real integral today is −21.3, not the −10 its comment claims) |

**Residual-history audit (rule 2 enforcement).** After the redesign, no kept bias may key on a hidden
internal counter. The counters retired or mirrored: `_consecutive_attack_repeats`,
`_consecutive_bounces`, `_consecutive_dead_matchup_stays`, `_consecutive_struggle` (→ the observed
`turns_since_progress`); `_prev_opp_boosts`, `_prev_opp_spikes`, `_prev_*_statused`,
`_prev_opp_se_threat` (→ current-obs reframes / transition events); `last_switch_turn` **[RT-R3]**
(→ drop the `switch_base` spam-gate); `_last_opp_seen_by` (→ drop the `se_switch` gate). `_prev_phi_*`
are PBRS state (pure state functions — fine). The shipped `_prev_active_ko_risk` is a deterministic
function of the in-obs belief block — fine.

---

## 4. The no-progress clock (the one Markovian anti-stall signal)

### 4.1 The progress predicate (exact) **[RT-2-blocker]**

`turns_since_progress` is an episode-scoped integer owned by `EpisodeTracker` (§5.1), updated once per
resolved decision window, clamped at `N_CAP = 10`. **Each window resolves to one of THREE outcomes**, not
two — `PROGRESS` (reset), `DENIED` (freeze, §4.1.1), or `NO_OP` (increment + charge):

```
PROGRESS(delta, live, prev_spikes) :=
      (delta.our_damaging_event is not None  AND  our move dealt ≥ PROGRESS_DMG_EPS to a non-fainted opp)   # (i)
   OR  delta.opp_status_applied is not None                       # (ii) a status LANDED on opp (the event)
   OR (opp_spikes_now − prev_spikes) > 0                          # (iii) a hazard LAYER was added
   OR  delta.opp_switch_to is not None                            # (iv) forced an opp commit (phaze / forced switch)

clock update:  PROGRESS → n = 0 ;  DENIED → n unchanged (no charge) ;  else NO_OP → n = min(n+1, N_CAP) + charge p(n)
```

with `PROGRESS_DMG_EPS = 0.03`.

**Critical: damage must be OUR-action-attributed, not net board change [RT-2-blocker].** The draft
keyed (i) on `opp_hp_delta.sum() ≤ −0.03`, but `opp_hp_delta` folds **all** opp-side damage regardless
of source — **Sandstorm (1/16 ≈ 0.0625) and Leech Seed (0.125) clear a 0.03 floor every turn and reset
the clock for free**, letting the agent stall indefinitely under passive chip. The fix: attribute
progress-damage to **our move** via `delta.our_damaging_event` (the protocol-truth damaging event for
*our* move) and its `target_hp_delta`, requiring `≥ PROGRESS_DMG_EPS` to a non-fainted target. Passive
chip the agent did not cause is **not** progress.

**Reconciliation with `_last_attack_had_effect` (`:1019`).** That flag asks "did our move have a
mechanical effect" (for routing the old repetition step); the clock asks the stricter "did this window
advance the game toward a win." Three deliberate divergences: (a) **our setup does NOT count** (below);
(b) damage must be **our-attributed and above a floor** (closes the chip hack); (c) status uses the
**transition event** `opp_status_applied`, robust to a cure-then-reapply no-op. A successful Roar/forced
opp pivot is new progress (not represented in the old flag).

**Setup is NOT progress (recommended) [RT-2].** Counting `our_boost_delta > 0` would let a Calm Mind ↔
Recover / Dragon Dance ↔ Substitute loop stall forever while "progressing" every other turn. Excluding
setup closes that loophole *at the predicate level* (no special-casing). Real setup is barely affected:
a setup sweep deals damage within 1–2 windows, resetting the clock; the toll at `n=1..2` is tiny
(§4.3), far under the eventual `boost_utilized` payoff. Setup that *never* converts within `N_CAP`
windows is, by construction, a stall — and should be taxed.

**Hardening summary (all baked into the predicate, all Markovian-from-the-obs):** damage → our-attributed
+ 3% floor (closes chip-reset); status → transition event (closes cure-reapply); hazard → strict layer
increase (closes Spikes-at-3-reset); setup → excluded (closes the setup-loop). Each progress type is
intrinsically un-spammable (status caps at one application, hazards at 3 layers, damage must be real),
so no "cap how often a type can reset" machinery is needed.

#### 4.1.1 DENIED — a missed / blocked / prevented attempt FREEZES the clock (does not increment)

A `NO_OP` (increment + charge) must be a **deliberate** failure to advance — a choice the agent could
have avoided, knowable from the obs at decision time (an immune attack at a 0× target, capped setup, a
redundant status, Spikes at 3). It must **not** include an attempt the agent made in good faith that was
nullified by **exogenous variance or the opponent's action**. Those are the `DENIED` case and they
**freeze** the clock (leave `n` unchanged, charge nothing):

```
DENIED(delta, live) :=  our move was progress-CAPABLE this window (a damaging move at a damageable
                        target — effectiveness > 0 — OR a status/hazard our action would have applied)
                        AND it did not land because of:
                          • a MISS  (delta.our_move_outcome == miss — accuracy/evasion roll failed), OR
                          • a BLOCK (delta.our_move_outcome == fail because opp Protect/Detect/Endure/Substitute absorbed it), OR
                          • PREVENTION (delta.our_failed_to_move / our_cant_reason ∈ {par, slp, frz, flinch} — our chosen move never fired)
```

**Why freeze, not increment or reset.** A miss is *attempted progress denied by RNG*, not stalling — the
agent chose to advance and an uncontrollable event nullified it. **Increment** would punish the agent for
variance it could not control (and double-hits: a miss already pays implicitly — 0 damage dealt, the
opponent got a free turn that shows in `Φ_mat`); it also injects avoidable variance into the penalty,
against the Markovian-cleanliness the clock exists to provide (the penalty should fire only on
*deterministic*, obs-knowable no-ops). **Reset** would over-credit a window where nothing actually
advanced, and is a mild dodge-the-clock surface. **Freeze** is the principled middle — "trying, not yet
succeeding": pressure neither grows nor vanishes. This is the same anti-over-reach principle as
eliminating the 14% productive-repeat tax (§3.1): do not punish a good attempt that didn't pan out.

**Safe in gen3 OU specifically:** evasion (Double Team / Minimize) is **Evasion-Clause-banned**, so a
miss is *always* accuracy-based ("good move, unlucky roll"), never an opponent-erected evasion wall.
There is therefore no "spam a guaranteed-missing move to freeze the clock forever" surface: a damaging
move at a non-immune target hits within a few turns (`PROGRESS` resets), and an immune move is a
deterministic `NO_OP` (increments). The `our_failed_to_move` cases are already exempted from
`futile_attack` today (`:790`); this extends the same exemption to the clock and to a miss/block.

**The kept `futile_attack` gets the same exemption.** Today `_compute_futile_attack_penalty`
(`:784–813`) exempts `our_failed_to_move` but **not** a miss, so a missed Fire Blast still eats `−0.05`.
Exempt misses there too (read `our_move_outcome`), by the identical logic — the futile family fires only
on *deterministic* no-ops (immune, out-healed-on-HIT, capped, redundant), never on RNG misses.

**Honest counter-argument (logged):** a miss *is* a wasted turn in pure game-flow, and charging it adds
a mild accuracy-aversion pressure (prefer reliable moves in clutch spots) that is arguably slightly
beneficial. But that pressure is small, the implicit `Φ_mat` penalty already covers the wasted tempo,
and it is outweighed by not rewarding/penalizing uncontrollable RNG. Lean freeze; it is a judgment call,
not a clear bug — A/B can revisit if accuracy discipline degrades.

#### 4.1.2 Defensive / support actions — the clock stays out of anything a Φ potential prices

The `PROGRESS` predicate (§4.1) is **offense-centric** (damage / status-on-opp / hazard / commit). That is
correct for *offense*, but a naive "anything not offense = `NO_OP`" wrongly taxes legitimate
**defensive/support** play (Recover, Aromatherapy, Rest, Wish, Protect). The governing principle — the same
no-double-count rule the PBRS potentials follow — resolves it: **the clock must not increment on an action
whose value (or scheduled value) a Φ potential already accounts for.** The full intent-aware classification:

| intent of the chosen action | outcome | clock | rationale |
|---|---|---|---|
| **offense** lands (dmg ≥3% / status / hazard / commit) | progress | **RESET** | game moved toward a resolution |
| **offense/active attempt** denied (miss / Protect-block / cant) | denied | **FREEZE** | §4.1.1 — uncontrollable |
| **offense/setup/hazard** achieves nothing & moves no Φ (immune attack, <3% chip, capped setup, redundant status, Spikes-at-3) | futile active | **INCREMENT + futile** | a deliberate, obs-knowable wheel-spin |
| **support/defense** that registered value in another reward component (heal → `Φ_mat`; self-cure → the status term [additive at `λ=1`, `Φ_status` at `λ<1`]; pending Wish) | productive support | **FREEZE** | the value is *already* priced elsewhere; the clock punishing it double-counts and nukes walling |
| **support/defense** that achieved nothing (full-HP Recover, no-target cleric, spam-failed Protect) | wasted support | **INCREMENT + futile** | a genuinely wasted move (parity with capped setup) |

The discriminator is **"did a Φ potential register this action, or was it a denied attempt"** → freeze;
**"deliberate, achieved-nothing, moved-no-Φ"** → increment. Setup is the one deliberate exception that
*always* tolls (the §4.1 setup-loop closure) — but cheaply. This keeps the clock entirely out of the
material/status/belief signals (no double-count) and out of legitimate defense.

**Worked edge cases (the ones that motivated this):**
- **Recover / Wish / Moonlight / Softboiled etc.** — heals real HP → `Φ_mat` ↑ → FREEZE (the heal is the
  reward; the clock is silent). At full HP → moves no Φ → INCREMENT + a `futile_heal` (mirror of
  `futile_setup`). **Wish** is the subtle one: the heal lands *next* turn, so the use-turn moves no
  immediate `Φ_mat` — treat a scheduled-benefit support move (`is_heal` flag, or the §5.3 pending-Wish
  state) as FREEZE on the use-turn (it is deliberate support, not a wheel-spin), and the landing turn
  freezes via the `Φ_mat` jump.
- **Aromatherapy / Heal Bell** — cures real status → `Φ_status` ↑ (proportional to statuses cleared, a
  strict improvement over today's flat `±0.3`) → FREEZE; nothing to cure → INCREMENT + futile.
- **Seismic Toss / Night Shade / fixed-damage** — a connecting hit is ≥3% always → RESET (the
  `our_damaging_event` fires on the protocol `-damage`, independent of the dex `base_power=0` tag). Into a
  Ghost (0×) → a deterministic `NO_OP` + `futile_immune`. ⚠️ **Bug to fix:** the current `futile_attack`
  gates on `md.base_power == 0` (`:804`), which *wrongly exempts* a Seismic-Toss-into-Ghost (it reads BP 0)
  — use the `incoming_damage.FIXED_DAMAGE` set (already maintained for the belief block) so fixed-damage
  moves are treated as damaging for the immune/futile check.
- **Rest (Rest-zap / Sleep-Talk Suicune)** — the full heal → `Φ_mat` ↑ → FREEZE (the Toxic-clock-clearing
  is under-credited by uniform `Φ_status`, the §2.7 caveat, but the heal carries it); the asleep turns →
  FREEZE (cant=`slp`, §4.1.1); **Sleep Talk** → RESET if its random call deals damage, FREEZE if not (the
  agent chose its only useful asleep action; the call is exogenous, like a miss).
- **Failed Focus Punch** — breaks via `cant_reason=focuspunch` → already FREEZE (§4.1.1). The *misplay
  cost* is priced where it belongs — `Φ_mat` (we ate the opponent's hit and dealt nothing). It is more
  *foreseeable* than a pure miss, so a tunable exists: treat `cant=focuspunch` as `NO_OP` instead of
  `DENIED` if A/B shows reckless Focus-Punching. Default freeze + let `Φ_mat` price the eaten hit.

**Two gates on the penalty [RT-2-major×2]:**

- **Forced-switch windows are no-ops.** A post-faint / phaze replacement allows only switches, so it
  fails the predicate spuriously. Gate on `delta.phase_is_forced_switch`: do **not** increment and do
  **not** charge on a forced window.
- **Trapped-vs-wall is not charged.** A trapped mon with no progress move has switches masked illegal,
  so a penalty would be unavoidable by *any* policy (punishing a state, not a choice). Gate the
  **charge** on a switch being legal this decision (`legal.switches` non-empty). The obs counter may
  still tick (so the model sees the stuck state), but no return is levied for helplessness.

**Productive repeats/bounces escape entirely.** A damage-dealing repeated attack
(`our_damaging_event ≥ 3%` each window) pins the clock at 0 → zero penalty (fixing the 14% over-reach).
A matchup-improving A→B→A pivot that lands damage/status/commit resets; a pure tempo-pivot that lands
nothing pays only the front-loaded toll once (correctly — it *was* a no-progress window) without a
separate bouncing tax.

### 4.2 Front-loading shape — decision

Goal: **"stop useless turns ASAP"** ⟹ the marginal penalty must be **non-increasing** in
`n = turns_since_progress` (the first useless turn hurts at least as much as any later one; a long
pointless game is never "cheaper per turn" early). Three shapes, `p(n)` = penalty charged when the
clock reaches `n`:

**(a) LOG (front-loaded, saturating).** `p(n) = −c·(log(1+n) − log(n)) = −c·log(1 + 1/n)`,
`p(1) = −c·log 2`; cumulative `P(n) = −c·log(1+n)`. With `c = 0.30`: `p(1)=−0.208`, decaying to
`p(10)=−0.029`; cumulative `−0.72` at n=10. Marginal at n=1 is **7.3×** n=10; 58% of the first-10 cost
in the first 3 turns. Strongly front-loaded.

**(b) FLAT.** `p(n) = −c`; cumulative `−c·n`. With `c = 0.15`: constant `−0.15`/turn, cumulative `−1.5`
at n=10. The first useless turn already hurts as much as any later one (no early discount) — satisfies
the goal, just not *strictly* front-loaded.

**(c) CURRENT ESCALATING (wrong).** `p(n) ≈ −step·n`; cumulative quadratic. The marginal *grows* with n,
so the first useless turn is the *cheapest* — the model learns "early useless turns are nearly free,"
the opposite of "stop on turn 1."

**Recommendation: (b) FLAT at `c = 0.15`, with (a) LOG as the documented fallback.** Both (a) and (b)
satisfy the hard goal; (c) fails it. (b) is preferred for *this* environment because under
`gae_lambda = 0.80` (~5-window credit horizon) the LOG shape's *decaying* marginal can read as
"continuing is getting cheaper" within the credit window — a miniature of the (c) pathology — whereas
FLAT carries one fully-legible `−0.15` per useless turn with **zero variance** for the critic to learn.
It also needs one tunable, not `c` + a `p(1)` convention. **Switch to (a)** if A/B shows the policy
"nibbles" (repeated isolated single useless turns because one `−0.15` is cheap) — the LOG's heavier n=1
marginal (`−0.21`) discourages even starting. Same constant plumbing either way.

**Cap & cumulative vs ±30.** Clamp `n` at 10; past the cap, keep charging `−0.15` (do not stop — a
committed staller would go free; do not escalate). A 40-useless-turn stretch = `−6.0`; reaching `−30`
would need 200 consecutive useless turns, which the **turn-250 forfeit (scored a −30 loss** via the
stall config, `gen3_env.py:106–108`) makes impossible. The clock is structurally bounded well under the
±30 terminal — it can never dominate win/loss or make a stall-loss preferable to a forfeit.

### 4.3 `stall_tax` — retain a GENTLE version (revised — the progress clock cannot catch defensive stalls)

The progress clock and `stall_tax` are **complementary, not redundant** — this is a deliberate revision of
an earlier draft that recommended *replacing* `stall_tax`. The §4.1.2 analysis shows the clock is
**offense-centric and cannot judge defensive progress**: a productive-heal stall (a Recover war under
Sandstorm — each heal moves `Φ_mat` → FREEZE) is exactly the case the clock is silent on, by design (it
must not punish legitimate walling). So:

- **The progress clock** handles **active wheel-spinning** (immune attacks, capped setup, struggle,
  move-spam, Spikes-at-3, wasted support) — progress-conditioned and Markovian.
- **A gentle absolute-turn `stall_tax`** handles the **defensive/global stall** the clock can't see (two
  walls trading heals/Protects, a Wish-stall). It keys on `battle.turn` (in the obs via global-env
  log-turn → still Markovian) and pressures *length itself*, which is the right signal when neither side
  is doing anything the clock can register.
- **The turn-250 −30 forfeit** is the hard backstop (a stall-to-cap is a loss).

So **keep `stall_tax`, but make it gentle** — fix it to its *intended* magnitude. **Doc bug to fix in the
same pass:** `reward_manager.py:153` claims `stall_tax` cumulates to "≈10 over a 190-turn game"; the actual
integral of `min(0.05·(t−60)/20, 0.5)` over turns 61–190 is **≈ −21.3** (it saturates to the −0.5/turn
clamp at turn 70). Re-tune it to the intended ~−10 ceiling (raise the start turn / lower the ramp+clamp)
so it is a soft length-pressure, not a second heavy tax double-charging the clock on active stalls. (The
two only overlap on a turn that is *both* late *and* an active wheel-spin; the gentle `stall_tax` keeps
that overlap small.) This restores the original term-classification position (`stall_tax` stays as the
obs-keyed turn clock); the edge-case analysis vindicates it over the "replace" draft.

### 4.4 Markovian property

The penalty is `p(turns_since_progress)` — a deterministic function of one integer, which (§5) is in
the obs. So `V(s)` predicts the expected no-progress penalty with **no irreducible variance** (it
*improves* `explained_variance`, unlike a hidden-counter penalty that injects noise the critic cannot
fit), and the policy learns the exact contingency ("if `turns_since_progress` is high and my only repeat
is no-progress, switching/attacking-for-damage dominates"). New field: `no_progress_tax`. It is **not**
a PBRS term — a stall *should* cost return — so it deliberately does not telescope; it is orthogonal to
both `_prev_phi_mat` and `_prev_phi_belief`.

---

## 5. Feature-encoder enrichment (deliverable B)

### 5.1 `turns_since_progress` — the progress clock (the one mandatory feature)

**Sourcing [RT-2-blocker, env.py-verified].** It is a **cross-turn counter**, so it **cannot** live in
`LiveView` (invariant: primitives only, no past-turn state — `live_view.py:18–22`). The exact precedent
is `HiddenPowerTracker`: an episode-scoped accumulator owned by `EpisodeTracker` (`:102`), exposed as a
property, threaded into `encode()` as the `hp_tracker` arg (`gen3_env.py:74`), reset in
`EpisodeTracker.reset()`. `turns_since_progress` follows this template exactly — a new `ProgressClock`
owned by `EpisodeTracker`, **read by both the obs encoder and the reward** so they share **one value**
(the entire point).

**Timing — update at `record()` (embed time), NOT at `process_turn_reward`.** This is the corrected
plumbing **[RT-2-blocker]**. Within a single `env.step()`, poke-env runs `embed_battle` (the *next*
obs, `env.py:591`) **before** `calc_reward` (the *current* reward, `env.py:600`). So if the clock were
updated inside `process_turn_reward` (calc_reward), the just-built obs would have already been encoded
with the **stale** pre-window value — the model would be blind to its own most recent no-op until a
window later. The fix mirrors the HP tracker: update the clock inside `EpisodeTracker.record()` (called
at the top of `embed_battle`, before `encode`), folding the just-completed window's `TurnDelta` (the
same fold `_maybe_observe_hidden_power` already does there). Consequences, with `n_D` = the value in the
obs the model sees at decision `D`:

- `embed_battle(D)`: `record()` folds window `D−1` → clock = `n_D`; `encode()` reads `n_D` → **obs is
  always fresh** (reflects everything resolved before `D`).
- `process_turn_reward(D)` (later in the same step) reads the clock value for the no-op streak length
  and charges `p(streak)` — the **marginal** cost. Because `record()` (embed for `D+1`) runs *before*
  `calc_reward` in the same step, the penalty for window `D` keys on `n_{D+1}` = `f(n_D, action_D)`, a
  deterministic function of the obs the model saw at `D` and its action. `V(obs_D)` predicts its
  expectation → **Markovian**. For FLAT this is simply `−0.15` per gated no-op window; for LOG it is
  `p(streak length)`.

So **the obs at every decision is current, and the penalty is recoverable from `(obs, action)`** — the
off-by-one is gone. The progress predicate is computed once in `ProgressClock.update(delta, live)` and
is the single source of truth shared by obs and reward (no `_last_attack_had_effect` duplication).

**Encoding — one scalar, log-saturated.** `PROGRESS_CLOCK_CAP = 10`; encode as
`log(1 + min(n, CAP)) / log(1 + CAP) ∈ [0,1]` — the same form `global_env.py` already uses for the turn
clock (`log(1+turn)/log(1+MAX_TURNS)`), so it reuses a normalization the extractor handles well. Log is
right because the marginal of "one more no-progress turn" matters most early (1→3) and saturates late
(8 vs 10 are both "stalling"); a one-hot would waste 10 dims for a monotone scalar.

**Layout — widen `REACTIVE_SCALAR_DIM` 14 → 15.** The scalar joins the reactive scalar lead block (a
"reactive momentum" gauge like fainted-counts / trapped), **before** the matchups so it flows to both
heads + the global token via `non_matchup_rest` automatically. Cascade (all auto-computed from
`REACTIVE_SCALAR_DIM` in `constants.py:110–114`):

| constant | before | after |
|---|---|---|
| `REACTIVE_SCALAR_DIM` | 14 | **15** |
| `INCOMING_DMG_OFFSET` | 50 | **51** |
| `REACTIVE_MATCHUP_OFFSET` | 83 | **84** |
| `REACTIVE_DIM` | 371 | **372** |
| base_dim / full obs | 1789 / 3390 | **1790 / 3391** |

`ReactiveEncoder.encode`/`get_layout`/`describe_vector` write/declare the scalar at its index; the
prober's pinned offset test (`prober/engine_test.py::test_offsets_resolve_matches_layout`) shifts by +1
and must be re-pinned (the engine resolves from `get_layout()` at runtime — only the *test* pins break).

### 5.2 `last_mon_switched_from` / `oscillation_depth` — DROP

The progress clock subsumes bouncing (an A↔B oscillation makes no board progress → the clock climbs),
and the model can already *detect* bouncing from the 10-turn TurnDelta history (per-slot `our_switch_to`
/ `our_actor`). There is no reward term that keys on the *identity* of the mon we left (the bounce tax
keyed on the *equality* `target == _last_switched_from`, a relationship the history already expresses).
**Add an `oscillation_depth` scalar only if** A/B reveals a residual clock-dodging oscillation exploit;
default = drop, for minimality.

### 5.3 Pending delayed effects — Wish & Future Sight (obs-only enrichment, lowest priority)

These **are** current-board pending state (a scheduled future event true *now*), so they belong in
`LiveView` (extend `LiveSide` / `global_env`), **not** the EpisodeTracker. No reward term keys on them
(and none should — that would hand-code the game, against "provide facts, let it learn"); they are
**obs-only enrichment**. The justification (rule 3 in miniature): a Wish cast on turn `T` heals at the
end of `T+1`, but the history window only shows Wish was *used* — forcing the transformer to detect the
move, count turns, and know the delay, an inference chain across attention positions for a deterministic
scheduled event. An explicit `wish_pending = [present, turns_until]` scalar hands it the resolved fact.

**Caveat — verify the fork first.** From the code read, `LivePokemon.volatiles` is built from
`mon.effects` and side conditions from `battle.side_conditions`. Wish/Future Sight are tracked as
side/slot conditions with a countdown; **presence + turns-until is the reliable part**, but the
**amount** (Future Sight damage = attacker SpA at fire time; Wish heal = 50% of the wisher's max HP) is
likely *not* retained as a pending scalar. Specify **presence + turns-until** (in the 18-dim global env
block, where weather/screens/spikes live — near-zero cost, `side_conditions` already in hand), and add
`amount` only if the fork verifiably exposes it. A bridge-backed `*_fuzz_test.py` must validate presence
against protocol truth. **This is deferrable** — it does not gate the Markovian-reward work; include it
only if the FPS gate has headroom.

### 5.4 Sweep — every other kept Markovian key is already observable

| reward key | observable? | verdict |
|---|---|---|
| `stall_tax` turn count | **yes** (global-env log-turn) | no new feature (kept as a gentle term, §4.3) |
| `_prev_active_ko_risk` (re-gate) | **yes** — `max(pko_phys, pko_spec)·(1−outspeed)` from the belief block at offset 51 | no new feature |
| `Φ_belief` (shipped) | **yes** — pure function of HP fracs + belief block | no new feature (PBRS: model needs the state, not Φ) |
| `Φ_mat` (new) | **yes** — HP at `POKEMON_HP_OFFSET=67` + alive bits | no new feature |
| status counts | **yes** (per-mon condition one-hots + transition events) | reframe in place |

**Net: the only genuinely un-observable key a kept Markovian reward needs is `turns_since_progress`.**
Every other escalating counter is a specialization of "no progress," unified under the one clock.

### 5.5 Retrain-class checklist

1. **ARCH_SIGNATURE bump** — `model_version.py`, `gen3_incoming_damage_v2` → e.g.
   `gen3_markovian_progress_v1` (obs dim changes 3390→3391 — a weight-shape change).
2. **Golden fixture regen** — `src/agents/training/golden_obs_fixture.json` (byte-for-byte pin).
3. **Mandatory obs-build benchmark** — `obs_build_benchmark.py --turn 25 --reps 400 --top 22`,
   before/after, same session; gate **<10% calls/encode**. The clock scalar is ~free (one int read +
   one `math.log`) → expect ≈0%, but the gate is mandatory for any `observation/` edit.
4. **Prober offset re-pin** — `prober/engine_test.py::test_offsets_resolve_matches_layout`.
5. **Stale-comment cleanup [RT-R3-nit]** — fix `constants.py:116–121` comments (they say base 1579 /
   OFFSET_OPP_TEAM 594; live code is base 1790 / full 3391 post-change). Only the named constants are
   load-bearing.
6. **Reward-value changes** (the term collapse, renames, reframes, the no-progress tax) are retrain-class
   but need **no** ARCH bump — they land together with the obs change in one fresh run, so one bump
   covers both.

---

## 6. Coexistence / constraints

### 6.1 The shipped switch PBRS + the naming migration **[RT-7]**

`design_reward_switching.md` (commit `7483dd1`) shipped the **belief** PBRS (stored mis-named in
`pbrs_material`) + the belief re-gate of `escape_threat_switch`/`matchup_penalty`. This design **keeps**
both and **adds** `Φ_mat`. The **always-on PBRS class** is exactly these two potentials, coexisting by
simple addition:

```
Φ_total(s) = Φ_mat(s) + Φ_belief(s)                                        (the always-on PBRS class)
F_total,t  = γ·Φ_total(s_{t+1}) − Φ_total(s_t) = F_mat,t + F_belief,t      (telescopes as a sum)
```

`Φ_hazard` / `Φ_status` are **NOT** in this always-on sum — they are the *telescoping forms of BIAS
terms* (§2.6–2.7), active only as `bias_additivity → 0`; at the default `λ=1` `spikes`/`status` are
additive (today) and contribute no potential. **Bookkeeping:** two always-on `_prev_phi` carries
(`_prev_phi_mat`, `_prev_phi_belief`), both `None` at `reset()`, `Φ(terminal)=0`, keyed on `PBRS_GAMMA`;
plus the BIAS accumulators (`Φ_hazard`/`Φ_status` and any swept term) which only carry state when
`bias_additivity < 1`. `RewardBreakdown` fields in the `shaping` group: `pbrs_material` = `Φ_mat`,
`pbrs_belief` = the shipped belief term (always-on); `pbrs_hazard`/`pbrs_status`/… = the BIAS refund
components (zero at `λ=1`). The telescoping/terminal/falsifier machinery (§7) covers the always-on pair
plus the parameterized BIAS refund. With `Φ_total(s_0) ≈ 0` (declared full teams), the per-episode
constant `−Φ_total(s_0) ≈ 0` with small cross-episode variance — the dense `ΔΦ` still raises the shaping
magnitude the value head tracks, reinforcing the PopArt pairing (§6.3). **The rename is
a recorded-schema migration, not a "pure rename"** [RT-7]: `RewardBreakdown.to_dict()` emits these into
every `eval_traces/*/summary.json`, which the prober + TD-residual eval metric consume. Pre-rename traces
carry the belief term under `pbrs_material`; post-rename traces carry it under `pbrs_belief`. The offline
falsifier (§7.1) must branch on trace vintage (the manifest's `arch_signature`/`git_hash`); cleanest, since
this is a fresh run, is to validate the belief term only on pre-rename traces with pre-rename code, and
run the new falsifier only on traces from a build that has *both* fields. Grep the prober / `eval_callback`
/ TD-residual code for the literal `"pbrs_material"` and confirm nothing hardcodes it as "the belief term."

### 6.2 No reward annealing — and why

This design **does not** adopt `design_reward_annealing.md` for the shaping half, and the reason is
structural, not a deferral:

- **Annealing exists to DE-BIAS a return that PBRS never biases.** Annealing scales the *shaping* terms
  (Tier A/C heuristics) toward 0/floor late, because those terms **change the episode return** and a
  strong policy can farm them. **A potential-based term has no bias to anneal**: `F_t` telescopes to a
  single policy-invariant per-episode constant `−Φ(s_0)`. It does not shift the optimum and cannot be
  farmed. Annealing a PBRS term toward zero would only re-introduce the credit-assignment lag the term
  bridges, with **no de-biasing benefit** (there was no bias).
- **In one line:** annealing de-biases *late, by scaling shaping toward 0*; PBRS is **de-biased from
  step 1, by construction** — the bias is zero at every timestep, not just at `anneal_end`. Converting
  the material spine to `Φ_mat` makes the annealing machinery (the `set_attr` plumbing, the cosine
  schedule, the Tier-A/B/C taxonomy) unnecessary for the shaping half. The only terms left to *possibly*
  anneal are the small, bounded, observable Markovian biases — and their de-biasing is a separate, later
  decision, not part of this design. (`design_reward_annealing.md`'s Tier-B "keep base, never anneal"
  maps onto "base → `Φ_mat` PBRS"; its Tier-A/C onto the Markovian biases this doc classifies.)
- **Boundary [RT-honesty]:** annealing's *second* motivation — a pure win-probability `V_θ` for v6 MCTS
  leaf evaluation — is only *partially* addressed by PBRS. PBRS removes the *shaping* bias, but the kept
  Tier-B outcome proxies (`Φ_mat` is still material) leave `V_θ` a material-biased return estimator. That
  residual is the same one annealing's "optional final outcome-only phase" targets, and remains a
  deferred v6 item. This design does not claim to solve it.
- **`bias_additivity` IS the de-bias knob annealing wanted — but as a per-run CONSTANT, not a schedule.**
  Where annealing scales BIAS shaping toward 0 *over a step window* (a moving reward target that drifts
  the value head, §design_reward_annealing.md), `--bias-additivity` sets, **once per run**, how much the
  BIAS class biases the objective (`λ·acc`). At `λ=0` a BIAS term is a pure telescoping hint — exactly
  what annealing converges to at `anneal_end`, but *stationary from step 1* (no drifting target, no
  cosine plumbing, no `set_attr` push). So a run that wants late-stage de-biased shaping sets a low
  `λ` and gets it from the start, policy-invariantly. The accumulate-refund (§1.2) is the static
  analogue of annealing's per-tier coefficient; the difference is stationarity (which the value head
  prefers) vs a schedule (which annealing needed only because it scaled per-turn values directly).

### 6.3 PopArt pairing **[RT-6]**

The `−Φ_total(s_0)` constant lands in the **value-loss target** (per-batch advantage normalization
cancels it in the *advantage*, but the value head regresses the *unnormalized return*). The declared-team
`Φ_mat` (§2.2) makes `Φ_mat(s_0) ≈ 0` with near-zero variance, minimizing this. Still, because the new
material density adds return-target structure to the trunk this run is already fighting (`project_popart`,
`grad_balance.py value_share`), **pair this design with `--use-popart`** when run, and add
`grad/value_share` + `train/return_std` to the pre-registered guards (§7.4) — they must not regress vs
baseline.

### 6.4 Out of scope

`VICTORY_VALUE = 30` (the ±30 terminal) — untouched; both potentials use `Φ(terminal) = 0`.

---

## 7. Verification / falsification plan

### 7.0 Prerequisite (gates everything)

Resolve the naming collision FIRST: `RewardBreakdown` has BOTH `pbrs_material` (`Φ_mat`) and
`pbrs_belief` (shipped), both auto-summed into `.total` (the `fields()`-based property at `:110` already
does this — pin it with a sentinel-field test), both in `_GROUPS` "shaping". Catches the most likely
impl bug (one potential shadowing/dropping the other).

### 7.0a Registry tests (the framework itself)

- **Registry coverage (`reward_registry_test.py`).** Every current `RewardBreakdown` field maps to
  **exactly one** registry entry with **exactly one** class (TERMINAL / PBRS / BIAS) — **no orphans, no
  double-counts**. Assert the registry's term set == the dataclass field set; assert the fold loop's
  output `.total` == the sum over registry entries (the fold is registry-driven, no hidden term). Assert
  each class's membership matches §3 (PBRS = #1–4,#31; TERMINAL = #5; BIAS = the rest).
- **BIAS no-op equivalence (`bias_additivity_noop_test.py` + the offline replay below).** At
  `bias_additivity = 1.0`, every BIAS-class per-turn reward is **byte-identical (float tol)** to the
  current `reward_manager` — both as a unit test on constructed turns and as the offline replay (§7.1a).
  This is the proof the flag default changed nothing for the bias class (the accumulate-refund emits the
  current value, refunds 0).
- **Parameterized BIAS telescoping (`bias_additivity_blend_test.py`).** Over a synthetic episode with a
  known BIAS accumulator `acc`, assert the episode-summed BIAS contribution `== λ·acc` for
  `λ ∈ {0, 0.5, 1}`: `0` at `λ=0` (fully refunded), `acc` at `λ=1` (no refund), `0.5·acc` at `λ=0.5`.
  Assert the **per-turn** values are identical across all `λ` (only the refund differs) — the
  accumulate-refund property. Assert the low-variance refund path (accumulator-potential, §1.2) gives
  the same episode sum as the terminal-lump reference, to tolerance.

### 7.1 PBRS unit + offline falsifier

**Telescoping unit test (`pbrs_telescope_test.py`).** Over a synthetic, decision-window-indexed
trajectory **[RT-1]** (HP drains, a switch that moves `Φ_belief` but not `Φ_mat` and vice-versa, a
faint where an alive bit flips — assert `Φ_mat` drops, never rises — and a terminal state; lengths
`T ≈ 40` and `T = 250`), run the **exact** loop the manager runs (`_prev_phi=None` first-window skip;
`Φ(terminal)=0`) and assert `Σ_t γ^t F_t` equals the closed-form telescope of the *same* emitted `F_t`
sequence, for `Φ_mat`, `Φ_belief`, and `Φ_total`. **Numerical care:** float64; `atol=1e-6, rtol=1e-5`;
construct the trajectory to keep the `Φ` clamp **inert** (a clamped step breaks the identity) and assert
it never activated. Includes the **terminal-window guard [RT-2]** as the *first* test written: a scripted
6-0 winning KO where `Φ_mat(post-win) ≈ +19.5` must yield `bd.pbrs_material == −_prev_phi_mat` (small
negative), and the episode `base + terminal` total must equal `+30 − Φ_mat(s_0)` **regardless of margin**.

**Offline reward-replay falsifier (`pbrs_replay_falsifier.py`, script, no sim).** `eval_traces` holds
everything needed (verified against `battle_recorder.py` + `prober/engine.py`): `summary.json
invocations[i].outcome.reward` = `RewardBreakdown.to_dict()` (per-decision `total` + named components);
`states.npz obs[i]` = the full obs the model saw; `values[i]` = recorded `V(s_i)`. Two modes:

1. **Telescoping check on the shipped term (pure arithmetic, run first).** Sum the recorded
   `pbrs_material`/`pbrs_belief` column across a game; assert `|Σ_t pbrs + Φ_belief(s_0)| < ε` where
   `Φ_belief(s_0)` is recomputed from `obs[0]` (`decode_incoming_belief`). **Tolerance `ε ≈ 0.05`, not
   1e−6:** the saved per-step term is `γ·Φ′ − Φ` (γ applied inside), so the *undiscounted* sum telescopes
   to `γ^T Φ_T − Φ_0 + (γ−1)Σ_interior Φ` — the `(γ−1)` residual is `O(10⁻⁴·T·Φ̄) ≈ O(0.01)` over 100
   turns. This is a **coarse** "did the band collapse to a small constant" check (catches sign/init/
   terminal bugs producing `O(game-length)` drift); the exact-telescope check is §7.1's job.
2. **Re-score `Φ_mat` from saved obs.** Decode `Σ our_hp` / `Σ opp_hp` / alive from `obs[i]`
   (`POKEMON_HP_OFFSET=67`; opp at `OFFSET_OPP_TEAM=642`; opp unrevealed slots → full-HP per §2.2), form
   `F_mat,i`, assert `Σ_i F_mat,i ≈ −Φ_mat(s_0)` (bounded, `ε≈0.05`). **Falsification:** if `Σ F_mat`
   scales with game length, `Φ_mat` is not a clean potential — a hard FAIL before any GPU training.

Run against an absolute `models/run_<…>/eval_traces/...` path (models/ lives in the main checkout).

### 7.1a The two default-run replay checks (the single-variable proof)

The default run changes **exactly one thing** vs the live baseline; the offline replay proves it by
splitting the two halves:

- **BIAS no-op equivalence (must be IDENTICAL).** Re-score the saved `eval_traces` at
  `bias_additivity = 1.0` and assert the per-turn **BIAS-class** rewards are byte-identical (float tol)
  to the recorded `reward_manager` values. This proves the flag default + the registry fold changed
  nothing for the bias class — the necessary control for clean attribution.
- **MATERIAL clutch-fix (must DIFFER, in the predicted way).** This is the one intended change: re-score
  with `Φ_mat` replacing the unconditional base spine and confirm **win-game returns collapse toward
  `+30` and loss-game returns toward `−30`** (the material margin removed), i.e. the dominant-vs-clutch
  spread (≈+47 vs +26 today) flattens to ≈+30 for all wins. Decode `Σ our_hp`/`Σ opp_hp`/alive from the
  saved `obs[i]` (no sim). **This SHOULD differ from today** — it is the falsifiable signature of the
  clutch-fix; if returns do *not* flatten, the `Φ_mat` plumbing is broken.

Together these two replays *are* the "single-variable change" evidence: bias unchanged, material fixed.

**`PBRS_GAMMA == model.gamma` invariant [RT-8].** Currently `PBRS_GAMMA = 0.9999` is a hardcoded module
constant **never asserted against `model.gamma`** (the shipped doc *claims* the assertion exists — it does
not). **It cannot live in the reward-manager constructor** (the manager is built at env construction,
`gen3_env.py:51`, before the PPO model exists). Thread `model.gamma` into the reward manager **after both
are built** in `train_rl_agent.py`, assert `PBRS_GAMMA == model.gamma` there and in the existing
`_run_roundtrip_test` smoke (verify the exact line — do not trust the unconfirmed `:195` cite). This now
guards **both** potentials. **Pre-registered:** because both PBRS terms are policy-invariant, an **ELO
regression vs the kept-alive baseline is a BUG SIGNAL** (broken telescoping or gamma mismatch), not an
experimental outcome — with PBRS-only, halt and debug before attributing to the biases.

### 7.2 Markovian-bias tests

- **`markovian_obs_share_test.py`** — decode `turns_since_progress` from its obs slot and assert it
  equals the counter the reward keyed on, and **pin the phase**: the value shown at `embed_battle(D)` is
  the count of no-progress windows resolved before `D`, and the penalty for window `D` is the marginal
  `f(that value, action_D)` (§5.1). A refactor that reintroduces the off-by-one fails here.
- **`no_progress_clock_test.py`** — construct a no-op stall (capped setup; immune attack;
  Protect/Recover loop) → clock increments, penalty fires; construct a productive repeat / chip ≥3% /
  matchup-improving bounce → clock RESETS, no penalty. Assert **Sandstorm/Leech-Seed-only** windows do
  NOT reset (the `our_damaging_event` attribution [RT-2-blocker]). Assert forced-switch windows and
  trapped-no-switch windows are not charged. Assert the clock and any retained futile term do **not**
  double-charge a single no-op window beyond the intended (one-shot futile + one clock marginal).
  **Three-outcome / miss-freeze (§4.1.1):** assert a **missed** damaging move at a damageable target, a
  **Protect-blocked** move, and a **full-para/sleep-prevented** move each **FREEZE** the clock (`n`
  unchanged, no charge) — distinct from both an immune attack (increments) and a landed hit (resets).
  Assert `futile_attack` also no longer fires on a miss (the same exemption). Assert a damaging move that
  *hits* at the same target still resets, so freeze is not a reset (no clock-dodge surface).
  **Defensive/support (§4.1.2):** assert a **productive Recover** (heals real HP → `Φ_mat` moves) and a
  **productive Aromatherapy** (cures a real status → `Φ_status` moves) each FREEZE (not increment, not
  reset); a **full-HP Recover** and a **no-target cleric** INCREMENT + a `futile_heal`/futile penalty; a
  **Rest** (heal → freeze) followed by **asleep turns** (cant=slp → freeze); and a **Seismic Toss into a
  Ghost** triggers `futile_immune` (the fixed-damage move is NOT exempted by the `base_power==0` gate —
  the `FIXED_DAMAGE`-set fix). Assert the gentle `stall_tax` still pressures a **productive-heal stall**
  (a Recover-war the clock freezes through) so the defensive-stall hole is covered.
- **`faint_material_handoff_test.py`** — (i) `faint_ours` no longer contains the `0.75`; (ii) on a faint
  `Φ_mat` drops by exactly `2·hp_before + MAT_ALIVE_WEIGHT` (material conserved, not double-removed);
  (iii) the `finishing_blow` `we_fainted` guard still makes a healthy Explosion trade net negative under
  `Φ_mat`; (iv) `explosion`'s literal is gone but `explosion_block` still fires (the nesting [RT-4]).

### 7.3 / 7.4 Pre-registered behavioral metrics (declare BEFORE the run, vs the kept-alive baseline)

| metric | definition | prediction |
|---|---|---|
| **ELO** | `eval/elo` (anchored BT) at matched steps | **≥ baseline within CI.** PBRS contributes 0 in expectation; the biases should help. A PBRS-only regression = a bug (§7.1). |
| **clutch-win conversion ↑** | among **won** games, `1 − fraction` where the win included an our-faint at `hp_before ≥ 0.8` while a non-suicidal KO/switch existed (the inverse of the §1 under-switch scan; decode `hp_before` + pivot existence from the belief block) | **UP** — replacing the flat 0.75 with graded `Φ_mat` stops pricing a clutch sacrifice like a careless one |
| **useless-turn-rate ↓** | `(decisions with no our-attributed damage/status/hazard/commit, action = attack/setup) / non-forced decisions` | **DOWN** — the clock's direct target; report the curve over steps |
| **under-switching not ↑** | the shipped prober diagnostic — switch-prob-mass-vs-P(KO) curve + count of "high P(KO), alive low-pko pivot existed, stayed and died" | **no rise** (don't undo the shipped switch work) |
| **over-switching/bouncing not ↑** | bounce-rate + voluntary-switch fraction at **low** P(KO) (Φ_mat rising when a doomed active goes to bench could over-reward switching) | **no rise** — symmetric guard |
| **reckless-trade not ↑** [RT-9] | high-HP mutual-KO / unfavorable-trade rate (the symmetric opp-alive term raises immediate KO credit) | **no rise** |
| **status/hazard rate not ↓** (§2.6/§2.7) | status-application rate + spikes-set rate vs baseline (PBRS removes the *net* incentive these biases gave; the tempo-status signal is diffuse) | **no collapse** — a drop means the bridged realized signal is too weak; restore a small standing bias for the non-damaging statuses only |
| **value-scale not ↑** [RT-6] | `grad/value_share`, `train/return_std` | **no regression** (the new return-target structure must not worsen the trunk-swamping) |
| win-margin flattening | mons-remaining-on-win / shaped-return spread | **SANITY CHECK ONLY, NOT a result** — PBRS makes every win's return collapse to `+30 − Φ_total(s_0)` *by construction*; a flattened distribution confirms the plumbing (the macroscopic shadow of §7.1's telescoping), not a better policy. If it does NOT flatten, that's a PBRS bug the falsifier should have caught. |

### 7.5 Ablations

- **10-vs-N turn-history.** `turns_since_progress` as a scalar is NOT window-limited (its *key* is
  observable at any streak length), so this tests whether the model needs the *raw* 10-turn history to
  *learn* to act on the clock. Prefer **ablation-by-zeroing** (fixed dim, no ARCH bump) over retraining
  at `N ∈ {5, 20}`; compare useless-turn-rate + ELO.
- **Belief-toggle × material-PBRS independence (`belief_ablation_pbrs_interaction_test.py`).** Coordinate
  with the shipped belief-toggle coupling (`design_reward_switching.md` §6.1, `project_belief_toggle_flags`):
  the `enabled_beliefs` mask zeros disabled belief obs features and makes `Φ_belief` + the re-gate read 0.
  **`Φ_mat` is material (HP/alive), NOT belief-derived — it must stay ACTIVE under belief ablation**, as
  must the no-progress clock. Assert: with beliefs ablated, (i) `pbrs_belief` = 0, (ii) `pbrs_material`
  ≠ 0 and unchanged, (iii) the no-progress penalty fires unchanged. Prevents a copy-paste bug where the
  new material PBRS inherits the belief-ablation zeroing it must NOT have; the `ModelVersion` value-check
  must distinguish the two potentials' enable state.

### 7.6 Smoke

`--debug --steps 10000` → `[ModelVersion] Round-trip smoke test PASSED` (new config keys: the ARCH bump,
the two PBRS fields, `--use-popart` if paired), episodes finish, no NaN, the gamma assertion passes.

---

## 8. Open questions + retrain-class notes

1. **Front-loading shape (§4.2):** FLAT recommended; LOG fallback. Decide at run-time A/B if "nibbling"
   appears.
2. **`stall_tax` gentle-retain (§4.3, revised):** keep a soft absolute-turn term alongside the progress
   clock — the clock is offense-centric and cannot pressure *defensive* stalls (heal/Protect wars), which
   the edge-case analysis (§4.1.2) exposed. Re-tune `stall_tax` to its intended ~−10 ceiling. Open: the
   exact gentle re-tune (start turn / ramp / clamp).
3. **`MAT_ALIVE_WEIGHT` (§2.4):** 1.25 from the stated invariant. Resume-immutable hparam
   (`--mat-alive-weight`), value-checked by `ModelVersion` like `--pbrs-risk-weight` / `--vf-coef`.
4. **`se_switch` gate (§3, #25):** drop the once-per-matchup gate and fold the offensive-threat tilt into
   `pivot_damage`, vs keep a flat `se_switch`. Recommend fold (avoids the `_last_opp_seen_by` residual).
4b. **`status` PBRS — tempo-status weight/guard (§2.7):** `Φ_status` is **adopted** (the inflict/receive
   credit is a material bridge, same as hazards). The only open part is the heterogeneity: paralysis /
   sleep / freeze are non-damaging tempo, only diffusely material, so `Φ_status` bridges a noisier signal
   for them than for Toxic/burn. The open decision is purely the **hedge** — pre-register
   status-application-rate-must-not-collapse (§7.4); if it does, restore a small standing bias for the
   non-damaging statuses only. Mechanics are settled; this is the one judgment call.
5. **Wish/Future-Sight (§5.3):** gated on fork support (presence + turns-until certain; amount only if
   resolvable) + FPS headroom. Deferrable — does not gate the Markovian-reward work.
6. **PopArt pairing (§6.3):** recommend `--use-popart` for this run; not strictly required given
   `Φ_mat(s_0) ≈ 0`, but the value-scale guards (§7.4) decide.
7. **Material is always-on (RESOLVED — no flag).** `Φ_mat` is **not** gated behind a flag: we have decided
   to just fix the clutch-vs-dominant bias. The default run *is* the material-PBRS run (§1.3). So
   `--use-material-pbrs` is **dropped**; material lives in the PBRS registry class unconditionally.
8. **Retrain-class:** obs change (the clock scalar) is ARCH-bumped + golden-regenerated + benchmark-gated
   (§5.5); reward-value changes are retrain-class, no ARCH bump; both land in one fresh run. PBRS leaves
   the optimal policy unchanged, so the effect is a *cleaner, faster* material/credit signal + a legible
   anti-stall — not a new equilibrium. **Resume-immutable, fresh-run-only, `ModelVersion`-value-checked
   flags** (recorded in `model_config.json`, same machinery as `--vf-coef`): **`--bias-additivity`**
   (default `1.0`, §1.2 — the per-run BIAS additive↔telescoping knob, NOT annealed within a run),
   `--mat-alive-weight` (default `1.25`), `--no-progress-penalty` (+ shape). The bias-redesign enable
   (clock-replaces-anti-spam + reframes, §1.3) is a separate staged toggle, also resume-immutable.

---

## 9. Adversarial-review findings ledger

This design was red-teamed before being written. The corrections folded in above:

| ref | severity | finding | resolution |
|---|---|---|---|
| RT-1 | blocker | Φ updates per **decision window**, not game-turn; a faint splits a turn into ≥2 updates | §1, §2.3 re-indexed to decision windows; falsifier iterates `states.npz` rows |
| RT-2 | blocker | terminal window: `Φ_mat(post-win) ≈ +19.5` must be zeroed via `is_terminal→0` or it becomes a +19.5 dominant-win bonus | §2.3 walked-through proof + the **first** guard test (§7.1) |
| RT-3 | blocker | "`+30` propagates to the sacrifice" is backwards under `gae_lambda=0.80`; faint pain is ~43% lighter at full HP | §2.4 corrected argument (immediate `F_mat` density) + the faint-rate guard (§7.4) |
| RT-2 | blocker | progress damage floor is source-agnostic — Sandstorm/Leech Seed reset the clock free | §4.1 — our-action-attributed damage (`our_damaging_event`), not net `opp_hp_delta` |
| RT-2 | blocker | obs/reward off-by-one (env runs `embed_battle` before `calc_reward`) | §5.1 — update the clock at `record()` (embed) time, HP-tracker precedent; obs always fresh |
| RT-5/6 | major | `Φ_mat` jumps on opp reveals (%-based, revealed-only HP) + large `Φ_mat(s_0)` variance feeds the value-loss target | §2.2 — compute over **declared team size**, unrevealed = full-HP → no jumps, `Φ_mat(s_0)≈0` |
| RT-4 | major | Explosion deletion could nuke `explosion_block`; "old terms cancel" framing stale | §2.5 — delete only the `+2.0` line, keep the nesting; derive cancellation from `Φ_mat` |
| RT-7 | major | `pbrs_material→pbrs_belief` rename is a recorded-schema migration (eval_traces/prober/TD-residual) | §6.1 — branch the falsifier on trace vintage; keep both fields distinct |
| RT-2 | major | forced-switch + trapped-wall windows wrongly taxed | §4.1 — no-op forced windows; gate the charge on a legal switch |
| RT-8 | minor | `PBRS_GAMMA==model.gamma` genuinely unguarded; can't be a constructor assert | §7.1 — thread `model.gamma` post-build; assert there + in roundtrip |
| RT-9 | minor | `MAT_ALIVE_WEIGHT` from "midpoint feel"; symmetric alive term raises KO credit | §2.4 — from stated invariant (1.25); framed as credit-density, not aggression bias; guard pre-registered |
| RT-R3 | minor | `switch_base` spam-gate keys on hidden `last_switch_turn` | §3 #18 — drop the spam-gate; added to the residual-history audit |
| RT-R2 | minor | `stall_tax` integral is −21.3, not −10 (comment wrong) | §4.3 — corrected; re-tune to the intended ~−10 (kept gentle, not replaced) |
| edge | — | defensive/support actions (Recover/Aromatherapy/Rest/Wish/Focus Punch) wrongly taxed by an offense-only predicate; `stall_tax` can't be fully replaced | §4.1.2 freeze-if-Φ-priced rule + §4.3 keep a gentle `stall_tax`; fixed-damage `futile_attack` BP-gate bug noted |
| owner | — | **reward registry + bias-additivity flag added; material resolved as always-on** | §1.1 registry (3 classes, one fold loop, breakdown derived); §1.2 `--bias-additivity` (accumulate-refund, default 1.0); §1.3 single-variable default run; `--use-material-pbrs` dropped (PBRS class, always-on); `spikes`/`status` reclassified BIAS (telescope at `λ→0`); §6.1 always-on `Φ_total = Φ_mat+Φ_belief`; §6.2 flag replaces annealing; §7.0a registry/no-op/blend tests + §7.1a single-variable replay |
| RT-R3 | nit | stale `constants.py:116–121` comments | §5.5 — fix in the same pass |
