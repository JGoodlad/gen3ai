# AI v6 — Differentiable Damage Operator ("compute the physics, learn the belief")

A **fixed, differentiable damage calculator that runs in the GPU forward pass**, fed by the
belief's soft species/move distributions, emitting per-candidate damage as tokens the
transformer attends over — instead of computing damage on the CPU (where the belief isn't
visible yet) or hoping the trunk learns it (it doesn't).

> **Status:** design + de-risking proof complete (2026-06-14); **NOT built into the model.**
> **Class:** L3 amortized-anticipation lever (`designs/research_state/`) — it is *computed
> physics*, no search, no sim call at inference. Sibling to
> `design_latent_predictive_representation.md`.
> **Scope of THIS doc:** the **damage-computation math**. The belief / latent representation
> that *feeds* it (species role-geometry, continuous-vs-discrete moves, the un-revealed
> inference learnability gate) is **out of scope here** — treated as an input interface;
> see §11 and the hidden-team-belief lever.
> **Build posture:** the riskiest engineering question is already answered by the proof in §2
> (byte-exact + differentiable). What remains is encoding more physics, tier by tier (§10), and
> a hard *belief-helps* gate that governs whether any of it ships (§10, GATE).

---

## 1. Motivation & framing

### 1.1 The phase split you can't compute around

Today damage/effectiveness is computed in the **CPU observation-build phase** — `state_encoder.encode`,
which `trainer_turn_benchmark` clocks at ~80% of all our CPU (~88% counting the rest of obs build)
— and it runs in the **env-worker subprocesses**. The policy forward runs on the **GPU in the main
process**. The belief is a *model output*: it does not exist when the CPU builds the obs, and it
lives in a different process. So:

> A deterministic CPU damage calculator **cannot** see the model's belief within a single
> decision. Computing "damage against the believed (hidden) mon" with the existing CPU calc would
> require a cross-process round-trip and a second forward — a two-pass architecture that loads the
> exact resource we are short on.

### 1.2 The trunk does not learn damage magnitude (measured)

A representation probe (2026-06-14, ai_v5_11 ~52M ckpt, 400 attacker×defender actives, ground-truth
via a verified gen3 calc) found the trunk recovers **outgoing** expected-damage *worse than the raw
obs ingredients*:

| representation | R²(best-move dmg frac) | AUC(is-KO) |
|---|---|---|
| pi / vf / active-token (internal) | 0.30–0.32 | ~0.80 |
| **raw obs ingredients (control)** | **0.42** | **0.885** |

The transformer composes a *partial* matchup signal (pre-transformer 0.12 → 0.30 by attending to
the defender) but the final representation is **lossier than its inputs** — it compresses
damage-magnitude away. This reproduces the earlier "damage-spread r²≈0.06" finding on the outgoing
side. The pattern across probes: the trunk **preserves low-dimensional categorical structure**
(move role, type) but **compresses away high-resolution continuous magnitudes** that require
multiplicative composition (BP × STAB × effectiveness × stat-ratio).

### 1.3 The principle

Three different kinds of object, each handled the right way:

- **Epistemic — *what the opponent has*** (hidden identity / moves / item / spread): genuinely
  uncertain, *reducible* by inference → **LEARN it** (the belief; out of scope here).
- **Deterministic — *the damage / effectiveness it implies*** (a known function of identity +
  type chart + stats): → **COMPUTE it** (this op). Known physics belongs as a fixed/differentiable
  op, not a learned approximation the trunk fumbles.
- **Aleatoric — *the dice*** (damage roll, crit, secondary procs, accuracy): irreducible → represent
  as **distributions**, never "believe" it away.

> Belief predicts *identity*; damage is a *function* of identity; the dice are *distributions*.
> The op exists to do the middle one, on the GPU, fed by the first.

### 1.4 Systems: it uses the resource we have spare

py-spy / benchmark posture (`designs/research_state/`, `project_throughput_profile`): the rollout is
latency-bound, the **GPU is ~86% idle**, and **CPU obs-build is the bottleneck**. Putting the calc
on the GPU therefore (a) uses the free resource, (b) stays single-pass, and (c) is differentiable, so
gradients flow back into the belief head. A handful of extra damage tokens costs almost nothing on an
idle GPU.

---

## 2. De-risking proof (DONE — both tests pass)

Proof code (worktree-isolated): `gen3_damage_calc_torch.py` + test; mirrors the numpy oracle
`src/main/gen3_damage_calc.py`.

- **Byte-for-byte equivalence (float64).** 600 real OU `(attacker, move, defender)` triples, torch
  float64 vs the numpy oracle: **max abs diff 0.0, max rel diff 0.0, 600/600 exact.** All five hard
  type-immunity cases (Ground→Flying, Normal→Ghost, Electric→Ground, Ghost→Normal, Psychic→Dark)
  return exactly `0.0` on both sides. float32-CPU drift ~2e-7 (irrelevant on a [0,1] fraction); the
  on-GPU number was not measured because a live training run owned the card (correctly not touched) —
  identical device-agnostic kernel, same float32 class, and the op runs on GPU in production anyway.
- **Differentiable through the belief.** A soft distribution over 8 candidate defenders
  (`requires_grad`) yields a finite, nonzero gradient (L2 ≈ 0.10 on the logits). Pushing belief mass
  toward bulky/resistant defenders lowers expected damage, toward frail ones raises it — a usable
  training signal into the belief head.
- **The Jensen gap (the load-bearing finding).** Damage-of-**average**-inputs (mean-field,
  blend-then-compute) = **0.894** vs **average**-of-damage (marginalized, `Σ P(s)·dmg(s)`,
  simulate-then-average) = **0.615** → a **0.28 systematic overestimate**. Damage is nonlinear (the
  `1/Def` term, the multiplicative type product, bimodal immunity), so blending inputs first *lies*.
  **Never blend-then-compute.** This drives §4.

**Verdict:** "can we do this?" is *proven yes*. The remaining work is encoding more physics (§10),
which is bounded — a vectorize-and-soften of an existing spec (§3.1), not a rebuild.

---

## 3. The core math

### 3.1 Reuse, don't rebuild

There are four existing damage implementations; the **side-neutral `damage_belief.py`** (currently
on branch `claude/outgoing-ko`) is the de-facto spec — pure numpy, computes both incoming and
outgoing, and already applies essentially the full edge-case matrix (§7). The port is
**vectorize + soften**, not a rewrite. The live shipped incoming-only core is
`src/agents/observation/incoming_damage.py` (`gen3_incoming_crit_split_v1`). The effectiveness
primitive everyone shares is `agents/gen3_mechanics.py:effective_multiplier_by_types` / `_CHART`.

### 3.2 The gen3 formula (L100, max-roll)

```
core = ((42 * BP * A) // Def) // 50 + 2      # 42 = floor(2*100/5 + 2)
mod  = type_eff * weather * STAB * screen * burn
dmg_max = int(core * mod)
```

- **Type split.** The physical/special class is set by the **move's TYPE** (the gen3 type-split),
  NOT a per-move category field: `{NORMAL, FIGHTING, FLYING, GROUND, ROCK, BUG, GHOST, POISON, STEEL}`
  are physical (Atk vs Def); the rest are special (SpA vs SpD).
- **Type effectiveness** is an `[18, 18]` tensor `T[def_type, att_type]` (the JSON chart, uppercase).
  Dual-type defender = product of the two rows. The **8 immunities are the `0` entries** — they fall
  out of the product, no branch.
- **STAB** = ×1.5 gate iff the move type is one of the attacker's species types.
- **The roll** is **16 discrete values** `R ∈ [85,100]`; `damage(R) = floor(dmg_max · R/100)`. Mean
  roll ≈ 0.925.
- **Crit** = **×2** (gen3, not ×1.5), **ignores screens and the attacker's negative / defender's
  positive boosts** — so a crit is **not** simply `2 × no-crit` (behind a screen it can be more). It
  must be its own computed value. Base crit rate ~1/16, **higher (~1/4) for high-crit moves**
  (Slash/Crabhammer/Razor Leaf/Aeroblast) → crit probability is move-dependent.

---

## 4. Do NOT collapse — feed per-candidate tokens to attention

The Jensen gap (§2) is not an argument to "collapse correctly" — it is an argument against
*premature* collapse. The cleanest design does not emit a single expected-damage scalar at all.

### 4.1 The reframe

`E[damage]` is a *choice*, and a lossy one (it discards the distribution's shape: variance,
bimodality, the "could be immune" mass). **Attention is a learned aggregator** — `softmax(w)·values`
*is* a weighted sum, i.e. marginalization. So feeding per-candidate damages + the belief as weights
**recovers** `Σ P(s)·dmg(s)` *and* does better: context-dependent aggregation (the value head can
weight toward the OHKO tail; the policy toward the mean). Pre-collapsing does the model's *strong
suit* (aggregation) for it, rigidly.

### 4.2 The division of labor (the principle)

> The trunk is **bad at physics** (§1.2) but **good at aggregation** (attention is its native
> operation). So **compute the physics it can't (per candidate), feed it raw, let attention
> aggregate.** Don't pre-aggregate — that hands off the model's strength and bakes in the Jensen
> bias.

### 4.3 The only real constraint: cardinality

The joint belief is ~400 species × movesets — you cannot feed the Cartesian product. So "feed raw"
in practice = **top-k candidate tokens per hidden slot**, tagged with belief `P`. The axis is not
collapse-or-not, it is *how much*: **scalar → top-k → full**; **top-k is the sweet spot** (keeps
bimodality, bounded). Recommended **hybrid**: top-k raw tokens (expressive) **+** an optional cheap
pre-computed `E[damage]` / `P(KO)` summary scalar as a **guaranteed-correct strong-prior anchor**
(helps sample-efficiency and gives the value head a direct KO-tail read without re-deriving it each
forward).

---

## 5. The per-candidate token

Each top-k candidate `(hidden slot s → believed move m)` contributes one token:

**Damage distribution (computed by the op):**
- `mid` — the mean-roll damage fraction.
- `min`, `max` — the roll band (`0.85`–`1.0`×dmg_max). The **KO-margin reliability**: if remaining
  HP ≤ `min` → guaranteed KO; HP > `max` → can't KO; in between → roll-dependent. (Equivalent to
  `mid ± half-range`; `(min,max)` is the directly-consumable form for the HP comparison.)
- `crit_value` — computed separately (ignores screens/boosts; not `2×mid`).
- `crit_prob` — move-dependent (high-crit moves ~1/4 vs base 1/16).

**Effect flags (looked up, §6):** `status` type + `secondary%`, `boost`, `hazard`, `flinch%`,
`recoil`/`drain` (these two are damage-derived → computed), `substitute`, etc.

**Belief weight:** `P` (the candidate's probability — the natural attention weight).

> Note this is the **raw distribution**, not a pre-collapsed `P(KO)` — consistent with §4. The
> remaining step, "is `damage ≥ remaining HP`", is a **threshold/subtraction** — far simpler than the
> multiplicative physics the op already did — so the model can do it from these values + the (known)
> HP, even though it is bad at composing damage from scratch.

---

## 6. Side effects — provide flags, learn value (do NOT simulate)

The op computes **damage magnitude**. Side effects are a different object and are handled by
*provision + learning*, not by the op:

- **Categorical effects** (Toxic/Will-O-Wisp/Thunder Wave status, Body Slam's 30% paralysis, Spikes,
  Dragon Dance's +1, Rock Slide flinch): **provide the flags** (already in the move data / obs), and
  **let RL learn their value.** The move-role probe showed the trunk is *good* at categorical move
  structure precisely because those flags are provided. The *value* of a status or a setup move is
  strategic and **sequential** — exactly RL's job; computing it would mean simulating future turns,
  i.e. the MuZero/world-model/search path the project **ruled out**. Do not turn the op into a turn
  simulator.
- **Damage-derived effects** (recoil = ⅓·dmg, drain = ½·dmg): these *are* functions of the damage
  the op already computes → **compute them in the op** (free byproducts).
- **Hidden moves:** the effect flags arrive with the **move-id belief** (predict the discrete move →
  look up its flags), the same path as its damage. "Simulate the moves we haven't seen" covers side
  effects too — each candidate token carries damage *and* effects.

---

## 7. Edge-case matrix

Verified against the four data files (`gen3_type_chart.json`, `gen3_abilities.json`,
`gen3_items.json`, `gen3_moves.json`). `lifeorb` confirmed absent (no Life Orb in gen3); the chart
includes a dead `FAIRY` row/col (no gen3 mon has it).

### 7.1 CLEAN — pure tensor ops on known board state
Type effectiveness + the 8 immunities (one `[18,18]` chart tensor); STAB; **weather** (Rain
×1.5W/×0.5F, Sun opposite); **screens** (Reflect halves physical, Light Screen special; crit
ignores); **burn** (×0.5 physical); **crit** (×2 mixture, screen/boost-ignoring). All fixed
multiplicative gates keyed on observed state.

### 7.2 MESSY — gates / piecewise / replacement branches (attribute KNOWN)
- **Ability immunities/resists** → an `[ability × move-type → mult]` table: Levitate (Ground=0),
  Volt Absorb (Electric=0), Water Absorb (Water=0), Flash Fire (Fire=0), Thick Fat (Fire/Ice=0.5).
- **Wonder Guard** → a piecewise gate *after* the type lookup: `mult ×= [type_eff > 1]` (only
  super-effective lands).
- **Stat-mod abilities** → scalars on the Atk/Def input: Huge/Pure Power (×2 Atk), Hustle (×1.5),
  Guts (×1.5 when statused, + cancels burn), Overgrow/Blaze/Torrent/Swarm (×1.5 at ≤⅓ HP), Marvel
  Scale (×1.5 Def when statused), Intimidate (−1 Atk via the boost table).
- **Sturdy** → gates **OHKO moves only** in gen3 (not the gen4 "survive at 1 HP").
- **~24 formula-breaker moves** → an explicit `move-id → handler` enum (the data has no category
  tag): fixed-damage (Seismic Toss/Night Shade = level, Sonic Boom 20, Dragon Rage 40, Super Fang
  ½HP) — bypass Atk/Def/STAB/crit but **still respect type immunity**; OHKO (Fissure/Horn Drill/
  Guillotine/Sheer Cold) → a **separate KO-probability output**, not a damage value; variable
  (Low Kick by weight, Flail/Reversal by HP, Magnitude, Eruption/Waterspout = `150·hp_frac` —
  smooth, Return/Frustration ≈ 102); Counter/Mirror Coat (reactive, function of damage taken);
  multi-hit (price `E[n_hits]`, ~3.0 for the 2–5 spread; Triple Kick explicit).
- **Hidden Power** → the one move whose **type AND BP come from IVs**. `P(hiddenpower)` alone is
  insufficient; it needs a **type/BP rider** from the IV belief (`gen3_hidden_power_priors.json` +
  the HP tracker + the exact `gen3_hidden_power(ivs)` decode already exist).

### 7.3 BELIEF-COUPLED — the attribute is itself uncertain (hidden mon)
Gen3 has **no team preview**, so a hidden defender/attacker's **ability / item / spread / type** are
distributions. The op **marginalizes** over the belief (§4) and the discrete cases stay **bimodal**:
a Ground move into an unrevealed mon is *full damage OR 0* (Levitate), not "0.5× effectiveness." Keep
the mean **and** the tail (per §5); never smear the bimodality into a misleading mean. (Item note:
species-locked items — Thick Club/Light Ball/Soul Dew — collapse to deterministic once species is
known; Choice Band ×1.5 Atk is the main free-floating one; none of the existing calcs apply items
today — a design choice, "let the model learn the CB effect from the obs item flag".)

---

## 8. Differentiability — the two non-smooth spots

Only two pieces of the physics are not naturally differentiable:
1. The **integer floors** in `gen3_damage_max`.
2. The **16-roll `P(KO)` step function**.

Handling: the **byte-exact forward** can keep the floors (`torch.floor` matches numpy `//`) and the
exact 16-roll count — useful for a faithful feature. The **gradient path** uses a smooth surrogate:
straight-through floors (or simply omit the inner floors — a sub-HP error) and a **sum-of-shifted-
sigmoids** for `P(KO)` over the 16 rolls. The proof's differentiable path already used the smooth
un-floored formula on the belief side; the equivalence claim lives on the integer path.

---

## 9. Architecture integration

- **Attach point:** `Gen3FeaturesExtractor.forward_internal`, **after `MoveBelief` (~`:1126`) and
  before `CLSPool` (~`:1130`)** — where the belief logits are materialized.
- **Inputs (believed-opp side, leak-free):** `softmax(self.last_belief_logits["species"])`
  `[B,6,400]` and `sigmoid(self.last_belief_logits["moves"])` / `self.last_move_belief_logits`
  `[B,6,400]`.
- **Inputs (known-our side, from `ctx`):** `ctx.{species_ids, all_move_ids, all_move_type_ids,
  type1_ids, type2_ids, item_ids, hp_and_active, our_active_idx, opp_active_local}` + the base-stat
  slice of `ctx.pokemon_part`.
- **Lookup buffers** (`[400, …]`, indexed by national-dex `num` — the same axis the belief logits
  live on, since `species_id == entry["num"]` is fed directly as the embedding row): base stats
  (`gen3_data.species`), BP + move-type (`gen3_data.moves`), the `[18,18]` chart
  (`gen3_data.type_chart`). **DATA GAP:** species→types is **not** in the `gen3_data` facade today
  (`SpeciesData` has only `num`, `name`, `base_stats`); types live per-mon in the obs via
  `TypeEncoder`. Extend `tools/pokemon_data_extractor` to emit `types` into `gen3_species.json` and
  add the field to `SpeciesData`.
- **Output → BOTH heads.** Emit a small **set of per-candidate damage-annotated tokens** (top-k per
  slot, §4–5) and inject via the existing `hidden_opp_belief` append seam in
  `ProjectionAssembler.forward` (`:961-963`, appended last; projection dims auto-discovered by the
  dummy forward). Both heads: the **policy** needs OUR→THEIR damage to pick the KO move; the **value**
  head needs THEIR→OUR damage to price the incoming-KO tail (the `value_active_readout` motivation).
- **Leak-safety (verified).** The forward reads **only** `obs["observation"]`; the privileged label
  keys (`belief_species`, `belief_moves`, `known_moves`) are read **only** in the PPO loss
  (`instrumented_ppo.py:489-493, 510-513`). The op consumes the **predicted** (non-privileged) belief
  → leak-free; gradients into the belief only sharpen an already-leak-free prediction. A new op is a
  structural change → **bump `ARCH_SIGNATURE`**.

---

## 10. Staged plan + the gate

- **T0 — proven kernel.** Type-split, STAB, effectiveness + the 8 immunities (the `[18,18]` chart),
  **marginalized** over the species belief (→ stats/types) and move belief (→ BP/type) as **top-k
  per-candidate tokens** injected into both heads. (This is what §2 validated, plus the marginalize/
  feed-raw plumbing.)
- **T1 — clean gates + free byproducts.** Crit (×2 mixture), weather, screens, burn, recoil/drain.
- **T2 — `P(KO)` + abilities.** Soften the floors + 16-roll step (§8); the `[ability × move-type]`
  table + Wonder Guard gate + stat-mod abilities, **marginalized over the believed-ability
  distribution** (the bimodal "could be Levitate" case — where §4 pays off).
- **T3 — the messy tail.** Items (Choice Band etc.), the formula-breaker `move-id → handler` enum,
  the Hidden Power type/BP rider, and the species→types facade extension.

> **GATE (do not skip).** The op makes the belief **actionable**; it does not make the belief
> **useful.** Whether the full belief→damage stack earns a retrain still hinges on the
> hidden-team-belief gate: *can the trunk infer an unrevealed mon's role/threat above the usage
> prior, and does acting on it move the loss-crater share + win-rate?* **Run that gate probe before
> T0.** This op is downstream of a belief worth feeding it.

---

## 11. Out of scope (this doc)

- **The belief / latent representation** — the species role-geometry (continuous, soft distribution),
  the discrete multi-label move belief, continuous-vs-discrete by *blur-tolerance*, the BYOL/
  contrastive question, and the **un-revealed-inference learnability gate**. This doc treats the
  belief's output (`softmax(species_logits)`, `sigmoid(move_logits)`, the HP-type rider) purely as an
  **input interface**. See the hidden-team-belief lever / its own doc.
- **MuZero / search / turn-simulation.** Ruled out (`designs/research_state/README.md`). This op is
  **1-ply amortized physics**, fed by a belief — never a planner. Side-effect *value* (status, setup)
  is RL's job, not the op's (§6).

---

## 12. References

- Proof: `gen3_damage_calc_torch.py` (+ test) — byte-exact vs `src/main/gen3_damage_calc.py`;
  differentiable; the Jensen measurement.
- Spec to port: `damage_belief.py` (branch `claude/outgoing-ko`, side-neutral full physics);
  live core `src/agents/observation/incoming_damage.py`; effectiveness primitive
  `agents/gen3_mechanics.py:effective_multiplier_by_types`.
- Probe (trunk can't learn outgoing damage): the move-token + outgoing-damage representation probe,
  2026-06-14.
- Attach point + leak-safety: `src/agents/model/features_extractor.py` (`forward_internal`,
  `ProjectionAssembler`), `src/agents/training/instrumented_ppo.py` (label-only-in-loss).
- Sibling design: `designs/ai_v6/design_latent_predictive_representation.md` (the value-path-leak
  STOP-flag there is why this op stays leak-free and injects the *predicted* belief only).
- Principle / posture: `designs/research_state/README.md` (no search on the model), the
  hidden-team-belief lever, `project_gpu_damage_op` / `project_belief_latent_role_probe` (memory).
```
