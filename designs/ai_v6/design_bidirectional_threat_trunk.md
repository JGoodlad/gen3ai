# Design — Bidirectional in-trunk threat field (ai_v6)

**Status:** DESIGN (not built). Retrain-class, flag-gated, **OFF byte-identical**.
Target `MODEL_CONFIG_VERSION` — **v36** (sequenced after the per-move matrices = v35);
`ARCH_SIGNATURE` unchanged. Proposed tag: `gen3_bidir_threat_trunk_v1`.

Diagrams (render with `dot -Tsvg`): [`threat_now`](threat_now.svg) · [`threat_future`](threat_future.svg)
· [`threat_delta`](threat_delta.svg) · the integrated **full-model** diagram + the latent-defender /
warm-start / slow-start explainers live in `gen3ai/tmp/` (`model_v36_full`, `latent_expected_defender`,
`warm_start`, `slow_start_recap`). (`.png` siblings for quick viewing.)

**Build = three pieces:** **#1** outgoing residual into the trunk (onto opp tokens, symmetric to incoming);
**#2** expected LATENT defender — keep hidden slots latent, marginalize the belief into expected
multiplier + bulk, **P(KO) nulled for unrevealed**; **#3** probabilistic outspeed for the active matchup.
New toggles: `--threat-refine-outgoing` (reuses `--damage-refine-rounds N`) and `--threat-prob-outspeed`;
both OFF byte-identical.

---

## 1. The goal (owner's words)

> "Make the model's threat — both ours and the opponent's — **dynamic**, fusing **what is known
> (poke-env)** with **what the model believes**: believed *moves* for **revealed** mons, and believed
> *mons* (species/spread) for **unrevealed** mons **without predicting their moves**. Infuse this into
> the **trunk** so attention can do its work."

The load-bearing word is **trunk**. The verified data-path probe (below) shows today's threat physics
is computed but lands almost entirely at the **projection heads** (post-pool, no attention). The CPU
incoming-damage obs only reaches the transformer as **one diluted global token** (mashed in with
weather/hazards/screens), never as a per-mon signal. So the transformer cannot *attend over* the
threat field — exactly the lever this design adds.

## 2. Target decomposed into four legs — and where each stands (verified)

| Leg | What it is | Current state (file:line) |
|---|---|---|
| **1. Incoming · revealed · known+believed moves** | opp active's *known* moves ⊕ *believed* unseen-slot moves → damage to our 6 | **✓ exists.** `MoveBelief.move_logits` pins revealed (`_REVEAL_LOGIT`) + fuses Smogon prior ⊕ learned delta (`features_extractor.py:1209-1230`); the op reads it for the opp active (`:2380`). **In-trunk** via the `--damage-refine-rounds N` between-layers `refine_cb` → `discrete_incoming` → residual onto OUR tokens `[0:6]` (`:3197`, `refine_proj` zero-init `:3005-3014`). |
| **2. Outgoing · revealed** | our *known* moves → revealed opp mons | **⚠ computed, heads-only.** `_outgoing_matrix` (`:1890-1986`) prices our 4 known moves vs opp 6, revealed types + SpreadBelief bulk — but the block only ever lands at `ProjectionAssembler` (`:2712-2714`). **No trunk path.** |
| **3. Outgoing · unrevealed** | our moves → *hidden* opp mons from species/spread belief, no move prediction | **✗ missing.** Unrevealed columns **hard-zeroed**: `revealed = (~ctx.opp_believed_mask)`, `def_gate = revealed * (hp>0)` (`:1951-1952`, `:1984`). Believed **bulk** *does* flow (`SpreadBelief` covers all 6 slots, `:1939`), believed **types** do **not** — the op reads real observed types `ctx.type1_ids[:, opp]` (`:1949`), which are unknown for hidden mons. |
| **4. Whole field in the trunk** | both directions on the right tokens | **◐ half.** Incoming residual injects onto OUR tokens ✓; **nothing** injects onto OPP tokens. The refine loop and `--damage-reattend` are both incoming-only (`discrete_incoming` → `[B,6,4]`, `:2237-2253`; reattend projects incoming rows onto our tokens then re-attends, `:3286-3291`). |

**Net:** Leg 1 is done; Leg 3's *bulk* half is done. The genuinely missing work is two bounded pieces.

## 3. The two new pieces

### A — Outgoing-in-trunk injector (the symmetric half of Leg 4)
Mirror the incoming refine onto the **opp** token slice:
- **New op method `discrete_outgoing(ctx, …) → [B, TEAM_SIZE, _DMG_OUT_REFINE]`** — a lean per-opp-mon
  summary of "how hard our active hits this opp mon" (the attacker is fixed = our active, only 4 known
  moves to iterate, so it is even cheaper than `discrete_incoming`). Reuses the validated `_rolls`
  physics primitive; no reimplementation.
- **New `outgoing_proj` = `Linear(_DMG_OUT_REFINE, D_MODEL)`, zero-init** → true identity-at-init,
  gradient still flows. Injected as a pure residual onto `tokens[:, TEAM_SIZE:2*TEAM_SIZE]` (the opp
  slice, `_their_token_slice`, `:797`).
- **Same `between_layers` callback** the incoming refine already uses — one extra branch, called before
  each of the first N layers. After injection, attention reasons over *our threat to each opp mon* as a
  property of that opp's token (e.g. "this opp mon is in KO range → safe to stay in; that one walls us").

### B — Expected LATENT defender (Leg 3, owner-simplified: keep latent, P(KO) NULLED)
Do **not** decode hidden slots to a discrete species. Keep them **latent** and marginalize the belief
into **expected** quantities — fully differentiable, no argmax, and (owner decision) **P(KO) is nulled
(=0) for unrevealed defenders** (you almost never OHKO a full-HP switch-in, so the OHKO bit isn't a
useful signal and the threshold marginalization isn't worth the complexity). We keep only the expected
**magnitude**.

The marginalization folds to **two precomputed buffers × one matmul** with `P(species)`:
- **`SPECIES_EXP_MULT[n_species, 19]`** (non-persistent, data-built) = `typechart(types(s), t) ·
  Σ_a P(a|s)·ABILITY_DAMAGE_MULT[a, t]` — the **expected damage multiplier** of species `s` vs each
  attacking type `t`, *with ability immunity folded in* (Levitate→Ground 0×, Water/Volt Absorb→Water/
  Electric 0×, Flash Fire→Fire 0×; `P(a|s)` from `gen3_ability_priors.json`). Then
  `E[mult vs move type t] = Σ_s P(s)·SPECIES_EXP_MULT[s, t]` — a single `P(species)[B,6,n_species] @
  SPECIES_EXP_MULT[n_species,19]` matmul, gathered by move type. The gradient rides `P(species)`
  (decorrelated — not the damage magnitude).
- **`SPECIES_PRIOR_BULK[n_species, …]`** = the `spread_prior` mean def/spd → `E[bulk] = Σ_s P(s)·bulk(s)`.
  This also **fixes a confirmed gap**: today the hidden-slot bulk is `spread_prior[species_id]` keyed by
  the *unknown sentinel* (`features_extractor.py:1285`), i.e. a placeholder, not belief-weighted.
- **`P(species)`** per hidden slot comes from a factored **`BeliefHead.species_logits(tokens)`** (mirrors
  the `MoveBelief.move_logits` factor), so it is available **both** per-round in the trunk refine **and**
  at the final post-transformer op. Also fixes a second confirmed gap: ability immunity is **revealed-only**
  today (`opp_ability = 0 if unrevealed → no Levitate/Absorb mult`, `:1846`) — the expected-ability fold
  covers hidden mons AND revealed-species-but-unknown-ability.
- **Output cell for unrevealed:** `[low, high, crit, pko=0, type_mult]` — expected magnitude, **pko nulled**.
  No move prediction for the defender (outgoing needs only types + bulk), so "don't predict their moves"
  holds by construction.

> **Incoming stays revealed-only by design.** Threat *from* a hidden mon would require predicting its
> moves — explicitly excluded. So Leg 3 is outgoing-only; nothing new on the incoming side beyond Leg 1.

### C — Probabilistic outspeed (#3)
Replace the point-estimate `p_outspeed` (`our_spe > opp_spe`, hard) with a **soft** `P(our_spe > opp_spe)`
under the spread belief's speed **mean ± std** (`spread_prior` carries `[mean, std]` per stat) — a smooth
CDF, differentiable. Scoped to the **active matchup** (where KO ordering — "do I KO first or eat the hit?"
— actually drives the decision). For unrevealed defenders the point is moot (pko nulled), so #3 is the
revealed/active KO-order lever, not an unrevealed one.

## 4. Mechanism summary (what the diagrams show)

- **NOW** (`threat_now`): incoming reaches the trunk (green refine box on the OUR slice); the op's
  outgoing matrix and the whole `damage_block` only reach the heads (red edge "OUTGOING is HEADS-ONLY");
  unrevealed outgoing columns are zeroed (red box); CPU incoming reaches the heads (flat concat) and the
  trunk only as a dotted "1 diluted global token."
- **FUTURE** (`threat_future`): a second gold `discrete_outgoing` box injects onto the OPP slice
  `[6:12]`; `BeliefHead P(species)` + the new `species→types` buffer feed both the in-trunk outgoing
  residual and the now-priced unrevealed columns of `_outgoing_matrix`.
- **DELTA** (`threat_delta`): NEW = `discrete_outgoing`, `outgoing_proj`, `SPECIES_TYPE1/2`,
  marginalized-effectiveness path; CHANGED = `refine_cb` (incoming → incoming+outgoing), `_outgoing_matrix`
  (hard-zero → belief-priced); versioning 35→36.

## 5. Versioning / safety
- **`threat_refine_outgoing`** (bool) and the belief-priced-unrevealed sub-toggle: STRUCTURAL (adds
  `outgoing_proj`; changes forward), gated in `check_compatible` with an unconditional compare (like
  `damage_refine_rounds` / `opp_belief_cls_k`). OFF (default) byte-for-byte — no `ARCH_SIGNATURE` bump.
  `MODEL_CONFIG_VERSION` 35→36; `_migrate_config` `setdefault` off.
- **Requires** `--damage-op` (the physics + matrices). The belief-priced-unrevealed leg additionally
  requires `--opp-belief-aux-coef > 0` (the source of `BeliefHead` species logits) and `--spread-belief`
  (the believed bulk); guard at extractor build + CLI. The in-trunk outgoing residual reuses
  `--damage-refine-rounds N` as its round count (one knob for both directions) — proposed flag
  `--threat-refine-outgoing` simply turns the opp-side injection on.
- **PopArt strongly recommended** — a second per-round residual adds shared-trunk value-gradient load
  (same caveat as `--damage-reattend` / refine). Soft-warn without `--use-popart`.
- Threaded through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` (the 4
  opp-load sites) + both `extractor_kwargs` build sites in `train_rl_agent.py`.

## 6. Differentiability / decorrelation
- The per-round outgoing read flows gradient into nothing new on *our* move side (our moves are KNOWN,
  no belief — correct, `:1770-1776`); for hidden defenders the gradient flows into `BeliefHead` species
  logits via the marginalized `Σ P(s)·dmg`, sharpening the *species* belief toward KO-relevant
  identities. Decorrelate the belief gradient onto the **probability weight**, not the damage value
  (the same pattern as the top-K block), so the head learns "which species" without the physics
  back-propagating spurious magnitude.
- Residual carries gradient into `outgoing_proj`; zero-init ⇒ identity-at-init ⇒ ON forward == OFF at
  step 0 (a `torch.equal` identity test, as for `refine_proj`).

## 7. Compute
- `discrete_outgoing` is **cheaper** than `discrete_incoming`: the attacker is fixed (our active) and
  there are only 4 known moves × 6 opp defenders, vs the incoming top-K sweep. The full `_outgoing_matrix`
  already runs once post-transformer; the per-round lean kernel is a small add (GPU idle in rollout,
  small in the update). The `Σ P(s)·dmg` marginalization is a `[B, 6_slots, K_species]` gather over the
  type chart — bounded by a top-K_species cap (e.g. 8) to stay cheap.

## 8. The "closest with what we have TODAY" config (no new code)
Until v36 lands, the closest approximation of the goal with existing flags (fresh run, all structural):
- `--damage-refine-rounds 2` — **incoming in the trunk** (Leg 1+4 incoming), physics-in-the-loop.
- `--damage-reattend` — damage-AWARE pooling: the one extra attention pass lets opp tokens see the
  incoming-enriched our tokens (the closest existing thing to attention reasoning over the field).
- `--damage-matrices both` — richest **heads** representation of both directions (outgoing per-(move,opp
  mon) + rich incoming top-K, reusing `--damage-topk 5` as K).
- Keep the CPU incoming obs **visible** (do NOT add `--mask-incoming-damage-obs` yet — that's the
  deprecation endpoint, run it only after the GPU path carries the field into the trunk).

What this still **cannot** do (and v36 adds): outgoing as a direct trunk residual on opp tokens (A), and
outgoing-vs-unrevealed pricing (B).

## 9. Test plan (to build with the feature)
`damage_op_test`: off-builds-no-`outgoing_proj` + projection dims unchanged; requires-`damage-op` guard;
**identity-at-init** (`outgoing_proj` zero-init → ON == OFF, `torch.equal`); `discrete_outgoing` shape +
no-opp/fainted gating; `discrete_outgoing` matches `_outgoing_matrix`'s per-cell high-roll for a revealed
defender (kernel reuses the validated physics); **belief-priced unrevealed** == marginalized
`Σ P(s)·dmg` for a concentrated species belief (collapses to the single-species cell); grad flows to
`BeliefHead` species logits (marginalization) AND to `outgoing_proj`. `snapshot_test`: structural gate
reject/accept; kwargs read; v35→v36 migration; `arch_toggles_from_model` round-trips the toggle. Full
unit suite + roundtrip smoke + serverless `--debug --use-showdown-bridge --unified-moves both
--damage-refine-rounds 2 --threat-refine-outgoing` → `[ModelVersion] Round-trip smoke test PASSED` +
`Training complete`.

## 10. Deferred (v2+, not in v1)
- **Incoming-from-hidden** (would need hidden-mon move prediction — excluded by the goal).
- **Per-bench switch pointer head** (first-class per-candidate switch scoring, the `damage_reattend`
  follow-up) — orthogonal; the opp-token outgoing residual is a board-level enrichment, not a pointer.
- **Choice-lock / item-reveal** sharpening of the believed defender profile.
- The CPU **deprecation A/B** (`--mask-incoming-damage-obs`) — run *after* v36 proves the GPU field
  carries the trunk signal (see §8 and `design_unified_damage_system.md`).
