# Design — Iterative damage refinement: recompute the GPU damage as tokens are enriched (ai_v6)

**Status:** BUILT (this worktree). Retrain-class, flag-gated, **OFF byte-identical**.
`MODEL_CONFIG_VERSION` — **as landed: v33** (sequenced after the already-shipped reattend=v31 + prefuse=v32);
`ARCH_SIGNATURE` unchanged.
Tag: `gen3_iterative_damage_v1`. Flag: `--damage-refine-rounds N` (0 = off). Requires `--damage-op`.

> **As-built note.** The build reuses the validated `_damage_rolls` physics (no reimplementation) and
> simplifies the injection vs the original sketch: the residual is a **single zero-init `refine_proj`
> Linear** (NO `refine_norm` — a LayerNorm of a ~0 residual is ill-conditioned and breaks identity-at-init;
> post-LN transformer layers renormalize downstream anyway). The lean kernel is `discrete_incoming` returning
> `[B, 6, _DMG_REFINE_FEATS]` = `[phys_high, spec_high, phys_pko, spec_pko]` (per-channel worst-case damage +
> P(KO), `_DMG_REFINE_FEATS=4`). It requires **only `--damage-op`** (which already pulls in `--move-belief-mode
> revealed|both`), NOT `--move-latent` (the refine summary carries no move-identity latent — that's the top-K
> block's job). `refine_proj` is **weight-tied across rounds**, so its shape is N-independent; the int is gated
> in `check_compatible` regardless (0↔N a state_dict change, N↔M a forward change).

## Motivation (the principle)

The damage physics (BP×stat×eff×roll, the OHKO threshold) is the thing the trunk is **bad at composing**
and the op already **computes** it. The remaining lever is *plumbing*: today the op runs ONCE
post-transformer, so the damage it computes reflects a belief read from the **final** tokens, and the
result is only seen post-attention. Empirically (this run series) feeding computed damage **pre-attention**
helps. This generalizes that: **recompute the (discrete) damage every time the tokens are enriched by an
attention layer**, and feed it back, so each layer attends over physics derived from the *current* refined
belief — physics-in-the-loop, not a one-shot post-hoc read. We do the hard part (compute) and hand it to
the existing aggregator (attention) repeatedly, rather than adding generic attention capacity.

## Mechanism (v1)

`--damage-refine-rounds N` (int, 0 = off; capped at `TRANSFORMER_N_LAYERS`). When N>0, the `TeamTransformer`
runs its encoder layers one at a time (it already iterates a `ModuleList`), and **before each of the first
N layers** a refine step injects a refreshed discrete-damage residual into the 6 our-team token positions:

1. **Belief from the current opp tokens** — `MoveBelief.move_logits(opp_tokens, species_ids, move_ids)` →
   the per-slot POSTERIOR move logits (prior⊕delta + revealed-pinning, NO reinjection — factored out of
   `MoveBelief.forward` so the refinement re-reads the belief without re-running the soft-embed). REUSE the
   existing head; the per-round gradient sharpens it. The opp-active slot's logits drive the op.
2. **Lean discrete damage** — `DamageOperator.discrete_incoming(ctx, move_belief_logits)`: select the top
   `_DMG_REFINE_K` (=8) candidates over `w_all` (real moves ⊕ typed HP, selection DETACHED), compute
   incoming damage for **only those K** vs our 6 mons (the lean per-K mirror of `_damage_rolls`, reusing the
   shared `_rolls` formula) → reduce to `[B, 6, _DMG_REFINE_FEATS]` = `[phys_high, spec_high, phys_pko,
   spec_pko]` (per-channel worst-case max-roll fraction + accuracy-folded P(KO), belief-weighted via the same
   `_chan_max` hard-max). Cost ∝ K (≈8), not C (≈416) — the ~50× cheaper primitive. v1 uses the LEGACY
   de-timid attacker offense (NO spread belief / boost / burn / weather / fixed-damage — the coarse
   refinement signal; the FINAL post-transformer op carries the full physics and is authoritative).
3. **Inject** — `refine_proj` (`Linear(_DMG_REFINE_FEATS, D_MODEL)`, **zero-init**) as a pure residual onto
   the our-mon token positions `tokens[:, 0:TEAM_SIZE]`. Zero-init ⇒ the residual is EXACTLY 0 at init →
   **true identity-at-init**, while the gradient still flows (∂out/∂W = the damage feats). NO LayerNorm on
   the residual branch (a LayerNorm of a ~0 vector is ill-conditioned and would break identity-at-init;
   the post-LN transformer layers renormalize downstream).
4. The next encoder layer attends over the damage-enriched tokens → refined tokens → repeat.

After the loop, the existing path is unchanged: MoveBelief reinjection (final) + the full `[B,6,C]` op
(worst-case anchor + the projection append). So the heavy full sweep runs **once**; the per-round
recomputes are the lean discrete primitive.

## What it reuses / what's new

- **Reuses**: `TeamTransformer`'s `ModuleList` loop (add a `between_layers` callback called before each of the
  first N layers), `MoveBelief.move_logits` (per-round belief — factored out of `forward`), the op's physics
  buffers (CHART/BASE_STATS/MOVE_BP/…) + the shared `_rolls` formula for the lean kernel.
- **New module** (built only when N>0; gated like `opp_belief_cls_k`): `refine_proj`
  (`Linear(_DMG_REFINE_FEATS, D_MODEL)`, **zero-init**). One projection reused across rounds (weight-tied,
  so its SHAPE is N-independent) — the *recompute* is what varies, not the projection. No `refine_norm`.
- **New op method**: `discrete_incoming(ctx, move_belief_logits)` → `[B,6,_DMG_REFINE_FEATS]` (lean,
  top-`_DMG_REFINE_K`-only; no full `[B,6,C]`).
- **Refactor**: `MoveBelief.move_logits(...)` extracted from `MoveBelief.forward` (byte-identical — `forward`
  now calls it then does the soft-embed reinject).

## Differentiability / gradient
- The per-round belief read flows gradient into `move_belief.move_head` **at every round** — so the belief
  is sharpened by the downstream policy/value loss through each refinement (more signal than the one-shot
  post-transformer read).
- The injected damage is w-independent physics (decorrelated); the residual carries gradient into
  `refine_proj`. The belief gradient rides the `move_head` read, not the damage values (same decorrelation
  as the topk block).

## Versioning / safety
- `damage_refine_rounds: int = 0` — STRUCTURAL int (adds `refine_proj`; 0↔N a state_dict change, N↔M a
  forward-behavior change since `refine_proj` is weight-tied), gated in `check_compatible` with an
  unconditional int compare (like `opp_belief_cls_k`). OFF (0) byte-for-byte (no `ARCH_SIGNATURE` bump).
  `MODEL_CONFIG_VERSION` 30→31; `_migrate_config` `setdefault 0`.
- Requires `--damage-op` ONLY (which itself requires `--move-belief-mode revealed|both`, the per-round belief
  source) — enforced at extractor build + CLI. NOT `--move-latent` (the refine summary has no move-identity
  latent). NOT auto-enabled by `--unified-moves` (an explicit A/B lever). PopArt strongly recommended
  (per-round reads add shared-trunk gradient load — same caveat as `damage_reattend`).
- Threaded through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` (the 4
  opp-load sites) + both `extractor_kwargs` build sites in `train_rl_agent.py`.

## Compute (from the measured analysis)
The full op is bandwidth-bound (~1 GB `[B,6,416]` activations, ~2–3 ms/pass at batch 16384). The **lean
discrete recompute is ~50× cheaper** (`[B,6,K]`, K=`_DMG_REFINE_K`=8), so N rounds add ~N × (a cheap kernel
+ the existing layer) — negligible in rollout (GPU idle) and small in the update. The full sweep still runs
once. The per-round physics is the lean top-K, NOT the full `[B,6,C]` — so refinement does NOT recompute the
expensive sweep per round (the user's compute-not-memory priority).

## Deferred (v2+, not in v1)
- **belief-earlier**: v1 reads the move_head per round on the *transformer's* opp tokens (which are being
  refined) — it does NOT relocate the MoveBelief *phase*. A fuller version moves the belief module itself
  earlier. (Separate change.)
- **Per-candidate switch pointer head** (the `damage_reattend` follow-up): a per-bench scoring head reading
  each bench token + its discrete incoming. Orthogonal; layer on after.
- **SpreadBelief per round** (v1 uses the post-transformer spread for the final op only).

## Test plan (BUILT — all green)
`damage_op_test` (6 new): off-builds-no-module + projection dims unchanged by refine ON; requires-damage-op
guard; **identity-at-init** (refine_proj zero-init → ON forward == the same model with the callback disabled,
`torch.equal`); `discrete_incoming` shape + no-opp/fainted gating; `discrete_incoming` **matches the full op's
per-mon channel worst-case** (`last_raw_block` phys_high/phys_pko) for a concentrated belief — the kernel
reuses the validated physics; grad flows to the belief logits (kernel) AND to `refine_proj` (extractor
end-to-end). `snapshot_test` (4 new): int gate reject (0↔N AND N↔M) / accept; kwargs read; v30→v31 migration;
`arch_toggles_from_model` round-trips `damage_refine_rounds`. Full unit suite 2761 passed. Roundtrip smoke +
serverless `--debug --use-showdown-bridge --unified-moves both --damage-refine-rounds 2` → `[ModelVersion]
Round-trip smoke test PASSED` + `Training complete` (the refine forward+backward + save/reload exercised in
the real PPO loop).
