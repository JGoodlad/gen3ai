---
name: project_opp_hp_immune_bug
description: "Opp Hidden Power read \"immune\"/0-damage in the damage op — FIXED (gen3_opp_hp_type_belief_v1, config v38, worktree, NOT shipped)"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — GIGO fixed and shipped a4aa2bc; the class rule lives in feedback_gigo_order_bugs_asap. Preserved verbatim; nothing below is current.

**SHIPPED (a4aa2bc) + RAN + BEHAVIORALLY VERIFIED 2026-06-20** (ai_v6_11_typed_hp_0619 @34M, prober traces). The immune-GIGO is CONCLUSIVELY GONE in production: on a real decision the op renders `hiddenpower(grass)` (typed num 363, NEVER bare 237) with all 6 our mons taking NONZERO damage and the type-multiplier multiset {2,2,0.5,0.5,0.25,1.0} EXACTLY matching the data-layer type chart (type-correct physics, not merely nonzero). The learned HP-type head holds a genuine per-slot DISTRIBUTION (e.g. grass 41%/rock 38%/ghost 19%) and across 1638 decisions top-1 HP-type spans the full spread (grass 3797…ice 699…electric 92) — it genuinely weights many HP types. No residual HP-special prober weirdness (the model-vs-observability ambiguity is resolved). hptype_acc 0.91 (chance 0.0625), stable. NOTE: `--unified-obs` is ON, so the op is the policy's ONLY damage view → this fix is the sole damage signal, not cosmetic. (The HP run otherwise plateaued ~1880 ELO = genuine convergence, see [[project_value_dist_head_status]] for the next-lever read.)

**FIXED + UNIFIED 2026-06-19** (worktree, NOT shipped — awaiting /gen3ai-ship). FINAL design = the clean
unification (user wanted "make 355-370 typed moves, super clean — the prober has HP-special bugs"):
PULLED FROM MAIN (main had shipped `gen3_typed_hidden_power_ids_v1` = our-side typed HP nums 355-370; FF +
reconciled my opp-belief WIP). Then made HP **16 ordinary typed moves end-to-end**: the op uses the REAL nums
355-370 as candidates (C=n_moves; the synthetic appended-16 expansion REMOVED — value-preserving, every
MOVE_*[355-370]==the appended values, verified incl. gen3 type-based phys/spec), bare 237 masked as the
presence token, the per-type belief `index_add`-scattered onto HP_TYPED_NUMS. ARCH_SIGNATURE BUMPED →
`gen3_opp_hp_typed_candidates_v1` (op forward-math change, out_dim unchanged so not shape-caught). Data-derived
HP_TYPED_NUMS + a THROWING GIGO guard (MOVE_TYPE_IDX[355+j]==HP_TYPE_IDX[j], MOVE_BP[237]==0). PROBER FIXED:
`_topk_move_names` decodes typed nums via the NORMAL move path (hiddenpower(ice)), no HP-special collapse — the
ambiguity is GONE. tri-state `--hp-type-belief off/prior/learned`; learned = HPTypeBelief head (prior⊕delta,
cold-start==prior) + token reinjection (presence-gated expected-type emb → opp token) + CE on agent2's typed
label (leak-safe, opp obs stays 237). Verified: full suite 2889 + 87 op/hp (hp-ice 365 + hp-grass 363 both
surface distinctly) + 248 prober + smoke (acc learning) + live fuzz (3302 HP slots == true type, no-leak).
config v38. (Superseded the earlier appended-16 + HP_BARE_MASK design below.)
ORIGINAL fix (`gen3_opp_hp_type_belief_v1`, config v38).
Shared `DamageOperator._opp_candidate_weights` (single source for all 3 candidate sites): when `hp_type_fix` on,
MASKS the bare num-237 (`HP_BARE_MASK`) + resolves the typed-HP belief = learned posterior ⊕ Smogon
`SPECIES_HP_PRIOR` floor (`build_hp_type_prior`), NARROWED by the obs `hp_probs` effectiveness survivors
(uniform-over-survivors fallback so an off-meta HP never re-collapses to immune). Tri-state
`--hp-type-belief {off,prior,learned}` (off byte-identical; requires --damage-op); `learned` adds `HPTypeBelief`
(prior⊕delta, zero-init→cold-start==prior) + leak-safe CE (`_hp_type_belief_loss`, label from agent2's typed
move-id, training-only `hp_type_label`/`hp_type_mask`, metrics belief/hptype_*). STRING-gated in check_compatible;
no ARCH/obs-dim change. Verified: 14 unit + 2875 suite + smoke (acc 0.1→0.5) + flagless-resume inherits learned +
belief_labels fuzz (4711 live HP slots == agent2 true type, no-leak). 25-agent review caught+FIXED 2 HIGH (CLI
dest≠field flagless-resume FATAL; narrow-renorm under-normalization re-immune). Use: --damage-op --move-belief-mode
revealed --hp-type-belief learned --hp-type-belief-coef 0.05. Presence (w_hp, move belief) × type (this belief) factored.

----- ORIGINAL DIAGNOSIS -----
2026-06-19 (research workflow). The prober "opp moves (matrix)" view showed the opponent's `hiddenpower` as "safe (immune)" / 0 damage vs all 6 of our mons. Root cause was TWO stacked bugs in `DamageOperator` (features_extractor.py):

1. **Opp typed-HP belief is all zeros.** `hp_type_belief = ctx.hp_probs[opp]` (features_extractor.py:2693) ← `HiddenPowerTracker.get_probs(species)` returns `np.zeros(16)` UNTIL the opp actually FIRES Hidden Power + an effectiveness tier is observed (hidden_power_tracker.py:163-173; state_encoder.py:198-206). The Smogon HP prior is used ONLY inside `observe()` as a seed — it NEVER reaches the opp obs. So the 16 typed-HP candidate rows get weight `w_hp·0 = 0` → 0 damage.
2. **Bare num-237 unmasked + dead.** Not zeroed before `w_all = cat([w, w_hp·hp_type_belief])` (features_extractor.py:2708, also :2360, :2536); carries full presence belief `w[237]=w_hp` (~0.41 unrevealed, ~1.0 revealed via `_REVEAL_LOGIT=10.0`), but `MOVE_BP[237]=0` and the kernel zeros BP-0 moves (damage_tables.py:141; features_extractor.py:1786). So it's the ONLY weighted HP candidate → belief-ranked top-K (`w_all.detach().topk(K)`) selects it → 0 damage → "immune".

KEY: masking the bare-237 ALONE is INSUFFICIENT (HP just vanishes — no nonzero typed candidates to replace it). FIX needs BOTH: inject a real opp HP-type distribution (Smogon prior on reveal, or a learned head) AND mask the bare-237. The collapsed `_chan_max` worst-case summary is NOT corrupted (bare-237's `0.41·0=0` doesn't lower the max) — bug is confined to the belief-ranked top-K / matrix views.

NOT fixed by worktree `01MgZb` (`gen3_typed_hidden_power_ids_v1` = OUR-side typed HP nums 355-370, opp folded to 237 by explicit no-leak fuzz invariant) NOR `optimistic-allen-c14db0` (`SpeedBelief` = opp speed-bin, zero HP-type inference). Both verified orthogonal.

GIGO-class per [[feedback_gigo_order_bugs_asap]]. The richer lever = a learned opp-HP-type head supervised on reveal (privileged true type from agent2's team, leak-safe belief-label pattern) — none exists today. Pairs w/ [[project_topk_incoming_moves]] (the top-K block this surfaces in) + [[project_incoming_damage_outcome]].
