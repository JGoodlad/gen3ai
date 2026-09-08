---
name: project_topk_incoming_moves
description: "Discrete top-K incoming move-space DamageOperator block BUILT (v30, gen3_unified_topk_incoming_v1, NOT run/shipped)"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — v30 build record; the top-K block is production and stated in designs/ARCHITECTURE.md. Preserved verbatim; nothing below is current.

2026-06-16 (worktree bridge-cse, branch worktree-bridge-cse..., **BUILT, NOT run/shipped**). Gave the
`DamageOperator` a DISCRETE top-K incoming block so the policy reasons over the opp active's INDIVIDUAL
likely moves instead of the worst-phys/worst-spec `_chan_max` collapse (which hid WHICH move + the
per-pivot consequences). Uses the [[project_unified_move_system]] MoveLatentEncoder + status machinery.

**WHAT:** for the opp active's K most-believed CANDIDATE moves (over `w_all` = belief ⊕ 16 typed HP;
`torch.topk` indices DETACHED), per move emit: move **LATENT** identity (gathered from the candidate
latent table — real ⊕ typed-HP via new `MoveLatentEncoder.hp_latent_block`; differentiable → sharpens
the latent) + belief `w` (differentiable → sharpens the move belief) + accuracy + is_phys (opp-property,
`_DMG_TOPK_MOVE`=35), then **per OUR mon** `[high, pko, status_lands]` (`_DMG_TOPK_DMG_PER`=3). `out_dim`
grows `K·53`. Damage gathered from RAW `_damage_rolls` `[B,6,C]` (w-INDEPENDENT → decorrelated; the
Jensen/"provide facts" principle); `_incoming_status_lands` = the immunity-folded incoming status threat
(dedicated status move OR damaging-move major-status secondary, gated by damage landing).

**OWNER DECISIONS (3 rounds of input):** (1) gradient = DECORRELATED (belief via the `w` feature +
retained `_chan_max`, NOT `w·damage`); (2) the real concern was IMMUNITY for safe switches — damage
immunity already preserved (per-pivot high/pko=0 from the chart, e.g. Ground→Flying/Gengar-Levitate,
Normal↔Ghost), PLUS a NEW per-pivot `status_lands` for status-move immunity (**Thunder Wave→Ground=0**,
the owner's canonical case); (3) K is a CONFIGURABLE int `damage_topk_k` (default 5, "reason about the
4th/5th move"), AUTO-on under `--unified-moves` (explicit `--damage-topk 0` disables = clean A/B); (4)
**meaningful-K gate** — zero the 5th slot once all 4 opp moves are revealed (moveset closed).

**VERSIONING:** v30, STRUCTURAL int gated in `check_compatible` like `opp_belief_cls_k`; OFF (0)
byte-identical, NO `ARCH_SIGNATURE` bump; requires `--damage-op`+`--move-latent`; threaded through the 4
opp-load sites; `decode_damage_block(topk_k=K)` SoT mirror; prober resolves EXACT move names from stashed
`last_topk_idx` (typed HP → `hiddenpower(type)`) + an "opp likely (op)" render. CRITICAL build detail:
the candidate latent table is built in `forward_internal` UNCONDITIONALLY when topk on (NOT
`is_grad_enabled`-gated, unlike `last_move_latent_table`), else the op output differs train vs rollout.

**VERIFIED:** 2745 unit tests + 10 new top-K op tests (distinct-threats, pivot damage-immunity, status-
immunity, meaningful-K gate, grad-flows-to-belief+latent, leak-free, typed-HP, purely-additive) + the
authoritative omniscient Showdown probe `damage_op_probe_fuzz_test.py` 22/22 (19 existing + 3 new
per-pivot) + roundtrip smoke + serverless `--debug --use-showdown-bridge --unified-moves both` →
Training complete + a 25-agent adversarial review (4 dims × verify) = 0 confirmed bugs.

**HONESTY GATE (learns ≠ helps, the op-feature precedent — see [[project_gpu_damage_op]],
[[project_incoming_damage_outcome]]):** wired + differentiable but UNMEASURED if it helps the POLICY →
fresh-run A/B (`--unified-moves both` vs `+ --damage-topk 0`): under-switching falls (prober
`switch-vs-info` / `human_agreement`) AND surprise-OHKO/wrong-pivot crater share falls (`falsify-scan`)
AND win-rate non-regress. Directly attacks the [[project_incoming_damage_outcome]] under-switching gap
(the belief was calibrated but the policy under-switched — this makes "which pivot is safe" decidable).
Design: `designs/ai_v6/design_topk_incoming_moves.md`.
