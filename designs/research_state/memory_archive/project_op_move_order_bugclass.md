---
name: project_op_move_order_bugclass
description: "2026-06-18 op move-order misalignment fix (gen3_op_move_align_v1) + the obs-ordering bug class + why GPU had it but CPU didn't"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — the durable rule survives as feedback_gigo_order_bugs_asap; the guards are in-tree. Preserved verbatim; nothing below is current.

The DamageOperator's OUTGOING per-move methods (_outgoing_block v23, _status_landing v27, _outgoing_matrix v34, + the v36/v37 refine discrete_outgoing/discrete_outgoing_status) read OUR active's moves from `ctx.all_move_ids[our_active]` — the PER-MON obs block, which is **sorted-by-id** (`get_sorted_moves`) — but their per-move output is consumed by the ACTION-aligned policy head (feature k ↔ action logit 6+k = request order = `legal.move_slots[k]`). Sorted-by-id ≠ request order in ~96% of decisions → the outgoing tie-break / status-landing / switch-in-KO matrix were positionally MISALIGNED with the actions they inform (a real correctness bug, worst under `--unified-obs` where the CPU per-move blocks are masked).

**Why GPU had it but CPU didn't (the reconciliation):** the CPU per-move obs features (move-power vec[0:4], type-mult vec[4:8], the 44-dim move-effect block) were EXPLICITLY aligned to request order by `gen3_move_slot_align_v1` (sourced via `reactive._request_slot_moves(battle, legal)` iterating `legal.move_slots`, disabled-kept, typed-HP resolved). The GPU op was added LATER and reused the convenient `all_move_ids` per-mon block — which is sorted-by-id and CANNOT be reordered (it feeds the role token via an order-sensitive concat, and exists for all 12 mons incl. bench, which have no request order). It even gated with `ctx.move_mask` (the prev-turn mask, ALSO reordered to sorted-by-id) → internally self-consistent (sorted ids + sorted mask), which HID the bug, but the OUTPUT was misaligned with the action-ordered logits. Two valid orderings coexist (sorted for role-token; request for action-aligned features); a new consumer picked the wrong one.

**THE FIX (gen3_op_move_align_v1, retrain-class, ARCH bump, obs 3457→3469, REACTIVE_DIM 402→414):** added a 12-dim request-ordered OUR-ACTIVE obs slice in reactive.py (`active_req_moves` = [move_num×4, resolved_type_id×4, legal_now×4], from `legal.move_slots` — the SAME source the CPU features + action mask use), placed AFTER the matchups (offsets + non_matchup_rest undisturbed; sliced in ObsUnpack into `ctx.our_active_req_move_{ids,type_ids,legal}`, excluded from the raw-scalar path). All 5 op methods now read that slice; the ones with a legality gate use the CURRENT `legal_now` (was the stale prev-turn `move_mask`). So CPU features AND the GPU op now both read request order → both action-aligned.

**GUARDS ("throw an error next time"):** move_alignment_fuzz_test asserts the obs `active_req_moves` block == `legal.move_slots` order (num+type+legality) over real battles (PRODUCER); damage_op_test::test_outgoing_reads_request_order_not_per_mon_block asserts the op output follows the request slice when it differs from the per-mon block (CONSUMER).

**Lesson:** any NEW GPU/model consumer that produces ACTION-aligned per-move output MUST source from the request-ordered obs (`legal.move_slots`), never the per-mon sorted block. Verified by a 4-lens adversarial audit workflow — the order-mismatch sweep CLEARED the rest of the pipeline (switches, matchups, incoming, topk all correct). Paired bug: the bridge `[miss]` parse crash — fragile single-position `event[-1]==` suffix strip stranded `[miss]` behind a later `[anim]`; fixed with a robust final loop + hardened the same pattern in the `|-activate|` `[of]` handlers (Trick/Mummy/Wandering-Spirit/Symbiosis). See [[feedback_provide_vs_learn]].
