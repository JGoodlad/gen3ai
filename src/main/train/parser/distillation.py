"""The `# --- ADVANTAGE-GATED / ACTION-FORM DISTILLATION + the RANK TRIPWIRE ---`
section — the largest one, carrying the distillation flags, the search-teacher
tuning, the aux/belief heads and the remaining arch switches declared under it.

Lifted VERBATIM out of the old single-file `parser.py` (lines 980-1596); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from agents.model.features_extractor import BELIEF_GRAD_MODES
from main.train.parser.base import BoolFlag


def add_distillation_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- ADVANTAGE-GATED / ACTION-FORM DISTILLATION + the RANK TRIPWIRE
    # (gen3_distill_target_gate_v1; designs/ai_v10/design_advantage_gated_distillation.md
    # §3.1/§3.3/§4.1/§7). ALL TRAINING-only, the td_aux_coef provenance class (config v103):
    # argparse default None so `_resolve` can inherit on a flagless resume; recorded on
    # ModelVersion for provenance; never gated. Defaults are byte-identical to today.
    parser.add_argument("--distill-target", "--distill_target", dest="distill_target",
                        choices=["kl", "action"], default=None,
                        help="TARGET FORM of the exploiter-distillation policy term (rung (c), the "
                             "axis no arm has ever manipulated). 'kl' (default) = today's "
                             "full-distribution forward KL, byte-identical. 'action' = distil the "
                             "teacher's top-K probabilities renormalized over the legal set "
                             "(--distill-topk; K=1 = pure argmax CE — one bit of ordering, no tail "
                             "shape), AWR-weighted w = clamp(exp(|adv|/--distill-beta), 20). "
                             "Requires --distill-coef > 0. Watch distill/gated_frac + "
                             "distill/gate_agree_rate + grad/distill_share (the §6.2 dose meter).")
    parser.add_argument("--distill-topk", "--distill_topk", dest="distill_topk",
                        type=int, default=None,
                        help="With --distill-target action: distil toward the teacher's top-K "
                             "probabilities renormalized over the legal set (default 1 = pure argmax "
                             "CE; K >= n_actions recovers the full KL — the D-F dial, K=3 the "
                             "defensible middle). Requires --distill-target action.")
    parser.add_argument("--distill-gate", "--distill_gate", dest="distill_gate",
                        choices=["none", "advantage"], default=None,
                        help="THE JUDGE (rung (a)). 'none' (default) = every on-pin row, exactly the "
                             "rows the KL fired on (arm G1). 'advantage' = keep only rows where the "
                             "teacher DISAGREES with the sampled action AND the student's own "
                             "NORMALIZED advantage reads it as a mistake (adv < -tau) — the distill "
                             "gradient then pushes a logit PPO is already pushing down, by "
                             "construction. Requires --distill-target action. Watch distill/n_gated "
                             "(0 is a reading, not an absence) and the §6.2 dose confound: G2's coef "
                             "is set by grad/distill_share, never by eye.")
    parser.add_argument("--distill-gate-tau", "--distill_gate_tau", dest="distill_gate_tau",
                        type=float, default=None,
                        help="Advantage-gate threshold tau (default 0.0): a row contributes only when "
                             "adv(s,a) < -tau, in NORMALIZED advantage units (the same normalization "
                             "the clip objective uses). Requires --distill-gate advantage.")
    parser.add_argument("--distill-beta", "--distill_beta", dest="distill_beta",
                        type=float, default=None,
                        help="AWR temperature beta for the action-form target's |adv| weight, "
                             "w = clamp(exp(|adv|/beta), max=20) — mirrors --search-teacher-beta "
                             "(default 1.0). Only used with --distill-target action.")
    parser.add_argument("--rank-tripwire", "--rank_tripwire", dest="rank_tripwire",
                        choices=["off", "warn", "abort"], default=None,
                        help="RANK TRIPWIRE (§4.1 — no fold runs blind again): watch the existing "
                             "rank/policy_pr probe as an EMA (half-life 10 train() calls) against the "
                             "run's own baseline (median over readings [5,25), logged as "
                             "rank/policy_pr_baseline). WARN (launcher event + rank/policy_pr_ratio) "
                             "at ema < (1 - drop/2)*base for 3 consecutive readings; TRIP at "
                             "ema < (1 - drop)*base x3 (loud event + rank/tripwire_fired latched; "
                             "under 'abort' the callback stops learn() cleanly, checkpoint saved). "
                             "Default 'warn' — a tripwire that ends a run by default would be a new "
                             "way to lose a training window. A missing reading is 'no reading' "
                             "(rank/tripwire_no_reading), never a trip and never an all-clear.")
    parser.add_argument("--rank-tripwire-drop", "--rank_tripwire_drop", dest="rank_tripwire_drop",
                        type=float, default=None,
                        help="TRIP threshold as a fractional drop from baseline (default 0.20: trip "
                             "at 0.80x base, WARN at half the drop = 0.90x). Calibrated against the "
                             "record: every KL-collapse arm fell 38-43%%, every control 0%% — 20%% "
                             "fires on all five known-bad arms and no known-good control; 20-38%% is "
                             "the honest margin.")
    parser.add_argument("--search-teacher-batch-size", "--search_teacher_batch_size",
                        dest="search_teacher_batch_size", type=int, default=None,
                        help="Corrections sampled per train() for the AWR forward (default 256).")
    parser.add_argument("--search-teacher-buffer-size", "--search_teacher_buffer_size",
                        dest="search_teacher_buffer_size", type=int, default=20000,
                        help="Correction ring capacity (recency; default 20000).")
    parser.add_argument("--teacher-search-budget", "--teacher_search_budget", dest="teacher_search_budget",
                        type=int, default=200, help="Candidates searched per cycle (budget cap; default 200).")
    parser.add_argument("--teacher-confirm-rollouts", "--teacher_confirm_rollouts",
                        dest="teacher_confirm_rollouts", type=int, default=8,
                        help="Monte-Carlo confirm games per candidate for the Wilson-CI strictly-better gate.")
    parser.add_argument("--teacher-search-workers", "--teacher_search_workers",
                        dest="teacher_search_workers", type=int, default=3,
                        help="Search-teacher worker subprocesses per cycle (default 3).")
    parser.add_argument("--teacher-search-freq", "--teacher_search_freq", dest="teacher_search_freq",
                        type=int, default=0, help="Steps between search-teacher cycles (0 = use the eval freq).")
    parser.add_argument("--teacher-persistent", "--teacher_persistent", dest="teacher_persistent",
                        action="store_true",
                        help="PERSISTENT-pool mode (the supply lever): long-lived workers GENERATE their "
                             "own fresh losses (frozen trainee vs current opponents) and search them "
                             "CONTINUOUSLY, dripping corrections into the buffer — instead of the bursty "
                             "per-cycle eval-trace scan. Higher, fresher supply; recommended once enabled.")
    parser.add_argument("--teacher-refresh-steps", "--teacher_refresh_steps", dest="teacher_refresh_steps",
                        type=int, default=500_000,
                        help="Persistent mode: re-freeze the trainee snapshot the workers use every N "
                             "steps (so long-lived workers track the moving policy). Default 500k.")
    parser.add_argument("--teacher-gen-battles", "--teacher_gen_battles", dest="teacher_gen_battles",
                        type=int, default=12, help="Persistent mode: battles generated per worker iteration.")
    parser.add_argument("--intent-move-cell", "--intent_move_cell",
                        dest="intent_move_cell", action=BoolFlag, default=None,
                        help="G3 (gen3_intent_move_cell_v1, design_conditional_execution.md): the "
                             "POLICY-side alpha consumer — the c2 status-consequence family "
                             "re-delivered through the pointer MOVE cell as a per-action absolute, "
                             "alpha-conditioned (burn/sleep channels become unrenormalized "
                             "alpha-expectations over the op's top-K seat candidates; the seat "
                             "mass rides as a decorrelated alpha_stay channel). Zero-init "
                             "projection => identity at init. Requires --opp-intent-coef>0, "
                             "--damage-op and --damage-topk-k>0. STRUCTURAL, version-checked.")
    parser.add_argument("--value-entity-pool-full", "--value_entity_pool_full",
                        dest="value_entity_pool_full", action=BoolFlag, default=None,
                        help="gen3_unified_value_readout_v2 (v82): the entity pool's COMPLETE "
                             "row set — + the refined GLOBAL token and the hidden-opp belief "
                             "queries. Requires --value-entity-pool. The Stage-3 successor for "
                             "every condemnable vf route. STRUCTURAL, version-checked.")
    parser.add_argument("--pair-outcome-cell", "--pair_outcome_cell",
                        dest="pair_outcome_cell", action=BoolFlag, default=None,
                        help="gen3_pair_outcome_v1 (v93, design_opponent_intent.md §5.1/§5.3 + "
                             "design_pair_reduction.md §2.1/§9a): the UNIFIED per-pair OUTCOME "
                             "VECTOR — one pair_in[their believed move k, our mon j] carrying "
                             "damage AND status BY IDENTITY (par/brn/frz/slp/psn/tox) AND "
                             "neutralization (how much of the mon is destroyed without a KO) AND "
                             "tempo_cost (turns spent undoing it), all in the same vector — "
                             "reduced by ONE α over the move axis (Contract W: one distribution, "
                             "every channel, so per-channel maxima are a shape error) and "
                             "delivered to the pointer MOVE cell. Closes the CURRENCY failure "
                             "§2.1 names: status reached the policy only as a softmax-normalised "
                             "edge RATIO, so \"35%% of my HP\" and \"80%% chance of burn\" never "
                             "met in one vector. Phase A = the MOVE-cell half (the switch cell "
                             "and the β cells are Phase B). Requires --damage-op and "
                             "--damage-topk-k>0; --opp-intent-coef>0 is OPTIONAL — without it α "
                             "falls back to the R1 belief_mean rung (α := w/Σw), so the DELIVERY "
                             "claim is testable apart from the DISTRIBUTION claim. Zero-init "
                             "projection so ON-at-init is bit-identical. STRUCTURAL, "
                             "version-checked.")
    parser.add_argument("--pair-outcome-switch", "--pair_outcome_switch",
                        dest="pair_outcome_switch", action=BoolFlag, default=None,
                        help="gen3_pair_outcome_switch_v1 (v94, substrate Phase B, "
                             "design_pair_reduction.md §2.1): deliver the SAME α-reduced unified "
                             "outcome row, PER DEFENDER, to the pointer SWITCH cell — the sink "
                             "§2.1 says the decision is actually made at. Today that cell holds "
                             "ten damage numbers, one speed number, two belief-mass numbers and "
                             "NO status coordinate in any currency, so \"they will click "
                             "Will-O-Wisp, so bring the Natural Cure mon\" is unrepresentable "
                             "there; status reaches the policy only as a softmax-normalised s3 "
                             "edge RATIO. Adds one per-defender coordinate of its own, "
                             "spin_denied = is_ghost(our mon j) · α(their Rapid Spin) · the "
                             "hazard stake on THEIR side — the defensive half of the Pursuit "
                             "mirror. The FIRST module ever to widen the switch cell. Requires "
                             "--damage-op and --damage-topk-k>0; NOT --pair-outcome-cell (the two "
                             "deliver one tensor to two sinks and coupling them would make a "
                             "result unattributable), NOT --opp-intent-coef (same R1 belief_mean "
                             "fallback as Phase A). Zero-init projection so ON-at-init is "
                             "bit-identical. STRUCTURAL, version-checked.")
    parser.add_argument("--switch-branch-cell", "--switch_branch_cell",
                        dest="switch_branch_cell", action=BoolFlag, default=None,
                        help="gen3_switch_branch_v1 (v94, substrate Phase B, "
                             "design_conditional_opponent_cells.md §2 = OA2): per-request-slot "
                             "content for the branch in which the OPPONENT SWITCHES. Gen-3 is "
                             "simultaneous-move so P(they switch) is ONE scalar for the turn, but "
                             "switches resolve FIRST — our move lands on the ARRIVAL, which β "
                             "names. Delivers E[high]/E[pko]/E[type mult] contracted over β, the "
                             "shared α_SWITCH scalar and wasted_ko = pko_stay·α_SWITCH "
                             "(\"don't click the KO into the obvious switch\"), all kept "
                             "DECORRELATED from the stay branch per §2.3 — never the collapsed "
                             "(1−p)·stay + p·switch. Plus two mechanics of the same shape: the "
                             "RAPID SPIN spinblock (p_spin_blocked = is_ghost(their active)·P(stay) "
                             "+ α_SWITCH·Σβ·P(arrival is Ghost) — the REVERSE of Pursuit "
                             "trapping, since gen3 Rapid Spin is Normal and a Ghost final defender "
                             "means no damage AND no hazard removal) and PROTECT's α-derived "
                             "attack mass (the c4 successor: its cell carries the consecutive-use "
                             "decay and never asked whether they will attack at all). Requires "
                             "--opp-intent-coef>0 with NO fallback — the R1 belief_mean rung is a "
                             "presence belief over their MOVES and has no switch class, so "
                             "α_SWITCH would be identically 0 and every coordinate would assert "
                             "\"they never switch\" — plus --damage-op, --damage-matrices "
                             "outgoing and --damage-topk-k>0. Zero-init projection so ON-at-init "
                             "is bit-identical. STRUCTURAL, version-checked.")
    parser.add_argument("--conditional-threat-cell", "--conditional_threat_cell",
                        dest="conditional_threat_cell", action=BoolFlag, default=None,
                        help="gen3_conditional_threat_v1 (v95, substrate Phase C, "
                             "design_conditional_opponent_cells.md §1 = OA1): the CONDITIONAL "
                             "THREAT CELL — \"they'll Ice Beam my Salamence; switch to the mon "
                             "that eats Ice Beam\". Four α-contracted coordinates on the pointer "
                             "SWITCH cell, each one a quantity Phase B's reduced outcome row "
                             "structurally cannot carry: e_pko_acc = Σα·ko_ramp·acc (the product "
                             "§0.2(2) says the OPERATOR must form — the two ride decorrelated and "
                             "a thin tanh scorer does not multiply its own inputs), e_type_mult "
                             "(the one cell channel NOT divided by the defender's own bulk, so a "
                             "structural immunity reads apart from an incidental zero), and the "
                             "two §0.2(3) MARGINS Σα·high − hp and Σα·crit − hp (>0 ⇒ dead) that "
                             "say by how much a saturated P(KO) saturates. §1.2's λ-weighted `w` "
                             "is deliberately NOT built: `pair_alpha` is the shipped distribution "
                             "and a second one would be a second α. Requires --damage-op, "
                             "--damage-matrices incoming and --damage-topk-k>0; NOT "
                             "--opp-intent-coef (the R1 belief_mean fallback is meaningful here — "
                             "every coordinate is a \"what lands on me if they attack\" "
                             "contraction) and NOT --pair-outcome-switch (two quantities, one "
                             "sink, attributable separately). Zero-init projection so ON-at-init "
                             "is bit-identical. STRUCTURAL, version-checked.")
    parser.add_argument("--pair-value-route", "--pair_value_route",
                        dest="pair_value_route", action=BoolFlag, default=None,
                        help="gen3_pair_value_route_v1 (v95, substrate Phase C, "
                             "design_opponent_intent.md §7a(2) = PV): the α-reduced unified "
                             "outcome row for our mon j injected as TOKEN CONTENT on mon j's own "
                             "token inside CLSPool, on the VALUE pool's copy ONLY — so pi is "
                             "bit-identical at ANY weight. It is the first per-entity route by "
                             "which the CRITIC reads the status / neutralization / tempo currency "
                             "at all (today incoming status reaches vf only as the s3 edge "
                             "family's softmax-normalised RATIO). Token content rather than the "
                             "v89 value-route seam: a post-pool additive route must collapse the "
                             "team axis, and the only equivariant collapse is a sum — which "
                             "cannot tell one mon losing 90%% of its bar from six losing 15%%. "
                             "⚠️ α is the R1 belief_mean rung UNCONDITIONALLY — ORDERING, not "
                             "preference: value_cls pools BEFORE the α/β heads are scored. "
                             "⚠️ C4 RE-ENTRY CONDITION: any α/β-critic route may be BUILT opt-in "
                             "but its ENABLING owes the C4-style offline gate first (ledger C6 — "
                             "the delivery line is EXHAUSTED). Requires --damage-op and "
                             "--damage-topk-k>0. Zero-init so ON-at-init is bit-identical. "
                             "STRUCTURAL, version-checked.")
    parser.add_argument("--intent-threshold", "--intent_threshold",
                        dest="intent_threshold", action=BoolFlag, default=None,
                        help="gen3_intent_threshold_v1 (v84, design_conditional_execution.md §3.0 "
                             "step 3): the α-weighted THRESHOLD operator p_thresh(τ,⋛) — five "
                             "mechanics through the pointer MOVE cell at once (Focus Punch "
                             "executes / Substitute survives / Endure·p_KO / Destiny Bond·p_KO / "
                             "Endeavor survives-to-act), plus p_KO — the calibrated am-I-about-"
                             "to-die — appended to the CRITIC (the ledger-H1 payoff; the critic "
                             "previously read a hard max). One contraction over the op's existing "
                             "per-candidate cells; both projections zero-init so ON-at-init is "
                             "bit-identical. Requires --opp-intent-coef>0, --damage-op and "
                             "--damage-topk-k>0. STRUCTURAL, version-checked.")
    parser.add_argument("--op-drop-renders", "--op_drop_renders",
                        dest="op_drop_renders", action=BoolFlag, default=None,
                        help="gen3_op_lean_forward_v1 (v86, design_op_tensors step 3): drop the op "
                             "flat block's three RENDER regions (outgoing matrix / incoming matrix "
                             "/ OAX) from the forward — serialization-only since the concat's "
                             "deletion; every consumer value survives as a typed stash and every "
                             "surviving offset is unchanged, so ON at init is bit-identical. "
                             "Shrinks out_gain (state_dict). STRUCTURAL, version-checked.")
    parser.add_argument("--op-believed-lean", "--op_believed_lean",
                        dest="op_believed_lean", action=BoolFlag, default=None,
                        help="gen3_op_lean_forward_v1 (v86): the lean d3 edge physics price the "
                             "attacker from the BELIEVED spread instead of the legacy de-timid "
                             "252-EV/boosting-nature fiction — the B-spread correctness fix at the "
                             "last de-timid site the edges read. Requires --spread-belief and "
                             "--damage-op. Forward-math only. STRUCTURAL, version-checked.")
    parser.add_argument("--intent-conditional", "--intent_conditional",
                        dest="intent_conditional", action=BoolFlag, default=None,
                        help="gen3_intent_conditional_v1 (v85, design_conditional_execution.md "
                             "steps 4+7): the remaining α-conditioned mechanic cells — Counter/"
                             "Mirror Coat's category test (unplayable without an intent model), "
                             "flinch's missing (1−α_SWITCH) term, Explosion's p_executes + "
                             "into-switch facts (the H1 companions), Pursuit's ×2 never-miss "
                             "doubling trigger (port-verified departing-target rule), Protect's "
                             "α-weighted avoided damage/status beside c4's mechanical odds, Magic "
                             "Coat's oracle-verified reflect set, and Explosion's β-weighted trade "
                             "KO — the first forward-side β consumer (β published like α). One "
                             "zero-init projection over tensors the op already stashes. Requires "
                             "--opp-intent-coef>0, --damage-op, --damage-outgoing, "
                             "--damage-matrices outgoing|both and --damage-topk-k>0. STRUCTURAL, "
                             "version-checked.")
    parser.add_argument("--item-belief", "--item_belief",
                        dest="item_belief", action=BoolFlag, default=None,
                        help="gen3_item_belief_v1 (v83): a learned posterior over each opp slot's "
                             "HIDDEN item (Smogon usage prior ⊕ zero-init trunk delta; BeliefBank's "
                             "seventh row supervises it at revealed slots via --item-belief-coef). "
                             "The op's Choice-Band-conditional tail consumes P(CB) from the "
                             "published posterior at the UNREVEALED branch (revealed stays exact "
                             "0/1), replacing the static SPECIES_CB_PRIOR scalar there. Cold start "
                             "posterior == the Smogon prior exactly (zero-init delta), whose CB "
                             "column sits within ~0.6%% of the static table (the row floor's renorm), "
                             "so enabling is ~behavior-preserving at init. STRUCTURAL, "
                             "version-checked.")
    parser.add_argument("--history-events", "--history_events",
                        dest="history_events", action=BoolFlag, default=None,
                        help="gen3_event_window_v1 (v81, Tier H-B of design_history_entity.md): "
                             "the last-32 event records join the trunk as EVENT SEATS — typed, "
                             "entity-content (shared species/move embeddings), time as content "
                             "(log recency + forced-window tag), appended after the E5 seats. "
                             "The obs block is unconditional; this builds the consumer. "
                             "STRUCTURAL, version-checked.")
    parser.add_argument("--value-entity-pool", "--value_entity_pool",
                        dest="value_entity_pool", action=BoolFlag, default=None,
                        help="gen3_unified_value_readout_v1 (v80, design_unified_belief.md §3 / "
                             "Stage-3 T3-DELIVER): ONE attention pool over the critic's entity "
                             "rows — the 12 post-transformer team tokens + the op's per-our-mon "
                             "incoming rows — K learned queries, per-source type embeddings, "
                             "ZERO-INIT output projection riding vf only (the policy is untouched "
                             "at any weight). The designed successor of the bolt-on vf routes the "
                             "critic_route_audit adjudicates. Works with or without --damage-op "
                             "(the row set shrinks to the team tokens). STRUCTURAL, "
                             "version-checked.")
    # `--opp-intent-grad-mode` was DEMOTED to config_only on 2026-08-23 (registry sweep #2). It is
    # frozen at "detached", still recorded in model_config.json and still version-checked; the
    # "shaping" arm remains constructible via the extractor kwarg. Census: unanimous across the 24
    # runs that record it, and typed in ZERO of 107 recorded launcher commands.
    parser.add_argument("--beta-setvalued-coef", "--beta_setvalued_coef",
                        dest="beta_setvalued_coef", type=float, default=None,
                        help="SET-VALUED partial credit for beta on switch-ins we did not believe "
                             "(gen3_beta_setvalued_v1). Today those rows are MASKED, discarding a "
                             "true fact: they brought a mon we had not revealed. This grades the "
                             "coarse call -log(sum of believed-slot mass) without asserting WHICH "
                             "member, which is the part we cannot label. Scales on top of "
                             "--opp-intent-coef. 0.0 = OFF (byte-identical). Training-only.")
    parser.add_argument("--intent-label-bot-weight", "--intent_label_bot_weight",
                        dest="intent_label_bot_weight", type=float, default=None,
                        help="Per-sample weight on the OPPONENT-INTENT (alpha/beta) labels produced "
                             "against a heuristic BOT (gen3_intent_label_bot_weight_v1); every other "
                             "opponent class (pool / stable / exploiter) stays 1.0. Bots play "
                             "strategies that are not the meta, and the self-play ramp trains 100%% vs "
                             "bots until the pool seeds, so early intent supervision is "
                             "bot-DOMINATED (gen-11: 100%% of supervised rows at 2M, ~7%% from 6M on) "
                             "and the head can imprint on a decision tree. Folded BEFORE the mean at "
                             "the same n_sup denominator, so the --opp-intent-coef semantics are "
                             "unchanged; 0.0 trains on no bot rows at all. Applies to alpha/beta ONLY "
                             "— never to the species/move/item/spread/HP-type belief labels, which "
                             "are TEAM truth and valid whoever pilots the team. Watch "
                             "opp_intent/label_bot_frac (the exposure) and opp_intent/alpha_acc_pool "
                             "(the metric that must not fall). 1.0 = OFF (loss bit-identical). "
                             "TRAINING-only (not version-locked; inherited on a flagless resume).")
    parser.add_argument("--opp-intent-coef", "--opp_intent_coef", dest="opp_intent_coef",
                        type=float, default=None,
                        help="OPPONENT-INTENT aux (gen3_opp_intent_v1, v67): supervise ALPHA — a "
                             "distribution over the opponent's K believed threat-move seats PLUS "
                             "SWITCH — and BETA — which of their mons comes in — against what they "
                             "ACTUALLY did. Both are POINTER heads (equivariant over their moves / "
                             "their bench) and see a DETACHED input, so a null says the head cannot "
                             "predict them rather than that predicting them hurt the policy. "
                             "Measured headroom (gen-8): the belief's top-K contains their move 85.8%% "
                             "of the time but ranks it first only 51.8%% — 34pp of mis-ranked mass. "
                             "Requires --entity-topk-seats>0. 0.0 = OFF (no heads, byte-identical). "
                             "STRUCTURAL + version-checked; the coef itself is training-only.")
    parser.add_argument("--value-threat-inject", "--value_threat_inject",
                        dest="value_threat_inject", action=BoolFlag, default=None,
                        help="CRITIC THREAT INJECTION (gen3_value_threat_inject_v1, v64): add the "
                             "DamageOperator's alpha-weighted incoming-threat row for each of OUR "
                             "mons to that mon's token on the VALUE POOL's copy only, so value_cls "
                             "pools per-entity threat MAGNITUDES instead of the softmax RATIOS the "
                             "d3 edge family can carry. vf-ONLY: the policy reads the unaugmented "
                             "tokens, so pi is bit-identical at any weight (gated). Forces the op's "
                             "pair reduction to the R1 belief_mean rung (hard_max builds no reducer "
                             "and would leave nothing to inject). Zero-init => ON starts identical "
                             "to OFF. STRUCTURAL + version-checked: fixed for a run's lifetime.")
    parser.add_argument("--value-dist-mode", "--value_dist_mode", dest="value_dist_mode",
                        choices=("none", "read_only", "shaping"), default=None,
                        help="Distributional VALUE head (v29): an interpretability readout off the value "
                             "pool emitting --value-dist-bins logits over [--value-dist-vmin, "
                             "--value-dist-vmax] — softmax = the critic's predicted RETURN DISTRIBUTION "
                             "(sharp=confident, wide=uncertain, bimodal=coinflip), reviewable per-decision "
                             "in the prober. 'none' (default) = no module (baseline byte-for-byte). "
                             "'read_only' = the head trains on a STOP-GRAD value pool (a risk-free "
                             "diagnostic that CANNOT perturb the policy). 'shaping' = its gradient also "
                             "shapes the shared trunk. STRUCTURAL + resume-IMMUTABLE (version-checked). A "
                             "SIDE readout (never in pi/vf — leak-safe). "
                             "Design: designs/ai_v6/design_distributional_value_critic.md.")
    parser.add_argument("--value-dist-bins", "--value_dist_bins", dest="value_dist_bins",
                        type=int, default=None,
                        help="Atom count for --value-dist-mode (the head's output width; weight-shape, "
                             "version-checked). Recommended 32 (readable). Required > 0 when the mode is "
                             "on; ignored (must be 0) when none.")
    parser.add_argument("--value-dist-vmin", "--value_dist_vmin", dest="value_dist_vmin",
                        type=float, default=None,
                        help="Lower edge of the value-dist atom support (the return range the atoms span). "
                             "Resume-immutable (version-checked). Required when --value-dist-mode is on.")
    parser.add_argument("--value-dist-vmax", "--value_dist_vmax", dest="value_dist_vmax",
                        type=float, default=None,
                        help="Upper edge of the value-dist atom support. Resume-immutable "
                             "(version-checked). Required when --value-dist-mode is on (must be > vmin).")
    parser.add_argument("--value-dist-coef", "--value_dist_coef", dest="value_dist_coef",
                        type=float, default=None,
                        help="Loss weight for the value-dist head's HL-Gauss CE (value_dist_coef * CE), "
                             "like --win-prob-coef. Default 1.0. TRAINING-only (not version-locked; "
                             "inherited on a flagless resume). Ignored when --value-dist-mode none. Lower "
                             "it if 'shaping' fights the policy (watch grad/value_dist_share / "
                             "grad/value_dist_policy_cosine — this head's own shared-trunk pull).")
    parser.add_argument("--td-aux-coef", "--td_aux_coef", dest="td_aux_coef",
                        type=float, default=None,
                        help="TD-CONSISTENCY auxiliary weight (gen3_td_consistency_aux_v1): add "
                             "coef * mean[(V(s_t) - r_t - gamma*V(s_t+1))^2] over CONTIGUOUS rollout "
                             "pairs, on top of the per-state value loss. The per-state MSE never "
                             "constrains adjacent-state DIFFERENCES, so dV inherits ~2x the state "
                             "noise where the truth is nearly constant; this is the Bellman identity "
                             "the critic already owes, made explicit. 0.0 = OFF (loss byte-identical). "
                             "Pre-registered band 1.0-3.0 (3.0 is the favourite); coef <= 0.1 measured "
                             "WORSE than control offline, so avoid the small-coef regime. TRAINING-only "
                             "(not version-locked; inherited on a flagless resume). Costs one extra "
                             "512-state critic forward per minibatch. Watch td_aux/resid_rms fall and "
                             "td_aux/resid_mean stay near 0.")
    parser.add_argument("--win-prob-pbrs-coef", "--win_prob_pbrs_coef", dest="win_prob_pbrs_coef",
                        type=float, default=None,
                        help="WIN-PROB PBRS reward shaping (gen3_winprob_pbrs_v1; ai_v12 route 1, "
                             "designs/ai_v12/design_winprob_behavior_coupling.md). Adds "
                             "coef * (gamma*phi(s') - phi(s)) to every transition's reward, with "
                             "phi(s) = sigmoid of the win-prob head's logit, DETACHED. --win-prob-mode "
                             "'shaping' is REPRESENTATION shaping and carries NO behavioral force (the "
                             "head is a side readout with no gradient path to the acting head); this is "
                             "the reward-level route that gives it force, so a whiff that drops the "
                             "model's own P(win) costs literal reward. Protected by the "
                             "potential-based-shaping invariance theorem -- a miscalibrated phi costs "
                             "learning SPEED, not correctness -- but our phi is a LEARNED, DRIFTING "
                             "head, so that holds exactly per rollout and only approximately across "
                             "them (prefer a MATURE base; see the doc's SS2.4). 0.0 = OFF, "
                             "byte-identical (the module is not even imported). REQUIRES "
                             "--win-prob-mode read_only|shaping. Applied trainer-side to the rollout "
                             "buffer before GAE (env workers hold no model); covers --async-rollout. "
                             "TRAINING-only (not version-locked; recorded for provenance and inherited "
                             "on a flagless resume, the td_aux_coef class). Watch "
                             "train/pbrs_reward_share -- the shaping's share of the reward stream.")
    parser.add_argument("--win-prob-pbrs-source", "--win_prob_pbrs_source",
                        dest="win_prob_pbrs_source", type=str, default=None,
                        help="FROZEN phi for --win-prob-pbrs-coef (gen3_winprob_pbrs_source_v1): a "
                             "checkpoint .zip or run dir whose win-prob head supplies the potential, "
                             "instead of the LIVE (training, drifting) head. This is what makes the "
                             "PBRS invariance theorem hold EXACTLY rather than approximately -- the "
                             "theorem assumes phi is a FIXED function of state, and our live head is "
                             "a module inside the network being trained, so the per-start-state "
                             "constant moves across rollouts. A frozen mature phi removes that "
                             "caveat entirely. Costs one extra frozen extractor on the training "
                             "device (the --distill-teacher class) and one no_grad forward per "
                             "rollout, which REPLACES the live-phi forward rather than adding to it. "
                             "Requires --win-prob-pbrs-coef > 0; the source must share our "
                             "arch_signature (an obs FAMILY check, so a prior-generation phi is "
                             "viable). Loaded eagerly -- never torch.compile'd, and never pickled "
                             "into our checkpoint. A bad path is a FATAL_CONFIG exit at startup, "
                             "never a crash-restart loop. TRAINING-only, recorded for provenance and "
                             "inherited on a flagless resume (a resume that silently reverted to "
                             "live-phi would change the objective mid-run with nothing saying so).")
    parser.add_argument("--policy-grad-coef", "--policy_grad_coef", dest="policy_grad_coef",
                        type=float, default=None,
                        help="POLICY-GRADIENT term weight (gen3_policy_grad_coef_v1): multiplies ONLY the "
                             "clipped PPO surrogate `policy_loss` in the loss fold — never entropy "
                             "(--ent-coef), never the value term (--vf-coef), never any aux/distill "
                             "coefficient. Default 1.0 = the upstream expression, byte-identical "
                             "(the unscaled tensor is used). 0.0 removes the policy-gradient "
                             "contribution entirely — the pure-distill/aux phase (arm F of "
                             "design_advantage_gated_distillation.md §5): every other term keeps "
                             "training while PPO's own policy pull is off. TRAINING-only (not "
                             "version-locked; recorded for provenance and inherited on a flagless "
                             "resume, the td_aux_coef class). Watch grad/policy_share read ~0 at "
                             "0.0 — the live confirmation the term is actually gone.")
    parser.add_argument("--move-latent", "--move_latent", dest="move_latent",
                        action=BoolFlag, default=None,
                        help="MoveLatentEncoder (gen3_unified_move_system_v1): a context-free, "
                             "mechanics-grounded per-move latent (move/type embeddings + structured "
                             "MOVE_ATTR — BP / category / accuracy / priority / drain / per-status secondary "
                             "chances) concatenated into the move network, so the model reads a richer move "
                             "identity AND the SAME latent is the similarity-grading target (Rock Slide ~= "
                             "Hidden Power Rock). STRUCTURAL (widens the move-network input; version-checked, "
                             "fresh-only). Off by default.")
    parser.add_argument("--move-belief-latent-coef", "--move_belief_latent_coef",
                        dest="move_belief_latent_coef", type=float, default=None,
                        help="Latent-space grading weight for the move belief: coef * (cosine of the "
                             "predicted move distribution's expected move-latent toward the true moveset's "
                             "mean latent + VICReg floor) on revealed slots — the soft complement to the "
                             "per-ID BCE so near-moves grade as near. REQUIRES --move-latent (reads its "
                             "latent table) and a move-belief mode that scores revealed slots. TRAINING-only "
                             "(not version-locked; inherited on a flagless resume). 0.0 = OFF.")
    parser.add_argument("--unified-moves", "--unified_moves", dest="unified_moves",
                        choices=["off", "incoming", "both"], default=None,
                        help="ONE knob for the WHOLE unified move system: sets --unified-damage to the same "
                             "level (move belief + prior fusion + the GPU damage op, incl. its per-status "
                             "secondary/Serene-Grace effects; 'both' adds the outgoing direction) AND turns "
                             "on --move-latent + a default --move-belief-latent-coef 0.05 + the DISCRETE "
                             "incoming move-space at K=5 (--damage-topk, which implies --damage-matrices "
                             "incoming). DEFAULT: 'both' on a FRESH run (the unified system IS the model — "
                             "without it the op has no belief to price and the policy loses the whole "
                             "believed-move threat read); a RESUME (--model) inherits the checkpoint's saved "
                             "component toggles verbatim, so old configs keep working. 'off' is DEPRECATED — "
                             "it survives only as an explicit ablation baseline and warns at startup. Compose "
                             "the pieces by hand for finer control (e.g. --damage-topk 0 to A/B the discrete "
                             "move-space off under --unified-moves).")
    parser.add_argument("--damage-topk", "--damage_topk", dest="damage_topk_k",
                        type=int, default=None,
                        help="K for the DISCRETE incoming move-space: the number of the opp ACTIVE's "
                             "most-believed CANDIDATE moves the INCOMING per-move damage matrix surfaces "
                             "INDIVIDUALLY (vs the worst-case max collapse that loses WHICH move it is) — "
                             "per move its LATENT identity + belief + acc + is_phys + per-move effect/"
                             "secondary bits, then per OUR mon [low, high, crit, P(KO), type_mult, "
                             "status_lands], the read that makes 'anticipate the move / pick the safe "
                             "switch' decidable (damage-immunity AND status-immunity both = 0, e.g. "
                             "Thunder-Wave→Ground). 0 = off. STRUCTURAL int (scales both projections; "
                             "version-checked, fresh-only). REQUIRES --damage-op + --move-latent, and "
                             "IMPLIES --damage-matrices incoming (gen3_op_block_trim_v1 deleted the lean "
                             "top-K block K used to select — the matrix is its strict superset, and the "
                             "profiler measured the lean block at 0 calls/forward). AUTO-set to 5 by "
                             "--unified-moves (the moveset is 4, so the 5th slot is the surprise candidate); "
                             "the 5th is zeroed once all 4 opp moves are revealed. Default off.")
    parser.add_argument("--damage-matrices", "--damage_matrices", dest="damage_matrices",
                        choices=["off", "incoming", "outgoing", "both"], default=None,
                        help="Per-move DAMAGE MATRICES (gen3_per_move_matrices_v1). 'outgoing': OUR 4 moves × "
                             "the opp's 6 mons (active + REVEALED bench) — per (move, opp mon) "
                             "[low,high,crit,pko,type_mult] + a revealed bit (price a KO on a SWITCH-IN). "
                             "'incoming': the ENRICHED top-K — per opp move a header [latent, belief, acc, "
                             "is_phys, EXPLICIT effect bits(6), secondary chances(10)] + per (OUR mon, move) "
                             "cell [low,high,crit,pko,type_mult,status_lands] (the un-collapsed evolution of "
                             "--damage-topk; it REUSES --damage-topk K as its K — one knob, try 4/5/6, default "
                             "5 — and REPLACES the lean top-K block at that K; requires --move-latent). "
                             "'both' = incoming + outgoing. Unrevealed opp slots zeroed (belief-driven = TODO). "
                             "STRUCTURAL (version-checked, fresh-only). REQUIRES --damage-op. 'off' (default) = "
                             "baseline byte-identical.")
    # gen3_bidir_threat_trunk_v1 (v36): the uncertainty-aware P(outspeed).
    parser.add_argument("--threat-prob-outspeed", "--threat_prob_outspeed", dest="threat_prob_outspeed",
                        action=BoolFlag, default=None,
                        help="#3 UNCERTAINTY-AWARE P(outspeed): divide the speed gap by the believed speed STD "
                             "(SPECIES_SPREAD_PRIOR; sigmoid≈normal-CDF) instead of a fixed scale — a high-variance "
                             "opp speed reads ~0.5, a pinned one reads sharp. FORWARD-behavior (version-checked, "
                             "fresh-only). REQUIRES --damage-op. Default off (byte-identical).")
    parser.add_argument("--spread-belief", "--spread_belief", dest="spread_belief",
                        action=BoolFlag, default=None,
                        help="SpreadBelief (gen3_unified_spread_belief_v1): the THIRD belief leg — predict "
                             "the opponent's hidden SPREAD (the 5 derived stats atk/def/spa/spd/spe) per "
                             "slot from a usage PRIOR + a learned head, reinject into the opp token, and "
                             "feed the DamageOperator so it consumes BELIEVED opp stats instead of its "
                             "hand-coded de-timid/neutral constants (offense, bulk, speed). STRUCTURAL "
                             "(version-checked, fresh-only). Off by default.")
    parser.add_argument("--spread-belief-coef", "--spread_belief_coef", dest="spread_belief_coef",
                        type=float, default=None,
                        help="Spread-belief SUPERVISION weight (gen3_unified_spread_belief_v1): coef * "
                             "smooth_l1(believed derived stats {atk,def,spa,spd,spe}, TRUE derived stats) "
                             "over the REVEALED opp slots, so the SpreadBelief head LEARNS the opponent's "
                             "hidden EV spread (privileged training-only label from agent2's own team) "
                             "instead of sitting at the usage-mean prior (which over-estimates the largest-EV "
                             "stat → mis-priced damage/outspeed). The DamageOperator then prices damage "
                             "against the opponent's REAL bulk/offense/speed. 0.0 = OFF (byte-identical loss; "
                             "the head gets only the indirect op-damage gradient). REQUIRES --spread-belief. "
                             "TRAINING-only (not version-locked); metrics ride belief/spread_* "
                             "(mae, largest_bias→0, n_slots).")
    parser.add_argument("--spread-belief-nature", "--spread_belief_nature", dest="spread_belief_nature",
                        action=BoolFlag, default=None,
                        help="NATURE/EV generative spread head (gen3_nature_ev_belief_v1): swap SpreadBelief's "
                             "additive point-estimate for a head that predicts a NATURE categorical ⊕ its "
                             "Smogon prior + per-stat EVs ⊕ their prior (prior-fusion), assumes IV 31, and "
                             "COMPUTES the derived stat. The nature coupling (one stat ×1.1, one ×0.9) + the EV "
                             "budget are STRUCTURAL → the head can't inflate every stat, fixing the "
                             "'over-estimates the largest EV' order-statistic bias at the source. Supervised by "
                             "nature CE + EV regression (privileged inverted label) folded at --spread-belief-coef; "
                             "metrics ride belief/natureev_* (nature_acc, ev_mae). STRUCTURAL (version-checked, "
                             "fresh-only). REQUIRES --spread-belief. Off by default.")
    parser.add_argument("--hp-belief-mode", "--hp_belief_mode", dest="hp_belief_mode",
                        choices=["composed", "flat"], default=None,
                        help="How the opponent's 16 TYPED Hidden-Power channels are produced "
                             "(gen3_hp_belief_ablation_v1). BOTH arms reason over discrete TYPED HP "
                             "(355-370) and mask the typeless BP-0 num 237 — that is not the variable, "
                             "it is the 'opp HP reads immune' bug. "
                             "'composed' (DEFAULT) factors the belief as P(HP_t) = presence x P(type), "
                             "which makes 'a REVEALED Hidden Power must exist as SOME type' structural "
                             "(Sum_t P(HP_t) = presence, reveal-pinned), and applies the two certain-fact "
                             "eliminations: moveset exhaustion (4 moves seen, none is HP => ruled out) and "
                             "effectiveness narrowing (the HiddenPowerTracker's hard zeros). "
                             "'flat' is the ABLATION: no HPTypeBelief head — the multi-label move head "
                             "predicts the 16 typed channels INDEPENDENTLY off their own real per-typed "
                             "Smogon usage priors, i.e. Hidden Power is treated exactly like any other "
                             "move, with no factorisation, no constraint and no narrowing. Use it to "
                             "measure what the factorisation is worth. STRUCTURAL (version-checked, "
                             "fresh-only).")
    parser.add_argument("--hp-type-belief-coef", "--hp_type_belief_coef", dest="hp_type_belief_coef",
                        type=float, default=None,
                        help="HP-type-belief SUPERVISION weight (gen3_opp_hp_type_belief_v1): coef * "
                             "cross_entropy(HPTypeBelief posterior, TRUE opp HP type) over the REVEALED opp "
                             "slots that run Hidden Power (privileged training-only label from agent2's team — "
                             "Gen 3 never reveals the opp HP type). 0.0 = the head still runs and still gets "
                             "the op's damage gradient + the move-belief BCE through its typed channels; it "
                             "just has no direct CE, so it stays near the Smogon prior. gen3_typed_hp_belief_v1 "
                             "removed the old --hp-type-belief mode flag: the head is UNCONDITIONAL whenever "
                             "there is a move belief, because its 'off' state made the model reason over a "
                             "typeless BP-0 Hidden Power and priced a REVEALED HP as nonexistent. "
                             "TRAINING-only (not version-locked); metrics ride belief/hptype_* (acc, n_slots).")
    parser.add_argument("--item-belief-coef", "--item_belief_coef", dest="item_belief_coef",
                        type=float, default=None,
                        help="ITEM-belief SUPERVISION weight (gen3_item_belief_v1): coef * "
                             "cross_entropy(ItemBelief posterior, TRUE opp item num) over the REVEALED "
                             "opp slots (privileged training-only label from agent2's team — Gen 3 "
                             "reveals an item only when it acts, and NEVER a Choice Band). 0.0 = the "
                             "head still runs and still gets the op's p_cb damage gradient; it just "
                             "has no direct CE, so it stays near the Smogon prior. Requires "
                             "--item-belief (auto-zeroed with a warning otherwise). TRAINING-only "
                             "(not version-locked); metrics ride belief/item_* (acc, n_slots).")
    parser.add_argument("--value-from-dist", "--value_from_dist", dest="value_from_dist",
                        action=BoolFlag, default=None,
                        help="Phase B (gen3_dist_critic_v1): make the DISTRIBUTIONAL value head the critic "
                             "— GAE/bootstrap/deployment read E[Z] and the HL-Gauss CE is the primary value "
                             "loss (vf_coef weight); the scalar value_net freezes as a fallback. Requires "
                             "--value-dist-mode shaping. Resume-immutable (the belief-grad-mode class); flip "
                             "on a warm-started run with --allow-value-from-dist-change.")
    parser.add_argument("--allow-value-from-dist-change", "--allow_value_from_dist_change",
                        dest="allow_value_from_dist_change", action="store_true", default=False,
                        help="Permit the INTENTIONAL Phase-B critic-source migration on resume (the v45 gate "
                             "otherwise FATALs a drift). The offline probe confirmed E[Z]≈V, so the swap is "
                             "near-seamless. Loud notice; next save records the new mode. Needed once.")
    parser.add_argument("--allow-belief-grad-mode-change", "--allow_belief_grad_mode_change",
                        dest="allow_belief_grad_mode_change", action="store_true", default=False,
                        help="Permit an INTENTIONAL belief-grad-mode migration on resume (the v41 gate "
                             "otherwise makes a drift FATAL). detach() is value-preserving, so flipping "
                             "shaping<->detached on a converged checkpoint is weight-safe — only future "
                             "gradients change. Prints a loud notice; the next checkpoint save records "
                             "the new mode, so this flag is needed once per migration.")
    parser.add_argument("--belief-grad-mode", "--belief_grad_mode", dest="belief_grad_mode",
                        choices=list(BELIEF_GRAD_MODES), default=None,
                        help="gen3_belief_grad_mode_v1: WHICH gradient arrow between the STATE-prediction "
                             "belief heads (move / spread / hp-type / the species-moves-latent aux) and the "
                             "rest of the net is cut. THE TWO NON-DEFAULT MODES CUT OPPOSITE ARROWS. "
                             "'shaping' (default) = nothing cut: the heads READ the live trunk, so their "
                             "supervised + reinject gradients reshape it, and PPO trains the heads. "
                             "'detached' = they READ a STOP-GRAD trunk, so NO belief gradient reshapes the "
                             "trunk — it can't drag the trunk toward predicting hidden state at the policy's "
                             "expense (eliminates belief->trunk interference). "
                             "'label_only' (gen3_belief_label_only_v1) = the opposite cut: the heads' outputs "
                             "are PUBLISHED stop-grad to every forward consumer, so NO policy/value gradient "
                             "reaches a belief head's PARAMETERS and the belief is trained by its supervised "
                             "labels ALONE. The belief is still computed, reinjected and consumed by the op — "
                             "the policy reads it, it just can't push it off-calibration. Its trunk READ stays "
                             "live, so the label loss still teaches the trunk to encode hidden state (cutting "
                             "both would leave a probe on a trunk with no reason to carry the information, "
                             "still feeding the policy — that combination is deliberately not offered). "
                             "In ALL modes detach() is value-preserving, so the FORWARD is bit-identical and "
                             "only the training gradient differs. RESUME-IMMUTABLE (like --vf-coef, "
                             "version-checked on resume only — a frozen opponent's forward is unaffected). The "
                             "win-aligned heads (--win-prob-mode / --value-dist-mode) keep their own read_only.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per environment per rollout")
    parser.add_argument("--grad-checkpointing", "--grad_checkpointing", dest="grad_checkpointing",
                        action=BoolFlag, default=False,
                        help="Gradient-checkpoint the transformer encoder layers during the PPO "
                             "update (bit-exact; trades one extra forward on the idle GPU for "
                             "~5GB less activation VRAM). Off by default; safe to toggle per run.")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay (L2 regularisation). Default 1e-5 is conservative for PPO.")
