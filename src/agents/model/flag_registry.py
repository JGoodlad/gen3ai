"""THE declarative table of every model-relevant extractor toggle — and the five surfaces it binds.

WHY THIS MODULE EXISTS. A toggle that changes the feature extractor has to be spelled out in FIVE
independent places, by hand, in three files:

  1. the ``argparse`` entry in ``main.train_rl_agent``           (how a human sets it)
  2. the ``_resolve("name", default)`` line beside it            (how a FLAGLESS resume inherits it)
  3. ``extractor_arch.ARCH_ARG_KEYS`` (or ``_DERIVED``)          (how it reaches the extractor)
  4. the ``snapshot.current_model_version()`` keyword            (how a WORKER rebuilds the gate)
  5. the ``ModelVersion`` dataclass field                        (how it is RECORDED + version-gated)

Nothing enforced the five agreeing. Every historical failure in this class had the same shape — a
toggle that reaches the extractor but not the recorded config (so a resume version-checks against
an architecture it does not build), or reaches the argparse but not ``_resolve`` (so a flagless
resume silently reverts it to OFF). ``extractor_arch``'s own docstring records one; the gen-3
launch crash recorded in ``consequence_edges_test`` records another.

WHAT THIS BUYS. ``REGISTRY`` below is the single declaration. From it:

  * ``ARCH_ARG_KEYS`` and ``FROZEN_ARCH_KWARGS`` are **GENERATED** in ``extractor_arch`` — surface
    3 can no longer drift, because it is not written twice.
  * ``flag_registry_test.py`` **VALIDATES** surfaces 1, 2, 4 and 5 against the table, and a toggle
    present in one but missing from another fails with a message naming the missing site.
  * ``designs/flag_registry.md`` is generated from it (``python -m agents.model.flag_registry``),
    so the human-readable table cannot go stale either (``--check`` is the gate).

SCOPE. Exactly the **feature-extractor architecture toggles** — the things that pass through
``build_extractor_arch_kwargs``. Training-only loss coefficients (``move_belief_coef``,
``opp_belief_aux_coef``, …) are recorded on ``ModelVersion`` for provenance but never reach the
extractor, and reward-config / PPO hparams (``vf_coef``, ``draw_penalty``, …) are a different
mechanism with their own ``check_*``; neither is in scope here. Two toggles ARE listed whose CLI
surface is a *coefficient* rather than a flag of their own (``opp_belief_slots``, ``opp_intent``):
they carry ``derived=True`` and name the coef in ``source_arg``.

THE THREE TIERS. A flag's three ROLES — SELECT (choose it at launch), RECORD (write it down), GATE
(refuse a mismatched resume) — are independent, and only SELECT needs a CLI entry. So a settled
toggle can lose its flag without losing its explicitness:

    cli                the full surface: argparse + _resolve + recorded + gated
    config_only        NO argparse, NO _resolve. Frozen at ``default`` for every CLI-launched run,
                       still recorded in model_config.json and still resume-gated. The extractor
                       CONSTRUCTOR kwarg survives, so it stays reachable for an experiment.
    constructor_only   not recorded, not gated — reachable only by constructing the module.
                       ``pair_reduce``'s ``reduce_how`` is the precedent; nothing here is one yet.

THE FOUR CLASSES say what a mismatch MEANS, which is what picks the gate:

    structural         weights and/or the trained forward differ  -> ``check_compatible`` (gates
                       EVERY load, including frozen eval/pool/distill opponents)
    resume_immutable   the FORWARD is identical; only training differs -> a dedicated ``check_*``
                       on the resume path only, EXCLUDED from ``check_compatible`` (gating a frozen
                       opponent on it would be a false rejection that breaks league play)
    training_coef      recorded for provenance, never gated (a resume may change it freely)
    runtime            a perf knob — never recorded, never gated, NOT inherited on resume

DEPENDENCIES (``requires``). A sixth surface used to exist and was not listed above, because it was
not a surface at all — it was ~30 hand-written ``raise ValueError`` lines in
``Gen3FeaturesExtractor.__init__`` saying things like *"intent_threshold requires opp_intent"*.
Nothing outside that function knew them, so the launcher could not warn about an unsatisfiable
combination, ``designs/flag_registry.md`` could not show the graph, and there was no way to ask
"what is the minimum config that turns X on?" without reading the constructor.

``requires`` is now that data. It names the flags that must be ENABLED for this flag to be enabled
— ``is_enabled`` below defines both ends of "enabled", and its OFF convention (``False`` / ``0`` /
``'off'`` / ``'none'``) is the same one the CLI already uses.

It is deliberately WEAKER than the constructor in two places, and the constructor keeps the
stronger form: ``requires`` can say *"damage_op needs move_belief_mode enabled"* but not *"…in
{revealed, both}"*, and it cannot express a per-VALUE dependency at all — which is why
``edge_bias_families`` (whose 17 family letters each carry their own requirement, and ``h`` carries
none) declares nothing and stays bespoke. ``flag_requires_test.py`` enforces BOTH directions: every
declared dependency must actually make the constructor raise, and every constructor raise that
couples two registry flags must be declared here or listed in that test's bespoke table. Neither
side is allowed to know something the other does not.
"""
from __future__ import annotations

import argparse
import difflib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils.paths import repo_path


class Tier(str, Enum):
    CLI = "cli"
    CONFIG_ONLY = "config_only"
    CONSTRUCTOR_ONLY = "constructor_only"


class Klass(str, Enum):
    STRUCTURAL = "structural"
    RESUME_IMMUTABLE = "resume_immutable"
    TRAINING_COEF = "training_coef"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class ModelFlag:
    """One extractor toggle, and everything the five surfaces need to agree on.

    ``name`` is deliberately ONE name used as the extractor kwarg, the ``ModelVersion`` field, the
    ``current_model_version`` keyword AND (for a cli-tier flag) the argparse ``dest``. Every row in
    the pre-registry ``ARCH_ARG_KEYS`` mapped a key to an identical value; making that an invariant
    rather than a coincidence is most of what removes the drift.
    """
    name: str
    default: Any
    tier: Tier
    klass: Klass
    since: int                      # the MODEL_CONFIG_VERSION that introduced the field
    meaning: str                    # one line, for designs/flag_registry.md
    derived: bool = False           # lives in extractor_arch._DERIVED (a callable over args)
    source_arg: Optional[str] = None    # the args attribute that feeds it; defaults to ``name``
    cli_name: Optional[str] = None  # the long flag when it is NOT `--<arg with dashes>`
    note: str = ""                  # anything a reader needs that ``meaning`` cannot carry
    requires: Tuple[str, ...] = ()  # flags that must be ENABLED for this one to be (see below)

    @property
    def arg(self) -> str:
        """The ``args`` attribute this toggle reads (== ``name`` unless it is derived)."""
        return self.source_arg or self.name

    @property
    def cli_flag(self) -> str:
        """The long-form flag, for the argparse cross-check and the generated doc.

        Usually `--<arg>` with underscores dashed, but three toggles are set by a flag with a
        different name — `--damage-topk` writes `damage_topk_k`, and the `--damage-matrices`
        MODE flag desugars into the two `damage_matrices_*` bools. Those name their flag
        explicitly rather than being exempted from the argparse check, which is the whole point:
        the check found all three the first time it ran.
        """
        return self.cli_name or ("--" + self.arg.replace("_", "-"))


# ---------------------------------------------------------------------------------- THE REGISTRY
# Ordered by `since`, so a new toggle appends and the diff reads as history. `default` is the
# value a CLI-launched run gets when the flag is absent — for a config_only row that is the FROZEN
# value, i.e. the only value the CLI can now produce.
REGISTRY: Tuple[ModelFlag, ...] = (
    ModelFlag("attend_unrevealed_opponents", True, Tier.CONFIG_ONLY, Klass.STRUCTURAL, 8,
              "keep the opponent's still-hidden party attendable instead of key-masking it",
              note="DEMOTED (config_only) and frozen ON: it is a hard prerequisite of "
                   "opp_belief_cls_k>0 / opp_belief_slots / move_belief_mode!=off, so no run since "
                   "v16 has turned it off. The extractor kwarg still defaults False — the OFF "
                   "baseline stays constructible, it is just no longer selectable from the CLI."),
    ModelFlag("opp_belief_cls_k", 0, Tier.CLI, Klass.STRUCTURAL, 9,
              "k learned query tokens summarising the unrevealed opp party into both heads",
              requires=("attend_unrevealed_opponents",)),
    ModelFlag("opp_belief_slots", False, Tier.CLI, Klass.STRUCTURAL, 16,
              "learned unknown-mon tokens in the un-revealed opp slots + the BeliefHead",
              derived=True, source_arg="opp_belief_aux_coef",
              note="coef>0 is the enable signal; the COEF is a training hparam, the BOOL is the "
                   "version-checked arch toggle.",
              requires=("attend_unrevealed_opponents",)),
    ModelFlag("move_belief_mode", "off", Tier.CLI, Klass.STRUCTURAL, 17,
              "predict + reinject each opp mon's moveset (off|revealed|unrevealed|both)",
              requires=("attend_unrevealed_opponents",)),
    ModelFlag("damage_op", False, Tier.CLI, Klass.STRUCTURAL, 19,
              "build the differentiable GPU DamageOperator",
              requires=("move_belief_mode",)),
    ModelFlag("move_prior_fusion", False, Tier.CLI, Klass.STRUCTURAL, 20,
              "fuse the Smogon move-frequency prior into the move belief as a log-odds delta",
              requires=("move_belief_mode",)),
    ModelFlag("win_prob_mode", "none", Tier.CLI, Klass.STRUCTURAL, 22,
              "auxiliary win-probability side head off value_pooled (none|read_only|shaping)"),
    ModelFlag("damage_outgoing", False, Tier.CLI, Klass.STRUCTURAL, 23,
              "the op's OUTGOING per-move direction (our active's moves -> the opp active)",
              requires=("damage_op",)),
    ModelFlag("move_candidate_floor", 0.02, Tier.CLI, Klass.STRUCTURAL, 23,
              "the LEGAL-BUT-UNOBSERVED base probability of the move prior",
              note="must equal damage_tables._PRIOR_FLOOR; legality itself is unconditional (v65)."),
    ModelFlag("move_latent", False, Tier.CLI, Klass.STRUCTURAL, 24,
              "the context-free MoveLatentEncoder concatenated into the move network"),
    ModelFlag("spread_belief", False, Tier.CLI, Klass.STRUCTURAL, 25,
              "predict + reinject the opponent's hidden spread (5 derived stats per slot)"),
    ModelFlag("value_dist_mode", "none", Tier.CLI, Klass.STRUCTURAL, 29,
              "distributional VALUE side head off value_pooled (none|read_only|shaping)",
              requires=("value_dist_bins",)),
    ModelFlag("value_dist_bins", 0, Tier.CLI, Klass.STRUCTURAL, 29,
              "atom count = the value-dist head's output Linear width"),
    ModelFlag("value_dist_vmin", 0.0, Tier.CLI, Klass.RESUME_IMMUTABLE, 29,
              "low end of the return range the value-dist atoms span",
              note="value-MEANING, so check_value_dist on the resume path only."),
    ModelFlag("value_dist_vmax", 0.0, Tier.CLI, Klass.RESUME_IMMUTABLE, 29,
              "high end of the return range the value-dist atoms span",
              note="value-MEANING, so check_value_dist on the resume path only."),
    ModelFlag("damage_topk_k", 0, Tier.CLI, Klass.STRUCTURAL, 30,
              "K = how many of the opp active's believed moves the incoming matrix surfaces",
              cli_name="--damage-topk",
              requires=("damage_op", "move_latent", "damage_matrices_incoming")),
    ModelFlag("damage_matrices_outgoing", False, Tier.CLI, Klass.STRUCTURAL, 34,
              "our active's 4 moves x the opp's 6 mons, per-(move, mon) rolls",
              cli_name="--damage-matrices",
              note="set by the `--damage-matrices {off,outgoing,incoming,both}` MODE flag, which "
                   "desugars into this bool and `damage_matrices_incoming` before `_resolve`.",
              requires=("damage_op",)),
    ModelFlag("damage_matrices_incoming", False, Tier.CLI, Klass.STRUCTURAL, 35,
              "the enriched top-K incoming matrix (per opp move x per our mon)",
              cli_name="--damage-matrices",
              note="the other half of the `--damage-matrices` mode desugar; it also REUSES "
                   "`damage_topk_k` as its K.",
              requires=("damage_op", "move_latent")),
    ModelFlag("threat_prob_outspeed", False, Tier.CLI, Klass.STRUCTURAL, 36,
              "uncertainty-aware P(outspeed): divide the speed gap by the believed speed std"),
    ModelFlag("spread_belief_nature", False, Tier.CLI, Klass.STRUCTURAL, 40,
              "swap SpreadBelief's additive head for the NATURE/EV generative head",
              requires=("spread_belief",)),
    ModelFlag("belief_grad_mode", "shaping", Tier.CLI, Klass.RESUME_IMMUTABLE, 41,
              "which gradient arrow between the belief heads and the trunk is cut",
              note="detach() is value-preserving => the forward is bit-identical in every mode, so "
                   "check_belief_grad_mode on the resume path only."),
    ModelFlag("damage_candidate_k", 0, Tier.CLI, Klass.STRUCTURAL, 49,
              "cap the op's incoming candidate sweep at the K most-believed opponent moves",
              requires=("damage_op",)),
    ModelFlag("hp_belief_mode", "composed", Tier.CLI, Klass.STRUCTURAL, 53,
              "how the 16 typed Hidden-Power channels are produced (composed|flat)"),
    ModelFlag("entity_topk_seats", 0, Tier.CLI, Klass.STRUCTURAL, 54,
              "E4 — the opp active's top-K believed threat-move attention seats",
              requires=("damage_op", "move_latent")),
    ModelFlag("edge_bias_families", "off", Tier.CLI, Klass.STRUCTURAL, 56,
              "which physics families are delivered as additive per-pair attention biases"),
    ModelFlag("entity_tail_seats", False, Tier.CLI, Klass.STRUCTURAL, 57,
              "E5 — 6 per-opp-mon seats summarising the beyond-top-K belief mass",
              requires=("damage_op", "entity_topk_seats")),
    ModelFlag("consequence_topk", 6, Tier.CLI, Klass.STRUCTURAL, 59,
              "the consequence kernels' believed-candidate axis (C1b/C2/C3 k_cand + D4 k_bench)"),
    ModelFlag("value_threat_inject", False, Tier.CLI, Klass.STRUCTURAL, 64,
              "add the op's alpha-weighted incoming row to our tokens on the VALUE pool's copy",
              requires=("damage_op",)),
    ModelFlag("opp_intent", False, Tier.CLI, Klass.STRUCTURAL, 68,
              "the alpha (their move) / beta (their switch-in) supervised pointer heads",
              derived=True, source_arg="opp_intent_coef",
              note="coef>0 is the enable signal, like opp_belief_slots.",
              requires=("entity_topk_seats",)),
    ModelFlag("species_prior_fusion", False, Tier.CLI, Klass.STRUCTURAL, 69,
              "read BeliefHead's species head as a DELTA on the team-composition prior",
              requires=("opp_belief_slots",)),
    ModelFlag("t0_species_prior", False, Tier.CLI, Klass.STRUCTURAL, 72,
              "feed the T1 physics the model's own species belief, not the static usage table"),
    ModelFlag("opp_intent_grad_mode", "detached", Tier.CONFIG_ONLY, Klass.STRUCTURAL, 73,
              "whether alpha/beta's gradient reaches the shared trunk (detached|shaping)",
              note="DEMOTED (config_only) 2026-08-23, sweep #2, frozen at its own default. "
                   "MEASURED over 107 archived run configs: the 24 runs recording it are "
                   "'detached' UNANIMOUSLY, and the flag appears in ZERO of the 107 recorded "
                   "launcher commands — nobody has ever typed it. The 'shaping' arm stays "
                   "CONSTRUCTIBLE (the extractor kwarg is untouched), so re-opening the "
                   "trunk-exposure question costs one constructor argument, not a revert. "
                   "The three other unanimous-at-default flags found by the same census "
                   "(consequence_topk=6, damage_candidate_k=0, hp_belief_mode=composed) were "
                   "NOT demoted: all three ARE typed in the live run's command, so removing "
                   "their argparse entries would make its launcher_command unlaunchable on "
                   "restart — the cleanup journey's own live-run exclusion."),
    ModelFlag("intent_move_cell", False, Tier.CLI, Klass.STRUCTURAL, 77,
              "G3 — the c2 status-consequence family re-delivered, alpha-conditioned, through "
              "the pointer MOVE cell",
              requires=("opp_intent", "damage_op")),
    ModelFlag("value_entity_pool_full", False, Tier.CLI, Klass.STRUCTURAL, 82,
              "the entity pool's COMPLETE row set: + the refined global token and the "
              "hidden-opp belief queries",
              note="requires value_entity_pool; a separate flag/shape so v80-table checkpoints "
                   "keep loading. It is the SUCCESSOR the critic-route deletion wave actually "
                   "landed on — the nmr vf concat, the hidden-opp vf half and the seed readout "
                   "are all deleted, and this pool carries 97% of the critic's route "
                   "dependence (gen-14, dV 5.490 of all_off 5.635).",
              requires=("value_entity_pool",)),
    ModelFlag("history_events", False, Tier.CLI, Klass.STRUCTURAL, 81,
              "Tier H-B: the obs event-window records join the trunk as event SEATS "
              "(shared species/move embeddings, recency as content, TOKEN_TYPE_HISTORY)",
              note="the obs BLOCK is unconditional (v81 widening); this flag builds only the "
                   "consumer. Gen-13 candidate arm, gated on H-A's gen-12 verdict."),
    ModelFlag("value_entity_pool", False, Tier.CLI, Klass.STRUCTURAL, 80,
              "Stage-3 T3-DELIVER: ONE attention pool over the critic's entity rows (12 team "
              "tokens + op incoming rows), zero-init, vf-only",
              note="the designed SUCCESSOR contract of the bolt-on vf routes, and the one the "
                   "critic_route_audit picked: gen-14 dV 5.490 vs threat 1.069 and every other "
                   "route below 0.32. The seed readout it succeeded is deleted; threat-inject "
                   "KEEPS (its deadline discharged at 1.0686)."),
    ModelFlag("item_belief", False, Tier.CLI, Klass.STRUCTURAL, 83,
              "the hidden-ITEM belief head: per-opp-slot posterior over item nums, Smogon "
              "usage prior ⊕ zero-init trunk delta; the op's p_cb unrevealed branch consumes "
              "its publication (revealed stays exact 0/1)",
              note="BeliefBank's seventh row (--item-belief-coef supervises the revealed "
                   "slots). Cold start posterior == the Smogon prior exactly; its CB column is "
                   "within ~0.6% of the static table (row-floor renorm), so enabling is "
                   "~behavior-preserving at init and the delta must EARN its movement."),
    ModelFlag("intent_threshold", False, Tier.CLI, Klass.STRUCTURAL, 84,
              "the α-weighted threshold operator p_thresh(τ,⋛): Focus Punch / Substitute / "
              "Endure / Destiny Bond / Endeavor through the pointer MOVE cell (+ p_KO as "
              "per-slot context)",
              note="design_conditional_execution.md §3.0 build-order step 3. Requires "
                   "opp_intent + damage_op (+ the top-K pair-cell stash at runtime). The "
                   "projection is zero-init ⇒ ON-at-init bit-identical. The flag used to "
                   "build a SECOND consumer, the p_KO vf route (the ledger-H1 payoff); the "
                   "critic-route deletion wave retired that half on dV 0.155/0.136 against a "
                   "0.39 bar. This flag is now POLICY-ONLY, and that is deliberate.",
              requires=("opp_intent", "damage_op")),
    ModelFlag("intent_conditional", False, Tier.CLI, Klass.STRUCTURAL, 85,
              "the remaining α-conditioned mechanic cells: Counter/Mirror Coat's category "
              "test, flinch's (1−α_SWITCH) term, Explosion's execute/into-switch facts + the "
              "β-weighted trade KO (the FIRST forward-side β consumer), Protect's α-weighted "
              "avoided quantities, Magic Coat's oracle-verified reflect set, Pursuit's ×2 "
              "doubling trigger (port-verified departing-target rule)",
              note="design_conditional_execution.md build steps 4+5+6+7. Requires opp_intent + "
                   "damage_op + damage_outgoing + damage_matrices_outgoing (the arrival pko "
                   "source). β is PUBLISHED like α (label_only cuts the PPO route at the same "
                   "boundary). Zero-init ⇒ ON-at-init bit-identical; G3-gated like "
                   "intent_threshold.",
              requires=("opp_intent", "damage_op", "damage_outgoing", "damage_matrices_outgoing")),
    ModelFlag("pair_outcome_cell", False, Tier.CLI, Klass.STRUCTURAL, 93,
              "the UNIFIED per-pair OUTCOME VECTOR + its α-weighted delivery: one "
              "pair_in[their move k, our mon j] carrying damage AND status-by-identity AND "
              "neutralization AND tempo_cost in the same currency, reduced by ONE α over the "
              "move axis (Contract W) and delivered to the pointer MOVE cell",
              note="design_opponent_intent.md §5.1/§5.3 + design_pair_reduction.md §2.1/§9a. "
                   "Phase A — the MOVE-cell half; the switch cell and the β cells are Phase B. "
                   "Requires damage_op (the physics has one source) but NOT opp_intent: with no "
                   "intent head α falls back to the shipped R1 belief_mean rung (α := w/Σw), so "
                   "the DELIVERY claim is testable apart from the DISTRIBUTION claim. Zero-init "
                   "⇒ ON-at-init bit-identical.",
              requires=("damage_op",)),
    ModelFlag("pair_outcome_switch", False, Tier.CLI, Klass.STRUCTURAL, 94,
              "Phase B — the SAME α-reduced unified outcome row, per DEFENDER, delivered to the "
              "pointer SWITCH cell (+ spin_denied: our Ghost candidate denying their believed "
              "Rapid Spin, priced by the hazard stake it preserves)",
              note="design_pair_reduction.md §2.1's CANONICAL defect, at its own sink: the switch "
                   "logit's cell holds ten damage numbers, one speed number, two belief-mass "
                   "numbers and NO status coordinate in any currency, so 'they will click "
                   "Will-O-Wisp, bring the Natural Cure mon' is unrepresentable. The FIRST module "
                   "to widen the switch cell. Requires damage_op but NOT pair_outcome_cell — the "
                   "two deliver one tensor to two sinks and coupling them would make a result "
                   "unattributable. Zero-init ⇒ ON-at-init bit-identical.",
              requires=("damage_op",)),
    ModelFlag("switch_branch_cell", False, Tier.CLI, Klass.STRUCTURAL, 94,
              "Phase B — OA2, the SWITCH-BRANCH move cell: E[our move | they switch] contracted "
              "over β (the arrival), kept DECORRELATED from the stay branch, plus the Rapid-Spin "
              "spinblock (the Pursuit mirror: α_SWITCH × β × P(arrival is Ghost)) and Protect's "
              "α-derived attack mass (the c4 successor — decay × will-they-attack)",
              note="design_conditional_opponent_cells.md §2 + the owner's Rapid Spin / Protect "
                   "specs. Requires opp_intent with NO fallback, and that is substantive: the R1 "
                   "belief_mean rung is a presence belief over their MOVES and carries no switch "
                   "class, so α_SWITCH would be identically 0 and every coordinate would assert "
                   "'they never switch'. §4.1's hard prerequisite is CLOSED "
                   "(gen3_unrevealed_outgoing_prior_v1 prices unrevealed arrivals against the "
                   "expected-latent defender); the one residue is that pko stays NULLED there, so "
                   "e_pko_switch is deflated in proportion to β's hidden mass while e_high_switch "
                   "carries the magnitude. Zero-init ⇒ ON-at-init bit-identical.",
              requires=("opp_intent", "damage_op", "damage_matrices_outgoing")),
    ModelFlag("conditional_threat_cell", False, Tier.CLI, Klass.STRUCTURAL, 95,
              "Phase C — OA1, the CONDITIONAL THREAT CELL (the defensive pivot): the four "
              "α-contracted coordinates the reduced outcome row structurally cannot carry — "
              "e_pko_acc (accuracy x P(KO), the product §0.2(2) says the OP must form), "
              "e_type_mult (the one channel not divided by the defender's own bulk) and the two "
              "§0.2(3) MARGINS against our own HP (max roll and crit roll), on the pointer SWITCH "
              "cell",
              note="design_conditional_opponent_cells.md §1 + §0.2. THREE of §1.2's clauses are "
                   "SUPERSEDED and the substitutions are recorded in conditional_threat.py: the "
                   "λ-weighted `w` is NOT built (pair_alpha is the shipped distribution; a second "
                   "one would be a second α), `high`/`pko`/`status_lands` are already delivered by "
                   "pair_outcome_switch, and §1.3's --damage-matrices-outgoing-all is VOID (deleted "
                   "at v88). Requires damage_op + damage_matrices_incoming (the only producer of "
                   "the per-(defender, seat) type multiplier AND of the top-K seat axis), NOT "
                   "opp_intent — the R1 belief_mean fallback is MEANINGFUL here because every "
                   "coordinate is a 'what lands on me if they attack' contraction. Independent of "
                   "pair_outcome_switch on purpose: two quantities, one sink, attributable "
                   "separately. Zero-init ⇒ ON-at-init bit-identical.",
              requires=("damage_op", "damage_matrices_incoming")),
    ModelFlag("pair_value_route", False, Tier.CLI, Klass.STRUCTURAL, 95,
              "Phase C — PV, the pair-VALUE CRITIC route: the α-reduced unified outcome row for "
              "our mon j injected as TOKEN CONTENT on mon j's own token inside CLSPool, on the "
              "VALUE pool's copy only — the first per-entity route by which the critic reads the "
              "status / neutralization / tempo currency at all (today it reaches vf only as the s3 "
              "edge family's softmax-normalised RATIO)",
              note="design_opponent_intent.md §7a(2). ⚠️ C4 RE-ENTRY CONDITION: any α/β-critic "
                   "route may be BUILT opt-in but its ENABLING owes the C4-style offline gate "
                   "first (ledger C6 — five v89 routes trained off zero with entity_pool carrying "
                   "decisively and the critic's stall over-confidence did NOT move; the delivery "
                   "line is EXHAUSTED). Token content, NOT the v89 _value_pooled_routes seam: a "
                   "post-pool additive route must collapse the J axis and the only equivariant "
                   "collapse is a sum, which cannot tell one mon losing 90% from six losing 15%. "
                   "α is the R1 belief_mean rung UNCONDITIONALLY — ORDERING, not preference: "
                   "value_cls pools before the α/β heads are scored, and §7a(2) pre-registers that "
                   "substitution as the DELIVERY-claim test. vf-only at ANY weight (the augmented "
                   "tensor is a local); zero-init ⇒ ON-at-init bit-identical. Requires damage_op.",
              requires=("damage_op",)),
    ModelFlag("op_drop_renders", False, Tier.CLI, Klass.STRUCTURAL, 86,
              "design_op_tensors step 3: the op's flat forward block loses the three RENDER "
              "regions (omx/imx/OAX — serialization-only since the concat's deletion); "
              "selection machinery + every consumer stash survive, out_gain shrinks",
              note="every surviving offset unchanged (renders appended last), so pi/vf at init "
                   "are bit-identical to renders-on — pinned by test. The prober decodes a "
                   "lean run's blocks with the run's own config flags."),
    ModelFlag("op_believed_lean", False, Tier.CLI, Klass.STRUCTURAL, 86,
              "the lean d3 physics price the attacker from the BELIEVED spread instead of the "
              "legacy de-timid fiction — the B-spread correctness fix at the last de-timid "
              "site the edges read",
              note="requires spread_belief + damage_op. Forward-math only (no state_dict "
                   "change): the version gate is the ONLY thing rejecting a mismatched resume.",
              requires=("spread_belief", "damage_op")),
    ModelFlag("cf_evidential", False, Tier.CLI, Klass.STRUCTURAL, 98,
              "the EVIDENTIAL Beta readout over P(win|state) off value_pooled — (α, β) via "
              "softplus+1, trained by the Beta-Binomial marginal likelihood of the counterfactual "
              "factory's rollout COUNTS",
              note="IN the registry because it is a `Gen3FeaturesExtractor` constructor kwarg that "
                   "builds a MODULE — the win_prob_mode / value_dist_mode precedent exactly, and "
                   "the registry's declared scope is 'the things that pass through "
                   "build_extractor_arch_kwargs'. Its two coefficients (--cf-evidential-coef / "
                   "--cf-evidential-reg) are NOT: they are training-only loss weights in the "
                   "--opd-coef / --cf-winprob-coef class, set on the MODEL rather than the "
                   "extractor. Stronger than its two precedents in one way: there is no "
                   "read_only/shaping split, because the head's input is detached "
                   "UNCONDITIONALLY — it is a pure supervised readout that feeds nothing forward "
                   "and is not even CALLED by the forward pass. So OFF is byte-identical AND "
                   "ON-at-coef-0 is bit-identical in pi/vf (it is built LAST, so no earlier "
                   "module's init RNG draw moves). No `requires`: it reads `value_pooled`, which "
                   "is unconditional."),
    ModelFlag("cf_twin_heads", False, Tier.CLI, Klass.STRUCTURAL, 99,
              "the TWIN win-prob heads B and C off value_pooled — B trained on the cf-labelled "
              "states with SINGLE-OUTCOME labels, C on the same states with TIGHT-MC labels, both "
              "additionally carrying head A's on-policy BCE",
              note="The owner-authorized amendment to the signed R1 pre-registration (ledger "
                   "2026-08-22 evening, 'Three owner sign-offs' item 3): the arm's primary "
                   "comparison becomes a WITHIN-RUN paired head difference instead of a run-vs-run "
                   "one, so B-A isolates coverage and C-B isolates pure variance reduction on an "
                   "identical trunk over identical states. IN the registry for the "
                   "`cf_evidential`/`win_prob_mode` reason: a Gen3FeaturesExtractor constructor "
                   "kwarg that builds MODULES. Its coefficient (--cf-twin-coef) is NOT — a "
                   "training-only loss weight in the --opd-coef class. Head-only ALWAYS in v1 "
                   "(both twins read a DETACHED value_pooled in every term), so this measures the "
                   "LABEL effect on a trunk frozen with respect to them; trunk exposure stays a "
                   "cross-run question. Never called by the forward, built LAST: OFF byte-identical, "
                   "ON-at-coef-0 bit-identical in pi/vf. No `requires` on cf_evidential (orthogonal "
                   "readouts of the same unconditional value_pooled). It DOES require "
                   "`win_prob_mode`, and that is declared rather than left to the CLI: head A is "
                   "`win_head`, so `--win-prob-mode none` leaves the factorial with no control arm "
                   "— and `checkargs` walks this column, so without it an operator validating a "
                   "recorded launcher_command gets exit 0 on a command the child then refuses, "
                   "which is the launch-crash-fix loop checkargs exists to end.",
              requires=("win_prob_mode",)),
    ModelFlag("cf_shadow_critic", False, Tier.CLI, Klass.STRUCTURAL, 99,
              "the passive SHADOW CRITIC off value_pooled — a value twin trained on tight-MC "
              "`mc_return` labels (the run's own shaped return, PopArt frame), which never "
              "computes an advantage and never enters GAE",
              note="The staged PROMOTION PATH for critic surgery, not the surgery: swapping the "
                   "live critic for an MC-grounded one is a critic ROUTE change and owes the C4 "
                   "offline gate, so this head accumulates the evidence (shadow-vs-live-V "
                   "divergence on the same minibatch states) as a published number instead of an "
                   "argument. The `pubval` structural precedent. Detached ALWAYS — there is no "
                   "read_only/shaping split — and never called by the forward, so OFF is "
                   "byte-identical and ON-at-coef-0 is bit-identical in pi/vf. Its coefficient "
                   "(--cf-shadow-coef) is training-only, the --opd-coef class. No `requires`: it "
                   "reads value_pooled, which is unconditional."),
)

BY_NAME: Dict[str, ModelFlag] = {f.name: f for f in REGISTRY}

if len(BY_NAME) != len(REGISTRY):                    # a duplicate would silently shadow a row
    _dupes = sorted({f.name for f in REGISTRY if sum(g.name == f.name for g in REGISTRY) > 1})
    raise AssertionError(f"duplicate flag_registry names: {_dupes}")

_unknown_req = sorted({(f.name, d) for f in REGISTRY for d in f.requires if d not in BY_NAME})
if _unknown_req:                                     # a typo'd dependency would silently never fire
    raise AssertionError(f"flag_registry `requires` names no such flag: {_unknown_req}")

for _f in REGISTRY:                                  # a self-requirement can never be satisfied
    if _f.name in _f.requires:
        raise AssertionError(f"flag_registry: {_f.name} requires itself")


# The OFF values, one convention for every type the registry carries. `False` for a bool, `0` for a
# width/count (`entity_topk_seats`, `value_dist_bins`), and the two mode-string spellings the CLI
# already uses. It is a MODULE-level rule rather than per-flag data because the CLI enforces it too:
# every mode flag in `train_rl_agent` spells its disabled state exactly one of these ways.
OFF_VALUES = (False, 0, "off", "none")


def is_enabled(value: Any) -> bool:
    """Is this flag value the ON state, for the purpose of ``requires``?

    Note this is NOT ``bool(value)``: a mode string's OFF state is the truthy ``'off'`` / ``'none'``,
    and reading it as enabled is a real bug this project has shipped before (the dead-kwarg
    sanitizer refused every OFF-mode checkpoint until it stopped testing truthiness). Floats are
    excluded on purpose — ``move_candidate_floor`` and the ``value_dist_v*`` bounds are magnitudes,
    not switches, and nothing depends on them.
    """
    if isinstance(value, str):
        return value not in ("off", "none")
    return bool(value)


def requirement_closure(name: str) -> Tuple[str, ...]:
    """Every flag that must be enabled to enable ``name``, transitively, excluding ``name``.

    Order is deterministic (depth-first over the declared order) so a caller building a minimal
    config gets a stable answer. Cycles are impossible — a cycle would make both flags
    unenableable, so ``flag_requires_test`` fails on one rather than this silently looping.
    """
    seen: List[str] = []

    def walk(n: str, stack: Tuple[str, ...]) -> None:
        for dep in BY_NAME[n].requires:
            if dep in stack:
                raise AssertionError(f"flag_registry `requires` cycle: {' -> '.join(stack + (dep,))}")
            if dep not in seen:
                seen.append(dep)
            walk(dep, stack + (dep,))

    walk(name, (name,))
    return tuple(seen)


def cli_flags() -> Tuple[ModelFlag, ...]:
    return tuple(f for f in REGISTRY if f.tier is Tier.CLI)


def config_only_flags() -> Tuple[ModelFlag, ...]:
    return tuple(f for f in REGISTRY if f.tier is Tier.CONFIG_ONLY)


def recorded_flags() -> Tuple[ModelFlag, ...]:
    """Rows that MUST have a ``ModelVersion`` field + a ``current_model_version`` keyword.

    Everything except the ``runtime`` class and the ``constructor_only`` tier — those are, by
    definition, the two states of "not written down".
    """
    return tuple(f for f in REGISTRY
                 if f.klass is not Klass.RUNTIME and f.tier is not Tier.CONSTRUCTOR_ONLY)


def extractor_kwarg_flags() -> Tuple[ModelFlag, ...]:
    """Rows that reach ``Gen3FeaturesExtractor`` — every tier except ``constructor_only``."""
    return tuple(f for f in REGISTRY if f.tier is not Tier.CONSTRUCTOR_ONLY)


# ------------------------------------------------------------- designs/flag_registry.md generator
_DOC = str(repo_path("designs", "flag_registry.md"))

SECTIONS = ("registry-table",)


def _begin(name: str) -> str:
    return f"<!-- BEGIN GENERATED: {name} -->"


def _end(name: str) -> str:
    return f"<!-- END GENERATED: {name} -->"


def extract_section(text: str, name: str) -> str:
    b, e = _begin(name), _end(name)
    if text.count(b) != 1 or text.count(e) != 1:
        raise AssertionError(
            f"marker pair for {name!r} must appear exactly once in {_DOC} "
            f"(BEGIN x{text.count(b)}, END x{text.count(e)})")
    return text.split(b, 1)[1].split(e, 1)[0].strip("\n")


def fill_section(text: str, name: str, body: str) -> str:
    b, e = _begin(name), _end(name)
    extract_section(text, name)                      # validates the pair exists exactly once
    return text.split(b, 1)[0] + b + "\n" + body.rstrip("\n") + "\n" + e + text.split(e, 1)[1]


def _cell(v: Any) -> str:
    return f"`{v!r}`" if isinstance(v, str) else f"`{v}`"


def registry_table_section() -> str:
    lines = [
        f"{len(REGISTRY)} toggles — "
        f"{len(cli_flags())} `cli`, {len(config_only_flags())} `config_only`, "
        f"{len([f for f in REGISTRY if f.tier is Tier.CONSTRUCTOR_ONLY])} `constructor_only`.",
        "",
        "| toggle | CLI | tier | class | default | since | requires | meaning |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in REGISTRY:
        cli = f"`{f.cli_flag}`" if f.tier is Tier.CLI else "—"
        if f.derived and f.tier is Tier.CLI:
            cli += " *(coef)*"
        req = ", ".join(f"`{d}`" for d in f.requires) or "—"
        lines.append(
            f"| `{f.name}` | {cli} | `{f.tier.value}` | `{f.klass.value}` | "
            f"{_cell(f.default)} | v{f.since} | {req} | {f.meaning} |")

    dependents = [f for f in REGISTRY if f.requires]
    lines += [
        "",
        f"**Dependencies.** {len(dependents)} of {len(REGISTRY)} toggles name a `requires`. The "
        "column lists only DIRECT dependencies; the transitive closure is "
        "`flag_registry.requirement_closure(name)` — e.g. enabling `intent_conditional` also "
        f"pulls in {', '.join('`%s`' % d for d in requirement_closure('intent_conditional'))}. "
        "\"Enabled\" follows `flag_registry.is_enabled`: `False` / `0` / `'off'` / `'none'` are "
        "OFF, everything else is ON.",
        "",
        "Two constructor checks are STRONGER than the column can say, and stay hand-written in "
        "`Gen3FeaturesExtractor.__init__`: `damage_op` needs `move_belief_mode` in "
        "*{revealed, both}* specifically (the column can only say \"enabled\"), and "
        "`edge_bias_families` carries a requirement PER FAMILY LETTER — most families need "
        "`damage_op`, `d1/s1/c1/c2` also need `damage_outgoing`, `d3/s3` need "
        "`entity_topk_seats > 0`, `r` needs `history_events`, and `h` needs nothing — which no "
        "flag-level declaration can represent. `flag_requires_test.py` holds that list and fails "
        "if a new coupling appears in neither place.",
    ]
    notes = [f for f in REGISTRY if f.note]
    if notes:
        lines += ["", "**Notes**", ""]
        lines += [f"- `{f.name}` — {f.note}" for f in notes]
    return "\n".join(lines)


def generate_sections() -> Dict[str, str]:
    return {"registry-table": registry_table_section()}


def render(text: str, sections: Dict[str, str]) -> str:
    for name in SECTIONS:
        text = fill_section(text, name, sections[name])
    return text


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true",
                   help="exit 1 with a diff summary if designs/flag_registry.md no longer matches "
                        "the table in this module (staleness gate)")
    a = p.parse_args(argv)

    with open(_DOC) as fh:
        current = fh.read()
    sections = generate_sections()

    if a.check:
        stale = 0
        for name in SECTIONS:
            have, want = extract_section(current, name), sections[name].strip("\n")
            if have != want:
                stale += 1
                print(f"STALE section {name!r} in {os.path.relpath(_DOC)}:")
                for line in difflib.unified_diff(have.splitlines(), want.splitlines(),
                                                 "committed", "generated", lineterm="", n=1):
                    print(f"  {line}")
        if stale:
            print(f"\n{stale} generated section(s) drifted — regenerate with:")
            print("  python -m agents.model.flag_registry")
            return 1
        print(f"OK {os.path.relpath(_DOC)} generated sections match the registry")
        return 0

    new = render(current, sections)
    if new != current:
        with open(_DOC, "w") as fh:
            fh.write(new)
        print(f"rewrote generated sections in {os.path.relpath(_DOC)}")
    else:
        print(f"{os.path.relpath(_DOC)} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
