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
                       EVERY load, including frozen eval/pool opponents)
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


class Family(str, Enum):
    """WHICH SURFACE a toggle belongs to — orthogonal to `Klass`, and read by the ARCH-SURFACE
    guard (`main.train.arch_surface`).

    `Klass` already separates the reward / training-coefficient / runtime families from the
    architecture: they are `training_coef` and `runtime`, and the guard drops them by class. The one
    split `Klass` cannot express is inside `structural`: the CRITIC READOUTS (`win_prob_mode`, the
    ride-along heads) build modules exactly the
    way an entity seat does, but they are the quantity an experiment is deliberately CHANGING —
    the win-prob critic implies one of them — so a guard that demanded they
    match the production mirror would refuse every critic arm it exists to protect.

    So the guard's key set is `structural` AND `ARCH`, and the exclusion is DECLARED here rather
    than hand-listed at the guard: a new critic readout marks itself `family=Family.CRITIC` in the
    same row that declares everything else about it, and `arch_surface_test` fails if a row is
    added to the surface without a decision being made.
    """
    ARCH = "arch"
    CRITIC = "critic"


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
    family: Family = Family.ARCH    # which SURFACE it belongs to — see `Family`
    #: For a DERIVED row only: the value to write into `source_arg` to turn this toggle ON when
    #: `designs/production_config.json` cannot supply one. A derived toggle's CLI surface is a
    #: training COEFFICIENT; the production mirror predates config v125 (which made
    #: `opp_intent_coef` a recorded `ModelVersion` field) and carries only the BOOL, so without this
    #: the umbrella would have no way to enable a toggle production has ON. The magnitude is a training dose, not an
    #: architecture fact: `--arch production` writes it and an explicit `--opp-intent-coef` wins.
    on_value: Optional[Any] = None
    #: The SEPARATE training-coefficient field that supervises this toggle's head, where the toggle
    #: is a mode/bool of its own (a `derived` row has no such split — its CLI surface IS the
    #: coefficient, so it declares `source_arg` instead). NOT part of the arch surface and never
    #: written by `--arch production`: a dose is training, not architecture, and pinning it would
    #: make the umbrella refuse the ablations it exists to leave free.
    #:
    #: It is declared here so the ARCH-SURFACE guard can NAME it. `--arch production` builds the
    #: production network with `move_belief_coef` 0.0 and `spread_belief_coef` 0.0 where production
    #: trains at 0.05 — the same class of silent-omission the guard exists to end, one layer down —
    #: and a block that listed only what it applied would read as coverage of what it did not.
    coef_arg: Optional[str] = None

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
              "the hidden-opponent belief: X5's hypothesis tokens in the un-revealed opp slots (the T0 "
              "hypothesis builder) + the BeliefHead",
              derived=True, source_arg="opp_belief_aux_coef",
              note="coef>0 is the enable signal; the COEF is a training hparam, the BOOL is the "
                   "version-checked arch toggle. Since the X5 VERSION BREAK (config v144) it builds X5's "
                   "hypothesis tokens (designs/endstate/design_x5_belief_tokens.md) — the only belief "
                   "representation: `--belief-tokens` and its blob arm (a constant learned token per hidden "
                   "slot, BeliefSlots) are DELETED. So it REQUIRES what X5 requires: t0_species_prior (the "
                   "scores are log P_T0 + a learned delta), move_belief_mode (the active's move group), "
                   "move_prior_fusion (a hypothesis slot's move prior is its species' Smogon row), "
                   "opp_intent (the flat opponent pointer reads the hypothesis set — the two are ONE switch: "
                   "opp_intent's own build refuses opp_belief_slots off, a coupling the acyclic `requires` "
                   "graph states from this side only), entity_topk_seats (>= 4: revealed moves take the "
                   "first seats) and entity_tail_seats (OTHER_move's token is the opponent active's E5 tail "
                   "seat). An OFF value of any of them with the belief family on is REFUSED at build.",
              requires=("attend_unrevealed_opponents", "t0_species_prior", "move_belief_mode", "move_prior_fusion",
                        "opp_intent", "entity_topk_seats", "entity_tail_seats")),
    ModelFlag("move_belief_mode", "off", Tier.CLI, Klass.STRUCTURAL, 17,
              "predict + reinject each opp mon's moveset (off|revealed|unrevealed|both)",
              requires=("attend_unrevealed_opponents",), coef_arg="move_belief_coef"),
    ModelFlag("damage_op", False, Tier.CLI, Klass.STRUCTURAL, 19,
              "build the differentiable GPU DamageOperator",
              requires=("move_belief_mode",)),
    ModelFlag("move_prior_fusion", False, Tier.CLI, Klass.STRUCTURAL, 20,
              "fuse the Smogon move-frequency prior into the move belief as a log-odds delta",
              requires=("move_belief_mode",)),
    ModelFlag("win_prob_mode", "none", Tier.CLI, Klass.STRUCTURAL, 22,
              "auxiliary win-probability side head off value_pooled (none|read_only|shaping)",
              family=Family.CRITIC),
    ModelFlag("damage_outgoing", False, Tier.CLI, Klass.STRUCTURAL, 23,
              "the op's OUTGOING per-move direction (our active's moves -> the opp active)",
              requires=("damage_op",)),
    ModelFlag("move_candidate_floor", 0.02, Tier.CLI, Klass.STRUCTURAL, 23,
              "the LEGAL-BUT-UNOBSERVED base probability of the move prior",
              note="must equal damage_tables._PRIOR_FLOOR; legality itself is unconditional (v65)."),
    ModelFlag("move_latent", False, Tier.CLI, Klass.STRUCTURAL, 24,
              "the context-free MoveLatentEncoder concatenated into the move network",
              coef_arg="move_belief_latent_coef"),
    ModelFlag("spread_belief", False, Tier.CLI, Klass.STRUCTURAL, 25,
              "predict + reinject the opponent's hidden spread (5 derived stats per slot)",
              coef_arg="spread_belief_coef"),
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
              "how the 16 typed Hidden-Power channels are produced (composed|flat)",
              coef_arg="hp_type_belief_coef"),
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
              requires=("damage_op",),
              note="ON in production. Architecture audit F10 (owner 2026-10-06) screens 'off': since v142 "
                   "(gen3_value_threat_inject_off_v1) an OFF build with the op on CONSTRUCTS the projection and the "
                   "policy RETIRES it after SB3's orthogonal re-init (`ExtractorApi.retire_value_threat_inject`), so "
                   "every other parameter's initial bytes equal production's and `--value-threat-inject off` is a "
                   "one-lever arm. The critic then reads the op's incoming rows through the trunk (`prefuse_proj`) "
                   "and `value_entity_pool`'s op-row source only."),
    ModelFlag("opp_intent", False, Tier.CLI, Klass.STRUCTURAL, 68,
              "the opponent-intent readout: X5's FLAT opponent pointer (their move seats + OTHER_move + "
              "each switch target + OTHER_species, one softmax), re-expressed as alpha / beta for the "
              "consumers",
              derived=True, source_arg="opp_intent_coef", on_value=0.05,
              coef_arg="intent_label_bot_weight",
              note="coef>0 is the enable signal, like opp_belief_slots. `coef_arg` is the heads' "
                   "BOT-label weight, which is a separate dose from the enable signal: production "
                   "trains it at 0.25 and a fresh run defaults to 1.0, so `--arch production` "
                   "names it rather than setting it. Since the X5 VERSION BREAK (config v144) the alpha / "
                   "beta HEADS are the deleted blob path's (still constructed for their init draws, retired "
                   "by the policy before the optimizer); the readout is X5's flat pointer, so opp_intent "
                   "also needs opp_belief_slots and X5's requirements (declared on opp_belief_slots; the "
                   "constructor refuses opp_intent without it).",
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
                   "~behavior-preserving at init and the delta must EARN its movement.",
              coef_arg="item_belief_coef"),
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
    ModelFlag("ridealong_ensemble", 0, Tier.CLI, Klass.STRUCTURAL, 126,
              "K DETACHED win-prob heads on value_pooled (bootstrap masks + randomized priors): "
              "their disagreement is V's EPISTEMIC uncertainty (0 = off)",
              note="The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface." " It REQUIRES win_prob_mode: the members predict V's own win "
                   "target.",
              requires=("win_prob_mode",),
              family=Family.CRITIC),
    ModelFlag("ridealong_rnd", False, Tier.CLI, Klass.STRUCTURAL, 126,
              "a DETACHED RND novelty head (frozen random target + trained predictor over the "
              "running-normalised RAW observation): its error is how rarely a state was seen",
              note="The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface." " The input is the observation, not the trunk features, so the "
                   "novelty is not confounded by representation drift.",
              family=Family.CRITIC),
    ModelFlag("ridealong_adv", 0, Tier.CLI, Klass.STRUCTURAL, 126,
              "K DETACHED per-action A heads over the pointer head's own tokens, centred under pi, "
              "regressed on the GAE advantage of the action taken (0 = off)",
              note="The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface.",
              family=Family.CRITIC),
    ModelFlag("ridealong_opp", 0, Tier.CLI, Klass.STRUCTURAL, 126,
              "K DETACHED opponent-effect B heads over alpha's support (believed move seats by "
              "move id + SWITCH), centred under alpha, regressed on the same advantage (0 = off)",
              note="The DETACHED RIDE-ALONG baseline (owner, 2026-09-30; EXPERIMENT_BACKLOG X25 / X4a). The extractor builds NOTHING for it: it records the kwarg, and `Gen3DualHeadMaskablePolicy` builds `policy.ridealong` after SB3's `_build` (outside `policy.optimizer` and the ortho-init apply), from a PRIVATE seed inside `fork_rng`. Every input is `.detach()`ed and the learner steps the heads with their own optimizer before PPO's loss is assembled, so a run with the heads learns EXACTLY what the same run without them learns (pinned bit-for-bit by `ridealong_heads_test`). STRUCTURAL all the same: the heads' parameters are the state_dict delta, so check_compatible compares the value. family=CRITIC: readouts an experiment varies, never on the production ARCH surface." " It REQUIRES opp_intent: B's columns and centring are alpha's. "
                   "The simple pre-X5 parameterisation, to be re-based onto X5's flat pointer.",
              requires=("opp_intent",),
              family=Family.CRITIC),
    ModelFlag("ridealong_rnd_variants", "off", Tier.CLI, Klass.STRUCTURAL, 127,
              "the DETACHED RND VARIANT ENSEMBLE beside --ridealong-rnd (base): a canonical comma "
              "list of fast,decay,small,feat ('all' = every one; 'off' = none), each its own "
              "predictor + optimizer + statistics, compared PAIRED against base",
              note="The X26 RND strategy comparison (owner, 2026-09-30: \"ensemble RND, toss one a different learning rate or something, so we knock them out all at once\"). `fast` = base's predictor at 10x the rate; `decay` = base's predictor pulled toward its init with a 10-update half-life; `small` = a 32-unit one-hidden-layer predictor; `feat` = base's shapes over the detached value_pooled. Declarations: `agents.model.ridealong_heads.RND_VARIANT_DECLS`. The observation variants share base's target and normalisation (paired). DETACHED exactly like the four heads (pinned bit-for-bit by `ridealong_update_test`, base itself included). STRUCTURAL: the variants' parameters are the state_dict delta. family=CRITIC: never on the production ARCH surface. It REQUIRES ridealong_rnd: base is the shared target and the reference.",
              requires=("ridealong_rnd",),
              family=Family.CRITIC),
    ModelFlag("oracle_reveal", "off", Tier.CLI, Klass.RESUME_IMMUTABLE, 137,
              "DIAGNOSTIC observation mode (X32; X5 A/B §7.6): how much of the opponent's true team the "
              "OBSERVATION states from turn 1 ('off' = production; 'species' = the six species, "
              "team-preview semantics; 'full' = the whole set: moves, item, ability, spread)",
              note="Never production. The Rust encoder writes the facts into the opponent team block of "
                   "the observation itself (the shared trunk), not a side input to any head: the seen mons keep their "
                   "reveal-order slots and bytes, the unseen ones follow in dex-num order as the encoder's own row "
                   "for a never-seen mon (`encoder::oracle`); at 'full' the set's facts are written too (an unseen "
                   "mon's slot is the row of an OWN mon of that set; a seen mon gains only the facts play has not "
                   "revealed). 'off' is byte-identical to the build without it, and 'species' to the build that "
                   "shipped it. "
                   "SYMMETRIC (the trainee and its self-play opponents each see the other side's team), the eval "
                   "core uses the run's recorded mode, scripted bots are unaffected. RESUME_IMMUTABLE: the forward "
                   "is bit-identical (no module, no weight) but what the input MEANS differs, so a resume that "
                   "flips it is refused (`check_oracle_reveal`) and a flagless resume inherits it; it is excluded "
                   "from `check_compatible` (a frozen opponent of the same run is built at the run's mode) and "
                   "from the ARCH SURFACE by class. Offline tools that play a checkpoint without the reveal "
                   "(`main.anchors`, `main.play`) REFUSE a recorded non-'off' mode "
                   "(`agents.model.oracle_reveal.refuse_if_revealed`); `main.h2h` plays it at its recorded "
                   "level through a PER-SIDE reveal (`--oracle-reveal-mode one_sided|both_sided`, "
                   "`main.h2h.reveal`)."),
    ModelFlag("policy_readout", "tower", Tier.CLI, Klass.STRUCTURAL, 138,
              "where the pointer head's decision context comes from: 'tower' (the flat SB3 policy "
              "tower: projection 1177->512 + mlp_extractor.policy_net 512->512->512 tanh -> latent_pi) "
              "or 'trunk' (that tower retired; one learned query over every refined trunk token)",
              note="Architecture audit F2 (designs/endstate/design_arch_audit.md; owner 2026-10-06: \"no need to keep the tower\"). A SCREENABLE behaviour change, OFF in production. 'tower' builds nothing new and is byte-identical to the build without the flag. 'trunk' builds `agents.model.pools.PolicyStateQuery` (a private seed, `IsolatedLinear`s: no non-lever init byte moves), RETIRES the extractor's `pre_proj_norm` / `projection` and SB3's `mlp_extractor.policy_net` after SB3's orthogonal re-init (the `retire_superseded_intent_heads` precedent), and widens the pointer scorers to `TRUNK_POINTER_HIDDEN`; each legal action's logit is still its OWN refined token through the same equivariant scorer. The value path is unchanged (the dead SB3 value tower is F1, a separate unit). Both modes build at ONE commit, so there is no ARCH_SIGNATURE bump while the screen runs; the string compare in check_compatible is the gate."),
    ModelFlag("token_encoding", "legacy", Tier.CLI, Klass.STRUCTURAL, 139,
              "the per-Pokemon token: 'legacy' (`PokemonEncoder`: one MLP over the mon's row AND the board "
              "context) or 'static' (S = the static identity: species + set + actual stats, the moves pooled "
              "as a set; D = the mon's own battle state; added; no board fact in either)",
              note="designs/endstate/design_static_tokens.md (the architecture audit's lead hypothesis §4 L1 "
                   "with F4 + F14 and the owner's widened spec, 2026-10-06). Production stays 'legacy' until its "
                   "screen rules. 'legacy' builds `PokemonEncoder` exactly as before (byte-identical); 'static' "
                   "builds `agents.model.static_tokens.StaticTokenEncoder` at the SAME attribute "
                   "(`pokemon_encoder`, with the submodule names kept), and under --belief-tokens fixed_mass "
                   "its hypothesis tokens are the dex table encoded once and gathered (exact). Builds on both "
                   "belief modes; no requirement. No ARCH_SIGNATURE bump while both encodings build at one "
                   "commit: the string compare in check_compatible is the gate."),
    ModelFlag("move_resolution", "off", Tier.CLI, Klass.STRUCTURAL, 141,
              "F11's MOVE-RESOLUTION family ('off' = the seven per-action blocks, production; 'on' = per legal "
              "action, P(it resolves as stated) by the exact gen-3 rules, intent-weighted, plus the seven "
              "blocks' FACTS consolidated and their JUDGMENTS dropped)",
              note="Architecture audit F11 §9 (owner 2026-10-06: facts kept, judgments dropped; Destiny Bond's "
                   "feature = P(the opponent KOs us), no threshold). 'off' builds nothing and sets no op seam: "
                   "byte-identical. 'on' builds `agents.model.move_resolution.MoveResolutionCell` LAST from "
                   "zero-init IsolatedLinears (no global RNG draw, skipped by SB3's orthogonal re-init) and the "
                   "policy RETIRES the seven blocks it replaces after SB3's re-init and before the optimizer "
                   "(`retire_superseded_action_cells`), so every other parameter's initial bytes equal "
                   "production's: `--arch production --move-resolution on` is the ONE-lever screen arm. "
                   "REQUIRES opp_intent (α / β weight every opponent-dependent fact), damage_op, damage_outgoing "
                   "and both per-move matrices (the KO / hit / immunity physics). Builds on BOTH belief modes "
                   "(gen3_move_resolution_x5_v1): under belief_tokens='fixed_mass' the same rules read the flat "
                   "pointer, OTHER_move priced as one seat per priority level and OTHER_species as a 7th mon "
                   "(the renormalised tail), never excluded.",
              requires=("opp_intent", "damage_op", "damage_outgoing", "damage_matrices_incoming",
                        "damage_matrices_outgoing")),
    ModelFlag("speed_physics", "off", Tier.CLI, Klass.STRUCTURAL, 143,
              "Architecture audit F7b's SPEED PHYSICS ('off' = the op's P(we act first) as a fixed logistic over the "
              "speed gap, production; 'on' = their speed as the species' discrete Smogon spreads mixture plus the exact "
              "gen-3 order rules)",
              note="Owner 2026-10-06 (F7b IN). No parameters: 'off' runs nothing new (byte-identical: graph, "
                   "state_dict, outputs); 'on' builds its inputs in `damage_op_speed` and calls ONE rule "
                   "(`move_order.p_first_same_priority`) at every op site that prices who moves first. Quick Claw "
                   "is format-gated (banned in gen3ou). The string compare in check_compatible is the only gate.",
              requires=("damage_op",)),
    ModelFlag("obs_facts", "off", Tier.CLI, Klass.STRUCTURAL, 144,
              "whether the model READS the observation's OBS-FACTS block ('off' = production, builds nothing; "
              "'v1' = the zero-init ObsFactsInject: what the opponent has seen of our team, the opponent "
              "active's Choice-lock evidence, the actives' Encore / Taunt / Disable / Uproar / partial-trap "
              "turns and each side's screen turns, added to the tokens of the entities they describe)",
              note="gen3_obs_facts_v1 (designs/endstate/design_entity_coverage_audit.md §8 + Decision record; "
                   "owner scope 2026-10-06), the X5 version break's part 3 (config v144). The observation carries "
                   "the block whatever this flag says (obs 2761 -> 2845, the `total_dim` break; the 2761-dim prefix "
                   "is byte-identical), so the lever is screened as its OWN arm. 'v1' is zero-init "
                   "(identity-at-init), built LAST (no existing parameter position moves) out of IsolatedLinears "
                   "(no RNG draw); under --token-encoding static the SIDE-class facts (screens) go to the side board "
                   "tokens, never a per-mon token (`obs_facts_inject.FACTS_TOKEN_CLASS`). STRUCTURAL: the module is "
                   "the state_dict delta, so check_compatible compares the value. OFF on the production surface."),
    ModelFlag("op_reduction", "max", Tier.CLI, Klass.STRUCTURAL, 146,
              "Architecture audit F6b's PRINCIPLED OPERATOR REDUCTIONS ('max' = the op's per-channel hard maxima over "
              "the opponent's believed moves, production; 'principled' = the alpha-weighted EXPECTATION in every "
              "channel (one mixture per attacker, no channel from another move) + the noisy-OR P(some move KOs) "
              "worst case)",
              note="Owner 2026-10-08 ('compute the FACTS exactly; the trunk's attention does the judgment'). "
                   "`op_reduction.py`. alpha = the attacker's presence / its total presence (R1 belief_mean: the "
                   "flat pointer's alpha is T2 and cannot weight the T1 op). Width-neutral flat block; 'principled' "
                   "adds ONE zero-init bias-free IsolatedLinear `op_worst_proj` (2 -> 128, built LAST, no RNG draw) "
                   "carrying the noisy-OR KO row onto our mon tokens, so every other initial byte equals 'max''s. "
                   "'max' is byte-identical (graph, state_dict, outputs). Screened in the bundle with "
                   "--token-encoding static, --move-resolution on, --speed-physics on as NON-INFERIORITY.",
              requires=("damage_op",)),
    ModelFlag("mon_hazard_cost", "off", Tier.CLI, Klass.STRUCTURAL, 147,
              "a NARROW per-mon fact for --token-encoding static ('off' = nothing built, production; 'on' = every "
              "mon's token, both sides, gets its own side's Spikes layers and the fraction of max HP it would lose "
              "switching in: 1/8, 1/6, 1/4, 0 for Flying / Levitate, Levitate read from the species)",
              note="gen3_static_port_v1 (`static_facts.py`; the static diagnostic's H2 / H3: our side's Spikes the "
                   "one fact whose gap to legacy GREW). The fraction is the damage operator's ONE entry rule "
                   "(`DamageOperator.spikes_entry`, which the `x` edge cell also reads). Zero-init bias-free "
                   "IsolatedLinear (2 -> 128) built LAST, no RNG draw: every other initial byte equals 'off''s, and "
                   "ON adds exactly 0 at init. Added as token content after the op (not into D's MLP: a hidden "
                   "slot's static token is a dex-table gather).",
              requires=("token_encoding", "damage_op")),
    ModelFlag("move_actor_state", "off", Tier.CLI, Klass.STRUCTURAL, 147,
              "a NARROW per-move fact for --token-encoding static ('off' = nothing built, production; 'on' = our "
              "active's 4 E3 move seats get its current HP fraction and status one-hot)",
              note="gen3_static_port_v1 (`static_facts.py`; the static diagnostic's H2: our active's HP on its move "
                   "tokens R² 0.59 vs legacy's 0.80, since static's move tokens carry no HP). Zero-init bias-free "
                   "IsolatedLinear (8 -> 128) added to the E3 seats (equivalently zero-init input columns of "
                   "`move_seat_proj`), built LAST, no RNG draw. The opponent's E4 seats get nothing: neither "
                   "encoding ever gave them their active's state.",
              requires=("token_encoding",)),
    ModelFlag("trunk_layers", 2, Tier.CLI, Klass.STRUCTURAL, 150,
              "the number of trunk self-attention rounds (2 = production, the two post-LN layers; 3 or 4 append "
              "1 or 2 IDENTITY-INIT pre-LN residual rounds after them, so the deeper network starts exactly equal "
              "to the 2-round one)",
              note="gen3_static_recovery_v1 (`trunk_depth.py`; the static diagnostic's H3: under static, board facts "
                   "reach the mon tokens only at the LAST of two rounds). Each extra round's attention out-proj and FFN "
                   "output are zero (weight and bias), so it adds exactly 0 at init (bitwise); its other projections "
                   "are IsolatedLinears built from a PRIVATE seed (TRUNK_EXTRA_INIT_SEED), so no other initial byte "
                   "moves. A post-LN round with zero outputs would NOT be the identity (LN of an LN output moves values "
                   "by ~eps), hence the pre-LN form. 2 builds nothing (byte-identical)."),
    ModelFlag("switch_hazard_cost", "off", Tier.CLI, Klass.STRUCTURAL, 150,
              "the switch pointer cell's entry-hazard cost ('off' = nothing built, production; 'on' = each switch "
              "target's [our side's Spikes layers / 3, the HP fraction it loses switching in: 1/8, 1/6, 1/4, 0 for "
              "Flying / Levitate] appended LAST to its switch cell)",
              note="gen3_static_recovery_v1 (`static_facts.switch_hazard_features`; design_hand_computed_features.md "
                   "§4 ADD rank 1, entity audit B3: the switch logit had no entry-chip coordinate). The values are the "
                   "op's ONE entry rule (`DamageOperator.spikes_entry`), our half of --mon-hazard-cost's fact. The "
                   "pointer head reads the block through its own zero-init bias-free IsolatedLinear "
                   "(`switch_extra_proj`, built LAST: zero-init input columns of `switch_proj`, whose own init is "
                   "unchanged). Composes with either token encoding.",
              requires=("damage_op",)),
    ModelFlag("eot_residual", "off", Tier.CLI, Klass.STRUCTURAL, 150,
              "every mon's END-OF-TURN HP change if it is the one on the field at the end of this turn ('off' = "
              "nothing built, production; 'on' = Leftovers, weather, Rain Dish, the status tick, Leech Seed drain / "
              "heal, Wish, Ingrain, Curse, Nightmare and the HP-clamped net, both sides, added to the mon tokens)",
              note="gen3_static_recovery_v1 (`eot_residual.py`; design_hand_computed_features.md §4 ADD rank 2: the "
                   "stall residual race static loses most). Mechanics verified in deps/pokemon-showdown (gen 3 Wish "
                   "heals half the RECIPIENT's max HP; burn 1/8; Toxic (n+1)/16 reset on switch; Liquid Ooze inverts "
                   "the seeder's heal; a timed weather's last turn has no chip). Unrevealed items / abilities are the "
                   "species' Smogon priors. Zero-init bias-free IsolatedLinear (11 -> 128) built LAST, no RNG draw; "
                   "the rule itself holds only constant tables. Composes with either token encoding.",
              requires=("damage_op",)),
    # gen3_endstate_facts_v1 (config v152; design_hand_computed_features.md §1's FACT / JUDGMENT test, §4 ranks 3-5,
    # §5 rank 1, finding 7): the FACT-COMPLETION levers, each OFF in production (byte-identical).
    ModelFlag("move_resolution_facts", "off", Tier.CLI, Klass.STRUCTURAL, 152,
              "move resolution's RESTORED facts ('off' = the family as built on the 2026-10-06 list, production; "
              "'full' = + Focus Punch survives, the Substitute survives, Endure x P(KO), Endeavor x P(not KO'd), "
              "the hazard layers a failed Rapid Spin leaves, the layers our Ghost switch-in preserves)",
              note="Owner 2026-10-09: these were dropped as judgments but are FACTS by §1's test. Two zero-init "
                   "bias-free IsolatedLinears inside MoveResolutionCell (the zero-init input columns of its "
                   "move_proj / switch_proj, built LAST, no RNG draw). Requires --move-resolution on.",
              requires=("move_resolution",)),
    ModelFlag("status_facts", "off", Tier.CLI, Klass.STRUCTURAL, 152,
              "the pair-outcome STATUS coordinates ('off' = neutralization / tempo_cost, production; 'exact' = the "
              "expected burn damage lost and the expected outspeed lost to paralysis in the same two positions, and "
              "every mon's cure-availability flags as token content)",
              note="Owner 2026-10-09: neutralization (a weighted sum across units) and tempo_cost (our cure plan) are "
                   "JUDGMENTS by §1's test. The swap is width-neutral (pair_outcome.STATUS_FACT_COORDS); the cure flags "
                   "(status_facts.CureFlags: a live Heal Bell / Aromatherapy user on the side, Natural Cure, Rest, Lum / "
                   "Chesto Berry; ours exact, theirs the move / ability / item beliefs) ride a zero-init bias-free "
                   "IsolatedLinear (5 -> 128) built LAST; under --move-resolution on the two facts also reach its "
                   "cells through their own zero-init projections. P(full para) = 1/4 is the rule constant times the "
                   "delivered p_par column.",
              requires=("damage_op",)),
    ModelFlag("ko_ramp", "ramp", Tier.CLI, Klass.STRUCTURAL, 152,
              "how every P(KO) is priced ('ramp' = the op's mean-roll 15%-window ramp with no crit, production, an "
              "APPROXIMATE fact; 'exact' = over the 16 gen-3 rolls and the move's crit chance, each roll's KO "
              "resolved over the observed HP interval)",
              note="ko_exact.py: one shared function at every site (DamageOperator._rolls, the inlined Choice-Band "
                   "ko_cb, intent_threshold's Substitute break, move resolution's Pursuit KO). Continuous and "
                   "differentiable (no new discrete site for K9(b)). No parameters; the string compare in "
                   "check_compatible is the only gate.",
              requires=("damage_op",)),
    ModelFlag("drop_progress_clock", "off", Tier.CLI, Klass.STRUCTURAL, 152,
              "whether the model reads the observation's turns_since_progress ('off' = it does, production; 'on' = "
              "it reads 0 there)",
              note="§5 rank 1: a hand definition of 'progress' (3% thresholds, eight reset conditions) is a JUDGMENT. "
                   "Zeroed at the model input (ObsUnpack), never in the Rust observation layout. The threshold-free "
                   "fact alternative ('turns since either side lost HP') is not built."),
    ModelFlag("g_ledger", "coarse", Tier.CLI, Klass.STRUCTURAL, 152,
              "which end-of-turn rule the op's g cell reads ('coarse' = the op's own older ledger, production; 'eot' = "
              "the ONE rule --eot-residual reads)",
              note="Finding 7 (two end-of-turn rules): under 'eot' the g cell, c4's nets and the static op content read "
                   "eot_residual.EotResidualRule grouped into the ledger's four columns (eot_residual.g_cells). No "
                   "parameters.",
              requires=("damage_op",)),
    # gen3_probe_facts_v1 (config v153; design_hand_computed_features.md §4 rows 12-13, the representation probe
    # battery's two candidates `measurements/probe_battery_2026-10-09/`), each OFF in production (byte-identical).
    ModelFlag("effective_stats", "off", Tier.CLI, Klass.STRUCTURAL, 153,
              "each side's ACTIVE mon's STAGE-APPLIED stats as token content ('off' = nothing built, production; 'on' = "
              "Atk / Def / SpA / SpD / Spe after the gen-3 stage multipliers (Spe after paralysis) / 500, and the "
              "accuracy / evasion multipliers; ours exact, theirs the believed spread)",
              note="The probe battery: a boost's SIZE reads at R2 <= 0.10 while 'is boosted' reads at AUC 0.97 (the role "
                   "network squashes the stage columns). static_facts.effective_stat_features; one zero-init bias-free "
                   "IsolatedLinear (7 -> 128) built LAST, no RNG draw. Composes with either token encoding.",
              requires=("damage_op",)),
    ModelFlag("move_target_state", "off", Tier.CLI, Klass.STRUCTURAL, 153,
              "THEIR active's HP fraction + status one-hot onto our 4 E3 move seats ('off' = nothing built, production; "
              "'on' = through a zero-init projection)",
              note="--move-actor-state's sibling (the TARGET of the moves). The probe battery: their HP reads at R2 0.85 at "
                   "its own token but 0.26-0.34 at our decision tokens. static_facts.move_target_features; one zero-init "
                   "bias-free IsolatedLinear (8 -> 128) built LAST, no RNG draw. Composes with either token encoding."),
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
# width/count (`entity_topk_seats`), and the two mode-string spellings the CLI
# already uses. It is a MODULE-level rule rather than per-flag data because the CLI enforces it too:
# every mode flag in `train_rl_agent` spells its disabled state exactly one of these ways.
OFF_VALUES = (False, 0, "off", "none")


#: The OFF spellings of a MODE string. ``'tower'`` is `policy_readout`'s (audit F2, v138): today's flat
#: policy tower, nothing new built. ``'legacy'`` is `token_encoding`'s OFF state (v139): today's
#: `PokemonEncoder`. ``'max'`` is `op_reduction`'s (v146, audit F6b): the op's legacy per-channel hard maxima.
#: (``'blob'``, `belief_tokens`' OFF state, left with the flag at the X5 version break.) ``'ramp'`` is `ko_ramp`'s and
#: ``'coarse'`` `g_ledger`'s (v152, gen3_endstate_facts_v1): the op's legacy KO ramp / end-of-turn ledger.
OFF_STRINGS: Tuple[str, ...] = ("off", "none", "tower", "legacy", "max", "ramp", "coarse")


def is_enabled(value: Any) -> bool:
    """Is this flag value the ON state, for the purpose of ``requires``?

    Note this is NOT ``bool(value)``: a mode string's OFF state is the truthy ``'off'`` / ``'none'``,
    and reading it as enabled is a real bug this project has shipped before (the dead-kwarg
    sanitizer refused every OFF-mode checkpoint until it stopped testing truthiness). Floats are
    excluded on purpose — ``move_candidate_floor`` is a magnitude,
    not a switch, and nothing depends on it.
    """
    if isinstance(value, str):
        return value not in OFF_STRINGS
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


def arch_surface_flags() -> Tuple[ModelFlag, ...]:
    """The ARCH SURFACE: the rows a fresh launch is judged against `designs/production_config.json`.

    `structural` AND `family=ARCH` AND recorded (never `constructor_only`) — i.e. the toggles whose
    mismatch means the run builds a DIFFERENT NETWORK, minus the critic readouts an experiment is
    deliberately varying. Everything else drops out by its own declaration: `training_coef` and
    `runtime` are not architecture, `resume_immutable` leaves the forward identical, and the reward
    / PPO fields were never in this registry at all.

    THE POINT is that this is DERIVED, not typed. The 2026-09-06 incident launched a near-bare
    architecture from a 38-token argv while all three validators said "it launches" (~7 GPU-hours,
    24.4M steps, 31 keys off the mirror), and a hand-written list of "the flags that matter" would
    have gone stale the first time a toggle landed.

    The COUNT is reconciled rather than merely smaller — `arch_surface.surface_partition()` prints
    `39 arch + 7 critic + 3 non-structural = 49 registry rows` in every block, because a guard that
    compares fewer keys than a reader's own count leaves them unable to tell an excluded row from a
    forgotten one.
    """
    return tuple(f for f in REGISTRY
                 if f.klass is Klass.STRUCTURAL
                 and f.family is Family.ARCH
                 and f.tier is not Tier.CONSTRUCTOR_ONLY)


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
        "| toggle | CLI | tier | class | family | default | since | requires | meaning |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in REGISTRY:
        cli = f"`{f.cli_flag}`" if f.tier is Tier.CLI else "—"
        if f.derived and f.tier is Tier.CLI:
            cli += " *(coef)*"
        req = ", ".join(f"`{d}`" for d in f.requires) or "—"
        lines.append(
            f"| `{f.name}` | {cli} | `{f.tier.value}` | `{f.klass.value}` | `{f.family.value}` | "
            f"{_cell(f.default)} | v{f.since} | {req} | {f.meaning} |")

    surface = arch_surface_flags()
    lines += [
        "",
        f"**The ARCH SURFACE.** {len(surface)} of {len(REGISTRY)} toggles are `structural` AND "
        "`family` = `arch`. That set is what `--arch production` applies and what the ARCH-SURFACE "
        "guard compares against `designs/production_config.json` on every FRESH launch "
        "(`main.train.arch_surface`, read by `--dry-run`, `python -m main.checkargs` and the "
        "launcher alike). `family` = `critic` marks a readout an experiment deliberately VARIES "
        "(the win-prob critic implies one and refuses two others), so it is excluded from both; "
        "`training_coef` / `runtime` / `resume_immutable` are excluded by CLASS.",
    ]

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
