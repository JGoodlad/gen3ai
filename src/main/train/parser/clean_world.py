"""The `# --- gen3_clean_world_config_v1 ... ---` section: the CLEAN-WORLD reward
switches, plus the PPO clip / PopArt, opponent-belief, damage-op, compile and
entity-seat flags declared under that heading. Kept whole and in order, because the
heading is where `--help` shows them.

Lifted VERBATIM out of the old single-file `parser.py` (lines 329-649); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from agents.training.value_sidecar import DEFAULT_SIDECAR_FRACTION
from main.train.constants import CLIP_RANGE_DEFAULT
from main.train.parser.base import BoolFlag, optional_float


def _rnd_variants_arg(value: str) -> str:
    """argparse ``type`` for --ridealong-rnd-variants: the CANONICAL recorded string (so the order
    a human types never reaches the version gate), or an argparse error naming the declared set."""
    from agents.model.ridealong_heads import canonical_rnd_variants

    try:
        return canonical_rnd_variants(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def add_clean_world_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- gen3_winprob_critic_mode_v1 (ai_v12, designs/ai_v12/design_winprob_only_critic.md): WHICH readout is
    #     the value function. THERE IS NO `--critic` FLAG (deletion pass P11b, 2026-10-03): the win-prob critic
    #     is the only trainable critic (`'shaped'` trained on the Python env core, deleted in U3), so the mode
    #     is a CONSTANT of the namespace (`parser/objective.py`) and a typed `--critic` is refused with the
    #     reason. The win-prob head's own readout mode stays a flag, `--win-prob-mode`, further down. ---
    parser.add_argument("--value-sidecar", "--value_sidecar", dest="value_sidecar",
                        type=str, choices=("auto", "on", "off"), default="auto",
                        help="Log the critic against its OWN TRAINING TARGET. Once per rollout a "
                             "seeded fraction of buffer states is appended to "
                             "<run>/value_sidecar/rows.jsonl (value, derived win logit, back-filled "
                             "win target, opponent class, turn, episode extent, timeout flag). "
                             "'auto' (the DEFAULT) = ON (the win-prob critic is the only critic), OFF otherwise: "
                             "every probe this project owns reads EVAL battles, so on a win-prob "
                             "arm the training distribution's calibration was simply never "
                             "measured. Read it with `python -m main.ops.value_sidecar_read <run>`. "
                             "Pure observability -- no forward pass, no extra battle, no env call, "
                             "no gradient path; every column is sliced out of arrays the rollout "
                             "buffer already holds.")
    parser.add_argument("--value-sidecar-fraction", "--value_sidecar_fraction",
                        dest="value_sidecar_fraction", type=float,
                        default=DEFAULT_SIDECAR_FRACTION,
                        help=f"Share of each rollout's buffer states sampled into the sidecar "
                             f"(default {DEFAULT_SIDECAR_FRACTION:.6g} = 1/64, ~2,048 rows and "
                             f"~0.6MB per rollout at --n-steps 2048 --n-envs 64). The sample is "
                             f"UNIFORM OVER BUFFER CELLS, so a long episode contributes more rows "
                             f"-- the right weighting for a per-decision calibration question, "
                             f"which is why the reader clusters its bootstrap by EPISODE.")
    parser.add_argument("--value-sidecar-seed", "--value_sidecar_seed",
                        dest="value_sidecar_seed", type=int, default=0,
                        help="Seed for the sidecar sampler (default 0). Seeded per (seed, rollout "
                             "index), not as one stream, so a restart re-draws the same states an "
                             "uninterrupted run would.")
    # --- gen3_clean_world_config_v1: the TERMINAL's switches (`--victory-value`, `--terminal-indicator` and `--draw-penalty` are
    #     DELETED, P11b: constants of the namespace, `parser/objective.py`; so are the no-progress clock's two opt-in variants,
    #     `--progress-decision-tense` / `--progress-switch-freeze`, P11d). (`--hand-shaping`, `--pbrs-material`,
    #     `--pbrs-belief` and `--arm-no-progress-tax` were DELETED with the shaped reward path,
    #     gen3_shaped_reward_deletion_v1, 2026-09-26 — the reward is the terminal alone.) ---
    parser.add_argument("--clip-range", type=float, default=CLIP_RANGE_DEFAULT, help="PPO policy clip range (default 0.15)")
    parser.add_argument("--clip-range-vf", type=optional_float, default=0.5, help="Value function clip range; pass 'none' to disable clipping (thesis used 0.0184)")
    parser.add_argument("--opp-belief-cls-k", "--opp_belief_cls_k", dest="opp_belief_cls_k",
                        type=int, default=None,
                        help="Hidden-opponent belief: number of distinct learned query tokens (DETR "
                             "object-query style) that summarise the unrevealed opp party and feed both "
                             "heads. 0 = OFF (default, baseline arch). 1 = a single 'hidden-opponent CLS' "
                             "set-summary; >1 = N distinct per-slot queries that coordinate + specialise. "
                             "k>0 REQUIRES --attend-unrevealed-opponents (else the queries read a board "
                             "with the hidden mons masked out) and is a weight-shape change (version-"
                             "checked, cannot change on a resume). NOTE: without a dedicated aux objective "
                             "(B3 — species-ID / BYOL) the RL gradient only weakly shapes these queries.")
    parser.add_argument("--opp-belief-aux-coef", "--opp_belief_aux_coef",
                        dest="opp_belief_aux_coef", type=float, default=None,
                        help="In-place hidden-opponent BELIEF AUX (the B3 objective). 0.0 = OFF (default). "
                             ">0 turns ON opp_belief_slots (fills the un-revealed opp team slots with "
                             "distinct learned unknown-mon tokens refined in-lineup by the transformer + a "
                             "BeliefHead) and AUTO-FORCES --attend-unrevealed-opponents, and adds "
                             "coef*(species_CE + moves_BCE) over the believed slots to the PPO loss. The "
                             "slot module is weight-shape (version-checked); the coef itself is a "
                             "TRAINING-only hparam like --ent-coef (NOT resume-locked). The privileged "
                             "belief obs labels exist only when >0.")
    parser.add_argument("--opp-belief-moves-weight", "--opp_belief_moves_weight",
                        dest="opp_belief_moves_weight", type=float, default=1.0,
                        help="Relative weight of the moves multi-label BCE vs the species CE inside the "
                             "belief aux term (aux = species_CE + w·moves_BCE; both on a per-believed-slot "
                             "scale). Default 1.0 — species dominates; raise to up-weight move prediction. "
                             "TRAINING-only, like --opp-belief-aux-coef. Ignored when the coef is 0. 0.0 zeroes "
                             "the hidden-mon move prediction (the deleted --no-predict-unrevealed-mon-moves "
                             "alias also set --move-belief-mode revealed).")
    parser.add_argument("--move-belief-mode", "--move_belief_mode", dest="move_belief_mode",
                        choices=("off", "revealed", "unrevealed", "both"), default=None,
                        help="MOVE-belief REINJECTION: predict each opp mon's moveset and FLOW it back into "
                             "the slot token (soft move-embedding added before the CLS pools), so the policy/"
                             "value heads reason about the believed moves — not a dead-end readout. 'off' "
                             "(default) = no module (baseline byte-for-byte). 'revealed' = seen mons only "
                             "(predict their still-UNREVEALED moves — the defensible, surprise-OHKO lever). "
                             "'unrevealed' = hidden mons (Hungarian-matched, omniscient — REQUIRES "
                             "--opp-belief-aux-coef>0, else the hidden slots are empty placeholders). 'both' "
                             "= all slots (also requires it). STRUCTURAL (a new head; version-"
                             "checked, fresh-only — cannot change on a resume) and AUTO-FORCES "
                             "--attend-unrevealed-opponents. Supervised by privileged labels (the model's own "
                             "full team), training-only. The known-vs-unknown axis is the defensible-vs-"
                             "omniscient A/B.")
    parser.add_argument("--move-belief-coef", "--move_belief_coef", dest="move_belief_coef",
                        type=float, default=None,
                        help="Loss weight for the move-belief head (move_belief_coef * BCE over the scored "
                             "opp slots), like --opp-belief-aux-coef. 0.0 = no supervised pull (the module "
                             "still reinjects, but only RL gradient shapes it). TRAINING-only (not version-"
                             "locked). Ignored when --move-belief-mode off.")
    parser.add_argument("--damage-op", "--damage_op", dest="damage_op",
                        action=BoolFlag, default=None,
                        help="Differentiable GPU damage operator: compute the believed-move incoming "
                             "damage the opp ACTIVE would deal to each of our mons, fed by the MOVE "
                             "belief's predicted moves (sigmoid logits), and append it to BOTH heads. "
                             "Differentiable, so gradients sharpen the move belief toward real KO "
                             "threats; replaces the CPU obs block's fixed usage-prior with the LEARNED "
                             "belief. STRUCTURAL (widens both projections; version-checked, fresh-only). "
                             "REQUIRES --move-belief-mode revealed|both (it reads the opp active's "
                             "predicted logits, supervised only for a revealed mon). Off by default.")
    parser.add_argument("--unified-damage", "--unified_damage", dest="unified_damage",
                        choices=["off", "incoming", "both"], default="off",
                        help="ONE knob for the unified damage system (desugars into the component flags at "
                             "parse time): 'off' = baseline; 'incoming' = move belief (revealed) + prior "
                             "fusion + the GPU damage op (opp active → our 6 mons, incl. the safe-switch "
                             "bench rows); 'both' = also the OUTGOING per-move block (our active → opp "
                             "active, action-aligned — the equal-effectiveness tie-break). Overrides "
                             "--move-belief-mode / --damage-op / --move-prior-fusion / --damage-outgoing "
                             "when not 'off'. Pair with --move-candidate-floor (the learnset/rarity gate) "
                             "and --move-belief-mode both (to also guess unrevealed mons' moves).")
    parser.add_argument("--damage-outgoing", "--damage_outgoing", dest="damage_outgoing",
                        action=BoolFlag, default=None,
                        help="OUTGOING per-move damage direction (our active → opp active), in REQUEST-slot "
                             "order so the policy head can compare move A vs B directly (the "
                             "equal-effectiveness tie-break: Earthquake vs Brick Break into a Rock). "
                             "STRUCTURAL (widens both projections; version-checked, fresh-only). REQUIRES "
                             "--damage-op. Off by default. (Usually set via --unified-damage both.)")
    parser.add_argument("--move-candidate-floor", "--move_candidate_floor", dest="move_candidate_floor",
                        type=float, default=None,
                        help="The LEGAL-BUT-UNOBSERVED base probability of the fused move prior (default "
                             "0.02). This is NOT an on/off switch: move LEGALITY is UNCONDITIONAL — a move a "
                             "species CANNOT learn always gets ~0 prior mass, and a legal move always keeps "
                             "its TRUE Smogon usage (rare techs stay rare-but-liftable, never pruned, so "
                             "surprise-move anticipation survives). This flag only sets how high a LEGAL move "
                             "with no recorded usage starts, so in-battle evidence can still lift it. Must be "
                             ">= 0.001 (0.0 would make legal-unobserved indistinguishable from impossible). "
                             "Forward-behavior value (version-checked, fresh-only); only read under "
                             "--move-prior-fusion, which is what builds the prior.")
    parser.add_argument("--move-prior-fusion", "--move_prior_fusion", dest="move_prior_fusion",
                        action=BoolFlag, default=None,
                        help="Unified two-part move belief: fuse the Smogon move-frequency PRIOR into the "
                             "move-belief head as a log-odds residual (posterior = prior + learned delta) "
                             "and PIN revealed moves certain — so the belief the damage op + BCE loss read "
                             "is one coherent posterior (priors ⊕ prediction unified), anchored at the "
                             "prior at cold-start. Forward-behavior toggle (no weight-shape change; "
                             "version-checked, fresh-only). REQUIRES --move-belief-mode != off. Off by default.")
    # (`--belief-tokens {blob,fixed_mass}` was DELETED at the X5 version break, config v144: X5's hypothesis
    # tokens are the only belief representation, built whenever the opponent-belief family is on. A typed
    # one is refused at parse time WITH its reason — `designs/deleted_flags.md` via `deleted_flag_reasons`.)
    from agents.model.pools import POLICY_READOUT_MODES
    parser.add_argument("--policy-readout", "--policy_readout", dest="policy_readout",
                        choices=POLICY_READOUT_MODES, default=None,
                        help="Where the pointer head's decision context comes from "
                             "(gen3_policy_readout_trunk_v1, v138; architecture audit F2, "
                             "designs/endstate/design_arch_audit.md). 'tower' (default; production): "
                             "the flat SB3 policy tower, byte-identical. 'trunk': that tower (the "
                             "1177->512 projection + mlp_extractor.policy_net 512->512->512 tanh) is "
                             "RETIRED and the context is one learned attention query over every refined "
                             "trunk token; each action is still scored from its own token by the same "
                             "equivariant pointer scorer. STRUCTURAL, version-checked, fresh-only.")
    from agents.model.static_tokens import TOKEN_ENCODING_MODES
    parser.add_argument("--token-encoding", "--token_encoding", dest="token_encoding",
                        choices=TOKEN_ENCODING_MODES, default=None,
                        help="The per-Pokemon token (gen3_static_tokens_v1, v139; "
                             "designs/endstate/design_static_tokens.md). 'legacy' (default; production): "
                             "today's PokemonEncoder, byte-identical. 'static': each token is S (the static "
                             "identity: species, item, ability, the four moves pooled as a set, the actual "
                             "stats; for the opponent the Smogon stat prior until revealed) + D (the mon's own "
                             "battle state: HP, status, boosts, volatiles, PP, ...); the clock, weather, faint "
                             "counts, hazards and screens reach a token only by attention and the damage "
                             "operator. Builds with or without X5's hypothesis tokens. STRUCTURAL, version-checked, "
                             "fresh-only.")
    from agents.model.move_resolution_rules import MOVE_RESOLUTION_MODES
    parser.add_argument("--move-resolution", "--move_resolution", dest="move_resolution",
                        choices=MOVE_RESOLUTION_MODES, default=None,
                        help="F11's MOVE-RESOLUTION family (gen3_move_resolution_v1, v141; "
                             "designs/endstate/design_arch_audit.md F11 §9). 'off' (default; production): the "
                             "seven per-action blocks, byte-identical. 'on': per legal action, P(it resolves as "
                             "stated) by the exact gen-3 rules (lands / not blocked / not immune / not a no-op), "
                             "intent-weighted, plus the seven blocks' FACTS consolidated and their JUDGMENTS "
                             "(tempo_cost, wasted_ko, the hand thresholds) dropped; the seven blocks are retired. "
                             "STRUCTURAL, version-checked, fresh-only. Requires --opp-intent-coef > 0, "
                             "--damage-op, --damage-outgoing and both per-move matrices.")
    from agents.model.move_order import SPEED_PHYSICS_MODES
    parser.add_argument("--speed-physics", "--speed_physics", dest="speed_physics",
                        choices=SPEED_PHYSICS_MODES, default=None,
                        help="Architecture audit F7b's SPEED PHYSICS (gen3_speed_physics_v1, v143; "
                             "designs/endstate/design_arch_audit.md F7). 'off' (default; production): the damage "
                             "operator's P(we act first) is a fixed logistic over the speed gap, byte-identical. "
                             "'on': the integral over the speed BELIEF (the spread belief's believed speed and the "
                             "Smogon prior's spread) plus the exact gen-3 rules (speed tie = coin flip, paralysis, "
                             "stat stages, our exact stat arithmetic; Quick Claw format-gated, banned in gen3ou), "
                             "at every op site. No parameters. STRUCTURAL, version-checked, fresh-only. Requires "
                             "--damage-op.")
    from agents.model.obs_facts_inject import OBS_FACTS_MODES
    parser.add_argument("--obs-facts", "--obs_facts", dest="obs_facts",
                        choices=OBS_FACTS_MODES, default=None,
                        help="gen3_obs_facts_v1 (v144, the X5 version break's part 3): whether the model READS the "
                             "observation's OBS-FACTS block (always in the observation, its last 84 dims) — what "
                             "the opponent has seen of our team, the opponent active's Choice-lock evidence, the "
                             "actives' Encore / Taunt / Disable / Uproar / partial-trap turns, each side's screen "
                             "turns. 'off' (default; production): builds nothing. 'v1': the zero-init "
                             "ObsFactsInject adds each fact to its entity's token (under --token-encoding static "
                             "the screens go to the side board tokens). STRUCTURAL, version-checked, fresh-only.")
    from agents.model.op_reduction import OP_REDUCTION_MODES
    parser.add_argument("--op-reduction", "--op_reduction", dest="op_reduction",
                        choices=OP_REDUCTION_MODES, default=None,
                        help="Architecture audit F6b's PRINCIPLED OPERATOR REDUCTIONS (gen3_op_reduction_principled_v1, "
                             "v146; designs/endstate/design_arch_audit.md F6). 'max' (default; production): the damage "
                             "operator collapses the opponent's believed moves with a hard maximum taken separately per "
                             "channel (each channel may describe a different move), byte-identical. 'principled': every "
                             "such maximum becomes the alpha-weighted EXPECTATION (alpha = each attacker's move "
                             "presence normalised; one mixture for every channel), and the noisy-OR P(some move of "
                             "theirs KOs this mon) is added to our mon tokens through a zero-init projection. "
                             "STRUCTURAL, version-checked, fresh-only. Requires --damage-op.")
    from agents.model.static_facts import MON_HAZARD_COST_MODES, MOVE_ACTOR_STATE_MODES
    parser.add_argument("--mon-hazard-cost", "--mon_hazard_cost", dest="mon_hazard_cost",
                        choices=MON_HAZARD_COST_MODES, default=None,
                        help="A NARROW per-mon fact for --token-encoding static (gen3_static_port_v1, v147; "
                             "designs/endstate/design_static_tokens.md §12). 'off' (default; production): builds "
                             "nothing. 'on': every mon's token (both sides) gets its own side's Spikes layers and the "
                             "fraction of max HP it would lose switching in (the damage operator's one Spikes entry "
                             "rule: 1/8, 1/6, 1/4; 0 for Flying / Levitate), through a zero-init projection. "
                             "STRUCTURAL, version-checked, fresh-only. Requires --token-encoding static and --damage-op.")
    parser.add_argument("--move-actor-state", "--move_actor_state", dest="move_actor_state",
                        choices=MOVE_ACTOR_STATE_MODES, default=None,
                        help="A NARROW per-move fact for --token-encoding static (gen3_static_port_v1, v147). 'off' "
                             "(default; production): builds nothing. 'on': our active's 4 move seats (E3) get its "
                             "current HP fraction and status one-hot, through a zero-init projection. STRUCTURAL, "
                             "version-checked, fresh-only. Requires --token-encoding static.")
    parser.add_argument("--t0-species-prior", "--t0_species_prior",
                        dest="t0_species_prior", action=BoolFlag, default=None,
                        help="T0 SPECIES belief for the physics (gen3_t0_species_prior_v1, v72): price "
                             "unrevealed opponent mons from the model's own team-composition belief "
                             "(naive-Bayes over the revealed team, Species-Clause floored) instead of "
                             "the STATIC gen3ou usage prior. The belief already existed at T2 "
                             "(BeliefHead) where the T1 DamageOperator could not read it; this "
                             "re-homes it to T0. Parameter-free, no state_dict change. STRUCTURAL and "
                             "version-checked: it re-means every damage number against a hidden slot, "
                             "so it cannot be flipped on resume.")
    parser.add_argument("--species-prior-fusion", "--species_prior_fusion",
                        dest="species_prior_fusion", action=BoolFlag, default=None,
                        help="SPECIES belief prior fusion (gen3_species_prior_fusion_v1, v68): fuse a "
                             "TEAM-COMPOSITION prior into BeliefHead's species head as a log-prob "
                             "residual (posterior = prior + learned delta), the same two-part shape "
                             "--move-prior-fusion gives the move belief. The prior is naive Bayes over "
                             "pairwise co-occurrence in the data/teams/ pool — 'given the opponent mons "
                             "already revealed, what is likely in a hidden slot' — with Species Clause "
                             "as a hard constraint. The species head was the ONE belief leg with no "
                             "prior, so it cold-started ~uniform over ~400 nums. Measured on the pool, "
                             "5-fold held out: top-1 0.106 with nothing revealed, and with 3 revealed "
                             "0.189 conditional vs 0.156 marginal-only (top-3 0.449 vs 0.345) — vs "
                             "~0.0025 for uniform. The delta head is ZERO-INIT, so the cold-start "
                             "posterior EQUALS the prior. Adds NO parameters (the co-occurrence tables "
                             "are non-persistent buffers), but STRUCTURAL + version-checked all the "
                             "same: flipping it re-means every species logit. REQUIRES "
                             "--opp-belief-aux-coef>0. Off by default (byte-identical).")
    parser.add_argument("--compile-trainer", "--compile_trainer", dest="compile_trainer",
                        action=BoolFlag, default=None,
                        help="torch.compile the LEARNER's feature extractor — the GPU forward AND "
                             "backward that the PPO train step runs. Measured on v76 at the production "
                             "shape (batch 4096, PopArt on, real MaskablePPO path): "
                             "155.1 -> 88.5 ms per minibatch = 1.75x, i.e. ~+62%% end-to-end FPS at the "
                             "~89%% train share. CUDA ONLY and FAIL-LOUD by design — a silent fall back "
                             "to eager would be an invisible 1.75x regression, and the CPU backward "
                             "provably does not lower (Inductor's C++ backend refuses an atomic_add "
                             "scatter). **DEFAULT: AUTO — ON when the resolved device is cuda, OFF on "
                             "cpu and OFF under --debug**, so a working CPU invocation can never be "
                             "turned into a refusal by a default. An EXPLICIT --compile-trainer on cpu "
                             "still refuses, loudly (that contract is unchanged). "
                             "--no-compile-trainer opts out. RUNTIME PERF KNOB: not versioned; with the auto default "
                             "a flagless cuda resume gets it ON.")
    parser.add_argument("--consequence-topk", "--consequence_topk", dest="consequence_topk",
                        type=int, default=None,
                        help="v59: the CONSEQUENCE kernels' believed-candidate axis — C1b/C2/C3's "
                             "k_cand + D4's k_bench in one knob (how many candidates the belief-"
                             "weighted worst-case max covers per opp mon). Default 6 (4 real moves "
                             "+ 2 surprise slots; pre-v59 models trained at 4). FORWARD-BEHAVIOR "
                             "(no params) but version-checked — a frozen opponent's forward "
                             "changes with it.")
    parser.add_argument("--entity-topk-seats", "--entity_topk_seats", dest="entity_topk_seats",
                        type=int, default=None,
                        help="gen3_entity_move_seats_v1 (v54, Stage 1 of the entity generation): the E4 "
                             "THREAT-MOVE seat count — the opp active's top-K believed candidate moves "
                             "enter the trunk as attention SEATS ([move latent ⊕ belief w ⊕ acc ⊕ "
                             "is_phys] per seat; the op's refine_candidates definition, one source). "
                             "0 (default) = E3-only: our active's 4 request-ordered move seats, which "
                             "are UNCONDITIONAL in this generation (the pointer head reads the REFINED "
                             "seats). STRUCTURAL int (version-checked, fresh-only). >0 REQUIRES "
                             "--damage-op + --move-latent (--unified-moves).")
    parser.add_argument("--entity-tail-seats", "--entity_tail_seats", dest="entity_tail_seats",
                        action=BoolFlag, default=None,
                        help="gen3_entity_tail_seats_v1 (v57, E5): 6 per-opp-mon TAIL-THREAT seats — "
                             "the truncation insurance summarizing the beyond-top-K belief mass every "
                             "candidate consumer drops ([p_tail, worst_phys, worst_spec, revealed]). "
                             "STRUCTURAL (version-checked, fresh-only). REQUIRES --damage-op "
                             "AND --entity-topk-seats > 0.")
    parser.add_argument("--edge-bias-families", "--edge_bias_families", dest="edge_bias_families",
                        type=str, default=None,
                        help="gen3_edge_bias_trunk_v1 (v56, Stage 2 of the entity generation): deliver "
                             "computed physics as per-pair per-head additive ATTENTION BIASES. 'off' "
                             "(default) | 'd' (= d1,d3) | a comma list. d1 = our active's moves x the "
                             "opp's 6 mons (the outgoing-matrix kernel) at the (E3 seat, opp-mon seat) "
                             "pairs — requires --damage-op + --damage-outgoing; d3 = the opp's top-K "
                             "believed moves x our 6 mons (the pre-collapse incoming kernel, the SAME "
                             "candidates as the E4 seats) at the (E4 seat, our-mon seat) pairs — "
                             "requires --entity-topk-seats > 0. c1 = the CONSEQUENCE edge: post-"
                             "setup-move damage/outspeed DELTAS (SD/DD/CM/Agility hypothetical "
                             "kernel re-runs) at the (E3 setup seat, opp-mon) pairs — requires "
                             "--damage-op + --damage-outgoing. Zero-init maps: identity at init. "
                             "STRUCTURAL (version-checked, fresh-only). The op head-concat stays "
                             "(deprecation playbook: bias-ablation audit before deletion).")
    parser.add_argument("--damage-candidate-k", "--damage_candidate_k", dest="damage_candidate_k",
                        type=int, default=None,
                        help="Cap the DamageOperator's INCOMING candidate sweep at the K most-believed "
                             "opponent moves (0 = the full ~400-wide sweep, byte-identical). NO tail "
                             "bound - the truncated mass is DROPPED, so a rare-but-lethal candidate "
                             "below rank K is simply not priced (the on-policy probe measured top-16 "
                             "owning 94.2%% of channels, with misses BIMODAL). Payoff is learner-side: "
                             "measured +11.4%% forward / +63.5%% op at B=256, but only +0.3%% at B=1 "
                             "(the CPU opponent is dispatch-bound, not tensor-size bound). "
                             "Forward-behavior (version-checked, fresh-only). REQUIRES --damage-op.")
    # gen3_pointer_native_v1: --pointer-head is GONE — the pointer head is THE action head,
    # unconditionally (no flat action_net exists in this generation; see Gen3DualHeadMaskablePolicy).
    parser.add_argument("--win-prob-mode", "--win_prob_mode", dest="win_prob_mode",
                        choices=("none", "read_only", "shaping"), default=None,
                        help="Auxiliary WIN-PROBABILITY head: a calibrated P(win|state) readout off the "
                             "value pool, supervised by the Monte-Carlo episode outcome (win=1/loss=0) — "
                             "the shaped critic's V is expected RETURN, not win odds, so this gives an "
                             "interpretable P(win) (and ΔP(win) per move). 'none' (default) = no module "
                             "(baseline byte-for-byte). 'read_only' = the head trains on a STOP-GRAD value "
                             "pool — a pure, risk-free diagnostic that CANNOT perturb the policy. 'shaping' "
                             "= its gradient also shapes the shared trunk (the win objective improves the "
                             "representation; A/B it vs read_only). STRUCTURAL + resume-IMMUTABLE "
                             "(version-checked: any change FATALs on resume). The head is a SIDE readout "
                             "(never in pi/vf — leak-safe).")
    # --- gen3_ridealong_heads_v1 (2026-09-30, owner): the DETACHED RIDE-ALONG baseline heads.
    #     Each OFF by default; the baseline argv turns all four on. None of them can change what the
    #     run learns (detached inputs, their own optimizer, a private init RNG) — they OBSERVE. ---
    _ra_common = (" DETACHED: every input is stop-grad and the heads train on their own optimizer, "
                  "so the run learns bit-for-bit what it would without them. STRUCTURAL: fixed for "
                  "a run's lifetime; a flagless resume inherits it. TB: ridealong/*.")
    parser.add_argument("--ridealong-ensemble", "--ridealong_ensemble", dest="ridealong_ensemble",
                        type=int, default=None,
                        help="K win-probability heads on the detached value_pooled (V's own target), "
                             "each with a bootstrap mask and a randomized prior: their disagreement "
                             "is V's EPISTEMIC uncertainty. 0 (default) = off; the baseline uses 5. "
                             "Requires --win-prob-mode != none." + _ra_common)
    parser.add_argument("--ridealong-rnd", "--ridealong_rnd", dest="ridealong_rnd",
                        action=BoolFlag, default=None,
                        help="An RND novelty head (Burda et al. 2018): a frozen random network of "
                             "the running-normalised RAW observation and a trained predictor; the "
                             "error is how rarely the state was seen." + _ra_common)
    parser.add_argument("--ridealong-adv", "--ridealong_adv", dest="ridealong_adv",
                        type=int, default=None,
                        help="K per-action A heads over the pointer head's own action tokens, "
                             "centred under pi and regressed on the GAE advantage of the action "
                             "taken; their spread is per-ACTION uncertainty (starved moves). 0 "
                             "(default) = off; the baseline uses 5." + _ra_common)
    parser.add_argument("--ridealong-opp", "--ridealong_opp", dest="ridealong_opp",
                        type=int, default=None,
                        help="K opponent-effect B heads over alpha's support (their believed move "
                             "seats by move id + SWITCH), centred under alpha and regressed on the "
                             "same advantage where their actual action is named. Q = V + A + B is "
                             "logged as a derived readout. 0 (default) = off; the baseline uses 5. "
                             "Requires --opp-intent-coef > 0." + _ra_common)
    parser.add_argument("--ridealong-rnd-variants", "--ridealong_rnd_variants",
                        dest="ridealong_rnd_variants", type=_rnd_variants_arg, default=None,
                        help="gen3_ridealong_rnd_variants_v1 (v127): the RND VARIANT ENSEMBLE beside "
                             "--ridealong-rnd (base, the reference). A comma list of fast (10x the "
                             "predictor rate: forgetting by fast tracking), decay (pulled toward its "
                             "init, half-life 10 PPO updates: forgetting by shrinkage), small (a "
                             "32-unit predictor that cannot fingerprint battles), feat (over the "
                             "detached trunk features: live representation drift); 'all' = every "
                             "one; 'off' (default) = none. Each is its own predictor, optimizer and "
                             "statistics; the observation variants share base's target (paired). "
                             "The X26 baseline uses 'all'. Requires --ridealong-rnd." + _ra_common)
    # --- gen3_fork_v1 (2026-09-14, the FORK ARM): CONTESTED-STATE EXPLORING STARTS. Fork a
    #     contested decision, play the branches to a terminal under common random numbers, and put
    #     their transitions in the SAME PPO buffer. Registered by
    #     designs/research_state/measurements/paired_refit_discrimination_2026-09-14/ --- 
    parser.add_argument("--fork-fraction", "--fork_fraction", dest="fork_fraction",
                        type=float, default=None,
                        help="Fraction of the rollout buffer's decisions that are FORKED, in "
                             "[0, 1]. 0.0 (the DEFAULT) = OFF and BIT-identical -- no module "
                             "imported, no obs key declared, no callback attached, no row "
                             "injected. WHY: the promoted win-prob critic ranks two successors ONE "
                             "MOVE APART at CHANCE (pairwise accuracy 0.5169 [0.4800, 0.5524]), "
                             "while a frozen-trunk refit on COUNTERFACTUAL SUCCESSOR states "
                             "reaches 0.6032 -- so the trunk already holds the ordering and the "
                             "DATA is what the on-policy stream never supplies. A rollout visits "
                             "exactly ONE successor per decision; this manufactures the siblings. "
                             "At a contested decision the battle is cloned, each branch is played "
                             "to a terminal by the CURRENT policy, and the branch's transitions "
                             "enter the buffer with the FORK STEP masked out of the policy term "
                             "and the shared prefix counted ONCE. The value target is the ordinary "
                             "GAE/lambda-return of that branch -- plain BCE, NO ranking term "
                             "(CLOSED as a lever: -0.0107 [-0.0249, +0.0028], NOT DETECTED). "
                             "🚨 COST IS LINEAR AND LARGE: fraction x branches x remaining "
                             "decisions, so 0.02 at 3 branches ASKS for ~2.1x the run's own "
                             "simulation. 🚨 BUT THE ROW BUDGET BINDS FIRST: the injection is "
                             "capped at one buffer's worth of rows (~790 forks at the production "
                             "shape), so above ~0.008 this flag is INERT and the delivered count "
                             "is the budget's -- read fork/requested against fork/forks. Watch "
                             "fork/sim_steps_share, fork/branch_share, fork/rate and "
                             "fork/pairwise_acc. "
                             "TRAINING-only, resume-inherited.")
    parser.add_argument("--fork-branches", "--fork_branches", dest="fork_branches",
                        type=int, choices=(2, 3), default=None,
                        help="How many branches a fork plays (default 3). 3 = the policy's top-2 "
                             "candidates + ONE uniformly random legal action. The random branch is "
                             "where the new information is: on 5,076 measured forks top-1 and "
                             "top-2 were outcome-INTERCHANGEABLE (0.7082 vs 0.7078, a gap of "
                             "0.0004), throwing the decision away cost 2.9 pp, and in 4.5%% "
                             "[3.99, 5.16] of forks the random alternative beat BOTH policy "
                             "candidates. 2 = the top-2 alone, the CONTROL that isolates that "
                             "4.5%%; expected to buy little, and fork/random_wins is the meter "
                             "that says so. INERT at --fork-fraction 0.")
    parser.add_argument("--fork-contested-gap", "--fork_contested_gap",
                        dest="fork_contested_gap", type=float, default=None,
                        help="The QUANTILE, over this rollout's own candidate pool, of the "
                             "policy's top-2 masked-logit gap below which a decision counts as "
                             "CONTESTED (default 0.40). Taken verbatim from the paired-refit "
                             "dataset's --gap-quantile so the arm forks the population its 0.5169 "
                             "baseline was measured on. 🚨 A QUANTILE and not an absolute gap: the "
                             "gap's SCALE moves as a run's logits sharpen, so a fixed threshold "
                             "would fork 40%% of decisions early and ~0%% late -- the treatment "
                             "would anneal itself off in silence. INERT at --fork-fraction 0.")
    parser.add_argument("--fork-contested-absv", "--fork_contested_absv",
                        dest="fork_contested_absv", type=float, default=None,
                        help="ALSO admit a decision whose |V - 0.5| is below this (default 0.0 = "
                             "OFF). 🚨 OFF ON PURPOSE. The paired-refit selector never reads V, in "
                             "its own words because 'selecting on V would make the held-out read "
                             "partly a measurement of the selector' -- and this arm's registered "
                             "endpoint is held-out pairwise accuracy AGAINST that baseline. "
                             "Turning this on forfeits the comparison. INERT at --fork-fraction 0.")
    parser.add_argument("--fork-max-per-battle", "--fork_max_per_battle",
                        dest="fork_max_per_battle", type=int, default=None,
                        help="At most this many forks per EPISODE SLICE (default 1). Two forks of "
                             "one game share a prefix, an opponent and a team draw, so they are "
                             "far more correlated than two forks of different games -- the "
                             "bits-per-state argument applied to the sample itself. INERT at "
                             "--fork-fraction 0.")
    parser.add_argument("--fork-crn", "--fork_crn", dest="fork_crn",
                        choices=("dice", "dice_and_draws"), default=None,
                        help="WHAT the branches share after the fork. `dice_and_draws` (the "
                             "DEFAULT): one sim seed for the whole line AND both players' policy "
                             "sampling streams seeded identically per branch, so the k-th decision "
                             "of every branch consumes the SAME uniform and the branches differ in "
                             "exactly one thing -- the action at the fork. `dice`: the sim seed "
                             "only, which is the `cf_q_labels` regime and the CONTROL. That "
                             "distinction is a concrete, testable account of the cf-labels null: "
                             "that factory paired the dice and left both sides sampling at "
                             "temperature 1.0, so it may have been teaching the head noise. "
                             "INERT at --fork-fraction 0.")
