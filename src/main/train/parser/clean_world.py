"""The `# --- gen3_clean_world_config_v1 ... ---` section: the CLEAN-WORLD reward
switches, plus the PPO clip / PopArt, opponent-belief, damage-op, compile and
entity-seat flags declared under that heading. Kept whole and in order, because the
heading is where `--help` shows them.

Lifted VERBATIM out of the old single-file `parser.py` (lines 329-649); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from agents.model.critic_mode import CRITIC_MODES
from agents.training.value_sidecar import DEFAULT_SIDECAR_FRACTION
from main.train.constants import CLIP_RANGE_DEFAULT
from main.train.parser.base import BoolFlag, optional_float


def add_clean_world_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- gen3_winprob_critic_mode_v1 (ai_v12, designs/ai_v12/design_winprob_only_critic.md): WHICH
    #     readout is the value function. Declared FIRST in this family because it governs the
    #     reward composition, the PopArt switch and the win-prob head below it. ---
    parser.add_argument("--critic", dest="critic", choices=CRITIC_MODES, default=None,
                        help="WHICH readout is the critic. 'shaped' (the DEFAULT, and every "
                             "generation through gen-16) = the scalar value_net (or the "
                             "distributional E[Z] under --value-from-dist), de-normalized through "
                             "PopArt into raw SHAPED-RETURN units, with the win-prob head an "
                             "auxiliary BCE at --win-prob-coef. 'winprob' = THE WIN-PROB HEAD IS "
                             "THE CRITIC: V(s) = sigmoid(logit) in [0,1], the value loss IS that "
                             "head's BCE against the terminal outcome (weighted by --vf-coef, NOT "
                             "--win-prob-coef -- one critic, one coefficient), the reward stream is "
                             "the TERMINAL WIN INDICATOR alone (--no-hand-shaping implied, "
                             "--victory-value 1.0 required, so V(s) == E[return] exactly), PopArt "
                             "is OFF (a bounded stationary Bernoulli payoff has no scale to track) "
                             "and --gamma defaults to 1.0 (a win on turn 200 is worth a win on turn "
                             "20), which makes V(s) EXACTLY P(win|s) with no approximation term. "
                             "It requires --win-prob-mode read_only|shaping (unset defaults to "
                             "'shaping' under this critic) and REFUSES the flags whose job it "
                             "subsumes -- see the refusals `python -m main.checkargs` prints. "
                             "STRUCTURAL + resume-IMMUTABLE (a different set of heads carries the "
                             "value, so a mid-run flip is a different training problem).")
    # --- gen3_value_sidecar_v1 (2026-09-08): the TRAINING-SIDE value read. Declared here because
    #     it is only meaningful relative to --critic above: `v` is a probability under `winprob`
    #     and a shaped return under `shaped`, and the sidecar's header records which. ---
    parser.add_argument("--value-sidecar", "--value_sidecar", dest="value_sidecar",
                        type=str, choices=("auto", "on", "off"), default="auto",
                        help="Log the critic against its OWN TRAINING TARGET. Once per rollout a "
                             "seeded fraction of buffer states is appended to "
                             "<run>/value_sidecar/rows.jsonl (value, derived win logit, back-filled "
                             "win target, opponent class, turn, episode extent, timeout flag). "
                             "'auto' (the DEFAULT) = ON under --critic winprob, OFF otherwise: "
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
    parser.add_argument("--arm-no-progress-tax", "--arm_no_progress_tax",
                        dest="no_progress_tax_armed", action=BoolFlag, default=False,
                        help="Keep the anti-stall `no_progress_tax` BIAS term ARMED even under "
                             "--no-hand-shaping (default OFF = today's behaviour exactly: the "
                             "master switch zeroes the whole BIAS class, tilt included). It exists "
                             "because the clean-world composition and the win-prob critic each drop "
                             "an anti-stall defence -- --no-hand-shaping drops the tilt, and a "
                             "critic bounded in [0,1] CANNOT express 'a timeout is worse than a "
                             "loss' the way --draw-penalty -35 does -- so this is the CONTINGENCY "
                             "for a run whose stall rate rises, re-armable without reviving the "
                             "other 24 BIAS terms. Resume-immutable, value-checked.")
    # --- gen3_clean_world_config_v1 (ai_v12 build wave A): the CLEAN-WORLD reward switches. Every
    #     default below is today's behaviour, so a flagless launch is byte-identical. ---
    parser.add_argument("--hand-shaping", "--hand_shaping", dest="hand_shaping",
                        action=BoolFlag, default=True, help="MASTER switch for every HAND-DESIGNED "
                        "dense reward term. Default ON = today's reward. --no-hand-shaping is the "
                        "CLEAN-WORLD composition: all EIGHT PBRS potentials off (material and belief "
                        "included) AND the WHOLE BIAS class zeroed, no_progress_tax included, leaving "
                        "1 TERMINAL + 0 PBRS + 0 BIAS. It exists because --no-all-shaping-pbrs cannot "
                        "get you there: that flag is ALSO the BIAS class's master gate, so disabling "
                        "it silences 5 potentials while REVIVING 25 BIAS terms. NOTE, and state it in "
                        "any write-up: every PBRS term is policy-INVARIANT, so removing them cannot "
                        "change the optimal policy -- it changes learning dynamics and conceptual "
                        "complexity. Resume-immutable, value-checked.")
    parser.add_argument("--pbrs-material", "--pbrs_material", dest="pbrs_material",
                        action=BoolFlag, default=True, help="Fold the material PBRS potential "
                        "Phi_mat (default ON = today's reward). --no-pbrs-material drops it; the "
                        "field stays 0.0 and its carry-over stays unset, the same shape every other "
                        "PBRS fold's off-state takes. INDEPENDENT of --all-shaping-pbrs on purpose "
                        "(that flag is anti-correlated -- see --hand-shaping). Resume-immutable, "
                        "value-checked.")
    parser.add_argument("--pbrs-belief", "--pbrs_belief", dest="pbrs_belief",
                        action=BoolFlag, default=True, help="Fold the incoming-KO belief PBRS "
                        "potential Phi_belief (default ON = today's reward). --no-pbrs-belief drops "
                        "the EMITTED term only: the decision-time KO-risk / safe-pivot snapshots it "
                        "also computes still run, because the belief-scaled BIAS terms read them and "
                        "a gate here must skip a compute, never a cross-turn mutation. INDEPENDENT "
                        "of --all-shaping-pbrs on purpose. Resume-immutable, value-checked.")
    parser.add_argument("--victory-value", "--victory_value", dest="victory_value", type=float,
                        default=30.0, help="TERMINAL magnitude: a win scores +V, a decisive loss and "
                        "a rare pre-cap tie score -V, a 250-turn TIMEOUT scores --draw-penalty. "
                        "Default 30.0 = the historical reward_weights.VICTORY_VALUE constant. Pass "
                        "1.0 for the clean-world +-1 terminal. THE OUTCOME ORDERING IS LOAD-BEARING: "
                        "--draw-penalty must stay <= -V, or a 250-turn stall becomes the best "
                        "non-winning outcome and a losing agent's optimal play is to run out the "
                        "clock. Pair --victory-value 1.0 with --draw-penalty -1.0 (draw = loss) and "
                        "make stall rate + mean game length a PRIMARY endpoint. Note MAT_HP_WEIGHT / "
                        "MAT_ALIVE_WEIGHT are calibrated against the 30 scale, so a +-1 terminal "
                        "wants Phi_mat off (--no-hand-shaping does that). Resume-immutable, "
                        "value-checked.")
    parser.add_argument("--terminal-indicator", "--terminal_indicator",
                        dest="terminal_indicator", action=BoolFlag, default=False,
                        help="TERMINAL SHAPE: OFF (default, and every generation to date) pays "
                             "+V on a win, -V on a decisive loss and a pre-cap tie, and "
                             "--draw-penalty on a 250-turn TIMEOUT. ON pays +V on a WIN and 0.0 on "
                             "EVERYTHING else, so the undiscounted return is V*1{win} and at "
                             "--victory-value 1.0 the return IS the win indicator. That is what "
                             "makes V(s) == P(win|s) exactly under --critic winprob, which IMPLIES "
                             "this flag; you rarely set it by hand. ⚠️ It makes --draw-penalty and "
                             "the draw<=loss ORDERING inapplicable, not merely inert -- a [0,1] "
                             "critic cannot represent 'a timeout is worse than a loss' -- so the "
                             "anti-stall pressure must come from the obs deadline clock and, if "
                             "the stall rate rises, --arm-no-progress-tax. Resume-immutable, "
                             "value-checked.")
    parser.add_argument("--progress-decision-tense", "--progress_decision_tense",
                        dest="progress_decision_tense", action=BoolFlag, default=False,
                        help="No-progress clock: read BOTH window gates (the forced-switch sit-out "
                        "and the trapped-vs-wall charge suppression) off the decision BEING CHARGED "
                        "instead of the one after it. Today's off-by-one exempts 13.2%% of decisions "
                        "that had full agency and charges the zero-agency post-faint replacement "
                        "63.9%% of the time (36.3%% of all charges). Default OFF = the shipped "
                        "behaviour. ON changes the turns_since_progress OBS scalar too, so it is "
                        "retrain-class. Resume-immutable, value-checked.")
    parser.add_argument("--progress-switch-freeze", "--progress_switch_freeze",
                        dest="progress_switch_freeze", action=BoolFlag, default=False,
                        help="No-progress clock: a VOLUNTARY switch that fails the progress "
                        "predicate FREEZES the window (no increment, no charge) instead of being "
                        "taxed. The predicate is offense-only, so no switch can satisfy it by its "
                        "own doing — the tax prices the action KIND (-0.101 per voluntary switch vs "
                        "-0.010 per move) and within the branch its discrimination is INVERTED. "
                        "42.7%% of all charges. Anti-stall survives via the move turns between "
                        "pivots + --draw-penalty + the 250-turn forfeit; a pure A-B switch-loop "
                        "becomes free, so watch the stall-rate canary. Default OFF = the shipped "
                        "behaviour. Retrain-class. Resume-immutable, value-checked.")
    parser.add_argument("--clip-range", type=float, default=CLIP_RANGE_DEFAULT, help="PPO policy clip range (default 0.15)")
    parser.add_argument("--clip-range-vf", type=optional_float, default=0.5, help="Value function clip range; pass 'none' to disable clipping (thesis used 0.0184)")
    parser.add_argument("--use-popart", "--use_popart", dest="use_popart", action=BoolFlag, default=None,
                        help="Enable PopArt value-target normalization (adaptive (mu,sigma) on the "
                             "value head; keeps the value gradient O(1) so it stops swamping the "
                             "shared trunk). Requires an explicit --clip-range-vf none (value "
                             "clipping is unnecessary with normalization). Version-checked: cannot "
                             "be toggled on a resumed model.")
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
                             "TRAINING-only, like --opp-belief-aux-coef. Ignored when the coef is 0. The "
                             "explicit --[no-]predict-unrevealed-mon-moves knob below is the clear on/off.")
    parser.add_argument("--predict-unrevealed-mon-moves", "--predict_unrevealed_mon_moves",
                        dest="predict_unrevealed_mon_moves", action=BoolFlag, default=None,
                        help="EXPLICIT clarity knob: should the model predict the MOVES of opponent mons it "
                             "has NOT even seen (the hidden bench)? Default (unset) = yes (current behavior). "
                             "--no-predict-unrevealed-mon-moves turns it OFF — zeros BOTH hidden-mon "
                             "move-prediction paths: the BeliefHead's hidden-slot moves-BCE "
                             "(--opp-belief-moves-weight → 0) AND any MoveBelief unrevealed leg "
                             "(--move-belief-mode 'unrevealed'/'both' → 'revealed'). The REVEALED-mon move "
                             "belief (a SEEN mon's unseen slots) and the SPECIES belief on hidden mons are "
                             "UNTOUCHED. A desugar into existing fields — no version field.")
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
    parser.add_argument("--compile-opponents", "--compile_opponents", dest="compile_opponents",
                        action=BoolFlag, default=True,
                        help="torch.compile each frozen SELF-PLAY OPPONENT's feature extractor in the "
                             "env workers (CPU, B=1 — the measured 68%% of rollout worker time). "
                             "Measured 6.53x on the real forward; value-preserving to ~5e-7 with 0/16 "
                             "argmax flips. This is the CPU/ROLLOUT half; --compile-trainer is the "
                             "GPU/LEARNER half and they are independent. **DEFAULT ON** — pass "
                             "--no-compile-opponents to fall back to eager. The default failure mode "
                             "is still warn-and-fall-back (--compile-opponents-strict promotes it to "
                             "a hard error). RUNTIME PERF KNOB: not versioned, not in "
                             "check_compatible; with the default ON a flagless resume gets it ON. "
                             "Hides CUDA in the (CPU) workers first, because compiling in a "
                             "CUDA-visible process costs ~252 MiB of card per worker.")
    parser.add_argument("--compile-opponents-preload", "--compile_opponents_preload",
                        dest="compile_opponents_preload", action=BoolFlag, default=None,
                        help="gen3_forkserver_preload_v1: compile the extractor ONCE in the "
                             "multiprocessing FORKSERVER so every env worker inherits the traced "
                             "graph by fork (~0.12 s/worker instead of ~30 s against a warm disk "
                             "cache). Possible since the lazy poke_env __init__ made the extractor "
                             "import single-threaded (compile_prewarm.extractor_import_is_fork_safe). "
                             "FAIL-LOUD: a preload that cannot prove the forkserver is "
                             "single-threaded after the compile RAISES, killing env construction "
                             "with a traceback instead of the silent 2-of-48-workers wedge the "
                             "2026-08 attempt caused. **DEFAULT: FOLLOWS --compile-opponents** (so "
                             "ON by default, OFF whenever the opponent compile is off); "
                             "--no-compile-opponents-preload keeps the opponent compile but reverts "
                             "to the per-worker in-trainer cache prewarm. Runtime perf knob (never "
                             "versioned, not inherited on resume).")
    parser.add_argument("--compile-opponents-strict", "--compile_opponents_strict",
                        dest="compile_opponents_strict", action="store_true", default=False,
                        help="Turn a failed or ineffective OPPONENT compile into a hard error instead "
                             "of a warning. Without --compile-opponents this does nothing. Falling "
                             "back to eager is a ~6.5x regression on the opponent forward that is "
                             "otherwise invisible (the run just produces fewer steps/hour forever), so "
                             "use this when you would rather fail at startup than discover it in the "
                             "FPS graph a day later. (--compile-trainer needs no such flag: it is "
                             "ALWAYS fail-loud, see its help.)")
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
                             "--no-compile-trainer opts out and is also how you KEEP the "
                             "ObservationDebugger, which the compile drops (dynamo cannot trace its "
                             "numpy asserts). RUNTIME PERF KNOB: not versioned; with the auto default "
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
    # --- gen3_winprob_strata_weight_v1 (2026-09-09, the critic ladder's arm 7): OPPONENT-STRATIFIED
    #     weighting of the win-prob BCE. Declared beside the head's own coefficient because it is a
    #     second knob on the SAME loss — one scales it, this one re-prices its MIX. ---
    parser.add_argument("--win-prob-strata-weight", "--win_prob_strata_weight",
                        dest="win_prob_strata_weight", type=float, default=None,
                        help="OPPONENT-STRATIFIED weighting of the win-prob BCE, in [0, 1]. 0.0 "
                             "(the DEFAULT) = OFF and the loss is bit-identical. 1.0 = every "
                             "opponent CLASS (bot / pool / stable / exploiter, the `opp_class` "
                             "label key) contributes to the objective in EQUAL proportion instead "
                             "of in episode proportion; intermediate values interpolate the "
                             "exponent (w = freq ** -s), capped at 8x and renormalised so the mean "
                             "weight over the rollout buffer is 1 -- so it moves the MIX without "
                             "moving the loss scale. WHY: only 10-14%% of the terminal 0/1 label's "
                             "variance lies BETWEEN (cycle, opponent) cells, so a head minimising "
                             "BCE buys its resolution from the board and its own team and never "
                             "conditions on the opponent; the refit showed the same head on a "
                             "target with a higher between-cell share DOES condition "
                             "(designs/research_state/measurements/winprob_head_refit_2026-09-09/). "
                             "This raises that share directly, with no new labels and no rollout "
                             "cost. REQUIRES --critic winprob (refused otherwise -- the BCE is an "
                             "auxiliary readout under `shaped`, not the objective the measurement "
                             "indicts). Watch win_prob/strata_share_*, strata_w_entropy and "
                             "loss vs loss_unweighted. TRAINING-only, resume-inherited.")
    # --- gen3_winprob_lambda_v1 (2026-09-09, the critic ladder's arm 8): λ-RETURN targets for the
    #     win-prob BCE. The third knob on the SAME loss — one scales it, `--win-prob-strata-weight`
    #     re-prices its MIX, and this one changes what it REGRESSES TOWARD. ---
    parser.add_argument("--win-prob-lambda", "--win_prob_lambda",
                        dest="win_prob_lambda", type=float, default=None,
                        help="λ-RETURN targets for the win-prob BCE, in [0, 1]. 1.0 (the DEFAULT) "
                             "= OFF and BIT-identical: every state of an episode is trained "
                             "against the episode's terminal 0/1 outcome, as today. Below 1.0 the "
                             "target becomes a backward blend of the network's OWN recorded "
                             "later estimates -- G[t] = (1-lambda)*V(s[t+1]) + lambda*G[t+1], "
                             "anchored at G = y on the state that ENDS the episode (gamma = 1 and "
                             "the clean-world stream is terminal-only, so an n-step return IS "
                             "V(s[t+n])); a state d steps from its terminal keeps weight "
                             "lambda**d on the outcome. WHY: one bit copied to ~30 states is a "
                             "very noisy target, and the head refit showed the win-prob critic's "
                             "conditional miscalibration is a TARGET defect -- only 10-14%% of "
                             "that label's variance lies BETWEEN (cycle, opponent) cells, so an "
                             "on-policy learner shrinks the weak axes toward the marginal and the "
                             "turn-1 value barely separates opponents (spread ratio ~0.1 at turn "
                             "1 vs ~0.5-0.8 over all states). Later values already separate them, "
                             "so this moves that information BACKWARD within the episode along a "
                             "far less noisy channel "
                             "(designs/research_state/measurements/winprob_head_refit_2026-09-09/). "
                             "REQUIRES --critic winprob (refused otherwise -- under `shaped` the "
                             "buffer's values are a shaped return in PopArt units, not a "
                             "probability). Watch win_prob/lambda_target_shift, "
                             "lambda_bootstrap_frac and lambda_loss vs lambda_loss_terminal. "
                             "TRAINING-only, resume-inherited.")
    parser.add_argument("--win-prob-lambda-truncated", "--win_prob_lambda_truncated",
                        dest="win_prob_lambda_truncated", choices=("bootstrap", "mask"),
                        default=None,
                        help="How a TRUNCATED episode (still running when the rollout buffer "
                             "filled) is targeted under --win-prob-lambda < 1. `bootstrap` (the "
                             "DEFAULT) gives its states the bootstrap value V(s_T) from the same "
                             "post-rollout forward SB3's own GAE bootstrap uses, which UNMASKS "
                             "rows that carry no target today; `mask` leaves them excluded exactly "
                             "as they are now. It is a FLAG rather than a constant so a read can "
                             "attribute an effect to the target change rather than to the extra "
                             "rows -- win_prob/lambda_unmasked counts them per rollout. INERT at "
                             "--win-prob-lambda 1.0 (the recursion is skipped whole).")
    # --- gen3_dense_aux_v1 (2026-09-10, the critic ladder's arm 9): DENSE AUXILIARY targets. The
    #     fourth knob on the same objective — one scales it, strata re-prices its MIX, lambda
    #     re-aims it, and this one ADDS 25 dense targets beside it that share the win's cause. ---
    parser.add_argument("--win-prob-dense-aux", "--win_prob_dense_aux",
                        dest="win_prob_dense_aux", type=float, default=None,
                        help="DENSE AUXILIARY targets for the win-prob critic's trunk, coefficient "
                             ">= 0. 0.0 (the DEFAULT) = OFF and BIT-identical: the head is not "
                             "BUILT, so there is no module, no obs key, no callback and no loss "
                             "term. Above 0 it builds a small MLP on `value_pooled` -- the SAME "
                             "tensor the win head reads -- predicting, for every state, the "
                             "episode's END-OF-BATTLE facts back-filled the way the win bit is: "
                             "SURVIVAL of each of the 12 slots (our 6 then theirs, in the "
                             "observation's own team order), each slot's FINAL HP FRACTION, and "
                             "the scaled TURNS LEFT -- 25 sigmoid outputs, three masked-mean BCE "
                             "terms averaged, folded at this coefficient. The per-side KO counts "
                             "are DERIVED from survival and published as meters, never predicted. "
                             "WHY: one terminal bit copied to ~30 states carries only ~10%% of its "
                             "variance BETWEEN opponents, so the head shrinks the weak axes "
                             "(opponent, own team) toward the marginal although its features "
                             "carry them, and four 10M levers moved nothing at +-0.01. The "
                             "literature's answer to a one-bit terminal signal is KataGo's (Wu "
                             "2019 section 3): dense auxiliary targets that share the win's CAUSE "
                             "-- ownership of every point and the final score beside the win -- "
                             "reported as a large gain in learning efficiency. Per-Pokemon "
                             "end-of-battle outcomes are our analogue, and each one is a fact "
                             "about a NAMED ENTITY, so its gradient runs along exactly the axes "
                             "the pooled bit cannot separate. An opponent slot that was never "
                             "revealed, and any slot the state's own observation does not carry, "
                             "is MASKED rather than fabricated. REQUIRES --critic winprob "
                             "(refused otherwise). Watch win_prob/aux_auc_own vs aux_auc_opp, "
                             "aux_hp_mae, aux_masked_frac and grad/dense_aux_share. STRUCTURAL: "
                             "the head is fixed for a run's lifetime (a resume may re-dose it, "
                             "not add or remove it); the dose is resume-inherited.")
    # --- gen3_winprob_rollout_target_v1 (2026-09-10, the critic ladder's arm 10): R-ROLLOUT
    #     MONTE-CARLO targets for the win-prob BCE. The fourth knob on the SAME loss — one scales
    #     it, `--win-prob-strata-weight` re-prices its MIX, `--win-prob-lambda` re-aims it at the
    #     network's own later estimates, and this one BUYS NEW BITS by playing the state forward. ---
    parser.add_argument("--win-prob-rollout-target", "--win_prob_rollout_target",
                        dest="win_prob_rollout_target", type=float, default=None,
                        help="Fraction of the rollout buffer's states whose win-prob target is "
                             "replaced by an R-ROLLOUT MONTE-CARLO win fraction, in [0, 1]. 0.0 "
                             "(the DEFAULT) = OFF and BIT-identical. WHY: the terminal label is ONE "
                             "outcome bit copied to ~30 states -- one bit about the GAME and none "
                             "about the STATE -- and the head refit showed that is a TARGET defect "
                             "(only 10-14%% of the label's variance lies BETWEEN (cycle, opponent) "
                             "cells). R continuations from a state give R bits about THAT state. "
                             "Each sampled state is replayed to its own turn from the "
                             "--cf-records ring and played forward R times by the CURRENT policy on "
                             "both sides at temperature 1.0; the target becomes wins/R. "
                             "🚨 COST IS LINEAR AND LARGE: fraction x R x ~104 decisions per "
                             "continuation, against the n_envs x n_steps decisions the trainee "
                             "itself makes, so 1/32 at R=8 is ~26x the run's whole simulation "
                             "budget and the fraction that costs 1x is ~1/(R*104). Watch "
                             "win_prob/rollout_budget_multiple, rollout_seconds and rollout_mass. "
                             "REQUIRES --critic winprob AND --cf-records. TRAINING-only, "
                             "resume-inherited.")
    parser.add_argument("--win-prob-rollout-r", "--win_prob_rollout_r",
                        dest="win_prob_rollout_r", type=int, default=None,
                        help="Continuations per sampled state under --win-prob-rollout-target "
                             "(default 8). The label's standard error is ~0.5/sqrt(R) at p = 0.5, "
                             "and the COST is exactly linear in it -- R and the fraction trade "
                             "against each other at constant budget, so raising R means lowering "
                             "the fraction. INERT at --win-prob-rollout-target 0.")
    parser.add_argument("--win-prob-rollout-mode", "--win_prob_rollout_mode",
                        dest="win_prob_rollout_mode", choices=("replace", "blend"), default=None,
                        help="What a labelled state's target BECOMES. `replace` (the DEFAULT): the "
                             "rollout win fraction, full stop. `blend`: the mean of the rollout "
                             "fraction and the episode's terminal bit -- half the target shift, and "
                             "it keeps some of the RECORDED ecology, because the terminal bit was "
                             "played against the real opponent while the continuation was not (the "
                             "record carries no opponent identity, so every continuation plays a "
                             "self-like opponent; see win_prob/rollout_bot_share). INERT at "
                             "--win-prob-rollout-target 0.")
    parser.add_argument("--win-prob-rollout-weight", "--win_prob_rollout_weight",
                        dest="win_prob_rollout_weight", type=float, default=None,
                        help="Per-row LOSS WEIGHT on the rows --win-prob-rollout-target anchored, "
                             ">= 1.0. 1.0 (the DEFAULT) = OFF and BIT-identical. WHY: the fraction "
                             "that costs 1x the run's own simulation budget is ~0.0012, so the "
                             "rollout-labelled rows are ~0.12%% of the win-prob BCE's mass and the "
                             "head cannot move BY ARITHMETIC, whatever the new labels say. This is "
                             "the only lever that raises the treatment's share of the objective at "
                             "FIXED simulation cost -- no extra continuations, no changed labels. "
                             "The weight vector is renormalised to mean 1 over the scored rows, so "
                             "the loss SCALE does not move (the same convention "
                             "--win-prob-strata-weight uses, and it MULTIPLIES with it rather than "
                             "replacing it). At fraction 0.0012 a weight of 64 puts the anchors at "
                             "~7.1%% of the weighted mass. 🚨 ONLY THE ANCHOR ROWS are weighted, "
                             "never the rows that bootstrap toward them under "
                             "--win-prob-lambda < 1: their target is a MIXTURE of the anchor and "
                             "the network's own later values, so weighting them would dose arm 8's "
                             "channel under arm 10's flag. That propagated influence is MEASURED "
                             "instead -- read win_prob/rollout_mass_weighted (the anchors' share) "
                             "and win_prob/rollout_influence_lambda (anchors + their lambda^k "
                             "reach). REQUIRES --win-prob-rollout-target > 0. TRAINING-only, "
                             "resume-inherited.")
    parser.add_argument("--win-prob-coef", "--win_prob_coef", dest="win_prob_coef",
                        type=float, default=None,
                        help="Loss weight for the win-prob head's BCE (win_prob_coef * BCE), like "
                             "--opp-belief-aux-coef. Default 1.0. TRAINING-only (not version-locked; "
                             "inherited on a flagless resume). Ignored when --win-prob-mode none. Lower it "
                             "if 'shaping' fights the policy (watch grad/win_prob_share).")
