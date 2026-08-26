"""THE argparse surface — `build_parser()` and its three custom argparse pieces.

Split out of `train_rl_agent.py` so the parser can be read (and inspected by `main.checkargs`)
without loading the training entry point. `train_rl_agent` re-exports every name here, so
`from main.train_rl_agent import build_parser` still resolves.

**On the length** (~1,600 lines — inside the 2,000 hard bound, and in `file_size_gate_test.py`'s
REPORTED 1,000-2,000 band): that is a decision, not an oversight. This file is 197 flags declared
once each, read top-to-bottom exactly as `--help` renders them; splitting it into
`parser_reward.py` / `parser_arch.py` / ... would fragment the one surface a reader comes here to
see whole, and the natural section boundaries are wildly uneven (71 lines to 595). The gate agrees
by construction: the band is reported rather than failed, which is what a target means.
"""
import argparse

from agents.model.features_extractor import BELIEF_GRAD_MODES
from agents.training.eval_callback import _EVAL_SUBPROCESS_CONCURRENCY, EVAL_SHARD_GAMES
from agents.training.snapshot_pool import HEURISTIC_FLOOR, SELF_PLAY_FULL, SELF_PLAY_START
from agents.training.wrappers import STABLE_CHALLENGE_SHARE
from main.train.constants import (
    CLIP_RANGE_DEFAULT, SMOKE_EVAL_BATTLES, SMOKE_STEPS,
)

__all__ = ["optional_float", "str2bool", "BoolFlag", "build_parser",
           "_BOOL_TRUE", "_BOOL_FALSE"]


def optional_float(s: str) -> float | None:
    """argparse `type=` converter for an optional float (`float | None`).

    Returns `None` for the sentinels `none`/`null`/`""` (case-insensitive),
    otherwise parses a float. A bad value raises `ValueError`, which argparse
    turns into a clean usage error. Used by `--clip-range-vf` so `none`
    disables value-function clipping (SB3 branches on `clip_range_vf is None`).
    """
    if s.strip().lower() in ("none", "null", ""):
        return None
    return float(s)


_BOOL_TRUE = ("true", "t", "yes", "y", "1", "on")
_BOOL_FALSE = ("false", "f", "no", "n", "0", "off")


def str2bool(s: str) -> bool:
    """Parse a human boolean: true/false, yes/no, 1/0, on/off (case-insensitive)."""
    v = s.strip().lower()
    if v in _BOOL_TRUE:
        return True
    if v in _BOOL_FALSE:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean ({'/'.join(_BOOL_TRUE)} or {'/'.join(_BOOL_FALSE)}), got {s!r}")


class BoolFlag(argparse.Action):
    """Boolean flag accepting BOTH the bare/`--no-` form AND an explicit value.

    Registers a generated `--no-<flag>` for every `--<flag>` (like
    argparse.BooleanOptionalAction) but ALSO takes an optional value:
        --foo               -> True
        --no-foo            -> False
        --foo true | false  -> parsed (also yes/no, 1/0, on/off; --foo=false too)
    Passing a value to the negation (`--no-foo true`) is a usage error.
    """

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        opts, self._negatives = [], set()
        for opt in option_strings:
            opts.append(opt)
            if opt.startswith("--"):
                neg = "--no-" + opt[2:]
                opts.append(neg)
                self._negatives.add(neg)
        super().__init__(option_strings=opts, dest=dest, nargs="?", default=default,
                         required=required, help=help, metavar="{true,false}")

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self._negatives:
            if values is not None:
                raise argparse.ArgumentError(
                    self, f"{option_string} is a negation and does not take a value")
            setattr(namespace, self.dest, False)
        elif values is None:            # bare `--foo`
            setattr(namespace, self.dest, True)
        else:                           # `--foo <value>` / `--foo=<value>`
            setattr(namespace, self.dest, str2bool(values))



def build_parser() -> argparse.ArgumentParser:
    """THE argument parser, as data — built outside `main()` so it can be INSPECTED without
    running a training job.

    Extracted for two reasons, both of which cost real time before it existed:

    * A run's recorded `launcher_command` outlives the flags in it. Relaunching gen-12's
      argv on v89 died on `--pubval-*` (deleted at v88) — one flag at a time, since argparse
      reports only the first error, and only by actually starting the trainer.
    * `--help` was itself broken (an unescaped `%` rendered as a `%o` conversion), so there
      was no offline way to ask what the parser accepts. Nothing rendered the help strings,
      so nothing caught it.

    `python -m main.checkargs` is the consumer; `checkargs_test.py` renders every help string.
    """
    parser = argparse.ArgumentParser(description="Train or Evaluate Gen 3 OU RL Agent")
    
    # --- Operational Flags ---
    parser.add_argument("--model", type=str, help="Path to existing model to load")
    parser.add_argument("--run-dir", type=str, help="Run folder to write checkpoints into (set by launcher on resume)")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default=None,
                        help="A MEMORABLE name for a fresh run → writes to models/<name>/ instead of "
                             "a date-stamped models/run_<timestamp>/. Must be a single name "
                             "(letters/digits/._-, no slashes). Refuses to overwrite an existing run "
                             "of that name (pick another, or --model to resume it). Ignored when "
                             "--run-dir is set (launcher resume). For --exploiter, defaults to "
                             "'exploiter_vs_<target>' if you don't name it.")
    parser.add_argument("--eval-only", action=BoolFlag, default=False, help="Skip training and only evaluate")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--debug", action=BoolFlag, default=False, help="Use DummyVecEnv (1 env) for debugging")
    parser.add_argument("--debug-eval", "--debug_eval", dest="debug_eval", action=BoolFlag, default=False,
                        help="Run evaluation under --debug. By default a --debug smoke run skips ALL eval "
                             "(both the periodic eval callback AND the final win-rate eval) so it needs no "
                             "eval opponents / Showdown eval connection and stays light on CPU. Pass "
                             "--debug-eval to exercise the eval pipeline in a smoke run. No effect on real "
                             "(non-debug) runs, which always eval.")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--async-rollout", "--async_rollout", dest="async_rollout",
                        action=BoolFlag, default=False,
                        help="Non-barrier async rollout collection: keep every env worker "
                             "continuously in-flight and forward whichever are ready, instead of "
                             "barriering on the slowest env each step (AsyncSubprocVecEnv + an "
                             "on-policy async collect_rollouts that overlaps the GPU forward with "
                             "CPU env-stepping). Off by default; ignored under --debug. With async, "
                             "right-size --n-envs nearer the core count (16) rather than oversubscribing.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, or auto)")
    parser.add_argument("--showdown-port", type=int, default=None,
                        help="Local Showdown server port (default 8000). Sets the port for the trainee, "
                             "eval, and self-play clients. Start the server on the matching port, "
                             "e.g. npm run showdown -- <port>.")
    parser.add_argument("--use-bridge", type=str, default="rust",
                        choices=["off", "node", "rust"],
                        help="In-process BattleStream bridge transport for BOTH training AND eval "
                             "(no Showdown server, no port, no /challenge storm, deterministic). "
                             "'rust' (DEFAULT) = the byte-compatible src/rust_sim sim_bridge binary "
                             "(built via cargo; override with POKESIM_SIM_BRIDGE_BIN) — measured "
                             "1.41x node's throughput at --n-envs 48 with a ~25x smaller child "
                             "(9 MB RSS vs ~224 MB). 'node' = the Node local_sim_bridge.js, kept as "
                             "the explicit A/B arm and for the parity harness. 'off' = the websocket "
                             "transport, which needs a running Showdown server on --showdown-port. "
                             "NOTE: 'rust' now emits __RECON__ (gen3_bridge_recon_record_v1, on a "
                             "seedless battle too) and supports resumeReseed "
                             "(gen3_bridge_resume_reseed_v1), so the forensic reconstruction and "
                             "counterfactual paths work on rust. The OFFLINE search/replay drivers "
                             "are on rust too (gen3_rust_search_driver_v1 / "
                             "gen3_rust_replay_driver_v1 — one search_driver binary serves both "
                             "verb families), so --search-teacher no longer requires 'node'; the "
                             "run's impl is threaded into the teacher workers. 'rust' also "
                             "fail-louds on an unmodeled move.")
    parser.add_argument(
        "--self-play-use-cpu",
        action=BoolFlag,
        default=True,
        help="Load self-play opponent snapshots on CPU instead of the training device. "
             "Default True: avoids one CUDA context per SubprocVecEnv worker (~300-600 MB each), "
             "which would otherwise OOM the GPU at high --n-envs. Opponent inference is batch-1 "
             "no_grad, so CPU is plenty fast. Pass --no-self-play-use-cpu to load them on --device.",
    )
    parser.add_argument("--eval-battles", type=int, default=None,
                        help="Battles per FINAL-evaluation opponent. Default 100, but AUTO-SCALED "
                             f"down to {SMOKE_EVAL_BATTLES} when --steps < {SMOKE_STEPS:,} (a smoke "
                             "run), because a 9-opponent x 100-battle final eval costs many minutes "
                             "and a 2k-step policy produces no signal worth that. An explicit value "
                             "always wins.")
    parser.add_argument("--eval-concurrency", type=int, default=100, help="Concurrent battles during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="periodic", choices=["quiet", "periodic", "detailed", "debug"], help="Logging verbosity level")

    # --- Hyperparameter Flags (Optimized for GPU) ---
    parser.add_argument("--batch-size", type=int, default=4096, help="PPO mini-batch size")
    parser.add_argument("--grad-accum-steps", "--grad_accum_steps", dest="grad_accum_steps",
                        type=int, default=1,
                        help="Gradient accumulation: sum the gradients of K --batch-size MICRO-batches "
                             "and step the optimizer ONCE per group of K, giving the EXACT gradient of a "
                             "(batch_size·K) batch at the GPU-memory cost of batch_size (only one "
                             "micro-batch's activations are ever held). 1 = OFF (one step per minibatch, "
                             "byte-identical to stock). Use it to keep a large effective batch when the "
                             "full minibatch OOMs: e.g. --batch-size 4096 --grad-accum-steps 4 ≈ "
                             "--batch-size 16384 at ¼ the activation peak. A train-loop knob (not "
                             "version-locked); pass it on every resume like --batch-size.")
    parser.add_argument("--checkpoint-every-steps", "--checkpoint_every_steps",
                        dest="checkpoint_every_steps", type=int, default=None,
                        help="ENV-STEP interval between periodic checkpoints. Default None = the "
                             "historical hardcoded cadence, which is 50000 VEC-ENV CALLS and "
                             "therefore 50000 x --n-envs ENV STEPS (2,400,000 at --n-envs 48) — a "
                             "multiplier that was invisible until it starved the counterfactual "
                             "label path. A value here is converted back to vec-calls by "
                             "ceil-division, so it is honoured to within one rollout. Lower it "
                             "when an out-of-process consumer reloads the newest checkpoint (the "
                             "cf label producer): --cf-label-lag-steps divided by this is the "
                             "label DUTY CYCLE the launch announces, and a value under 25%% is "
                             "refused. A train-loop knob (not version-locked); pass it on every "
                             "resume like --batch-size.")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO optimization epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (AdaptiveLRCallback adjusts from here)")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Hard lower bound on adaptive LR")
    parser.add_argument("--max-lr", type=float, default=None, help="Hard upper bound on adaptive LR (default: 2× --lr)")
    parser.add_argument("--anneal-lr-start-steps", type=int, default=None,
                        help="Absolute global step at which cosine LR decay begins. "
                             "Duration = --steps minus this value. Pass the same value on every resume.")
    parser.add_argument("--anneal-min-lr", type=float, default=None,
                        help="LR floor for annealing (required with --anneal-lr-start-steps). "
                             "Separate from --min-lr used by AdaptivePPO.")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient (exploration bonus)")
    parser.add_argument("--defensive-entropy-boost", "--defensive_entropy_boost", dest="defensive_entropy_boost",
                        type=float, default=1.0,
                        help="STATE-CONDITIONED entropy boost (gen3_defensive_entropy_v1): multiply the "
                             "per-decision entropy bonus by this factor ON decisions where the active mon has a "
                             "productive defensive move legal (HP-recovery with HP to restore, or a self/team "
                             "status-cure with a status to clear). Keeps the policy EXPLORING defensive moves "
                             "(Recover/Soft-Boiled/Wish/Refresh/Heal Bell) instead of collapsing to attacking, "
                             "WITHOUT touching the reward (no stall incentive — the draw penalty + no-progress "
                             "clock stay the guardrail; the model only keeps healing if the returns reward it). "
                             "1.0 = OFF (byte-identical). Try 3.0. TRAINING-only (not version-locked).")
    parser.add_argument("--defensive-entropy-anneal-frac", "--defensive_entropy_anneal_frac",
                        dest="defensive_entropy_anneal_frac", type=float, default=0.0,
                        help="Anneal --defensive-entropy-boost linearly back to 1.0 over this FRACTION of total "
                             "--steps (e.g. 0.5 = boost fades to off by the halfway point). 0.0 = constant boost "
                             "(default). Lets exploration fade as the policy learns defensive value.")
    parser.add_argument("--bait-entropy-boost", "--bait_entropy_boost", dest="bait_entropy_boost",
                        type=float, default=1.0,
                        help="STATE-CONDITIONED entropy boost (gen3_bait_entropy_v1): multiply the "
                             "per-decision entropy bonus by this factor ON bait-opportunity decisions — the "
                             "attack we would click deals ZERO damage to an alive, revealed opponent BENCH mon "
                             "(the board the bait loop is fired from). This is the SAMPLING-side probe of the "
                             "bait verdict: the whiff sits at p~0.97, so the alternatives at p~0.01-0.03 are "
                             "never sampled and their advantage is never realized. Does NOT touch the reward "
                             "and does not tell the policy which action to take. 1.0 = OFF (byte-identical). "
                             "Try 3.0. TRAINING-only (not version-locked, settable on resume).")
    parser.add_argument("--bait-entropy-anneal-frac", "--bait_entropy_anneal_frac",
                        dest="bait_entropy_anneal_frac", type=float, default=0.0,
                        help="Anneal --bait-entropy-boost linearly back to 1.0 over this FRACTION of total "
                             "--steps. 0.0 = constant boost (default). This is what makes the probe TWO-SIDED: "
                             "a whiff rate that falls and STAYS down past the anneal means sampling was the "
                             "block; one that reverts convicts CREDIT (and the off-policy levers inherit).")
    parser.add_argument("--vf-coef", "--vf_coef", dest="vf_coef", type=float, default=0.5,
                        help="PPO value-loss coefficient (default 0.5, the SB3 default). Fixed for a "
                             "run's lifetime: it is recorded in model_config.json and resuming with a "
                             "different value is a FATAL error (it silently rescales the value head's "
                             "gradient on the shared trunk — tune it on a fresh run). Watch "
                             "grad/value_policy_logratio (the aux-independent value-vs-policy balance).")
    parser.add_argument("--value-tail-weight", "--value_tail_weight", dest="value_tail_weight",
                        type=float, default=0.0,
                        help="Tail-weighted value loss β∈[0,1] (default 0.0 = plain MSE, byte-identical). "
                             ">0 blends in the CVaR of the worst ~10%% value misses: (1-β)·MSE + β·CVaR, "
                             "so the critic prioritises the big over-claim craters it under-prices (a "
                             "probe found VF→incoming-KO AUC 0.79 vs the policy's 0.90). Symmetric in "
                             "error sign → V stays unbiased (GAE advantages unaffected). Watch "
                             "eval/td_resid_tail fall. Resume-immutable (recorded + FATAL to change).")
    # --- Reward config (design_markovian_reward_and_features.md). Resume-immutable, value-checked. ---
    parser.add_argument("--bias-additivity", "--bias_additivity", dest="bias_additivity", type=float,
                        default=1.0, help="BIAS-class additive↔telescoping knob λ∈[0,1] (default 1.0 = "
                        "fully additive, byte-identical to today's biases). 0.0 = fully telescoping "
                        "(pure PBRS hint). Per-run constant (NOT annealed). Resume-immutable.")
    parser.add_argument("--mat-alive-weight", "--mat_alive_weight", dest="mat_alive_weight", type=float,
                        default=1.25, help="Material PBRS Φ_mat per-mon-alive weight (default 1.25). "
                        "Resume-immutable.")
    parser.add_argument("--no-progress-penalty", "--no_progress_penalty", dest="no_progress_penalty",
                        type=float, default=0.15, help="Flat per-no-progress-window penalty magnitude "
                        "(default 0.15; only charged when --bias-redesign).")
    parser.add_argument("--bias-redesign", "--bias_redesign", dest="bias_redesign", action=BoolFlag,
                        default=False, help="Enable the staged BIAS redesign: the no-progress clock "
                        "replaces the anti-spam taxes + the obs-keyed reframes apply. Default OFF = the "
                        "single-variable run (material clutch-fix only). Pass --no-bias-redesign (or "
                        "--bias-redesign false) to set it off explicitly. Resume-immutable.")
    parser.add_argument("--switch-bias-weight", "--switch_bias_weight", dest="switch_bias_weight",
                        type=float, default=0.0, help="Belief-risk-scaled stay-into-KO BIAS lever for "
                        "the under-switch pathology (design_reward_switching.md §7). 0.0 = OFF "
                        "(default; behavior unchanged). >0 taxes staying in a high-P(KO) spot when a "
                        "safe pivot exists (−w·risk) + rewards escaping it. BIAS-class, so it also "
                        "rides --bias-additivity (λ=1 additive vs λ=0 telescoping A/B). Resume-immutable.")
    parser.add_argument("--draw-penalty", "--draw_penalty", dest="draw_penalty", type=float,
                        default=-35.0, help="Terminal reward for a DRAW / 250-turn timeout (no "
                        "winner). DEFAULT -35.0 = the validated ai_v8 value: stalling to the turn "
                        "cap is strictly worse than losing cleanly, which cancels the discount-driven "
                        "micro-incentive to delay an inevitable loss. A decisive loss stays -30. Pass "
                        "-30 for the historical default (a tie scored as a decisive loss), which was "
                        "tuned under the additive-BIAS regime --all-shaping-pbrs replaces. "
                        "Resume-immutable (recorded + value-checked in model_config.json).")
    parser.add_argument("--self-ko-hp-penalty", "--self_ko_hp_penalty", dest="self_ko_hp_penalty",
                        type=float, default=0.0,
                        help="Decision-time-HP-scaled penalty (-w*hp) for self-KOing a mon via "
                        "Explosion/Self-Destruct. Default 0.0 = OFF (behavior unchanged). The symmetric "
                        "material PBRS prices a healthy 1-for-1 trade at ~0, so the critic learns to "
                        "value a full-HP self-KO positively and the policy throws away healthy mons "
                        "(measured: ~38%% of explosions are on >=80%%-HP mons). >0 (e.g. 2.5) charges "
                        "the squandered HP, sparing legitimate low-HP sac-for-KO. Resume-immutable "
                        "(recorded + value-checked in model_config.json).")
    parser.add_argument("--drop-redundant-bias", "--drop_redundant_bias", dest="drop_redundant_bias",
                        action=BoolFlag, default=False, help="De-bias cleanup: zero BIAS terms REDUNDANT "
                        "with an existing PBRS/terminal term — stall_tax (covered by the no-progress clock "
                        "+ --draw-penalty; it also taxed winning long games on raw turn count) and "
                        "matchup_penalty (the same incoming-KO threat signal as pbrs_belief, but additive "
                        "not telescoping). Default OFF = byte-identical. Resume-immutable, value-checked.")
    parser.add_argument("--drop-switch-bias", "--drop_switch_bias", dest="drop_switch_bias",
                        action=BoolFlag, default=False, help="De-bias cleanup: zero the HAND-CODED "
                        "switch-strategy subsidy (switch_base, switch_bouncing_tax, escape_threat_switch, "
                        "se_switch, pivot_*, sleep_in/out) — switching value is LEARNABLE from Φ_mat + "
                        "pbrs_belief + win/loss, so hand-rewarding it distorts the objective. Default OFF "
                        "= byte-identical. Resume-immutable, value-checked.")
    parser.add_argument("--all-shaping-pbrs", "--all_shaping_pbrs", dest="all_shaping_pbrs",
                        action=BoolFlag, default=True, help="END-STATE PBRS, 'everything but stall': "
                        "fold Φ_hazard/Φ_boost/Φ_opp_boosts + Φ_status (telescoping, policy-invariant) "
                        "and ZERO every BIAS term EXCEPT the anti-stall tilt no_progress_tax — so all "
                        "non-stall shaping is policy-invariant (the bad turn-ramp stall_tax is zeroed). "
                        "DEFAULT ON = the validated ai_v8 composition (1 TERMINAL + 7 PBRS + 1 BIAS); "
                        "--no-all-shaping-pbrs is the fallback and restores the fully-additive "
                        "26-term BIAS objective every ai_v9 run drifted into. Pair with --stall-pbrs for "
                        "a FULLY-PBRS reward, or use alone to keep the no_progress stall tilt as the one "
                        "acknowledged BIAS. Resume-immutable, value-checked.")
    parser.add_argument("--stall-pbrs", "--stall_pbrs", dest="stall_pbrs",
                        action=BoolFlag, default=False, help="END-STATE PBRS, 'stall': fold Φ_progress "
                        "(telescoping anti-stall over the turns_since_progress clock) and ZERO "
                        "no_progress_tax + stall_tax, so the anti-stall signal is policy-invariant too. "
                        "Default OFF. Run --all-shaping-pbrs WITH --stall-pbrs ⇒ the whole BIAS class is "
                        "zero (TERMINAL + PBRS only); WITHOUT it ⇒ keep the no_progress stall tilt as "
                        "insurance against stall-regression (watch the stall-rate canary). Resume-"
                        "immutable, value-checked.")
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
    parser.add_argument("--win-prob-coef", "--win_prob_coef", dest="win_prob_coef",
                        type=float, default=None,
                        help="Loss weight for the win-prob head's BCE (win_prob_coef * BCE), like "
                             "--opp-belief-aux-coef. Default 1.0. TRAINING-only (not version-locked; "
                             "inherited on a flagless resume). Ignored when --win-prob-mode none. Lower it "
                             "if 'shaping' fights the policy (watch grad/win_prob_share).")
    # --- SEARCH-AS-TEACHER (offline ExIt plateau-breaker; designs/ai_v6/design_search_teacher.md) ---
    # All TRAINING-only (no version bump; coef 0 / flag absent = byte-identical). The coefs are
    # _resolve'd (flagless-resume-inherited); the operational knobs are forwarded by the launcher.
    parser.add_argument("--search-teacher", "--search_teacher", dest="search_teacher",
                        action="store_true",
                        help="Enable the search-teacher: each cycle, search + rollout-confirm the worst "
                             "falsify-flagged loss craters (EXACT reloaded opponent), CI-gate strictly-"
                             "better corrections, and distil them into the policy via an AWR aux loss. "
                             "Non-blocking (subprocess workers). Recommended at PLATEAU. Re-pass on resume.")
    parser.add_argument("--search-teacher-coef", "--search_teacher_coef", dest="search_teacher_coef",
                        type=float, default=None,
                        help="AWR policy-distillation weight (search_teacher_coef * advantage-weighted CE "
                             "toward the verified-better action). Default 0.0 = OFF (loss byte-identical). "
                             "Training-only (inherited on a flagless resume). Watch grad/searchteacher_share "
                             "+ teacher/agree_rate.")
    parser.add_argument("--search-teacher-value-coef", "--search_teacher_value_coef",
                        dest="search_teacher_value_coef", type=float, default=None,
                        help="OFF by default (0.0) — the off-policy value term (the search value is V^π*, "
                             "which biases the GAE critic). Only for the joint-ExIt A/B.")
    parser.add_argument("--search-teacher-beta", "--search_teacher_beta", dest="search_teacher_beta",
                        type=float, default=None, help="AWR temperature β (default 1.0).")
    # ON-POLICY SELF-DISTILLATION (OPD) — upgrades the distillation TARGET from the single action A*
    # (AWR) to the FULL improved distribution π' via KL(π' ‖ π_student). Training-only, modelled EXACTLY
    # on --search-teacher-coef (0 = byte-identical; NOT version-locked). REQUIRES --search-teacher (it
    # fills the correction buffer + its workers build π'). A run carries BOTH targets → A/B AWR vs KL.
    parser.add_argument("--opd-coef", "--opd_coef", dest="opd_coef", type=float, default=None,
                        help="ON-POLICY SELF-DISTILLATION weight (opd_coef * KL(π' ‖ π_student) toward the "
                             "beam's improved distribution). Default 0.0 = OFF (loss byte-identical). "
                             "Requires --search-teacher. Training-only (inherited on a flagless resume). "
                             "Watch grad/opd_share + opd/kl / opd/agree_rate.")
    parser.add_argument("--opd-beta", "--opd_beta", dest="opd_beta", type=float, default=None,
                        help="OPD softmax temperature β for π' over the per-action backed-up values "
                             "(default 1.0). Higher β → flatter target.")
    # --- COUNTERFACTUAL VALUE GROUNDING (gen3_cf_label_plumbing_v1; G3 of
    # designs/ai_v10/design_counterfactual_value_grounding.md, rung R1). An OUT-OF-PROCESS producer
    # re-rolls recorded training decisions to termination and drops tight Monte-Carlo P(win) labels
    # into <run_dir>/cf_labels/; the trainer rings the reconstruction records the producer needs
    # (--cf-records) and folds the labels into the win-prob head's BCE (--cf-winprob-coef).
    # ALL TRAINING-ONLY: no weight shape, no forward change — the `td_aux_coef` class. Every
    # default is OFF, and an off run is byte- AND file-identical to today.
    # INHERITED on a FLAGLESS resume since config v100 (gen3_cf_coef_provenance_v1): every one of
    # them is a recorded `ModelVersion` field with a `_resolve` line, so a resume that re-types
    # nothing keeps the coefficients it was launched with. They are recorded for PROVENANCE only
    # and never gated — a resume may still change any of them freely.
    parser.add_argument("--cf-records", "--cf_records", dest="cf_records",
                        action=BoolFlag, default=None,
                        help="Ring each training episode's __RECON__ reconstruction record into "
                             "<run_dir>/cf_records/ (newest --cf-records-keep only) so an offline "
                             "counterfactual LABEL PRODUCER can replay those decisions. Default OFF "
                             "— training discards the records today. Costs one small file write per "
                             "episode per env worker; requires --use-bridge (node or rust).")
    parser.add_argument("--cf-records-keep", "--cf_records_keep", dest="cf_records_keep",
                        type=int, default=None,
                        help="GLOBAL cap on <run_dir>/cf_records/ (default 512). Every env worker "
                             "prunes the shared dir to the newest N, so this is a total, not a "
                             "per-worker count, and it holds across launcher restarts.")
    parser.add_argument("--cf-winprob-coef", "--cf_winprob_coef", dest="cf_winprob_coef",
                        type=float, default=None,
                        help="COUNTERFACTUAL win-prob grounding weight: cf_winprob_coef * "
                             "BCE(win_head(s), tight-MC P(win) label) over labels the producer left "
                             "in <run_dir>/cf_labels/. Default 0.0 = OFF (no poll, no forward, loss "
                             "byte-identical). Requires --win-prob-mode != none (there must be a head "
                             "to supervise). Watch cf/buffer_fill (0 = the producer is starving you), "
                             "train/cf_loss and train/cf_grad_share.")
    parser.add_argument("--cf-head-only", "--cf_head_only", dest="cf_head_only",
                        action=BoolFlag, default=None,
                        help="Stop-grad the win-prob head's input for the CF term, so it trains the "
                             "HEAD ONLY and cannot perturb the trunk (train/cf_grad_share reads 0.0 "
                             "by construction). Default TRUE — the safe first stage the design's R1 "
                             "prescribes. --no-cf-head-only (or --cf-head-only false) lets the "
                             "ground-truth objective shape the shared trunk. Independent of "
                             "--win-prob-mode, which governs the ON-POLICY win-prob BCE, not this.")
    parser.add_argument("--cf-label-lag-steps", "--cf_label_lag_steps", dest="cf_label_lag_steps",
                        type=int, default=None,
                        help="STALENESS BOUND in policy steps: a label whose policy_step is older "
                             "than this is dropped (counted in cf/labels_expired_total). Default "
                             "150000 ≈ one PPO iteration at production shapes, so a label is "
                             "consumed by roughly the policy that produced it. 0 disables expiry.")
    parser.add_argument("--cf-label-likelihood", "--cf_label_likelihood",
                        dest="cf_label_likelihood", type=str, default=None,
                        choices=["binomial", "bce"],
                        help="WHICH likelihood the counterfactual win-prob term uses. 'binomial' "
                             "(default) is the exact binomial NLL of the row's win COUNT "
                             "(w=round(label*n_rollouts), folded as sum(NLL)/sum(n)), so an R=16 "
                             "label pulls 4x an R=4 one — correct evidence weighting, not an "
                             "emphasis choice. 'bce' is the flat per-row BCE on the scalar label "
                             "(the pre-2026-08-22 form, kept as the A/B arm). The two are EXACTLY "
                             "equal when every n_rollouts == 1. Training-only.")
    # --- THE EVIDENTIAL BETA HEAD (gen3_cf_evidential_head_v1). G0 convicted the scalar win-prob
    # head of RESOLUTION (within-decile true spread 0.11-0.36), not of an optimism offset. This head
    # cannot fix the blur — same input — but it can CONFESS it, as a Beta posterior whose width the
    # factory's priority sampler and the awareness stack can read. `--cf-evidential` is STRUCTURAL
    # (v98, version-gated, in flag_registry); the two coefficients are training-only.
    parser.add_argument("--cf-evidential", "--cf_evidential", dest="cf_evidential",
                        action=BoolFlag, default=None,
                        help="BUILD the evidential Beta head (α, β via softplus+1) off "
                             "value_pooled. STRUCTURAL and version-gated: its params are in the "
                             "state_dict, so a resume must match. Its input is detached "
                             "UNCONDITIONALLY — a pure supervised readout that feeds nothing "
                             "forward and is not even called by the forward, so OFF is "
                             "byte-identical and ON at coefficient 0 is bit-identical in pi/vf.")
    parser.add_argument("--cf-evidential-coef", "--cf_evidential_coef",
                        dest="cf_evidential_coef", type=float, default=None,
                        help="Weight on the evidential term: the Beta-Binomial MARGINAL "
                             "log-likelihood of the label's counts (p integrated out, the correct "
                             "evidential loss for count data), normalized per rollout like the "
                             "scalar term. Default 0.0 = OFF. Requires --cf-evidential. Watch "
                             "cf/evid_epistemic_std_mean (the confessed width — the headline) and "
                             "cf/evid_precision_mean (α+β, the claimed evidence).")
    parser.add_argument("--cf-evidential-reg", "--cf_evidential_reg",
                        dest="cf_evidential_reg", type=float, default=None,
                        help="Weight of the KL(Beta(α,β) ‖ Beta(1,1)) pull, RELATIVE to the NLL "
                             "(it rides inside --cf-evidential-coef, so coefficient 0 kills it "
                             "too). The standard evidential-overconfidence guard: nothing in the "
                             "likelihood bounds α+β on locally-consistent data, and an inflated "
                             "precision makes the width — the entire product — meaningless. "
                             "Default 1e-3.")
    # --- THE TWIN HEADS + THE SHADOW CRITIC (gen3_cf_twin_heads_v1, v99). The owner-authorized
    # amendment to the signed R1 pre-registration (ledger 2026-08-22 evening, "Three owner
    # sign-offs" item 3): the arm's primary comparison becomes a WITHIN-RUN paired head difference
    # instead of a run-vs-run one. Three win-prob heads on ONE trunk — A (control, on-policy BCE
    # only), B (+ the cf states with SINGLE-OUTCOME labels), C (+ the same states with TIGHT-MC
    # labels) — so B−A isolates coverage and C−B isolates pure variance reduction with every random
    # draw held identical. The two structural flags are version-gated; the coefficients are not.
    parser.add_argument("--cf-twin-heads", "--cf_twin_heads", dest="cf_twin_heads",
                        action=BoolFlag, default=None,
                        help="BUILD the TWIN win-prob heads B and C off value_pooled (the "
                             "within-run paired R1 comparison). STRUCTURAL and version-gated: "
                             "their params are in the state_dict, so a resume must match, and "
                             "because the forward never calls them nothing else would catch a "
                             "flip. Head-only ALWAYS in v1 (both read a DETACHED value_pooled), so "
                             "OFF is byte-identical and ON at coefficient 0 is bit-identical in "
                             "pi/vf. Requires --win-prob-mode read_only|shaping (head A must exist "
                             "for the twins to mirror its loss).")
    parser.add_argument("--cf-twin-coef", "--cf_twin_coef", dest="cf_twin_coef",
                        type=float, default=None,
                        help="Weight on BOTH twins' cf folds — ONE knob on purpose: B and C must "
                             "differ in their LABEL STREAM and in nothing else. B eats the row's "
                             "outcome_label at n=1, C eats its tight-MC label at n=R, through the "
                             "SAME per-rollout-normalized binomial NLL, so the two pull equally "
                             "hard and C−B reads label PRECISION rather than learning rate. Their "
                             "share of head A's own on-policy BCE rides --win-prob-coef, not this. "
                             "Default 0.0 = OFF (whole block skipped, byte-identical). Requires "
                             "--cf-twin-heads. Read cf/twin_b_coverage FIRST.")
    parser.add_argument("--cf-shadow-critic", "--cf_shadow_critic", dest="cf_shadow_critic",
                        action=BoolFlag, default=None,
                        help="BUILD the passive SHADOW CRITIC off value_pooled — a value twin "
                             "trained on tight-MC mc_return labels (the run's own shaped return). "
                             "It NEVER computes an advantage and NEVER enters GAE: it is the "
                             "staged promotion path for critic surgery (a critic route change owes "
                             "the C4 gate), so it accumulates evidence rather than changing the "
                             "critic. STRUCTURAL and version-gated; detached always; OFF "
                             "byte-identical, ON at coefficient 0 bit-identical in pi/vf.")
    parser.add_argument("--cf-shadow-coef", "--cf_shadow_coef", dest="cf_shadow_coef",
                        type=float, default=None,
                        help="Weight on the shadow critic's masked MSE against mc_return, computed "
                             "in the PopArt-normalized frame (the value loss's frame, so the "
                             "coefficient is scale-comparable with it). Default 0.0 = OFF. "
                             "Requires --cf-shadow-critic. THE METER is cf/shadow_shadow_vs_live_v "
                             "— the signed real-unit gap between the MC-grounded twin and the live "
                             "critic on the same states.")
    # --- LIVE CAPACITY TELEMETRY (gen3_capacity_telemetry_v1) --------------------------------
    # Three continuous saturation early-warnings that ride the train loop instead of being probed
    # offline: the PLASTICITY CANARY, the HALF-BATCH TRUNK-GRADIENT COSINE, and the FIXED-PROBE
    # FEATURE VELOCITY. All TRAINING-only and, uniquely, not even a loss weight — nothing here
    # enters `loss` or writes `.grad`, so the policy's updates are bit-identical on or off. They
    # are recorded on `ModelVersion` for PROVENANCE and `_resolve`-inherited on a flagless resume
    # (the `td_aux_coef` / `cf_records` class), never gated. Detail:
    # `agents/training/capacity_telemetry.py` and `src/agents/training/CLAUDE.md`.
    parser.add_argument("--capacity-telemetry", "--capacity_telemetry", dest="capacity_telemetry",
                        action=BoolFlag, default=None,
                        help="Log the `capacity/*` saturation early-warnings: the PLASTICITY CANARY "
                             "(a detached head refitting K=4 seeded synthetic targets, one of which "
                             "is re-seeded every --canary-reset-steps), the HALF-BATCH TRUNK COSINE "
                             "(does the batch fight itself?) and the FEATURE VELOCITY on a frozen "
                             "probe batch (do the functions still move?). Default OFF — off is "
                             "byte- AND cost-identical (no head, no optimizer, no probe batch, no "
                             "extra forward or backward). ON costs <3%% of the train step. Read every "
                             "scalar as a TREND, never as a level. ⚠️ The canary's state is NOT "
                             "checkpointed: it re-inits on every resume/launcher restart, so its "
                             "curves restart there too.")
    parser.add_argument("--canary-reset-steps", "--canary_reset_steps", dest="canary_reset_steps",
                        type=int, default=None,
                        help="ENV steps between plasticity-canary resets (default 1000000). Each "
                             "reset re-seeds ONE of the K=4 synthetic targets, round-robin; the "
                             "re-fit that follows is the supply-side measurement, read off "
                             "capacity/canary_recovery at a MATCHED capacity/canary_age. Too small "
                             "and the head never converges between resets; too large and the run "
                             "yields two points.")
    parser.add_argument("--capacity-cosine-every", "--capacity_cosine_every",
                        dest="capacity_cosine_every", type=int, default=None,
                        help="Minibatches between half-batch trunk-gradient cosine measurements "
                             "(default 50). The probe costs two half-batch forward+backwards ≈ one "
                             "extra full one, so 50 amortizes it to ~2%% of the train step. 0 = off "
                             "(the other two probes keep running).")
    parser.add_argument("--capacity-velocity-every", "--capacity_velocity_every",
                        dest="capacity_velocity_every", type=int, default=None,
                        help="train() calls between feature-velocity measurements (default 50). One "
                             "no_grad forward of a frozen 256-row probe batch. 0 = off.")
    # EXPLOITER DISTILLATION (gen3_exploiter_distill_v1) — pour a frozen per-team SPECIALIST (an
    # --exploiter checkpoint) into the generalist via an ON-POLICY KL, masked to the states where the
    # trainee pilots the teacher's team; the other (pool) states are the anti-forgetting rehearsal.
    # Training-only (0 = byte-identical; NOT version-locked). designs/learning/generalist_specialist_amortization_gap.md
    parser.add_argument("--distill-teacher", "--distill_teacher", dest="distill_teacher", type=str, default=None,
                        help="Frozen exploiter teacher(s) to distil into the trainee, as "
                             "'TEACHER:TEAM' pairs (KL(π_teacher ‖ π_student) on that teacher's team states). "
                             "TEACHER = a checkpoint dir/.zip, TEAM = its Showdown team file. Comma-separated "
                             "for N teachers (joint multi-teacher distillation), e.g. "
                             "'models/expA:data/teams/specialist/a.txt,models/expB:data/teams/specialist/b.txt'. "
                             "The colon pairing binds each teacher to its team — no misalignment possible.")
    parser.add_argument("--distill-coef", "--distill_coef", dest="distill_coef", type=float, default=None,
                        help="Exploiter-distillation KL weight (default 0.0 = OFF, loss byte-identical). "
                             "Requires --distill-teacher ('TEACHER:TEAM' pairs). Training-only (inherited on "
                             "a flagless resume). Watch distill/kl (the mean over active teachers) FALL "
                             "and the per-teacher distill/t<k>_agree_rate RISE, with distill/t<k>_coverage "
                             "confirming the trainee actually pilots that teacher's team, and "
                             "grad/distill_share (gen3_grad_distill_share_v1) reading the KL's own "
                             "shared-trunk gradient share — the dose meter G1/G2 arms are matched on "
                             "(design_advantage_gated_distillation.md §6.2).")
    parser.add_argument("--distill-value-coef", "--distill_value_coef", dest="distill_value_coef",
                        type=float, default=None,
                        help="VALUE-distillation weight (gen3_exploiter_value_distill_v1): also pour the "
                             "teacher's per-team VALUE into the student — MSE(V_teacher, V_student) on the "
                             "teacher-team states, in the PopArt-normalized frame. Default 0.0 = OFF "
                             "(byte-identical; no teacher predict_values forward). Requires --distill-coef > 0 "
                             "(the policy KL validates the value target). Training-only, inherited on resume. "
                             "The A/B lever for 'does distilling the value enrich it' — watch distill/value_mse ↓ "
                             "and the value_cls effective-rank probe rise. Distributional-value distill is future.")
    parser.add_argument("--distill-value-feat-coef", "--distill_value_feat_coef", dest="distill_value_feat_coef",
                        type=float, default=None,
                        help="FITNETS VALUE-FEATURE distillation weight (gen3_exploiter_value_feat_distill_v1): "
                             "match the teacher's INTERMEDIATE 128-dim value-CLS pool (the hint layer) instead of "
                             "the collapsed scalar V — 1−cos(value_pooled_student, value_pooled_teacher) on the "
                             "teacher-team states, so the trunk inherits the teacher's per-team value STRUCTURE "
                             "(scalar value-distill CRYSTALLIZES the critic — value_cls rank DROPS). Default 0.0 = "
                             "OFF (byte-identical; no teacher value_pooled read). Requires --distill-coef > 0. "
                             "Training-only, inherited on resume. Composes with / is an A/B alternative to "
                             "--distill-value-coef — watch distill/value_feat_dist (the cosine DISTANCE 1-cos, so "
                             "LOWER = better aligned; the legacy alias distill/value_feat_cos holds the same "
                             "value and reads as its own opposite) fall + the value_cls rank probe.")
    parser.add_argument("--distill-team-bias", "--distill_team_bias", dest="distill_team_bias",
                        type=float, default=0.4,
                        help="Fraction of trainee episodes biased to the teacher's team (rest = pool "
                             "rehearsal). Default 0.4. Only used when --distill-coef > 0.")
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
    parser.add_argument("--pg-coef", "--pg_coef", dest="pg_coef",
                        type=float, default=None,
                        help="POLICY-GRADIENT term weight (gen3_pg_coef_v1): multiplies ONLY the "
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

    # --- Subprocess eval ---
    parser.add_argument("--eval-workers", "--eval_workers", dest="eval_workers", type=int, default=5,
                        help="Number of parallel eval-worker subprocesses per cycle (default 5 for bot "
                             "eval; self-play doubles this to 10). Workers work-steal opponents from a "
                             "shared pool, so uneven per-opponent cost self-balances. Capped at the "
                             "opponent count.")
    parser.add_argument("--eval-device", "--eval_device", dest="eval_device", type=str, default="cpu",
                        help="Device for the eval-worker subprocess inference (default cpu, to decouple from the training GPU).")
    parser.add_argument("--eval-concurrency-per-worker", "--eval_concurrency_per_worker",
                        dest="eval_concurrency_per_worker", type=int, default=_EVAL_SUBPROCESS_CONCURRENCY,
                        help="Battles each eval worker overlaps at once within its claimed opponent (default 1 = "
                             "sequential). Single-thread asyncio latency-hiding (not multi-core): overlaps the "
                             "bridge/server I/O wait with other battles' forwards. A single-core bridge benchmark "
                             "measured ~2x decisions/sec at 3 on spare cores (less under live training contention); "
                             "the plateau is ~3. Cross-opponent parallelism is still --eval-workers.")
    parser.add_argument("--eval-shard-games", "--eval_shard_games",
                        dest="eval_shard_games", type=int, default=EVAL_SHARD_GAMES,
                        help="Games per work-steal shard unit (battle-level work-stealing, default 25 → ~4 shards "
                             "per opponent). Each opponent's eval games split into chunks any idle worker can drain, "
                             "so one straggler no longer pins a whole opponent on a single worker — the long tail "
                             "collapses to one shard. Smaller = finer tail collapse but more player builds / (on "
                             "websocket) more connection churn; the in-process bridge (--use-bridge, the default) is "
                             "preferred for fine shards. >= the per-opponent game count disables sharding (one shard "
                             "per opponent = the original opponent-level behaviour).")
    parser.add_argument("--bait-bot-share", "--bait_bot_share", dest="bait_bot_share",
                        type=float, default=0.0,
                        help="Add the scripted BaitBot to the TRAINING opponent roster with this "
                             "share of the sampling mass (default 0.0 = ABSENT, byte-identical). "
                             "BaitBot pivots into an immune bench mon with probability "
                             "--bait-bot-p; it exists to put PUNISHMENT FREQUENCY on the bait habit "
                             "on a controlled dial (the habit is exploration starvation at a "
                             "saturated action, so punishment frequency is the one signal it cannot "
                             "seal off). The weight is sized from the ACTUAL sum of the other "
                             "roster weights, so the declared share stays exact even when "
                             "--bot-weights re-weights the heuristics. Keep it a MINORITY share to "
                             "avoid script-sniping.")
    parser.add_argument("--bait-bot-p", "--bait_bot_p", dest="bait_bot_p", type=float, default=0.6,
                        help="BaitBot's pivot probability when a bait is available (default 0.6). "
                             "The gate's held-out generalization read uses a DIFFERENT value, so an "
                             "arm cannot pass by memorising this one. Ignored unless "
                             "--bait-bot-share > 0.")
    parser.add_argument("--eval-freq", "--eval_freq", dest="eval_freq", type=int, default=None,
                        help="Steps between eval cycles (default None = EVAL_FREQ_STEPS, 2,000,000 — "
                             "byte-identical to every pre-existing command). Lower it for SHORT arms: a "
                             "3M exploiter-gate fork at the 2M default gets 1-2 cycles, which cannot meet a "
                             ">=4-cycle reading discipline; --eval-freq 750000 gives 4. Applies to BOTH the "
                             "per-opponent bot eval and the self-play eval, so their cadences cannot drift. "
                             "Eval is non-blocking and skips a cycle while the previous one is still running, "
                             "so a too-small value self-throttles rather than stalling training.")
    parser.add_argument("--eval-games", "--eval_games", dest="eval_games", type=int, default=None,
                        help="Games per OPPONENT per eval cycle (default: the module EVAL_GAMES, 100). "
                             "Per-cell 95%% CI: n=100 -> +/-0.098, n=200 -> +/-0.069 — raise for tighter "
                             "sentinel/promotion reads at proportionally more eval compute (work-stolen "
                             "across --eval-workers, off the training path). Shards per opponent = "
                             "eval-games / --eval-shard-games.")
    parser.add_argument("--snapshot-ladder-games", "--snapshot_ladder_games",
                        dest="snapshot_ladder_games", type=int, default=100,
                        help="Frozen-snapshot ELO ladder: games per pair for the per-promotion "
                             "round-robin tax (0 = disable). On each promotion a DETACHED bridge "
                             "subprocess plays the new frozen snapshot vs the current pool and "
                             "appends to <run>/snapshot_ladder/games.jsonl (measured once, kept "
                             "forever) — a dense, high-resolution internal ladder the saturated "
                             "bots can't provide. Off the training path.")
    parser.add_argument("--keep-eval-snapshots", "--keep_eval_snapshots", dest="keep_eval_snapshots",
                        type=int, default=10,
                        help="Retain the N most-recent eval weight snapshots in eval_traces/step_<N>/snapshot.zip "
                             "so the prober can reload the bit-exact model that produced a cycle's traces "
                             "(~27MB each; default 10 ≈ 270MB). 0 only writes the identity manifest; the prober "
                             "then falls back to the nearest persisted checkpoint.")
    parser.add_argument("--keep-eval-trace-steps", "--keep_eval_trace_steps", dest="keep_eval_trace_steps",
                        type=int, default=20,
                        help="The trainer grooms the forensic traces it writes: after each eval cycle it "
                             "keeps only the N most-recent eval step dirs under eval_traces/ (0 = keep all). "
                             "`python -m main.prober.groom` is the manual fallback for finished runs.")
    parser.add_argument("--keep-stalls", "--keep_stalls", dest="keep_stalls", type=int, default=50,
                        help="Bound the run's stalls/ dir: each eval cycle keep only the N most-recent "
                             "stall_*.html replays (0 = keep all). `python -m agents.training.artifact_retention` "
                             "is the manual fallback / cross-run sweep.")
    parser.add_argument("--keep-crashes", "--keep_crashes", dest="keep_crashes", type=int, default=10,
                        help="Bound the run's crashes/ dir: each eval cycle keep only the N most-recent "
                             "launcher restart_err_*.txt files (0 = keep all).")
    parser.add_argument("--self-play", action=BoolFlag, default=False, help="Enable self-play snapshot pool as training opponents")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Pool directory (default: <run_dir>/snapshots)")
    parser.add_argument("--promote-threshold", type=float, default=None,
                        help="Win rate vs. pool to trigger snapshot promotion. Default 0.65 with "
                             "stochastic sentinels; auto-lowered to 0.55 under --eval-sentinel-greedy "
                             "(greedy-vs-greedy removes the temperature handicap, so a genuinely-ahead "
                             "trainee wins the pool by a smaller margin — 0.65 would freeze the pool). "
                             "An explicit value always wins.")
    parser.add_argument("--eval-sentinel-greedy", "--eval_sentinel_greedy", dest="eval_sentinel_greedy",
                        action=BoolFlag, default=False,
                        help="Eval the self-play pool sentinels GREEDY (argmax) instead of stochastic. "
                             "Removes the greedy-trainee-vs-stochastic-sentinel handicap so win_rate_vs_pool "
                             "/ snapshot ELO reflect real best-vs-best skill (≈50%% vs a recent self, ramping "
                             "with sentinel age) instead of a flat temperature offset. Eval-only — TRAINING "
                             "opponents stay stochastic. Metric discontinuity vs prior cycles; pair with the "
                             "auto-lowered --promote-threshold (0.55).")
    parser.add_argument("--self-play-temp", type=float, default=1.0,
                        help="Sampling temperature for self-play TRAINING opponents (they sample, "
                             "not argmax, so the learner faces the policy's full action distribution). "
                             "1.0 = the policy's own distribution; >1 flatter/more random; lower → toward "
                             "greedy. Eval opponents stay deterministic regardless.")
    # ── Bot-mix curriculum (#2): keep the coverage-punishing bots in the TRAINING mix ──
    parser.add_argument("--bot-weights", "--bot_weights", dest="bot_weights", type=str, default=None,
                        help="Bias the per-episode HEURISTIC opponent pick toward chosen archetypes, "
                             "e.g. 'aggressive_v2=3,heuristic2=3'. Unlisted bots default to weight 1.0. "
                             "Names: heuristic, heuristic2, staller, staller_v2, aggressive, aggressive_v2, "
                             "setup_sweep, setup_sweep_v2. Omitted → uniform (current behavior). Only biases "
                             "WHICH heuristic an episode draws; the pool-vs-heuristic fraction is unaffected.")
    parser.add_argument("--heuristic-floor", "--heuristic_floor", dest="heuristic_floor",
                        type=float, default=None,
                        help="Minimum fraction of training episodes vs real bots once self-play saturates "
                             f"(default {HEURISTIC_FLOOR:g}). Raise it (e.g. 0.25) to keep a bigger permanent "
                             "bot slice so the coverage blindspot keeps getting exercised under self-play.")
    parser.add_argument("--self-play-start-wr", "--self_play_start_wr", dest="self_play_start_wr",
                        type=float, default=None,
                        help=f"win_rate_vs_bots at which self-play begins to ramp in (default {SELF_PLAY_START:g}).")
    parser.add_argument("--self-play-full-wr", "--self_play_full_wr", dest="self_play_full_wr",
                        type=float, default=None,
                        help=f"win_rate_vs_bots at which self-play reaches the floor (default {SELF_PLAY_FULL:g}); "
                             "raise it to ramp slower / stay bot-heavier for longer.")
    # ── PFSP / league-lite (prioritized fictitious self-play) — both OFF by default (byte-identical) ──
    parser.add_argument("--pfsp-scale", "--pfsp_scale", dest="pfsp_scale", type=float, default=0.0,
                        help="PFSP hardness weighting for self-play pool sampling (default 0.0 = off, "
                             "pure recency). >0 oversamples the pool selves the trainee is LOSING to "
                             "(weight ×(1 + pfsp_scale·(1−win_rate))) while never starving the ones it "
                             "beats — turns the recency window into a prioritised curriculum. The live "
                             "per-snapshot win-rates are measured at each self-play eval (EMA-smoothed) "
                             "and pushed to the training envs. Try 1.0–2.0. Pairs with --pool-spread so "
                             "PFSP has a diverse ladder of selves, not a recent-selves echo chamber.")
    parser.add_argument("--n-sentinels", "--n_sentinels", dest="n_sentinels", type=int, default=5,
                        help="Number of evenly-spaced pool snapshots eval'd as sentinels each self-play "
                             "cycle (default 5). Each gets a FRESH win-rate, which is what --pfsp-scale "
                             "weights the pool by — so a higher count re-prioritises MORE of the pool per "
                             "cycle (cuts the 'only ~¼ of the pool re-measured' staleness on a deep pool). "
                             "Cost: each extra sentinel is +100 games/cycle, work-stolen by the doubled "
                             "eval pool; eval is non-blocking + skip-while-running so it self-throttles. "
                             "Pairs with a larger --max-snapshots. Training-only (not version-locked).")
    parser.add_argument("--pool-spread", "--pool_spread", dest="pool_spread",
                        action=BoolFlag, default=False,
                        help="Self-play pool retention: keep a temporally-DIVERSE ladder (newest + "
                             "oldest + an even interior spread) instead of the oldest-evicted sliding "
                             "window, so PFSP (--pfsp-scale) has a real range of past selves to "
                             "up-weight. Default off = the legacy sliding window (byte-identical).")
    # ── Team-side PFSP: variance-weighted TEAM sampling by self-play win-rate (OFF by default) ──
    parser.add_argument("--team-pfsp", "--team_pfsp", dest="team_pfsp",
                        choices=["off", "measure", "var", "onesided"], default="off",
                        help="Per-team self-play win-rate tracking for the trainee's pool teams (default "
                             "off = uniform random.choice, byte-identical). 'measure' TRACKS + persists "
                             "the per-team self-play win-rate to <run>/team_winrates.json (the offline "
                             "'which team is the generalist weakest on → next exploiter target' artifact) "
                             "WITHOUT biasing sampling. 'var' additionally weights each pool team by floor "
                             "+ p*(1-p) (p = the win-rate EMA, seed 0.5), capped at --team-pfsp-cap x the "
                             "uniform share — so the trainee drills the teams it wins ~half the time (max "
                             "variance) and stops over-sampling the ones it crushes / always loses. "
                             "'onesided' keeps the LOSING side at MAX weight instead — w(p)=0.25 for p<0.5, "
                             "else p*(1-p) (continuous at 0.5): every sub-50%% team stays maximally sampled "
                             "and only mastery retires a team (under the z_arch/FiLM conditioning "
                             "hypothesis the weak tail is the learnable headroom, so 'truly lost' is the "
                             "claim under test, not a sampling prior). Measured on SELF-PLAY pool battles "
                             "only (bots excluded). Training-only, NOT version-locked.")
    parser.add_argument("--team-pfsp-cap", "--team_pfsp_cap", dest="team_pfsp_cap",
                        type=float, default=3.0,
                        help="Over-representation cap for --team-pfsp: no team is sampled more than "
                             "this multiple of the uniform share (weight ≤ cap×mean(raw)). Default 3.0.")
    parser.add_argument("--team-pfsp-floor", "--team_pfsp_floor", dest="team_pfsp_floor",
                        type=float, default=0.05,
                        help="Weight floor for --team-pfsp (raw_i = floor + p*(1-p)) so a fully-won / "
                             "fully-lost team is never starved to zero. Default 0.05.")
    parser.add_argument("--team-block-episodes", "--team_block_episodes", dest="team_block_episodes",
                        type=int, default=1,
                        help="Hold each drawn TRAINEE team for N consecutive episodes before redrawing "
                             "(1 = off, byte-identical). The per-team gradient-density counter to the "
                             "measured FiLM sample starvation (film/noise_scale ~8-9x the batch): at "
                             "~64 (~one rollout of episodes) per-update per-team density rises ~15x AND "
                             "blocks span an update boundary, so an env replays its team right after "
                             "that team's gradient landed (the exploiter-style learn-and-retest loop). "
                             "Composes with --team-pfsp (weights apply at each redraw; outcomes "
                             "attribute to the blocked team). Trainee side only; training-only, NOT "
                             "version-locked, resume-forwarded.")
    parser.add_argument("--team-wr-tracking", "--team_wr_tracking", dest="team_wr_tracking",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Track a running per-team win rate for the TRAINEE's piloted teams "
                             "(keyed by team_sha, stratified by opponent class): sparse TB summaries "
                             "(teams/wr_top_k, teams/wr_bottom_k, teams/n_teams_seen, teams/wr_mean) "
                             "plus a periodic full-table <run>/team_win_rates.json, archetype-joined "
                             "and restart-safe. PURE INSTRUMENTATION — nothing prioritizes on it, and "
                             "nothing should without normalizing against a team-strength baseline "
                             "first (a raw per-team win rate conflates pilot competence with team "
                             "strength). ON by default; --no-team-wr-tracking opts out. Distinct from "
                             "--team-pfsp, which measures self-play POOL battles only in order to "
                             "weight SAMPLING. Training-only, NOT version-locked, resume-forwarded.")
    # ── Stable (cross-run) opponents: load a model from ANOTHER run as a fixed opponent ──
    parser.add_argument("--stable-opponents", "--stable_opponents", dest="stable_opponents",
                        type=str, default=None,
                        help="Foreign model(s) from ANOTHER run to use as fixed eval opponents, "
                             "comma-separated. Simplest form is just the run dir: "
                             "'models/ai_v5_5_popart_N_0607' — the opponent is then labelled by that "
                             "dir name (ai_v5_5_popart_N_0607). Optional per-entry suffixes: "
                             "'@<step>' picks a specific checkpoint (default: best_model); "
                             "':<name>' renames it. (Per-opponent weights are NOT supported yet — "
                             "they only matter for the training mix, which is Stage 2.) Each model "
                             "must share this run's arch_signature (= observation layout) — a "
                             "mismatch is a startup FATAL surfaced to the TUI. Default None (off).")
    parser.add_argument("--stable-opponent-temp", "--stable_opponent_temp", dest="stable_opponent_temp",
                        type=float, default=1.0,
                        help="TRAINING-mix play temperature for stable opponents (default 1.0 = the "
                             "policy's own distribution). Stochastic (not greedy) so a fixed opponent "
                             "is a moving target — harder to over-exploit. (In EVAL they always play "
                             "greedy/temp-0 for a clean win-rate yardstick.)")
    parser.add_argument("--stable-opponent-mastered-wr", "--stable_opponent_mastered_wr",
                        dest="stable_opponent_mastered_wr", type=float, default=0.80,
                        help="Win rate at which a stable opponent is considered MASTERED and moves "
                             "from the challenge bucket (played alongside the self-play pool) to the "
                             "coverage floor (played alongside the bots) — it 'becomes another bot'. "
                             "Default 0.80. One-way per run. Only active under --self-play.")
    parser.add_argument("--stable-opponent-selfplay-share", "--stable_opponent_selfplay_share",
                        dest="stable_opponent_selfplay_share", type=float,
                        default=STABLE_CHALLENGE_SHARE,
                        help="Fraction of SELF-PLAY (challenge) episodes spent vs stable opponents — "
                             "the rest go to the self-play pool. Caps how much a fixed opponent "
                             "occupies training so a single one can't dominate; multiple un-mastered "
                             f"stable opponents SHARE this slice. Default {STABLE_CHALLENGE_SHARE:g}. "
                             "Only active under --self-play.")
    parser.add_argument("--stable-opponent-pfsp", "--stable_opponent_pfsp",
                        dest="stable_opponent_pfsp", action="store_true",
                        help="DYNAMIC stable-opponent selection: within the capped stable challenge "
                             "slice, pick WEIGHTED by how much the trainee is LOSING to each "
                             "(1 - win_rate) instead of uniformly — spend the exploiter budget on the "
                             "axis it's failing worst, and let each fade as it's mastered. The TOTAL "
                             "pool-vs-stable share is unchanged. Training-only; OFF = uniform "
                             "(byte-identical). Pairs with a raised --stable-opponent-selfplay-share.")
    parser.add_argument("--exploiter", dest="exploiter", type=str, default=None,
                        help="EXPLOITER MODE: train against ONE fixed foreign model as the SOLE "
                             "opponent every episode — the league 'exploiter' role (learn to beat a "
                             "specific target, e.g. the current main agent). Takes a run dir / "
                             "checkpoint spec exactly like --stable-opponents (e.g. "
                             "'models/ai_v6_13_outgoing_dmg_0620'), must share this run's "
                             "arch_signature (startup FATAL otherwise). This is a clean opponent-mix "
                             "front-end: it needs NO --self-play / --stable-opponents / share "
                             "fiddling — just point it at the target. Mutually exclusive with "
                             "--self-play. Recommended: init the exploiter from a strong checkpoint "
                             "(--model <target's checkpoint>) so it has a baseline to exploit from. "
                             "Default None (off).")
    parser.add_argument("--warmstart-consensus", "--warmstart_consensus", dest="warmstart_consensus",
                        type=str, default=None,
                        help="EXPLOITER MODE (requires --exploiter): before training, build a competent, "
                             "archetype-NEUTRAL warm start by disagreement-gated CONSENSUS distillation of "
                             "N mature teacher exploiters (comma-separated run-dirs) into --model (the "
                             "generalist init), then init the exploiter from it. The BC target is SHARP "
                             "where the teachers AGREE (universal decisions inherited) and FLAT where they "
                             "DISAGREE (archetype forks left high-entropy → the new exploiter specializes "
                             "FREELY, unbiased). Built ONCE into <run>/warmstart/ (idempotent across "
                             "launcher restarts; skipped once a training checkpoint exists). Deliberately "
                             "NOT valid for generalist/self-play runs (whose job is the opposite — absorb "
                             "divergence via --distill-teacher). See agents.training.warmstart. Default off.")
    parser.add_argument("--warmstart-battles", dest="warmstart_battles", type=int, default=200,
                        help="On-policy battles to collect for the --warmstart-consensus BC dataset (200).")
    parser.add_argument("--warmstart-bc-steps", dest="warmstart_bc_steps", type=int, default=4000,
                        help="BC gradient steps for --warmstart-consensus (early-stops on gated-KL; 4000).")
    parser.add_argument("--exploiter-keep-bots", dest="exploiter_keep_bots", action="store_true",
                        help="EXPLOITER MODE (requires --exploiter): mix the heuristic bots BACK IN "
                             "alongside the exploiter target instead of playing the target as the sole "
                             "opponent. Per episode, the target is faced with prob "
                             "(1 - --exploiter-bot-fraction), else a random floor/heuristic bot. Lets a "
                             "from-scratch specialist keep a bot floor while it learns to beat one strong "
                             "target. Off (default) = the target is the sole opponent (byte-identical).")
    parser.add_argument("--exploiter-bot-fraction", dest="exploiter_bot_fraction", type=float,
                        default=0.5,
                        help="Under --exploiter-keep-bots, the per-episode probability of facing a "
                             "heuristic bot instead of the exploiter target (default 0.5). The exploiter "
                             "target is faced with the complementary probability (1 - this).")
    parser.add_argument("--exploiter-temp-start", dest="exploiter_temp_start", type=float, default=None,
                        help="EXPLOITER MODE (requires --exploiter): ANNEAL the target opponent's sampling "
                             "temperature over training — a difficulty curriculum via opponent STOCHASTICITY. "
                             "Setting this (a positive float, e.g. 2.0) starts the target at this temperature "
                             "(flatter logits → noisier/weaker play, so a from-scratch trainee can win some "
                             "games and get a learning signal) and linearly anneals it to --exploiter-temp-end "
                             "over --exploiter-temp-anneal-frac of training, held after. None (default) = OFF: "
                             "the target plays at --stable-opponent-temp the whole run (byte-identical). "
                             "Training-only (not version-locked; forwarded verbatim on resume, where the anneal "
                             "continues from the resumed step).")
    parser.add_argument("--exploiter-temp-end", dest="exploiter_temp_end", type=float, default=1.0,
                        help="EXPLOITER MODE: the target opponent's temperature at the END of the anneal window "
                             "(default 1.0 = the policy's own distribution, i.e. the target's true strength as a "
                             "stochastic training opponent). Only used when --exploiter-temp-start is set. Set "
                             "below 1.0 to push the target toward greedy (harder) by the end.")
    parser.add_argument("--exploiter-temp-anneal-frac", dest="exploiter_temp_anneal_frac", type=float,
                        default=0.2,
                        help="EXPLOITER MODE: fraction of total --steps over which to linearly anneal the target "
                             "temperature from --exploiter-temp-start to --exploiter-temp-end (default 0.2 = the "
                             "first 20%% of training; held at the end temp after). 0 = constant at "
                             "--exploiter-temp-start (a fixed hotter opponent, no anneal). Only used in the FIXED "
                             "temp mode (--exploiter-temp-mode fixed).")
    parser.add_argument("--exploiter-temp-mode", dest="exploiter_temp_mode",
                        choices=["fixed", "ratchet"], default="fixed",
                        help="EXPLOITER MODE (with --exploiter-temp-start): how the target temperature is "
                             "controlled. 'fixed' (default) = the linear time schedule (--exploiter-temp-anneal-frac). "
                             "'ratchet' = DYNAMIC win-rate-driven: start at --exploiter-temp-start (set it HIGH, e.g. "
                             "5.0, so early games are trivially winnable) and ratchet the temperature DOWN toward "
                             "--exploiter-temp-end only when the trainee's measured TRAINING win-rate vs the target "
                             "clears --exploiter-temp-ratchet-wr — a ONE-WAY auto-curriculum that tracks the trainee's "
                             "competence frontier (never weakens the target, so no comfort-trap). Resume-safe (the "
                             "ratcheted temp is persisted to <run>/exploiter_temp_state.json).")
    parser.add_argument("--exploiter-temp-ratchet-wr", dest="exploiter_temp_ratchet_wr", type=float,
                        default=0.55,
                        help="RATCHET mode: the trainee TRAINING-WR vs the target at which the temperature ratchets "
                             "DOWN (harder). Default 0.55 (keeps play near the ~0.5 max-advantage-signal zone). "
                             "Measured per window of --exploiter-temp-ratchet-games target games.")
    parser.add_argument("--exploiter-temp-ratchet-factor", dest="exploiter_temp_ratchet_factor",
                        type=float, default=0.9,
                        help="RATCHET mode: multiply the temperature by this (<1) on each ratchet (default 0.9 = 10%% "
                             "harder steps). Floored at --exploiter-temp-end.")
    parser.add_argument("--exploiter-temp-ratchet-games", dest="exploiter_temp_ratchet_games",
                        type=int, default=500,
                        help="RATCHET mode: min target-games per decision window before a ratchet check (default 500 "
                             "— the noise guard; larger = smoother/slower).")
    parser.add_argument("--trainee-team", dest="trainee_team", type=str, default=None,
                        help="SPECIALIST MODE: pin the TRAINEE's team pool to the ONE team in this file "
                             "(a Showdown EXPORT string, like data/teams/sample/*.txt), so the agent "
                             "always plays that exact 6-mon team. The OPPONENTS still draw the full "
                             "diverse pool. Use to train a single-team specialist (e.g. --trainee-team "
                             "data/teams/specialist/tss_starmie.txt). Default None = the full trainee "
                             "pool (byte-identical).")
    parser.add_argument("--trainee-teams", dest="trainee_teams", type=str, default=None,
                        help="MULTI-TEAM SPECIALIST MODE: pin the TRAINEE's team pool to the SMALL "
                             "FIXED SET of teams in these files (comma-separated Showdown-export paths), "
                             "sampled UNIFORMLY per episode — a z-near multi-team exploiter (the "
                             "1-vs-3-team A/B). Opponents still draw the full pool. Mutually exclusive "
                             "with --trainee-team; under --exploiter EVERY member must be a sample team. "
                             "Default None.")
    parser.add_argument("--allow-nonsample-trainee", dest="allow_nonsample_trainee", action="store_true",
                        help="RESEARCH override: skip the exploiter vetted-SAMPLE gate so --trainee-team(s) "
                             "may pin NON-sample POOL teams (anchor on a sample, nearest neighbors from all "
                             "719 pool teams → a tighter z-cluster than the 32 samples allow). For FiLM "
                             "capacity / count-vs-diversity studies; NOT for a teacher you'll distil as-is. "
                             "Training-only, not version-locked. Default off (gate enforced).")
    return parser
