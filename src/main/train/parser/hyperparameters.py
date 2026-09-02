"""The `# --- Hyperparameter Flags (Optimized for GPU) ---` section: the PPO/optimizer
knobs and the vec-env shape.

Lifted VERBATIM out of the old single-file `parser.py` (lines 179-259); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse


def add_hyperparameter_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
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
    parser.add_argument("--fork-lr", "--fork_lr", dest="fork_lr", type=float, default=None,
                        help="RESUME-ONLY: pin the step size of a FORK. On a resume the optimizer's "
                             "saved LR wins and --lr is INERT, so a distillation fold silently "
                             "inherits whatever rate the parent's KL controller had annealed to "
                             "(measured: 5.8e-5 / 2.8e-5 / 1.0e-4 across three folds that were all "
                             "launched with the same --lr). This sets the optimizer LR *and* "
                             "model.lr_schedule at load, and seeds the KL controller from it, so the "
                             "fold's DOSE (lr x n_epochs / (batch_size*grad_accum_steps)) is a chosen "
                             "quantity rather than an inherited one. Applied ONLY on a genuine fork — "
                             "a checkpoint from OUTSIDE the target run dir; a launcher PERIODIC "
                             "RESTART (same run, same argv, its own checkpoint) never re-applies it, "
                             "so the controller keeps adapting from where it was. Refused on a fresh "
                             "run (use --lr there). Recorded in metadata.json's `dose` block.")
    parser.add_argument("--fork-lr-freeze", "--fork_lr_freeze", dest="fork_lr_freeze",
                        action="store_true",
                        help="With --fork-lr: also DISABLE the KL-driven LR controller for this run, "
                             "so the LR stays at --fork-lr exactly (the [--min-lr, --max-lr] bounds "
                             "still apply to the pinned value, and the two-phase cosine is held too). "
                             "A fold experiment wants a CONSTANT, recordable step size — an adapting "
                             "LR makes the dose a per-rollout variable nothing records. Unlike the pin "
                             "itself the freeze is a property of the RUN and DOES survive every "
                             "periodic restart.")
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
