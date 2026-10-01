"""The `# --- COUNTERFACTUAL VALUE GROUNDING ---` section
(gen3_cf_label_plumbing_v1; G3).

Lifted VERBATIM out of the old single-file `parser.py` (lines 724-778); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from main.train.parser.base import BoolFlag


def add_cf_grounding_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
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
    # --- THE LABEL SUPPLY (gen3_supply_guard_v1) — a DECLARED startup resource, guarded in flight.
    # Operational, not recorded as a ModelVersion field: the launcher's own restarts forward the argv
    # verbatim, and a bare flagless resume lands on the fail-closed defaults.
    parser.add_argument("--cf-label-supply", "--cf_label_supply", dest="cf_label_supply",
                        choices=["producer", "external"], default="producer",
                        help="WHO supplies <run_dir>/cf_labels/ when any cf-buffer coefficient "
                             "(--cf-winprob-coef / --cf-evidential-coef / --cf-twin-coef / "
                             "--cf-shadow-coef / --q-winprob-coef / --q-winprob-onpolicy-coef) is "
                             "live. 'producer' (default): the trainer STARTS cf_producer itself at "
                             "startup, as its child (log <run>/cf_producer.log; it dies with the "
                             "trainer) — requires --cf-records. 'external': an operator-run "
                             "producer, verified at startup by its lock <run>/cf_producer.lock; "
                             "none held → FATAL_CONFIG naming the command. Either way a supply "
                             "that delivers nothing in flight is FATAL_SUPPLY (exit 5).")
    parser.add_argument("--cf-producer-args", "--cf_producer_args", dest="cf_producer_args",
                        type=str, default=None,
                        help="Extra cf_producer flags for the SPAWNED producer, one shell-quoted "
                             "string (e.g. \"--rollouts 16 --top-n 4\"). The trainer already adds "
                             "--q-labels when a Q coefficient is live, and --parent-pid.")
    parser.add_argument("--cf-supply-starve-cycles", "--cf_supply_starve_cycles",
                        dest="cf_supply_starve_cycles", type=int, default=5,
                        help="IN-FLIGHT supply floor: once a checkpoint exists, a live cf "
                             "coefficient's label stream must accept at least one row within this "
                             "many consecutive train() cycles (AND --cf-supply-starve-minutes), "
                             "else the run exits FATAL_SUPPLY (5) and the launcher does not "
                             "restart it. Default 5. 0 DISABLES the guard (announced at start).")
    parser.add_argument("--cf-supply-starve-minutes", "--cf_supply_starve_minutes",
                        dest="cf_supply_starve_minutes", type=float, default=30.0,
                        help="The wall-clock half of the in-flight floor (default 30): both it and "
                             "--cf-supply-starve-cycles must be exceeded, so a fast-iterating run "
                             "does not trip before a healthy producer's first labels (~1-2 min of "
                             "compile + anchor) and a slow one is not held hostage by the count.")
