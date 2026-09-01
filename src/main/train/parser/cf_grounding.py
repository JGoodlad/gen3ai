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
