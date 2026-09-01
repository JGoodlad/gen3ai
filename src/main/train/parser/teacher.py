"""The two teacher sections — `# --- SEARCH-AS-TEACHER ---` (9 lines) and
`# --- THE WIN-PROB ONE-PLY TEACHER ---` — merged, being neighbours on the same
subject.

Lifted VERBATIM out of the old single-file `parser.py` (lines 650-723); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from agents.training.teacher.modes import TEACHER_MODES


def add_teacher_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- SEARCH-AS-TEACHER (offline ExIt plateau-breaker; designs/ai_v6/design_search_teacher.md) ---
    # All TRAINING-only (no version bump; coef 0 / flag absent = byte-identical). The coefs are
    # _resolve'd (flagless-resume-inherited); the operational knobs are forwarded by the launcher.
    parser.add_argument("--search-teacher", "--search_teacher", dest="search_teacher",
                        action="store_true",
                        help="Enable the search-teacher: each cycle, search + rollout-confirm the worst "
                             "falsify-flagged loss craters (EXACT reloaded opponent), CI-gate strictly-"
                             "better corrections, and distil them into the policy via an AWR aux loss. "
                             "Non-blocking (subprocess workers). Recommended at PLATEAU. Re-pass on resume.")
    # --- THE WIN-PROB ONE-PLY TEACHER (gen3_winprob_oneply_teacher_v1; ai_v12 routes 2+3,
    # designs/ai_v12/design_winprob_behavior_coupling.md). A new SUPPLY of corrections on the
    # existing ExIt seam, not a new pipeline: same Correction record, same ring buffer, same AWR
    # loss at --search-teacher-coef. All three are OPERATIONAL (like --teacher-search-budget):
    # re-pass on resume. `crater` is the default and is byte-identical to the pre-flag behaviour.
    parser.add_argument("--search-teacher-mode", "--search_teacher_mode", dest="search_teacher_mode",
                        choices=TEACHER_MODES, default="crater",
                        help="WHICH teacher produces the corrections. 'crater' (default, the "
                             "historical behaviour) asks 'where did the model lose the most value, "
                             "and is there a strictly better LINE' -- value craters, falsify-gated to "
                             "reducible mistakes, then a depth-2 beam over the CRITIC. "
                             "'winprob_oneply' asks a different question: 'at a decision the model's "
                             "own win-prob head calls CONTESTED, does a one-ply successor read prefer "
                             "another action by a margin that survives paired-rollout confirmation?' "
                             "-- the head's knowledge turned into a training target (it is a BAROMETER "
                             "otherwise: --win-prob-mode shaping exerts no force on behavior). "
                             "Requires --search-teacher; 'winprob_oneply' also needs --win-prob-mode "
                             "read_only|shaping (the ranking IS the head). OPERATIONAL -- re-pass on "
                             "resume, like --search-teacher itself.")
    parser.add_argument("--winprob-teacher-band", "--winprob_teacher_band",
                        dest="winprob_teacher_band", type=float, default=0.15,
                        help="CONTESTED-gate half-width for --search-teacher-mode winprob_oneply: a "
                             "decision qualifies when it has >= 2 legal actions AND "
                             "|P(win|s) - 0.5| < band. Default 0.15 -- probe H's measured operating "
                             "point, shared with the defensive searcher's own gate (the same code "
                             "decides both, so the two notions of 'contested' cannot drift apart). "
                             "Model-FREE: read off the trace's recorded win_probs/action_mask, so "
                             "widening it costs selection time, not search time.")
    parser.add_argument("--winprob-teacher-margin", "--winprob_teacher_margin",
                        dest="winprob_teacher_margin", type=float, default=0.02,
                        help="One-ply delta-P(win) a preferred action must clear before it is even "
                             "sent for confirmation, under --search-teacher-mode winprob_oneply. "
                             "Distinct from --search-teacher's margin_min, which gates the CRATER "
                             "mode's Wilson bound in win-RATE units. Default 0.02 is a WORKING "
                             "value, not the measured floor: defensive-search iter 2 put the leaf's "
                             "residual DIFFERENTIAL bias at RMS 0.122, larger than most true gaps, "
                             "and running at that floor collapses target volume by roughly an order "
                             "of magnitude (the design doc's E4 arm is the one that asks which is "
                             "right). The real defence is --teacher-confirm-rollouts: statistical "
                             "separation of a BIASED reader is not correctness, and a distillation "
                             "target has no invariance shield to fall back on.")
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
