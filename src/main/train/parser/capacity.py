"""The `# --- LIVE CAPACITY TELEMETRY (gen3_capacity_telemetry_v1) ---` section.

Lifted VERBATIM out of the old single-file `parser.py` (lines 892-979); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from main.train.parser.base import BoolFlag


def add_capacity_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
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
    # THE OFF-SLICE ANCHOR (gen3_distill_offslice_anchor_v1) — a trust region to the FROZEN fold
    # PARENT on the states no teacher covers, plus the licensing probe's collateral meters, live.
    # A fold's net is teacher content MINUS overshoot damage off the taught slice, and the
    # 2026-08-31 probe measured that damage as a systematic direction, not noise.
    parser.add_argument("--distill-anchor-coef", "--distill_anchor_coef", dest="distill_anchor_coef",
                        type=float, default=None,
                        help="OFF-SLICE TRUST-REGION weight (gen3_distill_offslice_anchor_v1): fold "
                             "coef * KL(pi_parent || pi_student) over the rollout states that are NOT on "
                             "any teacher's pinned teams, against the FROZEN fold parent. Default 0.0 = "
                             "OFF (byte-identical; no parent loaded, no forward). Requires "
                             "--distill-coef > 0 (the slice is the `distill_mask` the teacher term "
                             "already uses). This is a small-coefficient REGULARISER toward the "
                             "starting policy, NOT the R3-SELF self-distillation TARGET that measured "
                             "-9pp at production dose — size it as a FRACTION of --distill-coef. Watch "
                             "distill/collateral_kl fall while distill/teacher_agreement_on_slice holds, "
                             "and grad/distill_anchor_share against grad/distill_share for the dose.")
    parser.add_argument("--distill-anchor-mode", "--distill_anchor_mode", dest="distill_anchor_mode",
                        choices=["off_slice", "all"], default=None,
                        help="WHICH rows the anchor applies to. 'off_slice' (default) = only where "
                             "distill_mask == 0, so the anchor never fights the teacher on the states "
                             "the teacher owns. 'all' = every row; it exists so an arm can TEST whether "
                             "excluding the taught slice is what makes the trust region work rather "
                             "than assuming it. The METERS are unaffected — collateral_kl is always "
                             "the off-slice mean and on_slice_kl always the on-slice one.")
    parser.add_argument("--distill-anchor-monitor", "--distill_anchor_monitor",
                        dest="distill_anchor_monitor", action="store_true", default=False,
                        help="Attach the frozen fold parent and emit every distill/collateral_kl "
                             "meter even at --distill-anchor-coef 0 — the PURE-INSTRUMENT arm: no "
                             "loss term, no parameter changed, one frozen no_grad forward per "
                             "minibatch. Use it to measure a fold's off-slice damage before deciding "
                             "whether to penalise it.")
    parser.add_argument("--distill-anchor-parent", "--distill_anchor_parent",
                        dest="distill_anchor_parent", type=str, default=None,
                        help="Explicitly pin the anchor's frozen parent checkpoint (a run dir or "
                             ".zip). Normally UNSET: the parent is re-resolved on every launch from "
                             "<run>/metadata.json's immutable `original_command` --model, falling back "
                             "to this process's --model on a fork's first launch — because an "
                             "idempotent fork's --model is swapped to the fork's OWN latest checkpoint "
                             "on every restart, and anchoring to that would let the trust region drift "
                             "along with the student. Pass this only to override that resolution.")
    parser.add_argument("--distill-team-bias", "--distill_team_bias", dest="distill_team_bias",
                        type=float, default=None,
                        help="Fraction of trainee episodes biased to the teacher TEAMS (rest = pool "
                             "rehearsal). Default 0.4. Applies whenever --distill-teacher is given — "
                             "INCLUDING --distill-coef 0, which is the CONTROL-ARM shape: the loss is "
                             "off but the team distribution is held constant against the treatment arm. "
                             "Requires --distill-teacher (there is no team to bias toward without one). "
                             "The argparse default is None so an explicitly-typed value is "
                             "distinguishable from the unset flag; it resolves to 0.4 in resolve_config.")
