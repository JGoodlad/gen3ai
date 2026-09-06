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
                             "'TEACHER:TEAM' groups (KL(π_teacher ‖ π_student) on that teacher's team states). "
                             "TEACHER = a RUN SPEC '<run|zip>[@<step>]' ('@<step>' pins "
                             "<run>/checkpoints/checkpoint_<step>_steps.zip instead of best_model), TEAM = its "
                             "Showdown team file, or '*'/'auto' for exactly the teams that run recorded training "
                             "on. ';' separates teachers, ',' separates one teacher's teams, e.g. "
                             "'models/expA:data/teams/specialist/a.txt;models/expB@26267760:*'. "
                             "The colon pairing binds each teacher to its team — no misalignment possible; a "
                             "teacher that resolves to zero teams is REFUSED at launch, never run silently.")
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
                        choices=["off_slice", "all", "grad_project"], default=None,
                        help="WHICH rows the anchor applies to. 'off_slice' (default) = only where "
                             "distill_mask == 0, so the anchor never fights the teacher on the states "
                             "the teacher owns. 'all' = every row; it exists so an arm can TEST whether "
                             "excluding the taught slice is what makes the trust region work rather "
                             "than assuming it. The METERS are unaffected — collateral_kl is always "
                             "the off-slice mean and on_slice_kl always the on-slice one. "
                             "'grad_project' (gen3_distill_grad_project_v1) is SOURCE-SEPARATED and "
                             "is a different mechanism, not a third row set: every optimizer step it "
                             "projects the DISTILL gradient off the subspace spanned by "
                             "grad log pi(argmax) at --distill-anchor-proj-samples sampled OFF-SLICE "
                             "states, and steps with g_ppo + P_perp g_distill — PPO's gradient is "
                             "never read or scaled. The point is the one thing an OUTPUT anchor "
                             "cannot do: a fold's GIFT (an off-slice habit change orthogonal to the "
                             "taught content, PPO-driven, +5-10pp untaught) and its LEAK (the taught "
                             "content arriving on untaught boards, -5.7pp) are the same displacement "
                             "at the output and DIFFERENT SOURCES at the update. Its own output half "
                             "is 'off_slice', so --distill-anchor-coef 0 is projection-only and a "
                             "positive coefficient COMPOSES an output anchor on top (the projection "
                             "is per-step first-order; the output anchor catches the accumulation). "
                             "The frozen parent is attached in this mode even at coefficient 0, so "
                             "distill/collateral_kl_vs_parent reads the effect live. Costs the m "
                             "constraint backwards per micro-batch — see distill/proj_ms.")
    parser.add_argument("--distill-anchor-proj-samples", "--distill_anchor_proj_samples",
                        dest="distill_anchor_proj_samples", type=int, default=None,
                        help="m for --distill-anchor-mode grad_project (default 16): how many "
                             "OFF-SLICE rows of each micro-batch constrain that step's distill "
                             "gradient. Each costs one backward over an m-ROW graph (the obs are "
                             "sliced before the forward, so this is ~m/batch_size of a full "
                             "backward) and m x |params| floats of peak memory, both freed at the "
                             "end of the micro-batch. Higher m constrains more of the off-slice "
                             "behaviour manifold per step; watch distill/proj_rank, which is the "
                             "number of directions that SURVIVED Gram-Schmidt and is what the "
                             "projection actually removed along — if it saturates well below m the "
                             "sampled states are asking for the same change and more samples buy "
                             "nothing. Ignored (and refused) outside grad_project.")
    parser.add_argument("--distill-anchor-monitor", "--distill_anchor_monitor",
                        dest="distill_anchor_monitor", action=BoolFlag, default=None,
                        help="Attach the frozen fold parent and emit every distill/collateral_kl "
                             "meter even at --distill-anchor-coef 0 — the PURE-INSTRUMENT arm: no "
                             "loss term, no parameter changed, one frozen no_grad forward per "
                             "minibatch. Use it to measure a fold's off-slice damage before deciding "
                             "whether to penalise it. **ON BY DEFAULT WHENEVER A FOLD IS RUNNING** "
                             "(gen3_distill_instruments_default_v1): --distill-teacher names at "
                             "least one teacher AND --distill-coef > 0 AND no anchor coefficient is "
                             "already attaching the parent. It was opt-in until 2026-09-03, and a "
                             "batch of seven fold arms carried it on three argvs and not the other "
                             "four — which made the pre-registered cross-check unrunnable on the "
                             "arms that mattered, because an ABSENT meter reads like a zero. "
                             "--no-distill-anchor-monitor opts out; the default never turns a "
                             "launch into a FATAL — an unresolvable fold parent WARNS and leaves "
                             "the instrument off (recorded as such in metadata\'s cli_args).")
    parser.add_argument("--distill-anchor-ref", "--distill_anchor_ref", dest="distill_anchor_ref",
                        choices=["parent", "ema", "periodic"], default=None,
                        help="WHICH policy the anchor is measured against. 'parent' (DEFAULT, and "
                             "byte-identical to what this feature shipped with) = the FIXED frozen "
                             "fold parent, which is Learning-without-Forgetting's design: PPO's clip "
                             "bounds the per-update RATE against the data-collecting policy and "
                             "re-reads it every rollout, but the anchor bounds the ACCUMULATED "
                             "DISPLACEMENT from the fold start — the quantity the licensing probe "
                             "measured and rev-4's untaught robbery is made of — and that collateral "
                             "is SYSTEMATIC (the same off-slice direction every step), which a "
                             "following reference barely resists. 'ema' = a Polyak average of the "
                             "student (ACER's average-policy trust region), the arm to have if the "
                             "fixed anchor suppresses a GIFT: a fixed reference cannot tell v8's "
                             "+5.4pp off-slice switching change from a robbery — both are "
                             "displacement — while an average lets slow consistent improvement "
                             "through and still taxes fast overshoot. 'periodic' = re-snapshot the "
                             "student every --distill-anchor-refresh-every rollouts. Every mode is "
                             "INITIALISED FROM THE PARENT, so all three coincide at fold start.")
    parser.add_argument("--distill-anchor-ema-tau", "--distill_anchor_ema_tau",
                        dest="distill_anchor_ema_tau", type=float, default=None,
                        help="Polyak coefficient for --distill-anchor-ref ema (default 0.99): "
                             "ref <- tau*ref + (1-tau)*student, once per train() call. The window is "
                             "~1/(1-tau) train() CALLS, and at the production shape one call is one "
                             "rollout = n_envs*n_steps = 48*2048 = 98,304 env steps — so 0.99 is ~100 "
                             "calls ~ 9.8M env steps, 0.9 is ~10 calls ~ 983k, and 0.999 is ~1000 "
                             "calls ~ 98M, i.e. longer than any run here and effectively 'parent'. "
                             "tau=1.0 IS 'parent' (the reference never moves); tau=0.0 makes the "
                             "reference the current student, so the anchor loss goes to ~0.")
    parser.add_argument("--distill-anchor-refresh-every", "--distill_anchor_refresh_every",
                        dest="distill_anchor_refresh_every", type=int, default=None,
                        help="Rollouts between re-snapshots under --distill-anchor-ref periodic "
                             "(default 8; one rollout = one train() call). 0 = never = 'parent', "
                             "collapsed to that mode at construction so nothing downstream carries "
                             "the special case. The snapshot and its refresh counter are PERSISTED "
                             "beside every checkpoint, so a launcher restart does not re-anchor to a "
                             "drifted policy — nor reset the cadence.")
    parser.add_argument("--distill-anchor-parent", "--distill_anchor_parent",
                        dest="distill_anchor_parent", type=str, default=None,
                        help="Explicitly pin the anchor's frozen parent checkpoint (a run dir or "
                             ".zip). Normally UNSET: the parent is re-resolved on every launch from "
                             "<run>/metadata.json's immutable `original_command` --model, falling back "
                             "to this process's --model on a fork's first launch — because an "
                             "idempotent fork's --model is swapped to the fork's OWN latest checkpoint "
                             "on every restart, and anchoring to that would let the trust region drift "
                             "along with the student. Pass this only to override that resolution.")
    # gen3_distill_stop_rule_v1 — THE DUAL. Turn the coefficient above into a CONSTRAINT with a
    # readable budget (PPO-penalty's adaptive beta / MPO's Lagrangian dual). Pure controller +
    # every justification: agents/training/distill_stop_callback.py::AnchorDualAscent.
    parser.add_argument("--distill-anchor-target-kl", "--distill_anchor_target_kl",
                        dest="distill_anchor_target_kl", type=float, default=None,
                        help="DUAL ASCENT on --distill-anchor-coef (default 0.0 = OFF, the "
                             "coefficient is the constant it always was). Every rollout: "
                             "coef <- coef * exp(eta * (kl_ema/target - 1)), clamped into "
                             "[--distill-anchor-coef-min, --distill-anchor-coef-max], where kl_ema "
                             "is an EMA (alpha 0.2, half-life ~3 rollouts) of "
                             "distill/collateral_kl_vs_parent under --distill-anchor-ref parent, "
                             "and of distill/anchor_kl under a MOVING reference (there the anchor "
                             "is DESIGNED not to resist parent-displacement, so a dual budgeted on "
                             "it could never satisfy its constraint and would sit at the max clamp "
                             "forever). This makes the anchor a BUDGET you can read rather than a "
                             "coefficient nobody can tune. Requires --distill-anchor-coef > 0: the "
                             "update is multiplicative, so 0 is a fixed point. The coefficient is "
                             "PERSISTED in the checkpoint sidecar and restored on a launcher "
                             "restart. Watch distill/anchor_coef against distill/anchor_dual_kl_ema.")
    parser.add_argument("--distill-anchor-dual-lr", "--distill_anchor_dual_lr",
                        dest="distill_anchor_dual_lr", type=float, default=None,
                        help="Dual-ascent step eta (default 0.1). The update is an INTEGRATOR, so "
                             "eta alone sets the response timescale: a sustained 2x overshoot moves "
                             "the coefficient +10.5%% per rollout, ~7 rollouts to double. There is "
                             "deliberately NO cooldown (unlike the KL lr ladder, which is bang-bang "
                             "and compounds) — adding one to an integrator inserts dead time, which "
                             "is what causes the oscillation a cooldown is meant to prevent.")
    parser.add_argument("--distill-anchor-coef-min", "--distill_anchor_coef_min",
                        dest="distill_anchor_coef_min", type=float, default=None,
                        help="Lower clamp on the dual-driven anchor coefficient (default 0.0). "
                             "Under a MULTIPLICATIVE update 0.0 is unreachable from above, so the "
                             "default means 'no floor'; set it to pin a minimum trust region.")
    parser.add_argument("--distill-anchor-coef-max", "--distill_anchor_coef_max",
                        dest="distill_anchor_coef_max", type=float, default=None,
                        help="Upper clamp on the dual-driven anchor coefficient (default: 10x "
                             "--distill-anchor-coef). The anchor is documented as a FRACTION of "
                             "--distill-coef and a coefficient at distill scale is R3-SELF, which "
                             "measured -9pp — so an unbounded dual could walk into exactly the "
                             "misuse the feature warns about.")
    parser.add_argument("--distill-team-bias", "--distill_team_bias", dest="distill_team_bias",
                        type=float, default=None,
                        help="Fraction of trainee episodes biased to the teacher TEAMS (rest = pool "
                             "rehearsal). Default 0.4. Applies whenever --distill-teacher is given — "
                             "INCLUDING --distill-coef 0, which is the CONTROL-ARM shape: the loss is "
                             "off but the team distribution is held constant against the treatment arm. "
                             "Requires --distill-teacher (there is no team to bias toward without one). "
                             "The argparse default is None so an explicitly-typed value is "
                             "distinguishable from the unset flag; it resolves to 0.4 in resolve_config.")
