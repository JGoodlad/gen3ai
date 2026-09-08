#!/usr/bin/env bash
#
# marginal_fps.sh — MARGINAL fps for one run, from CHECKPOINT MTIMES only.
#
#     scripts/ops/marginal_fps.sh <run-name|run-dir> [--target N[:label]]... [--no-selfplay]
#     scripts/ops/marginal_fps.sh --help
#
# THE STANDARD METER (owner, 2026-08-23; `designs/ops/TRAINING_RUN_SOP.md` §2). Never the log's
# `fps` line: that is a CUMULATIVE average including compile and startup, so it lags every regime
# change — a fresh run's bots-only warmup held it at ~950 while the true rate had already fallen
# to ~525 as self-play ramped. And never the `T = step / fps` derivative: `fps` is logged as an
# INTEGER, so a 1-unit change moves the derived elapsed time by ~30 s against the ~200 s between
# rows, the marginals explode (6,008 fps observed, physically impossible), and the error GROWS
# with the run — plausible early, rotten later.
#
# With `--checkpoint-every-steps N` the files land every exactly N env-steps, so the wall-clock
# gap between consecutive checkpoints is a rounding-free rate:
#
#     marginal = (steps_N - steps_{N-1}) / (mtime_N - mtime_{N-1})
#
# which excludes startup by construction.
#
# AN fps FIGURE IS NOT INTERPRETABLE WITHOUT `train/selfplay_fraction` BESIDE IT. On the arm this
# was written for, the fraction stepped 0.0 -> 0.9 at the first pool promotion and the rollout
# cadence fell 107 s -> 163 s (919 -> 605 fps) with zero compile in the window. That is the
# REGIME, not a dip, and "promotion-free" is the wrong word for the 953 fps figure — it invites
# the reader to think the cost is a per-promotion tax that amortizes. It does not. So the
# fraction is printed at every artifact, unless --no-selfplay says the run has no such tag.
#
# Promoted 2026-09-07 from the Training Run session's `marginal_fps.sh` (run directory and the
# four ETA targets were literals).
set -u

usage() {
    cat <<'EOF'
marginal_fps.sh — marginal fps for one run, from checkpoint mtimes only.

USAGE
    scripts/ops/marginal_fps.sh <run-name|run-dir> [--target N[:label]]... [--no-selfplay]
    scripts/ops/marginal_fps.sh --help

ARGUMENTS
    <run-name|run-dir>   a bare name is looked up in the run archive (models/); anything
                         with a slash is taken as a directory and must exist.

OPTIONS
    --target N[:label]   print an ETA to step N at the LATEST marginal rate. Repeatable.
                         With no --target the table is printed and no ETA is claimed — an
                         ETA is an extrapolation, not part of the measurement.
    --no-selfplay        skip the train/selfplay_fraction table (a run that never logs it).

OUTPUT
    one row per consecutive checkpoint pair, then the latest marginal, the spread across
    all segments, any ETAs, and the selfplay_fraction at each artifact.

ENVIRONMENT
    GEN3AI_MODELS_DIR  the run archive, if it is not the main checkout's models/
    GEN3AI_PYTHON      interpreter

EXIT
    0  printed        2  bad usage, or the run could not be resolved
EOF
}

case "${1:-}" in
    -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

# shellcheck source=scripts/ops/_common.sh
. "$(dirname "$(readlink -f "$0")")/_common.sh"

RUN_ARG="$1"; shift
TARGETS=(); SELFPLAY=1
while [ $# -gt 0 ]; do
    case "$1" in
        --target)       TARGETS+=("$2"); shift 2 ;;
        --no-selfplay)  SELFPLAY=0; shift ;;
        *) echo "REFUSING: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

R="$(ops_resolve_run "$RUN_ARG")" || exit 2
PY="$(ops_python)"

"$PY" - "$R" "${TARGETS[@]+"${TARGETS[@]}"}" <<'PY'
import sys, os, glob, re, datetime as dt
R = sys.argv[1]
targets = []
for spec in sys.argv[2:]:
    step, _, label = spec.partition(":")
    targets.append((int(step), label or f"{int(step):,}"))
cks = sorted(glob.glob(os.path.join(R, 'checkpoints', 'checkpoint_*_steps.zip')),
             key=lambda p: int(re.search(r'checkpoint_(\d+)_steps', p).group(1)))
if len(cks) < 2:
    print(f"  NOT COMPUTABLE — {len(cks)} checkpoint(s); marginal fps needs 2."); sys.exit(0)
rows = [(int(re.search(r'checkpoint_(\d+)_steps', p).group(1)), os.path.getmtime(p), p) for p in cks]
print(f"  {'from':>12} {'to':>12} {'dsteps':>10} {'dt (s)':>9} {'marginal fps':>13}")
seg = []
for (s0, t0, _), (s1, t1, _) in zip(rows, rows[1:]):
    f = (s1 - s0) / (t1 - t0); seg.append(f)
    print(f"  {s0:>12,} {s1:>12,} {s1-s0:>10,} {t1-t0:>9.0f} {f:>13.1f}")
print(f"\n  latest marginal   {seg[-1]:.1f} fps")
if len(seg) > 1:
    print(f"  all segments      {min(seg):.1f} .. {max(seg):.1f} fps  (n={len(seg)})")
else:
    print("  ONE segment only — a single interval, not a rate estimate with spread.")
last_s, last_t, _ = rows[-1]
if not targets:
    print("  no --target given — no ETA claimed (an ETA is an extrapolation, not the measurement)")
for tgt, name in targets:
    if tgt > last_s:
        eta = dt.datetime.fromtimestamp(last_t + (tgt - last_s) / seg[-1])
        print(f"  ETA {name:<12} {tgt:>12,} -> {eta:%m-%d %H:%M}  (at the LATEST marginal rate; not a duration claim)")
    else:
        print(f"  target {name} ({tgt:,}) is already PASSED (latest {last_s:,}) — no ETA")
PY

[ "$SELFPLAY" -eq 1 ] || exit 0
echo
echo "selfplay_fraction at each artifact (an fps number here is a function of it):"
"$PY" - "$R" <<'PY' 2>/dev/null
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, sys
R = os.path.join(sys.argv[1], 'tb')
if not os.path.isdir(R):
    print("    NO tb/ directory  <- absence is not a zero"); raise SystemExit(0)
subs = [os.path.join(R, d) for d in os.listdir(R) if os.path.isdir(os.path.join(R, d))] or [R]
v = []
for d in subs:
    ea = EventAccumulator(d, size_guidance={'scalars': 0}); ea.Reload()
    if 'train/selfplay_fraction' in ea.Tags()['scalars']:
        v += [(e.step, e.value) for e in ea.Scalars('train/selfplay_fraction')]
if not v:
    print("    NO train/selfplay_fraction scalar  <- absence is not a zero")
for s, x in sorted(set(v)):
    print(f'    step {s:>10,}  selfplay_fraction {x:.2f}')
PY
