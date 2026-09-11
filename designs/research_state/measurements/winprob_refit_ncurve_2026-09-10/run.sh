#!/usr/bin/env bash
# THE N-CURVE, end to end. Offline, CPU only, niced, nothing written under models/.
#
#   1. extract    one frozen forward of `step_10000032` over THREE eval trees (seeds 20260909 /
#                 20260910 / 20260911, 400 + 800 + 800 games x 12 opponents) -> ~24,000 battles,
#                 ~720,000 states, ~15 min per substrate. The three snapshot.zip files are
#                 md5-compared and a mismatch REFUSES: the dataset must be ONE frozen policy.
#   2. ncurve     nested-by-battle subsets at N = 1k / 2k / 4k / 8k / 16k / all, four conditions
#                 each, scored on ONE fixed held-out set of 2,400 battles; plus the 5x-steps,
#                 no-early-stop control at the largest N.
#   3. readout    the mixture identity, the turn-1 decodes, Brier and the calibration slope, with
#                 battle-clustered bootstrap CIs on every delta.
#   4. summarize  the reading rule applied IN CODE, both spread keyings.
#
# Neither the 128-dim feature tensor nor the prediction columns are committed; this regenerates
# them in about 40 minutes per substrate.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(cd ../../../.. && pwd)/src"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}
HP=${HP_EVAL:-/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval}
OUT=${NCURVE_OUT:-/home/goodlad/.claude/jobs/9ab51de6/tmp/ncurve}
mkdir -p "$OUT"

for RUN in ai_v12_11_ladder_ctrl10M ai_v12_15_ladder_ctrl10M_b; do
  nice -n 19 "$PY" extract.py \
    --tree "cyc400=$HP/$RUN/eval_traces/step_10000032" \
    --tree "cyc800=$HP/hp800/$RUN/eval_traces/step_10000032" \
    --tree "cyc800b=$HP/hp800b/$RUN/eval_traces/step_10000032" \
    --snapshot "$HP/$RUN/eval_traces/step_10000032/snapshot.zip" \
    --out-dir "$OUT/$RUN" --threads 2 --batch 256

  nice -n 19 "$PY" ncurve.py --dir "$OUT/$RUN" --out "$OUT/ncurve_$RUN.npz" \
    --ns 1000,2000,4000,8000,16000,0 --holdout 2400 --threads 2

  nice -n 19 "$PY" readout.py --dir "$OUT/$RUN" --preds "$OUT/ncurve_$RUN.npz" \
    --out "$OUT/read_$RUN.json"

  for KEY in cell opponent; do
    nice -n 19 "$PY" summarize.py --read "$OUT/read_$RUN.json" --name "$RUN" --key "$KEY" \
      --out-md "tables_${RUN}_${KEY}.md" --out-json "reading_${RUN}_${KEY}.json"
  done
  cp "$OUT/$RUN/extract_meta.json" "extract_meta_${RUN}.json"
  "$PY" - "$OUT/read_$RUN.json" "ncurve_stats_${RUN}.json" <<'PYEOF'
import json, sys
# the COMMITTED summary: points, CIs and deltas — never an array.
r = json.load(open(sys.argv[1]))
r.pop("fit_log", None)
json.dump(r, open(sys.argv[2], "w"), indent=1)
PYEOF
done
