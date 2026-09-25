#!/usr/bin/env bash
# Round 2 §4.4 CONVERGENCE SIDE-CHECK (a DESCRIPTOR; changes no branch and no count) — the one-liner:
#   bash designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/side_check.sh [OUT_DIR]
# Run from a WORKTREE root (never the main checkout: finding K-1). OUT_DIR defaults to this read dir.
# REFUSES unless RB+ (ai_v13_31_popr1_read_loop_ext) and RC+ (ai_v13_32_popr1_read_ctrl_ext) have
# BOTH finished: final_model.zip present and "Training complete" in the child log. The meter's own
# --check then confirms 0 mismatches at budget 4,030,464 before anything is read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(git -C "$HERE" rev-parse --show-toplevel)"
MAIN=/home/goodlad/dev/gen3ai
[ "$WT" != "$MAIN" ] || { echo "REFUSED: run from a worktree, not the main checkout (K-1)"; exit 2; }
OUT="${1:-$HERE/..}"
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}"
export PYTHONPATH="${PYTHONPATH:-}:$WT/src"
for r in ai_v13_31_popr1_read_loop_ext ai_v13_32_popr1_read_ctrl_ext; do
  [ -f "$MAIN/models/$r/final_model.zip" ] || { echo "REFUSED: $r has no final_model.zip yet"; exit 3; }
  grep -q "Training complete" "$MAIN/models/$r/launcher_child.full.log" \
    || { echo "REFUSED: $r has not logged 'Training complete'"; exit 3; }
done
cd "$WT"
run() { local name=$1 rc=0; shift; { echo "\$ python -m main.best_response_gap $*"; "$PY" -m main.best_response_gap "$@" || rc=$?; echo "rc=$rc"; } > "$OUT/$name.txt" 2>&1; }
run brgap_ext_check ai_v13_31_popr1_read_loop_ext ai_v13_32_popr1_read_ctrl_ext --check
grep -q "^rc=0$" "$OUT/brgap_ext_check.txt" || { cat "$OUT/brgap_ext_check.txt"; echo "VOID: --check refused"; exit 4; }
run brgap_ext ai_v13_32_popr1_read_ctrl_ext ai_v13_31_popr1_read_loop_ext \
    --rounds ai_v13_32_popr1_read_ctrl_ext=2 ai_v13_31_popr1_read_loop_ext=3 \
    --json "$OUT/brgap_ext.json" --md "$OUT/brgap_ext.md"
{ echo "\$ python scripts/convergence_rule.py --json convergence_rule.json"
  rc=0; "$PY" "$HERE/../../scripts/convergence_rule.py" --json "$OUT/convergence_rule.json" || rc=$?; echo "rc=$rc"; } \
  > "$OUT/convergence_rule.txt" 2>&1
cat "$OUT/brgap_ext_check.txt" "$OUT/convergence_rule.txt"
