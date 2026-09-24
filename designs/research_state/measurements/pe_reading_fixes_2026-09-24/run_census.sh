#!/usr/bin/env bash
# The golden diff census: capture the fixed battle set with the fork's three reading files at the
# BASE commit, then with the fix, and resolve every changed entry to a field. Run from the worktree.
# WIDE="N_BATTLES N_TEAMS" plays a wider deterministic set instead (no fixture comparison).
set -euo pipefail
WT=$(git rev-parse --show-toplevel); cd "$WT"
BASE=${BASE:-origin/main}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}
export PYTHONPATH=${PYTHONPATH:-}:src
D=designs/research_state/measurements/pe_reading_fixes_2026-09-24
OUT=${OUT:-/tmp/pefix}; mkdir -p "$OUT/fixed_files"
FILES="src/poke_env/battle/pokemon.py src/poke_env/battle/abstract_battle.py src/poke_env/battle/battle.py"
for f in $FILES; do cp "$f" "$OUT/fixed_files/"; done
trap 'for f in $FILES; do cp "$OUT/fixed_files/$(basename $f)" "$f"; done' EXIT
git show "$BASE:src/poke_env/battle/pokemon.py" > src/poke_env/battle/pokemon.py
git show "$BASE:src/poke_env/battle/abstract_battle.py" > src/poke_env/battle/abstract_battle.py
git show "$BASE:src/poke_env/battle/battle.py" > src/poke_env/battle/battle.py
nice -n 10 "$PY" $D/golden_diff_census.py capture "$OUT/golden_base${WIDE:+_wide}.npz" ${WIDE:-}
for f in $FILES; do cp "$OUT/fixed_files/$(basename $f)" "$f"; done
nice -n 10 "$PY" $D/golden_diff_census.py capture "$OUT/golden_fixed${WIDE:+_wide}.npz" ${WIDE:-}
git show "$BASE:src/agents/training/golden_obs_fixture.json" > "$OUT/fixture_base.json"
nice -n 10 "$PY" $D/golden_diff_census.py diff "$OUT/golden_base${WIDE:+_wide}.npz" "$OUT/golden_fixed${WIDE:+_wide}.npz" $([ -z "${WIDE:-}" ] && echo "$OUT/fixture_base.json")
