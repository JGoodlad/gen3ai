#!/usr/bin/env bash
# E2 §1.1: the GAMES of one checkpoint, generated at its pin by the pin's own `main.ops.eval_trace_gen`, one generated
# cycle per live cycle that fed the registered bank (the bit-exact trace players, the live sentinel sets, the recorded
# regime; the registered seeds). Run through run_e2.sh at the gen checkout:
#
#   run_e2.sh gen_<cp> 2 8 12h gen_games.sh [--extension]
#
# Resumable: a step whose cycle already carries a COMPLETE generated manifest is skipped; an incomplete one is replaced
# (`--force`). `--extension` runs the pre-registered extension batch instead (seed + 1000, written beside the first;
# PREREGISTRATION §1.3 — only when a quintile could not be filled).
set -uo pipefail
case "${HIB_E2_KIND:-}" in gen) ;; *) echo "gen_games runs at the gen checkout only (run_e2.sh gen_<cp>)" >&2; exit 2 ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP="$HIB_E2_CP"
EXT=0; [ "${1:-}" = "--extension" ] && EXT=1
ARCH="${HIB_E2_ARCHIVE:-$HOME/gen3ai_archive/hidden_info_e2}"
LOGS="$ARCH/$CP/logs"; mkdir -p "$LOGS" "$ARCH/$CP/gen"
read -r RUN STEPS SEEDS < <("$VDP_PY" - "$HERE" "$CP" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import e2_common as E
c = E.CPS[sys.argv[2]]
print(c["run"], ",".join(map(str, c["steps"])), ",".join(map(str, c["seeds"])))
EOF
)
IFS=, read -r -a ST <<< "$STEPS"; IFS=, read -r -a SD <<< "$SEEDS"
read -r -a FLAGS < <("$VDP_PY" -c "import sys; sys.path.insert(0, '$HERE'); import e2_common as E; print(' '.join(E.GEN_FLAGS))")
for i in 0 1 2; do
  step="${ST[$i]}"; seed="${SD[$i]}"; name="e2_${CP}_${step}"
  if [ "$EXT" = 1 ]; then seed=$((seed + 1000)); name="${name}_ext"; fi
  out="$ARCH/$CP/gen/$name"
  man="$out/eval_traces/step_${step}/eval_manifest.json"
  if [ -f "$man" ] && "$VDP_PY" -c "import json,sys; m=json.load(open('$man')); sys.exit(0 if m.get('generated_by',{}).get('complete') is True else 1)"; then
    echo "$(date '+%F %T') $name complete, skipped" >> "$LOGS/gen_games.log"; continue
  fi
  echo "$(date '+%F %T') $name: seed $seed" >> "$LOGS/gen_games.log"
  "$VDP_PY" -m main.ops.eval_trace_gen "/home/goodlad/dev/gen3ai/models/$RUN@$step" --out "$out" --seed "$seed" \
    "${FLAGS[@]}" --nice 19 --force >> "$LOGS/gen_${name}.log" 2>&1 || { echo "$(date '+%F %T') $name FAILED rc=$?" >> "$LOGS/gen_games.log"; exit 1; }
  echo "$(date '+%F %T') $name done" >> "$LOGS/gen_games.log"
done
touch "$ARCH/$CP/GEN_DONE$([ "$EXT" = 1 ] && echo _ext)"
