#!/usr/bin/env bash
# Copy each finished unit's result + analysis (and its box-state file) from the archive into results/.
L=/home/goodlad/gen3ai_archive/lane_c_perf_guard/bank
D=$(dirname "$(readlink -f "$0")")/results
mkdir -p "$D"
for u in $(ls "$L" | grep -E '^[BPL][0-9]+$'); do
  f=$(ls -d "$L/$u"/*/ 2>/dev/null | tail -1); [ -n "$f" ] || continue
  [ -f "$f/time_result.json" ] && cp "$f/time_result.json" "$D/${u}_time_result.json"
  [ -f "$f/time_analysis.json" ] && cp "$f/time_analysis.json" "$D/${u}_time_analysis.json"
  [ -f "$L/$u.env" ] && cp "$L/$u.env" "$D/${u}_box.txt"
done
cp "$L/status" "$D/status.txt" 2>/dev/null
