#!/usr/bin/env bash
# X65-F1 contention workload (orchestrator 2026-11-05): 6 parallel offline S′ (r0s4_rolls@10) bank-tooth loops over
# dlive's banked U states, one per agent core (5,6,7,13,14,15), nice 19. Counts toward NO read: every --out is under the
# phase dir given here (outside the repo). Each worker re-runs the same hashed sample (78 scored states) pass after pass
# until the deadline; the sampler logs the total rows appended once a minute to <phase>/counts.tsv.
#   group_v1.sh <phase dir> <minutes>
set -u
PH="${1:?phase dir}"; MIN="${2:?minutes}"
REPO=/home/goodlad/dev/gen3ai; PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
S=/home/goodlad/gen3ai_archive/dlive_read_2026-10-28/live
MODEL=$REPO/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip
SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
ENG="port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129"
mkdir -p "$PH"; cd "$REPO" || exit 2
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO/src"
END=$(( $(date +%s) + MIN * 60 ))
date -Is > "$PH/start.txt"
worker() {   # <core> <idx>
    local core=$1 i=$2 n=0 left
    while :; do
        left=$(( END - $(date +%s) )); [ "$left" -le 5 ] && break
        n=$((n + 1))
        timeout "$left" nice -n 19 taskset -c "$core" "$PY" -m main.value_search.live_identity bank-tooth \
            --selected "$S/selected.jsonl" --scored "$S/scored.jsonl" --search r0s4_rolls@10 --model "$MODEL" \
            --model-sha256 "$SHA" --engine "$ENG" --out "$PH/w${i}_p${n}.json" --max 78 >> "$PH/w${i}.log" 2>&1
    done
}
i=0; for c in 5 6 7 13 14 15; do i=$((i + 1)); worker "$c" "$i" & done
while [ "$(date +%s)" -lt "$END" ]; do
    echo -e "$(date +%s)\t$(cat "$PH"/w*_p*.json.rows.jsonl 2>/dev/null | wc -l)\t$(cut -d' ' -f1-3 /proc/loadavg)" >> "$PH/counts.tsv"
    sleep 60
done
wait
echo -e "$(date +%s)\t$(cat "$PH"/w*_p*.json.rows.jsonl 2>/dev/null | wc -l)\t$(cut -d' ' -f1-3 /proc/loadavg)" >> "$PH/counts.tsv"
date -Is > "$PH/end.txt"
