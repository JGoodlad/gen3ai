#!/usr/bin/env bash
# Race-free alert watch (class fix, 10-15): per-file LINE OFFSETS persist in $STATE across re-arms, so a line written between
# one watch's exit and the next one's arming is never marked seen. First use of a file starts at its current end.
# Usage: alert_watch.sh <state_file> <ERE pattern> <ignore ERE> <timeout_s> <file>...
set -u
STATE=$1; PAT=$2; IGN=$3; TO=$4; shift 4
touch "$STATE"
end=$(( $(date +%s) + TO ))
while :; do
  hits=""
  for f in "$@"; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    o=$(awk -v f="$f" -F'\t' '$1==f{print $2}' "$STATE")
    if [ -z "$o" ]; then o=$n; fi
    [ "$n" -lt "$o" ] && o=0            # rotated / truncated
    if [ "$n" -gt "$o" ]; then
      new=$(sed -n "$((o + 1)),${n}p" "$f" | grep -E "$PAT" | grep -vE "$IGN")
      [ -n "$new" ] && hits="$hits$new"$'\n'
    fi
    grep -v -F "$f	" "$STATE" > "$STATE.tmp"; printf '%s\t%s\n' "$f" "$n" >> "$STATE.tmp"; mv "$STATE.tmp" "$STATE"
  done
  [ -n "$hits" ] && { printf '%s' "$hits" | cut -c1-500 | head -10; exit 0; }
  [ "$(date +%s)" -ge "$end" ] && { echo "watch_timeout"; exit 124; }
  sleep 20
done
