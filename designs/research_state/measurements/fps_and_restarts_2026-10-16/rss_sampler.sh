#!/bin/bash
# Read-only /proc sampler of the live X26 trainer child and its children (no signals).
P=${1:-209527}
while [ -d "/proc/$P" ]; do
  ts=$(date +%s)
  r=$(awk '/VmRSS|RssAnon|VmHWM/{printf "%s=%s ", $1, $2}' /proc/$P/status 2>/dev/null)
  kids=$(ps -o pid=,rss=,comm= --ppid "$P" | awk '{printf "%s:%s:%s ", $1, $3, $2}')
  echo "$ts $r $kids" >> /tmp/fpsr/rss.log
  sleep 120
done
