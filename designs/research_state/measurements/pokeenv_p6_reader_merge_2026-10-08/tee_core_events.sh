#!/bin/bash
# capture --obs-stream stdin, then run the baseline binary
if [ "$1" = "--obs-stream" ]; then
  f=/tmp/p6coord/s4/corpus/$(date +%s%N)_$$.in
  tee "$f" | /tmp/p6coord/s4/core_events_old "$@"
else
  exec /tmp/p6coord/s4/core_events_old "$@"
fi
