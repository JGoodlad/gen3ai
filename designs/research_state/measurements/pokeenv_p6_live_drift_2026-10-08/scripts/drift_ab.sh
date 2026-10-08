#!/bin/bash
# old (Python layer) vs new (Rust reader) drift scan on the same 200 cached replays + master's source
cd /home/goodlad/dev/gen3ai/.claude/worktrees/agent-add90bfa9edcdb538
M=/home/goodlad/.cache/gen3ai/p4/showdown-master
timeout 1500 /tmp/p6coord/py.sh /tmp/p6coord/s3/ladder_drift_scan_old.py --offline --cache /tmp/p6coord/s3/drift_cache --n 200 --showdown $M > /tmp/p6coord/s3/drift_old.log 2>&1
echo "old exit $?" >> /tmp/p6coord/s3/drift_old.log
timeout 1500 /tmp/p6coord/py.sh src/main/ladder_drift_scan.py --offline --cache /tmp/p6coord/s3/drift_cache --n 200 --showdown $M > /tmp/p6coord/s3/drift_new.log 2>&1
echo "new exit $?" >> /tmp/p6coord/s3/drift_new.log
