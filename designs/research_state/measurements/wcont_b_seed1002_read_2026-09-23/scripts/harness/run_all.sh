#!/usr/bin/env bash
R=/home/goodlad/.claude/jobs/wcontb_read_2026-09-23
bash $R/run_untaught.sh > $R/untaught.log 2>&1
bash $R/run_away_cell.sh > $R/away.log 2>&1
echo "READ ALL DONE $(date -Is)" >> $R/status.txt
