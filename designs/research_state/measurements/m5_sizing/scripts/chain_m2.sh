#!/usr/bin/env bash
# chain_m relaunch (2026-10-01 ~22:30): B's and C's meters start only AFTER arm C's training ends, so neither
# learning arm trains beside 6 meter workers (O8's train_ms clause; coordinator 22:2x). Bounded 12 h.
L=/home/goodlad/.cache/gen3ai/tmp/sizing
e=$((SECONDS + 43200))
until grep -q "CHAINBC C exit" $L/chain_bc.log || [ $SECONDS -ge $e ]; do sleep 60; done
grep -q "CHAINBC C exit" $L/chain_bc.log || { echo "CHAINM2_FAIL C never ended $(date -Is)"; exit 1; }
echo "CHAINM2 C ended; meters B then C $(date -Is)"
LABS="B:sizing_B_n256_e10_s1001 C:sizing_C_n256_e5_s1001" exec bash $L/chain_m.sh
