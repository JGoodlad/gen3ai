#!/usr/bin/env bash
# X26 checkpoints (and the ai_v14 lineage's N0@75M / K3 for scale) on the PRE-ERA ground truth: Lane S truth_v2
# (bank_v1, 1,600 turns, 64 seeds, three pre-era greedy continuations). The reads dir holds the X26 reads of bank_v1
# (reads.sh) beside symlinks to ~/gen3ai_archive/policy_spectrum/baseline_2026-09-29/<label>.probs.npz.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=designs/research_state/measurements/m5_laneS/truth_v2
exec "$HERE/run.sh" "$HERE/starvation_readout.py" --bank designs/research_state/measurements/m5_laneS/bank_v1 \
  --rows "K2final=$T/rows_K2final_S64.compact.jsonl.gz" \
  --rows "N0final75M=$T/rows_N0final75M_S64.compact.jsonl.gz" \
  --rows "C_fixfinal=$T/rows_C_fixfinal_S64.compact.jsonl.gz" \
  --reads "${1:-$HOME/gen3ai_archive/collapse_pilot_2026-10-17/reads_bankv1_combined}" \
  --labels N0@75.0M,K3@99.2M,X26@10M,X26@30M,X26@90M,X26@150M,X26@160M \
  --pair X26@10M:X26@160M --pair X26@30M:X26@160M --pair K3@99.2M:X26@160M --pair N0@75.0M:X26@10M \
  --out "$HERE/readout_oldtruth_bankv1.json"
