#!/usr/bin/env bash
# The battery's READS from chunk 1's outputs (no forward passes; light): B1 (spectrum / entropy on X26's bank),
# B2 (starvation on the X26@150M truth), B3 (starvation on the pre-era truth_v2), (f) churn + the read-overlap check.
# Early context points (10M / 30M / 90M, and 160M on bank_v1) are the collapse pilot's reads, symlinked in.
#     summarize.sh        (from the worktree root, after chunk 1)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
A="${A:-$HOME/gen3ai_archive/x26_collapse_battery_2026-10-20}"
PA="$HOME/gen3ai_archive/collapse_pilot_2026-10-17"
P=$ROOT/designs/research_state/measurements/collapse_pilot_2026-10-17
L=$ROOT/designs/research_state/measurements/m5_laneS
R=$HERE/results
mkdir -p "$R" "$A/reads_x26bank_all" "$A/reads_bankv1_all"
cd "$ROOT"

# combined read dirs: this unit's reads + the pilot's early ones (never overwriting one of ours)
for f in "$A"/reads_x26bank/*; do ln -sf "$f" "$A/reads_x26bank_all/$(basename "$f")"; done
for lab in 10M 30M 90M; do for ext in json probs.npz; do
  [ -e "$A/reads_x26bank_all/X26@$lab.$ext" ] || ln -s "$PA/reads_x26bank/X26@$lab.$ext" "$A/reads_x26bank_all/X26@$lab.$ext"
done; done
for f in "$A"/reads_bankv1/*; do ln -sf "$f" "$A/reads_bankv1_all/$(basename "$f")"; done
for f in "$PA"/reads_bankv1_combined/{N0@75.0M,K3@99.2M,X26@10M,X26@160M}.probs.npz; do
  [ -e "$A/reads_bankv1_all/$(basename "$f")" ] || ln -s "$(readlink -f "$f")" "$A/reads_bankv1_all/$(basename "$f")"
done
for lab in 10M 160M; do   # the pilot's read JSONs, for `policy_spectrum report` on bank_v1 (README)
  [ -e "$A/reads_bankv1_all/X26@$lab.json" ] || ln -s "$PA/reads_bankv1/X26@$lab.json" "$A/reads_bankv1_all/X26@$lab.json"
done

LAB=X26@10M,X26@30M,X26@90M,X26@160M,X26@180M,X26@200M,X26@220M,X26@230M,X26@240M,X26@260M,X26@280M,X26@300M,X26@320M,X26@330M,X26@340M,X26@360M,X26@380M
echo "=== B1 spectrum"
"$HERE/run.sh" -m main.policy_spectrum report --bank "$P/bank_x26" --reads "$A/reads_x26bank_all" --labels "$LAB" \
  --pair X26@180M:X26@380M --pair X26@180M:X26@280M --pair X26@280M:X26@380M --pair X26@160M:X26@380M \
  --pair X26@30M:X26@380M --title "X26 policy spectrum on its own bank, 10M-380M" \
  --out "$R/b1_spectrum_x26bank.md" --json "$R/b1_spectrum_x26bank.json"

echo "=== B2 starvation, X26@150M truth"
"$HERE/run.sh" "$P/starvation_readout.py" --bank "$P/bank_x26" --rows "X26@150M=$PA/rows_X26at150M_S32.jsonl" \
  --reads "$A/reads_x26bank_all" --labels X26@10M,X26@160M,X26@180M,X26@230M,X26@280M,X26@330M,X26@380M \
  --pair X26@180M:X26@380M --pair X26@180M:X26@280M --pair X26@280M:X26@380M --pair X26@160M:X26@380M \
  --pair X26@10M:X26@380M --out "$R/b2_readout_x26truth.json"

echo "=== B3 starvation, pre-era truth_v2"
T=$L/truth_v2
"$HERE/run.sh" "$P/starvation_readout.py" --bank "$L/bank_v1" \
  --rows "K2final=$T/rows_K2final_S64.compact.jsonl.gz" --rows "N0final75M=$T/rows_N0final75M_S64.compact.jsonl.gz" \
  --rows "C_fixfinal=$T/rows_C_fixfinal_S64.compact.jsonl.gz" --reads "$A/reads_bankv1_all" \
  --labels N0@75.0M,K3@99.2M,X26@10M,X26@160M,X26@180M,X26@230M,X26@280M,X26@330M,X26@380M \
  --pair X26@180M:X26@380M --pair X26@160M:X26@380M --pair X26@10M:X26@380M --pair K3@99.2M:X26@380M \
  --out "$R/b3_readout_oldtruth.json"

echo "=== (f) churn + overlap"
"$HERE/run.sh" "$HERE/churn.py" "$A/reads_x26bank_all" \
  X26@160M,X26@180M,X26@200M,X26@220M,X26@240M,X26@260M,X26@280M,X26@300M,X26@320M,X26@340M,X26@360M,X26@380M \
  "$R/f_churn.json" --overlap "$PA/reads_x26bank"
echo "=== DONE"
