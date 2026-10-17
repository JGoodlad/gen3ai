#!/usr/bin/env bash
# The policy reads (forward passes only, CPU) of X26's eval snapshots, on a bank.
#     reads.sh <bank dir> <out dir>
# The checkpoints at the eval cycles' steps (each cycle's snapshot.zip, where it exists, is a hard link to it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BANK="$1"; OUT="$2"
C=/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints
args=()
for s in 2000128:2M 10000089:10M 20000043:20M 30000045:30M 50000106:50M 70000008:70M 90000067:90M \
         110000269:110M 130000065:130M 150000100:150M 160000085:160M; do
  step="${s%%:*}"; lab="${s##*:}"
  args+=(--ckpt "$C/checkpoint_${step}_steps.zip=X26@${lab}")
done
exec "$HERE/run.sh" -m main.policy_spectrum read --bank "$BANK" --out "$OUT" --threads 1 --workers 1 "${args[@]}"
