#!/usr/bin/env bash
# Regenerate designs/research_state/ledger_index.md from this worktree (python -m main.ledger_index --write).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
cd "$WT"
export PYTHONPATH="$WT/src"
exec /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.ledger_index --write
