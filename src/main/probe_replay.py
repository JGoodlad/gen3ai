"""Forensic replay probe: feed a REAL observation (captured during a battle) back
through a checkpoint and inspect why the policy chose what it did.

For a saved (summary.json, states.npz) pair and an invocation index, prints:
  - the recorded action distribution vs the live re-run (faithfulness check),
  - the active mon's per-move type-matchup the model saw,
  - an intervention sweep on the chosen move's matchup (0x..4x),
  - gradient saliency of the chosen action's logit over obs blocks.

The analysis itself lives in ``main.prober.engine`` (pure, framework-agnostic) so
the interactive Textual prober (``python -m main.prober``) and this CLI share one
source of truth. This module is just argv parsing + text rendering.

Usage:
  python -m main.probe_replay <ckpt.zip> <battle_summary.json> <battle_states.npz> <inv_index>
"""
import json
import sys

import numpy as np

from main.prober.engine import InvocationAnalysis, analyze_invocation
from main.prober.model import ProbeModel


def _render_text(a: InvocationAnalysis) -> str:
    """Reproduce the original CLI stdout for one (captured) invocation."""
    lines = []
    lines.append(f"=== {a.meta.summary_path}  inv {a.inv_index}  turn {a.turn} ===")
    lines.append(f"our={a.our_species}  opp={a.opp_species}  chosen={a.chosen}")

    lines.append("\naction            recorded   re-run(real obs)")
    for row in a.actions:
        if row.valid:
            lines.append(f"  {row.label:18s} {row.recorded * 100:>6.1f}%   {row.rerun * 100:6.1f}%")

    mults = np.round(np.array(a.matchups.multipliers), 2).tolist()
    lines.append(f"\nactive-move type multipliers the model saw (×4 normalised): {mults}")

    if a.sweep is not None and a.sweep.applicable:
        lines.append(f"\nP(all switches) at real obs = {a.sweep.baseline_p_switches * 100:.1f}%")
        lines.append(f"intervention — sweep '{a.chosen}' (request slot {a.sweep.request_slot}) multiplier:")
        for r in a.sweep.rows:
            lines.append(
                f"   set {r.multiplier:.0f}x -> P({a.chosen})={r.p_chosen * 100:5.1f}%"
                f"  P(switches)={r.p_switches * 100:4.1f}%"
            )

    lines.append(f"\ngradient saliency of P({a.chosen}) logit "
                 f"(per-dim mean over ALL obs={a.saliency.overall_mean_abs:.4f}):")
    for b in a.saliency.blocks:
        lines.append(f"   {b.name:30s} |grad|/dim={b.mean_abs:.4f}  sum={b.total_abs:.2f}")

    return "\n".join(lines)


def main():
    ckpt, summ_path, npz_path, inv_i = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    summ = json.load(open(summ_path))
    npz = np.load(npz_path)
    if "has_state" in npz and not npz["has_state"][inv_i]:
        print(f"WARNING: invocation {inv_i} has no captured state.")
        return

    model = ProbeModel.load(ckpt, device="cpu")
    analysis = analyze_invocation(model, summ, npz, inv_i, summary_path=summ_path, npz_path=npz_path)
    print(_render_text(analysis))


if __name__ == "__main__":
    main()
