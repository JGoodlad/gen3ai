"""Generate FRESH searchable losses for the teacher — the supply lever (§Phase-1 follow-on).

The per-cycle teacher reads eval traces (a trickle, every ~2M steps). This plays NEW battles —
the FROZEN trainee vs a chosen opponent — and records each as a full trace (summary + npz +
``*_reconstruction.json``), exactly the way eval does (``begin_forensic_cycle`` +
``run_local_battles``), so the persistent worker has a CONTINUOUS stream to search. It never touches
the training hot path (the battles are a frozen-snapshot side activity, like eval), and — because the
worker CHOSE the opponent — the exact-opponent is known directly (no sentinel-resolution fragility).
Returns the loss traces it produced + the opponent each was played against.
"""

from __future__ import annotations

import asyncio
import glob
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class GeneratedTrace:
    summary_path: str
    recon_path: str
    opponent_label: str
    opponent_ckpt: Optional[str]   # the EXACT opponent the worker played (None for a bot)
    opponent_source: str           # 'ckpt' | 'bot'


def generate_loss_traces(trainee, opponent, *, out_dir: str, n_battles: int, step: int,
                         opponent_label: str, opponent_ckpt: Optional[str], opponent_source: str,
                         use_bridge: bool = True, concurrency: int = 1,
                         loss_quota: int = 1000, impl: str = "node") -> List[GeneratedTrace]:
    """Play ``n_battles`` (frozen ``trainee`` vs ``opponent``) recording every LOSS as a trace into
    ``out_dir``, and return those traces tagged with the (known) opponent. ``trainee`` is an
    ``EvalRLPlayer`` (built once by the worker, reused across calls); ``opponent`` is its poke-env
    player. Reuses the eval forensic-capture + bridge-battle path wholesale.

    ``impl`` (``"node"`` default | ``"rust"``) picks the in-process sim-bridge child these battles
    run on — the same selector as ``--use-bridge={node,rust}``. It used to be unset here, which
    silently pinned generation to node even on a rust run."""
    from utils.bridge.local_battle_runner import run_local_battles

    os.makedirs(out_dir, exist_ok=True)
    # Capture LOSSES only (those are the teacher's candidates); a high quota = keep them all this batch.
    trainee.begin_forensic_cycle(out_dir, step, trace_tag="", win_quota=0, loss_quota=loss_quota)
    trainee.reset_battles()
    if hasattr(opponent, "reset_battles"):
        opponent.reset_battles()

    async def _play():
        await run_local_battles(trainee, opponent, n_battles, concurrency=concurrency, impl=impl)
    asyncio.run(_play())

    traces: List[GeneratedTrace] = []
    for sp in sorted(glob.glob(os.path.join(out_dir, "loss_*_summary.json"))):
        recon = sp[: -len("_summary.json")] + "_reconstruction.json"
        if os.path.exists(recon):                  # the search needs the reconstruction sibling
            traces.append(GeneratedTrace(
                summary_path=sp, recon_path=recon, opponent_label=opponent_label,
                opponent_ckpt=opponent_ckpt, opponent_source=opponent_source))
    return traces
