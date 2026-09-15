"""``--leaf-head`` — swap the SEARCH LEAF's win-prob head without touching how either side PLAYS.

Why this is a legal move at all, and why it is a narrow one.

The `WinProbHead` is a **leak-safe SIDE readout**: it reads `value_pooled` and its logit is stashed
for the aux loss and the offline instruments, and it is **never concatenated into pi or vf**
(`agents/model/aux_value_heads.py`). So on a `--critic winprob` checkpoint, where
`defensive.resolve_for_critic` makes `win_prob` the only leaf, replacing this head's four tensors
changes **what search believes about a candidate** and changes **nothing** about the action either
player takes when it is not searching. In the MIRROR cell that is exactly one thing: the unsearched
side is untouched, the null stays 0.50 by construction, and the contrast is the LEAF.

It is narrow on purpose:

* the head must be the SAME MODULE — same keys, same shapes. A head fitted against a different
  `D_MODEL`, or a `ValueDistHead` saved by accident, is REFUSED by name rather than loaded into a
  shape-compatible slot and quietly scored.
* a checkpoint with **no** win head (``--win-prob-mode none``) is REFUSED: there is nothing to
  replace and the cell would silently be the unmodified one.
* the swap is reported, with the file's sha1, so a results file's log names the leaf that produced
  it. (The 2026-09-11 battery's hazard 1: a results ROW records neither the checkpoint nor the
  defensive deadline, so cells are told apart by FILE NAME only — a swapped head is one more thing
  a row cannot say about itself, and the startup line is where it is said.)

The counterpart refusal matters as much as the load: if a future caller points this at a
``--critic shaped`` checkpoint, the leaf is `value`, not `win_prob`, and swapping the win head
would change NOTHING while looking like it changed something. `resolve_for_critic` already narrows
the leaf at startup; :func:`refuse_if_score_is_not_winprob` makes the silent case loud.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any


def head_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def refuse_if_score_is_not_winprob(score: str) -> None:
    """``--leaf-head`` only means something where the LEAF SCORE is the win-prob head.

    The check is on the resolved ``--score``, not on ``--defensive-leaf``: ``--score`` is what
    every root strategy reads (``grid`` never consults ``--defensive-leaf`` at all), so a guard on
    the leaf flag would pass on a ``grid`` cell that is actually ranking on the scalar value head.
    """
    if str(score) != "win_prob":
        raise ValueError(
            f"--leaf-head replaces the WIN-PROB head, but this cell's resolved --score is "
            f"{score!r} — the swap would change nothing while looking like it changed something. "
            f"Run it on a `--critic winprob` checkpoint (where `resolve_for_critic` resolves "
            f"`auto` to `win_prob`), or drop --leaf-head.")


def install_leaf_head(model: Any, path: str, *, score: str = "win_prob") -> dict:
    """Replace ``model.policy.features_extractor.win_head``'s weights from ``path``.

    ``path`` is a torch-saved ``state_dict`` for a `WinProbHead` — either bare, or under a
    ``"state_dict"`` / ``"win_head"`` key (what this project's refit writer emits). Returns a small
    dict describing what was swapped; raises ``ValueError`` on anything that would make the cell a
    lie (no head, wrong keys, wrong shapes).
    """
    import torch

    refuse_if_score_is_not_winprob(score)
    if not os.path.exists(path):
        raise ValueError(f"--leaf-head {path!r} does not exist")
    fe = getattr(model.policy, "features_extractor", None)
    head = getattr(fe, "win_head", None) if fe is not None else None
    if head is None:
        raise ValueError(
            "--leaf-head: this checkpoint has no `win_head` (win_prob_mode='none'), so there is "
            "nothing to replace and the cell would silently be the unmodified one")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "win_head" in blob:
        blob = blob["win_head"]
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    if not isinstance(blob, dict):
        raise ValueError(f"--leaf-head {path!r} is not a state_dict (got {type(blob).__name__})")
    sd = {str(k): v for k, v in blob.items()}

    cur = head.state_dict()
    if set(sd) != set(cur):
        raise ValueError(
            f"--leaf-head key mismatch: file has {sorted(sd)} but this checkpoint's win_head has "
            f"{sorted(cur)} — a head fitted against a different module is refused, never coerced")
    for k in cur:
        if tuple(sd[k].shape) != tuple(cur[k].shape):
            raise ValueError(
                f"--leaf-head shape mismatch on {k!r}: file {tuple(sd[k].shape)} vs checkpoint "
                f"{tuple(cur[k].shape)}")
    head.load_state_dict(sd)
    head.eval()
    return {"leaf_head": path, "leaf_head_sha1": head_sha1(path), "keys": sorted(cur)}
