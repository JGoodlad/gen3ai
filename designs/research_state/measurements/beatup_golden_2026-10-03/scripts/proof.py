"""The K9 learner-golden re-bake PROOF for gen3_beatup_exact_v1 (X5 Tier 0 finding F8's Beat Up residual).

Runs the learner golden's update (`agents.training.learner_golden.compute`) on whichever tree is on
PYTHONPATH — the OLD tree (the parent commit, where Beat Up is priced as one 10-BP Dark special hit) or the
NEW tree (the exact multi-hit typeless model) — optionally with BEAT UP NEUTRALISED in the op's tables.

NEUTRALISED means the move's row is the SAME on both trees and prices nothing:

  * `MOVE_BP[beatup]   := 0`   (so the formula's BP gate zeroes it in every kernel, and an outgoing slot is
                                 `usable == 0`),
  * `MOVE_BEATUP[...]  := 0`   (NEW tree only — the flag that swaps the formula's inputs),
  * `MOVE_TYPE_IDX[beatup] := DARK` (NEW tree only — the new tree writes '???' there; the old tree's row is
                                 DARK already, and the incoming matrix's `type_mult` cell reads that row).

The claim: OLD-neutralised and NEW-neutralised produce BYTE-IDENTICAL post-update parameters, per-group
hashes and losses. Then the two trees differ ONLY through Beat Up's table entries (every kernel change is a
`where` / gather that selects exactly the old value for a non-Beat-Up cell), and so every element the re-bake
moves is caused by Beat Up. The script also records which Beat Up exposures the buffer's forwards actually saw
(our request slots, every visible move slot, the op's believed top-K candidates).

    PYTHONPATH=<tree>/src python proof.py --label new [--neutralise] [--buffer <npz>] --out <json>
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Set

import torch


def _neutralise(model: Any) -> int:
    from agents import gen3_data
    from agents.observation.types import TypeEncoder
    n = gen3_data.moves.get("beatup").num
    dark = TypeEncoder.TYPE_TO_IDX["DARK"]
    count = 0
    for mod in model.policy.modules():
        if not hasattr(mod, "MOVE_BP") or not hasattr(mod, "MOVE_TYPE_IDX"):
            continue
        count += 1
        with torch.no_grad():
            mod.MOVE_BP[n] = 0.0
            mod.MOVE_TYPE_IDX[n] = dark
            if hasattr(mod, "MOVE_BEATUP"):
                mod.MOVE_BEATUP.zero_()
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--neutralise", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer", default=None, help="a golden buffer .npz other than the tree's own")
    a = ap.parse_args()

    from agents import gen3_data
    from agents.training import learner_golden as LG
    from agents.model.damage_op import DamageOperator

    if a.buffer:
        import functools
        from pathlib import Path
        LG.load_buffer_into = functools.partial(LG.load_buffer_into, path=Path(a.buffer))

    bu = gen3_data.moves.get("beatup").num
    seen: Dict[str, Set[int]] = {"request_slots": set(), "visible_move_slots": set(), "believed_topk": set()}

    def pre(mod: Any, args: Any) -> None:
        ctx = args[0]
        seen["request_slots"] |= set(int(x) for x in ctx.our_active_req_move_ids.flatten().tolist())
        seen["visible_move_slots"] |= set(int(x) for x in ctx.all_move_ids.flatten().tolist())

    def post(mod: Any, args: Any, out: Any) -> None:
        idx = getattr(mod, "last_topk_idx", None)
        if idx is not None:
            seen["believed_topk"] |= set(int(x) for x in idx.flatten().tolist())

    with LG._one_thread():
        model = LG.build_learner()
    n_mod = _neutralise(model) if a.neutralise else 0
    for mod in model.policy.modules():
        if isinstance(mod, DamageOperator):
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
    fp = LG.compute(model)
    res = {
        "label": a.label, "neutralised": a.neutralise, "buffer": a.buffer or "the tree's own",
        "modules_neutralised": n_mod, "torch": torch.__version__, **fp,
        "beatup_seen_in": {k: (bu in v) for k, v in seen.items()},
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("label", "neutralised", "modules_neutralised",
                                          "init_params_sha256", "post_params_sha256",
                                          "beatup_seen_in")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
