"""The K9 learner-golden re-bake PROOF for gen3_nonformula_damage_v1 (X5 Tier 0 F8).

Runs the learner golden's update (`agents.training.learner_golden.compute`) on whichever tree is on
PYTHONPATH (the OLD = parent commit, or the NEW = the fix), optionally with the non-formula CLASS
NEUTRALISED in the op's tables:

  * OLD tree, --neutralise: MOVE_FIXED_DAMAGE := 0 (the only class table the old op had);
  * NEW tree, --neutralise: every gen3_nonformula_damage_v1 table := 0 (fixed / target-frac / Endeavor /
    nonformula / Flail / HP-scaled), MOVE_BP at the declared-BP rows := the dex 0, and MOVE_PHYS at the
    dex-damaging BP-0 rows := the facade-derived 0 (their old value).

The claim: OLD-neutralised and NEW-neutralised produce BYTE-IDENTICAL post-update parameters and losses.
Then the code paths differ ONLY through the class-move table entries, so every element the re-bake moves
is caused by a non-formula-class move. The script also records which class moves the buffer's forwards
actually saw (our request slots, every visible move slot, the op's believed top-K candidates).

    PYTHONPATH=<tree>/src python proof.py --label new [--neutralise] --out <json>
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Set

import torch


def _class_nums() -> Dict[int, str]:
    from agents import gen3_data
    out: Dict[int, str] = {}
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        if md is None:
            continue
        cat = getattr(md, "dex_category", "")
        if (cat in ("Physical", "Special") and md.base_power == 0) or mid in (
                "eruption", "waterspout", "beatup"):
            out[md.num] = mid
    # the OLD data has no dex category: always union the declared list (identical on the new tree)
    if True:
        for mid in ("seismictoss", "nightshade", "dragonrage", "sonicboom", "psywave", "superfang",
                    "guillotine", "horndrill", "fissure", "sheercold", "endeavor", "flail", "reversal",
                    "eruption", "waterspout", "return", "frustration", "magnitude", "present",
                    "hiddenpower", "beatup", "counter", "mirrorcoat", "bide", "lowkick", "spitup"):
            out[gen3_data.moves.get(mid).num] = mid
    return out


def _neutralise(model: Any) -> int:
    n = 0
    for mod in model.policy.modules():
        if not hasattr(mod, "MOVE_FIXED_DAMAGE"):
            continue
        n += 1
        with torch.no_grad():
            mod.MOVE_FIXED_DAMAGE.zero_()
            for k in ("MOVE_TARGET_HP_FRAC", "MOVE_ENDEAVOR", "MOVE_NONFORMULA", "MOVE_BP_FLAIL",
                      "MOVE_BP_HP_SCALED"):
                if hasattr(mod, k):
                    getattr(mod, k).zero_()
            try:
                from agents.model import damage_tables as dt
                models = dt.DAMAGE_MODELS
            except AttributeError:
                continue                                   # the OLD tree: nothing else to undo
            from agents import gen3_data
            for mid, (kind, _v, _c) in models.items():
                md = gen3_data.moves.get(mid)
                if kind in ("bp", "bp_flail"):
                    mod.MOVE_BP[md.num] = float(md.base_power)       # the dex 0
                if md.base_power == 0 and md.num != dt.HIDDEN_POWER_NUM:
                    mod.MOVE_PHYS[md.num] = 0.0                       # the facade's derived STATUS
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--neutralise", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer", default=None, help="a golden buffer .npz other than the tree's own")
    a = ap.parse_args()

    from agents.training import learner_golden as LG
    from agents.model.damage_op import DamageOperator

    if a.buffer:
        import functools
        from pathlib import Path
        LG.load_buffer_into = functools.partial(LG.load_buffer_into, path=Path(a.buffer))

    classes = _class_nums()
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
        "label": a.label, "neutralised": a.neutralise, "buffer": a.buffer or "the tree's own", "modules_neutralised": n_mod,
        "torch": torch.__version__, **fp,
        "class_moves_seen": {k: sorted(classes[n] for n in v if n in classes) for k, v in seen.items()},
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("label", "neutralised", "modules_neutralised",
                                          "init_params_sha256", "post_params_sha256",
                                          "class_moves_seen")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
