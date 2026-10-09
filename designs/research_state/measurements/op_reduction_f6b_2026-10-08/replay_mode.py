"""The K9(b) flip-judge replay (`designs/research_state/measurements/k9_flip_judge_2026-10-08/replay.py`) under a
chosen `--op-reduction`, at the SAME weights: the u1480 HEAD reconstruction, plus `op_worst_proj` at its
init (all zeros — exactly what a `principled` build starts from) when the weights lack it.

    python tmp/f6b/replay_mode.py <max|principled> --weights-from <pt> --out <json>
"""
import importlib.util
import os
import sys

mode = sys.argv.pop(1)
ROOT = os.getcwd()
p = os.path.join(ROOT, "designs/research_state/measurements/k9_flip_judge_2026-10-08/replay.py")
spec = importlib.util.spec_from_file_location("k9_flip_replay", p)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
R.m.old.ARMS["static_fm"] = dict(R.m.old.ARMS["static_fm"], op_reduction=mode)

_orig_state_dict = R.m._state_dict


def _state_dict_with_init(path):
    import torch as th
    sd = _orig_state_dict(path)
    if mode == "principled":
        for k in [k for k in list(sd) if k.endswith("damage_op.out_gain")]:
            base = k[: -len("damage_op.out_gain")]
            sd[base + "op_worst_proj.weight"] = th.zeros(128, 2)
    return sd


R.m._state_dict = _state_dict_with_init
sys.exit(R.main())
