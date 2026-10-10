"""K9(b) recorder over the END-STATE forward with the four-move closure (the P_end amendment, 2026-10-10): no undeclared
discrete op; the excluded share at FP32_TIE_EPS. `../../endstate_facts_2026-10-09/k9_tie_margins.py` with the configs
production / the 95d014fa overlay / the new overlay.

    python <this file>          # from any cwd; uses THIS checkout's src
"""
import json
import os
import sys

W = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..'))
sys.path.insert(0, W + '/src')
os.chdir(W)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from agents.model.identity_init_test import _build_real_policy  # noqa: E402
from agents.model.compile_parity_fixture import load_parity_rows  # noqa: E402
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings  # noqa: E402
from agents.training.rust_rollout.tie_margins import TieMargins  # noqa: E402
from agents.training.rust_rollout.consistency import FP32_TIE_EPS  # noqa: E402
from main.train.arch_arms import arm_overlay  # noqa: E402

cfg = json.load(open('designs/production_config.json'))
obs, _m = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
x = torch.as_tensor(obs)
END = arm_overlay("endstate")
for name, over in [("production", {}),
                   ("endstate @95d014fa overlay (closure off)", {k: v for k, v in END.items() if k != "move_set_closure"}),
                   ("endstate (P_end overlay, closure on)", END)]:
    tog = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    tog.update(over)
    pol = _build_real_policy(**tog)[0].policy
    fe = pol.features_extractor
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():   # planted weights: every zero-init route live
        for _n, m in fe.named_modules():
            wt = getattr(m, "weight", None)
            if isinstance(wt, torch.nn.Parameter) and wt.dim() == 2 and not bool(wt.any()):
                wt.copy_(torch.randn(wt.shape, generator=g) * 0.1)
    mode = TieMargins(x.shape[0])
    with torch.no_grad(), mode:
        fe({"observation": x})
    try:
        mode.check()
        ok = "no undeclared op"
    except Exception as e:
        ok = f"CHECK FAILED: {e}"
    excl = float(np.mean(~(mode.margin >= FP32_TIE_EPS)))
    print(json.dumps({"config": name, "move_set_closure": fe.move_set_closure, "check": ok,
                      "excluded_share": round(excl, 4), "eps": FP32_TIE_EPS, "rows": int(x.shape[0]),
                      "sites_seen": len(mode.sites_seen)}), flush=True)
