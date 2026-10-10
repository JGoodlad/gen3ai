"""K9(b) recorder over the endstate forward: no undeclared discrete op; the excluded share at FP32_TIE_EPS."""
import sys, os, json  # run: python <this file> (from any cwd; uses this checkout's src)
W = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, W + '/src')
os.chdir(W)
import numpy as np
import torch
torch.set_num_threads(4)
from agents.model.identity_init_test import _build_real_policy
from agents.model.compile_parity_fixture import load_parity_rows
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.rust_rollout.tie_margins import TieMargins
from agents.training.rust_rollout.consistency import FP32_TIE_EPS
from main.train.arch_arms import arm_overlay

cfg = json.load(open('designs/production_config.json'))
obs, _m = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
x = torch.as_tensor(obs)
for name, over in [("production", {}), ("E (no new levers)", {k: v for k, v in arm_overlay("endstate").items()
                    if k not in ("move_resolution_facts", "status_facts", "ko_ramp", "drop_progress_clock", "g_ledger")}),
                   ("endstate", arm_overlay("endstate"))]:
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
    print(f"{name}: {ok}; excluded share at eps {FP32_TIE_EPS}: {excl:.3f}; sites seen {len(mode.sites_seen)}")
