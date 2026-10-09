"""Static port cost per row (CPU): matmul FLOPs (FlopCounterMode, grad enabled, as design_static_tokens.md §6.2),
parameters, eager forward wall (median of 7, 1 thread), on the compile parity fixture's 64 real rows, the PRODUCTION
toggles built through a real SB3 policy (`identity_init_test._build_real_policy`).

    python -E cost.py --src <checkout>/src [--parent]   # --parent: only the configs the parent commit can build

Prints one JSON object: per config {policy_params, fe_params, matmul_flops_per_row, fwd_ms_per_64_median7}."""
import argparse
import json
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--parent", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.src)

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from agents.model.compile_parity_fixture import load_parity_rows  # noqa: E402
from agents.model.identity_init_test import _build_real_policy  # noqa: E402
import agents.model.extractor_build as EB  # noqa: E402
from utils.paths import repo_path  # noqa: E402

assert EB.__file__.startswith(a.src), EB.__file__
torch.set_num_threads(1)

BUNDLE = dict(token_encoding="static", move_resolution="on", speed_physics="on", value_threat_inject=False,
              op_reduction="principled", obs_facts="v1")
CONFIGS = {
    "legacy (production)": {},
    "static": dict(token_encoding="static"),
    "static + mon_hazard_cost": dict(token_encoding="static", mon_hazard_cost="on"),
    "static + move_actor_state": dict(token_encoding="static", move_actor_state="on"),
    "static + both facts": dict(token_encoding="static", mon_hazard_cost="on", move_actor_state="on"),
    "bundle (no facts)": dict(BUNDLE),
    "bundle + both facts": dict(BUNDLE, mon_hazard_cost="on", move_actor_state="on"),
}


def toggles(extra):
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    tog = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    tog.update(extra)
    return tog


out = {}
for name, extra in CONFIGS.items():
    if a.parent and ("facts" in name and "no facts" not in name):
        continue
    m, enc = _build_real_policy(**toggles(extra))
    pol = m.policy
    fe = pol.features_extractor
    obs, _ = load_parity_rows(enc.dimension)
    x = {"observation": torch.as_tensor(obs[:64])}
    fc = FlopCounterMode(display=False)
    with fc:
        fe(x)
    with torch.no_grad():
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            fe(x)
            ts.append(time.perf_counter() - t0)
    out[name] = {"policy_params": sum(p.numel() for p in pol.parameters()),
                 "fe_params": sum(p.numel() for p in fe.parameters()),
                 "matmul_flops_per_row": fc.get_total_flops() / 64,
                 "fwd_ms_per_64_median7": round(1e3 * statistics.median(ts), 1)}
    print(name, out[name], file=sys.stderr, flush=True)
print(json.dumps(out, indent=1))
