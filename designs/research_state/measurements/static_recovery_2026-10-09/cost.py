"""Static-recovery levers' cost per row (CPU; gen3_static_recovery_v1): parameters, matmul FLOPs (FlopCounterMode, grad
enabled, as design_static_tokens.md §6.2 / the static port's `cost.py`), and the eager forward wall (no_grad, 1 thread,
the median of `--reps` timed calls after 3 warm-ups, the configs INTERLEAVED round-robin so a load drift hits every
config alike), on the compile parity fixture's 64 real rows, each config built through a real SB3 policy
(`identity_init_test._build_real_policy`) on the PRODUCTION toggles.

    cd <checkout>/src && python -c "import runpy,sys; sys.argv=['cost.py','--reps','41']; \
        runpy.run_path('../designs/research_state/measurements/static_recovery_2026-10-09/cost.py', run_name='__main__')"

Prints one JSON object: per config {policy_params, fe_params, matmul_flops_per_row, fwd_ms_per_64_median, fwd_ms_iqr}."""
import argparse
import json
import statistics
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

import agents.model.extractor_build as EB
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.identity_init_test import _build_real_policy
from utils.paths import repo_path

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=41)
a = ap.parse_args()
torch.set_num_threads(1)

ALL = dict(token_encoding="static", mon_hazard_cost="on", move_actor_state="on", trunk_layers=3,
           switch_hazard_cost="on", eot_residual="on")
CONFIGS = {
    "legacy (production)": {},
    "legacy + trunk_layers 3": dict(trunk_layers=3),
    "static": dict(token_encoding="static"),
    "static + trunk_layers 3": dict(token_encoding="static", trunk_layers=3),
    "static + switch_hazard_cost": dict(token_encoding="static", switch_hazard_cost="on"),
    "static + eot_residual": dict(token_encoding="static", eot_residual="on"),
    "static + mon_hazard_cost + move_actor_state": dict(token_encoding="static", mon_hazard_cost="on",
                                                        move_actor_state="on"),
    "static_recovery (the combined arm)": ALL,
}


def toggles(extra):
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    tog = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    tog.update(extra)
    return tog


print("src:", EB.__file__, file=sys.stderr)
built, out = {}, {}
for name, extra in CONFIGS.items():
    m, enc = _build_real_policy(**toggles(extra))
    pol = m.policy
    fe = pol.features_extractor
    fe.eval()
    obs, _ = load_parity_rows(enc.dimension)
    x = {"observation": torch.as_tensor(obs[:64])}
    fc = FlopCounterMode(display=False)
    with fc:
        fe(x)
    built[name] = (fe, x)
    out[name] = {"policy_params": sum(p.numel() for p in pol.parameters()),
                 "fe_params": sum(p.numel() for p in fe.parameters()),
                 "matmul_flops_per_row": fc.get_total_flops() / 64, "_t": []}
with torch.no_grad():
    for fe, x in built.values():
        for _ in range(3):
            fe(x)
    for _ in range(a.reps):
        for name, (fe, x) in built.items():
            t0 = time.perf_counter()
            fe(x)
            out[name]["_t"].append(time.perf_counter() - t0)
for name, r in out.items():
    ts = sorted(r.pop("_t"))
    q = statistics.quantiles(ts, n=4)
    r["fwd_ms_per_64_median"] = round(1e3 * statistics.median(ts), 2)
    r["fwd_ms_iqr"] = [round(1e3 * q[0], 2), round(1e3 * q[2], 2)]
    print(name, r, file=sys.stderr, flush=True)
print(json.dumps(out, indent=1))
