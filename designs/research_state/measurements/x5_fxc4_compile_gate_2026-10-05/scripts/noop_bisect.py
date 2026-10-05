"""F-XC-4: which node's removal by Inductor's `remove_noop_ops` (joint graph) makes the compiled R1
backward non-finite? Binary search over the number of removals allowed (fixed_mass, fresh, CUDA)."""
import json
import sys

import torch as th
import torch._inductor.config as IC
import torch._functorch.config as FC
from torch._inductor.fx_passes import post_grad as PG

from agents.model import compile_regions as cr, parity_probe as PP
from agents.model.compile_control import apply_compile_config
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

B = 2048
th.set_float32_matmul_precision("highest")
PP.PERTURB_SCALE = 0.0
m = LG.build_arm_learner("fixed_mass")
m.policy.to("cuda")
m.device = th.device("cuda")
from agents.model.compile_cache import ensure_hermetic_cache  # noqa: E402
ensure_hermetic_cache("fxc4 noop bisect")
apply_compile_config()
IC.fx_graph_cache = False
FC.enable_autograd_cache = False
b = cr.r1_batch(m, B)
m.policy.set_training_mode(True)
args = cr._r1_args(m, b)

BUDGET = [10 ** 9]
LOG: list = []
_real_erase = th.fx.Graph.erase_node


def remove_noop_ops_budgeted(graph):
    """`post_grad.remove_noop_ops`, verbatim, but at most BUDGET[0] removals, each logged."""
    inputs, input_storages, output_storages = PG.OrderedSet(), PG.OrderedSet(), PG.OrderedSet()
    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(PG.get_node_storage(node))
    output_node = next(iter(reversed(graph.nodes)))
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, th.fx.Node):
            output_storages.add(PG.get_node_storage(out))
    for node in graph.nodes:
        if node.target in PG.noop_registry:
            cond, src_index = PG.noop_registry[node.target]
            src = node.args[src_index] if isinstance(src_index, int) else src_index(node.args)
            if not isinstance(src, th.fx.Node):
                continue
            node_storage = PG.get_node_storage(node)
            src_storage = PG.get_node_storage(src)
            node_is_view = node_storage == src_storage
            if (not node_is_view and node_storage in output_storages
                    and (src_storage in input_storages or src_storage in output_storages)):
                continue
            if node_is_view and node in output_node.args and (src in inputs or src in output_node.args):
                continue
            is_valid, a, k = PG.get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if PG.same_meta(node, src) and cond(*a, **k):
                if len(LOG) >= BUDGET[0]:
                    continue
                LOG.append({"i": len(LOG), "node": node.name, "target": str(node.target),
                            "args": [str(x) for x in node.args][:4], "src": src.name,
                            "src_target": str(getattr(src, "target", "")),
                            "stack": (node.meta.get("stack_trace") or "")[-600:]})
                node.replace_all_uses_with(src)
                graph.erase_node(node)


PG.remove_noop_ops = remove_noop_ops_budgeted


def finite_with(budget: int) -> bool:
    BUDGET[0] = budget
    LOG.clear()
    th._dynamo.reset()
    comp = th.compile(micro_step, fullgraph=True, dynamic=False)
    c = cr._r1_arm(m, comp, args)
    ok = bool(th.isfinite(c["grad"]).all())
    print(f"budget {budget}: removals {len(LOG)} finite={ok}", flush=True)
    return ok


full_ok = finite_with(10 ** 9)
full_log = list(LOG)
n = len(full_log)
json.dump(full_log, open(sys.argv[1] + ".all.json", "w"), indent=1)
zero_ok = finite_with(0)
print("full", full_ok, "n", n, "zero", zero_ok, flush=True)
if full_ok or not zero_ok:
    print("NOT MONOTONE-BISECTABLE", flush=True)
    sys.exit(0)
lo, hi = 0, n          # finite at lo removals, non-finite at hi
while hi - lo > 1:
    mid = (lo + hi) // 2
    if finite_with(mid):
        lo = mid
    else:
        hi = mid
culprit = full_log[hi - 1]
print("CULPRIT", json.dumps(culprit), flush=True)
json.dump({"n_removals": n, "first_bad_budget": hi, "culprit": culprit}, open(sys.argv[1], "w"), indent=1)
