"""CPU only: drive Inductor's CUDA (Triton) codegen of the production extractor WITHOUT a GPU, for the train-mode
forward + backward (`--mode train`, the fwd+bwd CUDA cell) or the no-grad forward (`--mode nograd`), then compile
every generated Triton kernel for sm_86 (`triton.compile` with a target needs no device).

    PYTHONPATH=src python3 designs/research_state/measurements/triton_xblock16_cat_read_2026-11-11/fake_cuda_codegen_fwdbwd.py \
        --batch 2 --mode train --out DIR

`../triton_xblock16_2026-10-25/fake_cuda_codegen.py` extended to the backward graph, with the failure handling
moved to `triton_diag_plugin.py` (the same patch a GPU pytest run loads as a plugin). A DIAGNOSTIC, not a gate
(README.md): dynamo + AOTAutograd run on the CPU module, each post-AOT aten graph (forward, then backward) has its
devices rewritten to cuda and is lowered by `compile_fx_inner` under fake CUDA device properties (RTX 3080 Ti,
sm_86), with every Triton kernel precompiled in-process and each failure recorded. No CUDA context is ever
created: CUDA_VISIBLE_DEVICES="" and every torch.cuda query it needs is faked. Torch 2.8.0 / triton 3.4.0 only.
"""
import argparse
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--speed-cum-mm", action="store_true",
                help="COUNTERFACTUAL: the speed mixture's fp64 cumsum spelled as a triangular matmul (no scan)")
ap.add_argument("--mode", choices=("train", "nograd"), default="train",
                help="train: train-mode forward + backward (the fwd+bwd CUDA cell); nograd: the 10-25 inference cell")
ap.add_argument("--override", action="append", default=[], help="k=v extractor override (json value)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(args.out, "inductor")
os.environ["TRITON_CACHE_DIR"] = os.path.join(args.out, "triton")

import torch  # noqa: E402

def _patch_cuda():
    global _props
    _props = types.SimpleNamespace(
        name="NVIDIA GeForce RTX 3080 Ti", major=8, minor=6, multi_processor_count=80,
        total_memory=12 * 1024 ** 3, regs_per_multiprocessor=65536,
        max_threads_per_multi_processor=1536, warp_size=32, gcnArchName="",
        L2_cache_size=6 * 1024 ** 2, shared_memory_per_block_optin=101376,
        shared_memory_per_multiprocessor=102400, max_threads_per_block=1024, uuid="fake",
        is_integrated=False)

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.set_device = lambda d: None
    torch.cuda.get_device_properties = lambda d=None: _props
    torch.cuda.get_device_capability = lambda d=None: (8, 6)
    torch.cuda.get_device_name = lambda d=None: _props.name
    torch.cuda.synchronize = lambda d=None: None
    torch.cuda.is_bf16_supported = lambda *a, **k: True
    torch.cuda._is_compiled = lambda: True
    torch.cuda.is_initialized = lambda: True
    torch.cuda._lazy_init = lambda: None
    torch.cuda._exchange_device = lambda d: 0
    torch.cuda._maybe_exchange_device = lambda d: 0
    torch.cuda.get_rng_state = lambda device="cuda": torch.zeros(16, dtype=torch.uint8)
    torch.cuda.set_rng_state = lambda new_state, device="cuda": None
    torch.cuda.random.get_rng_state = torch.cuda.get_rng_state
    torch.cuda.random.set_rng_state = torch.cuda.set_rng_state
    torch.cuda.memory_allocated = lambda d=None: 0
    torch.cuda.max_memory_allocated = lambda d=None: 0
    torch.cuda.mem_get_info = lambda d=None: (8 * 1024 ** 3, 12 * 1024 ** 3)
    torch._C._cuda_getDevice = lambda: 0
    torch._C._cuda_getDeviceCount = lambda: 1

    from torch._dynamo import device_interface as di
    C = di.CudaInterface
    for n in ("current_device", "set_device", "device_count", "synchronize", "get_device_properties",
              "is_bf16_supported", "exchange_device", "maybe_exchange_device"):
        src = {"exchange_device": "_exchange_device", "maybe_exchange_device": "_maybe_exchange_device"}.get(n, n)
        setattr(C, n, staticmethod(getattr(torch.cuda, src)))
    C.is_available = staticmethod(lambda: True)
    C.get_compute_capability = staticmethod(lambda d=None: 86)
    C.Worker.get_device_properties = staticmethod(lambda d=None: _props)
    C.Worker.current_device = staticmethod(lambda: 0)
    C.Worker.set_device = staticmethod(lambda d: None)
    import torch._subclasses.fake_tensor as _ft
    _ft.init_gpu_context = lambda d: None
    from torch.utils._mode_utils import no_dispatch
    import torch.utils._pytree as _pt

    def _fallback(fake_mode, func, flat_args, args_spec, orig_exc):
        print("FAKE FALLBACK on CPU:", func, flush=True)
        with no_dispatch():
            def to_real(e):
                if isinstance(e, torch.Tensor) and fake_mode.is_our_fake(e):
                    return torch.empty_strided(e.shape, e.stride(), dtype=e.dtype, device="cpu").zero_()
                if isinstance(e, torch.device) and e.type == "cuda":
                    return torch.device("cpu")
                return e
            a, k = _pt.tree_unflatten([to_real(x) for x in flat_args], args_spec)
            r = func(*a, **k)

        def map_out(e):
            if isinstance(e, torch.Tensor):
                meta = torch.empty_strided(e.shape, e.stride(), dtype=e.dtype, device="meta")
                return _ft.FakeTensor(fake_mode, meta, torch.device("cuda", 0))
            return e
        return _pt.tree_map(map_out, r)
    _ft.run_fallback_kernel = _fallback
    # python-level tensor METHODS on a fake cuda tensor hit a C++ CUDA device guard: give the prims whose
    # meta is written with methods a factory-only meta
    import torch._inductor.inductor_prims as _ip
    _ip.force_stride_order.prim_meta_impl = (
        lambda x, stride: torch.empty_strided(x.shape, stride, dtype=x.dtype, device=x.device))
    A = torch.ops.aten
    _ip.fma.prim_meta_impl = lambda a, b, c: A.add.Tensor(A.mul.Tensor(a, b), c)
    _ip.lookup_seed.prim_meta_impl = lambda seeds, index: A.clone.default(A.select.int(seeds, 0, index))

    def _psm(x, dim):
        m = A.amax.default(x, [dim], True)
        return m, A.sum.dim_IntList(A.exp.default(A.sub.Tensor(x, m)), [dim], True)
    _ip.prepare_softmax_online.prim_meta_impl = _psm
    import torch._inductor.fx_passes.reinplace as _rp

    def _igs(inp, src, view_ops):
        tmp = inp
        for view in view_ops:
            fa, fk = _pt.tree_map(lambda n: n.meta["val"] if isinstance(n, torch.fx.Node) else n, (view.args, view.kwargs))
            tmp = view.target(tmp, *fa, **fk)
        A.copy_.default(tmp, src)
        return inp
    _rp._inplace_generalized_scatter = _igs
    _rp._generalized_scatter = lambda inp, src, view_ops: _igs(A.clone.default(inp), src, view_ops)
    import torch.utils._triton as _ut
    _ut.has_triton.cache_clear()
    for name, mod in list(sys.modules.items()):
        if not name.startswith("torch._inductor") and not name.startswith("torch.utils"):
            continue
        try:
            f = mod.__dict__.get("has_triton")
            if f is not None and hasattr(f, "cache_clear"):
                f.cache_clear()
        except Exception:
            pass


import json  # noqa: E402

from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # noqa: E402

from agents.model.extractor_compiles_test import _build_production_extractor  # noqa: E402

overrides = {}
for kv in args.override:
    k, v = kv.split("=", 1)
    overrides[k] = json.loads(v)
if args.speed_cum_mm:
    import inspect as _insp
    import textwrap as _tw
    import agents.model.damage_op_speed as _dos
    _src = _tw.dedent(_insp.getsource(_dos.DamageOperatorSpeed._opp_speed_mix))
    _old = "w.to(torch.float64).cumsum(-1)"
    assert _old in _src
    _src = _src.replace(_old, "(w.to(torch.float64) @ torch.ones(w.shape[-1], w.shape[-1], dtype=torch.float64, "
                              "device=w.device).triu())")
    _ns = dict(vars(_dos))
    exec(compile(_src, _dos.__file__, "exec"), _ns)
    _dos.DamageOperatorSpeed._opp_speed_mix = _ns["_opp_speed_mix"]
    print("COUNTERFACTUAL: speed-mixture cumsum -> triangular matmul", flush=True)
fe, layout = _build_production_extractor(**overrides)
D = layout["total_dim"]
obs = {"observation": torch.rand(args.batch, D, generator=torch.Generator().manual_seed(7))}

import copy  # noqa: E402

captured = []   # one dict per inner compile: the forward graph, then (train mode) the backward graph


def cap_inner(gm, example_inputs, **kw):
    from functorch.compile import make_boxed_func
    captured.append({"gm": gm, "inputs": list(example_inputs), "kw": dict(kw)})
    print("CAPTURED inner compile", len(captured), {k: v for k, v in kw.items() if k in ("is_backward", "is_inference")},
          "nodes", len(gm.graph.nodes), flush=True)
    return make_boxed_func(gm.forward)


def cap(gm, example_inputs):
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs, inner_compile=cap_inner)


import torch._functorch.config as _fc  # noqa: E402
_fc.enable_autograd_cache = False
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = False
if args.mode == "train":
    import contextlib
    ctx_factory = contextlib.nullcontext
    fe.train()
    out = torch.compile(fe.forward, backend=cap, fullgraph=True)(obs)
    pi, vf = out
    (pi.square().mean() + vf.square().mean()).backward()     # the test's `_backward_through`
else:
    ctx_factory = torch.no_grad
    with torch.no_grad():
        torch.compile(fe.forward, backend=cap, fullgraph=True)(obs)
print("captured graphs:", len(captured), flush=True)
_patch_cuda()
cpu = torch.device("cpu")
cuda = torch.device("cuda", 0)


def swap(a):
    if isinstance(a, torch.device) and a.type == "cpu":
        return cuda
    if isinstance(a, (list, tuple)):
        return type(a)(swap(x) for x in a)
    if isinstance(a, dict):
        return {k: swap(v) for k, v in a.items()}
    return a


from torch._inductor import config as icfg  # noqa: E402
from torch._inductor.runtime import triton_heuristics as th  # noqa: E402

icfg.compile_threads = 1
_which = {"graph": "?"}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_diag_plugin as tdp  # noqa: E402  (the SAME patch the GPU script loads as a pytest plugin)

tdp.install(args.out, keep_going=True, label=lambda: _which["graph"], compile_only=True)
failures = tdp.STATE["failures"]
compiled_kernels = []
_patched = th.CachingAutotuner.precompile


def _count(self, *a, **k):
    compiled_kernels.append((_which["graph"], self.fn.__name__))
    return _patched(self, *a, **k)


th.CachingAutotuner.precompile = _count

from torch._inductor.compile_fx import compile_fx_inner  # noqa: E402
from torch._guards import TracingContext, tracing  # noqa: E402

fake = next((x.fake_mode for x in captured[0]["inputs"] if hasattr(x, "fake_mode")), None) or FakeTensorMode()
for gi, capd in enumerate(captured):
    gm = capd["gm"]
    label = "backward" if capd["kw"].get("is_backward") else "forward"
    _which["graph"] = label
    n_swapped = 0
    for node in gm.graph.nodes:
        na, nk = swap(node.args), swap(node.kwargs)
        if na != node.args or nk != node.kwargs:
            n_swapped += 1
        node.args, node.kwargs = na, nk
    # non-scalar tensor CONSTANTS cannot be real CUDA tensors here: lift them to graph inputs (fake cuda)
    extra_inputs = []
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    last_ph = placeholders[-1]
    for node in list(gm.graph.nodes):
        if node.op == "get_attr":
            val = getattr(gm, node.target)
            if isinstance(val, torch.Tensor) and val.dim() > 0:
                with gm.graph.inserting_after(last_ph):
                    ph = gm.graph.placeholder(f"lifted_{node.target}")
                last_ph = ph
                node.replace_all_uses_with(ph)
                gm.graph.erase_node(node)
                extra_inputs.append(val)
    gm.graph.lint()
    gm.recompile()
    print(f"[{label}] aten graph: {len(gm.graph.nodes)} nodes, {n_swapped} device rewrites, "
          f"{len(capd['inputs'])} inputs, lifted constants {len(extra_inputs)}", flush=True)
    with open(os.path.join(args.out, f"aten_graph_{label}.py"), "w") as fh:
        fh.write(gm.code)
    metas = [torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="meta") if isinstance(x, torch.Tensor) else x
             for x in list(capd["inputs"]) + extra_inputs]
    inputs = [FakeTensor(fake, m, cuda) if isinstance(m, torch.Tensor) else m for m in metas]
    kw = dict(capd["kw"])
    kw.pop("cudagraphs", None)
    n0 = len(compiled_kernels)
    with ctx_factory(), tracing(TracingContext(fake)), fake:
        try:
            compile_fx_inner(gm, inputs, **kw)
        except Exception:
            import traceback
            print(f"[{label}] COMPILE_FX_INNER raised:", flush=True)
            traceback.print_exc()
    print(f"[{label}] kernels compiled: {len(compiled_kernels) - n0}", flush=True)
print("KERNELS COMPILED", len(compiled_kernels))
print("FAILURES", len(failures))
for f in failures:
    print("  ", f["graph"], f["kernel"], f["configs"])
json.dump({"failures": failures, "kernels": compiled_kernels}, open(os.path.join(args.out, "failures.json"), "w"), indent=1)
