"""Compile one saved Inductor kernel (optionally hand-edited) for sm_86 on the CPU, one autotune config at a time.

    python3 kernel_configs.py failing_kernel_b2.txt failing_kernel_b2.meta.json [XBLOCK,R0_BLOCK,num_warps ...]

The source is a kernel as `triton_diag_plugin.py` saved it; the meta is its `ASTSource` inputs (signature, constants,
pointer attrs). With no configs it runs the kernel's own autotune list. Prints OK / FAIL per config; Triton's
verifier message goes to stderr.
"""
import json
import linecache
import os
import re
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="kernel_configs_triton_"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from torch._inductor.runtime import triton_helpers  # noqa: E402
from torch._inductor.runtime.triton_helpers import libdevice, math as tmath  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

src_path, meta_path = sys.argv[1], sys.argv[2]
with open(meta_path) as fh:
    meta = json.load(fh)
attrs = {tuple(k): v for k, v in meta["attrs"]}
cfgs = [tuple(int(x) for x in a.split(",")) for a in sys.argv[3:]] or [
    (c["XBLOCK"], c["R0_BLOCK"], c["num_warps"]) for c in meta["configs"]]
with open(src_path) as fh:
    src = re.sub(r"(?m)^def triton_", "@triton.jit\ndef triton_", fh.read(), count=1)
name = src_path + ".jit"
linecache.cache[name] = (len(src), None, src.splitlines(True), name)     # triton reads the source back by filename
ns = {"triton": triton, "tl": tl, "triton_helpers": triton_helpers, "libdevice": libdevice, "math": tmath,
      "tl_math": tmath, "torch": torch}
exec(compile(src, name, "exec"), ns)
fn = next(v for k, v in ns.items() if k.startswith("triton_") and hasattr(v, "arg_names"))
for xb, rb, nw in cfgs:
    consts = {k: v for k, v in {**meta["constants"], "XBLOCK": xb, "R0_BLOCK": rb}.items() if k in fn.arg_names}
    try:
        triton.compile(ASTSource(fn, meta["signature"], consts, attrs), target=GPUTarget("cuda", 86, 32),
                       options={"num_warps": nw, "num_stages": 1, "debug": True, "sanitize_overflow": False})
        print(f"XBLOCK {xb} R0_BLOCK {rb} warps {nw}: OK", flush=True)
    except Exception as e:  # noqa: BLE001 - a diagnostic records whatever the compiler raised
        print(f"XBLOCK {xb} R0_BLOCK {rb} warps {nw}: FAIL {type(e).__name__} {str(e)[-300:]!r}", flush=True)
