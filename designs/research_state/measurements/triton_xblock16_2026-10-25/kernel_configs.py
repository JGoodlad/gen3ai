"""Compile ONE Inductor-generated Triton kernel file (`<out>/inductor/xx/<hash>.py`) per autotune config for sm_86,
on the CPU: `python3 kernel_configs.py <kernel.py>` prints OK / FAIL per config."""
import importlib.util
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
torch.cuda.current_device = lambda: 0
torch._C._cuda_getDevice = lambda: 0
path = sys.argv[1]
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/claude-1000/kc_triton")
spec = importlib.util.spec_from_file_location("k", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
ks = [v for v in vars(m).values() if isinstance(v, CachingAutotuner)]
for k in ks:
    print(k.fn.__name__, "configs:", len(k.configs))
    for c in k.configs:
        try:
            k._precompile_config(c)
            print("  OK  ", c)
        except Exception as e:
            print("  FAIL", c, repr(e)[:100])
