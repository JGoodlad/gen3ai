"""THE COMPILE CONFIG ROW — the torch compile-config values `compile_control` PINS, per exact torch
version, and its hash (gen3_donated_buffer_off_v1, Lane K1b; split out of `compile_control`,
P10-D 2026-10-03).

WHY ITS OWN MODULE. The row's hash is one of the three fields of the hermetic compile cache's stamp
(`agents.model.compile_cache.cache_stamp`), and EVERY trainer declares its cache — compiled or not,
`--debug` included. While the row lived in `compile_control`, the stamp imported the adapter, so
every trainer ran the adapter's torch-internals check and died with an uncaught
`CompileSentinelError` on any torch other than the one recorded row (the P10 review's finding D).
This module holds DATA only: no `torch._dynamo`, no check, no torch import at module load. The
sentinel's checks run where a learner COMPILES (`compile_control.require_supported_torch`).
`compile_control` re-exports both names, so `compile_control._COMPILE_CONFIG` is this dict.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

# THE COMPILE CONFIG ROW (gen3_donated_buffer_off_v1, Lane K1b, 2026-09-29) — the torch compile-config
# values `compile_control` PINS, per exact torch version, keyed by their full dotted path; set at
# `install()` (so before the gate's first compile and before the post-gate prewarm) and restored at
# `uninstall()`.
#
#   torch._functorch.config.donated_buffer = False. Torch 2.6+ defaults it True (2.5.1: False):
#   AOTAutograd then lets the compiled BACKWARD reuse the forward's saved activations IN PLACE,
#   which is legal only when every backward through that graph is single-use. Our learner is not:
#   the read-only probes (`grad_balance._flat_grads`, the per-term noise-scale probe)
#   call `autograd.grad(..., retain_graph=True)` on the compiled graph. The donated
#   indices are collected when the graph compiles (`aot_dispatch_autograd`) and cleared only if the
#   FIRST backward through it retains the graph (`AOTDispatchAutograd.post_compile`'s lazy backward
#   compile); the sentinel's prewarm does a plain `.backward()` first, so on torch 2.8 the first
#   update's grad-balance probe raised "This backward function was compiled with non-empty donated
#   buffers ..." (the K1 learner-bench A/B, 2026-09-29, both workers). Off, the backward keeps its
#   saved tensors as eager does — same kernels, same numbers (`compile_control_test` pins the
#   gradient bit-for-bit on vs off); measured cost on 2.8: none in time, +0.3% peak memory
#   (`designs/research_state/measurements/m5_k1/`).
#
#   torch.compiler.config.cache_key_tag = "gen3_donated_buffer_off_v1" (2.8 only; 2.5.1 has no such
#   config and never donated). Setting donated_buffer alone was NOT enough — measured 2026-09-29: the
#   Inductor FX-graph cache key (`codecache.FxGraphHashDetails`) does not include the backward's
#   donated indices, so a backward compiled WITH donation by any earlier torch-2.8 process (the
#   crashed K1 A/B, any test that compiles a train graph without this adapter) is served from the
#   shared on-disk cache to a donation-OFF compile of the same graph. Its kernels still write into
#   the saved activations, so the probe's first `retain_graph` backward corrupted saved tensor
#   [2048, 128] (version 1 -> 2) and the second raised "modified by an inplace operation"; with a
#   fresh `TORCHINDUCTOR_CACHE_DIR` the same run completed. The tag is part of that key (and of the
#   AOTAutograd cache key), so our graphs can never be served an artifact compiled under the
#   default config. A kernel that reused a saved buffer WITHOUT an ATen op would not have bumped a
#   version counter — i.e. this could have been SILENT — which is why the tag is not optional.
#   Both readers are in the hashed row above (`FxGraphHashDetails.__init__` reads the tag).
COMPILE_CONFIG: Dict[str, Dict[str, Any]] = {
    "2.8.0+cu126": {"torch._functorch.config.donated_buffer": False,
                    "torch.compiler.config.cache_key_tag": "gen3_donated_buffer_off_v1"},
}


def _torch_version() -> str:
    import torch                                   # local: the module stays import-light
    return str(torch.__version__)


def config_row_hash(version: Optional[str] = None) -> str:
    """SHA256 of this torch version's `COMPILE_CONFIG` row (canonical JSON), or of `null` for a
    torch with no row (which `compile_control`'s `install()` refuses anyway). One of the three fields
    of the hermetic compile cache's stamp (`agents.model.compile_cache`, K3): a change to what the
    adapter pins changes what Inductor compiles, and the K1b fault proved the cache KEY can omit such
    a setting — so a changed row WIPES a run's cache instead of trusting the key."""
    row = COMPILE_CONFIG.get(str(version or _torch_version()))
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=repr).encode()).hexdigest()
