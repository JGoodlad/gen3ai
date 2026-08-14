"""Session-wide pytest config: hide the GPU from the whole test suite.

Our unit/integration tests never *require* CUDA — but SB3's ``MaskablePPO`` and
``load_model_snapshot`` default to ``device="auto"``, which SB3's ``get_device`` resolves to
CUDA whenever ``torch.cuda.is_available()``. A few constructions (e.g. in
``src/agents/model/snapshot_test.py``) omit ``device=``, so on a GPU box they build a policy
on the GPU and allocate a ~300 MB CUDA context — stealing VRAM from, and contending with, a
live training run.

Hiding the GPU here makes the suite deterministic, runnable anywhere, and incapable of touching
the GPU. With CUDA invisible, ``torch.cuda.is_available()`` is False, so every ``device="auto"``
resolves to CPU. (The device-less sites in ``snapshot_test.py`` ALSO pin ``device="cpu"``
explicitly — defense in depth: this conftest is the belt, those pins are the suspenders.)

This must run before torch initializes its CUDA driver. The root ``conftest.py`` is imported by
pytest at startup, before any test module is collected/imported and therefore before the first
``torch.cuda`` call — so setting the env var here is early enough.

Escape hatch: set ``GEN3AI_TEST_ALLOW_GPU=1`` to opt out (e.g. a deliberate GPU perf check).
Note this only affects pytest-collected tests; the ``*_fuzz_test.py`` / ``*_benchmark.py``
scripts are run directly (not via pytest), so they are unaffected and still use the GPU.
"""
import os

if not os.environ.get("GEN3AI_TEST_ALLOW_GPU"):
    # Empty string => no visible CUDA device => torch.cuda.is_available() is False
    # => SB3 device="auto" resolves to CPU. Hard-set (not setdefault) so an already-exported
    # CUDA_VISIBLE_DEVICES can't silently re-expose the GPU; GEN3AI_TEST_ALLOW_GPU is the one
    # opt-out.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- BLAS thread pinning: what makes `pytest -n` a speedup instead of a 6.5x SLOWDOWN ---
#
# Measured on this 16-core box, full unit suite:
#
#     serial, unpinned      167 s
#     serial, pinned        147 s
#     -n 8,   UNPINNED      389 s   <-- 6.5x slower than pinned; `user` time 68 min vs 3 min
#     -n 4,   pinned         56 s
#
# Same cliff, same cause as the one `src/main/thread_pinning_test.py` defends for env workers: at
# the library default of one BLAS thread per core, N pytest workers spawn N x 16 competing threads
# and the box thrashes. Nothing in the suite wants multi-threaded BLAS, and someone trying `-n auto`
# would otherwise measure a slowdown and conclude parallelism does not work here — so this is set
# for them rather than written in a doc they have to find first.
#
# Set before torch is imported (BLAS reads these at init, so setting them later is a no-op) — the
# root conftest is imported by pytest at startup, which is early enough. Escape hatch:
# GEN3AI_TEST_ALLOW_THREADS=1.
if not os.environ.get("GEN3AI_TEST_ALLOW_THREADS"):
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = "1"
