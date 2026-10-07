"""F-X5-4 REGRESSION PIN: `learner_golden.build_learner` builds the SAME initial state at any torch
thread count, and leaves the caller's thread count as it found it.

Mechanism (measured 2026-10-03, torch 2.8.0+cu126): SB3's `_build` re-initialises every Linear with
`torch.nn.init.orthogonal_`, a LAPACK QR whose blocked reduction order follows the BLAS thread count.
The RNG draws are identical; the QR's rounding is not (max |delta| ~1.1e-6, ~95% of a 512x512 matrix's
bytes differ). Built at 8 threads, 15 of the learner's 41 parameter groups moved, so a re-bake from a
CLI shell (no conftest pinning OMP to 1) recorded a different INIT. `build_learner` now builds inside
`_one_thread()` itself.

FAILS on revert: without the pin the 8-thread build's init hash differs from the 1-thread one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import json

import pytest

from agents.training import learner_golden as L

_PROBE = r"""
import json, sys
import torch as th
from agents.training import learner_golden as L
n = int(sys.argv[1])
th.set_num_threads(n)
m = L.build_learner()
print(json.dumps({"asked": n, "after": th.get_num_threads(), "init": L.params_sha256(m)}))
"""


def _build_at(threads: int) -> dict:
    """A FRESH interpreter, with BLAS free to use `threads` (the conftest pins OMP & co. to 1 for the
    test process; the env vars are read at BLAS init, so only a child can measure a multi-thread build)."""
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    r = subprocess.run([sys.executable, "-c", _PROBE, str(threads)], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_init_is_byte_identical_at_1_and_8_threads_and_is_the_recorded_one():
    one, eight = _build_at(1), _build_at(8)
    assert eight["after"] == 8 and one["after"] == 1, "build_learner must restore the caller's thread count"
    assert one["init"] == eight["init"], (
        "the learner golden's INIT depends on the torch thread count (F-X5-4): "
        f"1 thread {one['init'][:16]} vs 8 threads {eight['init'][:16]}")
    entry = L.load_golden()["entries"].get(L.torch_key())
    if entry is None:
        pytest.fail(f"no golden recorded for torch {L.torch_key()}")
    assert one["init"] == entry["init_params_sha256"], "the pinned build must reproduce the BANKED init bytes"


_GOLDEN_PROBE = r"""
import json, sys
import torch as th
from agents.training import learner_golden as L
th.set_num_threads(int(sys.argv[1]))
fp = L.compute(L.build_learner())
print(json.dumps({"after": th.get_num_threads(), "init": fp["init_params_sha256"],
                  "post": fp["post_params_sha256"], "losses": fp["losses"]}))
"""


def test_the_golden_is_identical_across_processes_hash_seeds_and_thread_counts():
    """X5 U6 / design §6.2: two processes with different ``PYTHONHASHSEED`` (one at 1 BLAS thread, one at
    8) compute the SAME golden (X5's fixed-mass surface) — initial and post-update bytes and every pinned loss — and it is
    the recorded one. FAILS if the build or update reads a hash-ordered set / dict iteration, or a
    thread-count-dependent reduction outside the pinned single thread."""
    runs = []
    for threads, hseed in ((1, "0"), (8, "4242")):
        env = dict(os.environ, PYTHONHASHSEED=hseed)
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[v] = str(threads)
        r = subprocess.run([sys.executable, "-c", _GOLDEN_PROBE, str(threads)], env=env,
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    a, b = runs
    assert (a["after"], b["after"]) == (1, 8), "compute must restore the caller's thread count"
    assert a["init"] == b["init"] and a["post"] == b["post"] and a["losses"] == b["losses"]
    entry = L.entry()
    assert a["init"] == entry["init_params_sha256"] and a["post"] == entry["post_params_sha256"]
