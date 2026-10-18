"""`remote_sanity.arch_runs` — the build check accepts a binary-compatible cubin (2026-10-18: the RTX 4090 is sm_89 and
the cu126 wheel ships sm_86; the exact-match check FAILED a working box and cost a rented setup step)."""
from __future__ import annotations

from tools.cloud.vast.remote_sanity import arch_runs

CU126 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
CU128 = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_exact_and_same_major_lower_minor_run():
    assert arch_runs((8, 6), CU126)          # 3080 Ti / 3090
    assert arch_runs((8, 9), CU126)          # 4090: sm_86 cubins run on sm_89
    assert arch_runs((8, 0), CU126)          # A100
    assert arch_runs((12, 0), CU128)         # 5090 / 5070 Ti / 5080


def test_blackwell_needs_the_cu128_build_and_a_higher_minor_never_runs():
    assert not arch_runs((12, 0), CU126)
    assert not arch_runs((8, 0), ["sm_86"])
    assert not arch_runs((8, 6), ["compute_86"])
