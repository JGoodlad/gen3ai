"""The reveal file's OFF/OFF play ALONE (`src/main/h2h/play_reveal_integration_test.py`: the conftest's checkpoints A, B,
12 pairs, 8 envs, its own engine and digests), under a FORCED tau solve — the cheap read of what FP32_ALT does to that
file's ``OFF_BITS`` / ``OFF_OUTCOME_ALL`` (a quarter of the file's plays), and the bisect: forced ``fp64`` must reproduce
the T62 record (`a5cd83ef…` / `26877212…`).

The solve is forced at the one call site every model path shares (`hypothesis_set.fixed_size_tau`, looked up as a
module global by `fixed_mass_presence`), before anything is built.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES= python3 reveal_off_digest.py fp32_alt
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> None:
    solve = sys.argv[1]
    from agents.model import hypothesis_set as HS
    orig = HS.fixed_size_tau
    calls = [0]

    def forced(scores, cand, k, n_iter=HS.BISECTION_ITERS, **_caller_solve):  # type: ignore[no-untyped-def]
        calls[0] += 1                                                       # the caller's `solve` is overridden
        return orig(scores, cand, k, n_iter, solve=solve)
    HS.fixed_size_tau = forced
    from agents.training import eval_ledger as L
    from agents.training.rust_eval import offline as PAR
    from agents.training.rust_rollout.testkit import build_selfcheck
    from main.h2h import play as PL
    from main.h2h import play_reveal_integration_test as T
    build_selfcheck()
    dst = Path(tempfile.mkdtemp(prefix="reveal_off_")) / "run_h2h_test"
    with PAR.declared_torch_state(1):
        a, sentinels, _cfg = PAR.build_models(dst, n_sentinels=1)
    b = sentinels[0]
    off, _ = T.run_engine([(a, b), (b, a)], "off", [])
    games = list(off[(a, b)][0].values())
    bits = T.bits_digest(games)
    outcome = L.outcome_digest(PL.outcome_vector(games), [])
    print(f"solve {solve}: OFF_BITS {bits} (recorded {T.OFF_BITS[:8]}…: {'SAME' if bits == T.OFF_BITS else 'MOVED'}); "
          f"OFF_OUTCOME_ALL {outcome} ({'SAME' if outcome == T.OFF_OUTCOME_ALL else 'MOVED'}); "
          f"fixed_size_tau calls {calls[0]}", flush=True)


if __name__ == "__main__":
    main()
