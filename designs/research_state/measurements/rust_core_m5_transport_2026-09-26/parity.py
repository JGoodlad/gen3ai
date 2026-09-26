"""PARITY of the two front ends (and of thread counts, and of a rerun): the same pool seed and the
same seeded random actions must give BYTE-identical columns at every step. Exit 0 iff all equal.

    python3 parity.py [steps] [n]
"""
import hashlib
import sys

import numpy as np

import m5_loader as m5


def trace(kind, n, threads, seed, opp_external, steps, succ_every=0):
    pool = m5.make(kind, n, threads, seed, opp_external)
    rng = np.random.default_rng(seed)
    digests, decisions, episodes = [], 0, 0
    try:
        pool.reset()
        for t in range(steps):
            h = hashlib.sha256()
            for a in (pool.obs, pool.mask, pool.need, pool.reward, pool.done):
                h.update(np.ascontiguousarray(a).tobytes())
            digests.append(h.hexdigest())
            decisions += int(pool.need.sum())
            m5.random_actions(pool, rng)
            pool.step()
            episodes += int(pool.done.sum())
        if succ_every:
            pass
        digests.append(f"refusals={pool.refusals}")
    finally:
        pool.close()
    return digests, decisions, episodes


def succ_trace(kind, seed, calls, k):
    pool = m5.make(kind, 1, 1, seed, True)
    rng = np.random.default_rng(seed)
    out = []
    try:
        pool.reset()
        for _ in range(calls):
            if pool.need[0, 0]:
                pool.successors(k)
                out.append(hashlib.sha256(pool.rows[:k].tobytes() + pool.ok[:k].tobytes()).hexdigest())
            m5.random_actions(pool, rng)
            pool.step()
    finally:
        pool.close()
    return out


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    ok = True
    for opp in (True, False):
        ref, dec, eps = trace("ffi", n, 1, 7, opp, steps)
        print(f"opp_external={opp}: ffi T=1 reference: {steps} steps, {dec} decisions, {eps} episodes ended")
        for kind, t in (("ffi", 4), ("proc", 1), ("proc", 4), ("ffi", 1)):
            got, _, _ = trace(kind, n, t, 7, opp, steps)
            same = got == ref
            ok &= same
            first = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
            print(f"  {kind:4s} T={t}: {'IDENTICAL' if same else f'DIFFERS from step {first}'}")
        other, _, _ = trace("ffi", n, 1, 8, opp, steps)
        print(f"  teeth: a different seed differs: {other != ref}")
        ok &= other != ref
    a = succ_trace("ffi", 11, 60, 48)
    b = succ_trace("proc", 11, 60, 48)
    print(f"successors k=48 over {len(a)} roots: ffi == proc: {a == b}")
    ok &= a == b and len(a) > 0
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
