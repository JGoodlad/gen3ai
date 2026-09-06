"""CORRECTION 3 — a cluster bootstrap that CANNOT under-sample its own array.

The v1 scripts drew one index matrix per *expected* cluster count and reused it by name. In
`content_locality/v8_era_locality.py` that went wrong: `own_all` holds one cell per
(teacher, taught team) pair — 10 + 3 + 10 = **23** — while `bsT` was drawn as
`rng.integers(0, nT, ...)` with `nT = len(taught_union) = 22` (the DEDUPED union, because one team
is taught by two teachers). So the 23rd cell, `defensive10`'s last team, could never be drawn and
the pooled-L CI (`primary_A_era`) was computed on 22 of its 23 clusters. The headline
sibling-control R was correctly sized and is unaffected.

The fix is structural rather than a corrected literal: the index matrix is DERIVED from the array
being resampled, cached per size, and asserted to span exactly `[0, n-1]`. A caller cannot pass a
mismatched matrix because it never passes one.

Deterministic: the matrix for size `n` depends only on `(seed, n)`, so two arrays of the same
length always get the SAME draws — which is what a paired difference needs — regardless of the
order the calls happen in.
"""
import numpy as np

N_BOOT = 20000


class Boot:
    def __init__(self, seed: int = 20260905, n_boot: int = N_BOOT):
        self.seed, self.n_boot, self._cache = seed, n_boot, {}

    def idx(self, n: int) -> np.ndarray:
        """A cached ``(n_boot, n)`` resampling matrix over ``n`` clusters."""
        if n not in self._cache:
            ix = np.random.default_rng(self.seed * 1000 + n).integers(0, n, (self.n_boot, n))
            # The registered assertion: the drawn index range must equal the cluster count.
            assert ix.shape == (self.n_boot, n), f"boot matrix shape {ix.shape} != {(self.n_boot, n)}"
            assert int(ix.min()) == 0 and int(ix.max()) == n - 1, (
                f"boot indices span [{int(ix.min())},{int(ix.max())}] but there are {n} clusters "
                "— an under-sampled bootstrap silently drops a cluster from every CI")
            self._cache[n] = ix
        return self._cache[n]

    def ci(self, vals):
        """(lo, hi) 95% percentile CI of the mean, resampling over ``len(vals)`` clusters."""
        v = np.asarray(vals, dtype=float)
        b = v[self.idx(len(v))].mean(axis=1)
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    def mean_ci(self, vals):
        v = np.asarray(vals, dtype=float)
        lo, hi = self.ci(v)
        return float(v.mean()), lo, hi

    def dist(self, vals):
        """The bootstrap distribution of the mean — for differencing two arms."""
        v = np.asarray(vals, dtype=float)
        return v[self.idx(len(v))].mean(axis=1)
