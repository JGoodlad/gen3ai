"""Make a PARITY check bite on FRESH weights, and REFUSE one that cannot (gen3_fresh_parity_probe_v1).

WHY THIS EXISTS. A fresh production-arch policy is a VACUOUS parity probe. The pointer head's three
scorers are zero-init (``pointer_head.py``: the cold-start policy is uniform-over-legal), so every
legal log-prob on a row is ``-log(n_legal)`` WHATEVER the extractor computed — a compiled-vs-eager
comparison of legal log-probs passes on any miscompile at all. Measured 2026-09-29 (M5 T2): an AOT
miscompile read max|dlogp| 0.0 on a fresh policy and 0.68 on a real one. It is not only the head:
on the fresh production policy 52 of 232 extractor parameters get ZERO gradient from the compile
gate's train loss (zero-init projections and the stash paths that only the pointer head reads), so
the features and gradient checks are blind to them too; a seeded perturbation brings the legal
log-prob spread from 0.0 to ~0.95 nats and routes every stash path into the decision readout.

THE TWO HALVES.

* ``require_informative`` — the FAIL-CLOSED guard: a compared quantity whose spread (within-row over
  the legal entries for log-probs, across rows for everything else) does not exceed the check's own
  tolerance cannot tell a miscompile from a match. It raises `VacuousParityError` instead of passing.
* ``perturbed_parameters`` / ``perturbed_copy`` — the deterministic PERTURBATION that makes a fresh
  policy informative: every parameter plus seeded Gaussian noise (``PERTURB_SCALE``), drawn from a
  PRIVATE CPU ``torch.Generator`` (the global RNG stream — training's — is never touched). The
  in-place form restores every parameter BIT-EXACTLY on exit and re-checks it, raising if not; it
  exists so a gate can run the SAME installed compiled graph (parameters are graph INPUTS, so a
  value change neither recompiles nor escapes the graph) on informative weights.

Every caller that compares two execution paths of a policy on weights that may be fresh uses both:
the startup compile gate (``compile_trainer``), the inference service's parity gate
(``agents.inference.service``), and the tests that build a fresh production policy.
"""
from __future__ import annotations

import contextlib
import copy
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch

#: The perturbation's seed and scale. Fixed so a gate's verdict is reproducible run to run; the
#: scale is T2's (``agents.inference.service.fixtures``), measured to make every head informative
#: on FRESH weights.
PERTURB_SEED = 20260929
PERTURB_SCALE = 0.05
#: The DECLARED perturbation LADDER (gen3_parity_perturb_ladder_v1): a gate whose comparison is
#: vacuous on the real weights tries these RUNGS in order — ``(scale, seed offset)``, the seed being
#: ``PERTURB_SEED + offset`` — and judges at the FIRST one whose comparison is informative (every
#: quantity's spread above its bar); none ⇒ it refuses. The first rung is ``(PERTURB_SCALE, 0)``, so
#: a fresh policy's verdict is unchanged.
#:
#: Why rungs above the first: a COLLAPSED critic — a win-prob head saturated at logit ≈ −8…−10 (a
#: trainee losing ~97%; ``~/gen3ai_archive/cutover_prep/fresh3``, 2026-09-30) — has real V spread
#: 6e-6, and the first rung only reaches 9.6e-5 < the 1e-4 bar; another seed at the same scale
#: (rung (0.05, 2)) reads 2.4e-4 on the gate's smallest fills.
#:
#: Why the scale is CAPPED at 0.1 and more SEEDS are tried instead: the compare bars are absolute
#: (legal log-prob 1e-3, V 1e-4), calibrated near the fresh / trained operating point, and the
#: compiled-vs-eager noise grows steeply with the scale. Measured on the GPU (RTX 3080 Ti, fp32, the
#: graph backend's compiled ``decide`` vs eager, 48 fixture rows, fresh / collapsed / trained
#: weights, 3 seeds, 2026-09-30): max |Δ log π| 9.5e-7 at 0.05, 4.5e-6 at 0.1, but 9e-5…**1.7e-3**
#: at 0.2 (over the bar: a FALSE refusal), 2.8e-2 at 0.3 and 0.66 at 0.5. So no rung goes above 0.1,
#: where the noise sits 200x under the bar. A deeply SATURATED critic (win logit ≈ −12 on every
#: row, or +9 on some bases) stays vacuous on V at every rung and is REFUSED — never passed. Table:
#: ``designs/research_state/measurements/m5_t2/PROGRESS.md`` "Flat weights".
PERTURB_MAX_SCALE = 0.1
PERTURB_LADDER = tuple((scale, k) for scale in (0.05, 0.1) for k in range(8))
#: ONE TABLE, KEYED BY THE FLOAT32 MATMUL PRECISION a gate runs at (``torch.get_float32_matmul_
#: precision()`` — the trainer's ``--matmul-precision``; gen3_precision_keyed_parity_v1, 2026-09-30).
#: Per precision: the legal LOG-PROB BAR (a healthy compiled-vs-eager |Δ log π| sits under it; the
#: greedy check's near-tie band is 2x it — a legitimate flip needs a top-2 margin below
#: |Δ top-1| + |Δ top-2|) and the ladder's SCALE CAP (rungs above it are skipped at that precision).
#:
#: * ``highest`` (fp32): bar 1e-3 — the compile gate's legal log-prob bar (>= 37x the healthy max
#:   2.7e-5 on 3,840 real rows, ``compile_trainer._FP32_TOL``, which reads it from here); cap 0.1
#:   (`PERTURB_MAX_SCALE`'s evidence above).
#: * ``high`` (TF32): bar 0.071 = 1.75 x 0.040, the LARGER of two measured healthy TF32 maxima of
#:   compiled-vs-eager |Δ log π| on trained weights — 0.040 on the learner forward over 147,456
#:   rollout rows (Lane K, K9's behaviour-check calibration, 2026-09-30) and 1.4e-2 on T2's served
#:   decision over the gate's own fixture rows (4 trained checkpoints x buckets 8 / 48 + every 0.05
#:   rung, 2,072 rows; RTX 3080 Ti, ``designs/research_state/measurements/m5_t2/PROGRESS.md``
#:   "TF32"). The 1.75x is K9's multiple on the same quantity. Cap 0.05: at the 0.1 rungs TF32's
#:   compiled graph drifts 6–24x eager's own TF32 error (5 of 138 groups), which the TF32 log-prob
#:   rule (<= 4x) refuses — a false refusal of a correct graph.
#:
#: Any other precision (``medium``) is REFUSED by every consumer: nothing has been measured there.
PRECISION_BARS: Dict[str, Tuple[float, float]] = {
    "highest": (1e-3, PERTURB_MAX_SCALE),
    "high": (0.071, 0.05),
}


def precision_bars(precision: str) -> Tuple[float, float]:
    """``(log-prob bar, ladder scale cap)`` at ``precision``; `KeyError` naming the measured ones."""
    if precision not in PRECISION_BARS:
        raise KeyError(f"float32 matmul precision {precision!r} has no measured parity bars "
                       f"(measured: {sorted(PRECISION_BARS)}) — refusing to judge with a guess")
    return PRECISION_BARS[precision]


def tie_band(precision: str) -> float:
    """The greedy check's near-tie band at ``precision``: 2 x that precision's log-prob bar."""
    return 2.0 * precision_bars(precision)[0]


def ladder_at(precision: str, ladder: "Tuple[Tuple[float, int], ...]" = PERTURB_LADDER
              ) -> "Tuple[Tuple[float, int], ...]":
    """``ladder`` without the rungs above ``precision``'s scale cap."""
    cap = precision_bars(precision)[1]
    return tuple((float(sc), int(k)) for sc, k in ladder if float(sc) <= cap)


#: A rung's seed offset k moves the seed by k x this, so the inference service's CONCURRENT gate
#: (per-slot seeds ``rung_seed(k) + slot``) never reuses a seed across rungs.
PERTURB_SEED_STRIDE = 1000


def rung_seed(k: int) -> int:
    """The perturbation seed of a ladder rung with seed offset ``k``."""
    return PERTURB_SEED + PERTURB_SEED_STRIDE * int(k)


class VacuousParityError(RuntimeError):
    """A parity check was about to compare a quantity that cannot distinguish a defect."""


def spread(key: str, t: "torch.Tensor") -> float:
    """How much ``t`` varies in the way a parity check on ``key`` needs it to.

    ``legal_logprob`` (illegal entries zeroed, the gate's convention): the max over rows of the
    WITHIN-ROW spread of the non-zero (legal) entries — a fresh pointer head gives exactly 0.0, since
    each row is ``-log(n_legal)`` everywhere. Anything else: the max over columns of the ACROSS-ROW
    spread (a 1-D tensor is one column) — a quantity that is the same on every row says nothing
    about the rows.
    """
    x = t.detach().float()
    if x.numel() == 0:
        return 0.0
    if key == "legal_logprob":
        x = x.reshape(x.shape[0], -1)
        nz = x != 0
        big = torch.where(nz, x, torch.full_like(x, float("-inf"))).max(dim=1).values
        small = torch.where(nz, x, torch.full_like(x, float("inf"))).min(dim=1).values
        row = torch.where(nz.sum(dim=1) >= 2, big - small, torch.zeros_like(big))
        return float(row.max())
    if key == "grad":
        return float(x.abs().max())
    x = x.reshape(x.shape[0], -1)
    return float((x.max(dim=0).values - x.min(dim=0).values).max())


def vacuous_keys(quantities: Mapping[str, "torch.Tensor"],
                 bars: Mapping[str, float]) -> Dict[str, float]:
    """``{key: spread}`` for every quantity in ``quantities`` whose spread is NOT above its bar
    (``bars[key]``, the check's own pass tolerance). Empty ⇔ every compared quantity is informative.
    Written ``not (s > bar)`` so a NaN spread counts as vacuous. An across-row quantity on fewer
    than two rows is not judged (it has no spread to have); the within-row log-prob rule still is."""
    out: Dict[str, float] = {}
    for key, t in quantities.items():
        if key not in bars:
            continue
        if key not in ("legal_logprob", "grad") and (t.dim() == 0 or t.shape[0] < 2):
            continue       # an ACROSS-row spread needs two rows (a 1-row padding check has one)
        s = spread(key, t)
        if not (s > float(bars[key])):
            out[key] = s
    return out


def require_informative(quantities: Mapping[str, "torch.Tensor"], bars: Mapping[str, float],
                        *, where: str = "parity check") -> None:
    """Raise `VacuousParityError` if any compared quantity's spread is not above its bar."""
    bad = vacuous_keys(quantities, bars)
    if bad:
        detail = ", ".join(f"{k} spread {v:.2e} <= bar {float(bars[k]):g}" for k, v in bad.items())
        raise VacuousParityError(
            f"{where}: VACUOUS — {detail}. A quantity that does not vary cannot distinguish a "
            f"miscompile from a match (a FRESH policy's zero-init pointer head makes every legal "
            f"log-prob -log(n_legal)). Run the check on a seeded perturbed copy "
            f"(`agents.model.parity_probe.perturbed_parameters`) instead of passing it.")


def _noise(module: torch.nn.Module, seed: int, scale: float) -> Dict[str, "torch.Tensor"]:
    """Deterministic per-parameter noise from a PRIVATE CPU generator, in ``named_parameters`` order
    (drawn on CPU so the values are device-independent and the global RNG is never advanced)."""
    g = torch.Generator().manual_seed(int(seed))
    return {n: (torch.randn(p.shape, generator=g, dtype=torch.float32) * float(scale))
            .to(device=p.device, dtype=p.dtype)
            for n, p in module.named_parameters()}


def perturb_(module: torch.nn.Module, *, seed: int = PERTURB_SEED,
             scale: float = PERTURB_SCALE) -> torch.nn.Module:
    """Add the seeded noise to every parameter of ``module`` IN PLACE (a copy you own). Returns it."""
    noise = _noise(module, seed, scale)
    with torch.no_grad():
        for n, p in module.named_parameters():
            p.add_(noise[n])
    return module


def perturbed_copy(module: torch.nn.Module, *, seed: int = PERTURB_SEED,
                   scale: float = PERTURB_SCALE) -> torch.nn.Module:
    """A deep copy of ``module`` with every parameter perturbed; ``module`` is untouched."""
    return perturb_(copy.deepcopy(module), seed=seed, scale=scale)


@contextlib.contextmanager
def perturbed_parameters(module: torch.nn.Module, *, seed: int = PERTURB_SEED,
                         scale: float = PERTURB_SCALE) -> Iterator[None]:
    """Perturb ``module``'s parameters in place for the block, then restore them BIT-EXACTLY.

    For a gate that must test the graph it will actually ship (a compiled forward, a CUDA-graph
    slot) and so cannot use a copy. Restoration is ``copy_`` from a saved clone in ``finally`` —
    also on an exception — and is then VERIFIED with ``torch.equal`` per parameter; a mismatch
    raises rather than leaving a run on weights that are not its own. ``.grad`` is left to the
    caller (a gate zeroes it anyway); no optimizer state is read or written.
    """
    params = dict(module.named_parameters())
    saved = {n: p.detach().clone() for n, p in params.items()}
    try:
        perturb_(module, seed=seed, scale=scale)
        yield
    finally:
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(saved[n])
        wrong = [n for n, p in params.items() if not torch.equal(p.detach(), saved[n])]
        if wrong:
            raise RuntimeError(f"perturbed_parameters: {len(wrong)} parameters were NOT restored "
                               f"bit-exactly (e.g. {wrong[:3]}) — refusing to continue on them")


def fresh_reason(quantities: Mapping[str, "torch.Tensor"],
                 bars: Mapping[str, float]) -> Optional[str]:
    """``None`` when every compared quantity is informative, else a one-line reason naming the
    vacuous ones — the "these weights are FRESH (or otherwise degenerate)" detector."""
    bad = vacuous_keys(quantities, bars)
    if not bad:
        return None
    return ", ".join(f"{k} spread {v:.2e}" for k, v in bad.items())
