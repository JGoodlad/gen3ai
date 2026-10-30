"""THE PER-PARAMETER DITHER NULL (`gen3_canary_param_null_v1`, 2026-10-30): the canary's bf16 per-parameter rule
divides by the function's own rounding sensitivity, not by one eager draw.

WHY. The bf16 rule's per-parameter envelope (`compile_regions._r1_verdict_bf16`) held every judged parameter p to

    d(C, R)_p <= K x max(d(E, R)_p, f_p)                      K = `BF16_ENVELOPE_K["param"]` 12, f_p the fp32 bar

with C the compiled bf16 gradient, E the eager bf16 one, R the fp32 eager reference and d the per-parameter relative
L2. The denominator is ONE eager draw. On a parameter whose judged gradient one to three fixture rows carry, the fp32
function itself moves that gradient, under a half-bf16-ulp weight dither, by about as much as the compiled arm does,
while the eager draw lands far inside that move: a lucky denominator, not a miscompile
(`designs/research_state/measurements/canary_slice2_2026-10-29/`). The GPU replay of 2026-10-30
(`main.ops.canary_slice_read --dither 16` on the five refusing states: X48 s3502 u200, N1-P s3505 u100, N1-P s3501
u200, X26c @482M at margin 1.2, A1-arch s3503 u200) read every state CONDITIONING: the compiled median sits at 0.09-0.55
of the fp32 null's maximum, so a real compiled-numerics fault is ruled out at those states.

THE RULE (this pin's ``canary_rule_version``, `canary_row_bar.CANARY_RULE_VERSION`). In the canary's checks, a judged
parameter's bar is

    K x max(d(E, R)_p, null_p, f_p)          null_p = max over `PARAM_NULL_SEEDS` (16) seeds s of d(R(w_s), R(w))_p

R(w_s) being the fp32 reference arm (`compile_regions._r1_reference`) on the check's OWN rows (relocated on the live
weights, as the check's arms are) at weights w_s = w x (1 + 2^-9 N(0, 1)): `main.ops.diag_bf16_canary.dithered`'s
dither exactly (name-keyed seeds, drawn on the CPU, so a GPU null here equals the replay's census null for the same
weights and rows). K, f and every other rule are unchanged.

EXACTLY LAZY. The new bar is never below the old one (it is the old denominator with one more term in the max), so a
parameter the old rule passes passes this one. The null is therefore computed only when some parameter's OLD ratio
exceeds K, and then once for every judged parameter (one fp32 forward + backward per seed). A canary the old rule
passes pays nothing and its verdict is unchanged; the verdict equals computing the null at every check.

THE FALLBACK. The null runs under a wall cap (`PARAM_NULL_CAP_S`, ``$GEN3AI_CANARY_PARAM_NULL_CAP_S``; 0 = off).
Off, truncated by the cap, in error, or short of `PARAM_NULL_SEEDS` seeds: the bars are the OLD ones (strictly
tighter), and the refusal and the record name why. A non-finite null on a parameter reads as 0 (that parameter keeps
its old bar).

WHERE. The scheduled canary and both confirmation halves, on the LIVE rung only (`compile_regions.r1_rungs
(param_null=True)`, from `compile_canary`), under the bf16 rule. The startup gate, the perturbed and accumulation
rungs, the fp32 rule, the global-gradient, loss and drift rules and the clip margin keep their bars.

THE BASIS (`testdata/canary_param_null_basis.json`, extracted by
`designs/research_state/measurements/canary_slice2_2026-10-29/extract_param_null_basis.py`): over the replay's 62,880
dithered GPU reads (5 states x 3 slices x every judged parameter x 16 seeds) the new statistic's maximum is 9.18 (X48
slice 2, the value path's one-seed compiled excursion, at its FLOOR: the old statistic reads the same), against the old
statistic's 27.9; the 14 refusing reads at the five states' own weights read at most 1.60 under it (old 12.2-52.4).

COST. Only on a would-be per-parameter refusal (4 of 166 production-lineage checks at margin 0.72): `PARAM_NULL_SEEDS` fp32 arms at the
canary's B, plus one copy of the policy's parameters and one flat gradient at a time. **UNVERIFIED on CUDA**: ~0.45 s
per arm at B = 2,048, inferred from the replay's dither read (32 s for 16 seeds x four arms and a relocation); the card's
transient peak is the reference arm's, already measured by every check.

WHAT IT DOES NOT DO: change training. Every parameter is restored bit-exactly in a ``finally``, the stashes are
released, nothing is seeded globally (each seed's noise has its own generator), and an error is recorded, never raised.
"""
from __future__ import annotations

import math
import os
import time
import zlib
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch

#: The seeds of the null: the replay's census was taken at 16 (`main.ops.canary_slice_read --dither 16`).
PARAM_NULL_SEEDS = 16
#: The dither: relative 2^-9, half a bf16 ulp (`main.ops.diag_bf16_canary.DITHER_REL`).
PARAM_NULL_REL = 2.0 ** -9
#: The wall cap on one null (seconds). A null runs only on a would-be refusal; ~10 s on CUDA (UNVERIFIED).
PARAM_NULL_CAP_S = 180.0
CAP_ENV = "GEN3AI_CANARY_PARAM_NULL_CAP_S"
#: The null's own version, in every record that carries one.
PARAM_NULL_VERSION = "gen3_canary_param_null_v1"

NullFn = Callable[[List[int]], Dict[str, Any]]


def cap_seconds() -> float:
    """``$GEN3AI_CANARY_PARAM_NULL_CAP_S`` (a non-negative number; 0 = the null OFF, the old bars), else
    `PARAM_NULL_CAP_S`. A malformed value raises (a cap is never guessed)."""
    v = os.environ.get(CAP_ENV, "").strip()
    if not v:
        return PARAM_NULL_CAP_S
    try:
        c = float(v)
    except ValueError as exc:
        raise ValueError(f"${CAP_ENV}={v!r} is not a number of seconds") from exc
    if not (c >= 0.0 and math.isfinite(c)):
        raise ValueError(f"${CAP_ENV}={v!r} is not a non-negative number of seconds")
    return c


def noise_seed(name: str, seed: int) -> int:
    """Parameter ``name``'s noise seed on dither ``seed``: `diag_bf16_canary.dithered`'s formula."""
    return (zlib.crc32(name.encode()) ^ (int(seed) * 2654435761)) & 0x7FFFFFFF


def _dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """``||a - b|| / ||b||`` in float64; NaN when ``b`` is all zero or either side is non-finite."""
    nb = float(b.norm())
    v = float((a - b).norm()) / nb if nb > 0 else float("nan")
    return v if math.isfinite(v) else float("nan")


def compute(model: Any, args: Tuple[Any, ...], ref: Mapping[str, torch.Tensor], judged: Sequence[int], *,
            seeds: Optional[int] = None, rel: float = PARAM_NULL_REL, cap_s: Optional[float] = None,
            clock: Callable[[], float] = time.perf_counter) -> Dict[str, Any]:
    """The null of every parameter index in ``judged`` (module docstring): ``{"null": {index: max over seeds},
    "per_seed": [[...] per seed], "seeds_done", "seconds", ...}``, or with ``skipped`` / ``truncated`` / ``error``.
    ``ref`` is the check's fp32 reference arm at the live weights (its ``grad`` and ``grad_sizes``); ``args`` the
    check's rows. Never raises; every parameter is restored bit-exactly."""
    from agents.model import compile_regions as cr
    k = PARAM_NULL_SEEDS if seeds is None else int(seeds)
    cap = cap_seconds() if cap_s is None else float(cap_s)
    out: Dict[str, Any] = {"version": PARAM_NULL_VERSION, "seeds": k, "rel": float(rel), "cap_s": cap}
    if cap <= 0.0:
        out["skipped"] = f"off (cap {cap:g} s; ${CAP_ENV})"
        return out
    idx = [int(i) for i in judged]
    params = list(model.policy.named_parameters())
    saved: Dict[str, torch.Tensor] = {}
    per_seed: List[List[float]] = []
    t0 = clock()
    try:
        sizes = [int(x) for x in ref["grad_sizes"].tolist()]
        base = torch.split(ref["grad"].detach().double(), sizes)
        with torch.no_grad():
            saved = {n: p.detach().clone() for n, p in params}
        for s in range(k):
            with torch.no_grad():
                for n, p in params:
                    g = torch.Generator().manual_seed(noise_seed(n, s))
                    noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
                    p.copy_(saved[n])
                    p.mul_(1.0 + rel * noise.to(p.device))
            got = torch.split(cr._r1_reference(model, args)["grad"].detach().double(), sizes)
            per_seed.append([_dist(got[i], base[i].to(got[i].device)) for i in idx])
            cr.release_gate_memory(model)
            spent = clock() - t0
            if s + 1 < k and spent * (s + 2) / (s + 1) > cap:
                out["truncated"] = (f"the cap {cap:g} s would pass at seed {s + 2} of {k} ({spent:.3g} s after "
                                    f"{s + 1}; ${CAP_ENV})")
                break
    except Exception as exc:                               # recorded, never raised: the old bars then judge
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if saved:
            with torch.no_grad():
                for n, p in params:
                    p.copy_(saved[n])
        for _n, p in params:
            p.grad = None
        out["seconds"] = clock() - t0
    out["seeds_done"] = len(per_seed)
    out["per_seed"] = per_seed
    if per_seed:
        out["null"] = {i: max(row[j] if math.isfinite(row[j]) else float("inf") for row in per_seed)
                       for j, i in enumerate(idx)}
    return out


def bind(raw: Optional[Mapping[str, Any]]) -> Tuple[Optional[Dict[int, float]], str]:
    """The null that binds (per index; a non-finite value reads 0, its old bar), or None with the fallback's reason."""
    if raw is None:
        return None, "fallback: the per-parameter null is off for this check"
    for key in ("error", "skipped", "truncated"):
        if raw.get(key):
            return None, f"fallback: the null {key}: {str(raw[key])[:200]}"
    if int(raw.get("seeds_done", 0)) != PARAM_NULL_SEEDS or raw.get("null") is None:
        return None, (f"fallback: the null has {int(raw.get('seeds_done', 0))} seeds; the census was taken at "
                      f"{PARAM_NULL_SEEDS}")
    null = {int(i): (float(v) if math.isfinite(float(v)) else 0.0) for i, v in raw["null"].items()}
    return null, "null"


class Judgement(NamedTuple):
    """The per-parameter envelope's verdict. ``worst`` = (ratio, index, compiled, eager); ``bad`` the failing
    indices; ``null`` the binding null (None = the old bars); ``source`` where the bars came from; ``record`` the
    watch's ``param_null`` block (None when no parameter needed the null)."""
    worst: Tuple[float, int, float, float]
    bad: List[int]
    null: Optional[Dict[int, float]]
    source: str
    record: Optional[Dict[str, Any]]
    ratios: Dict[int, float]

    def denominator(self) -> str:
        return ("max(eager bf16's own error, its fp32 dither null)" if self.null is not None
                else "eager bf16's own error")

    def describe(self, names: Sequence[str], err_c: Mapping[int, float], err_e: Mapping[int, float]) -> List[str]:
        """One phrase per failing parameter, the worst first."""
        out = []
        for i in sorted(self.bad, key=lambda j: -self.ratios[j]):
            n = names[i] if 0 <= i < len(names) else f"#{i}"
            null = "" if self.null is None else f", null {self.null.get(i, 0.0):.2e}"
            out.append(f"{n} compiled {err_c[i]:.2e} vs eager {err_e[i]:.2e}{null}")
        return out

    def note(self) -> str:
        """The rule line's / refusal's note on the null (empty when no parameter needed it)."""
        if self.record is None:
            return ""
        r = self.record
        return (f" [per-parameter null ({PARAM_NULL_VERSION}): {self.source}, {r.get('seeds_done', 0)} seeds in "
                f"{float(r.get('seconds', 0.0)):.1f} s, {len(r.get('params', {}))} parameter(s) past K x max(eager, "
                f"floor)]")


def _ratio(c: float, e: float, null: float, floor: float) -> float:
    r = c / max(e, null, floor)
    return r if r == r else float("inf")                    # NaN sorts worst and fails


def judge(err_c: Mapping[int, float], err_e: Mapping[int, float], floors: Mapping[int, float], k: float,
          null_fn: Optional[NullFn] = None, names: Sequence[str] = ()) -> Judgement:
    """The per-parameter envelope (module docstring): the old ratio c / max(e, f) for every judged index; when some
    exceeds ``k`` and ``null_fn`` is given, the null of every judged index (``null_fn(sorted indices)``, raw
    `compute` output) and the ratio c / max(e, null, f). Deterministic in its inputs."""
    old = {i: _ratio(err_c[i], err_e[i], 0.0, floors[i]) for i in err_c}
    over = sorted(i for i, r in old.items() if not r <= k)
    null: Optional[Dict[int, float]] = None
    record: Optional[Dict[str, Any]] = None
    if not over:
        source = "not needed: every parameter within K x max(eager, floor)"
    elif null_fn is None:
        source = "off: the old bars (not a canary check)"
    else:
        raw = null_fn(sorted(err_c))
        null, source = bind(raw)
        record = {k_: raw[k_] for k_ in ("version", "seeds", "rel", "cap_s", "seeds_done", "skipped", "truncated",
                                          "error") if k_ in raw}
        record["seconds"] = round(float(raw.get("seconds", 0.0)), 3)
        record["source"] = source
        record["params"] = {
            (names[i] if 0 <= i < len(names) else f"#{i}"): {
                "compiled": err_c[i], "eager": err_e[i], "floor": floors[i], "ratio_eager": old[i],
                **({} if null is None else {"null": null.get(i, 0.0),
                                            "ratio": _ratio(err_c[i], err_e[i], null.get(i, 0.0), floors[i])})}
            for i in over}
    ratios = old if null is None else {i: _ratio(err_c[i], err_e[i], null.get(i, 0.0), floors[i]) for i in err_c}
    worst = (-1.0, -1, 0.0, 0.0)
    for i, r in ratios.items():
        if r > worst[0]:
            worst = (r, i, err_c[i], err_e[i])
    bad = [i for i, r in ratios.items() if not r <= k]
    return Judgement(worst, bad, null, source, record, ratios)


def annotate(watch: Any, j: Judgement, names: Sequence[str]) -> None:
    """Bring a `compile_regions.GradWatch` filled by the old statistic (``fill_bf16``) to the JUDGED one when a null
    bound: the worst parameter and every watched parameter's ``ratio`` (the old one kept as ``ratio_eager``), plus the
    ``param_null`` block whenever a parameter needed the null (bound or fallen back)."""
    if j.record is not None:
        watch.param_null = j.record
    if j.null is None:
        return
    idx = {n: i for i, n in enumerate(names)}
    for n, d in watch.params.items():
        i = idx.get(n)
        if i is not None and i in j.ratios and "ratio" in d:
            d.update(ratio_eager=d["ratio"], null=j.null.get(i, 0.0), ratio=j.ratios[i])
    r, i, c, e = j.worst
    if i >= 0:
        watch.worst = {"param": names[i] if i < len(names) else f"#{i}", "ratio": r, "compiled": c, "eager": e,
                       "null": j.null.get(i, 0.0)}


def rule_record() -> Dict[str, Any]:
    """The constants every canary record names (``canary_rule.param_null``)."""
    return {"version": PARAM_NULL_VERSION, "seeds": PARAM_NULL_SEEDS, "rel": PARAM_NULL_REL,
            "cap_s": cap_seconds(), "bar": "K x max(eager, null, floor)", "lazy": True}
