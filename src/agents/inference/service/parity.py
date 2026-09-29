"""The service's PARITY GATE: every served decision equals the eager policy's, at the compile gate's bars.

The reference is the policy's OWN sb3 path (``decision.policy_reference``) on the same weights and
device; the rows are the committed REAL-observation fixture (``compile_parity_obs.npz``, never zeros
— zeros hid a 70%-argmax-agreement miscompile, ledger 2026-09-28). The bars are
``compile_trainer.decision_verdicts``'s, imported rather than copied, so the service and the compile
gate cannot drift apart: legal log-probs 1e-3 and V 1e-4 at fp32 matmul precision ('highest'), the
TF32 rule against an fp32 eager reference otherwise.

Three further checks the compile gate does not need:

* the MASK CONTRACT: every illegal entry is exactly ``-inf`` and every legal one finite;
* GREEDY: the served argmax equals eager's on every row whose eager top-2 legal margin exceeds the
  log-prob bar. Rows inside the bar are NEAR-TIES (real and fresh policies both have EXACT ties,
  measured) — they are counted and reported, and the served action must still be one of eager's
  near-top actions, so a tie can never hide a wrong choice;
* PADDING: callers run every bucket full AND partially filled (pad rows are discarded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from agents.inference.service.decision import policy_reference
from agents.inference.service.spec import ParityFailure

#: The greedy rule's near-tie band = the legal log-prob bar at fp32.
_TIE_BAND = 1e-3


@dataclass
class ParityReport:
    where: str
    rows: int
    legal_logprob_max: float
    value_max: float
    near_ties: int
    lines: Tuple[str, ...]

    def line(self) -> str:
        return (f"{self.where}: {self.rows} rows, max|dlogp| {self.legal_logprob_max:.2e}, "
                f"max|dV| {self.value_max:.2e}, near-ties {self.near_ties}")


class _precision:
    def __init__(self, value: str):
        self.value = value

    def __enter__(self) -> None:
        self.prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(self.value)

    def __exit__(self, *exc: object) -> None:
        torch.set_float32_matmul_precision(self.prev)


def fixture_rows(obs_dim: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` rows of the committed real-obs fixture (cycled), or `ParityFixtureError`."""
    from agents.model.compile_parity_fixture import load_parity_rows

    obs, mask = load_parity_rows(obs_dim)
    idx = np.arange(int(n)) % len(obs)
    return obs[idx], mask[idx]


def judge(*, where: str, policy: object, obs: torch.Tensor, mask: torch.Tensor,
          served: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> ParityReport:
    """Compare ``served = (logp, value, greedy)`` for ``obs``/``mask`` against the eager reference.
    Raises `ParityFailure` naming ``where`` and the quantity; returns the report on a pass."""
    from agents.model.compile_trainer import CompileTrainerError, decision_verdicts

    s_logp, s_value, s_greedy = (t.detach() for t in served)
    precision = torch.get_float32_matmul_precision()
    e_logp, e_value = policy_reference(policy, obs, mask)
    reference: Optional[Dict[str, torch.Tensor]] = None
    if precision != "highest":
        with _precision("highest"):
            r_logp, r_value = policy_reference(policy, obs, mask)
        reference = {"legal_logprob": _legal(r_logp, mask), "value": r_value}

    illegal_ok = bool(torch.isneginf(s_logp[~mask]).all()) if (~mask).any() else True
    legal_ok = bool(torch.isfinite(s_logp[mask]).all())
    if not (illegal_ok and legal_ok):
        raise ParityFailure(f"{where}: the mask contract is broken (illegal entries all -inf: "
                            f"{illegal_ok}; legal entries all finite: {legal_ok})")
    eager = {"legal_logprob": _legal(e_logp, mask), "value": e_value}
    comp = {"legal_logprob": _legal(s_logp, mask), "value": s_value.float()}
    try:
        lines = decision_verdicts(eager=eager, compiled=comp, reference=reference,
                                  precision=precision)
    except CompileTrainerError as exc:
        raise ParityFailure(f"{where}: {exc}") from exc

    top2 = e_logp.topk(min(2, e_logp.shape[-1]), dim=-1).values
    margin = torch.where(torch.isfinite(top2[:, 1]), top2[:, 0] - top2[:, 1],
                         torch.full_like(top2[:, 0], float("inf")))
    decisive = margin > _TIE_BAND
    e_greedy = e_logp.argmax(-1)
    bad = decisive & (s_greedy != e_greedy)
    if bool(bad.any()):
        raise ParityFailure(f"{where}: greedy action differs from eager on {int(bad.sum())} "
                            f"decisive rows (top-2 margin > {_TIE_BAND})")
    chosen = e_logp.gather(1, s_greedy.view(-1, 1)).squeeze(1)
    off_top = (~decisive) & (chosen < top2[:, 0] - _TIE_BAND)
    if bool(off_top.any()):
        raise ParityFailure(f"{where}: on {int(off_top.sum())} near-tie rows the served action is "
                            "not one of eager's near-top actions")
    d_lp = float((comp["legal_logprob"] - eager["legal_logprob"]).abs().max())
    d_v = float((comp["value"] - eager["value"]).abs().max())
    return ParityReport(where=where, rows=int(obs.shape[0]), legal_logprob_max=d_lp,
                        value_max=d_v, near_ties=int((~decisive).sum()), lines=tuple(lines))


def _legal(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, logp, torch.zeros_like(logp)).float()


def gate_slot(*, where: str, policy: object, obs_dim: int, bucket: int, device: torch.device,
              serve: Callable[[np.ndarray, np.ndarray], Tuple[torch.Tensor, ...]]) -> Tuple[ParityReport, ...]:
    """Run ``serve`` (the backend under test, one slot) on the fixture at ``bucket`` rows AND at a
    partially-filled ``bucket - 1`` rows (pad rows), and judge both. Returns both reports."""
    reports = []
    for n in sorted({int(bucket), max(1, int(bucket) - 1)}, reverse=True):
        obs, mask = fixture_rows(obs_dim, n)
        served = serve(obs, mask)
        o = torch.as_tensor(obs, device=device)
        m = torch.as_tensor(mask, device=device)
        reports.append(judge(where=f"{where} rows={n}", policy=policy, obs=o, mask=m,
                             served=(served[0], served[1], served[2])))
    return tuple(reports)
