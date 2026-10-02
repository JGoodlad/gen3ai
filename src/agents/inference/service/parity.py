"""The service's PARITY GATE: every served decision equals the eager policy's, at the compile gate's bars.

The reference is the policy's OWN sb3 path (``decision.policy_reference``) on the same weights and
device; the rows are the committed REAL-observation fixture (``compile_parity_obs.npz``, never zeros
— zeros hid a 70%-argmax-agreement miscompile, ledger 2026-09-28). The bars are
``compile_trainer.decision_verdicts``'s, imported rather than copied, so the service and the compile
gate cannot drift apart: legal log-probs 1e-3 and V 1e-4 at fp32 matmul precision ('highest'), the only
precision (TF32 was retired, deletion pass K2) — a process at any other precision is REFUSED.

Three further checks the compile gate does not need:

* the MASK CONTRACT: every illegal entry is exactly ``-inf`` and every legal one finite;
* GREEDY: the served argmax equals eager's on every row whose eager top-2 legal margin exceeds the
  NEAR-TIE BAND — 2x the legal log-prob bar (``parity_probe.NEAR_TIE_BAND``). Rows inside the band are
  NEAR-TIES (real and fresh policies both have EXACT ties, measured) — they are counted and reported,
  and the served action must still be one of eager's near-top actions, so a tie can never hide a wrong
  choice;
* PADDING: callers run every bucket full AND partially filled (pad rows are discarded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch

from agents.inference.service.decision import policy_reference
from agents.inference.service.spec import ParityFailure, VacuousParity
from agents.model.parity_probe import NEAR_TIE_BAND, unmeasured_precision



@dataclass
class ParityReport:
    where: str
    rows: int
    legal_logprob_max: float
    value_max: float
    near_ties: int
    lines: Tuple[str, ...]
    #: WHICH PATH produced this verdict (gen3_parity_perturb_ladder_v1): ``"real"`` (the real
    #: weights, vacuity guard ON — informative on their own), ``"perturbed seed=S scale=X"`` (the
    #: informative rung of the ladder, guard ON), or ``"real (vacuity waived)"`` (the real weights
    #: judged after that rung passed on the same slot and graph).
    path: str = "real"

    def line(self) -> str:
        return (f"{self.where}: {self.rows} rows, max|dlogp| {self.legal_logprob_max:.2e}, "
                f"max|dV| {self.value_max:.2e}, near-ties {self.near_ties} [{self.path}]")


def fixture_rows(obs_dim: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` rows of the committed real-obs fixture (cycled), or `ParityFixtureError`."""
    from agents.model.compile_parity_fixture import load_parity_rows

    obs, mask = load_parity_rows(obs_dim)
    idx = np.arange(int(n)) % len(obs)
    return obs[idx], mask[idx]


def judge(*, where: str, policy: object, obs: torch.Tensor, mask: torch.Tensor,
          served: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
          allow_vacuous: bool = False, path: str = "real") -> ParityReport:
    """Compare ``served = (logp, value, greedy)`` for ``obs``/``mask`` against the eager reference.
    Raises `ParityFailure` naming ``where`` and the quantity; returns the report on a pass.

    A comparison that cannot bite — the eager legal log-probs constant within every row (a FRESH
    policy) or V constant across rows — raises `VacuousParity` (a `ParityFailure`) unless
    ``allow_vacuous``, which only a caller that has judged the SAME slot on a seeded perturbation of
    its weights may pass (`InferenceService._gate`). ``path`` is recorded on the report."""
    from agents.model.compile_trainer import (CompileTrainerError, VacuousCompileParityError,
                                              decision_verdicts)

    s_logp, s_value, s_greedy = (t.detach() for t in served)
    refusal = unmeasured_precision()
    if refusal is not None:
        raise ParityFailure(f"{where}: {refusal}")
    band = NEAR_TIE_BAND
    e_logp, e_value = policy_reference(policy, obs, mask)

    illegal_ok = bool(torch.isneginf(s_logp[~mask]).all()) if (~mask).any() else True
    legal_ok = bool(torch.isfinite(s_logp[mask]).all())
    if not (illegal_ok and legal_ok):
        raise ParityFailure(f"{where}: the mask contract is broken (illegal entries all -inf: "
                            f"{illegal_ok}; legal entries all finite: {legal_ok})")
    eager = {"legal_logprob": _legal(e_logp, mask), "value": e_value}
    comp = {"legal_logprob": _legal(s_logp, mask), "value": s_value.float()}
    try:
        lines = decision_verdicts(eager=eager, compiled=comp, allow_vacuous=allow_vacuous)
    except VacuousCompileParityError as exc:
        raise VacuousParity(f"{where}: {exc}") from exc
    except CompileTrainerError as exc:
        raise ParityFailure(f"{where}: {exc}") from exc

    top2 = e_logp.topk(min(2, e_logp.shape[-1]), dim=-1).values
    margin = torch.where(torch.isfinite(top2[:, 1]), top2[:, 0] - top2[:, 1],
                         torch.full_like(top2[:, 0], float("inf")))
    decisive = margin > band
    e_greedy = e_logp.argmax(-1)
    bad = decisive & (s_greedy != e_greedy)
    if bool(bad.any()):
        raise ParityFailure(f"{where}: greedy action differs from eager on {int(bad.sum())} "
                            f"decisive rows (top-2 margin > the near-tie band "
                            f"{band:g})")
    chosen = e_logp.gather(1, s_greedy.view(-1, 1)).squeeze(1)
    off_top = (~decisive) & (chosen < top2[:, 0] - band)
    if bool(off_top.any()):
        raise ParityFailure(f"{where}: on {int(off_top.sum())} near-tie rows the served action is "
                            "not one of eager's near-top actions")
    d_lp = float((comp["legal_logprob"] - eager["legal_logprob"]).abs().max())
    d_v = float((comp["value"] - eager["value"]).abs().max())
    return ParityReport(where=where, rows=int(obs.shape[0]), legal_logprob_max=d_lp,
                        value_max=d_v, near_ties=int((~decisive).sum()), lines=tuple(lines),
                        path=path)


def _legal(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, logp, torch.zeros_like(logp)).float()


def gate_slot(*, where: str, policy: object, obs_dim: int, bucket: int, device: torch.device,
              serve: Callable[[np.ndarray, np.ndarray], Tuple[torch.Tensor, ...]],
              allow_vacuous: bool = False, path: str = "real") -> Tuple[ParityReport, ...]:
    """Run ``serve`` (the backend under test, one slot) on the fixture at ``bucket`` rows AND at a
    partially-filled ``bucket - 1`` rows (pad rows), and judge both. Returns both reports.
    ``allow_vacuous`` and ``path`` are `judge`'s."""
    reports = []
    for n in sorted({int(bucket), max(1, int(bucket) - 1)}, reverse=True):
        obs, mask = fixture_rows(obs_dim, n)
        served = serve(obs, mask)
        o = torch.as_tensor(obs, device=device)
        m = torch.as_tensor(mask, device=device)
        reports.append(judge(where=f"{where} rows={n}", policy=policy, obs=o, mask=m,
                             served=(served[0], served[1], served[2]),
                             allow_vacuous=allow_vacuous, path=path))
    return tuple(reports)
