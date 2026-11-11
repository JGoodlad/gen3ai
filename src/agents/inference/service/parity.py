"""The service's PARITY GATE: every served decision equals the eager policy's, at the compile gate's bars.

The reference is the policy's OWN sb3 path (``decision.policy_reference``) on the same weights and
device; the rows are the committed REAL-observation fixture (``compile_parity_obs.npz``, never zeros
— zeros hid a 70%-argmax-agreement miscompile, ledger 2026-09-28). The bars are
``compile_trainer.decision_verdicts``'s, imported rather than copied, so the service and the compile
gate cannot drift apart: legal log-probs 1e-3 and V 1e-4 at fp32 matmul precision ('highest'), the only
precision (TF32 was retired, deletion pass K2) — a process at any other precision is REFUSED.

Four further checks the compile gate does not need:

* the MASK CONTRACT: every illegal entry is exactly ``-inf`` and every legal one finite;
* GREEDY: the served argmax equals eager's on every row whose eager top-2 legal margin exceeds the
  NEAR-TIE BAND — 2x the legal log-prob bar (``parity_probe.NEAR_TIE_BAND``). Rows inside the band are
  NEAR-TIES (real and fresh policies both have EXACT ties, measured) — they are counted and reported,
  and the served action must still be one of eager's near-top actions, so a tie can never hide a wrong
  choice;
* SELECTION TIES (`gen3_t2_parity_tie_judge_v1`, root rule 8; X60-T s3501's FATAL, 2026-11-06): the forward
  SELECTS (the X5 hypothesis order's stable argsort, the candidate top-k, thresholds — every one declared in
  `agents/model/selection_sites.py`). A fixture row whose eager forward sits within a rounding error of a
  cutoff resolves it either way under any other float arithmetic — a compiled graph whose autotuned launch
  configs pick another reduction order is enough — and its log-probs then move by far more than the bar while
  both forwards are correct (s3501: rows 11 / 12, an opponent's Hidden Power Grass vs Ice at the K = 6 move-seat cut,
  a relative margin of 1.6e-6, |dlogp| 0.14 — eager with the pair flipped equals the served rows to 1.2e-5). When
  the as-is judgement FAILS, the gate re-judges through `tie_margins.resolve_ties` (K9(b)'s rule, the one
  implementation the opponent gate shares, at `consistency.FP32_TIE_EPS`, read not copied): a row at
  no tie is judged exactly as before; a tied row whose only near element is one isolated sort pair or one
  threshold element (`TieMargins.flippable`) is judged against BOTH resolutions (`TieFlip`) and passes on
  either; any other tied row is EXCLUDED and counted (`ParityReport.tie_excluded`). The margins come from the
  EAGER forward on the gate's own weights and rows, so no miscompile can move a row into the exclusion;
* PADDING: callers run every bucket full AND partially filled (pad rows are discarded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    #: gen3_t2_parity_tie_judge_v1: rows that passed only against the OTHER resolution of their selection tie,
    #: and tied rows the flip cannot express (EXCLUDED, not judged). Both 0 unless the as-is judgement failed.
    tie_flipped: int = 0
    tie_excluded: int = 0

    def line(self) -> str:
        ties = (f", selection ties {self.tie_flipped} flip-judged / {self.tie_excluded} excluded"
                if self.tie_flipped or self.tie_excluded else "")
        return (f"{self.where}: {self.rows} rows, max|dlogp| {self.legal_logprob_max:.2e}, "
                f"max|dV| {self.value_max:.2e}, near-ties {self.near_ties}{ties} [{self.path}]")


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
    its weights may pass (`InferenceService._gate`). ``path`` is recorded on the report.

    A failure is re-judged once with the rows at a SELECTION TIE resolved by K9(b)'s rule (module docs,
    `gen3_t2_parity_tie_judge_v1`); it stands, naming both judgements, unless every judged row then passes."""
    s_logp, s_value, s_greedy = (t.detach() for t in served)
    refusal = unmeasured_precision()
    if refusal is not None:
        raise ParityFailure(f"{where}: {refusal}")
    e_logp, e_value = policy_reference(policy, obs, mask)

    illegal_ok = bool(torch.isneginf(s_logp[~mask]).all()) if (~mask).any() else True
    legal_ok = bool(torch.isfinite(s_logp[mask]).all())
    if not (illegal_ok and legal_ok):
        raise ParityFailure(f"{where}: the mask contract is broken (illegal entries all -inf: "
                            f"{illegal_ok}; legal entries all finite: {legal_ok})")
    every = torch.ones(mask.shape[0], dtype=torch.bool, device=mask.device)
    try:
        lines, near, d_lp, d_v = _judge_rows(where, mask, (s_logp, s_value, s_greedy), (e_logp, e_value),
                                             every, allow_vacuous)
        flipped = excluded = 0
    except VacuousParity:
        raise
    except ParityFailure as first:
        try:
            tied = _resolve_ties(where, policy, obs, mask, (s_logp, s_value), (e_logp, e_value))
        except ParityFailure as why:        # the first failure stands, with the reason it could not be re-judged
            raise ParityFailure(f"{first} (its selection ties could not be re-judged: {why})") from first
        if tied is None:
            raise
        ref_logp, ref_value, rows, flipped, excluded = tied
        try:
            lines, near, d_lp, d_v = _judge_rows(where, mask, (s_logp, s_value, s_greedy), (ref_logp, ref_value),
                                                 rows, allow_vacuous)
        except ParityFailure as second:     # a vacuous remainder is no pass either: the first failure stands
            raise ParityFailure(f"{first} — and with its selection ties resolved by K9(b)'s rule ({flipped} row(s) "
                                f"flip-judged, {excluded} excluded): {second}") from first
    return ParityReport(where=where, rows=int(obs.shape[0]), legal_logprob_max=d_lp,
                        value_max=d_v, near_ties=near, lines=tuple(lines),
                        path=path, tie_flipped=flipped, tie_excluded=excluded)


def _judge_rows(where: str, mask: torch.Tensor, served: Tuple[torch.Tensor, ...],
                ref: Tuple[torch.Tensor, torch.Tensor], rows: torch.Tensor, allow_vacuous: bool
                ) -> Tuple[List[str], int, float, float]:
    """The bars and the greedy rule on ``rows`` (bool ``[n]``) against ``ref = (logp, value)``: returns
    ``(rule lines, near-ties, max|dlogp|, max|dV|)`` or raises `ParityFailure` / `VacuousParity`."""
    from agents.model.compile_trainer import (CompileTrainerError, VacuousCompileParityError,
                                              decision_verdicts)

    s_logp, s_value, s_greedy = served[0][rows], served[1][rows], served[2][rows]
    e_logp, e_value, m = ref[0][rows], ref[1][rows], mask[rows]
    if not m.shape[0]:
        raise VacuousParity(f"{where}: every row sits at a selection tie the flip cannot express — nothing to judge")
    eager = {"legal_logprob": _legal(e_logp, m), "value": e_value}
    comp = {"legal_logprob": _legal(s_logp, m), "value": s_value.float()}
    try:
        lines = decision_verdicts(eager=eager, compiled=comp, allow_vacuous=allow_vacuous)
    except VacuousCompileParityError as exc:
        raise VacuousParity(f"{where}: {exc}") from exc
    except CompileTrainerError as exc:
        raise ParityFailure(f"{where}: {exc}") from exc

    band = NEAR_TIE_BAND
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
    return lines, int((~decisive).sum()), d_lp, d_v


def _rows_within(a: Tuple[torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor],
                 mask: torch.Tensor) -> np.ndarray:
    """[n] bool: the row's legal log-probs AND its V agree within the decision bars (`compile_trainer._FP32_TOL`)."""
    from agents.model.compile_trainer import _FP32_TOL

    d_lp = (_legal(a[0], mask) - _legal(b[0], mask)).abs().amax(dim=-1)
    d_v = (a[1].float().reshape(-1) - b[1].float().reshape(-1)).abs()
    ok = (d_lp < _FP32_TOL["legal_logprob"]) & (d_v < _FP32_TOL["value"])
    return ok.cpu().numpy()


def _resolve_ties(where: str, policy: object, obs: torch.Tensor, mask: torch.Tensor,
                  served: Tuple[torch.Tensor, torch.Tensor], eager: Tuple[torch.Tensor, torch.Tensor]
                  ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
    """K9(b)'s tie rule on the gate's rows (`tie_margins.resolve_ties`, the ONE implementation the opponent gate
    shares): ``(ref logp, ref value, judged rows, flip-judged, excluded)``, or None when no row is tied. Raises
    `ParityFailure` when the re-read cannot be trusted (`TieMarginError`)."""
    from agents.training.rust_rollout.tie_margins import TieMarginError, resolve_ties

    def run(mode: object) -> Dict[str, torch.Tensor]:
        with mode:                                              # type: ignore[attr-defined]
            lp, v = policy_reference(policy, obs, mask)
        return {"logp": lp, "value": v}

    def within(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> np.ndarray:
        return _rows_within((a["logp"], a["value"]), (b["logp"], b["value"]), mask)

    try:
        res = resolve_ties(int(obs.shape[0]), run, {"logp": served[0], "value": served[1]},
                           {"logp": eager[0], "value": eager[1]}, within)
    except TieMarginError as exc:
        raise ParityFailure(f"{where}: {exc}") from exc
    if res is None:
        return None
    rows = torch.as_tensor(res.judged, device=mask.device)
    return res.ref["logp"], res.ref["value"], rows, res.flipped, res.excluded


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
