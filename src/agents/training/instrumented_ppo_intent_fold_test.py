"""K8 — `intent_fold` (the static, traceable opponent-intent fold) EQUALS the legacy inline block.

THE ORACLE is a VERBATIM copy (taken 2026-09-30 from `f8092cbc`) of `ppo.py::train()`'s
opponent-intent block and of every `agents.model.opp_intent` function it calls — copied, not
imported, so the oracle survives the refactor that replaces the inline block. Only non-semantic text
was dropped: docstrings and full-line comments, and the block's two
`from agents.model.opp_intent import (...)` statements, whose names resolve to the copies below.

Acceptance (`designs/endstate/program_rust_core.md` K8): in float64 every present metric and both
terms equal the oracle to 1e-12 and the gradient w.r.t. the alpha/beta logits to 1e-12; the key set
the oracle emits is exactly the set of weight-1 metrics; in float32 terms agree to 1e-6 relative
and gradients to 1e-5. The ONE float64 exception is a quantity the LEGACY code computes in float32
by an explicit `.float()` (the set-valued loss): its float32 reduction order is not reproducible
without the boolean subset, so it is held to float32 precision (`_SV_F64_TOL`, see the test).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch as th

from agents.training.instrumented_ppo.intent_fold import (IntentFoldOut, intent_fold,
                                                          intent_fold_tensors)

# =================================================================================== THE ORACLE
INTENT_IGNORE = -100
OPP_CLASS_NAMES = {0: "bot", 1: "pool", 2: "stable", 3: "exploiter"}
OPP_CLASS_BOT = 0


def match_seats_to_move_num(seat_nums: torch.Tensor, chosen_num: torch.Tensor,
                            kind: torch.Tensor, n_seats: int) -> torch.Tensor:
    B = chosen_num.shape[0]
    tgt = torch.full((B,), INTENT_IGNORE, dtype=torch.long, device=seat_nums.device)
    hit = seat_nums == chosen_num[:, None]                                       # [B,K]
    any_hit = hit.any(dim=-1)
    first = hit.float().argmax(dim=-1)                                           # first match
    is_move = kind == 0
    tgt = torch.where(is_move & any_hit, first.long(), tgt)
    tgt = torch.where(kind == 1, torch.full_like(tgt, n_seats), tgt)
    return tgt


def resolve_believed_slot_by_content(
    species_logits: torch.Tensor,
    believed_mask: torch.Tensor,
    switch_species: torch.Tensor,
    min_prob: float = 0.05,
) -> torch.Tensor:
    B, n_slots, _ = species_logits.shape
    probs = torch.softmax(species_logits.float(), dim=-1)
    idx = switch_species.clamp(min=0, max=probs.shape[-1] - 1)
    p_species = probs.gather(-1, idx[:, None, None].expand(B, n_slots, 1)).squeeze(-1)  # [B,slots]
    p_species = p_species.masked_fill(believed_mask < 0.5, -1.0)
    best_p, best_j = p_species.max(dim=-1)
    ok = (switch_species > 0) & (best_p >= min_prob)
    return torch.where(ok, best_j, torch.full_like(best_j, INTENT_IGNORE))


def info_gain_nats(logits: torch.Tensor, target: torch.Tensor) -> float:
    tgt = target.reshape(-1)
    n_classes = logits.shape[-1]
    counts = torch.bincount(tgt, minlength=n_classes).float()
    p = counts / counts.sum().clamp_min(1.0)
    nz = p > 0
    h_marginal = float(-(p[nz] * p[nz].log()).sum())
    ce_model = float(torch.nn.functional.cross_entropy(logits, tgt))
    return h_marginal - ce_model


def intent_label_weights(opp_class: Optional[torch.Tensor], bot_weight: float,
                         like: torch.Tensor) -> Optional[torch.Tensor]:
    if opp_class is None or bot_weight == 1.0:
        return None
    oc = opp_class.reshape(-1).to(like.device)
    ones = torch.ones(oc.shape, dtype=like.dtype, device=like.device)
    return torch.where(oc == OPP_CLASS_BOT, ones * float(bot_weight), ones)


def set_valued_switch_loss(beta_logits: torch.Tensor, believed_mask: torch.Tensor,
                           rows: torch.Tensor) -> Optional[torch.Tensor]:
    finite = ~torch.isneginf(beta_logits)
    avail = (believed_mask > 0.5) & finite                       # [B,6] believed AND legal
    rows = rows & (avail.sum(dim=-1) > 0)
    if not bool(rows.any()):
        return None
    p = torch.softmax(beta_logits[rows].float(), dim=-1)         # illegal slots are -inf -> 0
    mass = (p * avail[rows].float()).sum(dim=-1)
    return -(mass.clamp_min(1e-8).log()).mean()


def _alpha_subset_metrics(logits: torch.Tensor, tgt: torch.Tensor, k: int,
                          sfx: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    pred = logits.argmax(dim=-1)
    out[f"opp_intent/alpha_acc{sfx}"] = float((pred == tgt).float().mean())
    is_sw = tgt == k
    said_sw = pred == k

    if int(is_sw.sum()):
        out[f"opp_intent/alpha_switch_recall{sfx}"] = float((pred[is_sw] == k).float().mean())
    if int(said_sw.sum()):
        out[f"opp_intent/alpha_switch_precision{sfx}"] = float((tgt[said_sw] == k).float().mean())
    if int((~is_sw).sum()):
        out[f"opp_intent/alpha_move_kind_recall{sfx}"] = float((pred[~is_sw] != k).float().mean())
    if int((~said_sw).sum()):
        out[f"opp_intent/alpha_move_kind_precision{sfx}"] = float(
            (tgt[~said_sw] != k).float().mean())
    out[f"opp_intent/alpha_pred_switch_rate{sfx}"] = float(said_sw.float().mean())

    if int((~is_sw).sum()):
        mv = ~is_sw
        move_logits = logits[mv][:, :k]
        mv_tgt = tgt[mv]
        out[f"opp_intent/alpha_move_recall_top1{sfx}"] = float(
            (move_logits.argmax(dim=-1) == mv_tgt).float().mean())
        if move_logits.shape[-1] >= 2:
            top2 = move_logits.topk(2, dim=-1).indices
            out[f"opp_intent/alpha_move_recall_top2{sfx}"] = float(
                (top2 == mv_tgt[:, None]).any(dim=-1).float().mean())
        out[f"opp_intent/alpha_move_baseline_argmax_w{sfx}"] = float((mv_tgt == 0).float().mean())
    out[f"opp_intent/alpha_switch_rate{sfx}"] = float(is_sw.float().mean())

    out[f"opp_intent/alpha_info_gain_nats{sfx}"] = info_gain_nats(logits, tgt)
    if int((~is_sw).sum()) > 1:
        out[f"opp_intent/alpha_info_gain_nats_move{sfx}"] = info_gain_nats(
            logits[~is_sw], tgt[~is_sw])
    return out


def _beta_subset_metrics(logits: torch.Tensor, tgt: torch.Tensor,
                         sfx: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    out[f"opp_intent/beta_recall_top1{sfx}"] = float((logits.argmax(dim=-1) == tgt).float().mean())
    if logits.shape[-1] >= 2:
        b2 = logits.topk(2, dim=-1).indices
        out[f"opp_intent/beta_recall_top2{sfx}"] = float((b2 == tgt[:, None]).any(dim=-1)
                                                         .float().mean())
    out[f"opp_intent/beta_info_gain_nats{sfx}"] = info_gain_nats(logits, tgt)
    return out


def switch_coverage_metrics(kind: torch.Tensor, need: Optional[torch.Tensor],
                            content: Optional[torch.Tensor],
                            rows: Optional[torch.Tensor] = None,
                            sfx: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if rows is not None:
        kind = kind[rows]
        need = None if need is None else need[rows]
        content = None if content is None else content[rows]
    n_sw = float((kind == 1).float().sum())
    if n_sw <= 0:
        return out
    want = 0.0 if need is None else float(need.float().sum())
    got = 0.0 if (need is None or content is None) else float((need & (content >= 0)).float().sum())
    out[f"opp_intent/beta_switch_n{sfx}"] = n_sw
    out[f"opp_intent/beta_switch_to_revealed{sfx}"] = (n_sw - want) / n_sw
    out[f"opp_intent/beta_switch_to_hidden_found{sfx}"] = got / n_sw
    out[f"opp_intent/beta_switch_to_hidden_missed{sfx}"] = (want - got) / n_sw
    if want > 0:
        out[f"opp_intent/beta_belief_miss_rate{sfx}"] = 1.0 - got / want
    return out


def intent_losses(alpha_logits: Optional[torch.Tensor], alpha_target: Optional[torch.Tensor],
                  beta_logits: Optional[torch.Tensor], beta_target: Optional[torch.Tensor],
                  opp_class: Optional[torch.Tensor] = None,
                  bot_label_weight: float = 1.0,
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    total = None
    w_all = intent_label_weights(
        opp_class, bot_label_weight,
        alpha_logits if alpha_logits is not None else (
            beta_logits if beta_logits is not None else torch.zeros(())))

    if alpha_logits is not None and alpha_target is not None:
        sup = alpha_target != INTENT_IGNORE
        n_sup = int(sup.sum())
        metrics["opp_intent/alpha_mask_rate"] = 1.0 - n_sup / max(alpha_target.numel(), 1)
        metrics["opp_intent/alpha_n_supervised"] = float(n_sup)
        if n_sup > 0:
            if w_all is None:
                la = torch.nn.functional.cross_entropy(
                    alpha_logits, alpha_target, ignore_index=INTENT_IGNORE)
            else:
                per = torch.nn.functional.cross_entropy(
                    alpha_logits[sup], alpha_target[sup], reduction="none")
                la = (per * w_all[sup]).sum() / n_sup
            total = la if total is None else total + la
            with torch.no_grad():
                tgt = alpha_target[sup]
                k = alpha_logits.shape[-1] - 1
                metrics["opp_intent/alpha_loss"] = float(la.detach())
                if opp_class is not None:
                    metrics["opp_intent/label_bot_frac"] = float(
                        (opp_class.reshape(-1)[sup] == OPP_CLASS_BOT).float().mean())
                metrics.update(_alpha_subset_metrics(alpha_logits[sup], tgt, k))
                if opp_class is not None:
                    oc = opp_class.reshape(-1)[sup]
                    for _code, _name in OPP_CLASS_NAMES.items():
                        m = oc == _code
                        n_m = int(m.sum())
                        if n_m < 2:                       # a 1-row marginal has zero entropy
                            continue
                        metrics[f"opp_intent/alpha_n_supervised_{_name}"] = float(n_m)
                        metrics.update(_alpha_subset_metrics(
                            alpha_logits[sup][m], tgt[m], k, f"_{_name}"))

    if beta_logits is not None and beta_target is not None:
        sup = beta_target != INTENT_IGNORE
        n_sup = int(sup.sum())
        metrics["opp_intent/beta_mask_rate"] = 1.0 - n_sup / max(beta_target.numel(), 1)
        metrics["opp_intent/beta_n_supervised"] = float(n_sup)
        if n_sup > 0:
            if w_all is None:
                lb = torch.nn.functional.cross_entropy(
                    beta_logits, beta_target, ignore_index=INTENT_IGNORE)
            else:
                per = torch.nn.functional.cross_entropy(
                    beta_logits[sup], beta_target[sup], reduction="none")
                lb = (per * w_all[sup]).sum() / n_sup
            total = lb if total is None else total + lb
            with torch.no_grad():
                metrics["opp_intent/beta_loss"] = float(lb.detach())
                _bl, _bt = beta_logits[sup], beta_target[sup]
                metrics.update(_beta_subset_metrics(_bl, _bt))
                if opp_class is not None:
                    _oc = opp_class.reshape(-1)[sup]
                    for _code, _name in OPP_CLASS_NAMES.items():
                        _m = _oc == _code
                        _n_m = int(_m.sum())
                        if _n_m < 2:
                            continue
                        metrics[f"opp_intent/beta_n_supervised_{_name}"] = float(_n_m)
                        metrics.update(_beta_subset_metrics(_bl[_m], _bt[_m], f"_{_name}"))
                if alpha_logits is not None:
                    with torch.no_grad():
                        p_sw = torch.softmax(alpha_logits.float(), dim=-1)[:, -1]   # [B]
                        b_ok = _bl.argmax(dim=-1) == _bt
                        conf = p_sw[sup] >= 0.5
                        for _name, _m in (("alpha_confident", conf), ("alpha_unsure", ~conf)):
                            if int(_m.sum()) > 1:
                                metrics[f"opp_intent/beta_recall_{_name}"] = float(
                                    b_ok[_m].float().mean())

    if total is None:
        total = torch.zeros((), device=(alpha_logits.device if alpha_logits is not None
                                        else torch.device("cpu")))
    return total, metrics


def _legacy_block(self: Any, rollout_data: Any, aux_metrics: Dict[str, List[Any]], _ntg: Any) -> Any:
    """`ppo.py::train()`'s opponent-intent block, verbatim (see the module docstring)."""
    loss: Any = 0.0
    opp_intent_term = None
    if self.opp_intent_coef > 0.0:
        _al = self.policy.features_extractor.belief_supervision("alpha_logits")
        _bl = self.policy.features_extractor.belief_supervision("beta_logits")
        _sn = self.policy.features_extractor.last_alpha_seat_nums
        _obs = rollout_data.observations
        if _al is not None and _sn is not None and "opp_action_kind" in _obs:
            pass  # ORACLE: names bound to the verbatim copies above
            _kind = _obs["opp_action_kind"].long().flatten()
            _num = _obs["opp_action_num"].long().flatten()
            _atgt = match_seats_to_move_num(_sn, _num, _kind, _sn.shape[-1])
            _btgt = _obs["opp_switch_slot"].long().flatten()
            _btgt = th.where(_kind == 1, _btgt,
                             th.full_like(_btgt, INTENT_IGNORE))
            _sp = _obs.get("opp_switch_species")
            _bel = getattr(self.policy.features_extractor,
                           "last_opp_believed_mask", None)
            _blog = getattr(self.policy.features_extractor,
                            "last_belief_logits", None)
            if _sp is not None and _bel is not None and _blog is not None \
                    and "species" in _blog:
                pass  # ORACLE: names bound to the verbatim copies above
                _content = resolve_believed_slot_by_content(
                    _blog["species"].detach(), _bel.float(),
                    _sp.long().flatten())
                _need = (_kind == 1) & (_btgt < 0)
                _btgt = th.where(_need, _content, _btgt)
                oi_extra_believed = float(
                    ((_need) & (_content >= 0)).float().sum())
                oi_wanted_content = float(_need.float().sum())
            else:
                oi_extra_believed = 0.0
                oi_wanted_content = 0.0
                _content = None
                _need = None
            if _bl is not None:
                _safe = _btgt.clamp(min=0, max=_bl.shape[-1] - 1)
                _reach = ~th.isneginf(_bl.detach().gather(1, _safe[:, None]).squeeze(1))
                _btgt = th.where(_reach, _btgt, th.full_like(_btgt, INTENT_IGNORE))
            _sv, oi_m_extra_rows = None, 0.0
            if self.beta_setvalued_coef > 0.0 and _bl is not None \
                    and _bel is not None and _need is not None \
                    and _content is not None:
                _miss = _need & (_content < 0)
                _sv = set_valued_switch_loss(_bl, _bel.float(), _miss)
                oi_m_extra_rows = float(_miss.float().sum())
            _ocls = _obs.get("opp_class")
            oi_loss, oi_m = intent_losses(
                _al, _atgt, _bl, _btgt,
                opp_class=(_ocls.long() if _ocls is not None else None),
                bot_label_weight=float(
                    getattr(self, "intent_label_bot_weight", 1.0)))
            oi_m["opp_intent/beta_believed_targets"] = oi_extra_believed
            oi_m["opp_intent/beta_wanted_content"] = oi_wanted_content
            oi_m.update(switch_coverage_metrics(_kind, _need, _content))
            if _ocls is not None:
                _ocf = _ocls.long().reshape(-1)
                for _code, _name in OPP_CLASS_NAMES.items():
                    _rows = _ocf == _code
                    if int(_rows.sum()) < 2:
                        continue
                    oi_m.update(switch_coverage_metrics(
                        _kind, _need, _content, _rows, f"_{_name}"))
            if _sv is not None:
                loss = loss + _ntg.add(
                    "aux", self.opp_intent_coef * self.beta_setvalued_coef * _sv)
                oi_m["opp_intent/beta_setvalued_loss"] = float(_sv.detach())
                oi_m["opp_intent/beta_setvalued_rows"] = oi_m_extra_rows
            opp_intent_term = self.opp_intent_coef * oi_loss
            loss = loss + _ntg.add("aux", opp_intent_term)
            for _ok, _ov in oi_m.items():
                aux_metrics.setdefault(_ok, []).append(_ov)
    del loss
    return opp_intent_term


# =================================================================================== HARNESS
class _FE:
    """A stand-in extractor exposing exactly the stash surface both folds read."""

    def __init__(self, inp: Dict[str, Any]):
        self._sup = {"alpha_logits": inp["al"], "beta_logits": inp["bl"]}
        self.last_alpha_seat_nums = inp["sn"]
        self.last_opp_believed_mask = inp["bel"]
        self.last_belief_logits = None if inp["species"] is None else {"species": inp["species"]}

    def belief_supervision(self, name: str) -> Optional[torch.Tensor]:
        return self._sup.get(name)


class _Tagger:
    """`_ntg`: records every term the block adds to `loss`, in order."""

    def __init__(self) -> None:
        self.terms: List[torch.Tensor] = []

    def add(self, group: str, term: torch.Tensor) -> torch.Tensor:
        assert group == "aux"
        self.terms.append(term)
        return term


def _leaves(inp: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
    """Fresh grad-carrying leaves for alpha/beta (and a cast species posterior), in `dtype`."""
    out = dict(inp)
    for k in ("al", "bl"):
        if inp[k] is not None:
            out[k] = inp[k].detach().to(dtype).clone().requires_grad_(True)
    if inp["species"] is not None:
        out["species"] = inp["species"].detach().to(dtype)
    return out


def _run_legacy(inp: Dict[str, Any], coef: float, sv_coef: float, bot_w: float):
    tag, aux = _Tagger(), {}
    owner = SimpleNamespace(opp_intent_coef=coef, beta_setvalued_coef=sv_coef,
                            intent_label_bot_weight=bot_w,
                            policy=SimpleNamespace(features_extractor=_FE(inp)))
    term = _legacy_block(owner, SimpleNamespace(observations=inp["obs"]), aux, tag)
    return term, tag.terms, {k: v[0] for k, v in aux.items()}


def _run_new(inp: Dict[str, Any], coef: Any, sv_coef: Any, bot_w: float) -> Optional[IntentFoldOut]:
    return intent_fold(_FE(inp), inp["obs"], intent_coef=coef, setvalued_coef=sv_coef,
                       setvalued_on=float(sv_coef) > 0.0, bot_label_weight=bot_w)


def _grads(total: Any, inp: Dict[str, Any]) -> List[Optional[torch.Tensor]]:
    leaves = [inp[k] for k in ("al", "bl") if inp[k] is not None]
    if not (isinstance(total, torch.Tensor) and total.requires_grad):
        return [torch.zeros_like(x) for x in leaves]
    gs = torch.autograd.grad(total, leaves, allow_unused=True)
    return [torch.zeros_like(x) if g is None else g for g, x in zip(gs, leaves)]


def _close(a: float, b: float, tol: float) -> bool:
    if a != a or b != b:                                     # NaN must match NaN
        return a != a and b != b
    return abs(a - b) <= tol * max(1.0, abs(b))


def compare(inp0: Dict[str, Any], *, dtype: torch.dtype, coef: float = 0.05, sv_coef: float = 0.05,
            bot_w: float = 0.25, coef_as_tensor: bool = False) -> Dict[str, Any]:
    """Run both folds on identical inputs; assert equivalence; return the measured max deltas."""
    f64 = dtype == torch.float64
    m_tol, t_tol, g_tol = (1e-12, 1e-12, 1e-12) if f64 else (1e-5, 1e-6, 1e-5)
    a, b = _leaves(inp0, dtype), _leaves(inp0, dtype)
    term, tags, leg_m = _run_legacy(a, coef, sv_coef, bot_w)
    c, s = (th.tensor(coef, dtype=th.float64), th.tensor(sv_coef, dtype=th.float64)) \
        if coef_as_tensor else (coef, sv_coef)
    out = _run_new(b, c, s, bot_w)
    if term is None:
        assert out is None and not tags and not leg_m
        return {}
    assert out is not None
    rep: Dict[str, Any] = {"metric": 0.0, "sv_metric": 0.0, "intent_term": 0.0, "sv_term": 0.0,
                           "grad": 0.0}

    # ---- the key set, the weights, the values
    for k, (v, w) in out.metrics.items():
        assert v.dim() == 0 and w.dim() == 0 and float(w) in (0.0, 1.0), k
        if float(w) == 0.0:
            assert float(v) == 0.0, f"{k}: a weight-0 metric must read exactly 0.0"
    present = {k for k, (_v, w) in out.metrics.items() if float(w) == 1.0}
    assert present == set(leg_m), (sorted(present - set(leg_m)), sorted(set(leg_m) - present))
    for k in present:
        got, want = float(out.metrics[k][0]), float(leg_m[k])
        tol = _SV_F64_TOL if (f64 and k.endswith("beta_setvalued_loss")) else m_tol
        assert _close(got, want, tol), (k, got, want)
        if want == want:
            slot = "sv_metric" if k.endswith("beta_setvalued_loss") else "metric"
            rep[slot] = max(rep[slot], abs(got - want) / max(1.0, abs(want)))

    # ---- the terms and their presence
    has_sv = "opp_intent/beta_setvalued_loss" in leg_m
    assert bool(out.setvalued_present) == has_sv
    leg_sv = tags[0] if has_sv else None
    leg_it = tags[-1]
    assert leg_it is term
    n_a = leg_m.get("opp_intent/alpha_n_supervised", 0.0)
    n_b = leg_m.get("opp_intent/beta_n_supervised", 0.0)
    assert bool(out.intent_present) == (n_a > 0 or n_b > 0)
    if not bool(out.intent_present):
        assert float(out.intent_term.detach()) == 0.0 and float(term) == 0.0
    if leg_sv is None:      # exactly 0.0; its gradient (compared below) is exactly 0 too
        assert float(out.setvalued_term.detach()) == 0.0
    for slot, got_t, want_t, tol in (("intent_term", out.intent_term, term, t_tol),
                                     ("sv_term", out.setvalued_term, leg_sv,
                                      _SV_F64_TOL if f64 else t_tol)):
        if want_t is None:
            continue
        g, w = float(got_t.detach()), float(want_t.detach())
        assert _close(g, w, tol), (g, w)
        if w == w:
            rep[slot] = max(rep[slot], abs(g - w) / abs(w) if w else abs(g))

    # ---- the gradient of everything the block adds to the loss
    leg_total = sum(tags[1:], tags[0])
    g_leg = _grads(leg_total, a)
    g_new = _grads(out.setvalued_term + out.intent_term, b)
    for gl, gn in zip(g_leg, g_new):
        scale = float(gl.nan_to_num(0.0, 0.0, 0.0).abs().max()) or 1.0
        assert torch.allclose(gn, gl, rtol=g_tol, atol=g_tol * scale, equal_nan=True), \
            float((gn - gl).nan_to_num(0.0).abs().max())
        fin = gl.isfinite() & gn.isfinite()
        assert torch.equal(gl.isnan(), gn.isnan())
        if fin.any():
            rep["grad"] = max(rep["grad"], float((gn - gl)[fin].abs().max()) / scale)
    return rep


#: The set-valued loss is computed in float32 BY THE LEGACY CODE (`beta_logits[rows].float()`), even
#: under float64 inputs; the static version keeps that cast, so the two agree to float32 rounding.
_SV_F64_TOL = 1e-6


# =================================================================================== INPUTS
def synth(B: int, seed: int, *, class_probs=(0.4, 0.3, 0.2, 0.1), kind_probs=(0.55, 0.3, 0.15),
          K: int = 6, n_species: int = 400, with_class: bool = True) -> Dict[str, Any]:
    """A realistic synthetic micro-batch: every label kind, revealed / hidden-found / hidden-missed
    switches, -inf on unreachable / invalid slots, dead beta rows, garbage slots on non-switch rows."""
    g = th.Generator().manual_seed(seed)

    def r(*s):
        return th.rand(*s, generator=g)

    def ri(lo, hi, s):
        return th.randint(lo, hi, s, generator=g)

    seat_valid = r(B, K) < 0.85
    seat_valid[:, 0] = True
    sn = ri(1, n_species, (B, K)) * seat_valid
    al = th.randn(B, K + 1, generator=g) * 1.5
    al[:, :K] = al[:, :K].masked_fill(~seat_valid, float("-inf"))
    kind = th.multinomial(th.tensor(kind_probs), B, replacement=True, generator=g)
    pick = ri(0, K, (B,))
    hit = (r(B) < 0.8) & seat_valid.gather(1, pick[:, None]).squeeze(1)
    num = th.where(hit, sn.gather(1, pick[:, None]).squeeze(1), ri(1, n_species, (B,)))
    num = th.where(kind == 0, num, th.zeros_like(num))
    cand = r(B, 6) < 0.6
    bl = (th.randn(B, 6, generator=g)).masked_fill(~cand, float("-inf"))
    dead = cand.sum(-1) == 0
    bl = th.where(dead[:, None], th.zeros_like(bl), bl)        # BetaSwitchHead's flat dead rows
    revealed = r(B) < 0.5
    slot = th.where((kind == 1) & revealed, ri(0, 6, (B,)), th.full((B,), INTENT_IGNORE))
    junk = (kind != 1) & (r(B) < 0.1)                          # the label builder masks these
    slot = th.where(junk, ri(0, 6, (B,)), slot)
    sp = th.where(kind == 1, ri(1, n_species, (B,)), th.zeros(B, dtype=th.long))
    bel = (r(B, 6) < 0.5).float()
    spl = th.randn(B, 6, n_species, generator=g)
    found = r(B) < 0.5
    j = ri(0, 6, (B,))
    rows = th.arange(B)
    spl[rows[found], j[found], sp[found]] += 12.0
    bel[rows[found], j[found]] = 1.0
    obs = {"opp_action_kind": kind[:, None], "opp_action_num": num[:, None],
           "opp_switch_slot": slot[:, None], "opp_switch_species": sp[:, None]}
    if with_class:
        obs["opp_class"] = th.multinomial(th.tensor(class_probs), B, replacement=True,
                                          generator=g)[:, None]
    return {"al": al, "bl": bl, "sn": sn, "bel": bel, "species": spl, "obs": obs}


@pytest.fixture(scope="module")
def real() -> Dict[str, Any]:
    """The REAL stash shapes and labels: the seeded production-surface learner's extractor forward
    over the committed 64-row learner-golden buffer (CPU)."""
    from agents.training.learner_golden import build_learner, load_buffer_into

    m = build_learner()
    load_buffer_into(m)
    data = next(m.rollout_buffer.get(64))
    with th.no_grad():
        m.policy.evaluate_actions(data.observations, data.actions.long().flatten())
    fe = m.policy.features_extractor
    det = (lambda t: None if t is None else t.detach().clone())
    inp = {"al": det(fe.belief_supervision("alpha_logits")),
           "bl": det(fe.belief_supervision("beta_logits")),
           "sn": det(fe.last_alpha_seat_nums), "bel": det(fe.last_opp_believed_mask),
           "species": det(fe.last_belief_logits["species"]),
           "obs": {k: v.clone() for k, v in data.observations.items()}}
    for k in ("al", "bl", "sn", "bel", "species"):
        assert inp[k] is not None, f"the production surface stashes {k}"
    return {"inp": inp, "model": m, "data": data}


def tiled(real_inp: Dict[str, Any], reps: int, seed: int) -> Dict[str, Any]:
    """The real batch tiled to `64 * reps` rows, logits jittered and opponent classes reshuffled."""
    g = th.Generator().manual_seed(seed)
    out = {k: (None if v is None or k == "obs" else v.repeat(reps, *([1] * (v.dim() - 1))))
           for k, v in real_inp.items()}
    for k in ("al", "bl", "species"):
        x = out[k]
        out[k] = th.where(th.isfinite(x), x + 0.3 * th.randn(x.shape, generator=g), x)
    obs = {k: v.repeat(reps, 1) for k, v in real_inp["obs"].items() if k.startswith("opp_")}
    obs["opp_class"] = th.randint(0, 4, obs["opp_class"].shape, generator=g)
    out["obs"] = obs
    return out


# =================================================================================== EQUIVALENCE
DTYPES = [th.float64, th.float32]
CONFIGS = [(0.25, 0.05), (1.0, 0.05), (0.25, 0.0), (1.0, 0.0)]      # (bot_label_weight, setvalued)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bot_w,sv", CONFIGS)
def test_real_batch_equals_legacy(real, dtype, bot_w, sv):
    rep = compare(real["inp"], dtype=dtype, bot_w=bot_w, sv_coef=sv)
    assert rep


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bot_w,sv", CONFIGS)
def test_tiled_real_2048_equals_legacy(real, dtype, bot_w, sv):
    compare(tiled(real["inp"], 32, seed=7), dtype=dtype, bot_w=bot_w, sv_coef=sv)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bot_w,sv", CONFIGS)
@pytest.mark.parametrize("B,seed", [(64, 1), (64, 2), (2048, 3)])
def test_synthetic_equals_legacy(dtype, bot_w, sv, B, seed):
    compare(synth(B, seed), dtype=dtype, bot_w=bot_w, sv_coef=sv)


def test_the_real_extractor_through_the_adapter(real):
    """`intent_fold(fe, ...)` on the REAL extractor (not a stand-in) reads what the legacy block
    reads: the same forward, both folds, values equal."""
    m, data = real["model"], real["data"]
    m.policy.evaluate_actions(data.observations, data.actions.long().flatten())
    fe = m.policy.features_extractor
    tag, aux = _Tagger(), {}
    owner = SimpleNamespace(opp_intent_coef=0.05, beta_setvalued_coef=0.05,
                            intent_label_bot_weight=0.25, policy=m.policy)
    term = _legacy_block(owner, data, aux, tag)
    out = intent_fold(fe, data.observations, intent_coef=0.05, setvalued_coef=0.05,
                      setvalued_on=True, bot_label_weight=0.25)
    assert term is not None and out is not None
    assert {k for k, (_v, w) in out.metrics.items() if float(w) == 1.0} == set(aux)
    for k, v in aux.items():
        assert _close(float(out.metrics[k][0]), float(v[0]), 1e-5), k
    assert _close(float(out.intent_term.detach()), float(term.detach()), 1e-6)
    assert out.intent_term.requires_grad


def test_class_strata_with_zero_one_and_many_rows():
    """Class 3 has no row, class 2 exactly one (no per-class key), classes 0/1 many."""
    inp = synth(64, 11)
    oc = th.zeros(64, dtype=th.long)
    oc[30:] = 1
    oc[5] = 2
    inp["obs"]["opp_class"] = oc[:, None]
    for dtype in DTYPES:
        compare(inp, dtype=dtype)
    out = _run_new(_leaves(inp, th.float64), 0.05, 0.05, 0.25)
    assert out is not None
    assert float(out.metrics["opp_intent/alpha_n_supervised_stable"][1]) == 0.0
    assert float(out.metrics["opp_intent/beta_switch_n_exploiter"][1]) == 0.0


def test_all_unsupervised_batch_has_both_terms_absent():
    inp = synth(64, 12)
    inp["obs"]["opp_action_kind"] = th.full((64, 1), 2)          # UNKNOWN: nothing nameable
    for dtype in DTYPES:
        compare(inp, dtype=dtype)
    out = _run_new(_leaves(inp, th.float32), 0.05, 0.05, 0.25)
    assert out is not None and not bool(out.intent_present) and not bool(out.setvalued_present)
    assert float(out.intent_term) == 0.0 and float(out.setvalued_term) == 0.0


@pytest.mark.parametrize("drop", ["opp_class", "belief", "beta", "species_key"])
def test_static_paths_off_equal_legacy(drop):
    inp = synth(256, 13)
    if drop == "opp_class":
        del inp["obs"]["opp_class"]
    elif drop == "belief":
        inp["bel"] = None
    elif drop == "beta":
        inp["bl"] = None
    else:
        del inp["obs"]["opp_switch_species"]
    for dtype in DTYPES:
        for bot_w in (1.0, 0.25):
            compare(inp, dtype=dtype, bot_w=bot_w)


def test_legacy_skip_is_none():
    inp = synth(64, 14)
    inp["al"] = None
    assert _run_new(inp, 0.05, 0.05, 0.25) is None
    inp = synth(64, 14)
    del inp["obs"]["opp_action_kind"]
    compare(inp, dtype=th.float64)


def test_coefficients_as_tensors():
    """The compiled region takes the coefficients as 0-d tensors (no recompile when a dose moves)."""
    for dtype in DTYPES:
        compare(synth(512, 15), dtype=dtype, coef_as_tensor=True)


def _nan_row(inp: Dict[str, Any], key: str, supervised: bool) -> Dict[str, Any]:
    """Put a NaN in one alpha/beta row that the legacy block does / does not supervise."""
    obs = inp["obs"]
    kind = obs["opp_action_kind"].flatten()
    if key == "al":
        tgt = intent_fold_rows(inp)["alpha"]
    else:
        tgt = intent_fold_rows(inp)["beta"]
    sup = tgt != INTENT_IGNORE
    rows = th.nonzero(sup if supervised else (~sup & (kind != 1)))
    i = int(rows[0])
    x = inp[key].clone()
    x[i] = float("nan")
    out = dict(inp)
    out[key] = x
    return out


def intent_fold_rows(inp: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """The legacy supervision targets (alpha/beta), read off the oracle's own helpers."""
    obs = inp["obs"]
    kind = obs["opp_action_kind"].long().flatten()
    at = match_seats_to_move_num(inp["sn"], obs["opp_action_num"].long().flatten(), kind,
                                 inp["sn"].shape[-1])
    bt = th.where(kind == 1, obs["opp_switch_slot"].long().flatten(), th.full_like(kind, -100))
    content = resolve_believed_slot_by_content(inp["species"], inp["bel"].float(),
                                               obs["opp_switch_species"].long().flatten())
    bt = th.where((kind == 1) & (bt < 0), content, bt)
    safe = bt.clamp(0, inp["bl"].shape[-1] - 1)
    reach = ~th.isneginf(inp["bl"].gather(1, safe[:, None]).squeeze(1))
    return {"alpha": at, "beta": th.where(reach, bt, th.full_like(bt, -100))}


@pytest.mark.parametrize("key", ["al", "bl"])
@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("bot_w", [1.0, 0.25])
def test_nan_reaches_the_term_only_from_a_supervised_row(key, supervised, bot_w):
    """K9(c): a NaN in a SUPERVISED row reaches the term (and so the loss check); a NaN in a masked
    row does not — in both folds, with matching gradients (NaN where the legacy gradient is NaN)."""
    inp = _nan_row(synth(256, 16), key, supervised)
    for dtype in DTYPES:
        compare(inp, dtype=dtype, bot_w=bot_w, sv_coef=0.0)
    out = _run_new(_leaves(inp, th.float64), 0.05, 0.0, bot_w)
    assert out is not None
    assert bool(th.isnan(out.intent_term)) == supervised


# =================================================================================== THE TRACE
_ON_28 = th.__version__.startswith("2.8")


def _tensor_args(inp: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    o = inp["obs"]
    return (inp["al"], inp["bl"], inp["sn"], o["opp_action_kind"], o["opp_action_num"],
            o["opp_switch_slot"], o["opp_switch_species"], inp["bel"], inp["species"], o["opp_class"])


def _wrapper(bot_w: float) -> Callable[..., Any]:
    def fold(al, bl, sn, kind, num, slot, sp, bel, spl, oc, ci, cs):
        out = intent_fold_tensors(al, bl, sn, kind=kind, num=num, switch_slot=slot,
                                  switch_species=sp, believed_mask=bel, species_logits=spl,
                                  opp_class=oc, intent_coef=ci, setvalued_coef=cs,
                                  setvalued_on=True, bot_label_weight=bot_w)
        return out.setvalued_term + out.intent_term, out.intent_present, out.metrics
    return fold


@pytest.mark.skipif(not _ON_28, reason="fullgraph compile is only required on torch 2.8 (the target)")
@pytest.mark.parametrize("bot_w", [0.25, 1.0])
def test_fullgraph_trace_equals_eager(real, bot_w):
    import torch._dynamo

    fold = _wrapper(bot_w)
    compiled = th.compile(fold, fullgraph=True, backend="aot_eager", dynamic=False)
    ci, cs = th.tensor(0.05), th.tensor(0.05)
    try:
        for inp0 in (real["inp"], tiled(real["inp"], 32, seed=9)):
            a, b = _leaves(inp0, th.float32), _leaves(inp0, th.float32)
            t_e, p_e, m_e = fold(*_tensor_args(a), ci, cs)
            t_c, p_c, m_c = compiled(*_tensor_args(b), ci, cs)
            assert _close(float(t_c), float(t_e), 1e-6) and bool(p_c) == bool(p_e)
            assert set(m_c) == set(m_e)
            for k in m_e:
                assert float(m_c[k][1]) == float(m_e[k][1]), k
                assert _close(float(m_c[k][0]), float(m_e[k][0]), 1e-6), k
            for ge, gc in zip(th.autograd.grad(t_e, [a["al"], a["bl"]]),
                              th.autograd.grad(t_c, [b["al"], b["bl"]])):
                assert th.allclose(gc, ge, rtol=1e-6, atol=1e-6 * float(ge.abs().max()))
    finally:
        torch._dynamo.reset()
