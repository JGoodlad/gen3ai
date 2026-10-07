"""THE LEARNER MICRO-STEP — the compile REGION R1 of M5 Lane K8 (`gen3_learner_micro_step_v1`).

ONE function computes, for one micro-batch, everything `train()` folded INLINE before the eager tail:
the policy forward (`evaluate_actions`, functional masking — `masked_categorical`) and fold steps 1–3a
of the FOLD ORDER contract (`ppo.train`'s docstring; `src/agents/training/CLAUDE.md`):

  1. the upstream PPO loss: `policy_grad_coef · policy_loss + ent_coef · entropy + vf_term` (the fork
     mask, plain / clipped value loss);
  2. the BELIEF bank's `hidden_move` site (hidden-team set BCE, move belief) → the OPPONENT-INTENT
     fold (the flat pointer CE) → the `latent` site (move latent) → the `revealed` site (spread,
     nature/EV, HP type, item);
  3a. the WIN-PROB BCE (the value loss under the win-prob critic, else an aux term).

in that order, as ONE straight line — the float-addition order of the inline fold is preserved term
by term. Everything after 3a (the ride-along heads' own-forward update, the capacity probes) is the DECLARED EAGER TAIL, folded in
contract order by `train()` onto this region's loss.

WHY ONE FUNCTION. `train()` handed to dynamo is 514–646 graphs (the K8 inventory); this function is
written so it traces as ONE `fullgraph=True` graph: static shapes only (`belief_bank_static`,
`intent_fold`), no host read (every diagnostic is a `(value, weight)` pair of 0-d tensors, weight 1.0
exactly where the inline fold appended to its metric list), no Python branch on a tensor value (the
flags are `MicroStatic`, resolved once per `train()`), no numpy, no mutation outside its own locals
(the extractor's own per-forward stash aside, which the compiled extractor has always written).

THE CONTRACT WITH `train()`. The host reads ONE flat tensor per micro-batch (`pack`'s bundle: every
metric value + weight, the loss's finiteness, the approx-KL), then appends each present metric to the
same per-update lists the inline fold filled — so every TB tag keeps its meaning (mean over the
micro-batches that produced it).
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch as th
import torch.nn.functional as F

from agents.model.region_calls import note_eager_body
from agents.training import belief_bank_static as _bbs
from agents.training.fork_arm import PG_MASK_KEY as FORK_PG_MASK_KEY
from agents.training.instrumented_ppo.constants import _WIN_CONTESTED_TAU
from agents.training.instrumented_ppo.intent_fold import intent_fold

Metric = Tuple[th.Tensor, th.Tensor]

#: Per-update metric LIST names the PPO core fills (`train()`'s locals of the same names).
PPO_LISTS = ("pg_losses", "clip_fractions", "value_losses", "entropy_losses", "vf_clip_fractions",
             "approx_kl")


class MicroStatic(NamedTuple):
    """Everything the region branches on — Python constants, resolved ONCE per `train()` (dynamo
    specialises on them; none changes within a run except by a deliberate flag change)."""
    discrete: bool
    normalize_advantage: bool
    clip_range: float
    clip_range_vf: Optional[float]
    value_mode: str                  # "plain" | "clipped"
    critic_winprob: bool
    vf_coef: float
    ent_coef: float
    policy_grad_coef: float
    fork_pg_mask: bool
    belief_aux_on: bool
    move_belief_on: bool
    move_latent_on: bool
    spread_belief_on: bool
    hp_type_belief_on: bool
    item_belief_on: bool
    belief_coefs: Tuple[Tuple[str, float], ...]
    moves_weight: float
    intent_on: bool
    intent_coef: float
    bot_label_weight: float
    win_prob_on: bool


class MicroOut(NamedTuple):
    loss: th.Tensor
    values: th.Tensor                # flattened critic values (with grad)
    log_prob: th.Tensor
    entropy: Optional[th.Tensor]
    logp: Optional[th.Tensor]        # masked normalised log-probs (the ride-along heads' stash)
    masks_bool: Optional[th.Tensor]
    advantages: th.Tensor            # normalised
    entropy_loss: th.Tensor          # -mean(entropy), unweighted (the grad-balance probe's policy side)
    value_loss: th.Tensor            # the scalar value loss (the probe's value side off winprob)
    terms: Dict[str, th.Tensor]      # probe name -> live term (the grad-balance / noise probes)
    term_groups: Dict[str, str]      # probe name -> noise-scale group ("policy" / "entropy" / ...)
    present: Dict[str, th.Tensor]    # probe name -> 0-d bool (the inline fold's `is not None`)
    metrics: Dict[str, Metric]       # "<list or group>/<key>" -> (value, weight)


def _m(v: th.Tensor, w: Optional[th.Tensor] = None) -> Metric:
    v = v.detach().to(th.float32).reshape(())
    return v, (th.ones((), device=v.device) if w is None else w.to(th.float32).reshape(()))


def win_prob_terms(logits: Optional[th.Tensor], target: Optional[th.Tensor],
                   mask: Optional[th.Tensor], margin: Optional[th.Tensor]
                   ) -> Tuple[th.Tensor, th.Tensor, Dict[str, Metric]]:
    """`ValueTerms._win_prob_loss`, static: (loss, present, metrics). Absent inputs (a static fact)
    -> present False and no metrics; nothing scored (a data fact) -> present False, loss 0.0."""
    if logits is None or target is None or mask is None:
        z = th.zeros(())
        return z, th.zeros((), dtype=th.bool), {}
    logits = logits.reshape(-1)
    target = target.to(logits.device).reshape(-1)
    mask = mask.to(logits.device).reshape(-1)
    n_known = mask.sum()
    present = n_known > 0
    nk = n_known.clamp(min=1)                  # == n_known whenever present (a 0/1 mask)
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (per * mask).sum() / nk
    loss = th.where(present, loss, loss.new_zeros(()))
    wp = present
    mets: Dict[str, Metric] = {}
    with th.no_grad():
        p = th.sigmoid(logits)
        sq = (p - target) ** 2
        correct = ((p > 0.5).float() == target).float()
        brier = (sq * mask).sum() / nk
        mets["loss"] = _m(loss, wp)
        mets["acc"] = _m((correct * mask).sum() / nk, wp)
        mets["brier"] = _m(brier, wp)
        mets["pred_mean"] = _m((p * mask).sum() / nk, wp)
        mets["label_mean"] = _m((target * mask).sum() / nk, wp)
        mets["coverage"] = _m(n_known / mask.numel(), wp)
        if margin is not None:
            mg = margin.to(logits.device).reshape(-1)
            spread = (mg.max() - mg.min()) > 0.0
            on = wp & spread
            close = (mg.abs() < _WIN_CONTESTED_TAU).float() * mask
            n_close = close.sum()
            nc = n_close.clamp(min=1)
            mets["contested_frac"] = _m(n_close / nk, on)
            has_close = on & (n_close > 0)
            mets["brier_contested"] = _m((sq * close).sum() / nc, has_close)
            mets["acc_contested"] = _m((correct * close).sum() / nc, has_close)
            mets["contested_label_mean"] = _m((target * close).sum() / nc, has_close)
            p_mat = (0.5 + 0.5 * mg).clamp(1e-6, 1.0 - 1e-6)
            brier_mat = (((p_mat - target) ** 2) * mask).sum() / nk
            mets["brier_material"] = _m(brier_mat, on)
            skill = th.where(brier_mat > 0.0, 1.0 - brier / brier_mat.clamp(min=1e-30),
                             brier.new_zeros(()))
            mets["skill_vs_material"] = _m(skill, on)
    return loss, present, mets


def micro_step(policy: Any, obs: Dict[str, th.Tensor], actions: th.Tensor,
               action_masks: Optional[th.Tensor], old_log_prob: th.Tensor, old_values: th.Tensor,
               advantages: th.Tensor, returns: th.Tensor, st: MicroStatic) -> MicroOut:
    """R1 — see the module docstring. ``st`` carries every static flag and coefficient."""
    note_eager_body("R1")      # a no-op under a dynamo trace; counts an EAGER run (gen3_no_silent_eager_v1)
    fe = policy.features_extractor
    mets: Dict[str, Metric] = {}
    terms: Dict[str, th.Tensor] = {}
    groups: Dict[str, str] = {}
    present: Dict[str, th.Tensor] = {}
    if st.discrete:
        actions = actions.long().flatten()
    if hasattr(policy, "evaluate_actions_functional"):
        values, log_prob, entropy, logp, masks_bool = policy.evaluate_actions_functional(
            obs, actions, action_masks)
    else:                         # a stock sb3 policy (unit-test toys; never compiled — no extractor
        # region): its own distribution object, no masked-logit stash
        values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=action_masks)
        logp, masks_bool = None, None
    values = values.flatten()
    adv = advantages
    if st.normalize_advantage and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ---- 1. the upstream PPO loss ------------------------------------------------------------
    ratio = th.exp(log_prob - old_log_prob)
    policy_loss_1 = adv * ratio
    policy_loss_2 = adv * th.clamp(ratio, 1 - st.clip_range, 1 + st.clip_range)
    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
    if st.fork_pg_mask:
        _fk_m = obs[FORK_PG_MASK_KEY].reshape(-1)
        policy_loss = -((th.min(policy_loss_1, policy_loss_2) * _fk_m).sum()
                        / _fk_m.sum().clamp(min=1.0))
    mets["pg_losses/"] = _m(policy_loss)
    mets["clip_fractions/"] = _m(th.mean((th.abs(ratio - 1) > st.clip_range).float()))
    if st.value_mode == "plain":
        value_loss = ((returns - values) ** 2).mean()
    else:
        assert st.clip_range_vf is not None
        values_pred = old_values + th.clamp(values - old_values, -st.clip_range_vf, st.clip_range_vf)
        mets["vf_clip_fractions/"] = _m(th.mean(
            (th.abs(values - old_values) > st.clip_range_vf).float()))
        value_loss = ((returns - values_pred) ** 2).mean()
    mets["value_losses/"] = _m(value_loss)
    ent_per = -log_prob if entropy is None else entropy
    entropy_loss = -th.mean(ent_per)
    mets["entropy_losses/"] = _m(entropy_loss)
    vf_term: Any = 0.0 if st.critic_winprob else st.vf_coef * value_loss
    pg_term = policy_loss if st.policy_grad_coef == 1.0 else st.policy_grad_coef * policy_loss
    ent_term = st.ent_coef * entropy_loss
    loss = pg_term + ent_term + vf_term
    terms["policy"], groups["policy"] = pg_term, "policy"
    terms["entropy"], groups["entropy"] = ent_term, "entropy"
    if th.is_tensor(vf_term):
        terms["value"], groups["value"] = vf_term, "value"
    with th.no_grad():
        log_ratio = log_prob - old_log_prob
        mets["approx_kl/"] = _m(th.mean((th.exp(log_ratio) - 1) - log_ratio))

    coefs = dict(st.belief_coefs)

    def _bank(site: str, gates: Dict[str, bool]) -> th.Tensor:
        nonlocal loss
        for t in _bbs.compute_static(fe, obs, coefs=coefs, gates=gates, site=site,
                                     params={"moves_weight": st.moves_weight}):
            loss = loss + t.term
            terms[t.row.probe], groups[t.row.probe], present[t.row.probe] = t.term, "aux", t.present
            for k, v in t.metrics.items():
                mets[f"belief/{k}"] = v
        return loss

    # ---- 2. the belief bank (hidden_move) -> opponent intent -> latent -> revealed ---------------
    # gen3_x5_hypothesis_set_v1: the hidden-team supervision is X5's set BCE row (`hidden_team_set`);
    # it is absent (no term) when the extractor built no hypothesis set.
    _bank("hidden_move", {"hidden_team_set": st.belief_aux_on, "move_belief": st.move_belief_on})
    io = (intent_fold(fe, obs, intent_coef=st.intent_coef, bot_label_weight=st.bot_label_weight)
          if st.intent_on else None)                     # None <=> the inline block was skipped
    if io is not None:
        loss = loss + io.intent_term
        # REGISTERED whenever the block ran — the inline fold's `opp_intent_term` existed (a 0.0
        # when nothing was supervised) and the grad-balance probe and K9(c) saw it.
        terms["opp_intent"], groups["opp_intent"] = io.intent_term, "aux"
        present["opp_intent"] = th.ones((), dtype=th.bool, device=io.intent_term.device)
        for k, v in io.metrics.items():
            mets[f"aux/{k}"] = v
    _bank("latent", {"move_latent": st.move_latent_on})
    _bank("revealed", {"spread": st.spread_belief_on, "hp_type": st.hp_type_belief_on,
                       "item": st.item_belief_on})

    # ---- 3a. the win-prob BCE -----------------------------------------------------------------
    if st.win_prob_on:
        wl, wpres, wm = win_prob_terms(
            fe.last_win_prob_logits, obs.get("win_target"), obs.get("win_mask"),
            obs.get("win_margin"))
        if wm:
            if st.critic_winprob:
                wterm, grp = st.vf_coef * wl, "value"
            else:
                wterm, grp = wl, "aux"
            loss = loss + wterm
            terms["win_prob"], groups["win_prob"], present["win_prob"] = wterm, grp, wpres
            for k, v in wm.items():
                mets[f"win_prob/{k}"] = v
    return MicroOut(loss=loss, values=values, log_prob=log_prob, entropy=entropy, logp=logp,
                    masks_bool=masks_bool, advantages=adv, entropy_loss=entropy_loss,
                    value_loss=value_loss, terms=terms, term_groups=groups,
                    present=present, metrics=mets)


# --------------------------------------------------------------------------- the host side
def pack(out: MicroOut) -> Tuple[List[str], List[str], th.Tensor]:
    """The ONE device->host read per micro-batch: ``(metric_keys, present_keys, flat)`` where
    ``flat`` = [every metric value, every metric weight, every present flag, loss finite], float32.
    Key order is the dict order (static per run)."""
    mkeys = list(out.metrics)
    pkeys = list(out.present)
    parts = [th.stack([out.metrics[k][0] for k in mkeys]) if mkeys else None,
             th.stack([out.metrics[k][1] for k in mkeys]) if mkeys else None,
             th.stack([out.present[k].to(th.float32) for k in pkeys]) if pkeys else None,
             th.isfinite(out.loss.detach()).to(th.float32).reshape(1)]
    flat = th.cat([p.reshape(-1) for p in parts if p is not None])
    return mkeys, pkeys, flat.cpu()


def unpack(mkeys: List[str], pkeys: List[str], flat: th.Tensor
           ) -> Tuple[Dict[str, float], Dict[str, bool], bool]:
    """``(present metric values, present flags, loss finite)`` from `pack`'s host tensor."""
    a = flat.tolist()
    n = len(mkeys)
    vals = {k: a[i] for i, k in enumerate(mkeys) if a[n + i] > 0.5}
    pres = {k: a[2 * n + j] > 0.5 for j, k in enumerate(pkeys)}
    return vals, pres, a[-1] > 0.5
