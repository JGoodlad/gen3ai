"""The DETACHED RIDE-ALONG heads (`gen3_ridealong_heads_v1`) — baselines that observe, never steer.

The owner's ask (2026-09-30): *"establish a baseline just with the ensemble of value heads to test
uncertainty, the trick where you predict a random output to see how often you've seen the state,
and the Q, A and B values. Start getting baselines before we add more complicated arms."* So the
first GPU run on the M5 core is the production recipe PLUS these heads, and every later arm
(X23 first) is compared against it. Rows X25 (epistemic confidence) and X4a (the A/B ride-along)
of `designs/research_state/EXPERIMENT_BACKLOG.md`; the Q = V + A + B decomposition is
`designs/endstate/design_q_head.md` §4.

THE CORE RULE IS STOP-GRAD, and it is structural rather than a convention:

  * every input is `.detach()`ed (`RideAlongBatch.detached`), so no ride-along loss can put a
    gradient on a trunk, policy or V parameter;
  * the heads are NOT in `policy.optimizer`'s param groups (they are built after SB3's `_build`
    made it) and the learner steps them with their OWN optimizer right after the minibatch's
    forward, then sets their grads back to None — so PPO's global `clip_grad_norm_`, the noise-scale
    probes and the distill projector never see a ride-along gradient;
  * they are built inside `torch.random.fork_rng` from a PRIVATE seed and draw no random number at
    train time (the bootstrap masks are a HASH of the observation), so heads ON and OFF leave the
    global RNG stream — the policy's own init, every rollout sample, every minibatch shuffle —
    bit-identical;
  * they are NOT called by the forward, so a rollout, an eval, the prober and an opponent pay
    nothing for them and `pi`/`vf` are bit-identical at an arbitrary weight in them.

`ridealong_heads_test.py` proves (a) no gradient from any ride-along loss reaches a non-ride-along
parameter and (b) one real update with heads ON vs OFF at the same seed leaves the policy, trunk and
V parameters (and the optimizer state, and the RNG) BIT-IDENTICAL; both fail on revert.

THE FOUR HEADS (each its own STRUCTURAL flag, all OFF by default):

  * **V ENSEMBLE** (`--ridealong-ensemble K`): K small heads on LayerNorm(detached `value_pooled`),
    each predicting V's own target (the win bit, BCE). Diversity is forced three ways, because
    heads that share a detached trunk otherwise collapse together: a different init seed per
    member, a per-member BOOTSTRAP mask (Bernoulli 0.5 per STATE, a hash of the observation so it
    is identical in every epoch and consumes no RNG — bootstrapped DQN, Osband et al. 2016), and a
    RANDOMIZED PRIOR function per member (a frozen random net added to the member's logit; Osband
    et al. 2018) that keeps members apart exactly where data is thin. Disagreement = std of the
    members' probabilities.
  * **RND novelty** (`--ridealong-rnd`): a frozen random target net and a trained predictor over the
    RAW OBSERVATION (Burda et al. 2018), per-dimension running normalisation clipped to ±5. The
    input is the observation, not the trunk features: the features drift as the trunk trains, so a
    feature-space novelty would confound "rarely seen" with "the representation moved"; the
    observation is fixed, and the reader measures both. The squared error is z-scored with running
    statistics. Every head trains on each rollout row ONCE (PPO's epoch 0 — the learner's
    `RIDEALONG_EPOCHS`), so the RND predictor's familiarity is a visitation count.
  * **A** (`--ridealong-adv K`): per-action scores from the SAME per-action tokens the pointer head
    scores (`QWinProbHead`'s shared-scorer structure, value context), centred under the policy
    (Σ_a π(a)·A(s,a) = 0), regressed (MSE) on the GAE advantage of the action actually taken — the
    only label PPO produces. K bootstrapped members with randomized priors, so an action the policy
    never plays (a STARVED move) keeps its members apart: per-action uncertainty.
  * **B** (`--ridealong-opp K`): the opponent's main effect over α's own support — their K believed
    move seats (scored from the seat's concrete MOVE ID through the head's own embedding) plus
    SWITCH — centred under α (stop-grad), regressed (MSE) on the same advantage at the rows where
    the opponent's actual action is named in that support (α's label, `match_seats_to_move_num`).
    Because a and b are chosen simultaneously, E[adv | s, b] = B(s, b) once A is centred under π and
    B under α, so A and B are identified by two MARGINAL regressions on one label. This is the
    SIMPLE parameterisation that does not need X5; it is to be RE-BASED onto X5's flat opponent
    pointer (seats + switch targets + OTHER) when X5 lands. Its declared limit: a move outside the
    believed seats (α's mask rate) is not a label, so B is conditional on the opponent choosing a
    listed option.

Q = V + A(a) + B(b) is a DERIVED readout (no I term); losses are MSE, linear in the labels
(`design_q_head.md` §5.1's M = 1 rule).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from agents.action.constants import ACTION_SPACE_SIZE
from agents.model.arch_constants import (
    D_MODEL,
    POINTER_HIDDEN,
    RIDEALONG_HIDDEN,
    RIDEALONG_INIT_SEED,
    RIDEALONG_OPP_MOVE_EMB,
    RIDEALONG_PRIOR_SCALE_Q,
    RIDEALONG_PRIOR_SCALE_V,
    RIDEALONG_RND_HIDDEN,
    RIDEALONG_RND_OUT,
)
from agents.model.q_winprob_head import QWinProbHead

#: The four extractor kwargs (flag_registry rows) that declare the heads, in build order.
RIDEALONG_FLAGS: Tuple[str, ...] = ("ridealong_ensemble", "ridealong_rnd", "ridealong_adv",
                                    "ridealong_opp")

#: Bit offsets into the per-state hash, one block per ensemble so the V, A and B members' bootstrap
#: masks are independent of one another.
_BITS_V, _BITS_A, _BITS_B = 0, 12, 24
_MAX_MEMBERS = 12
#: RND observation normalisation clip (Burda et al. 2018).
RND_OBS_CLIP = 5.0
#: Running-statistics momentum for the RND error z-score (an EMA, so the scale follows the
#: predictor as it learns rather than averaging over the whole run).
RND_ERR_EMA = 0.01


@dataclass(frozen=True)
class RideAlongSpec:
    """Which heads exist. Read off the extractor's four ride-along kwargs."""
    ensemble: int = 0
    rnd: bool = False
    adv: int = 0
    opp: int = 0

    @property
    def any(self) -> bool:
        return bool(self.ensemble or self.rnd or self.adv or self.opp)

    @classmethod
    def from_extractor(cls, fe: object) -> "RideAlongSpec":
        return cls(ensemble=int(getattr(fe, "ridealong_ensemble", 0) or 0),
                   rnd=bool(getattr(fe, "ridealong_rnd", False)),
                   adv=int(getattr(fe, "ridealong_adv", 0) or 0),
                   opp=int(getattr(fe, "ridealong_opp", 0) or 0))


def validate_spec(spec: RideAlongSpec) -> None:
    for name, k in (("ridealong_ensemble", spec.ensemble), ("ridealong_adv", spec.adv),
                    ("ridealong_opp", spec.opp)):
        if not 0 <= int(k) <= _MAX_MEMBERS:
            raise ValueError(f"{name} must be in [0, {_MAX_MEMBERS}] members, got {k}")


def freeze_to_buffers(module: torch.nn.Module) -> torch.nn.Module:
    """Turn every PARAMETER of ``module`` into a BUFFER of the same name and value.

    A frozen random network (an RND target, a randomized prior) is never trained, so it must not be
    a parameter at all: a parameter with ``requires_grad=False`` still appears in
    ``policy.parameters()`` — in the compile gate's coverage count, in every "which parameters
    moved" read — and invites someone to hand it to an optimizer. Buffers ride the state_dict (so a
    checkpoint restores the SAME prior) and ``.to(device)`` moves them."""
    for mod in list(module.modules()):
        for name, p in list(mod.named_parameters(recurse=False)):
            data = p.detach().clone()
            delattr(mod, name)
            mod.register_buffer(name, data)
    return module


def _mlp(dims: List[int]) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def _scale_last_linear_(net: torch.nn.Module, std: float) -> None:
    """Re-draw a prior's LAST linear so its output has ~``std`` spread (normal init, zero bias) —
    the prior's scale is what sets how far apart untrained members start."""
    last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.normal_(last.weight, 0.0, std / math.sqrt(last.in_features))
    torch.nn.init.zeros_(last.bias)


# ── the bootstrap hash ───────────────────────────────────────────────────────────────────────────
_MIX_C1 = -49064778989728563      # 0xff51afd7ed558ccd as a signed int64 (murmur3 fmix64)
_MIX_C2 = -4265267296055464877    # 0xc4ceb9fe1a85ec53


def state_hash(obs: torch.Tensor, mult: torch.Tensor) -> torch.Tensor:
    """[B] int64 — a hash of each observation ROW, exact integer arithmetic (wrapping int64), so it
    is identical on CPU and CUDA, in every epoch and whatever minibatch the row lands in. The
    bootstrap masks are bits of it: a per-STATE Bernoulli(0.5) that draws no random number."""
    bits = obs.detach().float().contiguous().view(torch.int32).to(torch.int64)
    h = (bits * mult[: bits.shape[-1]]).sum(dim=-1)
    h = h ^ (h >> 33)
    h = h * _MIX_C1
    h = h ^ (h >> 33)
    h = h * _MIX_C2
    return h ^ (h >> 33)


def bootstrap_mask(h: torch.Tensor, k: int, offset: int) -> torch.Tensor:
    """[B, k] float {0, 1}: bit ``offset + j`` of the state hash for member j."""
    shifts = torch.arange(offset, offset + k, device=h.device, dtype=torch.int64)
    return ((h[:, None] >> shifts[None, :]) & 1).to(torch.float32)


# ── the heads ─────────────────────────────────────────────────────────────────────────────────────
class ValueEnsemble(torch.nn.Module):
    """K win-probability members over LayerNorm(detached `value_pooled`), each = trained MLP +
    RIDEALONG_PRIOR_SCALE_V · frozen random prior, in LOGIT units."""

    def __init__(self, k: int, in_dim: int = D_MODEL, hidden: int = RIDEALONG_HIDDEN) -> None:
        super().__init__()
        self.k, self.in_dim = int(k), int(in_dim)
        self.members = torch.nn.ModuleList()
        self.priors = torch.nn.ModuleList()
        for j in range(self.k):
            torch.manual_seed(RIDEALONG_INIT_SEED + 101 * (j + 1))
            self.members.append(_mlp([self.in_dim, hidden, 1]))
            prior = _mlp([self.in_dim, hidden, 1])
            _scale_last_linear_(prior, 1.0)
            self.priors.append(freeze_to_buffers(prior))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(pooled, (self.in_dim,))
        return torch.cat([m(x) + RIDEALONG_PRIOR_SCALE_V * p(x)
                          for m, p in zip(self.members, self.priors)], dim=-1)   # [B, K]


class RndNovelty(torch.nn.Module):
    """Random Network Distillation over the raw observation (see the module docstring for why the
    observation and not the trunk features)."""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        torch.manual_seed(RIDEALONG_INIT_SEED + 7)
        target = _mlp([self.obs_dim, RIDEALONG_RND_HIDDEN, RIDEALONG_RND_OUT])
        self.target = freeze_to_buffers(target)
        self.predictor = _mlp([self.obs_dim, RIDEALONG_RND_HIDDEN, RIDEALONG_RND_HIDDEN,
                               RIDEALONG_RND_OUT])
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_var", torch.ones(self.obs_dim))
        self.register_buffer("obs_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("err_mean", torch.zeros(()))
        self.register_buffer("err_var", torch.ones(()))
        self.register_buffer("err_seen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update_obs_stats(self, obs: torch.Tensor) -> None:
        """Chan et al.'s parallel mean/variance merge of one batch into the running statistics."""
        x = obs.detach().float()
        n_b = float(x.shape[0])
        if n_b == 0:
            return
        m_b = x.mean(0)
        v_b = x.var(0, unbiased=False)
        n_a = float(self.obs_count.item())
        n = n_a + n_b
        delta = m_b - self.obs_mean
        self.obs_mean.add_(delta * (n_b / n))
        self.obs_var.copy_((self.obs_var * n_a + v_b * n_b + delta * delta * (n_a * n_b / n)) / n)
        self.obs_count.fill_(n)

    def normalise(self, obs: torch.Tensor) -> torch.Tensor:
        if float(self.obs_count.item()) == 0.0:
            x = obs.float()
        else:
            x = (obs.float() - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        return x.clamp(-RND_OBS_CLIP, RND_OBS_CLIP)

    def error(self, obs: torch.Tensor) -> torch.Tensor:
        """[B] mean squared prediction error — the raw novelty (graph through the predictor only)."""
        x = self.normalise(obs.detach())
        with torch.no_grad():
            tgt = self.target(x)
        err: torch.Tensor = (self.predictor(x) - tgt).pow(2).mean(dim=-1)
        return err

    @torch.no_grad()
    def update_err_stats(self, err: torch.Tensor) -> None:
        e = err.detach().float()
        if e.numel() == 0:
            return
        m, v = e.mean(), e.var(unbiased=False)
        if not bool(self.err_seen.item()):
            self.err_mean.copy_(m)
            self.err_var.copy_(torch.clamp(v, min=1e-12))
            self.err_seen.fill_(True)
            return
        self.err_mean.mul_(1 - RND_ERR_EMA).add_(RND_ERR_EMA * m)
        self.err_var.mul_(1 - RND_ERR_EMA).add_(RND_ERR_EMA * v)

    def zscore(self, err: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = (err.detach() - self.err_mean) / torch.sqrt(self.err_var + 1e-12)
        return z


class AdvantageEnsemble(torch.nn.Module):
    """K per-action A members (the `QWinProbHead` shared-scorer structure over the pointer head's
    own per-action tokens, value context), each + RIDEALONG_PRIOR_SCALE_Q · a frozen random copy.
    Output RAW [B, K, 11]; the centring under π is applied by `RideAlongHeads.readout`."""

    def __init__(self, k: int, *, move_token_dim: int, move_cell_dim: int,
                 switch_cell_dim: int) -> None:
        super().__init__()
        self.k = int(k)
        kw = dict(move_token_dim=move_token_dim, d_model=D_MODEL, ctx_dim=D_MODEL,
                  move_cell_dim=move_cell_dim, switch_cell_dim=switch_cell_dim,
                  hidden=POINTER_HIDDEN)
        self.members = torch.nn.ModuleList()
        self.priors = torch.nn.ModuleList()
        for j in range(self.k):
            torch.manual_seed(RIDEALONG_INIT_SEED + 1009 * (j + 1))
            self.members.append(QWinProbHead(**kw))   # zero-init scorer: A = 0 + prior at init
            prior = QWinProbHead(**kw)
            torch.nn.init.normal_(prior.q_score.weight, 0.0, 1.0 / math.sqrt(POINTER_HIDDEN))
            self.priors.append(freeze_to_buffers(prior))

    def forward(self, ctx: torch.Tensor, pointer: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        tok, valid, team, mcell, scell = pointer
        return torch.stack([m(ctx, tok, valid, team, mcell, scell)
                            + RIDEALONG_PRIOR_SCALE_Q * p(ctx, tok, valid, team, mcell, scell)
                            for m, p in zip(self.members, self.priors)], dim=1)   # [B, K, 11]


class _OppScorer(torch.nn.Module):
    """One B member: seat k scored from its concrete MOVE ID (own embedding) + the value context,
    SWITCH from the context alone — α's support, α's shared-scorer shape."""

    def __init__(self, n_moves: int, hidden: int = POINTER_HIDDEN) -> None:
        super().__init__()
        self.n_moves = int(n_moves)
        self.ctx_proj = torch.nn.Linear(D_MODEL, hidden)
        self.move_emb = torch.nn.Embedding(self.n_moves, RIDEALONG_OPP_MOVE_EMB)
        self.move_proj = torch.nn.Linear(RIDEALONG_OPP_MOVE_EMB, hidden)
        self.seat_score = torch.nn.Linear(hidden, 1)
        self.switch_score = torch.nn.Linear(hidden, 1)
        for lin in (self.seat_score, self.switch_score):
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, ctx: torch.Tensor, seat_nums: torch.Tensor) -> torch.Tensor:
        c = self.ctx_proj(ctx)                                                  # [B,H]
        ids = seat_nums.long().clamp(0, self.n_moves - 1)
        h = torch.tanh(self.move_proj(self.move_emb(ids)) + c[:, None, :])      # [B,K,H]
        seat = self.seat_score(h).squeeze(-1)                                   # [B,K]
        sw = self.switch_score(torch.tanh(c))                                   # [B,1]
        return torch.cat([seat, sw], dim=-1)                                    # [B,K+1]


class OppEffectEnsemble(torch.nn.Module):
    def __init__(self, k: int, n_moves: int) -> None:
        super().__init__()
        self.k = int(k)
        self.members = torch.nn.ModuleList()
        self.priors = torch.nn.ModuleList()
        for j in range(self.k):
            torch.manual_seed(RIDEALONG_INIT_SEED + 10007 * (j + 1))
            self.members.append(_OppScorer(n_moves))
            prior = _OppScorer(n_moves)
            for lin in (prior.seat_score, prior.switch_score):
                torch.nn.init.normal_(lin.weight, 0.0, 1.0 / math.sqrt(POINTER_HIDDEN))
            self.priors.append(freeze_to_buffers(prior))

    def forward(self, ctx: torch.Tensor, seat_nums: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(ctx, seat_nums) + RIDEALONG_PRIOR_SCALE_Q * p(ctx, seat_nums)
                            for m, p in zip(self.members, self.priors)], dim=1)   # [B, K, S+1]


# ── the batch, the readout and the losses ────────────────────────────────────────────────────────
@dataclass
class RideAlongBatch:
    """Everything one ride-along step reads, ALREADY DETACHED (`detached()` builds it). Labels are
    None where the run does not emit them; a head whose label is absent trains nothing."""
    obs: torch.Tensor                        # [B, obs_dim] the raw observation vector
    pooled: torch.Tensor                     # [B, D_MODEL] value_pooled
    pointer: Optional[Tuple[torch.Tensor, ...]]   # PointerInputs, detached
    pi: Optional[torch.Tensor]               # [B, 11] the policy's masked probabilities
    logits: Optional[torch.Tensor]           # [B, 11] the policy's raw pointer logits
    legal: Optional[torch.Tensor]            # [B, 11] bool
    values: Optional[torch.Tensor]           # [B] V (the critic's value)
    actions: Optional[torch.Tensor] = None   # [B] long
    advantages: Optional[torch.Tensor] = None  # [B] the GAE advantage (raw, un-normalised)
    win_target: Optional[torch.Tensor] = None  # [B]
    win_mask: Optional[torch.Tensor] = None    # [B]
    alpha_logits: Optional[torch.Tensor] = None    # [B, K+1]
    alpha_seat_nums: Optional[torch.Tensor] = None  # [B, K]
    opp_kind: Optional[torch.Tensor] = None     # [B] long (0 move, 1 switch, 2 unknown)
    opp_num: Optional[torch.Tensor] = None      # [B] long

    @staticmethod
    def detached(**kw: object) -> "RideAlongBatch":
        """Build a batch with every tensor `.detach()`ed — THE stop-grad seam. Nothing downstream of
        this constructor can reach a graph the policy or V is part of."""
        out: Dict[str, object] = {}
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach()
            elif isinstance(v, tuple):
                out[k] = tuple(t.detach() for t in v)
            else:
                out[k] = v
        return RideAlongBatch(**out)  # type: ignore[arg-type]


def _flat(t: torch.Tensor) -> torch.Tensor:
    """[B, ...] -> [B]: the first column of a per-row label (a Dict-obs label is [B, 1])."""
    return t.reshape(t.shape[0], -1)[:, 0]


class RideAlongHeads(torch.nn.Module):
    """The container: whichever of the four heads the spec builds, plus the bootstrap hash."""

    def __init__(self, spec: RideAlongSpec, *, obs_dim: int, move_token_dim: int,
                 move_cell_dim: int, switch_cell_dim: int, n_moves: int) -> None:
        super().__init__()
        validate_spec(spec)
        self.spec = spec
        self.obs_dim = int(obs_dim)
        torch.manual_seed(RIDEALONG_INIT_SEED)
        mult = torch.randint(1, 2 ** 62, (self.obs_dim,), dtype=torch.int64) | 1
        self.register_buffer("hash_mult", mult)
        self.ensemble = ValueEnsemble(spec.ensemble) if spec.ensemble else None
        self.rnd = RndNovelty(self.obs_dim) if spec.rnd else None
        self.adv = (AdvantageEnsemble(spec.adv, move_token_dim=move_token_dim,
                                      move_cell_dim=move_cell_dim,
                                      switch_cell_dim=switch_cell_dim) if spec.adv else None)
        self.opp = OppEffectEnsemble(spec.opp, n_moves) if spec.opp else None

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def readout(self, b: RideAlongBatch) -> Dict[str, torch.Tensor]:
        """Every head's output on one batch (graph through the HEADS' OWN parameters only).

        Keys: ``ens_logits`` [B,K] · ``ens_p`` [B] (member mean) · ``ens_std`` [B] (disagreement) ·
        ``rnd_err`` [B] · ``rnd_z`` [B] · ``adv`` [B,K,11] centred under π · ``adv_mean`` /
        ``adv_std`` [B,11] · ``opp`` [B,K,S+1] centred under α · ``opp_mean`` [B,S+1]."""
        out: Dict[str, torch.Tensor] = {}
        if self.ensemble is not None:
            lg = self.ensemble(b.pooled)
            p = torch.sigmoid(lg)
            out["ens_logits"], out["ens_p"] = lg, p.mean(-1)
            out["ens_std"] = p.std(-1, unbiased=False)
            # The members' spread in LOGIT space. A std of PROBABILITIES is mechanically largest
            # where the logits sit near 0, i.e. where V itself is near 0.5, so it partly re-reads V's
            # own (aleatoric) uncertainty. The logit spread is not squashed by the sigmoid, and it
            # is the disagreement the uncertainty meters score.
            out["ens_logit_std"] = lg.std(-1, unbiased=False)
        if self.rnd is not None:
            err = self.rnd.error(b.obs)
            out["rnd_err"] = err
            out["rnd_z"] = self.rnd.zscore(err)
        if self.adv is not None and b.pointer is not None and b.pi is not None:
            raw = self.adv(b.pooled, b.pointer)                                  # [B,K,11]
            centred = raw - (raw * b.pi[:, None, :]).sum(-1, keepdim=True)
            out["adv"] = centred
            out["adv_mean"] = centred.mean(1)
            # Per-ACTION member spread, measured after centring each member on its OWN mean over
            # the LEGAL actions (uniform weights) — NOT on the π-centred values. π-centring pulls the
            # members together on exactly the actions π plays (each member's offset is mostly set
            # by them), so a π-centred spread reads "starved ⇒ uncertain" even on untrained heads
            # (the reader measured it: AUROC 0.73–0.75 at init). Uniform centring removes each
            # member's free offset without favouring any action.
            legal = (b.legal.float() if b.legal is not None else (b.pi > 0).float())[:, None, :]
            n_legal = legal.sum(-1, keepdim=True).clamp(min=1.0)
            uni = raw - (raw * legal).sum(-1, keepdim=True) / n_legal
            out["adv_std"] = uni.std(1, unbiased=False)
        if (self.opp is not None and b.alpha_logits is not None
                and b.alpha_seat_nums is not None):
            alpha = torch.softmax(b.alpha_logits.float(), dim=-1)                # [B,S+1]
            raw = self.opp(b.pooled, b.alpha_seat_nums)                          # [B,K,S+1]
            centred = raw - (raw * alpha[:, None, :]).sum(-1, keepdim=True)
            out["opp"] = centred
            out["opp_mean"] = centred.mean(1)
        return out

    def losses(self, b: RideAlongBatch, out: Dict[str, torch.Tensor], *,
               train_rnd: bool = True) -> Dict[str, torch.Tensor]:
        """Per-head scalar losses (MSE / BCE, LINEAR in the labels' first moment — §5.1). A head
        with no usable label this batch contributes no key."""
        from agents.model.opp_intent import INTENT_IGNORE, match_seats_to_move_num

        losses: Dict[str, torch.Tensor] = {}
        h = state_hash(b.obs, self.hash_mult)
        if "ens_logits" in out and b.win_target is not None:
            z = _flat(b.win_target).float()
            m = (_flat(b.win_mask).float() if b.win_mask is not None
                 else torch.ones_like(z))
            w = bootstrap_mask(h, self.spec.ensemble, _BITS_V) * m[:, None]      # [B,K]
            if bool((w.sum(0) > 0).any()):
                bce = F.binary_cross_entropy_with_logits(
                    out["ens_logits"], z[:, None].expand_as(out["ens_logits"]), reduction="none")
                losses["ens"] = ((bce * w).sum(0) / w.sum(0).clamp(min=1.0)).sum()
        if "rnd_err" in out and train_rnd:
            losses["rnd"] = out["rnd_err"].mean()
        if "adv" in out and b.actions is not None and b.advantages is not None:
            a = b.actions.long().reshape(-1)
            y = _flat(b.advantages).float()
            pred = out["adv"].gather(-1, a[:, None, None].expand(-1, self.spec.adv, 1)).squeeze(-1)
            w = bootstrap_mask(h, self.spec.adv, _BITS_A)
            se = (pred - y[:, None]).pow(2)
            losses["adv"] = ((se * w).sum(0) / w.sum(0).clamp(min=1.0)).sum()
        if ("opp" in out and b.advantages is not None and b.opp_kind is not None
                and b.opp_num is not None and b.alpha_seat_nums is not None):
            n_seats = int(b.alpha_seat_nums.shape[-1])
            idx = match_seats_to_move_num(b.alpha_seat_nums, b.opp_num.long().reshape(-1),
                                          b.opp_kind.long().reshape(-1), n_seats)
            ok = (idx != INTENT_IGNORE).float()
            if bool(ok.any()):
                safe = torch.where(idx == INTENT_IGNORE, torch.zeros_like(idx), idx)
                pred = out["opp"].gather(
                    -1, safe[:, None, None].expand(-1, self.spec.opp, 1)).squeeze(-1)
                y = _flat(b.advantages).float()
                w = bootstrap_mask(h, self.spec.opp, _BITS_B) * ok[:, None]
                se = (pred - y[:, None]).pow(2)
                losses["opp"] = ((se * w).sum(0) / w.sum(0).clamp(min=1.0)).sum()
        return losses


def build_ridealong(fe: object, *, obs_dim: int,
                    spec: Optional[RideAlongSpec] = None) -> Optional[RideAlongHeads]:
    """Build the heads the extractor's kwargs declare (or an explicit ``spec`` — the offline readers
    attach FRESH heads to a checkpoint trained without them), or None. PRIVATE RNG: the global
    stream is saved and restored around the whole construction (`fork_rng`), so building them draws
    nothing the policy's own init or any later sample would see."""
    spec = RideAlongSpec.from_extractor(fe) if spec is None else spec
    if not spec.any:
        return None
    if spec.ensemble and getattr(fe, "win_head", None) is None:
        raise ValueError("ridealong_ensemble requires win_prob_mode != 'none': the members predict "
                         "V's own win-probability target, which only a win head's run emits.")
    if spec.opp and not bool(getattr(fe, "opp_intent", False)):
        raise ValueError("ridealong_opp requires opp_intent: B's columns are α's support (their "
                         "believed move seats + SWITCH) and its centring reads α.")
    layout = getattr(fe, "layout", None) or {}
    # α's seat nums index the extractor's own move table (`layout['max_moves']` rows).
    n_moves = int(layout.get("max_moves", 400)) if isinstance(layout, dict) else 400
    with torch.random.fork_rng(devices=[]):
        return RideAlongHeads(
            spec, obs_dim=int(obs_dim),
            move_token_dim=int(getattr(fe, "pointer_move_token_dim", D_MODEL)),
            move_cell_dim=int(getattr(fe, "pointer_move_cell_dim", 0)),
            switch_cell_dim=int(getattr(fe, "pointer_switch_cell_dim", 0)),
            n_moves=n_moves)


__all__ = ["RIDEALONG_FLAGS", "RideAlongSpec", "RideAlongHeads", "RideAlongBatch",
           "build_ridealong", "freeze_to_buffers", "state_hash", "bootstrap_mask",
           "ACTION_SPACE_SIZE"]
