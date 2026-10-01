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

THE RND VARIANT ENSEMBLE (`--ridealong-rnd-variants`, `gen3_ridealong_rnd_variants_v1`, config
v127; owner 2026-09-30: *"ensemble RND, toss one a different learning rate or something, so we knock
them out all at once … and we can compare RND strategies"*). Plain RND counts CUMULATIVE visitation,
so it can saturate. "Recently seen" needs FORGETTING, and the lever for forgetting is the PREDICTOR's
memory, not the target. Each variant is its own detached predictor with its own optimizer state,
its own error z-score and its own logged statistics, beside the unchanged `--ridealong-rnd` head
(`base`, the reference). `RND_VARIANT_DECLS` holds every hyperparameter and the reason for it:

  * ``fast``  — base's predictor (same init) at RND_FAST_LR_MULT × base's rate: forgetting by fast
    tracking.
  * ``decay`` — base's predictor (same init), pulled toward its OWN INIT once per PPO update with a
    half-life of RND_DECAY_HALF_LIFE_UPDATES updates: forgetting by shrinkage ("shrink toward init",
    L2-Init — Ash & Adams 2020, Kumar et al. 2023). A state not revisited drifts back to the
    untrained error, i.e. back to "novel".
  * ``small`` — a one-hidden-layer predictor of RIDEALONG_RND_SMALL_HIDDEN units (11.5 % of base's
    parameters): too little capacity to fingerprint battles, so it should track coarse density.
  * ``feat``  — base's shapes over the detached trunk features (`value_pooled`), its own frozen
    target and its own feature normalisation: measures LIVE the representation drift the offline
    read predicted (2.5–8×).

The three OBSERVATION variants share base's frozen target AND base's observation normalisation
(the same stream gives identical running statistics, so sharing is exact and makes every comparison
PAIRED: one target output per row, several predictors). `fast` and `decay` also start from base's
exact predictor weights, so at step 0 they ARE base and every later difference is the strategy.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

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
    RIDEALONG_RND_SMALL_HIDDEN,
)
from agents.model.q_winprob_head import QWinProbHead
from agents.training.lifecycle_decl import startup_builder

#: The four extractor kwargs (flag_registry rows) that declare the heads, in build order.
RIDEALONG_FLAGS: Tuple[str, ...] = ("ridealong_ensemble", "ridealong_rnd", "ridealong_adv",
                                    "ridealong_opp", "ridealong_rnd_variants")

#: Bit offsets into the per-state hash, one block per ensemble so the V, A and B members' bootstrap
#: masks are independent of one another.
_BITS_V, _BITS_A, _BITS_B = 0, 12, 24
_MAX_MEMBERS = 12
#: RND observation normalisation clip (Burda et al. 2018).
RND_OBS_CLIP = 5.0
#: Running-statistics momentum for the RND error z-score (an EMA, so the scale follows the
#: predictor as it learns rather than averaging over the whole run).
RND_ERR_EMA = 0.01

# ── the RND variant ensemble's DECLARATIONS (gen3_ridealong_rnd_variants_v1, v127) ───────────────
#: `fast`'s predictor learning rate as a multiple of base's (`RIDEALONG_LR`, 3e-4 → 3e-3). WHY 10:
#: Adam's per-step move is ~lr whatever the gradient's scale, so a predictor's tracking timescale in
#: steps scales ~1/lr, and 10× puts `fast`'s memory roughly an order of magnitude shorter than base's.
#: That is the top of the 5–10× range, chosen so that the contrast with base is large enough to be
#: detectable on one run. 3e-3 is still an ordinary Adam rate for an MLP regression on inputs clipped
#: to ±5 with the gradient clipped to 1.0; a non-finite step disables `fast` alone, never the run.
RND_FAST_LR_MULT = 10.0
#: `decay`'s half-life toward its own init, in PPO UPDATES. WHY 10: one eval cycle is 2,000,000 env
#: steps (`EVAL_FREQ_STEPS`) ≈ 20 updates of the production 98,304-row rollout, so 10 updates is half
#: an eval cycle. Learning that is not renewed is 25 % left after one cycle and 6 % after two, and at
#: equilibrium the predictor holds about 1 / (1 − γ) ≈ 15 updates of learning. So `decay`'s
#: "familiar" means "visited during roughly the current eval cycle", which is the cadence the
#: saturation read (X26 amendment (c)) is aggregated at.
RND_DECAY_HALF_LIFE_UPDATES = 10.0


@dataclass(frozen=True)
class RndVariantDecl:
    """One declared RND variant: what it reads, how it forgets, and the predictor-vs-target design."""
    name: str
    input: str                         # "obs" (base's normalised observation) | "feat" (value_pooled)
    lr_mult: float                     # × RIDEALONG_LR, the variant's own Adam rate
    decay_half_life_updates: Optional[float]   # pull toward init per PPO update; None = no pull
    predictor_vs_target: str           # the architecture relation, recorded (owner's question)
    why: str


#: The declared variants, in CANONICAL order (the recorded flag value is this order's comma join).
#: `base` is NOT one of them: it is the unchanged `--ridealong-rnd` head, the reference every variant
#: is compared with, and it is REQUIRED (the observation variants share its target).
RND_VARIANT_DECLS: Tuple[RndVariantDecl, ...] = (
    RndVariantDecl(
        "fast", "obs", RND_FAST_LR_MULT, None,
        "SAME family as base: base's own predictor (obs→256→256→64 ReLU MLP, one layer deeper "
        "than the obs→256→64 target), started from base's exact weights",
        "forgetting by fast tracking: a 10x rate overwrites old fits faster"),
    RndVariantDecl(
        "decay", "obs", 1.0, RND_DECAY_HALF_LIFE_UPDATES,
        "SAME family as base: base's own predictor, started from base's exact weights, pulled back "
        "toward them",
        "forgetting by shrinkage: unrenewed learning decays back to the untrained (novel) error"),
    RndVariantDecl(
        "small", "obs", 1.0, None,
        "DELIBERATELY LESS EXPRESSIVE than the target: obs→32→64 (one hidden layer of 32) against "
        "the target's 256 hidden units, so it cannot represent the target exactly",
        "no fingerprinting: too little capacity to memorise battles; novelty tracks coarse density"),
    RndVariantDecl(
        "feat", "feat", 1.0, None,
        "SAME family as base, over value_pooled: target D→256→64, predictor D→256→256→64 (D = "
        "D_MODEL), its own frozen target and its own feature normalisation",
        "the trunk-feature input, live: measures the representation drift the offline read predicted"),
)
RND_VARIANTS: Tuple[str, ...] = tuple(d.name for d in RND_VARIANT_DECLS)
RND_VARIANT_BY_NAME: Dict[str, RndVariantDecl] = {d.name: d for d in RND_VARIANT_DECLS}
#: The OFF value of the recorded string (the registry's OFF convention, `flag_registry.is_enabled`).
RND_VARIANTS_OFF = "off"


def parse_rnd_variants(value: Any) -> Tuple[str, ...]:
    """The declared variant names in CANONICAL order. Accepts None / '' / 'off' / 'none' (no
    variants), 'all', a comma list, or an iterable of names. An unknown or repeated name RAISES —
    a typo must never quietly drop a variant from a pre-registered comparison."""
    if value is None:
        return ()
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("", "off", "none"):
            return ()
        if v == "all":
            return RND_VARIANTS
        names = [x.strip() for x in v.split(",") if x.strip()]
    else:
        names = [str(x).strip().lower() for x in value]
    bad = sorted(set(names) - set(RND_VARIANTS))
    if bad:
        raise ValueError(f"unknown RND variant(s) {bad}: the declared set is {list(RND_VARIANTS)} "
                         "('all' = every one; 'base' is --ridealong-rnd itself, not a variant)")
    if len(set(names)) != len(names):
        raise ValueError(f"an RND variant is repeated in {names}")
    return tuple(n for n in RND_VARIANTS if n in names)


def canonical_rnd_variants(value: Any) -> str:
    """The RECORDED form: 'off', or the canonical comma join (so 'decay,fast' and 'fast,decay'
    record — and version-gate — identically)."""
    names = parse_rnd_variants(value)
    return ",".join(names) if names else RND_VARIANTS_OFF


def obs_block_edges(layout: Optional[Mapping[str, Any]], obs_dim: int) -> Tuple[int, ...]:
    """The observation's BLOCK boundaries, read from the encoder's layout (never hardcoded): every
    `parts` block start plus the event window, in [0, obs_dim]. Fallback (no layout, e.g. a toy test
    policy): 7 equal blocks. The identification probes build their off-manifold rows from these."""
    edges = {0, int(obs_dim)}
    parts = (layout or {}).get("parts") if isinstance(layout, Mapping) else None
    if isinstance(parts, Mapping):
        for p in parts.values():
            st = int(p.get("start", -1)) if isinstance(p, Mapping) else -1
            if 0 < st < obs_dim:
                edges.add(st)
        ew = (layout or {}).get("event_window_offset")
        if isinstance(ew, int) and 0 < ew < obs_dim:
            edges.add(int(ew))
    if len(edges) < 3:
        edges |= {int(round(obs_dim * k / 7)) for k in range(1, 7)}
    return tuple(sorted(edges))


def block_chimera(x: torch.Tensor, edges: Sequence[int]) -> torch.Tensor:
    """The OFF-MANIFOLD probe generator (`chimera_v1`, frozen). Row j's block k is copied from row
    ``(j + k · stride) mod B`` (stride = max(1, B // n_blocks); block 0 stays row j's own), so every
    block is a REAL block of a real observation while the combination is one no battle produced:
    our team from one battle, the opponent's from another, the field and history from others. It is
    DETERMINISTIC and draws no random number. Over a SHUFFLED batch (a PPO minibatch, or the
    reader's seeded probe order) the donors are other battles. Per-dimension normalisation commutes
    with it, so it can be applied to normalised rows."""
    n = int(x.shape[0])
    nb = len(edges) - 1
    stride = max(1, n // max(1, nb))
    out = x.clone()
    for k in range(1, nb):
        lo, hi = int(edges[k]), int(edges[k + 1])
        out[:, lo:hi] = torch.roll(x[:, lo:hi], shifts=-(k * stride) % max(1, n), dims=0)
    return out


@dataclass(frozen=True)
class RideAlongSpec:
    """Which heads exist. Read off the extractor's five ride-along kwargs."""
    ensemble: int = 0
    rnd: bool = False
    adv: int = 0
    opp: int = 0
    #: The RND VARIANTS beside base (`RND_VARIANTS` names, canonical order; () = none).
    rnd_variants: Tuple[str, ...] = ()

    @property
    def any(self) -> bool:
        return bool(self.ensemble or self.rnd or self.adv or self.opp or self.rnd_variants)

    @classmethod
    def from_extractor(cls, fe: object) -> "RideAlongSpec":
        return cls(ensemble=int(getattr(fe, "ridealong_ensemble", 0) or 0),
                   rnd=bool(getattr(fe, "ridealong_rnd", False)),
                   adv=int(getattr(fe, "ridealong_adv", 0) or 0),
                   opp=int(getattr(fe, "ridealong_opp", 0) or 0),
                   rnd_variants=parse_rnd_variants(getattr(fe, "ridealong_rnd_variants", None)))


def validate_spec(spec: RideAlongSpec) -> None:
    for name, k in (("ridealong_ensemble", spec.ensemble), ("ridealong_adv", spec.adv),
                    ("ridealong_opp", spec.opp)):
        if not 0 <= int(k) <= _MAX_MEMBERS:
            raise ValueError(f"{name} must be in [0, {_MAX_MEMBERS}] members, got {k}")
    if tuple(spec.rnd_variants) != parse_rnd_variants(spec.rnd_variants):
        raise ValueError(f"rnd_variants must be declared names in canonical order "
                         f"{list(RND_VARIANTS)}, got {spec.rnd_variants!r}")
    if spec.rnd_variants and not spec.rnd:
        raise ValueError("ridealong_rnd_variants requires ridealong_rnd: the observation variants "
                         "share base's frozen target and normalisation, and base is the reference "
                         "every variant is compared with.")


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


@startup_builder
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


class _RndErrStats(torch.nn.Module):
    """The error z-score's running statistics (an EMA of the batch mean and variance). Each RND
    head — base and every variant — owns ONE, because their error scales differ by construction."""

    # Registered buffers, declared for the type checker only (torch >= 2.8 types an undeclared
    # Module attribute as `Tensor | Module`); a bare annotation creates no class attribute.
    err_mean: torch.Tensor
    err_var: torch.Tensor
    err_seen: torch.Tensor

    def _register_err_stats(self) -> None:
        self.register_buffer("err_mean", torch.zeros(()))
        self.register_buffer("err_var", torch.ones(()))
        self.register_buffer("err_seen", torch.zeros((), dtype=torch.bool))

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


class RndNovelty(_RndErrStats):
    """Random Network Distillation over the raw observation (see the module docstring for why the
    observation and not the trunk features). ``seed`` defaults to base's; the `feat` variant is this
    class over `value_pooled` with its own seed."""

    obs_mean: torch.Tensor      # registered buffers — typing only (see _RndErrStats)
    obs_var: torch.Tensor
    obs_count: torch.Tensor

    def __init__(self, obs_dim: int, seed: Optional[int] = None) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        #: Set when this module is the `feat` VARIANT (its declaration); None for base.
        self.decl: Optional[RndVariantDecl] = None
        torch.manual_seed(RIDEALONG_INIT_SEED + 7 if seed is None else int(seed))
        target = _mlp([self.obs_dim, RIDEALONG_RND_HIDDEN, RIDEALONG_RND_OUT])
        self.target = freeze_to_buffers(target)
        self.predictor = _mlp([self.obs_dim, RIDEALONG_RND_HIDDEN, RIDEALONG_RND_HIDDEN,
                               RIDEALONG_RND_OUT])
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_var", torch.ones(self.obs_dim))
        self.register_buffer("obs_count", torch.zeros((), dtype=torch.float64))
        self._register_err_stats()

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

    def inputs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(x, target(x))``: the normalised input and the frozen target's output. Computed ONCE
        per batch and shared with the observation variants (paired comparisons)."""
        x = self.normalise(obs.detach())
        with torch.no_grad():
            tgt = self.target(x)
        return x, tgt

    def error_from(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        err: torch.Tensor = (self.predictor(x) - tgt).pow(2).mean(dim=-1)
        return err

    def error(self, obs: torch.Tensor) -> torch.Tensor:
        """[B] mean squared prediction error — the raw novelty (graph through the predictor only)."""
        return self.error_from(*self.inputs(obs))


class RndObsVariant(_RndErrStats):
    """An OBSERVATION RND variant: its own predictor and error statistics over BASE's normalised
    observation against BASE's frozen target (both shared, so the comparison is paired). With a
    declared half-life it keeps a frozen copy of its init (``anchor``, buffers) and is pulled back
    toward it once per PPO update (`shrink_toward_init_`)."""

    def __init__(self, decl: RndVariantDecl, predictor: torch.nn.Module) -> None:
        super().__init__()
        self.decl = decl
        self.predictor = predictor
        self.anchor: Optional[torch.nn.Module] = (
            freeze_to_buffers(copy.deepcopy(predictor))
            if decl.decay_half_life_updates is not None else None)
        self._register_err_stats()

    def error_from(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        err: torch.Tensor = (self.predictor(x) - tgt).pow(2).mean(dim=-1)
        return err

    @torch.no_grad()
    def shrink_toward_init_(self) -> None:
        """θ ← θ0 + γ (θ − θ0), γ = 2^(−1 / half-life): one PPO update's worth of the pull."""
        if self.anchor is None or self.decl.decay_half_life_updates is None:
            return
        gamma = 2.0 ** (-1.0 / float(self.decl.decay_half_life_updates))
        anchor = dict(self.anchor.named_buffers())
        for name, p in self.predictor.named_parameters():
            p.mul_(gamma).add_(anchor[name], alpha=1.0 - gamma)


@startup_builder
def build_rnd_variants(names: Sequence[str], base: RndNovelty,
                       feat_dim: int = D_MODEL) -> torch.nn.ModuleDict:
    """The declared variants beside ``base`` (call inside the heads' private RNG). `fast` and `decay`
    are DEEP COPIES of base's predictor (no draw); `small` and `feat` seed themselves privately, so
    adding or removing a variant never changes another's init."""
    out = torch.nn.ModuleDict()
    obs_dim = int(base.obs_dim)
    for name in names:
        decl = RND_VARIANT_BY_NAME[name]
        if name in ("fast", "decay"):
            out[name] = RndObsVariant(decl, copy.deepcopy(base.predictor))
        elif name == "small":
            torch.manual_seed(RIDEALONG_INIT_SEED + 7 + 31)
            out[name] = RndObsVariant(decl, _mlp([obs_dim, RIDEALONG_RND_SMALL_HIDDEN,
                                                  RIDEALONG_RND_OUT]))
        elif name == "feat":
            fv = RndNovelty(int(feat_dim), seed=RIDEALONG_INIT_SEED + 7 + 53)
            fv.decl = decl
            out[name] = fv
        else:                                    # pragma: no cover - parse_rnd_variants refuses
            raise ValueError(name)
    return out


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
    """The container: whichever of the four heads the spec builds, the RND variants beside base,
    plus the bootstrap hash."""

    hash_mult: torch.Tensor     # registered buffer — typing only (see _RndErrStats)

    def __init__(self, spec: RideAlongSpec, *, obs_dim: int, move_token_dim: int,
                 move_cell_dim: int, switch_cell_dim: int, n_moves: int,
                 block_edges: Optional[Sequence[int]] = None) -> None:
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
        # gen3_ridealong_rnd_variants_v1 (v127): built LAST, each from its own private seed (or a
        # deep copy of base's predictor), so every head above is bit-identical with or without them.
        self.rnd_variants: Optional[torch.nn.ModuleDict] = (
            build_rnd_variants(spec.rnd_variants, self.rnd)
            if spec.rnd_variants and self.rnd is not None else None)
        #: The observation's block boundaries, for the identification probes (not state: derived
        #: from the encoder's layout at every build).
        self.block_edges: Tuple[int, ...] = (tuple(int(e) for e in block_edges) if block_edges
                                             else obs_block_edges(None, self.obs_dim))

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """The FOUR heads' trainable parameters (base RND included) — the learner's shared optimizer.
        The RND variants are EXCLUDED: each has its own optimizer (`variant_parameters`)."""
        return [p for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("rnd_variants.")]

    def variant_names(self) -> Tuple[str, ...]:
        return tuple(self.rnd_variants.keys()) if self.rnd_variants is not None else ()

    def variant_parameters(self, name: str) -> List[torch.nn.Parameter]:
        assert self.rnd_variants is not None
        return [p for p in self.rnd_variants[name].parameters() if p.requires_grad]

    @torch.no_grad()
    def begin_update_(self, skip: Sequence[str] = ()) -> None:
        """Once per PPO update, before its first ride-along step: the `decay` variant's pull (not
        for a variant in ``skip`` — a disabled variant is frozen)."""
        for n, v in (self.rnd_variants.items() if self.rnd_variants is not None else ()):
            if isinstance(v, RndObsVariant) and n not in skip:
                v.shrink_toward_init_()

    def variant_losses(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``{"rndv_<name>": mean error}`` for every variant the readout scored — each is its own
        predictor's loss alone, stepped by its own optimizer."""
        return {f"rndv_{n}": out[f"rndv_{n}_err"].mean() for n in self.variant_names()
                if f"rndv_{n}_err" in out}

    @torch.no_grad()
    def identification_errors(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """The IN-RUN identification probe: base's and every OBSERVATION variant's error on the
        batch's `block_chimera` rows (off-manifold but plausible), against base's target. Compared
        with the same predictors' error on the real rows it says whether a predictor is learning the
        target EVERYWHERE (identification: the chimera error falls with the real-row error) rather
        than on the states it visits. `feat` is skipped (its chimera input would need a trunk
        forward); the offline reader scores it on fixed probes. Keys ``rnd`` / ``rndv_<name>``."""
        out: Dict[str, torch.Tensor] = {}
        if self.rnd is None:
            return out
        x = block_chimera(self.rnd.normalise(obs.detach()), self.block_edges)
        tgt = self.rnd.target(x)
        out["rnd"] = self.rnd.error_from(x, tgt)
        for n, v in (self.rnd_variants.items() if self.rnd_variants is not None else ()):
            if isinstance(v, RndObsVariant):
                out[f"rndv_{n}"] = v.error_from(x, tgt)
        return out

    def readout(self, b: RideAlongBatch,
                skip_variants: Sequence[str] = ()) -> Dict[str, torch.Tensor]:
        """Every head's output on one batch (graph through the HEADS' OWN parameters only).

        Keys: ``ens_logits`` [B,K] · ``ens_p`` [B] (member mean) · ``ens_std`` [B] (disagreement) ·
        ``rnd_err`` [B] · ``rnd_z`` [B] · ``rndv_<name>_err`` / ``rndv_<name>_z`` [B] per RND
        variant (not for a name in ``skip_variants``) · ``adv`` [B,K,11] centred under π ·
        ``adv_mean`` / ``adv_std`` [B,11] · ``opp`` [B,K,S+1] centred under α · ``opp_mean``
        [B,S+1]."""
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
            x, tgt = self.rnd.inputs(b.obs)
            err = self.rnd.error_from(x, tgt)
            out["rnd_err"] = err
            out["rnd_z"] = self.rnd.zscore(err)
            for n, v in (self.rnd_variants.items() if self.rnd_variants is not None else ()):
                if n in skip_variants:
                    continue
                # Observation variants: base's normalised rows and target output, shared (paired).
                # `feat`: its own normalisation and target over the detached value_pooled.
                e = (v.error_from(x, tgt) if isinstance(v, RndObsVariant)
                     else cast(RndNovelty, v).error(b.pooled))
                out[f"rndv_{n}_err"] = e
                out[f"rndv_{n}_z"] = cast(_RndErrStats, v).zscore(e)
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


@startup_builder
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
            n_moves=n_moves,
            block_edges=obs_block_edges(layout if isinstance(layout, dict) else None,
                                        int(obs_dim)))


__all__ = ["RIDEALONG_FLAGS", "RideAlongSpec", "RideAlongHeads", "RideAlongBatch",
           "build_ridealong", "freeze_to_buffers", "state_hash", "bootstrap_mask",
           "ACTION_SPACE_SIZE", "RND_VARIANTS", "RND_VARIANT_DECLS", "RND_VARIANT_BY_NAME",
           "RND_FAST_LR_MULT", "RND_DECAY_HALF_LIFE_UPDATES", "RndVariantDecl", "RndObsVariant",
           "parse_rnd_variants", "canonical_rnd_variants", "obs_block_edges", "block_chimera",
           "build_rnd_variants"]
