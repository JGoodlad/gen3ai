"""The DECISION forward every inference slot runs: ``(obs, mask) -> (legal log-probs, V)``.

One pure tensor function per policy (``DecisionModule``), so the service's eager, compiled and
ahead-of-time backends all trace, compile or export the SAME callable, and the parity gate compares
them against the SAME eager reference.

WHY NOT ``policy.forward`` / ``get_distribution``. Those build an sb3 ``MaskableCategorical``
object — a Python distribution whose constructor pops ``__dict__`` entries and validates its
arguments with data-dependent checks (four graph breaks under dynamo, an export failure under
``torch.export``). The arithmetic it performs is two lines, so this module performs it directly,
in the SAME order, so the result is BIT-IDENTICAL to the policy's own masked distribution on the
same device (pinned by ``decision_test.py``):

    l0   = raw - logsumexp(raw)              # Categorical(logits=raw) normalises once
    lm   = where(mask, l0, -1e8)             # MaskableCategorical.apply_masking (HUGE_NEG)
    logp = lm - logsumexp(lm)                # ... and re-initialises, normalising again

The value is the policy's own ``_critic_value`` (win-prob sigmoid under ``critic='winprob'``,
the scalar value net otherwise), so every critic mode the policy supports is served unchanged.

OUTPUT CONTRACT. ``logp`` is ``[B, A]`` float32 with every ILLEGAL entry set to ``-inf`` (the policy
itself carries ~-1e8 there; ``-inf`` makes an illegal action impossible to sample or argmax rather
than merely improbable, and makes "legal" readable off the output). ``value`` is ``[B]`` float32.
Greedy is ``logp.argmax(-1)`` (the caller's, or the service's ``greedy``).

The forward reads ``observation`` only. An architecture whose extractor reads a further Dict key
(``extra_obs_keys`` — empty today; the privileged ``opp_true_team`` value route was deleted) is REFUSED at slot
declaration (``DecisionModule.__init__``): serving it without the key would raise mid-run, and
serving it with a synthetic key would be a V stripped of its privilege — a different quantity.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator, Tuple

import torch
from torch import nn

#: sb3_contrib ``MaskableCategorical.apply_masking``'s HUGE_NEG — the masked logit before the
#: second normalisation. Kept literal so the arithmetic is the policy's, bit for bit.
HUGE_NEG = -1e8


class UnservableArchitecture(ValueError):
    """The policy's architecture cannot be served by the decision forward (named reason)."""


def _refuse_extra_obs_keys(policy: Any) -> None:
    from agents.model.extra_obs_keys import required_extra_obs_keys

    live = [k.key for k in required_extra_obs_keys(policy.features_extractor)]
    if live:
        raise UnservableArchitecture(
            f"the extractor reads Dict obs key(s) {live} beyond 'observation' (extra_obs_keys). "
            "The inference service serves observation-only forwards; a synthetic key would serve a "
            "V stripped of its privilege. Serve this checkpoint on its own path.")


class DecisionModule(nn.Module):
    """``forward(obs [B, D] f32, mask [B, A] bool) -> (logp [B, A] f32, value [B] f32)``.

    Wraps (does not copy) ``policy``: the extractor, the actor tower, the pointer head and the
    critic read are the policy's own modules, so a weight copied into the policy is the weight
    this module serves.
    """

    def __init__(self, policy: Any):
        super().__init__()
        for attr in ("features_extractor", "mlp_extractor", "pointer_head", "_critic_value"):
            if not hasattr(policy, attr):
                raise UnservableArchitecture(
                    f"policy {type(policy).__name__} has no {attr!r}: the decision forward serves "
                    "the Gen3 dual-head pointer policy only")
        _refuse_extra_obs_keys(policy)
        # The policy is held UNREGISTERED and its parts registered ONCE each. sb3 registers the
        # shared extractor under three names (features_extractor, pi_/vf_features_extractor), and
        # torch.export (2.5.1) then names a non-persistent buffer by one alias and checks it by
        # another (SpecViolationError "Buffer ...MOVE_ATTR is not in the state dict", measured
        # 2026-09-29). Every module the forward reaches is one of these, so a weight swap or an
        # export lifts each parameter exactly once.
        object.__setattr__(self, "_policy", policy)
        self.fe = policy.features_extractor
        self.mlp = policy.mlp_extractor
        self.ptr = policy.pointer_head

    @property
    def policy(self) -> Any:
        return self._policy

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fe = self.fe
        pi, vf = fe({"observation": obs})
        latent_pi = self.mlp.forward_actor(pi)
        # The policy's OWN critic read (the win-prob head's stashed logit), never a copy. `vf` is
        # `value_pooled`: no critic tower runs (audit F1, config v144).
        value = self._policy._critic_value(vf).reshape(-1).float()
        tok_req, valid, team_tokens, move_cells, switch_cells = fe.last_pointer_inputs
        raw = self.ptr(latent_pi, tok_req, valid, team_tokens, move_cells, switch_cells)
        return masked_logp(raw, mask), value


def masked_logp(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The policy's masked log-probabilities (see the module docstring), illegal entries ``-inf``."""
    l0 = raw - raw.logsumexp(dim=-1, keepdim=True)
    lm = torch.where(mask, l0, torch.full_like(l0, HUGE_NEG))
    logp = lm - lm.logsumexp(dim=-1, keepdim=True)
    return torch.where(mask, logp, torch.full_like(logp, float("-inf"))).float()


def _is_forward_state(v: Any) -> bool:
    """A module attribute a forward (re)writes: a tensor that is not a parameter or buffer (those live in
    ``_parameters`` / ``_buffers``, not ``vars()``), a tuple / list holding one (`EntitySeats.last_cand`),
    or a per-forward STASH dataclass (`ExtractorStashes`, `OpStashes`: `gen3_extractor_stashes_v1`)."""
    if torch.is_tensor(v) or type(v).__name__.endswith("Stashes"):
        return True
    return isinstance(v, (tuple, list)) and any(torch.is_tensor(x) for x in v)


@contextlib.contextmanager
def forward_state_released(policy: Any) -> Iterator[None]:
    """Run a forward whose per-forward Python state does not outlive it (`gen3_reference_state_released_v1`,
    F-XC-3). Every module attribute the forward REPLACED — its stash dataclass, a plain tensor attribute
    such as `PokemonEncoder.last_move_tokens`, a tuple of tensors such as `EntitySeats.last_cand` — is
    EMPTIED when it returns (a stash to a fresh instance, anything else to ``None``, an attribute the
    forward created is removed); an attribute the forward did not touch is left alone. So the tensors the
    forward produced are released, and the module holds no row-sized state afterwards.

    Without it a parity gate's eager reference left its stashes alive on a T2 slot's replica, sized by the
    gate's LAST row count: a slot load then read as a NEW allocation, growing with the arm's stash per row
    (fixed_mass ≈ 0.38 MB per row; blob far less). Emptying rather than restoring the previous objects also
    drops a stale stash an earlier forward (with grad) left there, which `copy.deepcopy` refuses."""
    saved = []
    for mod in policy.modules():
        d = vars(mod)
        saved.append((d, {k: v for k, v in d.items() if v is None or _is_forward_state(v)}))
    try:
        yield
    finally:
        for d, before in saved:
            for k in [k for k, v in d.items() if _is_forward_state(v) and d.get(k) is not before.get(k, d)]:
                v = d[k]
                if k not in before:
                    del d[k]
                elif type(v).__name__.endswith("Stashes"):
                    d[k] = type(v)()
                else:
                    d[k] = None


def policy_reference(policy: Any, obs: torch.Tensor, mask: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The EAGER reference, read through the policy's OWN sb3 path (``extract_features`` → towers →
    ``_get_action_dist_from_latent`` → ``apply_masking``), exactly what the rollout samples from.
    Same output contract as ``DecisionModule``. The parity gate's ground truth. It leaves the policy's
    per-forward state EMPTY (`forward_state_released`): a gate run on a served slot's replica acquires
    nothing that outlives the gate."""
    fe = policy.features_extractor
    with torch.no_grad(), forward_state_released(policy):
        pi, vf = fe({"observation": obs})
        latent_pi = policy.mlp_extractor.forward_actor(pi)
        value = policy._critic_value(vf).reshape(-1).float()
        dist = policy._get_action_dist_from_latent(latent_pi)
        dist.apply_masking(mask)
        logp = dist.distribution.logits
        logp = torch.where(mask, logp, torch.full_like(logp, float("-inf"))).float()
    return logp, value
