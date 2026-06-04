"""The two-stage distillation recipe (validated: ~4.67x cheaper, h2h ~0.44 vs the teacher).

``distill_snapshot(layout, teacher, obs, mask, config)`` ->  (DistilledStudent, metrics):
  STAGE 1  distill the student's CheapEncoder onto the teacher's FROZEN 12x128 role tokens (MSE).
  STAGE 2  freeze the encoder, distill the matchup head on the teacher's masked logits (soft-KL
           T=0.7 + label smoothing 0.02).
The student carries its OWN embeddings (copied from the teacher, then frozen) so it is self-sufficient
at inference. Capacity (``config``) is the escalation rung chosen by the manager.

Offline-GPU-heavy is fine (we're CPU-bound): pass ``device='cuda'`` to train; the deployed inference
cost is measured separately at 1-thread CPU.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from agents.model.features_extractor import Gen3FeaturesExtractor, ROLE_TOKEN_SIZE
from agents.training.distill.student import DistilledStudent

BIG = 1e9


def _teacher_fe(teacher) -> Gen3FeaturesExtractor:
    for m in teacher.policy.modules():
        if isinstance(m, Gen3FeaturesExtractor):
            return m
    raise ValueError("teacher has no Gen3FeaturesExtractor")


def _masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits + (mask - 1.0) * BIG


def fidelity(student: DistilledStudent, obs: np.ndarray, lg: np.ndarray, mk: np.ndarray) -> Dict[str, float]:
    student.eval()
    with torch.no_grad():
        s = student({"observation": torch.as_tensor(obs)}).cpu().numpy()
    tm, sm = _masked(torch.as_tensor(lg), torch.as_tensor(mk)), _masked(torch.as_tensor(s), torch.as_tensor(mk))
    tp = torch.softmax(tm, 1); slp = torch.log_softmax(sm, 1); sp = slp.exp()
    top1 = float((tm.argmax(1) == sm.argmax(1)).float().mean())
    kl = float((tp * (tp.clamp_min(1e-9).log() - slp)).sum(1).mean())
    H = lambda p: float(-(p * p.clamp_min(1e-9).log()).sum(1).mean())
    return dict(top1=top1, kl=kl, ent_ratio=H(sp) / max(H(tp), 1e-6))


@torch.no_grad()
def _cache_teacher_targets(teacher, fe, obs: np.ndarray, mask: np.ndarray, bs: int, device: str):
    """Teacher frozen role tokens [N,12,128] and RAW logits [N,11] for every obs row."""
    n = len(obs)
    tok = np.empty((n, 12, ROLE_TOKEN_SIZE), dtype=np.float32)
    lg = np.empty((n, 11), dtype=np.float32)
    Xo = torch.as_tensor(obs)
    for i in range(0, n, bs):
        ob = Xo[i:i + bs].to(device)
        mk = torch.as_tensor(mask[i:i + bs]).to(device)
        ctx = fe.unpack({"observation": ob})
        tok[i:i + bs] = fe.pokemon_encoder(ctx, fe.embeddings).cpu().numpy()
        lg[i:i + bs] = teacher.policy.get_distribution(
            {"observation": ob, "action_mask": mk}).distribution.logits.cpu().numpy()
    return tok, lg


def _train_stage1(student, obs, teacher_tok, tr, va, epochs, lr, bs, device):
    Xo, Tt = torch.as_tensor(obs), torch.as_tensor(teacher_tok)
    opt = torch.optim.Adam(student.encoder.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state = 1e9, None
    tok_var = float(Tt[torch.as_tensor(tr)].var())
    for ep in range(epochs):
        student.train()
        perm = tr[np.random.permutation(len(tr))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            ctx = student.unpack({"observation": Xo[b].to(device)})
            loss = ((student.role_tokens(ctx) - Tt[b].to(device)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        with torch.no_grad():
            vb = va[:4000]
            ctx = student.unpack({"observation": Xo[vb].to(device)})
            vmse = float(((student.role_tokens(ctx) - Tt[vb].to(device)) ** 2).mean())
        if vmse < best:
            best, best_state = vmse, copy.deepcopy(student.encoder.state_dict())
    student.encoder.load_state_dict(best_state)
    return best, best / max(tok_var, 1e-9)


def _train_stage2(student, obs, lg, mk, tr, va, epochs, lr, bs, T, eps, device):
    for p in student.encoder.parameters():
        p.requires_grad_(False)                       # freeze the cheap encoder
    Xo, Tl, Mk = torch.as_tensor(obs), torch.as_tensor(lg), torch.as_tensor(mk)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(epochs):
        student.train()
        perm = tr[np.random.permutation(len(tr))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            ob = Xo[b].to(device); mkb = Mk[b].to(device); tl = Tl[b].to(device)
            sl = torch.log_softmax(_masked(student({"observation": ob}), mkb) / T, 1)
            tp = torch.softmax(_masked(tl, mkb) / T, 1)
            n_legal = mkb.sum(1, keepdim=True).clamp_min(1.0)
            tp = (1 - eps) * tp + eps * (mkb / n_legal)
            loss = (tp * (tp.clamp_min(1e-9).log() - sl)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            f = fidelity(student, obs[va], lg[va], mk[va])
            if f["top1"] > best:
                best, best_state = f["top1"], copy.deepcopy(student.state_dict())
    student.load_state_dict(best_state)
    return best


def distill_snapshot(
    layout: Dict[str, Any],
    teacher,
    obs: np.ndarray,
    mask: np.ndarray,
    config: Dict[str, int],
    *,
    s1_epochs: int = 60,
    s2_epochs: int = 200,
    bs: int = 512,
    T: float = 0.7,
    eps: float = 0.02,
    device: str = "cpu",
    split: float = 0.85,
    seed: int = 0,
    log: Optional[Any] = None,
) -> Tuple[DistilledStudent, Dict[str, float]]:
    """Distill ``teacher`` into a DistilledStudent at the given capacity ``config``, on a dataset of
    ``obs``/``mask`` rows (the teacher's targets are computed internally). Returns (student, metrics);
    metrics = {top1, kl, ent_ratio, tok_nrmse, n_states}. The h2h gate is run separately (worker)."""
    torch.manual_seed(seed); np.random.seed(seed)
    fe = _teacher_fe(teacher)
    student = DistilledStudent(layout, **config).to(device)
    student.emb.load_state_dict(fe.embeddings.state_dict())   # own embeddings = teacher's, then frozen
    student.freeze_embeddings()

    rng = np.random.RandomState(seed); idx = rng.permutation(len(obs)); cut = int(split * len(obs))
    tr, va = idx[:cut], idx[cut:]

    tok, lg = _cache_teacher_targets(teacher, fe, obs, mask, bs, device)
    _, tok_nrmse = _train_stage1(student, obs, tok, tr, va, s1_epochs, 1e-3, bs, device)
    _train_stage2(student, obs, lg, mask, tr, va, s2_epochs, 1e-3, bs, T, eps, device)

    student.to("cpu").eval()
    m = fidelity(student, obs[va], lg[va], mask[va])
    m["tok_nrmse"] = tok_nrmse
    m["n_states"] = int(len(obs))
    if log:
        log(f"distill step done: top1={m['top1']:.3f} kl={m['kl']:.3f} ent={m['ent_ratio']:.2f} "
            f"tok_nrmse={tok_nrmse:.3f} (n={len(obs)}, cfg={config})")
    return student, m
