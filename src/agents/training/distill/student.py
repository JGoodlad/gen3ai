"""The distilled opponent network — self-contained (own ObsUnpack + Embeddings, no teacher
reference at inference) and a drop-in adapter so ``RLPlayer`` uses it unchanged.

Architecture (ported from the validated `cheaper_encoder` recipe):
  obs -> ObsUnpack -> CheapEncoder (per-slot MLP -> 12x128 role tokens)
      -> 1-layer TransformerEncoderLayer -> matchup-aware pool -> MLP head -> 11 raw logits.
The encoder is distilled (MSE) onto the teacher's frozen role tokens; the head by soft-KL.
Capacity is parameterised (``enc_hidden`` / ``head_hidden`` / ``tf_*``) so the manager can
escalate the student size when a snapshot fails the gate (`distill_integration.md` §8).
"""
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn

from agents.model.features_extractor import Embeddings, ObsUnpack, ROLE_TOKEN_SIZE

TEAM = 6  # our / opp team size


class CheapEncoder(nn.Module):
    """Shared per-slot MLP: raw pokemon_part + per-slot embeddings + a per-slot matchup summary
    -> a ``ROLE_TOKEN_SIZE`` role token, applied to all 12 slots. Much cheaper than the teacher's
    move-network + within-mon attention + role encoder, same 12x128 output shape."""

    def __init__(self, emb: Embeddings, hidden: int = 192):
        super().__init__()
        sd = emb.species_embedding.embedding_dim
        idim = emb.item_embedding.embedding_dim
        ad = emb.ability_embedding.embedding_dim
        td = emb.type_embedding.embedding_dim
        md = emb.move_embedding.embedding_dim
        self.match_dim = 8  # per-move (4) mean + per-move (4) max over opp slots
        slot_in = 107 + sd + idim + 2 * ad + 2 * td + 4 * md + 4 * md + self.match_dim
        self.net = nn.Sequential(
            nn.LayerNorm(slot_in),
            nn.Linear(slot_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ROLE_TOKEN_SIZE),
        )

    def _slot_input(self, ctx, emb: Embeddings) -> torch.Tensor:
        B = ctx.batch_size
        sp = emb.species_embedding(ctx.species_ids)
        it = emb.item_embedding(ctx.item_ids)
        a1 = emb.ability_embedding(ctx.ability1_ids)
        a2 = emb.ability_embedding(ctx.ability2_ids)
        t1 = emb.type_embedding(ctx.type1_ids)
        t2 = emb.type_embedding(ctx.type2_ids)
        mv = emb.move_embedding(ctx.all_move_ids).reshape(B, 12, -1)
        mt = emb.type_embedding(ctx.all_move_type_ids).reshape(B, 12, -1)
        m = ctx.matchups_all                                   # B,12,4,6
        match = torch.cat([m.mean(-1), m.amax(-1)], dim=-1)    # B,12,8
        return torch.cat([ctx.pokemon_part, sp, it, a1, a2, t1, t2, mv, mt, match], dim=2)

    def forward(self, ctx, emb: Embeddings) -> torch.Tensor:
        return self.net(self._slot_input(ctx, emb))            # B,12,128


class DistilledStudent(nn.Module):
    """A self-contained cheap opponent policy: forward({"observation": [B,3357]}) -> raw [B,11]
    logits (the caller applies the action mask, exactly as for the full model). Owns its own
    ObsUnpack (stateless) and Embeddings (weights copied from the teacher at distill time, then
    frozen) so it needs no teacher at inference."""

    def __init__(self, layout: Dict[str, Any], enc_hidden: int = 192, head_hidden: int = 384,
                 tf_heads: int = 8, tf_ffn: int = 512):
        super().__init__()
        self.config = dict(enc_hidden=enc_hidden, head_hidden=head_hidden,
                           tf_heads=tf_heads, tf_ffn=tf_ffn)
        self.unpack = ObsUnpack(layout)            # stateless
        self.emb = Embeddings(layout)              # own copy (frozen; filled from teacher)
        self.encoder = CheapEncoder(self.emb, hidden=enc_hidden)
        self.tf = nn.TransformerEncoderLayer(ROLE_TOKEN_SIZE, nhead=tf_heads, dim_feedforward=tf_ffn,
                                             dropout=0.0, batch_first=True, norm_first=False)
        self._obs_dim = layout["total_dim"]
        with torch.no_grad():
            d = self._pool({"observation": torch.zeros(2, self._obs_dim)}).shape[1]
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 11),
        )

    def freeze_embeddings(self) -> None:
        for p in self.emb.parameters():
            p.requires_grad_(False)

    def role_tokens(self, ctx) -> torch.Tensor:
        return self.encoder(ctx, self.emb)

    def _pool(self, od: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx = self.unpack(od)
        rt = self.tf(self.role_tokens(ctx))                    # B,12,128
        B = ctx.batch_size
        bidx = torch.arange(B)
        our, opp = rt[:, :TEAM], rt[:, TEAM:]
        oa = ctx.our_active_idx.long().clamp(0, TEAM - 1)
        pa = ctx.opp_active_local.long().clamp(0, TEAM - 1)
        our_act = our[bidx, oa]
        opp_act = opp[bidx, pa]
        m = ctx.matchups_all
        our_m, their_m = m[:, :TEAM], m[:, TEAM:]
        bI = torch.arange(B).view(B, 1, 1); sI = torch.arange(TEAM).view(1, TEAM, 1)
        mI = torch.arange(4).view(1, 1, 4)
        ours_vs_oppact = our_m[bI, sI, mI, pa.view(B, 1, 1)].reshape(B, TEAM * 4)
        theirs_vs_ouract = their_m[bI, sI, mI, oa.view(B, 1, 1)].reshape(B, TEAM * 4)
        return torch.cat([our_act, opp_act, our.mean(1), opp.mean(1),
                          ours_vs_oppact, theirs_vs_ouract,
                          ctx.our_ctx_raw, ctx.opp_ctx_raw, ctx.non_matchup_rest], dim=1)

    def forward(self, od: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self._pool(od))                       # raw [B,11] logits


# ── drop-in adapter: looks like ``model`` to RLPlayer (model.policy.get_distribution / .device) ──
class _Dist:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _DistWrap:
    def __init__(self, logits: torch.Tensor):
        self.distribution = _Dist(logits)


class _StudentPolicy:
    def __init__(self, student: DistilledStudent):
        self._student = student

    def get_distribution(self, policy_in: Dict[str, torch.Tensor]) -> _DistWrap:
        return _DistWrap(self._student(policy_in))

    def predict_values(self, policy_in: Dict[str, torch.Tensor]) -> torch.Tensor:
        # The opponent never reads value; return a benign zero so any caller is safe.
        return torch.zeros(policy_in["observation"].shape[0], 1, device=self._student_device())

    def _student_device(self) -> torch.device:
        return next(self._student.parameters()).device

    def set_training_mode(self, mode: bool) -> None:
        self._student.train(mode)


class DistilledOpponentModel:
    """Wraps a ``DistilledStudent`` so it is interface-compatible with the ``MaskablePPO`` that
    ``RLPlayer`` expects: ``model.policy.get_distribution({obs, action_mask}).distribution.logits``
    + ``model.device``. RLPlayer applies the action mask itself, so the student returns raw logits."""

    def __init__(self, student: DistilledStudent, device: str = "cpu"):
        student.eval()
        self.policy = _StudentPolicy(student)
        self.device = torch.device(device)


# ── persistence (config + weights; ObsUnpack is stateless, rebuilt from layout on load) ──
def save_distilled(student: DistilledStudent, path: str) -> None:
    torch.save({"config": student.config, "state_dict": student.state_dict()}, path)


def load_distilled(path: str, layout: Dict[str, Any], device: str = "cpu") -> DistilledStudent:
    blob = torch.load(path, map_location=device, weights_only=False)  # our own file (config + tensors)
    student = DistilledStudent(layout, **blob["config"])
    student.load_state_dict(blob["state_dict"])
    student.to(device).eval()
    return student
