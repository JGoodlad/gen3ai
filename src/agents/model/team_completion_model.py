"""
Team Completion Model: predicts masked Pokémon slots given revealed teammates.

Architecture:
  1. Frozen embedding tables loaded from a PPO backbone checkpoint
     (species, move, item, type — same IDs as the observation encoder)
  2. Per-slot encoder: concat(species_emb, move_pool, item_emb) → Linear → 128-dim token
     - move_pool: weighted mean of move embeddings using the multi-hot input as weights
  3. Masked slots replaced with a learned mask_token embedding
  4. 2-layer TransformerEncoder with no positional encodings (team order is irrelevant)
  5. Output heads applied only at masked positions:
     - species_head: CrossEntropy over num_species
     - item_head:    CrossEntropy over num_items (only where item_known)
     - move_head:    BCEWithLogitsLoss over num_moves (multi-label; always supervised)
"""

import io
import zipfile
from typing import Optional

import torch
import torch.nn as nn


ROLE_TOKEN_DIM = 128        # per-slot transformer token dim; independent of backbone
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_FF_DIM = 256
ARCH_SIGNATURE = "team_completion_v1"


def model_config(
    num_species: int,
    num_moves: int,
    num_items: int,
    backbone_path: Optional[str],
) -> dict:
    """Return the architecture config dict — written as model_config.json alongside checkpoints."""
    return {
        "arch_signature":    ARCH_SIGNATURE,
        "num_species":       num_species,
        "num_moves":         num_moves,
        "num_items":         num_items,
        "role_token_dim":    ROLE_TOKEN_DIM,
        "transformer_layers": TRANSFORMER_LAYERS,
        "transformer_heads":  TRANSFORMER_HEADS,
        "transformer_ff_dim": TRANSFORMER_FF_DIM,
        "backbone":          backbone_path,
    }


def _load_embedding_from_backbone(checkpoint_path: str, device: str = "cpu") -> dict[str, nn.Embedding]:
    """
    Extract the four shared embedding tables from an SB3 MaskablePPO checkpoint (.zip).
    The checkpoint zip contains a 'data' entry with the full PyTorch state dict.
    Returns a dict with keys: species, move, item, type.
    """
    # SB3 checkpoint layout: policy weights are in 'policy.pth'
    with zipfile.ZipFile(checkpoint_path) as zf:
        with zf.open("policy.pth") as f:
            policy_sd = torch.load(io.BytesIO(f.read()), map_location=device, weights_only=False)

    def _extract(key: str) -> nn.Embedding:
        weight_key = f"features_extractor.{key}_embedding.weight"
        if weight_key not in policy_sd:
            raise KeyError(f"Key '{weight_key}' not found in checkpoint. Available keys (sample): "
                           f"{list(policy_sd.keys())[:5]}")
        w = policy_sd[weight_key]
        emb = nn.Embedding(w.shape[0], w.shape[1])
        emb.weight = nn.Parameter(w.clone())
        return emb

    return {
        "species": _extract("species"),
        "move":    _extract("move"),
        "item":    _extract("item"),
        "type":    _extract("type"),
    }


class TeamCompletionModel(nn.Module):
    """
    Predict the species, moveset, and item of masked Pokémon slots given the
    revealed slots on the same team.

    Args:
        num_species:  vocabulary size for species classifier head
        num_moves:    vocabulary size for move multi-label head
        num_items:    vocabulary size for item classifier head
        backbone_path: path to MaskablePPO .zip checkpoint; embeddings are
                       loaded and frozen. If None, embeddings are randomly
                       initialized (for ablation or tests).
        device:       'cpu' or 'cuda'
    """

    def __init__(
        self,
        num_species: int,
        num_moves: int,
        num_items: int,
        backbone_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_species = num_species
        self.num_moves = num_moves
        self.num_items = num_items

        # ── Embedding tables (frozen if backbone provided) ──────────────────
        if backbone_path is not None:
            embs = _load_embedding_from_backbone(backbone_path, device=device)
            self.species_emb = embs["species"]
            self.move_emb    = embs["move"]
            self.item_emb    = embs["item"]
        else:
            self.species_emb = nn.Embedding(num_species, 32)
            self.move_emb    = nn.Embedding(num_moves,   16)
            self.item_emb    = nn.Embedding(num_items,   16)

        species_dim = self.species_emb.embedding_dim   # 32
        move_dim    = self.move_emb.embedding_dim       # 16
        item_dim    = self.item_emb.embedding_dim       # 16

        if backbone_path is not None:
            for emb in (self.species_emb, self.move_emb, self.item_emb):
                emb.weight.requires_grad_(False)

        # ── Per-slot encoder ─────────────────────────────────────────────────
        slot_input_dim = species_dim + move_dim + item_dim  # 32+16+16 = 64
        self.slot_encoder = nn.Sequential(
            nn.Linear(slot_input_dim, ROLE_TOKEN_DIM),
            nn.ReLU(),
        )

        # ── Learned [MASK] token ─────────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.randn(ROLE_TOKEN_DIM) * 0.02)

        # ── Transformer (no positional encodings — team order is irrelevant) ─
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ROLE_TOKEN_DIM,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Output heads (applied at masked positions only) ───────────────────
        self.species_head = nn.Linear(ROLE_TOKEN_DIM, num_species)
        self.item_head    = nn.Linear(ROLE_TOKEN_DIM, num_items)
        self.move_head    = nn.Linear(ROLE_TOKEN_DIM, num_moves)  # BCEWithLogitsLoss

    def encode_slots(
        self,
        species_ids:   torch.Tensor,   # [B, 6] long
        move_multihot: torch.Tensor,   # [B, 6, num_moves] float
        item_ids:      torch.Tensor,   # [B, 6] long
        is_masked:     torch.Tensor,   # [B, 6] bool
    ) -> torch.Tensor:
        """Encode all 6 slots → [B, 6, ROLE_TOKEN_DIM], substituting mask_token for masked slots."""
        B = species_ids.shape[0]

        sp_emb   = self.species_emb(species_ids)                        # [B, 6, 32]
        item_emb = self.item_emb(item_ids)                              # [B, 6, 16]

        # Move pool: weighted mean of move embeddings using multi-hot weights.
        # move_multihot has M entries where M = num_moves (max move ID + 1).
        # The backbone embedding may have more rows than M, so slice to M.
        move_weights = move_multihot                                              # [B, 6, M]
        M = move_weights.shape[-1]
        move_pool = torch.matmul(move_weights, self.move_emb.weight[:M])         # [B, 6, 16]
        denom = move_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        move_pool = move_pool / denom                                   # mean over known moves

        slot_input = torch.cat([sp_emb, move_pool, item_emb], dim=-1)  # [B, 6, 64]
        tokens = self.slot_encoder(slot_input)                          # [B, 6, 128]

        # Replace masked positions with the learned mask_token
        mask = is_masked.unsqueeze(-1).expand_as(tokens)               # [B, 6, 128]
        mask_broadcast = self.mask_token.view(1, 1, -1).expand(B, 6, -1)
        tokens = torch.where(mask, mask_broadcast, tokens)

        return tokens

    def forward(
        self,
        species_ids:   torch.Tensor,   # [B, 6] long
        move_multihot: torch.Tensor,   # [B, 6, num_moves] float
        item_ids:      torch.Tensor,   # [B, 6] long
        is_masked:     torch.Tensor,   # [B, 6] bool
        is_padding:    torch.Tensor,   # [B, 6] bool  — padding slots are ignored
    ) -> dict[str, torch.Tensor]:
        """
        Returns logits for each head at ALL positions; caller applies loss only
        at masked (and non-padding) positions.

        Output keys:
          species_logits  [B, 6, num_species]
          item_logits     [B, 6, num_items]
          move_logits     [B, 6, num_moves]
        """
        tokens = self.encode_slots(species_ids, move_multihot, item_ids, is_masked)

        # TransformerEncoder src_key_padding_mask: True = ignore this position
        tokens = self.transformer(tokens, src_key_padding_mask=is_padding)

        return {
            "species_logits": self.species_head(tokens),   # [B, 6, S]
            "item_logits":    self.item_head(tokens),      # [B, 6, I]
            "move_logits":    self.move_head(tokens),      # [B, 6, M]
        }

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        batch:  dict[str, torch.Tensor],
        move_pos_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the combined loss, supervising only masked non-padding slots.

        move_pos_weight: [num_moves] tensor from Mappings.move_pos_weight, passed through
            to BCEWithLogitsLoss to compensate for the severe neg:pos imbalance in move
            labels (~353 zeros vs ~2 ones per Pokémon slot in replay data).

        Returns (total_loss, {species_loss, item_loss, move_loss}).
        """
        is_masked  = batch["is_masked"]    # [B, 6]
        is_padding = batch["is_padding"]   # [B, 6]
        active = is_masked & ~is_padding   # [B, 6] — slots we actually predict

        if not active.any():
            zero = logits["species_logits"].sum() * 0
            return zero, {"species_loss": zero, "item_loss": zero, "move_loss": zero}

        # ── Species loss (always supervised for masked slots) ─────────────────
        sp_logits  = logits["species_logits"][active]      # [N, S]
        sp_targets = batch["target_species"][active]       # [N]
        species_loss = nn.functional.cross_entropy(sp_logits, sp_targets)

        # ── Move loss (multi-label BCE, always for masked slots) ──────────────
        mv_logits  = logits["move_logits"][active]         # [N, M]
        mv_targets = batch["target_move_multihot"][active] # [N, M]
        pw = move_pos_weight.to(mv_logits.device) if move_pos_weight is not None else None
        move_loss = nn.functional.binary_cross_entropy_with_logits(
            mv_logits, mv_targets, pos_weight=pw
        )

        # ── Item loss (only where item was revealed) ─────────────────────────
        item_active = active & batch["item_known"]         # [B, 6]
        if item_active.any():
            it_logits  = logits["item_logits"][item_active]   # [K, I]
            it_targets = batch["target_item"][item_active]    # [K]
            item_loss = nn.functional.cross_entropy(it_logits, it_targets)
        else:
            item_loss = logits["item_logits"].sum() * 0

        total = species_loss + move_loss + item_loss
        return total, {
            "species_loss": species_loss.detach(),
            "item_loss":    item_loss.detach(),
            "move_loss":    move_loss.detach(),
        }

    @torch.no_grad()
    def predict_top_k(
        self,
        species_ids:   torch.Tensor,
        move_multihot: torch.Tensor,
        item_ids:      torch.Tensor,
        is_masked:     torch.Tensor,
        is_padding:    torch.Tensor,
        k: int = 5,
    ) -> dict[str, torch.Tensor]:
        """Return top-k predictions for each masked slot."""
        training = self.training
        self.eval()
        try:
            logits = self.forward(species_ids, move_multihot, item_ids, is_masked, is_padding)
            return {
                "top_species": logits["species_logits"].softmax(-1).topk(k, dim=-1).indices,
                "top_items":   logits["item_logits"].softmax(-1).topk(k, dim=-1).indices,
                "top_moves":   logits["move_logits"].sigmoid().topk(k, dim=-1).indices,
            }
        finally:
            self.train(training)
