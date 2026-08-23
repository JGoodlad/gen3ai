"""`ExtractorStashes` — every per-forward SIDE VALUE the extractor exposes, as ONE typed unit.

Split out of `features_extractor.py` 2026-08-23 (one responsibility per file); that module
re-exports it, so `from agents.model.features_extractor import ExtractorStashes` still resolves.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from agents.model.extractor_ctx import PointerInputs
from agents.model.intent_threshold import ThresholdProbs

@dataclass
class ExtractorStashes:
    """Every per-forward SIDE VALUE the extractor exposes, as ONE typed unit
    (`gen3_extractor_stashes_v1` — the OpStashes recipe applied to the orchestrator). The v89
    lesson made the cost of the old pattern concrete: phases communicated through mutable
    `self.last_*` instance stashes, consumers — including other MODULES — read them with
    `getattr`, and nothing type-level connected producer to consumer, so a consumer rewiring
    silently orphaned five value routes for two generations. `forward_internal` replaces the
    whole container at ENTRY, so a stale cross-batch read is unrepresentable for ANY stash.
    Reads stay on the `fe.last_*` properties (the documented consumer surface — the policy's
    pointer head + dist critic, instrumented_ppo's aux losses, the prober and inference all
    keep their spelling); WRITES go through `fe.stash.<field>` — a stray write to a `last_*`
    name now fails loud (AttributeError) instead of silently forking the state.

    Every stash is leak-safe by construction: they are OUTPUTS of the forward, read by aux
    losses / the prober / the policy heads, never fed back as inputs that could carry a label."""
    # --- pointer / action head (read by Gen3DualHeadMaskablePolicy) -------------------------
    pointer_inputs: Optional[PointerInputs] = None   # request-ordered move tokens + valid mask
    #                                                  + our team tokens + the op's per-action cells
    # --- intent heads (T2 publications — stop-grad under belief_grad_mode=label_only) --------
    alpha_logits: Optional[torch.Tensor] = None      # [B,K+1] which believed move (or SWITCH)
    alpha_seat_nums: Optional[torch.Tensor] = None   # [B,K] seat move NUMS (detached; loss labels)
    beta_logits: Optional[torch.Tensor] = None       # [B,6] if they switch, to whom
    thresh_probs: Optional[ThresholdProbs] = None    # T2-computed, read by the vf route at T3
    # --- belief bank (publications; the LIVE views live in `belief_supervision`) -------------
    belief_logits: Optional[Dict[str, torch.Tensor]] = None  # species/moves aux dict (refined opp)
    opp_believed_mask: Optional[torch.Tensor] = None  # [B,6] bool: un-revealed opp slots
    opp_active_local: Optional[torch.Tensor] = None   # [B] opp active idx (prober belief-row decode)
    move_belief_logits: Optional[torch.Tensor] = None  # [B,6,n_moves] TYPED posterior (11 readers)
    move_latent_table: Optional[torch.Tensor] = None   # [n_moves,MOVE_LATENT_DIM] grading target
    spread_belief: Optional[torch.Tensor] = None       # [B,6,5] believed derived stats
    spread_nature_logits: Optional[torch.Tensor] = None  # [B,6,25] (gen3_nature_ev_belief_v1)
    spread_ev: Optional[torch.Tensor] = None           # [B,6,5] believed EVs
    item_logits: Optional[torch.Tensor] = None         # [B,6,n_items] hidden-item posterior
    hp_type_logits: Optional[torch.Tensor] = None      # [B,6,16] HP-type head (aux CE + prober)
    # --- op / physics -------------------------------------------------------------------------
    damage_block: Optional[torch.Tensor] = None        # [B,out_dim] prober decode; never read fwd
    # --- value-side readouts ------------------------------------------------------------------
    value_pooled: Optional[torch.Tensor] = None        # [B,D_MODEL] the FitNets HINT layer
    win_prob_logits: Optional[torch.Tensor] = None     # [B,1] P(win) logit (aux BCE + prober)
    value_dist_logits: Optional[torch.Tensor] = None   # [B,bins] dist-critic atoms (E[Z] source)
    # --- same-forward hand-offs (T0 producer → T1/T2 consumer; internal, no `last_*` name) ----
    t0_species_probs: Optional[torch.Tensor] = None    # T0 species resolve → every T1 pricing site
    entity_latent_table: Optional[torch.Tensor] = None  # LIVE latent table → the E4 seat builder
    # The LIVE (graph-carrying) belief outputs for the SUPERVISED aux losses only — see
    # `belief_supervision()`. Under shaping/detached these are the identical objects the `last_*`
    # stashes hold; under label_only the `last_*` stashes are their stop-grad publications.
    # Container replacement at forward entry is what makes "key absent ⇒ the head did not run
    # this forward" true (these hold graph-carrying tensors, so a stale one would backprop
    # through a freed — or worse, a different minibatch's — graph).
    belief_supervision: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
