"""
PyTorch Dataset for team completion training.

Each TeamRecord has 1–6 revealed Pokémon. A training example is formed by
randomly masking k of those slots and asking the model to predict them from
the remaining (k-1)+ context slots.

The dataset uses the same ID mappings as the observation encoder so embeddings
are shared between the team completion model and the PPO backbone.
"""

import json
import random
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass
class Mappings:
    """Resolved integer-ID lookups for species, moves, and items."""
    species_to_id: dict[str, int]   # normalized key -> integer ID
    move_to_id: dict[str, int]
    item_to_id: dict[str, int]
    num_species: int
    num_moves: int
    num_items: int
    move_pos_weight: Optional[torch.Tensor] = field(default=None, repr=False)
    # Per-move BCE pos_weight; computed from training data in build_datasets.

    @classmethod
    def from_data_dir(cls, repo_root: str) -> "Mappings":
        sys.path.insert(0, os.path.join(repo_root, "src"))
        from agents.observation.state_encoder import load_mappings
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            raw = load_mappings()
        finally:
            os.chdir(old_cwd)
        species_to_id = {k: int(v["num"]) for k, v in raw["species"].items()}
        move_to_id    = {k: int(v["num"]) for k, v in raw["moves"].items()}
        item_to_id    = {k: int(v["num"]) for k, v in raw["items"].items()}
        return cls(
            species_to_id=species_to_id,
            move_to_id=move_to_id,
            item_to_id=item_to_id,
            num_species=max(species_to_id.values()) + 1,
            num_moves=max(move_to_id.values()) + 1,
            num_items=max(item_to_id.values()) + 1,
        )


def _compute_move_pos_weight(records: list[list[dict]], mappings: Mappings) -> torch.Tensor:
    """
    Compute per-move positive frequency for BCEWithLogitsLoss pos_weight.
    pos_weight[i] = (n_slots - count_i) / count_i  — balances the ~353:2 neg:pos ratio.
    """
    counts = torch.zeros(mappings.num_moves)
    n_slots = 0
    for mons in records:
        for mon in mons:
            n_slots += 1
            for move in mon.get("moves", []):
                mid = mappings.move_to_id.get(move)
                if mid is not None:
                    counts[mid] += 1
    pos_weight = (n_slots - counts) / counts.clamp(min=1.0)
    return pos_weight.clamp(max=300.0)  # cap to prevent extreme values for very rare moves


TEAM_SIZE = 6   # pad all examples to this length


class TeamCompletionDataset(Dataset):
    """
    Each item is one (context, masked) pair derived from a TeamRecord.

    Tensors returned by __getitem__:
      species_ids     [6]        integer species ID; 0 for masked slots
      move_multihot   [6, M]     multi-hot over all move IDs; all-zeros for masked/unknown
      item_ids        [6]        integer item ID; 0 for masked/unknown
      is_masked       [6] bool   which slots the model must predict
      is_padding      [6] bool   which slots are padding (< 6 revealed Pokémon)
      target_species  [6]        ground-truth species ID (only used where is_masked)
      target_move_multihot [6, M] ground-truth multi-hot moves
      target_item     [6]        ground-truth item ID (0 = unknown)
      item_known      [6] bool   True if item was revealed in the replay
    """

    def __init__(
        self,
        jsonl_path: Optional[str],
        mappings: Mappings,
        min_revealed: int = 2,
        max_mask_fraction: float = 0.7,
        seed: Optional[int] = None,
        *,
        _records: Optional[list[list[dict]]] = None,
    ):
        self.mappings = mappings
        self.min_revealed = min_revealed
        self.max_mask_fraction = max_mask_fraction
        self._rng = random.Random(seed)

        if _records is not None:
            self._records = _records
        else:
            self._records: list[list[dict]] = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    mons = record.get("pokemon", [])
                    if len(mons) >= min_revealed:
                        self._records.append(mons)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mons = self._records[idx]
        n = len(mons)

        # Sample how many slots to mask: at least 1, at most floor(n * max_fraction)
        max_mask = max(1, int(n * self.max_mask_fraction))
        n_mask = self._rng.randint(1, max_mask)
        masked_indices = set(self._rng.sample(range(n), n_mask))

        M = self.mappings.num_moves

        # Build per-slot tensors (length = n, then padded to TEAM_SIZE)
        species_ids    = torch.zeros(TEAM_SIZE, dtype=torch.long)
        move_multihot  = torch.zeros(TEAM_SIZE, M, dtype=torch.float32)
        item_ids       = torch.zeros(TEAM_SIZE, dtype=torch.long)
        is_masked      = torch.zeros(TEAM_SIZE, dtype=torch.bool)
        is_padding     = torch.ones(TEAM_SIZE, dtype=torch.bool)
        t_species      = torch.zeros(TEAM_SIZE, dtype=torch.long)
        t_moves        = torch.zeros(TEAM_SIZE, M, dtype=torch.float32)
        t_item         = torch.zeros(TEAM_SIZE, dtype=torch.long)
        item_known     = torch.zeros(TEAM_SIZE, dtype=torch.bool)

        for slot, mon in enumerate(mons):
            sp_id = self.mappings.species_to_id.get(mon["species"])
            if sp_id is None:
                # Species not in vocabulary — treat slot as padding to avoid mislabelling
                continue
            is_padding[slot] = False
            item_k = mon.get("item")
            it_id  = self.mappings.item_to_id.get(item_k, 0) if item_k else 0
            m_hot  = self._encode_moves(mon.get("moves", []))

            # Ground truth is always filled
            t_species[slot]  = sp_id
            t_moves[slot]    = m_hot
            t_item[slot]     = it_id
            item_known[slot] = item_k is not None

            if slot in masked_indices:
                is_masked[slot] = True
                # Input keeps zeros (masked)
            else:
                # Revealed: fill input
                species_ids[slot]   = sp_id
                move_multihot[slot] = m_hot
                item_ids[slot]      = it_id

        return {
            "species_ids":          species_ids,
            "move_multihot":        move_multihot,
            "item_ids":             item_ids,
            "is_masked":            is_masked,
            "is_padding":           is_padding,
            "target_species":       t_species,
            "target_move_multihot": t_moves,
            "target_item":          t_item,
            "item_known":           item_known,
        }

    def _encode_moves(self, moves: list[str]) -> torch.Tensor:
        vec = torch.zeros(self.mappings.num_moves, dtype=torch.float32)
        for m in moves:
            mid = self.mappings.move_to_id.get(m)
            if mid is not None:
                vec[mid] = 1.0
        return vec


def build_datasets(
    jsonl_path: str,
    repo_root: str,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple["TeamCompletionDataset", "TeamCompletionDataset", Mappings]:
    """
    Load mappings, split records into train/val, and return separate dataset instances.

    Train and val get separate seeds so val masking is stable across epochs regardless
    of train access order.
    """
    mappings = Mappings.from_data_dir(repo_root)

    # Load all records once
    min_revealed = 2
    all_records: list[list[dict]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            mons = record.get("pokemon", [])
            if len(mons) >= min_revealed:
                all_records.append(mons)

    # Shuffle then split the record list — separate dataset instances, separate seeds
    rng_split = random.Random(seed)
    rng_split.shuffle(all_records)
    n_val = max(1, int(len(all_records) * val_split))
    val_records   = all_records[:n_val]
    train_records = all_records[n_val:]

    train_ds = TeamCompletionDataset(None, mappings, seed=seed,     _records=train_records)
    val_ds   = TeamCompletionDataset(None, mappings, seed=seed + 1, _records=val_records)

    # Compute per-move positive-frequency weight for the BCE move loss
    mappings.move_pos_weight = _compute_move_pos_weight(train_records, mappings)

    return train_ds, val_ds, mappings


if __name__ == "__main__":
    """Quick sanity check: print 3 examples."""
    from utils.git import get_repo_root
    repo_root = get_repo_root()
    mappings = Mappings.from_data_dir(repo_root)
    print(f"Vocabulary: {mappings.num_species} species, {mappings.num_moves} moves, {mappings.num_items} items")

    jsonl = os.path.join(repo_root, "data", "replay_teams.jsonl")
    ds = TeamCompletionDataset(jsonl, mappings, seed=0)
    print(f"Dataset size: {len(ds)} records")

    id_to_species = {v: k for k, v in mappings.species_to_id.items()}
    id_to_move    = {v: k for k, v in mappings.move_to_id.items()}
    id_to_item    = {v: k for k, v in mappings.item_to_id.items()}

    for i in range(3):
        ex = ds[i]
        print(f"\n--- Example {i} ---")
        for slot in range(TEAM_SIZE):
            if ex["is_padding"][slot]:
                continue
            sp   = id_to_species.get(ex["species_ids"][slot].item(), "?")
            t_sp = id_to_species.get(ex["target_species"][slot].item(), "?")
            masked = ex["is_masked"][slot].item()
            known_moves = [id_to_move.get(j, "?") for j, v in enumerate(ex["target_move_multihot"][slot]) if v > 0.5]
            item_id = ex["target_item"][slot].item()
            item_name = id_to_item.get(item_id, "none") if item_id else "none"
            print(
                f"  slot {slot}: {'[MASK]' if masked else sp:<12} "
                f"target={t_sp:<12} moves={known_moves} item={item_name}"
                f" item_known={ex['item_known'][slot].item()}"
            )
