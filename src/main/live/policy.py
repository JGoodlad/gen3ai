"""Policies a live session plays with — they read ONLY the reader's :class:`main.live.reader.Frame`.

* :class:`ModelPolicy` — a checkpoint on CPU (never the GPU: live play must not contend with a training lease),
  loaded through ``main.play.load_policy`` (the strict loader + the reveal refusal), forwarded on the frame's row
  and mask exactly as ``RLPlayer._predict_best_action`` forwards poke-env's: ``{"observation", "action_mask"}``
  → masked logits → argmax (greedy, the measurement setting) or a temperature sample from a SEEDED generator.
* :class:`RandomPolicy` — a seeded uniform draw over the legal actions (the gates' cheap, diverse driver; it also
  plays every action class the mask offers, which a greedy model does not).
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from main.live.reader import Frame


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def choose(self, frame: Frame) -> int:
        legal = [i for i in range(len(frame.mask)) if frame.mask[i] == 1 and i in frame.tokens]
        return self._rng.choice(legal)


class ModelPolicy:
    def __init__(self, path: str, *, temperature: float = 0.0, seed: Optional[int] = None,
                 device: str = "cpu") -> None:
        import torch

        from main.play import load_policy
        if device != "cpu":
            raise ValueError("live play forwards on CPU only (a training lease owns the GPU)")
        self.model = load_policy(path, device)
        # The obs keys the FORWARD reads beyond the row (the declared registry; label keys in the observation
        # SPACE are training targets the forward never reads). The reader produces the row and the mask only.
        from agents.model.extra_obs_keys import required_extra_obs_keys
        extra = [e.key for e in required_extra_obs_keys(self.model.policy.features_extractor)]
        if extra:
            raise ValueError(f"{path}: the policy's forward reads obs keys {extra} the live reader does not produce")
        self.temperature = float(temperature)
        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(int(seed))
        self.last_masked_logits: Optional[np.ndarray] = None

    def masked_logits(self, frame: Frame) -> "np.ndarray":
        import torch
        with torch.no_grad():
            obs = torch.as_tensor(np.array(frame.row, dtype=np.float32, copy=True)[None])
            mask = torch.as_tensor(frame.mask.astype(np.float32)[None])
            dist = self.model.policy.get_distribution({"observation": obs, "action_mask": mask})
            logits = dist.distribution.logits
            return (logits + (mask - 1.0) * 1e9)[0].cpu().numpy()

    def choose(self, frame: Frame) -> int:
        import torch
        ml = self.masked_logits(frame)
        self.last_masked_logits = ml
        if self.temperature <= 0.0:
            return int(np.argmax(ml))
        probs = torch.softmax(torch.as_tensor(ml) / self.temperature, dim=0)
        return int(torch.multinomial(probs, 1, generator=self._gen).item())
