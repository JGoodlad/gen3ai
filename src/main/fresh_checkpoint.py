"""A FRESHLY BUILT, seeded, untrained current-architecture checkpoint at the PRODUCTION arch
surface — the test fixture that is guaranteed to load on the tree under test.

Why it exists: a named baseline or an archived run is a checkpoint from SOME generation, and every
architecture boundary (``MIGRATION_FLOOR``) puts all of them behind the pre-generation wall at once.
A test whose point is that the pieces FIT — a policy drives real battles, a checkpoint loads, a
tool's output is reproducible — needs a POLICY, not a strong one. This builds that policy at the
current architecture with the production surface applied (``--arch production`` + the
``production_config.json`` mirror, as ``rust_core_cutover.envs.production_args`` resolves it — so
the belief heads a forensic trace reads exist), seeded so the same seed gives the same weights, and
saves it through the project's own save path (a real ``MaskablePPO.save`` plus
``save_model_snapshot``'s ``model_config.json`` / ``metadata.json`` beside it).

It generalises the shape the observation-architecture batch (``b0a28b5b``) used inline for the
anchors test and the baselines-loader test. A test that uses it for want of a current-generation
production node returns to the named baseline once one exists (the ``ai_v14_01_base`` lineage).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: The policy trunk — the value every training run builds with.
NET_ARCH = [512, 512]


def _production_policy_kwargs() -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """``(args, layout, policy_kwargs)`` for a fresh run at the production surface — the same
    fields ``main.train.model_build`` puts in a fresh run's ``policy_kwargs``."""
    import torch

    from agents.model.extractor_arch import build_extractor_arch_kwargs
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.policy import POLICY_ACTIVATION_FN
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from main.rust_core_cutover.envs import production_args

    args = production_args()
    enc = Gen3ObservationEncoder(load_mappings())
    fek = build_extractor_arch_kwargs(args, base=enc.get_features_extractor_kwargs())
    pk = {"features_extractor_class": Gen3FeaturesExtractor,
          "features_extractor_kwargs": fek,
          "net_arch": list(NET_ARCH),
          "activation_fn": POLICY_ACTIVATION_FN,
          "optimizer_class": torch.optim.AdamW,
          "optimizer_kwargs": {"weight_decay": args.weight_decay, "eps": 1e-5},
          "use_popart": args.use_popart,
          "value_from_dist": args.value_from_dist,
          "critic": args.critic}
    return args, fek["layout"], pk


def build_fresh_model(seed: int, *, num_timesteps: int = 0) -> Tuple[Any, Any, Dict[str, Any]]:
    """``(model, args, policy_kwargs)``: a seeded, untrained ``MaskablePPO`` +
    ``Gen3DualHeadMaskablePolicy`` at the production surface, on CPU. Same ``seed`` ⇒ same weights."""
    import gymnasium as gym
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.policy import Gen3DualHeadMaskablePolicy

    args, layout, pk = _production_policy_kwargs()
    total_dim = layout["total_dim"]
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (total_dim,), np.float32),
        "action_mask": gym.spaces.MultiBinary(11)})

    class _E(gym.Env):
        observation_space = obs_space
        action_space = gym.spaces.Discrete(11)

        def reset(self, **kwargs: Any) -> Any:
            return {"observation": np.zeros(total_dim, np.float32),
                    "action_mask": np.ones(11, np.int8)}, {}

        def step(self, action: Any) -> Any:
            return self.reset()[0], 0.0, False, False, {}

    torch.manual_seed(seed)
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, DummyVecEnv([_E]), policy_kwargs=pk,
                        verbose=0, device="cpu", seed=seed, vf_coef=args.vf_coef)
    model.num_timesteps = num_timesteps
    return model, args, pk


def save_fresh_checkpoint(run_dir: Path, seed: int, *, name: str = "final_model",
                          num_timesteps: int = 0,
                          config_extra: Optional[dict] = None) -> Path:
    """Build a fresh model (``build_fresh_model``), save it as ``run_dir/<name>.zip`` and write the
    run's ``model_config.json`` / ``metadata.json`` beside it. ``config_extra`` is merged into
    ``model_config.json`` (e.g. the recorded eval regime a reader requires). Returns the zip path."""
    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import save_model_snapshot

    model, args, pk = build_fresh_model(seed, num_timesteps=num_timesteps)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / name))
    version = ModelVersion.from_layout_and_policy_kwargs(
        pk["features_extractor_kwargs"]["layout"], pk, vf_coef=args.vf_coef,
        move_belief_coef=args.move_belief_coef, spread_belief_coef=args.spread_belief_coef,
        win_prob_coef=args.win_prob_coef, value_dist_coef=args.value_dist_coef,
        opp_belief_aux_coef=args.opp_belief_aux_coef,
        hp_type_belief_coef=args.hp_type_belief_coef, item_belief_coef=args.item_belief_coef)
    save_model_snapshot(str(run_dir), version, git_hash="test")
    if config_extra:
        cfg_path = run_dir / "model_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg.update(config_extra)
        cfg_path.write_text(json.dumps(cfg, indent=2))
    return run_dir / f"{name}.zip"


def load_fresh_policy(run_dir: Path, seed: int) -> Any:
    """Save a fresh seeded checkpoint to ``run_dir`` and load it back through the SAME loader a
    by-name baseline load uses (``load_foreign_opponent``, as ``baselines.load`` calls it), with
    the ObservationDebugger dropped exactly as ``rust_core_parity.load_production_policy`` does.
    The stand-in for the ``production`` policy while no current-generation production node exists."""
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    zip_path = save_fresh_checkpoint(run_dir, seed)
    model, _ = load_foreign_opponent(str(zip_path), current_model_version(load_mappings()),
                                     device="cpu", config_path=str(run_dir / "model_config.json"))
    for m in model.policy.modules():
        if hasattr(m, "_debugger"):
            setattr(m, "_debugger", None)
    return model
