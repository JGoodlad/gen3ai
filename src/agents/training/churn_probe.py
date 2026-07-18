"""Function-space CHURN probe — is the policy still MOVING, independent of weight-space drift?

The steady-state question for the z_arch/FiLM run cannot be answered from weight metrics:
Adam turns any small-but-consistent gradient into constant-speed weight motion, and PPO moves a
clip-bounded amount per update forever — so `film/dev` can grow linearly at full FUNCTIONAL
convergence (gauge drift: the FiLM magnitude and the downstream weights trade off). The direct
read is the policy KL between two checkpoints on a FIXED probe-state set:

  - overall mean/median KL → "is the function still moving" (falls toward the PPO noise floor
    as the run reaches functional steady state);
  - the PER-TEAM split (probe states grouped by OUR roster) → HOW the movement distributes:
    uniform churn = global drift; a heavy per-team tail with a quiet aggregate = per-team
    conditioning still being (re)written — the noise-fitting signature if it never settles.

Two modes (CLI):
  collect  — play bridge battles (ckpt vs itself on the pool) ONCE and save obs+mask to an npz;
             the set is then FROZEN so every later comparison is apples-to-apples.
  compare  — KL between two checkpoints on the frozen set, overall + per-roster-group.

  python -m agents.training.churn_probe collect <ckpt> <config> <out.npz> [--battles 40]
  python -m agents.training.churn_probe compare <ckpt_a> <ckpt_b> <config> <probe.npz>

Pure math (`masked_kl`, `roster_keys`) is bridge-free and unit-tested (churn_probe_test.py).
"""
import argparse
import asyncio

import numpy as np

_EPS = 1e-12


def masked_kl(pa: np.ndarray, pb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise KL(pa ‖ pb) over LEGAL actions only. ``pa``/``pb`` [N, A] are masked action
    probability rows (illegal entries carry ~0 mass by construction — see
    warmstart.masked_action_probs); ``mask`` [N, A] re-asserts legality so a numerically
    nonzero illegal remnant can never contribute. Returns [N] (nats), asymmetric by design
    (pa = the LATER policy: "how far did it move from pb's prediction")."""
    m = mask > 0.5
    pa = np.where(m, pa, 0.0)
    pb = np.where(m, pb, 0.0)
    return (pa * (np.log(pa + _EPS) - np.log(pb + _EPS))).sum(-1)


def roster_keys(obs: np.ndarray, layout: dict) -> list:
    """Per-row hashable key of OUR roster (sorted species ids of slots 0..5) — the per-team
    grouping axis. Uses the SAME layout-driven slicer the model uses (no hardcoded offsets)."""
    import torch as th
    from agents.model.features_extractor import slice_pokemon_categoricals, TEAM_SIZE, POKEMON_FULL_DIM
    part = th.tensor(obs[:, : 12 * POKEMON_FULL_DIM], dtype=th.float32).reshape(len(obs), 12, POKEMON_FULL_DIM)
    sp = slice_pokemon_categoricals(part, layout)["species_ids"][:, :TEAM_SIZE].numpy()
    return [tuple(sorted(row.tolist())) for row in sp]


def churn(ckpt_a: str, ckpt_b: str, config_path: str, probe_npz: str, top_groups: int = 10) -> dict:
    """KL(π_b ‖ π_a) on the frozen probe set (b = the LATER checkpoint) → overall + per-team."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.training.warmstart import masked_action_probs

    z = np.load(probe_npz)
    obs, mask = z["obs"], z["mask"]
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()["layout"]
    cv = current_model_version(mappings)
    ma, _ = load_foreign_opponent(ckpt_a, current_version=cv, device="cpu", config_path=config_path)
    mb, _ = load_foreign_opponent(ckpt_b, current_version=cv, device="cpu", config_path=config_path)
    pa = masked_action_probs(ma, obs, mask)
    pb = masked_action_probs(mb, obs, mask)
    kl = masked_kl(pb, pa, mask)                                  # later ‖ earlier
    keys = roster_keys(obs, layout)
    by = {}
    for k, v in zip(keys, kl):
        by.setdefault(k, []).append(float(v))
    groups = sorted(((len(v), float(np.mean(v))) for v in by.values()), reverse=True)
    gmeans = [g for _, g in groups if _ >= 5]                     # groups with enough rows
    return {
        "n_states": int(len(kl)),
        "kl_mean": float(np.mean(kl)),
        "kl_median": float(np.median(kl)),
        "kl_p95": float(np.percentile(kl, 95)),
        "n_team_groups": int(len(by)),
        "group_kl_mean": float(np.mean(gmeans)) if gmeans else None,
        "group_kl_max": float(np.max(gmeans)) if gmeans else None,
        "group_kl_spread": float(np.std(gmeans)) if gmeans else None,   # per-team unevenness
        "top_groups": groups[:top_groups],
    }


async def collect_probe_states(ckpt: str, config_path: str, out_npz: str, battles: int = 40) -> int:
    """Play bridge battles (ckpt vs itself, both on the full pool) and freeze obs+mask to npz."""
    from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
    from agents.inference.player import RLPlayer
    from agents.observation.state_encoder import load_mappings
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder
    from utils.bridge.local_battle_runner import run_local_battles

    mappings = load_mappings()
    model, _ = load_foreign_opponent(ckpt, current_version=current_model_version(mappings),
                                     device="cpu", config_path=config_path)
    fe = model.policy.features_extractor
    if hasattr(fe, "_debugger"):
        fe._debugger = None
    pool_tb = Gen3Teambuilder(TeamLoader().get_all_teams())

    class _Coll(RLPlayer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.O, self.M = [], []

        def embed_battle(self, b):
            d = super().embed_battle(b)
            self.O.append(np.asarray(d["observation"], np.float32).copy())
            self.M.append(np.asarray(d["action_mask"], np.float32).copy())
            return d

    _acct = lambda tag: AccountConfiguration(tag, "pw")
    c = _Coll(model=model, team=pool_tb, battle_format="gen3ou",
              server_configuration=LocalhostServerConfiguration, mappings=mappings,
              account_configuration=_acct("ChA"), stochastic=False, start_listening=False)
    o = RLPlayer(model=model, team=pool_tb, battle_format="gen3ou",
                 server_configuration=LocalhostServerConfiguration, mappings=mappings,
                 account_configuration=_acct("ChB"), stochastic=False, start_listening=False)
    await run_local_battles(c, o, battles, concurrency=2)
    obs, mask = np.stack(c.O), np.stack(c.M)
    np.savez_compressed(out_npz, obs=obs, mask=mask)
    print(f"[churn] froze {obs.shape[0]} probe states from {battles} battles -> {out_npz}", flush=True)
    return obs.shape[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    pc = sub.add_parser("collect")
    pc.add_argument("ckpt"); pc.add_argument("config"); pc.add_argument("out")
    pc.add_argument("--battles", type=int, default=40)
    pk = sub.add_parser("compare")
    pk.add_argument("ckpt_a"); pk.add_argument("ckpt_b"); pk.add_argument("config"); pk.add_argument("probe")
    args = p.parse_args()
    if args.mode == "collect":
        asyncio.run(collect_probe_states(args.ckpt, args.config, args.out, battles=args.battles))
    else:
        r = churn(args.ckpt_a, args.ckpt_b, args.config, args.probe)
        print(f"[churn] {args.ckpt_a} -> {args.ckpt_b}  ({r['n_states']} states, {r['n_team_groups']} team groups)")
        print(f"  KL mean {r['kl_mean']:.4f} | median {r['kl_median']:.4f} | p95 {r['kl_p95']:.4f}")
        print(f"  per-team group KL: mean {r['group_kl_mean']} | max {r['group_kl_max']} | spread {r['group_kl_spread']}")


if __name__ == "__main__":
    main()
