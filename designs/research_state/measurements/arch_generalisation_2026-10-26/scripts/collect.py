"""COLLECT: P* (greedy) self-play, p1 pilots a PANEL team, p2 pilots a home72 team (familiar). Saves every decision
row of BOTH sides with the core's belief / item labels and the game outcome. Usage: collect.py <panel> <n_games> <seed>

Deterministic: fixed teams (p1 cycles the panel's 72 teams in order, p2 a seeded home72 draw), fixed battle seeds,
greedy P* on both sides, one thread in the core, envs stepped in index order."""
from __future__ import annotations

import sys
import time

import numpy as np

import common as C


def main(panel: str, n_games: int, seed: int, n_envs: int = int(__import__("os").environ.get("PROBE_ENVS", "24"))) -> None:
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    rng = np.random.default_rng(seed)
    p1 = C.pack(C.panel_texts(panel))
    home = C.pack(C.panel_texts("home72"))
    teams = p1 + home
    model = C.load_model()
    spec = P.spec_json(n=n_envs, threads=1, teams=teams, names=("argena", "argenb"), turn_limit=250,
                       refusal_budget=4, bank_dir=None, labels=("belief", "item"),
                       opponents=[{"kind": "external"}])
    core = ffi.FfiCore(spec, lib=C.lib())
    c = core.cols
    started = [0]
    ep_meta = {}

    def stage(envs):
        for e in envs:
            i = started[0] % 72
            j = int(rng.integers(0, 72))
            if panel == "home72" and j == i:
                j = (j + 1) % 72
            c["ep_team"][e] = [i, 72 + j]
            c["ep_seed"][e] = rng.integers(0, 65536, 4)
            c["ep_opp"][e] = 0
            ep_meta[e] = (started[0], i, j)
            started[0] += 1

    stage(range(n_envs))
    cur = dict(ep_meta)              # the episodes RESET starts
    core.reset()
    stage(range(n_envs))             # staged one ahead: what each env's auto-reset starts next
    buf = {e: [] for e in range(n_envs)}
    out = {k: [] for k in ("obs", "mask", "side", "game", "p1team", "p2team", "turn", "known_moves",
                           "belief_species", "belief_moves", "item_label", "item_mask", "winner")}
    done = 0
    t0 = time.time()
    try:
        while done < n_games:
            pend = [(e, s) for e in range(n_envs) for s in (0, 1) if c["need"][e, s]]
            if pend:
                obs = np.stack([c["obs"][e, s].copy() for e, s in pend]).astype(np.float32)
                mask = np.stack([c["mask"][e, s].astype(bool) for e, s in pend])
                probs, _, _ = C.forward(model, obs, mask)
                acts = probs.argmax(-1)
                for k, (e, s) in enumerate(pend):
                    assert mask[k][acts[k]]
                    buf[e].append(dict(obs=obs[k], mask=mask[k], side=s, turn=int(c["turn"][e]),
                                       known_moves=c["known_moves"][e, s].copy(),
                                       belief_species=c["belief_species"][e, s].copy(),
                                       belief_moves=c["belief_moves"][e, s].copy(),
                                       item_label=c["item_label"][e, s].copy(),
                                       item_mask=c["item_mask"][e, s].copy()))
                    c["action"][e, s] = int(acts[k])
            core.step()
            fin = core.finished()
            for e in range(n_envs):
                if c["refused"][e]:
                    raise RuntimeError(f"env {e}: a battle was quarantined (refused); this probe does not re-route it")
            for f in fin:
                e = f["env"]
                rows, buf[e] = buf[e], []
                g, i, j = cur[e]
                if done < n_games and rows:
                    for r in rows:
                        for k in ("obs", "mask", "side", "turn", "known_moves", "belief_species", "belief_moves",
                                  "item_label", "item_mask"):
                            out[k].append(r[k])
                        out["game"].append(g)
                        out["p1team"].append(i)
                        out["p2team"].append(j)
                        out["winner"].append(int(f["winner"]))
                    done += 1
            # a finished env auto-reset at this step from the staged episode: it is now current; stage the next
            for f in fin:
                cur[f["env"]] = ep_meta[f["env"]]
            if fin:
                stage([f["env"] for f in fin])
    finally:
        core.close()
    arr = {k: np.asarray(v) for k, v in out.items()}
    np.savez_compressed(C.OUT / f"rows_{panel}{__import__('os').environ.get('PROBE_TAG', '')}.npz", **arr)
    print(f"{panel}: {done} games, {len(arr['obs'])} rows, {time.time() - t0:.0f}s, "
          f"p1 win {np.mean([w == 1 for w in arr['winner']]):.3f} (row-weighted)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
