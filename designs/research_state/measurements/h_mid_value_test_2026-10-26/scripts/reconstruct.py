"""Replay banked anchor captures through a PINNED sim_bridge and read P*'s critic and policy on every decision.

Run with the env interpreter, PYTHONPATH=<PIN>/src (the read's own code: the obs encoder, the model loader), and
POKESIM_SIM_BRIDGE_BIN=<PIN>/src/rust_sim/target/release/sim_bridge:

    python reconstruct.py --model ZIP --cells-root READ/cells --cells pstar_home72_00,... --out OUT.jsonl.gz
        [--limit-games N] [--no-values]

One output row per game: the capture's identity, our side, the fidelity checks (PREREG §3) and, per decision of our
side, ``[n, turn, V, argmax, chosen, margin]`` (V = the win-prob critic, sigmoid of the stashed logit; ``chosen`` = the
recorded choice's action index; ``margin`` = top-1 minus top-2 masked logit). ``--no-values`` drops V from the output
(the pre-registration smoke reads fidelity only).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def bridge_bin() -> str:
    b = os.environ.get("POKESIM_SIM_BRIDGE_BIN")
    if not b or not os.path.isabs(b) or not os.path.exists(b):
        raise SystemExit("set POKESIM_SIM_BRIDGE_BIN to the read pin's absolute sim_bridge path")
    return b


def replay(commands: list, binary: str) -> tuple:
    """Feed the recorded stream (minus the teardown END) and collect (chunks, frames, error)."""
    import base64

    cmds = [c for c in commands if c.strip() != "END"]
    proc = subprocess.run([binary], input=("\n".join(cmds) + "\n").encode(), capture_output=True, timeout=300,
                          check=False)
    chunks, frames, err, ended = [], [], None, False
    for raw in proc.stdout.decode().split("\n"):
        if not raw:
            continue
        if raw == "__END__":
            ended = True
            break
        if raw.startswith("__ERR__"):
            err = base64.b64decode(raw[len("__ERR__ "):]).decode("utf-8")
            break
        if raw.startswith("__RECON__"):
            continue
        if raw.startswith("__OBS__ "):
            _m, slot, payload = raw.split(" ", 2)
            frames.append((slot, json.loads(payload)))
            continue
        slot, b64 = raw.split(" ", 1)
        chunks.append((slot, base64.b64decode(b64).decode("utf-8")))
    if not ended and err is None:
        err = f"no __END__ (rc {proc.returncode}; stderr {proc.stderr.decode()[-300:]!r})"
    return chunks, frames, err


class Model:
    def __init__(self, zip_path: str):
        import torch

        from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict

        self.torch = torch
        self.model = load_checkpoint_strict(zip_path, device="cpu", **historical_load_kwargs(zip_path))
        self.policy = self.model.policy
        self.training_flag = bool(self.policy.training)

    def run(self, rows: np.ndarray, masks: np.ndarray) -> tuple:
        """(V [N], masked logits [N, 11]) for a batch, through the same calls `CorePolicy.decide` makes plus the
        critic's read of the logit the same extractor forward stashed."""
        torch = self.torch
        with torch.no_grad():
            obs_t = torch.as_tensor(rows, dtype=torch.float32)
            mask_t = torch.as_tensor(masks.astype(np.int8))
            obs = {"observation": obs_t, "action_mask": mask_t}
            pi_features, vf = self.policy.extract_features(obs)
            v = self.policy._critic_value(vf).reshape(-1)
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            dist = self.policy._get_action_dist_from_latent(latent_pi)
            masked = dist.distribution.logits + (mask_t - 1.0) * 1e9
        return v.numpy().astype(float), masked.numpy().astype(float)


def our_side(commands: list) -> str:
    start = json.loads(commands[0][len("START "):])
    sides = (start.get("core_obs") or {}).get("sides") or []
    if len(sides) != 1:
        raise ValueError(f"core_obs sides {sides!r}: expected exactly our one side")
    return sides[0]


def process(cap: dict, binary: str, model: Model, with_values: bool) -> dict:
    from agents.battle.core_obs import wrap_row
    from utils.bridge.ws_frontend_replay import side_text

    cmds = cap["commands"]
    side = our_side(cmds)
    ours = [c[len(f"CHOOSE {side} "):] for c in cmds if c.startswith(f"CHOOSE {side} ")]
    forcelose = any(c.startswith(f"FORCELOSE {side}") for c in cmds)
    chunks, frames, err = replay(cmds, binary)
    out = {"tag": cap["tag"], "side": side, "err": err, "n_choose": len(ours), "forcelose": forcelose}
    if err is not None:
        out["ok"] = False
        return out
    cap_txt = side_text([tuple(c) for c in cap["chunks"]])
    rep_txt = side_text(chunks)
    out["protocol_identical"] = cap_txt == rep_txt
    fr = [f for s, f in frames if s == side]
    out["n_frames"] = len(fr)
    out["frames_ok"] = len(fr) == len(ours) or (forcelose and len(fr) == len(ours) + 1)
    # result from the replayed protocol, our view
    names = json.loads(cmds[0][len("START "):])
    our_name = names[side]["name"]
    res = None
    for ln in rep_txt[side].split("\n"):
        if ln.startswith("|win|"):
            res = 1.0 if ln[len("|win|"):].strip() == our_name else 0.0
        elif ln == "|tie" or ln.startswith("|tie|"):
            res = 0.5
    out["result"] = res
    n_dec = min(len(fr), len(ours))
    if n_dec == 0:
        out["ok"] = False
        return out
    rows = np.stack([np.array(wrap_row(fr[i]["frame"]), dtype=np.float32) for i in range(n_dec)])
    masks = np.array([fr[i]["mask"] for i in range(n_dec)], dtype=np.int8)
    v, logits = model.run(rows, masks)
    decs, n_match, n_bad = [], 0, 0
    for i in range(n_dec):
        inv = {tok: int(k) for k, tok in (fr[i].get("tokens") or {}).items()}
        chosen = inv.get(ours[i], -1)
        order = np.argsort(-logits[i])
        am = int(order[0])
        margin = float(logits[i][order[0]] - logits[i][order[1]]) if masks[i].sum() > 1 else float("inf")
        match = am == chosen
        n_match += int(match)
        if not match and margin > 1e-3:
            n_bad += 1
        decs.append([int(fr[i]["n"]), int(fr[i]["turn"]), round(float(v[i]), 6) if with_values else None, am, chosen,
                     round(margin, 5) if np.isfinite(margin) else None])
    out.update(decisions=decs, n_match=n_match, n_dec=n_dec, n_bad_disagree=n_bad,
               ok=bool(out["protocol_identical"] and out["frames_ok"] and n_bad == 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells-root", required=True)
    ap.add_argument("--cells", required=True, help="comma-separated unit dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--no-values", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    binary = bridge_bin()
    model = Model(a.model)
    print(f"[reconstruct] policy.training={model.training_flag} bridge={binary}", flush=True)
    t0 = time.time()
    n = 0
    with gzip.open(a.out, "wt") as fh:
        for unit in a.cells.split(","):
            cap_dir = Path(a.cells_root) / unit / "capture"
            caps = sorted(cap_dir.glob("*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
            for p in caps:
                if a.limit_games and n >= a.limit_games:
                    break
                cap = json.loads(p.read_text())
                row = process(cap, binary, model, not a.no_values)
                row["unit"] = unit
                fh.write(json.dumps(row) + "\n")
                n += 1
                if not row.get("ok"):
                    print(f"[reconstruct] NOT OK {unit}/{p.name}: err={row.get('err')} "
                          f"proto={row.get('protocol_identical')} frames={row.get('frames_ok')} "
                          f"bad={row.get('n_bad_disagree')}", flush=True)
            print(f"[reconstruct] {unit} done, {n} games, {time.time() - t0:.0f} s", flush=True)
    print(f"[reconstruct] DONE {n} games in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
