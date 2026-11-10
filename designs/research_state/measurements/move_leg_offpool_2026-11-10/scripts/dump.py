"""Dump, per (bank row, REVEALED opponent mon with labelled moves), the move belief's raw logits of one checkpoint:
the posterior (belief-prior-mix 0) and the Smogon prior alone (mix 1). CPU only; reuses the meter's loader, forward
and observation slicing (main.offpool_belief.score). Output: one npz per panel, outside every checkout."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from main.offpool_belief import score as SC
from main.offpool_belief.bank import load_bank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--bank", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th GAME (a sample, by game index)")
    a = ap.parse_args()
    from utils.torch_state_guard import torch_globals

    out = Path(a.out)
    for d in (out, *out.resolve().parents):
        if (d / ".git").exists():
            raise SystemExit(f"{out} is inside a checkout: the dump states held-out teams")
    out.mkdir(parents=True, exist_ok=True)
    with torch_globals(num_threads=1):
        model, ck = SC.load_checkpoint(a.ckpt)
        fe = model.policy.features_extractor
        ix = SC.obs_index(fe.pokemon_encoder.layout)
        hb = fe.hypothesis_builder
        np.savez_compressed(out / f"{a.label}__tables.npz",
                            move_legal=hb.move_legal.numpy(), move_valid=hb.move_valid.numpy(),
                            meter_legal=(fe.move_belief.move_prior_probs > 2.0 * SC._illegal_prob()).numpy(),
                            prior_logits=fe.move_belief.move_prior_logits.numpy(),
                            ckpt_sha256=np.array(ck["sha256"]), ckpt_path=np.array(ck["path"]))
        for bp in a.bank:
            bank = load_bank(Path(bp))
            rows = bank.rows
            keep = np.flatnonzero(rows["game"] % a.stride == 0)
            t0 = time.time()
            acc = {k: [] for k in ("row", "slot", "team", "game", "turn", "species", "shown", "shown_flag", "truth",
                                   "n_revealed_mons", "post", "prior")}
            for a0 in range(0, len(keep), SC.CHUNK):
                idx = keep[a0:a0 + SC.CHUNK]
                obs, mask = rows["obs"][idx], rows["mask"][idx]
                post = SC._forward(model, obs, mask, 0.0)["move_logits"]
                prior = SC._forward(model, obs, mask, 1.0)["move_logits"]
                opp = obs[:, ix.opp_start:ix.opp_start + ix.n_slots * ix.slot_dim].reshape(-1, ix.n_slots, ix.slot_dim)
                species = np.rint(opp[:, :, ix.species]).astype(np.int64)
                shown_ids = np.rint(opp[:, :, list(ix.move_ids)]).astype(np.int64)
                shown_flag = opp[:, :, list(ix.move_known)]
                km = rows["known_moves"][idx]
                rr, ss = np.nonzero((km[:, :, 0] >= 0) & (species > 0))
                acc["row"].append(idx[rr]); acc["slot"].append(ss)
                acc["team"].append(rows["team"][idx][rr]); acc["game"].append(rows["game"][idx][rr])
                acc["turn"].append(rows["turn"][idx][rr]); acc["species"].append(species[rr, ss])
                acc["shown"].append(shown_ids[rr, ss]); acc["shown_flag"].append(shown_flag[rr, ss] > 0.5)
                acc["truth"].append(km[rr, ss])
                acc["n_revealed_mons"].append((species > 0).sum(-1)[rr])
                acc["post"].append(post[rr, ss].astype(np.float32)); acc["prior"].append(prior[rr, ss].astype(np.float32))
                if (a0 // SC.CHUNK) % 20 == 0:
                    print(f"[{a.label} {bank.panel}] rows {a0 + len(idx)}/{len(keep)} {time.time() - t0:.0f}s", flush=True)
            np.savez(out / f"{a.label}__{bank.panel}.npz", n_teams=np.array(bank.n_teams),
                     rows_sha256=np.array(bank.manifest["rows_sha256"]),
                     **{k: np.concatenate(v) for k, v in acc.items()})
            print(f"[{a.label} {bank.panel}] done: {sum(len(x) for x in acc['row'])} units, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
