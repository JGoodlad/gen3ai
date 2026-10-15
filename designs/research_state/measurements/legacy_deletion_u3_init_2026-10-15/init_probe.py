"""THE FRESH-INIT PROBE of legacy deletion U3 (D-LD7, `designs/ops/legacy_deletion_manifest_2026-10-12.md`).

Builds a fresh PRODUCTION-surface policy exactly as a fresh run seeds it (`main.fresh_checkpoint.build_fresh_model`:
`torch.manual_seed(seed)` + SB3 `seed=`, one torch thread, CPU) for each seed, and records

* every parameter's sha256 (float32 bytes) in `named_parameters()` order, with its shape — the policy AFTER SB3's
  orthogonal re-init and the policy's own retirements (what training starts from);
* the init FORWARD on the 64 compile-parity rows (`src/agents/model/compile_parity_obs.npz`, real Rust-collector
  rows): the extractor's two outputs, the masked action log-probs of every legal action, the entropy and the value
  (win-prob) output — each as a sha256 of its bytes plus its float64 sum, so two probes can be compared exactly.

Run it at the commit before U3 and at U3's commit, then `compare` the two JSONs:

    python designs/research_state/measurements/legacy_deletion_u3_init_2026-10-15/init_probe.py dump OUT.json
    python designs/research_state/measurements/legacy_deletion_u3_init_2026-10-15/init_probe.py compare A.json B.json

(PYTHONPATH must name the checkout's `src/`.) CPU only; ~1 min per seed on one core.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any, Dict, List

SEEDS = (1001, 2001, 4001)


def _sha(t: Any) -> str:
    import torch
    return hashlib.sha256(t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()).hexdigest()


def _probe(seed: int) -> Dict[str, Any]:
    import numpy as np
    import torch

    from main.fresh_checkpoint import build_fresh_model
    from utils.paths import src_path

    model, _args, _pk = build_fresh_model(seed)
    pol = model.policy.eval()
    params = [{"name": n, "shape": list(p.shape), "sha256": _sha(p)} for n, p in pol.named_parameters()]
    d = np.load(src_path("agents", "model", "compile_parity_obs.npz"))
    obs = torch.as_tensor(d["obs"])
    masks = torch.as_tensor(d["action_mask"]).bool()
    out: Dict[str, Any] = {"seed": seed, "n_params": len(params),
                           "n_elements": int(sum(p.numel() for p in pol.parameters())), "params": params}
    with torch.no_grad():
        pi, vf = pol.features_extractor({"observation": obs})
        fwd: Dict[str, Any] = {"pi_features": pi, "vf_features": vf}
        logps: List[Any] = []
        for a in range(masks.shape[1]):
            acts = torch.full((obs.shape[0],), a, dtype=torch.long)
            values, logp, ent = pol.evaluate_actions({"observation": obs, "action_mask": masks.float()}, acts,
                                                     action_masks=masks)
            logps.append(torch.where(masks[:, a], logp, torch.zeros_like(logp)))
        fwd["masked_log_probs"] = torch.stack(logps, dim=-1)
        fwd["entropy"] = ent
        fwd["value"] = values
    out["forward"] = {k: {"shape": list(v.shape), "sha256": _sha(v), "sum": float(v.double().sum())}
                      for k, v in fwd.items()}
    return out


def dump(path: str) -> None:
    from utils.torch_state_guard import single_thread_build
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True,
                                text=True).stdout.strip())
    with single_thread_build():
        res = {"commit": head, "src_dirty": dirty, "seeds": [_probe(s) for s in SEEDS]}
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {path} (commit {head[:8]}, src dirty: {dirty})")


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"A {a['commit'][:8]} (dirty {a['src_dirty']})  vs  B {b['commit'][:8]} (dirty {b['src_dirty']})")
    for sa, sb in zip(a["seeds"], b["seeds"]):
        assert sa["seed"] == sb["seed"]
        pa = {p["name"]: p for p in sa["params"]}
        pb = {p["name"]: p for p in sb["params"]}
        gone = [n for n in pa if n not in pb]
        new = [n for n in pb if n not in pa]
        same = [n for n in pa if n in pb and pa[n]["sha256"] == pb[n]["sha256"]]
        moved = [n for n in pa if n in pb and pa[n]["sha256"] != pb[n]["sha256"]]
        print(f"\nseed {sa['seed']}: params A {sa['n_params']} ({sa['n_elements']:,} el) / B {sb['n_params']} "
              f"({sb['n_elements']:,} el); only-A {len(gone)}, only-B {len(new)}, shared {len(same) + len(moved)}: "
              f"bit-identical {len(same)}, CHANGED {len(moved)}")
        if gone:
            print("  only in A:", gone)
        if new:
            print("  only in B:", new)
        if moved:
            print("  changed:", moved)
        for k in sa["forward"]:
            fa, fb = sa["forward"][k], sb["forward"].get(k)
            eq = fb is not None and fa["sha256"] == fb["sha256"]
            print(f"  forward {k:18s} {'IDENTICAL' if eq else 'DIFFERS'}  sum A {fa['sum']:.9g}  "
                  f"B {fb['sum'] if fb else None}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: init_probe.py dump OUT.json | compare A.json B.json")
