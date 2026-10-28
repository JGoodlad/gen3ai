"""X48 §8.7 / NOTE 1 (b): have the per-kind experts diverged from the Pokémon FFN by more than chance drift?

Weights only (CPU, no model build; `pin_worker depth` refuses kffn checkpoints, FINDING 8). For each X48 checkpoint, per
trunk round r (execution order: ``transformer_layers.0``, ``.1``, ``extra_rounds.0``), per non-Pokémon kind k and per
linear l in {linear1, linear2}:

    D_k = ||W_k - W_mon||_F / ||W_mon||_F          (W = the ``.weight`` matrix; ``mon`` = the round's own linear)

and beside it the NULL, two copies drifting independently by C0's own amount:

    null = sqrt(2) * mean over C0's three seeds of ||W_C0 - W_P*||_F / ||W_P*||_F   (the same round and linear, C0 at
           the SAME node: its final for X48's final, its +10M checkpoint for X48's +10M)

The label quantity (NOTE 1 (b), fixed here before any read game): at the final, for each (round in {0, 1}, kind k),
R_k = the mean over the three X48 seeds and over {linear1, linear2} of D_k / null. "experts not engaged" holds iff
R_k <= 1.0 for EVERY such (round, kind) (rule 8: a value within 1e-9 of 1.0 counts as <= 1.0). Round 2 is reported
and excluded (its board / threat / event experts get no policy or value gradient by construction). A seed-matched
null (C0 s350x for X48 s350x) is printed as a sensitivity line, never deciding. Also reported: each X48 expert's and
mon FFN's own drift from P* (||W - W_P*|| / ||W_P*||), and the bias-inclusive D_k.

Every input is sha256-checked against the registered value or recorded. ``features_extractor`` is read; the
``pi_`` / ``vf_`` aliases are asserted equal to it. Read-only on ``models/``. Prints one JSON document.

    python .../expert_divergence.py --models-root models > .../expert_divergence.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROUNDS = ("transformer_layers.0", "transformer_layers.1", "extra_rounds.0")
KINDS = ("board", "move", "threat", "event")
LINEARS = ("linear1", "linear2")
PFX = "features_extractor.team_transformer."
SEEDS = (3501, 3502, 3503)
EPS = 1e-9
P_STAR = ("rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip",
          "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed")
C0_FINAL_SHA = {3501: "09a6a2df48509a59fd9aefe6d20d9722aad6717f1730c6dfd75274446b283303",
                3502: "7e9a9890e2b397de6d8b87588c597d7f53e39b84efd3d1080305f77dbf22feec",
                3503: "99096be8eccfc1cef4268ab4d6de15063a386cda475e72340212fdbf3a1b7e26"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path) -> Dict[str, torch.Tensor]:
    with zipfile.ZipFile(p) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    for k, v in sd.items():
        if k.startswith(PFX):
            for alias in ("pi_", "vf_"):
                a = alias + k
                if a in sd and not torch.equal(sd[a], v):
                    raise SystemExit(f"{p}: {a} differs from {k}")
    return {k: v.double() for k, v in sd.items() if k.startswith(PFX)}


def node_zip(run: Path, node: str) -> Path:
    if node == "final":
        return run / "final_model.zip"
    zs = sorted(run.glob("checkpoints/checkpoint_29000*_steps.zip"))
    if len(zs) != 1:
        raise SystemExit(f"{run}: {len(zs)} +10M checkpoints, want 1")
    return zs[0]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def cat(sd: Dict[str, torch.Tensor], base: str) -> torch.Tensor:
    return torch.cat([sd[base + ".weight"].flatten(), sd[base + ".bias"].flatten()])


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-root", type=Path, required=True)
    a = ap.parse_args(argv)
    root = a.models_root
    inputs: Dict[str, Any] = {}
    pstar_p = root / P_STAR[0]
    inputs["pstar"] = {"path": str(pstar_p), "sha256": sha256(pstar_p)}
    if inputs["pstar"]["sha256"] != P_STAR[1]:
        raise SystemExit("P* sha256 mismatch")
    pstar = load(pstar_p)
    out: Dict[str, Any] = {"schema": "x48_expert_divergence_v1", "rounds": list(ROUNDS), "kinds": list(KINDS),
                           "linears": list(LINEARS), "inputs": inputs, "nodes": {}}
    for node in ("plus10m", "final"):
        c0_drift: Dict[Tuple[int, str, str], float] = {}
        for s in SEEDS:
            p = node_zip(root / f"rb_x35_c0_s{s}", "final" if node == "final" else "10m")
            h = sha256(p)
            if node == "final" and h != C0_FINAL_SHA[s]:
                raise SystemExit(f"C0 s{s} final sha256 mismatch")
            inputs[f"c0_s{s}_{node}"] = {"path": str(p), "sha256": h}
            sd = load(p)
            for r in ROUNDS:
                for lin in LINEARS:
                    k = f"{PFX}{r}.{lin}.weight"
                    c0_drift[(s, r, lin)] = rel(sd[k], pstar[k])
        null = {(r, lin): math.sqrt(2) * sum(c0_drift[(s, r, lin)] for s in SEEDS) / len(SEEDS)
                for r in ROUNDS for lin in LINEARS}
        rows: List[Dict[str, Any]] = []
        for s in SEEDS:
            p = node_zip(root / f"rb_x35_x48kffn_s{s}", "final" if node == "final" else "10m")
            inputs[f"x48_s{s}_{node}"] = {"path": str(p), "sha256": sha256(p)}
            sd = load(p)
            for ri, r in enumerate(ROUNDS):
                for lin in LINEARS:
                    mon_k = f"{PFX}{r}.{lin}"
                    w_mon = sd[mon_k + ".weight"]
                    for kind in KINDS:
                        ek = f"{PFX}{r}.kind_ffn.{kind}.{lin}"
                        d = rel(sd[ek + ".weight"], w_mon)
                        rows.append({
                            "seed": s, "round": ri, "kind": kind, "linear": lin, "D_k": d,
                            "null": null[(r, lin)], "ratio": d / null[(r, lin)],
                            "null_seed_matched": math.sqrt(2) * c0_drift[(s, r, lin)],
                            "ratio_seed_matched": d / (math.sqrt(2) * c0_drift[(s, r, lin)]),
                            "D_k_with_bias": rel(cat(sd, ek), cat(sd, mon_k)),
                            "expert_drift_from_pstar": rel(sd[ek + ".weight"], pstar[mon_k + ".weight"]),
                            "mon_drift_from_pstar": rel(w_mon, pstar[mon_k + ".weight"])})
        summary: Dict[str, Any] = {}
        for ri in range(len(ROUNDS)):
            for kind in KINDS:
                sel = [x for x in rows if x["round"] == ri and x["kind"] == kind]
                summary[f"r{ri}.{kind}"] = {
                    "R_k": sum(x["ratio"] for x in sel) / len(sel),
                    "R_k_seed_matched": sum(x["ratio_seed_matched"] for x in sel) / len(sel),
                    "mean_D_k": sum(x["D_k"] for x in sel) / len(sel)}
        label_keys = [f"r{ri}.{k}" for ri in (0, 1) for k in KINDS]
        not_engaged = all(summary[k]["R_k"] <= 1.0 + EPS for k in label_keys)
        not_engaged_sm = all(summary[k]["R_k_seed_matched"] <= 1.0 + EPS for k in label_keys)
        out["nodes"][node] = {
            "c0_drift": {f"s{s}.{r}.{lin}": v for (s, r, lin), v in c0_drift.items()},
            "null": {f"{r}.{lin}": v for (r, lin), v in null.items()},
            "rows": rows, "summary": summary,
            "label_rounds_0_1": {"keys": label_keys, "min_R_k": min(summary[k]["R_k"] for k in label_keys),
                                 "max_R_k": max(summary[k]["R_k"] for k in label_keys),
                                 "experts_not_engaged": not_engaged,
                                 "seed_matched_sensitivity": {"experts_not_engaged": not_engaged_sm,
                                                              "min_R_k": min(summary[k]["R_k_seed_matched"]
                                                                             for k in label_keys)}}}
    out["experts_not_engaged_at_final"] = out["nodes"]["final"]["label_rounds_0_1"]["experts_not_engaged"]
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
