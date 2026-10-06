"""The K9 learner-golden re-bake PROOF for gen3_x5_hyp_gather_v1: ONE golden update
(`learner_golden.compute`) for an ARM on its committed buffer, on this tree, optionally with the
hypothesis encoding put back to the PER-ROW pass (`--perrow`, a measurement-only patch in this process:
`extractor_forward.gathered_hypothesis_tokens` -> `pokemon_encoder(hypothesis_ctx(...))[:, 6:12]` with the
real pass's move tokens restored — exactly the parent's code path).

Claim: this tree + `--perrow` is BYTE-IDENTICAL to the recorded golden (every field), so the fix moves the
golden ONLY through the hypothesis encoding.

    proof.py --label X --arm fixed_mass [--perrow] --out X.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def perrow_patch() -> None:
    import agents.model.extractor_forward as EF
    from agents.model.hypothesis_tokens import hypothesis_ctx
    from agents.observation.constants import TEAM_SIZE
    real_build = EF.ExtractorForward._build_hypothesis_species
    box = {}

    def build_hs(self, ctx, role):
        box["hs"] = real_build(self, ctx, role)
        return box["hs"]

    def perrow_tokens(pe, embeddings, ctx, slot_species, dex_rows):
        keep = pe.last_move_tokens
        tok = pe(hypothesis_ctx(ctx, box["hs"], pe.layout), embeddings)
        pe.last_move_tokens = keep
        return tok[:, TEAM_SIZE:2 * TEAM_SIZE]
    EF.ExtractorForward._build_hypothesis_species = build_hs
    EF.gathered_hypothesis_tokens = perrow_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", default="fixed_mass")
    ap.add_argument("--perrow", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.perrow:
        perrow_patch()
    import torch
    from agents.training import learner_golden as LG
    fp = LG.compute(LG.build_arm_learner(a.arm), buffer=LG.arm_buffer(a.arm))
    res = {"label": a.label, "arm": a.arm, "perrow": a.perrow, "torch": torch.__version__, **fp}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("label", "arm", "perrow", "post_params_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
