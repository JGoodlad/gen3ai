"""The K9 learner-golden re-bake PROOF for gen3_smogon_species_usage_weighted_v1 (F-X5-47: the species-usage
marginal over the UNWEIGHTED Raw count).

Runs ONE golden update (`agents.training.learner_golden.compute`) for an ARM on a BUFFER, on whichever tree is
on PYTHONPATH — the OLD tree (the parent commit `7bed4347`, whose `species_usage()` is the Raw count) or the NEW
tree (W = sum Abilities) — optionally with the SPECIES-USAGE MARGINAL HELD AT A GIVEN FILE's values (`--hold`):
the facade's `gen3_data.priors.species_usage` is replaced, before the learner is built, by a loader of that file.
It is the only source of the marginal's values: `build_species_usage_prior` (the op's SPECIES_USAGE_PRIOR, the
T0 / BeliefHead marginal and lift baseline through `build_species_cooccur_prior`) reads
`gen3_data.priors.species_usage()` at call time.

The claim: NEW tree + `--hold <the parent's marginal>` is BYTE-IDENTICAL to the OLD tree (post-update params,
every init / post group hash, every pinned loss) on every buffer. Then the two trees differ in what one update
computes ONLY through the marginal's values, so every element the re-bake moves is the marginal's.

    PYTHONPATH=<tree>/src python proof.py --label X --arm blob --buffer <npz> \
        [--hold out/parent_species_usage.json] --out out/X.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", default="blob")
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--hold", default=None, help="hold the species-usage marginal at THIS file's values")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from agents.gen3_data import priors as P
    from agents.training import learner_golden as LG

    held = None
    if a.hold:
        raw = Path(a.hold).read_bytes()
        table = json.loads(raw)
        P.species_usage = lambda: table      # the facade's one source of the marginal's values
        held = hashlib.sha256(raw).hexdigest()
    import torch

    fp = LG.compute(LG.build_arm_learner(a.arm), buffer=Path(a.buffer))
    used = P.species_usage()
    res = {"label": a.label, "arm": a.arm, "buffer": a.buffer,
           "buffer_sha256": hashlib.sha256(Path(a.buffer).read_bytes()).hexdigest(),
           "usage_held_file_sha256": held,
           "usage_in_use_canonical_sha256": hashlib.sha256(json.dumps(used, sort_keys=True).encode()).hexdigest(),
           "tyranitar_usage_in_use": used["tyranitar"], "torch": torch.__version__, **fp}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("label", "arm", "tyranitar_usage_in_use", "init_params_sha256",
                                          "post_params_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
