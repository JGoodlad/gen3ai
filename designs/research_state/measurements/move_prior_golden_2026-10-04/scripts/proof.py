"""The K9 learner-golden re-bake PROOF for gen3_smogon_prior_denominator_v1 (F-X5-41: the deflated move prior).

Runs ONE golden update (`agents.training.learner_golden.compute`) for an ARM on a BUFFER, on whichever tree is
on PYTHONPATH — the OLD tree (the parent commit, whose `data/pokemon/gen3_move_priors.json` is Moves / Raw count)
or the NEW tree (Moves / W) — optionally with the MOVE PRIOR HELD AT A GIVEN FILE's values (`--prior`): the
facade's `gen3_data.priors.move_raw` is replaced, before the learner is built, by a loader of that file (the
only reader of the move prior's values: `build_move_prior_logits` -> `gen3_data.priors.moves` -> `move_raw`).

The claim: NEW tree + `--prior <the parent's file>` is BYTE-IDENTICAL to the OLD tree (post-update params, every
init / post group hash, every pinned loss) on every buffer. Then the two trees differ in what one update
computes ONLY through the move prior's values, so every element the re-bake moves is the prior's.

    PYTHONPATH=<tree>/src python proof.py --label new_oldprior --arm blob --buffer <npz> \
        [--prior <old gen3_move_priors.json>] --out <json>
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
    ap.add_argument("--prior", default=None, help="hold the move prior at THIS file's values")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from agents.gen3_data import priors as P
    from agents.training import learner_golden as LG

    held = None
    if a.prior:
        raw = Path(a.prior).read_bytes()
        table = json.loads(raw)
        P.move_raw = lambda: table          # the facade's one source of the prior's values
        held = hashlib.sha256(raw).hexdigest()
    import torch

    fp = LG.compute(LG.build_arm_learner(a.arm), buffer=Path(a.buffer))
    used = hashlib.sha256(json.dumps(P.move_raw(), sort_keys=True).encode()).hexdigest()
    res = {"label": a.label, "arm": a.arm, "buffer": a.buffer,
           "buffer_sha256": hashlib.sha256(Path(a.buffer).read_bytes()).hexdigest(),
           "prior_held_file_sha256": held, "prior_in_use_canonical_sha256": used,
           "skarmory_spikes_in_use": P.move_raw()["skarmory"]["spikes"],
           "torch": torch.__version__, **fp}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("label", "arm", "skarmory_spikes_in_use", "init_params_sha256",
                                          "post_params_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
