"""DESCRIPTIVE (Amendment 8: the v1 builder's Phase 2c rows, played before the R-GIGO FAIL stopped them and read only
AFTER the v2 read): R-MOVES on the v1 worlds, det_c − det_b at LEAF-K8, on the states both played (end2 complete;
prod partial, so end2 only).

    python -I score_moves_v1.py end2   →  printed (quoted in README §14)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S1  # noqa: E402
import score_belief as SB  # noqa: E402

KEYS = ("ARGMAX|CRITIC", "ARGMAX|LEAF-K8", "ARGMAX|TRUNC-4-K8")


def main():
    cp = sys.argv[1]
    SB.KEYS = KEYS
    st, Cw, _ = SB.per_state(cp, "det_c")
    _, Bw, _ = SB.per_state(cp, "det_b", set(Cw))
    ids = sorted(set(Cw) & set(Bw))
    E = st["states"]
    bt = S1.Boot([E[i]["battle"] for i in ids], [1.0] * len(ids))
    arr = lambda D, k: np.array([D[i][k] for i in ids])  # noqa: E731
    out = {"cp": cp, "n_states": len(ids), "builder": "v1 (descriptive)",
           "C LEAF-K8": bt.stat(arr(Cw, "ARGMAX|LEAF-K8")), "B LEAF-K8": bt.stat(arr(Bw, "ARGMAX|LEAF-K8")),
           "R-MOVES (v1): C - B LEAF-K8": bt.stat(arr(Cw, "ARGMAX|LEAF-K8") - arr(Bw, "ARGMAX|LEAF-K8")),
           "C CRITIC": bt.stat(arr(Cw, "ARGMAX|CRITIC"))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
