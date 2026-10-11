"""POST-HOC (not registered; read after Phase 1's verdicts): how hard must the completed-Q policy TARGET
trust the leaf? IMP of the Gumbel-score argmax, logit + (50 + K)·c_scale·q̃, as c_scale grows from the
`mctx` default 0.1 toward a pure value argmax, for three leaves; and the expected IMP of SAMPLING from the
improved policy softmax(logit + σ) (what a distilled policy would play at temperature 1), on the 24 truth seeds.

    python -I target_scale_read.py <cp> [<cp> ...]  →  target_scale_read.json beside it
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S1  # noqa: E402

SCALES = (0.1, 0.3, 1.0, 3.0, 10.0)
LEAVES = ("LEAF-K8", "LEAF-K4", "TRUNC-4-K4", "MIX-K8")


def main():
    out = {}
    for cp in sys.argv[1:]:
        st, rows, _, _ = S1.load(cp, False)
        E = st["states"]
        ids = sorted(i for i in E if rows[i].get("ok"))
        per = {f"{lf}|c={c}|{m}": [] for lf in LEAVES for c in SCALES for m in ("argmax", "sample")}
        for did in ids:
            S = S1.state_arrays(rows[did], E[did])
            sc = S1.scorers(S)
            T, t1, lg, order = S["truth"], S["top1"], S["lg"], S["order"]
            for lf in LEAVES:
                q = sc[lf]
                K = S1.n_of(lf)
                lo, hi = q.min(), q.max()
                qs = (q - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(q)
                for c in SCALES:
                    g = lg + (S1.C_VISIT + K) * c * qs
                    a = S1.pick(g, order)
                    per[f"{lf}|c={c}|argmax"].append(T[a] - T[t1])
                    z = np.exp(g - g.max())
                    p = z / z.sum()
                    per[f"{lf}|c={c}|sample"].append(float((p * T).sum() - T[t1]))
        bt = S1.Boot([E[i]["battle"] for i in ids], [1.0] * len(ids))
        out[cp] = {k: bt.stat(np.array(v)) for k, v in per.items()}
        # the raw policy sampled at T = 1, for scale
        pol = []
        for did in ids:
            S = S1.state_arrays(rows[did], E[did])
            z = np.exp(S["lg"] - S["lg"].max())
            p = z / z.sum()
            pol.append(float((p * S["truth"]).sum() - S["truth"][S["top1"]]))
        out[cp]["POLICY|sample"] = bt.stat(np.array(pol))
    (S1.HERE / "target_scale_read.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for cp, d in out.items():
        for k, v in d.items():
            print(cp, k, v["mean"], v["ci"])


if __name__ == "__main__":
    main()
