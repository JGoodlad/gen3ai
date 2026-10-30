"""Extract the per-parameter dither null's basis (`gen3_canary_param_null_v1`) from the 2026-10-30 GPU replay.

The replay (`python -m main.ops.canary_slice_read`, the `slice2-replay` window, 10-30 14:46-15:19, every step rc 0) wrote,
per state, ``dither.json`` (per slice, per judged parameter, 16 seeds of: null d(E32_s, E32_0), eager d(Ebf_s, E32_s),
compiled d(Cbf_s, E32_s)) and ``canary.json`` (the canary at the dump's own weights, slices 0 / 2 / 1, two repeats).

This script writes ``src/agents/model/testdata/canary_param_null_basis.json``:

  * ``census``: over EVERY judged parameter, slice and seed (62,880 reads), the old statistic c_s / max(e_s, f) and
    the new one c_s / max(e_s, n, f) with n the parameter's 16-seed null maximum: maxima, where, and the counts over K;
  * ``tail``: the raw per-seed numbers of every (state, slice, parameter) where some seed's OLD ratio exceeds K or the
    NEW one exceeds 3 (46 entries), so `canary_param_null_test` re-judges them through `canary_param_null.judge`;
  * ``undithered``: every parameter a canary at the dump's own weights refused (repeat 0, slices 0 / 2; parsed from
    the refusal text, unrounded where the watch holds it) with its null maximum.

    python designs/research_state/measurements/canary_slice2_2026-10-29/extract_param_null_basis.py \\
        ~/gen3ai_archive/slice2_replay_2026-10-30/slice2_replay src/agents/model/testdata/canary_param_null_basis.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

K = 12.0
TAIL_NEW = 3.0
PAT = re.compile(r"([\w.]+) compiled ([\d.e+-]+) vs eager ([\d.e+-]+)")


def r4(x: float) -> float:
    return float(f"{float(x):.4g}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(root: Path, out: Path) -> None:
    census = {"reads": 0, "entries": 0, "old_over_k": 0, "new_over_k": 0, "old_max": [-1.0], "new_max": [-1.0]}
    tail, und, files = [], [], {}
    for d in sorted(p for p in root.iterdir() if (p / "dither.json").is_file()):
        st = d.name
        files[st] = {"dither.json": sha(d / "dither.json"), "canary.json": sha(d / "canary.json"),
                     "verdict.json": sha(d / "verdict.json")}
        dither = json.loads((d / "dither.json").read_text())
        for sl, dr in dither.items():
            fl = float(dr["floor"])
            for p, v in dr["params"].items():
                n, e, c = (np.asarray(v[k], dtype=np.float64) for k in ("null", "eager", "compiled"))
                old = c / np.maximum(e, fl)
                new = c / np.maximum(np.maximum(e, n.max()), fl)
                census["reads"] += int(c.size)
                census["entries"] += 1
                census["old_over_k"] += int((old > K).sum())
                census["new_over_k"] += int((new > K).sum())
                if old.max() > census["old_max"][0]:
                    census["old_max"] = [r4(old.max()), st, int(sl), p]
                if new.max() > census["new_max"][0]:
                    census["new_max"] = [r4(new.max()), st, int(sl), p]
                if old.max() > K or new.max() > TAIL_NEW:
                    tail.append({"state": st, "slice": int(sl), "param": p, "floor": fl,
                                 "null": [r4(x) for x in n], "eager": [r4(x) for x in e],
                                 "compiled": [r4(x) for x in c]})
        canary = json.loads((d / "canary.json").read_text())
        for sl, rd in canary["repeats"][0].items():
            if not rd["refused"]:
                continue
            want = int(re.search(r"on (\d+) parameter", rd["text"]).group(1))
            found = PAT.findall(rd["text"])
            assert len(found) == want, (st, sl, want, found)
            for p, comp, eag in found:
                w = rd["watch"].get(p)
                c0, e0 = (w["compiled"], w["eager"]) if isinstance(w, dict) and "compiled" in w else (comp, eag)
                und.append({"state": st, "slice": int(sl), "param": p, "floor": float(dither[sl]["floor"]),
                            "compiled": r4(c0), "eager": r4(e0),
                            "null_max": r4(max(dither[sl]["params"][p]["null"]))})
    basis = {"source": str(root), "replay": "main.ops.canary_slice_read --dither 16, 2026-10-30 (window slice2-replay)",
             "k": K, "seeds": 16, "rel": 2.0 ** -9, "files_sha256": files, "census": census, "tail": tail,
             "undithered": und}
    out.write_text(json.dumps(basis, indent=1) + "\n")
    print(f"census {census}; tail {len(tail)}; undithered {len(und)} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
