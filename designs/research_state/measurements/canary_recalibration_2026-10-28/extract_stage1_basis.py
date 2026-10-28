"""EXTRACT THE STAGE-1 BASIS (2026-10-28): the committed inputs behind `agents.model.canary_row_bar` (k 6.0) and
`compile_regions.BF16_ENVELOPE_K["loss"]` (9.2), so the routine tests re-derive both and backtest the rule with no
access to the W0 census or `models/`.

It reuses `backtest.py`'s loaders (imported, never run: that script is the frozen record of the DRAFT and refuses at
any pin past the ruling, because its `check_constants` pins the pre-ruling bars). CPU, read-only on its inputs, seconds:

    python extract_stage1_basis.py --census ~/.claude/jobs/x35/w0_2026-10-28/census \\
        --xcheck ~/.claude/jobs/x35/w0_2026-10-28/xcheck --models <main checkout>/models \\
        --out <checkout>/src/agents/model/testdata/canary_stage1_basis.json

What it writes:
  * ``census``: per W0 checkpoint, the 16-seed fp32 dither null per fixture row and the compiled (Cbf) and eager (Ebf)
    bf16 shifts at B = 2,048 on CUDA (7 significant digits);
  * ``production_loss_envelope``: envelope_loss / max(drift_loss, floor) of every bf16 check (scheduled and each
    confirmation half) of every X35 and P*-lineage record;
  * ``anchors_loss_envelope``: the same ratio on the X-CHECK anchors' three slices (repeat 0), the out-of-sample test.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _backtest():
    spec = importlib.util.spec_from_file_location("canary_backtest_2026_10_28", HERE / "backtest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _g(x: float) -> float:
    """7 significant digits: the census values are float32 differences, so nothing below is signal."""
    return float(f"{float(x):.7g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--xcheck", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    bt = _backtest()
    cs = bt.load_census(a.census)
    census = [{"label": e["label"], "step": e["step"], "null_k16": [_g(x) for x in bt.null_k(e, 16)],
               "Cbf": [_g(x) for x in e["arms"]["Cbf"]], "Ebf": [_g(x) for x in e["arms"]["Ebf"]]}
              for e in cs]
    prod = []
    for r in bt.load_records(a.models):
        if r["_group"] not in ("x35", "pstar_lineage"):
            continue
        for chk, w in (r.get("watch") or {}).items():
            if isinstance(w, dict) and w.get("rule") == "bf16":
                v = bt.loss_envelope_ratio(w)
                if v is not None:
                    prod.append({"run": r["_run"], "update": r.get("update"), "verdict": r.get("verdict"),
                                 "check": chk, "ratio": _g(v)})
    anchors = []
    for d in sorted(p for p in a.xcheck.iterdir() if p.is_dir() and (p / "repeat_00.json").exists()):
        rep = json.load(open(d / "repeat_00.json"))
        for s, v in rep["slices"].items():
            r = bt.loss_envelope_ratio(v["canary"]["watch"])
            if r is not None:
                anchors.append({"anchor": d.name, "slice": int(s), "ratio": _g(r)})
    out = {"provenance": {"written": datetime.date.today().isoformat(),
                          "by": "designs/research_state/measurements/canary_recalibration_2026-10-28/"
                                "extract_stage1_basis.py",
                          "census": "W0 CUDA census (RTX 3080 Ti, torch 2.8.0+cu126), 20 X26-lineage checkpoints",
                          "records": "canary_verdicts.jsonl of every rb_x35*, rb_x26_s2601, rb_x26c_s2601 run",
                          "anchors": "main.ops.canary_dump_replay --anchor, repeat 0"},
           "census": census, "production_loss_envelope": prod, "anchors_loss_envelope": anchors}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(f" {json.dumps(k)}: " + ("[\n" + ",\n".join("  " + json.dumps(x) for x in v) + "\n ]"
                                                 if isinstance(v, list) else json.dumps(v)) for k, v in out.items())
    a.out.write_text("{\n" + body + "\n}\n")
    print(f"{len(census)} census checkpoints, {len(prod)} production checks (max loss ratio "
          f"{max(p['ratio'] for p in prod):.4f}), {len(anchors)} anchor slices -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
