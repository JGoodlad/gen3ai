"""X48 PREREG NOTE 1 (a): VOID 4 made exact, checked by a script at arm end (never by hand), before any read game.

For each X48 run, a seed is VOID unless ALL hold (the values MEASURED on P* at the pin, REVIEW §1):
- ``model_config.json`` ``growth`` AND ``metadata.json`` ``lineage.grow`` (the two stamps) are equal and each has
  ``fields.trunk_ffn_by_kind`` == {parent ``off``, child ``on``}, ``optimizer_state`` == {carried 280, no_state 0,
  fresh 0, copied 48}, ``new_parameters`` == {tensors 48, elements 791,040}, ``function_check.bit_identical`` true and
  ``max_abs_d_logits`` == ``max_abs_d_value`` == 0.0 (exact: rule 8 has no boundary here);
- the run's ``model_config.json`` ``trunk_ffn_by_kind`` is ``on`` and its growth's ``parent_model`` is P*;
- no launcher / child log in the run dir holds ``[GrowFork] FATAL``, ``[ModelVersion] FATAL`` or ``does NOT inherit``
  (the chain's ``banner.void``), and the FULL child log (``launcher_child.full.log``; ``launcher_child.log`` is a
  ~1 MiB ring buffer whose head is trimmed) holds the expected banner ``fresh for 0, COPIED for 48``.
Also printed (VOID 1, for the record): ``final_model.zip`` exists and ``metadata.json`` ``num_timesteps`` >= 300,000,170.

Read-only on ``models/``. Prints one JSON document; exit 0 iff every run passes, else 1.

    python designs/research_state/measurements/x48_kffn_production_2026-10-27/void4_growth_stamp.py \
        models/rb_x35_x48kffn_s3501 models/rb_x35_x48kffn_s3502 models/rb_x35_x48kffn_s3503
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

P_STAR = "rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"
WANT = {
    "fields": {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}},
    "optimizer_state": {"carried": 280, "no_state": 0, "fresh": 0, "copied": 48},
    "new_parameters": {"tensors": 48, "elements": 791040},
}
WANT_CHECK = {"bit_identical": True, "max_abs_d_logits": 0.0, "max_abs_d_value": 0.0}
VOID_LINES = ("[GrowFork] FATAL", "[ModelVersion] FATAL", "does NOT inherit")
BANNER = "fresh for 0, COPIED for 48"
MIN_STEPS = 300_000_170


def _stamp_problems(g: Any, where: str) -> List[str]:
    if not isinstance(g, dict):
        return [f"{where}: no growth stamp"]
    probs = [f"{where}.{k}: {g.get(k)!r} != {v!r}" for k, v in WANT.items() if g.get(k) != v]
    fc = g.get("function_check") or {}
    probs += [f"{where}.function_check.{k}: {fc.get(k)!r} != {v!r}" for k, v in WANT_CHECK.items()
              if fc.get(k) != v or type(fc.get(k)) is not type(v)]
    if not str(g.get("parent_model", "")).endswith(P_STAR):
        probs.append(f"{where}.parent_model {g.get('parent_model')!r} is not P*")
    return probs


def check(run: Path) -> Dict[str, Any]:
    cfg = json.loads((run / "model_config.json").read_text())
    meta = json.loads((run / "metadata.json").read_text())
    g_cfg = cfg.get("growth")
    g_lin = (meta.get("lineage") or {}).get("grow")
    probs = _stamp_problems(g_cfg, "model_config.growth") + _stamp_problems(g_lin, "lineage.grow")
    if g_cfg != g_lin:
        probs.append("model_config.growth != lineage.grow")
    if cfg.get("trunk_ffn_by_kind") != "on":
        probs.append(f"model_config.trunk_ffn_by_kind {cfg.get('trunk_ffn_by_kind')!r} != 'on'")
    logs = sorted(p for p in run.glob("*.log"))
    void_hits = {p.name: [ln for ln in VOID_LINES if ln in p.read_text(errors="replace")] for p in logs}
    void_hits = {k: v for k, v in void_hits.items() if v}
    if void_hits:
        probs.append(f"void lines in logs: {void_hits}")
    banner = {p.name: sum(1 for line in p.read_text(errors="replace").splitlines() if BANNER in line) for p in logs}
    if banner.get("launcher_child.full.log") != 1:
        probs.append(f"launcher_child.full.log holds {banner.get('launcher_child.full.log')} {BANNER!r} banners, not 1")
    steps = int(meta.get("num_timesteps") or 0)
    final = (run / "final_model.zip").is_file()
    void1 = [] if final and steps >= MIN_STEPS else [f"VOID 1: final {final}, num_timesteps {steps}"]
    return {"run": run.name, "pass": not probs and not void1, "problems": probs + void1,
            "growth": g_cfg, "lineage_grow_equal": g_cfg == g_lin, "logs_scanned": [p.name for p in logs],
            "banner_lines": banner, "num_timesteps": steps, "final_model": final}


def main(argv: List[str]) -> int:
    rows = [check(Path(a)) for a in argv]
    out = {"schema": "x48_void4_growth_stamp_v1", "want": dict(WANT, function_check=WANT_CHECK),
           "void_lines": list(VOID_LINES), "banner": BANNER, "runs": rows, "all_pass": all(r["pass"] for r in rows)}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if rows and out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
