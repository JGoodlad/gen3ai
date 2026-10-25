"""Tabulate margin_read.py's per-checkpoint JSONs (read_<label>.json) into one table: `python tabulate.py <dir>`."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    d = Path(sys.argv[1])
    recs = sorted((json.loads(p.read_text()) for p in d.glob("read_*.json")), key=lambda r: float(r["label"][:-1]))
    print("| ckpt | max dlogp Ebf (row) | median | trunk latent rel err median / max | pi_features median / max "
          "| corr(dlogp, 1/top2) P / S | fp32 dither max (row) | Ebf dither max (row) | row 50: dlogp, top2, H, p_taken "
          "| row 31: dlogp, top2, H, p_taken | logit spread median | latent norm median |")
    print("|" + "---|" * 12)
    for r in recs:
        s, c, dd = r["summary"], r["corr"], r.get("dither", {})
        ro = r["rows_of_interest"]

        def row(k: str) -> str:
            x = ro[k]
            return f"{x['abs_dlogp_Ebf']:.3f}, {x['top2_margin']:.3f}, {x['entropy']:.2f}, {x['p_taken']:.2g}"
        print(f"| {r['label']} | {s['abs_dlogp_Ebf']['max']:.3f} ({s['abs_dlogp_Ebf']['argmax']}) "
              f"| {s['abs_dlogp_Ebf']['median']:.4f} "
              f"| {s['trunk_rel_latent_pi']['median']:.4f} / {s['trunk_rel_latent_pi']['max']:.4f} "
              f"| {s['trunk_rel_pi_features']['median']:.4f} / {s['trunk_rel_pi_features']['max']:.4f} "
              f"| {c['dlogp_vs_inv_top2']['pearson']:+.2f} / {c['dlogp_vs_inv_top2']['spearman']:+.2f} "
              f"| {dd['E32~_max']['max']:.3f} ({dd['E32~_max']['argmax']}) | {dd['Ebf~_max']['max']:.3f} "
              f"({dd['Ebf~_max']['argmax']}) | {row('50')} | {row('31')} | {s['legal_logit_spread']['median']:.2f} "
              f"| {s['latent_pi_norm_median']:.1f} |")
    print()
    for r in recs:
        dd = r.get("dither", {})
        print(r["label"], "row50 E32~", [round(x, 3) for x in dd["rows"]["50"]["E32~"]], "Ebf~",
              [round(x, 3) for x in dd["rows"]["50"]["Ebf~"]])
        print("   corr", {k: round(v.get("spearman", 0), 2) for k, v in r["corr"].items()},
              "Ebf_vs_fp32dither", round(dd["corr_Ebf_dlogp_vs_E32dither_sensitivity"]["spearman"], 2),
              "trace_E32_vs_Ebf rows changed", r["trace_E32_vs_Ebf"].get("rows_with_any_change"),
              "det", r["determinism_E32_rerun_max"], "micro", r["E32_vs_micro_step_max"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
