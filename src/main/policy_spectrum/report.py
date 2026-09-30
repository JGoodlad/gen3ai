"""The TREND report: several reads of the same bank, side by side (markdown + JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from main.policy_spectrum import spectrum as S
from main.policy_spectrum.bank import Bank
from main.policy_spectrum.categories import CATEGORIES
from main.policy_spectrum.reader import load_probs


def _fmt(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _ci(ci) -> str:
    return "" if not ci else f" [{ci[0]:.3f}, {ci[1]:.3f}]"


def load_reads(reads_dir: Path, labels: Sequence[str]) -> List[dict]:
    out = []
    for lab in labels:
        safe = lab.replace("/", "_").replace(" ", "_")
        out.append(json.loads((reads_dir / f"{safe}.json").read_text()))
    return out


def spectrum_table(reads: Sequence[dict], stratum: str, key: str) -> List[str]:
    lines = ["| policy | n | rank 1 | rank 2 | rank 3 | ranks 4+ | entropy (nats) | eff. choices | top1 ≥ 0.9 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in reads:
        a = r["strata"][stratum].get(key, {"n": 0})
        if not a.get("n"):
            lines.append(f"| {r['label']} | 0 | | | | | | | |")
            continue
        sp = a["spectrum"]
        rest = sum(sp[3:]) + a["tail"]
        lines.append(f"| {r['label']} | {a['n']} | {sp[0]:.3f}{_ci(a['ci']['rank1'])} | "
                     f"{sp[1]:.3f} | {sp[2]:.3f} | {rest:.3f} | {a['entropy']:.3f} | "
                     f"{a['eff_choices']:.2f} | {a['top1_ge_0.9']:.3f} |")
    return lines


def category_table(reads: Sequence[dict], field: str, block: str = "categories_free",
                   phase: Optional[str] = None) -> List[str]:
    cats = [c for c in CATEGORIES]
    lines = ["| policy | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for r in reads:
        blk = r[block] if phase is None else r[block].get(phase, {})
        cells = []
        for c in cats:
            v = blk.get(c, {}).get(field)
            cells.append(_fmt(v))
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    ns = [str(reads[0][block if phase is None else block].get(c, {}).get("n", 0)) if phase is None
          else str(reads[0][block].get(phase, {}).get(c, {}).get("n", 0)) for c in cats]
    lines.append("| *(decisions where legal)* | " + " | ".join(ns) + " |")
    return lines


def rank_category_table(reads: Sequence[dict], rank: str) -> List[str]:
    cats = [c for c in CATEGORIES]
    lines = ["| policy | " + " | ".join(f"{c} share / mass" for c in cats) + " |",
             "|---|" + "---|" * len(cats)]
    for r in reads:
        row = r["rank_category_free"][rank]
        lines.append(f"| {r['label']} | " + " | ".join(
            f"{row[c]['share']:.3f} / {_fmt(row[c]['mass'])}" for c in cats) + " |")
    return lines


def paired_rows(bank: Bank, reads_dir: Path, pairs: Sequence[Tuple[str, str]]) -> List[dict]:
    idx_free = np.array([i for i, d in enumerate(bank.decisions) if d["kind"] == "free"])
    idx_forced = np.array([i for i, d in enumerate(bank.decisions) if d["kind"] == "forced_switch"])
    out = []
    for a, b in pairs:
        pa, pb = load_probs(reads_dir, a, bank), load_probs(reads_dir, b, bank)
        out.append({"a": a, "b": b,
                    "all": S.paired_delta(pa, pb, bank.decisions),
                    "free": S.paired_delta(pa, pb, bank.decisions, idx_free),
                    "forced_switch": S.paired_delta(pa, pb, bank.decisions, idx_forced)})
    return out


def render(bank: Bank, reads: Sequence[dict], paired: Sequence[dict], title: str,
           preamble: str = "") -> str:
    L: List[str] = [f"# {title}", ""]
    if preamble:
        L += [preamble, ""]
    m = bank.manifest
    L += [f"Bank `{m['content_sha256'][:12]}`: {m['counts']['decisions']:,} decisions in "
          f"{m['counts']['battles']} battles; reader commit(s): "
          f"{', '.join(sorted({r['reader_commit'][:8] for r in reads}))}; every read "
          f"{'ran on the RECORDED observations (re-encode byte-equal)' if all(r['reencode']['obs_as_recorded'] for r in reads) else 'did NOT run on byte-equal observations — see each JSON'}.",
          "", "Spectrum = mean probability on the policy's own 1st / 2nd / 3rd … choice; "
          "[ ] = 95 % battle-clustered bootstrap.", ""]
    L += ["## All decisions", ""] + spectrum_table(reads, "all", "all") + [""]
    L += ["## Free-choice turns", ""] + spectrum_table(reads, "kind", "free") + [""]
    L += ["## Forced switches", ""] + spectrum_table(reads, "kind", "forced_switch") + [""]
    for ph in ("opening", "midgame", "endgame"):
        L += [f"## Free turns, {ph}", ""] + spectrum_table(reads, "free_by_phase", ph) + [""]
    for oc in ("bot", "pool_snapshot", "exploiter"):
        L += [f"## Free turns vs {oc}", ""] + spectrum_table(reads, "free_by_opp_class", oc) + [""]
    L += ["## Free turns where the category is legal: spectrum", ""]
    for c in ("status", "setup", "hazard", "recovery", "switch"):
        L += [f"### {c} legal", ""] + spectrum_table(reads, "free_cat_legal", c) + [""]
    L += ["## Mass on each category (free turns where it is legal)", ""] + category_table(reads, "mass") + [""]
    L += ["## Share of free turns where a LEGAL category holds < 1 % of the mass", "",
          "A descriptor, not starvation: whether that category deserved mass needs gate ④'s ground truth.", ""]
    L += category_table(reads, "share_lt_1pct") + [""]
    L += ["## Share of free turns where the policy's top choice is the category", ""]
    L += category_table(reads, "share_top1") + [""]
    for rk in ("rank1", "rank2", "rank3"):
        L += [f"## The policy's {rk[-1]}{'st' if rk == 'rank1' else 'nd' if rk == 'rank2' else 'rd'} choice by category (free turns): share of turns / mean mass", ""]
        L += rank_category_table(reads, rk) + [""]
    if paired:
        L += ["## Paired Δ on the same decisions (b − a)", "",
              "| a → b | turns | Δ rank 1 | Δ rank 2 | Δ rank 3 | Δ entropy |", "|---|---|---|---|---|---|"]
        for p in paired:
            for part in ("all", "free", "forced_switch"):
                q = p[part]
                L.append(f"| {p['a']} → {p['b']} | {part} (n={q['n']}) | "
                         + " | ".join(f"{q[k]['delta']:+.3f}{_ci(q[k]['ci'])}"
                                      for k in ("rank1", "rank2", "rank3", "entropy")) + " |")
        L.append("")
    return "\n".join(L)
