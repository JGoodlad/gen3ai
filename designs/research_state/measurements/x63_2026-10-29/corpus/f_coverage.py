"""X63 review M3's re-measure: how much of each Kakuna panel's mon sets the union's components (and the pool) carry.

For each panel (forum72 = F, heldout72_hpfix = H, p72 = P72, home72) and each source (the pool's 719 teams, each union
component, the whole union): the share of the panel's mon SLOTS whose set (species + sorted moves, Hidden Power typed:
`main.anchors.team_overlap` MonSet.key, the matcher every build uses) occurs in the source, and the same at the strict
key (+ item, ability, nature, EVs). Deterministic, CPU, seconds:
    PYTHONPATH=src python designs/research_state/measurements/x63_2026-10-29/corpus/f_coverage.py > .../f_coverage.log
"""
import json

from main.anchors.team_overlap import parse_team
from utils.team_corpora import load_corpus
from utils.team_corpora.build_x62_recomb import panel_items


def keys(texts):
    k, s = set(), set()
    for t in texts:
        for m in parse_team(t).mons:
            k.add(m.key)
            s.add(m.strict_key)
    return k, s


def main():
    from main import heldout_panels as HPN
    from main.anchors.session import pool_team_texts
    pool = list(pool_team_texts())
    srcs = {f"pool ({len(pool)})": pool}
    for n, t in (("x62_recomb_v1", "ALL"), ("n1_proc_v1", "K3"), ("n1_proc_v1", "ALL"), ("d3_setmut_v1", "ALL"),
                 ("x63_union_v1", "ALL")):
        srcs[f"{n}@{t}"] = list(load_corpus(n, t).texts)
    skeys = {n: keys(v) for n, v in srcs.items()}
    panels = panel_items()
    try:
        panels["home72"] = list(HPN.panel_team_texts("home72"))
    except Exception as exc:  # noqa: BLE001 -- home72 may not be a registry panel; say so
        print(f"# home72 not loadable through the registry: {exc}")
    out = {}
    for p in ("forum72", "heldout72_hpfix", "p72", "home72"):
        if p not in panels:
            continue
        mons = [m for _, t in panels[p] for m in parse_team(t).mons]
        row = {}
        for n, (k, s) in skeys.items():
            row[n] = {"slots": len(mons), "set_share": round(sum(m.key in k for m in mons) / len(mons), 4),
                      "strict_share": round(sum(m.strict_key in s for m in mons) / len(mons), 4)}
        out[p] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
