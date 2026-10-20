"""L5, Long et al.'s (AAAI 2010) three PIMC properties measured on the banked rows (`../PREREG.md` §6). DESCRIPTIVE:
no rule, no verdict. Pure numpy over the cache `l1l5_data.py build` writes.

    python -I l5_props.py        → ../result_l5.json

Long et al.'s definitions (VERIFIED in the paper, p. 136 and "Real Games", p. 139):
  * leaf correlation lc: "the probability all sibling, terminal nodes have the same payoff value"; in real games,
    measured at "pre-terminal" nodes (every move leads directly to a terminal position): "correlated if all move values
    are the same";
  * bias b: "the fraction of correlated nodes which are soloist wins";
  * disambiguation df: "comparing the change in the number of possible (consistent) worlds since the current player was
    last to move"; in the synthetic model each information set is halved with probability df each time the player
    moves (recursively).

OUR ANALOGUES (PREREG §6; each a proxy, said so):
  lc, b   on the TRUTH rows (Phase 1: every legal action played out in the TRUE world on 40 dice seeds, the network
          playing both sides). A bank state is PRE-TERMINAL when every action's every playout ends in the CURRENT turn
          (end turn == the decision's turn). It is CORRELATED when every (action, seed) outcome is the same; lc = the
          share of pre-terminal states that are correlated; b = the share of correlated states won by the searched
          side. A dice-paired variant lc_seed = the mean over seeds of [every action's outcome on that seed is the
          same]. The same two quantities are reported for every state by the turns its playouts take to end
          (a gradient, not Long's lc).
  df      on the X26 interim bank's 513 battles (the traced side's own protocol view): per own turn t with hidden facts
          left, df_event = P(at least one new TRUE opponent fact is revealed during turn t) — in Long's generative model
          the probability of at least one halving per move IS df — and the mean revealed fraction R_t / H_t.
          Facts: the opponent's species and the moves of its true sets (primary); + items and abilities (secondary).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "result_l5.json"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1] / "hidden_info_tier1_2026-10-13"))
import t1_stats as ST  # noqa: E402

GAP_BINS = ((0, 0, "0 (ends this turn)"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 10 ** 6, ">=8"))


def norm_move(m: str) -> str:
    m = "".join(ch for ch in str(m).lower() if ch.isalnum())
    return "hiddenpower" if m.startswith("hiddenpower") else m


def state_props(r: dict) -> dict:
    Ob = np.asarray(r["Ob40"], dtype=np.int64)              # [A, 40] in {-1, 0, +1}
    end = r.get("end40")
    gap_max = int((np.asarray(end) - r["turn0"]).max()) if end is not None else None
    gap_med = float(np.median(np.asarray(end) - r["turn0"])) if end is not None else None
    corr = bool((Ob == Ob.flat[0]).all())
    seed_same = (Ob == Ob[0:1, :]).all(0)                  # [40]
    return {"pre_terminal": gap_max == 0 if gap_max is not None else None, "gap_max": gap_max, "gap_med": gap_med,
            "correlated": corr, "corr_payoff": int(Ob.flat[0]) if corr else None,
            "lc_seed": float(seed_same.mean()),
            "seed_same_win_share": float((seed_same & (Ob[0] == 1)).sum() / max(1, seed_same.sum())),
            "leaf_mean": float(((Ob + 1) / 2).mean())}


def leaf_props(recs: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    groups = {"all banks": recs}
    for bk in ("tier1", "e2", "x26"):
        groups[bk] = [r for r in recs if r["bank"] == bk]
    for name, rs in groups.items():
        P = [state_props(r) for r in rs]
        pre = [p for p in P if p["pre_terminal"]]
        cor = [p for p in pre if p["correlated"]]
        g = {"states": len(P), "with_end_turns": sum(p["gap_max"] is not None for p in P),
             "pre_terminal": len(pre),
             "lc (Long: share of pre-terminal states whose every action and seed gives one payoff)":
                 _share(len(cor), len(pre)),
             "b (Long: share of correlated pre-terminal states won by the searched side)":
                 _share(sum(p["corr_payoff"] == 1 for p in cor), len(cor)),
             "draw_share_of_correlated": _share(sum(p["corr_payoff"] == 0 for p in cor), len(cor)),
             "lc_seed (pre-terminal; dice-paired)": _mean([p["lc_seed"] for p in pre]),
             "leaf_win_rate (all states)": _mean([p["leaf_mean"] for p in P]),
             "by_turns_to_end (median over playouts)": {}}
        for lo, hi, lab in GAP_BINS:
            sel = [p for p in P if p["gap_med"] is not None and lo <= int(p["gap_med"]) <= hi]
            g["by_turns_to_end (median over playouts)"][lab] = {
                "n": len(sel), "all_outcomes_equal": _share(sum(p["correlated"] for p in sel), len(sel)),
                "lc_seed": _mean([p["lc_seed"] for p in sel]),
                "win_share_of_equal_seeds": _mean([p["seed_same_win_share"] for p in sel])}
        out[name] = g
    return out


def _share(a: int, n: int):
    return None if n == 0 else round(a / n, 4)


def _mean(xs):
    return None if not xs else round(float(np.mean(xs)), 4)


def disambiguation(rev: List[dict]) -> dict:
    out = {}
    for label, kinds in (("species+moves (primary)", ("species", "move")),
                         ("species+moves+items+abilities", ("species", "move", "item", "ability"))):
        rows = []                                   # (battle, stratum, turn, H, R)
        for b in rev:
            facts = set()
            for m in b["team"]:
                s = m["species"]
                facts.add(("species", s))
                if "move" in kinds:
                    facts |= {("move", s, norm_move(x)) for x in m["moves"]}
                if "item" in kinds and m.get("item"):
                    facts.add(("item", s))
                if "ability" in kinds and m.get("ability"):
                    facts.add(("ability", s))
            known = set()
            by_turn: Dict[int, set] = {}
            for (t, kind, actor, val) in b["events"]:
                if kind not in kinds or actor is None:
                    continue
                f = ("species", actor) if kind == "species" else (
                    ("move", actor, norm_move(val)) if kind == "move" else (kind, actor))
                if f in facts:
                    by_turn.setdefault(int(t), set()).add(f)
            known |= by_turn.get(0, set())
            T = max([int(e[0]) for e in b["events"]] + [0])
            for t in range(1, T):                   # own decisions at turns 1..T-1 have a next decision
                H = len(facts - known)
                new = by_turn.get(t, set()) - known
                if H > 0:
                    rows.append((b["battle"], b["stratum"], t, H, len(new)))
                known |= new
        if not rows:
            out[label] = {"turns": 0}
            continue
        bat = [x[0] for x in rows]
        ev = np.array([x[4] > 0 for x in rows], dtype=float)
        frac = np.array([x[4] / x[3] for x in rows])
        bt = ST.Boot(bat)
        res = {"battles": len(set(bat)), "turns": len(rows), "df_event": _s(bt, ev), "revealed_fraction": _s(bt, frac)}
        res["long_model_halvings_per_move_at_df_event (df / (1 - df); derived, not in the paper)"] = round(
            float(ev.mean() / (1 - ev.mean())), 3) if ev.mean() < 1 else None
        for st in ("bot", "snapshot"):
            m = np.array([x[1] == st for x in rows])
            res[f"stratum {st}"] = {"df_event": _s(bt, ev, m), "revealed_fraction": _s(bt, frac, m)}
        for lo, hi in ((1, 7), (8, 14), (15, 21), (22, 10 ** 6)):
            m = np.array([lo <= x[2] <= hi for x in rows])
            res[f"turn {lo}-{hi if hi < 10 ** 6 else ''}"] = {"df_event": _s(bt, ev, m),
                                                              "revealed_fraction": _s(bt, frac, m)}
        out[label] = res
    return out


def _s(bt, v, m=None) -> dict:
    est, vec = bt.mean(v, m)
    lo, hi = ST.ci(vec)
    return {"mean": round(est, 4), "ci95": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)],
            "n": int(len(v) if m is None else m.sum())}


def main() -> None:
    import l1l5_data as L

    cache = L.load_cache()
    res = {"status": "L5, DESCRIPTIVE (PREREG §6): Long et al. 2010's properties on the banked rows",
           "leaf_correlation_and_bias": leaf_props(cache["recs"]),
           "disambiguation (X26 interim-bank battles)": disambiguation(cache["reveals"]),
           "long_2010_reference": {"skat_and_hearts": "correlation 0.8 to nearly 1.0, bias varying widely, df very close "
                                                      "to 0.6 (p. 139, Fig. 6)"}}
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
