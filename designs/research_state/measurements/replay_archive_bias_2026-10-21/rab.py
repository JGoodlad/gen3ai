"""Replay-archive bias: archive-only windows vs neighbouring spectator days (2026-10-21). CPU only, read-only on the corpus.

    python rab.py extract --cache <dir> [--workers 3] [--windows W1,W2,W3,...]   # one jsonl.gz per date folder, resumable
    python rab.py analyze --cache <dir> --out <dir>                              # bootstrap tables -> results.json / results.md

The registered design (windows, bars, verdict rule) is the README's first section; the constants below restate it.
Pure text pass over each log (header ratings, last turn, ending message, switch-ins, Spikes): no Rust reader needed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CORPUS = Path(os.environ.get("RAB_CORPUS", "/home/goodlad/dev/gen3ai/replays/showdown/gen3ou"))


def days(a: str, b: str) -> List[str]:
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return [(d0 + dt.timedelta(n)).isoformat() for n in range((d1 - d0).days + 1)]


#: name -> (archive-only days, spectator comparator days [before], [after]); the README registration.
WINDOWS = {
    "W1": (days("2026-06-16", "2026-06-20"), days("2026-06-10", "2026-06-14"), days("2026-06-22", "2026-06-26")),
    "W2": (days("2026-08-03", "2026-08-17"), days("2026-07-19", "2026-08-01"), days("2026-08-19", "2026-09-01")),
    "W3": (days("2026-09-24", "2026-10-20"), days("2026-08-27", "2026-09-22"), []),
}
BOOT = 2000
SEED = 20261021
TOP_SPECIES, TOP_LEADS = 30, 10

_PLAYER = re.compile(r"^\|player\|(p[12])\|([^|]*)\|([^|]*)\|(\d*)")
_SWITCH = re.compile(r"^\|(?:switch|drag)\|(p[12])[ab]?: [^|]*\|([^|,]+)")
_FORFEIT = re.compile(r"^\|-message\|(.*) forfeited\.$")
_TIMER = re.compile(r"^\|-message\|(.*) lost due to inactivity\.$")


def parse(text: str) -> Optional[dict]:
    """One battle -> a small record (None if it is not a battle log). Lines common to archive and spectator logs only."""
    rated = False
    rating: Dict[str, Optional[int]] = {"p1": None, "p2": None}
    names: Dict[str, str] = {}
    turns = 0
    end, winner = "none", None
    species: Dict[str, List[str]] = {"p1": [], "p2": []}
    spikes = False
    forfeiter = None
    for ln in text.split("\n"):
        if not ln.startswith("|"):
            continue
        if ln.startswith("|move|"):
            if ln.split("|")[3:4] == ["Spikes"]:
                spikes = True
        elif ln.startswith(("|switch|", "|drag|")):
            m = _SWITCH.match(ln)
            if m and m.group(2) not in species[m.group(1)]:
                species[m.group(1)].append(m.group(2))
        elif ln.startswith("|turn|"):
            try:
                turns = int(ln[6:])
            except ValueError:
                pass
        elif ln.startswith("|player|"):
            m = _PLAYER.match(ln)
            if m and m.group(2) and m.group(2) not in names.values() and m.group(1) not in names:
                names[m.group(1)] = m.group(2)
                rating[m.group(1)] = int(m.group(4)) if m.group(4) else None
        elif ln == "|rated|" or ln.startswith("|rated|"):
            rated = True
        elif ln.startswith("|-message|"):
            m = _FORFEIT.match(ln)
            if m:
                forfeiter, end = m.group(1), "forfeit"
            else:
                m = _TIMER.match(ln)
                if m:
                    forfeiter, end = m.group(1), "timer"
        elif ln.startswith("|win|"):
            winner = ln[5:]
            if end == "none":
                end = "ko"
            break
        elif ln == "|tie":
            end = "tie"
            break
    if not names:
        return None
    if end in ("forfeit", "timer") and winner is None:
        end = "none"  # a forfeit message without a result line (the room died): no result
    win_seat = next((s for s, n in names.items() if n == winner), None) if winner else None
    return {"rated": rated, "r1": rating["p1"], "r2": rating["p2"], "turns": turns, "end": end, "win": win_seat,
            "sp1": species["p1"], "sp2": species["p2"], "spikes": spikes,
            "n1": names.get("p1"), "n2": names.get("p2")}


def extract_day(args) -> str:
    date, cache = args
    out = Path(cache) / f"{date}.jsonl.gz"
    if out.exists():
        return f"{date} cached"
    d = CORPUS / date
    rows = []
    for p in sorted(d.iterdir()):
        try:
            rec = parse(p.read_text(errors="replace"))
        except OSError:
            continue
        if rec is not None:
            rec["id"] = p.stem
            rows.append(rec)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(out)
    return f"{date} {len(rows)}"


def all_dates(names: List[str]) -> List[str]:
    out: List[str] = []
    for w in names:
        for part in WINDOWS[w]:
            out += part
    return sorted(set(out))


def cmd_extract(a) -> None:
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    names = a.windows.split(",")
    todo = [(d, a.cache) for d in all_dates(names)]
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(extract_day, todo):
            print(msg, flush=True)


# ---------------------------------------------------------------------------------------------------- analysis
def load(cache: str, dates: List[str]) -> List[dict]:
    rows: List[dict] = []
    for d in dates:
        with gzip.open(Path(cache) / f"{d}.jsonl.gz", "rt") as f:
            rows += [json.loads(x) for x in f]
    return rows


def features(rows: List[dict], species: List[str], leads: List[str]) -> Dict[str, np.ndarray]:
    """Per-battle columns (NaN = not defined for the battle); every statistic is a NaN-aware mean of one column."""
    n = len(rows)
    nan = np.full(n, np.nan)
    f: Dict[str, np.ndarray] = {}
    done = np.array([r["end"] != "none" for r in rows])
    f["no_result"] = (~done).astype(float)
    rated = np.array([r["rated"] for r in rows], float)
    f["rated"] = rated
    r1 = np.array([np.nan if r["r1"] is None else r["r1"] for r in rows], float)
    r2 = np.array([np.nan if r["r2"] is None else r["r2"] for r in rows], float)
    both = (rated == 1) & ~np.isnan(r1) & ~np.isnan(r2) & done
    f["rating_pair_mean"] = np.where(both, (r1 + r2) / 2, nan)
    f["rating_high"] = np.where(both, np.maximum(r1, r2), nan)
    f["rating_low"] = np.where(both, np.minimum(r1, r2), nan)
    t = np.array([r["turns"] for r in rows], float)
    f["log_turns"] = np.where(done & (t > 0), np.log(np.maximum(t, 1)), nan)
    for e in ("forfeit", "timer", "ko", "tie"):
        f["end_" + e] = np.where(done, np.array([r["end"] == e for r in rows], float), nan)
    w1 = np.array([r["win"] == "p1" for r in rows], float)
    haswin = np.array([r["win"] in ("p1", "p2") for r in rows])
    f["seat1_win"] = np.where(haswin, w1, nan)
    unequal = both & haswin & (r1 != r2)
    f["higher_rated_won"] = np.where(unequal, np.where(r1 > r2, w1, 1 - w1), nan)
    f["spikes"] = np.where(done, np.array([r["spikes"] for r in rows], float), nan)
    for s in species:
        f["sp:" + s] = np.where(done, np.array([((s in r["sp1"]) + (s in r["sp2"])) / 2 for r in rows], float), nan)
    for s in leads:
        f["lead:" + s] = np.where(done, np.array(
            [((bool(r["sp1"]) and r["sp1"][0] == s) + (bool(r["sp2"]) and r["sp2"][0] == s)) / 2 for r in rows], float), nan)
    return f


def top_names(rows: List[dict], k: int, lead: bool) -> List[str]:
    c: Dict[str, int] = {}
    for r in rows:
        if r["end"] == "none":
            continue
        for key in ("sp1", "sp2"):
            xs = r[key][:1] if lead else r[key]
            for s in xs:
                c[s] = c.get(s, 0) + 1
    return [s for s, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def boot_means(f: Dict[str, np.ndarray], rng: np.random.Generator, reps: int) -> Dict[str, np.ndarray]:
    """Battle-level bootstrap of every column's NaN-aware mean at once: multinomial resample counts @ the column matrix."""
    names = list(f)
    X = np.stack([f[k] for k in names], 1)
    M = (~np.isnan(X)).astype(np.float32)
    X0 = np.nan_to_num(X).astype(np.float32)
    n = len(X)
    out = np.empty((reps, len(names)))
    p = np.full(n, 1.0 / n)
    for i in range(0, reps, 100):
        m = min(100, reps - i)
        C = rng.multinomial(n, p, size=m).astype(np.float32)
        num, den = (C @ X0).astype(np.float64), (C @ M).astype(np.float64)
        out[i:i + m] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return {k: out[:, j] for j, k in enumerate(names)}


#: dimension -> (bar, scale to report in, interval level, unit label). Log-turns: bar 5 % = ln(1.05).
def spec(name: str):
    if name == "rated":
        return 0.03, 0.95
    if name.startswith("rating_"):
        return 20.0, 0.95
    if name == "log_turns":
        return float(np.log(1.05)), 0.95
    if name.startswith("end_") or name in ("seat1_win", "higher_rated_won", "spikes"):
        return 0.02, 0.95
    if name.startswith("sp:"):
        return 0.015, 0.998
    if name.startswith("lead:"):
        return 0.015, 0.995
    return None, 0.95  # no_result: reported, not judged


def compare(fa: Dict[str, np.ndarray], fb: Dict[str, np.ndarray], seed: int) -> Dict[str, dict]:
    rng = np.random.default_rng(seed)
    ba, bb = boot_means(fa, rng, BOOT), boot_means(fb, rng, BOOT)
    out = {}
    for name in fa:
        bar, level = spec(name)
        na, nb = int((~np.isnan(fa[name])).sum()), int((~np.isnan(fb[name])).sum())
        est = float(np.nanmean(fa[name]) - np.nanmean(fb[name]))
        d = ba[name] - bb[name]
        if min(na, nb) < 30 or np.isnan(d).any():
            lo = hi = float("nan")
        else:
            lo, hi = (float(x) for x in np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2]))
        if bar is None or np.isnan(lo):
            verdict = "n/a"
        elif -bar <= lo and hi <= bar:
            verdict = "within"
        elif lo > bar or hi < -bar:
            verdict = "OUTSIDE"
        else:
            verdict = "straddles"
        out[name] = {"diff": est, "lo": lo, "hi": hi, "bar": bar, "level": level, "verdict": verdict,
                     "n_a": na, "n_b": nb, "mean_a": float(np.nanmean(fa[name])), "mean_b": float(np.nanmean(fb[name]))}
    return out


def tvd(fa: Dict[str, np.ndarray], fb: Dict[str, np.ndarray], prefix: str) -> float:
    """Total variation over the listed top-k shares (a lower bound of the full TVD; read it against the control)."""
    ks = [k for k in fa if k.startswith(prefix)]
    return 0.5 * float(sum(abs(np.nanmean(fa[k]) - np.nanmean(fb[k])) for k in ks))


def verdict_rule(res: Dict[str, Dict[str, dict]]) -> dict:
    """README rule. res[window] = {"vs": compare(A,S), "drift": compare(S1,S2) or None}."""
    dims = [k for k in next(iter(res.values()))["vs"] if spec(k)[0] is not None]
    fair = all(all(w["vs"][d]["verdict"] == "within" for d in dims) and (
        w["drift"] is None or all(w["drift"][d]["verdict"] == "within" for d in dims)) for w in res.values())
    outside: Dict[str, List[str]] = {}
    for d in dims:
        for sign in (1, -1):
            ws = [wn for wn, w in res.items() if w["vs"][d]["verdict"] == "OUTSIDE" and np.sign(w["vs"][d]["diff"]) == sign
                  and not (w["drift"] is not None and w["drift"][d]["verdict"] == "OUTSIDE"
                           and np.sign(w["drift"][d]["diff"]) == sign)]
            if len(ws) >= 2:
                outside[d] = ws
    return {"fair": fair, "skewed_dims": outside,
            "verdict": "FAIR" if fair else ("SKEWED" if outside else "CAN'T TELL")}


def fmt(name: str, r: dict) -> str:
    if r["verdict"] == "n/a" and r["bar"] is None:
        return "all zero" if r["diff"] == 0 else f"{r['diff']:+.4f}"
    scale = 100 if r["bar"] is not None and r["bar"] < 1 and name != "log_turns" else 1
    unit = "pp" if scale == 100 else ("%" if name == "log_turns" else " Elo")
    if name == "log_turns":
        f = lambda x: f"{100 * (np.exp(x) - 1):+.2f}"
        return f"{f(r['diff'])} [{f(r['lo'])}, {f(r['hi'])}] %"
    return f"{r['diff'] * scale:+.2f} [{r['lo'] * scale:+.2f}, {r['hi'] * scale:+.2f}] {unit}"


def cmd_analyze(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: Dict[str, Dict[str, dict]] = {}
    extra: Dict[str, dict] = {}
    md = ["# Results (generated by `rab.py analyze`)\n"]
    pool_rows = load(a.cache, all_dates(list(WINDOWS)))  # ONE species / lead list for every window (pooled top-k)
    species, leads = top_names(pool_rows, TOP_SPECIES, False), top_names(pool_rows, TOP_LEADS, True)
    del pool_rows
    for wn, (adays, bdays, cdays) in WINDOWS.items():
        A, S = load(a.cache, adays), load(a.cache, bdays + cdays)
        fA, fS = features(A, species, leads), features(S, species, leads)
        vs = compare(fA, fS, SEED)
        drift = None
        if cdays:
            S1, S2 = load(a.cache, bdays), load(a.cache, cdays)
        else:
            S1, S2 = load(a.cache, bdays[: len(bdays) // 2]), load(a.cache, bdays[len(bdays) // 2:])
        g1, g2 = features(S1, species, leads), features(S2, species, leads)
        drift = compare(g1, g2, SEED + 1)
        res[wn] = {"vs": vs, "drift": drift}
        extra[wn] = {"n_archive": len(A), "n_spectator": len(S), "archive_per_day": len(A) / len(adays),
                     "spectator_per_day": len(S) / len(bdays + cdays), "tvd_species": tvd(fA, fS, "sp:"),
                     "tvd_species_drift": tvd(g1, g2, "sp:"), "tvd_lead": tvd(fA, fS, "lead:"),
                     "tvd_lead_drift": tvd(g1, g2, "lead:"), "n_s1": len(S1), "n_s2": len(S2)}
        e = extra[wn]
        md.append(f"\n## {wn}: archive {adays[0]}..{adays[-1]} (n={e['n_archive']:,}, {e['archive_per_day']:.0f}/day) vs "
                  f"spectator {bdays[0]}..{(cdays or bdays)[-1]} (n={e['n_spectator']:,}, {e['spectator_per_day']:.0f}/day)\n")
        md.append(f"Drift control groups: S1 n={e['n_s1']:,}, S2 n={e['n_s2']:,}. Top-{TOP_SPECIES} species TVD (lower bound): "
                  f"archive-vs-spectator {e['tvd_species']:.4f}, drift {e['tvd_species_drift']:.4f}; top-{TOP_LEADS} leads TVD: "
                  f"{e['tvd_lead']:.4f} vs drift {e['tvd_lead_drift']:.4f}.\n")
        md.append("| dimension | mean A | mean S | A - S [CI] | bar | verdict | drift S1-S2 [CI] | drift verdict |\n|---|---|---|---|---|---|---|---|")
        for k in vs:
            bar = vs[k]["bar"]
            btxt = "n/a" if bar is None else (f"±{bar * 100:.1f} pp" if k != "log_turns" and bar < 1 else
                                              ("±5 %" if k == "log_turns" else f"±{bar:.0f} Elo"))
            md.append(f"| {k} | {vs[k]['mean_a']:.4g} | {vs[k]['mean_b']:.4g} | {fmt(k, vs[k])} | {btxt} | {vs[k]['verdict']} | "
                      f"{fmt(k, drift[k])} | {drift[k]['verdict']} |")
    v = verdict_rule(res)
    md.append(f"\n## Verdict by the registered rule: **{v['verdict']}** (fair={v['fair']}; skewed dims={json.dumps(v['skewed_dims'])})\n")
    (out / "results.md").write_text("\n".join(md) + "\n" + diagnostics(a))
    (out / "results.json").write_text(json.dumps({"verdict": v, "extra": extra, "res": res}, indent=1, default=float))
    print("\n".join(md))


def band(r: dict) -> Optional[str]:
    if not r["rated"] or r["r1"] is None or r["r2"] is None or r["end"] == "none":
        return None
    m = (r["r1"] + r["r2"]) / 2
    return "<1200" if m < 1200 else "1200-1399" if m < 1400 else "1400-1599" if m < 1600 else "1600+"


def diagnostics(a) -> str:
    """POST-HOC (not in the registration): player concentration, rating-band-matched contrasts, weekly stability of W3."""
    md = ["\n# Post-hoc diagnostics (NOT registered; read as explanation, not as verdict)\n"]
    for wn, (adays, bdays, cdays) in WINDOWS.items():
        A, S = load(a.cache, adays), load(a.cache, bdays + cdays)
        md.append(f"\n## {wn}: who plays the games\n\n| group | games | distinct players | top-1 player's share of sides | top-10 | top-100 |\n|---|---|---|---|---|---|")
        for lab, G in (("archive", A), ("spectator", S)):
            c: Dict[str, int] = {}
            for r in G:
                for k in ("n1", "n2"):
                    if r.get(k):
                        c[r[k]] = c.get(r[k], 0) + 1
            tot = sum(c.values())
            v = sorted(c.values(), reverse=True)
            md.append(f"| {lab} | {len(G):,} | {len(c):,} | {v[0] / tot:.2%} | {sum(v[:10]) / tot:.2%} | {sum(v[:100]) / tot:.2%} |")
        md.append(f"\n{wn} rating-band-matched (rated, completed games; band = mean of the pair's ratings):\n\n"
                  "| band | n archive | n spectator | forfeit A / S | mean turns A / S | Spikes A / S |\n|---|---|---|---|---|---|")
        for b in ("<1200", "1200-1399", "1400-1599", "1600+"):
            ga, gs = [r for r in A if band(r) == b], [r for r in S if band(r) == b]
            if not ga or not gs:
                continue
            md.append(f"| {b} | {len(ga):,} | {len(gs):,} | {np.mean([r['end'] == 'forfeit' for r in ga]):.1%} / "
                      f"{np.mean([r['end'] == 'forfeit' for r in gs]):.1%} | {np.mean([r['turns'] for r in ga]):.1f} / "
                      f"{np.mean([r['turns'] for r in gs]):.1f} | {np.mean([r['spikes'] for r in ga]):.1%} / "
                      f"{np.mean([r['spikes'] for r in gs]):.1%} |")
    md.append("\n## Leave the 10 most frequent archive players out (their games removed from A; A - S point estimates)\n\n"
              "| window | A games kept | rated share | pair rating (Elo) | forfeit share | turns (log-mean ratio) | Spikes |\n|---|---|---|---|---|---|---|")
    for wn, (adays, bdays, cdays) in WINDOWS.items():
        A, S = load(a.cache, adays), load(a.cache, bdays + cdays)
        c = {}
        for r in A:
            for k in ("n1", "n2"):
                if r.get(k):
                    c[r[k]] = c.get(r[k], 0) + 1
        top = {n for n, _ in sorted(c.items(), key=lambda kv: -kv[1])[:10]}
        Ak = [r for r in A if r.get("n1") not in top and r.get("n2") not in top]
        fa, fs = features(Ak, [], []), features(S, [], [])
        dd = lambda k: np.nanmean(fa[k]) - np.nanmean(fs[k])
        md.append(f"| {wn} | {len(Ak):,} of {len(A):,} | {dd('rated') * 100:+.1f} pp | {dd('rating_pair_mean'):+.0f} | "
                  f"{dd('end_forfeit') * 100:+.1f} pp | {100 * (np.exp(dd('log_turns')) - 1):+.1f} % | {dd('spikes') * 100:+.1f} pp |")
    A = load(a.cache, WINDOWS["W3"][0])
    md.append("\n## W3 archive by week (is the shift steady?)\n\n| week of | games | mean pair rating | Zapdos lead (share of sides) | Zapdos on side | top-1 player share | rated |\n|---|---|---|---|---|---|---|")
    d0 = dt.date.fromisoformat("2026-09-24")
    day_rows: Dict[int, List[dict]] = {}
    for dday in WINDOWS["W3"][0]:
        wk = (dt.date.fromisoformat(dday) - d0).days // 7
        day_rows.setdefault(wk, []).extend(load(a.cache, [dday]))
    for wk, G in sorted(day_rows.items()):
        c: Dict[str, int] = {}
        for r in G:
            for k in ("n1", "n2"):
                if r.get(k):
                    c[r[k]] = c.get(r[k], 0) + 1
        rr = [(r["r1"] + r["r2"]) / 2 for r in G if r["rated"] and r["r1"] and r["r2"]]
        zl = np.mean([((bool(r["sp1"]) and r["sp1"][0] == "Zapdos") + (bool(r["sp2"]) and r["sp2"][0] == "Zapdos")) / 2 for r in G])
        zs = np.mean([(("Zapdos" in r["sp1"]) + ("Zapdos" in r["sp2"])) / 2 for r in G])
        md.append(f"| {(d0 + dt.timedelta(7 * wk)).isoformat()} | {len(G):,} | {np.mean(rr):.0f} | {zl:.1%} | {zs:.1%} | "
                  f"{max(c.values()) / sum(c.values()):.2%} | {np.mean([r['rated'] for r in G]):.1%} |")
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cache", required=True)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--windows", default="W1,W2,W3")
    z = sub.add_parser("analyze")
    z.add_argument("--cache", required=True)
    z.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
