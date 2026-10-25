"""REVIEW A (data / bias / leakage) of the hr_spec_v1 data card: the spot-checks behind REVIEW_A_data.md.

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 10 \
        $PY designs/research_state/measurements/human_replay_quality_2026-10-25/review_a_checks.py   # ~2 min, 5.7 GB peak

Read-only: the hrq extraction cache, the X50 rung-1b archive, the corpus (headers + a 400-file re-hash sample) and
the snapshot manifests. Prints one JSON line per check (the review quotes them).
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # hrq.py, beside this file
import hrq

CACHE = Path.home() / "gen3ai_archive/human_replay_quality/cache"
SNAP = Path.home() / "gen3ai_archive/human_replay_snapshots/hr_spec_v1.tsv.gz"
OUT: dict = {}


def say(k, v):
    OUT[k] = v
    print(k, json.dumps(v, default=float)[:1500], flush=True)


recs = [r for r in hrq.load(CACHE) if "err" not in r]
spec = [r for r in recs if r["pop"] == "spec"]
say("n_spec", len(spec))
per = hrq.accounts(spec)
flagged = set(hrq.bot_flags(per))

# ---------------------------------------------------------------- G: snapshot re-hash sample + digest
with gzip.open(SNAP, "rb") as fh:
    body = fh.read()
say("G_manifest_digest", hashlib.sha256(body).hexdigest())
lines = body.decode().strip().split("\n")
rng = random.Random(1)
bad = 0
for ln in rng.sample(lines, 400):
    rel, h, n = ln.split("\t")
    b = (hrq.CORPUS / rel).read_bytes()
    bad += hashlib.sha256(b).hexdigest() != h or len(b) != int(n)
say("G_rehash_sample_400_mismatch", bad)
manifest_rel = {ln.split("\t")[0] for ln in lines}
spec_manifest = hrq.manifest_set()
say("G_snapshot_eq_spectator_manifest", manifest_rel == spec_manifest)

sig = collections.Counter()
for snap in ("hr_spec_v1", "hr_arch_2026-10-24"):
    with gzip.open(SNAP.parent / f"{snap}.tsv.gz", "rt") as fh:
        snap_lines = fh.read().split("\n")
    for ln in snap_lines:
        if ln:
            with open(hrq.CORPUS / ln.split("\t")[0], "rb") as f:
                sig[(snap, f.read(12) == b"|init|battle")] += 1
say("G_init_header_by_snapshot", {f"{a}:{b}": v for (a, b), v in sig.items()})
# X50 battles inside the snapshot?
x50 = collections.Counter()
for sub in ("train", "heldout"):
    for js in sorted((hrq.X50 / sub).glob("humans_*.json")):
        for p in json.loads(js.read_text())["battles"]:
            rel = "/".join(Path(p).parts[-2:])
            x50[(sub, rel in manifest_rel)] += 1
say("G_x50_battles_in_snapshot", {f"{a}:{b}": v for (a, b), v in x50.items()})

# ---------------------------------------------------------------- A: bots below the bar
chat = collections.Counter()
for r in spec:
    for s in ("1", "2"):
        n = r.get("n" + s)
        if n:
            chat[hrq.toid(n)] += r.get("chat" + s, 0)
cand = []
for k, a in per.items():
    if k in flagged or a["games"] < 100:
        continue
    mh = max((len(v) for v in a["hours"].values()), default=0)
    md = max(a["days"].values()) if a["days"] else 0
    if chat[k] == 0 and (mh >= 14 or md >= 50):
        cand.append((a["games"], a["name"], md, mh, float(np.mean(a["rat"])) if a["rat"] else None))
cand.sort(reverse=True)
say("A_zero_chat_heavy_unflagged_n", len(cand))
say("A_zero_chat_heavy_unflagged_top", cand[:25])
fl_chat = [chat[k] / per[k]["games"] for k in flagged if per[k]["games"] >= 100]
say("A_flagged_ge100_chat_per_game_median", float(np.median(fl_chat)))
allchat = [chat[k] / a["games"] for k, a in per.items() if a["games"] >= 100]
say("A_all_ge100_zero_chat_share", float(np.mean(np.asarray(allchat) == 0)))
# >= 1500 side share from candidate accounts (train dates)
candk = {hrq.toid(c[1]) for c in cand}
tot = hit = hitf = 0
for r in spec:
    if r["date"] > hrq.TRAIN_LAST or not r["rated"]:
        continue
    for s in ("1", "2"):
        if (r.get("r" + s) or 0) >= 1500:
            tot += 1
            k = hrq.toid(r.get("n" + s) or "")
            hit += k in candk
            hitf += k in flagged
say("A_ge1500_train_sides", {"total": tot, "flagged_share": hitf / tot, "zero_chat_heavy_unflagged_share": hit / tot})

# ---------------------------------------------------------------- B: spectator saturation (concurrency of captured games)
ev = []
for r in spec:
    if r["t0"] and r["t1"] and r["t1"] >= r["t0"]:
        ev.append((r["t0"], 1))
        ev.append((r["t1"], -1))
ev.sort()
t_start, t_end = ev[0][0], ev[-1][0]
grid = np.arange(t_start, t_end, 60)
times = np.asarray([e[0] for e in ev])
cum = np.cumsum([e[1] for e in ev])
idx = np.searchsorted(times, grid, side="right") - 1
conc = np.where(idx >= 0, cum[np.clip(idx, 0, None)], 0)
live = conc > 0
c = conc[live]
say("B_concurrency_minutes_with_any", int(live.sum()))
say("B_concurrency_pct", {q: float(np.percentile(c, q)) for q in (10, 50, 90, 99, 99.9)} | {"max": int(c.max())})
say("B_concurrency_hist_top", {int(k): int(v) for k, v in sorted(collections.Counter(c.tolist()).items())[-12:]})
# games per LA hour-of-day
hod = collections.Counter(hrq.la_date_hour(r["t0"])[1] for r in spec if r["t0"])
say("B_games_by_LA_hour", [hod[h] for h in range(24)])
dur = np.asarray([(r["t1"] - r["t0"]) for r in spec if r["t0"] and r["t1"]])
say("B_duration_s_pct", {q: float(np.percentile(dur, q)) for q in (1, 5, 10, 50, 90, 99, 99.9)} | {"max": float(dur.max())})
daily = collections.Counter(r["date"] for r in spec)
dv = np.asarray([v for d, v in sorted(daily.items())][1:-1])
say("B_daily_games", {"min": int(dv.min()), "p10": float(np.percentile(dv, 10)), "p50": float(np.median(dv)),
                      "p90": float(np.percentile(dv, 90)), "max": int(dv.max())})
bymonth = collections.defaultdict(list)
for d, v in sorted(daily.items()):
    bymonth[d[:7]].append(v)
say("B_daily_median_by_month", {m: float(np.median(v)) for m, v in bymonth.items()})

# ---------------------------------------------------------------- teams
teams = hrq.Teams()

# ---------------------------------------------------------------- C: drift floor, account-clustered
groups = collections.defaultdict(list)
for r in spec:
    if not r["rated"]:
        continue
    for s in ("1", "2"):
        if r.get("r" + s) is not None and len(r["o" + s]) == 6:
            groups[r["date"][:7]].append((hrq.toid(r.get("n" + s) or ""), r["o" + s]))


def marg(ts):
    cc = collections.Counter(teams.canon(x) for t in ts for x in t)
    t_ = sum(cc.values())
    return {k: v / t_ for k, v in cc.items()}


def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


jul = groups["2026-07"]
accs = sorted({a for a, _ in jul})
fl_side, fl_acc = [], []
for seed in range(10):
    g = np.random.default_rng(seed)
    ix = g.permutation(len(jul))
    h = len(ix) // 2
    fl_side.append(tvd(marg([jul[i][1] for i in ix[:h]]), marg([jul[i][1] for i in ix[h:]])))
    half = set(g.permutation(len(accs))[: len(accs) // 2].tolist())
    A_ = {accs[i] for i in half}
    fl_acc.append(tvd(marg([o for a, o in jul if a in A_]), marg([o for a, o in jul if a not in A_])))
say("C_july_split_half_floor_by_side", {"mean": float(np.mean(fl_side)), "max": float(np.max(fl_side))})
say("C_july_split_half_floor_by_account", {"mean": float(np.mean(fl_acc)), "min": float(np.min(fl_acc)),
                                           "max": float(np.max(fl_acc))})
months = sorted(groups)
mm = {m: marg([o for _, o in groups[m]]) for m in months}
say("C_month_to_month_tvd", {f"{a}->{b}": tvd(mm[a], mm[b]) for a, b in itertools.pairwise(months)})
# Smogon prior (12 months 2025-05..2026-04, cutoff 1500) vs the spectator >= 1500 species usage, per month
sm = json.loads((hrq.REPO / "data/pokemon/gen3_smogon_stats.json").read_text())
smu = {teams.canon(k): v["usage"] for k, v in sm["data"].items()}
st = sum(smu.values())
smu = {k: v / st for k, v in smu.items()}
hi_m = collections.defaultdict(list)
for r in spec:
    if not r["rated"]:
        continue
    for s in ("1", "2"):
        if (r.get("r" + s) or 0) >= 1500 and len(r["o" + s]) == 6:
            hi_m[r["date"][:7]].append(r["o" + s])
say("J_smogon_prior_vs_spec_ge1500_tvd", {m: tvd(smu, marg(v)) for m, v in sorted(hi_m.items())})
ms = sorted(hi_m)
say("J_spec_ge1500_first_vs_last_month_tvd", tvd(marg(hi_m[ms[0]]), marg(hi_m[ms[-1]])))

# ---------------------------------------------------------------- E: panel exclusion share (train dates, battles with a >=1500 side)
HELD = ("heldout72_hpfix", "forum72", "p72")


def informative(mons):
    """only mons that revealed >= 1 move (a species-only reveal is dropped)"""
    return {k: v for k, v in mons.items() if v[0]}


ex = collections.Counter()
for r in spec:
    if r["date"] > hrq.TRAIN_LAST or not r["rated"]:
        continue
    if max(r.get("r1") or 0, r.get("r2") or 0) < 1500:
        continue
    w = max(r["turns"], 1)
    ex["battles"] += 1
    ex["turns"] += w
    best = {p: 0 for p in HELD}
    best_inf = {p: 0 for p in HELD}
    best_sp = {p: 0 for p in HELD}
    for s in ("1", "2"):
        m = r["m" + s]
        for p in HELD:
            sp, ft = teams.overlaps(m, p, min_sp=3)
            best[p] = max(best[p], int(ft.max()) if len(ft) else 0)
            best_sp[p] = max(best_sp[p], int(sp.max()) if len(sp) else 0)
            mi = informative(m)
            if mi:
                _, fi = teams.overlaps(mi, p, min_sp=1)
                best_inf[p] = max(best_inf[p], int(fi.max()))
    for thr in (3, 4):
        anyp = any(best[p] >= thr for p in HELD)
        ex[f"fit>={thr}_any"] += anyp
        ex[f"fit>={thr}_any_turns"] += w * anyp
        for p in HELD:
            ex[f"fit>={thr}_{p}"] += best[p] >= thr
    ex["fit>=4_any_but_<=1_moverevealed_fits"] += any(best[p] >= 4 for p in HELD) and all(best_inf[p] <= 1 for p in HELD)
    ex["fit<=3_but_species>=5_any"] += all(best[p] <= 3 for p in HELD) and any(best_sp[p] >= 5 for p in HELD)
    ex["moverevealed_fit>=3_any"] += any(best_inf[p] >= 3 for p in HELD)
n = ex["battles"]
say("E_panel_exclusion", {k: (v / n if k not in ("battles", "turns") and not k.endswith("_turns") else v)
                          for k, v in ex.items()} | {"turn_share_fit>=4": ex["fit>=4_any_turns"] / ex["turns"],
                                                     "turn_share_fit>=3": ex["fit>=3_any_turns"] / ex["turns"]})

# ---------------------------------------------------------------- D: familiarity strata contamination (held-out sides >= 1500)
D = collections.Counter()
for r in spec:
    if r["date"] <= hrq.TRAIN_LAST or not r["rated"]:
        continue
    for s in ("1", "2"):
        if (r.get("r" + s) or 0) < 1500:
            continue
        m = r["m" + s]
        sp, fits, _ = teams.side(m, "pool")
        nseen = len(r["o" + s])
        _, fi = teams.overlaps(informative(m), "pool", min_sp=1) if informative(m) else (None, np.zeros(1))
        fi = int(fi.max())
        if sp <= 3:
            D["sp<=3"] += 1
            D["sp<=3_with_<6_seen"] += nseen < 6
            D["sp<=3_with_<=4_seen"] += nseen <= 4
        if fits >= 4:
            D["fit>=4"] += 1
            D["fit>=4_movefit<=2"] += fi <= 2
        if fits <= 2:
            D["fit<=2"] += 1
            D["fit<=2_with_<6_seen"] += nseen < 6
say("D_familiarity_contamination", dict(D))

# ---------------------------------------------------------------- I: account / team overlap train vs held-out
train_acc = set()
train_acc_team = set()
for r in spec:
    if r["date"] <= hrq.TRAIN_LAST:
        for s in ("1", "2"):
            k = hrq.toid(r.get("n" + s) or "")
            train_acc.add(k)
            if len(r["o" + s]) == 6:
                train_acc_team.add((k, frozenset(teams.canon(x) for x in r["o" + s])))
I_ = collections.Counter()
for r in spec:
    if r["date"] > hrq.TRAIN_LAST and r["rated"]:
        for s in ("1", "2"):
            if (r.get("r" + s) or 0) >= 1500:
                k = hrq.toid(r.get("n" + s) or "")
                I_["sides"] += 1
                I_["acc_in_train"] += k in train_acc
                if len(r["o" + s]) == 6:
                    I_["full6"] += 1
                    I_["acc_team_in_train"] += (k, frozenset(teams.canon(x) for x in r["o" + s])) in train_acc_team
say("I_heldout_overlap", dict(I_))

# ---------------------------------------------------------------- F: the 2 % rating-header mismatch
by_path = {f"{r['date']}/battle-gen3ou-{r['id'].split('-')[-1]}.log": r for r in spec if r["date"] > hrq.TRAIN_LAST}
F = collections.Counter()
for js in sorted((hrq.X50 / "heldout").glob("humans_*.json")):
    meta = json.loads(js.read_text())
    z = np.load(js.with_suffix(".npz"))
    sel = (z["mode"] == 2) & (z["own_action"] >= 0) & (z["n_legal"] >= 2)
    base = int(js.stem.split("_")[-1]) * int(meta["part_size"])
    for b, st_, rv, rl in zip(z["battle"][sel], z["seat"][sel], z["rating_viewer"][sel], z["rating_labelled"][sel]):
        key = "/".join(Path(meta["battles"][int(b) - base]).parts[-2:])
        rec = by_path.get(key)
        s = "1" if int(st_) == 0 else "2"
        o = "2" if s == "1" else "1"
        if rec is None:
            F["rec_missing"] += 1
            continue
        mine, other = rec.get("r" + s), rec.get("r" + o)
        if mine == int(rv):
            F["match"] += 1
        else:
            F["mismatch"] += 1
            F["mm_equals_other_seat"] += other == int(rv)
            F["mm_card_rating_none"] += mine is None
            F["mm_rv<=0"] += int(rv) <= 0
            F["mm_stratum_changes_ge1500"] += ((mine or 0) >= 1500) != (int(rv) >= 1500)
            F["mm_abs_diff_le25"] += mine is not None and abs(mine - int(rv)) <= 25
say("F_rating_mismatch", dict(F))

