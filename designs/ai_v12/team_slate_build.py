"""ai_v12 TEAM SLATE BUILDER — regenerates designs/ai_v12/team_slate_40.{md,json} — consolidate every per-team win-rate estimate into one ranked slate.

Read-only: no battles, no models, no network. ~5 s.\n\nRun: python designs/ai_v12/team_slate_build.py\n(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import hashlib, json, math, os, sys, glob
from collections import Counter, defaultdict

P = "/home/goodlad/.claude/jobs/1046b1d6/tmp/probes"
CEILING = 0.6896          # set-mean of the 12 rev-3 teacher absolute cells (r3_admission)
CEIL_LO, CEIL_HI = 0.5925, 0.7750   # per-team observed teacher range

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from utils.paths import main_models_dir, repo_root               # noqa: E402
os.chdir(repo_root())
OUT = os.path.join(repo_root(), "designs", "ai_v12")
from utils.team_loader import TeamLoader                        # noqa: E402

if not os.path.isdir(P):
    raise SystemExit(
        f"\n[team_slate] the probe artifacts are GONE: {P}\n\n"
        "  They live in a SESSION-SCOPED job directory, not in the repo. This script cannot\n"
        "  regenerate the slate without them — but it does not need to: every number it derived\n"
        "  is already committed in designs/ai_v12/team_slate_40.json (per-team `evidence` rows,\n"
        "  the calibration block, the full 719-team table). Read that instead of re-deriving.\n\n"
        "  To rebuild from scratch you would have to re-run probes/{coverage_sweep,coverage_sample,\n"
        "  headroom_screen,fleet_admission,r3_admission}.py — thousands of battles, not minutes.\n")


def sha(text: str) -> str:
    """team_archetypes.team_sha convention — STRIP-normalized sha1[:10]."""
    return hashlib.sha1(text.strip().encode()).hexdigest()[:10]

def raw_sha(text: str) -> str:
    """coverage_sample.py's UNSTRIPPED variant (a derived-key defect; needed to rejoin it)."""
    return hashlib.sha1(text.encode()).hexdigest()[:10]

def wilson(k, n, z=1.96):
    if not n: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - hw, c + hw)

# ─── universe ────────────────────────────────────────────────────────────────
arch = json.load(open("data/teams/gen3_team_archetypes.json"))["teams"]
L = TeamLoader()
pool_txt = {sha(t): t for t in L.get_all_teams()}
cur_file, cur_raw = {}, {}
for f in sorted(glob.glob("data/teams/sample/*.txt")):
    txt = open(f).read()
    cur_file[sha(txt)] = f
    cur_raw[raw_sha(txt)] = sha(txt)
CURATED = set(cur_file)
assert len(pool_txt) == 719 and len(CURATED) == 32, (len(pool_txt), len(CURATED))

ALL_PATHS = {}
for _root, _d, _f in os.walk("data/teams"):
    for _n in _f:
        if not _n.endswith(".txt"): continue
        _p = os.path.join(_root, _n)
        try: ALL_PATHS.setdefault(sha(open(_p).read()), _p)
        except Exception: pass

def stem(s):  # sample-file stem, for the human-readable name
    return os.path.basename(cur_file[s])[:-4] if s in cur_file else None

# ─── the named sets ──────────────────────────────────────────────────────────
NAMED = {  # r3_admission.py TEAMS (the authority: probes/authoritative_teams.txt)
    "ZapDug": "eccfe630ec08de27", "JynxSO": "023a2d47648b85e6",
    "RaikouCelebi": "8e768980fc8f3b5f", "MixZap": "710d8d529538ff90",
    "BlueOffense": "63eda9d8d491d6a4", "MedichamCune": "f5a4f4f0f5dc49ce",
    "CBMetaCroCune": "e541f7be8713393c", "Q6a": "9eb3abdc52876a63",
    "Q6b": "e0d97b0ed592889d", "COV_82e97fe2": "82e97fe2bfbb6f69",
    "COV_460b8c99": "460b8c99ee366f6c", "COV_7036a0a1": "7036a0a1dcb59a19",
}
name2sha = {n: sha(open(f"data/teams/sample/{st}.txt").read()) for n, st in NAMED.items()}
sha2name = {v: k for k, v in name2sha.items()}
METER9 = [name2sha[n] for n in list(NAMED)[:9]]
COV3   = [name2sha[n] for n in list(NAMED)[9:]]
HELDOUT = [sha(open(f"{P}/offpin_{i}.txt").read()) for i in (0, 1)]

def argv_teams(tag):
    line = open(f"{P}/{tag}.argv").read()
    i = line.index("--trainee-teams ") + len("--trainee-teams ")
    files = line[i:].split()[0].split(",")
    return [sha(open(f).read()) for f in files], files

TAUGHT = {}   # sha -> [run tags]
for tag in ["F5a","F5b","F5c","F5d","F5e","F6a","F6b","F6c","F6d","F6e","F6f","F6CURR"]:
    for s in argv_teams(tag)[0]: TAUGHT.setdefault(s, []).append(tag)
REV4 = {}
for tag in ["R4S3a","R4S3b","R4S3c"]:
    for s in argv_teams(tag)[0]: REV4.setdefault(s, []).append(tag)

# ─── evidence sources ────────────────────────────────────────────────────────
# Each row: (sha, wins, games, scale, source). scale "gen15" = pilot is the CURRENT
# generation's product (R2-ACTION final). scale "rev1f" = pilot is rev-1 final (one
# generation older). Opponent is rev-1@24M except where noted; the two opponents are
# shown equivalent below.
ev = defaultdict(list)
def add(s, w, n, scale, src, opp):
    if n: ev[s].append({"wins": int(w), "games": int(n), "scale": scale, "source": src, "opponent": opp})

r3 = json.load(open(f"{P}/r3_admission.json"))
for nm, c in r3["target"].items():
    add(name2sha[nm], c["wins"], c["games"], "gen15", "r3_admission.target", "R2ACTION-final")
fa = json.load(open(f"{P}/fleet_admission.json"))
for nm, c in fa["rev1final"].items():
    add(name2sha[nm], c["wins"], c["games"], "rev1f", "fleet_admission.rev1final", "rev1@24M")
for tag, path, scale in [("R2ACTION", "pilot_R2ACTION_n300.json", "gen15"),
                         ("rev1fin", "pilot_rev1fin_n300.json", "rev1f")]:
    if not os.path.exists(f"{P}/{path}"): continue
    for nm, c in json.load(open(f"{P}/{path}")).items():
        if nm in ("_meta", "POOLED") or nm not in name2sha: continue
        add(name2sha[nm], c["wins"], c["games"], scale, path, "rev1@24M")
for tag, path, scale in [("R2ACTION", "cov_R2ACTION.json", "gen15"),
                         ("rev1fin", "cov_rev1fin.json", "rev1f")]:
    for nm, c in json.load(open(f"{P}/{path}")).items():
        if nm in ("_meta", "POOLED"): continue
        add(name2sha[nm], c["wins"], c["games"], scale, path, "rev1@24M")
hs = json.load(open(f"{P}/headroom_screen.json"))
HS_PATHS = {}
for ln in open(f"{P}/headroom_screen.py"):
    if ln.strip().startswith('("') and "data/teams/sample" in ln:
        k, v = ln.strip().strip("(),").split(",", 1)
        HS_PATHS[k.strip().strip('"')] = v.strip().strip(' "')
for nm, c in hs.items():
    if nm in ("_meta", "POOLED"): continue
    add(sha(open(HS_PATHS[nm]).read()), c["wins"], c["games"], "gen15", "headroom_screen", "rev1@24M")
cs = json.load(open(f"{P}/coverage_sample.json"))
for r in cs["confirm"]:
    add(cur_raw[r["sha"]], r["wins"], r["games"], "rev1f", "coverage_sample(n=200)", "rev1@24M")
cw = json.load(open(f"{P}/coverage_sweep.json"))
for r in cw["confirm"]:
    add(r["sha"], r["wins"], r["games"], "rev1f", "coverage_sweep.confirm(n=200)", "rev1@24M")
SCREEN40 = {r["sha"]: r for r in cw["screen"]}   # n=40 — nomination only, NOT pooled
for r in cw["screen"]:   # recorded as evidence under its OWN scale, so `pooled` can never mix it in
    add(r["sha"], r["wins"], r["games"], "rev1f_screen40", "coverage_sweep.screen(n=40)", "rev1@24M")

# training-time per-team WR (task #18 machinery) — a THIRD scale, never pooled with the above
TRAIN_RUNS = ["ai_v9_29_rev1_0823", "ai_v9_58_R2CTRL_0827", "ai_v9_59_R2ACTION_0827",
              "ai_v9_70_R3ACTION_0828", "ai_v9_71_R3ACTIONHI_0828", "ai_v9_72_R3SELF_0828"]
train = defaultdict(lambda: {"w": 0, "n": 0})
train_rev1 = {}
#: models/ is NOT committed and exists only in the MAIN checkout — a worktree has none, so this
#: reaches across via git's shared common dir. None => the training-time nomination column is
#: absent (recorded in the artifact), never silently zero.
MODELS = main_models_dir()
for r in TRAIN_RUNS:
    if MODELS is None: break
    p = os.path.join(str(MODELS), r, "metadata.json")
    if not os.path.exists(p): continue
    blk = (json.load(open(p)).get("team_win_rates") or {}).get("teams") or {}
    for s, v in blk.items():
        pc = v.get("by_class", {}).get("pool")
        if not pc: continue
        if r != "ai_v9_29_rev1_0823":
            train[s]["w"] += pc["wins"]; train[s]["n"] += pc["n"]
        else:
            train_rev1[s] = (pc["wins"], pc["n"])

# ─── pooling + calibration ───────────────────────────────────────────────────
def pooled(s, scale):
    rs = [r for r in ev[s] if r["scale"] == scale]
    if not rs: return None
    w = sum(r["wins"] for r in rs); n = sum(r["games"] for r in rs)
    lo, hi = wilson(w, n)
    return {"wr": w / n, "wins": w, "games": n, "ci95": [lo, hi],
            "sources": sorted({r["source"] for r in rs})}

both = [(s, pooled(s, "gen15"), pooled(s, "rev1f")) for s in CURATED]
both = [(s, a, b) for s, a, b in both if a and b]
gen_off = sum(a["wr"] - b["wr"] for _, a, b in both) / len(both)
# opponent-equivalence check: the 3 coverage teams have BOTH opponents on the gen15 scale
opp_pairs = []
for s in COV3:
    a = [r for r in ev[s] if r["source"] == "r3_admission.target"]
    b = [r for r in ev[s] if r["source"] == "cov_R2ACTION.json"]
    if a and b: opp_pairs.append((a[0]["wins"] / a[0]["games"], b[0]["wins"] / b[0]["games"]))
opp_off = sum(x - y for x, y in opp_pairs) / len(opp_pairs) if opp_pairs else None

def spearman(a, b):
    n = len(a)
    if n < 3: return None
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0] * n
        for i, j in enumerate(o): r[j] = i
        return r
    ra, rb = rk(a), rk(b); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((x - mb) ** 2 for x in rb))
    return num / den if den else None

# does the training-time WR PREDICT the fixed-reference piloting WR?  (the gate from the
# code-rank lesson: rank a lever on prediction, never on "the quantity is available")
vx, vy = [], []
for s in CURATED:
    g = pooled(s, "gen15")
    if g and s in train and train[s]["n"] >= 30:
        vx.append(g["wr"]); vy.append(train[s]["w"] / train[s]["n"])
rho_cur = spearman(vx, vy)
px, py = [], []
for s, r in list(SCREEN40.items()):
    b = pooled(s, "rev1f")
    if b and b["games"] >= 200 and s in train_rev1 and train_rev1[s][1] >= 30:
        px.append(b["wr"]); py.append(train_rev1[s][0] / train_rev1[s][1])
rho_pool = spearman(px, py)

# ═══ STAGE 2 — the slate ══════════════════════════════════════════════════════
def estimate(s):
    """Best available baseline WR on the GEN-15 scale, with its provenance and tier."""
    g = pooled(s, "gen15")
    if g:
        return dict(g, tier="A_direct", adjusted=False,
                    note="measured with the current generation's product as pilot")
    b = pooled(s, "rev1f")
    if b and b["games"] >= 150:
        lo, hi = b["ci95"]
        return {"wr": b["wr"] + gen_off, "wins": b["wins"], "games": b["games"],
                "ci95": [lo + gen_off, hi + gen_off], "sources": b["sources"],
                "tier": "B_shifted", "adjusted": True,
                "note": f"rev-1-final pilot, shifted by the measured generation offset {gen_off:+.4f}"}
    r = SCREEN40.get(s)
    if r:
        lo, hi = wilson(r["wins"], r["games"])
        return {"wr": r["wr"] + gen_off, "wins": r["wins"], "games": r["games"],
                "ci95": [lo + gen_off, hi + gen_off], "sources": ["coverage_sweep.screen(n=40)"],
                "tier": "C_screen_only", "adjusted": True,
                "note": "n=40 SCREEN, selected-on-noise-prone — nomination only, NOT a measurement"}
    return None

rows = []
for s, txt in pool_txt.items():
    a = arch.get(s, {})
    est = estimate(s)
    tr = train.get(s); trn = tr["n"] if tr else 0
    excl = []
    if s in METER9: excl.append("meter9")
    if s in HELDOUT: excl.append("held_out2")
    if s in TAUGHT: excl.append("taught:" + "/".join(TAUGHT[s]))
    if s in REV4: excl.append("rev4_pending:" + "/".join(REV4[s]))
    rows.append({
        "sha": s, "name": sha2name.get(s), "file": cur_file.get(s) or ALL_PATHS.get(s), "stem": stem(s),
        "path": ALL_PATHS.get(s),
        "curated": s in CURATED, "archetype": a.get("archetype"), "tags": a.get("tags", []),
        # RAW per-source cells, copied in so the slate outlives the session-scoped probes dir.
        "evidence": sorted(ev.get(s, []), key=lambda r: (r["scale"], r["source"])),
        "baseline_wr": (round(est["wr"], 4) if est else None),
        "n": (est["games"] if est else 0),
        "evidence_tier": (est["tier"] if est else "UNMEASURED"),
        "ci95": ([round(x, 4) for x in est["ci95"]] if est else None),
        "sources": (est["sources"] if est else []),
        "estimate_note": (est["note"] if est else "no fixed-reference piloting measurement exists"),
        "headroom": (round(CEILING - est["wr"], 4) if est else None),
        "train_wr_pool": (round(tr["w"] / tr["n"], 4) if trn >= 30 else None),
        "train_n_pool": trn,
        "excluded": excl,
    })
by_sha = {r["sha"]: r for r in rows}

# ── the slate: eligible = not meter, not held-out, not taught (incl. rev-4 pending) ──
def eligible(r): return not r["excluded"]
elig = [r for r in rows if eligible(r)]
measured = [r for r in elig if r["evidence_tier"] in ("A_direct", "B_shifted")]
screened = [r for r in elig if r["evidence_tier"] == "C_screen_only"]
unmeas   = [r for r in elig if r["evidence_tier"] == "UNMEASURED"]
for L_ in (measured, screened): L_.sort(key=lambda r: -r["headroom"])
# UNMEASURED ranked separately, by the weak training-time nomination signal (ascending WR),
# teams with no training games last.
unmeas.sort(key=lambda r: (r["train_wr_pool"] is None, r["train_wr_pool"] or 0))

CAP = 0.40
def pick(n=40):
    out, cnt = [], Counter()
    for src in (measured, screened, unmeas):
        for r in src:
            if len(out) >= n: break
            if cnt[r["archetype"]] + 1 > math.floor(CAP * n): continue
            out.append(r); cnt[r["archetype"]] += 1
        if len(out) >= n: break
    return out, cnt
slate, acount = pick(40)

def assign(teams, n_teachers, per):
    """Archetype-balanced round-robin: sort by archetype then headroom, deal into teachers."""
    order = sorted(teams, key=lambda r: (r["archetype"] or "zz", -(r["headroom"] or 0)))
    buckets = [[] for _ in range(n_teachers)]
    for i, r in enumerate(order): buckets[i % n_teachers].append(r)
    return buckets

A5 = assign(slate, 5, 8)
A20 = assign(slate, 20, 2)

def se_sp(n): return (1 / math.sqrt(n - 1)) if n and n > 2 else None
meta = {
    "generated_for": "S1 — candidate team slate for the 40-team flywheel revolution (ai_v12)",
    "ceiling_used": CEILING, "ceiling_per_team_observed_range": [CEIL_LO, CEIL_HI],
    "ceiling_provenance": "set-mean of the 12 rev-3 teacher ABSOLUTE cells in r3_admission.json",
    "headroom_definition": "ceiling - baseline_wr (both on the GEN-15 piloting scale)",
    "scales": {
        "gen15": "pilot = R2-ACTION final (the current generation's product); opponent draws the "
                 "full 719 pool; per-team seeded paired opponent sequence",
        "rev1f": "pilot = rev-1 final (one generation older); same opponent/harness",
        "train": "training-time per-team win rate from metadata.json team_win_rates (pool class) — "
                 "a MOVING self-play opponent, NOT comparable to either fixed-reference scale",
    },
    "calibration": {
        "gen15_minus_rev1f": round(gen_off, 4), "n_teams": len(both),
        "opponent_offset_R2ACTIONfinal_minus_rev1at24M": (round(opp_off, 4) if opp_off is not None else None),
        "n_opponent_pairs": len(opp_pairs),
        "spearman_train_vs_fixed_reference_curated": (round(rho_cur, 3) if rho_cur else None),
        "n_curated": len(vx), "se_curated": (round(se_sp(len(vx)), 3) if se_sp(len(vx)) else None),
        "spearman_train_vs_fixed_reference_pool": (round(rho_pool, 3) if rho_pool else None),
        "n_pool": len(px), "se_pool": (round(se_sp(len(px)), 3) if se_sp(len(px)) else None),
    },
    "coverage": {
        "pool_total": len(pool_txt), "curated_total": len(CURATED),
        "gen15_measured": sum(1 for r in rows if r["evidence_tier"] == "A_direct"),
        "rev1f_shifted": sum(1 for r in rows if r["evidence_tier"] == "B_shifted"),
        "screen_only_n40": sum(1 for r in rows if r["evidence_tier"] == "C_screen_only"),
        "unmeasured": sum(1 for r in rows if r["evidence_tier"] == "UNMEASURED"),
        "train_wr_available_n_ge_30": sum(1 for r in rows if r["train_n_pool"] >= 30),
        "models_dir": (str(MODELS) if MODELS else None),
    },
    "sets": {
        "meter9": METER9, "held_out2": HELDOUT, "coverage3": COV3,
        "taught_F5_F6": sorted(TAUGHT), "rev4_pending": sorted(REV4),
        "curated_untaught_after_rev4": sorted(CURATED - set(TAUGHT) - set(REV4) - set(HELDOUT)),
    },
    "sources": {
        "r3_admission": f"{P}/r3_admission.json", "fleet_admission": f"{P}/fleet_admission.json",
        "pilot_R2ACTION_n300": f"{P}/pilot_R2ACTION_n300.json",
        "cov_R2ACTION": f"{P}/cov_R2ACTION.json", "cov_rev1fin": f"{P}/cov_rev1fin.json",
        "headroom_screen": f"{P}/headroom_screen.json (IN FLIGHT at build time)",
        "coverage_sample": f"{P}/coverage_sample.json (identity recovered via UNSTRIPPED sha)",
        "coverage_sweep": f"{P}/coverage_sweep.json",
        "team_win_rates": "models/<run>/metadata.json:team_win_rates (TeamWinRateCallback)",
        "archetypes": "data/teams/gen3_team_archetypes.json",
    },
}

# ═══ STAGE 3 — render ════════════════════════════════════════════════════════
core, shortlist = measured, screened
os.makedirs(OUT, exist_ok=True)

# nomination score for the shortlist: the n=40 screen is the primary (a fixed-reference
# measurement, however thin); the training-time WR is a weak tie-break with a MEASURED
# rank correlation of rho~0.36-0.51, reported so the discount is visible.
for r in shortlist:
    t = r["train_wr_pool"]
    r["_nom"] = (r["headroom"] or 0) + (0.0 if t is None else 0.25 * (0.70 - t))
shortlist.sort(key=lambda r: -r["_nom"])

def pick_balanced(src, n, seed_counts=None):
    out, cnt = [], Counter(seed_counts or {})
    for r in src:
        if len(out) >= n: break
        if cnt[r["archetype"]] + 1 > math.floor(CAP * 40): continue
        out.append(r); cnt[r["archetype"]] += 1
    return out, cnt

CORE, ccnt = pick_balanced(core, 20)
PROV, pcnt = pick_balanced(shortlist, 20, ccnt)
SLATE = CORE + PROV

def assign(teams, k):
    order = sorted(teams, key=lambda r: (r["archetype"] or "zz", -(r["headroom"] or 0)))
    b = [[] for _ in range(k)]
    for i, r in enumerate(order): b[i % k].append(r)
    return b
A5, A20 = assign(SLATE, 5), assign(SLATE, 20)

def nm(r):
    return r["stem"] or r["sha"]
def tag(r):
    return ("curated" if r["curated"] else "POOL")

json.dump({"_meta": dict(meta, structure={
              "core_20": "eligible teams with a fixed-reference estimate at n>=150",
              "provisional_20": "n=40 screen only — MUST be re-screened on FRESH games before selection",
              "archetype_cap": f"<= {CAP:.0%} of the 40-team slate per class"}),
           "slate_40": [r["sha"] for r in SLATE],
           "core_20": [r["sha"] for r in CORE],
           "provisional_20": [r["sha"] for r in PROV],
           "screen_shortlist": [r["sha"] for r in shortlist],
           "unmeasured_nominations": [r["sha"] for r in unmeas[:40]],
           "archetype_counts": dict(Counter(r["archetype"] for r in SLATE)),
           "assignment_5x8": [[r["sha"] for r in b] for b in A5],
           "assignment_20x2": [[r["sha"] for r in b] for b in A20],
           "teams": rows},
          open(f"{OUT}/team_slate_40.json", "w"), indent=1)

L = []
w = L.append
c = meta["coverage"]; cal = meta["calibration"]
w("# ai_v12 — the 40-team candidate slate\n")
w("**Status: a SELECTION INPUT, not a launch order.** It is built to be read the day the rev-4")
w("shape discriminator lands, so the next fleet can be launched the same day. Nothing here has been")
w("trained; no battles were played to produce it.\n")
w("Sources are existing artifacts only. Every number carries the scale it was measured on and the")
w("n behind it; an unmeasured team is listed as UNMEASURED with n=0 and is **never imputed**.\n")
w("### The four things a reader needs before anything else\n")
w("1. **The curated-32 gate BINDS, and 40 > 32.** After rev-4 only **8** curated teams are")
w("   untaught. A 40-team fleet needs the sample set widened, the research override taken, or the")
w("   ambition cut to ≤32 — §1.")
w(f"2. **Coverage is thin: {c['gen15_measured'] + c['rev1f_shifted']} of {c['pool_total']} teams have a fixed-reference estimate**")
w("   (§2). Half the slate is nominated rather than measured and says so.")
w("3. **The wide training-time table does not substitute for one** — it was tested as a predictor")
w("   and came back ρ ≈ 0.4–0.5 (§3). It nominates; it never ranks.")
w("4. **The ranking is a ranking on `−baseline_WR`**, because the ceiling is treated as constant.")
w("   That is licensed by the ceiling reframe, and its limit is stated in §4.\n")

w("## 1. The constraint that shapes everything: the curated-32 cap\n")
w("`--exploiter` refuses a trainee team that is not one of the curated `data/teams/sample/` teams.\n")
w("| | |")
w("|---|---|")
w("| enforced in | `src/main/train/matchup_setup.py:127-140` — FATAL, `TrainExitCode.FATAL_CONFIG` |")
w("| predicate | `agents.training.matchup_spec.validate_exploiter_trainee_is_sample` (`matchup_spec.py:265`) |")
w("| scope | `mix_kind == 'exploiter'` with a PINNED trainee (`pinned` / `pin_biased` / `pin_multi`); a generalist exploiter is out of scope |")
w("| curated set size | **32** `.txt` files in `data/teams/sample/` (`teams.json` is the manifest, not a team) |")
w("| documented escape hatch | `--allow-nonsample-trainee` — prints a warning and skips the gate; its own comment says *\"Use for capacity studies … NOT for a teacher you intend to distil as-is\"* |\n")
w("**It binds, and it has already bitten once.** The rev-3 coverage picks were drawn from the full")
w("719-team pool, REJECTED by this gate, and re-picked from the curated set — and the admission")
w("harness's team dict was never updated, which is the 16/36-cell mid-flight save recorded at")
w("61608ac. `probes/coverage_sweep.json`'s three `picks` (`a05a190b50`, `27b7b27e8a`, `f9f8d0608a`)")
w("are still pool teams; the teams actually trained are the three curated `COV_*`.\n")
w("### The arithmetic that caps the ambition\n")
w("| set | n | note |")
w("|---|---|---|")
w("| curated teams | 32 | the whole legal exploiter-trainee universe |")
w("| taught by F5a–e + F6a–f (incl. the 9 meter) | 12 | rev-2 and rev-3 fleets |")
w("| newly taught by rev-4 R4S3a/b/c (frozen argvs) | 12 | 24 distinct once rev-4 lands |")
w("| **curated and still untaught after rev-4** | **8** | `ce35b736 · b89e1e37 · 9909f2e9 · e11829f0 · f7ba5702 · dbf81d8e · a04c29cf · 9f27f5d3` |")
w("| held out (`probes/offpin_{0,1}.txt`) | 2 | pool teams, not curated — the off-slice transfer instrument |\n")
w("> **A 40-team revolution cannot be built from the curated set.** Even ignoring the exclusions")
w("> entirely, 32 < 40; honoring them leaves **8**. Three ways forward, none of them free, and this")
w("> is a NAMED GAP for the design rather than something S1 fixes:")
w(">")
w("> 1. **Promote vetted teams into `data/teams/sample/`** — what the refusal message itself")
w(">    advises (*\"promote this one into the sample set first if it is proven\"*). ~32 promotions")
w(">    are needed. The gate's stated purpose is that a teacher pilots a *tournament-proven* team, so")
w(">    a promotion needs a vetting criterion that does not yet exist in the tree.")
w("> 2. **`--allow-nonsample-trainee`** — one flag, zero code. But its own comment excludes exactly")
w(">    this use (a teacher you intend to distil), so taking it means overriding a documented")
w(">    intent, deliberately and in writing.")
w("> 3. **Shrink the ambition to ≤ 32** — e.g. 4 teachers × 8, which still tests breadth against")
w(">    rev-3's 6×2 and fits the count-dominates N≤10 bound.\n")

w("## 2. Coverage — what is actually measured\n")
w(f"Pool: **{c['pool_total']}** teams. Fixed-reference piloting estimates exist for")
w(f"**{c['gen15_measured'] + c['rev1f_shifted']} of {c['pool_total']}** ({(c['gen15_measured']+c['rev1f_shifted'])/c['pool_total']:.1%}).\n")
w("| evidence tier | teams | n / team | what it is |")
w("|---|---|---|---|")
w(f"| **A — gen-15 direct** | {c['gen15_measured']} | 150–400 | R2-ACTION final pilots the pinned team; opponent draws the pool |")
w(f"| **B — rev-1 shifted** | {c['rev1f_shifted']} | 200–400 | rev-1 final pilots; shifted by the measured generation offset |")
w(f"| **C — n=40 screen** | {c['screen_only_n40']} | 40 | nomination only; quantized to ±0.025 and selection-prone |")
w(f"| **UNMEASURED** | {c['unmeasured']} | 0 | no fixed-reference measurement exists |")
w(f"| *(training-time WR, any tier)* | {c['train_wr_available_n_ge_30']} | ≥30 | a **different scale** — see §3 |\n")

w("## 3. Three scales, and why only two of them may be pooled\n")
w("| scale | pilot | opponent | comparable to the ~0.69 ceiling? |")
w("|---|---|---|---|")
w("| `gen15` | R2-ACTION final | rev-1@24M *or* R2-ACTION final, drawing the 719-pool | **yes** — the ceiling is the set-mean of the rev-3 teacher cells on this harness |")
w("| `rev1f` | rev-1 final | rev-1@24M | after the offset below |")
w("| `train` | the run itself, mid-training | a MOVING self-play pool | **no** |\n")
w("**Two calibrations were measured rather than assumed:**\n")
w("* **Opponent equivalence.** On the 3 coverage teams that carry both, R2-ACTION-final and")
w(f"  rev-1@24M as the *opponent* differ by **{cal['opponent_offset_R2ACTIONfinal_minus_rev1at24M']:+.4f}**")
w(f"  (n={cal['n_opponent_pairs']} teams, 300–400 games each) — indistinguishable from zero, so the two")
w("  gen-15 sub-harnesses are pooled.")
w(f"* **Generation offset.** Over **{cal['n_teams']}** curated teams carrying both pilots, gen-15 minus")
w(f"  rev-1-final is **{cal['gen15_minus_rev1f']:+.4f}**. Tier-B rows are shifted by exactly this.")
w("  *This is itself a finding:* on these teams the current generation pilots **no better than**")
w("  rev-1 did — the same 'meter teams are mined out / the fold redistributes' reading the rev-3")
w("  recap (ade78c1) arrived at from the other direction.\n")
w("**The training-time table was tested as a predictor and largely FAILED.** `team_win_rates`")
w("(the task-#18 machinery, `TeamWinRateCallback` → `metadata.json`) is by far the widest source —")
w("717 teams in `ai_v9_58_R2CTRL_0827`, 716 in `ai_v9_29_rev1_0823` at a median 582 pool games —")
w("but rank-correlating it against the fixed-reference estimates on the overlap gives")
w(f"**ρ = {cal['spearman_train_vs_fixed_reference_curated']} (n={cal['n_curated']}, SE ≈ {cal['se_curated']})**")
w("on curated teams and")
w(f"**ρ = {cal['spearman_train_vs_fixed_reference_pool']} (n={cal['n_pool']}, SE ≈ {cal['se_pool']})**")
w("on pool teams — i.e. **z ≈ 2.1 and z ≈ 1.2**. The curated arm is borderline; the pool arm is")
w("nothing. Even taking the larger at face value, ρ ≈ 0.5 accounts for a quarter of the rank")
w("variance, and the *levels* are not even close (the same teams read 0.55–0.80 on the training")
w("scale and 0.37–0.56 on the fixed-reference one). The artifact's own `notes` field predicted this")
w("— *\"a low win rate with team T may mean we pilot T badly OR that T is weak\"* — and a moving")
w("self-play opponent plus a bot-heavy early curriculum compound it. **So it nominates; it never")
w("ranks, and it is never pooled with a fixed-reference number.** (The code-rank lesson applied:")
w("gate a signal on *does it PREDICT?*, not on *is it available?*)\n")

w("## 4. Headroom\n")
w(f"`headroom(T) = {meta['ceiling_used']} − baseline_WR(T)`, both on the gen-15 scale.\n")
w("The ceiling is the set-mean of the **12 rev-3 teacher ABSOLUTE cells** in `r3_admission.json`")
w(f"(observed per-team range **{meta['ceiling_per_team_observed_range'][0]}–{meta['ceiling_per_team_observed_range'][1]}**,")
w("matching the admission record). The ceiling-reframe result (61608ac) is what makes this")
w("estimable with no teacher trained: teachers land at ~0.69 *regardless* of where the target")
w("starts (budget +67% ⇒ +0.0019 z=0.16; target start 0.46–0.61), so headroom is a property of the")
w("**baseline**, not of the teacher.\n")
w("⚠️ **The per-team ceiling spread is ±9pp and is NOT modeled here.** A team whose true ceiling is")
w("0.59 and whose baseline is 0.37 has 22pp of headroom, not 32pp. Ranking on a constant ceiling is")
w("therefore a ranking on `−baseline_WR`; it is defensible only because the baseline spread")
w("(0.35–0.65) is ~1.7× the ceiling spread. Treat the ordering as robust and the magnitudes as")
w("upper-ish bounds.\n")

w("## 5. The slate\n")
w("Split by evidence, deliberately — a 40-row table where half the rows are n=40 would read as one")
w("measurement.\n")
w("### 5a. CORE 20 — measured, launchable ordering\n")
w("| # | sha | file (under `data/teams/`) | set | archetype | baseline WR | n | 95% CI | headroom | tier |")
w("|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(CORE, 1):
    ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
    w(f"| {i} | `{r['sha']}` | `{(r['path'] or '?').replace('data/teams/','')}` | {tag(r)} | {r['archetype']} | "
      f"{r['baseline_wr']:.3f} | {r['n']} | {ci} | **{r['headroom']:.3f}** | {r['evidence_tier'][0]} |")
w("")
w(f"Of these, **{sum(1 for r in CORE if r['curated'])}** are curated (legal today); "
  f"**{sum(1 for r in CORE if not r['curated'])}** are pool teams requiring §1's widening.\n")
w("### 5b. PROVISIONAL 20 — nominated, **must be re-screened**\n")
w("Every row below rests on an n=40 screen (±0.08 at 1 SE) plus the weak training-time signal.")
w("The listed order is a nomination order, **not a measurement**; adjacent rows are tied within noise.\n")
w("| # | sha | file (under `data/teams/`) | archetype | screen WR (n=40) | headroom | train WR | train n |")
w("|---|---|---|---|---|---|---|---|")
for i, r in enumerate(PROV, 21):
    tw = "—" if r["train_wr_pool"] is None else f"{r['train_wr_pool']:.3f}"
    w(f"| {i} | `{r['sha']}` | `{(r['path'] or '?').replace('data/teams/','')}` | {r['archetype']} | {r['baseline_wr']:.3f} | {r['headroom']:.3f} | {tw} | {r['train_n_pool']} |")
w("")
w("> **Rev-4's 12 teams are excluded CONDITIONALLY and the exclusion is reversible.** Its argvs are")
w("> frozen but its teachers are not folded; if rev-4 is abandoned or its fleet is not admitted,")
w("> those 12 return to the candidate pool. They are tagged `rev4_pending:<arm>` in the JSON's")
w("> per-team `excluded` list, separately from `taught:<teacher>`, precisely so the two can be")
w("> reversed independently.\n")
w("### 5c. UNMEASURED — candidates too, ranked separately, n=0\n")
w(f"**{c['unmeasured']}** eligible pool teams have no fixed-reference measurement. They are")
w("candidates on equal footing once screened — a team is not weak because nobody has measured it.")
w("The 20 with the lowest training-time pool win rate (the weak nomination signal, ρ≈0.4) are:\n")
w("| sha | file (under `data/teams/`) | archetype | train WR (pool) | train n |")
w("|---|---|---|---|---|")
for r in unmeas[:20]:
    tw = "—" if r["train_wr_pool"] is None else f"{r['train_wr_pool']:.3f}"
    w(f"| `{r['sha']}` | `{(r['path'] or '?').replace('data/teams/','')}` | {r['archetype']} | {tw} | {r['train_n_pool']} |")
w("")
w("### Source-folder spread — checked, because archetype is not the only diversity axis\n")
w("`data/teams/others/` is five AUTHOR folders. A fleet drawn deep into one author inherits that")
w("author's building habits as surely as it would inherit a single archetype, and this tree has the")
w("precedent: `yak_attack` was 66% of team draws before the 1601→719 dedupe. So it was measured")
w("rather than assumed:\n")
w("| folder | in slate | slate share | pool share |")
w("|---|---|---|---|")
def _au(x):
    q = (x["path"] or "").split("/")
    return q[3] if len(q) > 4 else "sample"
_ps = Counter(_au(x) for x in rows); _ss = Counter(_au(x) for x in SLATE)
for _k, _v in _ss.most_common():
    w(f"| `{_k}` | {_v} | {_v/len(SLATE):.0%} | {_ps[_k]/len(rows):.0%} |")
w("")
w("**No cap was needed.** The largest folder (`giraffe`) is 38% of the slate against 51% of the")
w("pool — the headroom ranking already draws it BELOW proportion, so an author cap would only")
w("displace measured teams for nominated ones. Re-check this table if the slate is re-cut; the")
w("`path` field in the JSON carries the folder for exactly that purpose.\n")
w("### Archetype spread\n")
w("| class | in slate | share |")
w("|---|---|---|")
cnt = Counter(r["archetype"] for r in SLATE)
for k, v in cnt.most_common():
    w(f"| {k} | {v} | {v/len(SLATE):.0%} |")
w(f"\nMax class share **{max(cnt.values())/len(SLATE):.0%}** ≤ the 40% cap. "
  "(Pool-wide the classes run 12.5–28.7%, so the cap binds on `balance` alone.)\n")

w("## 6. Draft assignments\n")
w("Both drafts deal the same 40 teams; only the shape differs. Teams are ordered by archetype then")
w("headroom and dealt round-robin, so each teacher gets a spread rather than a monoculture — the")
w("v8 10-team structure logic (a teacher that only ever sees stall learns stall, and the fold")
w("inherits that narrowness).\n")
w("### 6a. WIDE — 5 teachers × 8 teams (the endorsed shape if rev-4's 3×8 wins)\n")
for i, b in enumerate(A5, 1):
    cn = Counter(x["archetype"] for x in b)
    w(f"**T{i}** ({', '.join(f'{k}×{v}' for k, v in sorted(cn.items()))})  ")
    w("```\n--trainee-teams " + ",".join(x["path"] or x["sha"] for x in b) + "\n```\n")
w("### 6b. NARROW — 20 teachers × 2 teams (if the discriminator picks narrow)\n")
w("| teacher | teams | archetypes |")
w("|---|---|---|")
for i, b in enumerate(A20, 1):
    w(f"| N{i} | `{(b[0]['path'] or b[0]['sha']).replace('data/teams/','')}`, `{(b[1]['path'] or b[1]['sha']).replace('data/teams/','')}` | {b[0]['archetype']} / {b[1]['archetype']} |")
w("")
w("### 6c. ⚠️ The budget arithmetic, which is the same for BOTH shapes\n")
w("Read from the frozen argvs, not assumed (`--steps` minus rev-1 final's 25,067,760):\n")
w("| fleet | teachers × teams | per teacher | **per team** | total |")
w("|---|---|---|---|---|")
w("| rev-3 (F6a–f) | 6 × 2 | 5.0M | **2.5M** | 30.0M |")
w("| rev-4 (R4S3a–c) | 3 × 8 | 10.0M | **1.25M** | 30.0M |")
w("| a 40-team fleet at the SAME total | 5 × 8 *or* 20 × 2 | 6.0M / 1.5M | **0.75M** | 30.0M |")
w("| a 40-team fleet at rev-4's per-team budget | 5 × 8 *or* 20 × 2 | 10.0M / 2.5M | **1.25M** | 50.0M |\n")
w("**The shape choice does not change the per-team budget** — 5×8 and 20×2 deal the same 40 teams,")
w("so at a fixed total they both land on 0.75M/team. What the discriminator decides is *breadth per")
w("teacher*, and that is orthogonal to this row. The real hazard is shared:\n")
w("> **0.75M/team is BELOW every budget ever measured here.** The budget-invariance result")
w("> (61608ac §2: +67% ⇒ +0.0019, z=0.16) was measured between **1.5M and 2.5M**, and rev-4 already")
w("> steps outside it at 1.25M. Invariance over a tested band is not licence to extrapolate below")
w("> it — the ceiling account predicts a floor somewhere, and nothing has looked for it. **Either")
w("> budget the 40-team fleet at 50M total (1.25M/team, matching rev-4), or measure the 0.75M point")
w("> first.** Rev-4's own absolute rows give that check for free: if its 1.25M teachers land at the")
w("> same ~0.69 as rev-3's 2.5M ones, the band widens downward by one point of evidence — read it")
w("> before committing 40 forks to 0.75M.\n")
w("## 7. What needs a FRESH screen before final selection\n")
w("**The rule, from the withdrawn §7 seniority split (61608ac):** selecting a team on an estimate")
w("and then reporting that same estimate is selection-on-the-minimum. The measured regression-to-mean")
w("there was **+0.061** — larger than most of the differences this slate ranks on. So:\n")
w("* **The 20 PROVISIONAL rows (§5b) and any UNMEASURED promotion (§5c) need a screen at n ≥ 200,**")
w("  played on **fresh games** with a seed family disjoint from `coverage_sweep`'s")
w("  (`52000 + idx`), `headroom_screen`'s (`1000 + 9 + idx`) and the admission's (`41000 + idx`).")
w("  Reusing any of those makes the confirm a re-report of the selection.")
w("* **`coverage_sample.json` is a live instance of the failure it warns about**: its `screen` and")
w("  `confirm` blocks are byte-identical for all 23 rows (`screen_wr == wr` exactly), so its")
w("  \"two-stage\" structure collapsed to one measurement. Its numbers are usable as a single n=200")
w("  estimate — which is how they are used here — but **not** as a confirmation of anything.")
w("* **`coverage_sweep.json` did it right**: n=40 screen → an INDEPENDENT n=200 confirm on the")
w("  bottom-12, and the confirm values move (0.275→0.380, 0.425→0.370). Those 12 rows are the")
w("  strongest pool-team evidence in the tree and are used at face value.")
w("* **The 8 untaught curated teams are being measured RIGHT NOW.** `probes/run_headroom.sh` is")
w("  running `headroom_screen.py` over the 20 non-taught curated teams at n=150 against R2-ACTION")
w("  final; all 8 sit at screen indices 8–19. **Re-run this slate when it finishes.** Note what")
w("  that actually buys: n=150 DIRECT is not a larger sample than the n=200 SHIFTED number they")
w("  carry here — it is a sample that needs no generation-offset assumption. Prefer the direct")
w("  one for that reason, not for precision, and treat agreement between the two as the real")
w("  signal (the offset was fitted on these same teams, so a large disagreement would indict it).")
w("* **The ceiling itself deserves one manipulation before 40 teachers are spent on it.** Every")
w("  headroom number is `0.69 − baseline`; if F6-CURR's absolute row (requested at 61608ac §9)")
w("  shows a curriculum that lifts 0.69, the whole ranking is against a moving bar.\n")

w("## 8. Provenance\n")
_WHAT = {
    "r3_admission": "the rev-3 admission battery — the `target` baseline (12 teams, n=400) AND the "
                    "12 teacher ABSOLUTE cells the ~0.69 ceiling is the mean of",
    "fleet_admission": "the rev-2 battery — the `rev1final` pilot row on the 9 meter teams (n=400)",
    "pilot_R2ACTION_n300": "current-generation piloting of the 9 meter teams (n=300)",
    "cov_R2ACTION": "current-generation piloting of the 3 coverage teams (n=300)",
    "cov_rev1fin": "the rev-1-final companion of the row above — the generation-offset anchor",
    "headroom_screen": "the LIVE n=150 screen over the 20 non-taught curated teams",
    "coverage_sample": "the 23 non-meter curated teams at n=200 under the rev-1 pilot",
    "coverage_sweep": "80 POOL teams at n=40 + an independent n=200 confirm on the bottom 12",
    "team_win_rates": "the task-#18 per-team tracking — 658 teams, nomination signal only",
    "archetypes": "the pace-class label and tags for all 719 teams",
}
w("| artifact | what it contributed |")
w("|---|---|")
for k, v in meta["sources"].items():
    w(f"| `{v.split(' (')[0]}` | {_WHAT.get(k, k)} |")
w("")
w(f"**Freshness of the live source:** {len([x for x in hs if x not in ('_meta', 'POOLED')])} of 20 "
  "`headroom_screen` rows had landed when this was built.\n")
w("The two admission artifacts and the piloting/coverage probes live in a **session-scoped job")
w("directory** (`~/.claude/jobs/1046b1d6/tmp/probes`), not in the repo. Every number that this slate")
w("depends on is copied into `team_slate_40.json` so the slate survives that directory's deletion.\n")
w("**Re-running this slate.** `designs/ai_v12/team_slate_build.py` regenerates both files from the")
w("artifacts above in ~5 s (read-only; no models, no battles). Run it after the in-flight headroom")
w("screen finishes — it promotes the 8 untaught curated teams from tier B to tier A and re-fits the")
w("generation offset.\n")
w("```bash")
w("export PYTHONPATH=$PYTHONPATH:src")
w("python designs/ai_v12/team_slate_build.py")
w("```\n")
w("**Key-convention warning, recorded because it cost a rejoin.** `coverage_sample.py` fingerprints")
w("a team with `sha1(text)` on the **UNSTRIPPED** text, while `team_archetypes.team_sha`,")
w("`MatchupSpec` pins, `TeamWinRateCallback` and this slate all use `sha1(text.strip())[:10]`. The")
w("tell is that all 23 of its rows carry `\"class\": \"?\"` — its own archetype lookup silently missed.")
w("The 23 identities were recovered by re-hashing the curated files under both conventions (23/23")
w("matched on `raw`). Sixth specimen of the recorded-vs-effective derived-key genre.\n")
open(f"{OUT}/team_slate_40.md", "w").write("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
print("core", len(CORE), "prov", len(PROV), "arch", dict(cnt))
