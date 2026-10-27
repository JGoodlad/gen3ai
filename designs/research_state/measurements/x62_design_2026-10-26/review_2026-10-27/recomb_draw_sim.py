"""X62 adversarial review (``designs/endstate/reviews/x62_review_2026-10-27.md``): a PYTHON-ONLY prototype of the
proposed ``x62_recomb_v1`` draw (design §2 steps 1-3), to check what the design asserts but has not built. Reads
committed files only; plays nothing, writes nothing, runs no Node validator.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 \
        python .../review_2026-10-27/recomb_draw_sim.py [N] [--no-rc72b-refusal] > recomb_draw_sim[_no_rc72b].log

The draw is rc72b's ``draw_team`` with the POOL's species counts as the profile (design §2 step 1), the pool-only
``<= 2`` rule, the held-out radius-3 rule (plus rc72b and n1p72), and the within-corpus ``<= 4`` cap applied in index
order. Step 4's validator is NOT run; the One Boost Passer Clause and the strict Hidden Power class are emulated
from the set text. Measured: acceptance, the within-corpus histogram (design FINDING 3), the species and the set
exposure against the pool's (design §2 "per-species exposure stays near the pool's", "a set is drawn as often as the
pool repeats it"), Baton Pass collisions, the no-IV-line Hidden Power share (PREREG Amendment 2), and the max number
of SPECIES shared with one pool team (a distance the design does not constrain), beside home72, F, H, P72 and rc72b.
"""
import collections
import importlib.util
import math
import random
import sys
import time

from main import heldout_panels as HPN
from main.anchors import team_overlap as TO
from main.anchors.session import load_team_texts, pool_team_texts
from utils.paths import repo_path

MEAS = "designs/research_state/measurements"
spec = importlib.util.spec_from_file_location(
    "build_rc72", repo_path(MEAS, "hctx_rc72_2026-10-26", "scripts", "build_rc72.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

BOOST = {'acidarmor', 'agility', 'amnesia', 'apicotberry', 'barrier', 'bellydrum', 'bulkup', 'calmmind', 'cosmicpower',
         'curse', 'defensecurl', 'dragondance', 'ganlonberry', 'growth', 'harden', 'howl', 'irondefense', 'liechiberry',
         'meditate', 'petayaberry', 'salacberry', 'sharpen', 'speedboost', 'starfberry', 'swordsdance', 'tailglow',
         'withdraw'}  # deps/pokemon-showdown data/rulesets.ts, One Boost Passer Clause


def boost_passer(m) -> bool:
    if "batonpass" not in m.moves:
        return False
    return sum(mv in BOOST for mv in m.moves) + (m.item in BOOST) + (m.ability in BOOST) >= 1


def no_iv_hp(block: str) -> bool:
    return "hidden power" in block.lower() and "ivs:" not in block.lower()


def tvd(a: collections.Counter, b: collections.Counter) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in set(a) | set(b))


def species_dist(teams, pool):
    idx = collections.defaultdict(list)
    for i, t in enumerate(pool):
        for s in t.species6:
            idx[s].append(i)
    out = collections.Counter()
    for t in teams:
        h = collections.Counter()
        for s in t.species6:
            for i in idx[s]:
                h[i] += 1
        out[max(h.values()) if h else 0] += 1
    return dict(sorted(out.items()))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_target = int(args[0]) if args else 12000
    t0 = time.time()
    pool_texts = sorted(pool_team_texts(), key=B.sha_text)
    pool = [TO.parse_team(t, "pool_" + B.sha_text(t)[:12]) for t in pool_texts]
    occ = B.pool_occurrences(pool_texts)
    profile = dict(collections.Counter(m.species for t in pool for m in t.mons))
    lift = B.lift_table(sorted(profile))
    held = {p: TO.Reference(TO.parse_items(HPN.panel_team_texts(p))) for p in HPN.panel_names(held_out_only=True)}
    held["n1p72"] = TO.Reference(TO.parse_items(load_team_texts(repo_path(B.N1P72_DIR))))
    rc72b = TO.parse_items(load_team_texts(repo_path(MEAS, "hctx_rc72_2026-10-26", "teams_rc72b")))
    if "--no-rc72b-refusal" not in sys.argv:
        held["rc72b"] = TO.Reference(rc72b)
    refs = {"train": {"pool": TO.Reference(pool)}, "held": held}
    pool_bp = sum(boost_passer(m) for t in pool for m in t.mons)
    pool_noiv = sum(no_iv_hp(b) for b, _m, _s in (x for v in occ.values() for x in v))
    print(f"pool: {len(pool)} teams, {sum(len(v) for v in occ.values())} mons, {len(profile)} species; "
          f"boost-passer mons {pool_bp}; teams with >= 2 boost passers {sum(sum(boost_passer(m) for m in t.mons) >= 2 for t in pool)}; "
          f"no-IV-line Hidden Power mons {pool_noiv}")

    fails = collections.Counter()
    acc, acc_blocks = [], []
    within_idx = collections.defaultdict(list)
    within_hist = collections.Counter()
    i = 0
    while len(acc) < n_target and i < 3 * n_target:
        got = None
        for a in range(B.MAX_ATTEMPTS):
            got = B.draw_team(i, a, profile, occ, lift, refs, fails)
            if got is None:
                continue
            mons = [m for _b, m, _s in got]
            if sum(boost_passer(m) for m in mons) >= 2:
                fails["one_boost_passer"] += 1
                got = None
                continue
            h = collections.Counter()
            for k in {m.key for m in mons}:
                for j in within_idx[k]:
                    h[j] += 1
            mx = max(h.values()) if h else 0
            if mx > 4:
                fails["within_gt4"] += 1
                got = None
                continue
            within_hist[mx] += 1
            break
        i += 1
        if got is None:
            fails["index_exhausted"] += 1
            continue
        t = TO.Team(label=f"x{i}", mons=tuple(m for _b, m, _s in got))
        for k in t.key:
            within_idx[k].append(len(acc))
        acc.append(t)
        acc_blocks.append([b for b, _m, _s in got])
        if len(acc) % 2000 == 0:
            print(f"  {len(acc)} accepted at index {i}, {time.time() - t0:.0f} s", flush=True)
    print(f"accepted {len(acc)} of {i} indices in {time.time() - t0:.0f} s; refusals {dict(sorted(fails.items()))}")
    print(f"within-corpus max shared with an EARLIER accepted team: {dict(sorted(within_hist.items()))}")
    rc_ref = TO.Reference(rc72b)
    print("max shared with ONE rc72b team (the design refuses only the RETIRED rc72): "
          f"{dict(sorted(collections.Counter(rc_ref.max_shared(t)[0] for t in acc).items()))}")

    sp_c = collections.Counter(m.species for t in acc for m in t.mons)
    sp_p = collections.Counter(m.species for t in pool for m in t.mons)
    set_c = collections.Counter(m.key for t in acc for m in t.mons)
    set_p = collections.Counter(m.key for t in pool for m in t.mons)
    print(f"species TVD corpus vs pool {tvd(sp_c, sp_p):.3f}; set-exposure TVD corpus vs pool occurrences "
          f"{tvd(set_c, set_p):.3f}; distinct sets {len(set_c)} of {len(set_p)}")
    ratio = sorted(((set_c[k] / sum(set_c.values())) / (set_p[k] / sum(set_p.values())), k) for k in set_p)
    q = [ratio[int(f * (len(ratio) - 1))][0] for f in (0.05, 0.25, 0.5, 0.75, 0.95)]
    print("set exposure ratio corpus/pool, quantiles 5/25/50/75/95 %: " + ", ".join(f"{x:.2f}" for x in q))
    top = sp_p.most_common(8)
    print("top pool species, share of slots pool -> corpus: " + ", ".join(
        f"{s} {sp_p[s] / sum(sp_p.values()):.3f}->{sp_c[s] / sum(sp_c.values()):.3f}" for s, _ in top))
    noiv = sum(no_iv_hp(b) for bl in acc_blocks for b in bl)
    print(f"corpus no-IV-line Hidden Power mons: {noiv} of {6 * len(acc)} ({noiv / (6 * len(acc)):.3f}); "
          f"teams with >= 1: {sum(any(no_iv_hp(b) for b in bl) for bl in acc_blocks)} of {len(acc)}")
    rows = [("x62_proto", acc), ("rc72b", rc72b)] + [
        (p, TO.parse_items(HPN.panel_team_texts(p))) for p in ("home72", "forum72", "heldout72_hpfix", "p72")]
    print("max SPECIES shared with ONE pool team (histogram); the design constrains SETS only:")
    for name, teams in rows:
        d = species_dist(teams, pool)
        ge5 = sum(v for k, v in d.items() if k >= 5) / len(teams)
        print(f"  {name}: {d}  (>= 5 species: {ge5:.3f})")


if __name__ == "__main__":
    main()
