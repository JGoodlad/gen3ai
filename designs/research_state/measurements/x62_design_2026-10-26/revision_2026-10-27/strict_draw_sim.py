"""X62 revision (``designs/endstate/design_x62_team_recombination.md`` §2, RESPONSE M1 / m2 / m8): the review's
Python-only prototype of the ``x62_recomb_v1`` draw (``../review_2026-10-27/recomb_draw_sim.py``) re-run under the
revised rules. Reads committed files only; plays nothing, writes nothing, runs no Node validator.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 \
        python .../revision_2026-10-27/strict_draw_sim.py [N] > strict_draw_sim.log

The revised rules, beside the review's prototype (rc72b's ``draw_team``, the pool's species profile, ``<= 2`` with
any pool team, radius 3 against every held-out panel + n1p72 + rc72b, ``<= 4`` within the corpus):
- **STRICT Hidden Power at the slot** (PREREG Amendment 2's rule, emulated from the text): a pool block whose Hidden
  Power names a type other than Dark and carries no IV line is never drawn (Dark at 70 is what all-31 IVs give, so
  that one plays as written). The builder's ``hp_set_check`` stays the authority; this sizes its effect.
- **No exact species six** of a pool team (the review's m2).
Measured: what STRICT removes from the draw (pool mons, distinct sets, species), the species and set exposure against
the pool's occurrences, the within-corpus histogram, the rc72b distance, the exact-species-six refusals, and the
NEAR-DUPLICATES against the other breadth corpora (D1 ``w_forum_v1@ABC``, D3 ``d3_setmut_v1``, N1-P ``n1_proc_v1``;
the review's m8), reported, not refused.
"""
import collections
import importlib.util
import re
import sys
import time

from main import heldout_panels as HPN
from main.anchors import team_overlap as TO
from main.anchors.session import load_team_texts, pool_team_texts
from utils import team_corpora as TC
from utils.paths import repo_path

MEAS = "designs/research_state/measurements"
spec = importlib.util.spec_from_file_location(
    "build_rc72", repo_path(MEAS, "hctx_rc72_2026-10-26", "scripts", "build_rc72.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
spec2 = importlib.util.spec_from_file_location(
    "recomb_draw_sim", repo_path(MEAS, "x62_design_2026-10-26", "review_2026-10-27", "recomb_draw_sim.py"))
R = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(R)

HP_RE = re.compile(r"hidden power\s*\[?\s*([a-z]+)", re.I)


def strict_bad(block: str) -> bool:
    """True iff the block names a Hidden Power type other than Dark and carries no IV line."""
    low = block.lower()
    if "hidden power" not in low or "ivs:" in low:
        return False
    m = HP_RE.search(block)
    return not (m and m.group(1).lower() == "dark")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_target = int(args[0]) if args else 12000
    t0 = time.time()
    pool_texts = sorted(pool_team_texts(), key=B.sha_text)
    pool = [TO.parse_team(t, "pool_" + B.sha_text(t)[:12]) for t in pool_texts]
    occ_all = B.pool_occurrences(pool_texts)
    n_all = sum(len(v) for v in occ_all.values())
    occ = {s: [x for x in v if not strict_bad(x[0])] for s, v in occ_all.items()}
    lost_sp = sorted(s for s, v in occ.items() if not v)
    occ = {s: v for s, v in occ.items() if v}
    n_ok = sum(len(v) for v in occ.values())
    sets_all = {m.key for v in occ_all.values() for _b, m, _s in v}
    sets_ok = {m.key for v in occ.values() for _b, m, _s in v}
    print(f"pool: {len(pool)} teams, {n_all} mons, {len(sets_all)} distinct sets, {len(occ_all)} species")
    print(f"STRICT removes {n_all - n_ok} of {n_all} pool mons ({(n_all - n_ok) / n_all:.3f}); drawable sets "
          f"{len(sets_ok)} of {len(sets_all)} (lost {len(sets_all) - len(sets_ok)}); species lost {lost_sp}")
    teams_touched = sum(any(strict_bad(b) for b in B.split_blocks(t)) for t in pool_texts)
    print(f"pool teams holding >= 1 strict-bad block: {teams_touched} of {len(pool_texts)}")
    profile = {s: c for s, c in collections.Counter(m.species for t in pool for m in t.mons).items() if s in occ}
    lift = B.lift_table(sorted(profile))
    held = {p: TO.Reference(TO.parse_items(HPN.panel_team_texts(p))) for p in HPN.panel_names(held_out_only=True)}
    held["n1p72"] = TO.Reference(TO.parse_items(load_team_texts(repo_path(B.N1P72_DIR))))
    rc72b = TO.parse_items(load_team_texts(repo_path(MEAS, "hctx_rc72_2026-10-26", "teams_rc72b")))
    held["rc72b"] = TO.Reference(rc72b)
    refs = {"train": {"pool": TO.Reference(pool)}, "held": held}
    pool_six = {frozenset(t.species6) for t in pool}

    fails = collections.Counter()
    acc = []
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
            if sum(R.boost_passer(m) for m in mons) >= 2:
                fails["one_boost_passer"] += 1
                got = None
                continue
            if frozenset(m.species for m in mons) in pool_six:
                fails["pool_species_six"] += 1
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
    print(f"accepted {len(acc)} of {i} indices in {time.time() - t0:.0f} s; refusals {dict(sorted(fails.items()))}")
    print(f"within-corpus max shared with an EARLIER accepted team: {dict(sorted(within_hist.items()))}")
    rc_ref = TO.Reference(rc72b)
    print("max shared with ONE rc72b team: "
          f"{dict(sorted(collections.Counter(rc_ref.max_shared(t)[0] for t in acc).items()))}")
    sp_c = collections.Counter(m.species for t in acc for m in t.mons)
    sp_p = collections.Counter(m.species for t in pool for m in t.mons)
    set_c = collections.Counter(m.key for t in acc for m in t.mons)
    set_p = collections.Counter(m.key for t in pool for m in t.mons)
    print(f"species TVD corpus vs pool {R.tvd(sp_c, sp_p):.3f}; set-exposure TVD corpus vs pool occurrences "
          f"{R.tvd(set_c, set_p):.3f}; distinct sets {len(set_c)} of {len(set_p)}")
    hp_p = sum(c for k, c in set_p.items() if any("hiddenpower" in str(x).replace(" ", "").lower() for x in k))
    hp_c = sum(c for k, c in set_c.items() if any("hiddenpower" in str(x).replace(" ", "").lower() for x in k))
    print(f"slots carrying Hidden Power: pool {hp_p / sum(set_p.values()):.3f} -> corpus {hp_c / sum(set_c.values()):.3f}")
    top = sp_p.most_common(8)
    print("top pool species, share of slots pool -> corpus: " + ", ".join(
        f"{s} {sp_p[s] / sum(sp_p.values()):.3f}->{sp_c[s] / sum(sp_c.values()):.3f}" for s, _ in top))
    print(f"exact pool species six in the corpus: {sum(frozenset(t.species6) in pool_six for t in acc)}")
    print("NEAR-DUPLICATES: max sets shared with ONE team of each breadth corpus (histogram; reported, not refused)")
    for name in sorted(TC.CORPORA):
        if name.startswith("x62"):
            continue
        c = TC.load_corpus(name, "ABC" if name == "w_forum_v1" else "ALL")
        ref = TO.Reference(TO.parse_items(list(zip(c.ids, c.texts))))
        d = collections.Counter(ref.max_shared(t)[0] for t in acc)
        print(f"  {name}: {dict(sorted(d.items()))}  (>= 5: {sum(v for k, v in d.items() if k >= 5)})")


if __name__ == "__main__":
    main()
