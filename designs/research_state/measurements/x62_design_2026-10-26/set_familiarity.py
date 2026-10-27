"""X62 design (``designs/endstate/design_x62_team_recombination.md`` §1): how FAMILIAR each training corpus's and each
panel's mon sets are to P*'s pool, and how far each team sits from its nearest pool team. Reads committed files only;
plays nothing, writes nothing.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 \
        python designs/research_state/measurements/x62_design_2026-10-26/set_familiarity.py > set_familiarity.log

A SET is ``gen3_team_overlap_v1``'s mon key (species + sorted moves, Hidden Power typed; ``main.anchors.team_overlap``).
The POOL is the 719 teams P* trained on (``main.anchors.session.pool_team_texts``). Per team: ``n_pool`` = how many of
its six sets occur ANYWHERE in the pool; ``max_shared`` = the most sets it shares with ONE pool team. A team is
RECOMBINED-LIKE when ``n_pool >= 5`` and ``max_shared <= 2`` (familiar sets in a new six: rc72's construction).
"""
import collections

from main import heldout_panels as HPN
from main.anchors import team_overlap as TO
from main.anchors.session import load_team_texts, pool_team_texts
from utils import team_corpora as TC
from utils.paths import repo_path

RC72 = "designs/research_state/measurements/hctx_rc72_2026-10-26/teams_rc72"


def main() -> None:
    pool = [TO.parse_team(t, "pool_%d" % i) for i, t in enumerate(pool_team_texts())]
    pool_sets = {k for t in pool for k in t.key}
    pool_species = {s for t in pool for s in t.species6}
    ref = TO.Reference(pool)
    print(f"pool: {len(pool)} teams, {len(pool_sets)} distinct sets, {len(pool_species)} species")
    sets_by_species = collections.defaultdict(set)
    team_share = collections.Counter()
    pairs = collections.Counter()
    for t in pool:
        for k in t.key:
            sets_by_species[k[0]].add(k)
        team_share.update(t.species6)
        ks = sorted(t.key, key=str)
        pairs.update((a, b) for i, a in enumerate(ks) for b in ks[i + 1:])
    print(f"species with >= 2 sets: {sum(1 for v in sets_by_species.values() if len(v) >= 2)}; "
          f"set pairs co-occurring in >= 5 pool teams: {sum(1 for v in pairs.values() if v >= 5)}; "
          "top species by pool-team share: " + ", ".join(
              f"{s} {c / len(pool):.3f}" for s, c in team_share.most_common(5)))
    print("name | teams | slot share with a pool set | species in pool | n_pool hist | max_shared hist | "
          "recombined-like (n_pool>=5, max<=2) | (n_pool>=4, max<=2)")
    sources = []
    for name in sorted(TC.CORPORA):
        c = TC.load_corpus(name, "ABC" if name == "w_forum_v1" else "ALL")
        sources.append((name, TO.parse_items(list(zip(c.ids, c.texts)))))
    for p in ("home72", "forum72", "heldout72_hpfix", "p72"):
        sources.append((p, TO.parse_items(HPN.panel_team_texts(p))))
    sources.append(("rc72", TO.parse_items(load_team_texts(repo_path(RC72)))))
    for name, teams in sources:
        n_hist, m_hist = collections.Counter(), collections.Counter()
        fam = slots = sp = rec5 = rec4 = 0
        for t in teams:
            n = sum(1 for k in t.key if k in pool_sets)
            m, _ = ref.max_shared(t)
            n_hist[n] += 1
            m_hist[m] += 1
            fam += n
            slots += len(t.key)
            sp += sum(1 for s in t.species6 if s in pool_species)
            rec5 += n >= 5 and m <= 2
            rec4 += n >= 4 and m <= 2
        print(f"{name} | {len(teams)} | {fam / slots:.3f} | {sp / slots:.3f} | {dict(sorted(n_hist.items()))} | "
              f"{dict(sorted(m_hist.items()))} | {rec5} | {rec4}")


if __name__ == "__main__":
    main()
