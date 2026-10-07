import collections
import sys

sys.path.insert(0, 'src')
from agents.gen3_data import team_legality as tl  # noqa: E402
from utils.team_sources import procedural_teams  # noqa: E402

n = int(sys.argv[1])
teams = procedural_teams(n)
bad = 0
rules = collections.Counter()
qc_sets = 0
for t in teams:
    sets = tl.parse_packed(t)
    assert len(sets) == 6, (len(sets), t[:200])
    qc_sets += sum(s.item == 'quickclaw' for s in sets)
    probs = tl.validate_team(sets)
    if probs:
        bad += 1
        for p in probs:
            rules[(p.rule, p.ladder_only, p.message.split(' ')[-1] if p.rule == 'banlist' else '')] += 1
print(f"procedural teams {n}: illegal {bad}; quick claw sets {qc_sets} of {6 * n}")
for k, v in rules.most_common():
    print(v, k)
