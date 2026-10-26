"""gen3_team_pack_oracle_v1 — the pack oracle (``pack_oracle.js``) that every ``Gen3Teambuilder`` construction runs
over its pool / bias / mix packs (2026-10-26 GIGO audit C-2 / C-3). Real Node, real Showdown: the oracle must SEE a
pack that is a legal team but not the authored one, and pass the authored one."""
import pytest

from utils.bridge.team_validator import PackIdentityError, check_pack_identity, pack_identity_defects
from utils.team_packing import Teambuilder

pytestmark = pytest.mark.integration

_PASTE = """Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Frustration
- Curse
- Rest
- Sleep Talk

Salamence @ Choice Band
Ability: Intimidate
Hidden Power: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Claw
- Earthquake
- Rock Slide
- Double-Edge"""


def test_the_oracle_passes_our_pack_of_the_authored_team():
    from utils.gen3_utils import fix_gen3_hp_ivs
    ours = Teambuilder.join_team(fix_gen3_hp_ivs(Teambuilder.parse_showdown_team(_PASTE)))
    assert pack_identity_defects("gen3ou", [_PASTE], [ours]) == [[]]


def test_the_oracle_sees_the_two_defects_the_audit_found():
    """The pre-fix packs: happiness omitted (=255) on the Frustration Snorlax, IVs left at 31 on the HP-Water line."""
    stale = Teambuilder.join_team(Teambuilder.parse_showdown_team(_PASTE)).split("]")
    snorlax = stale[0].split("|")
    snorlax[11] = ""                     # the misc field as the old packer wrote it: no happiness
    stale[0] = "|".join(snorlax)
    sala = stale[1].split("|")
    sala[8] = ""                         # all-31 IVs
    stale[1] = "|".join(sala)
    (d,) = pack_identity_defects("gen3ou", [_PASTE], ["]".join(stale)])
    assert any("happiness: Showdown plays 0, our pack 255" in x for x in d), d
    assert any("(salamence) ivs" in x for x in d), d
    with pytest.raises(PackIdentityError, match="gen3_team_pack_oracle_v1"):
        check_pack_identity("gen3ou", [_PASTE], ["]".join(stale)], "the test pool")
