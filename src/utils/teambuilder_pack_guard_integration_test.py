"""gen3_team_pack_oracle_v1 on the TRAINING path: every ``Gen3Teambuilder`` construction (the pool, the sample-bias
teams, the ``--team-mix`` corpus) refuses a pack that is not the authored team. The 2026-10-27 X35 pack-regime audit
(``designs/research_state/measurements/x35_pack_regime_2026-10-27/``) found that every X35-frame arm, P* and X26c
trained under the pre-``d8d674b9`` packer, which dropped an explicit ``Happiness: 0`` so Frustration hit at base power
1. Here that packer is re-created (the legacy ``happiness or ""``) and each builder path must RAISE on it; a revert of
any of the three ``check_pack_identity`` calls in ``utils.teambuilder`` (or of the oracle's happiness field) fails
this file. Real Node, real Showdown."""
import pytest

import utils.team_packing as team_packing
from utils.bridge.team_validator import PackIdentityError, check_pack_identity
from utils.team_packing import Teambuilder

pytestmark = pytest.mark.integration

# An EXPLICIT `Happiness: 0` beside Frustration (sfv2_0663's and forum72 sft431's shape), not the importer's
# "Frustration and no Happiness line" rule that pack_oracle_integration_test covers.
_HAPPINESS_0_FRUSTRATION = """Snorlax @ Leftovers
Ability: Thick Fat
Happiness: 0
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Frustration
- Curse
- Rest
- Sleep Talk

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Spikes
- Roar
- Toxic
- Drill Peck"""

_CLEAN = """Metagross @ Choice Band
Ability: Clear Body
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Rock Slide

Swampert @ Leftovers
Ability: Torrent
EVs: 252 HP / 4 Atk / 252 Def
Relaxed Nature
- Earthquake
- Surf
- Protect
- Toxic"""


@pytest.fixture
def legacy_packer(monkeypatch):
    """The packer as every pre-d8d674b9 pin had it: ``self.happiness or ""`` drops a 0."""
    monkeypatch.setattr(team_packing, "_packed_happiness", lambda h: h or "")


def _pack(paste):
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(paste))


def test_the_current_packer_plays_the_authored_team():
    assert _pack(_HAPPINESS_0_FRUSTRATION).split("]")[0].split("|")[11] == "0"
    check_pack_identity("gen3ou", [_HAPPINESS_0_FRUSTRATION], [_pack(_HAPPINESS_0_FRUSTRATION)], "the test pool")


def test_check_pack_identity_raises_on_the_legacy_pack_of_a_happiness_0_frustration_team(legacy_packer):
    stale = _pack(_HAPPINESS_0_FRUSTRATION)
    assert stale.split("]")[0].split("|")[11] == ""            # the old pack: no happiness, unpacked as 255
    with pytest.raises(PackIdentityError, match="happiness: Showdown plays 0, our pack 255"):
        check_pack_identity("gen3ou", [_HAPPINESS_0_FRUSTRATION], [stale], "the test pool")


@pytest.mark.parametrize("where, kwargs", [
    ("the team pool", {"teams": [_HAPPINESS_0_FRUSTRATION, _CLEAN]}),
    ("the bias teams", {"teams": [_CLEAN], "bias_teams": [_HAPPINESS_0_FRUSTRATION], "bias_prob": 0.1}),
    ("the --team-mix corpus", {"teams": [_CLEAN], "mix_teams": [_HAPPINESS_0_FRUSTRATION], "mix_prob": 0.5}),
])
def test_every_training_builder_path_refuses_the_legacy_pack(legacy_packer, where, kwargs):
    from utils.teambuilder import Gen3Teambuilder
    with pytest.raises(PackIdentityError, match=where.replace("-", r"\-")):
        Gen3Teambuilder(**kwargs)


def test_every_training_builder_path_accepts_the_current_pack():
    from utils.teambuilder import Gen3Teambuilder
    tb = Gen3Teambuilder([_HAPPINESS_0_FRUSTRATION, _CLEAN], bias_teams=[_HAPPINESS_0_FRUSTRATION], bias_prob=0.1,
                         mix_teams=[_HAPPINESS_0_FRUSTRATION], mix_prob=0.5)
    assert all(t.split("]")[0].split("|")[11] == "0" for t in (tb.packed_teams[0], tb.bias_packed_teams[0],
                                                               tb.mix_packed_teams[0]))
