"""A minimal LiveView stub for the material-margin rule (`win_prob_test.py`) — NOT a test module.

The battle / TurnDelta / legality stubs the deleted reward-manager and progress-clock tests shared went
with them (T27 P6 slice 6d-2).

The name deliberately does NOT match pytest's `python_files` patterns (`*_test.py`), so this module
is imported, never collected.
"""


class _Mon:
    """A board mon: what `material_margin` reads (`hp_fraction`, `fainted`), plus inert view fields."""
    def __init__(self, hp_fraction=1.0, species="mon", active=False):
        self.hp_fraction = hp_fraction
        self.fainted = hp_fraction <= 0.0
        self.species = species
        self.active = active
        self.status = None
        self.volatiles = {}
        self.types = ()
        self.move_ids = ()
        self.boosts = {}
        self.ability = None
        self.item = None
        self.consumed_item = None
        self.stats = {}
        self.current_hp = None
        self.max_hp = None


class _Side:
    def __init__(self, hps, team_size=6, spikes=0):
        self.mons = tuple(_Mon(h, active=(i == 0)) for i, h in enumerate(hps))
        self.active = self.mons[0] if (self.mons and not self.mons[0].fainted) else None
        self.team_size = team_size
        self.side_conditions = {"spikes": spikes} if spikes else {}


class _Live:
    def __init__(self, our_hps, opp_hps, opp_team_size=6, won=False, lost=False, finished=False):
        self.ours = _Side(our_hps)
        self.opp = _Side(opp_hps, team_size=opp_team_size)
        self.weather = None
        self.won, self.lost, self.finished = won, lost, finished


def _full_team_live(our_alive=6, opp_alive=6, our_hp=1.0, opp_hp=1.0, **kw):
    """A 6v6 LiveView with `our_alive`/`opp_alive` mons at the given HP, rest fainted."""
    our = [our_hp] * our_alive + [0.0] * (6 - our_alive)
    opp = [opp_hp] * opp_alive + [0.0] * (6 - opp_alive)
    return _Live(our, opp, **kw)
