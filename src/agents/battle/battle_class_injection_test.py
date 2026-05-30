"""Gen3Battle injection: our players/env build Gen3Battle (so .events / live_view()
exist), with None-safety for poke-env's env threading a None default.
"""

import logging

from poke_env.battle.battle import Battle
from poke_env.player.player import Player

from agents.battle.gen3_battle import Gen3Battle


def test_player_battle_class_defaults_to_battle():
    import inspect
    sig = inspect.signature(Player.__init__)
    assert sig.parameters["battle_class"].default is Battle


def test_none_battle_class_falls_back_to_battle():
    """PokeEnv threads battle_class=None down to its _EnvPlayer agents; Player must
    treat None as 'use the default Battle', not store None (which would crash
    _create_battle). Asserted on the source guard so we don't need a live socket."""
    import inspect
    src = inspect.getsource(Player.__init__)
    assert "battle_class or Battle" in src, (
        "Player.__init__ must guard None battle_class -> Battle"
    )


def test_gen3player_defaults_to_gen3battle():
    import inspect
    from agents.inference.player import Gen3Player
    sig = inspect.signature(Gen3Player.__init__)
    assert sig.parameters["battle_class"].default is Gen3Battle


def test_gen3env_defaults_to_gen3battle():
    import inspect
    from agents.training.gen3_env import Gen3Env
    sig = inspect.signature(Gen3Env.__init__)
    assert sig.parameters["battle_class"].default is Gen3Battle


def test_gen3battle_has_event_log_and_live_view():
    """A constructed Gen3Battle exposes the event log + cursor + live_view that the
    consumers (obs/turn-delta/reward/replay) will read."""
    b = Gen3Battle("battle-gen3ou-inj", "p1user", logging.getLogger("t"), gen=3)
    assert hasattr(b, "events")
    assert hasattr(b, "events_since")
    assert hasattr(b, "event_cursor")
    assert hasattr(b, "live_view")
    assert isinstance(b, Battle)  # still a poke-env Battle (drop-in)
