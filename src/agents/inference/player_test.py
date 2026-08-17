import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player import Player
from agents.inference.player import Gen3Player, RLPlayer
from agents.action.mapper import StaleDecisionError
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
from agents.training.stall import StallConfig
from agents.battle.strict_view import StrictBattleView


def _with_strict_view(battle):
    """Give a mock battle a real StrictBattleView so the player's migrated meta reads
    (battle.strict_view().turn / .battle_tag / .finished) pass straight through to the
    mock's set fields — the boundary the code now goes through."""
    battle.strict_view.return_value = StrictBattleView(battle)
    return battle


@contextmanager
def _stubbed_encode(player, base_obs, mask):
    """Stub encode() + Gen3ActionMasker so embed_battle can run on a bare mock.

    Mirrors the new embed_battle flow: get_mask → record → encode(hp_tracker=...).
    """
    player.observation_encoder.encode = MagicMock(return_value=base_obs)
    with patch(
        "agents.inference.player.Gen3ActionMasker.get_mask", return_value=mask
    ):
        yield


class _ConcretePlayer(Gen3Player):
    def choose_move(self, battle):
        pass

    def _battle_finished_callback(self, battle):
        # Provide the poke-env no-op that Gen3Player._battle_finished_callback calls via super()
        Gen3Player._battle_finished_callback(self, battle)


def _make_player(threshold=250):
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    player = _ConcretePlayer.__new__(_ConcretePlayer)
    player.observation_encoder = encoder
    player.mappings = mappings
    player._stall_config = StallConfig(threshold=threshold)
    player._stall_loggers = {}
    player._trackers = {}
    player._turn_delta_encoder = None
    return player, encoder


def _stall_battle(tag, turn):
    battle = MagicMock()
    battle.battle_tag = tag
    battle.turn = turn
    return _with_strict_view(battle)


def _make_battle(encoder):
    """Return a mock battle that satisfies embed_battle's requirements."""
    battle = MagicMock()
    battle.wait = False
    battle.finished = False
    battle.battle_tag = "battle-gen3ou-test"
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    battle.weather = None
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    battle.force_switch = False
    # LegalActions.from_battle (called in embed_battle) reads these legality flags.
    battle.available_switches = []
    battle.available_moves = []
    battle.trapped = False
    battle.maybe_trapped = False
    battle.wait = False
    battle.last_request = None
    battle.our_last_effectiveness = None
    battle.opp_last_effectiveness = None
    # Explicit None — otherwise MagicMock auto-creates a tree whose
    # .user_species / .target_species would leak into the TurnDelta encoder
    # and trip its strict species lookup.
    battle.our_last_damaging_move = None
    battle.opp_last_damaging_move = None
    battle.we_moved_first = None
    battle.turn = 1
    return _with_strict_view(battle)


def test_embed_battle_output_dim_matches_encoder():
    """
    Regression test: Gen3Player.embed_battle() must produce an observation whose
    length equals encoder.dimension (base + prev_mask + TurnDelta).
    Previously, TurnDelta was not appended, causing a shape mismatch at inference time.
    """
    player, encoder = _make_player()
    battle = _make_battle(encoder)

    base_obs = np.zeros(encoder.base_dimension, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    with _stubbed_encode(player, base_obs, mask):
        result = player.embed_battle(battle)

    assert result["observation"].shape == (encoder.dimension,), (
        f"embed_battle() produced {result['observation'].shape[0]} dims, "
        f"expected {encoder.dimension}. "
        f"Likely missing prev_mask or TurnDelta block in player.embed_battle()."
    )
    assert result["observation"].dtype == np.float32


# ---------------------------------------------------------------------------
# _handle_stall — basic behaviour
# ---------------------------------------------------------------------------

def test_no_forfeit_below_threshold():
    player, _ = _make_player(threshold=10)
    assert player._handle_stall(_stall_battle("A", turn=5), "STALL") is None


def test_forfeit_at_threshold():
    player, _ = _make_player(threshold=10)
    result = player._handle_stall(_stall_battle("A", turn=10), "STALL")
    assert isinstance(result, ForfeitBattleOrder)


def test_forfeit_above_threshold():
    player, _ = _make_player(threshold=10)
    result = player._handle_stall(_stall_battle("A", turn=999), "STALL")
    assert isinstance(result, ForfeitBattleOrder)


# ---------------------------------------------------------------------------
# _handle_stall — concurrent-battle isolation
# ---------------------------------------------------------------------------

def test_separate_stall_logger_per_battle():
    player, _ = _make_player(threshold=10)
    player._handle_stall(_stall_battle("A", turn=1), "STALL")
    player._handle_stall(_stall_battle("B", turn=1), "STALL")

    assert "A" in player._stall_loggers
    assert "B" in player._stall_loggers
    assert player._stall_loggers["A"] is not player._stall_loggers["B"]


def test_battle_a_stall_does_not_silence_battle_b():
    """A reaching threshold must not prevent B from getting its own independent logger."""
    player, _ = _make_player(threshold=5)

    # A stalls (ForfeitBattleOrder returned)
    result_a = player._handle_stall(_stall_battle("A", turn=10), "STALL")
    assert isinstance(result_a, ForfeitBattleOrder)

    # B also stalls: must get its own logger, not share A's
    result_b = player._handle_stall(_stall_battle("B", turn=10), "STALL")
    assert isinstance(result_b, ForfeitBattleOrder)

    assert "A" in player._stall_loggers
    assert "B" in player._stall_loggers
    assert player._stall_loggers["A"] is not player._stall_loggers["B"]


def test_battle_b_under_threshold_does_not_forfeit_battle_a():
    """Evaluating B (healthy) between A's checks must not affect A's outcome."""
    player, _ = _make_player(threshold=10)

    player._handle_stall(_stall_battle("A", turn=1), "STALL")   # A: fine
    player._handle_stall(_stall_battle("B", turn=1), "STALL")   # B: fine
    player._handle_stall(_stall_battle("A", turn=3), "STALL")   # A: still fine
    result_b = player._handle_stall(_stall_battle("B", turn=1), "STALL")  # B: still fine
    result_a = player._handle_stall(_stall_battle("A", turn=20), "STALL") # A: stalled

    assert result_b is None
    assert isinstance(result_a, ForfeitBattleOrder)


def test_interleaved_battles_independent_stall_decisions():
    """A stalling must not forfeit B; B healthy must not un-forfeit A."""
    player, _ = _make_player(threshold=10)

    # Interleave: A at turn 15 (stalled), B at turn 2 (healthy)
    result_a = player._handle_stall(_stall_battle("A", turn=15), "STALL")
    result_b = player._handle_stall(_stall_battle("B", turn=2), "STALL")

    assert isinstance(result_a, ForfeitBattleOrder)
    assert result_b is None


# ---------------------------------------------------------------------------
# _battle_finished_callback — stall logger cleanup
# ---------------------------------------------------------------------------

def test_battle_finished_removes_its_stall_logger():
    player, _ = _make_player(threshold=10)
    player._handle_stall(_stall_battle("A", turn=1), "STALL")
    assert "A" in player._stall_loggers

    player._battle_finished_callback(_stall_battle("A", turn=1))
    assert "A" not in player._stall_loggers


def test_battle_finished_does_not_remove_other_battles_logger():
    """Finishing A must leave B's stall logger intact."""
    player, _ = _make_player(threshold=10)
    player._handle_stall(_stall_battle("A", turn=1), "STALL")
    player._handle_stall(_stall_battle("B", turn=1), "STALL")

    player._battle_finished_callback(_stall_battle("A", turn=1))

    assert "A" not in player._stall_loggers
    assert "B" in player._stall_loggers


def test_stall_state_not_reset_when_different_battle_fires():
    """Old bug: switching battle tags reset the stall logger. Must not happen now."""
    player, _ = _make_player(threshold=10)

    # A makes progress toward threshold
    player._handle_stall(_stall_battle("A", turn=8), "STALL")
    # B fires (simulating interleaved concurrent battle)
    player._handle_stall(_stall_battle("B", turn=1), "STALL")
    # A's logger must still exist with its own state — not reset by B's call
    assert "A" in player._stall_loggers
    assert player._stall_loggers["A"] is not player._stall_loggers["B"]


# ---------------------------------------------------------------------------
# Per-battle EpisodeTracker isolation
# ---------------------------------------------------------------------------

def test_per_battle_trackers_are_independent():
    """Two concurrent battles must get separate EpisodeTracker instances."""
    player, _ = _make_player()

    battle_a = _make_battle(player.observation_encoder)
    battle_a.battle_tag = "battle-gen3ou-A"

    battle_b = _make_battle(player.observation_encoder)
    battle_b.battle_tag = "battle-gen3ou-B"

    base_obs = np.zeros(player.observation_encoder.base_dimension, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    with _stubbed_encode(player, base_obs, mask):
        player.embed_battle(battle_a)
        player.embed_battle(battle_b)

    assert "battle-gen3ou-A" in player._trackers
    assert "battle-gen3ou-B" in player._trackers
    assert player._trackers["battle-gen3ou-A"] is not player._trackers["battle-gen3ou-B"]


def test_battle_finished_removes_tracker():
    """_battle_finished_callback must clean up the tracker for the finished battle."""
    player, _ = _make_player()
    battle = _make_battle(player.observation_encoder)
    battle.battle_tag = "battle-gen3ou-cleanup"

    base_obs = np.zeros(player.observation_encoder.base_dimension, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    with _stubbed_encode(player, base_obs, mask):
        player.embed_battle(battle)
    assert "battle-gen3ou-cleanup" in player._trackers

    player._battle_finished_callback(battle)
    assert "battle-gen3ou-cleanup" not in player._trackers


def test_battle_finished_leaves_other_trackers_intact():
    """Finishing battle A must not remove battle B's tracker."""
    player, _ = _make_player()

    def _make_battle_tagged(tag):
        b = _make_battle(player.observation_encoder)
        b.battle_tag = tag
        return b

    base_obs = np.zeros(player.observation_encoder.base_dimension, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    battle_a = _make_battle_tagged("tag-A")
    battle_b = _make_battle_tagged("tag-B")
    with _stubbed_encode(player, base_obs, mask):
        player.embed_battle(battle_a)
        player.embed_battle(battle_b)

    player._battle_finished_callback(battle_a)
    assert "tag-A" not in player._trackers
    assert "tag-B" in player._trackers


def test_embed_battle_history_zero_on_first_call():
    """On the first embed_battle() call there is no prior context — history block must be zeros."""
    player, encoder = _make_player()
    battle = _make_battle(encoder)

    base_obs = np.zeros(encoder.base_dimension, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    with _stubbed_encode(player, base_obs, mask):
        result1 = player.embed_battle(battle)
    obs1 = result1["observation"]

    history_start = encoder.base_dimension + 11
    assert obs1.shape == (encoder.dimension,)
    assert np.all(obs1[history_start:] == 0), (
        "No prior context at turn 1 — history block must be all-zeros"
    )


def test_embed_battle_tracker_accumulates_across_calls():
    """After two embed_battle() calls the tracker has two contexts (one committed delta)."""
    player, encoder = _make_player()
    battle = _make_battle(encoder)

    base_obs = np.zeros(encoder.base_dimension, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    with _stubbed_encode(player, base_obs, mask):
        player.embed_battle(battle)
        player._get_tracker(battle).advance(0)
        battle.turn = 2
        result2 = player.embed_battle(battle)
    obs2 = result2["observation"]

    assert obs2.shape == (encoder.dimension,)
    tracker = player._get_tracker(battle)
    assert len(tracker._history) == 2
    assert len(tracker._actions) == 1


# ---------------------------------------------------------------------------
# Connect-or-raise guard: a dead connection must FAIL FAST, not hang.
# ---------------------------------------------------------------------------
def _make_test_player(name, connected, listen_done=None):
    """A real Gen3Player subclass instance built WITHOUT Player.__init__ (no real
    websocket), so super()._battle_against still resolves to Player._battle_against
    while we control the connection state. Must be called inside a running loop so
    the asyncio.Event binds to it.

    listen_done: None  → no _listening_coroutine (start_listening=False case);
                 True  → a *done* listen Future (listen() returned, the deterministic
                         'connection failed' signal); False → a *pending* listen Future.
    """
    import asyncio
    import concurrent.futures

    class _TestPlayer(Gen3Player):
        def __init__(self):  # deliberately skip Player.__init__ (no connection)
            ev = asyncio.Event()
            if connected:
                ev.set()
            fut = None
            if listen_done is not None:
                fut = concurrent.futures.Future()
                if listen_done:
                    fut.set_result(None)  # listen() returned without logging in
            self.ps_client = type("C", (), {
                "logged_in": ev,
                "websocket_url": "ws://localhost:1/showdown/websocket",
                "_listening_coroutine": fut,
            })()

        @property
        def username(self):  # Player.username is read-only; stub it for the guard
            return name

        def choose_move(self, battle):  # abstract on Player; unused by the guard
            raise NotImplementedError

    return _TestPlayer()


class TestConnectGuard:
    def test_raises_when_listen_exits_without_login(self):
        """The FUNDAMENTAL fix, DETERMINISTIC (no timeout): poke-env's listen()
        swallows a failed connection and returns, leaving logged_in unset forever.
        'listen task done && not logged in' is an exact failure signal → the guard
        raises ShowdownConnectionError instead of hanging on logged_in.wait()."""
        import asyncio
        from agents.inference.player import ShowdownConnectionError

        async def run():
            me = _make_test_player("trainee", connected=False, listen_done=True)
            opp = _make_test_player("opponent", connected=False, listen_done=True)
            with pytest.raises(ShowdownConnectionError) as exc:
                await me._battle_against(opp, n_battles=1)
            assert "ws://localhost:1" in str(exc.value)
            assert "listen task exited" in str(exc.value)

        asyncio.run(asyncio.wait_for(run(), timeout=5))  # must resolve instantly, never hang

    def test_raises_when_no_listening_task(self):
        """A client with no listening task can never log in — fail loudly, don't wait."""
        import asyncio
        from agents.inference.player import ShowdownConnectionError

        async def run():
            me = _make_test_player("trainee", connected=False, listen_done=None)
            with pytest.raises(ShowdownConnectionError) as exc:
                await me._battle_against(n_battles=1)
            assert "no listening task" in str(exc.value)

        asyncio.run(asyncio.wait_for(run(), timeout=5))

    def test_proceeds_when_logged_in(self):
        """When all participants are logged in, the guard delegates to the real
        _battle_against (stubbed) without raising."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        async def run():
            me = _make_test_player("trainee", connected=True)
            opp = _make_test_player("opponent", connected=True)
            with patch.object(Player, "_battle_against",
                              new=AsyncMock(return_value="ok")) as sup:
                result = await me._battle_against(opp, n_battles=3)
            assert result == "ok"
            sup.assert_awaited_once()

        asyncio.run(asyncio.wait_for(run(), timeout=5))


def _make_rl_opponent():
    """An RLPlayer built WITHOUT Player.__init__ (no model, no server) — just enough
    attributes to drive choose_move's decision path deterministically."""
    p = RLPlayer.__new__(RLPlayer)
    p._stochastic = False
    p._temperature = 1.0
    p._n_decisions = 0
    p._n_defaults = 0
    p._n_redecides = 0
    p._handle_stall = MagicMock(return_value=None)               # no stall forfeit
    p._predict_best_action = MagicMock(return_value=(3, None, None))  # a legal idx
    p._tracker = MagicMock()                                     # snapshot()/restore() rollback spy
    p._get_tracker = MagicMock(return_value=p._tracker)
    return p


class TestStaleDecisionRedecide:
    """choose_move RE-DECIDES on a stale decision (the live battle advanced under it during the
    model forward) rather than crashing. The opponent's decision is internal to
    SingleAgentWrapper.step, so it must always return a VALID order — SB3 has no failed-step
    path, and a raise would kill the worker. The TRAINEE stays strict (gen3_env): its action
    comes from SB3 and can't be re-run inside the step, so a stale trainee decision must crash
    rather than corrupt its (obs, action) → (reward, next_obs) transition."""

    def test_redecides_on_stale_then_succeeds(self):
        p = _make_rl_opponent()
        sentinel = object()
        p.action_to_order = MagicMock(side_effect=[StaleDecisionError("turn advanced"), sentinel])
        p.choose_default_move = MagicMock()
        result = RLPlayer.choose_move(p, MagicMock())
        assert result is sentinel               # re-decided on the now-current request
        assert p._n_redecides == 1
        # The superseded attempt's record() is rolled back so no phantom turn lingers in the
        # opponent's turn-history obs: one snapshot for the decision, one restore for the stale try.
        p._tracker.snapshot.assert_called_once()
        p._tracker.restore.assert_called_once_with(p._tracker.snapshot.return_value)
        p.choose_default_move.assert_not_called()   # the model's real move, NOT a default

    def test_redecide_budget_exhausted_falls_back_to_valid_default(self):
        from agents.inference.player import _OPP_REDECIDE_MAX
        p = _make_rl_opponent()
        p.action_to_order = MagicMock(side_effect=StaleDecisionError("never settles"))
        default = object()
        p.choose_default_move = MagicMock(return_value=default)
        result = RLPlayer.choose_move(p, MagicMock())
        assert result is default                # valid order — NEVER crashes the SB3 worker
        assert p._n_redecides == _OPP_REDECIDE_MAX
        # Every superseded attempt is rolled back, so an exhausted budget still leaves the
        # tracker exactly at its pre-decision state (no accumulated phantom turns).
        assert p._tracker.restore.call_count == _OPP_REDECIDE_MAX
        p.choose_default_move.assert_called_once()

    def test_clean_decision_returns_order_unchanged(self):
        p = _make_rl_opponent()
        sentinel_order = object()
        p.action_to_order = MagicMock(return_value=sentinel_order)
        p.choose_default_move = MagicMock()
        result = RLPlayer.choose_move(p, MagicMock())
        assert result is sentinel_order             # normal path — unchanged
        assert p._n_redecides == 0
        # A clean decision commits its record() — snapshot is taken, but never rolled back.
        p._tracker.snapshot.assert_called_once()
        p._tracker.restore.assert_not_called()
        p.choose_default_move.assert_not_called()

    def test_no_legal_action_restores_then_defaults(self):
        # idx=None (all-zero mask: polled during the other agent's forced-switch window).
        # embed_battle() skips recording, but we still restore() defensively before defaulting,
        # so any partial obs-build work (e.g. a memoized delta) never leaks into the history.
        p = _make_rl_opponent()
        p._predict_best_action = MagicMock(return_value=(None, None, None))
        default = object()
        p.choose_default_move = MagicMock(return_value=default)
        p.action_to_order = MagicMock()
        result = RLPlayer.choose_move(p, MagicMock())
        assert result is default
        assert p._n_redecides == 0
        p.action_to_order.assert_not_called()       # idx None → never serializes a stale ctx
        p._tracker.restore.assert_called_once()


# ── RLPlayer._value_dist (v29 distributional value head capture) ──────────────
# _value_dist only reads self.model.policy.features_extractor.last_value_dist_logits, so a mock
# `self` exercises it directly (no model/battle needed). The same trace-capture path as _win_prob.

def test_value_dist_reads_stash_into_distribution():
    import torch
    from types import SimpleNamespace
    logits = torch.tensor([[0.0, 10.0, 0.0]])   # peaked on atom 1
    fake = SimpleNamespace(model=SimpleNamespace(policy=SimpleNamespace(
        features_extractor=SimpleNamespace(last_value_dist_logits=logits))))
    out = RLPlayer._value_dist(fake)
    assert len(out) == 3 and out[1] > 0.99 and abs(sum(out) - 1.0) < 1e-6


def test_value_dist_none_when_head_off():
    from types import SimpleNamespace
    fake = SimpleNamespace(model=SimpleNamespace(policy=SimpleNamespace(
        features_extractor=SimpleNamespace(last_value_dist_logits=None))))
    assert RLPlayer._value_dist(fake) is None


# ── RLPlayer._opp_intent (v67 α/β capture — the interpretability payload) ─────
# Same shape as the two above: the method only reads the extractor stash, so a mock `self`
# exercises it. What it produces is what lands in the trace and therefore in the prober.

def _intent_self(*, alpha=None, seat_nums=None, beta=None, species=None):
    from types import SimpleNamespace
    ex = SimpleNamespace(last_alpha_logits=alpha, last_alpha_seat_nums=seat_nums,
                         last_beta_logits=beta,
                         last_belief_logits=({"species": species} if species is not None else None))
    fake = SimpleNamespace(model=SimpleNamespace(policy=SimpleNamespace(features_extractor=ex)))
    # `_opp_intent` calls the sibling helper on `self`; bind the real one so the mock exercises the
    # REAL naming path rather than a stub of it.
    fake._slot_species = lambda: RLPlayer._slot_species(fake)
    return fake


def test_opp_intent_none_when_heads_off():
    assert RLPlayer._opp_intent(_intent_self()) is None


def test_opp_intent_names_every_option_and_rounds_for_disk():
    """The owner constraint: the model may only point at things it can NAME. A seat the belief did
    not fill (num 0) is DROPPED rather than shown as an anonymous index."""
    import torch
    # Two real seats (Tackle=33, Thunder=87) + one unfilled, then the SWITCH column.
    alpha = torch.tensor([[2.0, 0.0, 0.0, 1.0]])
    seats = torch.tensor([[33, 87, 0]])
    out = RLPlayer._opp_intent(_intent_self(alpha=alpha, seat_nums=seats))
    names = [r["name"] for r in out["alpha"]]
    assert "SWITCH" in names and len(out["alpha"]) == 3, "the empty seat must be dropped"
    assert all(not n.startswith("move#") for n in names), "every option is named through the dex"
    assert out["alpha"][0]["p"] > out["alpha"][-1]["p"], "ranked highest-first"
    # Rounded on the way to disk — these ride every captured decision of every captured battle.
    assert all(r["p"] == round(r["p"], 4) for r in out["alpha"])
    assert "beta" not in out, "no β head → no β key, rather than an empty one"


def test_opp_intent_beta_is_named_by_the_species_posterior():
    """β points at a SLOT, and a believed slot is an anonymous query — the model's own species
    posterior is what makes 'they will switch to X' refer to anything. An illegal candidate (-inf,
    i.e. fainted or already active) is not listed at all."""
    import torch
    alpha, seats = torch.tensor([[1.0, 0.5]]), torch.tensor([[33]])
    beta = torch.tensor([[0.0, 5.0, float("-inf")]])
    species = torch.zeros(1, 3, 400)
    species[0, 1, 143] = 10.0          # slot 1 → snorlax (dex 143)
    out = RLPlayer._opp_intent(_intent_self(alpha=alpha, seat_nums=seats, beta=beta,
                                            species=species))
    assert [r["slot"] for r in out["beta"]] == [1, 0], "ranked, and the -inf candidate excluded"
    assert out["beta"][0]["species"] == "snorlax"


def test_opp_intent_beta_slot_stays_a_bare_index_without_a_species_head():
    """No species head means nothing in the model can say what that slot holds. A bare index is the
    honest answer; inventing a name would be worse than the index."""
    import torch
    out = RLPlayer._opp_intent(_intent_self(
        alpha=torch.tensor([[1.0, 0.5]]), seat_nums=torch.tensor([[33]]),
        beta=torch.tensor([[1.0, 0.0]])))
    assert [r["species"] for r in out["beta"]] == [None, None]
