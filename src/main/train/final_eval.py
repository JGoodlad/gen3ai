"""The POST-TRAINING win-rate evaluation — 9 fixed opponents, greedy policy.

Runs after `Training complete` is printed, which is why the banner it prints says the quantity of
work out loud: a working process and a wedged one were otherwise indistinguishable from the log,
and that misread cost six timeouts before `gen3_smoke_eval_scale_v1`.
"""
from datetime import datetime

from agents.inference.player import RLPlayer
from agents.opponents import (
    Gen3AggressivePlayer, Gen3AggressiveV2Player, Gen3HeuristicV2Player, Gen3SetupSweepPlayer,
    Gen3SetupSweepV2Player, Gen3StallerPlayer, Gen3StallerV2Player,
)
from agents.training.eval_callback import opponent_name
from main.train.constants import BATTLE_FORMAT
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from utils.bridge.local_battle_runner import run_local_battles


async def evaluate_model_random(model, *, args, mappings, trainee_teambuilder,
                                opponent_teambuilder, server_config):
    ts = datetime.now().strftime('%H%M%S')
    n = args.eval_battles
    # Bridge eval: build every player start_listening=False and play in-process via
    # run_local_battles (no server). Same flag as training; lets --debug + bridge run serverless.
    _eval_sl = not args.use_showdown_bridge
    _n_opp = 9
    print(f"\nFinal Evaluation (Session {ts}, Battles: {n}, Concurrency: {args.eval_concurrency})...")
    # Say the QUANTITY of work out loud. "Training complete" is printed by the caller BEFORE
    # this runs, so without a line here a still-working process is indistinguishable from a hung
    # one — six timeouts were spent proving exactly that.
    print(f"  ~{_n_opp * n} battles ({_n_opp} opponents x {n}). Training IS finished and the "
          f"model is saved; this is the post-training measurement and it can take minutes.",
          flush=True)

    rl_player = RLPlayer(
        model=model,
        team=trainee_teambuilder,
        battle_format=BATTLE_FORMAT,
        server_configuration=server_config,
        mappings=mappings,
        account_configuration=AccountConfiguration(f"RLFinal{ts}", "password"),
        max_concurrent_battles=args.eval_concurrency,
        stochastic=False,  # final eval = greedy policy
        start_listening=_eval_sl,
    )

    final_opponents = [
        (opponent_name(RandomPlayer), RandomPlayer(
            battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"FinalRand{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
            start_listening=_eval_sl,
        )),
        (opponent_name(SimpleHeuristicsPlayer), SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"FinalHeur{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
            start_listening=_eval_sl,
        )),
        (opponent_name(Gen3StallerPlayer), Gen3StallerPlayer(
            battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"FinalStall{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
            start_listening=_eval_sl,
        )),
        (opponent_name(Gen3AggressivePlayer), Gen3AggressivePlayer(
            battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"FinalAggr{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
            start_listening=_eval_sl,
        )),
        (opponent_name(Gen3SetupSweepPlayer), Gen3SetupSweepPlayer(
            battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"FinalSetup{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
            start_listening=_eval_sl,
        )),
    ]
    for _cls, _uname in [
        (Gen3HeuristicV2Player, f"FinalHeur2{ts}"),
        (Gen3StallerV2Player, f"FinalStallV2{ts}"),
        (Gen3AggressiveV2Player, f"FinalAggrV2{ts}"),
        (Gen3SetupSweepV2Player, f"FinalSetupV2{ts}"),
    ]:
        final_opponents.append((opponent_name(_cls), _cls(
            battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(_uname, "password"),
            max_concurrent_battles=args.eval_concurrency,
            start_listening=_eval_sl,
        )))

    win_rates: dict[str, float] = {}
    for name, opponent in final_opponents:
        if rl_player.n_finished_battles > 0:
            rl_player.reset_battles()
        print(f"  vs {name} [{n} battles]...")
        start_time = datetime.now()
        if args.use_showdown_bridge:
            # Overlap games like the server does; cap the Node-process fan-out (eval_concurrency
            # defaults to 100, which would spawn 100 sim children).
            await run_local_battles(rl_player, opponent, n,
                                    concurrency=min(args.eval_concurrency, 8),
                                    impl=args.bridge_impl)
        else:
            await rl_player.battle_against(opponent, n_battles=n)
        duration = datetime.now() - start_time
        wr = rl_player.n_won_battles / rl_player.n_finished_battles
        win_rates[name] = wr
        print(f"  Win rate vs {name}: {wr * 100:.1f}%  [{duration}]")
        model.logger.record(f"eval_final/win_rate_vs_{name}", wr)

    aggregate = sum(win_rates.values()) / len(win_rates)
    model.logger.record("eval_final/win_rate_mean", aggregate)
    model.logger.dump(model.num_timesteps)
    print(f"\nFinal aggregate win rate: {aggregate * 100:.1f}%")
