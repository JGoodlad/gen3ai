"""Phase 3 — THE TRAINING ENV FACTORY: one `_init` closure per `SubprocVecEnv` worker.

Everything the worker needs must either be picklable or be rebuilt inside `_init`, which is why
the foreign-opponent loads (stable / exploiter / pool) happen in the worker rather than here.

The keyword-only tail (`args`, `mappings`, `OPPONENT_CLASSES`, ...) is what used to be closure
state when this lived inside `main()`. `OPPONENT_CLASSES` keeps its shouting name deliberately:
the body is a verbatim move, and renaming the binding would have meant editing the body.
"""
import traceback
from datetime import datetime

from agents.training.gen3_env import Gen3Env
from agents.training.snapshot_pool import SnapshotPool
from agents.training.wrappers import MaskableAgentWrapper
from agents.inference.player import RLPlayer
from main.train.constants import BATTLE_FORMAT
from poke_env import AccountConfiguration
from stable_baselines3.common.monitor import Monitor
from utils.bridge.bridge_session import attach_bridge_transport
from utils.logging.levels import LogLevel


def create_training_env_random(idx, stall_config=None, opponent_device="auto",
                               opponent_version=None, snapshot_dir=None,
                               self_play_fraction=0.0, self_play=False,
                               heuristic_weights=None, stable_opponents=None,
                               exploiter_entry=None, cf_records_dir=None, *,
                               args, mappings, log_level, trainee_teambuilder,
                               opponent_teambuilder, server_config, OPPONENT_CLASSES,
                               reward_factory):
    def _init():
        try:
            # Defensive per-worker pin (the module-level env vars are the primary guard, but they
            # are `setdefault` — an explicit OMP_NUM_THREADS=8 for the learner must not turn every
            # worker's B=1 opponent forward into an N-thread contender). Mirrors
            # snapshot_ladder.py's pin for the same reason: B=1 CPU inference gets nothing from
            # intra-op parallelism, and the parallelism that matters is ACROSS workers.
            import torch as _torch
            _torch.set_num_threads(1)
            ts = datetime.now().strftime('%H%M%S')
            env_username = f"RLAgent{idx}{ts}"

            env_log_level = log_level if idx == 0 else LogLevel.QUIET

            env = Gen3Env(
                mappings,
                battle_format=BATTLE_FORMAT,
                team=trainee_teambuilder,
                log_level=env_log_level,
                stall_config=stall_config,
                reward_fn=reward_factory(log_level=env_log_level),
                server_configuration=server_config,
                account_configuration1=AccountConfiguration(env_username, "password"),
                # Bridge mode: don't open websockets — the in-process sim is the transport.
                start_listening=not args.use_showdown_bridge,
                # TRAINING-only privileged belief labels (only the trainee env; the model side
                # gates the BeliefHead on the same coef>0 signal). Eval/self-play opponents play
                # via RLPlayer, not Gen3Env, so they never emit them.
                emit_belief_labels=(args.opp_belief_aux_coef > 0.0),
                move_belief_mode=args.move_belief_mode,
                emit_win_target=(args.win_prob_mode != "none"),
                # SPREAD-belief supervision (gen3_unified_spread_belief_v1): emit the privileged
                # true-spread label only when the loss will consume it (coef>0; the CLI guards that
                # --spread-belief-coef requires --spread-belief, so the head is present to supervise).
                emit_spread_labels=(args.spread_belief and args.spread_belief_coef > 0.0),
                emit_opp_intent_labels=(getattr(args, 'opp_intent_coef', 0.0) > 0.0),
                # HP-TYPE-belief supervision (gen3_typed_hp_belief_v1): emit the privileged true-HP-type
                # label only when the CE will consume it (the head itself is unconditional under a move
                # belief; the CLI guards that the coef implies one).
                emit_hp_type_labels=(args.move_belief_mode != "off" and args.hp_belief_mode == "composed"
                                     and args.hp_type_belief_coef > 0.0),
                # ITEM-belief supervision (gen3_item_belief_v1): emit the privileged true-item
                # label only when the head exists AND the CE will consume it.
                emit_item_labels=(args.item_belief and args.item_belief_coef > 0.0),
                # DEFENSIVE-exploration flag (gen3_defensive_entropy_v1): emit only when the boost is on, so
                # the state-conditioned entropy term in the PPO loss can read it. Off = no key, no cost.
                emit_defensive_opportunity=(args.defensive_entropy_boost > 1.0),
                # EXPLOITER DISTILLATION (gen3_exploiter_distill_v1): the teacher team's species id-set
                # (None unless --distill-coef>0). The env emits `distill_mask`=1 on states where the
                # trainee pilots this team — the only states the distillation KL folds. None → no key.
                distill_team_species=getattr(args, "_distill_species", None),
                # The OPPONENT side's real team source (agent2 does the networking for every
                # per-episode opponent; the rotated Players are decision-functions whose own
                # builders are inert). Without this, PokeEnv fed `team=` (the TRAINEE builder)
                # to BOTH sides — a --trainee-team pin made every battle a single-team MIRROR.
                opponent_team=opponent_teambuilder,
            )
            if args.use_showdown_bridge:
                # gen3_cf_label_plumbing_v1: the OPT-IN reconstruction-record tap. The ring is
                # built HERE, inside the worker, so the transport module never imports the
                # training package and nothing unpicklable crosses the SubprocVecEnv boundary
                # (only the directory string does). None (the default) → no sink, no file.
                _recon_sink = None
                if cf_records_dir:
                    from agents.training.cf_records import CfRecordRing
                    _ring = CfRecordRing(cf_records_dir, keep=args.cf_records_keep)
                    _recon_sink = _ring.write_b64
                # Swap the two _EnvPlayer agents' websocket transport for a local
                # BattleStream subprocess. Everything above the transport (obs, reward,
                # mask, wrappers) is unchanged — see utils/bridge/bridge_session.py.
                attach_bridge_transport(env, battle_format=BATTLE_FORMAT,
                                        impl=args.bridge_impl, recon_sink=_recon_sink)

            # Opponents are pure DECISION FUNCTIONS over env.battle2 (env.agent1/agent2 do
            # the networking), so build them start_listening=False — no idle connections,
            # and we can hold several per env for live per-episode selection. The heuristic
            # roster is always built; self-play adds a per-worker pool + one reusable pool
            # RLPlayer whose .model is swapped per episode (see MaskableAgentWrapper).
            heuristic_opponents = [
                cls(
                    battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                    server_configuration=server_config,
                    account_configuration=AccountConfiguration(f"Opp{idx}h{i}{ts}", "password"),
                    start_listening=False,
                )
                for i, cls in enumerate(OPPONENT_CLASSES)
            ]

            pool = pool_player = None
            if self_play and snapshot_dir is not None:
                pool = SnapshotPool(
                    pool_dir=snapshot_dir, current_version=opponent_version,
                    device=opponent_device,
                    pfsp_scale=getattr(args, "pfsp_scale", 0.0),
                    pool_spread=getattr(args, "pool_spread", False),
                    compile_extractor=args.compile_opponents,
                    compile_hide_cuda=True,       # spawned env worker — never take a CUDA context
                    compile_strict=args.compile_opponents_strict,
                )
                # model=None placeholder — the wrapper swaps in a sampled snapshot before
                # ever using it. Stochastic + temperature so the learner trains against the
                # policy's full action distribution (richer, less exploitable than argmax).
                # Strict (crash-over-corruption) on a stale decision: the launcher restarts.
                pool_player = RLPlayer(
                    model=None, team=opponent_teambuilder, battle_format=BATTLE_FORMAT,
                    server_configuration=server_config, mappings=mappings,
                    account_configuration=AccountConfiguration(f"Opp{idx}p{ts}", "password"),
                    start_listening=False,
                    stochastic=True, temperature=args.self_play_temp,
                )

            # Stable cross-run opponents — one reusable RLPlayer each, loaded ONCE per worker
            # (foreign models don't change, so no per-episode reload). They join the TRAINING
            # mix only under self-play (the challenge/pool bucket); each plays stochastically at
            # its temperature (harder to over-exploit). Un-mastered → challenge peer of the pool;
            # mastered (pushed via set_stable_mastered) → floor peer of the bots.
            stable_players, stable_labels, stable_teams = [], [], []
            if self_play and stable_opponents:
                from agents.model.compile_opponents import maybe_compile_extractor
                from agents.model.snapshot import load_foreign_opponent
                from utils.teambuilder import Gen3Teambuilder as _G3TB
                for e in stable_opponents:
                    opp_model, _ = load_foreign_opponent(
                        e.zip_path, current_version=opponent_version,
                        device=opponent_device, config_path=e.config_path)
                    # hide_cuda=True: this runs in a spawned env worker, where a CUDA context
                    # would cost ~252 MiB of card per worker (the June 48× OOM).
                    maybe_compile_extractor(opp_model, args.compile_opponents,
                                            label=f"stable:{e.label}", hide_cuda=True,
                                            strict=args.compile_opponents_strict)
                    # Fold-back: a specialist opponent pilots ITS OWN pinned team (entry
                    # team_str from its run's metadata); the wrapper switches agent2._team to
                    # this builder on the episodes it plays. None = pool pilot (generalist).
                    _pin_tb = _G3TB(list(e.team_strs)) if e.team_strs else None   # multi-team specialist samples among ITS OWN teams
                    stable_players.append(RLPlayer(
                        model=opp_model, team=(_pin_tb or opponent_teambuilder),
                        battle_format=BATTLE_FORMAT,
                        server_configuration=server_config, mappings=mappings,
                        account_configuration=AccountConfiguration(
                            f"Opp{idx}s{len(stable_players)}{ts}", "password"),
                        start_listening=False,
                        stochastic=True, temperature=e.temperature,
                    ))
                    stable_labels.append(e.label)
                    stable_teams.append(_pin_tb)

            # EXPLOITER mode: one fixed target loaded ONCE per worker → the sole opponent. Same
            # foreign-load path as a stable opponent; stochastic at the stable-opponent temp so
            # it stays a moving target (harder to over-exploit a frozen target's quirks).
            exploiter_player = None
            exploiter_team = None
            if exploiter_entry is not None:
                from agents.model.compile_opponents import maybe_compile_extractor
                from agents.model.snapshot import load_foreign_opponent
                from utils.teambuilder import Gen3Teambuilder as _G3TB
                _ex_model, _ = load_foreign_opponent(
                    exploiter_entry.zip_path, current_version=opponent_version,
                    device=opponent_device, config_path=exploiter_entry.config_path)
                maybe_compile_extractor(_ex_model, args.compile_opponents,
                                        label="exploiter-target", hide_cuda=True,
                                        strict=args.compile_opponents_strict)
                # Fold-back: an exploiter-of-a-specialist faces the target ON ITS OWN pinned team.
                exploiter_team = (_G3TB(list(exploiter_entry.team_strs))
                                  if exploiter_entry.team_strs else None)
                exploiter_player = RLPlayer(
                    model=_ex_model, team=(exploiter_team or opponent_teambuilder),
                    battle_format=BATTLE_FORMAT,
                    server_configuration=server_config, mappings=mappings,
                    account_configuration=AccountConfiguration(f"Opp{idx}x{ts}", "password"),
                    start_listening=False,
                    stochastic=True, temperature=exploiter_entry.temperature,
                )

            wrapped = MaskableAgentWrapper(
                env, heuristic_opponents=heuristic_opponents, pool=pool,
                pool_player=pool_player, self_play_fraction=self_play_fraction, rng_seed=idx,
                heuristic_weights=heuristic_weights,
                stable_players=stable_players, stable_labels=stable_labels,
                stable_challenge_share=args.stable_opponent_selfplay_share,
                stable_pfsp=args.stable_opponent_pfsp,
                exploiter_player=exploiter_player,
                # Fold-back per-opponent teams: pinned builders (or None) parallel to
                # stable_players, the exploiter target's pin, and the pool builder to restore
                # on unpinned episodes. All-None → the wrapper never touches agent2._team.
                stable_teams=stable_teams, exploiter_team=exploiter_team,
                opponent_pool_team=opponent_teambuilder,
                # keep-bots: the heuristic roster (always built above) is mixed back in
                # per-episode alongside the exploiter target. No-op unless exploiter_player is set.
                exploiter_keep_bots=args.exploiter_keep_bots,
                exploiter_bot_fraction=args.exploiter_bot_fraction,
                team_wr_tracking=getattr(args, "team_wr_tracking", True),
            )

            # FORCE OVERRIDE: SingleAgentWrapper hardcodes 10 for gen3ou. We need 11.
            # Also ensure it propagates our Dict observation space natively.
            wrapped.action_space = env.action_space
            wrapped.observation_space = env.observation_space

            return Monitor(wrapped)
        except Exception as e:
            print(f"🛑 ERROR IN WORKER {idx}: {e}")
            traceback.print_exc()
            raise e
    return _init
