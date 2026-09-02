"""The `# --- Subprocess eval ---` section: the eval workers, the self-play pool and
promotion, the stable opponents, the exploiter/ladder, and the team pins.

Lifted VERBATIM out of the old single-file `parser.py` (lines 1597-1966); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from agents.training.eval_callback import _EVAL_SUBPROCESS_CONCURRENCY, EVAL_SHARD_GAMES
from agents.training.snapshot_pool import HEURISTIC_FLOOR, SELF_PLAY_FULL, SELF_PLAY_START
from agents.training.wrappers import STABLE_CHALLENGE_SHARE
from main.train.parser.base import BoolFlag


def add_eval_subprocess_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- Subprocess eval ---
    parser.add_argument("--eval-workers", "--eval_workers", dest="eval_workers", type=int, default=5,
                        help="Number of parallel eval-worker subprocesses per cycle (default 5 for bot "
                             "eval; self-play doubles this to 10). Workers work-steal opponents from a "
                             "shared pool, so uneven per-opponent cost self-balances. Capped at the "
                             "opponent count.")
    parser.add_argument("--eval-device", "--eval_device", dest="eval_device", type=str, default="cpu",
                        help="Device for the eval-worker subprocess inference (default cpu, to decouple from the training GPU).")
    parser.add_argument("--eval-concurrency-per-worker", "--eval_concurrency_per_worker",
                        dest="eval_concurrency_per_worker", type=int, default=_EVAL_SUBPROCESS_CONCURRENCY,
                        help="Battles each eval worker overlaps at once within its claimed opponent (default 1 = "
                             "sequential). Single-thread asyncio latency-hiding (not multi-core): overlaps the "
                             "bridge/server I/O wait with other battles' forwards. A single-core bridge benchmark "
                             "measured ~2x decisions/sec at 3 on spare cores (less under live training contention); "
                             "the plateau is ~3. Cross-opponent parallelism is still --eval-workers.")
    parser.add_argument("--eval-shard-games", "--eval_shard_games",
                        dest="eval_shard_games", type=int, default=EVAL_SHARD_GAMES,
                        help="Games per work-steal shard unit (battle-level work-stealing, default 25 → ~4 shards "
                             "per opponent). Each opponent's eval games split into chunks any idle worker can drain, "
                             "so one straggler no longer pins a whole opponent on a single worker — the long tail "
                             "collapses to one shard. Smaller = finer tail collapse but more player builds / (on "
                             "websocket) more connection churn; the in-process bridge (--use-bridge, the default) is "
                             "preferred for fine shards. >= the per-opponent game count disables sharding (one shard "
                             "per opponent = the original opponent-level behaviour).")
    parser.add_argument("--bait-bot-share", "--bait_bot_share", dest="bait_bot_share",
                        type=float, default=0.0,
                        help="Add the scripted BaitBot to the TRAINING opponent roster with this "
                             "share of the sampling mass (default 0.0 = ABSENT, byte-identical). "
                             "BaitBot pivots into an immune bench mon with probability "
                             "--bait-bot-p; it exists to put PUNISHMENT FREQUENCY on the bait habit "
                             "on a controlled dial (the habit is exploration starvation at a "
                             "saturated action, so punishment frequency is the one signal it cannot "
                             "seal off). The weight is sized from the ACTUAL sum of the other "
                             "roster weights, so the declared share stays exact even when "
                             "--bot-weights re-weights the heuristics. Keep it a MINORITY share to "
                             "avoid script-sniping.")
    parser.add_argument("--bait-bot-p", "--bait_bot_p", dest="bait_bot_p", type=float, default=0.6,
                        help="BaitBot's pivot probability when a bait is available (default 0.6). "
                             "The gate's held-out generalization read uses a DIFFERENT value, so an "
                             "arm cannot pass by memorising this one. Ignored unless "
                             "--bait-bot-share > 0.")
    parser.add_argument("--eval-freq", "--eval_freq", dest="eval_freq", type=int, default=None,
                        help="Steps between eval cycles (default None = EVAL_FREQ_STEPS, 2,000,000 — "
                             "byte-identical to every pre-existing command). Lower it for SHORT arms: a "
                             "3M exploiter-gate fork at the 2M default gets 1-2 cycles, which cannot meet a "
                             ">=4-cycle reading discipline; --eval-freq 750000 gives 4. Applies to BOTH the "
                             "per-opponent bot eval and the self-play eval, so their cadences cannot drift. "
                             "Eval is non-blocking and skips a cycle while the previous one is still running, "
                             "so a too-small value self-throttles rather than stalling training.")
    parser.add_argument("--eval-games", "--eval_games", dest="eval_games", type=int, default=None,
                        help="Games per OPPONENT per eval cycle (default: the module EVAL_GAMES, 100). "
                             "Per-cell 95%% CI: n=100 -> +/-0.098, n=200 -> +/-0.069 — raise for tighter "
                             "sentinel/promotion reads at proportionally more eval compute (work-stolen "
                             "across --eval-workers, off the training path). Shards per opponent = "
                             "eval-games / --eval-shard-games.")
    parser.add_argument("--snapshot-ladder-games", "--snapshot_ladder_games",
                        dest="snapshot_ladder_games", type=int, default=100,
                        help="Frozen-snapshot ELO ladder: games per pair for the per-promotion "
                             "round-robin tax (0 = disable). On each promotion a DETACHED bridge "
                             "subprocess plays the new frozen snapshot vs the current pool and "
                             "appends to <run>/snapshot_ladder/games.jsonl (measured once, kept "
                             "forever) — a dense, high-resolution internal ladder the saturated "
                             "bots can't provide. Off the training path.")
    parser.add_argument("--keep-eval-snapshots", "--keep_eval_snapshots", dest="keep_eval_snapshots",
                        type=int, default=10,
                        help="Retain the N most-recent eval weight snapshots in eval_traces/step_<N>/snapshot.zip "
                             "so the prober can reload the bit-exact model that produced a cycle's traces "
                             "(~27MB each; default 10 ≈ 270MB). 0 only writes the identity manifest; the prober "
                             "then falls back to the nearest persisted checkpoint.")
    parser.add_argument("--keep-eval-trace-steps", "--keep_eval_trace_steps", dest="keep_eval_trace_steps",
                        type=int, default=20,
                        help="The trainer grooms the forensic traces it writes: after each eval cycle it "
                             "keeps only the N most-recent eval step dirs under eval_traces/ (0 = keep all). "
                             "`python -m main.prober.groom` is the manual fallback for finished runs.")
    parser.add_argument("--keep-stalls", "--keep_stalls", dest="keep_stalls", type=int, default=50,
                        help="Bound the run's stalls/ dir: each eval cycle keep only the N most-recent "
                             "stall_*.html replays (0 = keep all). `python -m agents.training.artifact_retention` "
                             "is the manual fallback / cross-run sweep.")
    parser.add_argument("--keep-crashes", "--keep_crashes", dest="keep_crashes", type=int, default=10,
                        help="Bound the run's crashes/ dir: each eval cycle keep only the N most-recent "
                             "launcher restart_err_*.txt files (0 = keep all).")
    parser.add_argument("--self-play", action=BoolFlag, default=False, help="Enable self-play snapshot pool as training opponents")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Pool directory (default: <run_dir>/snapshots)")
    # ── The FORK pool-seed guard (gen3_fork_pool_seed_v1) ──────────────────────────────────────
    # A FORK starts in a new run dir whose snapshots/ is EMPTY, and an empty pool does not disable
    # --self-play: it silently falls back to the BOT pool. That confound voided a three-arm A/B
    # (2026-08-18) and nearly voided the three-dose cell (2026-09-02). A genuine fork now seeds its
    # parent's pool automatically, and a fork that still has no pool REFUSES. See
    # agents.training.pool_seed for the audited file list and the rule.
    # Declared POSITIVELY so `BoolFlag` generates the `--no-` form itself: the flag a launch types
    # is `--no-fork-pool-seed`, and declaring THAT name would generate `--no-no-fork-pool-seed`.
    parser.add_argument("--fork-pool-seed", "--fork_pool_seed", dest="fork_pool_seed",
                        action=BoolFlag, default=True,
                        help="Auto-seed a FORK's empty self-play pool from its fork parent "
                             "(snapshot_*.zip + summary.json / win_rate_vs_bots.txt / "
                             "model_config.json). ON by default, and it only ever fires on a "
                             "genuine fork with an EMPTY pool — a launcher restart and a non-empty "
                             "pool are never touched. Pass --no-fork-pool-seed to opt out; a "
                             "poolless --self-play fork is then REFUSED unless --allow-empty-pool.")
    parser.add_argument("--allow-empty-pool", "--allow_empty_pool", dest="allow_empty_pool",
                        action=BoolFlag, default=False,
                        help="Consent explicitly to running --self-play on a FORK with an EMPTY "
                             "snapshot pool — i.e. to training against the BOT fallback until this "
                             "run promotes its own first snapshot. Without it such a launch exits "
                             "FATAL_CONFIG. A FRESH run (no --model) never needs this: it starts "
                             "poolless by design.")
    parser.add_argument("--promote-threshold", type=float, default=None,
                        help="Win rate vs. pool to trigger snapshot promotion. Default 0.65 with "
                             "stochastic sentinels; auto-lowered to 0.55 under --eval-sentinel-greedy "
                             "(greedy-vs-greedy removes the temperature handicap, so a genuinely-ahead "
                             "trainee wins the pool by a smaller margin — 0.65 would freeze the pool). "
                             "An explicit value always wins.")
    parser.add_argument("--eval-sentinel-greedy", "--eval_sentinel_greedy", dest="eval_sentinel_greedy",
                        action=BoolFlag, default=False,
                        help="Eval the self-play pool sentinels GREEDY (argmax) instead of stochastic. "
                             "Removes the greedy-trainee-vs-stochastic-sentinel handicap so win_rate_vs_pool "
                             "/ snapshot ELO reflect real best-vs-best skill (≈50%% vs a recent self, ramping "
                             "with sentinel age) instead of a flat temperature offset. Eval-only — TRAINING "
                             "opponents stay stochastic. Metric discontinuity vs prior cycles; pair with the "
                             "auto-lowered --promote-threshold (0.55).")
    parser.add_argument("--self-play-temp", type=float, default=1.0,
                        help="Sampling temperature for self-play TRAINING opponents (they sample, "
                             "not argmax, so the learner faces the policy's full action distribution). "
                             "1.0 = the policy's own distribution; >1 flatter/more random; lower → toward "
                             "greedy. Eval opponents stay deterministic regardless.")
    # ── Bot-mix curriculum (#2): keep the coverage-punishing bots in the TRAINING mix ──
    parser.add_argument("--bot-weights", "--bot_weights", dest="bot_weights", type=str, default=None,
                        help="Bias the per-episode HEURISTIC opponent pick toward chosen archetypes, "
                             "e.g. 'aggressive_v2=3,heuristic2=3'. Unlisted bots default to weight 1.0. "
                             "Names: heuristic, heuristic2, staller, staller_v2, aggressive, aggressive_v2, "
                             "setup_sweep, setup_sweep_v2. Omitted → uniform (current behavior). Only biases "
                             "WHICH heuristic an episode draws; the pool-vs-heuristic fraction is unaffected.")
    parser.add_argument("--heuristic-floor", "--heuristic_floor", dest="heuristic_floor",
                        type=float, default=None,
                        help="Minimum fraction of training episodes vs real bots once self-play saturates "
                             f"(default {HEURISTIC_FLOOR:g}). Raise it (e.g. 0.25) to keep a bigger permanent "
                             "bot slice so the coverage blindspot keeps getting exercised under self-play.")
    parser.add_argument("--self-play-start-wr", "--self_play_start_wr", dest="self_play_start_wr",
                        type=float, default=None,
                        help=f"win_rate_vs_bots at which self-play begins to ramp in (default {SELF_PLAY_START:g}).")
    parser.add_argument("--self-play-full-wr", "--self_play_full_wr", dest="self_play_full_wr",
                        type=float, default=None,
                        help=f"win_rate_vs_bots at which self-play reaches the floor (default {SELF_PLAY_FULL:g}); "
                             "raise it to ramp slower / stay bot-heavier for longer.")
    # ── PFSP / league-lite (prioritized fictitious self-play) — both OFF by default (byte-identical) ──
    parser.add_argument("--pfsp-scale", "--pfsp_scale", dest="pfsp_scale", type=float, default=0.0,
                        help="PFSP hardness weighting for self-play pool sampling (default 0.0 = off, "
                             "pure recency). >0 oversamples the pool selves the trainee is LOSING to "
                             "(weight ×(1 + pfsp_scale·(1−win_rate))) while never starving the ones it "
                             "beats — turns the recency window into a prioritised curriculum. The live "
                             "per-snapshot win-rates are measured at each self-play eval (EMA-smoothed) "
                             "and pushed to the training envs. Try 1.0–2.0. Pairs with --pool-spread so "
                             "PFSP has a diverse ladder of selves, not a recent-selves echo chamber.")
    parser.add_argument("--n-sentinels", "--n_sentinels", dest="n_sentinels", type=int, default=5,
                        help="Number of evenly-spaced pool snapshots eval'd as sentinels each self-play "
                             "cycle (default 5). Each gets a FRESH win-rate, which is what --pfsp-scale "
                             "weights the pool by — so a higher count re-prioritises MORE of the pool per "
                             "cycle (cuts the 'only ~¼ of the pool re-measured' staleness on a deep pool). "
                             "Cost: each extra sentinel is +100 games/cycle, work-stolen by the doubled "
                             "eval pool; eval is non-blocking + skip-while-running so it self-throttles. "
                             "Pairs with a larger --max-snapshots. Training-only (not version-locked).")
    parser.add_argument("--pool-spread", "--pool_spread", dest="pool_spread",
                        action=BoolFlag, default=False,
                        help="Self-play pool retention: keep a temporally-DIVERSE ladder (newest + "
                             "oldest + an even interior spread) instead of the oldest-evicted sliding "
                             "window, so PFSP (--pfsp-scale) has a real range of past selves to "
                             "up-weight. Default off = the legacy sliding window (byte-identical).")
    # ── Team-side PFSP: variance-weighted TEAM sampling by self-play win-rate (OFF by default) ──
    parser.add_argument("--team-pfsp", "--team_pfsp", dest="team_pfsp",
                        choices=["off", "measure", "var", "onesided"], default="off",
                        help="Per-team self-play win-rate tracking for the trainee's pool teams (default "
                             "off = uniform random.choice, byte-identical). 'measure' TRACKS + persists "
                             "the per-team self-play win-rate to <run>/team_winrates.json (the offline "
                             "'which team is the generalist weakest on → next exploiter target' artifact) "
                             "WITHOUT biasing sampling. 'var' additionally weights each pool team by floor "
                             "+ p*(1-p) (p = the win-rate EMA, seed 0.5), capped at --team-pfsp-cap x the "
                             "uniform share — so the trainee drills the teams it wins ~half the time (max "
                             "variance) and stops over-sampling the ones it crushes / always loses. "
                             "'onesided' keeps the LOSING side at MAX weight instead — w(p)=0.25 for p<0.5, "
                             "else p*(1-p) (continuous at 0.5): every sub-50%% team stays maximally sampled "
                             "and only mastery retires a team (under the z_arch/FiLM conditioning "
                             "hypothesis the weak tail is the learnable headroom, so 'truly lost' is the "
                             "claim under test, not a sampling prior). Measured on SELF-PLAY pool battles "
                             "only (bots excluded). Training-only, NOT version-locked.")
    parser.add_argument("--team-pfsp-cap", "--team_pfsp_cap", dest="team_pfsp_cap",
                        type=float, default=3.0,
                        help="Over-representation cap for --team-pfsp: no team is sampled more than "
                             "this multiple of the uniform share (weight ≤ cap×mean(raw)). Default 3.0.")
    parser.add_argument("--team-pfsp-floor", "--team_pfsp_floor", dest="team_pfsp_floor",
                        type=float, default=0.05,
                        help="Weight floor for --team-pfsp (raw_i = floor + p*(1-p)) so a fully-won / "
                             "fully-lost team is never starved to zero. Default 0.05.")
    parser.add_argument("--team-block-episodes", "--team_block_episodes", dest="team_block_episodes",
                        type=int, default=1,
                        help="Hold each drawn TRAINEE team for N consecutive episodes before redrawing "
                             "(1 = off, byte-identical). The per-team gradient-density counter to the "
                             "measured FiLM sample starvation (film/noise_scale ~8-9x the batch): at "
                             "~64 (~one rollout of episodes) per-update per-team density rises ~15x AND "
                             "blocks span an update boundary, so an env replays its team right after "
                             "that team's gradient landed (the exploiter-style learn-and-retest loop). "
                             "Composes with --team-pfsp (weights apply at each redraw; outcomes "
                             "attribute to the blocked team). Trainee side only; training-only, NOT "
                             "version-locked, resume-forwarded.")
    parser.add_argument("--team-wr-tracking", "--team_wr_tracking", dest="team_wr_tracking",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Track a running per-team win rate for the TRAINEE's piloted teams "
                             "(keyed by team_sha, stratified by opponent class): sparse TB summaries "
                             "(teams/wr_top_k, teams/wr_bottom_k, teams/n_teams_seen, teams/wr_mean) "
                             "plus a periodic full-table <run>/team_win_rates.json, archetype-joined "
                             "and restart-safe. PURE INSTRUMENTATION — nothing prioritizes on it, and "
                             "nothing should without normalizing against a team-strength baseline "
                             "first (a raw per-team win rate conflates pilot competence with team "
                             "strength). ON by default; --no-team-wr-tracking opts out. Distinct from "
                             "--team-pfsp, which measures self-play POOL battles only in order to "
                             "weight SAMPLING. Training-only, NOT version-locked, resume-forwarded.")
    # ── Stable (cross-run) opponents: load a model from ANOTHER run as a fixed opponent ──
    parser.add_argument("--stable-opponents", "--stable_opponents", dest="stable_opponents",
                        type=str, default=None,
                        help="Foreign model(s) from ANOTHER run to use as fixed eval opponents, "
                             "comma-separated. Simplest form is just the run dir: "
                             "'models/ai_v5_5_popart_N_0607' — the opponent is then labelled by that "
                             "dir name (ai_v5_5_popart_N_0607). Optional per-entry suffixes: "
                             "'@<step>' picks a specific checkpoint (default: best_model); "
                             "':<name>' renames it. (Per-opponent weights are NOT supported yet — "
                             "they only matter for the training mix, which is Stage 2.) Each model "
                             "must share this run's arch_signature (= observation layout) — a "
                             "mismatch is a startup FATAL surfaced to the TUI. Default None (off).")
    parser.add_argument("--stable-opponent-temp", "--stable_opponent_temp", dest="stable_opponent_temp",
                        type=float, default=1.0,
                        help="TRAINING-mix play temperature for stable opponents (default 1.0 = the "
                             "policy's own distribution). Stochastic (not greedy) so a fixed opponent "
                             "is a moving target — harder to over-exploit. (In EVAL they always play "
                             "greedy/temp-0 for a clean win-rate yardstick.)")
    parser.add_argument("--stable-opponent-mastered-wr", "--stable_opponent_mastered_wr",
                        dest="stable_opponent_mastered_wr", type=float, default=0.80,
                        help="Win rate at which a stable opponent is considered MASTERED and moves "
                             "from the challenge bucket (played alongside the self-play pool) to the "
                             "coverage floor (played alongside the bots) — it 'becomes another bot'. "
                             "Default 0.80. One-way per run. Only active under --self-play.")
    parser.add_argument("--stable-opponent-selfplay-share", "--stable_opponent_selfplay_share",
                        dest="stable_opponent_selfplay_share", type=float,
                        default=STABLE_CHALLENGE_SHARE,
                        help="Fraction of SELF-PLAY (challenge) episodes spent vs stable opponents — "
                             "the rest go to the self-play pool. Caps how much a fixed opponent "
                             "occupies training so a single one can't dominate; multiple un-mastered "
                             f"stable opponents SHARE this slice. Default {STABLE_CHALLENGE_SHARE:g}. "
                             "Only active under --self-play.")
    parser.add_argument("--stable-opponent-pfsp", "--stable_opponent_pfsp",
                        dest="stable_opponent_pfsp", action="store_true",
                        help="DYNAMIC stable-opponent selection: within the capped stable challenge "
                             "slice, pick WEIGHTED by how much the trainee is LOSING to each "
                             "(1 - win_rate) instead of uniformly — spend the exploiter budget on the "
                             "axis it's failing worst, and let each fade as it's mastered. The TOTAL "
                             "pool-vs-stable share is unchanged. Training-only; OFF = uniform "
                             "(byte-identical). Pairs with a raised --stable-opponent-selfplay-share.")
    parser.add_argument("--exploiter", dest="exploiter", type=str, default=None,
                        help="EXPLOITER MODE: train against ONE fixed foreign model as the SOLE "
                             "opponent every episode — the league 'exploiter' role (learn to beat a "
                             "specific target, e.g. the current main agent). Takes a run dir / "
                             "checkpoint spec exactly like --stable-opponents (e.g. "
                             "'models/ai_v6_13_outgoing_dmg_0620'), must share this run's "
                             "arch_signature (startup FATAL otherwise). This is a clean opponent-mix "
                             "front-end: it needs NO --self-play / --stable-opponents / share "
                             "fiddling — just point it at the target. Mutually exclusive with "
                             "--self-play. Recommended: init the exploiter from a strong checkpoint "
                             "(--model <target's checkpoint>) so it has a baseline to exploit from. "
                             "Default None (off).")
    parser.add_argument("--warmstart-consensus", "--warmstart_consensus", dest="warmstart_consensus",
                        type=str, default=None,
                        help="EXPLOITER MODE (requires --exploiter): before training, build a competent, "
                             "archetype-NEUTRAL warm start by disagreement-gated CONSENSUS distillation of "
                             "N mature teacher exploiters (comma-separated run-dirs) into --model (the "
                             "generalist init), then init the exploiter from it. The BC target is SHARP "
                             "where the teachers AGREE (universal decisions inherited) and FLAT where they "
                             "DISAGREE (archetype forks left high-entropy → the new exploiter specializes "
                             "FREELY, unbiased). Built ONCE into <run>/warmstart/ (idempotent across "
                             "launcher restarts; skipped once a training checkpoint exists). Deliberately "
                             "NOT valid for generalist/self-play runs (whose job is the opposite — absorb "
                             "divergence via --distill-teacher). See agents.training.warmstart. Default off.")
    parser.add_argument("--warmstart-battles", dest="warmstart_battles", type=int, default=200,
                        help="On-policy battles to collect for the --warmstart-consensus BC dataset (200).")
    parser.add_argument("--warmstart-bc-steps", dest="warmstart_bc_steps", type=int, default=4000,
                        help="BC gradient steps for --warmstart-consensus (early-stops on gated-KL; 4000).")
    parser.add_argument("--exploiter-keep-bots", dest="exploiter_keep_bots", action="store_true",
                        help="EXPLOITER MODE (requires --exploiter): mix the heuristic bots BACK IN "
                             "alongside the exploiter target instead of playing the target as the sole "
                             "opponent. Per episode, the target is faced with prob "
                             "(1 - --exploiter-bot-fraction), else a random floor/heuristic bot. Lets a "
                             "from-scratch specialist keep a bot floor while it learns to beat one strong "
                             "target. Off (default) = the target is the sole opponent (byte-identical).")
    parser.add_argument("--exploiter-bot-fraction", dest="exploiter_bot_fraction", type=float,
                        default=0.5,
                        help="Under --exploiter-keep-bots, the per-episode probability of facing a "
                             "heuristic bot instead of the exploiter target (default 0.5). The exploiter "
                             "target is faced with the complementary probability (1 - this).")
    parser.add_argument("--exploiter-temp-start", dest="exploiter_temp_start", type=float, default=None,
                        help="EXPLOITER MODE (requires --exploiter): ANNEAL the target opponent's sampling "
                             "temperature over training — a difficulty curriculum via opponent STOCHASTICITY. "
                             "Setting this (a positive float, e.g. 2.0) starts the target at this temperature "
                             "(flatter logits → noisier/weaker play, so a from-scratch trainee can win some "
                             "games and get a learning signal) and linearly anneals it to --exploiter-temp-end "
                             "over --exploiter-temp-anneal-frac of training, held after. None (default) = OFF: "
                             "the target plays at --stable-opponent-temp the whole run (byte-identical). "
                             "Training-only (not version-locked; forwarded verbatim on resume, where the anneal "
                             "continues from the resumed step).")
    parser.add_argument("--exploiter-temp-end", dest="exploiter_temp_end", type=float, default=1.0,
                        help="EXPLOITER MODE: the target opponent's temperature at the END of the anneal window "
                             "(default 1.0 = the policy's own distribution, i.e. the target's true strength as a "
                             "stochastic training opponent). Only used when --exploiter-temp-start is set. Set "
                             "below 1.0 to push the target toward greedy (harder) by the end.")
    parser.add_argument("--exploiter-temp-anneal-frac", dest="exploiter_temp_anneal_frac", type=float,
                        default=0.2,
                        help="EXPLOITER MODE: fraction of total --steps over which to linearly anneal the target "
                             "temperature from --exploiter-temp-start to --exploiter-temp-end (default 0.2 = the "
                             "first 20%% of training; held at the end temp after). 0 = constant at "
                             "--exploiter-temp-start (a fixed hotter opponent, no anneal). Only used in the FIXED "
                             "temp mode (--exploiter-temp-mode fixed).")
    parser.add_argument("--exploiter-temp-mode", dest="exploiter_temp_mode",
                        choices=["fixed", "ratchet"], default="fixed",
                        help="EXPLOITER MODE (with --exploiter-temp-start): how the target temperature is "
                             "controlled. 'fixed' (default) = the linear time schedule (--exploiter-temp-anneal-frac). "
                             "'ratchet' = DYNAMIC win-rate-driven: start at --exploiter-temp-start (set it HIGH, e.g. "
                             "5.0, so early games are trivially winnable) and ratchet the temperature DOWN toward "
                             "--exploiter-temp-end only when the trainee's measured TRAINING win-rate vs the target "
                             "clears --exploiter-temp-ratchet-wr — a ONE-WAY auto-curriculum that tracks the trainee's "
                             "competence frontier (never weakens the target, so no comfort-trap). Resume-safe (the "
                             "ratcheted temp is persisted to <run>/exploiter_temp_state.json).")
    parser.add_argument("--exploiter-temp-ratchet-wr", dest="exploiter_temp_ratchet_wr", type=float,
                        default=0.55,
                        help="RATCHET mode: the trainee TRAINING-WR vs the target at which the temperature ratchets "
                             "DOWN (harder). Default 0.55 (keeps play near the ~0.5 max-advantage-signal zone). "
                             "Measured per window of --exploiter-temp-ratchet-games target games.")
    parser.add_argument("--exploiter-temp-ratchet-factor", dest="exploiter_temp_ratchet_factor",
                        type=float, default=0.9,
                        help="RATCHET mode: multiply the temperature by this (<1) on each ratchet (default 0.9 = 10%% "
                             "harder steps). Floored at --exploiter-temp-end.")
    parser.add_argument("--exploiter-temp-ratchet-games", dest="exploiter_temp_ratchet_games",
                        type=int, default=500,
                        help="RATCHET mode: min target-games per decision window before a ratchet check (default 500 "
                             "— the noise guard; larger = smoother/slower).")
    # gen3_exploiter_pool_ladder_v1 — the OTHER difficulty axis. --exploiter-temp-* makes ONE target
    # play noisily; this swaps in genuinely WEAKER frozen opponents and promotes on a win-rate gate,
    # ending at the --exploiter target itself. Training-only knob (never versioned).
    parser.add_argument("--exploiter-ladder", dest="exploiter_ladder", type=str, default=None,
                        help="EXPLOITER MODE (requires --exploiter): POOL-LADDER opponent curriculum — "
                             "train against progressively STRONGER frozen snapshots, promoting a rung "
                             "each time the trainee's training win-rate vs the CURRENT rung clears "
                             "--exploiter-ladder-gate. Two forms: an ORDERED comma-separated list of "
                             "checkpoint paths, weakest first (--stable-opponents grammar, "
                             "path[@step][:label]); or 'auto:<run_dir>', which draws "
                             "--exploiter-ladder-rungs evenly-ELO-spaced snapshots from that run's "
                             "snapshot_ladder/ladder.json. The --exploiter target is ALWAYS appended as "
                             "the terminal rung, and the ladder never demotes. Rung state survives a "
                             "launcher restart via <run>/exploiter_ladder_state.json. Default None = OFF "
                             "(byte-identical: the target is the sole opponent from step 0).")
    parser.add_argument("--exploiter-ladder-gate", dest="exploiter_ladder_gate", type=float,
                        default=0.55,
                        help="--exploiter-ladder: the trainee TRAINING win-rate vs the CURRENT rung at "
                             "which it is promoted to the next one (default 0.55 — keeps play near the "
                             "~0.5 max-advantage-signal zone, matching --exploiter-temp-ratchet-wr). "
                             "Measured per window of --exploiter-ladder-window games.")
    parser.add_argument("--exploiter-ladder-window", dest="exploiter_ladder_window", type=int,
                        default=500,
                        help="--exploiter-ladder: min games vs the CURRENT rung per promotion check "
                             "(default 500 — the same noise guard, and the same disjoint-window "
                             "semantics, as --exploiter-temp-ratchet-games). Bot episodes under "
                             "--exploiter-keep-bots do not count.")
    parser.add_argument("--exploiter-ladder-rungs", dest="exploiter_ladder_rungs", type=int,
                        default=4,
                        help="--exploiter-ladder auto:<run_dir> ONLY: how many evenly-ELO-spaced "
                             "snapshots to draw from that run's ladder (default 4), BEFORE the "
                             "--exploiter target is appended — so the default auto ladder is 5 rungs. "
                             "Ignored for an explicit rung list.")
    parser.add_argument("--trainee-team", dest="trainee_team", type=str, default=None,
                        help="SPECIALIST MODE: pin the TRAINEE's team pool to the ONE team in this file "
                             "(a Showdown EXPORT string, like data/teams/sample/*.txt), so the agent "
                             "always plays that exact 6-mon team. The OPPONENTS still draw the full "
                             "diverse pool. Use to train a single-team specialist (e.g. --trainee-team "
                             "data/teams/specialist/tss_starmie.txt). Default None = the full trainee "
                             "pool (byte-identical).")
    parser.add_argument("--trainee-teams", dest="trainee_teams", type=str, default=None,
                        help="MULTI-TEAM SPECIALIST MODE: pin the TRAINEE's team pool to the SMALL "
                             "FIXED SET of teams in these files (comma-separated Showdown-export paths), "
                             "sampled UNIFORMLY per episode — a z-near multi-team exploiter (the "
                             "1-vs-3-team A/B). Opponents still draw the full pool. Mutually exclusive "
                             "with --trainee-team; under --exploiter EVERY member must be a sample team. "
                             "Default None.")
    parser.add_argument("--allow-nonsample-trainee", dest="allow_nonsample_trainee", action="store_true",
                        help="RESEARCH override: skip the exploiter vetted-SAMPLE gate so --trainee-team(s) "
                             "may pin NON-sample POOL teams (anchor on a sample, nearest neighbors from all "
                             "719 pool teams → a tighter z-cluster than the 32 samples allow). For FiLM "
                             "capacity / count-vs-diversity studies; NOT for a teacher you'll distil as-is. "
                             "Training-only, not version-locked. Default off (gate enforced).")
