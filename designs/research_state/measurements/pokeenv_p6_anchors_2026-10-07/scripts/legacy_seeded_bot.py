"""THE OLD PATH of P6's bot identity proof — a COMPARISON-HARNESS hook, never production.

Runs ``python -m main.anchors <args>`` with the LEGACY client (pass ``--our-transport poke-env``) and a
``bot:<name>`` our-side, after wrapping ``main.anchors.runner.install_our_side`` so that the Python roster bot
each half builds gets its RNG streams SEEDED exactly as the Rust bot's are: half ``h`` (0 = ours_challenge,
1 = peer_challenge) → ``_choice_rng = random.Random(bot_stream_seed(S, h, 0))`` and, on a staller,
``_protect_rng = random.Random(bot_stream_seed(S, h, 1))`` — the env core's bot-route rule, whose Python twin is
``agents.training.rust_env_opponents.bot_stream_seed`` (the same seeding the Rust bot gate's corpus and
``main.eval_worker``'s per-game rule use). ``S`` is the ``--bot-seed`` in the argv (required).

In production the legacy bot drew from the process-wide, unseeded ``random``, so it could not be compared draw
for draw; this hook is the only change to the old path.

    PYTHONPATH=<wt>/src python legacy_seeded_bot.py --our-transport poke-env --our-side bot:staller \\
        --bot-seed 7 --opponent metamon:SmallRL --seed-base 914007 --capture-dir … --out …
"""
import random
import sys


def main() -> int:
    argv = sys.argv[1:]
    if "--bot-seed" not in argv:
        raise SystemExit("legacy_seeded_bot: pass --bot-seed S (the new path's declared stream seed)")
    seed = int(argv[argv.index("--bot-seed") + 1])

    import main.anchors.runner as R
    import main.play as play
    from agents.training.rust_env_opponents import bot_stream_seed

    orig = R.install_our_side
    calls = {"n": 0}

    def install(*a, **k):
        undo = orig(*a, **k)                      # installs play.build_model_player = build_bot_player
        env = calls["n"]
        calls["n"] += 1
        inner = play.build_model_player

        def build(*aa, **kk):
            p = inner(*aa, **kk)
            p._choice_rng = random.Random(bot_stream_seed(seed, env, 0))
            if hasattr(p, "_protect_rng"):
                p._protect_rng = random.Random(bot_stream_seed(seed, env, 1))
            print(f"[harness] seeded {type(p).__name__} for half-env {env}: choice="
                  f"{bot_stream_seed(seed, env, 0)} protect={bot_stream_seed(seed, env, 1)}", flush=True)
            return p

        play.build_model_player = build
        return undo                               # restores the saved original (our wrapper included)

    R.install_our_side = install
    sys.argv = ["main.anchors"] + argv
    from main.anchors.cli import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
