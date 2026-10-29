"""The PINNED trainer's startup TEAM PHASE on X63-U's argv, at the X63 pin (CPU, card hidden), with its time and the
process's peak RSS. Run from a detached checkout of the pin with ITS src/ on PYTHONPATH (so its own TeamLoader,
MatchupSpec, team_mix_guard and corpus registry run):

    cd <pin checkout> && PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" python <this> <argv file> [--no-mix]

It parses the argv file's tokens with the pin's `main.train.parser.build_parser` (the BARE argv: no launcher recipe
restoration, so the matchup hash and mix_kind are the harness's, as N1-P's `startup_team_phase_at_pin.txt`), calls the
pin's `main.train.matchup_setup.build_matchup_and_opponents`, and stops at the first error after the team phase (the
opponent resolution needs launcher-filled fields). `--no-mix` drops the four `--team-mix` tokens: the same harness on
X26c's argv, the baseline for the RSS and time deltas. Prints `[startup] <s> s` and `[startup] peak RSS <GB>`.
"""
import resource
import shlex
import sys
import time


def main() -> int:
    argv_file, no_mix = sys.argv[1], "--no-mix" in sys.argv[2:]
    toks = shlex.split(open(argv_file).read())
    for launcher_only in ("--restart-interval-hours", "--pin-commit"):
        if launcher_only in toks:
            i = toks.index(launcher_only)
            del toks[i:i + 2]
    if no_mix:
        for flag in ("--team-mix", "--team-mix-sides"):
            i = toks.index(flag)
            del toks[i:i + 2]
    from main.train.parser import build_parser
    from main.train.matchup_setup import build_matchup_and_opponents
    args = build_parser().parse_args(toks)
    t0 = time.perf_counter()
    try:
        build_matchup_and_opponents(args)
        print("[startup] the team phase and opponent resolution both completed")
    except Exception as exc:  # noqa: BLE001 -- the harness stops at the launcher-only field, as N1-P's did
        print(f"[startup] stopped after the team phase: {type(exc).__name__}: {exc}")
    print(f"[startup] {time.perf_counter() - t0:.1f} s")
    print(f"[startup] peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
