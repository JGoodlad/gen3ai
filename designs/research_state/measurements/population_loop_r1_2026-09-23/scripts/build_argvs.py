"""Build the round-1 population-loop argvs TOKEN-EXACTLY from the two recorded templates.

    B, C   <- ai_v13_12_plateau's recorded original_command (the plateau parent's own block)
    A2, R  <- ai_v13_18_teach5_offense_hidose's recorded original_command (round-0 exploiter A)

Every argv is written to ../argv_<arm>.txt and its diff against its template is printed, so the
registration can quote the diff instead of asserting it. Run from anywhere:

    python designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/build_argvs.py

It reads models/ (the MAIN checkout's archive) and writes only inside this directory.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
MODELS = os.environ.get("GEN3AI_MODELS_DIR", "/home/goodlad/dev/gen3ai/models")

PLATEAU = "models/ai_v13_12_plateau/final_model.zip"              # G0 @95,158,272
A_FINAL = "models/ai_v13_18_teach5_offense_hidose/final_model.zip"  # A (round-0 exploiter, 1.78x)
A13_FINAL = "models/ai_v13_13_exploit5_offense/final_model.zip"      # round-0 exploiter, 0.39x unnamed
STABLE_SET = f"{A_FINAL},{A13_FINAL}"

NAMES = {
    "B": "ai_v13_22_popr1_loop",
    "C": "ai_v13_23_popr1_ctrl",
    "RB": "ai_v13_24_popr1_read_loop",
    "RC": "ai_v13_25_popr1_read_ctrl",
    "A2": "ai_v13_26_popr0_exploit5_offense_s1002",
}
GEN_STEPS = 103158272          # a TOTAL: G0 95,158,272 + 8,000,000 -> lands 103,219,200 (82 x 98,304)
GEN_FINAL = 103219200
READER_STEPS = GEN_FINAL + 8000000   # 111,219,200 -> lands 111,280,128: budget 8,060,928 == A's


def recorded(run):
    m = json.load(open(os.path.join(MODELS, run, "metadata.json")))
    toks = m["original_command"]
    toks = toks.split() if isinstance(toks, str) else list(toks)
    if toks and toks[0].endswith("__main__.py"):
        toks = toks[1:]            # the launcher script path, not an argument
    return toks


def set_flag(toks, flag, value=None):
    """Replace the value of every occurrence of `flag` (or append it). value=None = a boolean."""
    out, seen, i = [], False, 0
    while i < len(toks):
        t = toks[i]
        if t == flag:
            seen = True
            out.append(t)
            has_val = i + 1 < len(toks) and not toks[i + 1].startswith("--")
            if value is not None:
                out.append(value)
            i += 2 if has_val else 1
            continue
        out.append(t)
        i += 1
    if not seen:
        out += [flag] + ([value] if value is not None else [])
    return out


def diff(a, b):
    """Flag-level diff: {flag: (old, new)} over the LAST occurrence of each flag."""
    def fl(toks):
        d, i = {}, 0
        while i < len(toks):
            if toks[i].startswith("--"):
                v = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else True
                d[toks[i]] = v
            i += 1
        return d
    da, db = fl(a), fl(b)
    return {k: (da.get(k), db.get(k)) for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)}


def write(arm, toks, template_name, template):
    path = os.path.join(OUT, f"argv_{arm}.txt")
    with open(path, "w") as fh:
        fh.write(" ".join(toks) + "\n")
    print(f"== {arm}  {NAMES.get(arm, arm)}  ({len(template)} -> {len(toks)} tokens vs {template_name})")
    for k, (o, n) in diff(template, toks).items():
        print(f"     {k}: {o!r} -> {n!r}")


def main():
    plateau = recorded("ai_v13_12_plateau")
    hidose = recorded("ai_v13_18_teach5_offense_hidose")

    # ---- B: the loop. G0 continued +8M at G0's OWN frozen dose, the two offense exploiters of G0
    #      as stable opponents at a raised, loss-weighted share, retirement disabled.
    b = list(plateau)
    b = set_flag(b, "--run-name", NAMES["B"])
    b = set_flag(b, "--model", PLATEAU)
    b = set_flag(b, "--steps", str(GEN_STEPS))
    b = set_flag(b, "--fork-lr", "2.8e-5")          # 0.20x, G0's own frozen value (named)
    b = set_flag(b, "--fork-lr-freeze")
    b = set_flag(b, "--distill-coef", "0.0")
    b = set_flag(b, "--stable-opponent-selfplay-share", "0.4")
    b = set_flag(b, "--stable-opponent-mastered-wr", "1.01")
    b = set_flag(b, "--stable-opponent-pfsp")
    b = set_flag(b, "--stable-opponents", STABLE_SET)
    write("B", b, "ai_v13_12_plateau", plateau)

    # ---- C: the control. B's argv with ONE value changed: the stable share 0.4 -> 0.0. The two
    #      specialists are LOADED and EVALUATED (so C's own series measures its rate against them,
    #      and eval load / RNG coin consumption match B's) but NEVER trained against.
    c = set_flag(list(b), "--stable-opponent-selfplay-share", "0.0")
    c = set_flag(c, "--run-name", NAMES["C"])
    write("C", c, "B", b)

    # ---- the readers: A's recipe TOKEN-EXACT, target and fork parent re-pointed, +8M.
    for arm, tgt in (("RB", NAMES["B"]), ("RC", NAMES["C"])):
        r = list(hidose)
        r = set_flag(r, "--run-name", NAMES[arm])
        r = set_flag(r, "--model", f"models/{tgt}/final_model.zip")
        r = set_flag(r, "--exploiter", f"models/{tgt}/final_model.zip")
        r = set_flag(r, "--steps", str(READER_STEPS))
        write(arm, r, "ai_v13_18_teach5_offense_hidose", hidose)

    # ---- A2: A's seed replicate against G0 — the reader instrument's RUN-level floor.
    a2 = set_flag(list(hidose), "--run-name", NAMES["A2"])
    a2 = set_flag(a2, "--seed", "1002")
    write("A2", a2, "ai_v13_18_teach5_offense_hidose", hidose)

    # ---- STAND-IN reader argvs for validation only: B/C do not exist yet. The stand-in target is
    #      ai_v13_16_teach5_offense_dist — a self-play, frozen-lr fork of G0 whose final sits at
    #      EXACTLY 103,219,200, the step B and C will land on.
    stand = "models/ai_v13_16_teach5_offense_dist/final_model.zip"
    for arm in ("RB", "RC"):
        r = open(os.path.join(OUT, f"argv_{arm}.txt")).read().split()
        r = set_flag(r, "--model", stand)
        r = set_flag(r, "--exploiter", stand)
        r = set_flag(r, "--run-name", NAMES[arm] + "_STANDIN")
        path = os.path.join(OUT, "scripts", f"argv_{arm}_STANDIN.txt")
        with open(path, "w") as fh:
            fh.write(" ".join(r) + "\n")
    print("stand-in reader argvs written to scripts/argv_*_STANDIN.txt (validation only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
