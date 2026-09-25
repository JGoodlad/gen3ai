"""Build the round-2 population-loop READER argvs TOKEN-EXACTLY from recorded templates.

    RB2, RC2  <- ai_v13_18_teach5_offense_hidose's recorded original_command (arm A, the reader recipe)
    RB+, RC+  <- ai_v13_24_popr1_read_loop's / ai_v13_25_popr1_read_ctrl's recorded original_command
                 (the round-1 readers themselves; the convergence side-check's FORKS of them)

B2 / C2 (the round-2 generalists) are NOT built here: they are the Training Run session's, copied
verbatim into ../generalists/. Every argv is written to ../argv_<arm>.txt and its diff against its
template is printed, so the registration can quote the diff instead of asserting it. Run from anywhere:

    python designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/build_argvs.py

It reads models/ (the MAIN checkout's archive) and writes only inside this directory.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
MODELS = os.environ.get("GEN3AI_MODELS_DIR", "/home/goodlad/dev/gen3ai/models")

NAMES = {
    "B2": "ai_v13_27_popr2_loop",            # round-2 loop generalist (Training Run's)
    "C2": "ai_v13_28_popr2_ctrl",            # round-2 no-exploiter control (Training Run's)
    "RB2": "ai_v13_29_popr2_read_loop",      # fresh offense reader of B2
    "RC2": "ai_v13_30_popr2_read_ctrl",      # fresh offense reader of C2
    "RBX": "ai_v13_31_popr1_read_loop_ext",  # RB+ : RB extended +50 % (convergence side-check)
    "RCX": "ai_v13_32_popr1_read_ctrl_ext",  # RC+ : RC extended +50 % (convergence side-check)
}
ROLLOUT = 48 * 2048                  # 98,304
GEN2_FINAL = 111280128               # B2 / C2 land here: 103,219,200 + 8,000,000 -> ceil to 1132 rollouts
READER2_STEPS = GEN2_FINAL + 8000000  # 119,280,128 -> lands 119,341,056: budget 8,060,928 == A's
R1_FINAL = 111280128                 # RB / RC finals
EXT_STEPS = R1_FINAL + 4000000       # 115,280,128 -> lands 115,310,592: +4,030,464 == +50 % of 8,060,928


def lands(total):
    return -(-total // ROLLOUT) * ROLLOUT


assert lands(READER2_STEPS) - GEN2_FINAL == 8060928
assert lands(EXT_STEPS) - R1_FINAL == 4030464 == 8060928 // 2


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


def flags(toks):
    d, i = {}, 0
    while i < len(toks):
        if toks[i].startswith("--"):
            v = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else True
            d[toks[i]] = v
        i += 1
    return d


def diff(a, b):
    """Flag-level diff: {flag: (old, new)} over the LAST occurrence of each flag."""
    da, db = flags(a), flags(b)
    return {k: (da.get(k), db.get(k)) for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)}


def write(arm, toks, template_name, template, path=None):
    path = path or os.path.join(OUT, f"argv_{arm}.txt")
    with open(path, "w") as fh:
        fh.write(" ".join(toks) + "\n")
    print(f"== {arm}  {NAMES.get(arm, arm)}  ({len(template)} -> {len(toks)} tokens vs {template_name})")
    for k, (o, n) in diff(template, toks).items():
        print(f"     {k}: {o!r} -> {n!r}")


def main():
    hidose = recorded("ai_v13_18_teach5_offense_hidose")   # arm A
    rb = recorded("ai_v13_24_popr1_read_loop")             # RB
    rc = recorded("ai_v13_25_popr1_read_ctrl")             # RC

    # ---- sanity: RB and RC are A's recipe with ONLY the four registered values moved (round 1).
    for name, r in (("RB", rb), ("RC", rc)):
        moved = set(diff(hidose, r))
        assert moved == {"--run-name", "--model", "--exploiter", "--steps"}, (name, moved)

    # ---- RB2 / RC2: A's recipe TOKEN-EXACT, target and fork parent re-pointed, +8M.
    for arm, tgt in (("RB2", NAMES["B2"]), ("RC2", NAMES["C2"])):
        r = list(hidose)
        r = set_flag(r, "--run-name", NAMES[arm])
        r = set_flag(r, "--model", f"models/{tgt}/final_model.zip")
        r = set_flag(r, "--exploiter", f"models/{tgt}/final_model.zip")
        r = set_flag(r, "--steps", str(READER2_STEPS))
        write(arm, r, "ai_v13_18_teach5_offense_hidose", hidose)

    # ---- RB+ / RC+: the round-1 reader's OWN recorded command, forked from its OWN final into a NEW
    #      run dir (never a resume: that would move RB's / RC's recorded budget). --exploiter is
    #      UNCHANGED (B's / C's final). --fork-lr 2.5e-4 --fork-lr-freeze is already in the template.
    for arm, src_name, src in (("RBX", "ai_v13_24_popr1_read_loop", rb),
                               ("RCX", "ai_v13_25_popr1_read_ctrl", rc)):
        r = list(src)
        r = set_flag(r, "--run-name", NAMES[arm])
        r = set_flag(r, "--model", f"models/{src_name}/final_model.zip")
        r = set_flag(r, "--steps", str(EXT_STEPS))
        f = flags(r)
        assert f["--fork-lr"] == "2.5e-4" and f["--fork-lr-freeze"] is True, arm
        assert f["--exploiter"] == flags(src)["--exploiter"], arm
        write(arm, r, src_name, src)

    # ---- STAND-IN reader argvs for validation only: B2 / C2 have not finished. The stand-in
    #      target is ai_v13_24_popr1_read_loop (RB), whose final sits at EXACTLY 111,280,128 — the
    #      step B2 and C2 will land on. It is an exploiter, not a generalist (finding K-2).
    stand = "models/ai_v13_24_popr1_read_loop/final_model.zip"
    for arm in ("RB2", "RC2"):
        r = open(os.path.join(OUT, f"argv_{arm}.txt")).read().split()
        r = set_flag(r, "--model", stand)
        r = set_flag(r, "--exploiter", stand)
        r = set_flag(r, "--run-name", NAMES[arm] + "_STANDIN")
        write(arm + "_STANDIN", r, f"argv_{arm}.txt", open(os.path.join(OUT, f"argv_{arm}.txt")).read().split(),
              path=os.path.join(HERE, f"argv_{arm}_STANDIN.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
