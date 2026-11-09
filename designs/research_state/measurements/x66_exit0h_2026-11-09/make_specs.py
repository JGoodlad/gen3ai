"""Writes X66's three read specs (PREREG §5, §6): look 1 (the banked S'-HONEST raw control behind G-BANK), look 1 with
a fresh control (only if G-BANK fails) and look 2 (the cumulative 700-784 pool, the raw arm banked throughout). Every
value the registration fills later is the literal PENDING (main.reads.spec): the students' zips and sha256s (the
students amendment), the play pin (e46e3ddf8, named by PREREG §5), the bank manifest sha256 (the bank amendment) and the earlier looks' spec sha256s.
    python3 make_specs.py            # rewrites read_spec_look1.json, read_spec_look1_fresh.json, read_spec_look2.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = "PENDING"
X = "/home/goodlad/gen3ai_archive/x66_exit0h_2026-11-09"
HON = "/home/goodlad/gen3ai_archive/sprime_honest_play_2026-11-06"
HON_SPEC = "9222393a14c79ff52e2a2bd315864626a9f6ea68ffeea709352c40e79d618afa"
PIN = None                 # e46e3ddf8 (PREREG §5): added with the students; a spec with no pin is reviewed by check-spec --pin-pending
E6 = "port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130"
PANELS = ["heldout72_hpfix", "forum72", "p72"]
STUDENTS = {"heldout72_hpfix": ("s_h", "S_H", 6611), "forum72": ("s_f", "S_F", 6612), "p72": ("s_p", "S_P", 6613)}
SHORT = {"heldout72_hpfix": "h", "forum72": "f", "p72": "p"}
L1, L2 = "700-744", "745-784"
PSTAR = {"arm": "pstar", "seed": 2601, "ref": {
    "kind": "pinned", "path": "models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip",
    "sha256": "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"}}
KAK = "metamon:Kakuna"


def models():
    m = {"pstar": PSTAR}
    for _p, (lab, st, seed) in STUDENTS.items():
        m[lab] = {"arm": lab, "seed": seed, "ref": {"kind": "pinned", "path": P, "sha256": P}}
    return m


def imp(state, spec, group, labels):
    return {"state": state, "spec_sha256": spec, "group": group, "models": labels}


def params(look, seeds, control, groups, gbank):
    return {"look": look, "u_panels": PANELS, "students": {p: STUDENTS[p][0] for p in PANELS}, "raw": "pstar",
            "groups": groups, "seeds": seeds, "control": control, "gbank": gbank, "engine": E6,
            "z_go": {"1": 2.576, "2": 2.0}, "z_null": 1.96, "null_bar_pp": 2.0,
            "bootstrap": {"B": 4000, "seed": 20261109}, "rule8_eps": 0.0005, "max_drop_frac": 0.005,
            "delivery": {"path": f"{X}/delivery.json", "B": 4000, "seed": 20261105}, "bank_manifest_sha256": P}


def spec(read, note, anchor, prm):
    doc = {"schema": "gen3_read_spec_v1", "read": read,
           "prereg": "designs/research_state/measurements/x66_exit0h_2026-11-09/PREREG.md"}
    if PIN is not None:
        doc["pin"] = PIN
    return {**doc, "note": note, "models": models(), "anchor": anchor, "decision": {"rule": "distill_play", "params": prm}}


def gbank_games(first, last, origin):
    return {str(s): [(s - origin) % 20] for s in range(first, last + 1)}


look1 = spec(
    "x66_exit0h_look1_2026-11-09",
    f"X66 ExIt-0H LOOK 1 (state {X}/look1; plays only on DELIVERED): each P* student distilled from S'-HONEST's honest "
    "labels on the other two panels (s_h / s_f / s_p) plays greedy, no search, lone one-game units vs Kakuna on its "
    f"held-out panel's 900 S'-HONEST deals (U unit seeds {L1}, 20 a seed), paired by deal with S'-HONEST's BANKED P* raw "
    "unit (imported); G-BANK re-plays 135 banked raw deals (game (seed-700) mod 20 per panel and seed), >= 133 "
    "byte-identical or the banked control is VOID (read_spec_look1_fresh.json); distill_play: NOT-DELIVERED by the "
    "delivery gate's file, GO iff LB(z 2.576) > 0, NULL iff UB(1.96) < +2 pp, else CONTINUE (look 2). PENDING values "
    "are filled by the bank and students amendments.",
    {"gbank": {"opponent": KAK, "teamsets": PANELS, "unit_seeds": L1, "per_game": True, "models": ["pstar"],
               "games": gbank_games(700, 744, 700)},
     **{SHORT[p]: {"opponent": KAK, "teamsets": [p], "unit_seeds": L1, "per_game": True,
                   "models": ["pstar", STUDENTS[p][0]], "import": imp(HON, HON_SPEC, "u", ["pstar"])} for p in PANELS}},
    params(1, {"1": L1}, "banked", {p: [SHORT[p]] for p in PANELS},
           {"group": "gbank", "origin": 700, "min_identical": 133, "n": 135}))

look1_fresh = spec(
    "x66_exit0h_look1_fresh_2026-11-09",
    f"X66 ExIt-0H LOOK 1, FRESH CONTROL (state {X}/look1_fresh; played ONLY if look 1's G-BANK fails, PREREG §5): P* "
    f"raw played fresh on look 1's 2,700 deals, the students imported from look 1 (state {X}/look1); no G-BANK; the "
    "rest as look 1. PENDING values are filled by the bank and students amendments and look 1's spec sha256.",
    {SHORT[p]: {"opponent": KAK, "teamsets": [p], "unit_seeds": L1, "per_game": True,
                "models": ["pstar", STUDENTS[p][0]],
                "import": imp(f"{X}/look1", P, SHORT[p], [STUDENTS[p][0]])} for p in PANELS},
    params(1, {"1": L1}, "fresh", {p: [SHORT[p]] for p in PANELS}, None))

look2 = spec(
    "x66_exit0h_look2_2026-11-09",
    f"X66 ExIt-0H LOOK 2 (state {X}/look2; plays only if look 1 is CONTINUE): the students play S'-HONEST's remaining "
    f"banked deals (U unit seeds {L2}, 800 a panel), paired with S'-HONEST's BANKED P* raw units, read with look 1's 900 "
    "a panel (the students and G-BANK's replays imported from look 1); one bootstrap pool of 85 seeds a panel; GO iff "
    "LB(z 2.0) > 0, NULL iff UB(1.96) < +2 pp, else UNDECIDED. PENDING values are filled by the bank and students "
    "amendments and look 1's spec sha256.",
    {"gbank1": {"opponent": KAK, "teamsets": PANELS, "unit_seeds": L1, "per_game": True, "models": ["pstar"],
                "games": gbank_games(700, 744, 700), "import": imp(f"{X}/look1", P, "gbank", ["pstar"])},
     "ctl": {"opponent": KAK, "teamsets": PANELS, "unit_seeds": "700-784", "per_game": True, "models": ["pstar"],
             "import": imp(HON, HON_SPEC, "u", ["pstar"])},
     **{f"{SHORT[p]}1": {"opponent": KAK, "teamsets": [p], "unit_seeds": L1, "per_game": True,
                         "models": [STUDENTS[p][0]], "import": imp(f"{X}/look1", P, SHORT[p], [STUDENTS[p][0]])}
        for p in PANELS},
     **{f"{SHORT[p]}2": {"opponent": KAK, "teamsets": [p], "unit_seeds": L2, "per_game": True,
                         "models": [STUDENTS[p][0]]} for p in PANELS}},
    params(2, {"1": L1, "2": L2}, "banked", {p: [f"{SHORT[p]}1", "ctl", f"{SHORT[p]}2"] for p in PANELS},
           {"group": "gbank1", "origin": 700, "min_identical": 133, "n": 135}))

for name, doc in (("read_spec_look1.json", look1), ("read_spec_look1_fresh.json", look1_fresh),
                  ("read_spec_look2.json", look2)):
    (HERE / name).write_text(json.dumps(doc, indent=1) + "\n")
    print(name)
