"""Resolve every fold's TEACHER SET and its taught teams from run metadata, and check the
untaught 8 is disjoint from all of them. Prints the distinct (parent, teacher-set) points.

Run: python resolve_sets.py   (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import json, os, re, sys

MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
HERE = os.path.dirname(os.path.abspath(__file__))

UNTAUGHT = ["61590463ee85d456", "9283210847f806ee", "ce35b7368c3d692e", "9909f2e98e981ccc",
            "9d5f845869e899ee", "f7ba5702fe856292", "90b94599967c6b77", "dbf81d8ecae51c39"]

SETS = {
    "R4set": ("ai_v9_76_R4ACTION_0830", f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"),
    "R3set": ("ai_v9_70_R3ACTION_0828", f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"),
    "FUND":  ("ai_v9_160_TCFUNDA_0903", f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"),
    "UNF":   ("ai_v9_162_TCUNFA_0903",  f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"),
    "R2set": ("ai_v9_59_R2ACTION_0827", f"{MD}/ai_v9_29_rev1_0823/final_model.zip"),
}


def teachers_and_teams(run):
    m = json.load(open(f"{MD}/{run}/metadata.json"))
    spec = re.findall(r"--distill-teacher\s+(\S+)", m["original_command"])[0].strip("'\"")
    out = []
    for chunk in spec.split(";"):
        rp, _, teams = chunk.partition(":")
        t = rp.split("/")[-1]
        if teams.strip() == "*":
            tm = json.load(open(f"{MD}/{t}/metadata.json"))
            teams = re.findall(r"--trainee-teams\s+(\S+)",
                               tm["original_command"])[0].strip("'\"")
        out.append((t, [x.split("/")[-1].replace(".txt", "") for x in teams.split(",") if x]))
    return out


def main():
    res, allteams = {}, set()
    for tag, (run, parent) in SETS.items():
        ts = teachers_and_teams(run)
        taught = sorted({b for _, bs in ts for b in bs})
        overlap = sorted(set(taught) & set(UNTAUGHT))
        missing = [b for b in taught if not os.path.exists(f"{MAIN}/data/teams/sample/{b}.txt")]
        best = [t for t, _ in ts if not os.path.exists(f"{MD}/{t}/best_model/best_model.zip")
                and not os.path.exists(f"{MD}/{t}/final_model.zip")]
        print(f"{tag:7s} parent={os.path.basename(os.path.dirname(parent)):26s} "
              f"n_teachers={len(ts):2d} n_taught={len(taught):2d} "
              f"overlap_with_untaught={overlap} missing_team_files={missing} "
              f"unloadable_teachers={best}")
        assert not overlap, f"{tag}: taught set OVERLAPS the untaught 8 -- {overlap}"
        assert not missing, f"{tag}: missing team files {missing}"
        assert not best, f"{tag}: teacher checkpoints not found {best}"
        res[tag] = {"parent": parent, "teachers": [t for t, _ in ts],
                    "per_teacher_taught": {t: bs for t, bs in ts}, "taught_union": taught}
        allteams |= set(taught)
    print(f"\nUNION of all taught teams across the gen-era sets vs R2ACTION: "
          f"{len(set().union(*[set(res[k]['taught_union']) for k in ('R4set','R3set','FUND','UNF')]))}")
    print(f"R2set taught (vs REV1FIN parent): {len(res['R2set']['taught_union'])}")
    json.dump(res, open(f"{HERE}/teacher_sets.json", "w"), indent=1)
    print(f"wrote {HERE}/teacher_sets.json")


if __name__ == "__main__":
    main()
