"""For each candidate flag: where it lives, how many source lines mention it, and the
ledger's verdict lines. Counts are grep counts, not a claim about deletable lines --
the estimate column in the report is derived from these plus the whole-module sizes.

Run:  PYTHONPATH=src python designs/research_state/measurements/era_boundary_2026-09-06/flag_footprint.py
"""
import json
import pathlib
import os
import subprocess

from utils.paths import repo_root

ROOT = repo_root()
SRC = os.path.join(ROOT, "src")
LEDGER = os.path.join(ROOT, "designs/research_state/ledger.md")
RESEARCH = os.path.join(ROOT, "designs/research_state")

# (config-field name, the dashed CLI spelling or None)
CANDIDATES = [
    ("value_dist_mode", "--value-dist-mode"),
    ("value_dist_bins", "--value-dist-bins"),
    ("value_dist_vmin", "--value-dist-vmin"),
    ("value_dist_vmax", "--value-dist-vmax"),
    ("value_dist_coef", "--value-dist-coef"),
    ("value_from_dist", "--value-from-dist"),
    ("value_tail_weight", "--value-tail-weight"),
    ("win_prob_coef", "--win-prob-coef"),
    ("use_popart", "--use-popart"),
    ("draw_penalty", "--draw-penalty"),
    ("win_prob_pbrs_coef", "--win-prob-pbrs-coef"),
    ("win_prob_pbrs_source", "--win-prob-pbrs-source"),
    ("win_prob_pbrs_frozen", "--win-prob-pbrs-frozen"),
    ("hand_shaping", "--hand-shaping"),
    ("pbrs_material", "--pbrs-material"),
    ("pbrs_belief", "--pbrs-belief"),
    ("stall_pbrs", "--stall-pbrs"),
    ("bias_redesign", "--bias-redesign"),
    ("drop_redundant_bias", "--drop-redundant-bias"),
    ("drop_switch_bias", "--drop-switch-bias"),
    ("switch_bias_weight", "--switch-bias-weight"),
    ("self_ko_hp_penalty", "--self-ko-hp-penalty"),
    ("mat_alive_weight", "--mat-alive-weight"),
    ("bias_additivity", "--bias-additivity"),
    ("all_shaping_pbrs", "--all-shaping-pbrs"),
    ("damage_candidate_k", "--damage-candidate-k"),
    ("pair_value_route", "--pair-value-route"),
    ("q_winprob_mode", "--q-winprob-mode"),
    ("cf_evidential", "--cf-evidential"),
    ("cf_twin_heads", "--cf-twin-heads"),
    ("cf_shadow_critic", "--cf-shadow-critic"),
    ("td_aux_coef", "--td-aux-coef"),
]

# the pre-pointer / already-dead residues the task asks about
RESIDUES = [
    "action_net", "zarch", "film_", "lut_", "pubval", "damage_reattend",
    "opp_belief_cls_k", "damage_refine_rounds", "threat_refine", "move_belief_prefuse",
]


def files_with(pattern, path, include="*.py", exclude_tests=False):
    cmd = ["grep", "-rIl", f"--include={include}", "-F", pattern, path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    fs = [os.path.relpath(f, ROOT) for f in p.stdout.splitlines() if f]
    if exclude_tests:
        fs = [f for f in fs if not f.endswith("_test.py")]
    return fs


def lines_with(pattern, path, include="*.py"):
    cmd = ["grep", "-rI", "-c", f"--include={include}", "-F", pattern, path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    tot = 0
    for line in p.stdout.splitlines():
        if ":" in line:
            tot += int(line.rsplit(":", 1)[1])
    return tot


def ledger_lines(pattern):
    p = subprocess.run(["grep", "-c", "-F", pattern, LEDGER], capture_output=True, text=True)
    return int(p.stdout.strip() or 0)


def research_files(pattern):
    p = subprocess.run(["grep", "-rIl", "-F", pattern, RESEARCH], capture_output=True, text=True)
    return len([f for f in p.stdout.splitlines() if f])


out = {}
hdr = (f"{'flag':26s} {'srcF':>5s} {'srcF!t':>7s} {'srcL':>6s} {'ledg':>5s} {'ledg-':>6s} "
       f"{'rs-files':>9s}")
print("=== CANDIDATE FLAGS (src = src/, excluding poke_env & rust_sim by convention) ===")
print(hdr)
print("-" * len(hdr))
for field, dashed in CANDIDATES:
    fs = files_with(field, SRC)
    fs_nt = [f for f in fs if not f.endswith("_test.py")]
    sl = lines_with(field, SRC)
    ll = ledger_lines(field)
    ld = ledger_lines(dashed) if dashed else 0
    rf = research_files(field)
    print(f"{field:26s} {len(fs):5d} {len(fs_nt):7d} {sl:6d} {ll:5d} {ld:6d} {rf:9d}")
    out[field] = {
        "cli": dashed,
        "src_files": len(fs), "src_files_nontest": len(fs_nt),
        "src_files_nontest_list": fs_nt,
        "src_line_mentions": sl,
        "ledger_lines_field": ll, "ledger_lines_dashed": ld,
        "research_state_files": rf,
    }

print("\n=== PRE-POINTER / ALREADY-DEAD RESIDUE GREPS (whole src/, all file types) ===")
rhdr = f"{'token':24s} {'py-files':>9s} {'py-lines':>9s} {'nontest':>8s}   files"
print(rhdr)
print("-" * len(rhdr))
res = {}
for tok in RESIDUES:
    fs = files_with(tok, SRC)
    fs_nt = [f for f in fs if not f.endswith("_test.py")]
    sl = lines_with(tok, SRC)
    print(f"{tok:24s} {len(fs):9d} {sl:9d} {len(fs_nt):8d}   {', '.join(fs_nt[:5])}")
    res[tok] = {"py_files": fs, "py_files_nontest": fs_nt, "py_line_mentions": sl}

with open("designs/research_state/measurements/era_boundary_2026-09-06/flag_footprint.json", "w") as fh:
    json.dump({"candidates": out, "residues": res}, fh, indent=2)
print("\nwrote designs/research_state/measurements/era_boundary_2026-09-06/flag_footprint.json")
