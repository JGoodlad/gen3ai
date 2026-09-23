"""Resolve B's --stable-opponents and every arm's --exploiter exactly as the launch path does
(`fixed_opponent_pool.resolve_stable_opponents`), and print what each entry RESOLVES to: the zip,
its recorded num_timesteps, the resolution rung, the label, and the pinned teams it will pilot.
Read-only. Run from the MAIN checkout root (relative models/ paths), with PYTHONPATH=<worktree>/src.
"""
import sys

from agents.model.model_version import ModelVersion
from agents.training.fixed_opponent_pool import resolve_stable_opponents

SPECS = {
    "B/C --stable-opponents": "models/ai_v13_18_teach5_offense_hidose/final_model.zip,"
                              "models/ai_v13_13_exploit5_offense/final_model.zip",
    "A2 --exploiter (G0)": "models/ai_v13_12_plateau/final_model.zip",
    "stand-in reader --exploiter": "models/ai_v13_16_teach5_offense_dist/final_model.zip",
}

current = ModelVersion.from_json_file("models/ai_v13_12_plateau/model_config.json")
for what, spec in SPECS.items():
    print(f"== {what}: {spec}")
    for e in resolve_stable_opponents(spec, current):
        teams = [f.rsplit("/", 1)[-1] for f in e.team_files] or ["(none — pilots the shared POOL)"]
        print(f"   {e.label}")
        print(f"      zip   {e.zip_path}")
        print(f"      step  {e.num_timesteps:,}   rung {e.resolution_rung} / {e.resolution_rule}")
        print(f"      arch  {e.arch_signature}   teams {', '.join(teams)}")
sys.exit(0)
