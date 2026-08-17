import os
import json
from src.utils.bridge.team_validator import validate_team_locally

def cleanup_teams():
    base_dir = "data/teams"
    format_id = "gen3ou"
    
    total_found = 0
    total_removed = 0

    for root, dirs, files in os.walk(base_dir):
        if "teams.json" in files:
            json_path = os.path.join(root, "teams.json")
            print(f"Checking {json_path}...")
            
            with open(json_path, "r") as f:
                meta = json.load(f)
            
            valid_meta = []
            for entry in meta:
                total_found += 1
                rel_file_path = entry.get("file")
                if not rel_file_path:
                    continue
                
                full_path = os.path.join("data", rel_file_path)
                if not os.path.exists(full_path):
                    print(f"  [MISSING] {rel_file_path}")
                    total_removed += 1
                    continue
                
                with open(full_path, "r") as f:
                    team_text = f.read()
                
                result = validate_team_locally(format_id, team_text)
                if result.get("valid"):
                    valid_meta.append(entry)
                else:
                    errors = result.get("errors", ["Unknown validation error"])
                    print(f"  [REMOVED] {entry.get('name', rel_file_path)}: {errors[0]}")
                    total_removed += 1
            
            # Save the filtered meta back to teams.json
            with open(json_path, "w") as f:
                json.dump(valid_meta, f, indent=2)

    print("\nCleanup Complete!")
    print(f"Total Teams Scanned: {total_found}")
    print(f"Total Illegal/Missing Teams Removed: {total_removed}")
    print(f"Remaining Valid Teams: {total_found - total_removed}")

if __name__ == "__main__":
    cleanup_teams()
