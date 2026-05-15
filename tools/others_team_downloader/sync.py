import hashlib
import os
import requests
import json
import re
import argparse
import sys
from typing import List, Dict, Any

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.bridge.team_validator import validate_teams_locally

DEFAULT_URL = "https://pokepast.es/aed6c2ad0c5c2593"
DEFAULT_OUTPUT_DIR = "data/teams/others/yak_attack"

def sync_dump(urls: List[str], output_dir: str, format_id: str = "gen3ou", validate: bool = True, append: bool = False):
    """Fetches, parses, and validates PokePaste team dumps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    metadata_path = os.path.join(output_dir, "teams.json")
    all_teams_metadata = []
    
    if append and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                all_teams_metadata = json.load(f)
            print(f"Appending to existing metadata with {len(all_teams_metadata)} teams.")
        except Exception as e:
            print(f"Warning: Could not load existing metadata: {e}")

    # Track existing IDs to avoid duplicates if appending
    existing_ids = {team["id"] for team in all_teams_metadata}
    
    for url in urls:
        raw_url = url.rstrip('/') + "/raw"
        print(f"\n--- Fetching dump from {raw_url} ---")
        try:
            res = requests.get(raw_url, timeout=60)
            if res.status_code != 200:
                print(f"Failed to fetch dump: {res.status_code}")
                continue
        except Exception as e:
            print(f"Error fetching dump: {e}")
            continue
        
        content = res.text
        
        pattern = r"===\s*(?:\[(.*?)\])?\s*(.*?)\s*===\s*\n(.*?)(?=\n===\s*(?:\[.*?\])?\s*.*?\s*===|$)"
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        if not matches:
            if content.strip():
                 matches = [(None, "Unnamed Team", content.strip())]
            else:
                print("Dump is empty.")
                continue

        print(f"Processing {len(matches)} potential teams...")
        
        teams_to_validate = []
        team_data = []
        
        for match in matches:
            if isinstance(match, tuple):
                team_format, full_name, team_text = match
            else:
                team_format = match.group(1)
                full_name = match.group(2).strip()
                team_text = match.group(3).strip()
            
            if not team_text:
                continue

            current_format = team_format if team_format else format_id
            team_id = hashlib.sha256(team_text.encode()).hexdigest()[:16]
            
            if team_id in existing_ids:
                # print(f"Skipping duplicate team: {full_name} ({team_id})")
                continue
                
            team_data.append({
                "id": team_id,
                "name": full_name,
                "format": current_format,
                "text": team_text
            })
            teams_to_validate.append(team_text)

        if not teams_to_validate:
            print("No new unique teams found in this dump.")
            continue

        # Perform batch validation
        validation_results = []
        if validate:
            print(f"Validating {len(teams_to_validate)} new teams in batch...")
            validation_results = validate_teams_locally(format_id, teams_to_validate)
        else:
            validation_results = [{"valid": True, "errors": []} for _ in teams_to_validate]

        valid_count = 0
        invalid_count = 0
        
        for i, data in enumerate(team_data):
            team_id = data["id"]
            full_name = data["name"]
            team_text = data["text"]
            current_format = data["format"]
            
            res = validation_results[i]
            is_valid = res.get("valid", False)
            validation_errors = res.get("errors", [])
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                # print(f"Invalid team: {full_name} ({team_id}) - {validation_errors[0] if validation_errors else 'Unknown error'}")

            filename = f"{team_id}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(team_text)
                
            all_teams_metadata.append({
                "id": team_id,
                "name": full_name,
                "format": current_format,
                "valid": is_valid,
                "errors": validation_errors,
                "file": os.path.relpath(filepath, "data"),
                "source": url
            })
            existing_ids.add(team_id)
        
        print(f"Synced {len(team_data)} new teams. {valid_count} valid, {invalid_count} invalid")
        
    with open(metadata_path, 'w') as f:
        json.dump(all_teams_metadata, f, indent=2)
    
    print(f"\nFinal Summary: {len(all_teams_metadata)} total teams in {output_dir}")
    print(f"Metadata index saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and validate teams from PokePaste")
    parser.add_argument("--url", type=str, nargs="+", default=[DEFAULT_URL], help="PokePaste URL(s)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--format", type=str, default="gen3ou", help="Default format if not specified in dump")
    parser.add_argument("--no-validate", action="store_true", help="Skip team validation")
    parser.add_argument("--append", action="store_true", help="Append to existing metadata instead of overwriting")
    
    args = parser.parse_args()
    
    sync_dump(args.url, args.output, args.format, not args.no_validate, args.append)

