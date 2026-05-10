import hashlib
import os
import requests
import json
import re

URL = "https://pokepast.es/c5929d2a3fc95749/raw"
OUTPUT_DIR = "data/teams/others/johnnyg2"
METADATA_PATH = os.path.join(OUTPUT_DIR, "teams.json")

def sync_dump():
    """Fetches and parses a PokePaste team dump."""
    print(f"Fetching dump from {URL}")
    try:
        res = requests.get(URL, timeout=30)
        if res.status_code != 200:
            print(f"Failed to fetch dump: {res.status_code}")
            return
    except Exception as e:
        print(f"Error fetching dump: {e}")
        return
    
    content = res.text
    
    # Pattern to match: === [gen3ou] name ===
    # It accounts for potential whitespace and the trailing ===
    # We use re.DOTALL to let .* match newlines
    pattern = r"=== \[gen3ou\] (.*?) ===\s*\n(.*?)(?=\n=== \[gen3ou\]|$)"
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        print("No teams found in the dump. Check the regex or source content.")
        # Try a more lenient pattern if the first one fails
        pattern = r"=== (.*?) ===\s*\n(.*?)(?=\n=== |$)"
        matches = list(re.finditer(pattern, content, re.DOTALL))
        if not matches:
             return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    teams_metadata = []
    for match in matches:
        full_name = match.group(1).strip()
        # Remove [gen3ou] if it was caught by the lenient pattern
        full_name = re.sub(r"^\[.*?\]\s*", "", full_name)
        
        team_text = match.group(2).strip()
        
        if not team_text:
            continue

        # ID is hash of team text to ensure uniqueness and stability
        team_id = hashlib.sha256(team_text.encode()).hexdigest()[:16]
        
        filename = f"{team_id}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(team_text)
            
        teams_metadata.append({
            "id": team_id,
            "name": full_name,
            "author": "JohnnyG2", 
            "category": "Team Dump - Uncategorized",
            "file": f"teams/others/johnnyg2/{filename}",
            "source": "https://pokepast.es/c5929d2a3fc95749"
        })
        
    with open(METADATA_PATH, 'w') as f:
        json.dump(teams_metadata, f, indent=2)
    
    print(f"Successfully synced {len(teams_metadata)} teams to {OUTPUT_DIR}")
    print(f"Metadata index saved to {METADATA_PATH}")

if __name__ == "__main__":
    sync_dump()
