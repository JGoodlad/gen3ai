import os
import json

class TeamLoader:
    """
    A utility class to load Pokémon teams from the data/teams directory.
    Supports categorized loading for sample teams and community/dump teams.
    """
    def __init__(self, base_dir="data/teams"):
        self.base_dir = base_dir
        self.sample_teams = []
        self.other_teams = []
        self._load_teams()

    def _load_teams(self):
        """Discovers all teams.json files and loads the corresponding team text."""
        if not os.path.exists(self.base_dir):
            print(f"Warning: Base directory {self.base_dir} does not exist.")
            return

        for root, dirs, files in os.walk(self.base_dir):
            if "teams.json" in files:
                json_path = os.path.join(root, "teams.json")
                try:
                    with open(json_path, "r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
                    
                for entry in meta:
                    # Skip invalid teams if the metadata flag is present
                    if entry.get("valid") is False:
                        continue
                        
                    rel_file_path = entry.get("file")
                    if not rel_file_path:
                        continue
                    
                    # Resolve path: entry['file'] is relative to 'data/'
                    # e.g., 'teams/sample/abc.txt' -> 'data/teams/sample/abc.txt'
                    full_path = os.path.join("data", rel_file_path)
                    
                    if not os.path.exists(full_path):
                        # Fallback for other potential path structures
                        full_path = os.path.join(self.base_dir, os.path.basename(rel_file_path))
                        if not os.path.exists(full_path):
                            # Try absolute or direct relative
                            full_path = rel_file_path
                    
                    try:
                        if os.path.exists(full_path):
                            with open(full_path, "r") as f:
                                team_text = f.read().strip()
                                
                            # Categorize based on the folder structure
                            if "sample" in root:
                                self.sample_teams.append(team_text)
                            else:
                                self.other_teams.append(team_text)
                        else:
                            print(f"Warning: Team file not found: {rel_file_path} (resolved to {full_path})")
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")

    def get_sample_teams(self):
        """Returns only the curated sample teams."""
        return self.sample_teams

    def get_other_teams(self):
        """Returns teams that are not in the sample category."""
        return self.other_teams

    def get_all_teams(self):
        """Returns all discovered teams."""
        return self.sample_teams + self.other_teams

    def __repr__(self):
        return f"<TeamLoader(samples={len(self.sample_teams)}, others={len(self.other_teams)})>"
