import subprocess
import json
import os
from typing import Dict, List, Any

def validate_team_locally(format_id: str, team_text: str) -> Dict[str, Any]:
    """
    Validates a Pokémon team using the local Node.js Showdown environment.
    """
    results = validate_teams_locally(format_id, [team_text])
    return results[0]

def validate_teams_locally(format_id: str, teams: List[str]) -> List[Dict[str, Any]]:
    """
    Validates multiple Pokémon teams in a single batch using the local Node.js Showdown environment.
    
    Args:
        format_id: The tier/format (e.g., 'gen3ou', 'gen9ou')
        teams: List of team texts in Showdown/PokePaste export format
        
    Returns:
        A list of dictionaries, each with 'valid' (bool) and 'errors' (List[str])
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    node_script_path = os.path.join(current_dir, 'validate_team.js')

    if not os.path.exists(node_script_path):
        return [{"valid": False, "errors": [f"Cannot find the Node bridge script at: {node_script_path}"]} for _ in teams]

    payload = {
        "format": format_id,
        "team": teams
    }
    
    try:
        process = subprocess.Popen(
            ['node', node_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=json.dumps(payload))
        
        if stderr:
            if "Error" in stderr or "cannot find module" in stderr.lower():
                return [{"valid": False, "errors": [f"Node execution error: {stderr}"]} for _ in teams]
            
        if not stdout:
            return [{"valid": False, "errors": ["No output from Node script"]} for _ in teams]
            
        return json.loads(stdout)
    except Exception as e:
        return [{"valid": False, "errors": [f"Subprocess failure: {str(e)}"]} for _ in teams]


if __name__ == "__main__":
    # Quick test
    test_format = "gen3ou"
    test_team = """
Tyranitar @ Leftovers  
Ability: Sand Stream  
EVs: 252 HP / 4 Atk / 252 SpA  
Quiet Nature  
- Rock Slide  
- Crunch  
- Pursuit  
- Fire Blast  
    """
    
    result = validate_team_locally(test_format, test_team)
    print(f"Result: {result}")
