import subprocess
import json
import os
import pytest
from src.utils.gen3_utils import GEN3_HP_IVS

def get_showdown_hp(ivs):
    """Calls the local Showdown bridge to get Hidden Power data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    node_script_path = os.path.join(current_dir, 'bridge', 'get_hp.js')
    
    process = subprocess.Popen(
        ['node', node_script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=json.dumps(ivs))
    
    if stderr:
        raise Exception(f"Node error: {stderr}")
    
    return json.loads(stdout)

@pytest.mark.parametrize("hp_type, ivs", GEN3_HP_IVS.items())
def test_hidden_power_mappings(hp_type, ivs):
    """
    Verifies that the IVs in GEN3_HP_IVS correctly result in the 
    specified Hidden Power type and 70 Power in Gen 3.
    """
    result = get_showdown_hp(ivs)
    
    assert "error" not in result, f"Error from Showdown bridge: {result.get('error')}"
    
    # Showdown returns type as a capitalized string, e.g., "Dark"
    assert result['type'] == hp_type, f"Expected {hp_type}, but got {result['type']} for IVs {ivs}"
    
    # All mapped IVs should result in 70 Power
    assert result['power'] == 70, f"Expected 70 Power, but got {result['power']} for {hp_type} with IVs {ivs}"

if __name__ == "__main__":
    # Manual run if not using pytest
    print("Verifying GEN3_HP_IVS mappings against Showdown...")
    failed = 0
    for hp_type, ivs in GEN3_HP_IVS.items():
        try:
            result = get_showdown_hp(ivs)
            if result['type'] == hp_type and result['power'] == 70:
                print(f"[PASS] {hp_type}: {ivs}")
            else:
                print(f"[FAIL] {hp_type}: Expected {hp_type} 70, got {result['type']} {result['power']}")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {hp_type}: {str(e)}")
            failed += 1
    
    if failed == 0:
        print("\nAll mappings verified successfully!")
    else:
        print(f"\nVerification failed with {failed} errors.")
        exit(1)
